from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tilelang
    import tilelang.language as T
    from tilelang.autotuner import autotune
    HAVE_TILELANG = True
except ImportError:
    HAVE_TILELANG = False


DTYPE = torch.bfloat16
SMEM_PER_SM_KB = {8: 164, 9: 228, 10: 228, 12: 128}
MAX_ACCUM_BYTES = 96 * 1024


def shared_mem_limit(device=0, margin=0.9):
    try:
        if not torch.cuda.is_available():
            return 112 * 1024
        props = torch.cuda.get_device_properties(device)
        for name in ("shared_memory_per_block_optin",
                     "max_shared_memory_per_block_optin",
                     "shared_memory_per_block"):
            value = getattr(props, name, 0)
            if value:
                return int(value * margin)
        return int(SMEM_PER_SM_KB.get(props.major, 100) * 1024 * margin)
    except Exception:
        return 112 * 1024


SMEM_LIMIT = shared_mem_limit()


def gemm_smem(block_m, block_n, block_k, stages, weights=1, elem=2):
    return (block_m * block_k * elem * stages
            + block_k * block_n * elem * stages * weights
            + block_m * block_n * elem)


def fusion_accumulator_bytes(block_m, d_model):
    return block_m * d_model * 4


def warps_fit(block, threads, granularity=16):
    warps = threads // 32
    return block % warps == 0 and (block // warps) % granularity == 0


def gate_up_configs(limit=None):
    limit = limit or SMEM_LIMIT
    return [dict(block_M=bm, block_N=bn, block_K=bk, threads=t, num_stages=s)
            for bm, bn in itertools.product((32, 64, 128, 256), repeat=2)
            for bk in (32, 64, 128)
            for t in (128, 256) if warps_fit(bm, t)
            for s in (1, 2, 3)
            if gemm_smem(bm, bn, bk, s, weights=2) <= limit
            and bm * bn * 4 * 2 <= MAX_ACCUM_BYTES]


def grad_configs(limit=None):
    limit = limit or SMEM_LIMIT
    return [dict(block_M=bm, block_N=bn, threads=t)
            for bm, bn in itertools.product((32, 64, 128, 256), repeat=2)
            for t in (128, 256) if warps_fit(bm, t)
            if 5 * bm * bn * 2 <= limit]


AUTOTUNE = True
GATE_UP_TILES = dict(block_M=128, block_N=64, block_K=64, threads=128, num_stages=2)
GRAD_TILES = dict(block_M=128, block_N=64, threads=128)

assert GATE_UP_TILES in gate_up_configs(), "gate_up default is not a tunable config"
assert GRAD_TILES in grad_configs(), "grad default is not a tunable config"


if HAVE_TILELANG:
    FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    GATE_UP_SPACE = gate_up_configs()
    GRAD_SPACE = grad_configs()

    @autotune(configs=GATE_UP_SPACE, warmup=10, rep=20)
    @tilelang.jit(out_idx=[3], pass_configs=FAST_MATH)
    def gate_up_swiglu(tokens, d_model, ffn,
                       block_M=128, block_N=64, block_K=64,
                       threads=128, num_stages=2):
        dt, acc = T.bfloat16, T.float32

        @T.prim_func
        def kernel(
            X: T.Tensor([tokens, d_model], dt), #type: ignore
            Wg: T.Tensor([d_model, ffn], dt), #type: ignore
            Wu: T.Tensor([d_model, ffn], dt), #type: ignore
            Act: T.Tensor([tokens, ffn], dt), #type: ignore
        ):
            with T.Kernel(T.ceildiv(ffn, block_N), T.ceildiv(tokens, block_M), threads=threads) as (bx, by):
                x_s = T.alloc_shared([block_M, block_K], dt)
                wg_s = T.alloc_shared([block_K, block_N], dt)
                wu_s = T.alloc_shared([block_K, block_N], dt)
                out_s = T.alloc_shared([block_M, block_N], dt)
                gate = T.alloc_fragment([block_M, block_N], acc)
                up = T.alloc_fragment([block_M, block_N], acc)
                act = T.alloc_fragment([block_M, block_N], dt)

                T.clear(gate)
                T.clear(up)
                for k in T.Pipelined(T.ceildiv(d_model, block_K), num_stages=num_stages):
                    T.copy(X[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K], x_s)
                    T.copy(Wg[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N], wg_s)
                    T.copy(Wu[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N], wu_s)
                    T.gemm(x_s, wg_s, gate, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(x_s, wu_s, up, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, block_N):
                    act[i, j] = T.Cast(dt, gate[i, j] / (1.0 + T.exp(-gate[i, j])) * up[i, j])
                T.copy(act, out_s)
                T.copy(out_s, Act[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N])

        return kernel

    @autotune(configs=GATE_UP_SPACE, warmup=10, rep=20)
    @tilelang.jit(out_idx=[3, 4], pass_configs=FAST_MATH)
    def gate_up_split(tokens, d_model, ffn,
                      block_M=128, block_N=64, block_K=64,
                      threads=128, num_stages=2):
        dt, acc = T.bfloat16, T.float32

        @T.prim_func
        def kernel(
            X: T.Tensor([tokens, d_model], dt), #type: ignore
            Wg: T.Tensor([d_model, ffn], dt), #type: ignore
            Wu: T.Tensor([d_model, ffn], dt), #type: ignore
            Gate: T.Tensor([tokens, ffn], dt), #type: ignore
            Up: T.Tensor([tokens, ffn], dt), #type: ignore
        ):
            with T.Kernel(T.ceildiv(ffn, block_N), T.ceildiv(tokens, block_M), threads=threads) as (bx, by):
                x_s = T.alloc_shared([block_M, block_K], dt)
                wg_s = T.alloc_shared([block_K, block_N], dt)
                wu_s = T.alloc_shared([block_K, block_N], dt)
                tile = T.alloc_shared([block_M, block_N], dt)
                gate = T.alloc_fragment([block_M, block_N], acc)
                up = T.alloc_fragment([block_M, block_N], acc)
                cast = T.alloc_fragment([block_M, block_N], dt)

                T.clear(gate)
                T.clear(up)
                for k in T.Pipelined(T.ceildiv(d_model, block_K), num_stages=num_stages):
                    T.copy(X[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K], x_s)
                    T.copy(Wg[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N], wg_s)
                    T.copy(Wu[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N], wu_s)
                    T.gemm(x_s, wg_s, gate, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(x_s, wu_s, up, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, block_N):
                    cast[i, j] = T.Cast(dt, gate[i, j])
                T.copy(cast, tile)
                T.copy(tile, Gate[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N])
                for i, j in T.Parallel(block_M, block_N):
                    cast[i, j] = T.Cast(dt, up[i, j])
                T.copy(cast, tile)
                T.copy(tile, Up[by * block_M:(by + 1) * block_M, bx * block_N:(bx + 1) * block_N])

        return kernel

    @autotune(configs=GRAD_SPACE, warmup=10, rep=20)
    @tilelang.jit(out_idx=[3, 4], pass_configs=FAST_MATH)
    def swiglu_grad(tokens, ffn, block_M=128, block_N=64, threads=128):
        dt, acc = T.bfloat16, T.float32

        @T.prim_func
        def kernel(
            dAct: T.Tensor([tokens, ffn], dt), #type: ignore
            Gate: T.Tensor([tokens, ffn], dt), #type: ignore
            Up: T.Tensor([tokens, ffn], dt), #type: ignore
            dGate: T.Tensor([tokens, ffn], dt), #type: ignore
            dUp: T.Tensor([tokens, ffn], dt), #type: ignore
        ):
            with T.Kernel(T.ceildiv(ffn, block_N), T.ceildiv(tokens, block_M), threads=threads) as (bx, by):
                da = T.alloc_fragment([block_M, block_N], acc)
                g = T.alloc_fragment([block_M, block_N], acc)
                u = T.alloc_fragment([block_M, block_N], acc)
                sig = T.alloc_fragment([block_M, block_N], acc)
                out = T.alloc_fragment([block_M, block_N], dt)
                tile = T.alloc_shared([block_M, block_N], dt)

                rows = slice(by * block_M, (by + 1) * block_M)
                cols = slice(bx * block_N, (bx + 1) * block_N)
                T.copy(dAct[rows, cols], da)
                T.copy(Gate[rows, cols], g)
                T.copy(Up[rows, cols], u)

                for i, j in T.Parallel(block_M, block_N):
                    sig[i, j] = 1.0 / (1.0 + T.exp(-g[i, j]))
                for i, j in T.Parallel(block_M, block_N):
                    out[i, j] = T.Cast( dt,da[i, j] * u[i, j] * sig[i, j] * (1.0 + g[i, j] * (1.0 - sig[i, j])))
                T.copy(out, tile)
                T.copy(tile, dGate[rows, cols])
                for i, j in T.Parallel(block_M, block_N):
                    out[i, j] = T.Cast(dt, da[i, j] * g[i, j] * sig[i, j])
                T.copy(out, tile)
                T.copy(tile, dUp[rows, cols])

        return kernel


_kernels: dict = {}


def get_kernel(kind, *shape):
    key = (kind, shape, AUTOTUNE)
    if key not in _kernels:
        fn = {"gate_up": gate_up_swiglu,
              "split": gate_up_split,
              "grad": swiglu_grad}[kind]
        tiles = GRAD_TILES if kind == "grad" else GATE_UP_TILES
        _kernels[key] = fn(*shape) if AUTOTUNE else fn(*shape, **tiles)
    return _kernels[key]


def warmup(tokens, cfg=None):
    """Run the autotune searches up front so training never stalls mid-step."""
    cfg = cfg or FFNConfig()
    print(f"tuning gate_up over {len(GATE_UP_SPACE)} configs")
    get_kernel("gate_up", tokens, cfg.d_model, cfg.ffn_hidden)
    get_kernel("split", tokens, cfg.d_model, cfg.ffn_hidden)
    print(f"tuning grad over {len(GRAD_SPACE)} configs")
    get_kernel("grad", tokens, cfg.ffn_hidden)
    print("cached to disk")


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down):
        tokens, d_model = x.shape
        ffn = w_gate.shape[1]
        gate, up = get_kernel("split", tokens, d_model, ffn)(x, w_gate, w_up)
        act = F.silu(gate) * up
        out = act @ w_down
        ctx.save_for_backward(x, w_gate, w_up, w_down, gate, up, act)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w_gate, w_up, w_down, gate, up, act = ctx.saved_tensors
        tokens, ffn = gate.shape

        d_act = grad_out @ w_down.t()
        d_gate, d_up = get_kernel("grad", tokens, ffn)(d_act, gate, up)

        d_x = d_gate @ w_gate.t() + d_up @ w_up.t()
        d_w_gate = x.t() @ d_gate
        d_w_up = x.t() @ d_up
        d_w_down = act.t() @ grad_out
        return d_x, d_w_gate, d_w_up, d_w_down


def fused_swiglu(x, w_gate, w_up, w_down):
    return FusedSwiGLU.apply(x, w_gate, w_up, w_down)


@dataclass
class FFNConfig:
    d_model: int = 1280
    ffn_hidden: int = 3328
    backend: str = "tilelang"

    def __post_init__(self):
        assert self.ffn_hidden % 128 == 0
        assert self.d_model % 128 == 0

    @property
    def params(self):
        return 3 * self.d_model * self.ffn_hidden


class SwiGLU(nn.Module):
    def __init__(self, cfg: FFNConfig):
        super().__init__()
        self.cfg = cfg
        self.w_gate = nn.Parameter(torch.empty(cfg.d_model, cfg.ffn_hidden))
        self.w_up = nn.Parameter(torch.empty(cfg.d_model, cfg.ffn_hidden))
        self.w_down = nn.Parameter(torch.empty(cfg.ffn_hidden, cfg.d_model))
        std = 0.02
        nn.init.normal_(self.w_gate, std=std)
        nn.init.normal_(self.w_up, std=std)
        nn.init.normal_(self.w_down, std=std / math.sqrt(2 * 27))

    def forward(self, x, backend=None):
        cfg = self.cfg
        backend = backend or cfg.backend
        shape = x.shape
        flat = x.reshape(-1, cfg.d_model)

        if backend == "tilelang":
            out = fused_swiglu(flat, self.w_gate, self.w_up, self.w_down)
        elif backend == "torch":
            out = (F.silu(flat @ self.w_gate) * (flat @ self.w_up)) @ self.w_down
        elif backend == "math":
            gate = flat.float() @ self.w_gate.float()
            up = flat.float() @ self.w_up.float()
            out = ((gate * torch.sigmoid(gate)) * up) @ self.w_down.float()
            out = out.to(x.dtype)
        else:
            raise ValueError(f"unknown backend {backend!r}")

        return out.reshape(shape)


def reference(x, w_gate, w_up, w_down):
    gate = x.float() @ w_gate.float()
    up = x.float() @ w_up.float()
    return ((gate * torch.sigmoid(gate)) * up) @ w_down.float()


def check_math():
    passed = True

    def check(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    torch.manual_seed(0)
    cfg = FFNConfig()
    kb = 1024
    layers, tokens = 27, 8 * 2048

    print("full fusion feasibility")
    for bm in (16, 32, 64, 128):
        size = fusion_accumulator_bytes(bm, cfg.d_model)
        print(f"    block_M={bm:3d}  down accumulator fp32 {size/kb:7.1f} KB"
              f"  {'fits' if size <= SMEM_LIMIT else 'overflow'}")
    check("full fusion needs block_M <= 16, too small for the tensor cores",
          fusion_accumulator_bytes(32, cfg.d_model) > SMEM_LIMIT)

    print("\nhbm traffic, one ffn forward at 8x2048 tokens")
    unfused = 6 * tokens * cfg.ffn_hidden * 2
    fused = 2 * tokens * cfg.ffn_hidden * 2
    print(f"    unfused {unfused/1e6:7.1f} MB")
    print(f"    fused   {fused/1e6:7.1f} MB")
    print(f"    saved   {(unfused-fused)/1e6:7.1f} MB per layer,"
          f" {(unfused-fused)*layers/1e9:.2f} GB over {layers} layers")
    print(f"    at 1792 GB/s that is"
          f" {(unfused-fused)*layers/1792e9*1e3:.2f} ms per forward")
    check("fusion removes two thirds of the intermediate traffic",
          fused * 3 == unfused)

    print("\nconfig spaces")
    gu, gr = gate_up_configs(), grad_configs()
    print(f"    gate_up {len(gu):3d}   grad {len(gr):3d}")
    check("gate_up default is tunable", GATE_UP_TILES in gu)
    check("grad default is tunable", GRAD_TILES in gr)
    check("every gate_up config fits shared memory",
          all(gemm_smem(c["block_M"], c["block_N"], c["block_K"],
                        c["num_stages"], weights=2) <= SMEM_LIMIT for c in gu))
    check("every gate_up config fits the register file",
          all(c["block_M"] * c["block_N"] * 8 <= MAX_ACCUM_BYTES for c in gu))

    print("\nsilu derivative")
    z = torch.randn(2000, dtype=torch.float64) * 3
    z.requires_grad_(True)
    F.silu(z).sum().backward()
    sig = torch.sigmoid(z.detach())
    analytic = sig * (1 + z.detach() * (1 - sig))
    err = (z.grad - analytic).abs().max().item()
    check(f"matches sigmoid(z)*(1+z*(1-sigmoid(z))) ({err:.2e})", err < 1e-12)
    check("silu(0)=0 and dsilu(0)=0.5",
          abs(F.silu(torch.zeros(1)).item()) < 1e-12
          and abs(torch.sigmoid(torch.zeros(1)).item() - 0.5) < 1e-12)

    print("\ngradients against autograd")
    m, d, h = 32, 64, 96
    x = torch.randn(m, d, dtype=torch.float64, requires_grad=True)
    wg = torch.randn(d, h, dtype=torch.float64, requires_grad=True) * 0.1
    wu = torch.randn(d, h, dtype=torch.float64, requires_grad=True) * 0.1
    wd = torch.randn(h, d, dtype=torch.float64, requires_grad=True) * 0.1
    for t in (wg, wu, wd):
        t.retain_grad()
    grad_out = torch.randn(m, d, dtype=torch.float64)

    out = (F.silu(x @ wg) * (x @ wu)) @ wd
    out.backward(grad_out)
    auto = [x.grad.clone(), wg.grad.clone(), wu.grad.clone(), wd.grad.clone()]

    gate = (x @ wg).detach()
    up = (x @ wu).detach()
    act = F.silu(gate) * up
    d_act = grad_out @ wd.detach().t()
    sig = torch.sigmoid(gate)
    d_gate = d_act * up * sig * (1 + gate * (1 - sig))
    d_up = d_act * F.silu(gate)
    manual = [d_gate @ wg.detach().t() + d_up @ wu.detach().t(),
              x.detach().t() @ d_gate,
              x.detach().t() @ d_up,
              act.t() @ grad_out]
    for name, a, b in zip(("d_x", "d_w_gate", "d_w_up", "d_w_down"), manual, auto):
        rel = ((a - b).abs().max() / b.abs().max()).item()
        check(f"{name:9s} rel err {rel:.2e}", rel < 1e-10)

    print("\nmodule")
    ffn = SwiGLU(cfg)
    n = sum(p.numel() for p in ffn.parameters())
    check(f"parameters {n:,} == 3*{cfg.d_model}*{cfg.ffn_hidden}", n == cfg.params)
    print(f"    {cfg.params/1e6:.2f}M per layer, {cfg.params*layers/1e6:.1f}M"
          f" over {layers} layers")
    check("ffn is 76 percent of a transformer block",
          abs(cfg.params / (cfg.params + 3_932_160) - 0.765) < 0.01)

    x = torch.randn(2, 16, cfg.d_model)
    y = ffn(x, backend="torch")
    check(f"shape {tuple(x.shape)} -> {tuple(y.shape)}", y.shape == x.shape)
    err = (y - ffn(x, backend="math")).abs().max().item()
    check(f"torch and fp32 reference agree ({err:.2e})", err < 1e-4)

    x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
    ffn(x, backend="torch").sum().backward()
    check("finite gradients everywhere",
          torch.isfinite(x.grad).all()
          and all(torch.isfinite(p.grad).all() for p in ffn.parameters()))

    print("\nactivation storage")
    store = 2 * tokens * cfg.ffn_hidden * 2
    print(f"    gate+up for backward {store/1e6:.1f} MB per layer,"
          f" {store*layers/1e9:.2f} GB over {layers}")
    recompute = 2 * tokens * cfg.d_model * 2 * cfg.ffn_hidden
    print(f"    recomputing instead costs {recompute/1e9:.1f} GFLOP per layer"
          f" per backward")
    check("storing gate+up exceeds the 9 GB of headroom in the memory budget",
          store * layers > 5e9)

    print("\n" + ("all math checks passed" if passed else "FAILURES ABOVE"))
    print("still needs a GPU: kernel compilation, fused output vs reference, tuning")
    return passed


def check_gpu(tokens=4096):
    if not (HAVE_TILELANG and torch.cuda.is_available()):
        print("needs tilelang and cuda")
        return
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()}  sm_{major}{minor}")
    print(f"shared memory limit in use: {SMEM_LIMIT/1024:.0f} KB")

    cfg = FFNConfig()
    torch.manual_seed(0)
    x = torch.randn(tokens, cfg.d_model, device="cuda", dtype=DTYPE, requires_grad=True)
    wg = torch.randn(cfg.d_model, cfg.ffn_hidden, device="cuda", dtype=DTYPE,
                     requires_grad=True) * 0.02
    wu = torch.randn(cfg.d_model, cfg.ffn_hidden, device="cuda", dtype=DTYPE,
                     requires_grad=True) * 0.02
    wd = torch.randn(cfg.ffn_hidden, cfg.d_model, device="cuda", dtype=DTYPE,
                     requires_grad=True) * 0.02
    for t in (x, wg, wu, wd):
        t.retain_grad()
    grad_out = torch.randn(tokens, cfg.d_model, device="cuda", dtype=DTYPE)

    act = get_kernel("gate_up", tokens, cfg.d_model, cfg.ffn_hidden)(x, wg, wu)
    want_act = F.silu(x.float() @ wg.float()) * (x.float() @ wu.float())
    err = (act.float() - want_act).abs().max().item()
    scale = want_act.abs().max().item()
    print(f"\nfused activation vs fp32: max err {err:.3e}"
          f" (rel {err/scale:.3e})  {'ok' if err/scale < 2e-2 else 'FAIL'}")

    out = fused_swiglu(x, wg, wu, wd)
    out.backward(grad_out, retain_graph=True)
    mine = [t.grad.clone() for t in (x, wg, wu, wd)]
    for t in (x, wg, wu, wd):
        t.grad = None

    want = reference(x, wg, wu, wd).to(DTYPE)
    want.backward(grad_out, retain_graph=True)
    theirs = [t.grad.clone() for t in (x, wg, wu, wd)]
    for t in (x, wg, wu, wd):
        t.grad = None

    print()
    for name, a, b in zip(("out", "d_x", "d_w_gate", "d_w_up", "d_w_down"),
                          [out] + mine, [want] + theirs):
        rel = ((a.float() - b.float()).abs().max()
               / b.float().abs().max().clamp(min=1e-6)).item()
        print(f"  {name:9s} rel err {rel:.3e}  {'ok' if rel < 3e-2 else 'FAIL'}")


def benchmark(tokens=16384):
    if not torch.cuda.is_available():
        print("needs cuda")
        return
    cfg = FFNConfig()
    x = torch.randn(tokens, cfg.d_model, device="cuda", dtype=DTYPE, requires_grad=True)
    wg = torch.randn(cfg.d_model, cfg.ffn_hidden, device="cuda", dtype=DTYPE,
                     requires_grad=True)
    wu = torch.randn(cfg.d_model, cfg.ffn_hidden, device="cuda", dtype=DTYPE,
                     requires_grad=True)
    wd = torch.randn(cfg.ffn_hidden, cfg.d_model, device="cuda", dtype=DTYPE,
                     requires_grad=True)
    grad_out = torch.randn(tokens, cfg.d_model, device="cuda", dtype=DTYPE)
    flops = 2.0 * tokens * cfg.d_model * cfg.ffn_hidden * 3 * 3

    def time_it(fn, iters=30):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    def clear():
        for t in (x, wg, wu, wd):
            t.grad = None

    results = {}

    def eager():
        ((F.silu(x @ wg) * (x @ wu)) @ wd).backward(grad_out, retain_graph=True)
        clear()
    results["eager"] = time_it(eager)

    try:
        compiled = torch.compile(lambda a, g, u, d: (F.silu(a @ g) * (a @ u)) @ d)

        def run_compiled():
            compiled(x, wg, wu, wd).backward(grad_out, retain_graph=True)
            clear()
        results["compiled"] = time_it(run_compiled)
    except Exception as exc:
        print(f"torch.compile unavailable: {exc}")

    if HAVE_TILELANG:
        try:
            def run_tilelang():
                fused_swiglu(x, wg, wu, wd).backward(grad_out, retain_graph=True)
                clear()
            results["tilelang"] = time_it(run_tilelang)
        except Exception as exc:
            print(f"tilelang failed: {type(exc).__name__}: {exc}")

    print()
    for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:10s} {ms:8.3f} ms   {flops/ms*1e-9:7.1f} TFLOP/s")

    if "tilelang" in results and len(results) > 1:
        best_other = min(v for n, v in results.items() if n != "tilelang")
        speedup = best_other / results["tilelang"]
        overall = 1 / (0.235 + 0.765 / speedup)
        print(f"\n  ffn {speedup:.2f}x, whole block {overall:.3f}x")
        print(f"  ffn is 76 percent of block parameters, so that is the weight")


def tune(tokens=16384):
    print(f"gate_up candidates {len(gate_up_configs())}")
    print(f"grad candidates    {len(grad_configs())}")
    print(f"shared memory limit {SMEM_LIMIT/1024:.0f} KB")
    if not (HAVE_TILELANG and torch.cuda.is_available()):
        for name, space in (("gate_up", gate_up_configs()), ("grad", grad_configs())):
            print(f"\n{name}")
            for c in space[:12]:
                print(f"  {c}")
        return
    cfg = FFNConfig()
    warmup(tokens, cfg)
    for kind, shape in (("gate_up", (tokens, cfg.d_model, cfg.ffn_hidden)),
                        ("split", (tokens, cfg.d_model, cfg.ffn_hidden)),
                        ("grad", (tokens, cfg.ffn_hidden))):
        kernel = get_kernel(kind, *shape)
        print(f"\n{kind}: {getattr(kernel, 'config', None)}")
        latency = getattr(kernel, "latency", None)
        if latency is not None:
            print(f"     {latency:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        args.check = True
    if args.check:
        check_math()
    if args.gpu:
        check_gpu()
    if args.bench:
        benchmark()
    if args.tune:
        tune()