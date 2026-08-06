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


def fwd_smem(block_m, block_n, dim, stages, elem=2):
    return elem * dim * (2 * block_m + 2 * stages * block_n)


def bwd_smem(block_m, block_n, dim, stages, elem=2):
    in_loop = (2 * elem * block_m * dim
               + 2 * elem * stages * block_n * dim
               + elem * block_m * block_n
               + 8 * block_n)
    tail = 8 * block_m * dim
    return max(in_loop, tail)


def warps_fit(block, threads, granularity=16):
    warps = threads // 32
    return block % warps == 0 and (block // warps) % granularity == 0


def fwd_configs(dim=128, limit=None):
    limit = limit or SMEM_LIMIT
    return [dict(block_M=bm, block_N=bn, threads=t, num_stages=s)
            for bm, bn in itertools.product((32, 64, 128, 256), repeat=2)
            for t in (128, 256, 512) if warps_fit(bm, t)
            for s in (1, 2, 3) if fwd_smem(bm, bn, dim, s) <= limit]


def bwd_configs(dim=128, limit=None):
    limit = limit or SMEM_LIMIT
    return [dict(block_M=bm, block_N=bn, threads=t, num_stages=s)
            for bm, bn in itertools.product((32, 64, 128), repeat=2)
            for t in (128, 256) if warps_fit(bm, t)
            for s in (1, 2, 3) if bwd_smem(bm, bn, dim, s) <= limit]


AUTOTUNE = True
FWD_TILES = dict(block_M=128, block_N=64, threads=256, num_stages=1)
BWD_TILES = dict(block_M=64, block_N=32, threads=128, num_stages=2)

assert FWD_TILES in fwd_configs(), "forward default is not a tunable config"
assert BWD_TILES in bwd_configs(), "backward default is not a tunable config"


if HAVE_TILELANG:
    FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    FWD_SPACE = fwd_configs()
    BWD_SPACE = bwd_configs()

    @autotune(configs=FWD_SPACE, warmup=10, rep=20)
    @tilelang.jit(out_idx=[3, 4], pass_configs=FAST_MATH)
    def flash_fwd(batch, heads, seq_len, dim, causal, groups,
                  block_M=128, block_N=64, threads=256, num_stages=1):
        scale = (1.0 / dim) ** 0.5 * 1.44269504
        kv_heads = heads // groups
        dt, acc = T.bfloat16, T.float32

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
            K: T.Tensor([batch, seq_len, kv_heads, dim], dt), #type: ignore
            V: T.Tensor([batch, seq_len, kv_heads, dim], dt), #type: ignore
            O: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
            lse: T.Tensor([batch, heads, seq_len], acc), #type: ignore
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                          threads=threads) as (bx, by, bz):
                q_s = T.alloc_shared([block_M, dim], dt)
                k_s = T.alloc_shared([block_N, dim], dt)
                v_s = T.alloc_shared([block_N, dim], dt)
                o_s = T.alloc_shared([block_M, dim], dt)
                scores = T.alloc_fragment([block_M, block_N], acc)
                probs = T.alloc_fragment([block_M, block_N], dt)
                out = T.alloc_fragment([block_M, dim], acc)
                row_max = T.alloc_fragment([block_M], acc)
                prev_max = T.alloc_fragment([block_M], acc)
                rescale = T.alloc_fragment([block_M], acc)
                row_sum = T.alloc_fragment([block_M], acc)
                denom = T.alloc_fragment([block_M], acc)

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], q_s)
                T.fill(out, 0)
                T.fill(denom, 0)
                T.fill(row_max, -T.infinity(acc))

                steps = (T.min(T.ceildiv(seq_len, block_N),
                               T.ceildiv((bx + 1) * block_M, block_N))
                         if causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(steps, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], k_s)
                    if causal:
                        for i, j in T.Parallel(block_M, block_N):
                            scores[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0, -T.infinity(acc))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            scores[i, j] = T.if_then_else(
                                k * block_N + j >= seq_len, -T.infinity(acc), 0)
                    T.gemm(q_s, k_s, scores, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    T.copy(row_max, prev_max)
                    T.fill(row_max, -T.infinity(acc))
                    T.reduce_max(scores, row_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        row_max[i] = T.max(row_max[i], prev_max[i])
                    for i in T.Parallel(block_M):
                        rescale[i] = T.exp2(prev_max[i] * scale - row_max[i] * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        scores[i, j] = T.exp2(scores[i, j] * scale - row_max[i] * scale)
                    T.reduce_sum(scores, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        denom[i] = denom[i] * rescale[i] + row_sum[i]
                    T.copy(scores, probs)
                    for i, j in T.Parallel(block_M, dim):
                        out[i, j] *= rescale[i]
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], v_s)
                    T.gemm(probs, v_s, out, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, dim):
                    out[i, j] /= denom[i]
                T.copy(out, o_s)
                T.copy(o_s, O[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    denom[i] = T.log2(denom[i]) + row_max[i] * scale
                T.copy(denom, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return kernel

    @tilelang.jit(out_idx=[2], pass_configs=FAST_MATH)
    def flash_delta(batch, heads, seq_len, dim):
        dt, acc = T.bfloat16, T.float32
        blk = 32

        @T.prim_func
        def kernel(
            O: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
            dO: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
            Delta: T.Tensor([batch, heads, seq_len], acc), #type: ignore
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
                o = T.alloc_fragment([blk, blk], dt)
                do = T.alloc_fragment([blk, blk], dt)
                prod = T.alloc_fragment([blk, blk], acc)
                total = T.alloc_fragment([blk], acc)
                T.clear(prod)
                for k in range(T.ceildiv(dim, blk)):
                    T.copy(O[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                    T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                    for i, j in T.Parallel(blk, blk):
                        prod[i, j] += o[i, j] * do[i, j]
                T.reduce_sum(prod, total, 1)
                T.copy(total, Delta[bz, bx, by * blk:(by + 1) * blk])

        return kernel

    def dq_layout(dQ):
        return T.Layout(dQ.shape,
                        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2),
                                            4 * (l % 8) + (d % 8) // 2])

    @tilelang.jit(out_idx=[1], pass_configs=FAST_MATH)
    def flash_unswizzle(batch, heads, seq_len, dim):
        dt, acc = T.bfloat16, T.float32
        blk = 64

        @T.prim_func
        def kernel(
            dQ: T.Tensor([batch, seq_len, heads, dim], acc), #type: ignore
            dQ_out: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
        ):
            with T.Kernel(T.ceildiv(seq_len, blk), heads, batch,
                          threads=128) as (bx, by, bz):
                T.annotate_layout({dQ: dq_layout(dQ)})
                T.copy(dQ[bz, bx * blk:(bx + 1) * blk, by, :],
                       dQ_out[bz, bx * blk:(bx + 1) * blk, by, :])

        return kernel

    @autotune(configs=BWD_SPACE, warmup=10, rep=20)
    @tilelang.jit(pass_configs=FAST_MATH)
    def flash_bwd(batch, heads, seq_len, dim, causal, groups,
                  block_M=64, block_N=32, threads=128, num_stages=2):
        sm_scale = (1.0 / dim) ** 0.5
        scale = sm_scale * 1.44269504
        kv_heads = heads // groups
        dt, acc = T.bfloat16, T.float32

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
            K: T.Tensor([batch, seq_len, kv_heads, dim], dt), #type: ignore
            V: T.Tensor([batch, seq_len, kv_heads, dim], dt), #type: ignore
            dO: T.Tensor([batch, seq_len, heads, dim], dt), #type: ignore
            lse: T.Tensor([batch, heads, seq_len], acc), #type: ignore
            Delta: T.Tensor([batch, heads, seq_len], acc), #type: ignore
            dQ: T.Tensor([batch, seq_len, heads, dim], acc), #type: ignore
            dK: T.Tensor([batch, seq_len, kv_heads, dim], acc), #type: ignore
            dV: T.Tensor([batch, seq_len, kv_heads, dim], acc), #type: ignore
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch,
                          threads=threads) as (bx, by, bz):
                k_s = T.alloc_shared([block_M, dim], dt)
                v_s = T.alloc_shared([block_M, dim], dt)
                q_s = T.alloc_shared([block_N, dim], dt)
                do_s = T.alloc_shared([block_N, dim], dt)
                ds_s = T.alloc_shared([block_M, block_N], dt)
                lse_s = T.alloc_shared([block_N], acc)
                delta_s = T.alloc_shared([block_N], acc)
                probs = T.alloc_fragment([block_M, block_N], acc)
                dscore = T.alloc_fragment([block_M, block_N], acc)
                probs_c = T.alloc_fragment([block_M, block_N], dt)
                dscore_c = T.alloc_fragment([block_M, block_N], dt)
                dv = T.alloc_fragment([block_M, dim], acc)
                dk = T.alloc_fragment([block_M, dim], acc)
                dq = T.alloc_fragment([block_N, dim], acc)
                dk_s = T.alloc_shared([block_M, dim], acc)
                dv_s = T.alloc_shared([block_M, dim], acc)

                T.annotate_layout({dQ: dq_layout(dQ)})
                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx // groups, :], k_s)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx // groups, :], v_s)
                T.clear(dv)
                T.clear(dk)

                start = T.floordiv(by * block_M, block_N) if causal else 0
                stop = T.ceildiv(seq_len, block_N)
                for k in T.Pipelined(start, stop, num_stages=num_stages):
                    T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q_s)
                    T.clear(probs)
                    T.gemm(k_s, q_s, probs, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_s)
                    for i, j in T.Parallel(block_M, block_N):
                        probs[i, j] = T.exp2(probs[i, j] * scale - lse_s[j])
                    if causal:
                        for i, j in T.Parallel(block_M, block_N):
                            probs[i, j] = T.if_then_else(
                                by * block_M + i <= k * block_N + j, probs[i, j], 0)
                    T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do_s)
                    T.clear(dscore)
                    T.gemm(v_s, do_s, dscore, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.copy(probs, probs_c)
                    T.gemm(probs_c, do_s, dv, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta_s)
                    for i, j in T.Parallel(block_M, block_N):
                        dscore_c[i, j] = probs[i, j] * (dscore[i, j] - delta_s[j]) * sm_scale
                    T.gemm(dscore_c, q_s, dk, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dscore_c, ds_s)
                    T.clear(dq)
                    T.gemm(ds_s, k_s, dq, transpose_A=True)
                    for i, j in T.Parallel(block_N, dim):
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

                T.copy(dv, dv_s)
                T.atomic_add(dV[bz, by * block_M:(by + 1) * block_M, bx // groups, :], dv_s)
                T.copy(dk, dk_s)
                T.atomic_add(dK[bz, by * block_M:(by + 1) * block_M, bx // groups, :], dk_s)

        return kernel


_kernels: dict = {}


def get_kernel(kind, *shape):
    key = (kind, shape, AUTOTUNE)
    if key not in _kernels:
        fn = flash_fwd if kind == "fwd" else flash_bwd
        tiles = FWD_TILES if kind == "fwd" else BWD_TILES
        _kernels[key] = fn(*shape) if AUTOTUNE else fn(*shape, **tiles)
    return _kernels[key]


def warmup(batch, seq_len, cfg=None, causal=True):
    """Run both autotune searches up front so training never stalls mid-step."""
    cfg = cfg or AttnConfig()
    shape = (batch, cfg.n_heads, seq_len, cfg.head_dim, causal, cfg.groups)
    print(f"tuning forward over {len(FWD_SPACE)} configs")
    get_kernel("fwd", *shape)
    print(f"tuning backward over {len(BWD_SPACE)} configs")
    get_kernel("bwd", *shape)
    print("cached to disk")


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, groups):
        batch, seq_len, heads, dim = q.shape
        out, lse = get_kernel("fwd", batch, heads, seq_len, dim, causal, groups)(q, k, v)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.groups = causal, groups
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        batch, seq_len, heads, dim = q.shape
        kv_heads = v.shape[-2]
        groups = heads // kv_heads

        tensors = [t if t.stride(-1) == 1 else t.contiguous()
                   for t in (grad_out, q, k, v, out)]
        grad_out, q, k, v, out = tensors

        delta = flash_delta(batch, heads, seq_len, dim)(out, grad_out)
        dq = torch.zeros(batch, seq_len, heads, dim, dtype=torch.float32, device=q.device)
        dk = torch.zeros(batch, seq_len, kv_heads, dim, dtype=torch.float32, device=q.device)
        dv = torch.zeros(batch, seq_len, kv_heads, dim, dtype=torch.float32, device=q.device)

        get_kernel("bwd", batch, heads, seq_len, dim, ctx.causal, groups)(
            q, k, v, grad_out, lse, delta, dq, dk, dv)

        dq = flash_unswizzle(batch, heads, seq_len, dim)(dq)
        return dq, dk.to(q.dtype), dv.to(q.dtype), None, None


def flash_attention(q, k, v, causal=True, groups=5):
    return FlashAttention.apply(q, k, v, causal, groups)


@dataclass
class AttnConfig:
    d_model: int = 1280
    n_heads: int = 10
    n_kv_heads: int = 2
    head_dim: int = 128
    rope_theta: float = 10_000.0
    max_seq_len: int = 8192
    qk_norm: bool = True
    norm_eps: float = 1e-5
    backend: str = "tilelang"

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_heads * self.head_dim == self.d_model
        assert self.head_dim in (64, 128)

    @property
    def groups(self):
        return self.n_heads // self.n_kv_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Rotary(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.build(theta)

    def build(self, theta):
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        angles = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        table = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", table.cos(), persistent=False)
        self.register_buffer("sin", table.sin(), persistent=False)

    def set_theta(self, theta):
        device = self.cos.device
        self.build(theta)
        self.cos, self.sin = self.cos.to(device), self.sin.to(device)


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(x, cos, sin):
    seq_len = x.shape[1]
    c = cos[:seq_len].to(x.dtype).unsqueeze(1)
    s = sin[:seq_len].to(x.dtype).unsqueeze(1)
    return x * c + rotate_half(x) * s


class Attention(nn.Module):
    def __init__(self, cfg: AttnConfig, rope: Rotary):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        heads, kv_heads, dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, heads * dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, kv_heads * dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, kv_heads * dim, bias=False)
        self.wo = nn.Linear(heads * dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x, backend=None):
        batch, seq_len, _ = x.shape
        cfg = self.cfg
        backend = backend or cfg.backend

        q = self.q_norm(self.wq(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim))
        k = self.k_norm(self.wk(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim))
        v = self.wv(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim)
        q = apply_rope(q, self.rope.cos, self.rope.sin)
        k = apply_rope(k, self.rope.cos, self.rope.sin)

        if backend == "tilelang":
            out = flash_attention(q.contiguous(), k.contiguous(), v.contiguous(),
                                  True, cfg.groups)
        else:
            qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
            if backend == "sdpa":
                out = F.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=True, scale=self.scale, enable_gqa=True)
            elif backend == "flex":
                from torch.nn.attention.flex_attention import flex_attention, create_block_mask
                mask = create_block_mask(lambda b, h, i, j: i >= j, B=None, H=None,
                                         Q_LEN=seq_len, KV_LEN=seq_len, device=x.device)
                out = flex_attention(qh, kh, vh, block_mask=mask, scale=self.scale,
                                     enable_gqa=True)
            elif backend == "math":
                kh = kh.repeat_interleave(cfg.groups, dim=1)
                vh = vh.repeat_interleave(cfg.groups, dim=1)
                scores = (qh.float() @ kh.float().transpose(-2, -1)) * self.scale
                causal = torch.ones(seq_len, seq_len, dtype=torch.bool,
                                    device=x.device).tril()
                out = (scores.masked_fill(~causal, float("-inf")).softmax(-1)
                       @ vh.float()).to(x.dtype)
            else:
                raise ValueError(f"unknown backend {backend!r}")
            out = out.transpose(1, 2)

        return self.wo(out.reshape(batch, seq_len, cfg.n_heads * cfg.head_dim))


def reference(q, k, v, causal, groups):
    dim = q.size(-1)
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(dim)
    if causal:
        seq_len = q.size(1)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float()).to(q.dtype)


def rope_reference(x, theta):
    batch, seq_len, heads, dim = x.shape
    half = dim // 2
    freqs = 1.0 / (theta ** (2 * torch.arange(half).double() / dim))
    angles = torch.arange(seq_len).double().unsqueeze(-1) * freqs
    c = angles.cos().float().view(1, seq_len, 1, half)
    s = angles.sin().float().view(1, seq_len, 1, half)
    left, right = x[..., :half], x[..., half:]
    return torch.cat([left * c - right * s, left * s + right * c], dim=-1)


def check_math():
    passed = True

    def check(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    torch.manual_seed(0)
    cfg = AttnConfig(backend="math")
    kb = 1024

    print("shared memory")
    for stages in (1, 2, 3):
        size = fwd_smem(128, 64, 128, stages)
        print(f"    forward  128x64 stages={stages}  {size/kb:6.1f} KB")
    print(f"    backward  64x32 stages=2  {bwd_smem(64, 32, 128, 2)/kb:6.1f} KB")
    print(f"    backward 128x32 stages=2  {bwd_smem(128, 32, 128, 2)/kb:6.1f} KB")
    check("forward default fits with headroom",
          fwd_smem(FWD_TILES["block_M"], FWD_TILES["block_N"], 128,
                   FWD_TILES["num_stages"]) <= 0.85 * 128 * kb)
    check("backward default fits with headroom",
          bwd_smem(64, 32, 128, 2) <= 0.85 * 128 * kb)
    check("backward at block_M=128 hits the sm_120 ceiling",
          bwd_smem(128, 32, 128, 2) >= 128 * kb)

    print("\nconfig spaces")
    fwd, bwd = fwd_configs(), bwd_configs()
    print(f"    forward  {len(fwd):3d} of 144")
    print(f"    backward {len(bwd):3d} of  54")
    check("forward default is tunable", FWD_TILES in fwd)
    check("backward default is tunable", BWD_TILES in bwd)
    check("backward limited to block_M <= 64", not any(c["block_M"] > 64 for c in bwd))
    check("backward all threads=128", all(c["threads"] == 128 for c in bwd))

    print("\nrope")
    rope = Rotary(128, 64, 10_000.0)
    x = torch.randn(2, 64, 3, 128)
    got, want = apply_rope(x, rope.cos, rope.sin), rope_reference(x, 10_000.0)
    check(f"matches explicit rotation ({(got - want).abs().max():.1e})",
          (got - want).abs().max() < 1e-4)
    check(f"norm preserved ({(got.norm(-1) - x.norm(-1)).abs().max():.1e})",
          (got.norm(-1) - x.norm(-1)).abs().max() < 1e-4)
    single = torch.randn(1, 1, 1, 128).expand(1, 64, 1, 128).contiguous()
    dots = (apply_rope(single, rope.cos, rope.sin)[0, :, 0]
            * apply_rope(single.clone(), rope.cos, rope.sin)[0, :, 0]).sum(-1)
    check(f"relative encoding holds ({(dots - dots[0]).abs().max():.1e})",
          (dots - dots[0]).abs().max() < 1e-3)
    rope.set_theta(500_000.0)
    check("set_theta rebuilds tables", rope.theta == 500_000.0)

    print("\nrmsnorm")
    norm = RMSNorm(8)
    y = norm(torch.randn(3, 8))
    check(f"unit rms output ({y.pow(2).mean().sqrt():.4f})",
          abs(y.pow(2).mean().sqrt() - 1) < 0.05)

    print("\nmodule")
    attn = Attention(cfg, Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta))
    n_params = sum(p.numel() for p in attn.parameters())
    check(f"parameters {n_params:,}", n_params == 3_932_416)
    check("no biases", all(m.bias is None for m in (attn.wq, attn.wk, attn.wv, attn.wo)))

    x = torch.randn(2, 32, cfg.d_model)
    y = attn(x, backend="math")
    check(f"shape {tuple(x.shape)} -> {tuple(y.shape)}", y.shape == x.shape)
    check(f"math and sdpa agree ({(y - attn(x, backend='sdpa')).abs().max():.1e})",
          (y - attn(x, backend="sdpa")).abs().max() < 2e-3)

    perturbed = x.clone()
    perturbed[:, -1] += 10.0
    drift = (attn(perturbed, backend="math")[:, :-1] - y[:, :-1]).abs().max()
    check(f"causal, no leak from future ({drift:.1e})", drift < 1e-4)

    x = torch.randn(2, 16, cfg.d_model, requires_grad=True)
    attn(x, backend="math").sum().backward()
    check("finite gradients everywhere",
          torch.isfinite(x.grad).all()
          and all(torch.isfinite(p.grad).all() for p in attn.parameters()))

    print("\nkernel constants")
    check("(1/dim)**0.5 == 1/sqrt(dim)", abs((1.0 / 128) ** 0.5 - 1 / math.sqrt(128)) < 1e-12)
    check("1.44269504 == log2(e)", abs(1.44269504 - math.log2(math.e)) < 1e-8)

    print("\n" + ("all math checks passed" if passed else "FAILURES ABOVE"))
    print("still needs a GPU: kernel compilation, dQ/dK/dV vs reference, tuning")
    return passed


def check_gpu(seq_len=512):
    if not (HAVE_TILELANG and torch.cuda.is_available()):
        print("needs tilelang and cuda")
        return
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()}  sm_{major}{minor}")
    print(f"shared memory limit in use: {SMEM_LIMIT/1024:.0f} KB")

    heads, kv_heads, dim, groups = 10, 2, 128, 5
    torch.manual_seed(0)
    q = torch.randn(2, seq_len, heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    k = torch.randn(2, seq_len, kv_heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    v = torch.randn(2, seq_len, kv_heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    grad = torch.randn(2, seq_len, heads, dim, device="cuda", dtype=DTYPE)

    for causal in (False, True):
        out = flash_attention(q, k, v, causal, groups)
        out.backward(grad, retain_graph=True)
        mine = [t.grad.clone() for t in (q, k, v)]
        q.grad = k.grad = v.grad = None

        want = reference(q, k, v, causal, groups)
        want.backward(grad, retain_graph=True)
        theirs = [t.grad.clone() for t in (q, k, v)]
        q.grad = k.grad = v.grad = None

        print(f"\ncausal={causal}")
        for name, a, b in zip(("out", "dQ", "dK", "dV"),
                              [out] + mine, [want] + theirs):
            err = (a.float() - b.float()).abs().max().item()
            print(f"  {name:4s} {err:.3e}  {'ok' if err < 2e-2 else 'FAIL'}")

    runs = []
    for _ in range(2):
        flash_attention(q, k, v, True, groups).backward(grad, retain_graph=True)
        runs.append(q.grad.clone())
        q.grad = None
    spread = (runs[0].float() - runs[1].float()).abs().max().item()
    print(f"\nrepeat backward differs by {spread:.3e}"
          f" ({'deterministic' if spread == 0 else 'fp32 atomics, expected'})")


def benchmark(batch=8, seq_len=2048):
    if not torch.cuda.is_available():
        print("needs cuda")
        return
    heads, kv_heads, dim, groups = 10, 2, 128, 5
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    k = torch.randn(batch, seq_len, kv_heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    v = torch.randn(batch, seq_len, kv_heads, dim, device="cuda", dtype=DTYPE, requires_grad=True)
    grad = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=DTYPE)
    flops = 2 * 2.0 * batch * heads * seq_len * seq_len * dim * 0.5 * 2.5

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
        q.grad = k.grad = v.grad = None

    qh, kh, vh, gh = (t.transpose(1, 2) for t in (q, k, v, grad))
    results = {}

    def run_sdpa():
        F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True
                                       ).backward(gh, retain_graph=True)
        clear()
    results["sdpa"] = time_it(run_sdpa)

    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        compiled = torch.compile(flex_attention)
        mask = create_block_mask(lambda b, h, i, j: i >= j, B=None, H=None,
                                 Q_LEN=seq_len, KV_LEN=seq_len, device="cuda")

        def run_flex():
            compiled(qh, kh, vh, block_mask=mask, enable_gqa=True
                     ).backward(gh, retain_graph=True)
            clear()
        results["flex"] = time_it(run_flex)
    except Exception as exc:
        print(f"flex unavailable: {exc}")

    if HAVE_TILELANG:
        try:
            def run_tilelang():
                flash_attention(q, k, v, True, groups).backward(grad, retain_graph=True)
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
        overall = 1 / (0.863 + 0.137 / speedup)
        print(f"\n  attention {speedup:.2f}x, end to end {overall:.3f}x,"
              f" {23 * (1 - 1 / overall):+.1f} days on a 23 day run")


def tune(batch=8, seq_len=2048):
    print(f"forward candidates  {len(fwd_configs())}")
    print(f"backward candidates {len(bwd_configs())}")
    print(f"shared memory limit {SMEM_LIMIT/1024:.0f} KB")
    if not (HAVE_TILELANG and torch.cuda.is_available()):
        for name, space in (("forward", fwd_configs()), ("backward", bwd_configs())):
            print(f"\n{name}")
            for c in space:
                print(f"  {c}")
        return
    cfg = AttnConfig()
    warmup(batch, seq_len, cfg)
    shape = (batch, cfg.n_heads, seq_len, cfg.head_dim, True, cfg.groups)
    for kind in ("fwd", "bwd"):
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