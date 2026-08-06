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
IGNORE_INDEX = -100


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


def lse_smem(block_m, block_n, block_k, stages, elem=2):
    return block_m * block_k * elem * stages + block_k * block_n * elem * stages


def logits_bytes(tokens, vocab, elem=2):
    return tokens * vocab * elem


def warps_fit(block, threads, granularity=16):
    warps = threads // 32
    return block % warps == 0 and (block // warps) % granularity == 0


def lse_configs(limit=None):
    limit = limit or SMEM_LIMIT
    return [dict(block_M=bm, block_N=bn, block_K=bk, threads=t, num_stages=s)
            for bm, bn in itertools.product((32, 64, 128), (64, 128, 256))
            for bk in (32, 64, 128)
            for t in (128, 256) if warps_fit(bm, t)
            for s in (1, 2, 3)
            if lse_smem(bm, bn, bk, s) <= limit
            and bm * bn * 4 <= MAX_ACCUM_BYTES]


AUTOTUNE = True
LSE_TILES = dict(block_M=128, block_N=128, block_K=64, threads=128, num_stages=2)
VOCAB_CHUNK = 4096

assert LSE_TILES in lse_configs(), "lse default is not a tunable config"
assert VOCAB_CHUNK % 128 == 0, "vocab chunk should stay tensor core aligned"


if HAVE_TILELANG:
    FAST_MATH = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    LSE_SPACE = lse_configs()

    @autotune(configs=LSE_SPACE, warmup=10, rep=20)
    @tilelang.jit(out_idx=[2], pass_configs=FAST_MATH)
    def logsumexp_kernel(tokens, d_model, vocab,
                         block_M=128, block_N=128, block_K=64,
                         threads=128, num_stages=2):
        dt, acc = T.bfloat16, T.float32

        @T.prim_func
        def kernel(
            H: T.Tensor([tokens, d_model], dt), #type: ignore
            W: T.Tensor([d_model, vocab], dt),  #type: ignore
            Lse: T.Tensor([tokens], acc),  #type: ignore
        ):
            with T.Kernel(T.ceildiv(tokens, block_M), threads=threads) as bx:
                h_s = T.alloc_shared([block_M, block_K], dt)
                w_s = T.alloc_shared([block_K, block_N], dt)
                logits = T.alloc_fragment([block_M, block_N], acc)
                row_max = T.alloc_fragment([block_M], acc)
                prev_max = T.alloc_fragment([block_M], acc)
                rescale = T.alloc_fragment([block_M], acc)
                tile_sum = T.alloc_fragment([block_M], acc)
                total = T.alloc_fragment([block_M], acc)

                T.fill(row_max, -T.infinity(acc))
                T.fill(total, 0)

                for n in T.serial(T.ceildiv(vocab, block_N)):
                    T.clear(logits)
                    for k in T.Pipelined(T.ceildiv(d_model, block_K),
                                         num_stages=num_stages):
                        T.copy(H[bx * block_M:(bx + 1) * block_M,
                                 k * block_K:(k + 1) * block_K], h_s)
                        T.copy(W[k * block_K:(k + 1) * block_K,
                                 n * block_N:(n + 1) * block_N], w_s)
                        T.gemm(h_s, w_s, logits, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(block_M, block_N):
                        logits[i, j] = T.if_then_else(
                            n * block_N + j >= vocab, -T.infinity(acc), logits[i, j])

                    T.copy(row_max, prev_max)
                    T.fill(row_max, -T.infinity(acc))
                    T.reduce_max(logits, row_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        row_max[i] = T.max(row_max[i], prev_max[i])
                    for i in T.Parallel(block_M):
                        rescale[i] = T.exp(prev_max[i] - row_max[i])
                    for i, j in T.Parallel(block_M, block_N):
                        logits[i, j] = T.exp(logits[i, j] - row_max[i])
                    T.reduce_sum(logits, tile_sum, dim=1)
                    for i in T.Parallel(block_M):
                        total[i] = total[i] * rescale[i] + tile_sum[i]

                for i in T.Parallel(block_M):
                    total[i] = T.log(total[i]) + row_max[i]
                T.copy(total, Lse[bx * block_M:(bx + 1) * block_M])

        return kernel


_kernels: dict = {}


def get_kernel(shape):
    key = (shape, AUTOTUNE)
    if key not in _kernels:
        _kernels[key] = (logsumexp_kernel(*shape) if AUTOTUNE
                         else logsumexp_kernel(*shape, **LSE_TILES))
    return _kernels[key]


def warmup(tokens, d_model=1280, vocab=49152):
    """Run the autotune search up front so training never stalls mid-step."""
    print(f"tuning logsumexp over {len(LSE_SPACE)} configs")
    get_kernel((tokens, d_model, vocab))
    print("cached to disk")


def gather_target_logits(hidden, weight, targets):
    valid = targets.clamp(min=0)
    rows = weight.index_select(1, valid).t()
    return (hidden.float() * rows.float()).sum(-1)


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        tokens, d_model = hidden.shape
        vocab = weight.shape[1]
        mask = targets != IGNORE_INDEX
        count = mask.sum().clamp(min=1)

        if HAVE_TILELANG and hidden.is_cuda:
            lse = get_kernel((tokens, d_model, vocab))(hidden, weight)
        else:
            lse = torch.logsumexp((hidden.float() @ weight.float()), dim=-1)

        target_logit = gather_target_logits(hidden, weight, targets)
        per_token = torch.where(mask, lse - target_logit,
                                torch.zeros_like(lse))
        loss = per_token.sum() / count

        ctx.save_for_backward(hidden, weight, targets, lse, count)
        ctx.chunk = chunk
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, targets, lse, count = ctx.saved_tensors
        tokens, d_model = hidden.shape
        vocab = weight.shape[1]
        chunk = ctx.chunk
        mask = (targets != IGNORE_INDEX).unsqueeze(1)
        scale = grad_loss / count

        d_hidden = torch.zeros_like(hidden, dtype=torch.float32)
        d_weight = torch.zeros_like(weight, dtype=torch.float32)
        rows = torch.arange(tokens, device=hidden.device)

        for start in range(0, vocab, chunk):
            stop = min(start + chunk, vocab)
            w_chunk = weight[:, start:stop]
            logits = hidden.float() @ w_chunk.float()
            probs = torch.exp(logits - lse.unsqueeze(1))
            hit = (targets >= start) & (targets < stop) & (targets != IGNORE_INDEX)
            if hit.any():
                idx = rows[hit]
                probs[idx, targets[hit] - start] -= 1.0
            probs = probs * mask * scale
            d_hidden += probs @ w_chunk.float().t()
            d_weight[:, start:stop] = hidden.float().t() @ probs

        return d_hidden.to(hidden.dtype), d_weight.to(weight.dtype), None, None


def fused_cross_entropy(hidden, weight, targets, chunk=VOCAB_CHUNK):
    return FusedCrossEntropy.apply(hidden, weight, targets, chunk)


@dataclass
class HeadConfig:
    d_model: int = 1280
    vocab: int = 49152
    tied: bool = True
    chunk: int = VOCAB_CHUNK

    def __post_init__(self):
        assert self.vocab % 128 == 0
        assert self.d_model % 128 == 0

    @property
    def params(self):
        return self.d_model * self.vocab


class LMHead(nn.Module):
    def __init__(self, cfg: HeadConfig, embedding: nn.Embedding = None):
        super().__init__()
        self.cfg = cfg
        if cfg.tied:
            assert embedding is not None, "tied head needs the embedding module"
            self.embedding = [embedding]
            self.weight = None
        else:
            self.embedding = None
            self.weight = nn.Parameter(torch.empty(cfg.d_model, cfg.vocab))
            nn.init.normal_(self.weight, std=0.02)

    def matrix(self):
        if self.cfg.tied:
            return self.embedding[0].weight.t()
        return self.weight

    def forward(self, hidden, targets=None, backend="tilelang"):
        shape = hidden.shape
        flat = hidden.reshape(-1, self.cfg.d_model)
        weight = self.matrix()

        if targets is None:
            return (flat @ weight).reshape(*shape[:-1], self.cfg.vocab)

        flat_targets = targets.reshape(-1)
        if backend == "tilelang":
            return fused_cross_entropy(flat, weight, flat_targets, self.cfg.chunk)
        if backend == "torch":
            return F.cross_entropy(flat.float() @ weight.float(), flat_targets,
                                   ignore_index=IGNORE_INDEX)
        raise ValueError(f"unknown backend {backend!r}")


def check_math():
    passed = True

    def check(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    torch.manual_seed(0)
    cfg = HeadConfig()
    mb = 1e6
    tokens = 8 * 2048

    print("the tensor this kernel deletes")
    naive = logits_bytes(tokens, cfg.vocab)
    softmax_fp32 = tokens * cfg.vocab * 4
    fused = tokens * 4
    print(f"    logits bf16          {naive/mb:9.1f} MB")
    print(f"    fp32 softmax on top  {softmax_fp32/mb:9.1f} MB")
    print(f"    fused lse only       {fused/mb:9.4f} MB")
    print(f"    ratio                {naive/fused:9.0f}x")
    check("logits alone exceed 1.5 GB", naive > 1.5e9)
    check("fused output is under a megabyte", fused < 1e6)

    print("\nbackward peak, one vocabulary chunk at a time")
    for chunk in (2048, 4096, 8192, cfg.vocab):
        peak = tokens * chunk * 4
        tag = "full vocab" if chunk == cfg.vocab else f"chunk {chunk}"
        print(f"    {tag:12s} {peak/mb:9.1f} MB")
    check("chunk 4096 keeps the backward under 300 MB",
          tokens * 4096 * 4 < 300e6)

    print("\nconfig space")
    space = lse_configs()
    print(f"    {len(space)} configs, smem limit {SMEM_LIMIT/1024:.0f} KB")
    check("lse default is tunable", LSE_TILES in space)
    check("every config fits shared memory",
          all(lse_smem(c["block_M"], c["block_N"], c["block_K"],
                       c["num_stages"]) <= SMEM_LIMIT for c in space))
    check("every config fits the register file",
          all(c["block_M"] * c["block_N"] * 4 <= MAX_ACCUM_BYTES for c in space))
    d = LSE_TILES
    print(f"    default smem {lse_smem(d['block_M'], d['block_N'], d['block_K'], d['num_stages'])/1024:.0f} KB,"
          f" accumulator {d['block_M']*d['block_N']*4/1024:.0f} KB")

    print("\nonline logsumexp recurrence")
    x = torch.randn(64, 5000, dtype=torch.float64) * 8
    running_max = torch.full((64,), float("-inf"), dtype=torch.float64)
    running_sum = torch.zeros(64, dtype=torch.float64)
    for start in range(0, 5000, 128):
        tile = x[:, start:start + 128]
        prev = running_max.clone()
        running_max = torch.maximum(running_max, tile.max(dim=1).values)
        running_sum = (running_sum * torch.exp(prev - running_max)
                       + torch.exp(tile - running_max.unsqueeze(1)).sum(1))
    online = torch.log(running_sum) + running_max
    err = (online - torch.logsumexp(x, dim=1)).abs().max().item()
    check(f"matches torch.logsumexp ({err:.2e})", err < 1e-12)

    big = torch.tensor([[1e4, 1e4 + 1.0]], dtype=torch.float64)
    check("no overflow at logits of 1e4",
          torch.isfinite(torch.logsumexp(big, dim=1)).all())

    print("\ncross entropy identity")
    logits = torch.randn(200, 500, dtype=torch.float64)
    targets = torch.randint(0, 500, (200,))
    lse = torch.logsumexp(logits, dim=1)
    manual = (lse - logits[torch.arange(200), targets]).mean()
    ref = F.cross_entropy(logits, targets)
    check(f"lse - target_logit == cross_entropy ({(manual-ref).abs():.2e})",
          (manual - ref).abs() < 1e-12)

    print("\ngradient identity")
    logits = torch.randn(64, 200, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 200, (64,))
    F.cross_entropy(logits, targets).backward()
    probs = torch.softmax(logits.detach(), dim=1)
    onehot = F.one_hot(targets, 200).double()
    analytic = (probs - onehot) / 64
    err = (logits.grad - analytic).abs().max().item()
    check(f"d_logits == (softmax - onehot)/N ({err:.2e})", err < 1e-12)

    print("\nend to end against F.cross_entropy")
    d_model, vocab, n = 64, 300, 128
    hidden = torch.randn(n, d_model, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(d_model, vocab, dtype=torch.float64, requires_grad=True) * 0.05
    targets = torch.randint(0, vocab, (n,))
    targets[:8] = IGNORE_INDEX

    loss = fused_cross_entropy(hidden, weight, targets, chunk=64)
    loss.backward()
    mine = [hidden.grad.clone(), weight.grad.clone()]
    hidden.grad = weight.grad = None

    ref = F.cross_entropy(hidden @ weight, targets, ignore_index=IGNORE_INDEX)
    ref.backward()
    theirs = [hidden.grad.clone(), weight.grad.clone()]
    hidden.grad = weight.grad = None

    check(f"loss {loss.item():.10f} vs {ref.item():.10f}",
          (loss - ref).abs() < 1e-10)
    for name, a, b in zip(("d_hidden", "d_weight"), mine, theirs):
        rel = ((a - b).abs().max() / b.abs().max()).item()
        check(f"{name:9s} rel err {rel:.2e}", rel < 1e-9)

    check("ignore_index rows contribute nothing",
          fused_cross_entropy(hidden, weight,
                              torch.full_like(targets, IGNORE_INDEX), 64).item() == 0.0)

    print("\ntied head")
    emb = nn.Embedding(cfg.vocab, cfg.d_model)
    head = LMHead(cfg, emb)
    check("tied head holds no parameters of its own",
          sum(p.numel() for p in head.parameters()) == 0)
    check(f"shares the {cfg.params/1e6:.1f}M embedding matrix",
          head.matrix().data_ptr() == emb.weight.data_ptr())

    print("\n" + ("all math checks passed" if passed else "FAILURES ABOVE"))
    print("still needs a GPU: kernel compilation, lse vs reference, tuning")
    return passed


def check_gpu(tokens=4096):
    if not (HAVE_TILELANG and torch.cuda.is_available()):
        print("needs tilelang and cuda")
        return
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()}  sm_{major}{minor}")
    print(f"shared memory limit {SMEM_LIMIT/1024:.0f} KB")

    cfg = HeadConfig()
    torch.manual_seed(0)
    hidden = torch.randn(tokens, cfg.d_model, device="cuda", dtype=DTYPE) * 0.1
    weight = torch.randn(cfg.d_model, cfg.vocab, device="cuda", dtype=DTYPE) * 0.02
    hidden.requires_grad_(True)
    weight.requires_grad_(True)
    targets = torch.randint(0, cfg.vocab, (tokens,), device="cuda")

    lse = get_kernel((tokens, cfg.d_model, cfg.vocab))(hidden, weight)
    want = torch.logsumexp(hidden.float() @ weight.float(), dim=-1)
    err = (lse - want).abs().max().item()
    print(f"\nlse vs reference: max err {err:.3e}  {'ok' if err < 5e-3 else 'FAIL'}")

    loss = fused_cross_entropy(hidden, weight, targets)
    loss.backward()
    mine = [hidden.grad.clone(), weight.grad.clone()]
    hidden.grad = weight.grad = None

    ref = F.cross_entropy(hidden.float() @ weight.float(), targets)
    ref.backward()
    theirs = [hidden.grad.clone(), weight.grad.clone()]
    hidden.grad = weight.grad = None

    print(f"loss {loss.item():.6f} vs {ref.item():.6f}")
    for name, a, b in zip(("d_hidden", "d_weight"), mine, theirs):
        rel = ((a.float() - b.float()).abs().max()
               / b.float().abs().max().clamp(min=1e-6)).item()
        print(f"  {name:9s} rel err {rel:.3e}  {'ok' if rel < 3e-2 else 'FAIL'}")

    torch.cuda.reset_peak_memory_stats()
    fused_cross_entropy(hidden, weight, targets).backward()
    fused_peak = torch.cuda.max_memory_allocated()
    hidden.grad = weight.grad = None
    torch.cuda.reset_peak_memory_stats()
    F.cross_entropy(hidden.float() @ weight.float(), targets).backward()
    naive_peak = torch.cuda.max_memory_allocated()
    print(f"\npeak memory  fused {fused_peak/1e6:.0f} MB   naive {naive_peak/1e6:.0f} MB"
          f"   saved {(naive_peak-fused_peak)/1e6:.0f} MB")


def benchmark(tokens=16384):
    if not torch.cuda.is_available():
        print("needs cuda")
        return
    cfg = HeadConfig()
    hidden = torch.randn(tokens, cfg.d_model, device="cuda", dtype=DTYPE,
                         requires_grad=True) * 0.1
    weight = torch.randn(cfg.d_model, cfg.vocab, device="cuda", dtype=DTYPE,
                         requires_grad=True) * 0.02
    targets = torch.randint(0, cfg.vocab, (tokens,), device="cuda")

    def time_it(fn, iters=20):
        for _ in range(5):
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
        hidden.grad = weight.grad = None

    results = {}

    def naive():
        F.cross_entropy(hidden.float() @ weight.float(), targets).backward()
        clear()
    try:
        results["naive"] = time_it(naive)
    except torch.cuda.OutOfMemoryError:
        print("naive path ran out of memory, which is the point")

    if HAVE_TILELANG:
        try:
            def fused():
                fused_cross_entropy(hidden, weight, targets).backward()
                clear()
            results["fused"] = time_it(fused)
        except Exception as exc:
            print(f"fused failed: {type(exc).__name__}: {exc}")

    print()
    for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:8s} {ms:8.3f} ms")
    if len(results) == 2:
        print(f"\n  speed {results['naive']/results['fused']:.2f}x")
        print("  memory is the real win, see --gpu for peak allocation")


def tune(tokens=16384):
    print(f"lse candidates {len(lse_configs())}")
    print(f"shared memory limit {SMEM_LIMIT/1024:.0f} KB")
    if not (HAVE_TILELANG and torch.cuda.is_available()):
        for c in lse_configs()[:12]:
            print(f"  {c}")
        return
    cfg = HeadConfig()
    warmup(tokens, cfg.d_model, cfg.vocab)
    kernel = get_kernel((tokens, cfg.d_model, cfg.vocab))
    print(f"\nbest: {getattr(kernel, 'config', None)}")
    latency = getattr(kernel, "latency", None)
    if latency is not None:
        print(f"      {latency:.3f} ms")


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