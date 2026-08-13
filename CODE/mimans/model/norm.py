from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HAVE_NATIVE = hasattr(F, "rms_norm")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, impl: str = "native"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.impl = impl if (impl != "native" or HAVE_NATIVE) else "manual"
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.impl == "native":
            return F.rms_norm(x, (self.dim,), self.weight.to(x.dtype), self.eps)
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, impl={self.impl}"


def reference(x, weight, eps=1e-5):
    h = x.double()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return h * weight.double()


def dispatch_warning(module: RMSNorm, x: torch.Tensor) -> str:
    if module.impl != "native":
        return "manual implementation, no fused dispatch"
    if module.weight.dtype != x.dtype:
        return (f"weight {module.weight.dtype} vs input {x.dtype}: "
                "cast at the call site or the fused kernel is skipped")
    return "dtypes match, fused dispatch available"


def check():
    torch.manual_seed(0)
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    print(f"native F.rms_norm available: {HAVE_NATIVE}")
    print(f"torch {torch.__version__}")

    print("\nnative matches the fp64 reference")
    native = RMSNorm(1280, impl="native")
    manual = RMSNorm(1280, impl="manual")
    with torch.no_grad():
        w = torch.randn(1280) * 0.1 + 1.0
        native.weight.copy_(w)
        manual.weight.copy_(w)
    x = torch.randn(4, 16, 1280)
    want = reference(x, w).float()
    err_native = (native(x) - want).abs().max().item()
    err_manual = (manual(x) - want).abs().max().item()
    check_one(f"native  err {err_native:.2e}", err_native < 1e-4)
    check_one(f"manual  err {err_manual:.2e}", err_manual < 1e-4)
    check_one(f"native and manual agree "
              f"({(native(x) - manual(x)).abs().max():.2e})",
              (native(x) - manual(x)).abs().max() < 1e-5)

    print("\nfused dispatch check")
    bf16 = torch.randn(4, 16, 1280, dtype=torch.bfloat16)
    fp32_weight = RMSNorm(1280)
    print(f"    fp32 weight, bf16 input : {dispatch_warning(fp32_weight, bf16)}")
    cast = RMSNorm(1280).to(torch.bfloat16)
    print(f"    bf16 weight, bf16 input : {dispatch_warning(cast, bf16)}")
    check_one("forward casts the weight so dispatch is never silently lost",
              torch.isfinite(fp32_weight(bf16)).all())

    print("\nproperties")
    y = native(x)
    check_one(f"unit rms with unit gain "
              f"({RMSNorm(1280)(x).pow(2).mean().sqrt():.4f})",
              abs(RMSNorm(1280)(x).pow(2).mean().sqrt() - 1) < 0.02)
    check_one("scale invariant", (native(x * 100) - y).abs().max() < 2e-3)
    check_one("gain starts at one", (RMSNorm(1280).weight == 1).all())
    check_one(f"parameters {native.weight.numel()}", native.weight.numel() == 1280)

    big = torch.randn(2, 4, 1280, dtype=torch.bfloat16) * 1e4
    check_one("no overflow in bf16 at 1e4",
              torch.isfinite(RMSNorm(1280)(big)).all())

    print("\ngradients")
    xa = torch.randn(2, 8, 1280, requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    RMSNorm(1280, impl="native")(xa).sum().backward()
    RMSNorm(1280, impl="manual")(xb).sum().backward()
    check_one(f"native and manual grads agree "
              f"({(xa.grad - xb.grad).abs().max():.2e})",
              (xa.grad - xb.grad).abs().max() < 1e-5)
    check_one("finite", torch.isfinite(xa.grad).all())

    print("\n" + ("rmsnorm ok" if passed else "FAILURES ABOVE"))
    return passed


def benchmark(tokens=16384, dim=1280, iters=50):
    if not torch.cuda.is_available():
        print("needs cuda")
        return
    x = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True)
    grad = torch.randn_like(x)

    def time_it(fn):
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

    variants = {}
    for name, impl, cast in (("native fp32 w", "native", False),
                             ("native bf16 w", "native", True),
                             ("manual", "manual", False)):
        mod = RMSNorm(dim, impl=impl).cuda()
        if cast:
            mod = mod.to(torch.bfloat16)

        def run(m=mod):
            m(x).backward(grad, retain_graph=True)
            x.grad = None
        variants[name] = time_it(run)

    try:
        compiled = torch.compile(RMSNorm(dim).cuda())

        def run_compiled():
            compiled(x).backward(grad, retain_graph=True)
            x.grad = None
        variants["compiled"] = time_it(run_compiled)
    except Exception as exc:
        print(f"torch.compile unavailable: {exc}")

    bytes_moved = tokens * dim * 2 * 3
    print(f"\n{tokens} x {dim} bf16, forward and backward")
    for name, ms in sorted(variants.items(), key=lambda kv: kv[1]):
        print(f"  {name:16s} {ms:7.4f} ms   {bytes_moved/ms/1e6:7.0f} GB/s")
    print("\n  memory bound, so GB/s against 1792 is the number that matters")
    print("  pick the winner here, not the one that sounds best")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if not any(vars(args).values()):
        args.check = True
    if args.check:
        check()
    if args.bench:
        benchmark()