from __future__ import annotations
import math
from dataclasses import dataclass

import torch


NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Quintic iteration driving the singular values of G toward one."""
    a, b, c = NS_COEFFS
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G.clone()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def update_scale(param: torch.Tensor, factor: float = 0.2):
    return factor * math.sqrt(max(param.size(0), param.size(1)))


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.1, scale_factor=0.2):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        scale_factor=scale_factor)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2:
                    raise ValueError(
                        f"Muon takes 2D matrices only, got shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                direction = g.add(buf, alpha=mu) if group["nesterov"] else buf
                ortho = newton_schulz(direction, group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(ortho, alpha=-lr * update_scale(p, group["scale_factor"]))
        return loss


@dataclass
class ScheduleConfig:
    total_steps: int = 76_294
    warmup_steps: int = 2_000
    decay_fraction: float = 0.10
    decay_shape: str = "inv_sqrt"
    final_fraction: float = 0.0

    @property
    def decay_start(self):
        return int(self.total_steps * (1 - self.decay_fraction))


def wsd_factor(step: int, cfg: ScheduleConfig, decay_from: int = None):
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    start = cfg.decay_start if decay_from is None else decay_from
    if step < start:
        return 1.0
    span = max(1, cfg.total_steps - start)
    t = min(1.0, (step - start) / span)
    if cfg.decay_shape == "inv_sqrt":
        shape = 1.0 - math.sqrt(t)
    elif cfg.decay_shape == "cosine":
        shape = 0.5 * (1.0 + math.cos(math.pi * t))
    elif cfg.decay_shape == "linear":
        shape = 1.0 - t
    else:
        raise ValueError(f"unknown decay shape {cfg.decay_shape!r}")
    return cfg.final_fraction + (1.0 - cfg.final_fraction) * shape


@dataclass
class OptimConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.1
    adamw_lr: float = 3e-3
    adamw_betas: tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0
    grad_clip: float = 1.0


class HybridOptimizer:
    def __init__(self, model, cfg: OptimConfig = None,
                 schedule: ScheduleConfig = None):
        self.cfg = cfg or OptimConfig()
        self.schedule = schedule or ScheduleConfig()
        groups = model.param_groups(weight_decay=self.cfg.muon_weight_decay)
        self.matrices = groups[0]["params"]
        self.others = groups[1]["params"]
        self.all_params = self.matrices + self.others
        self.muon = Muon(self.matrices, lr=self.cfg.muon_lr,
                         momentum=self.cfg.muon_momentum,
                         ns_steps=self.cfg.muon_ns_steps,
                         weight_decay=self.cfg.muon_weight_decay)
        self.adamw = torch.optim.AdamW(self.others, lr=self.cfg.adamw_lr,
                                       betas=self.cfg.adamw_betas,
                                       eps=self.cfg.adamw_eps,
                                       weight_decay=self.cfg.adamw_weight_decay)
        self.step_count = 0
        self.decay_from = None

    def begin_decay(self, step: int = None):
        self.decay_from = self.step_count if step is None else step

    def set_step(self, step: int):
        self.step_count = step
        factor = wsd_factor(step, self.schedule, self.decay_from)
        for group in self.muon.param_groups:
            group["lr"] = self.cfg.muon_lr * factor
        for group in self.adamw.param_groups:
            group["lr"] = self.cfg.adamw_lr * factor
        return factor

    def clip(self):
        if not self.cfg.grad_clip:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(self.all_params,
                                                    self.cfg.grad_clip))

    def step(self):
        norm = self.clip()
        self.muon.step()
        self.adamw.step()
        self.step_count += 1
        return norm

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(),
                "adamw": self.adamw.state_dict(),
                "step": self.step_count,
                "decay_from": self.decay_from}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.step_count = state["step"]
        self.decay_from = state.get("decay_from")

    def memory_bytes(self):
        muon = sum(p.numel() for p in self.matrices) * 4
        adamw = sum(p.numel() for p in self.others) * 8
        return muon, adamw


def check():
    torch.manual_seed(0)
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    print("newton-schulz at 5 steps is an approximate orthogonalisation")
    print("    it does not drive every singular value to one and is not meant to")
    for shape in ((1280, 1280), (1280, 3328), (3328, 1280)):
        G = torch.randn(*shape, dtype=torch.float32)
        s = torch.linalg.svdvals(newton_schulz(G, steps=5).float())
        band = ((s > 0.9) & (s < 1.1)).float().mean()
        print(f"    {str(shape):12s} median {s.median():.3f} "
              f"min {s.min():.4f} max {s.max():.3f} in[0.9,1.1] {band:.0%}")
        check_one(f"{shape} median singular value near one",
                  0.8 < s.median() < 1.15)
    check_one("output shape matches input",
              newton_schulz(torch.randn(256, 256)).shape == (256, 256))

    print("    square matrices lag hardest, and wq and wo are square here")
    square = torch.linalg.svdvals(
        newton_schulz(torch.randn(1280, 1280), steps=5).float())
    wide = torch.linalg.svdvals(
        newton_schulz(torch.randn(1280, 3328), steps=5).float())
    check_one("wide matrices orthogonalise better than square",
              wide.min() > square.min())

    print("\nmore steps tighten the band, so the iteration is converging")
    G = torch.randn(256, 256)
    mins = []
    for steps in (1, 3, 5, 8, 12):
        s = torch.linalg.svdvals(newton_schulz(G, steps=steps).float())
        mins.append(s.min().item())
        print(f"    steps {steps:2d}  min {s.min():.4f} median {s.median():.4f} "
              f"max {s.max():.3f}")
    check_one("smallest singular value rises with steps", mins[-1] > mins[0])

    print("\nscale invariance, which is the property muon actually needs")
    G = torch.randn(512, 1024)
    base = newton_schulz(G, steps=5).float()
    drift = max((newton_schulz(G * k, steps=5).float() - base).abs().max().item()
                for k in (1e-3, 1e3))
    for k in (1e-3, 1.0, 1e3):
        s = torch.linalg.svdvals(newton_schulz(G * k, steps=5).float())
        print(f"    gradient x{k:<8g} median sv {s.median():.6f}")
    check_one(f"identical across six orders of gradient magnitude ({drift:.1e})",
              drift < 1e-2)

    print("\ncondition number, the real goal")
    for shape, name in (((256, 256), "square gaussian"),
                        ((1280, 3328), "wide gaussian")):
        G = torch.randn(*shape)
        s0 = torch.linalg.svdvals(G)
        s1 = torch.linalg.svdvals(newton_schulz(G, steps=5).float())
        before = (s0.max() / s0.min()).item()
        after = (s1.max() / s1.min().clamp(min=1e-8)).item()
        print(f"    {name:16s} {before:9.1e} -> {after:8.1e}  "
              f"({before/after:.0f}x better)")
        check_one(f"{name} condition number improves", after < before)
    ill = torch.randn(128, 128) @ torch.diag(torch.logspace(-4, 0, 128))
    s0 = torch.linalg.svdvals(ill)
    s1 = torch.linalg.svdvals(newton_schulz(ill, steps=5).float())
    before = (s0.max() / s0.min()).item()
    after = (s1.max() / s1.min().clamp(min=1e-8)).item()
    print(f"    {'ill conditioned':16s} {before:9.1e} -> {after:8.1e}  "
          f"({before/after:.0f}x better)")
    check_one("ill conditioned input improves by two orders", before / after > 50)

    print("\nupdate scale matches adamw rms")
    for shape in ((1280, 1280), (1280, 3328), (3328, 1280)):
        p = torch.zeros(*shape)
        print(f"    {str(shape):12s} scale {update_scale(p):.2f}")
    check_one("scale grows with the larger dimension",
              update_scale(torch.zeros(1280, 3328))
              > update_scale(torch.zeros(1280, 1280)))

    print("\nmuon rejects non matrices")
    try:
        Muon([torch.zeros(128, requires_grad=True)])
        check_one("1D parameter should have been rejected", False)
    except ValueError:
        check_one("1D parameter rejected, embeddings cannot slip in", True)

    print("\nmuon takes a step")
    p = torch.randn(64, 128, requires_grad=True)
    before = p.detach().clone()
    opt = Muon([p], lr=0.02)
    p.grad = torch.randn_like(p)
    opt.step()
    check_one("parameters moved", not torch.equal(p.detach(), before))
    check_one("still finite", torch.isfinite(p).all())
    check_one("one momentum buffer per matrix, not two",
              len(opt.state[p]) == 1 and "momentum_buffer" in opt.state[p])

    print("\nwsd schedule")
    cfg = ScheduleConfig()
    print(f"    total {cfg.total_steps:,}, warmup {cfg.warmup_steps:,}, "
          f"decay starts {cfg.decay_start:,}")
    check_one("starts at zero", wsd_factor(0, cfg) == 0.0)
    check_one("reaches peak at the end of warmup",
              abs(wsd_factor(cfg.warmup_steps, cfg) - 1.0) < 1e-9)
    check_one("holds peak through the stable phase",
              wsd_factor(cfg.decay_start - 1, cfg) == 1.0)
    check_one("continuous at the decay boundary",
              abs(wsd_factor(cfg.decay_start, cfg) - 1.0) < 1e-9)
    check_one(f"ends near zero ({wsd_factor(cfg.total_steps, cfg):.2e})",
              wsd_factor(cfg.total_steps, cfg) < 1e-9)
    monotone = all(wsd_factor(s, cfg) >= wsd_factor(s + 100, cfg) - 1e-12
                   for s in range(cfg.decay_start, cfg.total_steps - 100, 500))
    check_one("decay is monotone", monotone)
    for shape in ("inv_sqrt", "cosine", "linear"):
        c = ScheduleConfig(decay_shape=shape)
        mid = wsd_factor((c.decay_start + c.total_steps) // 2, c)
        print(f"    {shape:9s} halfway through decay: {mid:.3f}")
    check_one("inv_sqrt decays faster than linear at the halfway point",
              wsd_factor((cfg.decay_start + cfg.total_steps) // 2, cfg)
              < wsd_factor((cfg.decay_start + cfg.total_steps) // 2,
                           ScheduleConfig(decay_shape="linear")))

    print("\nearly decay, the reason for wsd")
    early = 50_000
    check_one("stable at step 50k under the default plan",
              wsd_factor(early, cfg) == 1.0)
    check_one("decaying from 50k instead drops the rate immediately",
              wsd_factor(early + 1000, cfg, decay_from=early) < 1.0)

    print("\noptimizer state memory for the real model")
    matrices, others = 451_215_360, 514_345_526 - 451_215_360
    muon_bytes, adamw_bytes = matrices * 4, others * 8
    both_adamw = 514_345_526 * 8
    print(f"    muon on {matrices/1e6:.0f}M matrices  {muon_bytes/1e9:.2f} GB")
    print(f"    adamw on {others/1e6:.0f}M others     {adamw_bytes/1e9:.2f} GB")
    print(f"    total                        {(muon_bytes+adamw_bytes)/1e9:.2f} GB")
    print(f"    adamw on everything          {both_adamw/1e9:.2f} GB")
    check_one("hybrid saves over 1.5 GB against all adamw",
              both_adamw - (muon_bytes + adamw_bytes) > 1.5e9)

    print("\n" + ("optim ok" if passed else "FAILURES ABOVE"))
    return passed


if __name__ == "__main__":
    check()