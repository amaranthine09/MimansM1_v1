from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mimans.model.norm import RMSNorm


@dataclass
class ResidualState:
    running: torch.Tensor
    snapshots: list = field(default_factory=list)
    steps: int = 0

    def entries(self):
        return self.snapshots + [self.running]

    def advance(self, delta, block_sublayers):
        self.running = self.running + delta
        self.steps += 1
        if self.steps % block_sublayers == 0:
            self.snapshots.append(self.running)


class AttnResGate(nn.Module):
    """One gate per sublayer. Holds a query vector, a key norm, and a bias."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNorm(d_model, eps)
        self.recency_bias = nn.Parameter(torch.zeros(()))

    def forward(self, entries):
        stack = torch.stack(entries, dim=0)
        keys = self.key_norm(stack)
        logits = torch.einsum("d,nbtd->nbt", self.query.to(keys.dtype), keys)
        last = torch.zeros(stack.shape[0], 1, 1,
                           device=logits.device, dtype=logits.dtype)
        last[-1] = 1.0
        logits = logits + last * self.recency_bias.to(logits.dtype)
        weights = logits.softmax(dim=0)
        return torch.einsum("nbt,nbtd->btd", weights, stack)

    def weights(self, entries):
        stack = torch.stack(entries, dim=0)
        logits = torch.einsum("d,nbtd->nbt", self.query.to(stack.dtype),
                              self.key_norm(stack))
        last = torch.zeros(stack.shape[0], 1, 1,
                           device=logits.device, dtype=logits.dtype)
        last[-1] = 1.0
        return (logits + last * self.recency_bias.to(logits.dtype)).softmax(dim=0)


def gate_parameters(n_layers, d_model, sublayers_per_layer=2):
    return n_layers * sublayers_per_layer * (2 * d_model + 1)


def check():
    torch.manual_seed(0)
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    d_model, layers, per_block = 1280, 27, 3
    batch, seq = 2, 16
    gate = AttnResGate(d_model)

    print("shape and cost")
    state = ResidualState(running=torch.randn(batch, seq, d_model))
    for _ in range(4):
        state.snapshots.append(torch.randn(batch, seq, d_model))
    out = gate(state.entries())
    check_one(f"output {tuple(out.shape)}", out.shape == (batch, seq, d_model))
    n = sum(p.numel() for p in gate.parameters())
    check_one(f"gate parameters {n} == 2*d_model + 1", n == 2 * d_model + 1)
    total = gate_parameters(layers, d_model)
    print(f"    {layers} layers x 2 sublayers -> {total:,} parameters")
    check_one("under 0.03 percent of a 514M model", total / 514_345_526 < 0.0003)

    print("\nzero init behaviour")
    entries = state.entries()
    weights = gate.weights(entries)
    check_one(f"uniform at init ({weights.std():.2e})", weights.std() < 1e-6)
    check_one(f"weights sum to one ({(weights.sum(0) - 1).abs().max():.2e})",
              (weights.sum(0) - 1).abs().max() < 1e-5)
    mean = torch.stack(entries, 0).mean(0)
    check_one(f"uniform gives the MEAN not the SUM ({(out - mean).abs().max():.2e})",
              (out - mean).abs().max() < 1e-4)
    running_sum = torch.stack(entries, 0).sum(0)
    check_one("and the mean is not the sum, so init is not a no-op",
              (out - running_sum).abs().max() > 1e-2)

    print("\nrecency bias unit test")
    with torch.no_grad():
        gate.recency_bias.fill_(1e4)
    collapsed = gate(entries)
    check_one(f"softmax collapses onto the running state "
              f"({(collapsed - state.running).abs().max():.2e})",
              (collapsed - state.running).abs().max() < 1e-6)
    check_one("this reproduces a plain PreNorm residual exactly",
              torch.equal(collapsed, state.running))
    with torch.no_grad():
        gate.recency_bias.zero_()

    print("\nblock bookkeeping")
    state = ResidualState(running=torch.zeros(batch, seq, d_model))
    for _ in range(layers * 2):
        state.advance(torch.ones(batch, seq, d_model) * 0.01, per_block * 2)
    expected_blocks = (layers * 2) // (per_block * 2)
    check_one(f"{layers} layers, {per_block} per block -> {len(state.snapshots)} snapshots",
              len(state.snapshots) == expected_blocks == 9)
    check_one("running state is the cumulative sum",
              torch.allclose(state.running,
                             torch.full((batch, seq, d_model), 0.01 * layers * 2)))
    print(f"    memory for snapshots at 8x2048 bf16: "
          f"{9 * 8 * 2048 * d_model * 2 / 1e6:.0f} MB")

    print("\ngradients")
    gate = AttnResGate(d_model)
    entries = [torch.randn(batch, seq, d_model, requires_grad=True) for _ in range(3)]
    gate(entries).sum().backward()
    check_one("query receives gradient", gate.query.grad is not None
              and torch.isfinite(gate.query.grad).all())
    check_one("recency bias receives gradient",
              gate.recency_bias.grad is not None
              and torch.isfinite(gate.recency_bias.grad).all())
    check_one("all entries receive gradient",
              all(e.grad is not None and torch.isfinite(e.grad).all() for e in entries))

    print("\nkey normalisation")
    gate = AttnResGate(d_model)
    with torch.no_grad():
        gate.query.normal_(std=0.05)
    small = [torch.randn(batch, seq, d_model) for _ in range(3)]
    scaled = [small[0] * 1000, small[1], small[2]]
    w_small = gate.weights(small)
    w_scaled = gate.weights(scaled)
    check_one("rms normed keys stop a large entry dominating by magnitude alone",
              (w_small[0] - w_scaled[0]).abs().max() < 0.05)

    print("\n" + ("attnres ok" if passed else "FAILURES ABOVE"))
    return passed


if __name__ == "__main__":
    check()