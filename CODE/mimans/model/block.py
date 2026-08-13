from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn

from mimans.model.norm import RMSNorm
from mimans.model.attnres import AttnResGate, ResidualState
from mimans.model.attention_tilelang import Attention, AttnConfig, Rotary
from mimans.model.swiglu_tilelang import SwiGLU, FFNConfig


@dataclass
class BlockConfig:
    d_model: int = 1280
    n_heads: int = 10
    n_kv_heads: int = 2
    head_dim: int = 128
    ffn_hidden: int = 3328
    n_layers: int = 27
    layers_per_block: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_seq_len: int = 8192
    use_attnres: bool = True
    attn_backend: str = "tilelang"
    ffn_backend: str = "tilelang"

    def __post_init__(self):
        assert self.n_layers % self.layers_per_block == 0, \
            "layers must divide evenly into blocks"

    @property
    def sublayers_per_block(self):
        return self.layers_per_block * 2

    @property
    def n_blocks(self):
        return self.n_layers // self.layers_per_block

    @property
    def attn_config(self):
        return AttnConfig(d_model=self.d_model, n_heads=self.n_heads,
                          n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
                          rope_theta=self.rope_theta,
                          max_seq_len=self.max_seq_len,
                          norm_eps=self.norm_eps, backend=self.attn_backend)

    @property
    def ffn_config(self):
        return FFNConfig(d_model=self.d_model, ffn_hidden=self.ffn_hidden,
                         backend=self.ffn_backend)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: BlockConfig, rope: Rotary):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg.attn_config, rope)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.ffn_config)
        if cfg.use_attnres:
            self.attn_gate = AttnResGate(cfg.d_model, cfg.norm_eps)
            self.ffn_gate = AttnResGate(cfg.d_model, cfg.norm_eps)
        else:
            self.attn_gate = None
            self.ffn_gate = None
        self.zero_init_projections()

    def zero_init_projections(self):
        with torch.no_grad():
            self.attn.wo.weight.zero_()
            self.ffn.w_down.zero_()

    def combine(self, gate, state):
        if gate is None:
            return state.running
        return gate(state.entries())

    def forward(self, state: ResidualState, backend=None):
        span = self.cfg.sublayers_per_block

        h = self.combine(self.attn_gate, state)
        delta = self.attn(self.attn_norm(h), backend=backend or self.cfg.attn_backend)
        state.advance(delta, span)

        h = self.combine(self.ffn_gate, state)
        delta = self.ffn(self.ffn_norm(h), backend=backend or self.cfg.ffn_backend)
        state.advance(delta, span)

        return state


def block_parameters(cfg: BlockConfig):
    d, h, kv, hd = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    attn = d * h * hd + 2 * d * kv * hd + h * hd * d + 2 * hd
    ffn = 3 * d * cfg.ffn_hidden
    norms = 2 * d
    gates = 2 * (2 * d + 1) if cfg.use_attnres else 0
    return attn + ffn + norms + gates


def check():
    torch.manual_seed(0)
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    cfg = BlockConfig(attn_backend="math", ffn_backend="math")
    rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    block = TransformerBlock(cfg, rope)
    batch, seq = 2, 16

    print("structure")
    print(f"    {cfg.n_layers} layers, {cfg.layers_per_block} per block "
          f"-> {cfg.n_blocks} blocks, {cfg.sublayers_per_block} sublayers each")
    check_one("27 layers give 9 blocks", cfg.n_blocks == 9)
    check_one("6 sublayers per block", cfg.sublayers_per_block == 6)

    n = sum(p.numel() for p in block.parameters())
    expect = block_parameters(cfg)
    print(f"    block parameters {n:,}")
    check_one(f"matches the arithmetic {expect:,}", n == expect)
    check_one("matches the design report 16,719,618", n == 16_719_618)
    print(f"    x{cfg.n_layers} layers = {n * cfg.n_layers / 1e6:.1f}M")

    print("\nzero init means identity at step zero")
    check_one("attention output projection is zero", (block.attn.wo.weight == 0).all())
    check_one("ffn down projection is zero", (block.ffn.w_down == 0).all())
    x = torch.randn(batch, seq, cfg.d_model)
    state = ResidualState(running=x.clone())
    block(state)
    check_one(f"running state unchanged ({(state.running - x).abs().max():.2e})",
              (state.running - x).abs().max() < 1e-6)
    combined = block.combine(block.attn_gate, state)
    check_one(f"gate over identical entries returns them "
              f"({(combined - x).abs().max():.2e})",
              (combined - x).abs().max() < 1e-5)

    print("\nsublayer bookkeeping")
    state = ResidualState(running=torch.randn(batch, seq, cfg.d_model))
    block(state)
    check_one(f"one block advances 2 sublayers ({state.steps})", state.steps == 2)
    state = ResidualState(running=torch.randn(batch, seq, cfg.d_model))
    for _ in range(cfg.layers_per_block):
        block(state)
    check_one(f"{cfg.layers_per_block} layers close one block "
              f"({len(state.snapshots)} snapshot)", len(state.snapshots) == 1)
    state = ResidualState(running=torch.randn(batch, seq, cfg.d_model))
    for _ in range(cfg.n_layers):
        block(state)
    check_one(f"{cfg.n_layers} layers give {len(state.snapshots)} snapshots",
              len(state.snapshots) == cfg.n_blocks)
    check_one(f"entries at the end {len(state.entries())} = blocks + running",
              len(state.entries()) == cfg.n_blocks + 1)

    print("\nattnres reduces to plain prenorm under a large recency bias")
    torch.manual_seed(1)
    gated = TransformerBlock(BlockConfig(use_attnres=True, attn_backend="math",
                                         ffn_backend="math"), rope)
    with torch.no_grad():
        for p in gated.parameters():
            if p.dim() >= 2:
                p.normal_(std=0.02)
        gated.attn_gate.recency_bias.fill_(1e4)
        gated.ffn_gate.recency_bias.fill_(1e4)

    plain = TransformerBlock(BlockConfig(use_attnres=False, attn_backend="math",
                                         ffn_backend="math"), rope)
    with torch.no_grad():
        plain.attn_norm.weight.copy_(gated.attn_norm.weight)
        plain.ffn_norm.weight.copy_(gated.ffn_norm.weight)
        plain.attn.load_state_dict(gated.attn.state_dict())
        plain.ffn.load_state_dict(gated.ffn.state_dict())

    x = torch.randn(batch, seq, cfg.d_model)
    a = ResidualState(running=x.clone())
    b = ResidualState(running=x.clone())
    gated(a)
    plain(b)
    diff = (a.running - b.running).abs().max().item()
    check_one(f"gated and plain agree ({diff:.2e})", diff < 1e-5)
    check_one("so the ablation switch is a true control arm", diff < 1e-5)

    print("\ncausality")
    block = TransformerBlock(BlockConfig(attn_backend="math", ffn_backend="math"), rope)
    with torch.no_grad():
        for p in block.parameters():
            if p.dim() >= 2:
                p.normal_(std=0.02)
    x = torch.randn(batch, seq, cfg.d_model)
    s1 = ResidualState(running=x.clone())
    block(s1)
    perturbed = x.clone()
    perturbed[:, -1] += 10.0
    s2 = ResidualState(running=perturbed)
    block(s2)
    drift = (s1.running[:, :-1] - s2.running[:, :-1]).abs().max().item()
    check_one(f"future token does not reach the past ({drift:.2e})", drift < 1e-4)

    print("\ngradients")
    x = torch.randn(batch, 8, cfg.d_model, requires_grad=True)
    state = ResidualState(running=x)
    block(state)
    state.running.sum().backward()
    check_one("input receives gradient",
              x.grad is not None and torch.isfinite(x.grad).all())
    missing = [n for n, p in block.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    check_one(f"every parameter receives a finite gradient"
              f"{'' if not missing else ' -> ' + str(missing[:3])}", not missing)

    print("\n" + ("block ok" if passed else "FAILURES ABOVE"))
    return passed


if __name__ == "__main__":
    check()