from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mimans.model.norm import RMSNorm
from mimans.model.attnres import ResidualState
from mimans.model.attention_tilelang import Rotary
from mimans.model.block import BlockConfig, TransformerBlock
from mimans.model.lm_head_tilelang import HeadConfig, LMHead, IGNORE_INDEX


@dataclass
class ModelConfig:
    vocab: int = 49152
    d_model: int = 1280
    n_layers: int = 27
    n_heads: int = 10
    n_kv_heads: int = 2
    head_dim: int = 128
    ffn_hidden: int = 3328
    layers_per_block: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    max_seq_len: int = 8192
    init_std: float = 0.02
    use_attnres: bool = True
    tied: bool = True
    attn_backend: str = "tilelang"
    ffn_backend: str = "tilelang"
    head_backend: str = "tilelang"

    def block_config(self):
        return BlockConfig(
            d_model=self.d_model, n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim, ffn_hidden=self.ffn_hidden,
            n_layers=self.n_layers, layers_per_block=self.layers_per_block,
            norm_eps=self.norm_eps, rope_theta=self.rope_theta,
            max_seq_len=self.max_seq_len, use_attnres=self.use_attnres,
            attn_backend=self.attn_backend, ffn_backend=self.ffn_backend)

    def head_config(self):
        return HeadConfig(d_model=self.d_model, vocab=self.vocab, tied=self.tied)

    @property
    def n_blocks(self):
        return self.n_layers // self.layers_per_block


class CodeModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.d_model)
        self.rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        block_cfg = cfg.block_config()
        self.blocks = nn.ModuleList(
            TransformerBlock(block_cfg, self.rope) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = LMHead(cfg.head_config(), self.embedding)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embedding.weight, std=self.cfg.init_std)
        for block in self.blocks:
            for name, p in block.named_parameters():
                if p.dim() >= 2 and "w_down" not in name and "wo" not in name:
                    nn.init.normal_(p, std=self.cfg.init_std)
            block.zero_init_projections()

    def set_rope_theta(self, theta: float):
        self.rope.set_theta(theta)
        self.cfg.rope_theta = theta

    def hidden_states(self, tokens):
        h = self.embedding(tokens)
        state = ResidualState(running=h)
        for block in self.blocks:
            block(state)
        return self.final_norm(state.running)

    def forward(self, tokens, targets=None):
        hidden = self.hidden_states(tokens)
        if targets is None:
            return self.head(hidden, None)
        return self.head(hidden, targets, backend=self.cfg.head_backend)

    def param_groups(self, weight_decay=0.1):
        matrices, others = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and "embedding" not in name:
                matrices.append(p)
            else:
                others.append(p)
        return [
            {"params": matrices, "weight_decay": weight_decay, "optimizer": "muon"},
            {"params": others, "weight_decay": 0.0, "optimizer": "adamw"},
        ]

    def n_parameters(self, matmul_only=False):
        if not matmul_only:
            return sum(p.numel() for p in self.parameters())
        total = sum(p.numel() for n, p in self.named_parameters() if p.dim() >= 2)
        return total + self.cfg.vocab * self.cfg.d_model - (
            self.cfg.vocab * self.cfg.d_model if not self.cfg.tied else 0)

    def flops_per_token(self, seq_len=2048):
        cfg = self.cfg
        attn = (cfg.d_model * cfg.n_heads * cfg.head_dim
                + 2 * cfg.d_model * cfg.n_kv_heads * cfg.head_dim
                + cfg.n_heads * cfg.head_dim * cfg.d_model)
        ffn = 3 * cfg.d_model * cfg.ffn_hidden
        dense = 6 * ((attn + ffn) * cfg.n_layers + cfg.vocab * cfg.d_model)
        scores = 6 * seq_len * cfg.n_heads * cfg.head_dim * cfg.n_layers
        return dense, scores

    def training_days(self, tokens=40e9, seq_len=2048, peak_tflops=209.5, mfu=0.35,
                      overhead=1.05):
        dense, scores = self.flops_per_token(seq_len)
        per_token = dense + scores
        seconds = tokens * per_token / (peak_tflops * 1e12 * mfu)
        return seconds / 86400 * overhead


def check():
    torch.manual_seed(0)
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    cfg = ModelConfig(attn_backend="math", ffn_backend="math", head_backend="torch")
    model = CodeModel(cfg)

    print("parameter budget")
    total = model.n_parameters()
    body = sum(p.numel() for b in model.blocks for p in b.parameters())
    emb = model.embedding.weight.numel()
    print(f"    embedding (tied) {emb:,}")
    print(f"    27 blocks        {body:,}")
    print(f"    final norm       {model.final_norm.weight.numel():,}")
    print(f"    TOTAL            {total:,}  = {total/1e6:.1f}M")
    check_one("matches the design report 514,345,526", total == 514_345_526)
    check_one("head adds no parameters of its own",
              sum(p.numel() for p in model.head.parameters()) == 0)
    check_one("embedding is 12.2 percent of the model",
              abs(emb / total - 0.1223) < 0.001)

    print("\ntied weights")
    check_one("head matrix is the embedding transposed",
              model.head.matrix().data_ptr() == model.embedding.weight.data_ptr())
    names = [n for n, _ in model.named_parameters() if "embedding" in n]
    check_one(f"embedding appears once in named_parameters {names}", len(names) == 1)

    print("\ncompute")
    dense, scores = model.flops_per_token(2048)
    print(f"    dense {dense/1e9:.3f} GF/token")
    print(f"    attention scores {scores/1e9:.3f} GF/token "
          f"({scores/dense*100:.1f} percent overhead)")
    print(f"    total {(dense+scores)/1e9:.3f} GF/token")
    check_one("attention overhead near 14 percent at seq 2048",
              0.10 < scores / dense < 0.16)
    for seq in (2048, 8192):
        d, s = model.flops_per_token(seq)
        print(f"    seq {seq:5d}: {(d+s)/1e9:.3f} GF/token, "
              f"{s/d*100:5.1f} percent in attention")
    days = model.training_days()
    print(f"    40B tokens at 35 percent MFU: {days:.1f} days")
    check_one("lands in the 20 to 26 day window from the report", 20 < days < 26)

    print("\nforward")
    tokens = torch.randint(0, cfg.vocab, (2, 16))
    logits = model(tokens)
    check_one(f"logits {tuple(logits.shape)}", logits.shape == (2, 16, cfg.vocab))

    print("\nloss of an untrained model")
    targets = torch.randint(0, cfg.vocab, (2, 16))
    loss = model(tokens, targets)
    uniform = math.log(cfg.vocab)
    print(f"    loss {loss.item():.4f}, log(vocab) {uniform:.4f}")
    check_one(f"within 5 percent of log(vocab)",
              abs(loss.item() - uniform) / uniform < 0.05)
    check_one("this is the canary for a broken init", torch.isfinite(loss))

    print("\nidentity at init")
    h = model.hidden_states(tokens)
    emb_normed = model.final_norm(model.embedding(tokens))
    check_one(f"zero init projections pass embeddings through "
              f"({(h - emb_normed).abs().max():.2e})",
              (h - emb_normed).abs().max() < 1e-5)

    print("\ncausality")
    torch.manual_seed(1)
    live = CodeModel(ModelConfig(n_layers=3, attn_backend="math",
                                 ffn_backend="math", head_backend="torch"))
    with torch.no_grad():
        for n, p in live.named_parameters():
            if p.dim() >= 2:
                p.normal_(std=0.02)
    a = torch.randint(0, live.cfg.vocab, (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % live.cfg.vocab
    drift = (live(a)[:, :-1] - live(b)[:, :-1]).abs().max().item()
    check_one(f"changing the last token leaves earlier logits alone ({drift:.2e})",
              drift < 1e-4)

    print("\nparameter groups")
    groups = live.param_groups()
    muon = sum(p.numel() for p in groups[0]["params"])
    adamw = sum(p.numel() for p in groups[1]["params"])
    print(f"    muon  {muon:,} in {len(groups[0]['params'])} tensors, "
          f"decay {groups[0]['weight_decay']}")
    print(f"    adamw {adamw:,} in {len(groups[1]['params'])} tensors, "
          f"decay {groups[1]['weight_decay']}")
    check_one("groups cover every parameter", muon + adamw == live.n_parameters())
    check_one("no weight decay outside hidden matrices",
              groups[1]["weight_decay"] == 0.0)
    check_one("embedding is not in the muon group",
              all(p.data_ptr() != live.embedding.weight.data_ptr()
                  for p in groups[0]["params"]))

    print("\ngradients")
    small = CodeModel(ModelConfig(n_layers=3, attn_backend="math",
                                  ffn_backend="math", head_backend="torch"))
    t = torch.randint(0, small.cfg.vocab, (2, 8))
    y = torch.randint(0, small.cfg.vocab, (2, 8))
    small(t, y).backward()
    missing = [n for n, p in small.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    check_one(f"every parameter receives a finite gradient"
              f"{'' if not missing else ' -> ' + str(missing[:3])}", not missing)
    check_one("embedding gets gradient from both the lookup and the head",
              small.embedding.weight.grad.abs().sum() > 0)

    print("\nignore index")
    masked = torch.full_like(y, IGNORE_INDEX)
    check_one("all masked targets give zero loss",
              small(t, masked).item() == 0.0)

    print("\nstage 3 rope switch")
    model.set_rope_theta(500_000.0)
    check_one("theta updated on the shared table", model.rope.theta == 500_000.0)
    check_one("config follows", model.cfg.rope_theta == 500_000.0)
    check_one("every block shares one rope module",
              all(b.attn.rope is model.rope for b in model.blocks))

    print("\n" + ("model ok" if passed else "FAILURES ABOVE"))
    return passed


if __name__ == "__main__":
    check()