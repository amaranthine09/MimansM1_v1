"""Compare weight initialisation schemes on the real model and real data.

The question this answers: the stack zero-inits its two output projections, so
at step 0 only those projections and the embedding get any gradient at all --
every wq, wk, wv, w_gate, w_up, norm and gate sees exactly zero.  That is the
ReZero/Fixup behaviour, and it is defensible, but GPT-2 and most production
LLMs instead scale those projections by 1/sqrt(2*n_layers) so that everything
gets signal from the first step.

Rather than argue it, train each scheme on identical data with an identical
seed and read the loss.

    python -m mimans.model.init_study --steps 300
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mimans import paths
from mimans.data import Loader, Shard, Source
from mimans.model.model import CodeModel, ModelConfig
from mimans.training.optim import HybridOptimizer, OptimConfig, ScheduleConfig

SCHEMES = ("zero", "gpt2", "fanin_zero", "fanin_gpt2", "orthogonal")


def fan_in_of(name: str, p: torch.Tensor):
    """nn.Linear stores (out, in); the ffn holds raw parameters used as x @ W,
    so their fan in is the first dimension instead."""
    if name.endswith(("w_gate", "w_up", "w_down")):
        return p.shape[0]
    return p.shape[1]


def is_output_projection(name: str):
    return name.endswith(("wo.weight", "w_down"))


@torch.no_grad()
def apply_scheme(model: CodeModel, scheme: str, std: float = 0.02):
    """Re-initialise every block matrix under one scheme.  Embedding, norms and
    gates are left exactly as the model already sets them, so the comparison
    isolates the block matrices."""
    n_layers = model.cfg.n_layers
    residual_scale = 1.0 / math.sqrt(2 * n_layers)

    for block in model.blocks:
        for name, p in block.named_parameters():
            if p.dim() < 2:
                continue
            fan_in = fan_in_of(name, p)
            base = std if scheme in ("zero", "gpt2") else fan_in ** -0.5

            if is_output_projection(name):
                if scheme in ("zero", "fanin_zero", "orthogonal"):
                    p.zero_()
                else:                       # gpt2, fanin_gpt2
                    p.normal_(0, base * residual_scale)
            elif scheme == "orthogonal":
                # Orthogonal rows/columns give every singular value the same
                # size, which is the condition Muon's update already assumes.
                flat = p.reshape(p.shape[0], -1)
                torch.nn.init.orthogonal_(flat, gain=1.0)
                p.copy_(flat.reshape(p.shape))
            else:
                p.normal_(0, base)
    return model


def grad_health(model: CodeModel, x, y):
    """How many block parameters get a nonzero gradient on the very first
    backward pass, before any weight has moved."""
    model.zero_grad(set_to_none=True)
    model(x, y).backward()
    live = dead = 0
    for block in model.blocks:
        for _, p in block.named_parameters():
            if p.grad is None or p.grad.norm().item() == 0.0:
                dead += 1
            else:
                live += 1
    model.zero_grad(set_to_none=True)
    return live, dead


def open_shards(seq_len):
    sources = {}
    for name in ("code", "text", "math"):
        folder = paths.SHARDS / name
        shards = sorted(folder.glob("*.bin")) if folder.is_dir() else []
        if shards:
            sources[name] = Source(name, [Shard(p) for p in shards])
    if not sources:
        raise SystemExit(f"no shards in {paths.SHARDS}; run prepare_data first")
    return sources


def run_scheme(scheme, args, sources, weights):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = ModelConfig(n_layers=args.layers)
    model = CodeModel(cfg)
    apply_scheme(model, scheme)
    model = model.cuda()

    # Warmup is shortened in proportion to the run: with the real 2,000 step
    # warmup a 300 step probe would spend the whole time at a near zero
    # learning rate and every scheme would look identical.
    schedule = ScheduleConfig(total_steps=args.steps,
                              warmup_steps=max(1, args.steps // 10),
                              decay_fraction=0.1)
    opt = HybridOptimizer(model, OptimConfig(), schedule)

    loader = Loader(sources, batch_size=args.micro_batch, seed=args.seed)
    x0, y0 = loader.batch(0, weights)
    live, dead = grad_health(model,
                             torch.from_numpy(x0).cuda(),
                             torch.from_numpy(y0).cuda())

    losses, started = [], time.time()
    for step in range(args.steps):
        opt.set_step(step)
        opt.zero_grad()
        x, y = loader.batch(step, weights)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            print(f"    diverged at step {step}")
            break

    elapsed = time.time() - started
    tail = losses[-args.tail:] if len(losses) >= args.tail else losses
    result = {
        "scheme": scheme,
        "first_loss": losses[0],
        "final_loss": sum(tail) / len(tail),
        "best_loss": min(losses),
        "live_grads": live,
        "dead_grads": dead,
        "steps": len(losses),
        "seconds": round(elapsed, 1),
        "losses": [round(v, 4) for v in losses],
    }
    del model, opt
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tail", type=int, default=25,
                        help="average this many final steps for the verdict")
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--out", type=Path,
                        default=paths.RUNS / "init_study.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a gpu")

    from mimans.model import attention_tilelang, lm_head_tilelang, swiglu_tilelang
    attention_tilelang.AUTOTUNE = False
    lm_head_tilelang.AUTOTUNE = False
    swiglu_tilelang.AUTOTUNE = False

    weights = {"code": 0.6, "text": 0.3, "math": 0.1}
    sources = open_shards(2048)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    print(f"{args.layers} layers, {args.steps} steps, micro batch "
          f"{args.micro_batch} x 2048, seed {args.seed}")
    print(f"log(vocab) = {math.log(ModelConfig().vocab):.4f}\n")

    results = []
    for scheme in schemes:
        print(f"  {scheme} ...", flush=True)
        r = run_scheme(scheme, args, sources, weights)
        results.append(r)
        print(f"  {scheme:12s} first {r['first_loss']:7.4f}  "
              f"final {r['final_loss']:7.4f}  best {r['best_loss']:7.4f}  "
              f"live grads {r['live_grads']}/{r['live_grads']+r['dead_grads']}  "
              f"{r['seconds']:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n{'scheme':14s} {'first':>8s} {'final':>8s} {'best':>8s} {'live grads':>11s}")
    for r in sorted(results, key=lambda r: r["final_loss"]):
        print(f"{r['scheme']:14s} {r['first_loss']:8.4f} {r['final_loss']:8.4f} "
              f"{r['best_loss']:8.4f} {r['live_grads']:6d}/"
              f"{r['live_grads']+r['dead_grads']:<4d}")
    best = min(results, key=lambda r: r["final_loss"])
    print(f"\nlowest final loss: {best['scheme']}")
    print(f"written to {args.out}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
