"""Tune every kernel once, for the shapes the real run actually uses.

Kernels are compiled per shape, so a config that wins at 8x2048 is not
necessarily the one that wins at 2x8192 -- and the run uses both, because the
long context stage at 36B tokens switches the sequence length and halves the
micro batch to keep the global batch fixed.  Both are tuned here so that stage
does not silently fall back to tiles picked for the wrong shape.

Results go to the store in mimans.model.tuned, written after each kernel, so
an interrupted run keeps everything it had already measured and a re-run skips
what is already known.

    python -m mimans.training.tune              # both stages
    python -m mimans.training.tune --stage 0    # just the 2048 shape
    python -m mimans.training.tune --force      # re-measure even if known
"""
from __future__ import annotations

import argparse
import time

import torch

from mimans.model import attention_tilelang, lm_head_tilelang, swiglu_tilelang
from mimans.model import tuned
from mimans.model.model import ModelConfig
from mimans.training.train import TrainConfig


def stage_shapes(cfg: TrainConfig, model_cfg: ModelConfig):
    """(micro_batch, seq_len) for each distinct context length in the run,
    holding the global batch constant the way Trainer.maybe_switch_stage does.
    """
    from mimans.data import DEFAULT_STAGES

    seen, out = set(), []
    micro, seq = cfg.micro_batch, cfg.seq_len
    for stage in DEFAULT_STAGES:
        if stage.seq_len != seq:
            micro = max(1, micro * seq // stage.seq_len)
            seq = stage.seq_len
        if seq in seen:
            continue
        seen.add(seq)
        out.append((stage.name, micro, seq))
    return out


def tune_all(micro_batch, seq_len, model_cfg, force=False):
    tokens = micro_batch * seq_len
    heads, kv_heads = model_cfg.n_heads, model_cfg.n_kv_heads
    dim, groups = model_cfg.head_dim, heads // kv_heads

    jobs = [
        ("attention_fwd", (micro_batch, heads, seq_len, dim, True, groups),
         lambda s: attention_tilelang.get_kernel("fwd", *s)),
        ("attention_bwd", (micro_batch, heads, seq_len, dim, True, groups),
         lambda s: attention_tilelang.get_kernel("bwd", *s)),
        ("swiglu_split", (tokens, model_cfg.d_model, model_cfg.ffn_hidden),
         lambda s: swiglu_tilelang.get_kernel("split", *s)),
        ("swiglu_grad", (tokens, model_cfg.ffn_hidden),
         lambda s: swiglu_tilelang.get_kernel("grad", *s)),
        ("lm_head_lse", (tokens, model_cfg.d_model, model_cfg.vocab),
         lambda s: lm_head_tilelang.get_kernel(s)),
    ]

    for name, shape, build in jobs:
        if not force and tuned.lookup(name, shape) is not None:
            print(f"  {name:14s} already known, skipping")
            continue
        print(f"  {name:14s} tuning {shape} ...", flush=True)
        started = time.time()
        try:
            kernel = build(shape)
            config = tuned.config_of(kernel)
            if config:
                # get_kernel records it, but record again under --force so a
                # re-measured winner replaces the old entry.
                tuned.record(name, shape, config)
                terse = " ".join(f"{k}={v}" for k, v in sorted(config.items()))
                print(f"  {name:14s} {time.time()-started:6.0f}s  {terse}")
            else:
                print(f"  {name:14s} {time.time()-started:6.0f}s  "
                      f"built, but reported no config (default tiles in use)")
        except Exception as exc:
            print(f"  {name:14s} FAILED after {time.time()-started:.0f}s: "
                  f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=None,
                        help="tune only one context length, by index")
    parser.add_argument("--force", action="store_true",
                        help="re-measure kernels that are already known")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("tuning needs the gpu it will run on")

    # This is the one place autotuning is wanted.
    attention_tilelang.AUTOTUNE = True
    swiglu_tilelang.AUTOTUNE = True
    lm_head_tilelang.AUTOTUNE = True

    model_cfg = ModelConfig()
    cfg = TrainConfig()
    shapes = stage_shapes(cfg, model_cfg)
    if args.stage is not None:
        shapes = [shapes[args.stage]]

    print(f"tuning on {tuned.device_key()}")
    print(f"store: {tuned.STORE}\n")
    for name, micro, seq in shapes:
        print(f"{name}: micro_batch {micro} x seq {seq} "
              f"= {micro*seq:,} tokens per micro step")
        tune_all(micro, seq, model_cfg, force=args.force)
        print()

    print(tuned.summary())
    print("\nsessions will now use these and skip tuning entirely")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
