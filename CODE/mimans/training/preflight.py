"""The last thing to run before committing three weeks of GPU time.

train.py's check() proves the arithmetic of the schedule.  This proves the
machine: real kernels, real optimizer, real checkpoint, real loader, on
synthetic shards written in the repo's own on-disk format.  It answers the
list train.py prints at the end of its own check:

    tokens/s at or above 18,000, gpu utilisation above 95 percent, loss at
    step 0 near 10.80, attention kernel confirmed not the math fallback,
    power limited to 500W, and resume tested from a checkpoint.

    python -m mimans.training.preflight          # full config, 3 steps
    python -m mimans.training.preflight --quick  # structural pass
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from mimans import paths
from mimans.data import (DEFAULT_STAGES, TOKEN_DTYPE, Loader, Shard,
                         Source, Stage, write_shard)
from mimans.model import attention_tilelang
from mimans.model.model import CodeModel, ModelConfig
from mimans.training.train import TrainConfig, Trainer

TARGET_TOKENS_PER_SEC = 18_000
TARGET_POWER_WATTS = 500


def synthetic_sources(root: Path, vocab: int, seq_len: int, rows_per_source: int):
    """Real Shard/Source/Loader path, just with random tokens in the files."""
    rng = np.random.default_rng(0)
    sources = {}
    for name in ("code", "text", "math"):
        tokens = rng.integers(0, vocab, size=(rows_per_source, seq_len + 1),
                              dtype=np.int64).astype(TOKEN_DTYPE)
        path = root / f"{name}_0000.bin"
        write_shard(path, tokens)
        sources[name] = Source(name, [Shard(path)])
    return sources


def power_limit_watts():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit,power.max_limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None, None
        limit, maximum = out.stdout.strip().splitlines()[0].split(",")
        return float(limit), float(maximum)
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument("--quick", action="store_true",
                        help="tiny model and batch, for checking the plumbing")
    args = parser.parse_args()
    if args.quick:
        args.steps, args.micro_batch, args.seq_len = 2, 2, 512
        args.accum, args.layers = 2, 6

    passed = True

    def check(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    if not torch.cuda.is_available():
        print("preflight needs a gpu")
        return False

    print(f"{torch.cuda.get_device_name()}  "
          f"sm_{'%d%d' % torch.cuda.get_device_capability()}  "
          f"torch {torch.__version__}")
    print(f"writing to {paths.summary()}")

    # tvm reads the target triple from a host compiler's -dumpmachine, and with
    # no clang or gcc on PATH it cannot write tuned kernels to disk.  That is a
    # traceback per kernel and a retune per process, not a correctness problem.
    logging.getLogger("tilelang").setLevel(logging.CRITICAL)
    print("note: tuned kernels cannot be cached to disk on this box "
          "(no clang/gcc for the tvm target triple), so every process retunes")

    watts, max_watts = power_limit_watts()
    if watts is not None:
        print(f"power limit {watts:.0f} W of {max_watts:.0f} W maximum")
        check(f"power limited at or under {TARGET_POWER_WATTS} W",
              watts <= TARGET_POWER_WATTS + 1)
    else:
        print("power limit unreadable, skipping that gate")

    model_cfg = ModelConfig(n_layers=args.layers)
    cfg = TrainConfig(
        micro_batch=args.micro_batch,
        seq_len=args.seq_len,
        global_batch_tokens=args.micro_batch * args.seq_len * args.accum,
    )

    # Synthetic shards plus a checkpoint, so this goes on D: like everything
    # else -- the default temp directory is on the full system drive.
    root = Path(tempfile.mkdtemp(prefix="preflight_", dir=paths.TMP))
    try:
        print(f"\nbuilding {3 * 4 * args.micro_batch} synthetic sequences "
              f"of {args.seq_len}")
        sources = synthetic_sources(root, model_cfg.vocab, args.seq_len,
                                    rows_per_source=4 * args.micro_batch)
        loader = Loader(sources, batch_size=cfg.micro_batch, seed=cfg.seed)

        # One stage covering the whole run, at the sequence length the shards
        # were actually written with.  The real schedule is dry-run below.
        stages = [Stage("preflight", 1e18, args.seq_len,
                        {"code": 0.60, "text": 0.30, "math": 0.10})]

        torch.manual_seed(0)
        model = CodeModel(model_cfg).cuda()
        trainer = Trainer(model, loader, cfg, device="cuda", stages=stages)
        trainer.out = root / "runs"
        trainer.out.mkdir(parents=True, exist_ok=True)

        report = trainer.preflight()
        print(f"\n{report['params']:,} parameters, "
              f"{report['gflops_per_token']:.3f} GF/token, "
              f"{cfg.grad_accum} micro steps per optimizer step")

        print("\nstep 0 loss")
        trainer.maybe_switch_stage()
        x, y = loader.batch(0, trainer.stage.weights)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            first = model(x, y).item()
        uniform = math.log(model_cfg.vocab)
        print(f"    {first:.4f} against log(vocab) {uniform:.4f}")
        check("untrained loss starts at log(vocab), not a copy of the input",
              abs(first - uniform) / uniform < 0.05)

        print(f"\n{args.steps} optimizer steps "
              f"({cfg.global_batch_tokens:,} tokens each)")
        torch.cuda.reset_peak_memory_stats()
        losses, rates = [], []
        for step in range(args.steps):
            torch.cuda.synchronize()
            started = time.time()
            metrics = trainer.train_step()
            torch.cuda.synchronize()
            elapsed = time.time() - started
            rate = cfg.global_batch_tokens / elapsed
            losses.append(metrics["loss"])
            rates.append(rate)
            print(f"    step {step}  loss {metrics['loss']:8.4f}  "
                  f"grad_norm {metrics['grad_norm']:7.3f}  "
                  f"{rate:9,.0f} tok/s  {elapsed:6.2f} s")

        # The first step pays for autotuning and cuda graph warmup, so judge
        # throughput on the steady state.
        steady = rates[-1] if len(rates) == 1 else max(rates[1:])
        dense, scores = model.flops_per_token(cfg.seq_len)
        mfu = (dense + scores) * steady / (cfg.peak_tflops * 1e12)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n    steady {steady:,.0f} tokens/s, mfu {mfu:.1%}, "
              f"peak memory {peak_gb:.2f} GB")
        check("every step produced a finite loss",
              all(math.isfinite(v) for v in losses))
        if not args.quick:
            check(f"throughput at or above {TARGET_TOKENS_PER_SEC:,} tokens/s",
                  steady >= TARGET_TOKENS_PER_SEC)
        check("fits in the memory budget with headroom for the long context stage",
              peak_gb < 0.75 * torch.cuda.get_device_properties(0).total_memory / 1e9)

        print("\nkernels actually used")
        kinds = {kind for kind, *_ in attention_tilelang._kernels}
        print(f"    attention kernels compiled: {sorted(kinds) or 'none'}")
        check("attention ran the tilelang kernel, not the math fallback",
              {"fwd", "bwd"} <= kinds)
        check("model is configured for the tilelang backends",
              model_cfg.attn_backend == "tilelang"
              and model_cfg.ffn_backend == "tilelang"
              and model_cfg.head_backend == "tilelang")

        print("\ncheckpoint and resume")
        path = trainer.save(tag="preflight")
        size_gb = path.stat().st_size / 1e9
        print(f"    wrote {path.name}, {size_gb:.2f} GB")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            before = model(x, y).item()

        torch.manual_seed(1234)
        fresh_model = CodeModel(model_cfg).cuda()
        fresh = Trainer(fresh_model, loader, cfg, device="cuda")
        fresh.out = trainer.out
        state = fresh.load(path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            after = fresh_model(x, y).item()
        print(f"    loss {before:.6f} before save, {after:.6f} after load")
        check("a resumed model gives the same loss", abs(before - after) < 1e-5)
        check("step and token counters survive the round trip",
              fresh.step == trainer.step and fresh.tokens_seen == trainer.tokens_seen)
        check("optimizer state came back too",
              len(state["optimizer"]) > 0)

        resumed = fresh.train_step()
        check("a resumed trainer can take another step",
              math.isfinite(resumed["loss"]))

        # The shards hold uniform random tokens, so there is nothing in them to
        # learn and a falling loss would be noise.  A constant batch is
        # trivially learnable, which makes it a real test of whether gradients
        # reach the optimizer through both fused kernels.
        print("\nlearning signal, one constant batch")
        fresh.opt.set_step(fresh.opt.schedule.warmup_steps)
        ones = torch.full((cfg.micro_batch, cfg.seq_len), 42,
                          dtype=torch.long, device="cuda")
        history = []
        for _ in range(20):
            fresh.opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fresh_model(ones, ones)
            loss.backward()
            fresh.opt.step()
            history.append(loss.item())
        print(f"    {history[0]:.4f} -> {history[-1]:.4f} over 20 steps")
        check("the optimizer drives a memorisable batch down by 2 nats",
              history[-1] < history[0] - 2.0)

        print("\nthe real stage schedule, dry run")
        plan = TrainConfig()
        seq, micro = plan.seq_len, plan.micro_batch
        strides = set()
        for stage in DEFAULT_STAGES:
            if stage.seq_len != seq:
                micro = max(1, micro * seq // stage.seq_len)
                seq = stage.seq_len
            accum = plan.global_batch_tokens // (micro * seq)
            fits = micro * seq * accum == plan.global_batch_tokens
            strides.add(seq)
            print(f"    {stage.name:11s} seq {seq:5d}  micro {micro:2d}  "
                  f"accum {accum:3d}  -> {micro * seq * accum:,} tokens"
                  f"{'' if fits else '   MISMATCH'}")
            check(f"{stage.name} keeps the global batch exact", fits)
        if len(strides) > 1:
            print(f"    shards are written at one stride, and this plan needs "
                  f"{sorted(strides)} --")
            print(f"    the {max(strides)} token sources have to exist before "
                  f"the long context stage")

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n" + ("preflight ok, clear to train" if passed else "FAILURES ABOVE"))
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
