"""One training session: train for roughly N hours, then stop clean.

The 40B token run is ~76,000 optimizer steps, which is far more than one
sitting.  So the run is split into sessions: each resumes from the last
checkpoint, trains until its time budget is spent, saves, and exits.  Run it
again tomorrow and it picks up exactly where it left off.

The budget is deliberately soft.  It is checked *between* optimizer steps,
never inside one, so a session always ends on a clean boundary with a complete
checkpoint and a consistent optimizer state.  A step that starts at 15h59m
finishes normally.  Ctrl+C behaves the same way: it asks the loop to stop,
finishes the step in flight, saves, and exits -- so interrupting a session
never costs more than one step.

Autotuning is off by default.  Tuned kernels cannot be cached to disk on this
machine (tvm wants a target triple from a host compiler that is not
installed), so leaving it on would re-tune from scratch on every start -- over
an hour before the first token is trained, every session.  The defaults in the
kernel modules are used instead.  Pass --autotune to spend the time.

    python -m mimans.training.session --hours 16
    python -m mimans.training.session --hours 16 --synthetic   # no shards yet
    python -m mimans.training.session --status                 # where am I?
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mimans import paths
from mimans.data import (DEFAULT_STAGES, TOKEN_DTYPE, Loader, Shard, Source,
                         write_shard)
from mimans.model.model import CodeModel, ModelConfig
from mimans.training.train import TrainConfig, Trainer

SOURCES = ("code", "text", "math")


def find_checkpoints(out_dir: Path):
    """Newest last.  step*.pt is what Trainer.save writes."""
    return sorted(Path(out_dir).glob("step*.pt"))


def open_sources(shard_root: Path):
    """One Source per mixture name, over whatever shards prepare_data wrote."""
    sources = {}
    for name in SOURCES:
        folder = shard_root / name
        shards = sorted(folder.glob("*.bin")) if folder.is_dir() else []
        if not shards:
            continue
        sources[name] = Source(name, [Shard(p) for p in shards])
    return sources


def synthetic_sources(root: Path, vocab: int, seq_len: int, rows=256):
    """Random tokens in the real shard format, so the loop can be exercised
    before prepare_data.py exists.  Trains nothing worth keeping."""
    rng = np.random.default_rng(0)
    sources = {}
    for name in SOURCES:
        tokens = rng.integers(0, vocab, size=(rows, seq_len + 1),
                              dtype=np.int64).astype(TOKEN_DTYPE)
        path = root / f"{name}_synthetic.bin"
        write_shard(path, tokens)
        sources[name] = Source(name, [Shard(path)])
    return sources


def human(seconds: float):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def status(cfg: TrainConfig):
    out = Path(cfg.out_dir)
    saved = find_checkpoints(out)
    print(f"checkpoints in {out}")
    if not saved:
        print("  none yet -- the next session starts from scratch")
        return True
    for p in saved:
        print(f"  {p.name:24s} {p.stat().st_size/1e9:5.2f} GB  "
              f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime))}")
    state = torch.load(saved[-1], map_location="cpu", weights_only=False)
    step, seen = state["step"], state["tokens_seen"]
    total = cfg.total_steps
    print(f"\nresuming from {saved[-1].name}")
    print(f"  step {step:,} of {total:,}  ({step/total:.1%})")
    print(f"  {seen/1e9:.2f}B tokens of {cfg.total_tokens/1e9:.0f}B")
    return True


def recent_throughput(out_dir: Path, samples=40):
    """Tokens per second from the tail of the session log, so the plan below
    is based on what this machine actually did rather than on the planning
    assumption.  None until a session has logged something."""
    log = Path(out_dir) / "session.jsonl"
    if not log.exists():
        return None
    rates = []
    try:
        for line in log.read_text().splitlines()[-samples:]:
            record = json.loads(line)
            if record.get("tokens_per_sec"):
                rates.append(record["tokens_per_sec"])
    except (OSError, json.JSONDecodeError):
        return None
    return sum(rates) / len(rates) if rates else None


def ask_number(question, default, low, high):
    """Repeat the question until the answer is a number in range.  Enter takes
    the default; Ctrl+C or a closed stdin leaves without starting anything."""
    while True:
        try:
            raw = input(f"{question} [{default:g}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nnothing started")
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number")
            continue
        if not low <= value <= high:
            print(f"  needs to be between {low:g} and {high:g}")
            continue
        return value


def confirm(question):
    try:
        return input(f"{question} [Y/n]: ").strip().lower() in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nnothing started")


def plan(cfg, hours, checkpoint_minutes, out_dir, step_now):
    """What the answers just chosen actually mean, before anything starts."""
    measured = recent_throughput(out_dir)
    tps = measured or 18_000
    source = "measured last session" if measured else "assumed, nothing measured yet"
    tokens = hours * 3600 * tps
    steps = int(tokens // cfg.global_batch_tokens)
    steps = min(steps, cfg.total_steps - step_now)
    saves = int(hours * 60 // checkpoint_minutes)
    left = (cfg.total_steps - step_now - steps) * cfg.global_batch_tokens

    print(f"\n  this session")
    print(f"    {hours:g} hours, checkpoint every {checkpoint_minutes:g} min "
          f"({saves} checkpoints)")
    print(f"    at {tps:,.0f} tokens/s ({source})")
    print(f"    about {steps:,} steps, {steps*cfg.global_batch_tokens/1e9:.2f}B tokens")
    print(f"    step {step_now:,} -> {step_now+steps:,} of {cfg.total_steps:,}")
    if left > 0:
        print(f"    {left/1e9:.1f}B tokens would remain, "
              f"about {left/tps/3600/hours:.0f} more sessions")
    else:
        print(f"    this finishes the run")


def run_session(args):
    if not torch.cuda.is_available():
        raise SystemExit("training needs a gpu")

    if not args.autotune:
        # Import the kernel modules and pin them to their default tiles before
        # any kernel is built.  See the module docstring for why.
        from mimans.model import attention_tilelang, lm_head_tilelang, swiglu_tilelang
        attention_tilelang.AUTOTUNE = False
        lm_head_tilelang.AUTOTUNE = False
        swiglu_tilelang.AUTOTUNE = False

    model_cfg = ModelConfig()
    cfg = TrainConfig(micro_batch=args.micro_batch, seq_len=args.seq_len)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.status:
        return status(cfg)

    if args.synthetic:
        root = paths.TMP / "session_synthetic"
        root.mkdir(parents=True, exist_ok=True)
        sources = synthetic_sources(root, model_cfg.vocab, cfg.seq_len)
        print("SYNTHETIC DATA -- exercising the loop, not training anything real")
    else:
        sources = open_sources(paths.SHARDS)
        if not sources:
            raise SystemExit(
                f"no shards under {paths.SHARDS}.\n"
                f"expected {paths.SHARDS}/{{code,text,math}}/*.bin -- run the "
                f"data preparation step first, or pass --synthetic to test the "
                f"loop without them.")
    print(f"sources: {', '.join(f'{k} ({len(v):,} sequences)' for k, v in sources.items())}")

    loader = Loader(sources, batch_size=cfg.micro_batch, seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    model = CodeModel(model_cfg).cuda()
    trainer = Trainer(model, loader, cfg, device="cuda")

    saved = find_checkpoints(out)
    if saved and not args.fresh:
        print(f"resuming from {saved[-1].name}")
        trainer.load(saved[-1])
    elif saved and args.fresh:
        print(f"--fresh: ignoring {len(saved)} existing checkpoint(s), starting at step 0")
    else:
        print("no checkpoint found, starting from scratch")

    started_step, started_tokens = trainer.step, trainer.tokens_seen
    print(f"\nstep {trainer.step:,} of {cfg.total_steps:,}, "
          f"{trainer.tokens_seen/1e9:.2f}B tokens seen")
    print(f"budget {args.hours:g}h, checkpoint every {args.checkpoint_minutes:g}min, "
          f"log every {cfg.log_every} steps\n")

    deadline = time.time() + args.hours * 3600
    last_save = time.time()
    log_path = out / "session.jsonl"
    stopping = {"now": False}

    def on_interrupt(signum, frame):
        if stopping["now"]:
            raise KeyboardInterrupt("second interrupt, exiting immediately")
        stopping["now"] = True
        print("\ninterrupt: finishing this step, saving, then exiting", flush=True)

    signal.signal(signal.SIGINT, on_interrupt)

    session_start = time.time()
    window = time.time()
    reason = "budget spent"
    try:
        while trainer.step < cfg.total_steps:
            metrics = trainer.train_step()

            if metrics["stage_switch"]:
                print(f"[stage] {metrics['stage_switch']}", flush=True)

            now = time.time()
            if trainer.step % cfg.log_every == 0:
                elapsed = now - window
                window = now
                tps = cfg.global_batch_tokens * cfg.log_every / max(elapsed, 1e-9)
                dense, scores = model.flops_per_token(cfg.seq_len)
                mfu = (dense + scores) * tps / (cfg.peak_tflops * 1e12)
                remaining_steps = cfg.total_steps - trainer.step
                record = {
                    "step": trainer.step,
                    "tokens": trainer.tokens_seen,
                    "loss": round(metrics["loss"], 5),
                    "grad_norm": round(metrics["grad_norm"], 4),
                    "lr_factor": round(metrics["lr_factor"], 5),
                    "tokens_per_sec": int(tps),
                    "mfu": round(mfu, 4),
                    "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "session_elapsed": round(now - session_start, 1),
                }
                print(json.dumps(record), flush=True)
                with open(log_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                eta = remaining_steps * cfg.global_batch_tokens / max(tps, 1e-9)
                print(f"    {human(deadline - now)} left this session, "
                      f"{human(eta)} of training left overall", flush=True)

            if now - last_save >= args.checkpoint_minutes * 60:
                path = trainer.save()
                last_save = time.time()
                print(f"    checkpoint {path.name}", flush=True)

            if stopping["now"]:
                reason = "interrupted"
                break
            if now >= deadline:
                reason = "budget spent"
                break
    except KeyboardInterrupt:
        reason = "hard interrupt, checkpoint may be missing"
        raise
    finally:
        # Always land on a complete checkpoint, whatever ended the session.
        if trainer.step > started_step:
            path = trainer.save()
            print(f"\nfinal checkpoint {path.name}")

    wall = time.time() - session_start
    steps_done = trainer.step - started_step
    tokens_done = trainer.tokens_seen - started_tokens
    print(f"\nsession over: {reason}")
    print(f"  {human(wall)} wall, {steps_done:,} steps, {tokens_done/1e9:.3f}B tokens")
    if steps_done:
        rate = tokens_done / wall
        left = (cfg.total_steps - trainer.step) * cfg.global_batch_tokens
        print(f"  {rate:,.0f} tokens/s average")
        print(f"  {trainer.step:,}/{cfg.total_steps:,} steps done ({trainer.step/cfg.total_steps:.1%})")
        print(f"  {left/1e9:.1f}B tokens left, about {left/rate/3600:.1f}h "
              f"= {left/rate/3600/args.hours:.1f} more sessions of {args.hours:g}h")
    print(f"\nrun the same command again to continue from step {trainer.step:,}")
    return True


def main():
    parser = argparse.ArgumentParser()
    defaults = TrainConfig()
    # Left as None so that "not given" is distinguishable from "given the
    # default", which is what decides whether to ask.
    parser.add_argument("--hours", type=float, default=None,
                        help=f"soft budget, checked between steps "
                             f"(asked for if omitted, default {defaults.session_hours:g})")
    parser.add_argument("--checkpoint-minutes", type=float, default=None,
                        help=f"how much progress a crash may cost "
                             f"(asked for if omitted, default {defaults.save_every_minutes:g})")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="take the defaults without asking, for unattended runs")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--synthetic", action="store_true",
                        help="random tokens, for exercising the loop before shards exist")
    parser.add_argument("--autotune", action="store_true",
                        help="re-tune kernels; over an hour on this box, every start")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoints and start at step 0")
    parser.add_argument("--status", action="store_true",
                        help="print where the run is up to and exit")
    args = parser.parse_args()

    # Ask only when there is somebody there to answer.  Piped or redirected
    # stdin, or -y, falls back to the configured defaults so that an unattended
    # run never blocks on a prompt nobody will ever see.
    interactive = sys.stdin.isatty() and not args.yes and not args.status
    if interactive:
        cfg = TrainConfig()
        saved = find_checkpoints(Path(cfg.out_dir))
        step_now, tokens_now = 0, 0
        if saved:
            state = torch.load(saved[-1], map_location="cpu", weights_only=False)
            step_now, tokens_now = state["step"], state["tokens_seen"]

        print("mimans training session")
        if saved:
            print(f"  resuming {saved[-1].name}: step {step_now:,} of "
                  f"{cfg.total_steps:,} ({step_now/cfg.total_steps:.1%}), "
                  f"{tokens_now/1e9:.2f}B of {cfg.total_tokens/1e9:.0f}B tokens")
        else:
            print("  no checkpoint yet, this starts the run from scratch")
        print()

        if args.hours is None:
            args.hours = ask_number("how many hours should this session run?",
                                    defaults.session_hours, 0.01, 168)
        if args.checkpoint_minutes is None:
            args.checkpoint_minutes = ask_number(
                "checkpoint every how many minutes?",
                defaults.save_every_minutes, 1, 24 * 60)

        plan(cfg, args.hours, args.checkpoint_minutes, cfg.out_dir, step_now)
        if not confirm("\nstart?"):
            raise SystemExit("nothing started")
        print()

    if args.hours is None:
        args.hours = defaults.session_hours
    if args.checkpoint_minutes is None:
        args.checkpoint_minutes = defaults.save_every_minutes
    return run_session(args)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
