from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from mimans import paths
from mimans.model.model import ModelConfig, CodeModel
from mimans.training.optim import (OptimConfig, ScheduleConfig,
                                   HybridOptimizer, wsd_factor)
from mimans.data import DEFAULT_STAGES, stage_at, Loader


@dataclass
class TrainConfig:
    total_tokens: float = 40e9
    global_batch_tokens: int = 524_288
    micro_batch: int = 8
    seq_len: int = 2048
    seed: int = 0
    peak_tflops: float = 209.5
    log_every: int = 10
    eval_every: int = 2_000
    save_every: int = 2_000
    # A step is ~29 seconds, so save_every=2000 is about 16 hours -- a whole
    # session could finish without ever checkpointing, and a crash at hour 15
    # would cost the lot.  Wall clock saving is what actually bounds the loss.
    save_every_minutes: float = 45.0
    session_hours: float = 16.0
    keep_last: int = 5
    # One checkpoint every this many steps is kept for good, on top of
    # keep_last, so there is a way back after a loss spike.  8,000 steps is
    # about 4B tokens: ten permanent restore points across the run.
    milestone_steps: int = 8_000
    # Measured, not estimated: 4.37 GB per checkpoint and 18s to write.  Five
    # kept plus ten milestones is ~66 GB, which is why this lives on D:.
    out_dir: str = str(paths.RUNS / "smolcoder")
    power_limit_watts: int = 500

    @property
    def grad_accum(self):
        return self.global_batch_tokens // (self.micro_batch * self.seq_len)

    @property
    def total_steps(self):
        return int(self.total_tokens // self.global_batch_tokens)

    def __post_init__(self):
        assert self.global_batch_tokens % (self.micro_batch * self.seq_len) == 0, \
            "global batch must divide evenly into micro batches"


def bits_per_byte(loss_nats: float, tokens: int, num_bytes: int):
    return loss_nats * tokens / (num_bytes * math.log(2))


def model_flops_utilization(flops_per_token, tokens_per_second, peak_tflops):
    return flops_per_token * tokens_per_second / (peak_tflops * 1e12)


class Trainer:
    def __init__(self, model: CodeModel, loader: Loader, cfg: TrainConfig,
                 optim_cfg: OptimConfig = None, device="cuda", stages=None):
        self.model = model
        self.loader = loader
        self.cfg = cfg
        self.device = device
        self.stages = stages or DEFAULT_STAGES
        schedule = ScheduleConfig(total_steps=cfg.total_steps)
        self.opt = HybridOptimizer(model, optim_cfg or OptimConfig(), schedule)
        self.step = 0
        self.tokens_seen = 0
        self.stage = None
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def preflight(self):
        report = {}
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            report["arch"] = f"sm_{major}{minor}"
            report["device"] = torch.cuda.get_device_name()
            report["consumer_blackwell"] = (major == 12)
        report["grad_accum"] = self.cfg.grad_accum
        report["total_steps"] = self.cfg.total_steps
        report["params"] = self.model.n_parameters()
        dense, scores = self.model.flops_per_token(self.cfg.seq_len)
        report["gflops_per_token"] = (dense + scores) / 1e9
        report["projected_days"] = self.model.training_days(
            self.cfg.total_tokens, self.cfg.seq_len)
        report["expected_initial_loss"] = math.log(self.model.cfg.vocab)
        return report

    def current_stage(self):
        return stage_at(self.tokens_seen, self.stages)

    def maybe_switch_stage(self):
        stage = self.current_stage()
        if self.stage is not None and stage.name == self.stage.name:
            return None
        previous, self.stage = self.stage, stage
        if stage.rope_theta != self.model.cfg.rope_theta:
            self.model.set_rope_theta(stage.rope_theta)
        if stage.seq_len != self.cfg.seq_len:
            # Hold tokens per micro batch constant across the switch.  Leaving
            # micro_batch alone while the context goes 2048 -> 8192 quadruples
            # activation memory 20 days into the run and drops grad_accum from
            # 32 to 8 without saying so; if seq_len ever stops dividing the
            # global batch it silently becomes zero.  The loader hands out
            # micro_batch rows, so it has to move in step.
            per_micro = self.cfg.micro_batch * self.cfg.seq_len
            micro = max(1, per_micro // stage.seq_len)
            assert self.cfg.global_batch_tokens % (micro * stage.seq_len) == 0, (
                f"stage {stage.name}: {micro} x {stage.seq_len} does not divide "
                f"the {self.cfg.global_batch_tokens:,} token global batch")
            self.cfg.micro_batch = micro
            self.cfg.seq_len = stage.seq_len
            self.loader.batch_size = micro
        return {"from": previous.name if previous else None,
                "to": stage.name, "seq_len": stage.seq_len,
                "micro_batch": self.cfg.micro_batch,
                "grad_accum": self.cfg.grad_accum,
                "rope_theta": stage.rope_theta, "weights": stage.weights}

    def train_step(self):
        cfg = self.cfg
        switch = self.maybe_switch_stage()
        lr_factor = self.opt.set_step(self.step)
        self.opt.zero_grad()

        total = 0.0
        for micro in range(cfg.grad_accum):
            x, y = self.loader.batch(self.step * cfg.grad_accum + micro,
                                     self.stage.weights)
            x = torch.from_numpy(x).to(self.device, non_blocking=True)
            y = torch.from_numpy(y).to(self.device, non_blocking=True)
            with torch.autocast(self.device, dtype=torch.bfloat16):
                loss = self.model(x, y)
            (loss / cfg.grad_accum).backward()
            total += loss.detach().float().item()

        grad_norm = self.opt.step()
        self.step += 1
        self.tokens_seen += cfg.global_batch_tokens
        return {"loss": total / cfg.grad_accum, "grad_norm": grad_norm,
                "lr_factor": lr_factor, "stage_switch": switch}

    @torch.no_grad()
    def evaluate(self, batches, bytes_per_token=3.4):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for x, y in batches:
            x = torch.as_tensor(x, device=self.device)
            y = torch.as_tensor(y, device=self.device)
            with torch.autocast(self.device, dtype=torch.bfloat16):
                loss = self.model(x, y)
            total_loss += loss.float().item() * y.numel()
            total_tokens += y.numel()
        self.model.train()
        mean = total_loss / max(1, total_tokens)
        return {"loss": mean,
                "bpb": bits_per_byte(mean, 1, bytes_per_token),
                "ppl": math.exp(min(mean, 20))}

    def save(self, tag=None):
        tag = tag or f"step{self.step:07d}"
        path = self.out / f"{tag}.pt"
        torch.save({
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "cfg": asdict(self.cfg),
            "model_cfg": asdict(self.model.cfg),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        }, path)
        self.prune_checkpoints()
        return path

    @staticmethod
    def _step_of(path: Path):
        return int(path.stem.replace("step", ""))

    def prune_checkpoints(self):
        """Keep the last few, plus one every milestone_steps, forever.

        Checkpoints are written on a wall clock interval, so their step numbers
        are whatever the run happened to reach -- 93, 186, 279.  The old test
        here kept a checkpoint only if its step divided exactly into a
        milestone, which such numbers never do, so every older checkpoint was
        deleted and there was no way back to 10B tokens after a loss spike at
        20B.  Bucketing by step keeps one per milestone whatever the numbers
        land on.
        """
        saved = sorted(self.out.glob("step*.pt"), key=self._step_of)
        if not saved:
            return
        keep = set(saved[-self.cfg.keep_last:]) if self.cfg.keep_last else set()

        milestone = self.cfg.milestone_steps
        if milestone:
            first_in_bucket = {}
            for path in saved:
                bucket = self._step_of(path) // milestone
                first_in_bucket.setdefault(bucket, path)
            keep |= set(first_in_bucket.values())

        for path in saved:
            if path not in keep:
                path.unlink()

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        self.step = state["step"]
        self.tokens_seen = state["tokens_seen"]
        # map_location moved every tensor in the checkpoint to the device,
        # including the rng state, and set_rng_state only takes a cpu
        # ByteTensor.  Without these, every resume dies on the first restart.
        torch.set_rng_state(state["cpu_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        self.stage = None
        return state

    def latest_checkpoint(self):
        """Newest step*.pt in the output directory, by step number rather than
        mtime -- a resumed run rewrites older files and mtime lies."""
        saved = sorted(self.out.glob("step*.pt"),
                       key=lambda p: int(p.stem.replace("step", "")))
        return saved[-1] if saved else None

    def run(self, max_steps=None, max_seconds=None):
        cfg = self.cfg
        stop = min(max_steps or cfg.total_steps, cfg.total_steps)
        log_path = self.out / "log.jsonl"
        started = time.time()
        window = time.time()
        last_save = time.time()
        reason = "reached the step budget"

        # Ctrl+C mid step would throw away everything since the last save, so
        # catch it, finish the step in flight, and fall out of the loop into
        # the final checkpoint below.
        self._stopping = False

        def request_stop(signum, frame):
            if self._stopping:            # second Ctrl+C, let it through
                raise KeyboardInterrupt
            self._stopping = True
            print("\n[stopping] finishing this step, then checkpointing. "
                  "Ctrl+C again to abandon it.", flush=True)

        previous = signal.signal(signal.SIGINT, request_stop)

        while self.step < stop:
            if max_seconds is not None and time.time() - started >= max_seconds:
                reason = f"hit the {max_seconds/3600:.1f} hour session limit"
                break
            if self._stopping:
                reason = "interrupted"
                break

            metrics = self.train_step()

            if metrics["stage_switch"]:
                print(f"[stage] {metrics['stage_switch']}")

            if self.step % cfg.log_every == 0:
                elapsed = time.time() - window
                window = time.time()
                tps = cfg.global_batch_tokens * cfg.log_every / max(elapsed, 1e-9)
                dense, scores = self.model.flops_per_token(cfg.seq_len)
                mfu = model_flops_utilization(dense + scores, tps, cfg.peak_tflops)
                record = {
                    "step": self.step,
                    "tokens": self.tokens_seen,
                    "loss": round(metrics["loss"], 5),
                    "grad_norm": round(metrics["grad_norm"], 4),
                    "lr_factor": round(metrics["lr_factor"], 5),
                    "tokens_per_sec": int(tps),
                    "mfu": round(mfu, 4),
                    "hours": round((time.time() - started) / 3600, 3),
                }
                if torch.cuda.is_available():
                    record["gpu_gb"] = round(
                        torch.cuda.max_memory_allocated() / 1e9, 2)
                print(json.dumps(record))
                with open(log_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")

            by_step = cfg.save_every and self.step % cfg.save_every == 0
            by_time = (cfg.save_every_minutes
                       and time.time() - last_save >= cfg.save_every_minutes * 60)
            if by_step or by_time:
                self.save()
                last_save = time.time()

        signal.signal(signal.SIGINT, previous)

        # Always land on a checkpoint, whatever ended the loop.  Without this a
        # session that stops on the clock leaves up to save_every_minutes of
        # work on the floor.
        path = self.save()
        elapsed = time.time() - started
        print(f"\n[session] {reason} after {elapsed/3600:.2f} h, "
              f"step {self.step:,}, {self.tokens_seen/1e9:.3f}B tokens seen")
        print(f"[session] checkpoint {path}")
        return {"step": self.step, "tokens_seen": self.tokens_seen,
                "hours": elapsed / 3600, "reason": reason, "checkpoint": path}


def check():
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    cfg = TrainConfig()
    print("budget")
    print(f"    {cfg.total_tokens/1e9:.0f}B tokens, {cfg.global_batch_tokens:,} "
          f"per step -> {cfg.total_steps:,} steps")
    print(f"    micro batch {cfg.micro_batch} x seq {cfg.seq_len} "
          f"x accum {cfg.grad_accum} = {cfg.global_batch_tokens:,} tokens")
    check_one("accumulation reaches the global batch exactly",
              cfg.micro_batch * cfg.seq_len * cfg.grad_accum
              == cfg.global_batch_tokens)
    check_one("32 accumulation steps", cfg.grad_accum == 32)
    check_one(f"{cfg.total_steps:,} optimizer steps, floored so the run never "
              f"exceeds the token budget", cfg.total_steps == 76_293)
    check_one("floor undershoots 40B by less than one step",
              0 <= cfg.total_tokens - cfg.total_steps * cfg.global_batch_tokens
              < cfg.global_batch_tokens)
    try:
        TrainConfig(global_batch_tokens=500_000)
        check_one("uneven batch should have been rejected", False)
    except AssertionError:
        check_one("uneven global batch is rejected at construction", True)

    print("\ngradient accumulation equivalence")
    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 4, bias=False)
    data = [(torch.randn(8, 16), torch.randn(8, 4)) for _ in range(4)]

    layer.zero_grad()
    for x, y in data:
        (torch.nn.functional.mse_loss(layer(x), y) / len(data)).backward()
    accumulated = layer.weight.grad.clone()

    layer.zero_grad()
    big_x = torch.cat([x for x, _ in data])
    big_y = torch.cat([y for _, y in data])
    torch.nn.functional.mse_loss(layer(big_x), big_y).backward()
    single = layer.weight.grad.clone()

    err = (accumulated - single).abs().max().item()
    check_one(f"accumulated gradient equals one large batch ({err:.2e})", err < 1e-6)

    layer.zero_grad()
    for x, y in data:
        torch.nn.functional.mse_loss(layer(x), y).backward()
    unscaled = layer.weight.grad.clone()
    check_one("forgetting the division inflates the gradient by accum",
              abs((unscaled / single).mean().item() - len(data)) < 1e-4)

    print("\nbits per byte")
    check_one(f"loss 1.0 nat at 1 byte per token is 1/ln2 bits "
              f"({bits_per_byte(1.0, 1, 1):.4f})",
              abs(bits_per_byte(1.0, 1, 1) - 1 / math.log(2)) < 1e-9)
    check_one("more bytes per token lowers bpb at equal loss",
              bits_per_byte(2.0, 1, 3.4) < bits_per_byte(2.0, 1, 2.6))
    for loss in (10.80, 4.0, 2.0, 1.5):
        print(f"    loss {loss:5.2f} nats -> bpb {bits_per_byte(loss, 1, 3.4):.4f}")
    check_one("an untrained model sits above 4 bpb",
              bits_per_byte(math.log(49152), 1, 3.4) > 4)

    print("\nmfu")
    tps = 21_608
    mfu = model_flops_utilization(3.509e9, tps, 209.5)
    print(f"    {tps:,} tokens/s at 3.509 GF/token -> {mfu:.1%}")
    check_one("reproduces the 35 percent planning assumption",
              abs(mfu - 0.35) < 0.02)
    check_one("halving throughput halves mfu",
              abs(model_flops_utilization(3.509e9, tps / 2, 209.5) - mfu / 2) < 1e-6)

    print("\nstage transitions")
    seen = []
    for tokens in (0, 24e9, 36e9, 38e9, 39.9e9):
        stage = stage_at(tokens, DEFAULT_STAGES)
        seen.append((stage.name, stage.seq_len, stage.rope_theta))
        print(f"    {tokens/1e9:5.1f}B -> {stage.name:11s} seq {stage.seq_len:5d} "
              f"theta {stage.rope_theta:,.0f}")
    check_one("four distinct stages are reached", len({s[0] for s in seen}) == 4)
    check_one("sequence length only ever grows",
              all(seen[i][1] <= seen[i + 1][1] for i in range(len(seen) - 1)))
    check_one("rope theta changes exactly when the context does",
              all((a[1] == b[1]) == (a[2] == b[2]) for a, b in zip(seen, seen[1:])))

    print("\nschedule at the boundaries")
    sched = ScheduleConfig(total_steps=cfg.total_steps)
    print(f"    warmup ends {sched.warmup_steps:,}, decay starts "
          f"{sched.decay_start:,}, ends {sched.total_steps:,}")
    check_one("warmup is under 3 percent of the run",
              sched.warmup_steps / sched.total_steps < 0.03)
    check_one("decay covers the last 10 percent",
              abs((sched.total_steps - sched.decay_start)
                  / sched.total_steps - 0.10) < 0.01)
    tokens_in_decay = (sched.total_steps - sched.decay_start) * cfg.global_batch_tokens
    print(f"    decay phase sees {tokens_in_decay/1e9:.1f}B tokens, "
          f"which is where the annealing mix goes")
    check_one("decay phase is about 4B tokens", 3e9 < tokens_in_decay < 5e9)

    print("\ncheckpoint cadence")
    per_ckpt = 514_345_526 * 2 + 2.31e9
    print(f"    every {cfg.save_every:,} steps = "
          f"{cfg.save_every*cfg.global_batch_tokens/1e9:.2f}B tokens")
    print(f"    roughly {per_ckpt/1e9:.2f} GB each, keeping {cfg.keep_last}")
    check_one("checkpoints land about once per billion tokens",
              0.5e9 < cfg.save_every * cfg.global_batch_tokens < 2e9)

    print("\n" + ("train ok" if passed else "FAILURES ABOVE"))
    print("\npreflight on the real machine before committing three weeks:")
    print("  tokens/s at or above 18,000, gpu utilisation above 95 percent,")
    print("  loss at step 0 near 10.80, attention kernel confirmed not the math")
    print("  fallback, power limited to 500W, and resume tested from a checkpoint.")
    return passed


if __name__ == "__main__":
    check()