"""Remembered autotune results, so the search happens once and never again.

tilelang can autotune every kernel against the machine it is running on, which
is worth doing -- but its own on-disk cache does not work here (tvm wants a
target triple from a host compiler that is not installed), so left alone it
re-derives the same answer on every process start.  That is over an hour before
the first token of a session is trained.

The answer only depends on the GPU and the kernel shape, and neither changes
between sessions, so it is recorded here instead: a plain JSON file mapping
(gpu, kernel, shape) to the winning tile configuration.  Kernels consult it
before tuning.  A hit means the tuned tiles are used with no search at all; a
miss falls back to the module defaults unless tuning was explicitly asked for.

Config lives as data rather than as edited source so that re-tuning is a
command rather than a patch, and so a different GPU simply misses instead of
silently using tiles picked for hardware you no longer have.

    python -m mimans.training.tune          # fill it in, once
    python -m mimans.model.tuned            # show what is remembered
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mimans import paths

STORE = paths.CACHE / "tuned_kernels.json"

_cache = None


def device_key():
    """Tiles are only valid for the card they were measured on."""
    if not torch.cuda.is_available():
        return "no-cuda"
    name = torch.cuda.get_device_name().replace(" ", "-")
    major, minor = torch.cuda.get_device_capability()
    return f"{name}-sm{major}{minor}"


def _shape_key(kernel: str, shape) -> str:
    return f"{kernel}:{','.join(str(s) for s in shape)}"


def _load():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(STORE.read_text())
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def lookup(kernel: str, shape):
    """The remembered config for this kernel and shape, or None."""
    return _load().get(device_key(), {}).get(_shape_key(kernel, shape))


def record(kernel: str, shape, config: dict):
    """Remember a tuned config.  Written immediately, so a tuning run that is
    interrupted keeps whatever it had already measured."""
    if not config:
        return
    store = _load()
    store.setdefault(device_key(), {})[_shape_key(kernel, shape)] = dict(config)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True))
    tmp.replace(STORE)


def config_of(kernel):
    """The config an autotuned tilelang kernel settled on, if it reports one."""
    config = getattr(kernel, "config", None)
    if isinstance(config, dict):
        return config
    # Some versions expose it on the wrapped best result instead.
    best = getattr(kernel, "best", None)
    return getattr(best, "config", None) if best is not None else None


def summary():
    store = _load()
    if not store:
        return f"nothing tuned yet ({STORE})"
    lines = [f"{STORE}"]
    for device, entries in sorted(store.items()):
        lines.append(f"  {device}  ({len(entries)} kernels)")
        for key, config in sorted(entries.items()):
            kernel, shape = key.split(":", 1)
            terse = " ".join(f"{k}={v}" for k, v in sorted(config.items()))
            lines.append(f"    {kernel:14s} [{shape}]")
            lines.append(f"      {terse}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"this gpu: {device_key()}\n")
    print(summary())
