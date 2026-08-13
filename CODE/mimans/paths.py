"""Every path this project writes to, all on D:.

C: has under 11 GB free.  One checkpoint is 3.3 GB and five are kept, the tuned
kernel cache passed 140 MB in a single session, and the uint16 shards for a 40B
token run are 80 GB before counting the raw text they came from.  None of that
belongs on the system drive.

tilelang reads its cache location once, at import time, so this module has to be
imported before tilelang is -- which is why the three kernel modules import it
on their first line rather than taking the path as an argument.

Override the root with the MIMANS_ROOT environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = Path("D:/mimans")

_root = os.environ.get("MIMANS_ROOT")
if _root:
    ROOT = Path(_root)
elif DEFAULT_ROOT.drive and Path(DEFAULT_ROOT.drive + "/").exists():
    ROOT = DEFAULT_ROOT
else:
    # No D: on this machine.  Fall back next to the repo rather than silently
    # filling up the system drive from a path nobody looked at.
    ROOT = Path(__file__).resolve().parents[3] / "mimans_data"
    print(f"warning: {DEFAULT_ROOT.drive} not found, writing to {ROOT}")

DATA = ROOT / "data"            # raw downloaded corpora, full scale training data
SHARDS = ROOT / "shards"        # tokenized uint16 shards, ~80 GB at 40B tokens
RUNS = ROOT / "runs"            # checkpoints and log.jsonl
CACHE = ROOT / "cache"          # tuned kernels
TMP = ROOT / "tmp"              # scratch, including preflight's synthetic data
TOKENIZER = ROOT / "tokenizer"  # tokenizer.json

# A small, separate corpus just for training and checking the tokenizer.  Kept
# apart from DATA (the full 136 GB training corpus) because it is 2-3 orders
# of magnitude smaller and gets downloaded, inspected, and re-downloaded long
# before the real training data does.
TOKENIZER_DATA = ROOT / "tokenizer_data"
TOKENIZER_RAW = TOKENIZER_DATA / "raw"          # downloaded, untouched
TOKENIZER_CODE = TOKENIZER_RAW / "code"
TOKENIZER_TEXT = TOKENIZER_RAW / "text"
TOKENIZER_MATH = TOKENIZER_RAW / "math"
TOKENIZER_HOLDOUT = TOKENIZER_DATA / "holdout"  # never trained on, for --check

for _dir in (DATA, SHARDS, RUNS, CACHE, TMP, TOKENIZER,
            TOKENIZER_CODE, TOKENIZER_TEXT, TOKENIZER_MATH, TOKENIZER_HOLDOUT):
    _dir.mkdir(parents=True, exist_ok=True)

# setdefault, so an explicit environment variable still wins.
os.environ.setdefault("TILELANG_CACHE_DIR", str(CACHE / "tilelang"))
os.environ.setdefault("TILELANG_TMP_DIR", str(CACHE / "tilelang" / "tmp"))

# huggingface_hub defaults its cache to ~/.cache/huggingface on C:.  Dataset
# shards are hundreds of MB each, so left alone this fills the system drive
# even when the download target is on D:.
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "huggingface" / "hub"))

# Python's own tempfile default is %TEMP% on C:.  Anything here that writes a
# checkpoint or a synthetic shard through tempfile lands on the system drive
# unless it passes dir= explicitly, which is easy to forget.
os.environ.setdefault("TMPDIR", str(TMP))
os.environ.setdefault("TEMP", str(TMP))
os.environ.setdefault("TMP", str(TMP))
import tempfile
tempfile.tempdir = str(TMP)


def summary():
    import shutil
    free = shutil.disk_usage(ROOT).free / 1e9
    return f"{ROOT}  ({free:,.0f} GB free)"


if __name__ == "__main__":
    print(summary())
    for name, path in (("data", DATA), ("shards", SHARDS), ("runs", RUNS),
                       ("cache", CACHE), ("tmp", TMP), ("tokenizer", TOKENIZER),
                       ("tokenizer_data/raw/code", TOKENIZER_CODE),
                       ("tokenizer_data/raw/text", TOKENIZER_TEXT),
                       ("tokenizer_data/raw/math", TOKENIZER_MATH),
                       ("tokenizer_data/holdout", TOKENIZER_HOLDOUT)):
        print(f"  {name:10s} {path}")
    print(f"\n  TILELANG_CACHE_DIR {os.environ['TILELANG_CACHE_DIR']}")
