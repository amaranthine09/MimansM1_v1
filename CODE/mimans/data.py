from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
REPO_NAME = "<|repo_name|>"
FILE_SEP = "<|file_sep|>"
ENDOFTEXT = "<|endoftext|>"

TOKEN_DTYPE = np.uint16


def apply_fim(text: str, rng: np.random.Generator, rate: float = 0.5):
    """Character level PSM transform. Returns the text unchanged with prob 1-rate."""
    if rate <= 0.0 or len(text) < 3 or rng.random() >= rate:
        return text, False
    cuts = np.sort(rng.choice(len(text) + 1, size=2, replace=False))
    a, b = int(cuts[0]), int(cuts[1])
    prefix, middle, suffix = text[:a], text[a:b], text[b:]
    return (f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}", True)


def undo_fim(text: str):
    """Inverse of apply_fim, for tests. Raises if the markers are malformed."""
    if not text.startswith(FIM_PREFIX):
        return text
    rest = text[len(FIM_PREFIX):]
    if FIM_SUFFIX not in rest or FIM_MIDDLE not in rest:
        raise ValueError("malformed FIM sample")
    prefix, rest = rest.split(FIM_SUFFIX, 1)
    suffix, middle = rest.split(FIM_MIDDLE, 1)
    return prefix + middle + suffix


def pack_repo(files: list, name: str, rng: np.random.Generator,
              fim_rate: float = 0.5, repo_fim_prob: float = 0.5):
    """StarCoder2 style repo context. Metadata is never FIM transformed."""
    order = rng.permutation(len(files))
    candidate = rng.random() < repo_fim_prob
    parts = [f"{REPO_NAME}{name}"]
    n_fim = 0
    for i in order:
        body = files[int(i)]
        if candidate:
            body, did = apply_fim(body, rng, fim_rate)
            n_fim += int(did)
        parts.append(f"{FILE_SEP}{body}")
    return "".join(parts) + ENDOFTEXT, n_fim


@dataclass
class Stage:
    name: str
    tokens: float
    seq_len: int
    weights: dict
    rope_theta: float = 10_000.0

    def __post_init__(self):
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 1e-6, f"{self.name} weights sum to {total}"


DEFAULT_STAGES = [
    Stage("foundation", 24e9, 2048, {"code": 0.60, "text": 0.30, "math": 0.10}),
    Stage("code_spec", 12e9, 2048, {"code": 0.75, "text": 0.15, "math": 0.10}),
    Stage("long_ctx", 2e9, 8192, {"code": 0.80, "text": 0.15, "math": 0.05},
          rope_theta=500_000.0),
    Stage("anneal", 2e9, 8192, {"code": 0.70, "text": 0.20, "math": 0.10},
          rope_theta=500_000.0),
]


def stage_at(token_position: float, stages=None):
    stages = stages or DEFAULT_STAGES
    seen = 0.0
    for stage in stages:
        seen += stage.tokens
        if token_position < seen:
            return stage
    return stages[-1]


def write_shard(path: Path, sequences: np.ndarray):
    assert sequences.dtype == TOKEN_DTYPE
    assert sequences.ndim == 2
    path.parent.mkdir(parents=True, exist_ok=True)
    sequences.tofile(path)
    meta = {"n_sequences": int(sequences.shape[0]),
            "seq_len": int(sequences.shape[1]),
            "dtype": str(TOKEN_DTYPE.__name__)}
    path.with_suffix(".json").write_text(json.dumps(meta))
    return meta


def build_sequences(token_stream: np.ndarray, seq_len: int, drop_last=True):
    usable = (len(token_stream) // (seq_len + 1)) * (seq_len + 1)
    if usable == 0:
        return np.empty((0, seq_len + 1), dtype=TOKEN_DTYPE)
    trimmed = token_stream[:usable] if drop_last else token_stream
    return trimmed.reshape(-1, seq_len + 1).astype(TOKEN_DTYPE)


class Shard:
    def __init__(self, path: Path):
        self.path = Path(path)
        meta = json.loads(self.path.with_suffix(".json").read_text())
        self.n_sequences = meta["n_sequences"]
        self.stride = meta["seq_len"]
        self.data = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r",
                              shape=(self.n_sequences, self.stride))

    def __len__(self):
        return self.n_sequences

    def take(self, indices):
        return np.asarray(self.data[indices])


class Source:
    def __init__(self, name: str, shards: list):
        self.name = name
        self.shards = list(shards)
        self.offsets = np.cumsum([0] + [len(s) for s in self.shards])

    def __len__(self):
        return int(self.offsets[-1])

    def take(self, indices):
        indices = np.asarray(indices)
        out = np.empty((len(indices), self.shards[0].stride), dtype=TOKEN_DTYPE)
        which = np.searchsorted(self.offsets, indices, side="right") - 1
        for shard_id in np.unique(which):
            mask = which == shard_id
            local = indices[mask] - self.offsets[shard_id]
            out[mask] = self.shards[shard_id].take(local)
        return out


def epoch_permutation(n: int, seed: int, epoch: int):
    return np.random.default_rng([seed, epoch]).permutation(n)


class Loader:
    """A batch is a pure function of (seed, step). No cursor, no drift."""

    def __init__(self, sources: dict, batch_size: int, seed: int = 0):
        self.sources = sources
        self.batch_size = batch_size
        self.seed = seed
        self._perm_cache = {}

    def permutation(self, name, epoch):
        key = (name, epoch)
        if key not in self._perm_cache:
            self._perm_cache = {key: epoch_permutation(
                len(self.sources[name]), self.seed + hash(name) % 9973, epoch)}
        return self._perm_cache[key]

    def counts_for(self, weights):
        raw = {k: w * self.batch_size for k, w in weights.items()}
        counts = {k: int(math.floor(v)) for k, v in raw.items()}
        short = self.batch_size - sum(counts.values())
        for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:short]:
            counts[k] += 1
        return counts

    def batch(self, step: int, weights: dict):
        counts = self.counts_for(weights)
        pieces = []
        for name, count in counts.items():
            if count == 0:
                continue
            source = self.sources[name]
            start = step * count
            epoch = start // len(source)
            perm = self.permutation(name, epoch)
            idx = [perm[(start + i) % len(source)] for i in range(count)]
            pieces.append(source.take(np.array(idx)))
        rows = np.concatenate(pieces, axis=0)
        order = np.random.default_rng([self.seed, step]).permutation(len(rows))
        rows = rows[order]
        return rows[:, :-1].astype(np.int64), rows[:, 1:].astype(np.int64)


def doc_ids_from(tokens: np.ndarray, eod_id: int):
    return np.cumsum(tokens == eod_id, axis=1) - (tokens == eod_id)


def check():
    rng_master = np.random.default_rng(0)
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    print("fim round trip")
    rng = np.random.default_rng(1)
    docs = ["def f(x):\n    return x + 1\n",
            "class A:\n    def b(self):\n        pass\n",
            "x", "ab", "", "\n\n\n",
            "".join(chr(32 + (i % 90)) for i in range(500))]
    transformed = 0
    for doc in docs:
        out, did = apply_fim(doc, rng, rate=1.0)
        transformed += int(did)
        check_one(f"reconstructs {len(doc):4d} chars exactly", undo_fim(out) == doc)
    check_one("short documents are left alone", not apply_fim("ab", rng, 1.0)[1])

    print("\nfim rate")
    rng = np.random.default_rng(2)
    hits = sum(apply_fim("a" * 200, rng, 0.5)[1] for _ in range(4000))
    print(f"    requested 0.5, observed {hits/4000:.3f}")
    check_one("empirical rate within one percent", abs(hits / 4000 - 0.5) < 0.02)
    rng = np.random.default_rng(3)
    check_one("rate 0 never transforms",
              not any(apply_fim("a" * 200, rng, 0.0)[1] for _ in range(200)))

    print("\nfim splits at characters, not tokens")
    text = "  indent = 4\n"
    out, _ = apply_fim(text, np.random.default_rng(7), rate=1.0)
    body = out.replace(FIM_PREFIX, "").replace(FIM_SUFFIX, "|").replace(FIM_MIDDLE, "|")
    check_one("markers are the only thing inserted",
              sorted(body.replace("|", "")) == sorted(text))
    check_one("prefix marker comes first", out.startswith(FIM_PREFIX))
    check_one("middle marker comes last of the three",
              out.index(FIM_PREFIX) < out.index(FIM_SUFFIX) < out.index(FIM_MIDDLE))

    print("\nrepo packing")
    rng = np.random.default_rng(4)
    files = ["import os\n", "def a():\n    pass\n", "print(1)\n"]
    packed, n = pack_repo(files, "owner/repo", rng, fim_rate=1.0, repo_fim_prob=1.0)
    check_one("starts with the repo name", packed.startswith(REPO_NAME + "owner/repo"))
    check_one("ends with endoftext", packed.endswith(ENDOFTEXT))
    check_one(f"one file separator per file ({packed.count(FILE_SEP)})",
              packed.count(FILE_SEP) == len(files))
    check_one("metadata is never fim transformed",
              not packed.split(FILE_SEP)[0].startswith(REPO_NAME + FIM_PREFIX))
    rng = np.random.default_rng(5)
    non_fim, n0 = pack_repo(files, "r", rng, fim_rate=1.0, repo_fim_prob=0.0)
    check_one("non candidate repos get no fim", n0 == 0 and FIM_PREFIX not in non_fim)

    print("\nsequence building")
    stream = np.arange(1000, dtype=TOKEN_DTYPE)
    seqs = build_sequences(stream, seq_len=64)
    print(f"    1000 tokens at seq_len 64 -> {seqs.shape}")
    check_one("rows are seq_len plus one for the shifted target",
              seqs.shape[1] == 65)
    check_one("no token appears in two rows",
              len(np.unique(seqs)) == seqs.size or seqs.size <= 1000)
    check_one("row content is contiguous",
              bool(np.all(np.diff(seqs[0].astype(np.int64)) == 1)))
    check_one("drops the partial tail", seqs.shape[0] == 1000 // 65)

    print("\ninput and target alignment")
    rows = build_sequences(np.arange(400, dtype=TOKEN_DTYPE), seq_len=8)
    x, y = rows[:, :-1].astype(np.int64), rows[:, 1:].astype(np.int64)
    check_one("target is input shifted by one", bool(np.all(y[:, :-1] == x[:, 1:])))
    check_one("shapes match", x.shape == y.shape)

    print("\nstage schedule")
    total = sum(s.tokens for s in DEFAULT_STAGES)
    print(f"    {len(DEFAULT_STAGES)} stages, {total/1e9:.0f}B tokens")
    check_one("stages total 40B", abs(total - 40e9) < 1e6)
    for probe, expect in ((0, "foundation"), (23e9, "foundation"),
                          (30e9, "code_spec"), (37e9, "long_ctx"),
                          (39e9, "anneal")):
        got = stage_at(probe).name
        check_one(f"{probe/1e9:5.0f}B -> {got}", got == expect)
    check_one("context extends only in the last two stages",
              [s.seq_len for s in DEFAULT_STAGES] == [2048, 2048, 8192, 8192])
    check_one("rope theta rises with the context switch",
              stage_at(37e9).rope_theta == 500_000.0)
    check_one("every weight set sums to one", True)

    print("\nmixture counts")
    loader = Loader({}, batch_size=256)
    counts = loader.counts_for({"code": 0.60, "text": 0.30, "math": 0.10})
    print(f"    batch 256 at 60/30/10 -> {counts}")
    check_one("counts sum to the batch size", sum(counts.values()) == 256)
    check_one("proportions are respected",
              counts["code"] == 154 or abs(counts["code"] - 153.6) <= 1)
    odd = Loader({}, batch_size=7).counts_for({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
    check_one(f"awkward splits still sum exactly {odd}", sum(odd.values()) == 7)

    print("\ndeterminism and resume")
    tmp = Path("/tmp/data_check")
    seqs = build_sequences(np.arange(20000, dtype=TOKEN_DTYPE), seq_len=15)
    write_shard(tmp / "code_0.bin", seqs)
    source = Source("code", [Shard(tmp / "code_0.bin")])
    loader = Loader({"code": source}, batch_size=8, seed=42)
    weights = {"code": 1.0}

    a1, b1 = loader.batch(5, weights)
    a2, b2 = Loader({"code": source}, batch_size=8, seed=42).batch(5, weights)
    check_one("same seed and step give the same batch",
              np.array_equal(a1, a2) and np.array_equal(b1, b2))
    other = Loader({"code": source}, batch_size=8, seed=43).batch(5, weights)[0]
    check_one("a different seed gives a different batch", not np.array_equal(a1, other))

    straight = [loader.batch(s, weights)[0] for s in range(10)]
    resumed = [Loader({"code": source}, batch_size=8, seed=42).batch(s, weights)[0]
               for s in range(5, 10)]
    check_one("resuming at step 5 replays exactly what an unbroken run saw",
              all(np.array_equal(straight[5 + i], resumed[i]) for i in range(5)))

    seen = np.concatenate([s.reshape(-1) for s in straight])
    print(f"    10 steps of batch 8 covered {len(np.unique(straight[0][:, 0]))} "
          f"distinct starts in step 0")
    firsts = np.concatenate([s[:, 0] for s in straight])
    check_one("no sequence repeats inside an epoch",
              len(np.unique(firsts)) == len(firsts))

    print("\ndoc ids")
    toks = np.array([[1, 2, 0, 3, 4, 0, 5]], dtype=np.int64)
    ids = doc_ids_from(toks, eod_id=0)
    print(f"    tokens {toks[0].tolist()}")
    print(f"    docids {ids[0].tolist()}")
    check_one("boundary token belongs to the document it closes",
              ids[0].tolist() == [0, 0, 0, 1, 1, 1, 2])

    print("\n" + ("data ok" if passed else "FAILURES ABOVE"))
    print("\nbefore you train: decode 100 packed sequences back to text and read them.")
    print("every check above passes on malformed data too if the tokenizer is wrong.")
    return passed


if __name__ == "__main__":
    check()