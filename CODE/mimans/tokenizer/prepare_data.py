"""Raw text -> uint16 shards, the format mimans.data.Loader reads.

Documents are tokenized, separated by <|endoftext|>, concatenated into one
stream per source, and cut into rows of seq_len+1 so that x and y are the same
row shifted by one.  Code gets the FIM transform at the character level before
tokenization, which is what teaches the model to fill in a hole given both
sides of it.

Shards land in paths.SHARDS/<source>/, one .bin plus a .json sidecar each, and
the loader picks up whatever is there.  Writing is incremental: a shard is
flushed as soon as it is full, so an interrupted run leaves usable shards
behind rather than nothing.

    python -m mimans.tokenizer.prepare_data --tokens-per-source 50e6
    python -m mimans.tokenizer.prepare_data --seq-len 8192 --out long_ctx
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mimans import paths
from mimans.data import ENDOFTEXT, TOKEN_DTYPE, build_sequences, write_shard
from mimans.data import apply_fim
from mimans.tokenizer.train_tokenizer import (TOP_CODE_LANGUAGES, documents,
                                              select_code_dirs,
                                              select_code_parquets)

ENCODE_BATCH = 512          # documents per encode_batch call
SHARD_SEQUENCES = 20_000    # rows per shard file


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("needs the tokenizers library: pip install tokenizers") from exc
    if not path.exists():
        raise SystemExit(f"no tokenizer at {path}; train it first")
    return Tokenizer.from_file(str(path))


def source_paths(name: str, languages):
    """Where each mixture source reads from, with code narrowed to the
    languages the tokenizer was built for."""
    if name == "code":
        root = paths.TOKENIZER_CODE
        return select_code_dirs(root, languages) + select_code_parquets(root, languages)
    return {"text": paths.TOKENIZER_TEXT, "math": paths.TOKENIZER_MATH}[name]


def encode_stream(tokenizer, source, eod_id, target_tokens, fim_rate=0.0,
                  seed=0, report_every=25_000_000):
    """Yield token ids for one source until target_tokens is reached.

    Batched through encode_batch because per-document calls spend more time
    crossing into the tokenizer than tokenizing.
    """
    rng = np.random.default_rng(seed)
    produced, batch = 0, []
    started = time.time()
    next_report = report_every

    def encode(batch):
        """One batch of documents -> a flat array of ids, each doc terminated
        by <|endoftext|> so the model learns where documents end."""
        ids = []
        for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
            ids.extend(encoding.ids)
            ids.append(eod_id)
        return np.asarray(ids, dtype=np.int64)

    for text in documents(source):
        if fim_rate:
            text, _ = apply_fim(text, rng, fim_rate)
        batch.append(text)
        if len(batch) < ENCODE_BATCH:
            continue

        chunk = encode(batch)
        batch = []
        produced += len(chunk)
        yield chunk

        if produced >= next_report:
            rate = produced / max(time.time() - started, 1e-9)
            print(f"      {produced/1e6:7.1f}M tokens  {rate/1e3:6.1f}k tok/s",
                  flush=True)
            next_report += report_every
        if produced >= target_tokens:
            return

    if batch:
        yield encode(batch)


def build_source(tokenizer, name, source, out_dir: Path, seq_len: int,
                 target_tokens: int, fim_rate: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    eod_id = tokenizer.token_to_id(ENDOFTEXT)
    if eod_id is None:
        raise SystemExit(f"tokenizer has no {ENDOFTEXT}")

    row = seq_len + 1
    carry = np.empty(0, dtype=np.int64)
    pending = []          # rows waiting to be written
    shard_index, rows_written, tokens_written = 0, 0, 0
    started = time.time()

    def flush_shard(force=False):
        nonlocal pending, shard_index, rows_written
        while pending and (len(pending) >= SHARD_SEQUENCES or force):
            take = pending[:SHARD_SEQUENCES]
            pending = pending[SHARD_SEQUENCES:]
            block = np.stack(take).astype(TOKEN_DTYPE)
            path = out_dir / f"{name}_{shard_index:04d}.bin"
            write_shard(path, block)
            rows_written += len(block)
            shard_index += 1
            print(f"      wrote {path.name}  {len(block):,} rows", flush=True)
            if not force:
                continue
            if not pending:
                break

    for chunk in encode_stream(tokenizer, source, eod_id, target_tokens,
                               fim_rate=fim_rate, seed=seed):
        tokens_written += len(chunk)
        stream = np.concatenate([carry, chunk]) if carry.size else chunk
        usable = (len(stream) // row) * row
        if usable:
            rows = build_sequences(stream[:usable].astype(TOKEN_DTYPE), seq_len)
            pending.extend(rows)
            flush_shard()
        carry = stream[usable:]

    flush_shard(force=True)
    elapsed = time.time() - started
    print(f"    {name}: {rows_written:,} sequences in {shard_index} shards, "
          f"{rows_written*seq_len/1e6:.1f}M usable tokens, {elapsed/60:.1f} min")
    return rows_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-source", type=float, default=50e6,
                        help="token budget per source before shard cutting")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=None,
                        help="shard root (default paths.SHARDS)")
    parser.add_argument("--tokenizer", type=Path,
                        default=paths.TOKENIZER / "tokenizer.json")
    parser.add_argument("--fim-rate", type=float, default=0.5,
                        help="fraction of code documents given the FIM transform")
    parser.add_argument("--languages", default=",".join(TOP_CODE_LANGUAGES))
    parser.add_argument("--sources", default="code,text,math")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    root = args.out or paths.SHARDS
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    names = [s.strip() for s in args.sources.split(",") if s.strip()]

    print(f"tokenizer {args.tokenizer}  vocab {tokenizer.get_vocab_size():,}")
    print(f"shards to {root}, seq_len {args.seq_len}")
    print(f"{args.tokens_per_source/1e6:.0f}M tokens per source\n")

    total = 0
    for name in names:
        source = source_paths(name, languages)
        fim = args.fim_rate if name == "code" else 0.0
        print(f"  {name}" + (f"  (fim rate {fim})" if fim else ""))
        total += build_source(tokenizer, name, source, root / name,
                              args.seq_len, int(args.tokens_per_source),
                              fim, args.seed)
    print(f"\n{total:,} sequences of {args.seq_len} = "
          f"{total*args.seq_len/1e9:.2f}B tokens ready in {root}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
