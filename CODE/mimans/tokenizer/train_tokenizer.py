"""Train the byte level BPE this model is built around.

Every hard number here is read off the rest of the repo, not chosen:

  vocab 49152        ModelConfig.vocab, and HeadConfig asserts vocab % 128 == 0
                     so the lm head kernel tiles cleanly.  49152 = 384 * 128,
                     and it is what makes the tied embedding 62,914,560
                     parameters and the model 514,345,526.
  ids < 65536        shards are uint16 (data.TOKEN_DTYPE)
  six specials       data.py splices these into the character stream, so they
                     have to survive encoding as single atomic ids
  no pad token       sequences are packed and chopped, never padded; masking is
                     done with IGNORE_INDEX at the target level

Two things that are easy to get wrong:

  * FIM is a character level transform applied *before* tokenization, so the
    tokenizer routinely sees fragments that start mid-identifier, and the
    markers arrive with no whitespace around them.  Byte level BPE over the
    full 256 byte alphabet is what keeps that lossless.  Train on raw text --
    never run apply_fim over the tokenizer corpus.
  * Indentation is a quarter of Python.  The byte level regex keeps runs of
    whitespace together, which is why it is worth more than a plain whitespace
    split here.

    pip install tokenizers            # and pyarrow for .parquet corpora
    python -m mimans.tokenizer.train_tokenizer --train \
        --corpus code=D:/data/code --corpus text=D:/data/text
    python -m mimans.tokenizer.train_tokenizer --check --sample D:/data/holdout
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

from mimans import paths
from mimans.data import (ENDOFTEXT, FILE_SEP, FIM_MIDDLE, FIM_PREFIX,
                         FIM_SUFFIX, REPO_NAME, apply_fim, undo_fim)

VOCAB_SIZE = 49_152
MIN_FREQUENCY = 2

# Ids 0-5 are the tokens data.py depends on and must never move.  The rest are
# reserved so that adding a chat or instruct marker later does not mean
# retraining the tokenizer and re-tokenizing the corpus.
SPECIAL_TOKENS = [
    ENDOFTEXT,       # 0, also the eod_id that data.doc_ids_from wants
    FIM_PREFIX,      # 1
    FIM_MIDDLE,      # 2
    FIM_SUFFIX,      # 3
    REPO_NAME,       # 4
    FILE_SEP,        # 5
    "<|pad|>",       # 6, unused by the pipeline, conventional to have
] + [f"<|extra_{i}|>" for i in range(9)]

DEFAULT_WEIGHTS = {"code": 0.60, "text": 0.30, "math": 0.10}
TEXT_FIELDS = ("content", "text", "code")

# the-stack-smol's own folder names, lowercase, hyphenated where the language
# name has a space or symbol.  Five general purpose, high volume languages
# rather than all thirty: fewer merges get spent on syntax nobody trains on,
# which is what a 49152 slot vocab for a 514M model can actually afford.
TOP_CODE_LANGUAGES = ("python", "javascript", "typescript", "java", "c++")


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("this needs the huggingface tokenizers library:\n"
                         "    pip install tokenizers") from exc


# ----------------------------------------------------------------- corpus ---

def _field_of(record: dict):
    for name in TEXT_FIELDS:
        if name in record and isinstance(record[name], str):
            return record[name]
    return None


def _jsonl(handle):
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _field_of(record) if isinstance(record, dict) else None
        if text:
            yield text


def documents(path):
    """Yield raw document strings from a file, a directory tree, or a list of
    either -- the list form is how a source gets narrowed to a handful of
    subdirectories, e.g. select_code_dirs()'s output, without teaching
    mixed_corpus about the distinction.

    Understands jsonl (plain, .gz, .zst), parquet, and plain text files, which
    covers how the open datasets actually ship.
    """
    if isinstance(path, (list, tuple)):
        for item in path:
            yield from documents(item)
        return

    path = Path(path)
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield from documents(child)
        return

    suffixes = [s.lower() for s in path.suffixes]

    if ".parquet" in suffixes:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise SystemExit(f"{path.name} is parquet: pip install pyarrow")
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=1024):
            records = batch.to_pylist()
            for record in records:
                text = _field_of(record)
                if text:
                    yield text
        return

    if ".jsonl" in suffixes or ".json" in suffixes:
        if ".gz" in suffixes:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as h:
                yield from _jsonl(h)
        elif ".zst" in suffixes:
            try:
                import zstandard
            except ImportError:
                raise SystemExit(f"{path.name} is zstd: pip install zstandard")
            with open(path, "rb") as raw:
                stream = zstandard.ZstdDecompressor().stream_reader(raw)
                wrapper = io.TextIOWrapper(stream, encoding="utf-8",
                                           errors="replace")
                yield from _jsonl(wrapper)
        else:
            with open(path, "rt", encoding="utf-8", errors="replace") as h:
                yield from _jsonl(h)
        return

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if text.strip():
        yield text


# the-stack-v2 style parquet 'language' values use display names and symbols
# ("C++", "C#") where the-stack-smol's folder names are lowercase and
# hyphenated ("c++", "c-sharp").  Only the entries that actually differ need
# to be listed; anything else is matched by lowercasing both sides.
CODE_LANGUAGE_ALIASES = {"c-sharp": "c#", "shell": "bash", "visual-basic": "visual basic"}


def select_code_dirs(code_root: Path, languages=TOP_CODE_LANGUAGES):
    """The the-stack-smol layout is code_root/data/<language>/data.json*.
    Returning just those subdirectories, rather than code_root itself, is what
    excludes the other twenty five languages.  Loose parquet files sitting
    next to data/ are a second, already-mixed-language dataset with no per
    file split -- those go through select_code_parquets instead, which filters
    by row.
    """
    code_root = Path(code_root)
    data_dir = code_root / "data"
    if not data_dir.is_dir():
        return [code_root]  # not this layout, fall back to reading everything

    found, missing = [], []
    for lang in languages:
        d = data_dir / lang
        (found if d.is_dir() else missing).append(d if d.is_dir() else lang)
    if missing:
        available = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
        print(f"    warning: {missing} not found under {data_dir}; "
              f"have {available}")
    return found


def select_code_parquets(code_root: Path, languages=TOP_CODE_LANGUAGES):
    """Loose *.parquet files directly under code_root (not under data/) are a
    second dataset shipped in the same folder, e.g. a the-stack-v2 sample,
    with every language mixed into one file and picked out by a 'language'
    column.  Filter each down to the requested languages and cache the result
    next to the source, keyed off the source's mtime so re-running is cheap.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    code_root = Path(code_root)
    wanted = {CODE_LANGUAGE_ALIASES.get(l, l).lower() for l in languages}
    cache_dir = code_root / "_language_filtered"
    out = []

    for src in sorted(code_root.glob("*.parquet")):
        cache = cache_dir / f"{src.stem}.parquet"
        if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
            out.append(cache)
            continue

        schema_names = pq.ParquetFile(src).schema_arrow.names
        lang_col = next((c for c in ("language", "lang") if c in schema_names), None)
        if lang_col is None:
            print(f"    {src.name}: no language column, skipping "
                  f"(cannot filter, would pull in every language)")
            continue

        table = pq.read_table(src)
        norm = pc.utf8_lower(table.column(lang_col))
        mask = pc.is_in(norm, value_set=__import__("pyarrow").array(sorted(wanted)))
        filtered = table.filter(mask)
        if filtered.num_rows == 0:
            print(f"    {src.name}: 0 of {table.num_rows} rows match {languages}, skipping")
            continue

        cache_dir.mkdir(exist_ok=True)
        pq.write_table(filtered, cache)
        print(f"    {src.name}: kept {filtered.num_rows:,} of {table.num_rows:,} rows "
              f"matching {languages} -> {cache.name}")
        out.append(cache)
    return out


def mixed_corpus(sources: dict, weights: dict, budget_bytes: int, verbose=True):
    """Documents from every source, each capped at its share of the budget.

    BPE only ever sees aggregate counts, so the order does not matter -- what
    matters is that the proportions match the training mixture, since that is
    what the merges get tuned for.
    """
    taken = {}
    for name, path in sources.items():
        share = weights.get(name, 0.0)
        cap = int(budget_bytes * share)
        used = 0
        if cap <= 0:
            continue
        for text in documents(path):
            yield text
            used += len(text.encode("utf-8", errors="ignore"))
            if used >= cap:
                break
        taken[name] = used
        if verbose:
            print(f"    {name:6s} {used/1e9:6.3f} GB of {cap/1e9:6.3f} GB budget"
                  f"{'  (source exhausted)' if used < cap else ''}")
    if verbose:
        print(f"    total  {sum(taken.values())/1e9:6.3f} GB")


# ------------------------------------------------------------------ train ---

def build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    tok = Tokenizer(models.BPE(unk_token=None))
    # No normalizer at all.  NFC or lowercasing would make encode/decode lossy
    # on real source files, and the shards have to round trip exactly.
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        # Splitting every digit is what makes the 10 percent math mixture
        # learnable; without it "1024" and "1025" share nothing.
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def train(sources: dict, out: Path, budget_bytes: int, weights: dict,
          vocab_size=VOCAB_SIZE):
    _require_tokenizers()
    from tokenizers import pre_tokenizers, trainers

    assert vocab_size % 128 == 0, "the lm head kernel asserts vocab % 128 == 0"
    assert vocab_size <= 65_536, "shards are uint16"

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        # All 256 bytes present means there is no unknown token and no input
        # that fails to encode, which is the property FIM fragments rely on.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"training to {vocab_size:,} over {budget_bytes/1e9:.1f} GB, "
          f"mixture {weights}")
    tok.train_from_iterator(mixed_corpus(sources, weights, budget_bytes),
                            trainer=trainer)
    tok.add_special_tokens(SPECIAL_TOKENS)

    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    print(f"\nwrote {out}  vocab {tok.get_vocab_size():,}")
    return tok


# ------------------------------------------------------------------ check ---

def check(path: Path, sample: Path = None):
    """The acceptance gate.  Do not build shards until this is clean."""
    _require_tokenizers()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    passed = True

    def check_one(label, condition):
        nonlocal passed
        passed &= bool(condition)
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}")

    print("size")
    size = tok.get_vocab_size()
    print(f"    vocab {size:,}")
    check_one(f"exactly {VOCAB_SIZE:,}, the size the model is built for",
              size == VOCAB_SIZE)
    check_one("divisible by 128 for the lm head kernel", size % 128 == 0)
    check_one("every id fits in uint16", size <= 65_536)
    try:
        from mimans.model.model import ModelConfig
        check_one("agrees with ModelConfig.vocab", ModelConfig().vocab == size)
    except Exception:
        print("       (torch not importable here, skipping the ModelConfig cross check)")

    print("\nspecial tokens")
    for i, name in enumerate(SPECIAL_TOKENS[:6]):
        ids = tok.encode(name, add_special_tokens=False).ids
        check_one(f"{name:16s} -> {ids} at the expected id {i}",
                  ids == [i])
    glued = f"{REPO_NAME}owner/repo{FILE_SEP}import os\n{ENDOFTEXT}"
    ids = tok.encode(glued, add_special_tokens=False).ids
    check_one("markers stay atomic with no whitespace around them",
              ids[0] == 4 and ids.count(5) == 1 and ids[-1] == 0)

    print("\nround trip")
    samples = []
    if sample:
        for i, text in enumerate(documents(Path(sample))):
            samples.append(text)
            if i >= 400:
                break
    samples += [
        "def f(x):\n    return x + 1\n",
        "\tif (a && b) { return 'x\\n'; }\n",
        "# emoji \U0001f600 and accents: café, naïve, \u4e2d\u6587\n",
        "x = 0b1010 + 0x1F + 3.14159e-10\n",
        "\r\nwindows\r\nline\r\nendings\r\n",
        "".join(chr(i) for i in range(1, 256)),
    ]
    bad = [s for s in samples
           if tok.decode(tok.encode(s, add_special_tokens=False).ids,
                         skip_special_tokens=False) != s]
    check_one(f"{len(samples)} samples decode back exactly"
              f"{'' if not bad else f', {len(bad)} failed'}", not bad)

    print("\nfim, the way data.py builds it")
    rng = __import__("numpy").random.default_rng(0)
    doc = "class A:\n    def b(self):\n        return 42\n"
    transformed, did = apply_fim(doc, rng, rate=1.0)
    ids = tok.encode(transformed, add_special_tokens=False).ids
    back = tok.decode(ids, skip_special_tokens=False)
    check_one("a fim sample survives encode and decode", back == transformed)
    check_one("and still reconstructs the original file", undo_fim(back) == doc)
    check_one("all three markers are one id each",
              sum(i in (1, 2, 3) for i in ids) == 3)

    print("\nindentation")
    ids = tok.encode("    return x\n        deeper\n",
                     add_special_tokens=False).ids
    print(f"    two indented lines cost {len(ids)} tokens")
    check_one("indentation is not one token per space", len(ids) < 14)

    if samples:
        print("\nfertility")
        raw = sum(len(s.encode("utf-8")) for s in samples)
        count = sum(len(tok.encode(s, add_special_tokens=False).ids)
                    for s in samples)
        rate = raw / max(count, 1)
        print(f"    {rate:.2f} bytes per token over {raw/1e6:.1f} MB")
        print(f"    train.py assumes 3.4 for bits per byte; 40B tokens is "
              f"{40e9*rate/1e9:.0f} GB of text at this rate")
        check_one("in the range a code tokenizer should land in",
                  2.5 < rate < 5.0)

    print("\n" + ("tokenizer ok, safe to build shards"
                  if passed else "FAILURES ABOVE"))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--corpus", action="append", default=[],
                        metavar="NAME=PATH",
                        help="repeatable, e.g. --corpus code=D:/other/place. "
                             "Omit entirely to use paths.TOKENIZER_RAW/{code,text,math}.")
    parser.add_argument("--out", type=Path,
                        default=paths.TOKENIZER / "tokenizer.json")
    parser.add_argument("--budget-gb", type=float, default=2.0,
                        help="bytes of corpus to train on, split by the mixture")
    parser.add_argument("--sample", type=Path, default=paths.TOKENIZER_HOLDOUT,
                        help="held out files for the round trip and fertility checks")
    parser.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    parser.add_argument("--code-languages", default=",".join(TOP_CODE_LANGUAGES),
                        help="comma separated the-stack-smol folder names, "
                             "or 'all' to read the entire code source unfiltered")
    args = parser.parse_args()

    if args.train:
        sources = {}
        for item in args.corpus:
            if "=" not in item:
                raise SystemExit(f"--corpus wants NAME=PATH, got {item!r}")
            name, path = item.split("=", 1)
            sources[name] = Path(path)
        if not sources:
            # No --corpus given: use the standard layout directly.
            sources = {"code": paths.TOKENIZER_CODE, "text": paths.TOKENIZER_TEXT,
                      "math": paths.TOKENIZER_MATH}
            sources = {k: v for k, v in sources.items() if any(v.iterdir())}
        if "code" in sources and args.code_languages != "all":
            languages = [s.strip() for s in args.code_languages.split(",") if s.strip()]
            dirs = select_code_dirs(sources["code"], languages)
            parquets = select_code_parquets(sources["code"], languages)
            print(f"code restricted to {languages}: "
                  f"{[d.name for d in dirs]} + {[p.name for p in parquets]}")
            sources["code"] = dirs + parquets
        if not sources:
            raise SystemExit(
                f"nothing found in {paths.TOKENIZER_RAW} and no --corpus given.\n"
                f"put files under {paths.TOKENIZER_CODE}, {paths.TOKENIZER_TEXT}, "
                f"{paths.TOKENIZER_MATH}")
        weights = {k: DEFAULT_WEIGHTS.get(k, 0.0) for k in sources}
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            weights = {k: v / total for k, v in weights.items()}
            print(f"renormalised mixture to {weights}")
        train(sources, args.out, int(args.budget_gb * 1e9), weights, args.vocab)

    if args.check or not args.train:
        return check(args.out, args.sample)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
