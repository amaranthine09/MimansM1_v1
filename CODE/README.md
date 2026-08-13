# Mimans

A 514,345,526 parameter code model with hand written tilelang kernels, sized for a
40B token run on one RTX 5090.

```
CODE/
  mimans/
    paths.py                 every path the project writes to, all on D:
    data.py                  the on disk contract: FIM, repo packing, shards,
                             stages, loader.  shared by tokenizer and training
    model/                   the architecture, embedding through to logits
      norm.py                RMSNorm
      attention_tilelang.py  rope, gqa, flash attention kernel
      swiglu_tilelang.py     fused feed forward
      attnres.py             the residual gate carrying state between blocks
      block.py               one transformer block, three layers deep
      model.py               CodeModel: embedding -> 27 blocks -> tied head
      lm_head_tilelang.py    tied head and fused cross entropy
    tokenizer/
      train_tokenizer.py     trains and checks the 49152 vocab byte level BPE
    training/
      optim.py               Muon for matrices, AdamW for the rest, WSD schedule
      train.py               TrainConfig, Trainer, the loop
      preflight.py           the gate before committing three weeks of gpu time
  tools/
    tilelang_smoke.py        standalone tilelang gemm demo, not part of the model
```

Everything is run as a module, from this directory:

```
python -m mimans.model.model                    # architecture self check
python -m mimans.model.attention_tilelang --gpu # kernel against a reference
python -m mimans.training.train                 # schedule arithmetic
python -m mimans.training.preflight --quick     # end to end, seconds
python -m mimans.training.preflight             # end to end, real config
python -m mimans.tokenizer.train_tokenizer --check
python -m mimans.paths                          # where everything gets written
```

Every module has a `check()` that runs when you execute it. They are the
documentation that cannot go stale.

## Two things about this machine

**Write to D:.** C: has under 11 GB free and a single checkpoint is 3.3 GB.
`mimans/paths.py` sends checkpoints, shards, the tuned kernel cache and scratch
to `D:\mimans`. Override with `MIMANS_ROOT`. It has to be imported before
tilelang, which is why the kernel modules import it on their first line.

**Tuned kernels are not cached across processes.** tvm reads a target triple
from a host compiler's `-dumpmachine` and there is no clang or gcc on PATH, so
every process retunes. Correctness is unaffected. Installing LLVM and setting
`CXX` would fix it.
