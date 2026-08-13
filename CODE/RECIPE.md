# Training recipe

Every number here is read out of the config dataclasses, not chosen in this
document. If they disagree, the code is right and this file is stale.

    python -m mimans.training.train      # prints and checks the arithmetic

## Budget

| | |
|---|---|
| tokens | 40B |
| optimizer steps | 76,293 |
| global batch | 524,288 tokens |
| micro batch | 8 x 2048, accumulated 32x |
| parameters | 514,345,526 |
| compute | 3.509 GF/token at seq 2048 |

The global batch is held at 524,288 tokens for the whole run. When the context
length changes the micro batch moves inversely (8x2048 becomes 2x8192) so both
the batch and the activation memory stay put.

## Model

27 layers in 9 blocks of 3, d_model 1280, 10 query heads and 2 kv heads of
dim 128 (GQA, 5 queries per kv head), SwiGLU with hidden 3328, RMSNorm
throughout, rope on q and k, vocab 49,152 tied between embedding and head.

Initialisation: everything reading from the residual stream gets
`normal_(std=0.02)`; the two output projections (`attn.wo`, `ffn.w_down`) are
zeroed, so every block starts as an exact identity and the stack passes
embeddings through untouched. AttnRes gate queries start at zero, so the
softmax over residual snapshots is uniform at init. All RMSNorm gains start
at one.

The head scales logits by `1/sqrt(d_model)`. Without it a tied head over an
identity stack starts out certain that the answer is the token it was just
given, and step 0 costs 25.3 nats instead of log(vocab) = 10.80.

## Optimizer

Two groups, split by shape:

| | matrices (dim >= 2, not embedding) | everything else |
|---|---|---|
| optimizer | Muon | AdamW |
| lr | 0.02 | 0.003 |
| momentum / betas | 0.95, nesterov | (0.9, 0.95), eps 1e-8 |
| weight decay | 0.1 | 0.0 |
| tensors | 21 | 32 |

Muon runs 5 Newton-Schulz steps to push the singular values of the update
toward one, then scales by `0.2 * sqrt(max(fan_in, fan_out))`. Gradients are
clipped to global norm 1.0 across both groups before either steps.

The embedding is deliberately in the AdamW group, not Muon: it is a lookup
table, and orthogonalising it is meaningless.

## Learning rate schedule

Warmup-stable-decay, one factor applied to both optimizers.

| phase | steps | factor |
|---|---|---|
| warmup | 0 to 2,000 | linear 0 -> 1 |
| stable | 2,000 to 68,663 | 1.0 |
| decay | 68,663 to 76,293 | `1 - sqrt(t)` to 0 |

Warmup is 2.6% of the run, decay the last 10% (~4B tokens), which is where
the annealing data mix lands.

## Stages

| stage | tokens | seq | rope theta | code / text / math |
|---|---|---|---|---|
| foundation | 24B | 2048 | 10,000 | 60 / 30 / 10 |
| code_spec | 12B | 2048 | 10,000 | 75 / 15 / 10 |
| long_ctx | 2B | 8192 | 500,000 | 80 / 15 / 5 |
| anneal | 2B | 8192 | 500,000 | 70 / 20 / 10 |

Theta changes exactly when the context does, on the rope table shared by all
27 blocks.

## Tokenizer

Byte level BPE, 49,152 vocab, trained on 6 GB at the foundation mixture
(code 3.6 / text 1.8 / math 0.6), five languages: python, javascript,
typescript, java, c++. Digits split individually, no normalizer, all 256
bytes in the alphabet so any fragment round trips.

Ids 0-5 are `<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`,
`<|fim_suffix|>`, `<|repo_name|>`, `<|file_sep|>`, then `<|pad|>` and 9
reserved slots.

**3.79 bytes per token**, measured on held out data. 40B tokens is therefore
about 152 GB of raw text.

## Running it

The run is split into sessions rather than one continuous stretch:

    python -m mimans.training.tune              # once, per gpu
    python -m mimans.training.session --hours 16
    python -m mimans.training.session --status

Each session resumes from the last checkpoint, trains until its time budget
is spent, checkpoints every 30 minutes, and always ends on a clean step
boundary. At 18,000 tokens/s a 16 hour session is roughly 2,000 steps, so the
full run is about 38 sessions.

Kernels are tuned once and remembered in `cache/tuned_kernels.json`, keyed by
gpu and shape. tilelang's own disk cache does not work on this machine, so
without that store every session would re-tune for over an hour before
training anything.

## What is verified

- every module self check, and the three kernels against reference math
- the logsumexp kernel, after fixing a warp specialisation bug that silently
  kept only the last vocab tile
- checkpoint save and resume, across separate processes
- the tokenizer, 21 acceptance checks including FIM round trips

## What is not

- **throughput and MFU at full scale.** The preflight has never completed a
  run at 27 layers. The 18,000 tokens/s target is an assumption.
- **evaluation is not wired in.** `eval_every` exists in the config and
  `Trainer.evaluate` exists, but nothing calls it. As written the run
  reports training loss only, with no held out number.
- **the shards do not exist.** Nothing converts the tokenizer plus raw text
  into the uint16 shards the loader reads.
- **8192 stride shards.** `Loader` reads one stride, and the long context
  stage needs a second set. This stops the run at 36B tokens.
- **bits per byte is wrong.** `TrainConfig` still assumes 3.4 bytes per
  token; the tokenizer measures 3.79.
