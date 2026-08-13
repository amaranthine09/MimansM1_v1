"""Mimans: a 514M parameter code model with hand written tilelang kernels.

    mimans.paths       every path the project writes to, all on D:
    mimans.data        the on disk token format: FIM, shards, stages, loader
    mimans.model       the architecture, embedding through to logits
    mimans.tokenizer   training the byte level BPE the model is built around
    mimans.training    the optimizer, the loop, and the preflight

paths is imported before tilelang everywhere it matters, because tilelang reads
its cache location once at import time.
"""
