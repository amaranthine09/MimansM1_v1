"""The architecture, in the order a token moves through it.

    norm                 RMSNorm, used everywhere
    attention_tilelang   embedding lookup's first consumer: rope, gqa, flash
    swiglu_tilelang      the feed forward
    attnres              the residual gate that carries state between blocks
    block                one transformer block, three layers deep
    model                CodeModel: embedding -> 27 blocks -> tied lm head
    lm_head_tilelang     the tied head and its fused cross entropy
"""
