"""Substrate-canonical character embedding.

Plain nn.Embedding maps each char to a learned random d-dim vector.
The substrate ops downstream operate on this scrambled mapping. We
build a substrate-canonical embedding instead: each char maps to a
Fibonacci-frequency signature that's STRUCTURED, not random.

For char index c in [0, V), the embedding at dim i is:
    embed[c, i] = sin(2 * pi * c * F(i mod K) / V)        if i even
                  cos(2 * pi * c * F(i mod K) / V)        if i odd

where F(k) are the first K Fibonacci numbers. This puts similar
chars (alphabetically adjacent, sharing substrate position) near
each other in embedding space. The substrate operations downstream
now receive canonically-placed input.

Optional learnable scale (gamma) per dim lets the model fine-tune
the relative emphasis of each Fibonacci tier while keeping the
substrate structure intact.
"""

import math

import torch
import torch.nn as nn


_FIB_NUMS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


class SubstrateEmbedding(nn.Module):
    """Substrate-canonical char embedding via Fibonacci-frequency basis.

    Each char c gets a fixed sin/cos signature at Fibonacci frequencies.
    Optionally multiplied by a learnable per-dim gamma for fine-tuning.
    Tied LM head can use the same fixed basis (transpose).
    """

    def __init__(self, vocab_size: int, d_model: int, K: int = 7,
                 learnable_gamma: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.K = K
        # Pre-compute the canonical embedding [V, d_model].
        c_idx = torch.arange(vocab_size, dtype=torch.float)
        embed = torch.zeros(vocab_size, d_model)
        for i in range(d_model):
            k = i % K
            freq = _FIB_NUMS[k]
            # alternate sin/cos across consecutive dims
            angle = 2 * math.pi * c_idx * freq / vocab_size
            if i % 2 == 0:
                embed[:, i] = torch.sin(angle)
            else:
                embed[:, i] = torch.cos(angle)
        self.register_buffer("substrate_embed", embed)
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.ones(d_model))
        else:
            self.register_buffer("gamma", torch.ones(d_model))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [...], returns [..., d_model]
        embedded = self.substrate_embed[token_ids]      # [..., d_model]
        return embedded * self.gamma

    @property
    def weight(self) -> torch.Tensor:
        """Mimics nn.Embedding.weight for tied LM head compatibility."""
        return self.substrate_embed * self.gamma
