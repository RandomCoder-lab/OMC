"""Substrate-similarity attention: L1 distance in K-dim Fibonacci basis.

Per the user's "we need an architecture that extrapolates differently,
not just compresses": standard attention's Q·K^T dot product has nothing
substrate-aware about it. This module replaces it with L1 distance in a
K-dim Fibonacci-basis signature space — the substrate's canonical
nearness metric, the same one used for attractor snapping.

The architectural claim: nearness in K-dim Fibonacci basis IS the
substrate-aligned way to ask "do these two tokens share structure?"
The dot-product operator only knows about magnitudes and orientations
in a generic Euclidean space.

Attention computation:
    sig[t]   = W_sig · x[t]                    # [K]: substrate signature
    dist[i,j] = ||sig[i] - sig[j]||_1          # L1 in Fibonacci basis
    attn[i,j] = softmax(-dist[i,j] / sqrt(K))  # nearness ~ attention

Compute cost: O(T·d·K) for the projection + O(T²·K) for the pairwise
L1 (vs O(T·d²) + O(T²·d) for dense attention). At d=4096, K=32 the
L1-score computation is 128× cheaper than dense Q·K^T.

The model uses FibGen weights too (compressed storage). So we have
SUBSTRATE COMPRESSED WEIGHTS + SUBSTRATE NATIVE OPERATOR. Two distinct
substrate properties stacked.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from models_fibgen import FibGenLinear, FIBONACCI


class SubstrateSimilarityAttention(nn.Module):
    """L1-distance attention in K-dim Fibonacci-basis signature space.

    Substrate-native at TWO levels:
      - WEIGHTS: W_sig, W_v, W_out are FibGen (Fibonacci-basis seeds, ~100x
        smaller storage than dense).
      - OPERATOR: attention scores via L1 distance in the K-dim signature
        space, NOT Q·K^T. Tokens with matching Fibonacci signatures
        attend; tokens with disparate signatures are gated out.
    """

    def __init__(self, d_model: int, K: int = 32, seq_len: int = 128,
                 fibgen_K: int = 32, mode: str = "cross",
                 lazy_tier_dropout: bool = False,
                 lazy_K_active: int = 0):
        super().__init__()
        self.d_model = d_model
        self.K = K
        kw = dict(K=fibgen_K, mode=mode, bias=False,
                   lazy_tier_dropout=lazy_tier_dropout,
                   lazy_K_active=lazy_K_active)
        self.W_sig = FibGenLinear(d_model, K, **kw)
        self.W_v = FibGenLinear(d_model, d_model, **kw)
        self.W_out = FibGenLinear(d_model, d_model, **kw)
        # Standard causal mask; substrate-distance attention is dense in
        # principle. Could also use Fibonacci-offset mask for sparsity.
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        sig = self.W_sig(x)                                  # [B, T, K]
        v = self.W_v(x)                                       # [B, T, D]
        # Pairwise L1 distance across the T axis: [B, T, T]
        diff = sig.unsqueeze(2) - sig.unsqueeze(1)            # [B, T, T, K]
        dist = diff.abs().sum(dim=-1)                          # [B, T, T]
        scores = -dist / math.sqrt(self.K)
        # Causal mask: cells where mask=0 set to -inf so softmax zeros them.
        m = self.mask[:T, :T]
        scores = scores.masked_fill(m == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return self.W_out(out)


class SubsimBlock(nn.Module):
    """Substrate-similarity attention + FibGen FFN."""

    def __init__(self, d_model: int, seq_len: int, K: int = 32,
                 fibgen_K: int = 32, mode: str = "cross",
                 lazy_tier_dropout: bool = False,
                 lazy_K_active: int = 0):
        super().__init__()
        self.attn = SubstrateSimilarityAttention(
            d_model, K=K, seq_len=seq_len, fibgen_K=fibgen_K, mode=mode,
            lazy_tier_dropout=lazy_tier_dropout, lazy_K_active=lazy_K_active,
        )
        kw = dict(K=fibgen_K, mode=mode, lazy_tier_dropout=lazy_tier_dropout,
                   lazy_K_active=lazy_K_active)
        self.w1 = FibGenLinear(d_model, 4 * d_model, **kw)
        self.w2 = FibGenLinear(4 * d_model, d_model, **kw)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.w2(F.gelu(self.w1(self.ln2(x))))
        return x


class SubsimLM(nn.Module):
    """Char-level LM with:
      - Standard learned embedding (subspace defined by the input vocabulary)
      - CRT-Fibonacci positional encoding
      - SubstrateSimilarityAttention (L1-distance in K-dim Fibonacci basis)
      - FibGen FFN weights
      - Tied LM head
    """

    def __init__(self, vocab_size: int, d_model: int, n_blocks: int,
                 seq_len: int, K: int = 32, fibgen_K: int = 32,
                 mode: str = "cross", lazy_tier_dropout: bool = False,
                 lazy_K_active: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.K = K
        self.embed = nn.Embedding(vocab_size, d_model)
        pe = self._crt_pe(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            SubsimBlock(d_model, seq_len, K=K, fibgen_K=fibgen_K, mode=mode,
                          lazy_tier_dropout=lazy_tier_dropout,
                          lazy_K_active=lazy_K_active)
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    @staticmethod
    def _crt_pe(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float)
        moduli = [5, 8, 13, 21, 34, 55, 89, 144]
        n_pairs = d_model // 2
        for i in range(n_pairs):
            m = moduli[i % len(moduli)]
            angle = 2 * math.pi * (pos % m) / m
            pe[:, 2 * i] = torch.sin(angle)
            pe[:, 2 * i + 1] = torch.cos(angle)
        return pe

    def forward(self, token_ids):
        B, T = token_ids.shape
        h = self.embed(token_ids) + self.pe[:T]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)

    def storage_summary(self):
        stored = 0
        dense_eq = 0
        for m in self.modules():
            if isinstance(m, FibGenLinear):
                stored += m.n_stored_params
                dense_eq += m.n_dense_equivalent_params
        for n, p in self.named_parameters():
            # Approximation: any param not inside a FibGen counts as itself.
            # (The embedding and LayerNorms are intentionally not compressed.)
            if not any(s in n for s in ("W_sig", "W_v", "W_out", ".w1.", ".w2.")):
                stored += p.numel()
                dense_eq += p.numel()
        return {"stored": stored, "dense_equivalent": dense_eq,
                "compression": dense_eq / max(stored, 1)}
