"""Generator-from-seed weights: the inference-first thesis's Piece 3.

A linear layer's W ∈ R^{out × in} is not STORED but GENERATED at each
forward pass from a small Fibonacci-indexed seed. The seed has K
components, each contributing a separable Fibonacci-frequency mixing
of the integer position indices (i, j):

    W[i, j] = Σ_{k=1..K} [ a_k · cos(2π·F(k)·i/out) · cos(2π·F(k)·j/in)
                          + b_k · sin(2π·F(k)·i/out) · cos(2π·F(k)·j/in)
                          + c_k · cos(2π·F(k)·i/out) · sin(2π·F(k)·j/in)
                          + d_k · sin(2π·F(k)·i/out) · sin(2π·F(k)·j/in) ]

where F(k) is the k-th unique positive Fibonacci number, and the seed
is (a_k, b_k, c_k, d_k) for k = 1..K — 4K scalars per layer.

Total stored parameters per layer: 4K (regardless of in_features or
out_features). For K=16, that's 64 floats — vs 65,536 for a dense
128×128 Linear. 1024× compression.

Per-forward cost: ONE matrix construction (4K · in · out FLOPs) plus
the standard matmul (B · T · in · out FLOPs). For B·T >> 4K (typical
batch and sequence), the generator cost amortizes to negligible.

At inference: a single layer's generator can be PRECOMPUTED once and
cached, making per-token cost identical to a stored weight. The win
is storage: the cache is ephemeral, the seed is the only persistent
artifact.

This is the highest-risk piece in the transformerless thesis: whether
a model with ALL weights generated from Fibonacci bases can learn
anything useful at all. If it tanks, we know the substrate basis is
too restrictive at the weight level even though it works for positions.
If it learns even partially, we have a foothold for radical inference-
time compression.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Extended unique-positive Fibonacci table — 32 entries.
# Previous 16-entry version caused K>16 to silently clamp.
FIBONACCI = [
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578,
]


class FibGenLinear(nn.Module):
    """Drop-in replacement for nn.Linear where W is generated from a seed.

    Two generator modes:

    "separable" (the original): each component uses the SAME Fibonacci
        frequency on both axes. Generates rank-K terms.
            W[i,j] = Σ_k [a_k cos(F_k·i) cos(F_k·j) + ...]
        Seed: 4·K params.

    "cross" (new): each component uses INDEPENDENT Fibonacci frequencies
        on the two axes. Generates a full K_i × K_j grid of frequency
        pairs, so the matrix is a sum of K_i·K_j outer products of
        single-frequency 1-D bases.
            W[i,j] = Σ_{k_i, k_j} [a_{kk'} cos(F_{k_i}·i) cos(F_{k_j}·j) + ...]
        Seed: 4·K² params. Equal expressivity as separable at K_separable = K²,
        but with the substrate-canonical Fibonacci-coprime structure that
        makes the basis non-degenerate (Fibonacci frequencies are pairwise
        substrate-distinguishable).

    Args:
        in_features: input dim.
        out_features: output dim.
        K: number of Fibonacci frequencies per axis.
        mode: "separable" or "cross".
        bias: whether to include a learnable bias vector.
        init_scale: scales the seed initialization.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 16,
                 mode: str = "separable",
                 bias: bool = True, init_scale: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = min(K, len(FIBONACCI))
        if mode not in ("separable", "cross"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        n_components = self.K if mode == "separable" else self.K * self.K
        self.seed = nn.Parameter(
            torch.randn(n_components, 4) * (init_scale / max(1, math.sqrt(n_components)))
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Precompute cos/sin position·Fibonacci-frequency tables.
        i_idx = torch.arange(out_features).float()
        j_idx = torch.arange(in_features).float()
        freqs = torch.tensor(FIBONACCI[:self.K], dtype=torch.float)
        a_i = 2 * math.pi * i_idx.unsqueeze(1) * freqs.unsqueeze(0) / max(out_features, 1)
        a_j = 2 * math.pi * j_idx.unsqueeze(1) * freqs.unsqueeze(0) / max(in_features, 1)
        self.register_buffer("cos_i", torch.cos(a_i))   # [out, K]
        self.register_buffer("sin_i", torch.sin(a_i))
        self.register_buffer("cos_j", torch.cos(a_j))   # [in, K]
        self.register_buffer("sin_j", torch.sin(a_j))

    def generate_W(self) -> torch.Tensor:
        if self.mode == "separable":
            a, b, c, d = self.seed[:, 0], self.seed[:, 1], self.seed[:, 2], self.seed[:, 3]
            W = torch.einsum("ok,k,jk->oj", self.cos_i, a, self.cos_j)
            W = W + torch.einsum("ok,k,jk->oj", self.sin_i, b, self.cos_j)
            W = W + torch.einsum("ok,k,jk->oj", self.cos_i, c, self.sin_j)
            W = W + torch.einsum("ok,k,jk->oj", self.sin_i, d, self.sin_j)
            return W
        # mode == "cross": seed shape [K*K, 4], reshape to [K, K, 4]
        K = self.K
        seed = self.seed.view(K, K, 4)
        a, b, c, d = seed[..., 0], seed[..., 1], seed[..., 2], seed[..., 3]
        # W[i,j] = Σ_{k_i, k_j} [a · cos_i[i, k_i] cos_j[j, k_j] + ...]
        # einsum: cos_i [out, k_i] @ a [k_i, k_j] -> [out, k_j], then
        # · cos_j [in, k_j] -> [out, in].
        W = torch.einsum("ol,lm,jm->oj", self.cos_i, a, self.cos_j)
        W = W + torch.einsum("ol,lm,jm->oj", self.sin_i, b, self.cos_j)
        W = W + torch.einsum("ol,lm,jm->oj", self.cos_i, c, self.sin_j)
        W = W + torch.einsum("ol,lm,jm->oj", self.sin_i, d, self.sin_j)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.generate_W()
        return F.linear(x, W, self.bias)

    @property
    def n_stored_params(self) -> int:
        n = self.seed.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n

    @property
    def n_dense_equivalent_params(self) -> int:
        n = self.in_features * self.out_features
        if self.bias is not None:
            n += self.out_features
        return n


class FibGenAttention(nn.Module):
    """Single-head self-attention with all linear layers FibGen-generated."""

    def __init__(self, d_model: int, K: int = 16, mode: str = "separable"):
        super().__init__()
        self.d_model = d_model
        self.qkv = FibGenLinear(d_model, 3 * d_model, K=K, mode=mode)
        self.out = FibGenLinear(d_model, d_model, K=K, mode=mode)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return self.out(out)


class FibGenFeedForward(nn.Module):
    """FFN with FibGen-generated linear layers."""

    def __init__(self, d_model: int, expansion: int = 4, K: int = 16,
                 mode: str = "separable"):
        super().__init__()
        d_inner = d_model * expansion
        self.w1 = FibGenLinear(d_model, d_inner, K=K, mode=mode)
        self.w2 = FibGenLinear(d_inner, d_model, K=K, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class FibGenBlock(nn.Module):
    def __init__(self, d_model: int, K: int = 16, mode: str = "separable"):
        super().__init__()
        self.attn = FibGenAttention(d_model, K=K, mode=mode)
        self.ff = FibGenFeedForward(d_model, K=K, mode=mode)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class FibGenLM(nn.Module):
    """Char-level LM with EVERY linear layer FibGen-generated.

    Embedding is also FibGen: the "embedding table" is generated from
    a seed, so vocab_size × d_model storage becomes 4K · log_2(vocab)
    or similar.  For char-level vocab=65 this is a small win, but at
    LLM scale (vocab=32k+) the embedding is a major param sink.

    LM head tied to embedding (standard).
    """

    def __init__(self, vocab_size: int, d_model: int, n_blocks: int,
                 seq_len: int, K: int = 16, mode: str = "separable"):
        super().__init__()
        self.seq_len = seq_len
        self.K = K
        self.mode = mode
        self.embed_gen = FibGenLinear(vocab_size, d_model, K=K, mode=mode,
                                        bias=False)
        pe = self._crt_pe(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            FibGenBlock(d_model, K=K, mode=mode) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = FibGenLinear(d_model, vocab_size, K=K, mode=mode, bias=False)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)

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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        # Embedding via one-hot · FibGen-generated [vocab, d_model] table.
        # Equivalent to W[token_ids] for a stored embedding.
        W_emb = self.embed_gen.generate_W()        # [d_model, vocab]
        h = W_emb.t()[token_ids]                    # [B, T, d_model]
        h = h + self.pe[:T]
        mask = self.mask[:T, :T]
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_f(h)
        return self.head(h)

    def storage_summary(self) -> dict:
        """Stored param count + the dense-equivalent count."""
        stored = 0
        dense_eq = 0
        for m in self.modules():
            if isinstance(m, FibGenLinear):
                stored += m.n_stored_params
                dense_eq += m.n_dense_equivalent_params
        # Add bias/LN params (these are NOT FibGen-generated)
        for n, p in self.named_parameters():
            if "seed" in n or "bias" in n and any(
                m_name in n for m_name in ("embed_gen", "head", "qkv", "out", "w1", "w2")
            ):
                continue
            stored += p.numel()
            dense_eq += p.numel()
        return {
            "stored": stored,
            "dense_equivalent": dense_eq,
            "compression": dense_eq / max(stored, 1),
        }
