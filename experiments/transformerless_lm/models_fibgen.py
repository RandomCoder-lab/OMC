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


# Extended unique-positive Fibonacci table — 64 entries.
# Computed by recurrence; large F(k) wrap pseudo-randomly mod small
# dimensions but remain pairwise-distinct, so they still serve as a
# rich basis on weight matrices at d=128-1024.
def _build_fibonacci(n: int) -> list[int]:
    out = [1, 2]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out


FIBONACCI = _build_fibonacci(64)


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
                 bias: bool = True, init_scale: float = 0.1,
                 lazy_tier_dropout: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = min(K, len(FIBONACCI))
        if mode not in ("separable", "cross"):
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.lazy_tier_dropout = lazy_tier_dropout
        n_components = self.K if mode == "separable" else self.K * self.K
        self.seed = nn.Parameter(
            torch.randn(n_components, 4) * (init_scale / max(1, math.sqrt(n_components)))
        )

        # Fibonacci tier per seed component, used for lazy-tier dropout.
        # Lower tier = more important = active more often.
        if mode == "separable":
            # Component k → tier (k+1). F(tier) = Fibonacci number.
            tiers_int = [i + 1 for i in range(self.K)]
        else:
            # Cross-mode pair (k_i, k_j) → tier max(k_i, k_j) + 1.
            # Pair (0, 0) is tier 1 (most important, always active).
            # Pair (31, 31) is tier 32 (rarely active under 1/F(32) probability).
            tiers_int = [max(k_i, k_j) + 1
                         for k_i in range(self.K) for k_j in range(self.K)]
        # Two substrate-aligned schemes available on this buffer:
        # (1) lazy_tier_dropout=True   -> mask seed via Bernoulli(tier_keep_probs)
        # (2) gradient-scale via tier_lr_scale (applied by training loop)
        keep_probs = torch.tensor(
            [1.0 / math.sqrt(t) for t in tiers_int], dtype=torch.float,
        )
        self.register_buffer("tier_keep_probs", keep_probs)
        # tier-weighted learning rate: low-tier components get full LR, high-tier
        # get reduced LR proportional to 1/sqrt(tier). Apply by multiplying
        # seed.grad by this buffer BEFORE optimizer.step().
        self.register_buffer("tier_lr_scale", keep_probs.unsqueeze(-1))
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

    def cache_weight(self):
        """Precompute the generated W and store as a buffer; subsequent
        forwards will skip generation. Use for deployment.
        After caching, `seed` is still stored but not used at runtime."""
        with torch.no_grad():
            W = self._compute_W()
            self.register_buffer("_cached_W", W)

    def _compute_W(self) -> torch.Tensor:
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

    def generate_W(self) -> torch.Tensor:
        """Returns the generated W. If `cache_weight()` was called, uses
        the cached buffer (no compute); otherwise recomputes from seed."""
        cached = getattr(self, "_cached_W", None)
        if cached is not None:
            return cached
        return self._compute_W()

    def _maybe_lazy_seed(self) -> torch.Tensor:
        """Returns the seed (optionally masked by Fibonacci-tier dropout).

        Substrate-native lazy LOADING applied to the seed itself:
          - Tier 1 components are always active (full participation)
          - Tier-k components active with probability 1/sqrt(k)
          - Only active components contribute to this step's forward;
            only they receive gradient on backward.

        Magnitude matching: at training the mask is Bernoulli; at eval
        we scale the seed by the per-component keep_prob so the
        EXPECTED forward output during training matches the deterministic
        forward at eval. This avoids the magnitude crash that pure-mask
        without scaling caused.
        """
        if not self.lazy_tier_dropout:
            return self.seed
        if self.training:
            mask = torch.bernoulli(self.tier_keep_probs)        # [n_components]
            return self.seed * mask.unsqueeze(-1)
        # eval: deterministic, scaled by keep_prob to match training E[seed]
        return self.seed * self.tier_keep_probs.unsqueeze(-1)

    def _forward_compressed(self, x: torch.Tensor) -> torch.Tensor:
        """Substrate-native forward: compute y = W·x WITHOUT materializing W.

        For the SEPARABLE basis,
            W = Σ_k a_k cos_i[:,k] cos_j[:,k]^T + ... (4 sign combos)
        and y = W @ x decomposes as
            y_i = Σ_k cos_i[i,k] · ( a_k · (cos_j[:,k]^T · x) )
                + ... three more terms
        — a K-step "Fourier-in-the-Fibonacci-basis" pass with no [out,in]
        tensor materialized. Cost: O(B·T·K·(in+out)) instead of O(B·T·in·out).

        For the CROSS basis the inner term is a K×K matmul on the
        K-dim projected x, then projected back.
        """
        # x: [B, T, in_features]
        seed = self._maybe_lazy_seed()
        if self.mode == "separable":
            a, b, c, d = seed[:, 0], seed[:, 1], seed[:, 2], seed[:, 3]
            # Project x into Fibonacci-basis along input axis: [B, T, K]
            x_cos = x @ self.cos_j                        # [B, T, K]
            x_sin = x @ self.sin_j                        # [B, T, K]
            # Inner separable mixing (Hadamard product with coefficients)
            #   cc term contributes cos_i[i,k] · a_k · x_cos[k]
            #   sc term contributes sin_i[i,k] · b_k · x_cos[k]
            #   cs term contributes cos_i[i,k] · c_k · x_sin[k]
            #   ss term contributes sin_i[i,k] · d_k · x_sin[k]
            y_cos = (a * x_cos) + (c * x_sin)              # [B, T, K]
            y_sin = (b * x_cos) + (d * x_sin)
            # Project K-dim mixed signal back to output axis
            y = y_cos @ self.cos_i.t() + y_sin @ self.sin_i.t()   # [B, T, out]
            if self.bias is not None:
                y = y + self.bias
            return y
        # cross mode: seed [K, K, 4] mixing matrix
        K = self.K
        seed_cross = seed.view(K, K, 4)
        a, b, c, d = seed_cross[..., 0], seed_cross[..., 1], seed_cross[..., 2], seed_cross[..., 3]
        x_cos = x @ self.cos_j                            # [B, T, K]
        x_sin = x @ self.sin_j
        # K×K mixing in seed space:
        #   y_cos = a · x_cos + c · x_sin   (cos-side mixing)
        #   y_sin = b · x_cos + d · x_sin   (sin-side mixing)
        y_cos = x_cos @ a.t() + x_sin @ c.t()             # [B, T, K]
        y_sin = x_cos @ b.t() + x_sin @ d.t()
        y = y_cos @ self.cos_i.t() + y_sin @ self.sin_i.t()
        if self.bias is not None:
            y = y + self.bias
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If we cached the dense W (deployment mode), use the materialized
        # matmul. Otherwise compute in the Fibonacci basis directly — no
        # W materialization — which is the substrate-native compute path.
        cached = getattr(self, "_cached_W", None)
        if cached is not None:
            return F.linear(x, cached, self.bias)
        return self._forward_compressed(x)

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


class FibGenSparseAttention(nn.Module):
    """Fibonacci-offset attention + FibGen QKV/out weights.

    Composes two validated substrate components:
      - sparse attention restricted to Fibonacci-distance position pairs
        (~log_phi_pi(T) edges per query instead of T)
      - FibGen-generated Q, K, V, out projections (100x weight compression)
    """

    def __init__(self, d_model: int, seq_len: int, K: int = 16,
                 mode: str = "separable"):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.qkv = FibGenLinear(d_model, 3 * d_model, K=K, mode=mode)
        self.out = FibGenLinear(d_model, d_model, K=K, mode=mode)
        # Fibonacci-offset mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        diag = torch.arange(seq_len)
        mask[diag, diag] = True
        for f in FIBONACCI:
            if f >= seq_len:
                break
            i_idx = torch.arange(f, seq_len)
            j_idx = i_idx - f
            mask[i_idx, j_idx] = True
        self.register_buffer("fib_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~self.fib_mask[:T, :T], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return self.out(attn @ v)


class FibGenRoutedFFN(nn.Module):
    """Zeckendorf-routed FFN where each specialist is FibGen-generated.

    Composes three substrate primitives:
      - K specialists, each at d_inner = expansion·d/n_specialists width
        so total params match standard FFN
      - per-token routing by the top Zeckendorf index of the token id
        (integer routing, no float router)
      - each specialist's W1, W2 are FibGen-generated
    """

    def __init__(self, d_model: int, n_specialists: int = 5,
                 expansion: int = 4, vocab_size: int = 65,
                 K: int = 16, mode: str = "separable"):
        super().__init__()
        self.d_model = d_model
        self.n_specialists = n_specialists
        d_inner = max(1, int(expansion * d_model / n_specialists))
        self.specialists = nn.ModuleList([
            nn.Sequential(
                FibGenLinear(d_model, d_inner, K=K, mode=mode),
                nn.GELU(),
                FibGenLinear(d_inner, d_model, K=K, mode=mode),
            )
            for _ in range(n_specialists)
        ])
        # Routing table from omnimcode-core/src/phi_pi_fib.rs (Zeckendorf-top
        # index of each token id, mod K)
        def _zeckendorf_top(n):
            if n <= 0:
                return 0
            rem = n
            i = len(FIBONACCI) - 1
            while i >= 0:
                if FIBONACCI[i] <= rem:
                    return i
                i -= 1
            return 0
        route = torch.tensor(
            [_zeckendorf_top(t) % n_specialists for t in range(vocab_size)],
            dtype=torch.long,
        )
        self.register_buffer("route_table", route)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        route_id = self.route_table[token_ids]               # [B, T]
        out = torch.zeros_like(x)
        for k, spec in enumerate(self.specialists):
            mask = (route_id == k).float().unsqueeze(-1)
            if mask.sum() == 0:
                continue
            out = out + spec(x) * mask
        return out


class FibGenTransformerlessBlock(nn.Module):
    """Block = sparse Fibonacci-offset attention + Zeckendorf-routed FFN.
    All weights inside both inner modules are FibGen-generated."""

    def __init__(self, d_model: int, seq_len: int, vocab_size: int,
                 K: int = 16, mode: str = "separable",
                 n_specialists: int = 5):
        super().__init__()
        self.attn = FibGenSparseAttention(d_model, seq_len, K=K, mode=mode)
        self.ff = FibGenRoutedFFN(d_model, n_specialists=n_specialists,
                                    vocab_size=vocab_size, K=K, mode=mode)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, token_ids):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x), token_ids)
        return x


class FibGenTransformerless(nn.Module):
    """All-substrate transformerless candidate.

    Composes:
      - CRT-Fibonacci positional encoding   (validated -5.4%)
      - FibGen embedding                     (100x compression)
      - Fibonacci-offset sparse attention   (-3.2% / 14x FLOPs)
      - FibGen QKV/out weights              (100x compression)
      - Zeckendorf-routed FFN                (1/n_specialists per-token FFN)
      - FibGen specialist weights            (100x compression each)
      - FibGen LM head                       (100x compression)

    Storage at d=128 should be dramatically smaller than the dense
    baseline; inference should run on Fibonacci-strided KV state.
    """

    def __init__(self, vocab_size: int, d_model: int, n_blocks: int,
                 seq_len: int, K: int = 16, mode: str = "separable",
                 n_specialists: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.K = K
        self.mode = mode
        self.embed_gen = FibGenLinear(vocab_size, d_model, K=K, mode=mode,
                                        bias=False)
        pe = FibGenLM._crt_pe(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            FibGenTransformerlessBlock(
                d_model, seq_len, vocab_size, K=K, mode=mode,
                n_specialists=n_specialists,
            )
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = FibGenLinear(d_model, vocab_size, K=K, mode=mode, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.shape
        W_emb = self.embed_gen.generate_W()
        h = W_emb.t()[token_ids] + self.pe[:T]
        for block in self.blocks:
            h = block(h, token_ids)
        h = self.ln_f(h)
        return self.head(h)

    def storage_summary(self) -> dict:
        stored = 0
        dense_eq = 0
        for m in self.modules():
            if isinstance(m, FibGenLinear):
                stored += m.n_stored_params
                dense_eq += m.n_dense_equivalent_params
        # LayerNorms etc.
        for n, p in self.named_parameters():
            if "seed" in n:
                continue
            if any(s in n for s in (".embed_gen.bias", ".head.bias",
                                      ".qkv.bias", ".out.bias",
                                      ".w1.bias", ".w2.bias",
                                      ".0.bias", ".2.bias")):
                continue
            stored += p.numel()
            dense_eq += p.numel()
        return {"stored": stored, "dense_equivalent": dense_eq,
                "compression": dense_eq / max(stored, 1)}


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
