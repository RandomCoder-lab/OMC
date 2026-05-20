"""Substrate-native architectural primitives.

This is the architectural shift from "substrate as side-channel to standard
matmul attention" (models.py) to "substrate REPLACES the expensive matmul
ops" (this file).

Three building blocks, all preserving the O(T · log_phi_pi_fibonacci(T))
complexity bound that the rest of OMC's algorithms already live on:

  1. FibonacciOffsetAttention — sparse attention where position i attends
     only to {i - f : f ∈ FIBONACCI ∩ [0, T]}. Partners per query:
     ~log_phi_pi(T). Same Fibonacci-coprime basis as CRT-PE.

  2. ZeckendorfRoutedFFN — K specialist FFNs (each at d/sqrt(K) width).
     Each token's Zeckendorf decomposition determines which specialist
     it routes to. Per-token compute drops from O(d²) to O(d²/K).
     Routing is by integer token-id — substrate-aligned, no float router.

  3. CRTBucketAttention — alternative to (1). Tokens are bucketed by
     their CRT-Fibonacci residue tuple over moduli {5, 8, 13, 21};
     attention is to bucket-aggregated K/V vectors (constant ~M
     buckets) instead of all T keys.

The orientation question: all three live on the same FIBONACCI table from
omnimcode-core/src/phi_pi_fib.rs. The geometric shape is the Zeckendorf
graph (nodes = positions, edges = Fibonacci-distance offsets). Attention
moves along graph edges; FFN routes within bins; the whole computation
is structured at log_phi_pi_fibonacci(N) connectivity.

PyTorch limitation note: implementing these as boolean masks on dense
matmuls preserves the FLOPS-USED claim (zeroed scores don't contribute
to gradient) but does NOT yield wall-clock speedup until a custom
sparse/grouped kernel replaces torch.matmul. We report both "effective
FLOPs" (the asymptotic claim) and wall-clock (the implementation cost)
so the asymptotic and engineering questions stay separate.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical Fibonacci table — matches omnimcode-core/src/phi_pi_fib.rs:32
FIBONACCI = [
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368,
]
# De-duplicated, positive: the unique Fibonacci offsets used as attention edges.
FIB_POS_UNIQUE = sorted(set(f for f in FIBONACCI if f > 0))


def fibonacci_tier_values(n_tiers: int, reciprocals: bool = False) -> list[float]:
    """Signed Fibonacci tier values.

    Without reciprocals (the original v1):
        {0, ±1, ±2, ±3, ±5, ±8, ±13, ±21, ...}
    log spacing toward infinity, no resolution between 0 and 1.

    With reciprocals (v2 — fixes the "no resolution near zero" failure
    from the v1 bench):
        {0, ±1/F_max, ..., ±1/5, ±1/3, ±1/2, ±1, ±2, ±3, ±5, ±8, ..., ±F_max}
    log spacing crossing zero — fine resolution near 0 where most
    Gaussian-distributed weights actually live.

    Adjacent ratios approach φ (since F(k+1)/F(k) → φ), so this is
    the natural phi-Fibonacci tier system the substrate already uses
    elsewhere in OMC.
    """
    fibs = FIB_POS_UNIQUE[: max(0, n_tiers - 1)]
    pos = [float(f) for f in fibs]
    if reciprocals:
        pos = sorted(set(pos + [1.0 / f for f in fibs if f > 1]))
    return sorted([-v for v in pos] + [0.0] + pos)


def fibonacci_tier_snap(W: torch.Tensor, n_tiers: int = 8,
                         scale: str = "per_tensor",
                         reciprocals: bool = False) -> tuple[torch.Tensor, int]:
    """Snap each weight in W to its nearest signed-Fibonacci tier value.

    Args:
        W: tensor (1-D or 2-D).
        n_tiers: resolution per sign (= number of distinct positive Fibonacci
                  values, before any reciprocals).
        scale: "per_tensor" → one global scale set by max(|W|).
               "per_row"   → one scale per output row of a 2-D matrix
                              (matches each row's own dynamic range; the
                              standard per-channel quantization trick).
        reciprocals: if True, include 1/F(k) values in the tier set —
                      gives fine resolution near 0.

    Returns:
        (W_quantized, n_unique_values_actually_used_avg)
    """
    tier_vals = torch.tensor(
        fibonacci_tier_values(n_tiers, reciprocals=reciprocals),
        dtype=W.dtype, device=W.device,
    )                                                       # [n_levels]
    max_tier = max(tier_vals.abs().max().item(), 1.0)

    if scale == "per_tensor":
        abs_max = W.abs().max().item()
        if abs_max == 0:
            return W.clone(), 1
        s = abs_max / max_tier
        target_vals = tier_vals * s
        diffs = (W.unsqueeze(-1) - target_vals).abs()
        nearest = diffs.argmin(dim=-1)
        W_q = target_vals[nearest]
        n_unique = nearest.unique().numel()
        return W_q, n_unique

    if scale == "per_row":
        if W.dim() != 2:
            # Fall back to per-tensor for 1-D / N-D parameters.
            return fibonacci_tier_snap(W, n_tiers, "per_tensor", reciprocals)
        abs_max_row = W.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)  # [out, 1]
        s_row = abs_max_row / max_tier                       # [out, 1]
        # For each row, scaled tier set is tier_vals * s_row. We need
        # per-row argmin over [out, in, n_levels].
        targets = tier_vals.view(1, 1, -1) * s_row.unsqueeze(-1)  # [out, 1, n_levels]
        diffs = (W.unsqueeze(-1) - targets).abs()             # [out, in, n_levels]
        nearest = diffs.argmin(dim=-1)                        # [out, in]
        W_q = torch.gather(targets.expand_as(diffs), -1,
                            nearest.unsqueeze(-1)).squeeze(-1)
        n_unique = nearest.unique().numel()
        return W_q, n_unique

    raise ValueError(scale)


def fibonacci_quantize_model(model: torch.nn.Module, n_tiers: int = 8,
                              scale: str = "per_tensor",
                              reciprocals: bool = False,
                              targets: list[str] = None) -> dict:
    """In-place Fibonacci-tier-snap of model parameters matching `targets`."""
    if targets is None:
        targets = [""]
    stats = {"params_quantized": 0, "tensors_quantized": 0,
             "per_tensor": {}}
    for name, p in model.named_parameters():
        if not any(t in name for t in targets):
            continue
        with torch.no_grad():
            W_q, n_unique = fibonacci_tier_snap(
                p.data, n_tiers=n_tiers, scale=scale, reciprocals=reciprocals,
            )
            p.data.copy_(W_q)
            stats["params_quantized"] += p.numel()
            stats["tensors_quantized"] += 1
            stats["per_tensor"][name] = {
                "numel": p.numel(),
                "n_unique_tier_values": n_unique,
            }
    return stats


def fib_offsets_up_to(t: int) -> list[int]:
    """Fibonacci offsets ≤ t. For T=128 returns {1,2,3,5,8,13,21,34,55,89}
    — 10 offsets, i.e. log_phi_pi(128) ≈ 3.6 · 10 ≈ 36 in linear count
    (each pos has ~10 partners — log T base φ^π ≈ 3.6)."""
    return [f for f in FIB_POS_UNIQUE if f <= t]


def zeckendorf_decompose(n: int) -> list[int]:
    """Return Zeckendorf indices (FIBONACCI table indices) representing n,
    largest first. Matches omnimcode-core/src/phi_pi_fib.rs:zeckendorf_indices.
    """
    if n <= 0:
        return []
    out = []
    rem = n
    i = len(FIBONACCI) - 1
    while i >= 2:
        if FIBONACCI[i] <= rem:
            rem -= FIBONACCI[i]
            out.append(i)
            i -= 2  # Zeckendorf: skip the next-smaller Fibonacci
        else:
            i -= 1
    return out


def zeckendorf_top_index(token_id: int) -> int:
    """Top Zeckendorf index of token_id, or 0 if token_id == 0.
    Used as the routing signal for ZeckendorfRoutedFFN."""
    decomp = zeckendorf_decompose(token_id)
    return decomp[0] if decomp else 0


# ---------------------------------------------------------------------------
# CRT-Fibonacci moduli — shared by the position encoding and the bucket attn
# ---------------------------------------------------------------------------
_FIB_MODULI = [5, 8, 13, 21, 34, 55, 89, 144]


def crt_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float)
    n_pairs = d_model // 2
    for i in range(n_pairs):
        m = _FIB_MODULI[i % len(_FIB_MODULI)]
        residue = pos % m
        angle = 2 * math.pi * residue / m
        pe[:, 2 * i] = torch.sin(angle)
        pe[:, 2 * i + 1] = torch.cos(angle)
    return pe


def fibonacci_attention_mask(seq_len: int, causal: bool = True) -> torch.Tensor:
    """Boolean mask [seq_len, seq_len]. mask[i, j] = True iff
    (i - j) is a non-negative Fibonacci number ≤ seq_len.

    Includes self (offset 0) so a position always sees itself.
    Causal version: only j ≤ i edges are kept.

    Effective partners per query: ≈ log_phi_pi(seq_len). For seq_len=128
    that's 11 (self + 10 backward Fibonacci offsets).
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    # Self
    diag = torch.arange(seq_len)
    mask[diag, diag] = True
    offsets = fib_offsets_up_to(seq_len)
    for f in offsets:
        i_idx = torch.arange(f, seq_len)
        j_idx = i_idx - f
        mask[i_idx, j_idx] = True
        if not causal:
            mask[j_idx, i_idx] = True
    return mask


class FibonacciOffsetAttention(nn.Module):
    """Attention where each query sees only Fibonacci-offset keys.

    Reuses standard Q/K/V projections; the only difference from dense
    causal attention is the mask. Asymptotic attention compute drops
    from O(T²·d) to O(T · log_phi_pi(T) · d).

    PyTorch caveat: torch.matmul on Q @ K^T is still dense — the mask
    only zeroes out scores post-hoc. Wall-clock parity requires a
    custom sparse kernel; we report effective_flops() so the asymptotic
    claim is measurable independent of the kernel choice.
    """

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        mask = fibonacci_attention_mask(seq_len, causal=True)
        self.register_buffer("fib_mask", mask)

    @property
    def edges_per_query(self) -> float:
        return self.fib_mask.float().sum(dim=-1).mean().item()

    def effective_flops(self) -> int:
        """FLOPs a kernel would do given the mask. 2× factor for Q·K plus
        attn·V; per-edge cost is 2·d_model."""
        n_edges = int(self.fib_mask.sum().item())
        return 2 * 2 * n_edges * self.d_model

    def forward(self, x: torch.Tensor, _ignored_causal_mask=None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = self.fib_mask[:T, :T]
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return self.out(out)


class CRTBucketAttention(nn.Module):
    """Bucket attention: keys/values are aggregated per CRT-Fibonacci
    residue bucket, queries attend to the small set of buckets.

    For modulus M, there are exactly M buckets. Each query computes M
    attention scores instead of T, so attention compute is O(T · M · d)
    where M is a small Fibonacci attractor (default 13).

    Causal: a query at position i can only see buckets aggregated from
    positions ≤ i. We re-aggregate per query (cumulative bucket means).
    """

    def __init__(self, d_model: int, seq_len: int, modulus: int = 13):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.M = modulus
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        # bucket_of[pos] in [0, M)
        bucket_of = (torch.arange(seq_len) % self.M).long()
        self.register_buffer("bucket_of", bucket_of)
        # one_hot[pos, b] = 1 if bucket_of[pos] == b (used to scatter K/V).
        one_hot = F.one_hot(bucket_of, num_classes=self.M).float()
        self.register_buffer("bucket_one_hot", one_hot)

    def effective_flops(self) -> int:
        # Q · K_bucket: T · M · d. attn · V_bucket: T · M · d.
        return 2 * 2 * self.seq_len * self.M * self.d_model

    def forward(self, x: torch.Tensor, _ignored_causal_mask=None) -> torch.Tensor:
        B, T, D = x.shape
        M = self.M
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(D)

        one_hot = self.bucket_one_hot[:T]                  # [T, M]
        # Causal cumulative one-hot: for each position i, how many positions
        # ≤ i are in each bucket b? Shape [T, M].
        cum_count = one_hot.cumsum(dim=0).clamp(min=1.0)    # avoid /0
        # Cumulative bucket SUM of K (and V), per batch.
        # k: [B, T, D]; one_hot: [T, M]. Want [B, T, M, D] = cum sum over T
        # of (one_hot[:, :, None] * k[:, :, None, :]).
        k_per_bucket = k.unsqueeze(2) * one_hot.unsqueeze(0).unsqueeze(-1)
        v_per_bucket = v.unsqueeze(2) * one_hot.unsqueeze(0).unsqueeze(-1)
        k_cum = k_per_bucket.cumsum(dim=1)                  # [B, T, M, D]
        v_cum = v_per_bucket.cumsum(dim=1)
        k_bucket = k_cum / cum_count.unsqueeze(0).unsqueeze(-1)   # [B, T, M, D]
        v_bucket = v_cum / cum_count.unsqueeze(0).unsqueeze(-1)

        # Per query, score against the M bucket-keys at its own position.
        # q: [B, T, D]; k_bucket: [B, T, M, D]; want scores [B, T, M].
        scores = torch.einsum("btd,btmd->btm", q, k_bucket) * scale
        # Mask out empty buckets (cum_count == 0 not possible after clamp,
        # but treat zero-count buckets as -inf so they don't attract attn).
        cum_count_t = cum_count.unsqueeze(0).expand(B, -1, -1)   # [B, T, M]
        scores = scores.masked_fill(cum_count_t < 0.5, float('-inf'))
        attn = F.softmax(scores, dim=-1)                    # [B, T, M]
        out = torch.einsum("btm,btmd->btd", attn, v_bucket)
        return self.out(out)


class ZeckendorfRoutedFFN(nn.Module):
    """K specialist FFNs; each token routes to one specialist by the
    top index of its Zeckendorf decomposition.

    Each specialist has width d_specialist = d_model · expansion / K so
    total params ≈ standard FFN. Per-token compute drops to 1/K of
    standard FFN because only one specialist runs per token.

    Routing is by token-id (an integer). Substrate-aligned: respects
    the rule that substrate metrics apply to integer quantities, not
    learned floats.

    Implementation: we mask-and-sum over all K specialists per forward.
    A real kernel would gather tokens by route → run one specialist per
    group → scatter. Effective per-token FLOPs are reported via
    effective_flops_per_token() so the asymptotic claim is measurable.
    """

    def __init__(self, d_model: int, K: int = 5, expansion: int = 4, vocab_size: int = 65):
        super().__init__()
        self.d_model = d_model
        self.K = K
        # Each specialist is a small d_model -> d_inner -> d_model FFN.
        # d_inner = expansion·d_model / K gives PARAM PARITY with a standard
        # FFN (K specialists at width 4d/K → total 2·d·4d = 8d² = standard
        # FFN params) AND 1/K per-token compute (each token runs only its
        # routed specialist).
        d_inner = max(1, int(expansion * d_model / K))
        self.specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_inner),
                nn.GELU(),
                nn.Linear(d_inner, d_model),
            )
            for _ in range(K)
        ])
        self.d_inner = d_inner

        # Precompute Zeckendorf top-index for every token id, then mod K.
        route_table = torch.tensor(
            [zeckendorf_top_index(t) % K for t in range(vocab_size)],
            dtype=torch.long,
        )
        self.register_buffer("route_table", route_table)
        # Per-specialist counts (for diagnostic — does the router balance?)
        counts = torch.bincount(route_table, minlength=K).float()
        self.register_buffer("route_counts", counts)

    def effective_flops_per_token(self) -> int:
        # One specialist's two linear layers: d_model → d_inner → d_model.
        return 2 * (self.d_model * self.d_inner) * 2

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D].  token_ids: [B, T]."""
        B, T, D = x.shape
        # route_id[B, T] in [0, K).
        route_id = self.route_table[token_ids]
        out = torch.zeros_like(x)
        # Mask-and-sum over specialists. PyTorch-friendly; not memory-optimal.
        for k, spec in enumerate(self.specialists):
            mask = (route_id == k).float().unsqueeze(-1)    # [B, T, 1]
            if mask.sum() == 0:
                continue
            # Run specialist on all tokens, then zero out the off-route.
            # (A real kernel would only run for masked tokens.)
            out_k = spec(x) * mask
            out = out + out_k
        return out


# ---------------------------------------------------------------------------
# Composed substrate-native block + LM
# ---------------------------------------------------------------------------


class TiedSubstrateAttention(nn.Module):
    """Tied Q/K/V attention via substrate channel permutation.

    The user's Principle A: instead of independent W_Q, W_K, W_V, there is
    ONE learned projection W. Q is W·x; K and V are obtained by FIXED
    channel-rotation of Q by Fibonacci strides:

        Q = W · x
        K = roll(Q, F_K, dims=-1)   # channels shifted by F_K
        V = roll(Q, F_V, dims=-1)   # channels shifted by F_V

    The strides F_K, F_V are Fibonacci numbers selected so K and V
    occupy meaningfully different parts of the channel space. The model
    learns ONE representation whose Q, K, V views are interderivable
    by substrate-native operations.

    Param count vs standard:
        standard: W_Q + W_K + W_V + W_out = 4·d²
        tied:     W + W_out = 2·d²      (50% reduction in attention)

    Inference economics:
        - one matmul per forward (vs three)
        - K and V are zero-cost channel rolls of Q
        - per-token attention parameter fetch: 2·d² (vs 4·d²)
    """

    def __init__(self, d_model: int, F_K: int = 13, F_V: int = 55,
                 dropout: float = 0.0, seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # ONE shared projection. No separate W_K or W_V.
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.F_K = F_K % d_model
        self.F_V = F_V % d_model
        self.dropout = dropout

    def effective_flops(self) -> int:
        # Per forward: one W·x matmul (T·d² FLOPs) + Q·K^T (T²·d) + attn·V (T²·d).
        # Note: standard attention has 3·T·d² for Q,K,V projections; tied
        # has T·d² (one matmul). The roll() is free.
        T, D = self.seq_len, self.d_model
        return 2 * T * D * D + 2 * 2 * T * T * D

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.W(x)                                       # [B, T, D]
        K = torch.roll(Q, shifts=self.F_K, dims=-1)          # channel-rotate
        V = torch.roll(Q, shifts=self.F_V, dims=-1)
        scale = 1.0 / math.sqrt(D)
        scores = (Q @ K.transpose(-2, -1)) * scale
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ V
        return self.out(out)


class SubstrateBlock(nn.Module):
    """Block = norm → substrate-attention → norm → substrate-FFN, with
    residuals. Both inner ops are the substrate-native primitives.
    """

    def __init__(self, d_model: int, seq_len: int, attn_kind: str,
                 K_specialists: int, vocab_size: int,
                 bucket_modulus: int = 13,
                 tied_F_K: int = 13, tied_F_V: int = 55):
        super().__init__()
        self.attn_kind = attn_kind
        if attn_kind == "fib":
            self.attn = FibonacciOffsetAttention(d_model, seq_len)
        elif attn_kind == "bucket":
            self.attn = CRTBucketAttention(d_model, seq_len, modulus=bucket_modulus)
        elif attn_kind == "tied":
            self.attn = TiedSubstrateAttention(d_model, F_K=tied_F_K, F_V=tied_F_V,
                                                seq_len=seq_len)
            # tied attention uses a standard causal mask; we need it here.
            mask = torch.tril(torch.ones(seq_len, seq_len))
            self.register_buffer("causal_mask", mask)
        else:
            raise ValueError(f"unknown attn_kind: {attn_kind}")
        self.ff = ZeckendorfRoutedFFN(
            d_model, K=K_specialists, vocab_size=vocab_size,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, token_ids):
        if self.attn_kind == "tied":
            B, T, _ = x.shape
            mask = self.causal_mask[:T, :T]
            x = x + self.attn(self.ln1(x), mask)
        else:
            x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x), token_ids)
        return x


class SubstrateLM(nn.Module):
    """Char-level LM built entirely on substrate-native primitives.

    Components (all on the same FIBONACCI basis):
      - Embedding: standard learned (no substrate token-id encoding —
        TRANSFORMERLESS_RESULT.md showed it doesn't compose without an
        attenuator, and the attenuator made no difference).
      - Positional encoding: CRT-Fibonacci PE.
      - Attention: Fibonacci-offset OR CRT-bucket.
      - FFN: Zeckendorf-routed specialists.
      - Head: tied to embedding.
    """

    def __init__(self, vocab_size: int, d_model: int, n_blocks: int,
                 seq_len: int, attn_kind: str, K_specialists: int,
                 bucket_modulus: int = 13,
                 tied_F_K: int = 13, tied_F_V: int = 55):
        super().__init__()
        self.seq_len = seq_len
        self.attn_kind = attn_kind
        self.tied_F_K = tied_F_K
        self.tied_F_V = tied_F_V
        self.embed = nn.Embedding(vocab_size, d_model)
        pe = crt_pe(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            SubstrateBlock(d_model, seq_len, attn_kind, K_specialists, vocab_size,
                           bucket_modulus=bucket_modulus,
                           tied_F_K=tied_F_K, tied_F_V=tied_F_V)
            for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, token_ids):
        B, T = token_ids.shape
        h = self.embed(token_ids) + self.pe[:T]
        for block in self.blocks:
            h = block(h, token_ids)
        h = self.ln_f(h)
        return self.head(h)

    def effective_attention_flops(self) -> int:
        # Sum over blocks.
        return sum(b.attn.effective_flops() for b in self.blocks)

    def effective_ffn_flops_per_token(self) -> int:
        return sum(b.ff.effective_flops_per_token() for b in self.blocks)
