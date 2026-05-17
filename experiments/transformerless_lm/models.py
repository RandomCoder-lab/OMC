"""Three model architectures for the transformerless-LM bench.

All three share:
- Token embedding (d_model)
- N transformer blocks
- LM head tied to embedding
- Same parameter count (within rounding)

They differ ONLY in:
- Positional encoding (sinusoidal vs CRT-Fibonacci)
- Attention scoring (pure softmax vs softmax × HBit-tension gate)

Architectures:
    standard:   sinusoidal PE  + pure softmax attention
    crt_only:   CRT-Fib PE     + pure softmax attention
    hybrid:     CRT-Fib PE     + softmax × HBit-tension gate
                (this is the proposed transformerless-LM candidate)

A fourth, "harmonic_only" (CRT-Fib PE + substrate attention from
experiment 11) is omitted because experiment 11 showed substrate
attention loses architecturally — no point training it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encodings
# ---------------------------------------------------------------------------

def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Classical Vaswani-style PE. Returns [seq_len, d_model]."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# Fibonacci attractors used as CRT moduli. Pairwise coprime; any
# subset of size d_model/some_chunk is fine. We use 5, 8, 13, 21 as
# the "small" set (period 10920) and 34, 55, 89, 144 as the "large"
# set (period ~24M) — combined they give 8 channels.
_FIB_MODULI = [5, 8, 13, 21, 34, 55, 89, 144]


def crt_pe(seq_len: int, d_model: int) -> torch.Tensor:
    """Harmonic CRT-style PE: pos mod Fibonacci-attractor for each
    channel. Pairs each modulus with a sin/cos pair so the value is
    smooth (the residue itself is integer-stepped, which gives a
    poor gradient signal; we project residue to a sin/cos pair on
    a 2π * residue / modulus circle so the encoding is differentiable
    through the embedding distance metric).

    Returns [seq_len, d_model].
    """
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float)
    n_pairs = d_model // 2
    for i in range(n_pairs):
        m = _FIB_MODULI[i % len(_FIB_MODULI)]
        residue = pos % m  # [seq_len]
        angle = 2 * math.pi * residue / m
        pe[:, 2 * i] = torch.sin(angle)
        pe[:, 2 * i + 1] = torch.cos(angle)
    return pe


# ---------------------------------------------------------------------------
# HBit tension gate
# ---------------------------------------------------------------------------

# Pre-compute the small Fibonacci attractor table for nearest-attractor
# lookup in tensor space.
_FIBS = torch.tensor([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987], dtype=torch.float)


def attractor_distance(values: torch.Tensor) -> torch.Tensor:
    """For each scalar in `values`, return distance to the nearest
    Fibonacci attractor (or 0 if value <= 0).
    Shape preserved: input [...] -> output [...].
    """
    # Broadcast: |values - attractors| -> [..., n_attractors]; argmin
    abs_v = values.abs()
    diffs = (abs_v.unsqueeze(-1) - _FIBS.to(values.device)).abs()
    return diffs.min(dim=-1).values


def hbit_tension_gate(keys: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute a gate factor in [0, 1] for each scalar in `keys`.
    keys: arbitrary shape. Returns same shape.

    gate(k) = 1 / (1 + scale * attractor_distance(k))

    On-attractor keys → gate = 1.0 (full weight).
    Off-attractor keys → gate < 1.0 (attenuated).
    """
    return 1.0 / (1.0 + scale * attractor_distance(keys))


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Single-head attention. The gate_mode parameter selects how (if
    at all) the HBit-tension signal modulates attention:

      "none"     : pure softmax (standard / crt_only).
      "key"      : fixed gate on per-key magnitude (the falsified
                   distractor-mix formulation; kept for reference).
      "score"    : ADDITIVE log-gate on the score tensor pre-softmax.
                   Substrate-distance of the raw score values dampens
                   off-attractor logits before softmax normalizes. No
                   post-hoc renormalization needed.
      "learned"  : per-head learnable scalar (W, b) gate on per-key
                   magnitude. Initialized to approximate the fixed
                   1/(1+d) formula; trains to discover whether
                   substrate distance is a useful signal for the task.
    """

    def __init__(self, d_model: int, gate_mode: str = "none", dropout: float = 0.0):
        super().__init__()
        if gate_mode not in ("none", "key", "score", "learned"):
            raise ValueError(f"unknown gate_mode: {gate_mode}")
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.gate_mode = gate_mode
        self.dropout = dropout
        if gate_mode == "learned":
            # Initialize so sigmoid(W*d + b) ≈ 1/(1 + d) near d ≈ 0:
            # picking W = -1, b = 0 gives sigmoid(-d) ∈ (0, 0.5], a
            # softer version of the falsified gate. Both are learnable.
            self.gate_w = nn.Parameter(torch.tensor(-1.0))
            self.gate_b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(-2, -1)) * scale  # [B, T, T]

        if self.gate_mode == "score":
            # Gate on score VALUES (pre-mask). attractor_distance of
            # raw scores tells us whether the (q·k) magnitude lands
            # on a substrate attractor. Off-attractor scores get
            # additively penalized in log-space, so softmax handles
            # normalization natively.
            d = attractor_distance(scores * 10.0)  # [B, T, T]
            log_gate = -torch.log1p(d)
            scores = scores + log_gate

        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)  # [B, T, T]

        if self.gate_mode == "key":
            key_mag = k.abs().mean(dim=-1)
            gate = hbit_tension_gate(key_mag * 100.0)
            attn = attn * gate.unsqueeze(1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)
        elif self.gate_mode == "learned":
            key_mag = k.abs().mean(dim=-1)
            d = attractor_distance(key_mag * 100.0)  # [B, T]
            gate = torch.sigmoid(self.gate_w * d + self.gate_b)
            attn = attn * gate.unsqueeze(1)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = attn @ v
        return self.out(out)


# ---------------------------------------------------------------------------
# Block + LM
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model: int, gate_mode: str = "none"):
        super().__init__()
        self.attn = Attention(d_model, gate_mode=gate_mode)
        self.ff = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """Tiny char-level LM. Architecture selected via constructor flags."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_blocks: int,
        seq_len: int,
        pe_kind: str,             # "sinusoidal" or "crt"
        gate_mode: str,           # "none" | "key" | "score" | "learned"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        if pe_kind == "sinusoidal":
            pe = sinusoidal_pe(seq_len, d_model)
        elif pe_kind == "crt":
            pe = crt_pe(seq_len, d_model)
        else:
            raise ValueError(f"unknown pe_kind: {pe_kind}")
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            Block(d_model, gate_mode=gate_mode) for _ in range(n_blocks)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pe[:T]
        mask = self.mask[:T, :T]
        for block in self.blocks:
            h = block(h, mask)
        h = self.ln_f(h)
        return self.head(h)


def make_model(
    arch: str,
    vocab_size: int,
    seq_len: int,
    d_model: int = 64,
    n_blocks: int = 2,
) -> TinyLM:
    """Five architectures:
      standard       : sinusoidal PE + pure softmax
      crt_only       : CRT-Fib PE   + pure softmax
      hybrid         : CRT-Fib PE   + KEY-magnitude gate (falsified)
      hybrid_score   : CRT-Fib PE   + SCORE-level pre-softmax gate
      hybrid_learned : CRT-Fib PE   + LEARNED per-head gate on key magnitude
    """
    common = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        n_blocks=n_blocks,
        seq_len=seq_len,
    )
    if arch == "standard":
        return TinyLM(**common, pe_kind="sinusoidal", gate_mode="none")
    if arch == "crt_only":
        return TinyLM(**common, pe_kind="crt", gate_mode="none")
    if arch == "hybrid":
        return TinyLM(**common, pe_kind="crt", gate_mode="key")
    if arch == "hybrid_score":
        return TinyLM(**common, pe_kind="crt", gate_mode="score")
    if arch == "hybrid_learned":
        return TinyLM(**common, pe_kind="crt", gate_mode="learned")
    raise ValueError(f"unknown arch: {arch}")
