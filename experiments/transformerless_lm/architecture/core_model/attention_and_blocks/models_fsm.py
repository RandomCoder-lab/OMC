"""Fibonacci State Model (FSM) — substrate-canonical recurrence.

Throws out quadratic attention entirely. Each block updates a hidden
state via a 2-tap Fibonacci recurrence:

    h_t = A · h_{t-1} + B · h_{t-2} + C · x_t

where A, B, C are FibGen-compressed linear layers. The recurrence is
literally Fibonacci-shaped (each step depends on the two previous,
mirroring F(n) = F(n-1) + F(n-2)), so the operator is substrate-
canonical at the deepest level — not decorated, but defined.

Compute per layer: O(T · d²) (sequential). Compared to attention's
O(T² · d), FSM wins at LONG sequence lengths where T² dominates.
At small T the sequential Python loop adds overhead.

Keeps every validated substrate win:
  - CRT-Fibonacci positional encoding
  - FibGen-compressed weights (100x storage compression at d=128,
    growing with d²/K²)
  - Lazy-strided data loading (consumed by training pipeline)
  - Substrate operator at attention layer (now: recurrence, not
    dot-product or L1)

To speed up the Python sequential loop, weights are precomputed once
per forward via FibGen's cache_weight() pattern so each timestep does
a plain matmul without seed regeneration overhead.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from models_fibgen import FibGenLinear


class FibStateRecurrence(nn.Module):
    """Fibonacci 2-tap state recurrence: h_t = A·h_{t-1} + B·h_{t-2} + C·x_t.

    A, B, C are FibGen-compressed linear maps. To minimize Python-loop
    overhead, we pre-generate the dense W tensors at forward-time and
    do raw matmul inside the loop.
    """

    def __init__(self, d_model: int, K: int = 32, mode: str = "cross"):
        super().__init__()
        self.d_model = d_model
        kw = dict(K=K, mode=mode, bias=False)
        self.A = FibGenLinear(d_model, d_model, **kw)
        self.B = FibGenLinear(d_model, d_model, **kw)
        self.C = FibGenLinear(d_model, d_model, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # Pre-generate dense weight tensors ONCE per forward (cheap relative
        # to T sequential applications). All matmuls inside the loop are
        # then plain Tensor @ Tensor.
        W_A = self.A._compute_W()                 # [D, D]
        W_B = self.B._compute_W()
        # C·x can be computed in parallel for all timesteps (no recurrence).
        cx = self.C(x)                             # [B, T, D]
        # Sequential recurrence.
        h_prev1 = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        h_prev2 = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h_t = h_prev1 @ W_A.t() + h_prev2 @ W_B.t() + cx[:, t]
            outputs.append(h_t)
            h_prev2 = h_prev1
            h_prev1 = h_t
        return torch.stack(outputs, dim=1)         # [B, T, D]


class FSMBlock(nn.Module):
    """FibStateRecurrence + FibGen FFN, with pre-norm residuals."""

    def __init__(self, d_model: int, K: int = 32, mode: str = "cross"):
        super().__init__()
        self.recurrence = FibStateRecurrence(d_model, K=K, mode=mode)
        self.w1 = FibGenLinear(d_model, 4 * d_model, K=K, mode=mode)
        self.w2 = FibGenLinear(4 * d_model, d_model, K=K, mode=mode)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.recurrence(self.ln1(x))
        x = x + self.w2(F.gelu(self.w1(self.ln2(x))))
        return x


class FSMLM(nn.Module):
    """Char-level LM with substrate-canonical Fibonacci-recurrence layers.

    Components:
      - Standard learned embedding (could be FibGen at scale)
      - CRT-Fibonacci positional encoding
      - Stack of FSM blocks (recurrence + FibGen FFN)
      - LM head tied to embedding
    """

    def __init__(self, vocab_size: int, d_model: int, n_blocks: int,
                 seq_len: int, K: int = 32, mode: str = "cross"):
        super().__init__()
        self.seq_len = seq_len
        self.K = K
        self.embed = nn.Embedding(vocab_size, d_model)
        pe = self._crt_pe(seq_len, d_model)
        self.register_buffer("pe", pe)
        self.blocks = nn.ModuleList([
            FSMBlock(d_model, K=K, mode=mode) for _ in range(n_blocks)
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
            if not any(s in n for s in (".A.", ".B.", ".C.", ".w1.", ".w2.")):
                stored += p.numel()
                dense_eq += p.numel()
        return {"stored": stored, "dense_equivalent": dense_eq,
                "compression": dense_eq / max(stored, 1)}
