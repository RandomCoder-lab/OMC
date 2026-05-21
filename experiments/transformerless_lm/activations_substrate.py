"""Substrate-aware activation functions.

Every layer in the model passes activations through a nonlinearity
(GELU in FFN, softmax in attention). These are substrate-BLIND —
they produce continuous floats with no Fibonacci structure. Even
when weights and gradients are substrate-aware, the actual NUMBERS
flowing between layers live in arbitrary float space.

substrate_gelu makes the activations themselves substrate-aligned:
the forward output is snapped to the nearest Fibonacci-attractor
magnitude, while the gradient flows through the smooth GELU
(straight-through estimator). This forces every layer's output to
live near Fibonacci values while keeping training differentiable.

The model learns a per-layer scale so it can position its activations
where attractors land (e.g., scaling small post-GELU values up to
where the {±1, ±2, ±3, ±5, ...} attractors are meaningful).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_FIB_ATTRACTORS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]


def _signed_attractor_table(device, dtype):
    pos = torch.tensor(_FIB_ATTRACTORS, dtype=dtype, device=device)
    return torch.cat([-pos[1:].flip(0), pos])   # [-89, -55, ..., -1, 0, 1, ..., 89]


def attractor_snap(x: torch.Tensor) -> torch.Tensor:
    """Snap each scalar to its nearest signed-Fibonacci attractor."""
    table = _signed_attractor_table(x.device, x.dtype)
    diffs = (x.unsqueeze(-1) - table).abs()
    nearest_idx = diffs.argmin(dim=-1)
    return table[nearest_idx]


class SubstrateGELU(nn.Module):
    """GELU + attractor snap with straight-through gradient.

    Forward: gelu(x) → snap to nearest Fibonacci attractor (after scaling).
    Backward: gradient flows through GELU only (snap is an identity in
    the backward pass). Standard STE trick for non-differentiable ops.

    The per-layer `scale` parameter lets the model learn where its
    activations should sit relative to the fixed attractor table.
    Initialized to a value that maps typical post-GELU magnitudes
    (~0.1-1.0) into the attractor-meaningful range.
    """

    def __init__(self, init_scale: float = 3.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(x)
        # Bring h into the attractor-meaningful range, snap, scale back.
        h_scaled = h * self.scale
        snapped = attractor_snap(h_scaled) / self.scale
        # Straight-through estimator: forward = snapped, backward = identity through h
        return h + (snapped - h).detach()


class SubstrateGELUSoft(nn.Module):
    """Softer variant: blend GELU with attractor-snap by a learnable mix.
    At mix=0 it's pure GELU; at mix=1 it's full snap. The model can
    learn its own substrate-coupling strength."""

    def __init__(self, init_scale: float = 3.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.mix_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(x)
        h_scaled = h * self.scale
        snapped = attractor_snap(h_scaled) / self.scale
        mix = torch.sigmoid(self.mix_raw)
        # Blend: gradient flows through h always; the snap component is STE
        snap_path = h + (snapped - h).detach()
        return (1 - mix) * h + mix * snap_path
