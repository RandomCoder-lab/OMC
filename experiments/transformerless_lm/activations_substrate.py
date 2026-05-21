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

# Reciprocal Fibonacci attractors — dense near 0, sparse far from 0.
# Matches the actual distribution of post-GELU activations (small values).
_INV_FIB_ATTRACTORS = sorted(set([0.0] + [
    1.0 / f for f in [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]
] + [1.0, 2.0, 3.0]))    # keep a few positive integers for tail coverage


def _signed_attractor_table(device, dtype, inverse: bool = False):
    pos = torch.tensor(
        _INV_FIB_ATTRACTORS if inverse else _FIB_ATTRACTORS,
        dtype=dtype, device=device,
    )
    return torch.cat([-pos[1:].flip(0), pos])


def attractor_snap(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Snap each scalar to its nearest signed-Fibonacci attractor.
    inverse=True uses reciprocal Fibonacci values (dense near 0)."""
    table = _signed_attractor_table(x.device, x.dtype, inverse=inverse)
    diffs = (x.unsqueeze(-1) - table).abs()
    nearest_idx = diffs.argmin(dim=-1)
    return table[nearest_idx]


class SubstrateGELU(nn.Module):
    """GELU + attractor snap with straight-through gradient.

    inverse=True uses reciprocal Fibonacci attractors, which are dense
    in [-1, 1] — much better matched to typical post-GELU magnitudes
    than the forward Fibonacci attractors {1, 2, 3, 5, 8, ...} that
    sit OUTSIDE the typical activation range.
    """

    def __init__(self, init_scale: float = 3.0, inverse: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.inverse = inverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(x)
        h_scaled = h * self.scale
        snapped = attractor_snap(h_scaled, inverse=self.inverse) / self.scale
        return h + (snapped - h).detach()


class SubstrateGELUInverse(SubstrateGELU):
    """Convenience subclass: SubstrateGELU with inverse=True (reciprocal Fib)."""
    def __init__(self, init_scale: float = 3.0):
        super().__init__(init_scale=init_scale, inverse=True)


class SubstrateGELUSoft(nn.Module):
    """Softer variant: blend GELU with attractor-snap by a learnable mix.
    At mix=0 it's pure GELU; at mix=1 it's full snap. Uses reciprocal
    Fibonacci attractors by default since they match post-GELU magnitudes."""

    def __init__(self, init_scale: float = 3.0, inverse: bool = True):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        # Initialize at low coupling — sigmoid(-2) ≈ 0.12 so 88% GELU, 12% snap
        self.mix_raw = nn.Parameter(torch.tensor(-2.0))
        self.inverse = inverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(x)
        h_scaled = h * self.scale
        snapped = attractor_snap(h_scaled, inverse=self.inverse) / self.scale
        mix = torch.sigmoid(self.mix_raw)
        snap_path = h + (snapped - h).detach()
        return (1 - mix) * h + mix * snap_path
