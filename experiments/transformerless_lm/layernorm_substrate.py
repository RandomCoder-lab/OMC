"""Substrate-aware LayerNorm and Softmax.

Both standard ops use Euclidean / exponential bases that are
substrate-blind. The substrate replacements use its canonical metric
(L1 attractor distance) and canonical exponential (phi^pi base).

Refinement note: v1 versions mix L2 (mean) with L1 (MAD) in the LN,
and use a single fixed phi^pi temperature in the softmax. v2 versions
push further:
  - SubstrateMedianLN: median center + MAD spread = full L1 alignment
  - substrate_tier_softmax: F(k)/phi^(pi*k) weighted mixture of
    softmaxes at tier-scaled temperatures pi*log(phi)*phi^k
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PHI = (1.0 + 5.0 ** 0.5) / 2.0
PI_LOG_PHI = math.pi * math.log(PHI)   # log(phi^pi) = pi * log(phi) ≈ 1.5145
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


class SubstrateL1LN(nn.Module):
    """LayerNorm with L1 mean-absolute-deviation instead of L2 std.

    Standard LN: (x − mean) / sqrt(var + eps)
    Substrate LN: (x − mean) / (mean_abs_dev + eps)

    Same gamma/beta as standard. L1 deviation is the substrate's
    canonical metric. For Gaussian-distributed activations, MAD ≈
    0.7979 * std, so activations end up scaled slightly larger than
    with L2 — gamma can absorb the difference during training.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*self.normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        # L1 mean absolute deviation -- the substrate's canonical
        # nearness metric. Same operation Subsim attention uses on
        # signature differences.
        mad = diff.abs().mean(dim=-1, keepdim=True)
        return self.gamma * diff / (mad + self.eps) + self.beta


class SubstrateMedianLN(nn.Module):
    """LayerNorm with median center + MAD spread (full L1 alignment).

    SubstrateL1LN uses the mean (L2-optimal center, minimizes sum (x-c)^2)
    with the MAD (L1 spread). That mixes metrics. The L1-optimal center
    is the median (minimizes sum |x-c|), which pairs naturally with MAD.

    Standard LN:    (x - mean)   / std   -- L2 center + L2 spread
    SubstrateL1LN:  (x - mean)   / MAD   -- L2 center + L1 spread (mixed)
    MedianLN:       (x - median) / MAD   -- L1 center + L1 spread (canonical)

    Gradient note: torch.median back-props only through the median
    element per row (sparse, like maxpool). That's fine here.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.gamma = nn.Parameter(torch.ones(*self.normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*self.normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        median = x.median(dim=-1, keepdim=True).values
        diff = x - median
        mad = diff.abs().mean(dim=-1, keepdim=True)
        return self.gamma * diff / (mad + self.eps) + self.beta


def substrate_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Substrate-canonical softmax with base phi^pi instead of e.

    softmax_phi_pi(x_i) = (phi^pi)^x_i / sum_j (phi^pi)^x_j
                       = e^(x_i * pi * log(phi)) / sum_j e^(x_j * pi * log(phi))
                       = F.softmax(x * pi * log(phi), dim=dim)

    Effective temperature: 1 / (pi * log(phi)) ≈ 0.661 (sharper than
    standard softmax). Uses phi^pi -- substrate's canonical exponential
    -- as the implicit base.
    """
    return F.softmax(x * PI_LOG_PHI, dim=dim)


def substrate_tier_softmax(x: torch.Tensor, dim: int = -1,
                            K: int = 5) -> torch.Tensor:
    """F(k)/phi^(pi*k) weighted mixture of tier-scaled softmaxes.

    substrate_softmax uses a single phi^pi base (one temperature). The
    substrate's actual mass distribution is tier-decayed: F(k)/phi^(pi*k)
    per tier. Apply that decay to a mixture of softmaxes at tier-scaled
    temperatures:

        out = sum_k w_k * softmax(x * pi*log(phi) * phi^k)  /  sum_k w_k
        where w_k = F(k) / phi^(pi*k)

    Tier 0 has the highest weight (matches single-temp substrate_softmax);
    higher tiers contribute progressively sharper signal blended by
    substrate decay. K=5 default; weights {1, 0.225, 0.097, 0.032, 0.012}.

    Output is a valid probability distribution (convex combination of
    softmaxes), so it drops in wherever F.softmax is used.
    """
    # precompute weights
    weights = [FIB[k] / (PHI ** (math.pi * k)) for k in range(K)]
    w_total = sum(weights)
    out = None
    for k in range(K):
        temp = PI_LOG_PHI * (PHI ** k)
        sm_k = F.softmax(x * temp, dim=dim)
        contrib = (weights[k] / w_total) * sm_k
        out = contrib if out is None else out + contrib
    return out
