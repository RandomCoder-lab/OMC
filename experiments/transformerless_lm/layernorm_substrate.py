"""Substrate-aware LayerNorm and Softmax.

Both standard ops use Euclidean / exponential bases that are
substrate-blind. The substrate replacements use its canonical metric
(L1 attractor distance) and canonical exponential (phi^pi base).

SubstrateL1LN: normalize by mean-absolute-deviation instead of std.
    L1 is the substrate's metric throughout (attractor distance,
    Subsim attention, Zeckendorf decomposition). Replacing the L2-std
    in LayerNorm with L1-MAD aligns the normalization with the same
    metric the substrate uses everywhere else.

SubstrateSoftmax: use phi^(pi*x) as the base instead of e^x.
    Equivalent to standard softmax with temperature 1/(pi*log(phi))
    ≈ 0.661. The base phi^pi is the substrate's canonical exponential
    growth rate (used in phi_pi_fib_search_v2 split points).
    Implementation: F.softmax(x * pi * log(phi), dim=dim) -- a single
    constant scaling preserves softmax's smooth properties while
    aligning the implicit base.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PHI = (1.0 + 5.0 ** 0.5) / 2.0
PI_LOG_PHI = math.pi * math.log(PHI)   # log(phi^pi) = pi * log(phi) ≈ 1.5145


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
