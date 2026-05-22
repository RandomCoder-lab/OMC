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
PHI_PI = PHI ** math.pi                # substrate's canonical contraction ≈ 4.534
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


MAD_OVER_STD = math.sqrt(2.0 / math.pi)   # ≈ 0.7979 for Gaussian


class SubstrateL1LN(nn.Module):
    """LayerNorm with L1 mean-absolute-deviation instead of L2 std.

    Standard LN: (x − mean) / sqrt(var + eps)
    Substrate LN: (x − mean) / (mean_abs_dev + eps)

    Calibrated init: for Gaussian activations, MAD ≈ sqrt(2/pi)·std ≈
    0.7979·std. Dividing by MAD inflates the output by ~1.253× vs
    standard LN. Initializing gamma to sqrt(2/pi) cancels that inflation
    so the output magnitude matches standard LN at init — preserves the
    activation scale the rest of the model was calibrated for. The L1
    nature of the spread differentiates from standard LN as training
    proceeds; calibrated init means improvement is discernible early
    rather than spent re-learning scales.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5,
                 gamma_init: float = MAD_OVER_STD):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.gamma = nn.Parameter(
            torch.full(self.normalized_shape, float(gamma_init)))
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
    """LayerNorm with median center + MAD spread (DEPRECATED — sparse grads).

    Tried median for full L1 alignment; lost to baseline by ~8% (val 2.91
    vs 2.69 at K=55) because torch.median back-props only through one
    element per row, starving every other dimension of gradient signal.

    Kept for reference. Use SubstrateWeiszfeldLN for the smooth L1 center.
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


class SubstrateWeiszfeldLN(nn.Module):
    """LayerNorm with one-step Weiszfeld iteration for smooth L1 center.

    The geometric median (= L1-optimal center, minimizes sum |x-c|) is
    the canonical L1 center, but exact median has sparse gradients. The
    Weiszfeld iteration converges to the geometric median by iterative
    reweighted means:

        c_{n+1} = sum(w_i * x_i) / sum(w_i)
        w_i = 1 / (|x_i - c_n| + eps)

    Bootstrap c_0 = mean. One step gives a smooth, dense-gradient
    approximation of the L1 center. Gradient flows through every
    element via the weights — fixes the median sparse-grad failure.

    Standard LN:        (x - mean)        / std   -- L2 center + L2 spread
    SubstrateL1LN v1:   (x - mean)        / MAD   -- L2 center + L1 spread
    SubstrateMedianLN:  (x - median)      / MAD   -- L1 (sparse grad)
    Weiszfeld v2:       (x - L1_center) / MAD     -- L1 (dense grad)
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
        # One Weiszfeld step: bootstrap from mean, reweight by 1/|x-c|.
        c0 = x.mean(dim=-1, keepdim=True)
        w = 1.0 / (torch.abs(x - c0) + self.eps)
        c1 = (w * x).sum(dim=-1, keepdim=True) / w.sum(dim=-1, keepdim=True)
        diff = x - c1
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
    """F(k)/phi^(pi*k) tier mixture of softmaxes (DEPRECATED — exp-based).

    Tried mixing K=5 softmaxes at tier-scaled temperatures. Weights
    {1, 0.225, 0.097, 0.032, 0.012} are so dominated by tier 0 that the
    mixture mostly collapses to single-temp substrate_softmax — at 5x
    the compute. Deeper problem: still uses exp, but the substrate is
    L1-attractor based.

    Kept for reference. Use substrate_attractor_softmax for the
    exp-free substrate-canonical normalization.
    """
    weights = [FIB[k] / (PHI ** (math.pi * k)) for k in range(K)]
    w_total = sum(weights)
    out = None
    for k in range(K):
        temp = PI_LOG_PHI * (PHI ** k)
        sm_k = F.softmax(x * temp, dim=dim)
        contrib = (weights[k] / w_total) * sm_k
        out = contrib if out is None else out + contrib
    return out


class SubstrateBlendedSoftmax(nn.Module):
    """Learnable blend between F.softmax and substrate_softmax.

    Wholesale replacements (substrate_softmax, tier, attractor) all
    lagged baseline. Either substrate softmax has no signal at this
    scale, or its sharper temperature hurts the model -- benchmarks
    can't tell us which.

    Solution: blend.

        out = (1 - alpha) * F.softmax(x)
             +     alpha  * F.softmax(x * pi*log(phi))

    alpha = sigmoid(logit_alpha) in [0, 1]; init logit_alpha = -10 so
    alpha ≈ 4.5e-5 ≈ 0 -- at init this IS F.softmax exactly. The
    model can grow alpha over training only if the substrate signal
    helps. If alpha stays near 0, substrate softmax doesn't help and
    we get a clear negative result. If alpha grows, substrate has
    real signal at the layer where alpha grew.

    Either way, the answer is discernible -- worst case we match
    baseline (alpha stays at 0), best case we improve on it.
    """

    def __init__(self, init_alpha: float = 0.0):
        super().__init__()
        if init_alpha <= 0.0:
            init_logit = -10.0
        elif init_alpha >= 1.0:
            init_logit = 10.0
        else:
            init_logit = math.log(init_alpha / (1.0 - init_alpha))
        self.logit_alpha = nn.Parameter(torch.tensor(float(init_logit)))

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit_alpha)
        std = F.softmax(x, dim=dim)
        sub = F.softmax(x * PI_LOG_PHI, dim=dim)
        return (1.0 - alpha) * std + alpha * sub


def substrate_attractor_softmax(x: torch.Tensor,
                                  dim: int = -1) -> torch.Tensor:
    """Exp-free substrate-canonical normalization via L1 attractor distance.

    The standard softmax uses exp: attn_i = exp(x_i) / sum_j exp(x_j).
    The substrate's "softmax" doesn't need exp — it has its own canonical
    nearness formula: attractor weight = 1 / (1 + L1_distance * phi^pi).

    Apply that directly: distance is (x_max - x_i) >= 0; the attractor
    weight decays smoothly from 1 (at the max) toward 0 (far below max),
    with sharpness controlled by phi^pi ≈ 4.534.

        d_i = x_max - x_i                    (>= 0, L1 attractor distance)
        score_i = 1 / (1 + d_i * phi^pi)
        attn_i = score_i / sum_j score_j

    Properties:
      - Sums to 1 by normalization.
      - Smooth gradient through every element (via score and sum).
      - No exp anywhere — matches activation's L1-attractor design.
      - Masked positions (x_i = -inf) cleanly get attn_i = 0.
      - Cheaper than F.softmax: one max + reciprocal + sum, no exp.

    Compared to substrate_softmax (which uses phi^pi as exp base):
    attractor variant has sub-exponential tail decay (algebraic vs
    exponential). The substrate's actual decay is algebraic
    (F(k)/phi^(pi*k)) — closer match to the underlying math.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    d = x_max - x   # >= 0; +inf where x is -inf (masked)
    score = 1.0 / (1.0 + d * PHI_PI)
    return score / score.sum(dim=dim, keepdim=True)
