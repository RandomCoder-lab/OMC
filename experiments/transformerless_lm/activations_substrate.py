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


class SubstrateNegAsymmetric(nn.Module):
    """Substrate-canonical asymmetric activation.

    The substrate's contraction ratio φ handles BOTH branches without
    needing exp() / softplus / sigmoid:

        f(x) = x · (1 − φ^(−x))   if x > 0
        f(x) = −|x| / φ^|x|       if x < 0

    Properties:
      - Positive branch saturates toward identity as x grows
        (substrate-shaped soft ramp).
      - Negative branch DECAYS toward 0 as |x| grows
        (substrate contraction). Bounded magnitude |neg| ≤ 1/(e·ln φ) ≈ 0.76.
      - Continuous at x=0 (both branches → 0).
      - No divide-by-zero anywhere (φ^|x| ≥ 1).
      - Uses ONLY φ. No exp, no sigmoid, no GELU.

    The negative side's shape — peaks around |x|≈1.4 then decays — is
    impossible to express via standard activations without infinite
    terms. Substrate math gets it from a single closed form.
    """

    def __init__(self):
        super().__init__()
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        self.register_buffer("phi", torch.tensor(phi, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Positive branch: x · (1 − φ^(−x))
        pos = x * (1.0 - self.phi ** (-x.clamp(min=0.0)))
        # Negative branch: −|x| / φ^|x|
        neg_x = (-x).clamp(min=0.0)            # |x| for x < 0, else 0
        neg = -neg_x / (self.phi ** neg_x)
        # Combine via masks (avoids in-place ops, keeps gradients clean)
        pos_mask = (x > 0).to(x.dtype)
        return pos * pos_mask + neg * (1.0 - pos_mask)


class BinetFibActivation(nn.Module):
    """Pure substrate activation — Binet's Fibonacci interpolation curve.

    Replaces GELU entirely with the smooth continuous extension of the
    Fibonacci sequence:

        F(x) = (φ^x − cos(π·x)·φ^(−x)) / √5

    Passes through Fibonacci numbers at integer x (F(0)=0, F(1)=1,
    F(2)=1, F(3)=2, F(4)=3, F(5)=5, ...). Uses ONLY φ and π — the
    substrate's canonical constants. No GELU underneath.

    Bounded via tanh to ±φ^π (substrate's canonical contraction):

        f(x) = φ^π · tanh( F_binet(x) / (√5 · φ^π) )

    The tanh keeps activations finite and the gradient nonzero
    everywhere. Per-layer learnable scale lets the model position
    its input range relative to the curve.
    """

    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        # Substrate constants — store as buffers for device safety
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        phi_pi = phi ** math.pi
        self.register_buffer("phi", torch.tensor(phi, dtype=torch.float))
        self.register_buffer("phi_pi", torch.tensor(phi_pi, dtype=torch.float))
        self.register_buffer("sqrt5", torch.tensor(5.0 ** 0.5, dtype=torch.float))
        self.register_buffer("pi", torch.tensor(math.pi, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input — model learns where its activations sit relative to
        # the Binet curve's interesting region (small integers).
        z = x * self.scale
        # Clip z to a safe range so φ^z and φ^(-z) don't overflow.
        # For φ ≈ 1.618 and float32, φ^z is finite up to z ≈ 88.
        z = z.clamp(-30.0, 30.0)
        phi_z = self.phi ** z                      # φ^x
        phi_neg_z = self.phi ** (-z)                # φ^(-x)
        cos_pi_z = torch.cos(self.pi * z)            # cos(π·x)
        f_binet = (phi_z - cos_pi_z * phi_neg_z) / self.sqrt5
        return self.phi_pi * torch.tanh(f_binet / (self.sqrt5 * self.phi_pi))


class PhiPiFibActivation(nn.Module):
    """Substrate-CANONICAL activation: GELU + sum of sin(F(k)·x) terms,
    each weighted by the substrate's F(k)/φ^(π·k) probe-decay sequence
    from phi_pi_fib.rs.

        f(x) = GELU(x) + α · Σ_k [F(k)/φ^(π·k)] · sin(F(k)·x)

    Substrate-canonical FORMULA via F(k)/φ^(π·k), smooth basis (sin),
    gradient-friendly (no discretization), per-layer learnable substrate
    strength α (init small so it starts as nearly-GELU and grows toward
    full substrate coupling only if helpful).
    """

    def __init__(self, K: int = 5, init_alpha: float = 0.1):
        super().__init__()
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        phi_pi = phi ** math.pi      # ≈ 4.534
        # Substrate-canonical sequence F(k)/φ^(π·k) from phi_pi_fib.rs
        FIB = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]
        K = min(K, len(FIB))
        coeffs = torch.tensor([FIB[k] / (phi_pi ** (k + 1)) for k in range(K)],
                                dtype=torch.float)
        freqs = torch.tensor(FIB[:K], dtype=torch.float)
        self.register_buffer("substrate_coeffs", coeffs)   # F(k)/φ^(πk)
        self.register_buffer("substrate_freqs", freqs)     # F(k)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.gelu(x)
        # x: [..., D]. We add scalar wobble: sum_k coeffs[k] * sin(freqs[k] * x).
        # Each term sin(F(k) * x) is element-wise, then weighted by F(k)/φ^(π·k).
        # For numerical stability we evaluate all K terms in a vectorized way.
        # Shape: [..., K] via x.unsqueeze(-1) * freqs (broadcast).
        scaled = x.unsqueeze(-1) * self.substrate_freqs        # [..., K]
        sin_terms = torch.sin(scaled)                            # [..., K]
        correction = (sin_terms * self.substrate_coeffs).sum(dim=-1)  # [...]
        return base + self.alpha * correction


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
