"""Substrate-aware loss — incorporates phi_pi_fib attractor distance.

The standard cross-entropy loss only rewards correct token prediction.
It doesn't reward the MODEL'S OUTPUT to live in substrate-aligned space.
With substrate-compressed weights and a standard loss, the model has
no incentive to USE substrate structure in its activations / logits.

Substrate-aware loss adds the substrate's own canonical distance metric:

    L = CE(softmax(logits), target) + λ · attractor_distance(logits)

where attractor_distance is the substrate's nearest-Fibonacci-tier
distance (the same operation phi_pi_fib.rs uses everywhere internally).

This pulls the logits toward substrate-aligned magnitudes, so the
TRAINING SIGNAL itself rewards substrate-shaped outputs.
"""

import math

import torch
import torch.nn.functional as F


# Canonical Fibonacci attractor table — matches omnimcode-core's
# nearest_attractor lookup.
_FIB_ATTRACTORS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0]


def attractor_distance(x: torch.Tensor) -> torch.Tensor:
    """Per-element distance to the nearest signed-Fibonacci attractor.

    Mirrors phi_pi_fib's nearest_attractor_with_dist: for each scalar
    value, find the nearest |F(k)| ∈ {1,2,3,5,8,13,...}, return the
    absolute difference. Negative values are treated by their absolute
    magnitude (sign-symmetric).

    Returns a tensor of the same shape as x.
    """
    abs_x = x.abs()
    # Build attractors on the same device/dtype as x
    attractors = torch.tensor(_FIB_ATTRACTORS, dtype=x.dtype, device=x.device)
    # Each |x[i]| vs each attractor: shape [..., n_attractors]
    diffs = (abs_x.unsqueeze(-1) - attractors).abs()
    return diffs.min(dim=-1).values


def substrate_aware_loss(logits: torch.Tensor, targets: torch.Tensor,
                          vocab_size: int,
                          lambda_substrate: float = 0.01) -> torch.Tensor:
    """Cross-entropy + substrate-attractor regularization.

    Args:
        logits: [B, T, V]
        targets: [B, T]
        vocab_size: V
        lambda_substrate: weight of the substrate term. Small values
            (0.001–0.05) typically work; larger values dominate CE
            and produce garbage.

    Returns:
        scalar loss
    """
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    # Substrate regularization: pull logits toward Fibonacci-attractor magnitudes.
    # We measure on the post-softmax distribution to keep the term comparable
    # in scale to CE.
    probs = F.softmax(logits, dim=-1)
    # Scale probs to a meaningful magnitude (so attractors 1, 2, 3 are reachable).
    # Probs are in [0, 1]; multiplying by 10 puts them in [0, 10] where the
    # attractors 1, 2, 3, 5, 8 give meaningful nearest-neighbor distances.
    scaled = probs * 10.0
    substrate_term = attractor_distance(scaled).mean()
    return ce + lambda_substrate * substrate_term


def substrate_only_loss(logits: torch.Tensor, targets: torch.Tensor,
                          vocab_size: int) -> torch.Tensor:
    """PURE substrate loss — no CE component. Tests whether the substrate
    operator alone is enough to drive learning."""
    probs = F.softmax(logits, dim=-1)
    scaled = probs * 10.0
    return attractor_distance(scaled).mean()


def substrate_fft_loss(logits: torch.Tensor, targets: torch.Tensor,
                        vocab_size: int,
                        lambda_substrate: float = 0.01) -> torch.Tensor:
    """CE + Fibonacci-frequency decomposition mismatch.

    Decompose the logit vector via cosine projections at Fibonacci
    frequencies. The substrate term penalizes mismatch between the
    predicted distribution's Fibonacci spectrum and the target's.

    More expensive than attractor_distance (does T·K projections) but
    a different substrate signal.
    """
    ce = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    # Project logits and target one-hot onto Fibonacci frequencies
    fib_freqs = torch.tensor([1, 2, 3, 5, 8, 13, 21], dtype=logits.dtype,
                              device=logits.device)
    v_idx = torch.arange(vocab_size, dtype=logits.dtype, device=logits.device)
    # angles[v, k] = 2π · F_k · v / V
    angles = 2 * math.pi * v_idx.unsqueeze(1) * fib_freqs.unsqueeze(0) / vocab_size
    basis_cos = torch.cos(angles)  # [V, K]
    basis_sin = torch.sin(angles)
    # Project predicted dist and target dist
    pred = F.softmax(logits, dim=-1)                                     # [B, T, V]
    target_onehot = F.one_hot(targets, vocab_size).to(pred.dtype)        # [B, T, V]
    pred_cos = pred @ basis_cos     # [B, T, K]
    pred_sin = pred @ basis_sin
    tgt_cos = target_onehot @ basis_cos
    tgt_sin = target_onehot @ basis_sin
    fft_mismatch = ((pred_cos - tgt_cos) ** 2 + (pred_sin - tgt_sin) ** 2).mean()
    return ce + lambda_substrate * fft_mismatch
