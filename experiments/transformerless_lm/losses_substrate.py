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


PHI = (1.0 + 5.0 ** 0.5) / 2.0
PI_LOG_PHI = math.pi * math.log(PHI)


_FIB_FREQS = [1, 2, 3, 5, 8, 13, 21]
_FIB_NUMS = [1, 1, 2, 3, 5, 8, 13]   # F(k) for tier k = 0..K-1
_FIB_LAGS = [1, 2, 3, 5, 8, 13, 21]  # Fibonacci sequence lags for multi-scale


def corpus_char_signature(corpus_tokens: torch.Tensor,
                            vocab_size: int) -> torch.Tensor:
    """Char-level substrate signature of a corpus.

    Returns the normalized Fibonacci-frequency energy profile of the
    one-hot distribution implied by the actual tokens. This is the
    corpus's char-level substrate fingerprint -- target for the model
    to match instead of the abstract F(k)/phi^(pi*k) canonical.
    """
    fib_freqs = torch.tensor(_FIB_FREQS, dtype=torch.float,
                              device=corpus_tokens.device)
    K = fib_freqs.numel()
    v_idx = torch.arange(vocab_size, dtype=torch.float,
                          device=corpus_tokens.device)
    angles = 2 * math.pi * v_idx.unsqueeze(1) * fib_freqs.unsqueeze(0) / vocab_size
    basis_cos = torch.cos(angles)                                  # [V, K]
    basis_sin = torch.sin(angles)
    one_hot = F.one_hot(corpus_tokens, vocab_size).float()         # [N, V]
    proj_cos = one_hot @ basis_cos                                 # [N, K]
    proj_sin = one_hot @ basis_sin
    energy = (proj_cos ** 2 + proj_sin ** 2).mean(dim=0)            # [K]
    return energy / (energy.sum() + 1e-8)


def corpus_multiscale_signature(corpus_tokens: torch.Tensor,
                                  vocab_size: int,
                                  seq_len: int = 64) -> torch.Tensor:
    """Multi-scale substrate signature: self-similarity decay at Fib lags
    measured over windows of the actual corpus."""
    one_hot = F.one_hot(corpus_tokens, vocab_size).float()         # [N, V]
    N = one_hot.shape[0]
    K = len(_FIB_LAGS)
    sims = []
    for lag in _FIB_LAGS:
        if N <= lag:
            sims.append(torch.tensor(0.0, device=one_hot.device))
            continue
        # Dot product between token t and token t+lag, averaged
        p1 = one_hot[:-lag]
        p2 = one_hot[lag:]
        sim = (p1 * p2).sum(dim=-1).mean()
        sims.append(sim)
    sims = torch.stack(sims)
    return sims / (sims.sum() + 1e-8)


def substrate_harmony_loss_grounded(logits: torch.Tensor,
                                     vocab_size: int,
                                     target_signature: torch.Tensor,
                                     K_harmony: int = None,
                                     ) -> torch.Tensor:
    """Char-level harmony loss grounded against a TARGET signature.

    K_harmony: number of Fibonacci frequencies to USE in this loss. As
    the model's K-shrinks (basis collapses), the harmony's active
    frequency count should shrink with it -- the substrate's measuring
    stick must match the model's representational capacity. None = use
    all 7.
    """
    fib_freqs_all = _FIB_FREQS
    K_full = len(fib_freqs_all)
    K_use = K_full if K_harmony is None else min(K_harmony, K_full)
    fib_freqs = torch.tensor(fib_freqs_all[:K_use], dtype=logits.dtype,
                              device=logits.device)
    target = target_signature[:K_use]
    target = target / (target.sum() + 1e-8)        # renormalize at K_use
    v_idx = torch.arange(vocab_size, dtype=logits.dtype, device=logits.device)
    angles = 2 * math.pi * v_idx.unsqueeze(1) * fib_freqs.unsqueeze(0) / vocab_size
    basis_cos = torch.cos(angles)
    basis_sin = torch.sin(angles)
    pred = F.softmax(logits, dim=-1)
    pred_cos = pred @ basis_cos
    pred_sin = pred @ basis_sin
    energy = (pred_cos ** 2 + pred_sin ** 2).mean(dim=(0, 1))
    energy = energy / (energy.sum() + 1e-8)
    return (energy - target).abs().sum()


def substrate_multiscale_harmony_loss_grounded(logits: torch.Tensor,
                                                  vocab_size: int,
                                                  target_signature: torch.Tensor,
                                                  K_harmony: int = None,
                                                  ) -> torch.Tensor:
    """Multi-scale harmony loss grounded against a TARGET signature.

    K_harmony: shrinks lag set as model's K shrinks. None = all 7 lags.
    """
    fib_lags_all = _FIB_LAGS
    K_full = len(fib_lags_all)
    K_use = K_full if K_harmony is None else min(K_harmony, K_full)
    lags = fib_lags_all[:K_use]
    target = target_signature[:K_use]
    target = target / (target.sum() + 1e-8)
    probs = F.softmax(logits, dim=-1)
    T = probs.shape[1]
    sims = []
    for lag in lags:
        if T <= lag:
            sims.append(torch.tensor(0.0, dtype=logits.dtype,
                                      device=logits.device))
            continue
        p1 = probs[:, :-lag]
        p2 = probs[:,  lag:]
        sim = (p1 * p2).sum(dim=-1).mean()
        sims.append(sim)
    sims = torch.stack(sims)
    sims = sims / (sims.sum() + 1e-8)
    return (sims - target).abs().sum()


def substrate_harmony_loss(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """L1 distance from canonical F(k)/phi^(pi*k) decay at Fibonacci freqs.

    NO TARGET REQUIRED. Measures how well the predicted distribution's
    Fibonacci-frequency energy profile matches the substrate's canonical
    TIER-DECAY pattern F(k)/phi^(pi*k). This is the model's self-harmony
    score: lower = more in tune with the substrate prior.

    v1 used pure 1/phi^(pi*k) (geometric decay). v2 uses the full
    substrate-canonical F(k)/phi^(pi*k) — F(k) numerator preserves the
    Fibonacci tier structure, giving the higher tiers a bit more weight
    than pure geometric decay. Same formula the winning V2 activation
    used internally.

    Mechanism:
      1. Project the predicted distribution onto K Fibonacci frequencies.
      2. Compute energy per frequency: pred_cos^2 + pred_sin^2.
      3. Normalize energies to a distribution (sum=1).
      4. Compare to canonical F(k)/phi^(pi*k) tier decay (also normalized).
      5. L1 distance between the two.
    """
    fib_freqs = torch.tensor(_FIB_FREQS, dtype=logits.dtype,
                              device=logits.device)
    K = fib_freqs.numel()
    # Canonical substrate tier decay: F(k)/phi^(pi*k) -- the same formula
    # the winning V2 activation uses for its tier weights.
    canonical = torch.tensor(
        [_FIB_NUMS[k] / (PHI ** (math.pi * k)) for k in range(K)],
        dtype=logits.dtype, device=logits.device,
    )
    canonical = canonical / canonical.sum()

    v_idx = torch.arange(vocab_size, dtype=logits.dtype, device=logits.device)
    angles = 2 * math.pi * v_idx.unsqueeze(1) * fib_freqs.unsqueeze(0) / vocab_size
    basis_cos = torch.cos(angles)
    basis_sin = torch.sin(angles)

    pred = F.softmax(logits, dim=-1)                   # [B, T, V]
    pred_cos = pred @ basis_cos                         # [B, T, K]
    pred_sin = pred @ basis_sin

    energy = (pred_cos ** 2 + pred_sin ** 2).mean(dim=(0, 1))  # [K]
    energy = energy / (energy.sum() + 1e-8)

    return (energy - canonical).abs().sum()


def substrate_multiscale_harmony_loss(logits: torch.Tensor,
                                        vocab_size: int) -> torch.Tensor:
    """Multi-scale substrate harmony via self-similarity decay at Fib lags.

    The single-scale substrate_harmony_loss measures only char-level
    spectrum -- catches frequency patterns but NOT meter, rhyme, theme,
    or any structure that lives above the char tier. A model that
    optimizes only char-level harmony produces statistically-correct
    gibberish, not poetry.

    Multi-scale harmony measures the model's *self-similarity* at
    Fibonacci lags L ∈ {1, 2, 3, 5, 8, 13, 21}. Each lag is a
    different POETIC SCALE:
        lag 1   chars within a word (high similarity expected)
        lag 5   words within a line (~iambic pentameter range)
        lag 8   lines within a quatrain
        lag 13  across quatrains / stanzas
        lag 21  across sonnets / acts (low similarity expected)

    Substrate prior: this similarity should decay as F(k)/phi^(pi*k)
    across lags -- same canonical formula as char-level harmony, but
    applied to SCALES not frequencies. If the model produces output
    with the right decay across these lags, it's exhibiting hierarchical
    poetic structure: short-range cohesion (words) + long-range theme
    (stanzas).

    Formula:
        sim_L = mean over t of (probs[t] · probs[t+L])
              = expected next-token-agreement at lag L
        canonical_k = F(k) / phi^(pi*k), normalized to sum=1
        sim_k normalized to sum=1
        loss = L1 distance(sim, canonical)

    Note: this measures self-correlation patterns, not absolute output
    distributions. Combine with single-scale substrate_harmony_loss
    for both char-frequency AND multi-scale structure.
    """
    probs = F.softmax(logits, dim=-1)                  # [B, T, V]
    T = probs.shape[1]
    K = len(_FIB_LAGS)

    # Compute self-similarity at each Fibonacci lag.
    sims = []
    for lag in _FIB_LAGS:
        if T <= lag:
            # Sequence too short for this lag; substitute small similarity.
            sims.append(torch.tensor(0.0, dtype=logits.dtype,
                                      device=logits.device))
            continue
        p1 = probs[:, :-lag]                           # [B, T-lag, V]
        p2 = probs[:,  lag:]                           # [B, T-lag, V]
        sim = (p1 * p2).sum(dim=-1).mean()             # scalar
        sims.append(sim)
    sims = torch.stack(sims)
    sims = sims / (sims.sum() + 1e-8)

    canonical = torch.tensor(
        [_FIB_NUMS[k] / (PHI ** (math.pi * k)) for k in range(K)],
        dtype=logits.dtype, device=logits.device,
    )
    canonical = canonical / canonical.sum()

    return (sims - canonical).abs().sum()


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
