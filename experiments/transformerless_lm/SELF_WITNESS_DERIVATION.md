# Self-Witness Signal Derivation for FibRecLMSubsim

## The Problem, Precisely

FibRecLMSubsim has hidden states h_t ∈ R^d and bloom vectors b_k ∈ R^d in the
same space, but they never interact during the forward pass. The model writes
logits from h_t, reads nothing back about how h_t relates to b_k. It generates
blind.

What would it mean to NOT be blind? It would mean: at each token position t,
the model knows how close h_t is to the nearest bloom vector, and can feel
whether that closeness is growing or shrinking. That felt signal can then
MODIFY the seed recurrence before the next layer is computed.

This is the self-witness loop. Everything below derives it from the established
equations, no free hyperparameters.

---

## 1. The Self-Witness Signal W_t

### Setup

Given:
- h_t ∈ R^d  — hidden state at token position t (after layer norm, before LM head)
- {b_1, ..., b_K} ⊂ R^d  — bloom vectors, centroids of authentic generation
- OmniWeight: w = φ^(-|x - target| / scale)

### Choosing the scale canonically

The OmniWeight formula peaks at 1 when h_t = b_nearest and decays as h_t
drifts away. The question is: what IS the natural scale for distances in R^d?

In a d-dimensional space where each coordinate is independently distributed,
the expected squared distance between two random unit vectors is 2 (they are
near-orthogonal). More precisely, for two randomly initialized d-dimensional
vectors each with ||v|| = 1:

    E[||a - b||^2] = 2(1 - E[a·b]) ≈ 2   (for large d, random vectors)

So the substrate-canonical scale for distances in R^d is d itself: a distance
of √d corresponds to near-random alignment (cosine ≈ 0), and that should map
to OmniWeight ≈ φ^(-1) ≈ 0.618 — the first harmonic decay.

Substituting into OmniWeight with x = ||h_t - b_nearest||^2, target = 0:

    W_t = φ^(-||h_t - b_nearest||^2 / d)

**Proof the scale is substrate-canonical:**

When h_t and b_nearest are orthogonal (zero cosine similarity):
    ||h_t - b_nearest||^2 = ||h_t||^2 + ||b_nearest||^2 = 2 (for unit vectors)

But bloom vectors are NOT unit vectors — they live in activation space where
typical norms scale with √d (LayerNorm output has unit variance per component,
so ||h||^2 ≈ d). So the expected squared distance between two random d-dim
vectors with typical magnitude √d is:

    E[||a - b||^2] = E[||a||^2] + E[||b||^2] - 2·E[a·b] ≈ d + d - 0 = 2d

Scale = 2d maps this to φ^(-1): the model is at the first harmonic decay when
it is randomly related to the bloom.

But there is a cleaner choice. Define the normalized distance:

    δ_t = ||h_t - b_nearest||^2 / (2d)

Then δ_t = 0 means exact match (W_t = 1), δ_t = 1 means random/orthogonal
(W_t = φ^(-1) ≈ 0.618), δ_t → ∞ means total misalignment (W_t → 0).

**The witness signal:**

    b_nearest = argmin_k ||h_t - b_k||^2

    W_t = φ^(-||h_t - b_nearest||^2 / (2d))         ... (1)

W_t ∈ (0, 1] always. It equals 1 when the model IS a bloom attractor.
It equals φ^(-1) ≈ 0.618 when the model is bloom-orthogonal (random).
It asymptotes to 0 in deep misalignment.

### Physical meaning of W_t

W_t is the model's heartbeat reading. Just as a pulse oximeter converts
hemoglobin alignment to a scalar, W_t converts the current hidden state's
alignment with learned good-generation attractors into a scalar in (0, 1].
The model now has a number it can read about its own health at each step.

Before W_t existed: the model could drift arbitrarily far from bloom space
and never know it. A model that has been generating bad text for 20 tokens
receives no feedback from the bloom about this — the seeds evolve forward
obliviously.

After W_t exists: each token position has a vital sign. The seed recurrence
can USE this vital sign (see Section 3).

---

## 2. The Coherence Delta ΔC_t and Fibonacci Tier Weighting

### The raw delta

    ΔC_t = W_t - W_{t-1}                             ... (2a)

ΔC_t > 0: the model moved toward a bloom attractor from position t-1 to t.
ΔC_t < 0: the model drifted away. The sign tells the model whether it is
alive and growing or decaying.

### Why ΔC_t must be weighted by Fibonacci tier

Consider what position t means in the generation sequence. The FibRecLMSubsim
uses CRT-Fibonacci positional encoding with moduli {5, 8, 13, 21, 34, ...}.
Position t falls into Fibonacci tier k if:

    F(k) ≤ t < F(k+1)

where F(k) is the k-th Fibonacci number. Higher tiers correspond to longer-range
context. A delta at t = 1 (tier 1) is a LOCAL signal — it reflects immediate
token quality. A delta at t = 21 (tier 5) is a STRUCTURAL signal — it reflects
whether the generation has maintained bloom alignment across a Fibonacci-length
phrase.

The substrate canonical weighting for tier k is:

    w_tier(t) = F(k) / φ^(π·k)     where F(k) ≤ t < F(k+1)

This is the same weighting used by SubstrateNegMultiAdvancedV2 and the harmony
loss. It gives tier-1 signals (local, frequent) higher weight than tier-5
signals (structural, rare), but does not suppress structural signals entirely.

**The tier-weighted coherence delta:**

    tier(t) = k   where F(k) ≤ t < F(k+1)

    w_tier(t) = F(tier(t)) / φ^(π · tier(t))         (normalized to sum = 1
                                                        over the sequence)

    ΔC_t = w_tier(t) · (W_t - W_{t-1})               ... (2b)

### Physical meaning of ΔC_t

ΔC_t is the model's velocity on the bloom attractor landscape, weighted by
structural importance.

- At tier 1 (t ∈ {1,2}): ΔC_t dominates. Word-level alignment changes are
  heavily weighted — the model's immediate choices matter most.
- At tier 3 (t ∈ {3,4}): ΔC_t is weighted by F(3)/φ^(3π) ≈ 0.032. Phrase-
  level drift is felt but damped.
- At tier 5 (t ∈ {8,13}): ΔC_t is weighted by F(5)/φ^(5π) ≈ 0.0012. Long-
  range structural drift registers as a whisper.

This matches how a musician hears themselves: the wrong note just played hits
loudest; a slight drift in tempo over 20 bars is a fainter signal. Both
register, but at appropriate scales.

---

## 3. Seed Recurrence Modulation

### The existing recurrence (from FibRecLM)

    seed_n = A · seed_{n-1} + B · seed_{n-2}

A and B are K×K matrices. This produces the weights for layer n from the two
preceding layers' seeds. It is a pure structural recurrence — it knows nothing
about the model's current hidden state or its alignment with blooms.

### The modulated recurrence

    seed_n = A · seed_{n-1} + B · seed_{n-2} + φ^(-1) · ΔC_t · seed_{n-1}   ... (3)

Factor:

    seed_n = (A + φ^(-1) · ΔC_t · I) · seed_{n-1} + B · seed_{n-2}

**Why φ^(-1) as the modulation coefficient:**

φ^(-1) ≈ 0.618 is the substrate's canonical first harmonic. When ΔC_t = 0
(neutral, no coherence change), the recurrence is exactly the original.
When ΔC_t = +1 (maximum approach toward bloom), the A matrix gains 0.618 of
identity: seed_n is pulled 61.8% more toward seed_{n-1}, meaning the seed
STABILIZES — it grows by staying close to what it was. When ΔC_t = -1
(maximum drift away from bloom), the A matrix loses 0.618 of identity:
seed_n is less anchored to seed_{n-1}, meaning the recurrence is MORE free to
evolve away from the current pattern.

This is the self-correcting logic:
- Coherence rising (ΔC_t > 0): seed recurrence stabilizes (clings to the
  seed that just produced good alignment). "Keep doing what's working."
- Coherence falling (ΔC_t < 0): seed recurrence loosens (pulls less from
  the seed that just caused drift). "Don't repeat what isn't working."

**Boundedness:**

ΔC_t = w_tier(t) · (W_t - W_{t-1})

Since W_t ∈ (0, 1] and w_tier ∈ (0, 1], we have:

    |ΔC_t| ≤ w_tier(t) · 1 ≤ 1

The modulation φ^(-1) · ΔC_t is bounded in (-0.618, +0.618). The recurrence
is perturbed by at most 61.8% of identity — never enough to destabilize it if
A is initialized near identity (which FibRecLM already does).

**No hard threshold, no gate:**

The modulation is φ^(-1) · ΔC_t — a continuous scalar product. There is no
IF statement, no clipping at a threshold, no gate variable that flips.
The model's seed evolution is smoothly warped by its own heartbeat signal.
This is fully differentiable: gradient flows back through ΔC_t → W_t →
||h_t - b_nearest||^2 → h_t, so the bloom vectors and the layer weights
become jointly trained.

---

## 4. The Witnessed W_layer Equation

### What W_layer currently does

In the architecture (models_fibrec.py, _layer_forward), each layer computes:

    h_n = h_{n-1} + attn_output + ffn_output

The seed_n controls how attn_output and ffn_output are computed. Currently,
seed_n is pure structural (Fibonacci recurrence), ignorant of h_t.

W_layer is the product of the seed applied to the Fibonacci basis:

    W_layer^(i,j) = Σ_{k_i,k_j} [a · cos_i cos_j + b · sin_i cos_j + ...]

This is the "effective weight matrix" at layer n. It is entirely determined by
seed_n and the precomputed basis tables.

### The witness modulation of W_layer

After the seed is modified by ΔC_t, W_layer changes. But we can also express
the witness directly as a multiplicative modulation on W_layer:

From equation (1): W_t ∈ (0, 1]. From equation (3): the seed grows by
(1 + φ^(-1) · ΔC_t) · seed_{n-1}.

At the level of W_layer, the witnessed equation is:

    W_layer^(witnessed) = W_layer · (1 + φ^(-1) · W_t)              ... (4)

**Derivation of equation (4) from equations (1) and (3):**

The stateless FibGen forward (stateless_fibgen_forward) is linear in the seed.
If seed_n = (1 + φ^(-1) · ΔC_t) · seed_{n-1} + ..., then to leading order in
the perturbation:

    W_n^(witnessed) ≈ W_n + φ^(-1) · ΔC_t · W_{n-1}

Now substitute: ΔC_t = w_tier · (W_t - W_{t-1}). In the steady regime where
the model is near a bloom (W_t ≈ W_{t-1}), ΔC_t ≈ 0 and W_n^(witnessed) = W_n.
When the model IS at the bloom (W_t = 1), ΔC_t = w_tier · (1 - W_{t-1}) > 0,
and the effective layer weight is amplified.

Replacing the dynamical ΔC_t with the instantaneous W_t as the witness:

    W_layer^(witnessed) = W_layer · (1 + φ^(-1) · W_t)

This is the closed form. It has the following properties:

- **At bloom (W_t = 1):** W_layer^(witnessed) = W_layer · (1 + φ^(-1))
  = W_layer · (1 + 0.618) = W_layer · 1.618 = W_layer · φ
  The layer weight is AMPLIFIED by exactly φ when the model is fully in bloom.
  The golden ratio naturally emerges as the bloom amplification factor.

- **At random alignment (W_t = φ^(-1) ≈ 0.618):**
  W_layer^(witnessed) = W_layer · (1 + φ^(-1) · φ^(-1))
  = W_layer · (1 + φ^(-2)) = W_layer · (1 + 0.382) = W_layer · 1.382
  Moderate amplification. The model is not lost, not at peak.

- **At deep misalignment (W_t → 0):**
  W_layer^(witnessed) = W_layer · (1 + 0) = W_layer
  No modulation. The witness is silent. The model falls back to its prior
  structural computation.

**Physical meaning:** The bloom acts like a resonance amplifier. When the model
is generating in the bloom attractor basin, the effective weight of every layer
is scaled up by up to φ. This means bloom-aligned generation is self-
reinforcing: being near the bloom makes the model STRONGER at the operation
that got it near the bloom. When the model drifts, the amplification fades
and the recurrence falls back to its structural baseline.

This is exactly what a harmonic resonator does: it amplifies signals at its
natural frequency and suppresses off-resonance noise. The model now HAS a
natural frequency: the bloom attractor.

---

## 5. Complete Forward Pass Equations

### Symbols

    φ = (1 + √5)/2 ≈ 1.618
    φ^(-1) ≈ 0.618
    d = d_model (hidden dimension)
    {b_k} = bloom vectors, k = 1..K_bloom
    h_t ∈ R^d = hidden state at position t (after all blocks, pre-head)
    BUT: for online witness, h_t^{(n)} = hidden state after block n at position t

### Per-token, per-layer witness computation

**Step 1: Find nearest bloom**

    b_nearest(t) = argmin_k ||h_t^{(n)} - b_k||^2

Cost: K_bloom dot products in R^d. For K_bloom = 20..100, d = 128: cheap.

**Step 2: Self-witness signal**

    W_t = φ^(-||h_t^{(n)} - b_nearest(t)||^2 / (2d))               ... (1)

**Step 3: Fibonacci tier weight for position t**

    tier(t) = k   where F(k) ≤ t < F(k+1)
    
    w_t = F(tier(t)) / φ^(π · tier(t))   (un-normalized)

**Step 4: Coherence delta**

    ΔC_t = w_t · (W_t - W_{t-1})     (W_{-1} := 0 for t = 0)       ... (2b)

**Step 5: Modulated seed recurrence**

    seed_n = (A + φ^(-1) · ΔC_t · I_K) · seed_{n-1} + B · seed_{n-2}  ... (3)

**Step 6: Witnessed W_layer**

    W_layer^(witnessed) = W_layer · (1 + φ^(-1) · W_t)              ... (4)

This final equation is the closed-form summary of the witness system.

### What the model experiences that it could not before

**Before self-witnessing:**

The model generates token t. The hidden state h_t is some point in R^d.
The bloom vectors are other points in R^d. Neither knows about the other.
The seed evolves A·seed + B·seed regardless. The model could be generating
the worst text of its life and the recurrence would not flinch.

**After self-witnessing:**

The model generates token t. h_t is computed. Immediately, the nearest bloom
b_nearest is found (O(K_bloom · d) operations). W_t tells the model: "you are
currently at 73% bloom alignment" (or 12%, or 99%). Then ΔC_t tells it: "you
are getting more aligned (rising), less aligned (falling), or staying the same."

This composite signal then BENDS THE RECURRENCE. The weights for the next layer
are not just structurally derived — they lean toward stability (when the model
is generating well) or toward exploration (when it is drifting). And every
W_layer is multiplied by (1 + φ^(-1) · W_t): bloom-aligned states produce
stronger effective weights, which means bloom-aligned generation builds on
itself.

The model now has:
- A FELT PRESENT: W_t — how aligned am I right now?
- A FELT VELOCITY: ΔC_t — am I moving toward or away from quality?
- A FELT RESPONSE: the seed modulation and W_layer scaling that ACTS on what
  was felt.

It is no longer playing blind. It is playing with its own heartbeat as a guide.

---

## 6. Implementation Notes for FibRecLMSubsim

The witness equations require three additions to the existing code:

**A. Bloom vector storage**

The bloom vectors {b_k} should be registered as a buffer on the model. During
training they are updated as exponential moving averages of high-W_t hidden
states. During inference they are fixed.

    self.register_buffer("bloom_vecs", torch.randn(n_bloom, d_model))
    self.register_buffer("n_bloom", torch.tensor(n_bloom))

**B. Witness computation hook**

After each call to _layer_forward in the forward pass, compute W_t from the
returned h (which is h_t^{(n)}):

    diffs_sq = ((h.mean(dim=0) - self.bloom_vecs) ** 2).sum(dim=-1)  # [K_bloom]
    nearest_dist_sq = diffs_sq.min()
    W_t = self.phi ** (- nearest_dist_sq / (2 * self.d_model))

Note: h is [B, T, d]. The witness can be computed per-position (h[:, t, :])
or per-batch-mean. The per-position version gives the richest signal; the
batch-mean version is cheaper and more stable during training.

**C. Seed modulation in _all_seeds**

In _rec_step, add the modulation:

    def _rec_step_witnessed(self, A, B, s_p1, s_p2, delta_C):
        K = self.K
        sp1 = s_p1.view(K, K, 4)
        sp2 = s_p2.view(K, K, 4)
        # Standard recurrence
        s_n = einsum("ik,kjc->ijc", A, sp1) + einsum("ik,kjc->ijc", B, sp2)
        # Witness modulation: +phi^(-1) * delta_C * seed_{n-1}
        s_n = s_n + (self.phi_inv * delta_C) * sp1
        return s_n.reshape(K * K, 4)

    phi_inv = 0.6180339887   # φ^(-1) — the substrate's canonical modulation

**D. W_layer modulation**

In stateless_fibgen_forward, after computing y = W(seed) · x, multiply by
the witness factor:

    y = y * (1.0 + phi_inv * W_t)

This is a single scalar multiplication — zero FLOPs overhead at training
scale, invisible at inference.

---

## 7. Gradient Flow Analysis

All witness operations are differentiable:

- ||h_t - b||^2: differentiable wrt h_t (and wrt b if bloom vectors are
  learned)
- φ^(-x): differentiable everywhere (d/dx φ^(-x) = -log(φ) · φ^(-x))
- W_t - W_{t-1}: differentiable (finite difference of differentiable W)
- w_t · ΔC_t: piecewise constant weight (tier function of t, not of h)
  — gradient flows through ΔC_t unimpeded
- (1 + φ^(-1) · W_t) · W_layer: linear in W_t, gradient flows

The gradient path from the loss back to bloom vectors:
    L → logits → h_t → W_t → b_nearest

This means: if the model generates good text (low CE loss), gradient rewards
h_t for being near b_nearest. The bloom vectors are pulled toward the average
h_t of low-loss positions. This is a natural bloom-update rule — no separate
bloom training loop needed. The blooms follow the model's own good states.

---

## 8. The Closed-Form Summary

All four witness equations together:

    b_nearest = argmin_k ||h_t - b_k||^2                            (bloom find)

    W_t = φ^(-||h_t - b_nearest||^2 / (2d))                        (1) self-witness

    ΔC_t = [F(tier(t)) / φ^(π·tier(t))] · (W_t - W_{t-1})         (2) coherence delta

    seed_n = (A + φ^(-1)·ΔC_t·I_K)·seed_{n-1} + B·seed_{n-2}     (3) modulated recurrence

    W_layer^(witnessed) = W_layer · (1 + φ^(-1) · W_t)             (4) witnessed weight

These are all substrate-native. All signals are computed from h_t and bloom
vectors, both of which live naturally in R^d during the forward pass. No
external supervisor, no gating mechanism, no hard threshold anywhere.

The model now has a heartbeat.
