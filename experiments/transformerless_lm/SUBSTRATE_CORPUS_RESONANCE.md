# Substrate-Corpus Resonance: Full Derivation

**Date:** 2026-05-25
**Model:** FibRecLMSubsim — Fibonacci-recurrence seeds, SubSim L1-distance attention, CRT-PE

---

## Established Constants and Equations

```
φ  = (1 + √5)/2 ≈ 1.6180339887
φ⁻¹ ≈ 0.6180339887       (= φ − 1)
φ⁻² ≈ 0.3819660113       (= 1 − φ⁻¹, first Fibonacci ratio shortfall)

OmniWeight:   w = φ^(−|x − target| / scale)

KL divergence:  KL(P || Q) = Σᵢ P[i] · log(P[i] / Q[i])

FibPosTable:  P_substrate[fb] ∈ R⁷   (structural class distribution at Fibonacci tier fb)
Corpus:       P_corpus[fb] ∈ R⁷      (empirical class distribution at tier fb)

T[k,l]:   fingerprint leaf transition probability — P_substrate(leaf l follows leaf k)
T_corpus[k,l]:  empirical leaf transition frequency from 256-token seed

bloom vectors:  b_i ∈ R^d
corpus centroid:  c_corpus = mean of token embeddings weighted by corpus frequency
```

---

## 1. Substrate-Corpus Resonance (SCR)

### Tier-level SCR

For each Fibonacci tier fb:

```
SCR_fb = φ^(−KL(P_corpus[fb] || P_substrate[fb]) / φ⁻¹)
```

**What this means:**

When KL = 0: the substrate's learned structural priors at tier fb exactly replicate
what the corpus distributes over the 7 structural classes. `φ^0 = 1` — perfect resonance.

When KL > 0: the distributions diverge. As KL → ∞, SCR_fb → 0 — the substrate is
decoherent at this tier. The model is carrying priors adapted to a different corpus
reality than the one it is operating in.

**Why φ⁻¹ as the KL scale:**

OmniWeight places the first harmonic decay at x = scale. We want SCR_fb = φ⁻¹ ≈ 0.618
when the substrate is at "one natural deviation" from the corpus — the smallest
informatively meaningful KL. What IS one natural deviation?

The minimum meaningful KL between two distributions over 7 classes occurs when
one class probability differs by the Fibonacci-ratio jump φ⁻² ≈ 0.382 (the smallest
Fibonacci ratio, one position down from φ⁻¹). By inspection: for P_corpus and
P_substrate differing in one class by ~0.1 (a single-class 10% disagreement), KL ≈ 0.1
to 0.2 nats. Choosing scale = φ⁻¹ ≈ 0.618 maps KL = φ⁻¹ to SCR = φ⁻¹ — the substrate
is one golden-unit off from the corpus, and resonance reflects exactly one harmonic
decay. This is the substrate-canonical choice: the scale is the unit of harmonic distance,
not an arbitrary regularization constant.

### Global SCR via Fibonacci-Weighted Average

The Fibonacci numbers F[fb] (1, 1, 2, 3, 5, 8, 13, ...) increase with tier depth.
Higher tiers correspond to longer-range structural classes (8-gram, 13-gram, 21-gram)
that are harder to accumulate statistics for and more costly to get wrong, because
errors at tier fb propagate across F[fb] tokens. The Fibonacci-weighted average KL:

```
KL̄ = Σ_{fb} F[fb] · KL(P_corpus[fb] || P_substrate[fb]) / Σ_{fb} F[fb]
```

gives deeper tiers more influence on the global measure. This is correct because:

1. **Deeper tiers are more expensive to correct.** A mismatch at tier fb (a 5-gram
   structural class) manifests over 5-token windows. If the substrate overestimates
   5-gram structure while the corpus is dominated by bigrams, every 5-token generation
   window is being pushed in the wrong structural direction. This costs more than a
   1-gram mismatch which only affects a single token.

2. **The Fibonacci weights already appear in the substrate.** The FibPosTable itself
   uses Fibonacci-spaced tiers; the substrate ALREADY assigns greater structural depth
   to larger Fibonacci numbers. The Fibonacci-weighted global KL uses the same
   weighting the substrate uses internally — it measures the resonance in the
   substrate's own native units.

3. **It prevents shallow tiers from dominating.** Character-level statistics (tier 1)
   are abundant and easy to fit. If all tiers had equal weight, SCR would be dominated
   by how well the substrate fits character frequencies, not by how well it captures
   long-range structure. Fibonacci weighting ensures structural fidelity at all scales
   registers proportionally.

**Global SCR:**

```
SCR = φ^(−KL̄ / φ⁻¹)
    = φ^(−Σ_{fb} F[fb] · KL(P_corpus[fb] || P_substrate[fb]) / (φ⁻¹ · Σ_{fb} F[fb]))
```

**Properties:**
- `SCR ∈ (0, 1]` always, because KL ≥ 0 everywhere.
- `SCR = 1` iff KL̄ = 0 iff `P_corpus[fb] = P_substrate[fb]` for all fb — perfect resonance.
- `SCR = φ⁻¹` when `KL̄ = φ⁻¹` — the substrate is one golden-unit decoherent on average.
- `SCR → 0` as KL̄ → ∞ — the substrate is structurally decoherent.

---

## 2. Leaf Transition Decoherence (LTD)

```
LTD = φ^(−Σ_{k,l} T_corpus[k,l] · |T[k,l] − T_corpus[k,l]| / φ⁻²)
```

### What LTD measures

`Σ_{k,l} T_corpus[k,l] · |T[k,l] − T_corpus[k,l]|` is the **corpus-weighted
absolute deviation** of the fingerprint transition matrix from the actual corpus
transition matrix. It is a weighted L1 distance between two transition matrices,
weighted by how often each (k,l) pair actually occurs in the corpus.

This is the right weighting because:
- If T_corpus[k,l] = 0 (leaf l never follows leaf k in the corpus), a large deviation
  in T[k,l] is irrelevant — the model never encounters this pair in practice.
  The weight T_corpus[k,l] = 0 zeroes out this pair's contribution.
- If T_corpus[k,l] = 0.4 (leaf l often follows leaf k), even a small T[k,l] deviation
  matters greatly — the model's semantic grammar is routinely wrong for a common
  transition. The weight amplifies this contribution.

Corpus-weighted absolute deviation is therefore the right diagnostic for whether
the fingerprint's world model reflects what the corpus actually does.

### Why φ⁻² as the scale

φ⁻² ≈ 0.382 is the **first Fibonacci ratio** — the smallest of the golden-ratio
powers and the unit of minimum meaningful deviation in the Fibonacci metric.

To understand why this is the correct scale: consider two transition matrices that
differ by exactly one minimally meaningful amount. In the Fibonacci lattice, the
smallest non-trivial step between probability values is approximately F(k)/F(k+1) − F(k-1)/F(k)
which approaches φ⁻² − φ⁻³ = φ⁻²(1 − φ⁻¹) = φ⁻² · φ⁻² ≈ 0.146 for large tiers.
More directly: φ⁻² is the substrate's "quantization unit" — the ratio by which
consecutive Fibonacci probability steps differ. A corpus-weighted deviation of φ⁻²
means the transition matrix is off by one Fibonacci step on average.

Setting scale = φ⁻² in OmniWeight gives:
- LTD = 1.0 when deviation = 0 (T = T_corpus, exact match)
- LTD = φ⁻¹ ≈ 0.618 when deviation = φ⁻² (one Fibonacci step off on average)
- LTD = φ⁻² ≈ 0.382 when deviation = 2·φ⁻² (two Fibonacci steps off on average)
- LTD → 0 as deviation → ∞

The scale φ⁻² was already established in the homeostatic strength formula
(models_fibrec.py, `_homeostatic_strength`): "phi^{-2} ≈ 0.382 is the quantisation
step of the golden ratio — the smallest meaningful deviation unit in the Fibonacci
metric." The same reasoning applies identically to the transition matrix deviation.
Using a different scale here would introduce a competing measure of "small" with no
substrate justification.

**LTD interpretation:**
- `LTD = 1.0`: the fingerprint's semantic grammar exactly matches the corpus grammar.
- `LTD = φ⁻¹`: the transition matrix is off by one Fibonacci step on corpus-weighted average.
- `LTD → 0`: the fingerprint's world model is decoherent from the corpus it was built on.

---

## 3. Bloom-Corpus Resonance (BCR)

```
BCR = (1/|B|) · Σᵢ φ^(−‖bᵢ − c_corpus‖² / (d · φ⁻¹))
```

### What the scale d · φ⁻¹ means

Following the same dimensional analysis as the self-witness signal W_t
(from SELF_WITNESS_DERIVATION.md): in d-dimensional activation space, two randomly
initialized d-dim vectors each with `‖v‖² ≈ d` (LayerNorm output, unit variance per
component) have expected squared distance:

```
E[‖a − b‖²] = E[‖a‖²] + E[‖b‖²] − 2E[a·b] ≈ d + d − 0 = 2d
```

The scale 2d maps a randomly-aligned vector to `φ^(−1)`, i.e., the first harmonic decay.

For bloom vectors and the corpus centroid, we want the "ideal distance" to correspond
to SCR = φ⁻¹ at a scale of d · φ⁻¹ ≈ 0.618d. This choice means:

- Scale = d · φ⁻¹ maps `‖b − c_corpus‖² = d · φ⁻¹` to `BCR_i = φ⁻¹` per bloom vector.
- Scale = d maps `‖b − c_corpus‖² = d` to `BCR_i = φ⁻¹` per bloom vector.

The choice of `d · φ⁻¹` as scale (rather than `2d` or `d`) is deliberate: it places
the φ⁻¹ crossover at a distance CLOSER to the corpus centroid than the random baseline
(2d), ensuring that φ⁻¹ BCR means "near but not at the corpus centroid" — the sweet spot.

### The "golden distance" and why BCR = φ⁻¹ is ideal

**BCR = 1.0 (overfit — bad):** All bloom vectors bᵢ = c_corpus. Every bloom attractor
is identical to the corpus centroid. This means the model has only learned one attractor:
"generate something that looks like average Shakespeare." There is no diversity in the
bloom space. Self-witnessing (W_t = φ^(−‖h_t − b‖²/(2d))) collapses to a measurement
of "how far is h_t from the corpus average," which is trivially uninformative. A model
never generates h_t = c_corpus except by accident — so W_t would always be near 0,
meaning self-witnessing is always reporting deep misalignment even for good generation.
The bloom has absorbed no information about the variety of good generation states.

**BCR = φ⁻¹ (golden distance — ideal):** Each bloom vector is at the first harmonic
distance from c_corpus. This means:
1. The blooms are NOT at the corpus centroid — they have diverged into the space of
   specific good-generation patterns, not just "corpus average."
2. They are also NOT far away — they are still within one golden-unit of the corpus
   centroid, meaning they remain anchored to the corpus's territory, not drifted into
   arbitrary regions of R^d.

The golden distance is the substrate-canonical "one step away but still within the
harmonic orbit." It is exactly the distance at which OmniWeight produces its first
decay: `φ^(−1)`. Bloom vectors at golden distance from c_corpus are positioned to
create a maximally useful self-witness signal — close enough to the corpus to remain
meaningful, far enough to discriminate between different generation qualities.

**Derivation of the golden distance r*:**

Set BCR = φ⁻¹ for a single bloom vector at distance r from c_corpus:

```
φ^(−r² / (d · φ⁻¹)) = φ⁻¹

⟹  r² / (d · φ⁻¹) = 1

⟹  r² = d · φ⁻¹

⟹  r* = √(d · φ⁻¹) = √(d · 0.618...)
```

For d = 128: r* = √(128 · 0.618) ≈ √79.1 ≈ 8.89. The golden distance is
approximately 8.89 in a 128-dimensional space — about 56% of the random-vector
baseline distance of √(2d) = √256 ≈ 16.

**BCR → 0 (corpus-decoherent):** Bloom vectors have drifted far from the corpus
territory. This happens when bloom vectors are updated primarily from the model's
OWN generation (feedback loop without corpus anchoring), and the model's generation
is already decoherent from the corpus. Self-witnessing then measures "how close is
h_t to the current (decoherent) bloom" — it rewards staying near bad attractors.
The witness signal becomes self-referential noise.

---

## 4. Global Resonance Field Φ_R

```
Φ_R = SCR^φ · LTD^(φ⁻¹) · BCR^(φ⁻²)
```

### Proof that Φ_R ∈ (0, 1]

All three components are in (0, 1] (shown in Sections 1–3). For any X ∈ (0, 1]
and positive exponent α > 0: X^α ∈ (0, 1]. The product of three terms in (0, 1]
is in (0, 1]. Therefore Φ_R ∈ (0, 1]. QED.

**Φ_R = 1 iff perfect resonance:**

Φ_R = 1 iff SCR^φ = 1 AND LTD^(φ⁻¹) = 1 AND BCR^(φ⁻²) = 1.
For X ∈ (0, 1], X^α = 1 iff X = 1.
Therefore Φ_R = 1 iff SCR = 1 AND LTD = 1 AND BCR = 1.
- SCR = 1 iff `KL̄ = 0` iff substrate tier distributions exactly match corpus distributions.
- LTD = 1 iff corpus-weighted transition deviation = 0 iff T = T_corpus.
- BCR = 1 iff all bloom vectors = c_corpus (overfit, but mathematically perfect).

So Φ_R = 1 is the limit of complete resonance. QED.

### Why the exponents are φ, φ⁻¹, φ⁻²

These are the first three consecutive powers of φ⁻¹ in descending order:
φ¹ > φ⁰ = 1 > φ⁻¹ > φ⁻². The exponents form a Fibonacci-decaying sequence, which
provides the correct relative weighting for the three resonance components:

**SCR (positional structure) carries the highest exponent φ ≈ 1.618:**

SCR measures whether the substrate's structural priors at every Fibonacci tier
match the corpus's actual distributional reality. This is the FOUNDATION. Every
other component of the model — the transition matrix T, the bloom vectors, the
self-witnessing — is computed from hidden states h_t that were themselves produced
by layers whose weights were generated by seeds whose structure was set by the
Fibonacci-tier priors. If the positional priors are decoherent, everything downstream
is contaminated. A low SCR does not merely degrade one component — it degrades all
components simultaneously, because all of them flow through the substrate's structural
prior. SCR therefore deserves the highest weight.

Additionally, the exponent φ gives SCR the property that when SCR drops slightly
below 1 (small decoherence), the penalty is amplified: `(1 − ε)^φ ≈ 1 − φε`, so
even a small SCR drop inflates the global Φ_R drop by a factor φ. The model is
hypersensitive to positional structural mismatch — correctly so, because it is the
hardest to recover from.

**LTD (semantic grammar) carries exponent φ⁻¹ ≈ 0.618:**

LTD measures whether the fingerprint's world model (the T[k,l] matrix) matches
the corpus's actual leaf-transition frequencies. This is SECONDARY: while LTD
decoherence degrades semantic coherence (the CF_pos field from the Compositionality
Derivation), it does not corrupt the positional priors that generate the seeds.
A low LTD model can still have structurally sound layers — it simply has bad
semantic expectations about what token classes follow each other.

LTD decoherence is also more recoverable: the fingerprint transition matrix can
be updated from additional corpus exposure, while the positional priors in FibPosTable
are baked into the architecture's inductive bias.

**BCR (style attractor) carries exponent φ⁻² ≈ 0.382:**

BCR measures whether the bloom vectors have remained anchored to the corpus's
distributional territory. This is TERTIARY: bloom decoherence corrupts the
self-witness signal W_t, but only if the blooms have drifted far. The bloom vectors
learn from actual generation via the gradient path (L → h_t → W_t → b_nearest),
so they have an online correction mechanism that LTD and SCR do not. They are the
most recoverable of the three.

Furthermore, BCR ≠ 1 is actually DESIRED (golden distance, not collapse to
c_corpus), so the weighting must not be too severe — φ⁻² keeps BCR from
dominating the global measure.

**The Fibonacci exponent sequence is internally consistent:**

The three exponents φ, φ⁻¹, φ⁻² satisfy the Fibonacci recurrence in exponent
space: φ⁻² + φ⁻¹ = φ⁻¹(φ⁻¹ + 1) = φ⁻¹ · φ = 1 (the identity φ⁻¹ + φ⁻² = 1).
And φ⁻¹ + 1 = φ (the defining Fibonacci identity). The weight sequence is not
arbitrary — it is the Fibonacci sequence written in powers of φ⁻¹, the same
sequence that generates the entire substrate structure.

---

## 5. Decoherence Detection and Correction

### Per-tier decoherence vector

```
D_fb = P_substrate[fb] − P_corpus[fb]   ∈ R⁷
```

`D_fb[s] > 0`: the substrate overestimates class s at tier fb — it assigns too much
probability to the s-th structural class (e.g., 5-gram patterns) while the corpus
rarely uses that class at this position tier.

`D_fb[s] < 0`: the substrate underestimates class s at tier fb — it assigns too
little probability to a class the corpus frequently uses at this tier.

### Decoherence correction signal

```
Δ_fb = −φ⁻¹ · D_fb / (‖D_fb‖ + ε)
```

This is the unit-normalized direction of the correction (toward reducing D_fb),
scaled by φ⁻¹ so corrections are gentle. The normalization `D_fb / (‖D_fb‖ + ε)`
extracts the DIRECTION of decoherence without amplifying large deviations into
large corrections (which would risk overshooting into the opposite decoherence).

The factor φ⁻¹ ≈ 0.618 is the substrate-canonical correction rate. It is the
same coefficient used in the seed recurrence modulation (SELF_WITNESS_DERIVATION.md,
eq. 3): small enough to never overcorrect, large enough to make progress each step.

### Update rule that preserves Fibonacci-tier structure

The constraint is: `P_substrate[fb]` must remain a valid probability distribution
(all entries ≥ 0, sum = 1) and the SHAPE of the update must respect the Fibonacci-tier
hierarchy — deeper tiers (larger fb) should update more slowly, because they represent
longer-range structure that the model has less corpus evidence for.

```
P_substrate[fb] ← P_substrate[fb] + α_fb · Δ_fb
```

where the per-tier learning rate is:

```
α_fb = φ^(−F[fb] / φ) · SCR_fb
```

**Why this form for α_fb:**

The term `φ^(−F[fb] / φ)` is the OmniWeight applied to the Fibonacci number F[fb]
at scale φ. Tier 1 (F=1): `φ^(−1/φ) ≈ φ^(−0.618) ≈ 0.675` — update at 67.5%.
Tier 3 (F=3): `φ^(−3/φ) ≈ φ^(−1.854) ≈ 0.31` — update at 31%.
Tier 7 (F=13): `φ^(−13/φ) ≈ φ^(−8.03) ≈ 0.037` — update at 3.7%.

Deeper tiers (larger Fibonacci numbers) update more slowly, matching the intuition:
we have abundant evidence about character-level structure (tier 1) from the 256-token
seed, but sparse evidence about 21-gram structure (tier 7). Slow updates at deep tiers
prevent the model from overwriting long-range structural priors on the basis of
limited corpus evidence.

The term `SCR_fb` gates the correction by the current tier resonance: if SCR_fb is
already near 1.0 (this tier is resonant), α_fb ≈ φ^(−F[fb]/φ) — normal update.
If SCR_fb is near 0 (this tier is deeply decoherent), α_fb ≈ 0 — the tier is so
decoherent that our corpus statistics are unreliable enough that aggressive correction
could overshoot. The gate ensures that severely decoherent tiers receive gentle
correction while moderately decoherent tiers receive full correction. This is the
same OmniWeight immune response principle as `_homeostatic_strength` in models_fibrec.py.

**Post-step normalization to preserve probability simplex:**

After the additive update:

```
P_substrate[fb] ← max(P_substrate[fb], ε)       # clip to non-negative
P_substrate[fb] ← P_substrate[fb] / sum(P_substrate[fb])  # re-normalize to simplex
```

The ε clip prevents any class from being driven to negative probability (which can
happen if D_fb points too strongly against a class with already-small probability).
Re-normalization ensures the constraint `Σ_s P_substrate[fb][s] = 1` holds after each step.

**Full update rule:**

```
Δ_fb     = −φ⁻¹ · D_fb / (‖D_fb‖ + ε)
α_fb     = φ^(−F[fb] / φ) · SCR_fb
P_new    = P_substrate[fb] + α_fb · Δ_fb
P_new    = max(P_new, ε)
P_substrate[fb] ← P_new / sum(P_new)
```

**Why this preserves Fibonacci-tier structure:**

Each tier's update is independently gated by α_fb which decays with F[fb]. The
substrate's multi-tier hierarchy is intact after every update: tier 1 always
corrects faster than tier 7. The correction direction is unit-normalized so the
scale is always controlled by α_fb alone. The normalization step ensures that the
7 classes at each tier remain a properly normalized probability distribution over
structural categories — the update never distorts the tier's distributional form,
only its content.

---

## 6. The Master Insight: What Low Φ_R Explains

### 6.1 The T20 Bloom Ceiling

T20 refers to the empirically observed ceiling where generation quality stops improving
around 20 bloom vectors even when the model architecture is extended. The bloom ceiling
follows directly from SCR decoherence.

Bloom vectors are updated toward the model's own high-W_t hidden states:

```
b_i ← β · b_i + (1 − β) · h_{t*}   where t* = argmax_t W_t
```

But W_t = φ^(−‖h_t − b_nearest‖²/(2d)) — the self-witness signal. If the substrate
is positionally decoherent (low SCR), then the hidden states h_t are generated by
layers whose seeds were built from incorrect structural priors. The h_t vectors
systematically OCCUPY THE WRONG REGION of R^d — a region adapted to the substrate's
false priors, not to the corpus's actual structure.

The bloom vectors chase these misaligned h_t vectors. After convergence, the bloom
vectors cluster in the substrate's preferred region, not the corpus's preferred region.
With 20 bloom vectors all anchored to approximately the same decoherent region of R^d,
adding more bloom vectors (21, 22, ...) provides no additional coverage of the corpus's
actual good-generation territory. The ceiling is not a capacity limit — it is a
domain limit. The model has exhausted the decoherent region's representation, and
no more vectors within that region can meaningfully discriminate generation quality.

**In decoherence terms:** with low SCR, the bloom vectors all have BCR << φ⁻¹ — they
are far from c_corpus (because c_corpus represents the corpus's territory, not the
substrate's). The effective diversity of the bloom space collapses: all bloom vectors
are far from the corpus centroid and near each other. Self-witnessing no longer
provides useful discrimination because ALL generation states are equally far from
ALL bloom vectors — the witness signal is saturated at low values for everything.

T20 is not a Fibonacci-special number. It is the number of bloom vectors at which
the substrate's decoherent region fills up and adding more vectors ceases to
cover new territory.

### 6.2 The "against/upon/were/there" Attractor Loop

These specific tokens appear as generation attractors because they are high-frequency
English function words that the corpus assigns to dominant leaf clusters in the
fingerprint. The attractor loop arises from the interaction of three decoherence effects:

**Effect 1 — SCR decoherence at shallow tiers (char/bigram level):**

If P_substrate[fb=1] (the character-level structural prior) overestimates certain
character class frequencies relative to P_corpus[fb=1], the seeds at layer 0 and 1
are weighted toward generating representations that discriminate character-level
patterns the corpus doesn't actually have. The model's output distribution then
assigns inflated probability to words whose character structure matches the substrate's
false character-level priors.

Function words ("of," "the," "in," "upon," "were") are short, common, and character-
structurally simple. When the character-level prior is decoherent toward
common English character n-grams (which short function words provide), these words
become the degenerate attractors — not because they are meaningful completions but
because they match the decoherent prior's character-level expectations.

**Effect 2 — LTD decoherence in the T[k,l] matrix:**

If T[k,l] ≠ T_corpus[k,l] for the leaf clusters containing these function words,
the coherence field CF_pos (from the Compositionality Derivation) is based on wrong
transition probabilities. Specifically: if T overestimates the probability that
certain leaf clusters transition to the function-word cluster, CF_pos for sequences
ending in function words is artificially high. A high CF_pos means the compositionality
gain amplifies the weights for those paths:

```
W^(composed) = W · φ⁻¹ · (1 + ρ_n · CF_pos)
```

When CF_pos is artificially inflated for function-word paths, those paths are
disproportionately amplified relative to content-word paths. The model preferentially
generates into the function-word leaf cluster — and then T[k,l] says the function-word
cluster most likely transitions back to itself or another function-word cluster.
The loop is self-reinforcing.

**Effect 3 — BCR decoherence making bloom vectors cluster in the function-word region:**

When SCR and LTD are both decoherent, the hidden states h_t that receive the highest
W_t scores (closest to bloom vectors) are those in the function-word region of R^d
(because that's where the decoherent substrate tends to place its representations).
The bloom vectors converge toward this region. Self-witnessing then rewards generation
that stays in this region — which means staying near function words. The witness loop
has learned to maximize its own signal by staying in the region where its own signal
is high. This is decoherent self-amplification, not quality-aligned generation.

The "against/upon/were/there" tokens are the specific members of the function-word
cluster that happen to be equidistant from the decoherent bloom vectors. They form
the attractor because the substrate's wrong priors made the bloom vectors converge
to the function-word region, and the witness then locks generation into that region.

### 6.3 Self-Witnessing Under Bloom Decoherence

The self-witness equation:

```
W_t = φ^(−‖h_t − b_nearest‖² / (2d))
```

requires that b_nearest be a meaningful attractor — a position in R^d that corresponds
to good generation, anchored to the corpus's distributional territory (BCR ≈ φ⁻¹).

If bloom vectors are corpus-decoherent (BCR << φ⁻¹, meaning ‖bᵢ − c_corpus‖² >> d·φ⁻¹
for all i), then W_t is measuring proximity to attractors that ARE NOT IN THE CORPUS'S
TERRITORY. Concretely:

- A hidden state h_t that IS generating good, corpus-consistent text will have
  h_t near c_corpus (it is generating something that looks like Shakespeare).
  But b_nearest is far from c_corpus (it is the decoherent bloom vector in the
  wrong region of R^d). So ‖h_t − b_nearest‖² is large → W_t is small.
  Self-witnessing tells the model: "you are far from your attractors" — even though
  you are doing well by corpus standards.

- A hidden state h_t that IS generating bad, corpus-inconsistent text may happen
  to be near a decoherent bloom vector (the bloom vector that trained on bad-but-
  substrate-native states). So ‖h_t − b_nearest‖² is small → W_t is large.
  Self-witnessing tells the model: "you are near your attractors" — even though
  you are doing poorly by corpus standards.

**The catastrophic consequence:**

Self-witnessing, when bloom vectors are corpus-decoherent, is measuring the WRONG
THING. Instead of rewarding corpus-aligned generation and penalizing degenerate
generation, it is rewarding substrate-aligned generation — generation that is native
to the substrate's own (decoherent) territory. The seed modulation (equation 3 from
the Self-Witness Derivation):

```
seed_n = (A + φ⁻¹ · ΔC_t · I_K) · seed_{n-1} + B · seed_{n-2}
```

will STABILIZE seeds that produced decoherent states (those with high W_t due to
proximity to decoherent bloom vectors), and DESTABILIZE seeds that produced corpus-
consistent states (those with low W_t despite quality generation). The model's immune
system is inverted: it defends the disease and attacks the cure.

**In summary:**

Bloom-decoherent self-witnessing does not measure "how well am I generating corpus-
consistent text?" It measures "how well am I staying near my (decoherent) learned
attractors?" These two questions have the same answer only when the bloom vectors
are at golden distance from c_corpus (BCR = φ⁻¹). When BCR << φ⁻¹, the witness
signal is measuring adherence to the wrong reality — and the seed modulation acts
to strengthen that wrong reality, further cementing decoherence with each generation
step.

---

## Summary: The Resonance Diagnostics

| Quantity | Formula | Range | Perfect | Decoherent |
|---|---|---|---|---|
| SCR_fb | φ^(−KL(P_corpus[fb]‖P_substrate[fb]) / φ⁻¹) | (0, 1] | 1.0 | →0 |
| SCR | φ^(−KL̄ / φ⁻¹) | (0, 1] | 1.0 | →0 |
| LTD | φ^(−Σ T_corpus‖T−T_corpus‖ / φ⁻²) | (0, 1] | 1.0 | →0 |
| BCR | (1/|B|)·Σᵢ φ^(−‖bᵢ−c_corpus‖²/(d·φ⁻¹)) | (0, 1] | φ⁻¹ (golden) | →0 |
| Φ_R | SCR^φ · LTD^(φ⁻¹) · BCR^(φ⁻²) | (0, 1] | 1.0 | →0 |

**Computability:**

| Quantity | Source objects | Operation |
|---|---|---|
| P_corpus[fb] | tinyshakespeare 256-token seed | Count structural classes per position, bin by Fibonacci tier |
| P_substrate[fb] | FibPosTable.probs | Direct read |
| T_corpus[k,l] | 256-token seed + fingerprint.sub_clusters | Count leaf transition pairs |
| T[k,l] | fingerprint.cluster_transition | Direct read |
| c_corpus | substrate_embed.weight weighted by corpus frequency | Weighted mean of embedding rows |
| bᵢ | bloom_vecs (registered buffer) | Direct read |
| d | model.d_model | Constant |

All quantities are computable from existing objects without new trainable parameters.
The resonance measurement is a purely diagnostic operation — no forward pass, no gradient.

---

## The Organism Analogy

A fish adapted to cold water and placed in warm water does not die instantly. It can
survive for a time. But its metabolic enzymes are calibrated to the wrong temperature —
every biochemical reaction is slightly off, every protein's folding is slightly wrong.
The fish can eat, move, and respond. But it cannot thrive, and it cannot grow.

FibRecLMSubsim with a decoherent substrate is that fish. The model generates text —
sometimes reasonable text. But every structural prior is calibrated to a corpus reality
that does not match the actual corpus. The seeds produce layers whose spectral structure
is appropriate for the substrate's imagined corpus, not the actual one. The bloom vectors
converge to the wrong territory. The self-witness signal measures adherence to the
wrong attractors. The coherence field uses wrong transition probabilities.

The model is perfectly adapted to the wrong reality.

Φ_R is the thermometer. SCR_fb is the per-organ health reading. D_fb is the diagnosis.
Δ_fb is the prescription. The update rule is the treatment.

A Φ_R below 0.5 means the substrate is not resonating with the corpus — the model is
fighting its own biology. Above 0.8 means the model is in its environment. A Φ_R of 1.0
would mean the substrate's priors ARE the corpus — the model is the environment.
