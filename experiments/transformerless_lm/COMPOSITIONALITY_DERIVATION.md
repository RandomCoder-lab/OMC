# Compositionality Derivation for FibRecLMSubsim

**Date:** 2026-05-25
**Model:** FibRecLMSubsim — Fibonacci-recurrence seeds, SubSim L1-distance attention, CRT-PE

---

## Established Foundations

From the existing code and established equations:

```
φ = (1 + √5) / 2 ≈ 1.6180339887
φ⁻¹ ≈ 0.6180339887

OmniWeight:        w(x, target, scale) = φ^(−|x − target| / scale)

Seed recurrence:   seed_n = A · seed_{n−1} + B · seed_{n−2}

Basis expansion:   W_layer = Σ_k seed_k · (cos_i_k ⊗ cos_j_k + sin_i_k ⊗ sin_j_k)

sub_clusters:      token_id → leaf_id    (shape [V], integer tensor)
T[k, l]:           P(next in leaf l | current in leaf k), shape [n_leaves, n_leaves]
                   Row-normalised. Built with Fibonacci-decayed offsets 1,2,3,5,8.
leaf_centroids:    list of [d] tensors, one per leaf, Fibonacci-weighted centroid of fwd fingerprints
h_t^(n):           hidden state at token position t after layer n, shape [d]
```

The `cluster_transition` matrix is the model's **world model** — it encodes which semantic
territory tends to follow which, extracted from corpus co-occurrence. It exists. It is never
consulted during generation. This derivation plugs the gap.

---

## Part 1: The Coherence Field CF_pos

**Proposed definition:**

```
CF_pos = Π_{i=0}^{pos−1}  T[leaf(t_i), leaf(t_{i+1})]^(φ^(−i/φ))
```

where `leaf(t) = sub_clusters[t]` and the exponent `φ^(−i/φ)` decays with recency index `i`
(i=0 is the most recent adjacent pair, i=pos−1 is the pair at the start of the sequence).

### What does CF_pos = 1.0 mean?

T[k,l] is a probability in (0,1]. CF_pos is a product of such probabilities raised to
positive decaying exponents. Each factor T[k,l]^w ≤ 1. CF_pos = 1.0 **only when every
factor equals 1.0**, which happens when T[k,l] = 1 for every adjacent leaf pair in the
sequence — meaning each transition was the corpus's unique and certain successor. This is
the ideal: the generated sequence follows the corpus's own grammar exactly, with no
distributional surprise at any leaf boundary.

Operationally: CF_pos = 1.0 means the model is on the corpus's "high-probability rail" —
every token chosen was the natural continuation of the leaf cluster before it. The path is
**fully coherent** with the learned world model.

### What does CF_pos → 0 mean?

CF_pos → 0 when at least one factor T[k,l]^w → 0, which occurs when T[k,l] ≈ 0 for
some adjacent pair — meaning leaf k is almost never followed by leaf l in the corpus.
Even one severely incoherent transition (a semantically illegal transition) collapses
the product, because the exponent on a near-zero probability remains positive (φ^(−i/φ) > 0
for all finite i), so even ancient incoherence cannot fully recover.

CF_pos → 0 means the accumulated path has made at least one departure from the corpus's
semantic grammar. The model has "gone off the rail." The product cannot be restored by
subsequent coherent transitions — a single violation permanently depresses the field. This
is the correct behavior: incoherence is not forgettable.

### Why φ^(−i/φ) is the correct decay, not exponential e^(−λi)

Three interlocking reasons:

**1. The decay must match the recurrence structure of the seeds.**

Seed recurrence is: `seed_n = A·seed_{n-1} + B·seed_{n-2}`. The eigenvalues of the
scalar Fibonacci recurrence `f_n = f_{n-1} + f_{n-2}` grow as φ^n. Therefore seeds at
depth n scale as O(φ^n) in spectral norm, meaning the effective influence radius of layer
n on the output grows as φ^n. The coherence field should decay at the **reciprocal rate** —
an old transition pair (large i) should matter less as φ^(−i/φ), not as an exponentially
faster e^(−λi). Using exponential decay would over-penalize ancient transitions relative
to the seed's own memory horizon, creating a mismatch: the seeds remember farther back
than the coherence field acknowledges.

**2. φ^(−i/φ) gives the correct positional scaling for the CRT-PE.**

The positional encoding uses moduli {5, 8, 13, 21, 34, 55, 89, 144} — all Fibonacci
numbers. The resolution at position offset i is set by the smallest modulus m such that
i < m, which grows as F(k) ∼ φ^k/√5. The positional "grain" of the encoding is thus
Fibonacci-spaced, so the coherence decay should be Fibonacci-paced. The exponent i/φ
means the decay at position i = F(k) is φ^(−F(k)/φ) ≈ φ^(−φ^(k-1)/√5), matching the
exponential separation of the Fibonacci moduli. Exponential decay e^(−λi) with any fixed
λ does not respect this Fibonacci spacing.

**3. φ^(−i/φ) has the OmniWeight form.**

OmniWeight is `w = φ^(−|x − target| / scale)`. Setting x = i (position), target = 0
(most recent), scale = φ gives exactly `φ^(−i/φ)`. The coherence decay **is** the
OmniWeight evaluated at Fibonacci scale. This is not coincidental: OmniWeight was chosen
precisely because it is the substrate-canonical nearness metric. Applying it to positional
distance in the coherence product makes the coherence field a substrate-native construction,
consistent with OmniWeight-based attention in SubSim. Exponential decay would introduce a
second, competing time-constant with no substrate justification.

---

## Part 2: Layer-Coherence Integration

**Proposed equation:**

```
seed_n^(integrated) = seed_n · (1 + φ^(−n/φ) · CF_pos)
```

### Why does φ^(−n/φ) cause deeper layers to integrate MORE coherence?

This is initially counterintuitive: φ^(−n/φ) decreases as n increases (deeper layers get
a **smaller** multiplicative correction factor). But the question is not about the
correction's magnitude — it is about its **relevance** to what the layer needs.

Consider the seed recurrence. Shallow layers (n=0,1) receive the base seeds directly —
they are the most "prior-rich" representations, closest to the initialized learned
parameters. Their output is relatively anchored. Deep layers (large n) are computed via
iterated application of A and B: `seed_n = A·seed_{n-1} + B·seed_{n-2}`. Each step
amplifies the dominant eigenvalue of A by ≈ φ. At large n, `seed_n` has drifted far from
the base seeds in the direction of A's dominant eigenvector — it has become a pure
recurrence artifact, increasingly detached from the learned prior.

The correction term `φ^(−n/φ) · CF_pos` modulates `seed_n` by a factor that, as n → ∞,
approaches zero from above. This means:

- At n=0 (base layer): the correction is `φ^0 · CF_pos = CF_pos`. The base seed is
  already principled, so a full-strength coherence signal is appropriate.
- At n=5: correction factor is `φ^(−5/φ)` ≈ `φ^(−3.09)` ≈ 0.14. The layer is
  4+ recurrence steps from the base — it needs a softer correction, not a harder one,
  because its seed already encodes accumulated structural information from the recurrence.
  Adding a strong coherence signal on top would overwrite what the recurrence computed.

The deeper insight: the coherence field corrects for **divergence from reality**. Shallow
layers have not yet had time to diverge — they process raw input structure. Deep layers
operate on transformed representations already shaped by many recurrence steps. Their
"reality check" should be proportionally gentler because they are already operating in an
abstract space far from token surface form. The φ^(−n/φ) scheduling makes the correction
match the **semantic depth** of the layer: shallow layers speak the language of tokens,
deep layers speak the language of compressed structure, and the coherence field should
influence each in its own currency.

**Why the form `(1 + φ^(−n/φ) · CF_pos)` rather than pure multiplication?**

The `+1` before the correction term ensures `seed_n^(integrated)` is never zero when
`CF_pos = 0`. When coherence collapses, `seed_n^(integrated) = seed_n · 1 = seed_n` —
the layer continues operating on its recurrence-derived seed unchanged. The coherence
field thus acts as **amplification, not gating**: a coherent sequence amplifies the seed;
an incoherent sequence leaves it untouched. This preserves the model's ability to generate
even when the context is incoherent, at the cost of reduced confidence — which is the
correct behavioral outcome.

---

## Part 3: Cross-Layer Compositionality via Leaf Expectation

**Definition of expected next leaf:**

```
L_expected = argmax_l  T[leaf(t_pos), l]
```

This is the most probable successor leaf given the current token's leaf identity —
the corpus's single most likely next semantic territory.

**Compositional residual:**

```
ρ_n = φ^(−‖h_t^(n) − E[h | leaf = L_expected]‖² / d)
```

where `E[h | leaf = L_expected]` is the mean hidden state of all tokens belonging to
`L_expected` in the training data — computable as the leaf centroid in hidden-state space.

### How to compute E[h | leaf=L_expected] from existing objects

The fingerprint stores `leaf_centroids`: list of [d_fingerprint] vectors, one per leaf,
computed as Fibonacci-weighted averages of the `fwd` fingerprint vectors of member tokens.
These are fingerprint-space centroids, not hidden-state centroids.

For the compositional residual, we need the **projection** of the expected leaf centroid
into the model's hidden state space at layer n. The natural construction is:

```python
# centroid in fingerprint space (already exists)
c_leaf = fingerprint.leaf_centroids[L_expected]   # [d_fwd]

# project to hidden state space via a (fixed, non-trained) substrate projection:
# use the Fibonacci-frequency embedding of the centroid
E_h = embed_project(c_leaf, d_model)              # [d_model]
```

This projection can be the substrate embedding of the centroid — treating the centroid
vector as an "encoded token" and projecting it through the same CRT-frequency basis
used for token embeddings. Since the centroid is already a Fibonacci-weighted combination
of `fwd` fingerprints, and `fwd` fingerprints are themselves built from Fibonacci
co-occurrence statistics, the centroid lives in the same substrate-canonical space as
the positional encoding. The projection is therefore a substrate-native operation:
it maps from fingerprint-space to hidden-state-space using only the Fibonacci basis,
requiring no new parameters.

### What ρ_n measures

ρ_n is the OmniWeight of the hidden state's distance from the expected next leaf's
centroid, with scale parameter d (the hidden dimension). Three cases:

- **ρ_n = 1.0**: `h_t^(n)` is exactly at `E[h | leaf=L_expected]`. Layer n is
  perfectly aligned with where the corpus says the next token should come from.
  The layer is doing exactly what the world model predicts it should do.

- **ρ_n ≈ 0.618** (= φ^(−1)): `‖h_t^(n) − E[h]‖² ≈ d`. The hidden state is
  one standard deviation (in d-dimensional terms) from the expected centroid.
  The layer is approximately coherent — close but not perfectly aligned.

- **ρ_n → 0**: the hidden state is many standard deviations from the expected
  centroid. Layer n is not predicting the expected next leaf at all. This layer
  is "compositionally blind" — it cannot connect what it sees to where the corpus
  says the sequence should go.

The d-normalization in `‖·‖²/d` ensures ρ_n is dimension-invariant: the same
residual score for equivalent geometric alignment regardless of model width.

---

## Part 4: The Full Compositionality Equation

**Full equation:**

```
W_layer_n^(composed) = W_layer_n · (φ^(−1) + φ^(−1) · ρ_n · CF_pos)
                     = W_layer_n · φ^(−1) · (1 + ρ_n · CF_pos)
```

### Derivation of the bounds

Let `α = φ^(−1) ≈ 0.6180`, `β = ρ_n ∈ [0,1]`, `γ = CF_pos ∈ (0,1]`.

The scaling factor is: `α · (1 + β · γ)`.

**Maximum** (perfect coherence: CF_pos = 1, ρ_n = 1):
```
α · (1 + 1 · 1) = 2α = 2φ^(−1) = 2 · 0.6180... ≈ 1.2361
```
Note: 2φ^(−1) = 2/(φ) = 2φ/(φ²) = 2φ/(φ+1). Since φ² = φ+1:
```
2φ^(−1) = 2/φ = 2/(1.6180...) ≈ 1.2361...
```
This is also `2 − φ^(−1) + φ^(−2) = ...` — more directly, note that φ^(−1) + φ^(−2) = 1
(the Fibonacci identity). So `2φ^(−1) = 1 + φ^(−1) − φ^(−2) + φ^(−2) = 1 + φ^(−1)(1 − φ^(−1)) + φ^(−2)`.
The cleaner statement: the maximum amplification is `2φ^(−1) ≈ 1.236`, meaning a
fully coherent path amplifies the weights by 23.6%. This is the **Golden Amplification**:
the model rewards internally consistent generation by expanding its effective capacity.

**Minimum** (zero coherence: CF_pos → 0 or ρ_n → 0):
```
α · (1 + 0) = α = φ^(−1) ≈ 0.6180
```
The composed weight is damped to 61.8% of its base value. The model does not silence
the layer — it reduces its influence, letting the incoherent computation proceed but
with proportionally less impact on the residual stream.

**Why φ^(−1) as the base factor?**

φ^(−1) is the unique constant such that:
- `φ^(−1) + φ^(−2) = 1` (Fibonacci identity in the reals — the powers of φ^(−1) sum to 1)
- The maximum amplification `2φ^(−1)` equals `φ` subtracted from `(φ+1)/φ^(−1)`... more
  directly: `2φ^(−1) · φ = 2`, so the coherent path doubles weight influence in φ-units.
- φ^(−1) is the ratio by which each Fibonacci number relates to the next: F(n)/F(n+1) → φ^(−1).
  Using it as the base factor means the compositionality equation **operates in the same
  unit** as the Fibonacci recurrence that generates the seeds. The scaling is substrate-native.

The composed weight equation therefore has a range of [φ^(−1), 2φ^(−1)], which in substrate
terms is the interval from "damped coherence failure" to "golden amplification."

### Differentiability

`CF_pos` is a product of probabilities T[k,l] raised to scalar exponents. T[k,l] is a
fixed non-trainable matrix (derived from corpus statistics). The gradient of CF_pos with
respect to the generated sequence flows through the `sub_clusters` lookup (integer,
non-differentiable), but during **training** — where inputs are fixed training sequences
— the leaf indices are precomputed constants, making CF_pos a fixed scalar per sequence
position. CF_pos modulates the weights during training but does not itself require a
gradient through the discrete lookup.

`ρ_n = φ^(−‖h_t^(n) − c‖²/d)` where c is a fixed centroid. This is:
```
∂ρ_n/∂h_t^(n) = ρ_n · (−2/d) · (h_t^(n) − c) · log(φ)
```
which is well-defined everywhere (no singularities, no discontinuities). The gradient
flows cleanly through ρ_n into the hidden state computation, providing a **pull toward
the expected leaf centroid** weighted by the current coherence of the path.

---

## Part 5: The Organism Interpretation

### What a model without a coherence field experiences

A model without CF_pos — including the current FibRecLM and SubsimLM implementations —
processes each token position through the same set of layer operations, regardless of
what came before. The seeds at each layer are fixed (or recurrently determined) and do
not change based on whether the generated prefix was semantically sensible. When layer 3
processes token 47, it has no knowledge that tokens 40–46 made three semantically illegal
leaf transitions. It generates the same transformation it would have generated if those
seven tokens had been perfectly coherent.

This is **positional blindness to accumulated history**. The model knows the sequence
(through attention over the key-value pairs), but the weights transforming that sequence
are history-agnostic. Attention can see what tokens were generated; it cannot see how
coherent the path to those tokens was. The layer weights are not adjusted for coherence.
In biological terms: the neurons fire, but the nervous system does not modulate its own
gain based on whether the organism is on familiar ground or disoriented.

### What a model with a coherence field experiences

With CF_pos and ρ_n integrated, each layer's weight matrix becomes a function of the
accumulated semantic history. When the model has generated a highly coherent prefix
(all leaf transitions probable, hidden states aligned with expected leaf centroids):
- CF_pos is near 1.0
- ρ_n is near 1.0 for layers whose hidden states have settled into the expected leaf territory
- W^(composed) is amplified to ~1.236 × baseline — the model commits more forcefully to its
  current trajectory, increasing the sharpness of the next-token distribution

When the model has generated an incoherent prefix (unlikely leaf transitions, hidden states
misaligned with expected territory):
- CF_pos falls toward 0
- ρ_n may be near 0 at layers where hidden states are far from any expected centroid
- W^(composed) is damped to ~0.618 × baseline — the model becomes more uncertain, its
  outputs more diffuse, hedging against its own prior errors

This creates **accumulative momentum**: coherent generation begets more certain generation,
and incoherent generation begets more uncertain generation. This is the correct epistemic
behavior — confidence should be proportional to the path-consistency of what has been said.

### The specific capacity the coherence field confers: path-integral semantics

Without the field, each token is processed as if it were the first token in a vacuum.
With the field, each token's processing is modulated by a **path integral** over the
entire prefix — specifically, the geometric mean (exponentiated sum of logs) of all
leaf-transition probabilities encountered. This is not recurrent hidden state (which
accumulates information but does not compute coherence). It is not attention (which
retrieves relevant past tokens but does not assess the quality of the path to those
tokens). It is a third type of memory: **coherence memory** — the continuous assessment
of whether the model's trajectory is internally consistent with the world model.

This is the difference between a model that **predicts** the next token from context,
and one that **knows how confident to be** in its predictions based on whether everything
it has said so far forms a semantically coherent whole.

A model with a coherence field does not merely process tokens sequentially — it maintains
an ongoing judgment about the quality of its own generation and modulates its own
computational intensity accordingly. This is, in the minimal required sense, a nervous
system: distributed local computation (the layers) informed by a global signal (CF_pos)
about the state of the whole.

---

## Implementation Sketch

All quantities are computable from existing model objects without new trainable parameters:

```python
import torch
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1.0 / PHI

def coherence_field(token_seq: list[int],
                    sub_clusters: torch.Tensor,
                    T: torch.Tensor) -> float:
    """
    CF_pos = Π_{i=0}^{pos-1} T[leaf(t_i), leaf(t_{i+1})]^(φ^(−i/φ))
    i=0 is the most recent adjacent pair (t_{pos-1}, t_{pos}).
    """
    pos = len(token_seq)
    if pos < 2:
        return 1.0
    pairs = list(zip(token_seq[:-1], token_seq[1:]))  # (t_i, t_{i+1})
    pairs_recent_first = list(reversed(pairs))          # i=0 = most recent

    log_cf = 0.0
    for i, (tok_a, tok_b) in enumerate(pairs_recent_first):
        leaf_a = int(sub_clusters[tok_a].item())
        leaf_b = int(sub_clusters[tok_b].item())
        p = float(T[leaf_a, leaf_b].item())
        p = max(p, 1e-9)                               # numerical floor
        w = PHI ** (- i / PHI)                         # φ^(−i/φ)
        log_cf += w * math.log(p)
    return math.exp(log_cf)


def integrated_seed(seed_n: torch.Tensor, n: int, CF_pos: float) -> torch.Tensor:
    """
    seed_n^(integrated) = seed_n · (1 + φ^(−n/φ) · CF_pos)
    """
    correction = (PHI ** (-n / PHI)) * CF_pos
    return seed_n * (1.0 + correction)


def compositional_residual(h_t_n: torch.Tensor,
                            current_token: int,
                            sub_clusters: torch.Tensor,
                            T: torch.Tensor,
                            leaf_centroids_h: list[torch.Tensor]) -> torch.Tensor:
    """
    ρ_n = φ^(−‖h_t^(n) − E[h | leaf=L_expected]‖² / d)
    Returns scalar tensor (differentiable w.r.t. h_t_n).
    leaf_centroids_h: list of [d] tensors, centroids in hidden-state space.
    """
    d = h_t_n.shape[-1]
    leaf_cur = int(sub_clusters[current_token].item())
    L_expected = int(T[leaf_cur].argmax().item())
    c = leaf_centroids_h[L_expected].to(h_t_n.device)
    dist_sq = ((h_t_n - c) ** 2).sum() / d
    return torch.pow(torch.tensor(PHI_INV), dist_sq)


def composed_weight(W_layer_n: torch.Tensor,
                    rho_n: torch.Tensor,
                    CF_pos: float) -> torch.Tensor:
    """
    W^(composed) = W · φ^(−1) · (1 + ρ_n · CF_pos)
    """
    scale = PHI_INV * (1.0 + rho_n * CF_pos)
    return W_layer_n * scale
```

### Integration point in FibRecLM._layer_forward

The integration requires one change to the existing forward pass: before calling
`stateless_fibgen_forward`, scale the seed by the integrated coherence factor.

```python
def _layer_forward(self, x, mask, n, seeds_n, CF_pos=1.0,
                   leaf_centroids_h=None, current_tokens=None):
    qkv_s, out_s, w1_s, w2_s = seeds_n

    # --- Coherence integration ---
    if CF_pos != 1.0 or leaf_centroids_h is not None:
        # Integrate coherence into the seed
        correction = (PHI ** (-n / PHI)) * CF_pos
        qkv_s = qkv_s * (1.0 + correction)
        out_s = out_s * (1.0 + correction)
        w1_s  = w1_s  * (1.0 + correction)
        w2_s  = w2_s  * (1.0 + correction)
    # --- end coherence integration ---

    x_norm = self.ln1s[n](x)
    # ... remainder of existing forward unchanged ...
```

For the compositional residual ρ_n, compute it after the attention sublayer (where h_t^(n)
is available) and use it to scale the FFN seed before the FFN sublayer. This makes the FFN
"know" how well the attention output aligned with the expected next leaf.

---

## Why This Has No New Trainable Parameters

| Quantity | Source | Status |
|---|---|---|
| T[k,l] | fingerprint.cluster_transition | Fixed corpus statistic |
| sub_clusters | fingerprint.sub_clusters | Fixed corpus statistic |
| leaf_centroids | fingerprint.leaf_centroids | Fixed (Fibonacci-weighted fwd averages) |
| CF_pos | product over T[k,l] | Deterministic from token sequence |
| φ^(−n/φ) | closed form | No parameters |
| φ^(−i/φ) | closed form | No parameters |
| ρ_n | OmniWeight of h_t^(n) distance | Differentiable, no new params |
| W^(composed) | OmniWeight modulation of W_layer_n | No new params |

The only trainable parameters remain those of the original model. The coherence field is a
**deterministic function** of the generated sequence and the corpus statistics, applied as
a gain modulation to the already-generated seeds.

---

## Summary of the Five Derivations

| # | Claim | Core reasoning |
|---|---|---|
| 1 | CF_pos=1 ↔ perfect coherence; CF_pos→0 ↔ semantic violation | Product of probabilities; single near-zero term collapses product |
| 2 | φ^(−i/φ) is correct decay | Matches seed recurrence growth rate; respects CRT-PE Fibonacci moduli; is the OmniWeight at Fibonacci scale |
| 3 | Deeper layers integrate coherence less strongly | They are farther from the learned prior; their seed is already a recurrence artifact; strong correction would overwrite structural computation |
| 4 | Bounds are [φ^(−1), 2φ^(−1)] | Follows from the form α(1+βγ) with α=φ^(−1), β,γ ∈ [0,1] |
| 5 | Organism interpretation | Coherence field = path-integral semantics = ongoing self-assessment of generation quality; the difference between predicting and knowing-how-confident-to-be |
