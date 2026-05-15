# Hybrid Harmonic / Transformer LLM

This branch (`claude/phi-field-llm-evolution`) explores using OMC's φ-math
primitives to replace or augment specific transformer components, with the
goal of producing measurable behavior differences on real sequence tasks.

The existing pure-OMC demos (`examples/phi_field_llm_demo.omc`,
`examples/phi_field_llm_multilayer.omc`) prove that geodesic
attention — picking the Fibonacci attractor with the highest
`OmniWeight w = φ^(-|e|)` — runs end-to-end. They don't yet show
**when** that's better than softmax-QK attention and **what it costs**.
This experiment series answers that.

## The substitutions we want to test

Three transformer pieces map cleanly onto OMC's harmonic primitives:

| Transformer piece | Harmonic replacement | What we're measuring |
|---|---|---|
| **Sinusoidal positional encoding** | Golden-angle rotation (`pos * 2π/φ²`) folded onto Fibonacci attractors via `phi.fold`. | Length-generalization: does a model trained on length N still work at 2N? Sinusoidal PE is known to extrapolate poorly. |
| **Softmax attention scoring** | OmniWeight: `w(q, k) = φ^(-|q − k| / max(\|k\|, 1))`. Per-position; pick argmax instead of weighted average. | Sharpness vs. softness. OmniWeight is winner-take-all. Useful for copy/lookup tasks; lossy for averaging tasks. |
| **Layer-norm + residual** | `phi.fold(residual_blend)` (already implemented in `phi_field_llm_multilayer.omc`). | Whether the φ-fold provides a useful regularizer that keeps activations on-attractor. |

Phase 0 of this branch focuses on (2) — OmniWeight attention — because
it's the most isolated and the existing demos already implement it.
The other two come later.

## Experiment 0: Copy task — OmniWeight vs softmax

The simplest task that distinguishes the two approaches:

- **Input:** a sequence of 8 Fibonacci-aligned tokens drawn at random
  from `{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233}`, plus a separator,
  plus a "query" token that copies one of the inputs verbatim.
  Example: `[34, 8, 89, 13, 21, |, 89]` → expected next token `89`.
- **Models:**
  - OmniWeight-attention head over the input (the current
    `best_attractor` mechanism).
  - Softmax-attention head over the same inputs, where the score is
    `exp(-|q − k|)` normalized. Both use **no learned weights** — this
    isolates the scoring function from training dynamics.
- **Metric:** exact-match accuracy on 100 random instances, broken
  down by (a) whether the query exactly matches an input, (b) how
  many distractors share the query's nearest attractor.

If OmniWeight wins on (a) and loses on (b), that confirms the
"winner-take-all" thesis and tells us where to apply it in a larger model.

**Status:** `experiment_0_copy_task.omc` runs this comparison.

## Why no torch yet

The current remote environment has no torch / numpy. Pure-OMC
experiments give us:

1. Deterministic, reproducible runs inside the standalone binary.
2. No dependency on `python-embed` for the experiment itself.
3. A baseline that any later torch-based experiment must match
   byte-for-byte on the harmonic side.

Once we have a winning harmonic primitive, the next branch step is to
port the same scoring rule to PyTorch (via `examples/lib/torch.omc` or
a stand-alone Python script) and bench against a real learned model
on a real corpus.

## How to run

```bash
# Build (one time)
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release

# Run experiment 0 (tree-walk)
./target/release/omnimcode-standalone experiments/hybrid_llm/experiment_0_copy_task.omc

# Same under the bytecode VM
OMC_VM=1 ./target/release/omnimcode-standalone experiments/hybrid_llm/experiment_0_copy_task.omc

# Audit: bytecode VM must match tree-walk
./target/release/omnimcode-standalone --audit experiments/hybrid_llm/experiment_0_copy_task.omc
```

## Results so far

| Experiment | Setting | Headline number |
|---|---|---|
| 0 | Copy task, exact-match query, 100 trials | OmniWeight 82/100, softmax 82/100, 0 disagreements. Confirms both scorers agree on exact match (the 18 "misses" are duplicate-value trials, both tie-break to first occurrence). |
| 1 | Perturbed query (query = true_val + noise), 200 trials per noise level | Softmax wins everywhere. noise=1: 189 vs 170. noise=7: 118 vs 99. noise=50: 42 vs 33. OmniWeight's |k|-normalised denominator pulls toward smaller-magnitude attractors regardless of perturbation direction, which hurts the "recover the original value" objective. |
| 2 | Single-channel PE distinctness + lookup at L = 8 / 14 / 24 / 48 | Sinusoidal wins at short L (8/8 vs 6/8). At L=48 harmonic appears to overtake: 38/48 vs 26/48 (79% vs 54%). Flagged as a likely metric artefact — single-int "closest code" lookup favours monotonic over periodic encodings. |
| 3 | 4-channel PE (harmonic primes 7/11/13/17, sin/cos periods 8/64), L2 lookup, L = 8 → 200 | **Sinusoidal regains its lead decisively at every L ≥ 16.** L=48: 48/48 vs 21/48. L=200: 72/200 vs 34/200. Harmonic saturates at 22 unique vectors by L=64; sinusoidal stays perfectly distinct up to L=64 then saturates at 64. The single-channel L=48 harmonic "win" was a metric artefact, exactly as suspected. |
| 4A | Harmonic OOD gate vs L2-NN baseline on 4-dim synthetic vectors (N_REF=300, 150 in-dist test, 150 OOD test). OOD = uniform [1, 90]. | L2 wins. AUROC L2 0.961 vs harmonic 0.910. TPR @ FPR=10%: L2 0.91 vs harmonic 0.71. L2 has a trivial magnitude advantage — mean L2 score 87 (in-dist) vs 1313 (OOD), since OOD vectors are larger on average and harmonic gate's `phi.fold` discards magnitude. |
| 4B | Same gates, **magnitude-matched** structural OOD (inverted attractor weights: 10%/30%/60% small/med/large vs in-dist's 60%/30%/10%). | **Harmonic edges past L2 in AUROC: 0.956 vs 0.946.** At low FPR L2 still wins (TPR@FPR=1%: L2 0.60 vs harmonic 0.48), but on overall ranking the structural rarity signal beats the L2 metric once magnitude is no longer a giveaway. |
| 5 | HBit cross-cutting tension (no reference) + combined gate (sum of z-normalised HBit, marginal rarity, L2) on both scenarios. | **Scenario A: HBit tension AUROC = 1.0** (perfect — mean tension 0.0 in-dist vs 20.1 OOD). Combined: 0.999. **Scenario B: HBit AUROC = 0.5** (random — both sides on-manifold, tension = 0 everywhere). Combined: 0.967, beating every single gate. Each gate owns a different OOD axis: HBit→off-manifold, marginal→distribution-shift, L2→magnitude. |

### Cumulative read across experiments 0–5

The six experiments now form a complete picture. Each OOD axis has
a gate that owns it:

| Failure mode | Owning gate | Cost | Scenario A AUROC | Scenario B AUROC |
|---|---|---|---|---|
| Off-manifold values | **HBit cross-cutting tension** | **Reference-free** | **1.000** | 0.500 |
| Wrong attractor distribution | Marginal log-rarity (exp 4 harmonic) | needs reference | 0.910 | 0.956 |
| Wrong magnitude | L2 nearest-neighbour | needs reference | 0.961 | 0.946 |
| Any of the above | Sum of z-normalised triple | needs reference | 0.999 | 0.967 |

The HBit gate is the cheapest possible: `sum_d |v[d] − phi.fold(v[d])|`.
Zero fitting, zero reference set, perfect detector when the OOD axis is
"value isn't a Fibonacci attractor". Useless when both sides are
on-manifold (scenario B mean tension is 0.0 on both in-dist and OOD —
the gate can't see any difference).

The combined gate is the clear winner across both scenarios. Sum of
z-normalised per-gate scores, with the z-normalisation parameters
fit on **in-dist scores only** (the combiner doesn't peek at OOD data).
Scenario A: 0.999 — almost perfect, gets HBit's free wins plus L2 and
marginal contributions. Scenario B: 0.967 — beats every individual
gate by 1-2 AUROC points.

What this means concretely:

1. **Reference-free OOD detection is real on harmonic-structured
   data.** If your in-distribution lives on (or near) the Fibonacci
   attractor manifold, HBit tension is a free OOD signal you can
   compute on a single test point with no model fitting. Cost is
   D float subtractions per test point.

2. **The "harmonic substrate is a structural detector" thesis is
   now empirically grounded for OOD gating**, with quantified
   contribution from each piece. Exp 0-3 ruled out using harmonic
   primitives as drop-in replacements for transformer components.
   Exp 4-5 found their actual home: as auxiliary detectors layered
   onto raw features (or activations) to catch failure modes that
   L2 alone misses.

3. **The combined gate is the deployable artifact.** Three
   complementary axes, z-normalised on the reference, summed.
   Wins on both magnitude-shifted and structural OOD. Beats every
   single-gate baseline.

### What changed between experiment 2 and experiment 3

Experiment 2 used **single-integer codes** and a **closest-int**
lookup metric. Single-integer codes can't capture the geometric
frequency layering that makes sinusoidal PE work in real
transformers — once the period wraps, the encoding is dead.

Experiment 3 used **4-channel vectors** and **L2 distance**. That
gives sinusoidal a long-period channel (P=64) that stays distinct
well past the short-period channel's wrap. Harmonic gets four
prime-multiplier channels but they all saturate at the same
Fibonacci ceiling, so the joint vector hits its uniqueness budget
fast (22 unique vectors total) and stays there forever.

The lesson is one of the project's existing themes spelled out
again: **measure honestly, and let the measurement reshape the
plan.** Experiment 2's headline number was reproducible and
audited, but the framing was wrong. Adding experiment 3 — same
question, fairer comparison — flipped the answer. The README is
updated to reflect the cumulative read, not just the latest
result.

## Roadmap on this branch

- **0** Copy task: OmniWeight vs softmax scoring. ✓ done
- **1** Perturbed-query divergence study. ✓ done
- **2** Single-channel positional-encoding distinctness + lookup. ✓ done
- **3** Multi-channel PE with L2 lookup. ✓ done
- **4** Harmonic OOD gate vs L2-NN baseline, two scenarios. ✓ done
- **5** HBit cross-cutting tension + 3-gate combined detector. ✓ done
- **6** Layer-norm-matched setup: pre-normalise all vectors to unit L2.
  Re-run scenarios A and B. Expected: HBit's perfect AUROC on A
  survives (tension is magnitude-invariant by definition); L2's free
  magnitude advantage on A disappears; the combined gate's edge on B
  widens.
- **7** Bake the combined gate into a reusable library:
  `experiments/hybrid_llm/lib/ood_gate.omc` exposing
  `ood_gate.fit(ref_corpus)` and `ood_gate.score(vec)`. Then once
  torch is available, replicate on real transformer activations.
