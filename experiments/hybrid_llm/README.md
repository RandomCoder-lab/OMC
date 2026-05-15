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

### Cumulative read across experiments 0–4

The five experiments converge on a clear, conditional picture:

> **The harmonic substrate is a structural detector, not a primary
> computation.** It loses head-to-head against softmax attention
> (exp 1) and multi-channel sinusoidal PE (exp 3). It loses to a
> trivial L2-NN OOD gate when raw magnitude separates the
> distributions (exp 4A). **It edges past L2-NN when magnitude is
> matched and only structural distribution differs (exp 4B).**

Experiment 4 is the first comparison so far where the harmonic
approach has won under any setup. The condition under which it
wins is the exact condition the project README already documents:
*structural* differences in *multi-dim* feature distributions.
That replicates the credential-stuffing benchmark's regime in a
new domain (synthetic OOD gating) and confirms it wasn't a one-off.

What this means concretely:

1. **The right hybrid architecture is auxiliary, not
   substitution.** Don't replace softmax with OmniWeight (exp 1
   said no). Don't replace sinusoidal PE with phi-fold (exp 3 said
   no). Add a harmonic structural-anomaly head as an *extra*
   signal alongside a standard transformer.

2. **Real LM activations are layer-normalised.** That strips the
   magnitude advantage L2 had in experiment 4A, putting us closer
   to experiment 4B's regime — where harmonic wins. So the
   theoretical case for harmonic OOD gating on transformer
   activations (rather than raw features) is stronger than the
   experiment 4A headline suggests.

3. **The right gate is probably a combination.** Experiment 4A says
   L2 owns the magnitude axis; 4B says harmonic owns the
   structural axis. A combined gate (e.g., logistic regression
   over `[l2_score, harmonic_score]` or simple multiplicative
   combination) should beat either alone, and that's the next
   experiment.

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
- **5** Combined gate: feed both `harmonic_score` and `l2_score` into a
  simple linear combiner (no learning — just `α·harmonic + (1−α)·l2`,
  sweep α) and re-run experiments 4A and 4B. Expectation: combined
  gate matches L2 on 4A, beats both on 4B.
- **6** Realistic regime: pre-normalise reference and test vectors to
  unit L2 norm (simulating layer-norm). Re-run 4A under that
  normalisation. Hypothesis: harmonic gate's loss in 4A was driven
  entirely by magnitude separation, which vanishes under L2-norm.
  If harmonic's AUROC stays flat or rises past L2's after
  normalisation, that's strong evidence for the structural-detector
  thesis on transformer activations.
- **7** Once torch is available: replicate experiment 6 on actual
  transformer activations from a tiny pretrained model. In-dist =
  activations from typical input; OOD = activations from adversarial
  or shuffled input.
