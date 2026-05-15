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

### Cumulative read across experiments 0–3

The four experiments converge on a single picture:

> **At every fair comparison so far, the standard transformer
> building blocks (softmax attention scoring, sinusoidal positional
> encoding) beat the harmonic alternatives on the specific tasks
> tested.** The harmonic substrate's only apparent wins have been
> traceable to metric artefacts that vanish under proper vector
> similarity.

This doesn't mean OMC's harmonic primitives are useless — the
project's documented wins (the credential-stuffing detector beating
IsolationForest at multi-dim structural anomalies, README's
benchmark table) are real and reproducible. But it does mean:

1. **Drop-replacing transformer components with harmonic ones is the
   wrong play.** Per-head softmax → OmniWeight loses (exp 1).
   Sinusoidal PE → multi-channel harmonic PE loses (exp 3). The
   harmonic substrate is not a better-on-everything substitute.

2. **The harmonic substrate's home is structural-anomaly detection,
   not next-token prediction.** That's where it's already shown
   measurable wins in the existing codebase, and it's a different
   computational task from "rank candidates by relative similarity".

3. **The right hybrid architecture is auxiliary, not substitution.**
   Use a standard transformer for the main next-token loss, and add
   a harmonic structural-anomaly head as an auxiliary signal — for
   detecting OOD inputs, attention-pattern anomalies, or
   activations that have drifted off-attractor. That's the pivot
   from the roadmap below.

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

- **0** Copy task: OmniWeight vs softmax scoring (no learning). ✓ done
- **1** Perturbed-query divergence study. ✓ done
- **2** Single-channel positional-encoding distinctness + lookup. ✓ done
- **3** Multi-channel PE with L2 lookup — fair comparison. ✓ done
- **4** *Pivot.* Stop trying to substitute transformer components.
  Build a harmonic structural-anomaly head as an auxiliary signal:
  given a sequence of intermediate activations from a tiny
  transformer, flag tokens whose `harmonic_index` score against a
  reference distribution is anomalous. Re-use the credential-stuffing
  detector machinery (`harmonic_anomaly.omc`) over activation vectors
  instead of request features. Pure-OMC first, torch port second.
- **5** With torch available: train a 2-layer transformer on a tiny
  char-level corpus. Add the experiment-4 anomaly head as an
  auxiliary loss term and measure whether it improves loss curves,
  OOD detection, or attention sharpness.
