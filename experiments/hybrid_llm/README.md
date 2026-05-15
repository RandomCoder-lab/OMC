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
| 1 | Perturbed query (query = true_val + noise), 200 trials per noise level | Softmax wins everywhere. noise=1: 189 vs 170. noise=7: 118 vs 99. noise=50: 42 vs 33. OmniWeight's |k|-normalised denominator pulls predictions toward smaller attractors regardless of perturbation direction, which hurts on a "recover the original value" metric. |

The headline lesson from experiment 1: **OmniWeight is scale-aware,
softmax is scale-agnostic, and which one wins depends entirely on
whether your target is "recover the true value" or "rank by relative
error".** For a real LM token-prediction task we'd expect softmax to win
straight-up here — the harmonic primitive's wins live elsewhere
(multi-dim structural anomalies, exactly as the project README
already documents). That gives us a clearer picture of where to plug
OmniWeight into a larger architecture: not at the per-head attention
scorer, but at a structural-anomaly gate or a regularisation term.

## Roadmap on this branch

- **0** Copy task: OmniWeight vs softmax scoring (no learning). ✓ done
- **1** Perturbed-query divergence study. ✓ done
- **2** `phi.fold`-based positional encoding vs sinusoidal PE on a sequence-repeat task. Test length extrapolation (train at len=N, eval at 2N) — sinusoidal is known to extrapolate poorly; does φ-fold do better because the attractor set is discrete?
- **3** Extend `examples/lib/torch.omc` with embedding, softmax, layer-norm, cross-entropy. Port the experiment 2 result to a *learned* tiny-transformer setting (requires torch in the host env).
- **4** Hybrid: standard softmax attention with an OmniWeight-based attention-entropy regulariser. Loss = CE + λ · (1 − mean(OmniWeight of attention peaks)). Test whether nudging attention toward harmonic peaks helps small models.
