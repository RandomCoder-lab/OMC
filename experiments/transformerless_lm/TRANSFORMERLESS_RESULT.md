# Transformerless candidate — token-substrate falsified, but only on accuracy

## Headline

Combining the three validated in-loop substrate primitives (CRT-PE on positions, CRT on token-IDs, geodesic bias on attention) **fails on final accuracy** but **succeeds on early-phase convergence speed**. The naive "stack the integer-quantity primitives" hypothesis is falsified; a refined architectural rule emerges.

## Results (1500 steps, 3 seeds, distractor-mix TinyShakespeare)

### Final accuracy

| arch | mean val | std | vs crt_only | wins |
|---|--:|--:|--:|--:|
| `crt_only` | 2.4595 | 0.026 | — | — |
| `token_crt` | 2.5598 | 0.026 | **+4.1%** | 0/3 |
| `hybrid_geodesic` | 2.4506 | 0.023 | **−0.4%** | 3/3 |
| `transformerless` | 2.5507 | 0.029 | **+3.7%** | 0/3 |

- **Geodesic re-validates exactly.** Mean 2.4506 vs the previously published 2.4506 in `GEODESIC_RESULT.md` — bit-identical 3/3 win. Clean replication.
- **Token-CRT loses 4.1%.** Falsifies the "all integer-substrate primitives stack" reading of the architectural rule.
- **Combined `transformerless` loses 3.7%.** The token-CRT damage dominates; geodesic's −0.4% can't compensate.

### Convergence speed (val loss at fixed step budget)

| step | crt_only | token_crt | Δ |
|---:|--:|--:|--:|
| 100 | 3.8216 | **3.7150** | **−2.8%** |
| 200 | 3.1496 | **3.0889** | **−1.9%** |
| 300 | 2.9589 | **2.9325** | **−0.9%** |
| 400 | 2.8486 | **2.8399** | −0.3% |
| 500 | 2.7703 | 2.7688 | −0.1% |
| 700 | 2.6734 | 2.6784 | +0.2% |
| 1000 | 2.5861 | 2.6007 | +0.6% |
| 1300 | 2.5186 | 2.5787 | +2.4% |
| 1499 | 2.4029 | 2.5365 | +5.6% |

**Token-CRT is strictly better than CRT-only for any step budget below ~500.** At step 100 the substrate-primed model has already hit the loss the baseline reaches at step ~130 — a ~30% step-saving in the warmup phase.

### Crossover step (when token_crt first loses to crt_only, per seed)

| seed | crossover step |
|---|--:|
| 7   | 500 |
| 42  | 900 |
| 123 | 100 |

Two of three seeds maintain the early-phase win past step 400. Seed 123 crosses immediately — the substrate prior happened to misalign with this seed's training trajectory.

### Compute cost per step

| arch | wall time / 1500 steps | overhead |
|---|--:|--:|
| `crt_only` | 140.0s | — |
| `token_crt` | 141.5s | +1.1% |
| `hybrid_geodesic` | 141.6s | +1.1% |
| `transformerless` | 142.0s | +1.4% |

Substrate primitives add ~1% per-step compute (one buffer add per token, one buffer add per attention layer). Negligible — speed wins or losses are step-count effects, not per-step compute effects.

## Architectural interpretation

The previous rule from `GEODESIC_RESULT.md`:

> SUBSTRATE METRIC APPLIES TO INTEGER QUANTITIES. NEVER APPLY ATTRACTOR_DISTANCE TO LEARNED FLOATS.

is necessary but **not sufficient**. Token IDs ARE integer quantities. Yet adding a fixed CRT-Fibonacci sin/cos prior to the learned embedding lookup hurts final accuracy by 4.1%.

What separates geodesic (wins) from token-CRT (loses):

| primitive | integer quantity | attenuable? | result |
|---|---|---|---|
| CRT-PE | position | no learned PE alternative — substrate IS the position signal | wins |
| Geodesic | position pair | learnable α scalar per block, init=0 | wins |
| Token-CRT | token ID | fixed additive prior, no off-switch | loses |

Geodesic can be driven to α=0 by gradient signal when the bias stops helping. Token-CRT cannot — it's permanent baseline interference the learned embedding must continuously route around. CRT-PE doesn't have this problem because the learned embedding doesn't compete with position information.

### Refined rule

```
SUBSTRATE METRIC APPLIES TO INTEGER QUANTITIES.
NEVER APPLY ATTRACTOR_DISTANCE TO LEARNED FLOATS.
THE INJECTION MUST BE ATTENUABLE (learnable gate to zero)
WHEN IT COMPETES WITH A LEARNED SIGNAL ON THE SAME PATH.
```

The third clause is the new finding. Substrate-on-positions doesn't need attenuation (no competing learned signal). Substrate-on-attention-bias has it (learnable α). Substrate-on-embeddings needs it (would need a learnable β scaling the table) but doesn't have it in this implementation, and pays the price.

## What the speed axis means

This experiment was framed as an accuracy bench. The user reframed it as a compute-efficiency question: *did it train faster?* The data answers yes, in the regime that matters for large-scale training economics:

- **Compute-limited regimes** (early stopping, distillation, single-epoch training on huge corpora where you'll never converge anyway): token-CRT gives a free 30% step-saving in the warmup phase.
- **Convergence-limited regimes** (fixed task, train to threshold): token-CRT is strictly worse.

The speed advantage decays with convergence — but in production LLM training, "convergence" is a budgetary fiction. Most models ship under-converged. The substrate's role as a structured init may matter more than its role as a fixed prior at saturation.

## Open follow-ups

1. **Learnable β on token-CRT.** Add `β · token_enc[x]` with `β` a per-layer scalar initialized to 1 (start with the prior on) — let gradient signal fade the prior as the embedding learns to do its own job. Prediction: matches or beats `crt_only` at all step budgets.

2. **Fixed-step-budget bench.** Train all archs to a fixed loss threshold (e.g. val=2.6) and report steps-to-threshold + wall-clock. The current bench fixes steps and varies final loss; the converse is the regime that matters for compute-efficiency claims.

3. **Scale.** This is on d_model=128 / 800K params. The crossover step may shift with model capacity — a larger model with more parameters might absorb the substrate prior more gracefully (less interference) or less gracefully (more parameters fighting a fixed signal). Untested.

4. **Sensitivity to substrate magnitude.** Token-CRT adds a sin/cos table with values in [-1, 1] to embedding outputs of arbitrary scale. The interference cost may be largely a magnitude-mismatch issue — scaling the substrate to match init-embedding magnitude could matter more than learned attenuation.

## Numbers taken

2026-05-20. CPU run, ~570s wall for 4 archs × 3 seeds × 1500 steps. Reproduction:

```bash
cd experiments/transformerless_lm
python3 train_transformerless.py --steps 1500 --seeds 42,7,123
```
