# v0.8.10 — substrate-aware backward gradients: TRIED, falsified at this scale

## Headline

Built and tested `tape_substrate_grad_mod(x, scale, alpha)` — a fused
tape op with identity forward but **substrate-shaped backward**. The
gradient is amplified when it pulls θ toward the nearest Fibonacci
attractor, dampened when it pushes θ away. The substrate as a gradient-
flow preconditioner instead of (or in addition to) a forward modulator.

**Result**: training is **+8.4% worse** at d_model=32 with substrate
backward applied to Q and V. The loss landscape pulls harder than
substrate alignment can resist. **Hypothesis falsified at this scale.**

Three reformulations are scoped for future chapters (none rushed today).

## Construction

The op is mathematically:

```
forward:   y = x                                    # identity
backward:
  for each cell:
    xs = round(x · scale)
    (attractor, dist) = nearest_attractor_with_dist(xs)
    if dist == 0:    dx = dy                        # on attractor, passthrough
    else:
      dir = sign(attractor - xs)
      pulls_toward = sign(g) · dir < 0              # update -lr·g moves toward attractor
      dx = dy · (1 + alpha) if pulls_toward         # amplify
           else dy · 1/(1 + alpha)                  # dampen
```

The sign math: parameter update is `θ ← θ − lr · grad`. If attractor is
above x (`dir > 0`), the update must be NEGATIVE → grad must be POSITIVE.
Amplifying grad in that case = good. If grad is negative when attractor
is above, the update pushes x further from attractor → dampen.

**Smoke test verifies math** (scale=10, alpha=0.5):

| x | xs | nearest_attractor | dist | dir | grad | result | expected |
|---|---|---|---|---|--:|--:|--:|
| 0.6 | 6 | 5 | 1 | -1 | +1 | **1.5** | 1.5 (amplify) ✓ |
| 0.7 | 7 | 8 | 1 | +1 | +1 | **0.667** | 0.667 (dampen) ✓ |
| 0.5 | 5 | 5 | 0 | — | +1 | **1.0** | 1.0 (passthrough) ✓ |

Math correct end-to-end.

## A/B at d_model=32, 250 steps, 3 seeds

Wrapped Q and V projection params in `tape_substrate_grad_mod(node, 64, 0.5)`
before the matmul (forward unchanged; backward biased).

| arm | mean tail loss | Δ vs baseline | wins |
|---|--:|--:|--:|
| baseline | 1.998 | — | — |
| + substrate gm | 2.165 | **+8.4%** | 1/3 |
| + substrate gm + Q6 | 2.157 | **+7.9%** | 1/3 |

**Falsified.** Substrate-shaped gradient bias hurts training at this
scale. The hypothesis was that pulling Q/V toward attractor positions
during training would regularize like substrate-init was supposed to,
without the rigidity of init-time snapping. The result says: the loss
landscape gradient is informative and biasing it toward substrate-
aligned positions costs more than it gains.

This mirrors the v0.8.8 substrate-init falsification — both "constrain
toward substrate" hypotheses fail. The substrate is good at:
- **Forward modulation** (Q6, S-MOD, V-resample) — explicit substrate
  shaping of activations
- **Architectural priors** (CRT-PE, fibonacci attractor table) —
  substrate in the data and structure
- **Post-training pattern** (v0.8.8 finding) — substrate emerges in
  attention after Q6 training

The substrate is NOT good at:
- **Init-time constraint** (v0.8.8 #3 falsified)
- **Gradient-time bias** (v0.8.10 falsified)

Pattern: **the substrate works when applied to outputs (forward modulation)
or revealed by training (post-train alignment), but NOT when forced on
inputs or gradients.** The information flow direction matters.

## What's NOT ruled out (future chapter reformulations)

1. **Different scale**: scale=64 may be too coarse. scale=1024 or scale
   per-layer (computed from param magnitude statistics) may give
   gentler bias that the loss can integrate.

2. **Apply to FF instead of attention**: attention Q/V are loss-critical;
   FF down-projection weights may be more tolerant of substrate bias.

3. **Decay alpha during training**: start with strong substrate bias
   (alpha=0.5), decay linearly to 0 over training. Substrate as a
   warm-start regularizer.

4. **Substrate as REGULARIZATION TERM, not gradient bias**: add
   `sum(attractor_distance(param)) · lambda` to the loss. Gradient
   then has substrate component naturally; doesn't override the loss.

Each is its own chapter. v0.8.10 ships the negative honestly.

## Where it lands in the substrate-IS-architecture map

The substrate has been validated at 5 layers across v0.8:
1. **Data** — CRT-PE positional encoding (cross-validates)
2. **Algorithm** — substrate-K + S-MOD + V-resample (cross-validates)
3. **Hardware tile** — 8×32 wavefront-aligned (cross-validates +38-61%)
4. **Post-training attention pattern** — Q6 → 8.3× concentration
   (v0.8.8 finding)
5. **Multi-head Q6 compound** — −3.57% vs baseline (v0.8.9 confirms)

Now-falsified attempts:
- **Init-time substrate-snap** — substrate-init regularization
  (v0.8.8 #3)
- **Gradient-time substrate-pull** — substrate backward modulation
  (v0.8.10 this chapter)

The empirical map is: substrate at OUTPUTS or in STRUCTURE works.
Substrate as INPUT constraint or BACKWARD bias does not (at current
scales, with current scale parameter, on current architectures).

## Files

- `omnimcode-core/src/interpreter.rs` — `TapeOp::SubstrateGradMod`
  variant + `tape_substrate_grad_mod` dispatch + substrate-aware
  backward
- `examples/prometheus_substrate_grad_mod_xval.omc` — 3-arm A/B
- `experiments/prometheus_parity/V0810_SUBSTRATE_AWARE_BACKWARD.md`

## Tests

**1111/1111 OMC tests pass.**
