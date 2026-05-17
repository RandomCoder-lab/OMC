# Substrate-Q (post-projection resample) LOSES — V's recipe doesn't generalize to Q

## Headline

The v0.1 chapter's substrate-V finding used `substrate_resample(x @ W_v)` (snap-to-attractor post-projection) and won −2.52% val. The natural hypothesis was that the same recipe would generalize to Q, giving a 4th stacked substrate-component. It didn't.

3-seed TinyShakespeare experiment, L1 multi-head + S-MOD α=1.0 + substrate-V (V1) production baseline, varying ONLY the Q recipe:

| Variant | Q formula | mean val | std | vs Q0 |
|---|---|--:|--:|--:|
| Q0 (baseline) | `q = x @ W_q` | 3.0059 | 0.200 | — |
| Q1 (resample) | `q = substrate_resample(x @ W_q)` | 3.1654 | 0.306 | **+5.31%** |
| Q2 (gate) | `q = (x @ W_q) * (1 + γ·near_attractor(x))` | 3.1213 | 0.194 | +3.84% |

Both substrate-Q variants LOSE. Q0 (unmodified learned projection) wins decisively.

## Why this is informative

The v0.1 chapter derived a principle: "substrate modulation works when applied to a quantity that has integer-coherent structure; substrate replacement of learned projections does not." The substrate-V win confirmed it on the value path. The substrate-Q failure SHARPENS it:

The principle wasn't "post-projection modulation works for any attention matrix." It was specific to where the substrate's integer-coherent structure aligns with the quantity's downstream role.

- **V's downstream role**: get aggregated INTO the attention output via `attn @ v`. Substrate-snap dampens off-attractor magnitudes → cleaner aggregated signal.
- **Q's downstream role**: STEER the attention pattern via `q @ k.T`. Substrate-snap dampens query diversity → the attention head's ability to discriminate positions weakens.

In other words: V is on the receiving end of attention (substrate cleans the signal); Q is on the steering end (substrate kills the variance you need to steer with).

## What this means for the substrate-attention stack

The production stack stays at three components:
1. K = CRT-Fibonacci substrate (no learnable W_K)
2. softmax → S-MOD α=1.0
3. V = `substrate_resample(x @ W_v)` post-projection

Q stays learned. The −8.94% cumulative win from v0.1 is the ceiling for the "post-projection modulation" recipe; further substrate gains would need a different mechanism for Q.

## Open question — different phi_pi_fib primitives

The Q1 experiment tested the SAME operation (substrate_resample = post-projection snap-to-attractor) as V1. The user pointed out that other phi_pi_fib primitives might apply differently:

- **Q3 (pre-projection)**: `q = (substrate_resample(x)) @ W_q`. Snap the input, then project. Different from snapping after.
- **Q4 (harmonic_align)**: use the existing `harmonic_align` primitive instead of attractor-distance modulation.
- **Q5 (phi_pi_log_distance)**: scale Q by `1 / log_phi_pi_fibonacci(|q|)` — substrate-aligned magnitudes get boosted, not dampened.
- **Q6 (zeckendorf snap)**: decompose Q components into nearest Zeckendorf representations.

These would test the broader hypothesis that SOME phi_pi_fib operation on Q produces a win, even if `substrate_resample` doesn't. Listed as v0.6.1-substrate-q-broader candidate in the next experiment cycle.

## Tests

3-seed PyTorch sweep at TinyShakespeare scale. Standard config (top_k attention, 4 heads × 4 blocks, seq=32, d_model=32, 1500 steps).

## Files

- `torch_substrate_q.py` — the experiment script (mirrors `torch_substrate_v.py`)
- `results_torch_substrate_q.json` — raw 3-seed result data

## Reproduction

```bash
cd experiments/prometheus_parity
python3 torch_substrate_q.py
```
