# Substrate-Q wins -12.15% via phi_pi_fib log-distance modulation (6/6 seeds)

## Headline

The first substrate-Q recipe (Q1 post-projection resample) lost on 3 seeds (+5.31% val). The user's note "Possible outcomes may relate to different integral pieces to phi_pi_fib" pointed to trying other operations. The broader sweep over five Q recipes found **one decisive winner**: Q6, the phi_pi_fib log-distance scaling.

```
3-seed broader sweep:
  Q0 (baseline)              3.0059
  Q3 (pre-projection snap)   3.1670  (+5.36% loses)
  Q4 (boost-not-dampen)      3.3346  (+10.94% loses)
  Q5 (signed-snap)           2.9833  (-0.75% ties)
  Q6 (log-distance scale)    2.6959  (-10.31% wins, std 0.42)

6-seed Q6 confirmation:
  Q0  3.1277 ± 0.20
  Q6  2.7477 ± 0.29  (-12.15%, 6/6 seeds beat baseline)
```

Q6 beats Q0 on every one of the 6 confirmation seeds:

| seed | Q0 | Q6 | Q6 wins? |
|---|--:|--:|:-:|
| 42 | 2.964 | 2.770 | ✓ |
| 7 | 3.223 | 3.075 | ✓ |
| 123 | 2.830 | 2.243 | ✓ |
| 2026 | 3.370 | 2.660 | ✓ |
| 99 | 3.202 | 2.959 | ✓ |
| 1 | 3.176 | 2.779 | ✓ |

The win is decisive.

## The recipe

```python
def phi_pi_log_distance(x, scale=10.0):
    """Approximate log_phi_pi_fibonacci(|x|)."""
    abs_x = (x * scale).abs() + 1.0
    return abs_x.log() / (math.pi * math.log(PHI))

q_proj = x @ self.W_q                 # standard learned projection
log_d = phi_pi_log_distance(q_proj)
modulation = (-gamma * log_d).exp()    # gamma=0.5 default
q_full = q_proj * modulation
```

Effectively scales each Q component by `(|q_proj| + 1)^(-γ/(π·ln φ))` — large magnitudes get dampened along the substrate's log-distance metric, not the linear attractor-distance metric V1 used.

## Why log-distance and not attractor-distance

The substrate-V finding worked via `substrate_resample` — snap each component toward its nearest Fibonacci attractor by multiplying with `1/(1 + d)` where `d = attractor_distance(x·scale)`. Q1 used the same operation and lost.

The HONEST principle that emerges from Q1 vs Q6: **Q's role is to STEER the attention pattern, not to be aggregated.** Snap-to-attractor (Q1) reduces the diversity of queries — every query gets pulled toward the same discrete set of attractor values, so heads can't discriminate positions. The attention pattern collapses.

**Log-distance modulation (Q6) is different**: it's a smooth magnitude regularizer keyed on substrate structure, not an attractor snap. It dampens LARGE-magnitude queries more than small ones (because log grows slowly), preserving the relative ordering and steering capability of the head while keeping query magnitudes in a substrate-friendly range. The head still discriminates; the magnitudes just get a soft cap.

This adds nuance to the v0.1 principle:
- **Substrate snap-to-attractor**: helps for quantities being AGGREGATED (V, K)
- **Substrate log-distance scaling**: helps for quantities that STEER (Q)

Both are "substrate modulation" — they just use different phi_pi_fib operations to match the role of the quantity being modulated.

## Cumulative substrate-attention stack

With Q6 added to the v0.1 production stack:

| Stack | mean val |
|---|--:|
| L0 (vanilla softmax + learned V + learned Q) | 3.301 |
| L1-MH + S-MOD α=1.0 (v0.0.6 + S-MOD) | 3.084 |
| + V1 substrate-resample (v0.1) | 3.006 |
| **+ Q6 phi_pi_log-distance (v0.8)** | **2.748** |
| | **−16.7% cumulative vs L0** |

Up from v0.1's -8.94% to **-16.7%**. Four substrate-attention components now stack: K (CRT-Fibonacci substrate, no learnable W_K), softmax (S-MOD α=1.0), V (substrate_resample), Q (phi_pi_log-distance modulation).

## Tests

- 5-variant 3-seed exploratory sweep (`torch_substrate_q_broader.py`): Q3/Q4 lose, Q5 ties, **Q6 wins**.
- 6-seed Q6 confirmation: 6/6 seeds beat baseline, mean -12.15%.

## What's NOT yet wired into production OMC

The Q6 win is established in PyTorch parity. Wiring it into OMC's `prom_attention_substrate_k_forward` requires `tape_abs` and `tape_log` ops (which the OMC tape autograd may or may not have today). That's the v0.8.1 follow-up: extend the tape, port Q6 into pure-OMC Prometheus, re-verify the win in OMC space the same way substrate-V was cross-validated.

## What's still open

- **Larger scale**: the win is at TinyShakespeare (1.1MB). Whether it holds at 10-100MB is the question that determines whether substrate-attention is a real physical inductive bias or a small-scale curiosity.
- **γ tuning**: γ=0.5 was the first guess from the sweep. A γ sweep might find a stronger setting.
- **OMC-side cross-validation**: the substrate-V finding was reproduced in both PyTorch and pure-OMC Prometheus. Same parity check is needed for Q6.

## Files

- `torch_substrate_q_broader.py` — the 5-variant Q sweep
- `results_torch_substrate_q_broader.json` — 3-seed exploratory data
- `results_torch_substrate_q6_confirm.json` — 6-seed Q6 confirmation data

## Reproduction

```bash
cd experiments/prometheus_parity
# 3-seed exploratory sweep across 5 Q variants:
python3 torch_substrate_q_broader.py
# 6-seed Q6 confirmation:
python3 torch_substrate_q_broader.py --seeds 42,7,123,2026,99,1 --variants Q0,Q6 \
    --out results_torch_substrate_q6_confirm.json
```
