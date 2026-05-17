# Substrate-V wins −2.52% on top of L1-MH + S-MOD α=1.0

## Headline

Applying `substrate_resample(x) = x * (1 / (1 + attractor_distance(10·x)/10))`
to V *after* the learned projection — keeping the L1 substrate-K and
S-MOD α=1.0 softmax — wins on TinyShakespeare.

- **V1 val: 3.006** (vs V0 baseline 3.084, **−2.52%**)
- **wins 3/3 seeds** ([42, 7, 123])
- **no parameter cost** — V projection still learned; substrate is a
  pure post-projection modulation
- **third substrate-component win** stacked on the attention block

Cumulative vs the vanilla baseline (L0 + vanilla softmax + learned V):
**L0 3.301 → L1-MH+S-MOD α=1.0 + V1 3.006 = −8.94%**.

## The three V variants tested

All on L1 multi-head (Q learned, K = CRT-Fibonacci frozen) +
S-MOD softmax α=1.0 (today's production default).

| Variant | V formula | mean val | std | vs V0 |
|---|---|--:|--:|--:|
| V0 (baseline) | `v = x @ W_v` | 3.0837 | 0.218 | — |
| **V1 (resample)** | **`v = substrate_resample(x @ W_v)`** | **3.0059** | **0.200** | **−2.52%** |
| V2 (gate) | `v = (x @ W_v) * (1 + γ·near_attractor(x))` | 3.3599 | 0.034 | +8.96% |

Where:
```python
def substrate_resample(x, scale=10.0):
    scaled = x * scale
    d = attractor_distance(scaled)
    modulation = 1.0 / (1.0 + d / scale)
    return x * modulation

def near_attractor_signal(x, scale=10.0):
    return 1.0 / (1.0 + attractor_distance(x * scale))
```

## Why V1 (resample) wins

The mechanism: `substrate_resample(x @ W_v)` dampens components of the
projected V whose magnitudes land far from any Fibonacci attractor.
Components already on-attractor pass through unchanged; off-attractor
components are scaled down toward attractor alignment.

Combined with **S-MOD softmax** suppressing off-attractor attention
weights, the substrate now constrains **both axes** of the attention
output:
- attention pattern → off-attractor positions weighted less (S-MOD)
- value content → off-attractor magnitudes weighted less (substrate-V)

The two modulations compose multiplicatively in `attn @ v`, so the
final output is biased toward attractor-aligned contributions on both
the position axis and the magnitude axis. The model learns to route
information through substrate-aligned channels.

## Why V2 (gate) loses

V2 multiplies V by a gate derived from the **input** `x`, not the
projected `v`. The gate `1 + γ·near_attractor(x)` peaks where `x` itself
is near an attractor, which has no necessary alignment with where the
PROJECTED V components land. The gate adds noise without aligning to
the substrate signal in V — and it kills variance (std=0.034 vs ~0.2
for the other variants), suggesting it collapses the V space.

Predicted failure mode: substrate metric applied to a quantity whose
relevant integer-coherent structure lives *somewhere else*. V's
substrate alignment is on `x @ W_v`, not on `x`.

## Why L4 (yesterday) lost but V1 (today) wins

Yesterday's L4 replaced V entirely with `substrate_resample(x)` — no
learned projection, no attention modulation. It lost because:
1. No learned projection meant V couldn't capture task-specific
   linear combinations of x.
2. Vanilla softmax over substrate-K scores had no off-attractor
   dampening, so off-attractor attention rows multiplied through
   raw substrate V values without alignment between the two.

V1 fixes both: keeps the learned W_v (captures domain projection),
applies the substrate as a modulation (not a replacement), and pairs
with S-MOD softmax (aligned modulation on both axes).

**The substrate rule restated:** substrate metric applied to a
quantity that has integer-coherent structure helps; applied without
preserving the learned domain structure, it doesn't.

## Updated substrate scoreboard

| Component | Substrate variant | Status |
|---|---|---|
| Positional encoding | CRT-Fibonacci PE | WINS −5.4% (TinyShakespeare) |
| OOD detection | HBit cross-cutting tension | WINS AUROC 1.0 |
| Attention K matrix | CRT-PE addressing | WINS −6.3% val (multi-head, TinyShakespeare) |
| Attention softmax | S-MOD α=1.0 | WINS −6.57% val (3-seed sweep) |
| **Attention V projection** | **post-projection substrate_resample** | **WINS −2.52% val (3/3 seeds)** |
| Geodesic attention bias | additive position bias | WINS 3/3 (single-block) |
| Optimizer | Harmonic SGD | WINS −13.2% (tiny-scale tinyLM) |

**Seven substrate-component wins across the transformer.** Three of
them (K + softmax + V) stack at TinyShakespeare scale for a combined
**−8.94%** val vs the vanilla L0 baseline.

## What's NOT in this run

- Single corpus (TinyShakespeare). Generalization unmeasured.
- 3 seeds — minimum for "majority vote"; more would tighten variance.
- scale=10.0 is a guess; not swept. A scale sweep on
  `substrate_resample` might find a stronger modulation strength.
- V1 only tested at α=1.0 S-MOD. May behave differently at smaller α.
- Substrate-resample applied to V only, not Q or output projection.

## Production recommendation update

The substrate-aware attention block in Prometheus should now use:
- **K = CRT-Fibonacci** (substrate-K, validated)
- **Q = learned per-head** (validated)
- **V = `substrate_resample(x @ W_v)`** (new, validated)
- **Normalization = S-MOD softmax α=1.0** (validated)
- Output projection learned

Three component swaps, ~9% cumulative val improvement on real corpus,
~10% parameter reduction from K removal alone.

## Code

```python
class AttentionL1V(nn.Module):
    """L1 multi-head + S-MOD softmax + post-projection substrate-V."""
    def forward(self, x):
        q = (x @ self.W_q).view(T, H, dh).transpose(0, 1)
        v_full = substrate_resample(x @ self.W_v)          # ← the change
        v = v_full.view(T, H, dh).transpose(0, 1)
        scores = (q @ self.K_const_mh.transpose(-2, -1)) / (dh ** 0.5)
        attn = softmax_smod(scores, alpha=1.0)
        out = attn @ v
        return out.transpose(0, 1).reshape(T, D) @ self.W_o
```

One additional line vs the L1-MH+S-MOD baseline. See
`experiments/prometheus_parity/torch_substrate_v.py` for the full
A/B harness and `results_torch_substrate_v.json` for raw 3-seed data.
