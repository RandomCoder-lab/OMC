# Gate reformulation — both follow-on architectures falsified

## Context

`distractor_mix_README.md` reported the original `hybrid` arch
(CRT-PE + KEY-magnitude HBit gate) losing 0/3 to `crt_only` at
distractor_frac=0.20. The writeup proposed two concrete
reformulations that kept CRT-PE and changed only the gate:

1. **`hybrid_score`** — gate on raw attention SCORES additively in
   log-space pre-softmax, instead of post-softmax renormalization
   on key magnitudes.
2. **`hybrid_learned`** — replace fixed `1/(1+d)` with
   `sigmoid(W*d + b)` where W, b are learned per-head. Lets the
   model discover its own threshold and slope.

Both kept CRT-PE intact. The hypothesis was that the original gate's
formulation was too rigid, and a softer or learnable variant might
earn its keep.

## Setup

Identical to `train_distractor_mix.py`:
- TinyShakespeare, 90/10 split
- 20% of training chunks char-shuffled (within-vocab distractors)
- Validation on PURE shakespeare (the actual task we care about)
- d_model=128, n_blocks=4, seq_len=128, ~801K params (+8 for learned gate)
- 1500 steps, batch=32, AdamW lr=3e-4
- 3 seeds: 42, 7, 123
- CPU, ~30 min total wall-clock for 3 archs × 3 seeds

## Results

| arch | mean | std | wins vs crt_only | rel |
|---|--:|--:|:-:|--:|
| `crt_only` | **2.4595** | 0.0257 | — | — |
| `hybrid_score` | 2.5488 | 0.0239 | **0/3** | **+3.6%** |
| `hybrid_learned` | 2.5607 | 0.0179 | **0/3** | **+4.1%** |

### Per-seed

| seed | crt_only | hybrid_score | hybrid_learned |
|---|--:|--:|--:|
| 42 | 2.489 | 2.562 | 2.567 |
| 7  | 2.443 | 2.521 | 2.540 |
| 123| 2.446 | 2.564 | 2.574 |

### Combined with the original

| arch | mean | wins vs crt_only |
|---|--:|:-:|
| `crt_only` | 2.4595 | — |
| `hybrid` (key-gate, original)     | 2.5379 | 0/3 |
| `hybrid_score` (score-gate)       | 2.5488 | 0/3 |
| `hybrid_learned` (learned-thresh) | 2.5607 | 0/3 |

**Three different gate formulations, three falsifications. Same
+3-4% loss magnitude across all three.**

## Interpretation

The architectural read consolidates: **HBit tension is not a useful
attention modulator at this scale and data regime**, regardless of
where in the attention path the gate fires or whether its threshold
is learnable.

Why this is a stronger negative than the original single failure:
- We tested two DIFFERENT failure modes the README proposed
- The learned-threshold variant had every chance to recover —
  the model could simply learn `gate_w ≈ 0, gate_b ≈ large` to
  disable the gate entirely. It did not converge there; it
  converged to a gate setting that costs ~4% on val loss.
- The score-level variant operates at the correct layer of the
  computation (logits, not key-magnitudes), removing the
  "wrong-layer-of-abstraction" objection.

This suggests the failure isn't a formulation bug — the underlying
substrate-distance signal on `q@k^T / sqrt(d)` values just doesn't
correlate with what the model needs to focus on. The Fibonacci
attractor structure of OMC's `HInt` doesn't transfer to attention
score tensors which have totally different distributional
properties (Gaussian-ish, scaled by `1/sqrt(d_head)`, drawn from
learned projections of token embeddings).

## What this means for the transformerless LM

The substrate's role in a transformer replacement is now empirically:

| Component | Substrate variant | Status |
|---|---|:-:|
| Positional encoding | CRT-Fibonacci PE | **Wins** (−5.4% clean, −2.9% distractor mix; 3/3 + 4/5 seeds) |
| OOD detection | HBit cross-cutting tension | **Wins** (AUROC 1.0 on scenario A) |
| Attention modulation (key-mag gate) | `1/(1+d)` on `\|k\|.mean` | **Falsified** (0/3) |
| Attention modulation (score-level gate) | `1/(1+d)` on logits pre-softmax | **Falsified** (0/3, this writeup) |
| Attention modulation (learned threshold) | `sigmoid(W*d+b)` on `\|k\|.mean` | **Falsified** (0/3, this writeup) |

**The substrate's home in this architecture is positional and
distributional, not as an attention-score shaper.** Three independent
attempts to make it work there have all failed by similar margins.

## What's left to try (the new menu)

Since attention-gate variants are exhausted at this scale, the
remaining places to introduce substrate signal are all
out-of-attention:

### A. FFN substrate gate (vs attention gate)
The FFN block doesn't have softmax — substrate signal there is
unmediated. Apply `attractor_distance` to the post-GELU
activations or to one of the linear projections. The FFN
operates on per-position vectors with no cross-position coupling,
so the substrate distance is computed in the same per-position
basis OMC's HInt was designed for.

### B. Auxiliary substrate loss (regularizer, not forward signal)
Add `lambda * attractor_distance(activations).mean()` as an
auxiliary loss term. Gradients pull the network toward
substrate-aligned representations without affecting the forward
pass. Closest analog: weight decay, but in attractor-distance
space instead of L2.

### C. Substrate-curriculum sampling (training order, not architecture)
Sort training batches by attractor-distance of their token IDs:
on-attractor samples first, off-attractor later. The substrate
becomes a curriculum signal, not an architecture change. Cheap
to test (no model change needed).

### D. Per-head selective gating
Some heads get the substrate gate, some don't. Train one of the
existing falsified variants but applied to only 1-of-4 heads.
This is the weakest "maybe" — if the gate fails on all heads it
will likely fail on one too — but worth ruling out cleanly.

### E. Honest pivot
Accept that HBit-as-attention-gate is dead at the scales we can
test. Ship the transformerless prototype with CRT-PE + standard
softmax attention + substrate-aware tokenization (which is the
biggest unexplored axis — the substrate at the EMBEDDING layer,
not the attention layer). This is the path that respects what
we've actually measured.

## Recommendation

E first, then A in parallel. Stop investing in attention-gate
formulations — three failures with consistent magnitude is a
saturation signal. The pivot toward substrate-aware tokenization
hasn't been measured yet and has a stronger architectural basis
(OMC's tokenizer is already substrate-routed; using it as the LM's
input tokenizer is a small change with potentially large effect).

Numbers taken on 2026-05-16. Same hardware as the original
distractor-mix experiment. Per-seed wall-clock ~10 min for 3 archs.
