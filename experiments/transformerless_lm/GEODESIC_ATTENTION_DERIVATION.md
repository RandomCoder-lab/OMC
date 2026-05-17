# Geodesic attention — deriving from what we've measured

## What we actually know (not what we hoped)

After CRT-PE (2 wins) + HBit OOD (1 win) + three falsified attention
gates, the empirical map is:

| Where substrate applied | Basis | Result |
|---|---|---|
| Position → CRT-PE | integer position `i` | **WINS** −5.4% / −2.9% |
| Reference-free OOD score | per-sample HBit tension | **WINS** AUROC 1.0 |
| Attention KEY magnitude gate | learned float `\|k\|.mean(-1)` | FAILS 0/3 |
| Attention SCORE gate | learned float `q @ k^T / √d` | FAILS 0/3 |
| Same with learned threshold | same float quantity | FAILS 0/3 |

**The common failure pattern**: every loss applied
`attractor_distance(·)` to a *continuous, Gaussian-ish, learned*
quantity. Those quantities have no architectural reason to land
on Fibonacci attractors — those attractors live in integer ID
space (the basis that CRT-PE actually uses).

**The wins share a pattern**: substrate signal applied to a
quantity that's *intrinsically integer-valued* (positions in
CRT-PE) or *aggregated cross-position* (HBit OOD over a sample).
The substrate's lattice lives in those bases.

## The right basis for attention bias

Attention has TWO sources of structure:
1. **The query/key activations** (continuous, learned, no substrate
   structure → all three previous attempts)
2. **The query/key POSITIONS** (integer, indexed 0..T, *is*
   meaningful in substrate space — that's why CRT-PE works)

We've been adding the substrate signal to source #1. The right move
is to add it to source #2. Specifically: **attention bias should be
a function of geodesic distance between positions i and j in the
same CRT-Fibonacci-moduli space CRT-PE already uses.**

## The formula

For positions i, j and Fibonacci moduli M = {5, 8, 13, 21, 34, 55, 89, 144}:

```
d_circ(i, j, m) = min(|(i % m) − (j % m)|, m − |(i % m) − (j % m)|)
geodesic(i, j) = Σ_{m ∈ M} d_circ(i, j, m) / m       # normalize to [0, ~|M|/2]
```

Each per-modulus term is a circular distance on a ring of size `m`
(positions sharing the same residue contribute 0; antipodal residues
contribute `m/2`). The total is the L1 sum over moduli — the
geodesic length in the CRT-Fibonacci lattice.

Why circular: positions on a ring of size `m` should be treated as
adjacent at the wrap. This matches CRT-PE which uses
`sin(2π·pos%m/m)` — same circularity.

## The attention modification

Pre-softmax additive bias (the form that works for ALiBi):

```
scores_ij = (q_i · k_j) / √d − α · geodesic(i, j)
attn = softmax(scores)
```

α is a learned scalar per head (initialized to 0 — model can disable
substrate signal if loss says to; same fairness as
`hybrid_learned`).

## Why this should work where the previous three failed

| Property | Previous gates | Geodesic |
|---|:-:|:-:|
| Substrate metric applied to integer quantities | ✗ | ✓ |
| Same basis as CRT-PE (proven to work) | ✗ | ✓ |
| Composes additively with softmax | partly | ✓ |
| Model can disable via single learnable | ✓ | ✓ |
| Computable once at init (not per-batch) | ✗ | ✓ |
| Independent of token content | ✗ | ✓ |

The last two are important: the geodesic table is `[T, T]`
precomputed at model construction. Forward pass adds the bias
without computing anything per-batch. This is essentially **ALiBi
with substrate-geodesic distances instead of plain absolute
distance** — and ALiBi itself is known to work, so the prior on
this formulation is much stronger than another activation gate.

## Falsifiable prediction

- If geodesic attention WINS vs crt_only on the distractor mix:
  substrate IS useful as an attention modulator, but the basis
  matters. The transformerless thesis gets a third architectural
  win.
- If geodesic attention LOSES: attention modulation in OMC's
  substrate is truly dead at this scale, regardless of basis.
  Honest pivot to tokenizer-layer substrate becomes the only
  remaining substrate-in-attention story.

Either way, this is the final attention-side experiment. After
this we're moving the substrate's role away from attention
unless this works.

## Init details (matters for fair comparison)

- α = 0.0 per head (disabled gate at init — the model has to
  *find* the bias useful from gradient signal alone)
- Geodesic table normalized so its mean over (i, j) for i ≠ j
  is approximately 1.0 (so α has interpretable units)
- All other hyperparameters identical to
  `train_gate_reformulation.py` (d_model=128, n_blocks=4,
  seq_len=128, 1500 steps, distractor_frac=0.20, 3 seeds)

The only architectural variable changed from `crt_only` is the
addition of the geodesic bias to attention scores. Everything else
identical.
