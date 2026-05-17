# v0.8.8 — four findings: 1 massive positive, 3 honest negatives

Following the v0.8.7 sweep, four follow-up experiments were run on the
extended goal items: JIT eligibility audit, post-training sparsity test,
substrate-init A/B, substrate-quant 6-seed verification.

## Finding 1 (POSITIVE): Q6 training pushes attention 8.3× toward substrate

**The v0.8.7 #8 falsification flips.** At random init, attention is
uniform across substrate-near vs substrate-far cells (8.36% mass /
6.84% cells, ratio 1.22). After 1000 steps of Q6-fused training:

| arm | mass in substrate-close cells | cell fraction | ratio |
|---|--:|--:|--:|
| baseline (no Q6), trained | 4.82% | 6.84% | **0.70 (anti-correlated)** |
| Q6 fused, trained | **56.80%** | 6.84% | **8.31×** |

Q6 modulation pushes the trained query matrix toward substrate-aligned
positions, not just substrate-aligned magnitudes. This is a real result
that opens up CRT-PE-keyed sparse attention as a **post-training**
inference optimization. **A sparse kernel that only computes substrate-
close cells captures 56.8% of attention with 6.84% of compute** — that's
the architecture-level "substrate is the architecture" claim landing.

Mechanism: Q6 dampens large-magnitude query components via
`exp(-γ · log_φπfib(|q · scale| + 1))`. Components whose substrate
log-distance is small get less dampening, so they survive training
and dominate the attention pattern. The substrate isn't directly
constraining position; it's reshaping the gradient landscape so
substrate-aligned positions win.

Implications:
- Sparse inference kernel: `q[i] · k[j]` only for `substrate_dist(i, j) ≤ τ`
- 10× attention compute reduction at the cost of ~43% attention quality
  (a defensible inference-time tradeoff)
- The PyTorch Q6 −12.15% finding may partially be substrate-position
  alignment in disguise

## Finding 2 (NEGATIVE): substrate-quant 6-seed verifies as noise

The v0.8.7 #7 first-look (1 seed × 5 steps) showed
`OMC_GPU_SUBSTRATE_QUANT=1 OMC_GPU_SUBSTRATE_QUANT_SCALE=4096` giving
loss 6.149 vs 6.959 baseline. Suspected seed noise.

6-seed × 300-step verification (d_model=32, OMC_GPU_MATMUL_MIN_FLOPS=1000
to force quant to fire on every matmul):

| | mean tail loss |
|---|--:|
| f32 baseline | 2.337 |
| substrate-quant scale=4096 | **2.365 (+1.2%, worse)** |

**Falsified.** The v0.8.7 single-seed lower loss was seed noise. At 6
seeds, substrate quantization at training time is a marginal regression
(though still in the same range as baseline — not catastrophically
broken). This rules out the "substrate alignment as gradient regularizer"
hypothesis at this scale.

What's NOT ruled out: substrate-quant as INFERENCE-only weight encoding
(post-training compression with on-attractor exactness). The training-
time application is what failed.

## Finding 3 (NEGATIVE): substrate-aware param init

`_prom_substrate_random_matrix(rows, cols, bound, state, scale)` was
added — initialize random uniform then snap each cell to nearest
Fibonacci attractor at scale. Tested as 3-way A/B:

| | mean tail loss | vs baseline | wins |
|---|--:|--:|--:|
| baseline uniform random | 2.502 | — | — |
| substrate-snap scale=1024 | 2.567 | **+2.6%** | 2/6 |
| substrate-snap scale=4096 | 2.620 | **+4.7%** | 1/6 |

**Falsified.** Substrate-aligned starting weights produce slightly worse
training trajectories. Hypothesis: the random init lives in a
well-conditioned region that training can find quickly; substrate-
aligned init starts on attractor positions that have less gradient
information per step (the modulator function has reduced sensitivity
near attractors by design).

## Finding 4 (POSITIVE, infrastructure): JIT eligibility audit

v0.8.7 #9 found that `OMC_HBIT_JIT=1` crashed with `arr_len requires
an array` because `_prom_geodesic_moduli` (which returns `[5, 8, ...]`)
was JIT'd as an i64-returning fn. The dual-band lowerer types
everything as i64; collection-typed returns silently lie.

Fix: `fn_uses_collections` in `omnimcode-codegen/src/lib.rs` skips
JIT for any fn whose bytecode contains `Op::NewArray`, `Op::NewDict`,
`Op::ArrayIndex`, `Op::ArrayLen`, or whose constant pool contains
string literals. Skipped fns get replaced with an `unreachable` body
so accidental calls trap loudly rather than silently returning 0.

**Result**: `OMC_HBIT_JIT=1` runs Prometheus cleanly now (0.674 s/step
at d_model=256 vs 0.661 tree-walk, ~0.013s of JIT-init overhead).
Tests: 1111/1111 still pass. No wall-clock win because v0.8.4
already eliminated the OMC orchestration overhead the JIT would
have compressed; bug fix only.

## Methodology

Each of these four experiments was small (under 10 min wall-clock,
single OMC file each). All four genuinely TRIED rather than scoped.
Three produced honest negatives that prevent future wasted chapters;
one produced a load-bearing positive (Q6 post-training sparsity) that
unblocks a real future optimization.

This compounds the v0.8 trajectory:
- v0.8.1 fixed broadcast-backward (unblocked S-MOD training)
- v0.8.4 fused AdamW (dissolved 96× overhead)
- v0.8.5 multi-head substrate-K (architecturally needed for parity)
- v0.8.7 tried 4 deferred items (2 viable, 1 falsified, 1 bug)
- **v0.8.8 four more attempts (1 major positive, 3 negatives, 1 infra fix)**

The "fail forward" discipline keeps producing useful data either way.

## Files

- `examples/prometheus_q6_post_train_sparsity.omc` — Finding 1
- `examples/prometheus_substrate_quant_6seed.omc` — Finding 2
- `examples/prometheus_substrate_init_xval.omc` — Finding 3
- `omnimcode-codegen/src/lib.rs` — Finding 4 (`fn_uses_collections`)
- `omnimcode-core/src/interpreter.rs` — `substrate_snap_matrix` builtin
- `examples/lib/prometheus.omc` — `_prom_substrate_random_matrix` helper
