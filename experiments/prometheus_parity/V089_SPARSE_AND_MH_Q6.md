# v0.8.9 — MH+Q6 compound confirmed + sparse attention kernel shipped

Two of three goal items landed with hard data; the third (d_model=128
larger-scale bench) is still running and will close in v0.8.10.

## Item #3: MH+Q6 compound — v0.8.8 finding scales

The v0.8.8 measurement showed Q6 training pushes attention 8.3× toward
substrate positions in single-head mode. Hypothesis for #3: if Q6
sculpts attention per-head, then MH+Q6 should compound harder than
SH+Q6.

**Result** (d_model=32, n_heads=4, 250 steps, 3 seeds):

| arm | mean tail loss | Δ from SH | (%) |
|---|--:|--:|--:|
| SH (single head) | 2.0309 | — | — |
| SH + Q6 fused | 1.9865 | **−0.0444** | **−2.19%** |
| MH (4 heads) | 2.0486 | +0.0177 | +0.87% |
| **MH (4h) + Q6 fused** | **1.9754** | **−0.0555** | **−2.73%** |

**Compound analysis**:
- `SH → SH+Q6`: −2.19% (Q6 alone)
- `MH → MH+Q6`: **−3.57%** (Q6 in MH is *larger* than Q6 in SH)
- `SH → MH+Q6`: −2.73% (compound, dominated by Q6 not MH)

**Confirmed**: Q6 gets more leverage in MH than in SH (−3.57% vs −2.19%).
Each head has its own Q to sculpt; Q6 modulation operates independently
per head and the per-head substrate alignment compounds at attention
time. **The v0.8.8 attention-shaping finding scales architecturally.**

What this implies for PyTorch parity: the PyTorch Q6 finding was
−12.15% at L1-MH on TinyShakespeare. OMC at much smaller scale (32-dim
single block, 250 steps, 165-char corpus) gets −2.73%. The directional
relationship holds; the magnitude will scale with capacity.

## Item #1: sparse substrate attention kernel — mechanism works, no speedup at this scale

**Shipped**: `tape_substrate_sparse_scores(q_id, k_id, threshold)` op
in `omnimcode-core::interpreter`. Forward computes scores only at
cells where `substrate_dist(i, j) ≤ threshold` (CRT moduli
{5, 8, 13, 21}), masks the rest to −∞ so subsequent softmax assigns
zero. Backward only flows through fired cells.

**Cell density telemetry** (set `OMC_GPU_VERBOSE=1`):
```
[sparse-scores] 70/1024 cells = 6.8%
```
**Exactly matches the v0.8.8 measurement** — 6.84% of cells have
substrate_dist ≤ 5 at seq_len=32 with CRT moduli {5, 8, 13, 21}.

### Wall-clock at seq_len=32, d_model=32 (10-iter avg, post-Q6 training)

| variant | forward ms/iter |
|---|--:|
| dense | 0.2723 |
| sparse | 0.2736 |
| **speedup** | **1.00×** |

**No speedup at this scale.** The dense path lives in `tape_matmul`'s
tight inner loop (or wgpu); the sparse path is a naive scalar
Rust triple-loop with per-cell substrate distance recomputation. At
seq_len=32 the savings on score computation (93% fewer MACs) are eaten
by the per-cell substrate-distance check and the cache-unfriendly
sparse access pattern.

L1 difference between dense softmax(q@k^T) and sparse softmax: 57.44
across 1024 cells (per-cell mean 0.056). Sparse captures the dominant
attention positions but with measurable divergence at the −∞-masked
cells.

### Reformulation for v0.8.10+ (path to real speedup)

The sparse kernel's mechanism is correct. The speedup needs:

1. **Larger seq_len** — at seq_len=64+, dense matmul cost is `seq²·d`
   while sparse is `(seq · density · seq)·d`. The 93% saved MACs
   start to dominate the constant per-cell overhead.
2. **Precomputed substrate mask** — the (i, j) → fired/not table is
   identical across batches and only depends on seq_len. Compute once,
   reuse forever.
3. **CSR / packed sparse format** — replace the dense `[N×N]` output
   matrix (most cells = -inf) with a compact list of (i, j, score)
   tuples and a per-row prefix index. Softmax becomes per-row over the
   fired cells only.
4. **WGSL implementation** — once shapes pass the GPU threshold, port
   to a sparse compute kernel. The 6.8% density is the substrate's
   architectural sparsity prior.

The v0.8.8 finding (substrate predicts where attention lives after
training) holds; the kernel landed but its speedup is a v0.8.10
follow-up. The chapter is **algorithmically validated, not yet
production-speed**.

## Item #2: d_model=128 larger-scale bench — in-flight

Background bench running task #265 (L0 vs B (L1+SMOD+V) vs B+Q6 fused
at d_model=128, 400 steps, 3 seeds, GPU). 13+ minutes in at chapter
write time; will land in v0.8.10 with the actual MH-at-128 datum.
This is the data point that would close PyTorch parity: their L1-MH
finding was −8.94% at TinyShakespeare scale.

## Compounding architecture continues

- v0.8.1 broadcast-backward unblocked S-MOD training
- v0.8.4 fused AdamW dissolved 96× overhead
- v0.8.5 multi-head substrate-K cross-validated
- v0.8.7 four deferred items each TRIED
- v0.8.8 Q6 post-training substrate alignment + JIT eligibility
- **v0.8.9 MH+Q6 compound confirmed + sparse kernel mechanism shipped**

The pattern: each chapter validates the previous chapter's hypothesis
or surfaces the next bottleneck. The Q6 attention-shaping finding from
v0.8.8 is the throughline — v0.8.9 #3 confirms it scales to MH and
v0.8.9 #1 ships the kernel that exploits it (mechanism only, speedup
pending).

## Files

- `omnimcode-core/src/interpreter.rs` — `TapeOp::SubstrateSparseScores`,
  `tape_substrate_sparse_scores` dispatch, sparse forward + backward
- `examples/prometheus_mh_q6_compound.omc` — #3 4-arm A/B
- `examples/prometheus_sparse_attn_bench.omc` — #1 dense-vs-sparse harness

## Tests

**1111/1111 OMC tests pass.**
