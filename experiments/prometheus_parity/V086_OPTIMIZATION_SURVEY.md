# v0.8.5 / v0.8.6 optimization sweep — what shipped, what's scoped, what's next

The user's optimization roadmap had 10 items. Five shipped in v0.8.5
(#1, #2, #4 negative, #5, #6) and one in v0.8.6 (#3, scaffold only).
Items #7-10 are each their own chapter. This doc records the honest state.

## Shipped

| # | item | status | notes |
|---|---|---|---|
| 1 | tape_cross_entropy_batch fused | v0.8.5 ✓ | closed-form (p−one_hot)/N backward, 5→1 tape nodes |
| 2 | tape_embedding_lookup direct gather | v0.8.5 ✓ | skips one-hot construction |
| 3 | route more tape ops through GPU | v0.8.6 scaffold | softmax hook in place, default declines |
| 4 | OMC_VM=1 on tape paths | v0.8.5 negative | 0.662 s/step vs 0.661 tree-walk |
| 5 | multi-head substrate-K | v0.8.5 ✓ | -0.25% MH vs SH, wins 2/3 seeds, d_model=32 |
| 6 | tape_substrate_resample fused | v0.8.5 ✓ | (tape_smod_softmax fusion deferred — bigger backward chain) |

## Scoped — each its own future chapter

### #7 Substrate-quantized GPU weights

**Goal**: encode f32 weights as substrate-shaped (attractor index + small delta) for smaller buffers and more bandwidth.

**What needs to happen**:
1. Rust quantizer: given an f64 cell, return `(u8 attractor_index, i16 delta)` where attractor_index is into the FIBONACCI table (40 entries, 6 bits used) and delta is a signed offset from the attractor.
2. Dequantizer: inverse. `attractor + (delta / scale)` reconstructs an approximate f64.
3. CPU-side validation: train a Prometheus model where every parameter goes through quantize→dequantize on each forward. Compare loss curve to baseline. If quality holds, the substrate encoding is doing useful work.
4. GPU port: a WGSL shader that takes packed u24-per-cell substrate-encoded buffer + emits f32 matmul inputs. Bench bandwidth-bound shapes (d_model=1024+).

**Expected payoff**: 1.3-2× on memory-bandwidth-bound matmuls. Substrate encoding has structured (not random) quant noise which the model may train around better than uniform i8 quantization.

**Why not shipped this chapter**: substantial cross-layer work — quantizer in Rust + WGSL changes + bench harness. Each piece is straightforward; together is ~half a day.

### #8 CRT-PE-keyed sparse attention matmul

**Goal**: for `scores = q @ k^T` where k is the CRT-PE table, only compute output cells where the CRT-substrate distance between (row, col) is small. Skip far pairs (they softmax to ~0 anyway).

**What needs to happen**:
1. CSR or coordinate-list sparse output buffer.
2. WGSL kernel that walks the query row, computes substrate-distance to each candidate col, skips above threshold.
3. Backward needs to scatter the sparse gradient back into a dense q grad. Doable but non-trivial.
4. Bench at seq_len=512+ where the sparsity payoff is large.

**Expected payoff**: 5-20× on attention computation at long sequences; minimal/negative at seq_len=64 because the substrate-distance check costs more than the saved MACs.

**Why not shipped**: real WGSL work for a sparse kernel + the OMC tape op needs sparse-aware backward. Half-day to a day of focused work.

### #9 omnimcode-codegen LLVM JIT for hot Prometheus paths

**Goal**: JIT-compile hot OMC functions (the `forward_window`, `train_arm` outer loops) to native via the existing omnimcode-codegen crate.

**What needs to happen**:
1. Identify Prometheus orchestration functions that JIT-elidigible (no tape mutation? no closures? need to check).
2. Currently the JIT path is opt-in via OMC_HBIT_JIT=1 — needs testing on tape-using code.
3. Tape ops are already in Rust; JIT'ing the OMC orchestration loop around them would compress the 10-50% of time still spent in OMC interp.

**Expected payoff**: 1.5-3× if the OMC orchestration overhead is non-trivial; near-zero if tape ops dominate (which v0.8.4 indicated they do at d_model=256).

**Why not shipped**: needs JIT compatibility audit of the Prometheus code path. Likely several hours of debugging if JIT chokes on prom_* fns.

### #10 f16/bfloat16 GPU paths

**Goal**: a second WGSL kernel variant taking f16 inputs. Halves the memory bandwidth, may halve the latency on bandwidth-bound shapes.

**What needs to happen**:
1. New WGSL kernel using `f16` type (or `i16`/`u16` packed).
2. f64 → f16 conversion at the boundary; verify training stability.
3. wgpu may need a feature flag for f16.

**Expected payoff**: ~2× on bandwidth-bound shapes (large weight matrices); training stability is the open question — PyTorch trains f16 with loss scaling, which we'd need to replicate.

**Why not shipped**: requires loss-scaling logic for training stability. Substantial cross-layer work.

## What the "try → if failed, reformulate → try again" record looks like

- #1 cross-entropy: tried (cheap), shipped, small visible wall-clock gain at vocab=32 (the test scale), bigger gain expected at vocab=10k+
- #2 embedding lookup: tried, shipped, same story (small at our vocabs, big at larger)
- #3 softmax through GPU: tried with the scaffold; **reformulated** the goal once measurement showed memory-bound element-wise ops won't benefit at our shapes; shipped the scaffold so larger-scale or different-hardware runs can opt in
- #4 OMC_VM=1: tried with zero code (free experiment), **negative result**, recorded and not pursued — that's the correct "fail forward"
- #5 multi-head substrate-K: tried, shipped, -0.25% with 2/3 wins (directionally consistent with PyTorch L1-MH -8.94%)
- #6 substrate_resample fused: tried, shipped, eliminates tape_value round-trip
- #7-10: scoped honestly above. Each is its own chapter.

## Velocity

Five items + scaffold of one = 6/10 of the v0.8.5 plan in one chapter. The 4 remaining are each substantial enough to deserve focused attention rather than being rushed in this same chapter.

Rome wasn't built overnight; v0.8 was built across 6 chapters this week.
