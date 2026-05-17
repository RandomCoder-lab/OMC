# v0.8.4 substrate-builtins: 40× CPU, 96× GPU, end-to-end on Prometheus

## Headline

Three Rust builtins replace the OMC-side inner-loop helpers that were
the v0.8.2 wall-clock bottleneck:

- `substrate_smod_matrix(scores, alpha)` — Rust port of `_prom_smod_matrix`
- `substrate_resample_matrix(v, scale)` — Rust port of `_prom_substrate_resample_matrix`
- `substrate_adamw_update(cur, grad, m, v, lr, b1, b2, eps, wd, step)` — fused AdamW per-parameter update

End-to-end on the same d_model=256 Prometheus training that v0.8.2 ran:

| version | CPU s/step | GPU s/step | total speedup vs v0.8.2 |
|---|--:|--:|--:|
| v0.8.2 (baseline, OMC-side helpers) | 25.81 | 25.88 | 1.00× |
| v0.8.4 (smod+resample Rust) | 26.38 | 26.28 | 0.98× ← no change |
| **v0.8.4 (+ fused AdamW)** | **0.65** | **0.27** | **40× / 96×** |

The first round of porting (smod + resample matrix construction) didn't
move the wall-clock at all — useful debugging finding. The real
bottleneck was `prom_adamw_step`, which ran ~15 OMC-side element-wise
loops per parameter per step. Replacing that inner block with one Rust
builtin produced the 40× CPU and 96× GPU speedup.

Loss agrees with v0.8.2 to 5e-5 (f32 GPU roundtrip noise); training
trajectory is identical.

## Why the first round didn't help

`_prom_smod_matrix` walks a 64×64 scores matrix per forward+backward, doing
4096 cells × 2 calls = 8192 attractor_distance + scalar arith calls per step.
That's milliseconds in the tree-walk interpreter — not nothing, but tiny
relative to the 25-second per-step cost.

`prom_adamw_step` walks every parameter (6 of them at d_model=256, sizes
ranging from 256² to 32×256) doing **15 element-wise loops per parameter**
in OMC: `_prom_zip(_prom_scale(...), _prom_scale(...), "add")` chained
through `_prom_zip(_prom_zip(...), _prom_zip(...), "div")` and so on. At
256² = 65k cells per param × 15 loops × 6 params × OMC tree-walk speed
(~10K ops/sec for nested-array iteration), that's tens of seconds per step.
Confirmed by the math; confirmed by the 40× drop after the fix.

## The fused AdamW builtin

```rust
substrate_adamw_update(cur, grad, m, v, lr, b1, b2, eps, wd, step)
```

- Takes OMC arrays for cur/grad/m/v (1D or 2D, same shape across all four)
- Flattens to `Vec<f64>` once, runs the inner loop entirely in Rust:
  ```
  m ← β₁·m + (1−β₁)·g
  v ← β₂·v + (1−β₂)·g²
  m̂ = m / (1 − β₁^step)
  v̂ = v / (1 − β₂^step)
  p ← cur − lr·wd·cur − lr · m̂ / (√v̂ + ε)
  ```
- Mutates `m` and `v` in place (Rc-shared OMC arrays — caller sees update)
- Returns the new parameter value as a freshly-allocated OMC array

OMC-side change is minimal — `prom_adamw_step` keeps its same outer loop
over parameters, just replaces the ~30-line inner block with one builtin
call. Existing callers (every Prometheus training script) pick up the
speedup automatically; the public AdamW interface is unchanged.

## The compound effect

v0.8.2 wired GPU in. v0.8.3 found the substrate-shaped 8×32 tile that
hit 114 GFLOPS. Neither moved end-to-end wall-clock because the OMC
overhead drowned everything. v0.8.4 removes the overhead — and now both
prior chapters' work actually pays out:

- **CPU**: 25.81 → 0.65 s/step = 40× speedup. AdamW reduction alone.
- **GPU**: 25.88 → 0.27 s/step = **96× speedup**. AdamW reduction + v0.8.3 substrate-tile win finally matters.
- **GPU vs CPU at v0.8.4**: 2.4× faster. This is what we'd expect from the matmul speedup at d_model=256.

The chapters are now compositional. Future scale-ups (d_model=512+,
batched inference, longer sequences) get *both* the OMC-overhead-gone
benefit AND the GPU acceleration that v0.8.2/3 enable.

## What this unlocks (immediately)

- **L1-MH + S-MOD α=1.0 in pure-OMC Prometheus** (task #264) — was unblocked by v0.8.1's broadcast-backward fix; was *impractical* until v0.8.4 made training take seconds rather than minutes per step.
- **Larger-scale substrate-attention** (task #265) — d_model=512, longer sequences, multi-block. Was 5-10 minutes per training step pre-v0.8.4; now sub-second.
- **Q6 cross-validation at real training length** — the v0.8.1 OMC-side Q6 finding was at 80 steps (the slowest we could afford). Can now run 5000+ step training in OMC and properly cross-validate the PyTorch -12.15% result.

## Tests

- `examples/tests/test_substrate_modulator_builtins.omc` — 8 tests: substrate_smod_matrix and substrate_resample_matrix forward correctness + equivalence vs the OMC wrapper helpers
- All 22 existing Prometheus tests still pass — fused AdamW produces identical training trajectories
- Full suite: **1111/1111 OMC tests pass**

## Files

- `omnimcode-core/src/interpreter.rs`
  - `substrate_smod_matrix` builtin
  - `substrate_resample_matrix` builtin
  - `substrate_adamw_update` builtin
  - Helpers: `flatten_2d_or_1d`, `write_back_1d_or_2d`, `rebuild_omc_array`,
    `was_2d`, `build_substrate_modulator_matrix`, `ModulatorKind`
- `examples/lib/prometheus.omc`
  - `_prom_smod_matrix` is now a wrapper around the builtin
  - `_prom_substrate_resample_matrix` is now a wrapper around the builtin
  - `prom_adamw_step` inner block replaced with `substrate_adamw_update` call
- `examples/tests/test_substrate_modulator_builtins.omc`

## Honest framing

The first round of porting (modulator matrices) didn't help end-to-end —
it was a hypothesis that turned out to be wrong about *where* the
bottleneck lived. Profiling-by-fixing found the real bottleneck in AdamW.
Both ports are shipped: the modulator builtins because they're
architecturally cleaner and verified correct, the AdamW builtin because
it's the actual win.

## Reproduction

```bash
cargo build --release -p omnimcode-cli --features gpu

# CPU baseline (now fast)
OMC_GPU_BACKEND=cpu ./target/release/omnimcode-standalone examples/bench_prometheus_gpu.omc

# GPU (now wins)
OMC_GPU_BACKEND=wgpu ./target/release/omnimcode-standalone examples/bench_prometheus_gpu.omc
```
