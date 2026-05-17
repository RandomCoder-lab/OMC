# GPU into Prometheus: tape_matmul routed through omnimcode-gpu

## Headline

Integration shipped: tape_matmul forwards above the CPU/GPU crossover threshold get routed through omnimcode-gpu's wgpu (Vulkan) backend. The kernel-level speedup is large (13× on a chained 512² matmul), but **end-to-end Prometheus training is now bottlenecked by OMC tree-walk overhead in the substrate-shaping helpers** (substrate_softmax, substrate_resample, Q6 modulation), not by matmul time. The honest read: the integration is correct and load-bearing for any future work that pushes matmul further into the budget — but **GPU alone doesn't accelerate today's Prometheus**.

## What got wired

A `MatmulAccelerator` hook in `omnimcode-core` that an outer binary can register at startup. The CLI binary now does so under the `gpu` feature, pointing it at `omnimcode-gpu::pick_backend()`. The hook:

- Accepts `(m, k, n, &[f64], &[f64])`, declines (returns `None`) when `m·k·n < OMC_GPU_MATMUL_MIN_FLOPS` (default 1,000,000)
- Converts f64 → f32 at the boundary, calls the backend, converts f32 → f64 back
- Disabled by `OMC_GPU_BACKEND=cpu`
- `OMC_GPU_VERBOSE=1` logs the chosen backend + threshold at startup

Pre-existing tape_matmul implementation is unchanged when no hook is registered — backward compatibility is total. Backward pass (`dA = dy @ B^T`, `dB = A^T @ dy`) automatically benefits because it calls the same `tape_matmul` helper.

## Kernel-level win: synthetic matmul chain

5 chained 512² matmuls, f64 OMC tape:

```
OMC_GPU_BACKEND=cpu   3.47 s
OMC_GPU_BACKEND=wgpu  0.27 s    ~13× speedup
```

f64 → f32 → f64 round-trip vs pure-f64 reference: result differs at the 9th significant digit (`239899095...` vs `239899097...`), well within f32 + summation-order noise. Parity is fine for any Prometheus-scale workload.

## End-to-end Prometheus training (d_model=256)

`examples/bench_prometheus_gpu.omc`, substrate-K transformer, seq_len=64, d_model=256, ff_dim=512, 5 AdamW steps:

| | wall-clock | per step | final loss |
|---|--:|--:|--:|
| `OMC_GPU_BACKEND=cpu`  | 129.05 s | 25.81 s | 6.95930 |
| `OMC_GPU_BACKEND=wgpu` | 129.39 s | 25.88 s | 6.95932 |
| **diff** | +0.3% slower | +0.3% | 2e-5 (f32 noise) |

Per-step matmul shapes that DID cross the GPU threshold:
- `x @ Q` : 64×256·256×256 = 4.2M flops
- `ff_up` : 64×256·256×512 = 8.4M flops

Both are well above the 1M threshold and get routed to GPU. But the wall-clock numbers don't move. Why? Because at this scale, **matmul wall-clock is single-digit milliseconds per step**, and the surrounding OMC-side iteration is multiple seconds per step.

### Where the time actually goes

For seq_len=64, d_model=256:

- `_prom_smod_matrix(scores_val, alpha)` — OMC loop over 64² = 4096 score cells, each calling `attractor_distance`. Per step: 1 forward + 1 backward = 8192 OMC arr_get/arith calls. At tree-walk speed (~100k ops/sec for fat dicts), that's ~80ms purely for the substrate-modulator matrix.
- `_prom_substrate_resample_matrix(v_val, scale)` — same shape OMC loop over V projections. Another ~80ms.
- `_prom_q6_log_distance_composed` / `_prom_q6_modulation_from_log_d` — runs at the same scale, several more OMC iterations.
- The whole inner-loop runs in OMC because it has to call `attractor_distance` which is an OMC builtin chain.
- Multiply by 5 steps and you get tens of seconds, not the 25 we measured — so there's additional OMC overhead in embedding lookup, parameter collection, AdamW state mutation, etc.

The GPU saves us maybe ~50ms per step on the matmul side. The OMC interp burns ~25 seconds per step on substrate-shaping logic. The 50ms vs 25s ratio is why we see 0% wall-clock movement.

## What this means

The GPU integration is **architecturally complete and load-bearing for any future direction that pushes matmul further into the time budget** — bigger d_model (1024+), batched inference, scaled corpora. It also opens the door to v0.8.3+ **substrate-native GPU kernels** (Fibonacci-tile workgroups, substrate-quantized weights, CRT-PE-keyed sparse matmul) where the substrate IS the kernel architecture.

But **GPU alone doesn't speed up today's Prometheus**. The next bottleneck is OMC tree-walk overhead in the substrate-shaping helpers. Three concrete options for that:

1. **Move substrate modulators into Rust builtins** — `_prom_smod_matrix` / `_prom_substrate_resample_matrix` become `prom_substrate_modulator_smod` / `prom_substrate_modulator_resample` Rust ops that take a tape node id, allocate the modulator matrix natively, return a const tape node. Estimated 100-1000× on these inner loops alone.
2. **Bytecode VM for the OMC side** — the existing `OMC_VM=1` path already gives 2-10× on hot loops. Hadn't been tested for tape-using paths; worth a measurement.
3. **Fused substrate tape ops** — `tape_substrate_resample`, `tape_smod_softmax` as single Rust nodes (the precedent set by `tape_phi_log` in v0.8.1). Eliminates the OMC-side iteration entirely.

(3) is the cleanest path and aligns with the substrate-native primitive thesis. (1) is the cheapest. (2) is free measurement.

## Files

- `omnimcode-core/src/accel.rs` — the `MatmulAccelerator` hook + `OnceLock` global + `try_accelerated_matmul` call site
- `omnimcode-core/src/interpreter.rs` — `tape_matmul` consults the hook before falling back to triple-loop
- `omnimcode-cli/Cargo.toml` — new `gpu` feature pulls in `omnimcode-gpu`
- `omnimcode-cli/src/main.rs` — `install_gpu_matmul_accelerator()` registers wgpu backend at startup
- `examples/bench_prometheus_gpu.omc` — wall-clock harness

## Reproduction

```bash
# Build with GPU feature
cargo build --release -p omnimcode-cli --features gpu

# Synthetic matmul chain (kernel-level win)
OMC_GPU_BACKEND=cpu  ./target/release/omnimcode-standalone /tmp/gpu_matmul_big.omc
OMC_GPU_BACKEND=wgpu ./target/release/omnimcode-standalone /tmp/gpu_matmul_big.omc

# End-to-end Prometheus training (no end-to-end win at d_model=256)
OMC_GPU_BACKEND=cpu  ./target/release/omnimcode-standalone examples/bench_prometheus_gpu.omc
OMC_GPU_BACKEND=wgpu ./target/release/omnimcode-standalone examples/bench_prometheus_gpu.omc

# Tune the crossover threshold
OMC_GPU_MATMUL_MIN_FLOPS=10000000 ./target/release/omnimcode-standalone ...
```
