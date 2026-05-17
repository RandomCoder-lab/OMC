# omnimcode-gpu

GPU compute scaffold for Prometheus. Pluggable backends behind a single trait; defaults to **wgpu (Vulkan / Metal / DX12 / OpenGL compute)** for cross-vendor portability without driver headaches.

## Status

**v0.7 — scaffold.** One operation (f32 matmul) implemented end-to-end on:

- `CpuBackend` — naive triple-loop, always available, ground-truth reference
- `WgpuBackend` (feature `wgpu`) — Vulkan / Metal / DX12 / OpenGL compute

ROCm and CUDA backends are stubbed in the trait but not implemented yet.

## Why wgpu over ROCm/CUDA

The user's primary target is an **AMD Radeon RX 580 (Polaris / gfx803)**. The honest situation there:

- **Official ROCm dropped Polaris support at version 4.0.** Newer ROCm (5.x, 6.x) doesn't ship gfx803 kernels.
- **Unofficial Polaris ROCm builds exist** but they're fragile — "Ollama gets fussy about it" was the user's verbatim description, which matches the broader community experience.
- **Vulkan compute works out of the box** on the same hardware via the open-source RADV driver. The Mesa-driven Vulkan path on this card is stable and well-tested.

So the default GPU backend is wgpu (Vulkan). ROCm/CUDA can be plugged in later via the same `ComputeBackend` trait when the user has supported hardware.

## Measured on the target hardware (AMD RX 580 / RADV Vulkan)

```
    size (m x k x n)       cpu ms      wgpu ms    speedup  parity
---------------------------------------------------------------------------
            64x64x64        0.052        0.228      0.23x  OK
         128x128x128        0.281        0.340      0.83x  OK
         256x256x256        1.966        0.880      2.24x  OK
         512x512x512       14.503        4.273      3.39x  OK
      1024x1024x1024      115.516       28.577      4.04x  OK
```

Crossover at ~128×128. By 1024×1024, GPU is 4× faster than the naive CPU baseline. Parity verified (GPU output matches CPU within f32 rounding) at every size.

## Build

```bash
# CPU-only (no GPU deps, builds everywhere)
cargo build --release -p omnimcode-gpu

# With wgpu Vulkan/Metal/DX12 backend
cargo build --release -p omnimcode-gpu --features wgpu
```

## Run the benchmark

```bash
cargo run --release -p omnimcode-gpu --features wgpu --example bench_matmul
```

## Pick a backend programmatically

```rust
use omnimcode_gpu::{pick_backend, Matrix};

let backend = pick_backend();    // wgpu if built+available, else CPU
let a = Matrix::new(128, 128, vec![0.5; 128 * 128]);
let b = Matrix::new(128, 128, vec![0.5; 128 * 128]);
let c = backend.matmul(&a, &b).unwrap();
```

Override via env:

```bash
OMC_GPU_BACKEND=cpu cargo run ...     # force CPU
OMC_GPU_BACKEND=wgpu cargo run ...    # force wgpu (errors if feature not built)
```

## How to add a new backend

Implement `ComputeBackend` for your type, gate it behind a Cargo feature, plumb it into `pick_backend()`. The trait is intentionally tiny (one method right now) so adding a new backend is mechanical.

```rust
pub struct CudaBackend { /* ... */ }
impl ComputeBackend for CudaBackend {
    fn name(&self) -> &'static str { "cuda" }
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, BackendError> {
        // cuBLAS sgemm call here
    }
}
```

## What's NOT in v0.7

- **Prometheus integration.** The tape ops in `examples/lib/prometheus.omc` still run pure-OMC. v0.8 would route `tape_matmul` through this backend when shapes exceed the CPU-crossover threshold.
- **Backward pass on GPU.** Only forward matmul. Backward requires the gradient autotape to live on GPU too.
- **Tiled / shared-memory kernels.** The wgpu shader is naive — one thread per output cell, no tiling. Tuned kernels would get more out of the hardware.
- **f16 / bfloat16.** f32 only for the v0.7 scaffold.
- **Multi-GPU.** Single device.

## Files

- `src/lib.rs` — `ComputeBackend` trait, `Matrix` type, `pick_backend`
- `src/cpu.rs` — `CpuBackend` (always available)
- `src/wgpu_backend.rs` — `WgpuBackend` (feature `wgpu`)
- `shaders/matmul.wgsl` — naive matmul compute kernel
- `examples/bench_matmul.rs` — CPU vs GPU bench harness
- `tests/integration.rs` — (none yet — unit tests in modules)

## ROCm / CUDA path (future)

For users on supported hardware (gfx900+ AMD, NVIDIA), the trait is ready for:

- **HIP / rocBLAS** via `hip-sys` + `rocblas-sys` — requires ROCm 5.x+ install
- **CUDA / cuBLAS** via `cust` + `cublas` — requires CUDA Toolkit
- **Apple MPS** via the `metal` crate — macOS-only

These would add ~2-10× over wgpu on appropriate hardware. None are in v0.7 because:

1. **Polaris (the user's hardware) doesn't get them** — wgpu is the right choice for this target
2. **Each requires a SDK install** that's risky on user machines (the "Ollama gets fussy" experience)
3. **Adding them is mechanical** once a real need on supported hardware appears
