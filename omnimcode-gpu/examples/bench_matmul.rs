//! GPU vs CPU matmul benchmark.
//!
//! Times matmul at several sizes on the CPU backend and (if the
//! `wgpu` feature is built in AND a GPU is available) the wgpu
//! backend. Reports wall-clock per-op + speedup ratio.
//!
//! Run:
//!     cargo run --release -p omnimcode-gpu --features wgpu --example bench_matmul
//!
//! Override the backend via OMC_GPU_BACKEND=cpu|wgpu.

use std::time::Instant;

use omnimcode_gpu::{ComputeBackend, Matrix, cpu::CpuBackend};

fn deterministic_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
    // Tiny LCG just so the data isn't all zeros — substance doesn't
    // matter, just shape + non-trivial values.
    let mut s = seed;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push(((s >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    Matrix::new(rows, cols, data)
}

fn time_matmul(backend: &dyn ComputeBackend, m: usize, k: usize, n: usize,
               warmup: usize, iters: usize) -> (f64, Matrix) {
    let a = deterministic_matrix(m, k, 42);
    let b = deterministic_matrix(k, n, 99);
    // Warmup — first call always pays kernel-compilation / buffer-alloc cost.
    let mut last = Matrix::zeros(m, n);
    for _ in 0..warmup {
        last = backend.matmul(&a, &b).expect("matmul");
    }
    let start = Instant::now();
    for _ in 0..iters {
        last = backend.matmul(&a, &b).expect("matmul");
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    (elapsed_ms, last)
}

fn main() {
    let cpu = CpuBackend;
    #[cfg(feature = "wgpu")]
    let wgpu = match omnimcode_gpu::wgpu_backend::WgpuBackend::new() {
        Ok(b) => {
            eprintln!("wgpu adapter: {}\n", b.describe_adapter());
            Some(b)
        }
        Err(e) => {
            eprintln!("wgpu unavailable on this machine: {} (CPU-only run)\n", e);
            None
        }
    };

    let sizes: &[(usize, usize, usize)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];

    println!("{:>20} {:>12} {:>12} {:>10}  parity",
             "size (m x k x n)", "cpu ms", "wgpu ms", "speedup");
    println!("{}", "-".repeat(75));
    for &(m, k, n) in sizes {
        let (cpu_ms, cpu_out) = time_matmul(&cpu, m, k, n, 1, 3);
        let label = format!("{}x{}x{}", m, k, n);

        #[cfg(feature = "wgpu")]
        {
            if let Some(ref g) = wgpu {
                let (gpu_ms, gpu_out) = time_matmul(g, m, k, n, 1, 3);
                let speedup = cpu_ms / gpu_ms;
                // Parity check — GPU output should match CPU output
                // within f32 rounding for these well-conditioned inputs.
                let diff = cpu_out.max_abs_diff(&gpu_out);
                let parity = if diff < 1e-3 { "OK".to_string() }
                             else { format!("diff={:.2e}", diff) };
                println!("{:>20} {:>12.3} {:>12.3} {:>9.2}x  {}",
                         label, cpu_ms, gpu_ms, speedup, parity);
                continue;
            }
        }
        let _ = cpu_out;
        println!("{:>20} {:>12.3} {:>12} {:>10}  -",
                 label, cpu_ms, "—", "—");
    }
    println!();
    println!("CPU backend: naive triple-loop f32, single-threaded.");
    println!("GPU backend: wgpu Vulkan/Metal/DX12, 16x16 workgroup, no tiling.");
    println!();
    println!("Honest framing: the CPU baseline is naive — a tuned BLAS");
    println!("would close most of the gap. The wgpu kernel is also untiled.");
    println!("The point is to verify the scaffold works end-to-end and to");
    println!("measure the CPU/GPU crossover point on this machine, not to");
    println!("claim cuBLAS-class performance.");
}
