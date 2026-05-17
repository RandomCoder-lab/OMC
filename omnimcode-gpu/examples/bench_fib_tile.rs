//! v0.8.3 Substrate-GPU formulation sweep.
//!
//! Three families of variants vs the conventional 16×16 linear-K reference:
//!
//! 1. **Square Fibonacci tiles**: 8×8 (1 wavefront, exact), 13×13, 21×21
//! 2. **Anisotropic Fibonacci tiles**: 8×32, 32×8 (Fib short dim,
//!    full-wavefront occupancy via the long dim)
//! 3. **Fibonacci K-stride**: 16×16 tile but inner K accumulation walks
//!    Fibonacci-sized chunks (1, 1, 2, 3, 5, 8, 13, 21, ...) instead of
//!    linear K. Substrate-shaped reduction order on the same hardware tile.
//!
//! Each configuration runs the same matmul at several sizes. Per-row:
//! warmup (1) + 3 timed iterations averaged. Parity is asserted against
//! the CPU reference (max abs diff).
//!
//! The goal: figure out which (if any) substrate-shaped GPU formulation
//! beats the conventional 16×16 linear-K on the user's AMD RX 580 / Vulkan.
//!
//! Run:
//!     cargo run --release -p omnimcode-gpu --features wgpu --example bench_fib_tile

use std::time::Instant;

use omnimcode_gpu::{ComputeBackend, Matrix, cpu::CpuBackend};
use omnimcode_gpu::wgpu_backend::{WgpuBackend, MatmulKernel};

fn deterministic_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
    let mut s = seed;
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        data.push(((s >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    Matrix::new(rows, cols, data)
}

fn time_matmul(
    backend: &dyn ComputeBackend, m: usize, k: usize, n: usize,
    warmup: usize, iters: usize,
) -> (f64, Matrix) {
    let a = deterministic_matrix(m, k, 42);
    let b = deterministic_matrix(k, n, 99);
    let mut last = Matrix::zeros(m, n);
    for _ in 0..warmup {
        last = backend.matmul(&a, &b).expect("matmul");
    }
    let start = Instant::now();
    for _ in 0..iters {
        last = backend.matmul(&a, &b).expect("matmul");
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    (ms, last)
}

struct Variant {
    label: String,
    backend: WgpuBackend,
}

fn try_variant(tx: u32, ty: u32, kernel: MatmulKernel, label: &str) -> Option<Variant> {
    match WgpuBackend::with_config(tx, ty, kernel) {
        Ok(b) => {
            eprintln!("{:<25} OK   ({}×{}, {:?})", label, tx, ty, kernel);
            Some(Variant { label: label.to_string(), backend: b })
        }
        Err(e) => {
            eprintln!("{:<25} SKIP ({})", label, e);
            None
        }
    }
}

fn main() {
    let cpu = CpuBackend;

    eprintln!("== variant initialization ==");
    let variants: Vec<Variant> = [
        // Square Fibonacci tiles
        try_variant( 8,  8, MatmulKernel::Linear,     " 8x8  linear-K  (1WF)"),
        try_variant(13, 13, MatmulKernel::Linear,     "13x13 linear-K  (3WF)"),
        try_variant(16, 16, MatmulKernel::Linear,     "16x16 linear-K  REF  "),
        try_variant(21, 21, MatmulKernel::Linear,     "21x21 linear-K  (7WF)"),
        // Anisotropic Fibonacci tiles (long dim picks 16/32 for cache-line fit)
        try_variant( 8, 32, MatmulKernel::Linear,     " 8x32 linear-K  aniso"),
        try_variant(32,  8, MatmulKernel::Linear,     "32x8  linear-K  aniso"),
        try_variant( 8, 16, MatmulKernel::Linear,     " 8x16 linear-K  aniso"),
        // Substrate-shaped reduction order, conventional tile
        try_variant(16, 16, MatmulKernel::FibKStride, "16x16 Fib-K-stride   "),
        // Larger substrate-K stride at larger tile
        try_variant( 8,  8, MatmulKernel::FibKStride, " 8x8  Fib-K-stride   "),
    ].into_iter().flatten().collect();
    eprintln!();

    if variants.is_empty() {
        eprintln!("no wgpu variants initialized — exit");
        std::process::exit(1);
    }

    let sizes: &[(usize, usize, usize)] = &[
        (256,  256,  256),
        (512,  512,  512),
        (1024, 1024, 1024),
    ];

    println!("{:>16}  {:<25} {:>10} {:>13}  parity", "size", "variant", "ms", "GFLOPS");
    println!("{}", "-".repeat(82));

    let mut wins: Vec<(String, String, f64)> = Vec::new();
    for &(m, k, n) in sizes {
        let (cpu_ms, cpu_out) = time_matmul(&cpu, m, k, n, 1, 2);
        let cpu_gflops = (2.0 * m as f64 * k as f64 * n as f64) / (cpu_ms / 1000.0) / 1e9;
        let label = format!("{}x{}x{}", m, k, n);
        println!("{:>16}  {:<25} {:>10.3} {:>13.3}  (baseline)",
                 label, "cpu reference", cpu_ms, cpu_gflops);

        let mut best_ms = f64::INFINITY;
        let mut best_label = String::new();
        for v in &variants {
            let (ms, gpu_out) = time_matmul(&v.backend, m, k, n, 1, 5);
            let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (ms / 1000.0) / 1e9;
            let diff = cpu_out.max_abs_diff(&gpu_out);
            let parity = if diff < 1e-2 { "OK".to_string() }
                         else { format!("diff={:.2e}", diff) };
            println!("{:>16}  {:<25} {:>10.3} {:>13.3}  {}",
                     "", v.label, ms, gflops, parity);
            if ms < best_ms { best_ms = ms; best_label = v.label.clone(); }
        }
        println!();
        wins.push((label, best_label, best_ms));
    }

    println!("== headline: winning variant per size ==");
    for (size, variant, ms) in &wins {
        println!("  {:>16}  →  {}  @ {:.3} ms", size, variant, ms);
    }
}
