//! GPU compute scaffold for Prometheus.
//!
//! Provides a `ComputeBackend` trait with multiple implementations:
//!
//! - **`CpuBackend`** (always available) — pure-Rust f32 matmul,
//!   used as the parity baseline and the fallback when no GPU is
//!   available. Single-threaded by design — this isn't BLAS, it's
//!   the "ground truth" output for comparing GPU results against.
//!
//! - **`WgpuBackend`** (feature `wgpu`) — Vulkan / Metal / DX12 /
//!   OpenGL compute via the `wgpu` crate. Cross-vendor; works on
//!   AMD Polaris (RX 580) via Vulkan without any ROCm install.
//!   Trades raw FLOPS for portability and stability.
//!
//! - **`RocmBackend`** (feature `rocm`, not yet implemented) — AMD
//!   HIP + rocBLAS. Best performance on supported AMD GPUs;
//!   Polaris (gfx803) requires unofficial ROCm builds and carries
//!   crash risk. Stub only.
//!
//! - **`CudaBackend`** (feature `cuda`, not yet implemented) —
//!   NVIDIA cuBLAS. Highest performance on NVIDIA hardware.
//!   Stub only.
//!
//! The trait + dispatch pattern means Prometheus can route its
//! `tape_matmul` (and other hot ops) through whichever backend the
//! user opts into at build time, without changing OMC-side code.
//!
//! ## Scope
//!
//! v0.7 is a SCAFFOLD: one operation (matmul) implemented end-to-end
//! on CPU + wgpu, with a benchmark harness that lets us measure GPU
//! speedup honestly. Real adoption (routing Prometheus's tape ops
//! through this layer) is the v0.8 candidate.

use std::fmt;

pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

/// Errors from a backend operation.
#[derive(Debug)]
pub enum BackendError {
    /// Shape mismatch (caller bug).
    ShapeMismatch { lhs: (usize, usize), rhs: (usize, usize) },
    /// Backend wasn't built into this binary (e.g. `wgpu` feature off).
    Unavailable(&'static str),
    /// Implementation-specific failure (driver, OOM, kernel error).
    Backend(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::ShapeMismatch { lhs, rhs } => write!(
                f, "shape mismatch: lhs {:?} vs rhs {:?}", lhs, rhs
            ),
            BackendError::Unavailable(name) => write!(
                f, "backend '{}' is not built into this binary; \
                    rebuild with --features {}", name, name
            ),
            BackendError::Backend(msg) => write!(f, "backend error: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// A row-major dense f32 matrix in host memory. The boundary type
/// between OMC's `Value` representation and the backend's native
/// layout (GPU buffer, ndarray, BLAS buffer, etc.).
///
/// Kept intentionally minimal: just `rows × cols` of `f32`. Sparse
/// matrices, integer types, and higher-dimensional tensors are out
/// of scope for the v0.7 scaffold.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Row-major: `data[r * cols + c]`.
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols,
                   "data len {} != rows*cols {} ({}x{})",
                   data.len(), rows * cols, rows, cols);
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }

    /// L∞ (max-elementwise) distance between two matrices of the same
    /// shape. Useful for asserting GPU results match CPU within
    /// floating-point rounding.
    pub fn max_abs_diff(&self, other: &Self) -> f32 {
        assert_eq!(self.shape(), other.shape(),
                   "max_abs_diff: shapes differ");
        self.data.iter().zip(other.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }
}

/// The compute backend trait — what every supported execution path
/// (CPU, wgpu, ROCm, CUDA) implements. v0.7 covers one operation:
/// matrix multiplication. The trait is open for extension as more
/// Prometheus tape ops migrate to GPU.
pub trait ComputeBackend: Send + Sync {
    /// Backend identifier ("cpu" / "wgpu" / "rocm" / "cuda"). Used in
    /// error messages and benchmark labels.
    fn name(&self) -> &'static str;

    /// Compute `c = a @ b`. `a` is `[m, k]`, `b` is `[k, n]`, `c` is
    /// `[m, n]`. Returns ShapeMismatch on a-cols != b-rows.
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, BackendError>;
}

/// Pick the best available backend at runtime, honoring the
/// `OMC_GPU_BACKEND` env var as an explicit override (`cpu` | `wgpu`).
/// Falls back to CPU if the requested backend isn't built in.
pub fn pick_backend() -> Box<dyn ComputeBackend> {
    let requested = std::env::var("OMC_GPU_BACKEND").ok();
    let want = requested.as_deref().unwrap_or(default_backend_name());
    match want {
        "cpu" => Box::new(cpu::CpuBackend),
        #[cfg(feature = "wgpu")]
        "wgpu" => match wgpu_backend::WgpuBackend::new() {
            Ok(b) => Box::new(b),
            Err(e) => {
                eprintln!("omc-gpu: wgpu init failed ({}); falling back to CPU", e);
                Box::new(cpu::CpuBackend)
            }
        },
        #[cfg(not(feature = "wgpu"))]
        "wgpu" => {
            eprintln!("omc-gpu: wgpu feature not built in; falling back to CPU");
            Box::new(cpu::CpuBackend)
        }
        other => {
            eprintln!("omc-gpu: unknown backend '{}'; falling back to CPU", other);
            Box::new(cpu::CpuBackend)
        }
    }
}

const fn default_backend_name() -> &'static str {
    // wgpu when available, CPU otherwise. The wgpu feature being on
    // at build time means the binary CAN talk to a GPU; whether one
    // is present at runtime is sorted out in pick_backend.
    #[cfg(feature = "wgpu")]
    { "wgpu" }
    #[cfg(not(feature = "wgpu"))]
    { "cpu" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_new_validates_shape() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.shape(), (2, 3));
    }

    #[test]
    #[should_panic]
    fn matrix_new_rejects_wrong_data_len() {
        let _ = Matrix::new(2, 3, vec![1.0, 2.0]);
    }

    #[test]
    fn max_abs_diff_zero_for_identical() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = a.clone();
        assert_eq!(a.max_abs_diff(&b), 0.0);
    }

    #[test]
    fn max_abs_diff_picks_largest_element_diff() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![1.1, 2.0, 3.0, 5.0]);
        let diff = a.max_abs_diff(&b);
        assert!((diff - 1.0).abs() < 1e-6, "max diff is 1.0 (the 5.0 vs 4.0 cell)");
    }

    #[test]
    fn pick_backend_returns_cpu_when_env_forces() {
        std::env::set_var("OMC_GPU_BACKEND", "cpu");
        let b = pick_backend();
        assert_eq!(b.name(), "cpu");
        std::env::remove_var("OMC_GPU_BACKEND");
    }
}
