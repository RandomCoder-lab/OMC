//! CPU backend — naive triple-loop f32 matmul.
//!
//! Not optimized. Its job is to be the GROUND TRUTH that GPU outputs
//! get compared against in tests + benchmarks. Real production CPU
//! matmul would use `ndarray-blas` or `matrixmultiply`; we don't pull
//! those in because the scaffold's compare-against-CPU semantics need
//! a deterministic, simple-to-reason-about reference.

use crate::{BackendError, ComputeBackend, Matrix};

pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str { "cpu" }

    fn matmul(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, BackendError> {
        if a.cols != b.rows {
            return Err(BackendError::ShapeMismatch {
                lhs: a.shape(), rhs: b.shape(),
            });
        }
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let mut c = vec![0.0_f32; m * n];
        // Loop order ikj (rather than ijk) keeps b indexing contiguous
        // in the inner loop — better cache behavior for row-major.
        for i in 0..m {
            for kk in 0..k {
                let aik = a.data[i * k + kk];
                for j in 0..n {
                    c[i * n + j] += aik * b.data[kk * n + j];
                }
            }
        }
        Ok(Matrix::new(m, n, c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_matmul_identity() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let id = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let out = CpuBackend.matmul(&a, &id).unwrap();
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_matmul_basic_2x3_3x2() {
        // a = [[1, 2, 3], [4, 5, 6]]
        // b = [[7, 8], [9, 10], [11, 12]]
        // c = [[58, 64], [139, 154]]
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = CpuBackend.matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_matmul_shape_mismatch_errors() {
        let a = Matrix::new(2, 3, vec![0.0; 6]);
        let b = Matrix::new(4, 2, vec![0.0; 8]);
        let res = CpuBackend.matmul(&a, &b);
        assert!(matches!(res, Err(BackendError::ShapeMismatch { .. })));
    }
}
