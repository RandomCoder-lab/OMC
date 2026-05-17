//! Pluggable accelerator hooks for hot tape ops.
//!
//! `omnimcode-core` is the bottom of the dependency stack — `omnimcode-gpu`
//! depends on -core, not the other way around. To route `tape_matmul`
//! through a GPU backend we need a hook that the higher-level binary
//! (omnimcode-cli, omnimcode-mcp, ...) can register at startup. This
//! module provides exactly that: a `OnceLock` global that holds an
//! optional matmul implementation, and a thin call-site wrapper that
//! invokes it when set and falls back to the in-core CPU loop otherwise.
//!
//! The hook signature uses raw `(m, k, n, &[f64], &[f64])` rather than
//! `TapeMat` so callers don't need to import any core-internal types.
//! Returning `None` means "decline this call, fall back to CPU" — used
//! to keep small matmuls on the CPU below the GPU crossover.
//!
//! See `omnimcode-cli/src/main.rs` for the wgpu-backed registration.

use std::sync::OnceLock;

/// A matmul accelerator. Receives `(m, k, n, a_row_major, b_row_major)`,
/// returns `Some(Ok(c_row_major))` to commit to handling the call,
/// `Some(Err(_))` to surface a backend error, or `None` to decline and
/// let the CPU path run.
pub type MatmulAccelerator = Box<
    dyn Fn(usize, usize, usize, &[f64], &[f64]) -> Option<Result<Vec<f64>, String>>
        + Send + Sync,
>;

static MATMUL_ACCELERATOR: OnceLock<MatmulAccelerator> = OnceLock::new();

/// Register a matmul accelerator. Idempotent — second call is a no-op,
/// matching `OnceLock::set` semantics. Call once during binary startup.
pub fn register_matmul_accelerator(f: MatmulAccelerator) -> Result<(), &'static str> {
    MATMUL_ACCELERATOR.set(f).map_err(|_| "matmul accelerator already registered")
}

/// Internal — used by `interpreter::tape_matmul`. Returns
/// `Some(Result<Vec<f64>, String>)` when the accelerator committed,
/// `None` when no accelerator is registered OR the registered one
/// declined this particular call (e.g. shape below GPU crossover).
pub(crate) fn try_accelerated_matmul(
    m: usize, k: usize, n: usize, a: &[f64], b: &[f64],
) -> Option<Result<Vec<f64>, String>> {
    MATMUL_ACCELERATOR.get().and_then(|f| f(m, k, n, a, b))
}
