// src/phi_pi_fib.rs - Fibonacci-Based Search Algorithms
//
// Two algorithms live here, exposed side-by-side so OMC code can pick
// (or benchmark) at runtime:
//
//   fibonacci_search       — Fibonacci-step search. Standard textbook
//                            algorithm. Comparison count tracks
//                            log_phi(n) ≈ 1.44 * log_2(n).
//
//   phi_pi_fib_search_v2   — The F(k) / φ^(π·k) split-point formula
//                            from PHI_PI_FIB_ALGORITHM.md. Probes at
//                            non-uniform fractions of the live range
//                            for early iterations, falls back to
//                            binary search when the offset would
//                            round to zero. Aimed at the theoretical
//                            log_φ_π_fibonacci(n) = ln(n) / ln(φ^π).
//
// binary_search is also exposed as a fair baseline. All three share
// global comparison counters via get_search_stats() / reset_search_stats().
//
// Whether v2 actually wins on compare count is an empirical question
// — see experiment_8_search_bench.omc for the head-to-head.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Import PHI from value.rs to maintain single source of truth
const PHI: f64 = 1.6180339887498948482045868343656;
const PI: f64 = std::f64::consts::PI;

/// Pre-computed Fibonacci sequence (first 40 terms fit in u64)
const FIBONACCI: &[u64] = &[
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
    2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418,
    317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465,
    14930352, 24157817, 39088169, 63245986,
];

/// Thread-safe statistics for search operations
pub struct SearchStats {
    pub total_searches: u64,
    pub total_comparisons: u64,
}

static TOTAL_SEARCHES: AtomicU64 = AtomicU64::new(0);
static TOTAL_COMPARISONS: AtomicU64 = AtomicU64::new(0);

/// Get current search statistics (thread-safe)
pub fn get_search_stats() -> SearchStats {
    SearchStats {
        total_searches: TOTAL_SEARCHES.load(Ordering::Relaxed),
        total_comparisons: TOTAL_COMPARISONS.load(Ordering::Relaxed),
    }
}

/// Reset search statistics
pub fn reset_search_stats() {
    TOTAL_SEARCHES.store(0, Ordering::Relaxed);
    TOTAL_COMPARISONS.store(0, Ordering::Relaxed);
}

/// Get Fibonacci number at index (clamped to sequence length)
fn get_fib(idx: usize) -> u64 {
    if idx >= FIBONACCI.len() {
        FIBONACCI[FIBONACCI.len() - 1]
    } else {
        FIBONACCI[idx]
    }
}

/// Find the Fibonacci index that bounds the array size
fn find_fib_index(n: usize) -> usize {
    for (i, &f) in FIBONACCI.iter().enumerate() {
        if f >= n as u64 {
            return i;
        }
    }
    FIBONACCI.len() - 1
}

/// Fibonacci-based search on a sorted array.
///
/// This is an alternative to binary search that uses Fibonacci numbers to
/// determine split points. In theory, it can be slightly more cache-efficient
/// for certain array sizes that match Fibonacci growth patterns.
///
/// In practice: Comparable performance to binary search, sometimes faster,
/// sometimes slower. Not worth using unless you have measured evidence it
/// helps on your specific workload.
///
/// # Arguments
/// * `arr` - Sorted array of comparable items
/// * `target` - Value to search for
/// * `cmp` - Comparison function: -1 if arr[i] < target, 0 if equal, 1 if arr[i] > target
///
/// # Returns
/// * `Ok(index)` - Index of target if found
/// * `Err(insert_pos)` - Insertion position if not found
pub fn fibonacci_search<T>(
    arr: &[T],
    target: &T,
    cmp: impl Fn(&T, &T) -> i32,
) -> Result<usize, usize> {
    if arr.is_empty() {
        return Err(0);
    }

    TOTAL_SEARCHES.fetch_add(1, Ordering::Relaxed);

    let mut fib_idx = find_fib_index(arr.len());
    let mut offset = 0usize;
    let mut comparisons = 0u64;

    // Standard Fibonacci search algorithm
    while fib_idx > 0 {
        comparisons += 1;

        let fib_val = get_fib(fib_idx) as usize;
        let mid = (offset + fib_val.min(arr.len() - offset - 1)).min(arr.len() - 1);

        let cmp_result = cmp(&arr[mid], target);

        match cmp_result {
            0 => {
                TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
                return Ok(mid);
            }
            n if n < 0 => {
                // arr[mid] < target, search right
                offset = mid + 1;
                fib_idx = fib_idx.saturating_sub(2);
            }
            _ => {
                // arr[mid] > target, search left
                fib_idx = fib_idx.saturating_sub(1);
            }
        }

        if offset >= arr.len() {
            break;
        }
    }

    TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
    Err(offset)
}

/// phi_pi_fib_search_v2 — F(k) / phi^(pi*k) split-point search.
///
/// Implements the algorithm described in `PHI_PI_FIB_ALGORITHM.md`:
/// at iteration k the probe offset (relative to the live range) is
/// `offset = (high - low) * F(k) / phi^(pi*k)` where F(k) is the
/// k-th Fibonacci number. F(k) grows like phi^k, so the ratio
/// `F(k)/phi^(pi*k) ~= phi^((1-pi)*k)` decays rapidly. The early
/// probes cluster near `low` at fractions 0.22, 0.049, 0.022,
/// 0.0071, ... of the live range.
///
/// When the offset would round to zero (range too small for the
/// current k), the search falls back to standard binary search on
/// the remaining range. This guarantees termination and bounds the
/// worst case by `binary_search` performance.
///
/// Whether the early iterations save enough work to beat binary
/// search overall is an empirical question — that's the point of
/// running the head-to-head benchmark in experiment 8.
pub fn phi_pi_fib_search_v2<T>(
    arr: &[T],
    target: &T,
    cmp: impl Fn(&T, &T) -> i32,
) -> Result<usize, usize> {
    if arr.is_empty() {
        return Err(0);
    }
    TOTAL_SEARCHES.fetch_add(1, Ordering::Relaxed);

    let mut low: usize = 0;
    let mut high: usize = arr.len();
    let mut k: usize = 1;
    let mut comparisons: u64 = 0;

    // Phase 1: phi-pi-fib probe-offset iterations.
    // Stop once the offset rounds to zero — then phase 2 binary-searches.
    while low + 1 < high {
        let range = (high - low) as f64;
        let fib_k = if k < FIBONACCI.len() {
            FIBONACCI[k] as f64
        } else {
            FIBONACCI[FIBONACCI.len() - 1] as f64
        };
        let denom = (PI * (k as f64) * PHI.ln()).exp(); // = φ^(π·k)
        let frac = (fib_k / denom).clamp(0.0, 0.999);
        let offset = (range * frac).round() as usize;
        if offset == 0 {
            break;
        }
        let mid = (low + offset).min(high - 1);

        comparisons += 1;
        match cmp(&arr[mid], target) {
            0 => {
                TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
                return Ok(mid);
            }
            n if n < 0 => low = mid + 1,
            _ => high = mid,
        }
        k += 1;
    }

    // Phase 2: fall through to binary search on the (smaller) live range.
    while low < high {
        comparisons += 1;
        let mid = low + (high - low) / 2;
        match cmp(&arr[mid], target) {
            0 => {
                TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
                return Ok(mid);
            }
            n if n < 0 => low = mid + 1,
            _ => high = mid,
        }
    }

    TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
    Err(low)
}

/// fibonacci_search_with_trace — same as fibonacci_search but also
/// returns the sequence of probed indices, in order. Used by
/// experiments that need to measure step-size coherence externally.
/// Counters are updated identically to fibonacci_search so combined
/// runs still report meaningful totals.
pub fn fibonacci_search_with_trace<T>(
    arr: &[T],
    target: &T,
    cmp: impl Fn(&T, &T) -> i32,
) -> (Result<usize, usize>, Vec<usize>) {
    let mut probes: Vec<usize> = Vec::new();
    if arr.is_empty() {
        return (Err(0), probes);
    }
    TOTAL_SEARCHES.fetch_add(1, Ordering::Relaxed);

    let mut fib_idx = find_fib_index(arr.len());
    let mut offset = 0usize;
    let mut comparisons = 0u64;

    while fib_idx > 0 {
        comparisons += 1;
        let fib_val = get_fib(fib_idx) as usize;
        let mid = (offset + fib_val.min(arr.len() - offset - 1)).min(arr.len() - 1);
        probes.push(mid);

        let cmp_result = cmp(&arr[mid], target);
        match cmp_result {
            0 => {
                TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
                return (Ok(mid), probes);
            }
            n if n < 0 => {
                offset = mid + 1;
                fib_idx = fib_idx.saturating_sub(2);
            }
            _ => {
                fib_idx = fib_idx.saturating_sub(1);
            }
        }

        if offset >= arr.len() {
            break;
        }
    }

    TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
    (Err(offset), probes)
}

/// Standard binary search (for comparison/benchmarking).
///
/// This is provided as a reference implementation to compare against
/// fibonacci_search on the same data.
pub fn binary_search<T>(
    arr: &[T],
    target: &T,
    cmp: impl Fn(&T, &T) -> i32,
) -> Result<usize, usize> {
    let mut low = 0usize;
    let mut high = arr.len();
    let mut comparisons = 0u64;

    TOTAL_SEARCHES.fetch_add(1, Ordering::Relaxed);

    while low < high {
        comparisons += 1;
        let mid = low + (high - low) / 2;

        let cmp_result = cmp(&arr[mid], target);

        match cmp_result {
            0 => {
                TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
                return Ok(mid);
            }
            n if n < 0 => {
                low = mid + 1;
            }
            _ => {
                high = mid;
            }
        }
    }

    TOTAL_COMPARISONS.fetch_add(comparisons, Ordering::Relaxed);
    Err(low)
}

/// log_phi_pi_fibonacci(n) — the theoretical compare-count bound for
/// the phi_pi_fib_search_v2 algorithm.
///
/// Derivation: the F(k)/phi^(pi*k) split-point formula reduces the
/// live range by a factor of ~phi^pi per iteration. Hence the
/// iteration count to converge on a target satisfies
/// `n / (phi^pi)^k = 1`, giving
/// `k = ln(n) / ln(phi^pi) = ln(n) / (pi * ln(phi))`.
///
/// Numerically: phi^pi ~= 4.534, ln(phi^pi) ~= 1.511, so
/// `log_phi_pi_fibonacci(n) ~= 0.459 * log2(n)`.
///
/// Whether the empirical compare count of phi_pi_fib_search_v2 actually
/// hits this bound depends on how often the offset rounds to zero and
/// the algorithm falls back to standard binary search; see the
/// experiment_8_search_bench.omc head-to-head.
pub fn log_phi_pi_fibonacci(n: f64) -> f64 {
    n.ln() / (PI * PHI.ln())
}

/// log_phi(n) — kept as a deprecated alias for backwards compatibility.
/// New code should use `log_phi_pi_fibonacci`. The naming change is
/// architectural: the phi-pi-fibonacci substrate is the unit of
/// measurement, not the golden ratio in isolation.
#[deprecated(note = "use log_phi_pi_fibonacci instead — phi-alone is the outdated baseline, phi-pi-fibonacci is the substrate")]
pub fn log_phi(n: f64) -> f64 {
    n.ln() / PHI.ln()
}

impl fmt::Display for SearchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.total_searches == 0 {
            return write!(f, "SearchStats {{ searches: 0, comparisons: 0 }}");
        }
        let avg = self.total_comparisons as f64 / self.total_searches as f64;
        write!(
            f,
            "SearchStats {{ searches: {}, total_comparisons: {}, avg: {:.2} }}",
            self.total_searches, self.total_comparisons, avg
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_search_found() {
        reset_search_stats();
        let arr = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

        let result = fibonacci_search(&arr, &7, |a, b| {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        });

        assert_eq!(result, Ok(3));
    }

    #[test]
    fn test_fibonacci_search_not_found() {
        reset_search_stats();
        let arr = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];

        let result = fibonacci_search(&arr, &6, |a, b| {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        });

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), 3); // Insert position between 5 and 7
    }

    #[test]
    fn test_binary_vs_fibonacci() {
        reset_search_stats();
        let arr: Vec<i32> = (0..100).collect();

        // Binary search
        let bin_result = binary_search(&arr, &50, |a, b| {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        });
        let bin_stats = get_search_stats();
        reset_search_stats();

        // Fibonacci search
        let fib_result = fibonacci_search(&arr, &50, |a, b| {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        });
        let fib_stats = get_search_stats();

        assert_eq!(bin_result, fib_result);
        // Both should find it; ratio can vary significantly depending on array size
        // Just verify both complete without panic
        assert!(bin_stats.total_comparisons > 0);
        assert!(fib_stats.total_comparisons > 0);
    }

    #[test]
    fn test_search_stats_thread_safe() {
        reset_search_stats();
        let _ = binary_search(&vec![1, 2, 3], &2, |a, b| a.cmp(b) as i32);
        let stats = get_search_stats();
        assert_eq!(stats.total_searches, 1);
        assert!(stats.total_comparisons > 0);
    }

    #[test]
    fn test_log_phi() {
        let val = log_phi(1000.0);
        assert!(val > 0.0);
        assert!(val < 20.0); // log_phi(1000) ≈ 6.8
    }
}
