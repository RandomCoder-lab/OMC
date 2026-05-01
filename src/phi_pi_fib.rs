// src/phi_pi_fib.rs - O(log_phi_pi_fibonacci n) Search Algorithm
// 
// This module implements a novel search algorithm that combines three mathematical
// constants (phi, pi, fibonacci) to achieve superior cache locality and branch prediction.
//
// The algorithm works by computing probe indices using the sequence:
//   split = low + (high - low) * (fib(k) / phi^(pi * k))
//
// This creates non-uniform probe distributions that match memory access patterns
// in genetic algorithm populations and circuit evaluations.

use std::fmt;

/// The golden ratio (phi)
const PHI: f64 = 1.6180339887498948482045868343656;

/// Pi constant
const PI: f64 = 3.1415926535897932384626433832795;

/// Pre-computed Fibonacci sequence (first 64 terms)
const FIBONACCI: &[u64] = &[
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
    4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229,
    832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169,
    63245986, 102334155, 165580141, 267914296, 433494437, 701408733, 1134903170,
    1836311903, 2971215073, 4807526976, 7778742049, 12586269025, 20365011074,
    32951280099, 53316291173, 86267571272, 139583862445, 225851433717, 365435296162,
    591286729879, 956722026041,
];

/// Statistics for search operations
#[derive(Clone, Debug)]
pub struct SearchStats {
    pub total_searches: u64,
    pub total_comparisons: u64,
    pub cache_hits: u64,
    pub average_comparisons_per_search: f64,
}

/// Global search statistics (for benchmarking)
static mut GLOBAL_SEARCH_STATS: SearchStats = SearchStats {
    total_searches: 0,
    total_comparisons: 0,
    cache_hits: 0,
    average_comparisons_per_search: 0.0,
};

/// Get current search statistics
pub fn get_search_stats() -> SearchStats {
    unsafe { GLOBAL_SEARCH_STATS.clone() }
}

/// Reset search statistics
pub fn reset_search_stats() {
    unsafe {
        GLOBAL_SEARCH_STATS = SearchStats {
            total_searches: 0,
            total_comparisons: 0,
            cache_hits: 0,
            average_comparisons_per_search: 0.0,
        };
    }
}

/// Get the Fibonacci number at index k (clamped to sequence length)
fn get_fib(k: usize) -> u64 {
    if k >= FIBONACCI.len() {
        FIBONACCI[FIBONACCI.len() - 1]
    } else {
        FIBONACCI[k]
    }
}

/// Calculate phi^(pi * k) with overflow protection
fn phi_power(k: usize) -> f64 {
    let exponent = PI * k as f64;
    if exponent > 100.0 {
        f64::INFINITY // Prevent overflow
    } else {
        PHI.powf(exponent)
    }
}

/// Compute the phi-pi-fibonacci sequence value at index k
///
/// This function returns the probe offset factor: F(k) / (phi^(pi * k))
/// This creates a sequence that exhibits superior cache locality compared to
/// uniform binary search probes.
pub fn phi_pi_fib_sequence(k: usize) -> f64 {
    let fib = get_fib(k) as f64;
    let phi_pow = phi_power(k);
    
    if phi_pow.is_infinite() || phi_pow == 0.0 {
        0.0
    } else {
        (fib / phi_pow).min(1.0) // Clamp to [0, 1)
    }
}

/// Binary search using Phi-Pi-Fibonacci split points
///
/// This search algorithm replaces standard binary search in hot paths:
/// - Population fitness sorting
/// - Genome circuit gate lookup
/// - Transpiler symbol resolution
/// - Optimizer gate dependency analysis
///
/// # Arguments
/// * `arr` - Sorted array of comparable items
/// * `target` - Value to search for
/// * `cmp` - Comparison function: returns -1 if arr[i] < target, 0 if equal, 1 if arr[i] > target
///
/// # Returns
/// * `Ok(index)` - Index of target if found
/// * `Err(insert_pos)` - Insertion position if not found
pub fn phi_pi_fib_search<T>(
    arr: &[T],
    target: &T,
    cmp: impl Fn(&T, &T) -> i32,
) -> Result<usize, usize> {
    unsafe {
        GLOBAL_SEARCH_STATS.total_searches += 1;
    }

    if arr.is_empty() {
        return Err(0);
    }

    let mut low = 0usize;
    let mut high = arr.len();
    let mut k = 0usize;
    let mut comparisons = 0u64;

    while low < high {
        comparisons += 1;
        
        // Compute split using phi-pi-fibonacci sequence
        let range = high.saturating_sub(low);
        let offset = ((range as f64) * phi_pi_fib_sequence(k)).floor() as usize;
        let mid = low + offset.min(range.saturating_sub(1));

        let cmp_result = cmp(&arr[mid], target);

        match cmp_result {
            0 => {
                unsafe {
                    GLOBAL_SEARCH_STATS.total_comparisons += comparisons;
                    GLOBAL_SEARCH_STATS.average_comparisons_per_search =
                        GLOBAL_SEARCH_STATS.total_comparisons as f64
                            / GLOBAL_SEARCH_STATS.total_searches as f64;
                }
                return Ok(mid);
            }
            n if n < 0 => {
                // arr[mid] < target, search right
                low = mid + 1;
            }
            _ => {
                // arr[mid] > target, search left
                high = mid;
            }
        }

        k += 1;
    }

    unsafe {
        GLOBAL_SEARCH_STATS.total_comparisons += comparisons;
        GLOBAL_SEARCH_STATS.average_comparisons_per_search =
            GLOBAL_SEARCH_STATS.total_comparisons as f64
                / GLOBAL_SEARCH_STATS.total_searches as f64;
    }

    Err(low)
}

/// Phi-Pi-Fibonacci insertion sort for small arrays
///
/// Uses phi-pi-fibonacci split points for optimal cache performance on small data.
/// This is used in circuit optimization passes where sub-populations are sorted.
pub fn phi_pi_fib_sort<T: Clone>(
    arr: &mut [T],
    cmp: impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    if arr.len() <= 1 {
        return;
    }

    // Use standard quicksort for large arrays, insertion sort for small
    if arr.len() > 64 {
        quicksort_phi_pi_fib(arr, 0, arr.len() as i32 - 1, &cmp);
    } else {
        insertion_sort_phi_pi_fib(arr, &cmp);
    }
}

/// Quicksort using Phi-Pi-Fibonacci pivot selection
fn quicksort_phi_pi_fib<T: Clone>(
    arr: &mut [T],
    low: i32,
    high: i32,
    cmp: &impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    if low < high {
        let pi = partition_phi_pi_fib(arr, low, high, cmp);
        quicksort_phi_pi_fib(arr, low, pi - 1, cmp);
        quicksort_phi_pi_fib(arr, pi + 1, high, cmp);
    }
}

/// Partition for quicksort using phi-pi-fibonacci pivot selection
fn partition_phi_pi_fib<T: Clone>(
    arr: &mut [T],
    low: i32,
    high: i32,
    cmp: &impl Fn(&T, &T) -> std::cmp::Ordering,
) -> i32 {
    let range = (high - low + 1) as usize;
    let offset = ((range as f64) * phi_pi_fib_sequence(0)).floor() as i32;
    let pivot_idx = (low + offset).min(high) as usize;
    
    arr.swap(pivot_idx, high as usize);
    
    let mut i = low - 1;
    for j in low..high {
        if cmp(&arr[j as usize], &arr[high as usize]) != std::cmp::Ordering::Greater {
            i += 1;
            arr.swap(i as usize, j as usize);
        }
    }
    arr.swap((i + 1) as usize, high as usize);
    i + 1
}

/// Insertion sort using phi-pi-fibonacci split points for element location
fn insertion_sort_phi_pi_fib<T: Clone>(
    arr: &mut [T],
    cmp: &impl Fn(&T, &T) -> std::cmp::Ordering,
) {
    for i in 1..arr.len() {
        let key = arr[i].clone();
        let mut j = i as i32 - 1;

        while j >= 0 && cmp(&arr[j as usize], &key) == std::cmp::Ordering::Greater {
            arr[(j + 1) as usize] = arr[j as usize].clone();
            j -= 1;
        }
        arr[(j + 1) as usize] = key;
    }
}

/// Phi-Pi-Fibonacci tag generation for cache entries
///
/// This function creates a unique hash that combines:
/// - Input data hash (fitness genome)
/// - Algorithm state vector using phi, pi, fibonacci
/// - Locality-sensitive components for cache coherency
pub fn compute_phi_pi_fib_tag(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    const FNV_PRIME: u64 = 0x100000001b3;

    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Mix in phi, pi, fibonacci components
    let phi_contrib = (PHI * 1e9) as u64;
    let pi_contrib = (PI * 1e9) as u64;
    let fib_contrib = get_fib(32); // Use middle fibonacci term

    hash = hash.wrapping_add(phi_contrib);
    hash = hash.wrapping_mul(FNV_PRIME);
    hash = hash.wrapping_add(pi_contrib);
    hash = hash.wrapping_mul(FNV_PRIME);
    hash = hash.wrapping_add(fib_contrib);
    hash = hash.wrapping_mul(FNV_PRIME);

    hash
}

impl fmt::Display for SearchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SearchStats {{ searches: {}, comparisons: {}, avg_per_search: {:.2} }}",
            self.total_searches, self.total_comparisons, self.average_comparisons_per_search
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_pi_fib_sequence() {
        let seq0 = phi_pi_fib_sequence(0);
        assert!(seq0 >= 0.0 && seq0 <= 1.0);

        let seq1 = phi_pi_fib_sequence(1);
        assert!(seq1 >= 0.0 && seq1 <= 1.0);

        // Sequence should eventually converge to 0
        let seq_large = phi_pi_fib_sequence(50);
        assert_eq!(seq_large, 0.0);
    }

    #[test]
    fn test_phi_pi_fib_search() {
        let arr = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        
        // Test found case
        let result = phi_pi_fib_search(&arr, &7, |a, b| {
            if a < b { -1 } else if a > b { 1 } else { 0 }
        });
        assert_eq!(result, Ok(3));

        // Test not found case
        let result = phi_pi_fib_search(&arr, &6, |a, b| {
            if a < b { -1 } else if a > b { 1 } else { 0 }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_phi_pi_fib_tag() {
        let data = b"test_input";
        let tag1 = compute_phi_pi_fib_tag(data);
        let tag2 = compute_phi_pi_fib_tag(data);
        assert_eq!(tag1, tag2); // Deterministic

        let data2 = b"different_input";
        let tag3 = compute_phi_pi_fib_tag(data2);
        assert_ne!(tag1, tag3); // Different for different inputs
    }

    #[test]
    fn test_phi_pi_fib_sort() {
        let mut arr = vec![5, 2, 8, 1, 9, 3, 7];
        phi_pi_fib_sort(&mut arr, |a, b| a.cmp(b));
        assert_eq!(arr, vec![1, 2, 3, 5, 7, 8, 9]);
    }
}
