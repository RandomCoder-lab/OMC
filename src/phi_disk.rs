// src/phi_disk.rs - In-Memory LRU Cache
//
// This is an in-memory LRU (Least Recently Used) cache with some phi/fibonacci-inspired
// tagging for content-addressable lookups. It does NOT persist to disk despite the name
// "Phi Disk" — that was aspirational. This is simply a cache that can accelerate
// repeated computations in evolutionary algorithms.
//
// The cache provides:
// - O(1) average lookup via HashMap
// - Simple LRU eviction policy when capacity is reached
// - Optional tag-based keying for semantic caching
// - Statistics for hit/miss tracking

use std::collections::HashMap;

/// Tag generation using FNV-1a hash mixed with a "phi-inspired" component.
/// This is just a deterministic hash; nothing magical about it.
pub fn compute_phi_pi_fib_tag(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    const FNV_PRIME: u64 = 0x100000001b3;

    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }

    // Mix in a "phi component" (just a constant, doesn't matter much)
    let phi_component = 1618033988u64; // phi * 1e9, rounded
    hash = hash.wrapping_add(phi_component);
    hash = hash.wrapping_mul(FNV_PRIME);

    hash
}

/// Cache entry: stored value + access metadata
#[derive(Clone, Debug)]
struct CacheEntry<T> {
    value: T,
    access_order: u64, // Lower = evict first
}

/// Simple in-memory LRU cache
pub struct PhiDiskCache<T: Clone> {
    entries: HashMap<u64, CacheEntry<T>>,
    max_capacity: usize,
    access_counter: u64, // Incremented on each access
    stats: CacheStats,
}

/// Cache statistics
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl<T: Clone> PhiDiskCache<T> {
    /// Create a new cache with specified capacity
    pub fn new(max_capacity: usize) -> Self {
        PhiDiskCache {
            entries: HashMap::new(),
            max_capacity: max_capacity.max(1),
            access_counter: 0,
            stats: CacheStats::default(),
        }
    }

    /// Insert or update a cache entry
    pub fn insert(&mut self, tag: u64, value: T) {
        self.access_counter = self.access_counter.wrapping_add(1);

        if self.entries.len() >= self.max_capacity && !self.entries.contains_key(&tag) {
            self.evict_lru();
        }

        self.entries.insert(
            tag,
            CacheEntry {
                value,
                access_order: self.access_counter,
            },
        );
    }

    /// Lookup a cache entry
    pub fn get(&mut self, tag: u64) -> Option<T> {
        self.access_counter = self.access_counter.wrapping_add(1);

        if let Some(entry) = self.entries.get_mut(&tag) {
            entry.access_order = self.access_counter;
            self.stats.hits += 1;
            Some(entry.value.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check if tag exists in cache
    pub fn contains(&self, tag: u64) -> bool {
        self.entries.contains_key(&tag)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    /// Clear the entire cache
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Evict the least-recently-used entry
    fn evict_lru(&mut self) {
        let lru_tag = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_order)
            .map(|(&tag, _)| tag);

        if let Some(tag) = lru_tag {
            self.entries.remove(&tag);
            self.stats.evictions += 1;
        }
    }
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CacheStats {{ hits: {}, misses: {}, hit_rate: {:.1}%, evictions: {} }}",
            self.hits,
            self.misses,
            self.hit_rate() * 100.0,
            self.evictions
        )
    }
}

// Concrete cache types for common use cases

pub type FitnessCache = PhiDiskCache<f64>;
pub type CircuitCache = PhiDiskCache<Vec<bool>>;
pub type TranspileCache = PhiDiskCache<String>;
pub type OptimizerCache = PhiDiskCache<(Vec<u8>, usize)>;

pub fn create_fitness_cache() -> FitnessCache {
    PhiDiskCache::new(10000)
}

pub fn create_circuit_cache() -> CircuitCache {
    PhiDiskCache::new(50000)
}

pub fn create_transpile_cache() -> TranspileCache {
    PhiDiskCache::new(5000)
}

pub fn create_optimizer_cache() -> OptimizerCache {
    PhiDiskCache::new(10000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_get() {
        let mut cache: PhiDiskCache<i32> = PhiDiskCache::new(10);
        let tag = compute_phi_pi_fib_tag(b"test");

        cache.insert(tag, 42);
        assert_eq!(cache.get(tag), Some(42));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache: PhiDiskCache<i32> = PhiDiskCache::new(10);
        let tag = compute_phi_pi_fib_tag(b"nonexistent");

        assert_eq!(cache.get(tag), None);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache: PhiDiskCache<i32> = PhiDiskCache::new(3);

        let tag1 = compute_phi_pi_fib_tag(b"entry1");
        let tag2 = compute_phi_pi_fib_tag(b"entry2");
        let tag3 = compute_phi_pi_fib_tag(b"entry3");
        let tag4 = compute_phi_pi_fib_tag(b"entry4");

        cache.insert(tag1, 1);
        cache.insert(tag2, 2);
        cache.insert(tag3, 3);

        // Access tag1 to make it most recently used
        let _ = cache.get(tag1);

        // Insert a 4th entry; LRU (tag2) should be evicted
        cache.insert(tag4, 4);

        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.get(tag1), Some(1)); // tag1 still there
        assert_eq!(cache.get(tag2), None); // tag2 was evicted
        assert_eq!(cache.get(tag4), Some(4)); // tag4 inserted
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: PhiDiskCache<i32> = PhiDiskCache::new(10);
        let tag = compute_phi_pi_fib_tag(b"test");

        cache.insert(tag, 42);
        let _ = cache.get(tag); // Hit
        let _ = cache.get(compute_phi_pi_fib_tag(b"miss")); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache: PhiDiskCache<i32> = PhiDiskCache::new(10);
        let tag = compute_phi_pi_fib_tag(b"test");

        cache.insert(tag, 42);
        assert!(!cache.entries.is_empty());

        cache.clear();
        assert!(cache.entries.is_empty());
    }
}
