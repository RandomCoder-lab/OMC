//! In-core **addressed memory** — a kNN datastore with a k-means IVF index. The Rust-native
//! substrate primitive behind the proven write-don't-train recipe (see written_memory.py and
//! GENERATOR_PLAN.md, 159+ pre-registered runs, 2026-05-30):
//!
//!   * WRITE (key_vector -> value) entries with NO training — O(1) per item, not gradient descent.
//!   * INDEX the keys with IVF (k-means Voronoi cells) — the frontier winner: ~99-100% of brute-force
//!     recall at a fraction of the comparisons (it beat fixed substrate geometry on learned-float keys,
//!     consistent with the integer-substrate law).
//!   * SEARCH the nearest written values for a query vector — addressing decides WHERE to look.
//!
//! Sits alongside the other addressing primitives in the core (haddr, cas, the addressable heap).
//! Stateful datastores live in a `thread_local` handle registry (the `python_embed` pattern): OMC
//! code holds an integer handle and calls `amem_write` / `amem_index` / `amem_search` on it.
//!
//! The generator (the char-LM that produces the key vectors) stays in Python; this is the memory and
//! index substrate it writes into and reads from.

use std::cell::RefCell;
use std::collections::HashMap;

/// A single addressed-memory datastore: parallel keys/values plus an optional IVF index.
pub struct Datastore {
    dim: usize,
    keys: Vec<Vec<f32>>,
    vals: Vec<i64>,
    centroids: Vec<Vec<f32>>, // empty until `build_index`
    cell_of_key: Vec<usize>,  // per-key centroid id; aligned with `keys` once indexed
}

#[inline]
fn dist2(a: &[f32], b: &[f32]) -> f32 {
    // squared euclidean; lengths are equal by construction (dim is fixed at first write)
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

impl Datastore {
    fn new() -> Self {
        Datastore { dim: 0, keys: Vec::new(), vals: Vec::new(), centroids: Vec::new(), cell_of_key: Vec::new() }
    }

    /// WRITE one (key -> value). The first write fixes the key dimension; later writes must match.
    fn write(&mut self, key: Vec<f32>, val: i64) -> Result<(), String> {
        if self.keys.is_empty() {
            self.dim = key.len();
        } else if key.len() != self.dim {
            return Err(format!("amem_write: key dim {} != store dim {}", key.len(), self.dim));
        }
        self.keys.push(key);
        self.vals.push(val);
        self.centroids.clear(); // a write invalidates any existing index
        self.cell_of_key.clear();
        Ok(())
    }

    fn nearest_centroid(&self, key: &[f32]) -> usize {
        let mut best = 0usize;
        let mut bd = f32::INFINITY;
        for (c, cen) in self.centroids.iter().enumerate() {
            let d = dist2(key, cen);
            if d < bd {
                bd = d;
                best = c;
            }
        }
        best
    }

    /// Build the IVF index: `nlist` k-means cells over the keys. Deterministic init (stride-spaced,
    /// `seed`-offset) so no RNG dependency — reproducible across runs.
    fn build_index(&mut self, nlist: usize, iters: usize, seed: usize) {
        let n = self.keys.len();
        if n == 0 {
            return;
        }
        let nlist = nlist.max(1).min(n);
        let stride = (n / nlist).max(1);
        self.centroids = (0..nlist).map(|i| self.keys[(seed + i * stride) % n].clone()).collect();
        for _ in 0..iters {
            let assign: Vec<usize> = self.keys.iter().map(|k| self.nearest_centroid(k)).collect();
            let mut sums = vec![vec![0.0f32; self.dim]; nlist];
            let mut cnts = vec![0usize; nlist];
            for (i, &c) in assign.iter().enumerate() {
                cnts[c] += 1;
                let row = &self.keys[i];
                for j in 0..self.dim {
                    sums[c][j] += row[j];
                }
            }
            for c in 0..nlist {
                if cnts[c] > 0 {
                    for j in 0..self.dim {
                        sums[c][j] /= cnts[c] as f32;
                    }
                    self.centroids[c] = std::mem::take(&mut sums[c]);
                }
                // empty cell keeps its previous centroid
            }
        }
        self.cell_of_key = self.keys.iter().map(|k| self.nearest_centroid(k)).collect();
    }

    /// SEARCH: the values of the `k` nearest written keys to `query`. Uses the IVF index (probing the
    /// `nprobe` nearest cells) when built, else brute-force over all keys (still correct).
    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<i64> {
        if self.keys.is_empty() {
            return Vec::new();
        }
        // candidate key indices
        let candidates: Vec<usize> = if self.centroids.is_empty() {
            (0..self.keys.len()).collect()
        } else {
            // nearest `nprobe` centroids
            let mut cd: Vec<(f32, usize)> =
                self.centroids.iter().enumerate().map(|(c, cen)| (dist2(query, cen), c)).collect();
            cd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let probe: Vec<usize> = cd.iter().take(nprobe.max(1)).map(|&(_, c)| c).collect();
            let mut out = Vec::new();
            for (i, &c) in self.cell_of_key.iter().enumerate() {
                if probe.contains(&c) {
                    out.push(i);
                }
            }
            if out.is_empty() {
                (0..self.keys.len()).collect()
            } else {
                out
            }
        };
        let mut scored: Vec<(f32, usize)> =
            candidates.iter().map(|&i| (dist2(query, &self.keys[i]), i)).collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.iter().take(k).map(|&(_, i)| self.vals[i]).collect()
    }
}

// ── thread_local handle registry (the python_embed pattern) ─────────────────────
thread_local! {
    static STORES: RefCell<HashMap<i64, Datastore>> = RefCell::new(HashMap::new());
    static NEXT_HANDLE: RefCell<i64> = const { RefCell::new(1) };
}

/// Allocate a fresh empty datastore; returns its handle.
pub fn amem_new() -> i64 {
    let h = NEXT_HANDLE.with(|n| {
        let mut n = n.borrow_mut();
        let h = *n;
        *n += 1;
        h
    });
    STORES.with(|s| s.borrow_mut().insert(h, Datastore::new()));
    h
}

/// WRITE (key -> value) into the datastore behind `handle`. Returns the new entry count.
pub fn amem_write(handle: i64, key: Vec<f32>, val: i64) -> Result<i64, String> {
    STORES.with(|s| {
        let mut s = s.borrow_mut();
        let ds = s.get_mut(&handle).ok_or_else(|| format!("amem_write: bad handle {handle}"))?;
        ds.write(key, val)?;
        Ok(ds.keys.len() as i64)
    })
}

/// Build the IVF index over the datastore (k-means, `nlist` cells). Returns the cell count.
pub fn amem_index(handle: i64, nlist: i64, iters: i64, seed: i64) -> Result<i64, String> {
    STORES.with(|s| {
        let mut s = s.borrow_mut();
        let ds = s.get_mut(&handle).ok_or_else(|| format!("amem_index: bad handle {handle}"))?;
        ds.build_index(nlist.max(1) as usize, iters.max(1) as usize, seed.max(0) as usize);
        Ok(ds.centroids.len() as i64)
    })
}

/// SEARCH the `k` nearest values to `query` (probing `nprobe` IVF cells; brute if not indexed).
pub fn amem_search(handle: i64, query: Vec<f32>, k: i64, nprobe: i64) -> Result<Vec<i64>, String> {
    STORES.with(|s| {
        let s = s.borrow();
        let ds = s.get(&handle).ok_or_else(|| format!("amem_search: bad handle {handle}"))?;
        Ok(ds.search(&query, k.max(1) as usize, nprobe.max(1) as usize))
    })
}

/// Entry count of the datastore behind `handle` (or error if the handle is unknown).
pub fn amem_len(handle: i64) -> Result<i64, String> {
    STORES.with(|s| {
        s.borrow().get(&handle).map(|ds| ds.keys.len() as i64).ok_or_else(|| format!("amem_len: bad handle {handle}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_index_search_roundtrip() {
        let h = amem_new();
        // 4 clusters in 2-D; value = cluster id. Many points so IVF has something to do.
        let centers = [[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        for (cid, c) in centers.iter().enumerate() {
            for j in 0..50 {
                let jitter = (j as f32) * 0.01;
                amem_write(h, vec![c[0] + jitter, c[1] - jitter], cid as i64).unwrap();
            }
        }
        assert_eq!(amem_len(h).unwrap(), 200);

        // brute-force search (no index): nearest to (9.9, 0.1) should be cluster 1
        let r = amem_search(h, vec![9.9, 0.1], 5, 1).unwrap();
        assert!(r.iter().all(|&v| v == 1), "brute search returned {:?}", r);

        // build IVF and search again — same answer, via the index
        let cells = amem_index(h, 4, 8, 0).unwrap();
        assert!(cells >= 1);
        let r2 = amem_search(h, vec![0.1, 9.9], 5, 2).unwrap();
        assert!(r2.iter().all(|&v| v == 2), "IVF search returned {:?}", r2);

        // a write invalidates the index; search still correct (falls back to brute)
        amem_write(h, vec![10.1, 9.9], 3).unwrap();
        let r3 = amem_search(h, vec![10.0, 10.0], 3, 1).unwrap();
        assert!(r3.iter().all(|&v| v == 3), "post-write search returned {:?}", r3);
    }

    #[test]
    fn dim_mismatch_errors() {
        let h = amem_new();
        amem_write(h, vec![1.0, 2.0], 0).unwrap();
        assert!(amem_write(h, vec![1.0, 2.0, 3.0], 1).is_err());
    }
}
