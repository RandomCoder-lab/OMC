//! Locality-bearing fingerprint — the SIMILARITY primitive (Phase 1.2).
//!
//! `haddr` (φ-address) is uniform but has NO content locality: a φ-hash of a slightly-changed
//! string is unrelated to the original, so φ-cosine retrieval ≈ random (proven: recall@5 0.016,
//! navigation 0.29). A byte-histogram fingerprint makes similar content *similar* → retrieval and
//! coarse-to-fine navigation work (proven: unigram recall@5 0.24, bigram 0.41; navigation 0.88).
//!
//! "Two fingerprints, two jobs": `haddr` for exact keys / uniform buckets, `locality` for
//! similarity / navigation. Universal — pure byte statistics, no vocab or word list.
//!
//! Ported from `experiments/transformerless_lm/locality_fp.py` (`hist_fp`). The core uses raw
//! bytes as the alphabet (no corpus vocab needed); same content-locality property.

const BIGRAM_DIM: usize = 4096;

/// Normalized byte-histogram fingerprint of `text`. unigram = 256-dim byte counts; bigram =
/// 4096-dim hashed consecutive-byte-pair counts (the proven-better retrieval variant on prose).
pub fn fingerprint(text: &str, bigram: bool) -> Vec<f64> {
    let bytes = text.as_bytes();
    let mut h = vec![0.0f64; if bigram { BIGRAM_DIM } else { 256 }];
    if bigram {
        for w in bytes.windows(2) {
            let idx = ((w[0] as usize) * 256 + w[1] as usize) % BIGRAM_DIM;
            h[idx] += 1.0;
        }
        // single-byte fallback so 1-char strings still get a fingerprint
        if bytes.len() == 1 {
            h[(bytes[0] as usize) % BIGRAM_DIM] += 1.0;
        }
    } else {
        for &b in bytes {
            h[b as usize] += 1.0;
        }
    }
    let norm = h.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in h.iter_mut() {
            *x /= norm;
        }
    }
    h
}

/// Cosine similarity of two equal-length fingerprints (normalized → just the dot product).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity of two strings' locality fingerprints, in [0, 1].
pub fn sim(a: &str, b: &str, bigram: bool) -> f64 {
    cosine(&fingerprint(a, bigram), &fingerprint(b, bigram))
}

/// Index of the candidate most similar to `query` by locality cosine, or -1 if empty.
pub fn nearest(query: &str, candidates: &[String], bigram: bool) -> i64 {
    let q = fingerprint(query, bigram);
    let mut best = -1i64;
    let mut best_sim = f64::NEG_INFINITY;
    for (i, c) in candidates.iter().enumerate() {
        let s = cosine(&q, &fingerprint(c, bigram));
        if s > best_sim {
            best_sim = s;
            best = i as i64;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic splitmix64
    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn upto(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn locality_beats_phi_on_corrupted_retrieval() {
        // The proven property, reproduced in-core: a slightly-corrupted query should still
        // retrieve its source document via LOCALITY (content survives), but NOT via φ/haddr
        // (a uniform hash scrambles under any change). Deterministic.
        const N: usize = 200;
        const LEN: usize = 14;
        let alpha: Vec<u8> = (b'a'..=b'r').collect(); // 18 letters
        let mut rng = R(0x1234_5678_9abc_def0);

        let docs: Vec<String> = (0..N)
            .map(|_| {
                (0..LEN)
                    .map(|_| alpha[rng.upto(alpha.len())] as char)
                    .collect()
            })
            .collect();

        // query_i = doc_i with 2 positions changed to a different letter
        let mut uni_hit = 0usize;
        let mut bi_hit = 0usize;
        let mut phi_hit = 0usize;
        for (i, d) in docs.iter().enumerate() {
            let mut q: Vec<u8> = d.bytes().collect();
            for _ in 0..2 {
                let p = rng.upto(LEN);
                let mut c = alpha[rng.upto(alpha.len())];
                if c == q[p] {
                    c = alpha[(rng.upto(alpha.len()) + 1) % alpha.len()];
                }
                q[p] = c;
            }
            let query = String::from_utf8(q).unwrap();

            if nearest(&query, &docs, false) as usize == i {
                uni_hit += 1;
            }
            if nearest(&query, &docs, true) as usize == i {
                bi_hit += 1;
            }
            // φ retrieval: nearest doc by haddr address-distance (the uniform hash)
            let qa = crate::address::haddr(&query);
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (j, dj) in docs.iter().enumerate() {
                let dd = crate::address::addr_distance(&qa, &crate::address::haddr(dj));
                if dd < best_d {
                    best_d = dd;
                    best = j;
                }
            }
            if best == i {
                phi_hit += 1;
            }
        }
        let (uni, bi, phi) = (
            uni_hit as f64 / N as f64,
            bi_hit as f64 / N as f64,
            phi_hit as f64 / N as f64,
        );
        println!(
            "[locality] corrupted-retrieval recall@1 (N={N}): locality-unigram={:.3} locality-bigram={:.3} φ/haddr={:.3} (chance≈{:.3})",
            uni, bi, phi, 1.0 / N as f64
        );
        assert!(uni > 0.5, "locality (unigram) should recover most: {uni}");
        assert!(phi < 0.15, "φ/haddr must be ~chance (no content locality): {phi}");
    }

    #[test]
    fn sim_orders_by_content() {
        // identical = 1.0; near-variant high; unrelated low.
        assert!((sim("fibonacci", "fibonacci", false) - 1.0).abs() < 1e-9);
        let near = sim("quicksort", "quick_sort", false);
        let far = sim("quicksort", "zzzzzzzzz", false);
        assert!(near > far, "near-variant ({near}) should beat unrelated ({far})");
        assert!(near > 0.6, "near-variant similarity unexpectedly low: {near}");
    }
}
