//! Dodecahedral content-addressing — the proven uniform φ-address, promoted to the core.
//!
//! Ported from the transformerless_lm research (`name_registry.py` / `addressed_memory.py` /
//! `hierarchical_address.py`), where `uniform_haddr` was measured **χ²≈17 (p≈0.11)** on the
//! function-name registry — statistically uniform face occupancy, versus χ²≈216 for the older
//! sin/cos fingerprint. This module ports the FACE/SUB_FACE mechanism bit-faithfully: two
//! fmix32-finalized golden-ratio LCG hashes → inverse-CDF uniform sphere point → read the face
//! off the 12 canonical icosahedral normals.
//!
//! Honest difference from the Python registry: the Python `zeck` level fed
//! `substrate_fingerprint(name).norm()` into Zeckendorf — but that φ-fingerprint has CONSTANT
//! norm √(d/2) (each sin/cos pair contributes 1), so `zeck` was effectively the constant
//! `zeckendorf(57) = {2,55}` for every name and carried no information. Here `zeck` is derived
//! from a THIRD content hash, so it is a genuine discriminating sub-bucket — strictly more
//! useful for the addressable heap, and it does not touch the proven face-uniformity result.
//!
//! Laws: pure content→address math, no dictionary/word-list/per-corpus table (law 1: universal);
//! addressing is an integer/identity quantity, exactly where substrate provably helps (law 2).

use std::sync::OnceLock;

const PHI_INT: u32 = 2_654_435_769; // floor(phi * 2^32)
const FMIX_C1: u32 = 0x85eb_ca6b;
const FMIX_C2: u32 = 0xc2b2_ae35;
const MOD32: f64 = 4_294_967_296.0; // 2^32

pub const N_FACES: usize = 12;

/// Fibonacci VALUES for the Zeckendorf sub-bucket (matches Python `hierarchical_address._FIBS`,
/// which returns values not indices). Covers integers 0..=610.
const FIBS: [i64; 14] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610];

/// The 12 icosahedral vertices = dodecahedral face normals: (0,±1,±φ) and cyclic permutations.
/// CANONICAL set — by icosahedral symmetry every face subtends an equal solid angle, so a
/// uniform point lands on each face with equal probability.
fn raw_normals() -> [[f64; 3]; N_FACES] {
    let p = (1.0 + 5.0_f64.sqrt()) / 2.0;
    [
        [0.0, 1.0, p], [0.0, 1.0, -p], [0.0, -1.0, p], [0.0, -1.0, -p],
        [1.0, p, 0.0], [1.0, -p, 0.0], [-1.0, p, 0.0], [-1.0, -p, 0.0],
        [p, 0.0, 1.0], [p, 0.0, -1.0], [-p, 0.0, 1.0], [-p, 0.0, -1.0],
    ]
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalized normals + the 3-nearest-neighbor table (for sub_face), computed once.
fn geometry() -> &'static ([[f64; 3]; N_FACES], [[usize; 3]; N_FACES]) {
    static G: OnceLock<([[f64; 3]; N_FACES], [[usize; 3]; N_FACES])> = OnceLock::new();
    G.get_or_init(|| {
        let mut norms = raw_normals();
        for v in norms.iter_mut() {
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n > 1e-12 {
                v[0] /= n;
                v[1] /= n;
                v[2] /= n;
            }
        }
        // neighbors: for each face, the 3 nearest OTHER faces by normal dot product
        // (descending by similarity, stable index tie-break — mirrors torch.argsort).
        let mut nbrs = [[0usize; 3]; N_FACES];
        for (i, row) in nbrs.iter_mut().enumerate() {
            let mut sims: Vec<(usize, f64)> = (0..N_FACES)
                .filter(|&j| j != i)
                .map(|j| (j, dot(&norms[i], &norms[j])))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            *row = [sims[0].0, sims[1].0, sims[2].0];
        }
        (norms, nbrs)
    })
}

/// Golden-ratio LCG over the chars of `text`, seeded per component.
fn lcg(text: &str, seed_i: u32) -> u32 {
    let mut h = seed_i.wrapping_add(1).wrapping_mul(PHI_INT);
    for c in text.chars() {
        h = h.wrapping_mul(PHI_INT).wrapping_add(c as u32);
    }
    h
}

/// murmur3 avalanche finalizer — decorrelates the LCG output.
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(FMIX_C1);
    h ^= h >> 13;
    h = h.wrapping_mul(FMIX_C2);
    h ^= h >> 16;
    h
}

/// Core: map three uniform [0,1) variates to a three-level dodecahedral address.
/// u0,u1 → an inverse-CDF uniform point on S² (face + sub_face); u2 → Zeckendorf magnitude.
fn address_from_uniforms(u0: f64, u1: f64, u2: f64) -> HAddr {
    let geo = geometry();
    let norms = &geo.0;
    let nbrs = &geo.1;
    let z = 2.0 * u0 - 1.0;
    let th = 2.0 * std::f64::consts::PI * u1;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let v = [r * th.cos(), r * th.sin(), z];

    // Level 1: face = argmax dot with the 12 normals.
    let mut face = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, nrm) in norms.iter().enumerate() {
        let s = dot(nrm, &v);
        if s > best {
            best = s;
            face = i;
        }
    }
    // Level 2: sub_face = which of the 3 neighbor normals it leans toward (0..3).
    let nb = &nbrs[face];
    let mut sub = 0usize;
    let mut bestn = f64::NEG_INFINITY;
    for (k, &nf) in nb.iter().enumerate() {
        let s = dot(&norms[nf], &v);
        if s > bestn {
            bestn = s;
            sub = k;
        }
    }
    // Level 3: Zeckendorf of a content-derived magnitude in [0, 610].
    let n = (u2 * 610.0).round() as i64;
    HAddr { face, sub_face: sub, zeck: zeckendorf_values(n) }
}

/// Greedy Zeckendorf as Fibonacci VALUES (clamped to [0, 610]), ascending.
pub fn zeckendorf_values(mut n: i64) -> Vec<i64> {
    if n <= 0 {
        return Vec::new();
    }
    if n > FIBS[FIBS.len() - 1] {
        n = FIBS[FIBS.len() - 1];
    }
    let mut out = Vec::new();
    for &f in FIBS.iter().rev() {
        if f <= n {
            out.push(f);
            n -= f;
        }
    }
    out.sort_unstable();
    out
}

/// A three-level dodecahedral address: face (0..12) × sub_face (0..3) × Zeckendorf magnitude.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HAddr {
    pub face: usize,
    pub sub_face: usize,
    pub zeck: Vec<i64>, // sorted Fibonacci values
}

/// Compute the equal-area dodecahedral address of the string `text`.
pub fn haddr(text: &str) -> HAddr {
    let u0 = fmix32(lcg(text, 0)) as f64 / MOD32;
    let u1 = fmix32(lcg(text, 1)) as f64 / MOD32;
    let u2 = fmix32(lcg(text, 2)) as f64 / MOD32;
    address_from_uniforms(u0, u1, u2)
}

/// Compute the dodecahedral address of a 64-bit content hash — for addressing
/// arbitrary values by content. Derives three decorrelated variates from the hash
/// halves, then the same uniform-sphere mapping as `haddr`.
pub fn haddr_hash(h: u64) -> HAddr {
    let lo = h as u32;
    let hi = (h >> 32) as u32;
    let u0 = fmix32(lo) as f64 / MOD32;
    let u1 = fmix32(hi) as f64 / MOD32;
    let u2 = fmix32(lo ^ hi.rotate_left(16)) as f64 / MOD32;
    address_from_uniforms(u0, u1, u2)
}

/// Coarse address distance: face mismatch 3.0, sub mismatch 1.0, 0.5 per Zeckendorf symdiff element.
pub fn addr_distance(a: &HAddr, b: &HAddr) -> f64 {
    let mut d = 0.0;
    if a.face != b.face {
        d += 3.0;
    }
    if a.sub_face != b.sub_face {
        d += 1.0;
    }
    let mut sym = 0usize;
    for x in &a.zeck {
        if !b.zeck.contains(x) {
            sym += 1;
        }
    }
    for x in &b.zeck {
        if !a.zeck.contains(x) {
            sym += 1;
        }
    }
    d + 0.5 * sym as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chi_square(counts: &[usize; N_FACES], n: usize) -> f64 {
        let exp = n as f64 / N_FACES as f64;
        counts.iter().map(|&c| { let d = c as f64 - exp; d * d / exp }).sum()
    }

    #[test]
    fn faces_are_equal_area_on_uniform_points() {
        // Feed TRULY uniform sphere points (seeded splitmix64) → tests the NORMALS directly
        // (the equal-solid-angle claim), independent of hash quality. Deterministic.
        let geo = geometry();
        let norms = &geo.0;
        let mut counts = [0usize; N_FACES];
        let n = 20_000usize;
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as f64 / u64::MAX as f64
        };
        for _ in 0..n {
            let u0 = next();
            let u1 = next();
            let z = 2.0 * u0 - 1.0;
            let th = 2.0 * std::f64::consts::PI * u1;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let v = [r * th.cos(), r * th.sin(), z];
            let mut face = 0usize;
            let mut best = f64::NEG_INFINITY;
            for (i, nrm) in norms.iter().enumerate() {
                let d = dot(nrm, &v);
                if d > best {
                    best = d;
                    face = i;
                }
            }
            counts[face] += 1;
        }
        let chi = chi_square(&counts, n);
        println!("[address] uniform-points face χ² = {:.2}  counts = {:?}", chi, counts);
        assert!(chi < 50.0, "faces not equal-area (regressed toward skew): χ²={chi} counts={counts:?}");
    }

    #[test]
    fn haddr_strings_are_uniform() {
        // Feed 20k varied identifier-like strings through haddr → tests fmix32+lcg+inverse-CDF.
        // The OLD sin/cos fingerprint scored χ²≈216 here; uniform expectation is ~11. Deterministic.
        let mut counts = [0usize; N_FACES];
        let n = 20_000usize;
        for i in 0..n {
            let s = format!("fn_{}_{}", i, i.wrapping_mul(2_654_435_761) % 100_000);
            counts[haddr(&s).face] += 1;
        }
        let chi = chi_square(&counts, n);
        println!("[address] haddr-strings face χ² = {:.2}  counts = {:?}", chi, counts);
        assert!(chi < 60.0, "haddr face hash not uniform (cf. old 216): χ²={chi} counts={counts:?}");
    }

    #[test]
    fn haddr_is_deterministic_and_discriminating() {
        assert_eq!(haddr("fibonacci"), haddr("fibonacci"));
        // distinct strings should not all collapse to one address
        let a = haddr("fibonacci");
        let b = haddr("quicksort");
        let c = haddr("gcd");
        assert!(!(a == b && b == c), "addresses degenerate: all equal");
    }

    #[test]
    fn addr_distance_basics() {
        let a = HAddr { face: 3, sub_face: 1, zeck: vec![2, 55] };
        assert_eq!(addr_distance(&a, &a.clone()), 0.0);
        let face_diff = HAddr { face: 7, sub_face: 1, zeck: vec![2, 55] };
        assert!(addr_distance(&a, &face_diff) >= 3.0);
        let zeck_diff = HAddr { face: 3, sub_face: 1, zeck: vec![3, 55] }; // symdiff {2,3} = 2 elems
        assert_eq!(addr_distance(&a, &zeck_diff), 1.0);
    }

    #[test]
    fn zeckendorf_matches_python() {
        assert_eq!(zeckendorf_values(57), vec![2, 55]); // the old vestigial constant
        assert_eq!(zeckendorf_values(0), Vec::<i64>::new());
        assert_eq!(zeckendorf_values(1), vec![1]);
        assert_eq!(zeckendorf_values(4), vec![1, 3]);
        assert_eq!(zeckendorf_values(1000), vec![610]); // clamps to 610
    }
}
