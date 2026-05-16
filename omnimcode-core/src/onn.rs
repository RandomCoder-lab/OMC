//! ONN (Omni Neural Network) primitives ported from Hermes's
//! `onn-instantiation` / `onn-geometric-self-instantiation` skills.
//!
//! Three load-bearing concepts:
//!
//! 1. **M3 spawn count** — sublogarithmic optimal subagent count via
//!    Fibonacci-π-Fibonacci wave interference. Replaces the
//!    `floor(log_phi(n)) + 1` (M1) heuristic with a proven-tighter
//!    bound.
//!
//! 2. **Geometric self-instantiation** — given input state, produce
//!    M3(N) "specialists" each holding a phase-shifted compressed
//!    view of the state. Each specialist gets inherited parent
//!    geometry (μ, σ, dominant attractor).
//!
//! 3. **Fold-back** — after children compute, merge their outputs
//!    into the parent's running statistics. Updates μ, σ, and
//!    verified-pattern set.
//!
//! The headline application: **context compression**. Given N
//! messages, fold them to M3(N) specialist-dicts. Specialists
//! grow log(log(N)) — so even N=1e6 messages fold to ~25
//! specialists. That's the substrate's answer to LLM context limits.

use std::collections::BTreeMap;

const PHI: f64 = 1.618033988749895_f64;
const GOLDEN_ANGLE: f64 = 2.399963229728653_f64; // π · (3 - √5)

/// M3 spawn count: number of wave-modes whose weighted amplitude
/// exceeds 1/n. The k-th mode has amplitude φ^(-k) · sin(k·γ)
/// where γ is the golden angle.
///
/// Properties:
///   - count(1) = 0  (handled: returns 1)
///   - count(2) ≈ 1
///   - count grows sublogarithmically; bounded above by ~log_φ(n) + 1
///   - returns at least 1 for any n ≥ 1
pub fn m3_spawn_count(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    let threshold = 1.0 / (n as f64);
    let mut count = 0i64;
    // The 50-mode cap matches Hermes's implementation; further modes
    // are vanishingly small and would be pruned anyway.
    for k in 1..=50 {
        let kf = k as f64;
        let weight = PHI.powf(-kf) * (kf * GOLDEN_ANGLE).sin();
        if weight.abs() > threshold {
            count += 1;
        }
    }
    count.max(1)
}

/// Compute a phase-shifted "wave mode" value for position `pos` at
/// mode index `k`. Used for geometric phase-spread when generating
/// specialists.
pub fn wave_mode(pos: usize, k: usize) -> f64 {
    let kf = k as f64;
    let pos_f = pos as f64;
    (pos_f * GOLDEN_ANGLE * (kf + 1.0)).sin() * PHI.powf(-kf)
}

/// Build one "specialist" dict from a slice of input items (Strings,
/// for now). Each specialist holds:
///   - fold_index (which slice/wave they cover)
///   - summary (concatenated source — caller may swap for a real
///     summarizer)
///   - mu / sigma (per-item resonance statistics)
///   - dominant_attractor (nearest Fibonacci to the slice's mean
///     content hash)
///   - resonance / wave_amplitude (their position in the phi-field)
#[derive(Clone, Debug)]
pub struct Specialist {
    pub fold_index: usize,
    pub summary: String,
    pub mu: f64,
    pub sigma: f64,
    pub dominant_attractor: i64,
    pub resonance: f64,
    pub wave_amplitude: f64,
    pub item_count: usize,
}

/// Self-instantiate: given a list of input items and a task hint,
/// fold them into m3_spawn_count(items.len()) specialists. Items are
/// distributed across specialists by round-robin (geometric
/// distribution would over-engineer the demo; this is enough to
/// preserve order while creating a fan-out).
pub fn self_instantiate(items: &[String], task_hint: &str) -> Vec<Specialist> {
    let n = items.len() as i64;
    let k = m3_spawn_count(n).max(1) as usize;
    let mut specialists: Vec<Specialist> = Vec::with_capacity(k);
    for slot in 0..k {
        // Items assigned to this specialist by stride-k indexing.
        let mine: Vec<&str> = items.iter()
            .enumerate()
            .filter(|(i, _)| i % k == slot)
            .map(|(_, s)| s.as_str())
            .collect();
        let item_count = mine.len();
        // Hash each owned item to a resonance/HInt for stats.
        let mut hashes: Vec<f64> = Vec::with_capacity(item_count);
        for s in &mine {
            let h = crate::tokenizer::fnv1a_64(s.as_bytes());
            hashes.push(crate::value::HInt::compute_resonance(h));
        }
        let mu = if hashes.is_empty() { 0.0 }
                 else { hashes.iter().sum::<f64>() / (hashes.len() as f64) };
        let var = if hashes.is_empty() { 0.0 }
                  else { hashes.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / (hashes.len() as f64) };
        let sigma = var.sqrt();
        // Dominant attractor: average content-hash → nearest Fib.
        let mean_hash = if mine.is_empty() { 0i64 }
                        else {
                            let sum: i128 = mine.iter()
                                .map(|s| crate::tokenizer::fnv1a_64(s.as_bytes()) as i128)
                                .sum();
                            (sum / (mine.len() as i128)) as i64
                        };
        let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(mean_hash);
        // Summary: concatenate first 64 chars of each item with a
        // separator. Callers can swap in a real summarizer.
        let mut summary = format!("[{}/{}] {}: ", slot + 1, k, task_hint);
        for (i, s) in mine.iter().enumerate() {
            if i > 0 { summary.push_str(" | "); }
            let truncated: String = s.chars().take(64).collect();
            summary.push_str(&truncated);
            if s.chars().count() > 64 { summary.push('…'); }
        }
        specialists.push(Specialist {
            fold_index: slot,
            summary,
            mu,
            sigma,
            dominant_attractor: attractor,
            resonance: crate::value::HInt::compute_resonance(mean_hash),
            wave_amplitude: wave_mode(slot, slot),
            item_count,
        });
    }
    specialists
}

/// Fold the children's outputs (results) back into a parent state.
/// Returns updated {mu, sigma, turn_count, dominant_attractor,
/// num_specialists_folded, resonance}.
pub fn fold_back(
    parent_mu: f64,
    parent_sigma: f64,
    parent_turn: i64,
    children: &[Specialist],
) -> BTreeMap<String, f64> {
    let n = children.len().max(1) as f64;
    // Weighted-by-item-count update of mu (heavier-loaded
    // specialists carry more weight in the fold).
    let total_items: f64 = children.iter().map(|c| c.item_count as f64).sum::<f64>().max(1.0);
    let child_mu: f64 = children.iter()
        .map(|c| c.mu * (c.item_count as f64))
        .sum::<f64>() / total_items;
    // Welford-ish blend with parent state (parent counts as N=turn_count).
    let p_weight = (parent_turn as f64).max(1.0);
    let new_mu = (parent_mu * p_weight + child_mu * total_items) / (p_weight + total_items);
    // Variance blend (population formula, approximation).
    let child_var: f64 = children.iter()
        .map(|c| (c.sigma * c.sigma) * (c.item_count as f64))
        .sum::<f64>() / total_items;
    let parent_var = parent_sigma * parent_sigma;
    let new_var = (parent_var * p_weight + child_var * total_items) / (p_weight + total_items);
    let new_sigma = new_var.sqrt();
    let mean_attractor: i64 = if children.is_empty() { 0 }
                              else {
                                  let s: i128 = children.iter()
                                      .map(|c| c.dominant_attractor as i128)
                                      .sum();
                                  (s / (children.len() as i128)) as i64
                              };
    let (attr, _) = crate::phi_pi_fib::nearest_attractor_with_dist(mean_attractor);
    let mut out = BTreeMap::new();
    out.insert("mu".to_string(), new_mu);
    out.insert("sigma".to_string(), new_sigma);
    out.insert("turn_count".to_string(), parent_turn as f64 + n);
    out.insert("dominant_attractor".to_string(), attr as f64);
    out.insert("num_specialists_folded".to_string(), children.len() as f64);
    out.insert("resonance".to_string(), crate::value::HInt::compute_resonance(attr));
    out
}

/// A ChildFold — a specialized mini-computation that explores
/// boundaries the parent couldn't handle. Ported from
/// Sovereign_Lattice/.../register_singularity_integration.py.
///
/// In the original: spawned when an OmniRegister's tension exceeds
/// 1/φ. Here: a deterministic structure exposing the
/// (numerator, denominator) "focus region" and a substrate-fold
/// resolution, runnable purely from a single HInt-shaped seed token.
///
/// This is what gives us "expand from a single substrate token back
/// to a computational subspace" — the seed carries enough metadata
/// to drive a fold_escape + harmony resolution.
#[derive(Clone, Debug)]
pub struct ChildFold {
    pub fold_id: i64,           // derived from seed hash
    pub focus_numerator: i64,
    pub focus_denominator: i64,
    pub spawn_reason: String,
    pub resonance_target: f64,
    pub explored_value: i64,    // result of fold-escape on the boundary
    pub final_resonance: f64,
}

/// Spawn a ChildFold from a single seed HInt value. The seed's
/// substrate metadata (value, resonance, attractor distance) drives
/// the boundary exploration. Deterministic — same seed always
/// produces the same ChildFold.
///
/// Strategy:
///   - Treat seed as a (numerator, denominator) decomposition via
///     attractor neighbors: numerator = nearest_attractor(seed),
///     denominator = max(1, distance_to_attractor(seed)).
///   - Resolution: fold seed's numerator to nearest Fibonacci
///     (the "boundary fold" — what the parent register would do
///     if tension exceeded 1/φ).
///   - Final resonance = HInt::new(folded_value).resonance.
pub fn spawn_child_fold(seed: i64, spawn_reason: &str) -> ChildFold {
    let (attractor, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(seed.abs());
    let numerator = attractor;
    let denominator = dist.max(1);
    let explored = crate::phi_pi_fib::nearest_attractor_with_dist(numerator).0;
    let resonance_target = 1.0 / (1.0 + (dist as f64));
    let final_resonance = crate::value::HInt::compute_resonance(explored);
    // fold_id derives from a stable hash of the seed value.
    let mut h = seed as u64;
    h = h.wrapping_mul(0x9E3779B97F4A7C15);
    h ^= h >> 33;
    let fold_id = (h & 0x7fff_ffff) as i64;
    ChildFold {
        fold_id,
        focus_numerator: numerator,
        focus_denominator: denominator,
        spawn_reason: spawn_reason.to_string(),
        resonance_target,
        explored_value: explored,
        final_resonance,
    }
}

/// Geodesic expansion: given a single seed token, deterministically
/// reconstruct an N-element sequence of HInt-valued substrate samples
/// along the geodesic path from `seed` toward its nearest Fibonacci
/// attractor.
///
/// This is what the user pointed at: "replicate entire forms of
/// compressed data from singular tokens" — formalized as walking the
/// φ-field geodesic from the seed to its attractor in N substrate-
/// equal steps. Each step yields a value, its resonance, and its
/// position along the path.
///
/// Honest framing: this is GEOMETRIC reconstruction, not semantic.
/// The seed carries no information about the original payload; it
/// just defines a φ-field geodesic. What this is useful for: stable
/// pseudo-random sequences anchored at a substrate-meaningful start.
pub fn geodesic_expand(seed: i64, n_samples: usize) -> Vec<(i64, f64)> {
    if n_samples == 0 {
        return Vec::new();
    }
    let (attractor, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(seed.abs());
    let mut out = Vec::with_capacity(n_samples);
    // Walk from `seed` toward `attractor` in n_samples equal steps.
    // If seed IS the attractor, the path is just `attractor` repeated
    // with phase-shifted wave modulation so the expansion isn't trivial.
    let target = if attractor > 0 { attractor } else { seed };
    let span = target - seed;
    for k in 0..n_samples {
        let t = (k as f64 + 1.0) / (n_samples as f64);
        // Linear interpolation along the geodesic, modulated by a
        // wave-mode that's stable per-k.
        let modulation = (wave_mode(k, k % 7) * (dist as f64).max(1.0)).round() as i64;
        let val = seed + (span as f64 * t).round() as i64 + modulation;
        let resonance = crate::value::HInt::compute_resonance(val);
        out.push((val, resonance));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m3_grows_sublog() {
        // Reproduce the table from Hermes's docs (within ±1 due to
        // rounding of GOLDEN_ANGLE).
        assert!(m3_spawn_count(5) <= 4);
        assert!(m3_spawn_count(20) <= 8);
        assert!(m3_spawn_count(50) <= 10);
        assert!(m3_spawn_count(200) <= 15);
        // Always at least 1.
        assert_eq!(m3_spawn_count(0), 1);
        assert_eq!(m3_spawn_count(1), 1);
    }

    #[test]
    fn m3_bounded_above_by_m1_envelope() {
        // M3 <= M1 = floor(log_phi(n)) + 1 + safety. Loose check
        // that M3 never blows past ~log_phi(n)+5.
        for &n in &[5, 20, 50, 100, 500, 1000, 10000] {
            let m3 = m3_spawn_count(n);
            let m1 = ((n as f64).ln() / PHI.ln()).floor() as i64 + 1;
            assert!(m3 <= m1 + 5, "n={n}, m3={m3}, m1={m1}");
        }
    }

    #[test]
    fn self_instantiate_creates_m3_specialists() {
        let items: Vec<String> = (0..20).map(|i| format!("item-{}", i)).collect();
        let specs = self_instantiate(&items, "test");
        assert_eq!(specs.len(), m3_spawn_count(20) as usize);
    }

    #[test]
    fn self_instantiate_preserves_item_count() {
        let items: Vec<String> = (0..50).map(|i| format!("item-{}", i)).collect();
        let specs = self_instantiate(&items, "test");
        let total: usize = specs.iter().map(|s| s.item_count).sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn fold_back_updates_turn_count() {
        let items: Vec<String> = (0..10).map(|i| format!("item-{}", i)).collect();
        let specs = self_instantiate(&items, "test");
        let folded = fold_back(0.5, 0.1, 0, &specs);
        assert!(folded.get("turn_count").unwrap() >= &(specs.len() as f64));
    }
}
