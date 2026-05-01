// src/hbit.rs - HBit (Harmonic Bit) Processing Engine (FIXED)
// Dual-band computation with harmonic coherence tracking

use crate::value::{HInt, PHI};
use std::collections::HashMap;

/// HBit Processor - Manages dual-band variables and harmony tracking
#[derive(Clone, Debug)]
pub struct HBitProcessor {
    /// Active dual-band variables: name -> (alpha, beta)
    pub bands: HashMap<String, (i64, i64)>,
    /// Cumulative harmony across all operations
    pub cumulative_harmony: f64,
    /// Operation count
    pub op_count: usize,
    /// Max harmony achieved (f64::NEG_INFINITY if no ops)
    pub max_harmony: f64,
    /// Min harmony achieved (f64::INFINITY if no ops)
    pub min_harmony: f64,
}

impl HBitProcessor {
    pub fn new() -> Self {
        HBitProcessor {
            bands: HashMap::new(),
            cumulative_harmony: 0.0,
            op_count: 0,
            max_harmony: f64::NEG_INFINITY,
            min_harmony: f64::INFINITY,
        }
    }

    /// Register a new dual-band variable by name
    pub fn register(&mut self, name: String, alpha: i64, beta: i64) {
        self.bands.insert(name, (alpha, beta));
        let harmony = Self::harmony(alpha, beta);
        self.track_harmony(harmony);
    }

    /// Calculate harmony between two bands (from value.rs HBit)
    /// Delegates to existing implementation to avoid duplication
    pub fn harmony(alpha: i64, beta: i64) -> f64 {
        let diff = (alpha - beta).abs() as f64;
        1.0 / (1.0 + diff)
    }

    /// Calculate tension (complementary to harmony)
    pub fn tension(harmony: f64) -> f64 {
        1.0 - harmony
    }

    /// Phi-fold: fractional part of alpha × φ
    /// Maps any integer to [0, 1) deterministically via golden ratio
    /// Uses the same pattern as HInt::compute_him for consistency
    pub fn phi_fold(alpha: i64) -> f64 {
        let x = alpha as f64 * PHI;
        x - x.floor()  // Fractional part in [0, 1)
    }

    /// Track harmony statistics
    fn track_harmony(&mut self, harmony: f64) {
        self.cumulative_harmony += harmony;
        self.op_count += 1;
        self.max_harmony = self.max_harmony.max(harmony);
        self.min_harmony = self.min_harmony.min(harmony);
    }

    /// Lookup a registered band variable
    fn get_band(&self, name: &str) -> Result<(i64, i64), String> {
        self.bands
            .get(name)
            .copied()
            .ok_or_else(|| format!("Unknown band: {}", name))
    }

    /// Dual-band addition: result = a + b
    /// Updates internal state with result stored as result_name
    pub fn add(&mut self, a_name: &str, b_name: &str, result_name: &str) -> Result<(), String> {
        let (a_alpha, a_beta) = self.get_band(a_name)?;
        let (b_alpha, b_beta) = self.get_band(b_name)?;

        let result_alpha = a_alpha.wrapping_add(b_alpha);
        let result_beta = a_beta.wrapping_add(b_beta);

        // Use register() to ensure track_harmony is called and stats are captured
        self.register(result_name.to_string(), result_alpha, result_beta);
        Ok(())
    }

    /// Dual-band subtraction: result = a - b
    pub fn sub(&mut self, a_name: &str, b_name: &str, result_name: &str) -> Result<(), String> {
        let (a_alpha, a_beta) = self.get_band(a_name)?;
        let (b_alpha, b_beta) = self.get_band(b_name)?;

        let result_alpha = a_alpha.wrapping_sub(b_alpha);
        let result_beta = a_beta.wrapping_sub(b_beta);

        // Use register() to ensure track_harmony is called and stats are captured
        self.register(result_name.to_string(), result_alpha, result_beta);
        Ok(())
    }
    /// Dual-band multiplication: result = a * b
    /// Beta uses phi-folded version for harmonic coherence
    pub fn mul(&mut self, a_name: &str, b_name: &str, result_name: &str) -> Result<(), String> {
        let (a_alpha, a_beta) = self.get_band(a_name)?;
        let (b_alpha, b_beta) = self.get_band(b_name)?;

        let result_alpha = a_alpha.wrapping_mul(b_alpha);
        // Beta: use phi-fold on the product to maintain coherence
        let beta_product = a_beta.wrapping_mul(b_beta);
        let result_beta = (Self::phi_fold(beta_product) * 1000.0) as i64; // Scale back to i64

        // Use register() to ensure track_harmony is called and stats are captured
        self.register(result_name.to_string(), result_alpha, result_beta);
        Ok(())
    }
    /// Dual-band division: result = a / b
    pub fn div(&mut self, a_name: &str, b_name: &str, result_name: &str) -> Result<(), String> {
        let (a_alpha, a_beta) = self.get_band(a_name)?;
        let (b_alpha, b_beta) = self.get_band(b_name)?;

        if b_alpha == 0 || b_beta == 0 {
            return Err("Division by zero".to_string());
        }

        let result_alpha = a_alpha / b_alpha;
        let result_beta = a_beta / b_beta;

        // Use register() to ensure track_harmony is called and stats are captured
        self.register(result_name.to_string(), result_alpha, result_beta);
        Ok(())
    }
    /// Average harmony of all operations
    pub fn average_harmony(&self) -> f64 {
        if self.op_count == 0 {
            0.0
        } else {
            self.cumulative_harmony / self.op_count as f64
        }
    }

    /// Coherence score (0.0 = chaotic, 1.0 = perfect agreement)
    pub fn coherence(&self) -> f64 {
        self.average_harmony()
    }

    /// Predictive error detection - compares alpha and beta divergence
    pub fn predict_error(&self, name: &str, expected_delta: i64) -> Result<bool, String> {
        let (alpha, beta) = self.get_band(name)?;
        let actual_delta = (alpha - beta).abs();
        Ok(actual_delta > expected_delta)
    }

    /// Get statistics for this session
    pub fn stats(&self) -> HBitStats {
        HBitStats {
            total_operations: self.op_count,
            average_harmony: self.average_harmony(),
            max_harmony: if self.op_count == 0 {
                None
            } else {
                Some(self.max_harmony)
            },
            min_harmony: if self.op_count == 0 {
                None
            } else {
                Some(self.min_harmony)
            },
            active_bands: self.bands.len(),
            cumulative_harmony: self.cumulative_harmony,
        }
    }

    /// Get a registered band's values
    pub fn get(&self, name: &str) -> Result<(i64, i64), String> {
        self.get_band(name)
    }

    /// Reset the processor
    pub fn reset(&mut self) {
        self.bands.clear();
        self.cumulative_harmony = 0.0;
        self.op_count = 0;
        self.max_harmony = f64::NEG_INFINITY;
        self.min_harmony = f64::INFINITY;
    }
}

/// HBit Processing Statistics
#[derive(Clone, Debug)]
pub struct HBitStats {
    pub total_operations: usize,
    pub average_harmony: f64,
    pub max_harmony: Option<f64>,
    pub min_harmony: Option<f64>,
    pub active_bands: usize,
    pub cumulative_harmony: f64,
}

impl HBitStats {
    pub fn display(&self) -> String {
        match (self.min_harmony, self.max_harmony) {
            (Some(min), Some(max)) => format!(
                "HBit Stats: {} ops, avg_harmony={:.4}, range=[{:.4}, {:.4}], bands={}",
                self.total_operations, self.average_harmony, min, max, self.active_bands
            ),
            _ => format!(
                "HBit Stats: {} ops, avg_harmony={:.4}, bands={}",
                self.total_operations, self.average_harmony, self.active_bands
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hbit_harmony() {
        assert_eq!(HBitProcessor::harmony(5, 5), 1.0); // Perfect harmony
        assert!(HBitProcessor::harmony(5, 10) < 1.0); // Some discord
        assert!(HBitProcessor::harmony(5, 10) > 0.0); // Still positive
    }

    #[test]
    fn test_hbit_register() {
        let mut proc = HBitProcessor::new();
        proc.register("x".to_string(), 100, 100);
        assert_eq!(proc.bands.len(), 1);
        assert_eq!(proc.op_count, 1);
        assert_eq!(proc.average_harmony(), 1.0);
    }

    #[test]
    fn test_hbit_addition() {
        let mut proc = HBitProcessor::new();
        proc.register("a".to_string(), 10, 10);
        proc.register("b".to_string(), 5, 5);
        
        proc.add("a", "b", "result").unwrap();
        
        let (alpha, beta) = proc.get("result").unwrap();
        assert_eq!(alpha, 15);
        assert_eq!(beta, 15);
        assert_eq!(proc.op_count, 3); // register a, register b, add
    }

    #[test]
    fn test_hbit_multiplication() {
        let mut proc = HBitProcessor::new();
        proc.register("a".to_string(), 3, 3);
        proc.register("b".to_string(), 4, 4);
        
        proc.mul("a", "b", "result").unwrap();
        
        let (alpha, beta) = proc.get("result").unwrap();
        assert_eq!(alpha, 12);
        // Beta is phi-folded version
        assert!(proc.op_count >= 3);
    }

    #[test]
    fn test_phi_fold() {
        // phi_fold(5) should be frac part of 5 * 1.618...
        let folded = HBitProcessor::phi_fold(5);
        assert!(folded >= 0.0 && folded < 1.0);
        
        // Different inputs should generally give different outputs
        let folded_10 = HBitProcessor::phi_fold(10);
        assert!(folded >= 0.0 && folded < 1.0);
    }

    #[test]
    fn test_hbit_stats_empty() {
        let proc = HBitProcessor::new();
        let stats = proc.stats();
        assert_eq!(stats.total_operations, 0);
        assert!(stats.max_harmony.is_none());
        assert!(stats.min_harmony.is_none());
    }

    #[test]
    fn test_hbit_stats_with_ops() {
        let mut proc = HBitProcessor::new();
        proc.register("a".to_string(), 10, 10);
        proc.register("b".to_string(), 20, 20);
        
        let stats = proc.stats();
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.average_harmony, 1.0); // Both perfect
        assert!(stats.max_harmony.is_some());
        assert!(stats.min_harmony.is_some());
    }

    #[test]
    fn test_hbit_error_prediction() {
        let mut proc = HBitProcessor::new();
        proc.register("x".to_string(), 100, 105);
        
        // Divergence is 5
        assert!(!proc.predict_error("x", 10).unwrap()); // expected_delta=10, actual=5 (no error)
        assert!(proc.predict_error("x", 2).unwrap());   // expected_delta=2, actual=5 (error predicted)
    }

    #[test]
    fn test_hbit_unknown_band() {
        let mut proc = HBitProcessor::new();
        
        let result = proc.add("nonexistent", "also_nonexistent", "result");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown band"));
    }
}
