// src/value.rs - OMNIcode runtime value types

use std::fmt;

/// Golden ratio constant
pub const PHI: f64 = 1.6180339887498948482045868343656;
pub const PHI_INV: f64 = 0.6180339887498943238644763136822;
pub const PHI_SQ: f64 = 2.6180339887498948482045868343656;

/// Harmonic Integer - Core numeric type with resonance tracking
#[derive(Clone, Debug)]
pub struct HInt {
    pub value: i64,
    pub resonance: f64,
    pub him_score: f64,
    pub is_singularity: bool,
}

impl HInt {
    pub fn new(value: i64) -> Self {
        let resonance = Self::compute_resonance(value);
        let him_score = Self::compute_him(value);
        HInt {
            value,
            resonance,
            him_score,
            is_singularity: false,
        }
    }

    /// Compute resonance (0-1) based on distance to nearest Fibonacci number
    pub fn compute_resonance(value: i64) -> f64 {
        let fibs: [i64; 16] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610];
        let abs_val = value.abs();
        
        // Find nearest Fibonacci
        let mut min_dist = i64::MAX;
        for &f in &fibs {
            let d = (f - abs_val).abs();
            if d < min_dist {
                min_dist = d;
            }
        }
        
        if min_dist == 0 {
            1.0
        } else {
            1.0 - (min_dist as f64) / (abs_val.max(1) as f64 + 1.0)
        }
    }

    /// Compute Harmonic Integer Map (0-1)
    pub fn compute_him(value: i64) -> f64 {
        let v = value as f64;
        let x = (v * PHI) - (v * PHI).floor();
        x.abs().min(1.0 - x.abs())
    }

    pub fn singularity() -> Self {
        HInt {
            value: 0,
            resonance: 0.0,
            him_score: 0.0,
            is_singularity: true,
        }
    }
}

impl fmt::Display for HInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_singularity {
            write!(f, "HInt(SINGULARITY)")
        } else {
            write!(
                f,
                "HInt({}, φ={:.3}, HIM={:.3})",
                self.value, self.resonance, self.him_score
            )
        }
    }
}

impl PartialEq for HInt {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

/// Harmonic Bit - Dual-band computing element
#[derive(Clone, Debug)]
pub struct HBit {
    pub b_alpha: i64,      // Classical band
    pub b_beta: i64,       // Harmonic band
    pub phase: f64,        // Wave phase
    pub weight: f64,       // Consensus weight
    pub tension: f64,      // Harmonic tension
}

impl HBit {
    pub fn new(alpha: i64, beta: i64) -> Self {
        let harmony = Self::harmony(alpha, beta);
        HBit {
            b_alpha: alpha,
            b_beta: beta,
            phase: 0.0,
            weight: harmony,
            tension: 1.0 - harmony,
        }
    }

    pub fn harmony(alpha: i64, beta: i64) -> f64 {
        let diff = (alpha - beta).abs() as f64;
        1.0 / (1.0 + diff)
    }
}

/// Harmonic Wave - Superposition of states
#[derive(Clone, Debug)]
pub struct HWave {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
}

impl HWave {
    pub fn new(amplitude: f64, frequency: f64, phase: f64) -> Self {
        HWave {
            amplitude,
            frequency,
            phase,
        }
    }

    pub fn collapse(&self) -> i64 {
        ((self.amplitude * self.frequency.cos()).round()) as i64
    }
}

/// Harmonic Singularity - Portal for undefined operations
#[derive(Clone, Debug)]
pub struct HSingularity {
    pub portal_id: u64,
    pub dimension: i64,
    pub stability: f64,
}

impl HSingularity {
    pub fn new(dimension: i64) -> Self {
        HSingularity {
            portal_id: rand_like(dimension as u64),
            dimension,
            stability: 0.0,
        }
    }
}

/// Array wrapper for homogeneous collections
#[derive(Clone, Debug)]
pub struct HArray {
    pub items: Vec<Value>,
}

impl HArray {
    pub fn new() -> Self {
        HArray { items: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        HArray {
            items: Vec::with_capacity(capacity),
        }
    }
}

/// Runtime value - Can be HInt, String, Boolean, Array, etc.
#[derive(Clone, Debug)]
pub enum Value {
    HInt(HInt),
    String(String),
    Bool(bool),
    Array(HArray),
    Circuit(crate::circuits::Circuit),  // NEW: Genetic logic circuits
    Null,
}

impl Value {
    pub fn to_int(&self) -> i64 {
        match self {
            Value::HInt(h) => h.value,
            Value::String(s) => s.parse().unwrap_or(0),
            Value::Bool(b) => if *b { 1 } else { 0 },
            Value::Null => 0,
            _ => 0,
        }
    }

    pub fn to_bool(&self) -> bool {
        match self {
            Value::HInt(h) => h.value != 0,
            Value::String(s) => !s.is_empty(),
            Value::Bool(b) => *b,
            Value::Array(a) => !a.items.is_empty(),
            Value::Circuit(_) => true, // Circuits are always "truthy"
            Value::Null => false,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Value::HInt(h) => h.to_string(),
            Value::String(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Circuit(c) => c.to_string(),
            Value::Null => "null".to_string(),
            Value::Array(a) => {
                let items: Vec<String> = a.items.iter().map(|v| v.to_string()).collect();
                format!("[{}]", items.join(", "))
            }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Simple pseudo-random generator (deterministic for reproducibility)
fn rand_like(seed: u64) -> u64 {
    let mut x = seed.wrapping_mul(6364136223846793005);
    x ^= x >> 33;
    x
}

/// Fibonacci sequence generation
pub fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0i64;
    let mut b = 1i64;
    for _ in 2..=n {
        let temp = a.wrapping_add(b);
        a = b;
        b = temp;
    }
    b
}

/// Check if a number is Fibonacci
pub fn is_fibonacci(n: i64) -> bool {
    let fibs: [i64; 20] = [
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181,
    ];
    fibs.contains(&n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hint_resonance() {
        let fib_89 = HInt::new(89);
        assert!(fib_89.resonance > 0.95);

        let nonfi = HInt::new(100);  // 100 is not a Fibonacci number
        assert!(nonfi.resonance < 0.95);
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
    }
}
