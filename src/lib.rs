// src/lib.rs - Library API for OMNIcode (mainly for benchmarking)
// Exposes the core modules for use in benches/ and tests

pub mod ast;
pub mod value;
pub mod parser;
pub mod interpreter;
pub mod runtime;
pub mod circuits;      // Genetic logic circuits
pub mod evolution;     // Genetic operators
pub mod circuit_dsl;   // Circuit DSL and transpiler [Tier 2]
pub mod optimizer;     // Circuit optimization engine [Tier 3]
pub mod hbit;          // HBit dual-band processing [Tier 2+]
pub mod phi_pi_fib;    // O(log_phi_pi_fibonacci n) search algorithm [Tier 4]
pub mod phi_disk;      // Phi Disk cache system [Tier 4]
