// src/lib.rs - Library API for OMNIcode (mainly for benchmarking)
// Exposes the core modules for use in benches/ and tests

pub mod ast;
pub mod value;
pub mod parser;
pub mod interpreter;
pub mod docs;
pub mod errors;
pub mod tokenizer;
pub mod canonical;
pub mod code_intel;
pub mod runtime;
pub mod circuits;      // Genetic logic circuits
pub mod evolution;     // Genetic operators
pub mod circuit_dsl;   // Circuit DSL and transpiler [Tier 2]
pub mod optimizer;     // Circuit optimization engine [Tier 3]
pub mod hbit;          // HBit dual-band processing [Tier 2+]
pub mod phi_pi_fib;    // O(log_phi_pi_fibonacci n) search algorithm [Tier 4]
pub mod phi_disk;      // Phi Disk cache system [Tier 4]
pub mod bytecode;      // VM bytecode + constant pool [Phase H]
pub mod compiler;      // AST -> bytecode lowering [Phase H]
pub mod vm;            // Stack-based VM execution loop [Phase H]
pub mod bytecode_opt;  // Constant folding + peephole optimizer [Phase K]
pub mod disasm;        // Bytecode disassembler [Phase P]
pub mod formatter;     // AST -> canonical OMC source (for --fmt)

// Embedded CPython: py_* builtins (numpy, pandas, ...). Default-on
// for desktop builds; downstream WASM / no_std crates can disable
// via `omnimcode-core = { default-features = false }`.
#[cfg(feature = "python-embed")]
pub mod python_embed;
