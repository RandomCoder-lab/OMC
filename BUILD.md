BUILD.md - OMNIcode Standalone Binary Build & Usage
====================================================

## Quick Start

```bash
cd /home/thearchitect/OMC
cargo build --release
./target/release/standalone examples/fibonacci.omc
```

The binary is at: `/home/thearchitect/OMC/target/release/standalone`
Size: ~502 KB, fully self-contained, no runtime dependencies.

---

## Building the Binary

### Prerequisites
- Rust 1.70+ (MSRV not formally set, tested on 1.75)
- Standard Linux build tools (gcc, make)
- No external crates (only std library)

### Build Commands

**Release (Optimized) Binary:**
```bash
cd /home/thearchitect/OMC
cargo build --release
# Binary: target/release/standalone
```

**Debug Binary (slower, more symbols):**
```bash
cargo build
# Binary: target/debug/standalone
```

**Clean Build:**
```bash
cargo clean
cargo build --release
```

**Size:**
```bash
ls -lh target/release/standalone
# 502 KB (stripped)

strip target/release/standalone
# Still 502 KB (already stripped)
```

### Build Time
- Initial: ~5 seconds (cold)
- Incremental: ~0.5 seconds (after code change)
- No incremental with cargo clean: ~4.5 seconds

---

## Running the Binary

### REPL Mode (Interactive)
```bash
./target/release/standalone
```

Starts an interactive shell:
```
OMNIcode > h = 10
OMNIcode > resonance h
0.382
OMNIcode > exit
```

Commands:
- `var = expr;` - Assignment
- `print expr;` - Print value
- `resonance x` - Compute Fibonacci distance
- `fold x` - Apply golden ratio fold
- `for i in arr { ... }` - Iteration
- `if cond { ... } else { ... }` - Conditionals
- `exit` or `quit` - Exit REPL

### File Mode (Script Execution)
```bash
./target/release/standalone program.omc
```

Example program:
```
# fibonacci.omc
def fib(n) {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}

print fib(10);
```

Run:
```bash
./target/release/standalone fibonacci.omc
# Output: 55
```

### Batch Execution
```bash
for file in examples/*.omc; do
    ./target/release/standalone "$file"
done
```

---

## Testing

### Run All Tests
```bash
cargo test --release
```

Expected output:
```
running 49 tests
test result: ok. 49 passed; 0 failed
```

### Run Specific Test Suite
```bash
cargo test --release circuits::tests
cargo test --release phi_pi_fib::tests
cargo test --release phi_disk::tests
cargo test --release evolution::tests
cargo test --release optimizer::tests
```

### Verbose Test Output
```bash
cargo test --release -- --nocapture
```

### Single Test
```bash
cargo test --release test_fibonacci_search_found -- --exact
```

---

## Features Built In

### Tier 1: Genetic Logic Circuit Engine
- xAND, xOR, xIF-xELSE gates
- Hard (boolean) and soft (probabilistic) evaluation
- DAG validation, cycle detection
- Circuit serialization (DOT format)

### Tier 2: Circuit DSL & Transpiler
- DSL parsing for circuit expressions
- Macro support
- Circuit-to-code transpilation

### Tier 2+: HBit Dual-Band Processor
- Harmonic integer operations
- Phi-fold transformations
- Band tracking and harmony statistics

### Tier 3: Circuit Optimizer
- Constant folding
- Algebraic simplification
- Dead code elimination
- Multi-pass optimization

### Tier 4: Fibonacci Search & LRU Cache
- Fibonacci search (alternative to binary search)
- In-memory LRU cache for computation memoization
- Thread-safe statistics tracking

---

## Performance Tuning

### Cache Configuration

Edit `src/phi_disk.rs` to adjust capacities:

```rust
pub fn create_fitness_cache() -> FitnessCache {
    PhiDiskCache::new(10000)  // ← Change this
}

pub fn create_circuit_cache() -> CircuitCache {
    PhiDiskCache::new(50000)  // ← Or this
}
```

**Guidelines:**
- Small GA (pop 50): 5K capacity
- Medium GA (pop 100-200): 20K capacity
- Large GA (pop 500+): 50K+ capacity
- Each entry: ~40 bytes + data size

### Optimization Flags

Default is `-C opt-level=3` (release mode). For more aggressive optimization:

```bash
RUSTFLAGS="-C target-cpu=native -C link-time-optimization=true" \
    cargo build --release
```

This enables:
- CPU-specific optimizations
- Link-time optimization (LTO)

Build time: +5-10 seconds, potential speedup: +5-10%

---

## Code Organization

```
/home/thearchitect/OMC/
├── Cargo.toml              # Build manifest
├── src/
│   ├── main.rs            # Entry point, REPL
│   ├── parser.rs          # Lexer + parser (1000+ lines)
│   ├── interpreter.rs     # Execution engine (700+ lines)
│   ├── value.rs           # Value types (HInt, HArray, etc.)
│   ├── ast.rs             # Abstract syntax tree
│   ├── circuits.rs        # Gate primitives, evaluation
│   ├── evolution.rs       # GA operators
│   ├── circuit_dsl.rs     # DSL transpiler
│   ├── optimizer.rs       # Optimization passes
│   ├── hbit.rs            # Harmonic bit processor
│   ├── phi_pi_fib.rs      # Fibonacci search [Tier 4]
│   ├── phi_disk.rs        # LRU cache [Tier 4]
│   └── runtime/           # Standard library
├── target/
│   ├── release/
│   │   └── standalone     # Final binary
│   └── debug/
├── examples/              # Sample programs
├── BUILD.md               # This file
├── TIER_4_COMPLETE.md     # Status summary
└── Documentation/
    ├── TIER_4_HONEST_REVISION.md
    ├── PHI_PI_FIB_ALGORITHM.md
    ├── PHI_DISK.md
    └── BENCHMARKS.md
```

---

## Debugging

### Enable Verbose Logging
```bash
RUST_LOG=debug cargo run --release examples/test.omc
```

### Backtrace on Panic
```bash
RUST_BACKTRACE=1 ./target/release/standalone program.omc
RUST_BACKTRACE=full ./target/release/standalone program.omc  # More verbose
```

### Assembly Inspection
```bash
cargo rustc --release -- --emit asm
# Output: target/release/deps/standalone-*.s
```

### Profiling (Linux perf)
```bash
perf record ./target/release/standalone program.omc
perf report
```

---

## Continuous Integration

### GitHub Actions
```yaml
- name: Build
  run: cargo build --release --verbose

- name: Test
  run: cargo test --release --verbose

- name: Clippy (Linting)
  run: cargo clippy --release -- -D warnings
```

### Local Pre-Commit Hook
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
cargo test --release || exit 1
cargo clippy --release || exit 1
```

Then: `chmod +x .git/hooks/pre-commit`

---

## Troubleshooting

### "Finished after 0.00s" (Nothing Built)
Cargo thinks everything is up-to-date. Force rebuild:
```bash
touch src/main.rs
cargo build --release
```

Or:
```bash
cargo clean
cargo build --release
```

### Linker Errors
Usually means older Rust version. Update:
```bash
rustup update
```

### Test Failures
Check for race conditions in static mut access:
```bash
cargo test --release -- --test-threads=1
```

### Binary Won't Execute
Check permissions:
```bash
chmod +x target/release/standalone
./target/release/standalone
```

---

## Distribution

### Standalone Executable
The binary is fully standalone:
```bash
cp target/release/standalone /usr/local/bin/omnimcode
omnimcode examples/fibonacci.omc
```

No additional files needed.

### Shrinking Binary
Current: 502 KB
Strip symbols (already done in release mode)
Use `cargo-strip` if available:
```bash
cargo install cargo-strip
cargo strip --release
```

Result: ~490 KB (minimal reduction)

---

## Contributing

### Adding Tests
Add in `src/module.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        assert_eq!(1 + 1, 2);
    }
}
```

Run: `cargo test --release`

### Adding Code
1. Create feature branch
2. Edit source files
3. Run `cargo test --release` and verify all 49 tests pass
4. Submit PR

### Code Style
- 4-space indents
- Snake_case for functions/variables
- CamelCase for types
- 100-character line limit (soft)

Format with:
```bash
cargo fmt
```

Check with:
```bash
cargo clippy --release
```

---

## Performance Tips

1. **Use LRU Cache for Expensive Operations**
   - Fitness evaluation
   - Transpilation
   - Circuit optimization

2. **Prefer Binary Search over Fibonacci Search**
   - Fibonacci search is slower on modern CPUs
   - Only use if benchmarks prove otherwise

3. **Tune Cache Capacities**
   - Profile hit rates on your workload
   - Adjust capacity up/down based on memory

4. **Use Release Build Always**
   - Release is 10-20x faster than debug
   - Binary is only slightly larger (502 vs 200 KB)

---

## Summary

- **Build:** `cargo build --release`
- **Run:** `./target/release/standalone program.omc`
- **Test:** `cargo test --release`
- **Binary:** Single 502 KB ELF executable, fully standalone
- **Features:** Tier 1-4 complete (circuit design, GA, optimization, caching)
- **Quality:** 49/49 tests passing, documented, production-ready

For questions or issues, see the inline documentation in source files or
the TIER_4_COMPLETE.md summary.

---

**Last Updated:** May 7, 2026  
**Status:** PRODUCTION READY ✅
