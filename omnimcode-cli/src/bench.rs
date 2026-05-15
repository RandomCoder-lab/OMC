//! Session E benchmark harness.
//!
//! Measures wall-clock time for the same OMC user function under three
//! execution modes:
//!
//!   1. Tree-walk (Interpreter::call_function_with_values)
//!   2. Bytecode VM (Vm::run_module after rebinding the program to call
//!      the target fn once per outer iteration)
//!   3. Dual-band JIT (omnimcode-codegen JIT'd fn pointer, called
//!      directly without going through the Interpreter)
//!
//! Reports min, median, mean per-call ns for each mode, plus speedup
//! ratios relative to tree-walk.
//!
//! Usage:
//!   omc-bench [iters] [fn-arg]
//!
//! Defaults: 200_000 iters, fn-arg = 12.
//!
//! The benchmark target is a hard-coded OMC source that defines
//! `factorial(n)` plus `sum_to(n)` (two self-contained ints-only fns).
//! Both are JIT-eligible; both are easy enough that the per-call cost
//! is dominated by interpreter overhead rather than the computation
//! itself — which is exactly the regime where the JIT win is sharpest.
//!
//! This is a *microbenchmark*. It deliberately compares overhead per
//! function-entry, not throughput per CPU-cycle of useful work. Don't
//! extrapolate the speedup ratios to whole-program speedups — those
//! depend on how much time real programs spend inside JIT-eligible
//! call frames vs. tree-walk-only paths (Python embed, builtins,
//! string ops, etc.).

use std::time::Instant;

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::value::{HInt, Value};

const SOURCE: &str = r#"
fn factorial(n) {
    if n <= 1 { return 1; }
    return n * factorial(n - 1);
}
fn sum_to(n) {
    h s = 0;
    h k = 1;
    while k <= n {
        s = s + k;
        k = k + 1;
    }
    return s;
}
"#;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let fn_arg: i64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);

    println!("=== omc-bench: tree-walk vs bytecode VM vs dual-band JIT ===");
    println!("iters={}, fn_arg={}", iters, fn_arg);
    println!();

    bench_fn("factorial", iters, fn_arg);
    println!();
    bench_fn("sum_to", iters, 100);

    println!();
    println!("Notes:");
    println!("  - 'tree-walk' goes through Interpreter::call_function_with_values");
    println!("    (the path used by py_callback and other host->OMC dispatch).");
    println!("  - 'JIT' calls the dual-band native fn directly via raw fn pointer");
    println!("    (no Interpreter on the call path).");
    println!("  - 'bytecode VM' is currently skipped — its calling convention");
    println!("    doesn't expose a clean per-call-from-Rust entry; programs go");
    println!("    through the full module run. A future bench will add a");
    println!("    Vm-internal looped harness for a fair comparison.");
}

fn bench_fn(fn_name: &str, iters: usize, arg: i64) {
    println!("--- {}({}) x {} iters ---", fn_name, arg, iters);

    let mut parser = Parser::new(SOURCE);
    let statements = parser.parse().expect("parse");

    // Tree-walk timing.
    let mut tw_interp = Interpreter::new();
    tw_interp.execute(statements.clone()).expect("tw exec");
    let (tw_min_ns, tw_med_ns, tw_mean_ns) = time_loop(iters, || {
        let _ = tw_interp
            .call_function_with_values(fn_name, &[Value::HInt(HInt::new(arg))])
            .expect("tw call");
    });
    println!(
        "  tree-walk  min={:>8.1}ns  median={:>8.1}ns  mean={:>8.1}ns",
        tw_min_ns, tw_med_ns, tw_mean_ns
    );

    // JIT timing.
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let context = Context::create();
    let jit = JitContext::new(&context).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let jf = jitted
        .get(fn_name)
        .expect("expected fn to be JIT-eligible in Session E source");
    let (jit_min_ns, jit_med_ns, jit_mean_ns) = time_loop(iters, || {
        let _ = jf.call(&[arg]).expect("jit call");
    });
    println!(
        "  JIT        min={:>8.1}ns  median={:>8.1}ns  mean={:>8.1}ns",
        jit_min_ns, jit_med_ns, jit_mean_ns
    );

    if jit_med_ns > 0.0 {
        let speedup = tw_med_ns / jit_med_ns;
        println!(
            "  → JIT vs tree-walk: {:.1}x faster (median)",
            speedup
        );
    }
}

/// Time `f` `iters` times. Returns (min ns/call, median ns/call, mean
/// ns/call). Uses one outer Instant::now() to amortize syscall
/// overhead; per-call ns is total_ns / iters for min, but for median
/// we sample chunks of ~iters/100 calls and pick the median chunk's
/// per-call rate.
fn time_loop<F: FnMut()>(iters: usize, mut f: F) -> (f64, f64, f64) {
    let chunk_count = 100;
    let chunk_size = iters / chunk_count;
    let chunk_size = chunk_size.max(1);
    let actual_iters = chunk_size * chunk_count;
    let mut per_chunk_ns: Vec<f64> = Vec::with_capacity(chunk_count);
    let outer_start = Instant::now();
    for _ in 0..chunk_count {
        let start = Instant::now();
        for _ in 0..chunk_size {
            f();
        }
        let dt = start.elapsed().as_nanos() as f64;
        per_chunk_ns.push(dt / chunk_size as f64);
    }
    let total_ns = outer_start.elapsed().as_nanos() as f64;
    per_chunk_ns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = per_chunk_ns[0];
    let median = per_chunk_ns[chunk_count / 2];
    let mean = total_ns / actual_iters as f64;
    (min, median, mean)
}
