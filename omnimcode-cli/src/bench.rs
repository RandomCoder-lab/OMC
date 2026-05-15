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

# --- Path A.1: harmony-gated branch elision ---
# Two execution paths: a cheap one (just doubles) and an expensive
# one (sum-to-100, ~100 iter loop). The `predicted` fn uses harmony
# of phi_shadow(x) as a runtime signal: if bands stay close to an
# attractor, take the cheap path; otherwise fall to expensive.
#
# `no_pred_always_expensive` runs the expensive path unconditionally
# (no harmony check, no shadow). Comparing predicted() to it tells
# us what @predict actually buys when the harmony signal is high.
fn cheap_path(x) {
    return x + x;
}
fn expensive_path(x) {
    h s = 0;
    h k = 1;
    while k <= 100 {
        s = s + k;
        k = k + 1;
    }
    return s + x;
}
fn predicted(x) {
    h y = phi_shadow(x);
    if harmony(y) >= 500 {
        return cheap_path(x);
    }
    return expensive_path(x);
}
fn no_pred_always_expensive(x) {
    return expensive_path(x);
}
fn no_pred_always_cheap(x) {
    return cheap_path(x);
}

# --- Path A.3: same workload, four execution modes ---
# A loop wrapper around factorial(12). Lets the VM and tree-walk
# benches measure on the same bytecode shape as JIT does. Per-iter
# time = total_call_time / N_INNER.
fn bench_loop(iters) {
    h sum = 0;
    h k = 0;
    while k < iters {
        sum = sum + factorial(12);
        k = k + 1;
    }
    return sum;
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
    println!("=== Path A.3: same workload, four execution modes ===");
    println!("Workload: bench_loop(N) = sum factorial(12) over N inner iters.");
    println!();
    bench_four_modes(50_000);

    println!();
    println!("=== Path A.1: harmony-gated branch elision ===");
    println!("Two regimes:");
    println!("  - HIGH-harmony input (x=0 → α=β=0 → harmony=1000)");
    println!("    `predicted` should take the cheap branch.");
    println!("  - LOW-harmony input (x=42 → α=42, β=phi_fold(42)*1000=957");
    println!("    → diff 915, near attractor 987 dist 72 → harmony ≈ 14)");
    println!("    `predicted` should fall to the expensive branch.");
    println!();
    bench_predict(iters);

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

fn bench_four_modes(n_inner: usize) {
    use omnimcode_codegen::JittedFn;
    use omnimcode_core::value::HInt;
    use std::collections::HashMap;
    use std::rc::Rc;

    let n_inner_i = n_inner as i64;
    println!("--- N_INNER = {} (inner loop count) ---", n_inner);

    // Mode 1: tree-walk only.
    {
        let mut parser = Parser::new(SOURCE);
        let statements = parser.parse().expect("parse");
        let mut interp = Interpreter::new();
        interp.execute(statements).expect("exec");
        let start = Instant::now();
        let _ = interp
            .call_function_with_values("bench_loop", &[Value::HInt(HInt::new(n_inner_i))])
            .expect("call");
        let total_ns = start.elapsed().as_nanos() as f64;
        println!(
            "  tree-walk          total={:>10.2}ms  per-iter={:>10.1}ns",
            total_ns / 1.0e6,
            total_ns / n_inner as f64
        );
    }

    // Mode 2: bytecode VM. Compose a tiny program whose `__main__` is
    // `bench_loop(N)` and run it through Vm::run_module. The VM sets
    // up its own scope/dispatch, so we measure the run_module call.
    {
        let mut parser = Parser::new(SOURCE);
        let mut statements = parser.parse().expect("parse");
        // Append a top-level call so __main__ runs bench_loop(N).
        let extra = format!("h __vm_result = bench_loop({});", n_inner_i);
        let mut extra_stmts = Parser::new(&extra).parse().expect("parse extra");
        statements.append(&mut extra_stmts);
        let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
        let mut vm = omnimcode_core::vm::Vm::new();
        vm.interp_mut().register_user_functions(&statements);
        let start = Instant::now();
        let _ = vm.run_module(&module).expect("run_module");
        let total_ns = start.elapsed().as_nanos() as f64;
        println!(
            "  bytecode VM        total={:>10.2}ms  per-iter={:>10.1}ns",
            total_ns / 1.0e6,
            total_ns / n_inner as f64
        );
    }

    // Mode 3: JIT-via-dispatch. Tree-walk runs the outer loop; each
    // factorial(12) call is intercepted by the JIT dispatch hook and
    // routed through native code. This is what the CLI's
    // OMC_HBIT_JIT=1 path produces for real OMC programs.
    {
        let mut parser = Parser::new(SOURCE);
        let statements = parser.parse().expect("parse");
        let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
        let context = Context::create();
        let jit = JitContext::new(&context).expect("jit");
        let jitted = jit.jit_module(&module).expect("jit_module");
        let jitted_for_hook: HashMap<String, JittedFn> = jitted.clone();
        let dispatch: omnimcode_core::interpreter::JitDispatch = Rc::new(
            move |name: &str, args: &[Value]| {
                let jf = jitted_for_hook.get(name)?;
                if args.len() != jf.arity {
                    return None;
                }
                let mut int_args = Vec::with_capacity(args.len());
                for a in args {
                    match a {
                        Value::HInt(h) => int_args.push(h.value),
                        Value::Bool(b) => int_args.push(if *b { 1 } else { 0 }),
                        _ => return None,
                    }
                }
                jf.call(&int_args).map(|r| Ok(Value::HInt(HInt::new(r))))
            },
        );
        let mut interp = Interpreter::new();
        interp.set_jit_dispatch(Some(dispatch));
        interp.execute(statements).expect("exec");
        let start = Instant::now();
        let _ = interp
            .call_function_with_values("bench_loop", &[Value::HInt(HInt::new(n_inner_i))])
            .expect("call");
        let total_ns = start.elapsed().as_nanos() as f64;
        println!(
            "  JIT-via-dispatch   total={:>10.2}ms  per-iter={:>10.1}ns  (loop is tree-walk, factorial is JIT)",
            total_ns / 1.0e6,
            total_ns / n_inner as f64
        );
    }

    // Mode 4: JIT-direct. Skip OMC entirely for the loop — call
    // factorial's fn pointer in a native Rust loop. This is the
    // theoretical best (no OMC dispatch on the hot path).
    {
        let mut parser = Parser::new(SOURCE);
        let statements = parser.parse().expect("parse");
        let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
        let context = Context::create();
        let jit = JitContext::new(&context).expect("jit");
        let jitted = jit.jit_module(&module).expect("jit_module");
        let factorial = jitted.get("factorial").expect("factorial JIT'd");
        let start = Instant::now();
        let mut sum: i64 = 0;
        for _ in 0..n_inner {
            sum = sum.wrapping_add(factorial.call(&[12]).expect("call"));
        }
        let total_ns = start.elapsed().as_nanos() as f64;
        let _ = sum;
        println!(
            "  JIT-direct         total={:>10.2}ms  per-iter={:>10.1}ns  (Rust loop, no OMC dispatch)",
            total_ns / 1.0e6,
            total_ns / n_inner as f64
        );
    }
}

fn bench_predict(iters: usize) {
    let mut parser = Parser::new(SOURCE);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let context = Context::create();
    let jit = JitContext::new(&context).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");

    let predicted = jitted.get("predicted").expect("predicted JIT'd");
    let always_exp = jitted
        .get("no_pred_always_expensive")
        .expect("no_pred_always_expensive JIT'd");
    let always_cheap = jitted
        .get("no_pred_always_cheap")
        .expect("no_pred_always_cheap JIT'd");

    println!("--- Direct path costs (no harmony check, no shadow) ---");
    let (_, cheap_med, _) = time_loop(iters, || {
        let _ = always_cheap.call(&[42]).expect("call");
    });
    println!("  cheap_path                 median={:>8.1}ns", cheap_med);
    let (_, exp_med, _) = time_loop(iters, || {
        let _ = always_exp.call(&[42]).expect("call");
    });
    println!("  expensive_path             median={:>8.1}ns", exp_med);
    let cost_ratio = exp_med / cheap_med.max(1.0);
    println!(
        "  expensive/cheap ratio: {:.1}x  (cost-cut ceiling for @predict)",
        cost_ratio
    );

    println!();
    println!("--- Predicted path (phi_shadow + harmony gate) ---");
    let (_, pred_high_med, _) = time_loop(iters, || {
        let _ = predicted.call(&[0]).expect("call");
    });
    println!(
        "  predicted(x=0)   high-harmony  median={:>8.1}ns  → expected: cheap branch",
        pred_high_med
    );
    let (_, pred_low_med, _) = time_loop(iters, || {
        let _ = predicted.call(&[42]).expect("call");
    });
    println!(
        "  predicted(x=42)  low-harmony   median={:>8.1}ns  → expected: expensive branch",
        pred_low_med
    );

    println!();
    println!("--- The honest cost analysis ---");
    let pred_overhead = pred_low_med - exp_med;
    let pred_overhead_pct = (pred_overhead / exp_med) * 100.0;
    println!(
        "  Overhead of phi_shadow+harmony+branch on the LOW path: +{:.1}ns (+{:.1}%)",
        pred_overhead, pred_overhead_pct
    );
    let pred_savings = exp_med - pred_high_med;
    let pred_savings_pct = (pred_savings / exp_med) * 100.0;
    println!(
        "  Savings on the HIGH path vs expensive: -{:.1}ns ({:.1}% reduction)",
        pred_savings, pred_savings_pct
    );

    println!();
    println!("--- Break-even analysis ---");
    // pred_low_med = expensive + overhead
    // pred_high_med = cheap + overhead
    // Break-even fraction p of inputs that hit cheap branch:
    //   p * pred_high_med + (1-p) * pred_low_med  <  exp_med  (always expensive)
    //   p * (pred_high_med - pred_low_med)  <  exp_med - pred_low_med
    //   p * (pred_low_med - pred_high_med)  >  pred_low_med - exp_med
    let numerator = pred_low_med - exp_med;
    let denom = pred_low_med - pred_high_med;
    if denom > 0.0 {
        let p_breakeven = numerator / denom;
        if p_breakeven < 0.0 {
            println!(
                "  Break-even fraction: predicted ALWAYS wins ({} < 0)",
                p_breakeven
            );
        } else if p_breakeven > 1.0 {
            println!(
                "  Break-even fraction: predicted NEVER wins ({:.2} > 1.0)",
                p_breakeven
            );
        } else {
            println!(
                "  Break-even fraction: predicted wins when ≥{:.1}% of inputs are high-harmony",
                p_breakeven * 100.0
            );
        }
    } else {
        println!("  (cheap and low paths timed identically — can't compute break-even)");
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
