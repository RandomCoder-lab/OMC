// omnimcode-core/benches/interpreter_bench.rs
//
// Real benchmarks (not ad-hoc `time` runs) comparing tree-walk vs VM,
// optimizer on/off, and showing the relative cost of the harmonic
// primitives. Driven by criterion so we get statistically stable
// numbers and HTML reports under target/criterion/.
//
// Run:  cargo bench --bench interpreter_bench
// View: open target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use omnimcode_core::bytecode_opt::optimize_module;
use omnimcode_core::compiler::compile_program;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::vm::Vm;

fn parse(src: &str) -> Vec<omnimcode_core::ast::Statement> {
    let mut parser = Parser::new(src);
    parser.parse().expect("parse failed")
}

// ---------- benchmark sources ----------

/// Recursive fibonacci(20) — call-heavy, exercises function-dispatch + scope.
const RECURSIVE_FIB: &str = r#"
fn fib(n) {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}
fib(20);
"#;

/// Tight loop with arithmetic — exercises Op::Add hot path.
const TIGHT_LOOP: &str = r#"
h sum = 0;
h i = 0;
while i < 10000 {
    sum = sum + i;
    i = i + 1;
}
sum;
"#;

/// Resonance check in a loop — exercises the inlined Op::Resonance.
const RESONANCE_LOOP: &str = r#"
h count = 0;
h i = 0;
while i < 5000 {
    h r = res(i);
    if r > 0.8 {
        count = count + 1;
    }
    i = i + 1;
}
count;
"#;

/// Mixed bitwise ops — exercises Op::BitAnd / Shl / etc.
const BITWISE_LOOP: &str = r#"
h acc = 0;
h i = 1;
while i < 1000 {
    acc = acc + ((i & 255) << 1);
    i = i + 1;
}
acc;
"#;

/// Quantization-heavy workload — exercises the Phase S primitives.
const QUANTIZE_HEAVY: &str = r#"
h xs = [85, 90, 142, 230, 235, 240, 375, 380, 605, 612, 100, 150, 200];
h sum = 0;
h i = 0;
while i < 200 {
    h q = quantize(xs, 0.5);
    h m = mean_omni_weight(q);
    sum = sum + i;
    i = i + 1;
}
sum;
"#;

// ---------- bench helpers ----------

fn bench_tree_walk(c: &mut Criterion, name: &str, src: &str) {
    let stmts = parse(src);
    c.bench_function(&format!("tree_walk/{}", name), |b| {
        b.iter(|| {
            let mut interp = Interpreter::new();
            interp.execute(black_box(stmts.clone())).unwrap();
        })
    });
}

fn bench_vm(c: &mut Criterion, name: &str, src: &str) {
    let stmts = parse(src);
    let module = compile_program(&stmts).expect("compile failed");
    c.bench_function(&format!("vm/{}", name), |b| {
        b.iter(|| {
            let mut vm = Vm::new();
            vm.run_module(black_box(&module)).unwrap();
        })
    });
}

fn bench_vm_opt(c: &mut Criterion, name: &str, src: &str) {
    let stmts = parse(src);
    let mut module = compile_program(&stmts).expect("compile failed");
    optimize_module(&mut module);
    c.bench_function(&format!("vm_opt/{}", name), |b| {
        b.iter(|| {
            let mut vm = Vm::new();
            vm.run_module(black_box(&module)).unwrap();
        })
    });
}

// ---------- benchmark groups ----------

fn bench_recursive_fib(c: &mut Criterion) {
    bench_tree_walk(c, "recursive_fib", RECURSIVE_FIB);
    bench_vm(c, "recursive_fib", RECURSIVE_FIB);
    bench_vm_opt(c, "recursive_fib", RECURSIVE_FIB);
}

fn bench_tight_loop(c: &mut Criterion) {
    bench_tree_walk(c, "tight_loop", TIGHT_LOOP);
    bench_vm(c, "tight_loop", TIGHT_LOOP);
    bench_vm_opt(c, "tight_loop", TIGHT_LOOP);
}

fn bench_resonance_loop(c: &mut Criterion) {
    bench_tree_walk(c, "resonance_loop", RESONANCE_LOOP);
    bench_vm(c, "resonance_loop", RESONANCE_LOOP);
    bench_vm_opt(c, "resonance_loop", RESONANCE_LOOP);
}

fn bench_bitwise_loop(c: &mut Criterion) {
    bench_tree_walk(c, "bitwise_loop", BITWISE_LOOP);
    bench_vm(c, "bitwise_loop", BITWISE_LOOP);
    bench_vm_opt(c, "bitwise_loop", BITWISE_LOOP);
}

fn bench_quantize_heavy(c: &mut Criterion) {
    bench_tree_walk(c, "quantize_heavy", QUANTIZE_HEAVY);
    bench_vm(c, "quantize_heavy", QUANTIZE_HEAVY);
    bench_vm_opt(c, "quantize_heavy", QUANTIZE_HEAVY);
}

// Microbenchmarks: pure parser/compiler/optimizer cost on a non-trivial
// program. These help diagnose where a slowdown originated.
fn bench_pipeline_cost(c: &mut Criterion) {
    let big_src = include_str!("../../examples/phi_field_llm_demo.omc");

    c.bench_function("pipeline/parse", |b| {
        b.iter(|| {
            let mut p = Parser::new(black_box(big_src));
            p.parse().unwrap();
        })
    });

    let stmts = parse(big_src);
    c.bench_function("pipeline/compile", |b| {
        b.iter(|| {
            let _ = compile_program(black_box(&stmts)).unwrap();
        })
    });

    c.bench_function("pipeline/compile_and_optimize", |b| {
        b.iter(|| {
            let mut m = compile_program(black_box(&stmts)).unwrap();
            optimize_module(&mut m);
        })
    });
}

criterion_group!(
    benches,
    bench_recursive_fib,
    bench_tight_loop,
    bench_resonance_loop,
    bench_bitwise_loop,
    bench_quantize_heavy,
    bench_pipeline_cost,
);
criterion_main!(benches);
