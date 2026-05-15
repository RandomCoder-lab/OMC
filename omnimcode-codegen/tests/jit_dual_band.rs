//! Session C — dual-band (HBit) lowering roundtrip tests.
//!
//! Each test builds a CompiledFunction, lowers it through BOTH the
//! scalar `lower_function` and the dual-band `lower_function_dual_band`,
//! JIT-compiles both, calls each with the same inputs, and asserts
//! they produce identical outputs.
//!
//! The dual-band version is also inspected at the LLVM IR level —
//! we verify the emitted IR contains `<2 x i64>` vector ops, proving
//! both bands are being computed in parallel.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::ast::Pos;
use omnimcode_core::bytecode::{CompiledFunction, Const, Op};

fn skeleton(name: &str, params: Vec<&str>, ops: Vec<Op>, constants: Vec<Const>) -> CompiledFunction {
    let n = ops.len();
    let param_types = vec![None; params.len()];
    CompiledFunction {
        name: name.to_string(),
        params: params.into_iter().map(String::from).collect(),
        param_types,
        return_type: None,
        op_positions: vec![Pos::unknown(); n],
        call_cache: (0..n).map(|_| std::cell::Cell::new(0)).collect(),
        ops,
        constants,
    }
}

#[test]
fn hbit_double_matches_scalar() {
    // fn double(x) { return x + x; } — should produce the same result
    // in both bands AND match the scalar lowering.
    let f = skeleton(
        "double",
        vec!["x"],
        vec![Op::LoadParam(0), Op::LoadParam(0), Op::Add, Op::Return],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("scalar lower");
    jit.lower_function_dual_band(&f).expect("hbit lower");

    unsafe {
        let scalar = jit.get_i64_i64("double").expect("scalar fn");
        let hbit = jit.get_i64_i64("double_hbit").expect("hbit fn");
        for x in &[0i64, 1, 21, -7, 1000, -1_000_000] {
            assert_eq!(scalar.call(*x), hbit.call(*x), "mismatch at x={}", x);
        }
    }
}

#[test]
fn hbit_factorial_matches_scalar() {
    // Recursive fn — the dual-band version internally calls back into
    // itself with scalar args (extracting α at the call boundary).
    let f = skeleton(
        "factorial",
        vec!["n"],
        vec![
            Op::LoadParam(0),
            Op::LoadConst(0),
            Op::Le,
            Op::JumpIfFalse(3),
            Op::Pop,
            Op::LoadConst(0),
            Op::Return,
            Op::Pop,
            Op::LoadParam(0),
            Op::LoadParam(0),
            Op::LoadConst(0),
            Op::Sub,
            Op::Call("factorial".into(), 1),
            Op::Mul,
            Op::Return,
        ],
        vec![Const::Int(1)],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("scalar lower");
    jit.lower_function_dual_band(&f).expect("hbit lower");

    unsafe {
        let scalar = jit.get_i64_i64("factorial").expect("scalar fn");
        let hbit = jit.get_i64_i64("factorial_hbit").expect("hbit fn");
        for n in 0..=12 {
            let s = scalar.call(n);
            let h = hbit.call(n);
            assert_eq!(s, h, "factorial({}) scalar={} hbit={}", n, s, h);
        }
    }
}

#[test]
fn hbit_sum_to_n_matches_scalar() {
    // While loop + locals (s and k) get exercised through allocas
    // of <2 x i64> type rather than i64.
    let f = skeleton(
        "sum_to_n",
        vec!["n"],
        vec![
            Op::LoadConst(0),
            Op::StoreVar("s".into()),
            Op::LoadConst(1),
            Op::StoreVar("k".into()),
            Op::LoadVar("k".into()),
            Op::LoadParam(0),
            Op::Le,
            Op::JumpIfFalse(10),
            Op::Pop,
            Op::LoadVar("s".into()),
            Op::LoadVar("k".into()),
            Op::Add,
            Op::AssignVar("s".into()),
            Op::LoadVar("k".into()),
            Op::LoadConst(1),
            Op::Add,
            Op::AssignVar("k".into()),
            Op::Jump(-14),
            Op::Pop,
            Op::LoadVar("s".into()),
            Op::Return,
        ],
        vec![Const::Int(0), Const::Int(1)],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("scalar lower");
    jit.lower_function_dual_band(&f).expect("hbit lower");

    unsafe {
        let scalar = jit.get_i64_i64("sum_to_n").expect("scalar fn");
        let hbit = jit.get_i64_i64("sum_to_n_hbit").expect("hbit fn");
        for n in &[0i64, 1, 10, 100, 1000] {
            assert_eq!(scalar.call(*n), hbit.call(*n), "sum_to_n({})", n);
        }
    }
}

#[test]
fn hbit_emitted_ir_contains_vector_ops() {
    // Architectural proof: the dual-band lowering really does emit
    // `<2 x i64>` ops, not scalar ones. Dump the module IR and inspect.
    let f = skeleton(
        "double",
        vec!["x"],
        vec![Op::LoadParam(0), Op::LoadParam(0), Op::Add, Op::Return],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function_dual_band(&f).expect("hbit lower");

    let ir = jit.module.print_to_string().to_string();
    assert!(
        ir.contains("<2 x i64>"),
        "expected dual-band IR to contain `<2 x i64>` vector type; got:\n{}",
        ir
    );
    // Vector add should be present as `add <2 x i64>` (LLVM textual form).
    assert!(
        ir.contains("add <2 x i64>"),
        "expected packed vector add; got:\n{}",
        ir
    );
    // The fn name should be suffixed with `_hbit` so it doesn't collide
    // with a scalar `double` in the same module.
    assert!(ir.contains("define i64 @double_hbit"), "expected _hbit fn; got:\n{}", ir);
}

#[test]
fn hbit_max_with_branches() {
    // if/else over <2 x i64> — the branch decision extracts α only
    // (since control flow is determined by the classical value), but
    // the operands and result are still vector-typed.
    let f = skeleton(
        "max",
        vec!["a", "b"],
        vec![
            Op::LoadParam(0),
            Op::LoadParam(1),
            Op::Gt,
            Op::JumpIfFalse(3),
            Op::Pop,
            Op::LoadParam(0),
            Op::Return,
            Op::Pop,
            Op::LoadParam(1),
            Op::Return,
        ],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("scalar lower");
    jit.lower_function_dual_band(&f).expect("hbit lower");

    unsafe {
        let scalar = jit.get_i64_i64_i64("max").expect("scalar fn");
        let hbit = jit.get_i64_i64_i64("max_hbit").expect("hbit fn");
        for &(a, b) in &[(7i64, 3i64), (3, 7), (5, 5), (-10, -3), (i64::MIN, 0)] {
            assert_eq!(scalar.call(a, b), hbit.call(a, b), "max({}, {})", a, b);
        }
    }
}
