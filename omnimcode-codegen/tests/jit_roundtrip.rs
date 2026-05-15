//! Session A end-to-end JIT roundtrip tests.
//!
//! Each test hand-builds a `CompiledFunction` (no parser, no compiler —
//! we want to isolate the lowering+JIT layer), lowers it through inkwell
//! into LLVM IR, JIT-compiles it, calls the resulting native function
//! pointer, and asserts the return value.
//!
//! These are the foundation of every later session: HBit/dual-band/SIMD
//! work all builds on the same lower-then-JIT pipeline.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::ast::Pos;
use omnimcode_core::bytecode::{CompiledFunction, Const, Op};

/// Construct an empty CompiledFunction skeleton. The bytecode-execution
/// path needs `call_cache` and `op_positions` parallel to `ops`; codegen
/// doesn't read them but we keep the struct well-formed in case the
/// fn ever round-trips through other code.
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
fn jit_double_x_returns_2x() {
    // fn double(x) { return x + x; }
    // Bytecode: LoadParam(0), LoadParam(0), Add, Return
    let f = skeleton(
        "double",
        vec!["x"],
        vec![Op::LoadParam(0), Op::LoadParam(0), Op::Add, Op::Return],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("lower");

    unsafe {
        let native = jit.get_i64_i64("double").expect("jit fn");
        assert_eq!(native.call(21), 42);
        assert_eq!(native.call(0), 0);
        assert_eq!(native.call(-5), -10);
        assert_eq!(native.call(i64::MAX / 2), (i64::MAX / 2) * 2);
    }
}

#[test]
fn jit_add_two_args() {
    // fn add(a, b) { return a + b; }
    // Bytecode: LoadParam(0), LoadParam(1), AddInt, Return
    let f = skeleton(
        "add",
        vec!["a", "b"],
        vec![
            Op::LoadParam(0),
            Op::LoadParam(1),
            Op::AddInt,
            Op::Return,
        ],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("lower");

    unsafe {
        let native = jit.get_i64_i64_i64("add").expect("jit fn");
        assert_eq!(native.call(2, 3), 5);
        assert_eq!(native.call(-100, 50), -50);
    }
}

#[test]
fn jit_const_arithmetic() {
    // fn answer() { return 6 * 7; } — but for i64-arg signature
    // compatibility, parameterize as a one-arg fn that ignores its arg:
    // fn answer(_unused) { return 6 * 7; }
    let f = skeleton(
        "answer",
        vec!["_unused"],
        vec![
            Op::LoadConst(0),
            Op::LoadConst(1),
            Op::MulInt,
            Op::Return,
        ],
        vec![Const::Int(6), Const::Int(7)],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("lower");

    unsafe {
        let native = jit.get_i64_i64("answer").expect("jit fn");
        assert_eq!(native.call(0), 42);
    }
}

#[test]
fn jit_mixed_const_and_param() {
    // fn shift_then_double(x) { return (x + 100) * 2; }
    // = LoadParam(0), LoadConst(0:=100), Add, LoadConst(1:=2), Mul, Return
    let f = skeleton(
        "shift_then_double",
        vec!["x"],
        vec![
            Op::LoadParam(0),
            Op::LoadConst(0),
            Op::Add,
            Op::LoadConst(1),
            Op::Mul,
            Op::Return,
        ],
        vec![Const::Int(100), Const::Int(2)],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("lower");

    unsafe {
        let native = jit.get_i64_i64("shift_then_double").expect("jit fn");
        assert_eq!(native.call(0), 200);
        assert_eq!(native.call(50), 300);
        assert_eq!(native.call(-50), 100);
    }
}

#[test]
fn jit_rejects_unsupported_op() {
    // Session A doesn't support Op::Print — make sure lowering errors
    // cleanly instead of producing broken IR.
    let f = skeleton(
        "broken",
        vec!["x"],
        vec![Op::LoadParam(0), Op::Print, Op::Return],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    let err = jit.lower_function(&f).expect_err("should fail");
    assert!(err.contains("doesn't yet lower op"), "got: {}", err);
}
