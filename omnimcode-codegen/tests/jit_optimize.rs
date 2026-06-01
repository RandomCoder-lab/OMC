//! JIT IR-optimization tests (JitContext::optimize).
//!
//! optimize() runs a curated scalar pass pipeline (mem2reg + instcombine/reassociate/
//! gvn/simplifycfg) via LLVM's new pass manager. The pipeline DELIBERATELY excludes Loop
//! Strength Reduction — LSR crashed at OptimizationLevel::Default on the dual-band lowerer's
//! LCSSA-form loops (why the engine runs at None). These tests prove the pipeline (a) runs
//! without crashing on real lowered IR including a LOOP-with-locals (the danger zone), and
//! (b) PRESERVES semantics — the optimized JIT'd function still returns the correct value.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::ast::Pos;
use omnimcode_core::bytecode::{CompiledFunction, Const, Op};
use omnimcode_core::parser::Parser;

fn skeleton(name: &str, params: Vec<&str>, ops: Vec<Op>, constants: Vec<Const>) -> CompiledFunction {
    let n = ops.len();
    let param_types = vec![None; params.len()];
    CompiledFunction {
        name: name.to_string(),
        params: params.into_iter().map(String::from).collect(),
        param_types,
        return_type: None,
        op_positions: vec![Pos::unknown(); n],
        pragmas: Vec::new(),
        call_cache: (0..n).map(|_| std::cell::Cell::new(0)).collect(),
        ops,
        constants,
    }
}

#[test]
fn optimize_preserves_simple_arithmetic() {
    let f = skeleton(
        "double",
        vec!["x"],
        vec![Op::LoadParam(0), Op::LoadParam(0), Op::Add, Op::Return],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("lower");
    jit.optimize().expect("optimize ok");
    unsafe {
        let native = jit.get_i64_i64("double").expect("jit fn");
        assert_eq!(native.call(21), 42, "optimized fn must still double");
        assert_eq!(native.call(-5), -10);
    }
}

#[test]
fn optimize_preserves_loop_with_locals() {
    // sum_to_n: allocas (StoreVar/LoadVar/AssignVar → mem2reg promotes) + a while LOOP
    // (the LSR danger zone). The scalar pipeline must survive it AND keep the result exact.
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
    jit.lower_function(&f).expect("lower");
    jit.optimize().expect("optimize must not error on a loop+locals fn");
    unsafe {
        let native = jit.get_i64_i64("sum_to_n").expect("jit fn");
        assert_eq!(native.call(10), 55, "optimized loop must still sum 1..10");
        assert_eq!(native.call(100), 5050);
        assert_eq!(native.call(0), 0);
        assert_eq!(native.call(1), 1);
    }
}

#[test]
fn jit_module_dispatch_with_opt_env_preserves_results() {
    // The DISPATCH path (jit_module) wires optimize() behind OMC_HBIT_JIT_OPT=1. With it on,
    // a cross-fn-call program must still build a correct dispatch table. Env leakage to other
    // parallel tests is benign — the passes preserve semantics, so any test still passes.
    std::env::set_var("OMC_HBIT_JIT_OPT", "1");
    let source = r#"
        fn helper(x) { return x * 2; }
        fn caller(x) { return helper(x) + 1; }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module (opt path)");
    let caller = jitted.get("caller").expect("caller fn");
    assert_eq!(caller.call(&[10]).expect("call"), 21, "opt dispatch must preserve caller=helper(x)+1");
    assert_eq!(caller.call(&[100]).expect("call"), 201);
    assert_eq!(caller.call(&[0]).expect("call"), 1);
    std::env::remove_var("OMC_HBIT_JIT_OPT");
}
