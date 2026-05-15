//! Session A + B end-to-end JIT roundtrip tests.
//!
//! Each test hand-builds a `CompiledFunction` (no parser, no compiler —
//! we want to isolate the lowering+JIT layer), lowers it through inkwell
//! into LLVM IR, JIT-compiles it, calls the resulting native function
//! pointer, and asserts the return value.
//!
//! Session A coverage: pure i64 arithmetic with no branches.
//! Session B coverage: locals (via allocas), conditionals (JumpIfFalse),
//! loops (Jump backward), comparisons, recursion.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::ast::Pos;
use omnimcode_core::bytecode::{CompiledFunction, Const, Op};

/// Construct an empty CompiledFunction skeleton. The bytecode-execution
/// path needs `call_cache` and `op_positions` parallel to `ops`; codegen
/// doesn't read them but we keep the struct well-formed.
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

// ---------- Session A regression tests ----------

#[test]
fn jit_double_x_returns_2x() {
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
        assert_eq!(native.call(-5), -10);
    }
}

#[test]
fn jit_add_two_args() {
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
    }
}

// ---------- Session B: locals + conditionals ----------

#[test]
fn jit_max_two_args() {
    // fn max(a, b) {
    //     if a > b { return a; }
    //     return b;
    // }
    //
    // Bytecode mirroring the compiler's emission:
    //   0: LoadParam(0)       # a
    //   1: LoadParam(1)       # b
    //   2: Gt                 # a > b -> stack: [0/1]
    //   3: JumpIfFalse(3)     # offset to op 7 (3+1+3)
    //   4: Pop                # true-path cleanup (suppressed)
    //   5: LoadParam(0)
    //   6: Return
    //   7: Pop                # false-path cleanup (suppressed)
    //   8: LoadParam(1)
    //   9: Return
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
    jit.lower_function(&f).expect("lower");
    unsafe {
        let native = jit.get_i64_i64_i64("max").expect("jit fn");
        assert_eq!(native.call(7, 3), 7);
        assert_eq!(native.call(3, 7), 7);
        assert_eq!(native.call(5, 5), 5); // tie -> b
        assert_eq!(native.call(-10, -3), -3);
    }
}

#[test]
fn jit_abs_single_arg() {
    // fn abs(x) {
    //     if x < 0 { return -x; }
    //     return x;
    // }
    //
    //   0: LoadParam(0)
    //   1: LoadConst(0:=0)
    //   2: Lt                 # x < 0
    //   3: JumpIfFalse(4)     # offset to op 8 (3+1+4)
    //   4: Pop
    //   5: LoadParam(0)
    //   6: Neg
    //   7: Return
    //   8: Pop
    //   9: LoadParam(0)
    //  10: Return
    let f = skeleton(
        "abs",
        vec!["x"],
        vec![
            Op::LoadParam(0),
            Op::LoadConst(0),
            Op::Lt,
            Op::JumpIfFalse(4),
            Op::Pop,
            Op::LoadParam(0),
            Op::Neg,
            Op::Return,
            Op::Pop,
            Op::LoadParam(0),
            Op::Return,
        ],
        vec![Const::Int(0)],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function(&f).expect("lower");
    unsafe {
        let native = jit.get_i64_i64("abs").expect("jit fn");
        assert_eq!(native.call(5), 5);
        assert_eq!(native.call(-5), 5);
        assert_eq!(native.call(0), 0);
        assert_eq!(native.call(-1_000_000), 1_000_000);
    }
}

// ---------- Session B: while loop + locals ----------

#[test]
fn jit_sum_to_n_while_loop() {
    // fn sum_to_n(n) {
    //     h s = 0;
    //     h k = 1;
    //     while k <= n {
    //         s = s + k;
    //         k = k + 1;
    //     }
    //     return s;
    // }
    //
    //   0: LoadConst(0:=0)
    //   1: StoreVar("s")
    //   2: LoadConst(1:=1)
    //   3: StoreVar("k")
    //   4: LoadVar("k")          # loop start
    //   5: LoadParam(0)          # n
    //   6: Le                    # k <= n
    //   7: JumpIfFalse(10)       # offset to op 18 (7+1+10)
    //   8: Pop                   # true cleanup (suppressed)
    //   9: LoadVar("s")
    //  10: LoadVar("k")
    //  11: Add
    //  12: AssignVar("s")
    //  13: LoadVar("k")
    //  14: LoadConst(1)
    //  15: Add
    //  16: AssignVar("k")
    //  17: Jump(-14)             # back to op 4 (17+1+(-14)=4)
    //  18: Pop                   # false cleanup (suppressed)
    //  19: LoadVar("s")
    //  20: Return
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
    unsafe {
        let native = jit.get_i64_i64("sum_to_n").expect("jit fn");
        assert_eq!(native.call(10), 55); // 1+2+...+10
        assert_eq!(native.call(100), 5050);
        assert_eq!(native.call(0), 0); // loop body never executes
        assert_eq!(native.call(1), 1);
    }
}

// ---------- Session B: recursive call ----------

#[test]
fn jit_factorial_recursion() {
    // fn factorial(n) {
    //     if n <= 1 { return 1; }
    //     return n * factorial(n - 1);
    // }
    //
    //   0: LoadParam(0)
    //   1: LoadConst(0:=1)
    //   2: Le                    # n <= 1
    //   3: JumpIfFalse(3)        # offset to op 7
    //   4: Pop                   # true cleanup (suppressed)
    //   5: LoadConst(0)          # 1
    //   6: Return
    //   7: Pop                   # false cleanup (suppressed)
    //   8: LoadParam(0)          # n  (the multiplier)
    //   9: LoadParam(0)          # n  (for n-1)
    //  10: LoadConst(0)          # 1
    //  11: Sub                   # n - 1
    //  12: Call("factorial", 1)  # recursive call
    //  13: Mul                   # n * factorial(n-1)
    //  14: Return
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
    jit.lower_function(&f).expect("lower");
    unsafe {
        let native = jit.get_i64_i64("factorial").expect("jit fn");
        assert_eq!(native.call(0), 1);
        assert_eq!(native.call(1), 1);
        assert_eq!(native.call(5), 120);
        assert_eq!(native.call(10), 3_628_800);
        assert_eq!(native.call(20), 2_432_902_008_176_640_000);
    }
}

// ---------- Session A negative test still applies ----------

#[test]
fn jit_rejects_unsupported_op() {
    // Print is not yet lowered.
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

#[test]
fn jit_rejects_cross_fn_call() {
    // Session B Call only handles recursion. A call to another fn name
    // should error cleanly.
    let f = skeleton(
        "caller",
        vec!["x"],
        vec![
            Op::LoadParam(0),
            Op::Call("some_other_fn".into(), 1),
            Op::Return,
        ],
        vec![],
    );
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    let err = jit.lower_function(&f).expect_err("should fail");
    assert!(err.contains("only supports recursive self-call"), "got: {}", err);
}
