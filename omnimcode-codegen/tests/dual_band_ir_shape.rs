//! Snapshot test — asserts the exact IR shape of a dual-band fn so a
//! regression that drops the vector type (or stops emitting parallel-
//! lane ops) breaks loud. Reference shape for `double(x) = x + x`:
//!
//!   - 2x `insertelement` splats per LoadParam (α slot, then β slot)
//!   - one `add <2 x i64>` doing the parallel addition
//!   - one `extractelement` pulling α for the return
//!
//! LLVM will lower the `add <2 x i64>` to a single SSE2 `paddq`
//! instruction on x86-64. That's the architectural payoff: both
//! bands compute in one machine instruction.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::ast::Pos;
use omnimcode_core::bytecode::{CompiledFunction, Op};

#[test]
fn dual_band_ir_shape_for_double() {
    let ops = vec![Op::LoadParam(0), Op::LoadParam(0), Op::Add, Op::Return];
    let n = ops.len();
    let f = CompiledFunction {
        name: "double".into(),
        params: vec!["x".into()],
        param_types: vec![None],
        return_type: None,
        op_positions: vec![Pos::unknown(); n],
        pragmas: Vec::new(),
        call_cache: (0..n).map(|_| std::cell::Cell::new(0)).collect(),
        ops,
        constants: vec![],
    };
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function_dual_band(&f).expect("hbit lower");
    let ir = jit.module.print_to_string().to_string();

    // Required IR markers (architecturally load-bearing).
    let must_contain = [
        "define i64 @double_hbit(i64",   // scalar-in/scalar-out fn signature
        "insertelement <2 x i64>",       // splat scalar -> vector
        "add <2 x i64>",                 // parallel-lane addition
        "extractelement <2 x i64>",      // unsplat for return
    ];
    for m in must_contain {
        assert!(
            ir.contains(m),
            "dual-band IR missing required pattern `{}`; got:\n{}",
            m,
            ir
        );
    }
}
