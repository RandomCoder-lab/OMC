//! Session F end-to-end: phi_shadow() intrinsic in dual-band JIT.
//!
//! Verifies:
//! 1. Calling `phi_shadow(x)` in JIT'd code returns x unchanged
//!    (α band is preserved — the user-visible value).
//! 2. The dual-band IR contains the phi-fold computation chain
//!    (sitofp → fmul PHI → llvm.floor.f64 → fsub → fmul → fptosi →
//!    insertelement) that replaces β.
//! 3. Tree-walk also treats phi_shadow as pass-through (semantic
//!    parity: programs using phi_shadow run identically in both
//!    modes, only the JIT actually populates β with the shadow).

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::value::Value;

#[test]
fn phi_shadow_jit_returns_alpha_unchanged() {
    // fn shadowed(x) { return phi_shadow(x); }
    // Should return x (α band is preserved).
    let source = r#"
        fn shadowed(x) {
            return phi_shadow(x);
        }
        h result = shadowed(42);
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = jitted.get("shadowed").expect("shadowed JIT'd");
    for x in &[0i64, 1, 42, -7, 1000, -1_000_000] {
        let r = f.call(&[*x]).expect("call");
        assert_eq!(r, *x, "phi_shadow({}) should return {} (α preserved)", x, x);
    }
}

#[test]
fn phi_shadow_tree_walk_is_pass_through() {
    // Tree-walk: phi_shadow returns x. Same as JIT's α band.
    let source = r#"
        fn shadowed(x) { return phi_shadow(x); }
        h result = shadowed(42);
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let mut interp = Interpreter::new();
    interp.execute(statements).expect("exec");
    let r = interp.get_var_for_testing("result").expect("result");
    assert_eq!(r.to_int(), 42);
}

#[test]
fn phi_shadow_emits_expected_ir_chain() {
    // Architectural snapshot: the dual-band IR for a fn that uses
    // phi_shadow must contain the canonical float-conversion chain.
    let source = r#"
        fn shadowed(x) {
            return phi_shadow(x);
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    jit.lower_function_dual_band(module.functions.get("shadowed").expect("fn"))
        .expect("lower");
    let ir = jit.module.print_to_string().to_string();
    // Required IR markers for the phi-shadow chain.
    let must_contain = [
        "sitofp i64",         // signed int → double
        "fmul double",        // multiply by PHI or by 1000.0
        "@llvm.floor.f64",    // floor intrinsic declared & called
        "fsub double",        // fractional part = x_phi - floor
        "fptosi double",      // float → signed int (back to β)
        "insertelement <2 x i64>", // β replacement in vector
    ];
    for m in must_contain {
        assert!(
            ir.contains(m),
            "phi_shadow IR missing `{}`; got:\n{}",
            m,
            ir
        );
    }
}

#[test]
fn phi_shadow_in_arithmetic_does_not_break_alpha() {
    // fn f(x) {
    //     h y = phi_shadow(x);     // β diverges
    //     return y + y;             // α propagates: both lanes get +y
    // }
    // After phi_shadow, β = phi_fold(α) * 1000. Adding y to itself:
    //   α' = α + α = 2α
    //   β' = β + β = 2β (NOT phi_fold(2α) — bands maintain their own paths)
    // The user-visible result (α') should still be 2x.
    let source = r#"
        fn f(x) {
            h y = phi_shadow(x);
            return y + y;
        }
        h result = f(21);
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");

    // Tree-walk run (phi_shadow is pass-through).
    let mut tw = Interpreter::new();
    tw.execute(statements.clone()).expect("tw exec");
    let tw_result = tw.get_var_for_testing("result").expect("tw result");
    assert_eq!(tw_result.to_int(), 42);

    // JIT run — α should still come out 42.
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = jitted.get("f").expect("f JIT'd");
    assert_eq!(f.call(&[21]).expect("call"), 42);
}

#[test]
fn phi_shadow_via_dispatch_hook() {
    // End-to-end through Interpreter + dispatch hook (matches the
    // CLI's OMC_HBIT_JIT=1 code path). Verifies the JIT'd phi_shadow
    // is callable transparently via interp.execute.
    use omnimcode_codegen::JittedFn;
    use omnimcode_core::value::HInt;
    use std::collections::HashMap;
    use std::rc::Rc;

    let source = r#"
        fn shadowed(x) {
            return phi_shadow(x);
        }
        h result = shadowed(89);
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    let jitted_map = jit.jit_module(&module).expect("jit_module");

    let jitted_for_hook: HashMap<String, JittedFn> = jitted_map.clone();
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
    let r = interp.get_var_for_testing("result").expect("result");
    assert_eq!(r.to_int(), 89);
}
