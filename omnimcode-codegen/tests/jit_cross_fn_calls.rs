//! Session H end-to-end: cross-fn calls in dual-band JIT.
//!
//! Verifies that an OMC fn JIT'd in dual-band mode can call ANOTHER
//! JIT'd OMC fn in the same module. Previously (Sessions C-G) only
//! recursive self-calls worked; cross-fn calls errored out and the
//! caller silently fell back to tree-walk.
//!
//! The negative-case test from Session D
//! (`jit_rejects_cross_fn_call` in jit_roundtrip.rs) used the
//! single-fn lowerer API — that one still rejects cross-fn calls.
//! The new path is via `JitContext::jit_module`, which now declares
//! every eligible fn up-front so cross-fn calls can resolve targets
//! by name.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::parser::Parser;

#[test]
fn cross_fn_call_in_jit_module() {
    // fn helper(x) { return x * 2; }
    // fn caller(x) { return helper(x) + 1; }
    // caller(10) → helper(10)*1 + 1 → 20 + 1 → 21
    let source = r#"
        fn helper(x) {
            return x * 2;
        }
        fn caller(x) {
            return helper(x) + 1;
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    assert!(jitted.contains_key("helper"), "helper should JIT");
    assert!(jitted.contains_key("caller"), "caller should JIT");
    let caller = jitted.get("caller").expect("caller fn");
    assert_eq!(caller.call(&[10]).expect("call"), 21);
    assert_eq!(caller.call(&[100]).expect("call"), 201);
    assert_eq!(caller.call(&[0]).expect("call"), 1);
}

#[test]
fn cross_fn_call_with_recursion() {
    // Mutual recursion-ish: caller dispatches to one of two helpers
    // based on a comparison, both helpers JIT'd alongside.
    let source = r#"
        fn double(x) { return x + x; }
        fn triple(x) { return x + x + x; }
        fn dispatch(x) {
            if x > 0 {
                return double(x);
            }
            return triple(0 - x);
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let dispatch = jitted.get("dispatch").expect("dispatch JIT'd");
    assert_eq!(dispatch.call(&[5]).expect("call"), 10);   // double(5) = 10
    assert_eq!(dispatch.call(&[-7]).expect("call"), 21);  // triple(7) = 21
    assert_eq!(dispatch.call(&[0]).expect("call"), 0);    // triple(0) = 0
}

#[test]
fn cross_fn_call_with_self_recursion_inside() {
    // The called fn is itself recursive. Tests that recursion still
    // works after the cross-fn-call refactor.
    let source = r#"
        fn factorial(n) {
            if n <= 1 { return 1; }
            return n * factorial(n - 1);
        }
        fn double_fact(n) {
            return factorial(n) + factorial(n);
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = jitted.get("double_fact").expect("double_fact JIT'd");
    assert_eq!(f.call(&[5]).expect("call"), 240);   // 120 + 120
    assert_eq!(f.call(&[10]).expect("call"), 7_257_600);  // 3.6M + 3.6M
}

#[test]
fn cross_fn_call_to_unsupported_fn_skips_caller() {
    // If `caller` calls `bad` which can't be JIT'd, then `caller`
    // can't be JIT'd either — its body references a target that
    // doesn't get declared. jit_module should silently skip the
    // caller; tree-walk runs it.
    let source = r#"
        fn bad(name) {
            # uses string concat, not yet JIT'able
            return concat_many("hello, ", name);
        }
        fn caller(x) {
            h s = bad("world");
            return x + 1;
        }
        fn pure(x) {
            return x * 3;
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    // `pure` should JIT (no string ops).
    assert!(jitted.contains_key("pure"), "pure should JIT");
    let pure = jitted.get("pure").expect("pure fn");
    assert_eq!(pure.call(&[7]).expect("call"), 21);
    // `bad` and `caller` should both be absent (bad uses strings;
    // caller calls bad).
    assert!(!jitted.contains_key("bad"), "bad should NOT JIT");
    assert!(!jitted.contains_key("caller"), "caller should NOT JIT (depends on bad)");
}
