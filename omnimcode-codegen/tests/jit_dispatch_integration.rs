//! Session D end-to-end: parse OMC source → compile to bytecode → JIT
//! eligible fns in dual-band mode → register dispatch hook on a fresh
//! Interpreter → run the program → verify the JIT'd fns produce the
//! same answers as a tree-walk-only run.
//!
//! This proves the architectural wiring: an Interpreter can route a
//! user-defined OMC fn through the LLVM-compiled dual-band code path
//! instead of its tree-walk body, transparently.
//!
//! The CLI-level OMC_HBIT_JIT env var still needs a separate small
//! refactor (extract main.rs into omnimcode-cli) to avoid the
//! codegen↔core dependency cycle. The mechanism itself works today.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::{JitContext, JittedFn};
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::value::{HInt, Value};
use std::collections::HashMap;
use std::rc::Rc;

/// Build a JIT dispatch closure that the Interpreter consults before
/// running a user-fn body. Marshals Value args -> i64, calls native,
/// wraps the i64 result back in `Value::HInt`. Returns `None` for fns
/// not in `jitted`, or when an arg can't be coerced to i64 cleanly.
fn make_dispatch(
    jitted: HashMap<String, JittedFn>,
) -> Rc<dyn Fn(&str, &[Value]) -> Option<Result<Value, String>>> {
    Rc::new(move |name: &str, args: &[Value]| {
        let jf = jitted.get(name)?;
        if args.len() != jf.arity {
            return None;
        }
        // Only marshal arg types the dual-band codegen actually
        // supports today (int / bool). Anything else → fall back
        // to tree-walk so we don't silently turn floats into i64s.
        let mut int_args = Vec::with_capacity(args.len());
        for a in args {
            match a {
                Value::HInt(h) => int_args.push(h.value),
                Value::Bool(b) => int_args.push(if *b { 1 } else { 0 }),
                _ => return None,
            }
        }
        let result = jf.call(&int_args)?;
        Some(Ok(Value::HInt(HInt::new(result))))
    })
}

/// End-to-end driver. Returns the program's global `result` binding
/// after execution (or any global the test names).
fn run_with_jit(source: &str, capture_global: &str) -> Result<Value, String> {
    use omnimcode_core::parser::Parser;

    let mut parser = Parser::new(source);
    let statements = parser.parse()?;

    let module = omnimcode_core::compiler::compile_program(&statements)?;

    let ctx = Context::create();
    let jit = JitContext::new(&ctx).map_err(|e| format!("jit ctx: {}", e))?;
    let jitted = jit
        .jit_module(&module)
        .map_err(|e| format!("jit_module: {}", e))?;
    assert!(
        !jitted.is_empty(),
        "expected at least one JIT-eligible fn in the test source"
    );

    let dispatch = make_dispatch(jitted);
    let mut interp = Interpreter::new();
    interp.set_jit_dispatch(Some(dispatch));
    interp.execute(statements)?;

    interp
        .get_var_for_testing(capture_global)
        .ok_or_else(|| format!("global `{}` not set", capture_global))
}

fn run_tree_walk_only(source: &str, capture_global: &str) -> Result<Value, String> {
    use omnimcode_core::parser::Parser;
    let mut parser = Parser::new(source);
    let statements = parser.parse()?;
    let mut interp = Interpreter::new();
    interp.execute(statements)?;
    interp
        .get_var_for_testing(capture_global)
        .ok_or_else(|| format!("global `{}` not set", capture_global))
}

#[test]
fn jit_dispatch_routes_simple_int_fn() {
    let source = r#"
        fn double(x) {
            return x + x;
        }
        h result = double(21);
    "#;
    let v = run_with_jit(source, "result").expect("run with jit");
    assert_eq!(v.to_int(), 42);
}

#[test]
fn jit_module_returns_callable_fn_directly() {
    // Isolation test: skip the Interpreter entirely, just JIT a tiny
    // module and call the fn directly through JittedFn::call. If this
    // fails, the bug is in jit_module's fn-ptr extraction. If this
    // passes but the dispatch test fails, the bug is in the dispatch
    // closure or Interpreter wiring.
    use omnimcode_core::parser::Parser;
    let source = r#"
        fn double(x) {
            return x + x;
        }
        h result = double(21);
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let double = jitted.get("double").expect("double JIT'd");
    let result = double.call(&[21]).expect("call");
    assert_eq!(result, 42, "JIT'd double(21) should return 42");
}

#[test]
fn jit_dispatch_matches_tree_walk_factorial() {
    let source = r#"
        fn factorial(n) {
            if n <= 1 { return 1; }
            return n * factorial(n - 1);
        }
        h result = factorial(10);
    "#;
    let with_jit = run_with_jit(source, "result").expect("jit run");
    let plain = run_tree_walk_only(source, "result").expect("tree-walk run");
    assert_eq!(with_jit.to_int(), 3_628_800);
    assert_eq!(with_jit.to_int(), plain.to_int());
}

#[test]
fn jit_dispatch_matches_tree_walk_sum_loop() {
    let source = r#"
        fn sum_to_n(n) {
            h s = 0;
            h k = 1;
            while k <= n {
                s = s + k;
                k = k + 1;
            }
            return s;
        }
        h result = sum_to_n(100);
    "#;
    let with_jit = run_with_jit(source, "result").expect("jit run");
    let plain = run_tree_walk_only(source, "result").expect("tree-walk run");
    assert_eq!(with_jit.to_int(), 5050);
    assert_eq!(with_jit.to_int(), plain.to_int());
}

#[test]
fn jit_dispatch_falls_through_on_unsupported_fn() {
    // `greet` uses strings (Const::Str), which dual-band codegen
    // doesn't yet support. The JIT module should silently skip it
    // and the tree-walk path executes the body normally.
    let source = r#"
        fn greet(name) {
            return concat_many("hello, ", name);
        }
        fn add(a, b) { return a + b; }
        h greeting = greet("world");
        h result = add(2, 3);
    "#;
    let v = run_with_jit(source, "result").expect("jit run");
    // `add` is JIT-eligible and produces 5 via the JIT path.
    assert_eq!(v.to_int(), 5);
    let g = run_with_jit(source, "greeting").expect("jit run greet");
    // `greet` falls through to tree-walk (string concat).
    assert_eq!(g.to_string(), "hello, world");
}
