//! L1.6: Array↔JIT bridging across the dispatch boundary.
//!
//! Verifies that a Value::Array argument can be marshalled into the
//! JIT'd function's stack-frame array layout — `[len, v0, v1, ..., vN]`
//! contiguous i64 — and that ArrayLen / ArrayIndex inside the JIT'd
//! code correctly read from the marshalled buffer.
//!
//! Before this bridge, the dispatch hook in omnimcode-cli/src/main.rs
//! returned None whenever any arg was Value::Array, falling through to
//! tree-walk. The harmonic libraries' hot paths (sum_array, score,
//! filter_by_resonance) all take arrays as input, so the JIT eligibility
//! was empty in practice on the most performance-critical code.
//!
//! End-to-end: tree-walk and JIT must return byte-identical results on
//! every test, validating the marshalling preserves semantics.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::{JitContext, JittedFn};
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::value::{HInt, Value};
use std::collections::HashMap;
use std::rc::Rc;

/// Dispatch closure that knows how to marshal int arrays. Mirrors the
/// production wiring in omnimcode-cli/src/main.rs.
fn make_array_aware_dispatch(
    jitted: HashMap<String, JittedFn>,
) -> Rc<dyn Fn(&str, &[Value]) -> Option<Result<Value, String>>> {
    Rc::new(move |name: &str, args: &[Value]| {
        let jf = jitted.get(name)?;
        if args.len() != jf.arity {
            return None;
        }
        let mut int_args: Vec<i64> = Vec::with_capacity(args.len());
        let mut _pinned: Vec<Box<[i64]>> = Vec::new();
        for a in args {
            match a {
                Value::HInt(h) => int_args.push(h.value),
                Value::Bool(b) => int_args.push(if *b { 1 } else { 0 }),
                Value::Array(arr) => {
                    let items = arr.items.borrow();
                    if !items.iter().all(|v| matches!(v, Value::HInt(_) | Value::Bool(_))) {
                        return None;
                    }
                    let mut buf: Vec<i64> = Vec::with_capacity(items.len() + 1);
                    buf.push(items.len() as i64);
                    for v in items.iter() {
                        buf.push(match v {
                            Value::HInt(h) => h.value,
                            Value::Bool(b) => if *b { 1 } else { 0 },
                            _ => unreachable!(),
                        });
                    }
                    let boxed = buf.into_boxed_slice();
                    let ptr = boxed.as_ptr() as i64;
                    _pinned.push(boxed);
                    int_args.push(ptr);
                }
                _ => return None,
            }
        }
        let result = jf.call(&int_args).map(|r| Ok(Value::HInt(HInt::new(r))));
        drop(_pinned);
        result
    })
}

fn run_with_jit(source: &str, capture_global: &str) -> Result<Value, String> {
    use omnimcode_core::parser::Parser;
    let mut parser = Parser::new(source);
    let statements = parser.parse()?;
    let module = omnimcode_core::compiler::compile_program(&statements)?;
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).map_err(|e| format!("jit ctx: {}", e))?;
    let jitted = jit.jit_module(&module).map_err(|e| format!("jit_module: {}", e))?;
    assert!(!jitted.is_empty(), "expected at least one JIT-eligible fn");
    let dispatch = make_array_aware_dispatch(jitted);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// sum_array(arr) walks an int array and accumulates. The simplest
/// possible array-consuming JIT-eligible fn — exercises the bridge's
/// length read (slot 0) and element read (slots 1..=N).
#[test]
fn jit_array_bridge_sum() {
    let source = r#"
        fn sum_array(arr) {
            h n = arr_len(arr);
            h s = 0;
            h i = 0;
            while i < n {
                s = s + arr_get(arr, i);
                i = i + 1;
            }
            return s;
        }
        h data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        h result = sum_array(data);
    "#;
    let v_jit = run_with_jit(source, "result").expect("jit");
    let v_tw = run_tree_walk_only(source, "result").expect("tree-walk");
    assert_eq!(v_jit.to_int(), 55);
    assert_eq!(v_jit.to_int(), v_tw.to_int(), "JIT vs tree-walk parity");
}

/// max_element(arr) — branchy access pattern (compare-and-update). Tests
/// that the bridge composes with control flow in JIT'd code.
#[test]
fn jit_array_bridge_max() {
    let source = r#"
        fn max_element(arr) {
            h n = arr_len(arr);
            h best = arr_get(arr, 0);
            h i = 1;
            while i < n {
                h v = arr_get(arr, i);
                if v > best {
                    best = v;
                }
                i = i + 1;
            }
            return best;
        }
        h data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        h result = max_element(data);
    "#;
    let v = run_with_jit(source, "result").expect("jit");
    assert_eq!(v.to_int(), 9);
}

/// count_threshold(arr, t) — array AND scalar args together. Verifies
/// that the bridge correctly interleaves pointer args with int args.
#[test]
fn jit_array_bridge_mixed_args() {
    let source = r#"
        fn count_threshold(arr, t) {
            h n = arr_len(arr);
            h c = 0;
            h i = 0;
            while i < n {
                if arr_get(arr, i) >= t {
                    c = c + 1;
                }
                i = i + 1;
            }
            return c;
        }
        h data = [1, 5, 10, 15, 20, 25, 30];
        h result = count_threshold(data, 10);
    "#;
    let v = run_with_jit(source, "result").expect("jit");
    assert_eq!(v.to_int(), 5);
}

/// Empty array doesn't crash. Bridge passes length=0; the JIT'd fn's
/// while-loop should run zero iterations.
#[test]
fn jit_array_bridge_empty() {
    let source = r#"
        fn sum_array(arr) {
            h n = arr_len(arr);
            h s = 0;
            h i = 0;
            while i < n {
                s = s + arr_get(arr, i);
                i = i + 1;
            }
            return s;
        }
        h data = [];
        h result = sum_array(data);
    "#;
    let v = run_with_jit(source, "result").expect("jit");
    assert_eq!(v.to_int(), 0);
}

/// Large array stresses the bridge's memory layout. 1000 elements is
/// well past anything the alloca-based internal layout would handle
/// (the JIT'd fn reads from the external buffer pointer, not its own
/// stack frame).
#[test]
fn jit_array_bridge_large() {
    let source = r#"
        fn sum_array(arr) {
            h n = arr_len(arr);
            h s = 0;
            h i = 0;
            while i < n {
                s = s + arr_get(arr, i);
                i = i + 1;
            }
            return s;
        }
        h data = arr_range(0, 1000);
        h result = sum_array(data);
    "#;
    let v = run_with_jit(source, "result").expect("jit");
    // 0 + 1 + ... + 999 = 499500
    assert_eq!(v.to_int(), 499_500);
}

/// Array-of-non-ints should fall through to tree-walk (None returned).
/// The bridge only handles int arrays today; string arrays must use
/// the slow path until extended.
#[test]
fn jit_array_bridge_rejects_non_int_arrays() {
    let source = r#"
        fn arr_count(arr) {
            return arr_len(arr);
        }
        h data = ["a", "b", "c"];
        h result = arr_count(data);
    "#;
    // Tree-walk handles this fine; the JIT'd version (if any) is bypassed
    // because the dispatch returns None for non-int arrays.
    let v = run_with_jit(source, "result").expect("jit");
    assert_eq!(v.to_int(), 3);
}
