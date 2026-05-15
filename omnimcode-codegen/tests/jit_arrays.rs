//! Path A.4 — read-only array support in the dual-band JIT.
//!
//! Arrays are represented as `alloca [N+1 x i64]` allocations in the
//! fn's stack frame. Slot 0 holds the length; slots 1..=N hold the
//! elements. Self-describing — ArrayLen needs no side-channel.
//!
//! On the operand stack, an array is the pointer cast to i64
//! (ptrtoint at NewArray, inttoptr at use). This fits the existing
//! Vec<VectorValue> stack convention without needing a typed enum.
//!
//! Out of scope for Path A.4 MVP:
//!   - ArrayIndexAssign (mutable writes)
//!   - Dynamic resize
//!   - Returning arrays from JIT'd fns (caller-facing signature is i64)
//!   - Multi-dimensional / nested arrays
//!
//! These are the next sessions' work. The MVP unlocks any pure-int OMC
//! fn that builds an array, reads from it, and returns a scalar.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::parser::Parser;

#[test]
fn jit_array_len_returns_correct_length() {
    let source = r#"
        fn arr5_len(unused) {
            h arr = [10, 20, 30, 40, 50];
            return arr_len(arr);
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = jitted.get("arr5_len").expect("arr5_len JIT'd");
    assert_eq!(f.call(&[0]).expect("call"), 5);
}

#[test]
fn jit_array_index_reads_correct_element() {
    let source = r#"
        fn arr5_at(idx) {
            h arr = [10, 20, 30, 40, 50];
            return arr_get(arr, idx);
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = jitted.get("arr5_at").expect("arr5_at JIT'd");
    assert_eq!(f.call(&[0]).expect("call"), 10);
    assert_eq!(f.call(&[1]).expect("call"), 20);
    assert_eq!(f.call(&[2]).expect("call"), 30);
    assert_eq!(f.call(&[3]).expect("call"), 40);
    assert_eq!(f.call(&[4]).expect("call"), 50);
}

#[test]
fn jit_array_sum_in_loop() {
    // The headline workload: sum the elements of a small array.
    // Exercises NewArray + ArrayLen + ArrayIndex inside a while loop.
    let source = r#"
        fn sum_arr(unused) {
            h arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            h sum = 0;
            h k = 0;
            while k < arr_len(arr) {
                sum = sum + arr_get(arr, k);
                k = k + 1;
            }
            return sum;
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = jitted.get("sum_arr").expect("sum_arr JIT'd");
    assert_eq!(f.call(&[0]).expect("call"), 55); // 1+2+...+10
}

#[test]
fn jit_array_via_dispatch_hook() {
    // End-to-end through Interpreter dispatch (matches CLI's
    // OMC_HBIT_JIT=1 path). Verifies arrays survive the JIT round-
    // trip when called from the user-facing tree-walk.
    use omnimcode_codegen::JittedFn;
    use omnimcode_core::interpreter::Interpreter;
    use omnimcode_core::value::{HInt, Value};
    use std::collections::HashMap;
    use std::rc::Rc;

    let source = r#"
        fn sum_arr(unused) {
            h arr = [100, 200, 300];
            h sum = 0;
            h k = 0;
            while k < arr_len(arr) {
                sum = sum + arr_get(arr, k);
                k = k + 1;
            }
            return sum;
        }
        h result = sum_arr(0);
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit = JitContext::new(&ctx).expect("jit");
    let jitted_map = jit.jit_module(&module).expect("jit_module");
    assert!(
        jitted_map.contains_key("sum_arr"),
        "sum_arr should JIT (uses NewArray, ArrayLen, ArrayIndex)"
    );
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
    assert_eq!(r.to_int(), 600);
}
