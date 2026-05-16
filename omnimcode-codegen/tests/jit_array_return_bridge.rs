//! L1.6 output-side bridge: a JIT'd fn marked with
//! `@jit_returns_array_int` allocates a frame array, calls
//! `omc_arr_heapify` before its `Op::Return`, returns the heap
//! pointer as i64. The dispatch boundary in omnimcode-cli/src/main.rs
//! materializes Value::Array from that pointer and calls
//! `omc_arr_free`.
//!
//! These tests don't go through the omnimcode-cli dispatch (that's
//! an integration boundary). They JIT the fn directly and call
//! through JittedFn::call, then materialize from the i64 return
//! using the same logic the dispatch uses.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::{JitContext, JittedFn};

/// Compile + JIT a single fn from source, return the JittedFn.
fn jit_one(source: &str, fn_name: &str) -> (Context, JittedFn) {
    use omnimcode_core::parser::Parser;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    // Move the context to the stack frame the caller owns, then
    // construct JitContext borrowing from it. We return both so
    // the caller can hold them lockstep.
    //
    // Inkwell needs ctx by reference with the same 'ctx lifetime
    // as the JitContext; the simplest correct lifetime story is:
    // build the JitContext inside this fn, leak via std::mem::transmute
    // so the returned JittedFn outlives the ctx ref. We accept that
    // unsafety locally — the test process exits shortly anyway.
    let jit_ctx: JitContext<'static> = unsafe {
        std::mem::transmute(JitContext::new(&ctx).expect("jit ctx"))
    };
    let jitted = jit_ctx.jit_module(&module).expect("jit_module");
    let jf = *jitted.get(fn_name).expect("fn JIT'd");
    // Leak the JitContext so the JittedFn's fn_ptr stays valid.
    Box::leak(Box::new(jit_ctx));
    (ctx, jf)
}

/// Materialize a Value::Array equivalent (Vec<i64>) from the heap pointer
/// returned by an @jit_returns_array_int fn. Mirrors the dispatch
/// closure's materialization logic; frees the heap allocation.
unsafe fn materialize(heap_ptr: i64) -> Vec<i64> {
    let p = heap_ptr as *const i64;
    let len = *p as usize;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(*p.add(i + 1));
    }
    omnimcode_codegen::omc_arr_free(heap_ptr);
    out
}

#[test]
fn jit_returns_array_int_singleton() {
    let source = r#"
        @jit_returns_array_int
        fn one_elem() {
            h arr = [42];
            return arr;
        }
    "#;
    let (_ctx, jf) = jit_one(source, "one_elem");
    assert!(jf.returns_array_int, "pragma should set the flag");
    let heap_ptr = jf.call(&[]).expect("call");
    let v = unsafe { materialize(heap_ptr) };
    assert_eq!(v, vec![42]);
}

#[test]
fn jit_returns_array_int_loop_built() {
    let source = r#"
        @jit_returns_array_int
        fn build_arr(n) {
            h arr = [0, 0, 0, 0, 0];
            h i = 0;
            while i < 5 {
                arr[i] = i * n;
                i = i + 1;
            }
            return arr;
        }
    "#;
    let (_ctx, jf) = jit_one(source, "build_arr");
    assert!(jf.returns_array_int);
    let heap_ptr = jf.call(&[3]).expect("call");
    let v = unsafe { materialize(heap_ptr) };
    assert_eq!(v, vec![0, 3, 6, 9, 12]);
}

#[test]
fn jit_returns_array_int_zeros() {
    let source = r#"
        @jit_returns_array_int
        fn make_zeros() {
            return [0, 0, 0, 0, 0, 0, 0, 0];
        }
    "#;
    let (_ctx, jf) = jit_one(source, "make_zeros");
    let heap_ptr = jf.call(&[]).expect("call");
    let v = unsafe { materialize(heap_ptr) };
    assert_eq!(v, vec![0; 8]);
}

#[test]
fn jit_returns_array_int_size_dependent() {
    // Allocate based on a param. Each call creates a fresh frame
    // array; heapify copies it independently per call.
    let source = r#"
        @jit_returns_array_int
        fn squares(k) {
            h arr = [0, 0, 0, 0];
            h i = 0;
            while i < 4 {
                arr[i] = (i + k) * (i + k);
                i = i + 1;
            }
            return arr;
        }
    "#;
    let (_ctx, jf) = jit_one(source, "squares");
    let h1 = jf.call(&[1]).expect("call(1)");
    let v1 = unsafe { materialize(h1) };
    assert_eq!(v1, vec![1, 4, 9, 16]);
    let h2 = jf.call(&[10]).expect("call(10)");
    let v2 = unsafe { materialize(h2) };
    assert_eq!(v2, vec![100, 121, 144, 169]);
}

#[test]
fn jit_no_pragma_returns_scalar() {
    // Sanity: without the pragma, the existing scalar-return
    // contract is preserved (returns_array_int is false).
    let source = r#"
        fn add(a, b) { return a + b; }
    "#;
    let (_ctx, jf) = jit_one(source, "add");
    assert!(!jf.returns_array_int);
    assert_eq!(jf.call(&[10, 20]).unwrap(), 30);
}
