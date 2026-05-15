//! Reverse-FFI host-builtin tests.
//!
//! Verifies that an embedder can register Rust closures as OMC-callable
//! builtins, and that the dispatch works for both the tree-walk
//! interpreter and the bytecode VM. Uses a shared cell to capture
//! side effects so we can assert the host code actually ran.

use omnimcode_core::bytecode_opt::optimize_module;
use omnimcode_core::compiler::compile_program;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::value::{HArray, HInt, Value};
use omnimcode_core::vm::Vm;
use std::cell::Cell;
use std::rc::Rc;

/// Run `source` through the tree-walk interpreter with `setup_host`
/// called on the interpreter before execution. Returns the final
/// value of `__result__`.
fn run_treewalk(
    source: &str,
    setup_host: impl FnOnce(&mut Interpreter),
) -> Result<Value, String> {
    let mut parser = Parser::new(source);
    let stmts = parser.parse()?;
    let mut interp = Interpreter::new();
    setup_host(&mut interp);
    interp.execute(stmts)?;
    interp
        .get_var_for_testing("__result__")
        .ok_or_else(|| "no __result__ variable".to_string())
}

/// Run `source` through the bytecode VM with `setup_host` called
/// before execution. The Vm's internal Interpreter is what the host
/// fns end up registered on.
///
/// The VM pops its top-level scope on exit so we can't read
/// `__result__` from it the way tree-walk can. Instead, the source
/// must end with `__capture(<expr>);` where `__capture` is a host
/// fn we register to stash the result in a returned Cell.
fn run_vm_with_capture(
    source: &str,
    setup_host: impl FnOnce(&mut Interpreter),
) -> Result<Value, String> {
    let mut parser = Parser::new(source);
    let stmts = parser.parse()?;
    let module = compile_program(&stmts)?;
    let mut module = module;
    optimize_module(&mut module);
    let mut vm = Vm::new();
    let captured: Rc<std::cell::RefCell<Option<Value>>> =
        Rc::new(std::cell::RefCell::new(None));
    let captured_clone = Rc::clone(&captured);
    {
        let interp = vm.interp_mut();
        setup_host(interp);
        interp.register_builtin("__capture", move |args| {
            *captured_clone.borrow_mut() = args.first().cloned();
            Ok(Value::Null)
        });
        interp.process_imports(&stmts)?;
        interp.register_user_functions(&stmts);
        for (lname, lparams, lbody) in &module.lambda_asts {
            interp.register_lambda(lname, lparams.clone(), lbody.clone());
        }
    }
    vm.run_module(&module)?;
    let result = captured.borrow().clone();
    result.ok_or_else(|| "no __capture(...) call in source".to_string())
}

#[test]
fn host_builtin_simple_int_double_treewalk() {
    let v = run_treewalk(
        r#"
        h __result__ = double(21);
        "#,
        |interp| {
            interp.register_builtin("double", |args| {
                Ok(Value::HInt(HInt::new(args[0].to_int() * 2)))
            });
        },
    )
    .unwrap();
    assert_eq!(v.to_int(), 42);
}

#[test]
fn host_builtin_simple_int_double_vm() {
    let v = run_vm_with_capture(
        r#"
        __capture(double(21));
        "#,
        |interp| {
            interp.register_builtin("double", |args| {
                Ok(Value::HInt(HInt::new(args[0].to_int() * 2)))
            });
        },
    )
    .unwrap();
    assert_eq!(v.to_int(), 42);
}

/// Confirm side effects propagate Rust-side. The host fn writes to a
/// shared Cell; we read it after OMC execution. This is the pattern
/// PyO3 will use for round-tripping data.
#[test]
fn host_builtin_side_effect_treewalk() {
    let captured: Rc<Cell<i64>> = Rc::new(Cell::new(0));
    let captured_clone = Rc::clone(&captured);
    let _ = run_treewalk(
        r#"
        capture(89);
        h __result__ = 1;
        "#,
        move |interp| {
            interp.register_builtin("capture", move |args| {
                captured_clone.set(args[0].to_int());
                Ok(Value::Null)
            });
        },
    )
    .unwrap();
    assert_eq!(captured.get(), 89);
}

#[test]
fn host_builtin_side_effect_vm() {
    let captured: Rc<Cell<i64>> = Rc::new(Cell::new(0));
    let captured_clone = Rc::clone(&captured);
    let _ = run_vm_with_capture(
        r#"
        capture(89);
        __capture(1);
        "#,
        move |interp| {
            interp.register_builtin("capture", move |args| {
                captured_clone.set(args[0].to_int());
                Ok(Value::Null)
            });
        },
    )
    .unwrap();
    assert_eq!(captured.get(), 89);
}

/// Host fn returns an array — the value flows back into OMC normally.
/// Tests the "I want my Python list to look like an OMC array" path.
#[test]
fn host_builtin_returns_array() {
    let v = run_treewalk(
        r#"
        h xs = numpy_arange(5);
        h __result__ = arr_len(xs);
        "#,
        |interp| {
            interp.register_builtin("numpy_arange", |args| {
                let n = args[0].to_int().max(0) as usize;
                let items: Vec<Value> = (0..n)
                    .map(|i| Value::HInt(HInt::new(i as i64)))
                    .collect();
                Ok(Value::Array(HArray::from_vec(items)))
            });
        },
    )
    .unwrap();
    assert_eq!(v.to_int(), 5);
}

/// Host fn errors propagate as OMC errors — catchable via try/catch.
#[test]
fn host_builtin_error_is_catchable() {
    let v = run_treewalk(
        r#"
        try {
            broken();
            h __result__ = 0;
        } catch e {
            h __result__ = e;
        }
        "#,
        |interp| {
            interp.register_builtin("broken", |_args| {
                Err("intentional host failure".to_string())
            });
        },
    )
    .unwrap();
    match v {
        Value::String(s) => assert!(s.contains("intentional host failure")),
        other => panic!("expected error string, got {:?}", other),
    }
}

/// Host fn shadows a stdlib name. Used for sandboxing — embedder hands
/// OMC a custom `read_file` that only sees a whitelisted directory.
#[test]
fn host_builtin_shadows_stdlib() {
    let v = run_treewalk(
        r#"
        h __result__ = now_ms();
        "#,
        |interp| {
            interp.register_builtin("now_ms", |_args| {
                Ok(Value::HInt(HInt::new(12345)))
            });
        },
    )
    .unwrap();
    assert_eq!(v.to_int(), 12345);
}

/// Same shadowing test under the VM — verifies vm_call_builtin checks
/// host_builtins BEFORE vm_fast_dispatch, which would otherwise win
/// for hot stdlib names.
#[test]
fn host_builtin_shadows_stdlib_under_vm() {
    let v = run_vm_with_capture(
        r#"
        __capture(str_len("ignored"));
        "#,
        |interp| {
            interp.register_builtin("str_len", |_args| {
                Ok(Value::HInt(HInt::new(999)))
            });
        },
    )
    .unwrap();
    assert_eq!(v.to_int(), 999);
}

/// unregister_builtin removes a previously-registered handler. The
/// next call resolves to the underlying stdlib (or fails if no
/// stdlib match).
#[test]
fn host_builtin_unregister() {
    let mut interp = Interpreter::new();
    interp.register_builtin("custom", |_args| Ok(Value::HInt(HInt::new(7))));
    assert!(interp.has_host_builtin("custom"));
    assert!(interp.unregister_builtin("custom"));
    assert!(!interp.has_host_builtin("custom"));
    assert!(!interp.unregister_builtin("custom"));
}
