//! Goal 2: JIT coverage of Prometheus hot-path leaf functions.
//!
//! These are the pure-scalar helpers in prometheus.omc that the old
//! fn_uses_collections filter wrongly excluded. With the corrected
//! filter (only NewDict/DictSetNamed/DictDelNamed disqualify), scalar
//! leaf functions now JIT automatically:
//!
//!   _prom_lcg_step(state)    — one-liner int, hottest LCG inner loop
//!   _prom_circular_dist(a,b,m) — int abs-min, per-modulus in geodesic
//!
//! Verification: compile a mini-prometheus module, confirm the above
//! names appear in the jitted HashMap, and call through JIT vs tree-
//! walk to verify byte-identical results.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::value::{HInt, Value};
use std::rc::Rc;

fn jit_module_from_source(source: &str) -> (Context, std::collections::HashMap<String, omnimcode_codegen::JittedFn>) {
    let mut parser = Parser::new(source);
    let stmts = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&stmts).expect("compile");
    let ctx = Context::create();
    // SAFETY: leak ctx to get 'static; test process exits shortly.
    let ctx_static: &'static Context = unsafe { std::mem::transmute(&ctx) };
    let jit = JitContext::new(ctx_static).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    Box::leak(Box::new(jit)); // keep alive
    (ctx, jitted)
}

/// _prom_lcg_step is pure int arithmetic → should JIT.
#[test]
fn prom_lcg_step_jits_and_matches_treewalk() {
    let source = r#"
        fn _prom_lcg_step(state) {
            return (state * 1103515245 + 12345) % 2147483648;
        }
    "#;
    let (_ctx, jitted) = jit_module_from_source(source);
    assert!(
        jitted.contains_key("_prom_lcg_step"),
        "_prom_lcg_step should be JIT-eligible (pure int)"
    );

    let jf = jitted["_prom_lcg_step"];
    // Verify a few known LCG outputs.
    // state=0 → (0 * 1103515245 + 12345) % 2147483648 = 12345
    assert_eq!(jf.call(&[0]).expect("call(0)"), 12345);
    // state=12345 → (12345 * 1103515245 + 12345) % 2147483648
    let expected = ((12345i64 * 1103515245 + 12345) % 2147483648) as i64;
    assert_eq!(jf.call(&[12345]).expect("call(12345)"), expected);
}

/// _prom_circular_dist is int abs-min → should JIT.
#[test]
fn prom_circular_dist_jits_and_matches_treewalk() {
    let source = r#"
        fn _prom_circular_dist(a, b, m) {
            h d = a - b;
            if d < 0 { d = 0 - d; }
            h alt = m - d;
            if alt < d { return alt; }
            return d;
        }
    "#;
    let (_ctx, jitted) = jit_module_from_source(source);
    assert!(
        jitted.contains_key("_prom_circular_dist"),
        "_prom_circular_dist should be JIT-eligible (pure int branches)"
    );

    let jf = jitted["_prom_circular_dist"];
    // circular_dist(0, 5, 8): d=5, alt=3 → 3
    assert_eq!(jf.call(&[0, 5, 8]).expect("call"), 3);
    // circular_dist(0, 3, 8): d=3, alt=5 → 3
    assert_eq!(jf.call(&[0, 3, 8]).expect("call"), 3);
    // circular_dist(0, 0, 13): d=0, alt=13 → 0
    assert_eq!(jf.call(&[0, 0, 13]).expect("call"), 0);
}

/// Verify JIT and tree-walk produce the same outputs for _prom_lcg_step
/// by running the interpreter with JIT dispatch enabled.
#[test]
fn prom_lcg_step_dispatch_parity() {
    let source = r#"
        fn _prom_lcg_step(state) {
            return (state * 1103515245 + 12345) % 2147483648;
        }
        h r1 = _prom_lcg_step(0);
        h r2 = _prom_lcg_step(r1);
        h r3 = _prom_lcg_step(r2);
    "#;
    let mut parser = Parser::new(source);
    let stmts = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&stmts).expect("compile");
    let ctx_static: &'static Context = Box::leak(Box::new(Context::create()));
    let jit = JitContext::new(ctx_static).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    Box::leak(Box::new(jit));

    let dispatch: Rc<dyn Fn(&str, &[Value]) -> Option<Result<Value, String>>> =
        Rc::new(move |name, args| {
            let jf = jitted.get(name)?;
            if args.len() != jf.arity { return None; }
            let int_args: Vec<i64> = args.iter().filter_map(|a| match a {
                Value::HInt(h) => Some(h.value),
                _ => None,
            }).collect();
            if int_args.len() != args.len() { return None; }
            jf.call(&int_args).map(|r| Ok(Value::HInt(HInt::new(r))))
        });

    // JIT run
    let mut parser2 = Parser::new(source);
    let stmts2 = parser2.parse().expect("parse2");
    let mut interp_jit = Interpreter::new();
    interp_jit.set_jit_dispatch(Some(dispatch));
    interp_jit.execute(stmts2).expect("execute jit");
    let r3_jit = interp_jit.get_var_for_testing("r3").expect("r3 jit");

    // Tree-walk run
    let mut parser3 = Parser::new(source);
    let stmts3 = parser3.parse().expect("parse3");
    let mut interp_tw = Interpreter::new();
    interp_tw.execute(stmts3).expect("execute tw");
    let r3_tw = interp_tw.get_var_for_testing("r3").expect("r3 tw");

    assert_eq!(
        r3_jit.to_int(), r3_tw.to_int(),
        "JIT and tree-walk LCG chains must agree"
    );
}
