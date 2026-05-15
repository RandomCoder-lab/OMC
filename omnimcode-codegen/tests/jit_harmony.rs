//! Session G end-to-end: harmony() intrinsic + harmony-gated branch.
//!
//! Verifies the architectural signal that makes "@predict cuts cost"
//! real:
//!
//! 1. Before phi_shadow, bands are matched (β=α from fn-entry splat)
//!    so harmony returns 1000 (perfect).
//! 2. After phi_shadow on a non-on-attractor value, β diverges from α
//!    by a substrate-distance > 0, so harmony returns < 1000.
//! 3. A JIT'd OMC fn can branch on harmony to skip work — the
//!    work-elision primitive that @predict needs.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::parser::Parser;

fn jit_fn(source: &str, fn_name: &str) -> (JitContext<'static>, omnimcode_codegen::JittedFn) {
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    // Box::leak the Context so the JitContext outlives this fn —
    // tests are short-lived so the leak is harmless.
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let jit = JitContext::new(context).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = *jitted.get(fn_name).expect("fn JIT'd");
    (jit, f)
}

#[test]
fn harmony_of_unshadowed_value_is_perfect() {
    // Without phi_shadow, β = α (matched-band fn entry). harmony
    // should return 1000 (perfect) for any input.
    let source = r#"
        fn read_harmony(x) { return harmony(x); }
    "#;
    let (_jit, f) = jit_fn(source, "read_harmony");
    for x in &[0i64, 1, 7, 42, 100, -50, 1000] {
        let r = f.call(&[*x]).expect("call");
        assert_eq!(
            r, 1000,
            "unshadowed harmony({}) should be 1000 (matched bands)",
            x
        );
    }
}

#[test]
fn harmony_of_shadowed_value_diverges() {
    // After phi_shadow, β = phi_fold(α) * 1000. For most α the diff
    // |α - β| lands OFF a Fibonacci attractor, so harmony < 1000.
    let source = r#"
        fn read_harmony_shadowed(x) {
            h y = phi_shadow(x);
            return harmony(y);
        }
    "#;
    let (_jit, f) = jit_fn(source, "read_harmony_shadowed");
    // Pick inputs whose phi-shadow diff is known to land off-attractor.
    // For α=42: β = phi_fold(42)*1000 = frac(67.957...)*1000 = 957.
    // diff = |42 - 957| = 915. Nearest attractor 987 (dist 72) →
    // harmony = 1/(1+72) ≈ 0.0137 → 14 in [0,1000].
    let r42 = f.call(&[42]).expect("call");
    assert!(
        r42 < 1000,
        "shadowed harmony(42) should be < 1000; got {}",
        r42
    );
    assert!(
        r42 < 100,
        "shadowed harmony(42) should be low (off-attractor); got {}",
        r42
    );
    // α=0 is a corner case: phi_fold(0) = 0, β = 0, diff = 0,
    // attractor 0, harmony = 1000.
    let r0 = f.call(&[0]).expect("call");
    assert_eq!(r0, 1000, "α=0 → β=0 → perfect harmony");
}

#[test]
fn harmony_gated_branch_elision() {
    // The cost-cut primitive: an OMC fn that uses harmony() to skip
    // expensive computation when bands are aligned.
    //
    // The pattern:
    //   if harmony(x) >= threshold {
    //       return cheap_path();
    //   }
    //   return expensive_path();
    //
    // Without phi_shadow, harmony is 1000 → cheap path wins.
    // With phi_shadow, harmony often < threshold → expensive path runs.
    let source = r#"
        fn gated(x) {
            if harmony(x) >= 500 {
                return 1;
            }
            return 0;
        }
        fn gated_shadowed(x) {
            h y = phi_shadow(x);
            if harmony(y) >= 500 {
                return 1;
            }
            return 0;
        }
    "#;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let context = Context::create();
    let jit = JitContext::new(&context).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let gated = jitted.get("gated").expect("gated JIT'd");
    let gated_shadowed = jitted.get("gated_shadowed").expect("gated_shadowed JIT'd");

    // Without phi_shadow: every input has perfect harmony → branch
    // taken, returns 1.
    for x in &[0i64, 7, 42, 89, 1000] {
        assert_eq!(
            gated.call(&[*x]).expect("call"),
            1,
            "unshadowed gated({}) should hit the high-harmony branch",
            x
        );
    }
    // With phi_shadow on a typical off-attractor input: harmony low,
    // expensive branch taken. (For α=0 phi_shadow still produces
    // perfect harmony.)
    assert_eq!(
        gated_shadowed.call(&[42]).expect("call"),
        0,
        "shadowed gated(42) should fall to the low-harmony branch"
    );
    assert_eq!(
        gated_shadowed.call(&[0]).expect("call"),
        1,
        "shadowed gated(0) is still perfect-harmony (α=β=0)"
    );
}
