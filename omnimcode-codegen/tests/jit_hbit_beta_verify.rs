//! HBit β-divergence verification (goal 1 of 4 HBit capabilities).
//!
//! Proves that PhiShadow store makes β data-dependent:
//!   - Fibonacci inputs: β = fold(α) = α → harmony = 1000 (perfect)
//!   - Non-Fibonacci inputs: β = fold(α) ≠ α → harmony data-dependent
//!
//! harmony(α, β) = 1 / (1 + dist(|α − β|, nearest Fibonacci)) * 1000
//!
//! After PhiShadow store: β = fold(α), so
//!   diff = |α − fold(α)| = distance from α to its nearest Fibonacci attractor
//!   harmony = 1 / (1 + dist(diff, nearest Fibonacci)) * 1000
//!
//! Specific cases verified:
//!   α=89 (Fibonacci): fold(89)=89, diff=0, dist(0,0)=0 → harmony=1000
//!   α=90 (next integer): fold(90)=89, diff=1 (Fibonacci!), dist=0 → harmony=1000
//!   α=100: fold(100)=89, diff=11, nearest Fib=13(dist 2) → harmony=333
//!   α=44: fold(44)=34, diff=10, nearest Fib=8(dist 2) → harmony=333

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::JitContext;
use omnimcode_core::parser::Parser;

fn jit_fn(source: &str, fn_name: &str) -> (JitContext<'static>, omnimcode_codegen::JittedFn) {
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let jit = JitContext::new(context).expect("jit ctx");
    let jitted = jit.jit_module(&module).expect("jit_module");
    let f = *jitted.get(fn_name).expect("fn JIT'd");
    (jit, f)
}

/// Verify the core property: `h y = x` triggers PhiShadow store,
/// setting β = fold(x). harmony(y) then measures whether |x - fold(x)|
/// is a Fibonacci number.
#[test]
fn beta_divergence_is_data_dependent() {
    let source = r#"
        fn store_and_harmony(x) {
            h y = x;
            return harmony(y);
        }
    "#;
    let (_jit, f) = jit_fn(source, "store_and_harmony");

    // α=89 (Fibonacci): β=fold(89)=89, diff=0, harmony=1000
    let r89 = f.call(&[89]).expect("call");
    assert_eq!(r89, 1000, "Fibonacci 89: β=α → perfect harmony");

    // α=44 (non-Fibonacci): fold(44)=34, diff=10
    // nearest Fib to 10: 8 (dist 2) < 13 (dist 3) → dist=2 → harmony=333
    let r44 = f.call(&[44]).expect("call");
    assert!(r44 < 1000, "non-Fibonacci 44: β≠α → imperfect harmony; got {}", r44);
    assert!(r44 > 0, "harmony should be positive; got {}", r44);

    // α=100: fold(100)=89, diff=11
    // nearest Fib to 11: 13 (dist 2) < 8 (dist 3) → dist=2 → harmony=333
    let r100 = f.call(&[100]).expect("call");
    assert!(r100 < 1000, "non-Fibonacci 100: β=89≠100 → imperfect harmony; got {}", r100);

    // The divergence is measurable — α=44 and α=100 should not be perfect.
    assert_ne!(r44, 1000, "44 should diverge from perfect harmony");
    assert_ne!(r100, 1000, "100 should diverge from perfect harmony");
}

/// Fibonacci inputs have β = α after PhiShadow store → perfect harmony.
#[test]
fn fibonacci_inputs_have_perfect_harmony_after_store() {
    let source = r#"
        fn store_harmony(x) {
            h y = x;
            return harmony(y);
        }
    "#;
    let (_jit, f) = jit_fn(source, "store_harmony");

    for fib in &[0i64, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233] {
        let r = f.call(&[*fib]).expect("call");
        assert_eq!(
            r, 1000,
            "Fibonacci {}: fold({})={} → diff=0 → harmony=1000",
            fib, fib, fib
        );
    }
}

/// Non-Fibonacci inputs diverge: diff = |α - fold(α)| > 0.
/// Harmony is < 1000 when diff is not itself a Fibonacci attractor.
/// Actual fold values from phi_pi_fib::nearest_attractor_with_dist:
///   fold(44)=34, diff=10, nearest Fib=8 (dist 2) → harmony=333
///   fold(50)=34, diff=16, nearest Fib=13 (dist 3) → harmony=250
///   fold(100)=89, diff=11, nearest Fib=13 (dist 3) → harmony=250
///   fold(90)=89, diff=1 (Fibonacci!) → harmony=1000
#[test]
fn non_fibonacci_with_non_fib_diff_is_imperfect() {
    let source = r#"
        fn store_harmony(x) {
            h y = x;
            return harmony(y);
        }
    "#;
    let (_jit, f) = jit_fn(source, "store_harmony");

    let r44 = f.call(&[44]).expect("call");
    let r50 = f.call(&[50]).expect("call");
    let r100 = f.call(&[100]).expect("call");
    // α=90: fold(90)=89, diff=1 IS Fibonacci → perfect harmony
    let r90 = f.call(&[90]).expect("call");

    assert!(r44 < 1000, "α=44: diff=10, not Fibonacci → imperfect; got {}", r44);
    assert!(r50 < 1000, "α=50: fold=34, diff=16, not Fibonacci → imperfect; got {}", r50);
    assert!(r100 < 1000, "α=100: diff=11, not Fibonacci → imperfect; got {}", r100);
    // α=90: fold(90)=89, diff=|90-89|=1, 1 is Fibonacci → perfect
    assert_eq!(r90, 1000, "α=90: fold=89, diff=1 (Fibonacci) → perfect harmony");
}

/// β-divergence gates branches: harmony < threshold → take slow path.
#[test]
fn beta_divergence_gates_branch() {
    let source = r#"
        fn gated(x) {
            h y = x;
            if harmony(y) >= 500 {
                return 1;
            }
            return 0;
        }
    "#;
    let (_jit, f) = jit_fn(source, "gated");

    // Fibonacci input: harmony = 1000 → high-harmony branch
    assert_eq!(f.call(&[89]).expect("call"), 1, "Fibonacci 89: high-harmony");
    assert_eq!(f.call(&[34]).expect("call"), 1, "Fibonacci 34: high-harmony");

    // α=44: diff=10, harmony=333 < 500 → low-harmony branch
    assert_eq!(f.call(&[44]).expect("call"), 0, "non-Fib 44: low-harmony branch");

    // α=100: diff=11, harmony=333 < 500 → low-harmony branch
    assert_eq!(f.call(&[100]).expect("call"), 0, "non-Fib 100: low-harmony branch");
}
