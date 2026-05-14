//! OMNIcode Conformance Golden Tests
//!
//! Lock the language's "physics" — mathematical and semantic behaviors that
//! must remain stable regardless of how the interpreter / compiler is
//! restructured. These tests are the contract between this Rust port and
//! the canonical Python omnicc at
//! `/home/thearchitect/Sovereign_Lattice/omninet_package/`.
//!
//! Modeled after `test_conformance_golden.omc` from the canonical tree.
//! If a test in this file fails, either:
//! (a) you genuinely changed the language's semantics — update Python too,
//!     and document the break in CHANGELOG.md, or
//! (b) something regressed — fix the regression, do NOT relax the test.

use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::value::Value;

fn run(source: &str) -> Result<Value, String> {
    let mut parser = Parser::new(source);
    let stmts = parser.parse()?;
    let mut interp = Interpreter::new();
    interp.execute(stmts)?;
    interp
        .get_var_for_testing("__result__")
        .ok_or_else(|| "no __result__ variable".to_string())
}

// ===========================================================================
// SECTION 1 — Fibonacci numbers must have HIGH resonance (>= 0.7)
// ===========================================================================

#[test]
fn fibonacci_1_has_high_resonance() {
    let v = run("__result__ = res(1);").unwrap();
    assert!(
        v.to_float() >= 0.7,
        "res(1) must be >= 0.7, got {}",
        v.to_float()
    );
}

#[test]
fn fibonacci_89_is_perfect() {
    let v = run("__result__ = res(89);").unwrap();
    assert!(
        (v.to_float() - 1.0).abs() < 1e-9,
        "res(89) must be 1.0 (perfect resonance), got {}",
        v.to_float()
    );
}

#[test]
fn fibonacci_610_is_perfect() {
    let v = run("__result__ = res(610);").unwrap();
    assert!(
        (v.to_float() - 1.0).abs() < 1e-9,
        "res(610) must be 1.0, got {}",
        v.to_float()
    );
}

#[test]
fn fibonacci_attractors_all_above_threshold() {
    // 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 are all Fibonacci
    for n in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610] {
        let src = format!("__result__ = res({});", n);
        let v = run(&src).unwrap();
        assert!(
            v.to_float() >= 0.7,
            "Fibonacci {} must have res >= 0.7, got {}",
            n,
            v.to_float()
        );
    }
}

// ===========================================================================
// SECTION 2 — Non-Fibonacci numbers have LOWER resonance
// ===========================================================================

#[test]
fn non_fibonacci_has_lower_resonance() {
    // 100 is far from any Fibonacci (89 and 144 nearest, dist 11)
    let v = run("__result__ = res(100);").unwrap();
    assert!(
        v.to_float() < 1.0,
        "res(100) must be < 1.0 (not perfect), got {}",
        v.to_float()
    );
}

// ===========================================================================
// SECTION 3 — fold() snaps to nearest Fibonacci attractor
// ===========================================================================

#[test]
fn fold_89_is_identity() {
    let v = run("__result__ = fold(89);").unwrap();
    assert_eq!(v.to_int(), 89);
}

#[test]
fn fold_90_snaps_to_89() {
    let v = run("__result__ = fold(90);").unwrap();
    assert_eq!(v.to_int(), 89);
}

#[test]
fn fold_negative_preserves_sign() {
    let v = run("__result__ = fold(-90);").unwrap();
    assert_eq!(v.to_int(), -89);
}

#[test]
fn fold_two_arg_string_mode_works() {
    // Canonical OMC: fold(x, "fibonacci")
    let v = run("__result__ = fold(90, \"fibonacci\");").unwrap();
    assert_eq!(v.to_int(), 89);
}

// ===========================================================================
// SECTION 4 — Division by zero produces a Singularity, NOT a crash
// ===========================================================================

#[test]
fn div_by_zero_is_singularity_not_crash() {
    let v = run("h x = 89 / 0; __result__ = x;").unwrap();
    assert!(
        matches!(v, Value::Singularity { numerator: 89, .. }),
        "89/0 must produce Singularity(89/0), got {:?}",
        v
    );
}

#[test]
fn is_singularity_returns_int_one_for_portal() {
    let v = run("h p = 7 / 0; __result__ = is_singularity(p);").unwrap();
    assert_eq!(v.to_int(), 1, "is_singularity must return int 1, not bool");
}

#[test]
fn is_singularity_returns_int_zero_for_normal() {
    let v = run("__result__ = is_singularity(42);").unwrap();
    assert_eq!(v.to_int(), 0);
}

#[test]
fn resolve_singularity_fold_mode_snaps_to_fibonacci() {
    let v = run("h p = 90 / 0; __result__ = resolve_singularity(p, \"fold\");").unwrap();
    assert_eq!(v.to_int(), 89);
}

#[test]
fn canonical_smart_divide_high_resonance_folds() {
    let src = r#"
        fn smart_divide(numerator, denominator) {
            h result = numerator / denominator;
            if is_singularity(result) == 1 {
                h num_res = res(numerator);
                if num_res >= 0.7 {
                    return resolve_singularity(result, "fold");
                } else {
                    return resolve_singularity(result, "invert");
                }
            } else {
                return result;
            }
        }
        __result__ = smart_divide(89, 0);
    "#;
    let v = run(src).unwrap();
    assert_eq!(v.to_int(), 89, "89/0 with high res folds to itself");
}

// ===========================================================================
// SECTION 5 — Arithmetic stability (int + int = int, mixed = float)
// ===========================================================================

#[test]
fn int_plus_int_is_int() {
    let v = run("__result__ = 21 + 34;").unwrap();
    assert!(matches!(v, Value::HInt(_)));
    assert_eq!(v.to_int(), 55, "21 + 34 must = 55 (Fibonacci)");
}

#[test]
fn float_plus_int_promotes_to_float() {
    let v = run("__result__ = 1.5 + 2;").unwrap();
    assert!(matches!(v, Value::HFloat(_)));
    assert_eq!(v.to_float(), 3.5);
}

#[test]
fn integer_division_by_nonzero_stays_int() {
    let v = run("__result__ = 89 / 2;").unwrap();
    assert!(matches!(v, Value::HInt(_)));
    assert_eq!(v.to_int(), 44);
}

// ===========================================================================
// SECTION 6 — phi.X module-qualified calls
// ===========================================================================

#[test]
fn phi_fold_one_arg_matches_fold() {
    let a = run("__result__ = fold(90);").unwrap();
    let b = run("__result__ = phi.fold(90);").unwrap();
    assert_eq!(a.to_int(), b.to_int(), "phi.fold(x) must match fold(x)");
}

#[test]
fn phi_res_returns_float() {
    let v = run("__result__ = phi.res(89);").unwrap();
    assert!(matches!(v, Value::HFloat(_)));
    assert!((v.to_float() - 1.0).abs() < 1e-9);
}

#[test]
fn phi_fold_with_dynamic_depth() {
    // Depth comes from a variable, not a literal — Phase 18 gotcha fix
    let v = run("h d = 3; __result__ = phi.fold(0.5, d);").unwrap();
    assert!(matches!(v, Value::HFloat(_)));
    let f = v.to_float();
    assert!(f >= 0.0 && f < 1.0, "phi.fold(float) result in [0,1)");
}

// ===========================================================================
// SECTION 7 — Built-in math identities
// ===========================================================================

#[test]
fn sqrt_144_is_12() {
    let v = run("__result__ = sqrt(144);").unwrap();
    assert!((v.to_float() - 12.0).abs() < 1e-9);
}

#[test]
fn pow_2_10_is_1024() {
    let v = run("__result__ = pow(2, 10);").unwrap();
    assert_eq!(v.to_int(), 1024);
}

#[test]
fn sigmoid_at_zero_is_half() {
    let v = run("__result__ = sigmoid(0.0);").unwrap();
    assert!((v.to_float() - 0.5).abs() < 1e-9);
}

#[test]
fn pi_constant_is_correct() {
    let v = run("__result__ = pi();").unwrap();
    assert!((v.to_float() - std::f64::consts::PI).abs() < 1e-12);
}

// ===========================================================================
// SECTION 8 — Arrays
// ===========================================================================

#[test]
fn arr_from_range_count_correct() {
    let v = run("h a = arr_from_range(1, 11); __result__ = arr_len(a);").unwrap();
    assert_eq!(v.to_int(), 10, "arr_from_range(1, 11) has 10 elements");
}

#[test]
fn arr_sum_of_1_through_10_is_55() {
    let v = run("h a = arr_from_range(1, 11); __result__ = arr_sum(a);").unwrap();
    assert_eq!(v.to_int(), 55, "sum(1..10) = 55 (Fibonacci coincidence)");
}

#[test]
fn arr_get_set_round_trip() {
    let src = "h a = arr_from_range(0, 5); arr_set(a, 2, 99); __result__ = arr_get(a, 2);";
    let v = run(src).unwrap();
    assert_eq!(v.to_int(), 99);
}

#[test]
fn arr_push_extends_length() {
    let src = "h a = arr_from_range(0, 3); arr_push(a, 100); __result__ = arr_len(a);";
    let v = run(src).unwrap();
    assert_eq!(v.to_int(), 4);
}

// ===========================================================================
// SECTION 9 — String operations
// ===========================================================================

#[test]
fn str_reverse_works() {
    let v = run("__result__ = str_reverse(\"hello\");").unwrap();
    assert_eq!(v.to_string(), "olleh");
}

#[test]
fn str_contains_finds_substring() {
    let v = run("__result__ = str_contains(\"hello world\", \"world\");").unwrap();
    assert_eq!(v.to_int(), 1);
}

#[test]
fn concat_many_joins_multiple_values() {
    let v = run("__result__ = concat_many(\"res=\", 89, \" \", \"phi=\", 1);").unwrap();
    assert_eq!(v.to_string(), "res=89 phi=1");
}

// ===========================================================================
// SECTION 10 — Recursion / control flow
// ===========================================================================

#[test]
fn recursive_fibonacci_matches_built_in() {
    let src = r#"
        fn fib(n) {
            if n <= 1 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        __result__ = fib(10);
    "#;
    let v = run(src).unwrap();
    assert_eq!(v.to_int(), 55);
}

// ===========================================================================
// SECTION 11 — Self-healing primitives (Phase O)
// ===========================================================================
//
// The ONN self-healing pattern: detect proximity to singularities BEFORE
// they occur via value_danger(x) = exp(-|x|), then preemptively fold to a
// Fibonacci attractor via fold_escape(x). This is the canonical "Fibonacci-
// alignment auto-repair" mechanism — code stays on the φ-geodesic without
// explicit if-then error handling.

#[test]
fn value_danger_at_zero_is_one() {
    let v = run("__result__ = value_danger(0);").unwrap();
    assert!((v.to_float() - 1.0).abs() < 1e-12);
}

#[test]
fn value_danger_at_one_is_exp_minus_one() {
    let v = run("__result__ = value_danger(1);").unwrap();
    let expected = (-1.0_f64).exp();
    assert!((v.to_float() - expected).abs() < 1e-12);
}

#[test]
fn value_danger_large_value_near_zero() {
    let v = run("__result__ = value_danger(89);").unwrap();
    assert!(v.to_float() < 1e-30, "danger of 89 must be vanishingly small");
}

#[test]
fn fold_escape_zero_becomes_one() {
    // The zero-trap escape: nearest Fibonacci to 0 is 0 itself, but
    // fold_escape jumps to 1 to actually escape the singularity.
    let v = run("__result__ = fold_escape(0);").unwrap();
    assert_eq!(v.to_int(), 1, "fold_escape must NEVER land on 0");
}

#[test]
fn fold_escape_safe_value_passthrough() {
    let v = run("__result__ = fold_escape(100);").unwrap();
    assert_eq!(v.to_int(), 100, "safe values must passthrough fold_escape");
}

#[test]
fn safe_divide_handles_zero_divisor() {
    // Without self-healing this would return a Singularity. With self-healing,
    // the divisor is folded away from zero BEFORE the operation.
    let v = run("__result__ = safe_divide(89, 0);").unwrap();
    assert!(
        !v.is_singularity(),
        "safe_divide must never produce a Singularity"
    );
    // 89 / 1 = 89 (zero was healed to nearest non-zero Fibonacci, which is 1)
    assert_eq!(v.to_int(), 89);
}

#[test]
fn safe_divide_normal_division_unchanged() {
    let v = run("__result__ = safe_divide(89, 2);").unwrap();
    assert_eq!(v.to_int(), 44);
}

#[test]
fn harmony_value_fibonacci_is_perfect() {
    let v = run("__result__ = harmony_value(89);").unwrap();
    assert!((v.to_float() - 1.0).abs() < 1e-9);
}

#[test]
fn harmony_value_non_fibonacci_is_lower() {
    let v89 = run("__result__ = harmony_value(89);").unwrap().to_float();
    let v100 = run("__result__ = harmony_value(100);").unwrap().to_float();
    assert!(
        v100 < v89,
        "harmony(100) {} must be < harmony(89) {}",
        v100, v89
    );
}

#[test]
fn while_loop_terminates_with_break() {
    let src = r#"
        h i = 0;
        h sum = 0;
        while i < 100 {
            sum = sum + i;
            i = i + 1;
        }
        __result__ = sum;
    "#;
    let v = run(src).unwrap();
    assert_eq!(v.to_int(), 4950, "sum(0..100) = 4950");
}
