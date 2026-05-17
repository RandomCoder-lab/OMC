//! Self-healing compiler pass tests.
//!
//! Exercises heal_ast directly (rather than end-to-end via --check) so a
//! regression in a single heal class shows up as a focused failing test.

use omnimcode_core::interpreter::{Interpreter, last_heal_counts};
use omnimcode_core::parser::Parser;

fn heal(source: &str) -> (Vec<String>, omnimcode_core::interpreter::HealClassCounts) {
    let mut parser = Parser::new(source);
    let stmts = parser.parse().expect("parse");
    let interp = Interpreter::new();
    let (_healed, diags) = interp.heal_ast(stmts);
    (diags, last_heal_counts())
}

// ---------- str_concat -----------------------------------------------------

#[test]
fn str_concat_string_plus_int_literal_rewrites() {
    let (diags, counts) = heal(r#"
        fn main() {
            h s = "loss: " + 5;
            return s;
        }
    "#);
    assert_eq!(counts.str_concat, 1, "exactly one str_concat heal");
    assert!(diags.iter().any(|d| d.contains("str-concat")),
            "diagnostic mentions str-concat: {:?}", diags);
}

#[test]
fn str_concat_string_plus_float_literal_rewrites() {
    let (_diags, counts) = heal(r#"
        fn main() {
            h x = 3.14 + "pi";
            return x;
        }
    "#);
    assert_eq!(counts.str_concat, 1, "float + string also rewrites");
}

#[test]
fn str_concat_does_not_rewrite_two_strings() {
    // Two-string concat is OMC's native Add behavior — must NOT be touched.
    let (_diags, counts) = heal(r#"
        fn main() {
            h s = "hello" + "world";
            return s;
        }
    "#);
    assert_eq!(counts.str_concat, 0, "string + string is left alone");
}

#[test]
fn str_concat_does_not_rewrite_two_numbers() {
    let (_diags, counts) = heal(r#"
        fn main() {
            h n = 1 + 2;
            return n;
        }
    "#);
    assert_eq!(counts.str_concat, 0, "number + number is left alone");
}

// ---------- var_typo --------------------------------------------------------

#[test]
fn var_typo_corrects_close_global_name() {
    // `helo` is one transposition away from `hello`. Heal should fix it.
    let (diags, counts) = heal(r#"
        h hello = 1;
        fn main() {
            return helo;
        }
    "#);
    assert!(counts.var_typo >= 1, "at least one var_typo fired: {:?}", diags);
}

#[test]
fn var_typo_does_not_flag_legit_local() {
    // `inner` is declared inside the body — the heal pass must collect it
    // into scope and NOT treat its reference as a typo of `outer`.
    let (_diags, counts) = heal(r#"
        h outer = 10;
        fn main() {
            h inner = 20;
            return inner;
        }
    "#);
    assert_eq!(counts.var_typo, 0, "local declaration must not false-positive");
}

#[test]
fn var_typo_does_not_flag_loop_var() {
    // For-loop iteration variable is a local binding — must be in scope.
    let (_diags, counts) = heal(r#"
        fn main() {
            h sum = 0;
            for i in range(0, 10) {
                sum = sum + i;
            }
            return sum;
        }
    "#);
    assert_eq!(counts.var_typo, 0, "for-loop var must be in scope");
}

#[test]
fn var_typo_total_includes_new_classes() {
    // Sanity: total() includes both new counters.
    let (_diags, counts) = heal(r#"
        h alpha = 1;
        fn main() {
            h s = "x: " + 5;
            return alph;
        }
    "#);
    assert!(counts.total() >= 2, "total counts both new heals: {:?}", counts);
}
