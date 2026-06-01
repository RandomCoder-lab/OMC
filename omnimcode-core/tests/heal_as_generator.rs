//! HEAL-as-generator-backend measurement.
//!
//! Hypothesis: a weak generator need not emit CORRECT OMC — only OMC inside HEAL's
//! basin of attraction. HEAL (substrate-addressed: typo→nearest-known-name) then
//! projects it onto the valid-program manifold. This harness measures that power
//! WITHOUT a generator confound: take known-good `__result__` programs, CORRUPT
//! them with the inverse of each heal class, then heal+run and compare to baseline.
//!
//! It reports, PER corruption class:
//!   raw_runs      — corrupted program runs as-is (no heal)        → the gap HEAL fills
//!   valid_recover — healed program runs without error             → VALIDITY recovery
//!   exact_recover — healed program returns the ORIGINAL result    → BEHAVIORAL recovery
//!
//! Honest expectation: typo/ref corruptions → high EXACT recovery (snap is invertible);
//! arity-drop / null-inject → high VALIDITY but low EXACT (0-pad / null→0 ≠ the original
//! operand). That contrast is the finding: HEAL is a true repair for lexical/reference
//! errors and a validity-restorer (not a mind-reader) for structural ones.

use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;

/// Parse + execute raw source; Some(result) if it ran and set __result__, else None.
fn run_raw(src: &str) -> Option<i64> {
    let mut p = Parser::new(src);
    let stmts = p.parse().ok()?;
    let mut interp = Interpreter::new();
    interp.execute(stmts).ok()?;
    interp.get_var_for_testing("__result__").map(|v| v.to_int())
}

/// Parse + HEAL-to-fixpoint + execute; Some(result) if the healed program ran.
fn run_healed(src: &str) -> Option<i64> {
    let mut p = Parser::new(src);
    let stmts = p.parse().ok()?;
    let interp = Interpreter::new();
    let (healed, _diags, _it, _outcome) = interp.heal_ast_until_fixpoint(stmts, 5);
    let mut interp2 = Interpreter::new();
    interp2.execute(healed).ok()?;
    interp2.get_var_for_testing("__result__").map(|v| v.to_int())
}

struct Corruption {
    class: &'static str,
    find: &'static str,
    replace: &'static str,
}

struct Case {
    src: &'static str,
    corruptions: &'static [Corruption],
}

fn cases() -> Vec<Case> {
    vec![
        // square(7)=49: typo the call (def intact) → snap back; drop arity → pad 0.
        Case {
            src: "fn square(x) { return x * x; } __result__ = square(7);",
            corruptions: &[
                Corruption { class: "typo", find: "square(7)", replace: "squar(7)" },
                Corruption { class: "arity_drop", find: "square(7)", replace: "square()" },
            ],
        },
        // cube(4)=64
        Case {
            src: "fn cube(x) { return x * x * x; } __result__ = cube(4);",
            corruptions: &[
                Corruption { class: "typo", find: "cube(4)", replace: "cueb(4)" },
            ],
        },
        // add3(2,3,4)=9: drop one arg → pad 0 (valid, wrong); typo the name.
        Case {
            src: "fn add3(a, b, c) { return a + b + c; } __result__ = add3(2, 3, 4);",
            corruptions: &[
                Corruption { class: "typo", find: "add3(2, 3, 4)", replace: "add33(2, 3, 4)" },
                Corruption { class: "arity_drop", find: "add3(2, 3, 4)", replace: "add3(2, 3)" },
            ],
        },
        // recursion: corrupt the RECURSIVE call (def name intact at top) → snap back.
        Case {
            src: "fn fact(n) { if n < 2 { return 1; } return n * fact(n - 1); } __result__ = fact(5);",
            corruptions: &[
                Corruption { class: "typo", find: "fact(n - 1)", replace: "factt(n - 1)" },
            ],
        },
        Case {
            src: "fn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); } __result__ = fib(10);",
            corruptions: &[
                Corruption { class: "typo", find: "fib(n - 1)", replace: "fibb(n - 1)" },
            ],
        },
        // null-inject into arithmetic: HEAL turns null→0 (valid, but ≠ original operand).
        Case {
            src: "__result__ = 10 + 5;",
            corruptions: &[
                Corruption { class: "null_inject", find: "10 + 5", replace: "null + 5" },
            ],
        },
        // str+num type slip: HEAL rewrites to concat (changes type → __result__ not int).
        Case {
            src: "fn dbl(x) { return x + x; } __result__ = dbl(21);",
            corruptions: &[
                Corruption { class: "typo", find: "dbl(21)", replace: "dlb(21)" },
            ],
        },
    ]
}

#[test]
fn heal_recovery_by_corruption_class() {
    use std::collections::BTreeMap;
    // class -> (n, raw_runs, valid_recover, exact_recover)
    let mut agg: BTreeMap<&str, [u32; 4]> = BTreeMap::new();

    for case in cases() {
        let baseline = run_raw(case.src).expect("baseline program must run");
        for c in case.corruptions {
            assert!(case.src.contains(c.find), "corruption target '{}' not in source", c.find);
            let corrupted = case.src.replacen(c.find, c.replace, 1);

            let raw = run_raw(&corrupted);
            let healed = run_healed(&corrupted);

            let e = agg.entry(c.class).or_insert([0; 4]);
            e[0] += 1; // n
            if raw == Some(baseline) {
                e[1] += 1; // raw still produced the right answer (corruption was benign)
            }
            if healed.is_some() {
                e[2] += 1; // VALIDITY recovery — healed program ran
            }
            if healed == Some(baseline) {
                e[3] += 1; // BEHAVIORAL recovery — healed matched original
            }
        }
    }

    println!("\n  HEAL-as-generator recovery (corrupt known-good → heal → run):");
    println!("  {:<12} {:>4} {:>9} {:>14} {:>14}", "class", "n", "raw_runs", "valid_recover", "exact_recover");
    let mut tot = [0u32; 4];
    for (class, v) in &agg {
        println!("  {:<12} {:>4} {:>9} {:>14} {:>14}", class, v[0], v[1], v[2], v[3]);
        for i in 0..4 { tot[i] += v[i]; }
    }
    println!("  {:<12} {:>4} {:>9} {:>14} {:>14}", "TOTAL", tot[0], tot[1], tot[2], tot[3]);

    // Contract (the honest, measured claims):
    // 1. Corruptions actually broke the programs (raw rarely produces the right answer).
    assert!(tot[1] * 2 < tot[0], "most corruptions should break the program (raw_runs low)");
    // 2. HEAL restores VALIDITY broadly (healed programs run again).
    assert!(tot[2] * 2 >= tot[0], "HEAL should restore validity for the majority");
    // 3. Lexical (typo) corruptions are EXACTLY recovered — the headline generator-repair.
    let typo = agg.get("typo").copied().unwrap_or([0; 4]);
    assert_eq!(typo[3], typo[0], "every typo corruption must be EXACTLY recovered (snap back to the real name)");
}
