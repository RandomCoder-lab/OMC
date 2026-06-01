//! MAPE-K self-healing loop — `heal_ast_until_fixpoint` conformance.
//!
//! The healer is a MAPE loop (Monitor=walk AST, Analyze=classify into heal classes,
//! Plan=per-class enable/budget, Execute=rewrite). These tests lock the "K" — the
//! persistent Knowledge the loop now retains across passes — and the convergence
//! contract that reasons over the diagnostic-set history rather than a bare count.

use omnimcode_core::interpreter::{Interpreter, last_fixpoint_knowledge, last_heal_counts};
use omnimcode_core::parser::Parser;

/// Run the fixpoint loop and snapshot both the per-pass counters (final pass) and
/// the cross-pass Knowledge, captured on this thread immediately after the run.
fn fixpoint(src: &str) -> (Vec<String>, &'static str, usize, u32, u32) {
    let mut p = Parser::new(src);
    let stmts = p.parse().expect("parse");
    let interp = Interpreter::new();
    let (_healed, diags, iters, outcome) = interp.heal_ast_until_fixpoint(stmts, 5);
    (diags, outcome, iters, last_fixpoint_knowledge().total(), last_heal_counts().total())
}

// ── The persistent K is the ONLY meaningful post-run signal ──────────────────
// Convergence is detected by a final clean pass, so `last_heal_counts()` (the
// per-pass counter) is ~always 0 after a converged run — it reports the empty
// pass, not the work. The cross-pass `last_fixpoint_knowledge()` is what retains
// what the loop actually healed. Without it, the loop's effort is invisible.
#[test]
fn fixpoint_knowledge_is_the_cross_pass_total_not_the_final_pass() {
    // typo (targt→target) + arity pad, both healed in pass 0; pass 1 is clean.
    let (diags, outcome, _it, fix_k, last_k) = fixpoint(
        r#"
        fn target(a, b) { return a + b; }
        fn main() { return targt(5); }
    "#,
    );
    assert_eq!(outcome, "converged", "a clean final pass = converged");
    // `diags` is the cumulative heal LOG (every pass's diagnostics), not residual
    // errors — converged means the last pass found nothing left to fix.
    assert_eq!(diags.len(), 2, "the heal report lists both fixes (typo + arity): {diags:?}");
    assert_eq!(fix_k, 2, "cross-pass Knowledge retains BOTH heals (typo + arity)");
    assert_eq!(last_k, 0, "per-pass counter sees only the final clean pass — hence K is needed");
}

#[test]
fn fixpoint_knowledge_sums_independent_heals() {
    // three independent classes in one pass: str_concat + null_arith + var_typo.
    let (_d, outcome, _it, fix_k, last_k) = fixpoint(
        r#"
        h alpha = 1;
        fn main() { h s = "x: " + 5; h y = null + 3; return alph; }
    "#,
    );
    assert_eq!(outcome, "converged");
    assert_eq!(fix_k, 3, "Knowledge accumulates all three independent heals");
    assert_eq!(last_k, 0, "final pass clean");
}

// ── Stuck detection survives the switch from count-equality to set-signature ──
// A non-rewriting diagnostic (`if 0` warns but is never rewritten) reproduces the
// SAME diagnostic set every pass → a true cycle. The signature-based check must
// still classify this as "stuck" (not loop to "exhausted"), and K must still
// accumulate on the stuck path.
#[test]
fn fixpoint_detects_stuck_on_a_persistent_non_rewriting_diagnostic() {
    let (diags, outcome, iters, fix_k, _last_k) =
        fixpoint("fn main() { if 0 { return 1; } return 2; }");
    assert_eq!(outcome, "stuck", "an unrewritable repeating diagnostic is a genuine fixpoint");
    assert!(iters <= 2, "the repeated signature is caught on the second pass, not at max_iter");
    assert!(!diags.is_empty(), "the residual diagnostic is reported");
    assert!(fix_k >= 2, "Knowledge accumulates across the stuck passes too: {fix_k}");
}

// ── A clean program does zero work and converges immediately ─────────────────
#[test]
fn fixpoint_clean_program_is_noop() {
    let (diags, outcome, iters, fix_k, last_k) =
        fixpoint("fn main() { h x = 5; if x > 3 { return 1; } return 0; }");
    assert_eq!(outcome, "converged");
    assert_eq!(iters, 0, "no diagnostics on the first pass → converged at iter 0");
    assert!(diags.is_empty());
    assert_eq!(fix_k, 0, "no heals → empty Knowledge");
    assert_eq!(last_k, 0);
}
