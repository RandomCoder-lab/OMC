//! Substrate-into-core (v1.8.x) conformance — locks the new primitives' behavior so future
//! interpreter restructuring can't silently regress them. Each test evals real OMC through the
//! parser + interpreter and asserts the contract. If one fails: fix the regression, don't relax it.

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

fn int(src: &str) -> i64 {
    run(src).unwrap().to_int()
}

// ── content-addressing (Phase 1.1) ──
#[test]
fn haddr_face_in_range_and_deterministic() {
    assert_eq!(int("__result__ = haddr_face(\"fibonacci\");"), int("__result__ = haddr_face(\"fibonacci\");"));
    let f = int("__result__ = haddr_face(\"gcd\");");
    assert!((0..12).contains(&f), "face out of range: {f}");
}

// ── O(1) semantic equality + content-addressed heap (Phase 2) ──
#[test]
fn same_value_structural_equality() {
    assert_eq!(int("__result__ = same_value([1,2,3], [1,2,3]);"), 1);
    assert_eq!(int("__result__ = same_value([1,2,3], [1,2,4]);"), 0);
    assert_eq!(int("__result__ = same_value({\"a\":1}, {\"a\":1});"), 1);
}

#[test]
fn cas_round_trip_and_dedup() {
    assert_eq!(int("h k = cas_put([10,20,30]); __result__ = cas_get(k)[1];"), 20);
    // identical content → same key
    assert_eq!(int("h a = cas_put([7,7]); h b = cas_put([7,7]); __result__ = same_value(a,b);"), 1);
}

// ── @memo (Phase 2.2): correctness + the purity gate ──
#[test]
fn memo_matches_plain() {
    let src = r#"
        @memo
        fn mf(n) { if n < 2 { return n; } return mf(n-1) + mf(n-2); }
        fn pf(n) { if n < 2 { return n; } return pf(n-1) + pf(n-2); }
        __result__ = same_value(mf(27), pf(27));
    "#;
    assert_eq!(int(src), 1);
    assert_eq!(int("@memo\nfn mf(n){ if n<2 {return n;} return mf(n-1)+mf(n-2);}\n__result__ = mf(40);"), 102334155);
}

#[test]
fn memo_refuses_impure() {
    let r = run("@memo\nfn bad(n){ print(n); return n; }\n__result__ = bad(1);");
    assert!(r.is_err(), "@memo on an impure fn must be refused");
}

// ── locality similarity + dispatch (Phase 1.2 / 3.1) ──
#[test]
fn locality_orders_by_content_and_routes() {
    // near-variant more similar than unrelated (×1000, integer compare to avoid float fmt)
    let src = r#"
        h near = locality_sim("quicksort","quick_sort");
        h far  = locality_sim("quicksort","zzzzzzzz");
        __result__ = near > far;
    "#;
    assert_eq!(int(src), 1);
    let route = run("fn quicksort(a){return a;} __result__ = nearest_fn(\"quicksrt\");").unwrap();
    assert_eq!(route.to_string(), "quicksort");
}

// ── verify-gated self-modification (Phase 3) ──
#[test]
fn fn_swap_verified_accepts_good_rejects_bad() {
    let good = r#"
        fn target(n) { return 0 - 1; }
        h cand = "fn target(n) { return n * n; }";
        h ok = fn_swap_verified("target", cand, "target(5) == 25");
        __result__ = ok["accepted"];
    "#;
    assert_eq!(int(good), 1);
    let bad = r#"
        fn target(n) { return n * n; }
        h cand = "fn target(n) { return n + 1; }";
        h r = fn_swap_verified("target", cand, "target(5) == 25");
        __result__ = r["accepted"];
    "#;
    assert_eq!(int(bad), 0, "a candidate failing its test must be rejected");
    // and rolled back: target still squares
    let rollback = r#"
        fn target(n) { return n * n; }
        h cand = "fn target(n) { return n + 1; }";
        fn_swap_verified("target", cand, "target(5) == 25");
        __result__ = target(6);
    "#;
    assert_eq!(int(rollback), 36, "rejected candidate must be rolled back");
}

// ── correct-by-construction synthesis (Phase 4) ──
#[test]
fn gen_omc_is_valid_by_construction() {
    // a generated program parses (code_parse_check ok) for several seeds
    for seed in [1, 7, 42, 256, 2026] {
        let src = format!("h p = gen_omc({seed}); h c = code_parse_check(p); __result__ = c[\"ok\"];");
        assert_eq!(int(&src), 1, "gen_omc({seed}) did not parse");
    }
    // same address → same program (gen_at determinism)
    assert_eq!(int("__result__ = same_value(gen_at(\"x\"), gen_at(\"x\"));"), 1);
}

// ── HBit dual-band at the Value level (Phase 6) ──
#[test]
fn dualband_rides_through_arithmetic_alpha_exact() {
    // phi_shadow(10) → β = nearest attractor (8); β rides through +3 → 11; α stays exact 13
    assert_eq!(int("h s = phi_shadow(10) + 3; __result__ = s;"), 13);
    assert_eq!(int("h s = phi_shadow(10) + 3; __result__ = bands(s)[1];"), 11);
    // on-lattice computation has zero divergence; off-lattice is positive
    assert_eq!(int("__result__ = value_divergence(phi_shadow(8) * 7);"), 0);
    assert!(int("__result__ = value_divergence((phi_shadow(50)+1)*3);") > 0);
    // ordinary (single-band) values are perfectly in tune and unchanged
    assert_eq!(int("__result__ = harmony(7 + 3);"), 1000);
    assert_eq!(int("__result__ = 7 + 3;"), 10);
}

#[test]
fn hbit_gate_separates_in_tune_from_divergent() {
    assert_eq!(int("__result__ = hbit_divergence(8, 8);"), 0);
    assert!(int("__result__ = hbit_divergence(8, 977);") > 500);
}

// ── CRT positional encoding (Phase 1.3) ──
#[test]
fn crt_pe_is_periodic_over_lcm() {
    assert_eq!(int("__result__ = same_value(crt_pe(0), crt_pe(10920));"), 1);
    assert_eq!(int("__result__ = same_value(crt_pe(7), crt_pe(8));"), 0);
}
