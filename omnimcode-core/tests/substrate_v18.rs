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

// ── verify-gate determinism hardening: the test is now run TWICE and must AGREE
//    before a swap is accepted (a candidate that passes only by luck/non-determinism
//    is rejected). OMC execution is deterministic by design (fixed RNG seed), so this
//    is a forward-guard (wall-clock / future non-deterministic builtins) + must not
//    regress the deterministic accept/reject behavior. ──
#[test]
fn fn_swap_verified_determinism_no_regression() {
    // deterministic + correct candidate: still accepted (gate's two agreeing runs pass)
    let det = r#"
        fn target(n) { return 0 - 1; }
        h cand = "fn target(n) { return n * n; }";
        h ok = fn_swap_verified("target", cand, "target(5) == 25");
        __result__ = ok["accepted"];
    "#;
    assert_eq!(int(det), 1, "a deterministic correct candidate must still be accepted");
    // deterministic + wrong candidate: still rejected
    let wrong = r#"
        fn target(n) { return n * n; }
        h cand = "fn target(n) { return n + 1; }";
        h r = fn_swap_verified("target", cand, "target(5) == 25");
        __result__ = r["accepted"];
    "#;
    assert_eq!(int(wrong), 0, "a deterministic wrong candidate must still be rejected");
}

// ── verify-gate invariant SET: the 3rd arg may be an ARRAY of invariants; the swap is
//    accepted iff EVERY invariant holds (deterministically + truthy), else rolled back.
//    A strong gate checks invariants, not a single example. ──
#[test]
fn fn_swap_verified_invariant_set() {
    // all invariants hold → accept
    let all_pass = r#"
        fn target(n) { return 0; }
        h cand = "fn target(n) { return n * n; }";
        h ok = fn_swap_verified("target", cand, ["target(5) == 25", "target(0) == 0", "target(3) == 9"]);
        __result__ = ok["accepted"];
    "#;
    assert_eq!(int(all_pass), 1, "swap accepted only when ALL invariants hold");
    // one invariant fails → reject + roll back
    let one_fails = r#"
        fn target(n) { return n * n; }
        h cand = "fn target(n) { return n * n; }";
        h r = fn_swap_verified("target", cand, ["target(5) == 25", "target(3) == 999"]);
        __result__ = r["accepted"];
    "#;
    assert_eq!(int(one_fails), 0, "swap rejected if ANY invariant fails");
}

// ── hierarchical addressing: fns_on_subface descends HAddr one level (face→sub_face),
//    surfacing the existing 3-level hierarchy that fns_on_face flattened. The 3 sub_faces
//    must EXACTLY partition a face's functions (disjoint cover) — without touching the
//    proven χ²-uniform face result. ──
#[test]
fn fns_on_subface_partitions_face() {
    let src = r#"
        fn alpha(n) { return n; }
        fn beta(n) { return n + 1; }
        fn gamma(n) { return n * 2; }
        fn delta(n) { return n - 1; }
        fn epsilon(n) { return n * n; }
        h face0 = arr_len(fns_on_face(0));
        h subs = arr_len(fns_on_subface(0, 0)) + arr_len(fns_on_subface(0, 1)) + arr_len(fns_on_subface(0, 2));
        __result__ = (subs == face0);
    "#;
    assert_eq!(int(src), 1, "the 3 sub_faces must exactly partition a face (disjoint cover)");
}

// ── bounded memo cache (memo_put): store + cache-hit must stay correct through the new
//    FIFO-eviction path. (Eviction itself is correctness-safe by construction — memo values
//    are pure + disk-persisted — and the 100k cap isn't reached in a unit test; this guards
//    against regressing the store/hit behavior.) ──
#[test]
fn memo_put_store_and_hit_correct() {
    let src = r#"
        @memo fn sq(n) { return n * n; }
        h a = sq(7);
        h b = sq(99);
        h c = sq(7);
        __result__ = (a == 49) and (b == 9801) and (c == 49);
    "#;
    assert_eq!(int(src), 1, "memo store + cache-hit must remain correct through memo_put");
}

// ── autograd numerical stability: tape_log is ε-clamped — log of a non-positive value
//    is a large FINITE number, never -∞ (a single -∞ would poison the tape into NaN
//    loss/grads and diverge training). ──
#[test]
fn tape_log_epsilon_clamped_no_neg_inf() {
    let src = r#"
        h x = tape_var(0 - 1);
        h y = tape_log(x);
        h v = tape_value(y);
        __result__ = (v > 0 - 1000000) and (v < 0);
    "#;
    assert_eq!(int(src), 1, "tape_log(x<=0) must be large-finite (eps-clamped), never -inf");
}

// ── autograd numerical stability: tape_exp is overflow-clamped — exp of a large value is
//    a finite number, never +∞ (which would poison the tape into NaN). (v-v is 0 for a
//    finite value but NaN for ∞, and NaN fails all comparisons → distinguishes the two.) ──
#[test]
fn tape_exp_clamped_no_overflow_inf() {
    let src = r#"
        h x = tape_var(1000);
        h y = tape_exp(x);
        h v = tape_value(y);
        h d = v - v;
        __result__ = (d < 1) and (d > 0 - 1);
    "#;
    assert_eq!(int(src), 1, "tape_exp(large) must be finite (clamped), never +inf");
}

// ── autograd numerical stability: tape_pow_int BACKWARD must not poison the tape. The naive
//    grad n·x^(n-1) computes 0·x^(-1) = 0·∞ = NaN for the constant case (n=0, x=0) — reachable
//    with perfectly ordinary inputs, no upstream blow-up needed — and ±∞ at the n<0 pole (x=0).
//    Guarded: n==0 ⇒ 0 (derivative of a constant); a non-finite x^(n-1) ⇒ 0. A single NaN/∞
//    grad would poison the whole tape. The fix must NOT distort legitimate gradients. ──
#[test]
fn tape_pow_int_backward_no_nan_or_inf() {
    // d/dx (x^0) at x=0: x^0 is constant ⇒ derivative 0, NOT NaN (the 0·∞ trap). g must be
    // finite (g-g == 0, which is NaN for a poisoned grad) and ~0.
    let constant = r#"
        h x = tape_var(0);
        h y = tape_pow_int(x, 0);
        tape_backward(y);
        h g = tape_grad(x);
        h d = g - g;
        __result__ = (d < 1) and (d > 0 - 1) and (g < 1) and (g > 0 - 1);
    "#;
    assert_eq!(int(constant), 1, "d/dx x^0 at x=0 must be a finite 0, never NaN (0*inf)");
    // and the fix must not distort real gradients: d/dx x^2 = 2x; at x=3 ⇒ exactly 6
    let normal = r#"
        h x = tape_var(3);
        h y = tape_pow_int(x, 2);
        tape_backward(y);
        __result__ = tape_grad(x);
    "#;
    assert_eq!(int(normal), 6, "d/dx x^2 at x=3 must be exactly 6 (fix must not distort real grads)");
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
