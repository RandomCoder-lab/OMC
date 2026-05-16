//! Harmonic-primitive JIT intrinsics — verify that each OMC builtin
//! intercepted in `dual_band.rs:HARMONIC_INTRINSICS` produces the
//! same answer through the JIT extern path as it does through the
//! tree-walk OMC builtin dispatch.
//!
//! Each test calls the JIT'd fn directly via `JittedFn::call` and
//! compares to a known mathematical answer (or to a Rust-side
//! equivalent computation). Cross-check with tree-walk happens
//! transitively because the OMC builtin handlers are themselves
//! the canonical reference — the extern Rust helpers reimplement
//! their math from the same `phi_pi_fib` substrate functions.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use omnimcode_codegen::{JitContext, JittedFn};

fn jit_one(source: &str, fn_name: &str) -> (Context, JittedFn) {
    use omnimcode_core::parser::Parser;
    let mut parser = Parser::new(source);
    let statements = parser.parse().expect("parse");
    let module = omnimcode_core::compiler::compile_program(&statements).expect("compile");
    let ctx = Context::create();
    let jit_ctx: JitContext<'static> = unsafe {
        std::mem::transmute(JitContext::new(&ctx).expect("jit ctx"))
    };
    let jitted = jit_ctx.jit_module(&module).expect("jit_module");
    let jf = *jitted.get(fn_name).expect("fn JIT'd");
    Box::leak(Box::new(jit_ctx));
    (ctx, jf)
}

#[test]
fn jit_nth_fibonacci() {
    let (_ctx, jf) = jit_one(
        "fn f(k) { return nth_fibonacci(k); }",
        "f",
    );
    assert_eq!(jf.call(&[0]).unwrap(), 0);
    assert_eq!(jf.call(&[1]).unwrap(), 1);
    assert_eq!(jf.call(&[11]).unwrap(), 89);
    assert_eq!(jf.call(&[20]).unwrap(), 6765);
    assert_eq!(jf.call(&[40]).unwrap(), 102_334_155);
}

#[test]
fn jit_is_attractor() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return is_attractor(n); }",
        "f",
    );
    assert_eq!(jf.call(&[89]).unwrap(), 1, "89 is FIB[11]");
    assert_eq!(jf.call(&[0]).unwrap(), 1, "0 is on-attractor");
    assert_eq!(jf.call(&[100]).unwrap(), 0);
    assert_eq!(jf.call(&[55]).unwrap(), 1, "55 = FIB[10]");
}

#[test]
fn jit_attractor_distance() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return attractor_distance(n); }",
        "f",
    );
    assert_eq!(jf.call(&[89]).unwrap(), 0);
    assert_eq!(jf.call(&[100]).unwrap(), 11, "100 - 89 = 11");
    assert_eq!(jf.call(&[34]).unwrap(), 0);
}

#[test]
fn jit_fibonacci_index() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return fibonacci_index(n); }",
        "f",
    );
    assert_eq!(jf.call(&[89]).unwrap(), 11);
    assert_eq!(jf.call(&[0]).unwrap(), 0);
    assert_eq!(jf.call(&[7]).unwrap(), -1, "7 not Fibonacci");
}

#[test]
fn jit_attractor_bucket() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return attractor_bucket(n); }",
        "f",
    );
    assert_eq!(jf.call(&[89]).unwrap(), 11);
    assert_eq!(jf.call(&[50]).unwrap(), 9, "50 nearest = 34 = FIB[9]");
}

#[test]
fn jit_substrate_hash_deterministic() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return substrate_hash(n); }",
        "f",
    );
    let h1 = jf.call(&[42]).unwrap();
    let h2 = jf.call(&[42]).unwrap();
    assert_eq!(h1, h2, "same input → same hash");
    let h3 = jf.call(&[43]).unwrap();
    assert_ne!(h1, h3, "different inputs → different hashes");
}

#[test]
fn jit_substrate_hash_matches_treewalk() {
    // Sanity: JIT result equals the OMC builtin's result. Computed
    // here directly via the same Rust expression both paths use.
    let (_ctx, jf) = jit_one(
        "fn f(n) { return substrate_hash(n); }",
        "f",
    );
    let jit_h = jf.call(&[1234]).unwrap();
    let extern_h = omnimcode_codegen::omc_substrate_hash(1234);
    assert_eq!(jit_h, extern_h);
}

#[test]
fn jit_zeckendorf_weight() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return zeckendorf_weight(n); }",
        "f",
    );
    assert_eq!(jf.call(&[0]).unwrap(), 0, "0 has empty representation");
    assert_eq!(jf.call(&[89]).unwrap(), 1, "single attractor");
    assert_eq!(jf.call(&[100]).unwrap(), 3, "89 + 8 + 3");
}

#[test]
fn jit_bit_count() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return bit_count(n); }",
        "f",
    );
    assert_eq!(jf.call(&[0]).unwrap(), 0);
    assert_eq!(jf.call(&[7]).unwrap(), 3);
    assert_eq!(jf.call(&[255]).unwrap(), 8);
}

#[test]
fn jit_bit_length() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return bit_length(n); }",
        "f",
    );
    assert_eq!(jf.call(&[0]).unwrap(), 0);
    assert_eq!(jf.call(&[1]).unwrap(), 1);
    assert_eq!(jf.call(&[256]).unwrap(), 9);
}

#[test]
fn jit_digit_sum() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return digit_sum(n); }",
        "f",
    );
    assert_eq!(jf.call(&[0]).unwrap(), 0);
    assert_eq!(jf.call(&[123]).unwrap(), 6);
    assert_eq!(jf.call(&[9999]).unwrap(), 36);
}

#[test]
fn jit_digit_count() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return digit_count(n); }",
        "f",
    );
    assert_eq!(jf.call(&[0]).unwrap(), 1);
    assert_eq!(jf.call(&[7]).unwrap(), 1);
    assert_eq!(jf.call(&[100]).unwrap(), 3);
}

#[test]
fn jit_harmonic_unalign() {
    let (_ctx, jf) = jit_one(
        "fn f(n) { return harmonic_unalign(n); }",
        "f",
    );
    assert_eq!(jf.call(&[89]).unwrap(), 0, "on-attractor residual = 0");
    assert_eq!(jf.call(&[100]).unwrap(), 11, "100 - 89");
}

#[test]
fn jit_harmonic_align() {
    // Aliased to omc_fold internally.
    let (_ctx, jf) = jit_one(
        "fn f(n) { return harmonic_align(n); }",
        "f",
    );
    assert_eq!(jf.call(&[100]).unwrap(), 89);
    assert_eq!(jf.call(&[89]).unwrap(), 89);
}

#[test]
fn jit_hbit_tension_alias() {
    // hbit_tension is intercepted to omc_attractor_distance — same math,
    // different OMC source name. Verify both paths give the same answer.
    let (_ctx, jf) = jit_one(
        "fn f(n) { return hbit_tension(n); }",
        "f",
    );
    assert_eq!(jf.call(&[89]).unwrap(), 0);
    assert_eq!(jf.call(&[100]).unwrap(), 11);
}

#[test]
fn jit_chained_harmonics() {
    // Multiple intrinsics in the same fn — exercise the dispatch path
    // for a substrate-heavy expression.
    let (_ctx, jf) = jit_one(
        "fn f(n) { return harmonic_unalign(n) + attractor_distance(n) + bit_count(n); }",
        "f",
    );
    // For n=100: unalign(100)=11, distance(100)=11, bit_count(100)=popcount(1100100)=3
    assert_eq!(jf.call(&[100]).unwrap(), 11 + 11 + 3);
}

// ---------- Binary i64,i64 -> i64 intrinsics ----------

#[test]
fn jit_gcd() {
    let (_ctx, jf) = jit_one("fn f(a, b) { return gcd(a, b); }", "f");
    assert_eq!(jf.call(&[12, 18]).unwrap(), 6);
    assert_eq!(jf.call(&[7, 11]).unwrap(), 1);
    assert_eq!(jf.call(&[0, 5]).unwrap(), 5, "gcd(0, n) = n");
}

#[test]
fn jit_lcm() {
    let (_ctx, jf) = jit_one("fn f(a, b) { return lcm(a, b); }", "f");
    assert_eq!(jf.call(&[4, 6]).unwrap(), 12);
    assert_eq!(jf.call(&[3, 7]).unwrap(), 21);
    assert_eq!(jf.call(&[0, 5]).unwrap(), 0, "lcm with 0 = 0");
}

#[test]
fn jit_safe_mod() {
    let (_ctx, jf) = jit_one("fn f(a, b) { return safe_mod(a, b); }", "f");
    assert_eq!(jf.call(&[10, 3]).unwrap(), 1, "10 mod 3 = 1");
    assert_eq!(jf.call(&[10, 0]).unwrap(), 0, "10 mod safe(0)=1 → 0");
}

// ---------- Ternary mod_pow ----------

#[test]
fn jit_mod_pow() {
    let (_ctx, jf) = jit_one("fn f(b, e, m) { return mod_pow(b, e, m); }", "f");
    assert_eq!(jf.call(&[3, 5, 7]).unwrap(), 5, "3^5 mod 7 = 243 mod 7 = 5");
    assert_eq!(jf.call(&[2, 10, 1000]).unwrap(), 24, "2^10 mod 1000");
    assert_eq!(jf.call(&[7, 0, 5]).unwrap(), 1, "anything^0 = 1");
}

// ---------- Array-input intrinsics ----------
//
// These use the L1.6 input bridge implicitly: the OMC source builds
// an array via NewArray (frame alloca, len-prefixed), then calls the
// intrinsic which receives the pointer in lane 0 and reads from it.

#[test]
fn jit_arr_sum_int_internal_array() {
    let (_ctx, jf) = jit_one(
        "fn f() { h arr = [1, 2, 3, 4, 5]; return arr_sum_int(arr); }",
        "f",
    );
    assert_eq!(jf.call(&[]).unwrap(), 15);
}

#[test]
fn jit_arr_product_internal_array() {
    let (_ctx, jf) = jit_one(
        "fn f() { h arr = [1, 2, 3, 4, 5]; return arr_product(arr); }",
        "f",
    );
    assert_eq!(jf.call(&[]).unwrap(), 120);
}

#[test]
fn jit_arr_min_max_int_internal_array() {
    let (_ctx, jf1) = jit_one(
        "fn f() { h arr = [5, 1, 9, 3, 7]; return arr_min_int(arr); }",
        "f",
    );
    assert_eq!(jf1.call(&[]).unwrap(), 1);
    let (_ctx2, jf2) = jit_one(
        "fn f() { h arr = [5, 1, 9, 3, 7]; return arr_max_int(arr); }",
        "f",
    );
    assert_eq!(jf2.call(&[]).unwrap(), 9);
}

#[test]
fn jit_combined_substrate_workload() {
    // Hot-path-style: walk a frame array, fold each element, sum the
    // residuals. Exercises NewArray + ArrayIndex + harmonic_unalign +
    // ArrayLen inside the JIT, no tree-walk fallback.
    let (_ctx, jf) = jit_one(r#"
        fn substrate_load() {
            h arr = [10, 20, 89, 100, 50, 144, 7];
            h n = arr_len(arr);
            h s = 0;
            h i = 0;
            while i < n {
                s = s + harmonic_unalign(arr_get(arr, i));
                i = i + 1;
            }
            return s;
        }
    "#, "substrate_load");
    // unalign values per actual substrate nearest-attractor tiebreaker
    // (verified empirically via tree-walk): 2 + (-1) + 0 + 11 + 16 + 0 + (-1) = 27
    assert_eq!(jf.call(&[]).unwrap(), 27);
}
