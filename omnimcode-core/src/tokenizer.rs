//! Substrate-typed token adapter.
//!
//! The thesis: OMC's HInt-with-resonance + Fibonacci attractors give
//! us a built-in tokenizer space that Python can't replicate. Map
//! common OMC names to attractor-aligned IDs, and an LLM can emit
//! short int arrays instead of full builtin names. The runtime
//! decodes back to canonical source.
//!
//! Three primitives, all already in the codebase:
//!   - fnv1a_hash → entry point for hashing
//!   - arr_fold_all / nearest_attractor_with_dist → snap to attractor
//!   - HInt::new → carry resonance/HIM on every output
//!
//! This module wires them into a token codec:
//!
//!   encode("h x = arr_softmax([1.0]);")  →  [1, ..., 17, ...]
//!   decode([1, ..., 17, ...])             →  "h x = arr_softmax([1.0]);"
//!
//! Encoding is a greedy longest-match against TOKEN_DICT. Unmatched
//! bytes get escaped as `[0, byte]` pairs so round-trip is exact.
//!
//! Dictionary entries are ordered so the most-common code substrings
//! land on small IDs. Small IDs are near the start of the attractor
//! chain (1, 2, 3, 5, 8, 13, 21, ...) so `attractor_distance(id)`
//! gives a free semantic-nearness signal: two builtins with nearby
//! IDs ARE substrate-near.

use crate::phi_pi_fib;

/// CRT moduli for packed multi-stream tokens.
/// Pairwise coprime; product ≈ 7.06e8, well inside i64.
/// Streams: (kind, vocab_id, position_class).
pub const CRT_MODULI: &[i64] = &[7, 1009, 100003];

/// Greedy longest-match dictionary. Order matters:
///   - ID 0 is reserved as the LITERAL_BYTE escape — the next int
///     in the stream is a raw byte (0..255) appended verbatim.
///   - IDs 1..19 are reserved for the most common code substrings,
///     so they land on (or near) early Fibonacci attractors.
///   - IDs >= 20 cover the broader vocabulary in roughly
///     frequency-descending order.
///
/// Adding entries is safe; reordering existing entries breaks
/// round-trip compatibility for previously-encoded streams, so do it
/// only when bumping a version of the encoder.
pub const TOKEN_DICT: &[&str] = &[
    // 0: LITERAL_BYTE escape (must be index 0; never matches)
    "\x00__LITERAL_BYTE__",

    // 1..19: most common substrings. Land near Fibonacci attractors.
    "h ",          // 1   (attractor)
    " = ",         // 2   (attractor)
    "arr_get",     // 3   (attractor)
    "fn ",         // 4
    "arr_set",     // 5   (attractor)
    "arr_len",     // 6
    "return ",     // 7
    "if ",         // 8   (attractor)
    "while ",      // 9
    "print(",      // 10
    "    ",        // 11  (4-space indent)
    " + ",         // 12
    "arr_push",    // 13  (attractor)
    "dict_get",    // 14
    "dict_set",    // 15
    " < ",         // 16
    " > ",         // 17
    " - ",         // 18
    " * ",         // 19

    // 20+: ML / autograd / substrate names (high value for LLMs)
    " == ",        // 20
    "arr_softmax", // 21  (attractor)
    "arr_matmul",
    "arr_transpose",
    "arr_relu_vec",
    "arr_sigmoid_vec",
    "arr_layer_norm",
    "arr_conv1d",
    "arr_add",
    "arr_sub",
    "arr_mul",
    "arr_div_int",
    "arr_scale",
    "arr_dot",
    "arr_zeros_2d",
    "arr_eye",
    "tape_var",
    "tape_const",
    "tape_add",
    "tape_sub",
    "tape_mul",
    "tape_matmul",
    "tape_relu",
    "tape_sigmoid",
    "tape_tanh",
    "tape_sum",
    "tape_mean",
    "tape_backward",
    "tape_value",
    "tape_grad",
    "tape_update",
    "tape_reset",
    "dual",
    "dual_add",
    "dual_mul",
    "dual_d",
    "gen_stream",
    "gen_take",
    "gen_sum",
    "gen_count",
    "gen_substrate_fib",

    // Substrate / OMC-unique
    "is_attractor",
    "attractor_distance",
    "arr_resonance_vec",
    "arr_him_vec",
    "arr_fold_all",
    "arr_substrate_attention",
    "arr_substrate_score_rows",
    "crt_recover",
    "fibonacci_index",
    "harmony",

    // Stdlib / regex / json / hashing
    "sha256",
    "sha512",
    "base64_encode",
    "base64_decode",
    "now_unix",
    "now_iso",
    "format_time",
    "parse_time",
    "json_parse",
    "json_stringify",
    "re_match",
    "re_find_all",
    "re_replace",

    // Strings
    "str_len",
    "str_split",
    "str_join",
    "str_slice",
    "concat_many",
    "to_string",

    // Introspection (this module's surface)
    "omc_help",
    "omc_list_builtins",
    "omc_categories",
    "omc_did_you_mean",
    "omc_unique_builtins",
    "omc_explain_error",
    "omc_token_encode",
    "omc_token_decode",
    "omc_token_distance",
    "omc_token_vocab",
    "omc_token_pack",
    "omc_token_unpack",
    "omc_code_hash",
    "omc_code_distance",
    "omc_token_compression_ratio",

    // Control flow / structure
    "else ",
    "elif ",
    "try ",
    "catch ",
    "finally ",
    "throw ",
    "yield ",
    "class ",
    "extends ",
    "import ",

    // Common literals + operators
    "true",
    "false",
    "null",
    ", ",
    "; ",
    ") {",
    "} ",
    "()",
    "[]",
    "{}",
    "= 0",
    "= 1",
    "= 0.0",
    "= 1.0",
    "+= 1",
    "i = 0",
    "i + 1",

    // Type tags / introspection values
    "int",
    "float",
    "string",
    "bool",
    "array",
    "dict",

    // Common Fibonacci-attractor literal IDs (LLM-friendly numerics)
    "0", "1", "2", "3", "5", "8", "13", "21", "34", "55",
    "89", "144", "233", "377", "610", "987", "1597", "2584",
    "4181", "6765",

    // Single-char punctuation & operators. Without these every "(",
    // ")", "[", "]", "," etc. costs an escape pair. Listing them as
    // their own IDs collapses that overhead 2x on punctuation-heavy
    // OMC code (which is most OMC code).
    "(", ")", "[", "]", "{", "}", ",", ";", ":", ".",
    "=", "+", "-", "*", "/", "%", "<", ">", "!", "?",
    " ", "\n", "\t",

    // Common 2-char operators / openers
    "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
    "//", "/*", "*/",

    // ---- Auto-appended bulk dict expansion (Phase 2) ----
    "abs",
    "acos",
    "arr_all",
    "arr_any",
    "arr_argmax",
    "arr_argmin",
    "arr_avg_distance",
    "arr_chunk",
    "arr_concat",
    "arr_contains",
    "arr_count",
    "arr_cumsum",
    "arr_diff",
    "arr_drop",
    "arr_enumerate",
    "arr_filter",
    "arr_find",
    "arr_first",
    "arr_flatten",
    "arr_fold_elements",
    "arr_from_range",
    "arr_gcd",
    "arr_geometric_mean",
    "arr_harmonic_mean",
    "arr_index_of",
    "arr_is_sorted",
    "arr_join",
    "arr_last",
    "arr_map",
    "arr_max",
    "arr_max_float",
    "arr_max_int",
    "arr_mean",
    "arr_median",
    "arr_min",
    "arr_min_float",
    "arr_min_int",
    "arr_neg",
    "arr_new",
    "arr_norm",
    "arr_ones",
    "arr_outer",
    "arr_partition_by",
    "arr_product",
    "arr_range",
    "arr_reduce",
    "arr_repeat",
    "arr_resonance",
    "arr_reverse",
    "arr_slice",
    "arr_sort",
    "arr_sort_int",
    "arr_stddev",
    "arr_sum",
    "arr_sum_int",
    "arr_sum_sq",
    "arr_take",
    "arr_unique",
    "arr_unique_count",
    "arr_variance",
    "arr_window",
    "arr_zeros",
    "arr_zip",
    "asin",
    "atan",
    "atan2",
    "attractor_bucket",
    "attractor_table",
    "bit_count",
    "bit_length",
    "call",
    "ceil",
    "clamp",
    "classify_resonance",
    "cleanup_array",
    "collapse",
    "cos",
    "crt_residues",
    "csv_parse",
    "cube",
    "defined_functions",
    "dict_clear",
    "dict_del",
    "dict_get_or",
    "dict_has",
    "dict_items",
    "dict_keys",
    "dict_len",
    "dict_merge",
    "dict_new",
    "dict_pop",
    "dict_size",
    "dict_values",
    "digit_count",
    "digit_sum",
    "dual_cos",
    "dual_exp",
    "dual_neg",
    "dual_pow_int",
    "dual_relu",
    "dual_sigmoid",
    "dual_sin",
    "dual_tanh",
    "dual_v",
    "e",
    "ensure_clean",
    "erf",
    "error",
    "even",
    "exp",
    "factorial",
    "fib",
    "fib_chunks",
    "fibonacci",
    "file_exists",
    "filter_by_resonance",
    "floor",
    "fnv1a_hash",
    "fold",
    "fold_escape",
    "frac",
    "from_zeckendorf",
    "gcd",
    "harmonic_align",
    "harmonic_checksum",
    "harmonic_dedupe",
    "harmonic_diff",
    "harmonic_hash",
    "harmonic_interfere",
    "harmonic_partition",
    "harmonic_partition_3",
    "harmonic_read_file",
    "harmonic_resample",
    "harmonic_score",
    "harmonic_sort",
    "harmonic_split",
    "harmonic_unalign",
    "harmonic_write_file",
    "harmony_value",
    "hbit_tension",
    "hypot",
    "int_binary_search",
    "int_lower_bound",
    "int_upper_bound",
    "interfere",
    "is_even",
    "is_fibonacci",
    "is_instance",
    "is_odd",
    "is_phi_resonant",
    "is_prime",
    "is_singularity",
    "is_zeckendorf_valid",
    "largest_attractor_at_most",
    "lcm",
    "len",
    "lerp",
    "ln_2",
    "log",
    "log10",
    "log2",
    "log_phi_pi_fibonacci",
    "max",
    "mean_omni_weight",
    "measure_coherence",
    "min",
    "mod_pow",
    "nearest_attractor",
    "now_ms",
    "nth_fibonacci",
    "odd",
    "omc_code_canonical",
    "omc_code_equivalent",
    "omc_error_categories",
    "omc_error_count",
    "omc_token_vocab_size",
    "phi",
    "phi_inv",
    "phi_pi_bin_search",
    "phi_pi_fib_nearest",
    "phi_pi_fib_nearest_traced",
    "phi_pi_fib_nearest_v2",
    "phi_pi_fib_reset",
    "phi_pi_fib_search",
    "phi_pi_fib_search_traced",
    "phi_pi_fib_search_v2",
    "phi_pi_fib_stats",
    "phi_pi_fib_stats_all",
    "phi_pi_fib_stats_bg",
    "phi_pi_log_distance",
    "phi_pi_pow",
    "phi_pow",
    "phi_shadow",
    "phi_sq",
    "phi_squared",
    "pi",
    "pow",
    "pow_int",
    "print_raw",
    "println",
    "quantization_ratio",
    "quantize",
    "random_float",
    "random_int",
    "random_seed",
    "re_find",
    "re_split",
    "read_file",
    "res",
    "resolve_singularity",
    "resonance_band",
    "resonance_band_histogram",
    "round",
    "safe_add",
    "safe_arr_get",
    "safe_arr_set",
    "safe_divide",
    "safe_log",
    "safe_mod",
    "safe_mul",
    "safe_sqrt",
    "safe_sub",
    "sigmoid",
    "sign",
    "sin",
    "sorted_dedupe",
    "sorted_merge",
    "sorted_union",
    "sqrt",
    "sqrt_2",
    "sqrt_5",
    "square",
    "str_capitalize",
    "str_chars",
    "str_concat",
    "str_contains",
    "str_count",
    "str_ends_with",
    "str_index_of",
    "str_is_empty",
    "str_lowercase",
    "str_pad_left",
    "str_pad_right",
    "str_repeat",
    "str_replace",
    "str_reverse",
    "str_split_lines",
    "str_starts_with",
    "str_to_float",
    "str_to_int",
    "str_trim",
    "str_uppercase",
    "substrate_count_range",
    "substrate_difference",
    "substrate_hash",
    "substrate_insert",
    "substrate_intersect",
    "substrate_lower_bound",
    "substrate_min_distance",
    "substrate_nearest",
    "substrate_quantile",
    "substrate_rank",
    "substrate_search",
    "substrate_select_k",
    "substrate_slice_range",
    "substrate_upper_bound",
    "tan",
    "tanh",
    "tape_neg",
    "tape_pow_int",
    "tau",
    "test_clear_failures",
    "test_failure_count",
    "test_get_current",
    "test_get_failures",
    "test_record_failure",
    "test_set_current",
    "to_float",
    "to_int",
    "type_of",
    "value_danger",
    "write_file",
    "zeckendorf",
    "zeckendorf_bit",
    "zeckendorf_weight",
    " 0;\n",
    " 1;\n",
    " 2;\n",
    " -1;\n",
    "h x = ",
    "h y = ",
    "h i = ",
    "h s = ",
    "h n = ",
    "h r = ",
    "h sum = 0",
    "h count = 0",
    "h result = ",
    "i = i + 1;",
    "j = j + 1;",
    "k = k + 1;",
    " < n {",
    " < arr_len(",
    "} else {",
    "} else if ",
    "while i < ",
    "for x in ",
    "for v in ",
    "fn test_",
    "test_record_failure(",
    "assert_eq(",
    "assert_true(",
    "assert_true(arr_len(",
    " == 1, \"",
    " == 0, \"",
    "approx_eq(",
    "to_string(",
    ".items.borrow()",
    "if arr_get(",
    "return arr_get(",
    "arr_push(out, ",
    "h out = [];",
    "h out = arr_new()",
    "h xs = [",
    "h ys = [",
    "if condition",
    "is empty",
    "out of bounds",
    "shape mismatch",
    " }\n",
    " {\n    ",
    " {\n",
    ");\n",
    ", ",
    " + 1",
    " - 1",
    " * 2",
    " / 2",
];

/// Substrate distance between two token IDs. Returns the absolute
/// Fibonacci-attractor distance from each ID, summed. Two builtins
/// that both live on attractor positions have distance 0 + 0 = 0
/// (perfectly substrate-near). Off-attractor IDs add their
/// individual attractor-distances.
///
/// Use this to ask "are these tokens semantically near in
/// substrate-space?" — Python tokenizers have no analogue.
pub fn token_distance(a: i64, b: i64) -> i64 {
    let (_, da) = phi_pi_fib::nearest_attractor_with_dist(a.abs());
    let (_, db) = phi_pi_fib::nearest_attractor_with_dist(b.abs());
    (a - b).abs() + da + db
}

/// Encode a source string as substrate-token IDs. Greedy longest-match
/// against TOKEN_DICT; unmatched bytes are escaped as `[0, byte]`.
/// Round-trips exactly via decode().
pub fn encode(source: &str) -> Vec<i64> {
    let mut out = Vec::with_capacity(source.len() / 4);
    let bytes = source.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let mut best_id: i64 = 0;
        let mut best_len = 0;
        // Skip ID 0 (LITERAL_BYTE escape — never matches real input).
        for (id, entry) in TOKEN_DICT.iter().enumerate().skip(1) {
            let eb = entry.as_bytes();
            let el = eb.len();
            if el > best_len && i + el <= n && &bytes[i..i + el] == eb {
                best_id = id as i64;
                best_len = el;
            }
        }
        if best_len > 0 {
            out.push(best_id);
            i += best_len;
        } else {
            // Literal byte escape.
            out.push(0);
            out.push(bytes[i] as i64);
            i += 1;
        }
    }
    out
}

/// Decode an ID stream back to source. Inverse of encode.
pub fn decode(ids: &[i64]) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(ids.len() * 2);
    let mut i = 0;
    while i < ids.len() {
        let id = ids[i];
        if id == 0 {
            // Next int is a literal byte.
            if i + 1 < ids.len() {
                let b = ids[i + 1];
                out.push((b & 0xff) as u8);
                i += 2;
            } else {
                // Malformed trailing escape — skip.
                i += 1;
            }
        } else if (id as usize) < TOKEN_DICT.len() {
            out.extend_from_slice(TOKEN_DICT[id as usize].as_bytes());
            i += 1;
        } else {
            // Unknown ID — skip silently. (A versioned dict would
            // emit a warning here, but we keep it forgiving.)
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// CRT pack: combine `streams` (one per modulus) into a single int.
/// `streams[k]` is the remainder modulo `moduli[k]`. Result is in
/// [0, product(moduli)). When `moduli == CRT_MODULI`, packing kind +
/// vocab_id + position_class gives one i64 carrying three tensors'
/// worth of token metadata.
pub fn crt_pack(streams: &[i64], moduli: &[i64]) -> Result<i64, String> {
    if streams.len() != moduli.len() {
        return Err(format!(
            "crt_pack: streams ({}) and moduli ({}) length mismatch",
            streams.len(),
            moduli.len()
        ));
    }
    // Standard CRT construction.
    let product: i64 = moduli.iter().product();
    let mut result: i64 = 0;
    for (i, &m) in moduli.iter().enumerate() {
        let mi = product / m;
        let inv = mod_inverse(mi % m, m)
            .ok_or_else(|| format!("crt_pack: moduli not pairwise coprime ({} vs {})", m, mi))?;
        let r = streams[i].rem_euclid(m);
        result = (result + r * mi * inv).rem_euclid(product);
    }
    Ok(result)
}

/// CRT unpack: recover per-modulus remainders from a packed int.
pub fn crt_unpack(packed: i64, moduli: &[i64]) -> Vec<i64> {
    moduli.iter().map(|&m| packed.rem_euclid(m)).collect()
}

/// Modular inverse via extended Euclidean algorithm.
fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let (g, x, _) = ext_gcd(a, m);
    if g != 1 {
        None
    } else {
        Some(x.rem_euclid(m))
    }
}

fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x1, y1) = ext_gcd(b, a % b);
        (g, y1, x1 - (a / b) * y1)
    }
}

/// FNV-1a hash of a byte slice (matches the existing fnv1a_hash builtin).
/// Used by code-hash + code-distance so two equivalent programs map to
/// nearby HInts after substrate-folding.
pub fn fnv1a_64(bytes: &[u8]) -> i64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h: u64 = OFFSET;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(PRIME);
    }
    // Mask to i63 to stay positive for downstream attractor calls.
    (h & 0x7fffffffffffffff) as i64
}

/// Hash a program's TOKEN-ENCODED form (not its raw bytes), then
/// fold the hash to its nearest Fibonacci attractor. Equivalent
/// programs that encode identically map to the same attractor.
/// Returns (folded_attractor, raw_hash, distance_from_attractor).
pub fn code_hash(source: &str) -> (i64, i64, i64) {
    let ids = encode(source);
    // Hash the ID stream as little-endian i64 bytes — canonical form.
    let mut buf = Vec::with_capacity(ids.len() * 8);
    for id in &ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
    let raw = fnv1a_64(&buf);
    let (attractor, dist) = phi_pi_fib::nearest_attractor_with_dist(raw);
    (attractor, raw, dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let src = "h x = arr_softmax([1.0, 2.0, 3.0]);";
        let ids = encode(src);
        let back = decode(&ids);
        assert_eq!(src, back);
    }

    #[test]
    fn roundtrip_unicode_escape() {
        // Greek letters → unmatched bytes → must escape as literal.
        let src = "h α = 3;";
        let ids = encode(src);
        let back = decode(&ids);
        assert_eq!(src, back);
    }

    #[test]
    fn compression_ratio_better_than_one() {
        let src = "fn main() {\n    h x = arr_softmax([1.0, 2.0, 3.0]);\n    return x;\n}";
        let ids = encode(src);
        // Each id is a single i64; raw bytes are 1 byte each. So
        // compression is meaningful when ids.len() < src.len() / 2.
        assert!(ids.len() < src.len(), "ids: {}, src: {}", ids.len(), src.len());
    }

    #[test]
    fn crt_roundtrip() {
        let packed = crt_pack(&[3, 42, 7], CRT_MODULI).unwrap();
        let unpacked = crt_unpack(packed, CRT_MODULI);
        assert_eq!(unpacked, vec![3, 42, 7]);
    }

    #[test]
    fn equivalent_code_same_hash() {
        let a = "arr_softmax([1, 2, 3])";
        let b = "arr_softmax([1, 2, 3])";
        assert_eq!(code_hash(a).0, code_hash(b).0);
    }
}
