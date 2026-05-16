//! High-level LLM-workflow primitives.
//!
//! These combine multiple introspection / canonical / hash calls into
//! single operations an LLM actually performs:
//!   - "summarise this whole codebase"
//!   - "give me a cheatsheet for the substrate primitives"
//!   - "what changed between A and B, with explanations?"
//!   - "did anything break in my edit?"
//!
//! Each function returns a rich dict so the MCP / REPL surface gets
//! one round-trip instead of N.

use std::collections::BTreeMap;

use crate::canonical;
use crate::code_intel;
use crate::docs;
use crate::tokenizer;

/// Topic-keyed cheatsheets that bundle ~5-10 builtins per topic into
/// pre-rendered Markdown. LLMs can pull a cheatsheet for the area
/// they're working in and skip the per-builtin help round-trips.
pub fn cheatsheet(topic: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!("# OMC Cheatsheet: {}\n\n", topic));
    let cat_match = match topic {
        "ml" | "ml_kernels" => "ml_kernels",
        "substrate" => "substrate",
        "autograd" => "autograd",
        "duals" => "duals",
        "tokenizer" => "tokenizer",
        "code_intel" | "intel" => "code_intel",
        "generators" | "lazy" => "generators",
        "arrays" => "arrays",
        "dicts" => "dicts",
        "strings" => "strings",
        "stdlib" => "stdlib",
        "math" => "math",
        "regex" => "regex",
        "introspection" | "help" => "introspection",
        "io" => "io",
        "python" => "python",
        _ => "",
    };
    if cat_match.is_empty() {
        s.push_str("Available topics: ml, substrate, autograd, duals, tokenizer,\n");
        s.push_str("code_intel, generators, arrays, dicts, strings, stdlib,\n");
        s.push_str("math, regex, introspection, io, python.\n");
        return s;
    }
    let entries: Vec<&docs::BuiltinDoc> = docs::BUILTINS.iter()
        .filter(|b| b.category == cat_match)
        .collect();
    if entries.is_empty() {
        s.push_str(&format!("No entries documented for {} yet.\n", topic));
        return s;
    }
    for b in entries {
        s.push_str(&format!("## `{}`\n\n", b.name));
        s.push_str(&format!("**Sig**: `{}`\n\n", b.signature));
        s.push_str(&format!("{}\n\n", b.description));
        s.push_str(&format!("```omc\n{}\n```\n\n", b.example));
    }
    s
}

/// "Did anything break?" — diff + per-change metrics + suggested
/// regression tests to write. Returns a structured dict.
pub fn change_report(old: &str, new: &str) -> Result<BTreeMap<String, String>, String> {
    let d = code_intel::diff(old, new).map_err(|e| format!("change_report: {}", e))?;
    let new_metrics = code_intel::quick_metrics(new).map_err(|e| format!("change_report: {}", e))?;
    let mut out = BTreeMap::new();
    out.insert("added".to_string(), d.added.join(", "));
    out.insert("removed".to_string(), d.removed.join(", "));
    out.insert("modified".to_string(), d.modified.join(", "));
    out.insert("unchanged".to_string(), d.unchanged.join(", "));
    out.insert("new_complexity".to_string(),
        new_metrics.get("complexity").copied().unwrap_or(0.0).to_string());
    out.insert("new_ast_size".to_string(),
        new_metrics.get("ast_size").copied().unwrap_or(0.0).to_string());
    // Suggested action.
    let mut action = String::new();
    if !d.removed.is_empty() {
        action.push_str("Removed functions — confirm callers no longer reference them.\n");
    }
    if !d.modified.is_empty() {
        action.push_str("Modified functions — re-run tests covering them.\n");
    }
    if !d.added.is_empty() {
        action.push_str("Added functions — write tests asserting the new behaviour.\n");
    }
    if action.is_empty() {
        action.push_str("No functional changes detected (possibly whitespace/comments only).\n");
    }
    out.insert("suggested_action".to_string(), action);
    Ok(out)
}

/// "Where should I look to learn OMC's unique value?" — returns names
/// of every OMC-unique builtin, grouped by category, with one-line
/// descriptions. The canonical "this is OMC" overview.
pub fn unique_overview() -> String {
    let mut s = String::new();
    s.push_str("# OMC unique surface (no clean Python equivalent)\n\n");
    let mut by_cat: BTreeMap<&str, Vec<&docs::BuiltinDoc>> = BTreeMap::new();
    for b in docs::BUILTINS.iter().filter(|b| b.unique_to_omc) {
        by_cat.entry(b.category).or_default().push(b);
    }
    for (cat, list) in by_cat {
        s.push_str(&format!("## {}\n\n", cat));
        for b in list {
            s.push_str(&format!("- `{}` — {}\n", b.name, b.description));
        }
        s.push_str("\n");
    }
    s
}

/// Quick OMC vs Python translation table for common operations.
pub fn python_translation() -> String {
    let mut s = String::new();
    s.push_str("# Python → OMC translation\n\n");
    s.push_str("| Python | OMC |\n");
    s.push_str("|--------|-----|\n");
    let table = [
        ("len(xs)", "arr_len(xs) or len(xs)"),
        ("xs[0]", "arr_get(xs, 0)"),
        ("xs.append(v)", "arr_push(xs, v)"),
        ("xs[i] = v", "arr_set(xs, i, v)"),
        ("sum(xs)", "arr_sum_int(xs) or arr_sum(xs)"),
        ("max(xs) / min(xs)", "arr_max(xs) / arr_min(xs)"),
        ("d['k']", "dict_get(d, \"k\")"),
        ("d['k'] = v", "dict_set(d, \"k\", v)"),
        ("d.get(k, default)", "dict_get_or(d, k, default)"),
        ("k in d", "dict_has(d, k)"),
        ("d.keys() / d.values()", "dict_keys(d) / dict_values(d)"),
        ("s.split(',')", "str_split(s, \",\")"),
        ("','.join(xs)", "str_join(xs, \",\")"),
        ("s[1:4]", "str_slice(s, 1, 4)"),
        ("hash(s)", "fnv1a_hash(s) or harmonic_hash(s)"),
        ("import json; json.loads(s)", "json_parse(s)"),
        ("json.dumps(v)", "json_stringify(v)"),
        ("re.match(p, s)", "re_match(p, s)"),
        ("re.findall(p, s)", "re_find_all(p, s)"),
        ("re.sub(p, r, s)", "re_replace(p, s, r)"),
        ("numpy.dot(a, b)", "arr_dot(a, b)"),
        ("numpy.matmul(A, B)", "arr_matmul(A, B)"),
        ("numpy.softmax(xs)", "arr_softmax(xs)"),
        ("torch.tensor.backward()", "tape_backward(loss_id)"),
        ("torch.autograd.grad(y, x)", "tape_grad(x_id)"),
        ("hashlib.sha256(b).hexdigest()", "sha256(s)"),
        ("base64.b64encode(b)", "base64_encode(s)"),
        ("time.time()", "now_unix()"),
        ("# OMC-only — no Python", "is_attractor(n)"),
        ("# OMC-only — no Python", "arr_resonance_vec(xs)"),
        ("# OMC-only — no Python", "arr_substrate_attention(Q, K, V)"),
        ("# OMC-only — no Python", "tape_value(id) -> substrate-annotated HInt"),
    ];
    for (py, omc) in &table {
        s.push_str(&format!("| `{}` | `{}` |\n", py, omc));
    }
    s
}

/// Detailed builtin index — names + categories only, in markdown
/// list form. Helps an LLM scan the surface in one read.
pub fn builtin_index_markdown() -> String {
    let mut s = String::new();
    s.push_str("# OMC Builtin Index\n\n");
    let mut by_cat: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for b in docs::BUILTINS.iter() {
        by_cat.entry(b.category).or_default().push(b.name);
    }
    for (cat, names) in by_cat {
        s.push_str(&format!("## {} ({})\n\n", cat, names.len()));
        for n in names {
            s.push_str(&format!("- `{}`\n", n));
        }
        s.push_str("\n");
    }
    s
}

/// One-shot LLM bootstrap pack: index + cheatsheets for the OMC-unique
/// categories + python-translation table. Single string an LLM can
/// load at the start of a session.
pub fn bootstrap_pack() -> String {
    let mut s = String::new();
    s.push_str(&builtin_index_markdown());
    s.push_str("\n---\n\n");
    s.push_str(&unique_overview());
    s.push_str("\n---\n\n");
    s.push_str(&python_translation());
    s.push_str("\n---\n\n");
    for topic in ["substrate", "autograd", "code_intel", "tokenizer"] {
        s.push_str(&cheatsheet(topic));
        s.push_str("\n---\n\n");
    }
    s
}

/// Canonical OMC ID for a chunk of code: combines fingerprint +
/// canonical hash into one stable string identifier. Format:
/// "omcid-<fingerprint>-<short_hash>". Stable under cosmetic edits.
pub fn omc_id(source: &str) -> Result<String, String> {
    let fp = code_intel::substrate_fingerprint(source)?;
    let canon = canonical::canonicalize(source)?;
    let (_, raw, _) = tokenizer::code_hash(&canon);
    let short = format!("{:x}", raw & 0xffff_ffff);
    Ok(format!("omcid-{}-{}", fp, short))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cheatsheet_returns_substrate_entries() {
        let s = cheatsheet("substrate");
        assert!(s.contains("is_attractor") || s.contains("attractor"));
    }

    #[test]
    fn cheatsheet_unknown_topic_lists_options() {
        let s = cheatsheet("bogus");
        assert!(s.contains("Available topics"));
    }

    #[test]
    fn change_report_detects_modified() {
        let r = change_report(
            "fn f(x) { return x; }",
            "fn f(x) { return x + 1; }",
        ).unwrap();
        assert!(r.get("modified").unwrap().contains("f"));
    }

    #[test]
    fn unique_overview_lists_substrate() {
        let s = unique_overview();
        assert!(s.contains("substrate") || s.contains("attractor"));
    }

    #[test]
    fn python_translation_contains_arr_get() {
        assert!(python_translation().contains("arr_get"));
    }

    #[test]
    fn omc_id_is_stable_for_equivalent_code() {
        let a = omc_id("fn f(x) { return x; }").unwrap();
        let b = omc_id("fn f(a) { return a; }").unwrap();
        assert_eq!(a, b);
    }
}
