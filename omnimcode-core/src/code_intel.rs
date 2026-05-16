//! Code-intelligence primitives — the "what LLMs actually reach for"
//! layer on top of canonicalize + tokenize + hash.
//!
//! Each function here answers a question an LLM has when iterating on
//! code: "what's the signature surface?" "what does this depend on?"
//! "did my edit change the function shape?" "how complex is this?"
//!
//! All operations work on raw OMC source — parse + analyse + return.
//! No persistent state — the MCP / REPL caller layers session memory
//! on top if it wants.

use std::collections::BTreeSet;

use crate::ast::{Expression, ForIterable, Statement};
use crate::canonical;
use crate::parser::Parser;
use crate::tokenizer;

/// Result of extracting a function's surface: name, params, body line count.
#[derive(Clone, Debug)]
pub struct FnSummary {
    pub name: String,
    pub params: Vec<String>,
    pub param_types: Vec<Option<String>>,
    pub return_type: Option<String>,
    pub pragmas: Vec<String>,
    pub body_stmts: usize,
    /// Canonical hash of the function body — stable under renames.
    pub canonical_hash: i64,
}

/// Result of summarising a program: top-level functions + classes +
/// imports + dependencies (other builtins/fns this program calls).
#[derive(Clone, Debug)]
pub struct ProgramSummary {
    pub functions: Vec<FnSummary>,
    pub classes: Vec<String>,
    pub imports: Vec<String>,
    pub calls: BTreeSet<String>,
    pub stmt_count: usize,
}

/// Parse + summarise.
pub fn summarise(source: &str) -> Result<ProgramSummary, String> {
    let mut p = Parser::new(source);
    let stmts = p.parse().map_err(|e| format!("parse error: {}", e))?;
    let mut summary = ProgramSummary {
        functions: Vec::new(),
        classes: Vec::new(),
        imports: Vec::new(),
        calls: BTreeSet::new(),
        stmt_count: stmts.len(),
    };
    for stmt in &stmts {
        match stmt {
            Statement::FunctionDef { name, params, param_types, body, return_type, pragmas } => {
                let body_str = body_to_canonical(body);
                let (_, raw, _) = tokenizer::code_hash(&body_str);
                summary.functions.push(FnSummary {
                    name: name.clone(),
                    params: params.clone(),
                    param_types: param_types.clone(),
                    return_type: return_type.clone(),
                    pragmas: pragmas.clone(),
                    body_stmts: body.len(),
                    canonical_hash: raw,
                });
                collect_calls(body, &mut summary.calls);
            }
            Statement::ClassDef { name, methods, .. } => {
                summary.classes.push(name.clone());
                for m in methods {
                    if let Statement::FunctionDef { name: mn, params, param_types, body, return_type, pragmas } = m {
                        let body_str = body_to_canonical(body);
                        let (_, raw, _) = tokenizer::code_hash(&body_str);
                        summary.functions.push(FnSummary {
                            name: format!("{}.{}", name, mn),
                            params: params.clone(),
                            param_types: param_types.clone(),
                            return_type: return_type.clone(),
                            pragmas: pragmas.clone(),
                            body_stmts: body.len(),
                            canonical_hash: raw,
                        });
                        collect_calls(body, &mut summary.calls);
                    }
                }
            }
            Statement::Import { module, alias, selected: _ } => {
                summary.imports.push(match alias {
                    Some(a) => format!("{} as {}", module, a),
                    None => module.clone(),
                });
            }
            _ => {
                collect_calls(std::slice::from_ref(stmt), &mut summary.calls);
            }
        }
    }
    Ok(summary)
}

fn body_to_canonical(body: &[Statement]) -> String {
    // Canonicalize a body-as-prog so its hash is rename-invariant.
    use crate::formatter::format_program;
    // Wrap body in a fake fn so canonicalizer sees it as a scope.
    let wrapper = Statement::FunctionDef {
        name: "__body__".to_string(),
        params: vec![],
        param_types: vec![],
        body: body.to_vec(),
        return_type: None,
        pragmas: vec![],
    };
    let canon_stmts = vec![wrapper];
    let canonical_renamed = canonicalize_stmts(&canon_stmts);
    format_program(&canonical_renamed)
}

fn canonicalize_stmts(stmts: &[Statement]) -> Vec<Statement> {
    use crate::canonical::canonicalize;
    // Reuse the canonical module by going through a round trip.
    let src = crate::formatter::format_program(stmts);
    match canonicalize(&src) {
        Ok(canon_src) => {
            let mut p = Parser::new(&canon_src);
            p.parse().unwrap_or_else(|_| stmts.to_vec())
        }
        Err(_) => stmts.to_vec(),
    }
}

fn collect_calls(stmts: &[Statement], out: &mut BTreeSet<String>) {
    for s in stmts {
        match s {
            Statement::Print(e) | Statement::Expression(e) | Statement::Throw(e) | Statement::Yield(e) => collect_expr_calls(e, out),
            Statement::VarDecl { value, .. } | Statement::Parameter { value, .. } | Statement::Assignment { value, .. } => collect_expr_calls(value, out),
            Statement::IndexAssignment { index, value, .. } => {
                collect_expr_calls(index, out);
                collect_expr_calls(value, out);
            }
            Statement::If { condition, then_body, elif_parts, else_body } => {
                collect_expr_calls(condition, out);
                collect_calls(then_body, out);
                for (c, b) in elif_parts {
                    collect_expr_calls(c, out);
                    collect_calls(b, out);
                }
                if let Some(eb) = else_body { collect_calls(eb, out); }
            }
            Statement::While { condition, body } => {
                collect_expr_calls(condition, out);
                collect_calls(body, out);
            }
            Statement::For { iterable, body, .. } => {
                match iterable {
                    ForIterable::Range { start, end } => {
                        collect_expr_calls(start, out);
                        collect_expr_calls(end, out);
                    }
                    ForIterable::Expr(e) => collect_expr_calls(e, out),
                }
                collect_calls(body, out);
            }
            Statement::FunctionDef { body, .. } => collect_calls(body, out),
            Statement::Return(Some(e)) => collect_expr_calls(e, out),
            Statement::Try { body, handler, finally, .. } => {
                collect_calls(body, out);
                collect_calls(handler, out);
                if let Some(f) = finally { collect_calls(f, out); }
            }
            Statement::ClassDef { methods, .. } => collect_calls(methods, out),
            Statement::Match { scrutinee, arms } => {
                collect_expr_calls(scrutinee, out);
                for arm in arms { collect_calls(&arm.body, out); }
            }
            _ => {}
        }
    }
}

fn collect_expr_calls(e: &Expression, out: &mut BTreeSet<String>) {
    match e {
        Expression::Call { name, args, .. } => {
            out.insert(name.clone());
            for a in args { collect_expr_calls(a, out); }
        }
        Expression::Array(items) => for i in items { collect_expr_calls(i, out); }
        Expression::Dict(pairs) => for (k, v) in pairs { collect_expr_calls(k, out); collect_expr_calls(v, out); }
        Expression::Index { index, .. } => collect_expr_calls(index, out),
        Expression::Add(a, b) | Expression::Sub(a, b) | Expression::Mul(a, b) | Expression::Div(a, b) | Expression::Mod(a, b)
        | Expression::Eq(a, b) | Expression::Ne(a, b) | Expression::Lt(a, b) | Expression::Le(a, b) | Expression::Gt(a, b) | Expression::Ge(a, b)
        | Expression::And(a, b) | Expression::Or(a, b)
        | Expression::BitAnd(a, b) | Expression::BitOr(a, b) | Expression::BitXor(a, b)
        | Expression::Shl(a, b) | Expression::Shr(a, b) => {
            collect_expr_calls(a, out); collect_expr_calls(b, out);
        }
        Expression::Not(inner) | Expression::BitNot(inner)
        | Expression::Resonance(inner) | Expression::Fold(inner) | Expression::Safe(inner) => collect_expr_calls(inner, out),
        Expression::Lambda { body, .. } => collect_calls(body, out),
        _ => {}
    }
}

/// Cyclomatic complexity — count branch points + 1 per function.
/// Higher = more branchy = harder to test.
pub fn complexity(source: &str) -> Result<i64, String> {
    let mut p = Parser::new(source);
    let stmts = p.parse().map_err(|e| format!("parse error: {}", e))?;
    let mut score: i64 = 1;
    fn walk(stmts: &[Statement], score: &mut i64) {
        for s in stmts {
            match s {
                Statement::If { then_body, elif_parts, else_body, .. } => {
                    *score += 1;
                    *score += elif_parts.len() as i64;
                    walk(then_body, score);
                    for (_, b) in elif_parts { walk(b, score); }
                    if let Some(e) = else_body { walk(e, score); }
                }
                Statement::While { body, .. } | Statement::For { body, .. } => {
                    *score += 1;
                    walk(body, score);
                }
                Statement::Try { body, handler, finally, .. } => {
                    *score += 1;
                    walk(body, score);
                    walk(handler, score);
                    if let Some(f) = finally { walk(f, score); }
                }
                Statement::Match { arms, .. } => {
                    *score += arms.len() as i64;
                    for arm in arms { walk(&arm.body, score); }
                }
                Statement::FunctionDef { body, .. } => walk(body, score),
                Statement::ClassDef { methods, .. } => walk(methods, score),
                _ => {}
            }
        }
    }
    walk(&stmts, &mut score);
    Ok(score)
}

/// AST node count — proxy for code size that survives reformatting.
pub fn ast_size(source: &str) -> Result<i64, String> {
    let mut p = Parser::new(source);
    let stmts = p.parse().map_err(|e| format!("parse error: {}", e))?;
    let mut count: i64 = 0;
    fn walk_s(stmts: &[Statement], count: &mut i64) {
        for s in stmts {
            *count += 1;
            match s {
                Statement::If { condition, then_body, elif_parts, else_body, .. } => {
                    walk_e(condition, count);
                    walk_s(then_body, count);
                    for (c, b) in elif_parts { walk_e(c, count); walk_s(b, count); }
                    if let Some(e) = else_body { walk_s(e, count); }
                }
                Statement::While { condition, body, .. } => { walk_e(condition, count); walk_s(body, count); }
                Statement::For { body, iterable, .. } => {
                    match iterable {
                        ForIterable::Range { start, end } => { walk_e(start, count); walk_e(end, count); }
                        ForIterable::Expr(e) => walk_e(e, count),
                    }
                    walk_s(body, count);
                }
                Statement::FunctionDef { body, .. } => walk_s(body, count),
                Statement::ClassDef { methods, .. } => walk_s(methods, count),
                Statement::Print(e) | Statement::Expression(e) | Statement::Throw(e) | Statement::Yield(e) => walk_e(e, count),
                Statement::VarDecl { value, .. } | Statement::Parameter { value, .. } | Statement::Assignment { value, .. } => walk_e(value, count),
                Statement::IndexAssignment { index, value, .. } => { walk_e(index, count); walk_e(value, count); }
                Statement::Return(Some(e)) => walk_e(e, count),
                Statement::Try { body, handler, finally, .. } => {
                    walk_s(body, count); walk_s(handler, count);
                    if let Some(f) = finally { walk_s(f, count); }
                }
                Statement::Match { scrutinee, arms } => {
                    walk_e(scrutinee, count);
                    for arm in arms { walk_s(&arm.body, count); }
                }
                _ => {}
            }
        }
    }
    fn walk_e(e: &Expression, count: &mut i64) {
        *count += 1;
        match e {
            Expression::Call { args, .. } => for a in args { walk_e(a, count); }
            Expression::Array(items) => for i in items { walk_e(i, count); }
            Expression::Dict(pairs) => for (k, v) in pairs { walk_e(k, count); walk_e(v, count); }
            Expression::Index { index, .. } => walk_e(index, count),
            Expression::Add(a, b) | Expression::Sub(a, b) | Expression::Mul(a, b) | Expression::Div(a, b) | Expression::Mod(a, b)
            | Expression::Eq(a, b) | Expression::Ne(a, b) | Expression::Lt(a, b) | Expression::Le(a, b) | Expression::Gt(a, b) | Expression::Ge(a, b)
            | Expression::And(a, b) | Expression::Or(a, b)
            | Expression::BitAnd(a, b) | Expression::BitOr(a, b) | Expression::BitXor(a, b)
            | Expression::Shl(a, b) | Expression::Shr(a, b) => { walk_e(a, count); walk_e(b, count); }
            Expression::Not(inner) | Expression::BitNot(inner) | Expression::Resonance(inner) | Expression::Fold(inner) | Expression::Safe(inner) => walk_e(inner, count),
            Expression::Lambda { body, .. } => walk_s(body, count),
            _ => {}
        }
    }
    walk_s(&stmts, &mut count);
    Ok(count)
}

/// AST max-depth — proxy for nesting / readability.
pub fn ast_depth(source: &str) -> Result<i64, String> {
    let mut p = Parser::new(source);
    let stmts = p.parse().map_err(|e| format!("parse error: {}", e))?;
    fn d_s(stmts: &[Statement]) -> i64 {
        stmts.iter().map(|s| 1 + match s {
            Statement::If { then_body, elif_parts, else_body, .. } => {
                let m1 = d_s(then_body);
                let m2 = elif_parts.iter().map(|(_, b)| d_s(b)).max().unwrap_or(0);
                let m3 = else_body.as_ref().map(|b| d_s(b)).unwrap_or(0);
                m1.max(m2).max(m3)
            }
            Statement::While { body, .. } | Statement::For { body, .. } => d_s(body),
            Statement::FunctionDef { body, .. } => d_s(body),
            Statement::ClassDef { methods, .. } => d_s(methods),
            Statement::Try { body, handler, finally, .. } => {
                let mut m = d_s(body).max(d_s(handler));
                if let Some(f) = finally { m = m.max(d_s(f)); }
                m
            }
            Statement::Match { arms, .. } => arms.iter().map(|a| d_s(&a.body)).max().unwrap_or(0),
            _ => 0,
        }).max().unwrap_or(0)
    }
    Ok(d_s(&stmts))
}

/// Minify: re-emit canonical form with single-space normalization
/// (skipping newlines). Useful when bandwidth matters more than readability.
pub fn minify(source: &str) -> Result<String, String> {
    let canon = canonical::canonicalize(source)?;
    // Replace runs of whitespace with single space.
    let mut out = String::with_capacity(canon.len());
    let mut last_space = false;
    for c in canon.chars() {
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(c);
            last_space = false;
        }
    }
    Ok(out.trim().to_string())
}

/// Similarity between two programs in [0, 1]: fraction of canonical
/// tokens in common (Jaccard over multiset of token IDs).
pub fn similarity(a: &str, b: &str) -> Result<f64, String> {
    let ca = canonical::canonicalize(a)?;
    let cb = canonical::canonicalize(b)?;
    let ta = tokenizer::encode(&ca);
    let tb = tokenizer::encode(&cb);
    use std::collections::HashMap;
    let mut ca_counts: HashMap<i64, i64> = HashMap::new();
    let mut cb_counts: HashMap<i64, i64> = HashMap::new();
    for t in &ta { *ca_counts.entry(*t).or_insert(0) += 1; }
    for t in &tb { *cb_counts.entry(*t).or_insert(0) += 1; }
    let mut intersection: i64 = 0;
    let mut union: i64 = 0;
    let mut keys: BTreeSet<i64> = BTreeSet::new();
    keys.extend(ca_counts.keys().cloned());
    keys.extend(cb_counts.keys().cloned());
    for k in keys {
        let a = *ca_counts.get(&k).unwrap_or(&0);
        let b = *cb_counts.get(&k).unwrap_or(&0);
        intersection += a.min(b);
        union += a.max(b);
    }
    if union == 0 { Ok(1.0) } else { Ok(intersection as f64 / union as f64) }
}

/// Substrate-weighted fingerprint: short stable ID composed of the
/// 3 nearest Fibonacci attractors of the (canonical_hash, AST_size,
/// complexity) triple — uses CRT-pack to combine into one i64.
pub fn substrate_fingerprint(source: &str) -> Result<i64, String> {
    let canon = canonical::canonicalize(source)?;
    let (attr, _, _) = tokenizer::code_hash(&canon);
    let size = ast_size(&canon).unwrap_or(0);
    let cpx = complexity(&canon).unwrap_or(0);
    let moduli = [997i64, 991, 983]; // pairwise coprime, all <1000
    let streams = [attr.rem_euclid(moduli[0]), size.rem_euclid(moduli[1]), cpx.rem_euclid(moduli[2])];
    tokenizer::crt_pack(&streams, &moduli)
}

/// Structural diff between two programs: which functions appear only
/// in A, only in B, in both but with different bodies, or both with
/// same body. Compared after canonicalization so renames don't show
/// up as diffs.
#[derive(Clone, Debug, Default)]
pub struct CodeDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<String>,
    pub unchanged: Vec<String>,
}

pub fn diff(a: &str, b: &str) -> Result<CodeDiff, String> {
    let sa = summarise(a)?;
    let sb = summarise(b)?;
    use std::collections::HashMap;
    let a_map: HashMap<&str, i64> = sa.functions.iter()
        .map(|f| (f.name.as_str(), f.canonical_hash))
        .collect();
    let b_map: HashMap<&str, i64> = sb.functions.iter()
        .map(|f| (f.name.as_str(), f.canonical_hash))
        .collect();
    let mut diff = CodeDiff::default();
    for f in &sa.functions {
        match b_map.get(f.name.as_str()) {
            None => diff.removed.push(f.name.clone()),
            Some(&bh) if bh == f.canonical_hash => diff.unchanged.push(f.name.clone()),
            Some(_) => diff.modified.push(f.name.clone()),
        }
    }
    for f in &sb.functions {
        if !a_map.contains_key(f.name.as_str()) {
            diff.added.push(f.name.clone());
        }
    }
    diff.added.sort();
    diff.removed.sort();
    diff.modified.sort();
    diff.unchanged.sort();
    Ok(diff)
}

/// Match against a corpus of code chunks. Returns
/// Vec<(index_into_corpus, distance)> sorted by ascending distance.
///
/// **Honest framing**: distance == 0 means the corpus entry is
/// alpha-equivalent to `query` (same canonical form). Distance > 0
/// means "not equivalent" — but the *magnitude* of that distance is
/// essentially noise, because fnv1a hashes don't preserve a "nearness"
/// metric. Two programs that are structurally close can have wildly
/// different hash diffs; two programs that are structurally far apart
/// can have a small one. Treat as exact-match dedup, not as fuzzy
/// similarity ranking.
///
/// What Python's hash() can't do that this can: the *exact-match*
/// case is invariant under renames / whitespace / comments. Python's
/// hash(source) is sensitive to all three. For true fuzzy similarity,
/// use `omc_code_similarity` (Jaccard over canonical token IDs).
pub fn find_similar(query: &str, corpus: &[String]) -> Result<Vec<(usize, i64)>, String> {
    let canon_q = crate::canonical::canonicalize(query)
        .map_err(|e| format!("find_similar: query canonicalize: {}", e))?;
    let (_, raw_q, _) = crate::tokenizer::code_hash(&canon_q);
    let mut scored: Vec<(usize, i64)> = Vec::with_capacity(corpus.len());
    for (i, c) in corpus.iter().enumerate() {
        match crate::canonical::canonicalize(c) {
            Ok(canon_c) => {
                let (_, raw_c, _) = crate::tokenizer::code_hash(&canon_c);
                let d = (raw_q - raw_c).abs();
                scored.push((i, d));
            }
            Err(_) => {
                // Unparseable corpus entries get worst-case distance.
                scored.push((i, i64::MAX));
            }
        }
    }
    scored.sort_by_key(|(_, d)| *d);
    Ok(scored)
}

/// Quick metrics: substrate score + complexity + size all in one shot.
/// Computed in one parse-and-canonicalize pass each.
pub fn quick_metrics(source: &str) -> Result<std::collections::BTreeMap<String, f64>, String> {
    let mut out = std::collections::BTreeMap::new();
    let cpx = complexity(source)? as f64;
    let size = ast_size(source)? as f64;
    let depth = ast_depth(source)? as f64;
    out.insert("complexity".to_string(), cpx);
    out.insert("ast_size".to_string(), size);
    out.insert("ast_depth".to_string(), depth);
    out.insert("source_bytes".to_string(), source.len() as f64);
    let ids = crate::tokenizer::encode(source).len() as f64;
    out.insert("token_count".to_string(), ids);
    if source.len() > 0 {
        out.insert("compression_ratio".to_string(), source.len() as f64 / ids.max(1.0));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_extracts_functions() {
        let src = "fn f(x) { return x; } fn g(a, b) { return a + b; }";
        let s = summarise(src).unwrap();
        assert_eq!(s.functions.len(), 2);
        assert_eq!(s.functions[0].name, "f");
        assert_eq!(s.functions[1].name, "g");
        assert_eq!(s.functions[1].params, vec!["a", "b"]);
    }

    #[test]
    fn summary_collects_calls() {
        let src = "fn f(x) { return arr_softmax(arr_neg(x)); }";
        let s = summarise(src).unwrap();
        assert!(s.calls.contains("arr_softmax"));
        assert!(s.calls.contains("arr_neg"));
    }

    #[test]
    fn complexity_of_straight_line_is_1_plus_fn() {
        let src = "fn f(x) { return x; }";
        assert!(complexity(src).unwrap() >= 1);
    }

    #[test]
    fn complexity_grows_with_branches() {
        let simple = "fn f(x) { return x; }";
        let branchy = "fn f(x) { if x > 0 { return 1; } else { return 2; } while x > 0 { x = x - 1; } return x; }";
        assert!(complexity(branchy).unwrap() > complexity(simple).unwrap());
    }

    #[test]
    fn minify_strips_newlines() {
        let src = "fn f(x) {\n    return x;\n}";
        let m = minify(src).unwrap();
        assert!(!m.contains('\n'));
        assert!(m.contains("return"));
    }

    #[test]
    fn similarity_self_is_one() {
        let s = "fn f(x) { return arr_softmax(x); }";
        assert!((similarity(s, s).unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn similarity_different_is_less_than_one() {
        let a = "fn f(x) { return x; }";
        let b = "fn f(x) { return arr_softmax(arr_neg(x)); }";
        assert!(similarity(a, b).unwrap() < 1.0);
    }

    #[test]
    fn find_similar_perfect_match_first() {
        let q = "fn f(x) { return x + 1; }";
        let corpus = vec![
            "fn unrelated() { return 99; }".to_string(),
            "fn f(a) { return a + 1; }".to_string(),
        ];
        let r = find_similar(q, &corpus).unwrap();
        assert_eq!(r[0].0, 1);
        assert_eq!(r[0].1, 0);
    }

    #[test]
    fn find_similar_empty_corpus() {
        let r = find_similar("fn f() {}", &[]).unwrap();
        assert!(r.is_empty());
    }
}
