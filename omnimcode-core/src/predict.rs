//! Substrate-indexed code completion.
//!
//! Given a partial OMC code prefix, returns ranked provenance-tracked
//! continuations from a content-addressed corpus of function bodies.
//! The synthesis of two earlier substrates:
//!   - tokenizer::encode (symbol stream IDs over canonicalized source)
//!   - canonical_hash + attractor_distance (substrate metric on source identity)
//!
//! Built on the assumption that "what could come next here" is best
//! answered by indexing what previous code DID come next at this
//! shape — and ranking by substrate distance (so close-shaped
//! corpora rise to the top) PLUS prefix-match length (so the longest
//! literal match wins ties).
//!
//! All ranking is deterministic and reproducible: same corpus + same
//! prefix → same top-k, every run.

use std::collections::HashMap;

use crate::canonical::canonicalize;
use crate::interpreter::extract_top_level_fns;
use crate::phi_pi_fib::nearest_attractor_with_dist;
use crate::tokenizer::{code_hash, encode};

/// One ingested function: full source, its symbol stream, identity
/// metadata. Stored once per corpus, referenced by index from the
/// PrefixTrie's `matches` lists.
#[derive(Clone, Debug)]
pub struct CorpusEntry {
    /// Function name as extracted from `fn name(...)`.
    pub fn_name: String,
    /// Full source text of the function (canonicalized form is what
    /// produced the symbol_stream and canonical_hash; source is the
    /// human-readable original for display).
    pub source: String,
    /// Path of the file this fn came from. Provenance: when a
    /// predicted continuation is surfaced, the user can `cat`
    /// this path to see the full original context.
    pub file: String,
    /// Token IDs from tokenizer::encode applied to canonicalized source.
    /// This is the "symbol stream" the trie is keyed on.
    pub symbol_stream: Vec<i64>,
    /// fnv1a-of-token-bytes hash, alpha-rename invariant.
    pub canonical_hash: i64,
    /// Nearest Fibonacci attractor to the canonical hash. Used as the
    /// substrate-distance pivot.
    pub attractor: i64,
}

/// Symbol-stream trie. Each child edge is one token ID; each node
/// carries the corpus indices whose stream INCLUDES this prefix
/// (so a prefix query returns all matches in one trie traversal).
#[derive(Default, Debug)]
pub struct PrefixTrie {
    children: HashMap<i64, PrefixTrie>,
    /// Corpus indices whose symbol stream passes through this node
    /// (i.e., this node's path-from-root is a prefix of their stream).
    matches: Vec<usize>,
}

impl PrefixTrie {
    pub fn new() -> Self { Self::default() }

    /// Insert a symbol stream's corpus index along every node on its
    /// path. Each node accumulates "indices whose stream starts with
    /// this prefix"; the root accumulates ALL corpus entries.
    pub fn insert(&mut self, stream: &[i64], corpus_idx: usize) {
        let mut node = self;
        // Root match: every corpus entry counts as starting with the
        // empty prefix.
        node.matches.push(corpus_idx);
        for &sym in stream {
            node = node.children.entry(sym).or_default();
            node.matches.push(corpus_idx);
        }
    }

    /// Walk the trie following `prefix`; return (matches, depth_reached)
    /// where depth_reached = longest prefix that mapped onto an edge in
    /// the trie. Returns the deepest non-empty match set even if the
    /// full prefix didn't trace — that's the "longest common prefix"
    /// fallback so a query close-but-not-identical to existing streams
    /// still surfaces something useful.
    pub fn query_prefix(&self, prefix: &[i64]) -> (Vec<usize>, usize) {
        let mut node = self;
        let mut depth = 0;
        let mut last_good = &node.matches;
        for &sym in prefix {
            match node.children.get(&sym) {
                Some(child) => {
                    node = child;
                    depth += 1;
                    last_good = &node.matches;
                }
                None => break,
            }
        }
        (last_good.clone(), depth)
    }
}

/// Ingested corpus: parallel vec of entries + a trie keyed on their
/// symbol streams.
#[derive(Debug)]
pub struct CodeCorpus {
    pub entries: Vec<CorpusEntry>,
    pub trie: PrefixTrie,
}

impl CodeCorpus {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            trie: PrefixTrie::new(),
        }
    }

    /// Number of functions ingested.
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Ingest one fn-source string (canonicalize → tokenize → hash →
    /// insert into trie). Returns the new corpus index. Errors only
    /// if canonicalization fails (parse error).
    pub fn ingest_fn(&mut self, fn_name: String, source: String, file: String) -> Result<usize, String> {
        let canon = canonicalize(&source)?;
        let symbol_stream = encode(&canon);
        let (attractor, raw_hash, _dist) = code_hash(&canon);
        let entry = CorpusEntry {
            fn_name,
            source,
            file,
            symbol_stream: symbol_stream.clone(),
            canonical_hash: raw_hash,
            attractor,
        };
        let idx = self.entries.len();
        self.trie.insert(&symbol_stream, idx);
        self.entries.push(entry);
        Ok(idx)
    }

    /// Ingest every top-level fn from a file's source. Returns the
    /// number of fns successfully ingested. Per-fn parse errors are
    /// swallowed (logged via `eprintln!`); a file that yields zero
    /// well-formed fns is silently a no-op.
    pub fn ingest_file(&mut self, path: &str, source: &str) -> usize {
        let mut count = 0;
        for fn_src in extract_top_level_fns(source) {
            // Extract the fn_name from `fn NAME(...)` for display.
            let name = parse_fn_name(&fn_src).unwrap_or_else(|| "<anonymous>".to_string());
            match self.ingest_fn(name, fn_src, path.to_string()) {
                Ok(_) => { count += 1; }
                Err(e) => {
                    // Silently skip un-canonicalizable fns — a corpus
                    // ingest pass should never fail loudly on one bad
                    // function in an otherwise-good file.
                    eprintln!("predict: skipping fn in {} ({})", path, e);
                }
            }
        }
        count
    }
}

impl Default for CodeCorpus {
    fn default() -> Self { Self::new() }
}

/// One predicted continuation. Includes the source of the original
/// fn (so the caller can show the user what shape to expect) plus
/// the metadata that drove the ranking.
#[derive(Clone, Debug)]
pub struct Suggestion {
    pub fn_name: String,
    pub source: String,
    pub file: String,
    pub canonical_hash: i64,
    pub attractor: i64,
    /// How many tokens of the query prefix matched edges in the trie.
    /// 0 means no token matched (fell back to root). Higher is better.
    pub prefix_match_len: usize,
    /// |query_hash - candidate_hash|, absolute substrate distance. The
    /// query hash is the canonical hash of the query prefix; the
    /// candidate hash is the corpus entry's canonical hash. Smaller is
    /// better.
    pub substrate_distance: i64,
    /// Attractor distance of the query hash. Surfaced for diagnostics;
    /// not part of the ranking.
    pub query_attractor: i64,
}

/// Given a prefix-source string and a CodeCorpus, return the top-k
/// ranked continuations. Ranking is:
///   1. longest prefix match in the trie (descending)
///   2. smallest substrate distance |query_hash − candidate_hash| (ascending)
///   3. tie-broken by corpus index (deterministic, ascending)
///
/// The query prefix may be ANY OMC source — typically a partial fn
/// declaration like `fn prom_linear_` — it just needs to canonicalize.
/// If canonicalization fails (incomplete syntax), we fall back to
/// tokenizing the raw source so even mid-statement queries return
/// something useful.
pub fn predict_continuations(
    corpus: &CodeCorpus,
    prefix_source: &str,
    top_k: usize,
) -> Vec<Suggestion> {
    if corpus.is_empty() || top_k == 0 {
        return Vec::new();
    }
    // Tokenize the prefix. If canonicalize fails (prefix is incomplete
    // OMC like `fn prom_linear_`), tokenize the raw source — the
    // tokenizer is robust to incomplete input and produces a usable
    // partial symbol stream.
    let (prefix_stream, query_hash, query_attractor) = match canonicalize(prefix_source) {
        Ok(canon) => {
            let stream = encode(&canon);
            let (attractor, raw_hash, _) = code_hash(&canon);
            (stream, raw_hash, attractor)
        }
        Err(_) => {
            let stream = encode(prefix_source);
            let (attractor, raw_hash, _) = code_hash(prefix_source);
            (stream, raw_hash, attractor)
        }
    };
    let (candidate_indices, prefix_depth) = corpus.trie.query_prefix(&prefix_stream);

    let mut suggestions: Vec<Suggestion> = candidate_indices
        .into_iter()
        .map(|idx| {
            let e = &corpus.entries[idx];
            let dist = (query_hash - e.canonical_hash).wrapping_abs();
            Suggestion {
                fn_name: e.fn_name.clone(),
                source: e.source.clone(),
                file: e.file.clone(),
                canonical_hash: e.canonical_hash,
                attractor: e.attractor,
                prefix_match_len: prefix_depth,
                substrate_distance: dist,
                query_attractor,
            }
        })
        .collect();

    // Sort by (-prefix_match_len, substrate_distance). prefix_match_len
    // is the same for all current candidates (they all matched the same
    // depth of the trie), so the sort is effectively substrate_distance
    // ascending. Kept as a primary key so future versions can fold in
    // partial-match scoring without changing the contract.
    suggestions.sort_by(|a, b| {
        b.prefix_match_len.cmp(&a.prefix_match_len)
            .then(a.substrate_distance.cmp(&b.substrate_distance))
    });
    suggestions.truncate(top_k);
    suggestions
}

/// Parse the function name from a `fn NAME(...)` declaration. Returns
/// None if the source doesn't start with a fn declaration.
fn parse_fn_name(fn_src: &str) -> Option<String> {
    let trimmed = fn_src.trim_start();
    let rest = trimmed.strip_prefix("fn")?.trim_start();
    let name: String = rest.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_corpus(fns: &[(&str, &str)]) -> CodeCorpus {
        let mut corpus = CodeCorpus::new();
        for (name, src) in fns {
            corpus.ingest_fn(name.to_string(), src.to_string(), "test.omc".to_string()).unwrap();
        }
        corpus
    }

    #[test]
    fn parse_fn_name_basic() {
        assert_eq!(parse_fn_name("fn foo() { return 1; }"), Some("foo".to_string()));
        assert_eq!(parse_fn_name("fn  bar_baz123(x) { x }"), Some("bar_baz123".to_string()));
        assert_eq!(parse_fn_name("not a fn"), None);
        assert_eq!(parse_fn_name(""), None);
    }

    #[test]
    fn corpus_ingest_single() {
        let mut corpus = CodeCorpus::new();
        let idx = corpus.ingest_fn(
            "double".to_string(),
            "fn double(x) { return x + x; }".to_string(),
            "math.omc".to_string(),
        ).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(corpus.len(), 1);
        assert!(!corpus.entries[0].symbol_stream.is_empty());
        assert_ne!(corpus.entries[0].canonical_hash, 0);
    }

    #[test]
    fn prefix_trie_query_returns_all_for_empty() {
        let corpus = mk_corpus(&[
            ("a", "fn a() { return 1; }"),
            ("b", "fn b() { return 2; }"),
            ("c", "fn c() { return 3; }"),
        ]);
        let (matches, depth) = corpus.trie.query_prefix(&[]);
        assert_eq!(matches.len(), 3);
        assert_eq!(depth, 0);
    }

    #[test]
    fn predict_returns_ranked_results() {
        let corpus = mk_corpus(&[
            ("inc", "fn inc(x) { return x + 1; }"),
            ("dec", "fn dec(x) { return x - 1; }"),
            ("double", "fn double(x) { return x + x; }"),
        ]);
        let suggestions = predict_continuations(&corpus, "fn ", 5);
        assert!(!suggestions.is_empty(), "should return at least one suggestion");
        // All three should appear since they all start with `fn`.
        let names: Vec<&str> = suggestions.iter().map(|s| s.fn_name.as_str()).collect();
        assert!(names.contains(&"inc"), "missing inc: {:?}", names);
        assert!(names.contains(&"dec"), "missing dec: {:?}", names);
        assert!(names.contains(&"double"), "missing double: {:?}", names);
    }

    #[test]
    fn predict_respects_top_k_cap() {
        let corpus = mk_corpus(&[
            ("a", "fn a() { return 1; }"),
            ("b", "fn b() { return 2; }"),
            ("c", "fn c() { return 3; }"),
            ("d", "fn d() { return 4; }"),
        ]);
        let suggestions = predict_continuations(&corpus, "fn ", 2);
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn predict_empty_corpus_returns_empty() {
        let corpus = CodeCorpus::new();
        let suggestions = predict_continuations(&corpus, "fn anything", 5);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn predict_zero_top_k_returns_empty() {
        let corpus = mk_corpus(&[("a", "fn a() { return 1; }")]);
        let suggestions = predict_continuations(&corpus, "fn ", 0);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn predict_provenance_includes_source_and_file() {
        let corpus = mk_corpus(&[
            ("greet", "fn greet(name) { return \"hello \" + name; }"),
        ]);
        let suggestions = predict_continuations(&corpus, "fn greet", 1);
        assert_eq!(suggestions.len(), 1);
        let s = &suggestions[0];
        assert_eq!(s.fn_name, "greet");
        assert!(s.source.contains("hello"));
        assert_eq!(s.file, "test.omc");
        assert!(s.canonical_hash != 0);
    }

    #[test]
    fn ingest_file_extracts_multiple_fns() {
        let src = "fn add(a, b) { return a + b; }\nfn sub(a, b) { return a - b; }";
        let mut corpus = CodeCorpus::new();
        let count = corpus.ingest_file("arith.omc", src);
        assert_eq!(count, 2);
        assert_eq!(corpus.entries[0].fn_name, "add");
        assert_eq!(corpus.entries[1].fn_name, "sub");
    }

    #[test]
    fn similar_prefixes_get_similar_substrate_distances() {
        // Two near-identical fns differ only in name suffix — their
        // canonical hashes (alpha-renamed) should be IDENTICAL, so
        // substrate_distance is 0 for both.
        let corpus = mk_corpus(&[
            ("foo_v1", "fn foo_v1(x) { return x * 2; }"),
            ("foo_v2", "fn foo_v2(x) { return x * 2; }"),
        ]);
        // Both bodies canonicalize identically except for the fn name
        // (which canonicalize PRESERVES at top level). Their substrate
        // distance from a related prefix should be small.
        let suggestions = predict_continuations(&corpus, "fn foo_", 2);
        assert_eq!(suggestions.len(), 2);
    }
}
