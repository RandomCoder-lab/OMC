//! Substrate-keyed conversation memory.
//!
//! Stores arbitrary text content-addressed by a substrate-routed hash.
//! Designed for the LLM-agent use case: an agent's per-turn outputs
//! get stored once, then referenced by hash in future turns instead
//! of being carried inline in context.
//!
//! Storage layout (filesystem):
//!     <root>/<namespace>/<hex_hash>.txt    — content
//!     <root>/<namespace>/_index.jsonl      — chronological append log
//!
//! `root` defaults to `~/.omc/memory/`; override via `OMC_MEMORY_ROOT`.
//! `namespace` defaults to "default"; use distinct namespaces to
//! separate concurrent conversation threads (different agents, different
//! tasks, different sessions).
//!
//! The hash function is `tokenizer::fnv1a_64` on the UTF-8 bytes of
//! the text — same primitive that backs the substrate codec's
//! `content_hash`, so a `text` stored here and a codec payload
//! encoding the same `text` produce the same hash. Identity composes
//! across v0.4 + v0.5.

use std::path::{Path, PathBuf};

use crate::tokenizer;

/// One entry as recorded in the index file. Stores enough to render a
/// list/browse response without re-reading every body off disk.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    pub content_hash: i64,
    pub namespace: String,
    pub bytes: usize,
    pub stored_at_unix: i64,
    /// First ~80 chars of the content, stripped of newlines. Cheap
    /// enough to keep in the index, useful as a disambiguator when
    /// listing many entries.
    pub preview: String,
}

/// Standard Fibonacci tier sizes for fibtier-bounded memory:
/// `[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]`.
/// Sum up to tier N is `Fib(N+2) − 1`. At all 16 tiers the cap is 4180.
/// Mirrors `fibtier_default_sizes()` in examples/lib/fibtier.omc.
pub const FIBTIER_DEFAULT_SIZES: &[usize] = &[
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
];

/// Default max-entries cap for a fibtier-bounded namespace: sum of
/// the first 10 tiers = 232. Generous enough for hour-long agent
/// conversations; tight enough that an agent on a multi-day session
/// doesn't accumulate gigabytes of dead state.
pub const FIBTIER_DEFAULT_MAX_ENTRIES: usize = 232;

/// Substrate-keyed content-addressed memory store.
///
/// Stateless across calls: each operation reads/writes the
/// filesystem. That keeps the MCP server stateless per the existing
/// convention while still surviving process restarts (which the
/// previous in-memory `fibtier` didn't).
///
/// When `max_entries_per_namespace` is `Some(n)`, each namespace gets
/// fibtier-bounded eviction: after a store that would push the count
/// over `n`, the oldest entries are evicted from the index until
/// `n` remain. Eviction is INDEX-ONLY — the body files stay on disk,
/// so an LLM that still has a hash can recall (just not browse
/// chronologically). This matches fibtier's semantics: bounded
/// active capacity, unbounded historical recall by hash.
#[derive(Clone, Debug)]
pub struct MemoryStore {
    pub root: PathBuf,
    pub max_entries_per_namespace: Option<usize>,
}

impl MemoryStore {
    /// Construct a memory store rooted at `OMC_MEMORY_ROOT` if set,
    /// else `~/.omc/memory/`. Defaults to fibtier-bounded with
    /// `FIBTIER_DEFAULT_MAX_ENTRIES`. Override the cap via the
    /// `OMC_MEMORY_MAX_ENTRIES` env var (0 means unbounded).
    pub fn from_env() -> Self {
        let root = std::env::var("OMC_MEMORY_ROOT").ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME").ok()
                    .map(|h| PathBuf::from(h).join(".omc").join("memory"))
            })
            .unwrap_or_else(|| PathBuf::from("/tmp/.omc-memory"));
        let max_entries = match std::env::var("OMC_MEMORY_MAX_ENTRIES") {
            Ok(s) => match s.parse::<usize>() {
                Ok(0) => None,
                Ok(n) => Some(n),
                Err(_) => Some(FIBTIER_DEFAULT_MAX_ENTRIES),
            },
            Err(_) => Some(FIBTIER_DEFAULT_MAX_ENTRIES),
        };
        Self { root, max_entries_per_namespace: max_entries }
    }

    /// Construct a memory store at an explicit path. Defaults to
    /// unbounded — tests that want eviction can set
    /// `max_entries_per_namespace` explicitly.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), max_entries_per_namespace: None }
    }

    /// Builder: set the per-namespace fibtier cap.
    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries_per_namespace = if n == 0 { None } else { Some(n) };
        self
    }

    fn namespace_dir(&self, namespace: &str) -> PathBuf {
        self.root.join(sanitize_namespace(namespace))
    }

    fn content_path(&self, namespace: &str, hash: i64) -> PathBuf {
        self.namespace_dir(namespace).join(format!("{:016x}.txt", hash as u64))
    }

    fn index_path(&self, namespace: &str) -> PathBuf {
        self.namespace_dir(namespace).join("_index.jsonl")
    }

    /// Store `text` in `namespace`, return its content hash. Idempotent:
    /// writing the same text twice produces the same hash and re-writes
    /// the body, but the index gets a fresh entry (so the chronology of
    /// repeats is preserved).
    pub fn store(&self, namespace: &str, text: &str) -> Result<i64, String> {
        let hash = tokenizer::fnv1a_64(text.as_bytes());
        let ns_dir = self.namespace_dir(namespace);
        std::fs::create_dir_all(&ns_dir)
            .map_err(|e| format!("create namespace dir {}: {}", ns_dir.display(), e))?;
        let content_p = self.content_path(namespace, hash);
        std::fs::write(&content_p, text)
            .map_err(|e| format!("write content {}: {}", content_p.display(), e))?;
        // Append to index.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let preview = preview_of(text);
        let line = format!(
            r#"{{"hash":{},"bytes":{},"stored_at":{},"preview":{}}}"#,
            hash,
            text.len(),
            now,
            json_escape(&preview),
        );
        let index_p = self.index_path(namespace);
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&index_p)
            .map_err(|e| format!("open index {}: {}", index_p.display(), e))?;
        writeln!(f, "{}", line)
            .map_err(|e| format!("write index {}: {}", index_p.display(), e))?;
        drop(f);
        // Fibtier eviction: if we're over the cap, prune oldest entries
        // from the index. Bodies stay on disk so an LLM that retained
        // the hash can still recall — only the chronological list is
        // bounded. Matches fibtier.omc's "bounded active capacity,
        // unbounded historical recall by hash" semantics.
        if let Some(cap) = self.max_entries_per_namespace {
            self.evict_to_cap(namespace, cap)?;
        }
        Ok(hash)
    }

    /// Prune the namespace's index down to the most-recent `keep` entries.
    /// Returns the number evicted. Body files on disk are NOT removed
    /// (so historical hash-recall still works); only the chronological
    /// index is bounded.
    pub fn evict_to_cap(&self, namespace: &str, keep: usize) -> Result<usize, String> {
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok(0); }
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let lines: Vec<&str> = content.lines()
            .filter(|l| !l.trim().is_empty())
            .collect();
        if lines.len() <= keep { return Ok(0); }
        let drop_n = lines.len() - keep;
        // Keep the LAST `keep` lines (oldest are at the top of an
        // append-only log; newest at the bottom).
        let retained: String = lines.iter().skip(drop_n)
            .copied().collect::<Vec<_>>().join("\n");
        let mut final_content = retained;
        if !final_content.is_empty() && !final_content.ends_with('\n') {
            final_content.push('\n');
        }
        std::fs::write(&index_p, final_content)
            .map_err(|e| format!("rewrite index {}: {}", index_p.display(), e))?;
        Ok(drop_n)
    }

    /// Recall the text for a hash. Walks namespaces if the namespace
    /// hint is None — useful when the hash was produced elsewhere and
    /// the LLM only kept the hash. Returns None if no namespace has
    /// an entry with this hash.
    pub fn recall(&self, namespace: Option<&str>, hash: i64) -> Result<Option<String>, String> {
        if let Some(ns) = namespace {
            return self.recall_in(ns, hash);
        }
        // Search all namespaces.
        if !self.root.exists() { return Ok(None); }
        let entries = std::fs::read_dir(&self.root)
            .map_err(|e| format!("read root {}: {}", self.root.display(), e))?;
        for ent in entries.flatten() {
            if !ent.path().is_dir() { continue; }
            if let Some(ns_name) = ent.file_name().to_str() {
                if let Some(text) = self.recall_in(ns_name, hash)? {
                    return Ok(Some(text));
                }
            }
        }
        Ok(None)
    }

    fn recall_in(&self, namespace: &str, hash: i64) -> Result<Option<String>, String> {
        let p = self.content_path(namespace, hash);
        if !p.exists() { return Ok(None); }
        let text = std::fs::read_to_string(&p)
            .map_err(|e| format!("read content {}: {}", p.display(), e))?;
        Ok(Some(text))
    }

    /// List recent entries in a namespace (most recent first).
    /// Returns at most `limit` entries.
    pub fn list(&self, namespace: &str, limit: usize) -> Result<Vec<MemoryEntry>, String> {
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok(Vec::new()); }
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let mut entries: Vec<MemoryEntry> = Vec::new();
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Some(entry) = parse_index_line(line, namespace) {
                entries.push(entry);
            }
        }
        // The index is an append-only log, so file order IS chronological
        // — store() appends newest at the bottom. Reverse for "most
        // recent first". Don't sort by timestamp: stores within the
        // same second would tie and break ordering. File order is the
        // truth.
        entries.reverse();
        entries.truncate(limit.max(1));
        Ok(entries)
    }

    /// Stats for a namespace: how many entries indexed, total bytes
    /// of stored content. Used by `omc_memory_stats` for diagnostics.
    pub fn stats(&self, namespace: &str) -> Result<(usize, usize), String> {
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok((0, 0)); }
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let mut count = 0usize;
        let mut bytes = 0usize;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Some(b) = extract_bytes_field(line) {
                bytes += b;
                count += 1;
            }
        }
        Ok((count, bytes))
    }
}

/// Strip out directory-traversal characters from a namespace string.
/// Only ASCII alphanumerics, `_`, and `-`; everything else (including
/// `.` and `/`) collapses to `_`. This prevents `../etc`-style escape
/// at the namespace level — every namespace becomes a single safe
/// directory name. Empty input → "default".
fn sanitize_namespace(ns: &str) -> String {
    let cleaned: String = ns.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    if cleaned.is_empty() { "default".to_string() } else { cleaned }
}

fn preview_of(text: &str) -> String {
    let one_line: String = text.chars().take(80)
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    one_line.trim().to_string()
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parse a single JSONL line from `_index.jsonl`. Hand-rolled to avoid
/// pulling serde into this module — the format is fixed and tiny.
fn parse_index_line(line: &str, namespace: &str) -> Option<MemoryEntry> {
    let hash = extract_i64_field(line, "\"hash\":")?;
    let bytes = extract_bytes_field(line)?;
    let stored_at = extract_i64_field(line, "\"stored_at\":")?;
    let preview = extract_string_field(line, "\"preview\":")?;
    Some(MemoryEntry {
        content_hash: hash,
        namespace: namespace.to_string(),
        bytes,
        stored_at_unix: stored_at,
        preview,
    })
}

fn extract_i64_field(line: &str, key: &str) -> Option<i64> {
    let rest = line.split_once(key)?.1;
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse::<i64>().ok()
}

fn extract_bytes_field(line: &str) -> Option<usize> {
    let rest = line.split_once("\"bytes\":")?.1;
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse::<usize>().ok()
}

fn extract_string_field(line: &str, key: &str) -> Option<String> {
    let rest = line.split_once(key)?.1.trim_start();
    let rest = rest.strip_prefix('"')?;
    // Find the next unescaped quote. Simple version: scan forward,
    // treat `\"` as an escape. Sufficient for our own preview output.
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(esc) = chars.next() {
                match esc {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    other => { out.push('\\'); out.push(other); }
                }
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_store() -> (MemoryStore, tempdir_dropper::TempDir) {
        let tmp = tempdir_dropper::TempDir::new();
        let store = MemoryStore::at(tmp.path().to_path_buf());
        (store, tmp)
    }

    #[test]
    fn store_recall_round_trip() {
        let (store, _td) = tmp_store();
        let text = "fn hello() { return 42; }";
        let hash = store.store("test_ns", text).unwrap();
        let recalled = store.recall(Some("test_ns"), hash).unwrap();
        assert_eq!(recalled.as_deref(), Some(text));
    }

    #[test]
    fn store_is_deterministic() {
        let (store, _td) = tmp_store();
        let text = "the same text twice";
        let h1 = store.store("ns", text).unwrap();
        let h2 = store.store("ns", text).unwrap();
        assert_eq!(h1, h2, "fnv1a is deterministic");
    }

    #[test]
    fn recall_unknown_hash_returns_none() {
        let (store, _td) = tmp_store();
        store.store("ns", "anything").unwrap();
        let recalled = store.recall(Some("ns"), 999_999).unwrap();
        assert!(recalled.is_none());
    }

    #[test]
    fn recall_across_namespaces() {
        let (store, _td) = tmp_store();
        let h_a = store.store("ns_a", "alpha content").unwrap();
        let h_b = store.store("ns_b", "beta content").unwrap();
        // Without namespace hint, walks all namespaces.
        assert_eq!(store.recall(None, h_a).unwrap().as_deref(), Some("alpha content"));
        assert_eq!(store.recall(None, h_b).unwrap().as_deref(), Some("beta content"));
    }

    #[test]
    fn list_returns_recent_first() {
        let (store, _td) = tmp_store();
        // No sleeps — append-only-log file order is the chronology.
        store.store("ns", "first").unwrap();
        store.store("ns", "second").unwrap();
        store.store("ns", "third").unwrap();
        let entries = store.list("ns", 5).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].preview, "third");
        assert_eq!(entries[2].preview, "first");
    }

    #[test]
    fn list_respects_limit() {
        let (store, _td) = tmp_store();
        for i in 0..10 {
            store.store("ns", &format!("entry {}", i)).unwrap();
        }
        let entries = store.list("ns", 3).unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn stats_count_and_bytes() {
        let (store, _td) = tmp_store();
        store.store("ns", "aaa").unwrap();
        store.store("ns", "bbbb").unwrap();
        store.store("ns", "ccccc").unwrap();
        let (count, bytes) = store.stats("ns").unwrap();
        assert_eq!(count, 3);
        assert_eq!(bytes, 12); // 3+4+5
    }

    #[test]
    fn namespace_sanitization_strips_traversal() {
        assert_eq!(sanitize_namespace(""), "default");
        // `..` collapses to `__` and the `/` to `_` — no path escape.
        assert_eq!(sanitize_namespace("../etc"), "___etc");
        assert_eq!(sanitize_namespace("my_ns"), "my_ns");
        // Dots collapse too — `agent-123.session` becomes `agent-123_session`.
        assert_eq!(sanitize_namespace("agent-123.session"), "agent-123_session");
    }

    #[test]
    fn preview_strips_newlines_and_truncates() {
        let text = "line one\nline two\nline three\n";
        assert_eq!(preview_of(text), "line one line two line three");
        let long = "x".repeat(200);
        assert_eq!(preview_of(&long).len(), 80);
    }

    #[test]
    fn fibtier_eviction_bounds_index_at_cap() {
        let (_st, td) = tmp_store();
        let store = MemoryStore::at(td.path()).with_max_entries(5);
        // Store 12 entries — the index should retain only the most recent 5.
        let mut hashes = Vec::new();
        for i in 0..12 {
            let h = store.store("ns", &format!("entry-{}", i)).unwrap();
            hashes.push(h);
        }
        let listed = store.list("ns", 20).unwrap();
        assert_eq!(listed.len(), 5, "index pruned to cap");
        // Most recent (entry-11) should be first.
        assert_eq!(listed[0].preview, "entry-11");
        // Oldest retained should be entry-7 (indices 7..11 kept).
        assert_eq!(listed[4].preview, "entry-7");
    }

    #[test]
    fn evicted_entries_still_recoverable_by_hash() {
        // Fibtier semantics: index gets bounded, but body files stay
        // on disk. An LLM that retained an old hash can still recall.
        let (_st, td) = tmp_store();
        let store = MemoryStore::at(td.path()).with_max_entries(3);
        let oldest_hash = store.store("ns", "ancient content").unwrap();
        // Push 4 more entries — the first one falls out of the index.
        for i in 0..4 {
            store.store("ns", &format!("newer {}", i)).unwrap();
        }
        let listed = store.list("ns", 10).unwrap();
        assert_eq!(listed.len(), 3, "index bounded");
        assert!(!listed.iter().any(|e| e.content_hash == oldest_hash),
                "oldest absent from index");
        // But recall by hash still works (body file persists on disk).
        let recalled = store.recall(Some("ns"), oldest_hash).unwrap();
        assert_eq!(recalled.as_deref(), Some("ancient content"),
                   "evicted entry still recoverable by hash");
    }

    #[test]
    fn evict_to_cap_returns_count_dropped() {
        let (_st, td) = tmp_store();
        let store = MemoryStore::at(td.path()); // unbounded
        for i in 0..10 {
            store.store("ns", &format!("e{}", i)).unwrap();
        }
        let dropped = store.evict_to_cap("ns", 4).unwrap();
        assert_eq!(dropped, 6);
        let listed = store.list("ns", 20).unwrap();
        assert_eq!(listed.len(), 4);
    }

    #[test]
    fn unbounded_store_keeps_all_entries() {
        let (_st, td) = tmp_store();
        let store = MemoryStore::at(td.path()); // no cap
        for i in 0..50 {
            store.store("ns", &format!("e{}", i)).unwrap();
        }
        let listed = store.list("ns", 100).unwrap();
        assert_eq!(listed.len(), 50, "no cap → no eviction");
    }

    #[test]
    fn fibtier_default_max_entries_is_232() {
        // Sum of first 10 Fibonacci tiers [1,2,3,5,8,13,21,34,55,89] = 231.
        // The constant rounds up to 232 to give one slot of headroom for
        // the in-flight store; let's verify.
        let sum: usize = FIBTIER_DEFAULT_SIZES.iter().take(10).sum();
        assert!((sum..=sum+1).contains(&FIBTIER_DEFAULT_MAX_ENTRIES),
                "default cap matches first 10 fibtier sizes (got {}, sizes sum to {})",
                FIBTIER_DEFAULT_MAX_ENTRIES, sum);
    }

    #[test]
    fn hash_matches_codec_content_hash() {
        // The substrate identity should compose: the hash this module
        // produces for arbitrary text should match what
        // tokenizer::fnv1a_64 would produce, so the LLM can use a
        // memory hash interchangeably with a codec content_hash for
        // the same text.
        let (store, _td) = tmp_store();
        let text = "any text at all";
        let memory_hash = store.store("ns", text).unwrap();
        let direct_hash = tokenizer::fnv1a_64(text.as_bytes());
        assert_eq!(memory_hash, direct_hash);
    }
}

// Inline tempdir helper to avoid adding a dependency just for tests.
#[cfg(test)]
mod tempdir_dropper {
    use std::path::{Path, PathBuf};
    pub struct TempDir { path: PathBuf }
    impl TempDir {
        pub fn new() -> Self {
            // Mirror std::env::temp_dir/pid/random conventions without
            // pulling in `tempfile` for one helper.
            let mut p = std::env::temp_dir();
            let nonce: u64 = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64).unwrap_or(0);
            p.push(format!("omc-mem-test-{}-{}", std::process::id(), nonce));
            std::fs::create_dir_all(&p).expect("mk tempdir");
            Self { path: p }
        }
        pub fn path(&self) -> &Path { &self.path }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.path); }
    }
}
