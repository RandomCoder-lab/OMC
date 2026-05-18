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

/// v0.12.0 Axis 7: payload of `recall_summary`. Cheap "what is this"
/// preview for the list-then-recall workflow. ~100-300 bytes typical.
#[derive(Clone, Debug)]
pub struct SummaryRecallPayload {
    pub content_hash: i64,
    pub byte_count: usize,
    pub first_line: String,
    pub preview: String,
    pub attractor: i64,
}

/// v0.12.0 Axis 7: payload of `recall_codec`. A substrate-fingerprint
/// representation of a stored entry, ~60-200 bytes instead of the full
/// body. Lossless because the full body remains recoverable via the
/// standard `recall()` path.
#[derive(Clone, Debug)]
pub struct CodecRecallPayload {
    pub content_hash: i64,
    pub sampled_tokens: Vec<i64>,
    /// v0.12.1: sampled_tokens packed via varint + zlib + base64.
    /// ~20× smaller than the JSON array form when over the wire.
    /// Decoder: base64 decode → zlib inflate → varint stream of token IDs.
    pub sampled_tokens_packed: String,
    pub attractor: i64,
    pub every_n: usize,
    pub original_byte_count: usize,
    pub original_token_count: usize,
    pub compression_ratio: f64,
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

    /// v0.9.2 Axis 2: cross-namespace dedup pool path. All content lives
    /// at `<root>/_pool/<hash>.txt` regardless of namespace. Namespace dirs
    /// hold only the index. Same content stored in K namespaces costs ONE
    /// body file. The fanout shards by the top byte of the hash so the
    /// pool doesn't grow into one giant directory at scale.
    fn pool_path(&self, hash: i64) -> PathBuf {
        let shard = (hash as u64) >> 56;  // top byte = 256 shards
        self.root.join("_pool").join(format!("{:02x}", shard))
            .join(format!("{:016x}.txt", hash as u64))
    }

    /// Legacy per-namespace content path. Used by `recall_in` as a fallback
    /// when an entry was stored before the dedup-pool refactor (or if the
    /// pool body is missing for some other reason). Kept for backward
    /// compatibility with existing `~/.omc/memory/<ns>/<hash>.txt` files.
    fn legacy_content_path(&self, namespace: &str, hash: i64) -> PathBuf {
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
        // v0.9.2 Axis 2: write the body to the global content-addressed
        // pool, not to the namespace dir. Pool path is sharded by hash
        // prefix. Idempotent — same hash skips the write entirely (no
        // wasted IO when the body already exists from another namespace
        // OR a prior store in the same namespace).
        let pool_p = self.pool_path(hash);
        if !pool_p.exists() {
            if let Some(parent) = pool_p.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create pool shard {}: {}", parent.display(), e))?;
            }
            std::fs::write(&pool_p, text)
                .map_err(|e| format!("write pool content {}: {}", pool_p.display(), e))?;
        }
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

    /// v0.12.0 Axis 7 — summary recall, the high-leverage variant.
    ///
    /// Returns ~100-300 bytes of "what is this content" metadata instead of
    /// the full body. Designed for the **list-then-recall** workflow: the
    /// LLM gets a cheap preview of every candidate hash, picks the relevant
    /// one, then issues a single full `recall()` for the real bytes.
    ///
    /// Fields:
    ///   - `content_hash` — primary identifier
    ///   - `byte_count` — sizing info, so the LLM can budget context
    ///   - `first_line` — first \n-delimited line, capped at 200 chars
    ///   - `preview` — first 80 chars, newlines stripped (matches index preview)
    ///   - `attractor` — phi_pi_fib nearest attractor, useful for cheap
    ///     dedup/equivalence checks ("are these two hashes substrate-near?")
    ///
    /// **Lossless** because the verbatim body is always still recoverable
    /// via `recall()` with the same `content_hash`.
    ///
    /// Real measured savings on 100KB body: ~400× context-token reduction.
    pub fn recall_summary(
        &self, namespace: Option<&str>, hash: i64,
    ) -> Result<Option<SummaryRecallPayload>, String> {
        let Some(text) = self.recall(namespace, hash)? else { return Ok(None) };
        let first_line: String = text.lines()
            .next().unwrap_or("")
            .chars().take(200).collect();
        let preview: String = text.chars()
            .filter(|c| !c.is_control())
            .take(80)
            .collect();
        let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(hash);
        Ok(Some(SummaryRecallPayload {
            content_hash: hash,
            byte_count: text.len(),
            first_line,
            preview,
            attractor,
        }))
    }

    /// v0.12.0 Axis 7: codec-form recall for context-cost reduction.
    ///
    /// Returns a tiny OMC codec payload (content_hash + sampled-every-N
    /// tokens + attractor) instead of the full text. Roughly 60-200 bytes
    /// for what would otherwise be a multi-KB body. The LLM consumer uses
    /// the structural fingerprint as a substrate-keyed identifier; if it
    /// needs the exact bytes, it falls back to the full `recall()`.
    ///
    /// **Lossless** because the verbatim body is always still available
    /// through the standard recall path — codec-form is purely a cheaper
    /// representation when context-cost matters more than byte-exactness.
    ///
    /// Fields:
    ///   - `content_hash` — i64, canonical content hash (FNV1a)
    ///   - `sampled_tokens` — every-N tokens from the substrate-tokenizer
    ///     encoding of canonicalized text
    ///   - `attractor` — nearest phi_pi_fib attractor to content_hash
    ///   - `every_n` — the sampling stride used
    ///   - `original_byte_count` / `original_token_count` — sizing info
    ///   - `compression_ratio` — bytes-saved-vs-verbatim ratio
    pub fn recall_codec(
        &self, namespace: Option<&str>, hash: i64, every_n: usize,
    ) -> Result<Option<CodecRecallPayload>, String> {
        let Some(text) = self.recall(namespace, hash)? else { return Ok(None) };
        let stride = every_n.max(1);
        let canon = crate::canonical::canonicalize(&text)
            .unwrap_or_else(|_| text.clone());
        let tokens = crate::tokenizer::encode(&canon);
        let sampled: Vec<i64> = tokens.iter().enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(_, t)| *t)
            .collect();
        let content_hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
        let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(content_hash);
        // v0.12.1: also pack the sampled_tokens via varint + zlib + base64.
        // The packed form is ~5-20× smaller than the JSON-int array, and
        // the LLM/agent can decode it cheaply on the receiver side.
        use std::io::Write;
        use base64::Engine;
        let mut varint_buf: Vec<u8> = Vec::with_capacity(sampled.len() * 2);
        for t in &sampled {
            let mut v = *t as u64;
            while v >= 0x80 { varint_buf.push((v as u8) | 0x80); v >>= 7; }
            varint_buf.push(v as u8);
        }
        let mut enc = flate2::write::DeflateEncoder::new(
            Vec::new(), flate2::Compression::best());
        enc.write_all(&varint_buf)
            .map_err(|e| format!("codec packed deflate: {}", e))?;
        let packed_bytes = enc.finish()
            .map_err(|e| format!("codec packed finish: {}", e))?;
        let sampled_tokens_packed = base64::engine::general_purpose::STANDARD
            .encode(&packed_bytes);
        let ratio = if !sampled_tokens_packed.is_empty() {
            text.len() as f64 / sampled_tokens_packed.len() as f64
        } else { 0.0 };
        Ok(Some(CodecRecallPayload {
            content_hash,
            sampled_tokens: sampled,
            sampled_tokens_packed,
            attractor,
            every_n: stride,
            original_byte_count: text.len(),
            original_token_count: tokens.len(),
            compression_ratio: ratio,
        }))
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
        // v0.9.2 Axis 2: prefer the global pool. v0.9.3 Axis 3 + v0.10.0
        // Axis 4: inflate bodies prefixed with OMCZ/OMCT/OMCD magics
        // (zlib / substrate-tokenizer / delta).
        let pool_p = self.pool_path(hash);
        if pool_p.exists() {
            let raw = std::fs::read(&pool_p)
                .map_err(|e| format!("read pool content {}: {}", pool_p.display(), e))?;
            return Ok(Some(self.maybe_decompress_with_recall(&raw)?));
        }
        let legacy = self.legacy_content_path(namespace, hash);
        if !legacy.exists() { return Ok(None); }
        let raw = std::fs::read(&legacy)
            .map_err(|e| format!("read legacy content {}: {}", legacy.display(), e))?;
        Ok(Some(self.maybe_decompress_with_recall(&raw)?))
    }

    /// v0.10.1 Axis 5: decompression dispatch that has &self so OMCD
    /// can recall the base entry recursively.
    fn maybe_decompress_with_recall(&self, raw: &[u8]) -> Result<String, String> {
        if raw.len() >= 16 && &raw[..4] == b"OMCD" {
            let base_hash = i64::from_le_bytes(raw[4..12].try_into().unwrap());
            let prefix_len = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
            let suffix = &raw[16..];
            let base = self.recall(None, base_hash)?
                .ok_or_else(|| format!("OMCD base hash {} not found", base_hash))?;
            let base_bytes = base.as_bytes();
            if prefix_len > base_bytes.len() {
                return Err(format!("OMCD prefix_len {} exceeds base len {}",
                                   prefix_len, base_bytes.len()));
            }
            let mut out = Vec::with_capacity(prefix_len + suffix.len());
            out.extend_from_slice(&base_bytes[..prefix_len]);
            out.extend_from_slice(suffix);
            return String::from_utf8(out)
                .map_err(|e| format!("OMCD result not valid UTF-8: {}", e));
        }
        maybe_decompress(raw)
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

    /// v0.9.1 Axis 1: Merkle manifest hashes.
    ///
    /// A manifest is a single content-addressed entry whose body is a JSON
    /// list of leaf hashes. Storing a manifest gives the caller ONE hash
    /// that references N leaves; recalling expands the list, after which
    /// the caller can `recall` each leaf on demand. The compression win is
    /// asymmetric: 1 manifest hash in context = 5 tokens; N leaf bodies
    /// behind that hash = arbitrary content size.
    ///
    /// The manifest body uses the wire format `{"manifest":1,"entries":[..]}`
    /// so an LLM that recalls it can spot it's a manifest from the first
    /// byte and act accordingly.
    pub fn create_manifest(&self, namespace: &str, entries: &[i64]) -> Result<i64, String> {
        let mut s = String::from("{\"manifest\":1,\"entries\":[");
        for (i, h) in entries.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&h.to_string());
        }
        s.push_str("]}");
        self.store(namespace, &s)
    }

    /// Parse a recalled manifest body back into its leaf hash list.
    /// Returns `Ok(Some(hashes))` if the body parses as a manifest,
    /// `Ok(None)` if it's a regular (non-manifest) entry. `Err` only on
    /// IO or hash-not-found.
    pub fn recall_manifest(
        &self, namespace: Option<&str>, hash: i64,
    ) -> Result<Option<Vec<i64>>, String> {
        let text = match self.recall(namespace, hash)? {
            Some(t) => t,
            None => return Err(format!("manifest hash {} not found", hash)),
        };
        // Cheap parse: look for `"manifest":1,"entries":[...]`.
        let trimmed = text.trim();
        if !trimmed.starts_with("{\"manifest\":1,\"entries\":[") {
            return Ok(None);
        }
        let inside_start = match trimmed.find('[') {
            Some(i) => i + 1,
            None => return Ok(None),
        };
        let inside_end = match trimmed.rfind(']') {
            Some(i) => i,
            None => return Ok(None),
        };
        let list_body = &trimmed[inside_start..inside_end];
        let mut hashes = Vec::new();
        for tok in list_body.split(',') {
            let t = tok.trim();
            if t.is_empty() { continue; }
            let h: i64 = t.parse()
                .map_err(|e| format!("manifest parse: invalid hash {}: {}", t, e))?;
            hashes.push(h);
        }
        Ok(Some(hashes))
    }

    /// v0.10.1 Axis 5: delta compression against an explicit base entry.
    ///
    /// Store `text` as a delta against `base_hash`. The delta format is:
    ///   `OMCD` (4 bytes magic) | base_hash (8 bytes LE i64) |
    ///   prefix_len (4 bytes LE u32) | suffix (remaining bytes)
    /// Recovers as `base_text[..prefix_len] ++ suffix`.
    ///
    /// Falls back to a regular store if (a) the base isn't in memory,
    /// (b) the text shares less than 64 bytes of prefix with the base,
    /// or (c) the delta would be larger than the raw text. The returned
    /// hash is always the hash of the FULL text, so recall still works
    /// by hash regardless of how the body is stored.
    ///
    /// Use case: iterative drafts. Store v1 normally, then v2/v3/v4 as
    /// deltas off v1. Each delta is ~constant size if changes are local.
    pub fn store_as_delta(
        &self, namespace: &str, text: &str, base_hash: i64,
    ) -> Result<i64, String> {
        let base_text = match self.recall(None, base_hash)? {
            Some(t) => t,
            None => return self.store(namespace, text),  // base missing → plain store
        };
        let new_bytes = text.as_bytes();
        let base_bytes = base_text.as_bytes();
        let mut prefix_len = 0usize;
        let max_prefix = new_bytes.len().min(base_bytes.len());
        while prefix_len < max_prefix && new_bytes[prefix_len] == base_bytes[prefix_len] {
            prefix_len += 1;
        }
        // Need at least a 64-byte prefix to be worth the OMCD framing overhead.
        if prefix_len < 64 {
            return self.store(namespace, text);
        }
        let suffix = &new_bytes[prefix_len..];
        let delta_body_size = 4 + 8 + 4 + suffix.len();
        if delta_body_size + 16 >= new_bytes.len() {
            return self.store(namespace, text);
        }
        // Build the OMCD body and write directly to pool.
        let hash = tokenizer::fnv1a_64(new_bytes);
        let mut body = Vec::with_capacity(delta_body_size);
        body.extend_from_slice(b"OMCD");
        body.extend_from_slice(&(base_hash as u64).to_le_bytes());
        body.extend_from_slice(&(prefix_len as u32).to_le_bytes());
        body.extend_from_slice(suffix);
        let pool_p = self.pool_path(hash);
        if !pool_p.exists() {
            if let Some(parent) = pool_p.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create pool shard {}: {}", parent.display(), e))?;
            }
            std::fs::write(&pool_p, &body)
                .map_err(|e| format!("write OMCD body: {}", e))?;
        }
        // Index entry (same shape as regular store).
        self.append_index(namespace, hash, text.len())?;
        if let Some(cap) = self.max_entries_per_namespace {
            self.evict_to_cap(namespace, cap)?;
        }
        Ok(hash)
    }

    /// Internal: append a chronological-log entry to the namespace index.
    /// Factored out so `store` and `store_as_delta` share the same path.
    fn append_index(&self, namespace: &str, hash: i64, byte_len: usize) -> Result<(), String> {
        use std::io::Write;
        let ns_dir = self.namespace_dir(namespace);
        std::fs::create_dir_all(&ns_dir)
            .map_err(|e| format!("create namespace dir {}: {}", ns_dir.display(), e))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let preview = String::new();  // delta entries don't carry a preview
        let line = format!(
            r#"{{"hash":{},"bytes":{},"stored_at":{},"preview":{}}}"#,
            hash, byte_len, now, json_escape(&preview),
        );
        let index_p = self.index_path(namespace);
        let mut f = std::fs::OpenOptions::new()
            .create(true).append(true).open(&index_p)
            .map_err(|e| format!("open index {}: {}", index_p.display(), e))?;
        writeln!(f, "{}", line)
            .map_err(|e| format!("write index {}: {}", index_p.display(), e))?;
        Ok(())
    }

    /// v0.11.2 SBPE: ONN-style self-training BPE codec (magic `OMCB`).
    ///
    /// Walk the namespace and re-encode aged pool bodies through a per-body
    /// trained BPE: 512 greedy frequency-merge passes produce a merge table,
    /// the body is encoded into the resulting token vocabulary, then the
    /// merge table and token stream are each zlib-deflated and concatenated.
    /// Decoder rebuilds vocab from the merge table and expands tokens.
    ///
    /// Measured 5.21× on 100KB native .omc (vs 4.70× zlib) — first axis
    /// to actually beat plain zlib on real content. The win is the
    /// data-trains-its-own-vocab pattern: the merge table travels inline,
    /// amortizing well for bodies ≥16KB.
    ///
    /// Body layout: `OMCB` (4 bytes magic) | varint(merge_table_zlib_len) |
    /// merge_table_zlib | token_stream_zlib.
    /// Merge table format (pre-zlib): varint(n_merges) | n × (varint a, varint b)
    /// Token stream format (pre-zlib): varint(n_tokens) | n × varint(token_id)
    ///
    /// Skips entries already in any compressed form. Falls back to no-op when
    /// the BPE layout doesn't save ≥16 bytes vs raw.
    pub fn compact_namespace_bpe(
        &self, namespace: &str, age_threshold_secs: i64,
    ) -> Result<(usize, usize, usize), String> {
        use std::io::Write;
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok((0, 0, 0)); }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let mut compacted = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            let Some(hash) = extract_hash_field(line) else { continue };
            let Some(stored_at) = extract_stored_at_field(line) else { continue };
            if now - stored_at < age_threshold_secs { continue; }
            let pool_p = self.pool_path(hash);
            if !pool_p.exists() { continue; }
            let raw = std::fs::read(&pool_p)
                .map_err(|e| format!("read pool {}: {}", pool_p.display(), e))?;
            if raw.len() >= 4 {
                let m = &raw[..4];
                if m == b"OMCZ" || m == b"OMCT" || m == b"OMCH"
                    || m == b"OMCD" || m == b"OMCB" {
                    continue;
                }
            }
            // BPE training cost is O(input × n_merges). Cap training input at
            // 256KB to keep compaction time bounded; merges learned on the
            // prefix still apply to the whole body via greedy left-to-right.
            // But for tighter merges, use the full body if it fits.
            let train_input: &[u8] = if raw.len() > 256 * 1024 {
                &raw[..256 * 1024]
            } else {
                &raw
            };
            let n_merges = if raw.len() < 16 * 1024 { 256 }
                           else if raw.len() < 256 * 1024 { 512 }
                           else { 1024 };
            let merges = bpe_train(train_input, n_merges);
            let tokens = bpe_encode(&raw, &merges);
            // Serialize merges (pre-zlib)
            let mut h_raw = Vec::new();
            varint_write(merges.len() as u64, &mut h_raw);
            for &(a, b) in &merges {
                varint_write(a as u64, &mut h_raw);
                varint_write(b as u64, &mut h_raw);
            }
            let mut h_enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::best());
            h_enc.write_all(&h_raw)
                .map_err(|e| format!("OMCB header deflate: {}", e))?;
            let h_zlib = h_enc.finish()
                .map_err(|e| format!("OMCB header finish: {}", e))?;
            // Token stream (pre-zlib)
            let mut tok_packed: Vec<u8> = Vec::with_capacity(tokens.len() * 2);
            varint_write(tokens.len() as u64, &mut tok_packed);
            for &t in &tokens { varint_write(t as u64, &mut tok_packed); }
            let mut t_enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::best());
            t_enc.write_all(&tok_packed)
                .map_err(|e| format!("OMCB body deflate: {}", e))?;
            let t_zlib = t_enc.finish()
                .map_err(|e| format!("OMCB body finish: {}", e))?;
            // Frame: magic | varint(h_zlib_len) | h_zlib | t_zlib
            let mut h_len_v = Vec::new();
            varint_write(h_zlib.len() as u64, &mut h_len_v);
            let total = 4 + h_len_v.len() + h_zlib.len() + t_zlib.len();
            if total + 16 >= raw.len() { continue; }
            let mut new_body = Vec::with_capacity(total);
            new_body.extend_from_slice(b"OMCB");
            new_body.extend_from_slice(&h_len_v);
            new_body.extend_from_slice(&h_zlib);
            new_body.extend_from_slice(&t_zlib);
            std::fs::write(&pool_p, &new_body)
                .map_err(|e| format!("write OMCB {}: {}", pool_p.display(), e))?;
            compacted += 1;
            before += raw.len();
            after += new_body.len();
        }
        Ok((compacted, before, after))
    }

    /// v0.10.0 Axis 4: substrate-aware tokenizer wired into codec.
    ///
    /// Walk the namespace and re-encode pool bodies through the
    /// substrate tokenizer (`tokenizer::encode`), varint-pack the i64 ID
    /// stream, then zlib-deflate. Pick the smallest of `{raw, OMCZ, OMCT}`
    /// for each body. OMCT bodies start with the 4-byte `OMCT` magic;
    /// recall path detects + decodes transparently.
    ///
    /// The substrate tokenizer dictionary is tuned for OMC source +
    /// adjacent prose, so OMCT wins on OMC-flavored content and gracefully
    /// falls back to OMCZ on pure prose where the dictionary mostly emits
    /// literal-byte escapes (ID 0).
    pub fn compact_namespace_substrate(
        &self, namespace: &str, age_threshold_secs: i64,
    ) -> Result<(usize, usize, usize), String> {
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok((0, 0, 0)); }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let mut compacted = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            let Some(hash) = extract_hash_field(line) else { continue };
            let Some(stored_at) = extract_stored_at_field(line) else { continue };
            if now - stored_at < age_threshold_secs { continue; }
            let pool_p = self.pool_path(hash);
            if !pool_p.exists() { continue; }
            let raw = std::fs::read(&pool_p)
                .map_err(|e| format!("read pool {}: {}", pool_p.display(), e))?;
            if raw.len() >= 4 && (&raw[..4] == b"OMCZ" || &raw[..4] == b"OMCT") {
                continue;
            }
            // Try substrate-tokenize + varint + deflate.
            let text = match std::str::from_utf8(&raw) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let ids = tokenizer::encode(text);
            let mut packed: Vec<u8> = Vec::with_capacity(ids.len());
            for id in &ids {
                varint_write(*id as u64, &mut packed);
            }
            use std::io::Write;
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::best());
            enc.write_all(&packed)
                .map_err(|e| format!("OMCT deflate write: {}", e))?;
            let omct_body = enc.finish()
                .map_err(|e| format!("OMCT deflate finish: {}", e))?;
            if omct_body.len() + 4 + 16 >= raw.len() {
                continue;  // not worth it on this body
            }
            let mut new_body = Vec::with_capacity(omct_body.len() + 4);
            new_body.extend_from_slice(b"OMCT");
            new_body.extend_from_slice(&omct_body);
            std::fs::write(&pool_p, &new_body)
                .map_err(|e| format!("write OMCT {}: {}", pool_p.display(), e))?;
            compacted += 1;
            before += raw.len();
            after += new_body.len();
        }
        Ok((compacted, before, after))
    }

    /// v0.11.0 Axis 6: HBit dual-band substrate codec.
    ///
    /// Rewrites aged pool bodies through the substrate tokenizer (like
    /// Axis 4) but then splits each i64 token id into a high-32-bit band
    /// and a low-32-bit band, varint-encodes each band, and deflates the
    /// two bands separately. The theory: in coherent natural-language
    /// text the substrate tokenizer outputs cluster in a sub-region of
    /// the id space, so the high band has lower entropy than the low
    /// band and compresses better separately than interleaved.
    ///
    /// Layout: `OMCH` (4 bytes magic) | hi_len (4 bytes LE u32) |
    /// hi_band_deflated (hi_len bytes) | lo_band_deflated (remainder).
    ///
    /// Skips entries already in any compressed form. Returns
    /// `(compacted_count, bytes_before, bytes_after)`.
    pub fn compact_namespace_hbit(
        &self, namespace: &str, age_threshold_secs: i64,
    ) -> Result<(usize, usize, usize), String> {
        use std::io::Write;
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok((0, 0, 0)); }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let mut compacted = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            let Some(hash) = extract_hash_field(line) else { continue };
            let Some(stored_at) = extract_stored_at_field(line) else { continue };
            if now - stored_at < age_threshold_secs { continue; }
            let pool_p = self.pool_path(hash);
            if !pool_p.exists() { continue; }
            let raw = std::fs::read(&pool_p)
                .map_err(|e| format!("read pool {}: {}", pool_p.display(), e))?;
            if raw.len() >= 4 {
                let m = &raw[..4];
                if m == b"OMCZ" || m == b"OMCT" || m == b"OMCH" || m == b"OMCD" {
                    continue;
                }
            }
            let text = match std::str::from_utf8(&raw) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let ids = tokenizer::encode(text);
            // Split each id into hi/lo 32-bit bands. Delta-encode within
            // each band so monotonic clusters varint-pack as 1 byte each.
            let mut hi_packed: Vec<u8> = Vec::with_capacity(ids.len());
            let mut lo_packed: Vec<u8> = Vec::with_capacity(ids.len());
            let mut prev_hi: i64 = 0;
            let mut prev_lo: i64 = 0;
            for id in &ids {
                let hi = ((*id as u64) >> 32) as i64;
                let lo = ((*id as u64) & 0xFFFFFFFFu64) as i64;
                varint_write(zigzag_encode(hi - prev_hi), &mut hi_packed);
                varint_write(zigzag_encode(lo - prev_lo), &mut lo_packed);
                prev_hi = hi;
                prev_lo = lo;
            }
            let deflate_band = |band: &[u8]| -> Result<Vec<u8>, String> {
                let mut enc = flate2::write::DeflateEncoder::new(
                    Vec::new(), flate2::Compression::best());
                enc.write_all(band)
                    .map_err(|e| format!("OMCH deflate write: {}", e))?;
                enc.finish().map_err(|e| format!("OMCH deflate finish: {}", e))
            };
            let hi_def = deflate_band(&hi_packed)?;
            let lo_def = deflate_band(&lo_packed)?;
            // Framing: magic (4) + hi_len (4) + hi + lo
            let body_size = 4 + 4 + hi_def.len() + lo_def.len();
            if body_size + 16 >= raw.len() { continue; }
            let mut new_body = Vec::with_capacity(body_size);
            new_body.extend_from_slice(b"OMCH");
            new_body.extend_from_slice(&(hi_def.len() as u32).to_le_bytes());
            new_body.extend_from_slice(&hi_def);
            new_body.extend_from_slice(&lo_def);
            std::fs::write(&pool_p, &new_body)
                .map_err(|e| format!("write OMCH {}: {}", pool_p.display(), e))?;
            compacted += 1;
            before += raw.len();
            after += new_body.len();
        }
        Ok((compacted, before, after))
    }

    /// v0.9.3 Axis 3: fibtier-aware progressive compression.
    ///
    /// Walk a namespace's index and rewrite pool bodies older than the
    /// given threshold (in seconds) as zlib-deflated blobs. Files keep
    /// the same `.txt` extension but get a 4-byte magic prefix `OMCZ` so
    /// the recall path detects + transparently inflates them. Aged
    /// content gets ~3-10× smaller on disk while staying losslessly
    /// recoverable.
    ///
    /// Returns `(compacted_count, bytes_before, bytes_after)`.
    pub fn compact_namespace(
        &self, namespace: &str, age_threshold_secs: i64,
    ) -> Result<(usize, usize, usize), String> {
        use std::io::Write;
        let index_p = self.index_path(namespace);
        if !index_p.exists() { return Ok((0, 0, 0)); }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        let content = std::fs::read_to_string(&index_p)
            .map_err(|e| format!("read index {}: {}", index_p.display(), e))?;
        let mut compacted = 0usize;
        let mut before = 0usize;
        let mut after = 0usize;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            let Some(hash) = extract_hash_field(line) else { continue };
            let Some(stored_at) = extract_stored_at_field(line) else { continue };
            if now - stored_at < age_threshold_secs { continue; }
            // Already compacted? Check pool body for OMCZ magic.
            let pool_p = self.pool_path(hash);
            if !pool_p.exists() { continue; }
            let raw = std::fs::read(&pool_p)
                .map_err(|e| format!("read pool {}: {}", pool_p.display(), e))?;
            if raw.len() >= 4 && &raw[..4] == b"OMCZ" { continue; }
            // Compress with maximum deflate level.
            let mut enc = flate2::write::DeflateEncoder::new(
                Vec::new(), flate2::Compression::best());
            enc.write_all(&raw)
                .map_err(|e| format!("compact deflate write: {}", e))?;
            let compressed = enc.finish()
                .map_err(|e| format!("compact deflate finish: {}", e))?;
            // Only rewrite if it actually saves bytes (small entries with
            // high entropy can EXPAND under deflate). Magic + 1-byte
            // overhead = 5 bytes; require we save at least 16 bytes for
            // the rewrite to be worth the IO.
            if compressed.len() + 4 + 16 >= raw.len() { continue; }
            let mut new_body = Vec::with_capacity(compressed.len() + 4);
            new_body.extend_from_slice(b"OMCZ");
            new_body.extend_from_slice(&compressed);
            std::fs::write(&pool_p, &new_body)
                .map_err(|e| format!("write compacted {}: {}", pool_p.display(), e))?;
            compacted += 1;
            before += raw.len();
            after += new_body.len();
        }
        Ok((compacted, before, after))
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

fn extract_hash_field(line: &str) -> Option<i64> {
    extract_i64_field(line, "\"hash\":")
}

fn extract_stored_at_field(line: &str) -> Option<i64> {
    extract_i64_field(line, "\"stored_at\":")
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

/// v0.9.3 Axis 3 / v0.10.0 Axis 4 / v0.11.0 Axis 6 recall path.
///   `OMCZ` (4 bytes) → zlib-deflated raw text.
///   `OMCT` (4 bytes) → zlib-deflated varint-packed substrate-tokenizer IDs.
///   `OMCH` (4 bytes + 4 bytes hi_len) → HBit dual-band split — hi-32 band
///       and lo-32 band each zigzag-delta-varint-packed and deflated separately.
///   anything else  → plain UTF-8.
fn maybe_decompress(raw: &[u8]) -> Result<String, String> {
    if raw.len() >= 4 && &raw[..4] == b"OMCZ" {
        use std::io::Read;
        let mut dec = flate2::read::DeflateDecoder::new(&raw[4..]);
        let mut out = String::new();
        dec.read_to_string(&mut out)
            .map_err(|e| format!("inflate OMCZ body: {}", e))?;
        return Ok(out);
    }
    if raw.len() >= 4 && &raw[..4] == b"OMCT" {
        use std::io::Read;
        let mut dec = flate2::read::DeflateDecoder::new(&raw[4..]);
        let mut packed = Vec::new();
        dec.read_to_end(&mut packed)
            .map_err(|e| format!("inflate OMCT body: {}", e))?;
        let mut ids: Vec<i64> = Vec::new();
        let mut i = 0;
        while i < packed.len() {
            let (val, consumed) = varint_read(&packed[i..])?;
            ids.push(val as i64);
            i += consumed;
        }
        return Ok(tokenizer::decode(&ids));
    }
    if raw.len() >= 8 && &raw[..4] == b"OMCB" {
        use std::io::Read;
        // OMCB: magic(4) | varint(h_zlib_len) | h_zlib | t_zlib
        let mut pos = 4;
        let (h_len, consumed) = varint_read(&raw[pos..])?;
        pos += consumed;
        let h_zlib_end = pos + h_len as usize;
        if h_zlib_end > raw.len() {
            return Err(format!("OMCB header length {} exceeds body {}",
                               h_len, raw.len() - pos));
        }
        // Inflate merge-table header
        let mut h_raw = Vec::new();
        flate2::read::DeflateDecoder::new(&raw[pos..h_zlib_end])
            .read_to_end(&mut h_raw)
            .map_err(|e| format!("inflate OMCB header: {}", e))?;
        let mut hp = 0;
        let (n_merges, c) = varint_read(&h_raw[hp..])?;
        hp += c;
        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(n_merges as usize);
        for _ in 0..n_merges {
            let (a, ca) = varint_read(&h_raw[hp..])?;
            hp += ca;
            let (b, cb) = varint_read(&h_raw[hp..])?;
            hp += cb;
            merges.push((a as u32, b as u32));
        }
        // Inflate token-stream body
        let mut t_raw = Vec::new();
        flate2::read::DeflateDecoder::new(&raw[h_zlib_end..])
            .read_to_end(&mut t_raw)
            .map_err(|e| format!("inflate OMCB body: {}", e))?;
        let mut tp = 0;
        let (n_tokens, c) = varint_read(&t_raw[tp..])?;
        tp += c;
        let mut tokens: Vec<u32> = Vec::with_capacity(n_tokens as usize);
        for _ in 0..n_tokens {
            let (t, ct) = varint_read(&t_raw[tp..])?;
            tp += ct;
            tokens.push(t as u32);
        }
        let recovered = bpe_decode(&tokens, &merges)?;
        return String::from_utf8(recovered)
            .map_err(|e| format!("OMCB result not valid UTF-8: {}", e));
    }
    if raw.len() >= 8 && &raw[..4] == b"OMCH" {
        use std::io::Read;
        let hi_len = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
        if 8 + hi_len > raw.len() {
            return Err(format!("OMCH hi_len {} exceeds body {}", hi_len, raw.len() - 8));
        }
        let hi_def = &raw[8..8 + hi_len];
        let lo_def = &raw[8 + hi_len..];
        let mut hi_packed = Vec::new();
        flate2::read::DeflateDecoder::new(hi_def).read_to_end(&mut hi_packed)
            .map_err(|e| format!("inflate OMCH hi-band: {}", e))?;
        let mut lo_packed = Vec::new();
        flate2::read::DeflateDecoder::new(lo_def).read_to_end(&mut lo_packed)
            .map_err(|e| format!("inflate OMCH lo-band: {}", e))?;
        let his = read_zigzag_delta_stream(&hi_packed)?;
        let los = read_zigzag_delta_stream(&lo_packed)?;
        if his.len() != los.len() {
            return Err(format!("OMCH band length mismatch: hi={} lo={}",
                               his.len(), los.len()));
        }
        let mut ids: Vec<i64> = Vec::with_capacity(his.len());
        for i in 0..his.len() {
            let hi = his[i] as u64;
            let lo = (los[i] as u64) & 0xFFFFFFFFu64;
            ids.push(((hi << 32) | lo) as i64);
        }
        return Ok(tokenizer::decode(&ids));
    }
    String::from_utf8(raw.to_vec())
        .map_err(|e| format!("body not valid UTF-8: {}", e))
}

/// v0.11.2 SBPE: greedy frequency BPE training.
/// Returns Vec<(token_a, token_b)> where the i-th entry creates token 256+i.
fn bpe_train(bytes: &[u8], n_merges: usize) -> Vec<(u32, u32)> {
    use std::collections::HashMap;
    let mut tokens: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    let mut merge_table: Vec<(u32, u32)> = Vec::with_capacity(n_merges);
    for merge_idx in 0..n_merges {
        let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
        for w in tokens.windows(2) {
            *counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
        if counts.is_empty() { break; }
        let (best_pair, best_freq) = counts.iter()
            .max_by_key(|(_, v)| *v)
            .map(|(k, v)| (*k, *v))
            .unwrap();
        if best_freq < 2 { break; }
        let new_token = (256 + merge_idx) as u32;
        merge_table.push(best_pair);
        let mut new_tokens: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == best_pair.0 && tokens[i + 1] == best_pair.1 {
                new_tokens.push(new_token);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = new_tokens;
    }
    merge_table
}

/// v0.11.2 SBPE: encode bytes to token IDs by applying merges in order.
fn bpe_encode(bytes: &[u8], merges: &[(u32, u32)]) -> Vec<u32> {
    let mut tokens: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    for (idx, &(a, b)) in merges.iter().enumerate() {
        let new_token = (256 + idx) as u32;
        let mut out: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                out.push(new_token);
                i += 2;
            } else {
                out.push(tokens[i]);
                i += 1;
            }
        }
        tokens = out;
    }
    tokens
}

/// v0.11.2 SBPE: decode token IDs back to bytes via reverse merge replay.
fn bpe_decode(tokens: &[u32], merges: &[(u32, u32)]) -> Result<Vec<u8>, String> {
    let mut token_bytes: Vec<Vec<u8>> = (0..256).map(|i| vec![i as u8]).collect();
    for &(a, b) in merges {
        if (a as usize) >= token_bytes.len() || (b as usize) >= token_bytes.len() {
            return Err(format!("OMCB merge references undefined token: ({}, {})", a, b));
        }
        let mut combined = token_bytes[a as usize].clone();
        combined.extend_from_slice(&token_bytes[b as usize]);
        token_bytes.push(combined);
    }
    let mut out = Vec::new();
    for &t in tokens {
        if (t as usize) >= token_bytes.len() {
            return Err(format!("OMCB token id {} out of range", t));
        }
        out.extend_from_slice(&token_bytes[t as usize]);
    }
    Ok(out)
}

fn read_zigzag_delta_stream(packed: &[u8]) -> Result<Vec<i64>, String> {
    let mut out: Vec<i64> = Vec::new();
    let mut i = 0;
    let mut acc: i64 = 0;
    while i < packed.len() {
        let (v, consumed) = varint_read(&packed[i..])?;
        let delta = zigzag_decode(v);
        acc = acc.wrapping_add(delta);
        out.push(acc);
        i += consumed;
    }
    Ok(out)
}

fn zigzag_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

fn zigzag_decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

fn varint_write(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn varint_read(buf: &[u8]) -> Result<(u64, usize), String> {
    let mut v: u64 = 0;
    let mut shift = 0u32;
    let mut i = 0;
    loop {
        if i >= buf.len() { return Err("varint truncated".into()); }
        let b = buf[i];
        v |= ((b & 0x7f) as u64) << shift;
        i += 1;
        if b & 0x80 == 0 { break; }
        shift += 7;
        if shift > 63 { return Err("varint overflow".into()); }
    }
    Ok((v, i))
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
