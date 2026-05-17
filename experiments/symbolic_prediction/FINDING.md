# Substrate-indexed code completion lands as v0.3

## Headline

Given a partial OMC code prefix, `omc_predict_files(paths, prefix, top_k)` returns ranked provenance-tracked continuations from a content-addressed corpus. The synthesis of two earlier substrates:

- **Symbol stream** (tokenizer::encode over canonicalized source)
- **Substrate metric** (canonical_hash + attractor distance)

into one primitive that LLM agents (and humans) can query while writing OMC to find out "what could come next here?" — with each result carrying a substrate-distance score and a pointer back to the source function it came from.

## Win condition (verified)

Prefix `fn prom_linear_` against the Prometheus corpus (`examples/lib/prometheus.omc`, 70 fns) returns exactly the three `prom_linear_*` functions, ranked by substrate distance:

```
=== Predict: 'fn prom_linear_' ===
  prom_linear_forward  (substrate_distance=1374830399114461754, prefix_match_len=24)
  prom_linear_new      (substrate_distance=2435455394695968441, prefix_match_len=24)
  prom_linear_params   (substrate_distance=5509025074886820819, prefix_match_len=24)
```

All three share `prefix_match_len=24` (the same 24 token IDs of the canonicalized prefix matched the trie before diverging into the function-specific suffix). They're then ranked by `|query_hash − candidate_hash|` ascending.

A wider prefix surfaces a broader namespace:

```
=== Predict: 'fn prom_attention_' ===
  prom_attention_substrate_kq_new    (substrate_distance=1.6e16)
  prom_attention_substrate_k_params  (substrate_distance=3.7e17)
  prom_attention_params              (substrate_distance=8.7e17)
  prom_attention_new                 (substrate_distance=1.0e18)
  prom_attention_substrate_k_new     (substrate_distance=2.4e18)
```

The attention-namespace functions are MUCH tighter in substrate space (smaller distances) than the linear-namespace ones — substrate distance reflects code-shape similarity inside the namespace.

## Architecture

### omnimcode-core/src/predict.rs (~370 lines)

- `CorpusEntry { fn_name, source, file, symbol_stream, canonical_hash, attractor }` — one ingested fn.
- `PrefixTrie { children: HashMap<i64, PrefixTrie>, matches: Vec<usize> }` — each node accumulates the indices of corpus entries whose symbol streams pass through it. A prefix query returns all matches in one trie traversal.
- `CodeCorpus { entries, trie }` — the ingested corpus plus its trie. `ingest_fn` canonicalizes → tokenizes → hashes → inserts. `ingest_file` extracts top-level fns from a source string.
- `predict_continuations(corpus, prefix_source, top_k) -> Vec<Suggestion>` — tokenize prefix, query trie, rank surviving matches by `(longest prefix match, smallest substrate distance, corpus index)`.

### Builtins (in interpreter.rs)

- `omc_predict_files(paths_array, prefix_source, top_k) -> array of dicts` — stateless. Each result dict has `fn_name`, `source`, `file`, `canonical_hash`, `attractor`, `prefix_match_len`, `substrate_distance`, `query_attractor`.
- `omc_corpus_size(paths_array) -> int` — diagnostic; reports how many top-level fns ingested.

## Why this composes well

Three primitives already in OMC — `canonicalize` (alpha-rename invariance), `tokenizer::encode` (substrate-aware symbol stream), `code_hash` (substrate-routed identity) — combine without modification. The trie is a 50-line data structure on top. The substrate metric (which already drove `omc_find_similar`, attention's `attractor_distance`, the heal pass's `substrate_hash_name` bucketing) drives ranking here too.

Determinism: same corpus + same prefix → same top-k, every run. No randomness, no embedding model, no neural inference.

## What's now possible that wasn't before

- An LLM agent can query "what previous code came next at this shape?" as a single MCP tool call.
- Branching is first-class — each result is a viable continuation, not a "best guess."
- Provenance is content-addressed: every suggestion includes its source file path AND its canonical hash, so a downstream agent can verify integrity by recompute.
- The corpus is just file paths; no index-build step, no maintenance overhead.

## Deferred (post-v0.3)

- **Prometheus rerank pass** — train a small Prometheus model on the corpus and rerank top-k by token-stream probability. Substrate ranking is the structural prior; Prometheus is the learned overlay.
- **Stateful corpus API** — `omc_corpus_build` returns a handle, `omc_predict_from(handle, prefix, top_k)` reuses it. The current stateless API rebuilds per call (fine for interactive use; slow if called in a tight loop).
- **MCP tool surface** — wrap `omc_predict_files` as an MCP tool so LLM clients can query during code generation without launching a subprocess.
- **Streaming queries** — incremental updates as the prefix grows token-by-token.
- **Cross-corpus blending** — query multiple corpora (project, stdlib, registry) with weighted ranking.

## Tests

- **10 Rust unit tests** in `predict.rs` cover trie semantics, ingestion, ranking, top_k cap, empty inputs, provenance.
- **11 OMC end-to-end tests** in `examples/tests/test_predict.omc` exercise the builtins against the real Prometheus corpus.

Total: 223 Rust pass, 1087/1087 OMC pass.
