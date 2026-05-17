# OMC Roadmap

Current chapter: **v0.2-ergonomics** (shipped 2026-05-17).
Next chapter: **v0.3-symbolic-prediction** (in flight).

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/RandomCoder-lab/OMC/releases) for the chapter-by-chapter history of how OMC got here. This file describes what's on the path going forward.

---

## v0.3-symbolic-prediction (in flight)

**Substrate-indexed code completion: given a partial OMC prefix, return ranked provenance-tracked continuations from a content-addressed corpus.**

The synthesis of two earlier threads — substrate codec (symbolic context) and Prometheus (text prediction) — into a single primitive that LLM agents (and humans) can use to navigate "what could come next here?" while writing OMC. Branching is first-class: each result is a viable continuation with a substrate-distance score and a pointer back to the source function it came from.

### Architecture

- `omnimcode-core/src/predict.rs` — `CodeCorpus`, `PrefixTrie`, `predict_continuations`.
- Builtins: `omc_corpus_build(paths)` → handle, `omc_predict(prefix_source, corpus_handle, top_k)` → ranked dict.
- CLI subcommand: `omc --predict --files DIR --prefix "fn ..." --top-k 5 --json`.
- Win condition: prefix `fn prom_linear_` against the Prometheus corpus returns `prom_linear_new`, `prom_linear_forward`, `prom_linear_params` ranked by substrate distance, with provenance pointers to the source files.

### Phases

1. Symbol-stream encoding wrapper over the existing `tokenizer::encode` — already produces `Vec<i64>` symbol IDs; just expose a clean ingestion API.
2. `CodeCorpus` builder: parse each file in a path list, extract top-level fns via `extract_top_level_fns`, build entries `{fn_name, source, symbol_stream, canonical_hash, attractor}`.
3. `PrefixTrie` over symbol streams: insert each stream once, query a prefix to get matching corpus indices in O(prefix length).
4. `predict_continuations(corpus, trie, prefix_source, top_k)` — tokenize prefix, query trie, rank surviving matches by `(longest prefix match, smallest substrate distance)`.
5. Rust tests + OMC tests against the lib/ corpus.
6. CLI demo + writeup as `experiments/symbolic_prediction/FINDING.md`.
7. Tag as `v0.3-symbolic-prediction` with chapter release notes.

### Deferred (post-v0.3)

- **Prometheus rerank pass** — once the trie-based candidate list is solid, train a small Prometheus model on the corpus and rerank top-k by token-stream probability.
- **MCP tool surface** — expose `predict_omc_continuation(prefix, top_k)` as an MCP tool so LLM clients can query during code generation.
- **Streaming queries** — incremental updates as the prefix grows token-by-token.
- **Cross-corpus blending** — query multiple corpora (project, stdlib, registry) with weighted ranking.

---

## Beyond v0.3 (rough)

### Substrate-attention follow-ups

- Substrate-modulated Q projection. Q hasn't been swapped yet; the V resample recipe (post-projection modulation) may generalize.
- Substrate FF: dampen off-attractor activations in the feed-forward residual.
- Substrate LayerNorm: substrate-distance-weighted variance computation.
- Larger-scale validation: every substrate-attention claim was made at TinyShakespeare scale (1.1MB). Need to verify the stack holds at 10-100MB corpora.

### Transformerless LLM

The substrate-attention components stack to −8.94% inside one block. The path forward is a top-to-bottom harmonic-only architecture trained competitively. Open: how to handle non-integer-coherent quantities at this scale (the substrate metric only applies to integer-valued quantities, per the rule derived from the HBit-gate falsification).

### JIT path expansion

- AVX-512 widening — blocked on array-processing OMC fns to fill the wider lanes.
- JIT for float-returning harmonic primitives — `returns_float` dispatch flag mirroring `returns_array_int`.
- JIT for dict ops — currently pure tree-walk for string-keyed data; the L1 array-of-hashed-int rewrite avoided this for hot paths.

### Tooling polish

- Improved formatter (`--fmt`) — preserve comments, configurable line width.
- LSP improvements: completion (uses the v0.3 predict engine), hover with substrate signature.
- VS Code extension: snippet library, inline hint UI for the heal pass.

---

## Done (linked to chapter releases)

| Chapter | Key shipped items |
|---|---|
| [v0.2-ergonomics](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.2-ergonomics) | `+=` / `-=` / `*=` / `/=` / `%=`, `len`/`range`/`getenv`/`to_hex`/`parse_int`, negative array indexing, did-you-mean, traced errors, 11 heal classes |
| [v0.1-substrate-attention](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.1-substrate-attention) | Substrate-K + S-MOD softmax + substrate-V resample → −8.94% val on TinyShakespeare |
| [v0.0.6-prometheus](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.6-prometheus) | Tape autograd, AdamW, Embedding, LayerNorm, multi-block transformer, first substrate-K wins |
| [v0.0.5-codec-kernel-protocol](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.5-codec-kernel-protocol) | Substrate codec, `omc-kernel`, `omc-grep`, OMC-PROTOCOL v1, substrate-aware tokenizer |
| [v0.0.4-jit-and-dual-band](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.4-jit-and-dual-band) | LLVM JIT, dual-band SSE2 codegen, harmony-gated branch elision, array support |
| [v0.0.3-substrate-and-stdlib](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.3-substrate-and-stdlib) | Heal pass, substrate-routed search family, stdlib expansion, `--check` / `--fmt` |
| [v0.0.2-language-core](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.2-language-core) | Parser, two-engine interpreter, HInt, bytecode VM, self-hosting fixpoint |
| V0.0.1 | Genesis: circuit evolution engine, FFI, Unity/Unreal bindings |

`ROADMAP.json` is preserved for archaeology — it captured the state through v0.0.4. This file supersedes it as the canonical forward plan.
