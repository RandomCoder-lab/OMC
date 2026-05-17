# OMC Roadmap

Current chapter: **v0.5-substrate-memory** (shipped 2026-05-17).
Next chapter: open — candidates listed below. The five-chapter symbolic-context arc (v0.3 → v0.3.1 → v0.4 → v0.5) has landed with the 10× target hit (10.61× measured).

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/RandomCoder-lab/OMC/releases) for the chapter-by-chapter history of how OMC got here. This file describes what's on the path going forward.

---

## Post-v0.5 candidates (none committed yet)

### v0.6 candidate A — fibtier-bounded memory

v0.5 ships substrate-keyed memory but the store grows unbounded. Long-running agents need pruning. Wire fibtier's tier-bounded eviction into `MemoryStore`:

- Each namespace gets a tier-state file alongside the index
- Stores cascade into higher tiers via the fibtier fold mechanism
- Old entries get summarized/aggregated as they fold upward
- Bounded total entries across all tiers (default ~4180 = Fib(18))

### v0.6 candidate B — Prometheus rerank pass

The substrate-ranked predict candidates can be reranked by a learned probability overlay. Train a small Prometheus model on the corpus, score top-k candidates' next-token probabilities, blend with the substrate distance.

### v0.6 candidate C — substrate-attention follow-ups

- Substrate-modulated Q projection. Q hasn't been swapped yet; the V resample recipe (post-projection modulation) may generalize.
- Substrate FF: dampen off-attractor activations in the feed-forward residual.
- Substrate LayerNorm: substrate-distance-weighted variance computation.
- Larger-scale validation: every substrate-attention claim was made at TinyShakespeare scale (1.1MB). Need to verify the stack holds at 10-100MB corpora.

### Other deferred items

- **Stateful corpus API** — `omc_corpus_build` returns a handle, `omc_predict_from(handle, prefix, top_k)` reuses it. Saves the corpus-rebuild cost on repeated queries.
- **Streaming queries** — incremental updates as the prefix grows token-by-token.
- **Cross-corpus weighted blending** — give different paths different priority in the ranking.
- **Conversation-aware predict** — `omc_predict(..., context_hash=H)` where H references prior reasoning state, biasing the ranking by which fns the agent has already touched.

---

## v0.7+ candidates

### Substrate-attention follow-ups

- Substrate-modulated Q projection. Q hasn't been swapped yet; the V resample recipe (post-projection modulation) may generalize.
- Substrate FF: dampen off-attractor activations in the feed-forward residual.
- Substrate LayerNorm: substrate-distance-weighted variance computation.
- Larger-scale validation: every substrate-attention claim was made at TinyShakespeare scale (1.1MB). Need to verify the stack holds at 10-100MB corpora.

### Beyond (rough)

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
| [v0.5-substrate-memory](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.5-substrate-memory) | `omc_memory_store/recall/list/stats` + filesystem persistence + **10.61× LLM context-budget reduction** measured on a 20-turn agent task |
| [v0.4-substrate-context](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.4-substrate-context) | `omc_compress_context` / `omc_decompress` tools + `format=codec` thumbnails + directory ingest + measured 1.85×-2.81× LLM context-budget reduction |
| [v0.3.1-symbolic-compression](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.3.1-symbolic-compression) | `omc_predict` gains `format=hash`/`signature`/`full` (3.8× compression default) + `omc_fetch_by_hash` for on-demand recovery |
| [v0.3-symbolic-prediction](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.3-symbolic-prediction) | `omc_predict_files(paths, prefix, top_k)` returns ranked provenance-tracked continuations from a content-addressed corpus |
| [v0.2-ergonomics](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.2-ergonomics) | `+=` / `-=` / `*=` / `/=` / `%=`, `len`/`range`/`getenv`/`to_hex`/`parse_int`, negative array indexing, did-you-mean, traced errors, 11 heal classes |
| [v0.1-substrate-attention](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.1-substrate-attention) | Substrate-K + S-MOD softmax + substrate-V resample → −8.94% val on TinyShakespeare |
| [v0.0.6-prometheus](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.6-prometheus) | Tape autograd, AdamW, Embedding, LayerNorm, multi-block transformer, first substrate-K wins |
| [v0.0.5-codec-kernel-protocol](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.5-codec-kernel-protocol) | Substrate codec, `omc-kernel`, `omc-grep`, OMC-PROTOCOL v1, substrate-aware tokenizer |
| [v0.0.4-jit-and-dual-band](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.4-jit-and-dual-band) | LLVM JIT, dual-band SSE2 codegen, harmony-gated branch elision, array support |
| [v0.0.3-substrate-and-stdlib](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.3-substrate-and-stdlib) | Heal pass, substrate-routed search family, stdlib expansion, `--check` / `--fmt` |
| [v0.0.2-language-core](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.2-language-core) | Parser, two-engine interpreter, HInt, bytecode VM, self-hosting fixpoint |
| V0.0.1 | Genesis: circuit evolution engine, FFI, Unity/Unreal bindings |

`ROADMAP.json` is preserved for archaeology — it captured the state through v0.0.4. This file supersedes it as the canonical forward plan.
