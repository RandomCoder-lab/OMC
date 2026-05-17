# OMC Roadmap

Current chapter: **v0.3.1-symbolic-compression** (shipped 2026-05-17).
Next chapter: **v0.4-substrate-context** (planned — the symbolic-context compression thesis taken seriously).

See [CHANGELOG.md](CHANGELOG.md) and [GitHub Releases](https://github.com/RandomCoder-lab/OMC/releases) for the chapter-by-chapter history of how OMC got here. This file describes what's on the path going forward.

---

## v0.4-substrate-context (planned)

**Take the symbolic-context compression thesis end-to-end.** v0.3.1 added format options to omc_predict (3.8× compression on the predict response path). v0.4 generalizes: every LLM-facing OMC surface becomes substrate-aware about its context cost.

The substrate codec from v0.0.5 already does library-lookup compression (`omc_codec_encode` → 10-50× ratios when the receiver has the library). The v0.4 chapter wires it into the LLM flow as a first-class context-compression mechanism:

### Tracks

- **`omc_export_module(path, format=codec)`** — emit a module as a sampled-token codec payload. The LLM consumes the payload (a few hundred bytes) instead of the full source (several KB). Recovery is via library lookup against the LLM's known corpus, or via `omc_codec_decode_lookup` for explicit reconstruction.
- **Substrate-keyed conversation memory** — wire the `fibtier` memory primitive to store conversation entries as canonical hashes; fetch on demand via the kernel. An LLM's conversation history becomes a stream of hash references that recover into full content when reasoning needs it.
- **MCP tool: `omc_compress_context(text)`** — given a chunk of OMC code or prose, return a substrate-keyed compressed form the LLM can reference. The complement of `omc_fetch_by_hash`.
- **Cross-corpus blending** — query multiple corpora (project, stdlib, registry) with weighted ranking, return substrate-keyed identifiers that work across any of them.
- **Substrate-typed conversation transcripts** — every message in an agent conversation gets a canonical hash; threading + memory operations index by hash, not by string.
- **Benchmark: end-to-end context-budget reduction** — measure how many fns an LLM agent can hold "in mind" with v0.4 vs without. Hypothesis: 5-10× more candidates fit in the same context window.

### Win condition

An LLM agent solves a multi-step OMC authoring task using ~10% of the context budget a baseline agent would consume, with no loss in solution quality — because the predict engine's output, the conversation memory, and the codec payloads all compose through the substrate's content-addressed identity.

### Deferred from v0.3

- **Prometheus rerank pass** — train a small Prometheus model on the corpus and rerank top-k by token-stream probability.
- **Stateful corpus API** — `omc_corpus_build` returns a handle, `omc_predict_from(handle, prefix, top_k)` reuses it.
- **Streaming queries** — incremental updates as the prefix grows token-by-token.

---

## v0.5+ candidates

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
