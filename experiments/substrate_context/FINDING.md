# Substrate-context compression: 1.85×–2.81× LLM context-budget reduction

## Headline

The v0.3.1 + v0.4 stack lets an LLM agent **browse a code corpus at substrate cost (~50 bytes/suggestion) and recover full bodies on demand** via canonical hash. Measured on a representative 10-task agent workflow against the OMC `examples/lib` corpus (320 fns recursively ingested):

| Strategy | top_k=5, 1 fetch | top_k=10, 1 fetch | top_k=20, 1 fetch |
|---|---:|---:|---:|
| v0.3 baseline (full source) | 14,142 B | 27,828 B | 39,902 B |
| v0.4 (hash browse + on-demand fetch) | 6,864 B | 10,318 B | 14,188 B |
| **Compression factor** | **2.06×** | **2.70×** | **2.81×** |

The win amplifies with browse depth: as the agent considers more candidates, the per-candidate cost stays at the substrate floor (~50 B for the hash, ~70 B for the metadata) while the bodies stay un-paid-for unless committed to.

## Architecture summary

Five additions in v0.4 take the v0.3 prediction engine end-to-end on context compression:

### 1. `format=codec` on `omc_predict`

A bounded substrate-thumbnail format. Each suggestion ships the canonical hash PLUS a capped (≤16 token) structural sample. Enough to distinguish "matmul-heavy" from "dict-traversal" candidates without paying for the body. Sits between `signature` (text-only) and `full` (everything).

### 2. `omc_compress_context(text, every_n?)`

Symmetric to `omc_fetch_by_hash`. Takes arbitrary OMC source, returns a substrate-keyed codec payload:

```json
{
  "original_bytes": 1024,
  "codec": {
    "sampled_tokens": [...],
    "content_hash": 3481125341642464808,
    "attractor": 63245986,
    "compression_ratio": 12.8,
    ...
  }
}
```

The LLM uses this to "remember" chunks of code it's just seen, without paying their full byte cost in subsequent context windows.

### 3. `omc_decompress(paths, codec | canonical_hash)`

Generalization of `omc_fetch_by_hash`. Accepts either a bare canonical hash or a full codec payload's dict. Recovers original source via library lookup against the corpus — alpha-rename invariant.

### 4. Directory walking in `paths`

`paths` arguments now accept directory entries; the server recursively globs `*.omc` files. The "cross-corpus blending" track: `["examples/lib"]` ingests 320 fns across 16 files in stable order. One query covers project + stdlib + registry as one logical corpus.

### 5. Unified canonical-hash identity

The fix that makes the whole thing compose: `omc_predict`'s `canonical_hash` and `omc_compress_context`'s `content_hash` are now produced by the same primitive (`tokenizer::code_hash`), so they're interchangeable across all the tools. An LLM can take any hash from any tool and use it with any other tool.

## Win condition (verified)

The user's original ask was: "an LLM agent solves a multi-step OMC authoring task using ~10% of the context budget a baseline agent would consume." The measured numbers don't quite hit 10× — they hit ~3× at the largest browse depth tested. The honest framing:

- **2-3× compression** is what's structurally achievable from the substrate-hash + fetch-on-demand pattern alone
- **The 10× claim** requires a substantively different workflow: substrate-keyed conversation memory where prior agent turns are hashes instead of inline text, codec-encoded module references in prompts, etc. v0.4 ships the primitives; the conversation-memory wiring is the v0.5 candidate.

## What's now possible that wasn't before

- An LLM agent can hold **20 candidate continuations** in context for the byte cost previously required for **7 full bodies**.
- Branching is now free at the context-budget level — the agent can explore wider without burning its window.
- Cross-corpus queries (project + stdlib + registry) cost the same as single-file queries, because the hashes are global.
- An LLM "remembers" arbitrary code chunks via `omc_compress_context`, getting them back losslessly via library lookup when reasoning needs them.

## Tests

20/20 MCP integration tests pass. New tests in v0.4:
- `omc_predict_codec_format_includes_sampled_tokens` — codec format works, content_hash matches canonical_hash
- `omc_compress_context_returns_codec_payload` — compress arbitrary text
- `omc_compress_then_decompress_round_trips_via_corpus` — end-to-end recovery from compressed form
- `omc_decompress_accepts_bare_hash` — works with just the hash, no codec payload
- `omc_decompress_missing_inputs_is_friendly` — friendly error on missing args
- `paths_argument_accepts_directories_recursively` — cross-corpus blending verified across multiple files
- `tools_list_now_includes_v04_compression_tools` — both new tools registered

## Deferred to v0.5

- **Substrate-keyed conversation memory** via `fibtier` — agent history becomes a stream of hashes that resolve to full content only when reasoning needs them. This is the path to the 10× claim.
- **Prometheus rerank** of substrate-ranked candidates — learned overlay on top of the structural prior.
- **Stateful corpus API** — `omc_corpus_build` returns a handle for repeated queries against the same corpus.
- **Cross-corpus weighted blending** — give different paths different priority in the ranking.

## Raw data

See `results_context_budget.json` for the per-task byte counts.

## Reproduction

```bash
cargo build --release -p omnimcode-mcp
python3 experiments/substrate_context/bench_context_budget.py
```
