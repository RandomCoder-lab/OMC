# v0.5 substrate-memory: 10.61× LLM context-budget reduction on a 20-turn conversation

## Headline

**v0.5 hits the 10× target the v0.4 chapter fell short of.** Combining v0.3.1's hash-format predict with v0.5's substrate-keyed conversation memory, a 20-turn LLM agent task uses **9.4% of the prompt-token budget** a baseline (full transcript inline) agent would consume.

| Strategy | Cumulative bytes across 20 turns | vs baseline |
|---|---:|---:|
| Baseline (full transcript inline) | 869,761 | 100% |
| v0.4 only (compressed predict, full transcript) | 423,030 | 48.6% (2.06× smaller) |
| **v0.5 full (memory hashes + compressed predict)** | **82,008** | **9.4% (10.61× smaller)** |

The growth pattern makes the story:

- **Baseline grows quadratically** — each turn re-sends the entire conversation history inline. By turn 20 the prompt is ~70 KB; the cumulative bytes processed across the conversation is ~870 KB.
- **v0.4 also grows quadratically** but with a smaller constant — same transcript-carrying pattern, just with compressed predict responses.
- **v0.5 grows linearly** — each turn's prompt is constant (this turn's content + cheap hash refs to prior turns + 1 recalled body when needed). By turn 20 the prompt is ~4 KB. Cumulative across 20 turns is ~82 KB.

The crossover happens around turn 5 — that's the moment v0.5 starts paying off.

## Architecture

### New module: `omnimcode-core/src/memory.rs` (~370 lines, 10 unit tests)

- `MemoryStore { root }` — filesystem-backed substrate-keyed store at `~/.omc/memory/<namespace>/<hex_hash>.txt`
- `store(namespace, text)` — content-address by `tokenizer::fnv1a_64`, write body + append to `_index.jsonl`
- `recall(namespace?, hash)` — read body by hash; with no namespace hint, walks all
- `list(namespace, limit)` — recent entries first, each carries `{hash, bytes, stored_at, preview}` (no body — that's the compression)
- `stats(namespace)` — count + total bytes for diagnostics
- Namespace sanitization (alphanumeric + `_-` only) prevents path traversal
- `OMC_MEMORY_ROOT` env var for test isolation

### Four new MCP tools

- `omc_memory_store(text, namespace?)` → `{content_hash, namespace, bytes}`
- `omc_memory_recall(content_hash, namespace?)` → `{found, text, bytes}` or `{found: false}`
- `omc_memory_list(namespace?, limit?)` → `{namespace, count, entries: [{content_hash, bytes, stored_at_unix, preview}]}`
- `omc_memory_stats(namespace?)` → `{namespace, total_entries, total_bytes}`

### Tests

27/27 MCP integration tests pass (was 20 + 7 new memory). Plus 10 unit tests in the memory module.

## How the workflow looks

A 20-turn LLM agent task with v0.5:

```
TURN 1:
  agent reasoning      → ~400 B
  omc_predict (hash)   → ~700 B   (no full bodies)
  omc_fetch_by_hash    → ~300 B   (1 fetch)
  omc_memory_store     → just sends back the hash to remember this turn
  → PROMPT SIZE this turn: ~1.4 KB

TURN 20:
  agent reasoning      → ~400 B
  omc_predict (hash)   → ~700 B
  omc_fetch_by_hash    → ~300 B
  prior_turn_refs      → 19 × 20 B = ~400 B   (the cheap pointers)
  recalled (turn 19)   → ~3 KB                  (1 prior turn recovered)
  → PROMPT SIZE this turn: ~4.8 KB
```

Baseline at turn 20 would be ~70 KB just to carry the transcript.

## Why it composes

The substrate's identity primitive (`tokenizer::fnv1a_64` for arbitrary bytes, `tokenizer::code_hash` for canonical OMC source) is shared across all the chapters:

- v0.3 `omc_predict` returns `canonical_hash` for each suggestion
- v0.3.1 `omc_fetch_by_hash` recovers via canonical_hash
- v0.4 `omc_compress_context` produces `content_hash` (matches predict's canonical_hash for OMC source)
- v0.4 `omc_decompress` accepts either
- v0.5 `omc_memory_store` produces `content_hash` (matches the codec's content_hash for the same bytes)
- v0.5 `omc_memory_recall` accepts any hash

An LLM agent can mix tools freely — no tool needs to know which other tool produced a hash. That's what makes the 10× win compose across the chapters instead of being an isolated effect.

## Honest framing

- The 10× comes from the COMBINED v0.4 + v0.5 stack. v0.4 alone tops out near 2-3×; v0.5 alone (memory but full predict bodies) would top out near 3-4×; together they multiply because they target different cost components.
- The win scales with conversation length. At 5 turns the baseline hasn't grown enough for v0.5 to matter — it's at parity. The 10× kicks in around turn 15+.
- The benchmark uses synthetic reasoning blurbs (~400 B each). Real LLM agent traces are longer (typically 1-5 KB per turn), which would make baseline grow even faster and amplify v0.5's advantage further.
- Filesystem-backed memory survives MCP process restart — agents can be paused and resumed without losing their substrate-keyed conversation state.
- We did NOT wire fibtier's tier-bounded eviction in v0.5 (deferred). The memory store grows unbounded; a long-running agent should add its own pruning policy or wait for v0.5.1.

## Reproduction

```bash
cargo build --release -p omnimcode-mcp
python3 experiments/substrate_context/bench_multi_turn_memory.py
```

Configurable via `bench_multi_turn_memory.py`: `n_turns`, `top_k`, `recalls_per_turn`, `paths`. Default config produces the table above.

## Raw data

`results_multi_turn_memory.json` has per-turn byte counts for all three strategies.
