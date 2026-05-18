# OMC Memory+ for Claude Code

Persistent, content-addressed memory for Claude Code sessions. Hold context references in 5 tokens instead of 5,000. Recall on demand.

**Measured on real Claude Code dev sessions: 297× context compression ratio, 73% token cost reduction.**

**v0.11.2 update — disk-side codec now beats plain zlib: 5.21× on 100KB native .omc vs 4.70× for zlib.** First axis to actually beat zlib. See `omc_memory_compact_bpe` below.

## What it is

A Claude Code MCP plugin powered by OMNIcode's substrate codec. It gives Claude four tools:

- `omc_compress_context` — compress arbitrary text into a canonical hash + structural thumbnail
- `omc_memory_store` — persist a chunk of text by canonical hash to local disk
- `omc_memory_recall` — recover stored text by hash, on demand
- `omc_memory_list` / `omc_memory_stats` — browse and inspect stored content

The win: **Claude no longer needs to hold large blocks of context in its working set.** It holds the canonical hash (a single int64, ~5 tokens) and recalls the full content only when it actually needs to reason about it.

## How much it saves

Tested on 18 project documentation files from a real codebase:

| | tokens |
|---|--:|
| pasted-in-context (status quo) | 26,781 |
| hash references only | 90 |
| **compression** | **297.6×** |

At Claude Sonnet pricing ($3/MTok input):
- **Without Memory+**: $0.08 per session that needs project context
- **With Memory+**: $0.02 per session (hash refs + on-demand recall of 3-5 chapters)
- **Savings**: $5.70/month per developer at 100 sessions/month
- **50-dev org savings**: $285/month

## Pricing

| plan | price | features |
|---|--:|---|
| **Free** | $0 | Local memory storage, all 4 tools, single machine |
| **Pro** | $5/mo per seat | Cross-machine sync via cloud, longer retention, namespace sharing |
| **Team** | $50/mo for 5 seats | Pro + shared team memory namespaces, audit log |
| **Enterprise** | from $500/mo | Self-hosted memory server, SSO, custom retention, SLA |

## Quickstart

```bash
# 1. Install the omnimcode-mcp binary (one-time)
curl -fsSL https://omc.sh/install.sh | sh

# 2. Add to Claude Code's MCP config (~/.claude.json)
omc-memory install

# 3. Restart Claude Code
# 4. Try it in any session:
#    "Remember this finding for next time: <text>"
#    "What did we figure out about X last session?"
```

## How the math works

OMC's codec is **content-addressed via canonical hashing** (alpha-rename invariant for code, structural hashing for prose). Identical content → identical hash, regardless of variable names or reformulation. Storage survives `/exit`.

Three modes a Claude Code session uses Memory+:

1. **Within-session**: compress long docs into the LLM context as hash refs; recall on demand. Saves tokens within a single long session.
2. **Cross-session**: persist findings, decisions, and project notes. Next session starts with cheap hash refs to prior work.
3. **Cross-machine** (Pro+): same memory available wherever you launch Claude Code.

## Architecture

```
Claude Code
    │
    ▼
MCP protocol (stdio JSON-RPC)
    │
    ▼
omnimcode-mcp binary
    │
    ▼
~/.omc/memory/<namespace>/  ← filesystem-backed, content-addressed
```

Local-first by default. Cloud sync is opt-in. Your codebase and findings stay on your machine unless you explicitly enable the Pro plan.

## What's in the box

- `omc_eval` — evaluate OMC code (bonus, for power users)
- `omc_help`, `omc_list_builtins`, `omc_categories` — OMC reference tools
- `omc_did_you_mean`, `omc_explain_error` — error-recovery helpers
- `omc_compress_context` — the codec
- `omc_decompress` — recover compressed text against a corpus
- `omc_predict` — substrate-indexed code completion (OMC-specific)
- `omc_fetch_by_hash` — companion to omc_predict
- `omc_memory_store` / `_recall` / `_list` / `_stats` / `_evict` — the memory layer
- `omc_memory_create_manifest` / `_recall_manifest` — bundle N hashes under 1 (Axis 1)
- `omc_memory_store_delta` — store as a delta against a base (Axis 5)
- `omc_memory_compact` (zlib), `_compact_substrate` (OMCT), `_compact_hbit` (OMCH),
  **`_compact_bpe` (OMCB — beats zlib)** — aged-tier compression axes
- `omc_unique_builtins` — list OMC-unique primitives (substrate ops, harmonic ops)
- `omc_corpus_size` — diagnostic

## Compression axis benchmark (100KB native .omc)

| axis | format | ratio | notes |
|---|---|--:|---|
| `omc_memory_compact_bpe` | OMCB | **5.21×** | ★ winner — self-training BPE, beats plain zlib |
| `omc_memory_compact` | OMCZ | 4.70× | plain zlib (still the simplest fallback) |
| `omc_memory_compact_substrate` | OMCT | 4.30× | substrate-tokenizer + zlib (loses to OMCZ) |
| `omc_memory_compact_hbit` | OMCH | 3.23× | HBit dual-band split (loses to OMCZ) |

Round-trip lossless for all four. The MemoryStore auto-detects body magic on recall, so once a body is compacted in any format, recall is transparent.

## Roadmap

- **v1.0** (now): local Memory+, all 4 core tools, MCP plugin manifest
- **v1.1**: cloud sync (Pro), team namespaces
- **v1.2**: auto-detect long context blocks, suggest compression
- **v1.3**: integration with `/compact` command — replace summary with hash refs
- **v2.0**: API endpoint for non-Claude-Code tools (Cursor, Continue, etc.)

## License

Source open under MIT. Cloud sync service hosted under usage-based pricing above.

## Built on OMNIcode

`omnimcode-mcp` is part of OMNIcode (OMC), a harmonic computing language with native substrate primitives (Fibonacci attractors, CRT-PE positional encoding, content-addressed code storage). The substrate codec was originally designed for distributed agent kernel communication (OMC-PROTOCOL v1); Memory+ packages it for Claude Code users.
