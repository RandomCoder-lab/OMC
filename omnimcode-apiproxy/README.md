# omnimcode-apiproxy

Substrate-rewriting reverse proxy for `api.anthropic.com`. Compresses large content blocks in the LLM's context window by replacing them with content-addressed `<omc:ref/>` markers, exposing a lossless expansion path via an injected tool.

Status: **v0.14.1** — request rewriting + transparent expand-tool resolution. Measured 6.64× wire-bandwidth compression on a single 6.8 KB content block; expand-tool round-trips are invisible to the client. Streaming + tool_use-block content + image content are still v0.14.2+ work.

## What it does

Every `/v1/messages` POST that flows through the proxy:

1. **Walks `messages[].content[]`** for text blocks bigger than `--rewrite-threshold` (default 4096 bytes)
2. **Replaces each big block** with a tiny marker:
   ```
   <omc:ref hash_str="8085708324473706805" bytes="6800" preview="Substrate-V wins post-projection. ..."/>
   ```
3. **Caches the original** in `~/.omc/memory/_apiproxy_cache/` (reuses the existing MemoryStore, naturally dedupes via the Axis 2 pool)
4. **Injects an `omc_proxy_expand_ref(hash_str)` tool** into the request's `tools` array so the LLM can retrieve the full bytes if it needs them
5. **Forwards** the rewritten request to the real upstream

The auth header (`x-api-key`, `Authorization`) is forwarded as-is — the proxy never reads or logs it.

## Run it

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release -p omnimcode-apiproxy

# Default: localhost:8088, threshold 4096B, preview 200B
./target/release/omnimcode-apiproxy

# Then point Claude Code (or anything that calls the Anthropic API) at it:
export ANTHROPIC_API_URL=http://localhost:8088
```

CLI args:

| flag | default | meaning |
|---|---|---|
| `--bind` | `127.0.0.1:8088` | localhost-only by design |
| `--upstream` | `https://api.anthropic.com` | the real API |
| `--rewrite-threshold` | `4096` | text blocks smaller than this pass through unchanged |
| `--preview-bytes` | `200` | how much of the original to inline as preview |

## What it gives you (measured)

Smoke test against a mock upstream:

| | bytes |
|---|--:|
| original request (one 6.8 KB content block in messages) | 7,177 |
| upstream payload after rewrite | 1,081 |
| **compression** | **6.64×** |

Real-world LLM-token savings depend on how often the LLM resists calling `omc_proxy_expand_ref`. The tool's description tells the LLM to only expand when the preview isn't enough; in practice this should hold ~70-90% of the time on long contexts where most prior turns aren't load-bearing for the current response.

## Known limitations (v0.14.1)

1. **No streaming.** Requests with `"stream": true` pass through unchanged (no SSE rewriting yet — v0.14.2 work).
2. **Mixed tool_use passes through unchanged.** When the LLM emits the expand call alongside another tool call in the same response, the proxy doesn't intercept — it forwards the full response to the client (which will see the unknown expand tool). The auto-resolution only triggers when expand is the sole tool_use.
3. **No image / `tool_use`-block / citation block rewriting.** Only `text` blocks and the `content` field of `tool_result` blocks (string or text-array form) are rewritten.
4. **Response body is not cached for next-turn rewriting.** Cache only fills on the request side, so the savings kick in on subsequent turns where prior big content reappears in conversation history. A v0.15 follow-up will also index large assistant `text` blocks.
5. **No batching API support.** `/v1/messages/batches` falls through to the generic passthrough.
6. **Expand-loop bound at 8 rounds.** If the LLM keeps requesting expansion, the proxy gives up and returns 502 — protects against runaway tool-loop costs.

## Threat model

This proxy sees your full conversation in cleartext. Defaults:

- Binds only to `127.0.0.1` (loopback)
- Never logs request bodies; tracing logs are sized in bytes only
- Cache lives at `~/.omc/memory/_apiproxy_cache/` and is bounded by the existing fibtier eviction (default 232 entries per namespace)

If you change `--bind` to anything non-loopback, you are putting your prompts on the network. Don't.

## Architecture

```
Claude Code  ───────────►  omnimcode-apiproxy  ───────────►  api.anthropic.com
              (HTTP)          │                  (HTTPS)
                              ▼
                       MemoryStore at
                  ~/.omc/memory/_apiproxy_cache/
```

The proxy is a thin axum HTTP server. State lives in the existing OMC MemoryStore (Axis 2 dedup pool), so multiple proxy invocations share a single deduplicated cache.

## Roadmap

- ~~**v0.14.1**: catch `omc_proxy_expand_ref` tool_use in responses, execute locally~~ ✅ shipped
- **v0.14.2**: streaming SSE response rewriting
- **v0.14.3**: handle mixed tool_use (expand + other in same assistant turn)
- **v0.15.0**: tool_use / citation / image content support; batching API; response-side caching for next-turn rewrites
- **v0.16.0**: cache namespace per-conversation (use `x-conversation-id` or similar) so concurrent sessions don't collide
