# omnimcode-apiproxy

Substrate-rewriting reverse proxy for `api.anthropic.com`. Compresses large content blocks in the LLM's context window by replacing them with content-addressed `<omc:ref/>` markers, exposing a lossless expansion path via an injected tool.

Status: **v0.14.0-alpha** — proof of concept. Measured 6.64× compression on a single 6.8 KB content block in smoke test. Known sharp edges below.

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

## Known limitations (v0.14.0-alpha)

1. **No streaming.** Requests with `"stream": true` pass through unchanged (no SSE rewriting yet).
2. **The injected `omc_proxy_expand_ref` tool is not yet served by the proxy.** If the LLM actually emits a `tool_use` block calling it, the response flows back through Claude Code, which doesn't know the tool and will error. This means: in this alpha, the proxy works best in a "fire and forget" mode where the LLM responds without expanding markers. A v0.14.1 follow-up will catch tool_use for `omc_proxy_expand_ref` in the response stream, execute it locally from the cache, and inject the tool_result before returning to the client.
3. **No image / tool_use / citation block rewriting.** Only `text` blocks and the `content` field of `tool_result` blocks (string or text-array form) are rewritten.
4. **Response is forwarded unmodified.** Cache only fills on the request side, so the savings kick in on subsequent turns where prior big content reappears in conversation history.
5. **No batching API support.** `/v1/messages/batches` falls through to the generic passthrough.

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

- **v0.14.1**: catch `omc_proxy_expand_ref` tool_use in responses, execute locally
- **v0.14.2**: streaming SSE response rewriting
- **v0.15.0**: tool_use / citation / image content support; batching API
- **v0.16.0**: cache namespace per-conversation (use `x-conversation-id` or similar) so concurrent sessions don't collide
