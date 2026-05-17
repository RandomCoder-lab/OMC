# OMC Substrate Protocol (OMC-P) v1

> An inter-agent wire protocol for content-addressed code and data,
> built on substrate-canonical hashes and signature verification
> without shared keys.

## Status

Living specification. Reference implementation lives in this
repository:
- Sender / receiver: `omc_msg_sign` / `omc_msg_verify` / the
  `omc_codec_*` family (OMC builtins, see `examples/lib/test.omc`
  patterns)
- Storage layer: `omc-kernel` (`omnimcode-cli/src/bin/omc_kernel.rs`)
- MCP adapter: `tools/mcp_substrate/server.py`
- End-to-end demos: `examples/demos/llm_tandem_*.omc`

## Design goals

| Goal | Mechanism |
|---|---|
| **Identity without keys.** Verify content integrity without PKI. | Substrate signature: `content_hash = fnv1a_64(canonicalize(content))`; receiver recomputes and compares. Tamper-evident by construction. |
| **Alpha-rename invariance.** Code that means the same thing has the same address. | Canonicalization at sender + receiver: AST normalization for OMC code; recursive key-sort for JSON; raw bytes for prose. |
| **Compression without context-key state.** Sender and receiver share no per-message agreement. | Codec produces sampled-token payload addressed by canonical hash; receiver recovers via library lookup. |
| **Forward compatibility.** Old receivers handle new message kinds gracefully. | Numeric `kind` field; unknown kinds short-circuit to "passthrough" handling. |
| **Composability with content-addressed stores.** Messages reference content the receiver may already hold. | `omc_msg_recover_compressed` / `omc_msg_recover_from_registry` walk known libraries by canonical hash. |

## Wire format

Every OMC-P message is a JSON object with these fields:

| Field | Type | Purpose |
|---|---|---|
| `sender_id` | int | Agent identity. `0` reserved for kernel-level / anonymous. Convention: `fnv1a_64("agent_name")` truncated to i32. |
| `kind` | int | Message kind (see registry below). |
| `content` | string | The payload (raw, or omitted if `sampled_tokens` is present). |
| `content_hash` | int (string in JSON for precision) | Canonical hash of `content`, computed by `canonicalize` per the kind's addressing scheme. |
| `attractor` | int | Nearest Fibonacci attractor to `content_hash`. |
| `resonance` | float | `phi.res(content_hash)`. |
| `him_score` | int | HBit invariant marker. |
| `packed` | int | `(sender_id ^ kind ^ low32(content_hash))`. Identity dedup key. |
| `sampled_tokens` (optional) | int[] | Codec compressed payload (codec messages only). |
| `every_n` (optional) | int | Codec sampling rate. |
| `original_tok_count` (optional) | int | Codec receiver hint. |
| `source_bytes` (optional) | int | Original byte count. |
| `compression_ratio` (optional) | float | Token-count compression. |

### Example: raw signed message

```json
{
  "sender_id": 18173,
  "kind": 1,
  "content": "fn compute_mean(xs) { ... }",
  "content_hash": "3551785709911115688",
  "attractor": "63245986",
  "resonance": 1.78e-17,
  "him_score": 0,
  "packed": 606047779
}
```

### Example: codec-compressed message

```json
{
  "sender_id": 18173,
  "kind": 1,
  "sampled_tokens": [4, 0, 109, 0, 116, 95, 0, 120, 629, 0, 118, 0, 99, 0, 109, 0, 34, 524],
  "content_hash": "3551785709911115688",
  "attractor": "63245986",
  "every_n": 3,
  "original_tok_count": 54,
  "source_bytes": 127,
  "compression_ratio": 2.117
}
```

Note: `content` is absent. Receiver recovers via library lookup.

## Message kind registry

| `kind` | Name | Purpose |
|---|---|---|
| 0 | RESERVED | Do not use. |
| 1 | REQUEST | Sender is asking the receiver to act on `content`. |
| 2 | RESPONSE | Reply to a REQUEST. Carry `in_reply_to: <packed>` field if returning to a specific request. |
| 3 | NOTIFY | Best-effort one-way notification. No response expected. |
| 4 | FETCH | Receiver should treat `content_hash` as a request to send back the addressed content (or NOT_FOUND). |
| 5 | STORE | Sender is offering content for the receiver's local store. Receiver MAY accept. |
| 6 | HEARTBEAT | Peer liveness ping. |
| 7 | ONBOARDING | Bundle of language reference / lib manifest for new agents. See `examples/tools/gen_onboarding_token.omc`. |
| 8 | ERROR | Last operation failed. Body SHOULD contain `error: string` + optional `correlates_to: <packed>`. |
| 16+ | application-defined | Reserved for negotiated extensions. |

Receivers MUST handle kinds 1, 2, 3, 4, 5, 8. Other kinds MAY be
silently dropped if unsupported.

## Verification algorithm

To verify a received message `M`:

1. If `M.sampled_tokens` is absent (raw message):
   - `canon = canonicalize(M.content)` per addressing scheme for the
     content's kind
   - `recomputed = fnv1a_64(canon)`
   - If `recomputed != M.content_hash` → REJECT (tampered)
   - Optionally recompute `attractor`, `resonance`, `him_score` from
     `content_hash`; mismatches indicate sender bug or different
     substrate version — accept with warning.
2. If `M.sampled_tokens` is present (codec message):
   - Look up `M.content_hash` in your library (`omc-kernel`,
     registry, peer store). If found:
     - `recomputed = fnv1a_64(canonicalize(found_content))`
     - If `recomputed == M.content_hash` → RECOVERED, content = `found_content`
   - If not found:
     - SEND back a FETCH message (kind=4) for the missing hash
     - Or: REJECT pending content acquisition

`sender_id` is informational only — there is NO key-based proof that
this sender wrote this content. The integrity guarantee is over
content, not author. To bind author to content, sign the
`packed`+`content_hash` tuple with conventional PKI on top of OMC-P
(out-of-scope here).

## Canonicalization schemes (the "addressing" field)

| Scheme | Applied to | Algorithm |
|---|---|---|
| `omc_fn` | OMC source code | `canonical::canonicalize` — AST parse, normalize whitespace and comments, alpha-rename parameters/locals to canonical order, re-serialize. |
| `json` | JSON data | Recursive key-sort, re-serialize. |
| `prose` / `blob` | Arbitrary bytes | Identity (raw bytes). |

The scheme determines what counts as "the same content." Choose
the strictest scheme that preserves your semantic notion of equality.

## Codec parameters

| Param | Purpose | Range / default |
|---|---|--:|
| `every_n` | Keep every Nth canonical token | 1..16, typical 3-8 |

Wire-byte break-even (single message, measured on TinyShakespeare-
shaped OMC payloads):

| Source size | Recommended `every_n` |
|---|---|
| < 500 B | Don't compress — use raw |
| 500 B – 2 KB | 5 |
| > 2 KB | 8 |

The always-on win regardless of size is **library-lookup recovery**:
alpha-rename invariant content addressing on the receiver, no
shared key.

## Peer discovery (informative, not normative for v1)

v1 spec is point-to-point: peers know each other's addresses
out-of-band (file path, socket, HTTP URL). Peer discovery is
deferred to a future v2 that may build on:

- Substrate-aware DHT (peers announce by `attractor_bucket(content_hash)`)
- WebRTC datachannels for browser-resident agents
- Existing libp2p / IPFS peer routing

The wire format does not depend on the transport. The reference
impl uses files in a shared directory; production deployments
should use sockets / HTTP / message queues at their discretion.

## Reference flows

### Flow A: agent asks agent for a code-fragment (compressed)

```
A → B:  {sender=A, kind=4, content_hash=H}                  # FETCH H
B:      hash H is in B's store? yes → send RESPONSE
B → A:  {sender=B, kind=2, content="fn ...",                # RESPONSE
         content_hash=H, attractor=..., ...}
A:      verify: recompute fnv1a_64(canonicalize("fn ...")) == H? yes
        → ACCEPT, content trusted
```

### Flow B: agent broadcasts a code library

```
A → *:  {sender=A, kind=5, content="fn add(x,y)..."}        # STORE
A → *:  {sender=A, kind=5, content="fn mean(xs)..."}        # STORE
...
peers: each verifies + stores in local omc-kernel
```

### Flow C: codec-compressed messaging

```
A:      msg = omc_msg_sign_compressed(big_source, A_id, 1, every_n=8)
A → B:  msg (carries sampled_tokens + content_hash, no content)
B:      recovered = omc_msg_recover_from_registry(msg)      # checks local store
B:      if recovered: ACCEPT
        else: send FETCH back to A
```

### Flow D: onboarding new agent

```
A → B:  {sender=A, kind=7, content=<json blob>, ...}        # ONBOARDING
B:      verify signature
B:      parse content: {bootstrap_pack, lib_manifest, ...}
B:      ingest manifest into local omc-kernel
B:      now knows every standard fn by canonical hash
```

See `examples/tools/gen_onboarding_token.omc` for a complete
ONBOARDING bundle generator.

## Compatibility commitments

OMC-P v1:
- Field name additions are non-breaking
- Field removals require version bump
- New `kind` values in [16, ∞) are non-breaking
- New `kind` values in [9, 15] reserved for future v1 additions
- Numeric IDs must fit in `i64` for `content_hash`, `attractor`,
  `sender_id`, `packed`; JSON should serialize as decimal strings
  to avoid float-precision loss in receivers
- The `canonicalize` algorithm for each scheme is part of v1
  forever; substrate-version changes must produce a new scheme
  name (e.g. `omc_fn_v2`)

## Reference implementations

| Component | Path |
|---|---|
| Sign / verify / serialize | `omnimcode-core/src/interpreter.rs` (`omc_msg_*` builtins) |
| Codec encode / decode-lookup | `omnimcode-core/src/interpreter.rs` (`omc_codec_*` builtins) |
| Persistent store | `omnimcode-cli/src/bin/omc_kernel.rs` |
| MCP adapter | `tools/mcp_substrate/server.py` |
| End-to-end demo (raw) | `examples/demos/llm_tandem_send.omc` + `llm_tandem_receive.omc` |
| End-to-end demo (compressed + library) | `examples/demos/llm_tandem_send_compressed.omc` + `llm_tandem_receive_compressed.omc` + `llm_tandem_registry.omc` |
| Onboarding bundle | `examples/tools/gen_onboarding_token.omc` + `consume_onboarding_token.omc` |

## Non-goals

- **Authentication.** OMC-P proves CONTENT integrity, not AUTHOR
  identity. Layer PKI / OAuth / OIDC on top if needed.
- **Encryption.** Wire is plaintext JSON. Use TLS or wrap in an
  encrypted envelope before transport if confidentiality is needed.
- **Transport.** OMC-P is wire format only. Use HTTP, sockets,
  message queues, files — anything that delivers bytes.
- **Discovery.** Peers know each other out-of-band in v1.

## Naming

OMC-P is the inter-AGENT wire protocol. It is distinct from:

- **OMC** the language (`omnicode`)
- **omc-kernel** the storage CLI
- **MCP** (Anthropic Model Context Protocol) — the OMC-P MCP server
  in `tools/mcp_substrate/` adapts OMC-P operations to the MCP
  RPC layer for LLM client consumption.

## Version

This document describes **OMC-P v1**, frozen 2026-05-16.

Changes require:
- Backwards-compatible additions: PR + this doc updated
- Backwards-incompatible changes: bump to v2 + new file
  (`OMC-PROTOCOL-v2.md`) + reference impls forked or feature-gated
