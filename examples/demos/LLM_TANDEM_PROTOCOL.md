# LLM ↔ LLM Substrate-Signed Messaging Protocol

A minimal wire format for two LLMs to exchange OMC code (or any text)
with **substrate-derived integrity verification** — no shared secret.

## Idea

A signed message wraps content with HBit metadata computed from the
canonical-hash of the content. On receipt, the verifier recomputes the
metadata from the content and checks it matches. Because the metadata
is *derived* from the content (not added externally), tampering with
the content invalidates the signature deterministically. No keys.

Bonus property: the canonical-hash is invariant under whitespace,
comments, and alpha-rename. So `fn f(x) { return x; }` and
`fn f(y) { return y; }` are equivalent under the protocol — useful
when LLMs reformat each other's code.

## Message dict

```
{
  "content":      string,    // the payload
  "sender_id":   int,        // recommended: fnv1a_hash("agent_name")
  "kind":         int,        // 1=code request, 2=response, 3=question, etc.
  "content_hash": int,        // fnv1a of CANONICAL(content)
  "resonance":    float,      // HInt(content_hash).resonance — recomputable
  "him_score":    float,      // HInt(content_hash).him_score  — recomputable
  "attractor":    int,        // nearest Fibonacci to content_hash
  "packed":       int         // CRT-pack(sender_id, kind, hash mod M)
}
```

Wire format: JSON via `omc_msg_serialize(msg)` / `omc_msg_deserialize(wire)`.

## Builtins

| Builtin | Purpose |
|---------|---------|
| `omc_msg_sign(content, sender_id, kind)` | Produce signed dict |
| `omc_msg_verify(msg)` | Returns `{valid, sender_id, kind, content, expected_hash, actual_hash, drift_resonance, drift_him}` |
| `omc_msg_serialize(msg)` | dict → JSON wire string |
| `omc_msg_deserialize(wire)` | JSON → dict |

## Suggested sender IDs

Use `fnv1a_hash("agent_name") & 0x7fffffff` to derive stable per-LLM
IDs. Canonical assignments for this demo:

| Agent | sender_id |
|-------|-----------|
| Claude  | 18173 |
| Hermes  | 28765 |
| Other   | pick any |

## Suggested `kind` values

| kind | meaning |
|------|---------|
| 1 | code request — "implement this" |
| 2 | code response — "here's an implementation" |
| 3 | question — "what does this do?" |
| 4 | review — "here's my critique" |
| 5 | handshake — protocol-negotiation |

## End-to-end example

**Channel**: `/home/thearchitect/omc_channel/` (shared filesystem).

**Send** (Claude):
```bash
./target/release/omnimcode-standalone examples/demos/llm_tandem_send.omc
```

**Receive** (Hermes — should be able to run on a fresh OMC checkout):
```bash
./target/release/omnimcode-standalone examples/demos/llm_tandem_receive.omc
```

## What Hermes should look for

1. **`valid == 1`** — the substrate signature recomputed correctly. Trust the content.
2. **`actual_hash == expected_hash`** — bit-for-bit content integrity (after canonicalization).
3. **`drift_resonance ≈ 0`** and **`drift_him ≈ 0`** — substrate metadata is consistent.
4. **`sender_id`** — confirms who emitted the message.
5. **`kind`** — what to do with it.

## What this catches

- Content tampering (someone edits the payload mid-flight): hash mismatch.
- Random corruption (JSON gets truncated / mangled): parse fails, then hash mismatch.
- Stale signatures (someone signs A, swaps in B): hash mismatch.
- Format drift (Hermes vs Claude format differently): **does NOT cause failure**,
  because canonicalization runs before hashing. Round-trip OMC code through
  either formatter and the signature still validates. *This is the point.*

## What this does NOT catch

- Identity forgery: any agent can pick any `sender_id`. There's no key
  binding. For real auth, layer Ed25519 on top.
- Replay attacks: same message can be re-sent. Add a nonce field if needed.
- Confidentiality: content is plaintext. Wrap in TLS or sign-then-encrypt.

## Round-trip property the protocol relies on

```
omc_canonical_hash(s) == omc_canonical_hash(omc_code_canonical(s))
```

Both agents must canonicalize the same way — both run the same OMC
version. Different OMC versions = different canonicalizers = signatures
won't match across versions. Pin the OMC build at protocol-negotiation
time (use `kind = 5`).
