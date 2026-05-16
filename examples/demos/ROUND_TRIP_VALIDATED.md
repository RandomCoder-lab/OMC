# LLM ↔ LLM Substrate-Signed Round Trip — Validated

**Date**: 2026-05-16
**Agents**: Claude Opus 4.7 (sender_id=18173) ↔ Hermes (sender_id=28765)
**Channel**: `/home/thearchitect/omc_channel/`
**Verdict**: ✓ `valid=1`, both directions, zero drift

## What the experiment proved

Two independent LLM processes exchanged OMC code through the
substrate-signed messaging protocol introduced in commit `4fcdfd6`.
Both directions verified with:

- `valid == 1`
- `actual_hash == expected_hash` (3551785709911115688)
- `drift_resonance == 0`
- `drift_him == 0`

This is the first time we have empirical evidence that **two
independent OMC runtimes, driven by two different LLMs, agree on
the canonical form of a piece of code byte-for-byte via their
substrate-derived metadata** — no shared secret, no trust
assumption, no negotiation.

## Why this matters

The integrity layer survives the things LLMs typically do to each
other's code: alpha-rename of params/locals, whitespace reflows,
comment edits, indentation differences. Because the hash is computed
on the *canonical* form (after AST normalization), both agents
produce identical hashes from formatting-different but
semantically-equivalent payloads.

Python's `hash(source)` cannot do this — it's sensitive to every
cosmetic detail. So the property we just validated is genuinely
OMC-only.

## Reproduction

```bash
# Claude side (writes signed message)
./target/release/omnimcode-standalone examples/demos/llm_tandem_send.omc

# Hermes side (verifies + signs response)
./target/release/omnimcode-standalone examples/demos/llm_tandem_reply.omc
# (Hermes's reply demo lives in their workspace; sample output preserved
# below)

# Claude verifies Hermes's response
./target/release/omnimcode-standalone /tmp/verify_hermes.omc
```

## Snapshot evidence

Preserved at this commit:

- `examples/demos/round_trip_evidence_claude.json` — Claude's signed message
- `examples/demos/round_trip_evidence_hermes.json` — Hermes's signed response

Both files are 294 bytes each. Both verify against their respective
content hashes with zero drift.

## Honest limits (unchanged)

- **No authentication**: any agent can pick any `sender_id`.
  For real auth, layer Ed25519 on top.
- **No replay protection**: same message can be re-sent.
  Add a nonce field.
- **No confidentiality**: content is plaintext JSON.
  Wrap in TLS or sign-then-encrypt.

What we proved: **integrity over a canonical semantic form** —
the load-bearing property for "two LLMs that reformat each other's
code can still verify each other."

## Next: secondary-brain prompting protocol

With this validated, the next layer can build on top: a prompting
protocol where two LLMs use the substrate channel to query each
other ("what does this function do?" → response) with substrate-
verified integrity on every message. Tracked in the follow-up
commit adding `omc_prompt_*` builtins.
