# `.omcs` — substrate-keyed save file format (v1)

> A self-contained, integrity-verified bundle of canonical-hash-
> addressed content. Save your kernel store / agent memory /
> code library as one file; restore anywhere.

## Why a new format

Existing save formats — JSON, pickle, parquet, protobuf — address
content by name + position. `.omcs` addresses content by **canonical
hash**. Two `.omcs` files containing the same fn (under any
parameter renaming or whitespace edit) carry the same hash; merging
them is a set union without conflict resolution.

Combined with the kernel + codec + signed-envelope primitives we
already shipped, `.omcs` is the **distributed-friendly artifact
format** for substrate-aware agents.

## Use cases

| Use | How |
|---|---|
| Snapshot an agent's memory | `omc-kernel pack memory.omcs` |
| Ship a code library | pack the kernel store of registry libs, distribute the .omcs |
| Hot-swap LLM context | pack an in-progress conversation's canonical content, unpack on a different agent |
| Sync two agents | A packs → wire transfer → B unpacks. Tamper-evident by envelope hash. |
| Backup with dedup | Re-packing a store produces the same envelope hash as long as content is unchanged. |

## Format (v1)

A `.omcs` file is a single JSON object:

```json
{
  "omcs_version": 1,
  "created_at_unix": 1747500000,
  "entry_count": 193,
  "envelope_hash": "4281062401442748079",
  "envelope_attractor": "63245986",
  "entries": [
    {
      "canonical_hash": "02158af4e9935df8",
      "kind": "omc_fn",
      "attractor": "63245986",
      "size_bytes": 59,
      "content": "fn commit(conn) { return py_call(conn, \"commit\", []); }\n"
    },
    ...
  ]
}
```

### Fields

| Field | Type | Purpose |
|---|---|---|
| `omcs_version` | int | Format version (1) |
| `created_at_unix` | int | Pack timestamp |
| `entry_count` | int | Number of entries; matches `entries.len()` |
| `envelope_hash` | string-int | fnv1a of concatenated entry hashes; tamper-evident |
| `envelope_attractor` | string-int | Nearest Fibonacci attractor to envelope_hash |
| `entries[]` | array | One per stored item |

### Entry fields

| Field | Type | Purpose |
|---|---|---|
| `canonical_hash` | string (hex i64) | Primary address |
| `kind` | string | `omc_fn` / `json` / `prose` / `blob` |
| `attractor` | string-int | Nearest Fibonacci attractor |
| `size_bytes` | int | Raw content length |
| `content` | string | The actual content |

## Integrity model

Two-layer verification:

1. **Envelope hash.** `envelope_hash = fnv1a_64(concat(canonical_hash for each entry))`. Re-concatenate on unpack; if recomputed != claimed, the bundle's ENTRY SET was modified (entry added, removed, reordered).

2. **Per-entry canonical hash.** For each entry, recompute the canonical hash from its content using the appropriate canonicalizer for `kind`. If recomputed != `canonical_hash`, that ENTRY'S CONTENT was modified. Skip the entry; continue.

So:
- Adding/removing/reordering entries → fails envelope check (whole bundle rejected)
- Modifying one entry's content → that entry skipped; rest of bundle still ingested

This matches the substrate's design principle: content integrity is intrinsic to addressing.

## Operations

```bash
# Pack the current kernel store into a bundle.
omc-kernel pack OUT.omcs

# Unpack a bundle into the kernel store (additive — won't overwrite
# entries that already exist with matching hash).
omc-kernel unpack IN.omcs
```

Both operations are O(N) in the number of entries; pack is bottlenecked by disk write, unpack by canonicalization re-verification.

## End-to-end demo (numbers from a real run)

```
$ omc-kernel ingest examples/lib/
ingested 215 fns: 193 new, 22 already present in store

$ omc-kernel pack /tmp/lib.omcs
packed 193 entries into /tmp/lib.omcs (53530 bytes);
  envelope_hash=3b696392734696af

$ rm -rf ~/.omc/kernel       # wipe local store
$ omc-kernel unpack /tmp/lib.omcs
unpack: envelope verified (193 entries)
unpacked 193 entries: 193 new, 0 already in store, 0 tampered

$ omc-kernel ls | head
193 fn(s) in store at /home/user/.omc/kernel/store
canonical-hash        bytes  fn
02158af4e9935df8         59  fn commit
...
```

Tamper test: modify one entry's content, re-unpack:

```
$ # (modify entries[0].content)
$ omc-kernel unpack /tmp/lib.omcs
unpack: envelope verified (193 entries)
unpacked 193 entries: 192 new, 0 already in store, 1 tampered (skipped)
```

The envelope hash still verifies (entry set unchanged); the per-entry
recompute catches the modification of the one entry and skips it.

## Compose with the rest of OMC

- `.omcs` files are valid input to OMC-PROTOCOL.md kind=5 STORE
  messages: an agent can wrap a bundle in a substrate-signed
  message envelope and ship over any transport.
- The MCP server can expose `omc_pack(out_path)` / `omc_unpack(in_path)`
  as additional tools (one-liner adapters).
- The kernel's existing `ingest` is the OMC-source-tree input;
  `unpack` is the bundle input; together they cover both
  "ingest from disk" and "ingest from network/peer."

## What's NOT in v1

- **Binary encoding.** JSON is the v1 format. A binary encoding
  (CBOR or a custom framed format) is a future v2 — would shrink
  bundles ~30-50% and speed up unpack.
- **Per-entry codec compression.** Each entry's `content` is the
  raw bytes. Compressed entries (via `omc_codec_encode`) would shrink
  bundles further but require recovery via library lookup on the
  unpack side. v2 candidate.
- **Per-entry signatures.** The envelope is hashed but unsigned;
  trust comes from the substrate-recompute on unpack, not from PKI.
  Layer signing on top if needed.
- **Streaming.** v1 loads the entire bundle into memory. Streaming
  unpack for huge bundles is a v2 add.

Each is a separable extension that doesn't break v1 compatibility.

## Version commitment

v1 frozen 2026-05-16. Future v2 will live in a separate spec file
and v1 unpackers will refuse v2 bundles (and vice versa) until
upgraded. Additive fields within v1 are non-breaking; field
removals or semantic changes require v2.
