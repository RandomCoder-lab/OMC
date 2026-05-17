# omc-kernel — content-addressed code DAG over canonical hashes

> **Code as a content-addressed Merkle DAG over substrate-canonical
> addresses.** Version code the way IPFS versions files, except the
> addressing is semantic instead of byte-level — alpha-rename,
> whitespace, and comment edits all collapse to the same address.

## What it does

A standalone CLI that maintains a file-system content-addressed
store at `~/.omc/kernel/store/<canonical_hash_hex>.omc`. Every OMC
fn has a 64-bit canonical-hash identity (whitespace-stripped,
comments removed, parameter binding normalized), and the store is
keyed on that.

Subcommands:

| Command | Purpose |
|---|---|
| `omc-kernel ingest DIR` | walk DIR, extract every top-level fn from `.omc` files, store one entry per canonical hash |
| `omc-kernel fetch HASH` | retrieve stored source by canonical hash (hex) |
| `omc-kernel stat HASH` | substrate metadata: attractor, distance, fn name, origin file |
| `omc-kernel ls` | list stored hashes + fn names |
| `omc-kernel sign FILE` | emit a substrate-signed wire message for the OMC source in FILE |
| `omc-kernel verify` | read a wire message from stdin, recover content via store lookup |
| `omc-kernel demo` | end-to-end alpha-rename invariant recovery |

## End-to-end proof (the actual run)

```bash
$ omc-kernel ingest examples/lib/
ingested 206 fns: 184 new, 22 already present in store

$ cat > /tmp/renamed.omc <<'EOF'
fn commit(handle) { return py_call(handle, "commit", []); }
EOF

$ omc-kernel sign /tmp/renamed.omc | tee wire.json > /dev/null
# wire is a JSON dict with content_hash + sampled_tokens

$ omc-kernel verify < wire.json
verify: content_hash = 02158af4e9935df8
verify: store hash matches; recovered 59 bytes
fn commit(conn) {
    return py_call(conn, "commit", []);
}
```

Sender wrote `fn commit(handle)`; receiver recovered `fn commit(conn)`
— the canonical form already stored. Same canonical-hash address, no
shared key, no certificate authority. Alpha-rename + whitespace
invariance is intrinsic to the addressing.

## Why this is a kernel primitive

The store is a "kernel" in the OS sense: a single shared substrate
that holds canonical-form content and serves it to any agent that
asks for it by hash. The codec we shipped earlier (`omc_msg_*`)
produces wire messages keyed on the same canonical hash; the kernel
is the backing store that makes recovery work.

Combined, you get the building blocks for a distributed code DAG:
- Each fn has a 64-bit stable identity (canonical hash)
- Each fn's dependencies (callees) form an outgoing-edge set — also
  hashes
- The DAG is content-addressed end-to-end; no naming conflicts,
  no version-string negotiation
- Substrate signature verifies content integrity without a key
  (recompute and compare)

## What's NOT shipped (honest limits)

- **No daemon yet.** All operations are CLI-level on the store
  directly. Multi-process/multi-host access requires a wrapper
  (Unix domain socket / HTTP / gRPC — pick later).
- **No peer-to-peer discovery.** Single-node only. Cross-host
  replication is a follow-on layer: each peer holds its own store,
  fetches can fall back to peers on miss.
- **No outgoing-edge graph.** Each store entry has a sidecar with
  substrate metadata but no callee list. Building the Merkle DAG
  edges requires parsing each fn's calls and recording their
  canonical hashes. One-pass extension.
- **Garbage collection.** No reference counting; entries persist
  until manually deleted. Reasonable for the prototype.
- **Compression on disk.** Each entry is stored as raw source for
  human inspection. Could swap to the codec payload for ~5–7× disk
  savings on larger libraries (with the store itself as the
  recovery library — circular but the recovery path is unchanged).

Each limit is a separable extension that doesn't require redesign
of the address scheme.

## Building

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release --bin omc-kernel
./target/release/omc-kernel demo
```

No JIT or Python deps required.

## Environment

| Var | Purpose | Default |
|---|---|---|
| `OMC_KERNEL_ROOT` | override store location | `~/.omc/kernel` |

## Relationship to the existing pieces

| Layer | Provides | Used by |
|---|---|---|
| `canonical::canonicalize` | the canonical-form normalizer | omc-kernel, omc-grep, codec |
| `tokenizer::fnv1a_64` | 64-bit canonical hash | all three |
| `phi_pi_fib::nearest_attractor_with_dist` | Fibonacci-attractor metadata | omc-kernel `stat`, codec messages |
| `omc_msg_sign_compressed` / `_recover_*` | OMC-builtin wire codec | sender side of the kernel |
| **`omc-kernel`** | **persistent content-addressed store** | **receiver side** |

The four-layer stack: substrate primitives → tokenizer → codec →
kernel-store. Everything below the kernel exists already; the
kernel is the persistence + retrieval layer that makes them
compose into a real distributed-agent system.
