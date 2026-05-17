# omc-substrate MCP server

Expose the OMC kernel as MCP tools any MCP-aware LLM can invoke.
Compression and memory become **skills the model uses**, not
infrastructure the model has to understand.

No retraining required. The LLM just calls the tools.

## Tools

| Tool | Purpose |
|---|---|
| `omc_store(content, kind="prose")` | Store content; return canonical hex hash |
| `omc_lookup(hex_hash)` | Retrieve stored content by hash |
| `omc_canonicalize(content, kind)` | Compute hash without storing (dedup check) |
| `omc_stat(hex_hash)` | Sidecar metadata for a stored entry |
| `omc_list()` | Enumerate all stored entries |
| `omc_compress(content, every_n=3)` | Apply substrate codec to OMC source |

`kind` selects the canonicalizer:
- `omc_fn` — alpha-rename-invariant OMC canonical form
- `json` — recursive key-sort + re-serialize (semantic-equal JSON collapses)
- `prose` — raw bytes (exact-text dedup, default)
- `blob` — alias for prose

## Why this is the unlock

The MCP layer lets ANY existing LLM use canonical-hash addressing
for cost/memory/context without fine-tuning. The agent's loop becomes:

```
# Before: re-paste the same function body every iteration
> assistant: "let me write the fn... [500 bytes of source]"
> tool result: [output]
> assistant: "let me revise... [501 bytes of source]"

# After: store once, reference by hash
> assistant: omc_store(content="fn ...", kind="omc_fn")
> tool: "stored at hash 1a2b3c..."
> assistant: omc_lookup("1a2b3c...") if I need it again
```

Multiply this across an agentic session and the token-cost / context
savings are significant. Across multiple agents, the kernel is the
shared substrate memory.

## Install

```bash
# 1. Build the omc-kernel binary (one-time)
cd /path/to/OMC
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release --bin omc-kernel

# 2. Install Python deps for the server
pip install mcp

# 3. Register with your MCP-aware client (Claude Desktop, Cursor, etc).
#    Example claude_desktop_config.json:
{
  "mcpServers": {
    "omc-substrate": {
      "command": "python3",
      "args": ["/path/to/OMC/tools/mcp_substrate/server.py"],
      "env": {
        "OMC_KERNEL_BIN": "/path/to/OMC/target/release/omc-kernel",
        "OMC_KERNEL_ROOT": "/home/USER/.omc/kernel"
      }
    }
  }
}
```

## How it composes

The server shells out to `omc-kernel`, so the same backing store at
`~/.omc/kernel/store/` is shared with:

- Direct CLI use (`omc-kernel fetch <hash>`)
- Other MCP clients pointing at the same `OMC_KERNEL_ROOT`
- Future inter-LLM substrate protocol (peer agents)

This is the "content-addressed AI" surface, delivered as MCP. The
substrate is the namespace; the kernel is the database; the MCP
server is the API.

## Honest limits

- Server is stdio-only (the standard MCP transport)
- No auth — relies on filesystem permissions on `OMC_KERNEL_ROOT`
- `omc_compress` shells out to `omnimcode-standalone` per call;
  fine for occasional use, batch via OMC scripts for hot paths
- Prose canonicalization is byte-exact only (no semantic
  deduplication for natural-language content — that would require
  a content-canonicalizer which is a separate research problem)
