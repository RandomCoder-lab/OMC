# Install OMC Memory+

Three steps. Under 2 minutes.

## 1. Build or download `omnimcode-mcp`

**Option A — build from source** (current path):

```bash
git clone https://github.com/RandomCoder-lab/OMC.git
cd OMC
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release -p omnimcode-mcp
# Binary at target/release/omnimcode-mcp
```

**Option B — install script** (v1.1, not yet shipped):

```bash
curl -fsSL https://omc.sh/install.sh | sh
```

## 2. Register with Claude Code

Open `~/.claude.json` and add an `mcpServers` block (or merge into existing one):

```json
{
  "mcpServers": {
    "omc": {
      "command": "/absolute/path/to/omnimcode-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

Or run this one-liner (if you have `jq`):

```bash
BINPATH="/home/thearchitect/OMC/target/release/omnimcode-mcp"  # update path
jq --arg p "$BINPATH" \
  '.mcpServers.omc = {"command": $p, "args": [], "env": {}}' \
  ~/.claude.json > /tmp/claude.json.new && mv /tmp/claude.json.new ~/.claude.json
```

## 3. Restart Claude Code

`/exit` then relaunch. The MCP tools `mcp__omc__omc_compress_context`, `mcp__omc__omc_memory_store`, etc. are now available to Claude.

## Verify

In any new Claude Code session, ask Claude:

> Use omc_memory_store to remember "hello world", then omc_memory_list to confirm it was stored.

You should see a JSON response with a `content_hash` like `5144560189087515934`.

## Where memory lives

```
~/.omc/memory/
├── default/                  ← omc_memory_store with no namespace
├── omc_session_v08_findings/ ← per-session namespace
└── <your namespaces>/
```

Files are append-only logs with one entry per line: `{content_hash}\t{stored_at_unix}\t{base64_encoded_text}\n`. You can grep, diff, or delete them like any other file. Memory+ doesn't lock or encrypt them.

## Troubleshooting

**MCP tools don't appear after restart**

- Check `~/.claude.json` has valid JSON (run `jq . ~/.claude.json`)
- Check the `command` path resolves to the binary (run `<your_path> --version`)
- Check Claude Code launch logs for MCP server connection errors

**`mcp__omc__*` tools listed but `InputValidationError` when called**

- Schemas are deferred-loaded. Use ToolSearch with `query: "select:mcp__omc__omc_compress_context"` first (Claude does this automatically in normal use).

**Memory store grows unbounded**

- Default fibtier cap is 232 entries per namespace (sum of first 10 Fibonacci tier sizes). Older entries are evicted from the *index*; raw bodies stay on disk and remain recoverable by hash. Use `omc_memory_evict` to force-compact.
