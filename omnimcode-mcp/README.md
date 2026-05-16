# omnimcode-mcp

MCP server for OMC. Lets an LLM client (Claude Desktop, Cursor, any
JSON-RPC capable agent) call OMC as a runtime — eval code, look up
builtins, get structured error explanations.

Built so an LLM can write idiomatic OMC without it being in training
data: the introspection + error catalog tools give it everything it
needs to discover the language at runtime.

## Tools exposed

- `omc_eval(code)` — evaluate OMC source, return result value
- `omc_help(name)` — signature + description + example for a builtin
- `omc_list_builtins(category?)` — enumerate documented builtins
- `omc_categories()` — list builtin categories
- `omc_unique_builtins()` — OMC-only primitives (no NumPy equivalent)
- `omc_explain_error(message)` — pattern-match an error against the
  259-entry knowledge base; returns explanation + cause + fix
- `omc_did_you_mean(name)` — typo suggestions over the known surface

## Build

```bash
cargo build --release -p omnimcode-mcp
# Binary lands at target/release/omnimcode-mcp
```

## Claude Desktop config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or the equivalent on your platform:

```json
{
  "mcpServers": {
    "omc": {
      "command": "/absolute/path/to/target/release/omnimcode-mcp"
    }
  }
}
```

Restart Claude Desktop. The LLM can now call `omc_eval`, `omc_help`,
etc. directly.

## Why this matters for LLMs

OMC has ~200+ builtins, many added recently. Without a discoverable
surface, an LLM will hallucinate `numpy.dot` or invent `arr_multiply`.
With the MCP server wired in, the LLM:

1. Calls `omc_categories()` to see what's available
2. Calls `omc_list_builtins("substrate")` to find OMC-unique primitives
3. Calls `omc_help("arr_substrate_attention")` for signature + example
4. Writes code, calls `omc_eval`
5. On error, calls `omc_explain_error(msg)` for a one-line fix

The OMC-unique primitives — substrate-typed arrays, autograd that
preserves φ-resonance, native lazy generators, harmonic ops — are
the reason an LLM would pick OMC over NumPy/PyTorch. The MCP server
makes those discoverable.

## Protocol

Line-delimited JSON-RPC 2.0 over stdin/stdout. Implements:
- `initialize` (returns server info + capabilities)
- `tools/list` (returns the tool catalog above)
- `tools/call` (dispatches to a tool by name)

Notifications (no `id` field) are accepted silently. Anything else
gets a "Method not found" error.

## Example manual session

```
$ ./target/release/omnimcode-mcp
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"omc_eval","arguments":{"code":"is_attractor(8);"}}}
```

Returns:

```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"text":"HInt { value: 1, resonance: 1.000, him: 0.382 }","type":"text"}],"isError":false}}
```

Notice the substrate metadata in the response — that's the part Python
can't give you.
