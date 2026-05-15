# omnimcode-wasm

OMNIcode for WebAssembly. The standalone language + harmonic primitives,
running in browsers, Node, Deno, or any wasm-bindgen host. Excludes
`py_*` builtins (libpython doesn't link in wasm32) — everything else
is identical to the desktop binary.

**Bundle size:** ~530 KB before gzip, ~150-200 KB after.

## Building

```bash
# wasm32 target (one-time)
rustup target add wasm32-unknown-unknown

# Build the .wasm artifact
cargo build --release -p omnimcode-wasm --target wasm32-unknown-unknown
# → target/wasm32-unknown-unknown/release/omnimcode_wasm.wasm
```

For npm distribution, install [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```bash
cargo install wasm-pack
cd omnimcode-wasm
wasm-pack build --release --target web
# → pkg/ contains omnimcode_wasm.js + .wasm + package.json
```

To publish:

```bash
cd pkg
npm publish
```

## Using from JavaScript

```javascript
import init, { OmcRuntime, run_once, version } from 'omnimcode-wasm';

await init();          // load + initialise the wasm module
console.log(version()); // "1.0.0"

// Persistent runtime — state survives across calls
const omc = new OmcRuntime();
omc.run("h x = fold(7);");           // x = 8 (nearest Fibonacci)
omc.run("h y = harmony_value(89);"); // y ≈ 1.0 (89 IS Fibonacci)
console.log(omc.get_var("x"));        // "8"
console.log(omc.get_var("y"));        // "1.0"

// One-shot evaluation — returns the value as a string
console.log(omc.eval("3 + 4 * 2"));   // "11"

// Reset state
omc.reset();

// Stateless one-shot — no runtime needed for simple scripts
run_once("println(harmonic_partition([3, 7, 21, 22, 89]));");
```

## What works in WASM

- The full OMNIcode language: closures, pattern matching, try/catch, harmonic primitives
- Two-engine execution (tree-walk + bytecode VM, byte-identical output)
- All in-language libraries that don't need Python (the `harmonic_*` libs)
- Self-healing compiler pass (`OMC_HEAL`-equivalent via interpreter API)

## What doesn't work in WASM

- `py_import`, `py_call`, `py_eval`, `py_callback`, etc. — fail with `Undefined function`
- `--install` (uses `requests` for HTTP fetch)
- File I/O (`read_file`, `write_file`) — by design, browsers don't expose
  local FS to JS. Use `fetch` in JS, pass strings into OMC's `run` / `eval`.

For Python-dependent workloads, use the desktop standalone binary.

## Use cases

- **Live OMC REPL in a browser** — for documentation sites, tutorials, experimentation
- **Jupyter / Observable notebooks** — embed OMC alongside Python/JS cells
- **Edge functions** (Cloudflare Workers, Vercel Edge) — fast cold-start anomaly detection
- **Client-side data analysis** — run `harmonic_anomaly` on user data without a backend
