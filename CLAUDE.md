# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

OMNIcode (OMC) is a self-hosting harmonic-math language with a self-healing compiler, an embedded CPython runtime, and a package manager — shipped as a single Rust binary. The orientation doc `00-START-HERE.md` is the fastest path in; `README.md` is the marketing-facing version; `CHANGELOG.md` is the milestone-by-milestone design history (V.6 → H.5).

The repo is **two arms** that share the φ-math substrate (Fibonacci attractors, resonance, HIM score):

- **Arm 1 — Language (active).** Self-hosting compiler + self-healing diagnostics. Source under `omnimcode-core/src/{parser,ast,interpreter,vm,compiler,value,python_embed}.rs`. Demos in `examples/self_hosting_*.omc` and `examples/self_healing_*.omc`.
- **Arm 2 — Circuit evolution (frozen at v1.0.0).** Genetic algorithms over Boolean/float logic circuits, with FFI to Python/Unity/Unreal. Source in `omnimcode-core/src/{circuits,evolution,circuit_dsl,hbit,optimizer,phi_pi_fib,phi_disk}.rs`. Still here, but not where active development happens.

When in doubt about which arm a file belongs to: if it imports `parser`/`interpreter`/`vm` it's Arm 1; if it imports `circuits`/`evolution` it's Arm 2.

## Workspace layout

Cargo workspace (resolver = "2"). Members in `Cargo.toml`:

| Crate | What it is | Notes |
|---|---|---|
| `omnimcode-core` | The language: lexer, parser, AST, tree-walk interp, bytecode compiler, VM, optimizer, formatter, embedded CPython glue. Ships the `omnimcode-standalone` binary. | Default features = `python-embed`. ~15K lines of Rust, most density is `interpreter.rs` (~5K lines — every builtin dispatches here). |
| `omnimcode-ffi` | C ABI (`cdylib` + `staticlib`). Consumed by the Unity/Unreal packages. | |
| `omnimcode-wasm` | Browser/Node target. Pulls `omnimcode-core` with `default-features = false` (no libpython). | `opt-level = "z"`, ~150 KB wasm. |
| `omnimcode-lsp` | LSP via `tower-lsp`. Also disables `python-embed`. | |
| `omnimcode-python` | **Excluded** from the default workspace. The "Python embeds OMC" wrapper (extension-module). Conflicts with `python-embed` because both crates `links = "python"`. Build separately: `cargo build -p omnimcode-python`. | |

`omnimcode-core` is published-style (has `[lib]` + `[[bin]]`). The binary is `omnimcode-standalone` (renaming it breaks `build.sh`, the `standalone.omc` symlink, and `BUILD.md`).

## Build & test

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release           # standalone binary
cargo test --workspace --release                                       # all tests
cargo test --release -p omnimcode-core conformance                     # language-physics goldens
cargo test --release -- --nocapture test_name                          # single test, with output
cargo bench -p omnimcode-core                                          # criterion benches
./target/release/omnimcode-standalone examples/self_hosting_v9b.omc    # smoke test: must print "✓✓✓ ALL THREE FIXPOINTS REACHED"
```

The `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` is needed when the host Python is newer than pyo3 0.23's max supported version — pyo3 hard-errors at build time otherwise. If you genuinely don't want libpython linked (faster compile, smaller binary, but `py_*` builtins fail at runtime), build with `--no-default-features` on `omnimcode-core`.

`build.sh` does release build + runs every `examples/*.omc` as a smoke test and refreshes the `standalone.omc → target/release/omnimcode-standalone` symlink.

## CLI surface (omnimcode-standalone)

| Flag | Effect |
|---|---|
| (no args) | REPL — `:help`, `:reset`, `:quit`. Bare expressions without `;` auto-print (Python REPL style). |
| `FILE` | Run a program. |
| `--check FILE` | Parse + heal pass to fixpoint, print diagnostics, **never execute**. Exit code = 0 if clean, 1 if any diagnostic. CI-friendly. |
| `--fmt FILE` | Pretty-print AST as canonical OMC source to stdout. Lossy on whitespace/comments. |
| `--init` | Scaffold `omc.toml` + `main.omc`. Refuses to overwrite. |
| `--install [SPEC]` | SPEC = URL, registry short-name, or absent (reads `omc.toml` `[dependencies]`). Uses embedded Python for HTTP + sha256 — requires the `python-embed` feature. |
| `--list` | List `omc_modules/`. |
| `--test FILE` / `--bench FILE` | Run every `fn test_*()` / `fn bench_*()` in FILE. |
| `--audit FILE` | Run under both engines, exit 1 on output divergence. |

Driving env vars (set them at run time, not build time):

| Var | Effect |
|---|---|
| `OMC_VM=1` | Use the bytecode VM (compiler.rs → vm.rs) instead of tree-walk. |
| `OMC_HEAL=1` | Static heal-to-fixpoint pass over the AST before execution (max 5 iters). |
| `OMC_HEAL_RETRY=1` | After a runtime error, heal the AST and retry once. Independent of `OMC_HEAL`. |
| `OMC_HEAL_QUIET=1` | Suppress heal diagnostics. |
| `OMC_OPT=0` | Disable bytecode optimizer (on by default under `OMC_VM`). |
| `OMC_OPT_STATS=1` / `OMC_DISASM=1` | Print optimizer stats / pre-execution disassembly. |
| `OMC_NO_PYTHON=1` | Skip CPython init (also disables `--install`). |
| `OMC_REGISTRY=<url>` | Alternative package registry. |
| `OMC_STDLIB_PATH=<colon-sep>` | Extra import search paths. |

## Architecture invariants

These are properties of the system, not aspirations. Breaking them is the kind of "regression" that the conformance tests are designed to catch.

- **Two engines, byte-identical output.** Tree-walk (`interpreter.rs`) is the reference. Bytecode VM (`compiler.rs` lowers AST → `bytecode.rs`; `vm.rs` runs it; `bytecode_opt.rs` optimizes) must match. `--audit` enforces this on any file you point it at; the conformance suite locks it for the language's "physics" (φ-resonance values, Fibonacci attractors, healing semantics).
- **Conformance tests in `omnimcode-core/tests/conformance.rs` are a contract**, not regular tests. They lock semantics shared with the canonical Python omnicc. If one fails: either you genuinely changed semantics (update CHANGELOG and the Python side) OR you regressed — fix the code, never relax the assertion. The file's header comment spells this out.
- **`Value::Array` and `Value::Dict` are `Rc<RefCell<...>>`-shared.** This was the killer perf fix (commit `d3c29b6`). Mutating collections pass by reference like Python's mutable types, not by clone. Don't reintroduce defensive `.clone()` on collection values.
- **Healing is opt-in.** Production code shouldn't ship with `OMC_HEAL=1`. The default tree-walk path executes the parser's output as-is; healing is a separate AST-rewrite pass invoked by env var.
- **Embedded CPython is a feature-gate.** `omnimcode-core` builds and tests cleanly without it (`--no-default-features`). When adding code that touches Python: keep it behind `#[cfg(feature = "python-embed")]` and provide a stub for the negative branch — see `maybe_register_python` in `main.rs:84-98` for the pattern.

## Adding a builtin

The mechanical recipe (used dozens of times in the changelog):

1. Implement in `omnimcode-core/src/interpreter.rs` — add a match arm to the builtin dispatch in `call_function` / `call_builtin`. The function name string is the dispatch key; arguments arrive as already-evaluated `Value`s.
2. If the builtin should run under `OMC_VM=1` too, add a corresponding op to `bytecode.rs` (or route through the generic call path) and handle it in `vm.rs` / `compiler.rs`. Many builtins go through a generic `OpCallBuiltin` path; only hot-path ones (arr_get, dict_get, str_concat, …) get inlined VM ops.
3. Update `STDLIB.md` — the canonical built-in reference, organized by category.
4. If it's user-visible in healing diagnostics or type-inference, propagate to `compiler.rs`'s inference table.
5. Add a conformance test if the behavior is semantic (not just convenience).

For new language features (new keyword, syntax, AST node), the spread is `parser.rs` → `ast.rs` → `interpreter.rs` (+ `compiler.rs` if it should run on the VM) → `formatter.rs` (so `--fmt` round-trips it).

## Packages & libraries

OMC libraries live in `examples/lib/*.omc` and are registered in `registry/index.json` with sha256 verification. Hosting is decentralized (any HTTPS URL); the registry is just name→URL mapping. Six wrappers ship out of the box: `np`, `pd`, `sk`/`sklearn`, `requests`, `sqlite`, `torch`, plus the harmonic-native libs (`harmonic_anomaly`, `harmonic_clustering`, `harmonic_recommend`). To submit a package, PR `registry/index.json`.

`--install` is dog-fooded: HTTP fetch goes through embedded Python's `requests`, TOML parse through `tomllib`, sha256 through `hashlib`. There are **zero Rust HTTP/TOML/hash dependencies** as a deliberate choice — keep it that way.

## Branch convention for this session

Develop on `claude/add-claude-documentation-MNbSx`. Commit and push there; do NOT push to `master` without explicit permission.

## Doc map

`00-START-HERE.md` is the orientation. `README.md` is the pitch. `CHANGELOG.md` is the design history (the V.6 → H.5 entries are the most relevant for current code). `ARCHITECTURE.md` and `DEVELOPER.md` describe the v1.0.0-era circuit-evolution arm — useful for the `circuits.rs`/`evolution.rs` lane, partial relevance to the language lane (the parser/interpreter sections are still accurate; the "extension points" sections describe a now-implemented future). `STDLIB.md` is the builtin reference. `BENCHMARKS.md` has tree-walk vs VM numbers. `PHI_PI_FIB_ALGORITHM.md` is the math foundation. `PHI_DISK.md` and `TIER_4_HONEST_REVISION.md` describe the LRU/Fibonacci-search sub-component honestly (it didn't deliver what its early docs claimed — read TIER_4_HONEST_REVISION first).
