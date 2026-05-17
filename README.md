# OMNIcode (OMC)

> A harmonic-substrate programming language with first-class φ, dual-band execution, an LLVM-backed JIT, a self-healing compiler, an O(log_φπfib N) algorithm family, and a substrate-native ML framework whose substrate-aware transformer attention wins at TinyShakespeare scale.

OMC is built around **φ** (the golden ratio) and a canonical 40-entry Fibonacci attractor table reaching 63,245,986. Every harmonic operation in the language — `fold`, `phi.res`, `substrate_search`, the heal pass's literal-rewrite, attention layers in the Prometheus ML framework, the bucketing in the anomaly detector — routes through the same substrate. The substrate is a primitive of the language type system, not a library on top.

[![Latest release](https://img.shields.io/github/v/release/RandomCoder-lab/OMC?label=latest&color=blue)](https://github.com/RandomCoder-lab/OMC/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of contents

- [Why OMC](#why-omc)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Capabilities](#capabilities)
- [Use cases](#use-cases)
- [LLM integration](#llm-integration)
- [Documentation map](#documentation-map)
- [Reading the project's history](#reading-the-projects-history)
- [Repository layout](#repository-layout)
- [CLI reference](#cli-reference)
- [Status](#status)
- [Contributing](#contributing)
- [License](#license)

---

## Why OMC

These are concrete language features, not aspirations:

- **The substrate is a primitive, not a library.** `HInt`, OMC's integer type, carries a φ-resonance and HIM score computed at construction. Every `Value::HInt(_)` ever created has been routed through `compute_resonance` and `nearest_attractor_with_dist`. Substrate-ness is at the type level.

- **Dual-band executable code.** OMC values have a classical α-band and a harmonic shadow β-band, packed into LLVM `<2 x i64>` SSE2 vectors inside JIT'd functions. `phi_shadow(x)` makes β diverge; `harmony(x)` reads the substrate-routed coherence. **Branch elision based on harmony** is shipped: high-coherence inputs skip entire conditional blocks at native code speed (95.2% reduction on high-harmony inputs).

- **O(log_φπfib N) algorithm family.** `substrate_search` and friends use F(k)/φ^(π·k) split-points — each iteration shrinks the live range by **φ^π ≈ 4.534**, not 2. The canonical iteration bound is `log_φπfib(n) ≈ 0.459 · log₂ n`. A complete primitive family is exposed: `substrate_lower_bound`, `substrate_upper_bound`, `substrate_rank`, `substrate_count_range`, `substrate_slice_range`, `substrate_intersect`, `substrate_difference`, `substrate_insert`, `substrate_quantile`, `substrate_select_k`, `substrate_nearest`, `substrate_min_distance`, `substrate_hash`.

- **Zeckendorf as first-class integer encoding.** Every positive integer has a unique sum of non-consecutive Fibonaccis. OMC exposes the canonical encoder/decoder: `zeckendorf(n)`, `from_zeckendorf(idxs)`, `zeckendorf_weight`, `zeckendorf_bit`, `is_zeckendorf_valid`.

- **Substrate-aware ML framework (Prometheus).** Pure-OMC tape autograd, AdamW, Embedding, LayerNorm, CRT-Fibonacci PE, multi-head/multi-block attention, content-addressed checkpoints. Three substrate-attention component swaps (K, S-MOD softmax, V) stack inside one transformer block for **−8.94% val on TinyShakespeare**. Every result cross-validated in PyTorch.

- **Self-healing compiler** with 11 heal classes: typo correction (call-site + variable-position, substrate-bucketed), arity pad/truncate, divide/mod by zero, harmonic-index snap, missing-return, str-concat coercion, null-arithmetic coercion, if-numeric diagnostic. Pragma opt-outs available.

- **Content-addressed code storage.** `omc-kernel` stores OMC source by canonical hash (alpha-rename-invariant); two processes converging on the same canonical form produce the same address. `omc-grep` finds renamed-but-identical functions across the codebase. `omc_codec_encode/decode_lookup` compresses code 10–50× via library-lookup.

- **Substrate-signed messaging.** `OMC-PROTOCOL v1` is a wire format where integrity is verified by canonical-hash recompute. No PKI, no shared keys — agents trust messages because the substrate trusts them.

- **Two execution engines kept byte-identical.** Tree-walk interpreter + bytecode VM; `--audit FILE` verifies divergence-free output. Optional LLVM-18 JIT on top.

- **Forgiving by default.** Python users can sit down and write OMC reaching for familiar intuitions — `len(d)`, `range(0, 10, 2)`, `x += 1`, `xs[-1]`, `for key in dict` — and have it Just Work. Runtime errors include call-stack traces and did-you-mean hints.

---

## Installation

### Prerequisites

- **Rust** 1.75+
- **Python 3** (for embedded CPython interop — optional but enabled by default; use `OMC_NO_PYTHON=1` to skip)
- **LLVM 18 + libpolly-18 + libzstd** (only for the JIT path — optional)

### From source

```bash
git clone https://github.com/RandomCoder-lab/OMC.git
cd OMC
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release -p omnimcode-cli
./target/release/omnimcode-standalone --version
```

The binary is `target/release/omnimcode-standalone`. Symlink it into your `PATH` as `omc` if you'd like.

### With the LLVM JIT enabled

```bash
sudo apt install llvm-18-dev libpolly-18-dev libzstd-dev   # Debian/Ubuntu
# Or equivalent for your platform.

PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 \
    cargo build --release -p omnimcode-cli --features llvm-jit
```

Run with the JIT path:

```bash
OMC_HBIT_JIT=1 OMC_HBIT_JIT_VERBOSE=1 \
    ./target/release/omnimcode-standalone your_program.omc
```

Eligible user functions get compiled to dual-band native code. Documented benchmarks: **272× on factorial(12)**, 115× on array-sum hot loops, 10.6× on substrate-heavy mixed workloads vs the tree-walk path.

### Other targets

| Target | Crate | Build command |
|---|---|---|
| WebAssembly (browser) | `omnimcode-wasm` | `wasm-pack build omnimcode-wasm --target web` |
| LSP server (editors) | `omnimcode-lsp` | `cargo build --release -p omnimcode-lsp` |
| Godot 4 plugin | `omnimcode-gdextension` | `cargo build --release -p omnimcode-gdextension` |
| Python bindings | `omnimcode-python` | `cargo build --release -p omnimcode-python` |

### Editor support

- VS Code: install from `omnimcode-lsp/vscode-extension/`
- Any LSP-aware editor: point at `target/release/omnimcode-lsp`

---

## Quick start

A first program — `hello.omc`:

```omc
fn main() {
    h items = ["apple", "banana", "cherry"];
    for i in range(len(items)) {
        print("item " + to_string(i) + ": " + items[i]);
    }

    # Python-style negative indexing
    print("last: " + items[0 - 1]);

    # Built-in substrate primitives
    print("89 is Fibonacci → resonance: " + to_string(phi.res(89)));
}
main();
```

```bash
./target/release/omnimcode-standalone hello.omc
```

Or start the REPL:

```bash
./target/release/omnimcode-standalone
> h x = 89;
> phi.res(x)
1.0
> substrate_search([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144], 89)
9
> ^D
```

Scaffold a new project:

```bash
./target/release/omnimcode-standalone --init my_project
cd my_project
./target/release/omnimcode-standalone main.omc
```

---

## Capabilities

A high-level map of what OMC ships. Detailed claims in `docs/` and the per-release chapter notes.

### Language

- Two execution engines (tree-walk + bytecode VM), byte-identical via `--audit`
- Optional LLVM-18 JIT with dual-band SSE2 codegen, harmony-gated branch elision
- Self-hosting compiler — `gen2 == gen3` byte-identical
- Self-healing pass with 11 heal classes (typo, arity, div-zero, mod-zero, harmonic-index, missing-return, str-concat, null-arith, if-numeric, var-typo, plus substrate-bucketed lookup)
- Python-idiom builtins (`len`, `range`, `getenv`, `to_hex`, `parse_int`, `+=`, negative indexing)
- f-strings, generators, classes with inheritance, typed exceptions, closures
- Pragmas for per-fn optimization (`@hbit`, `@harmony`, `@predict`, `@no_heal`, …)

### Substrate primitives

- 40-entry Fibonacci attractor table reaching 63,245,986
- Substrate-routed search family (`substrate_search`, `_lower_bound`, `_quantile`, `_select_k`, …)
- Zeckendorf encoding (`zeckendorf`, `from_zeckendorf`, …)
- Substrate hashing (`substrate_hash`, `attractor_bucket`)
- Substrate analytics (`harmonic_align`, `harmonic_score`, `resonance_band_histogram`, `phi_pi_log_distance`)

### ML framework (Prometheus)

- Tape-based reverse-mode autograd in pure OMC (20+ tape ops)
- Optimizers (SGD, AdamW), Embedding, LayerNorm, Linear, ReLU, Softmax, MSE/CE loss
- Multi-head attention, multi-block transformer composition
- CRT-Fibonacci positional encoding
- **Substrate-K attention** (CRT-Fibonacci as K), **S-MOD softmax**, **substrate-V resample** — three substrate components that stack to −8.94% val on TinyShakespeare
- Content-addressed model checkpoints, substrate-cached inference
- Cross-framework parity: every result reproduced in both pure-OMC and PyTorch

### Infrastructure

- Content-addressed kernel (`omc-kernel`) with alpha-rename-invariant storage
- Code archaeology CLI (`omc-grep`) — finds renamed-but-identical functions
- Substrate codec (`omc_codec_encode/decode_lookup`) for 10–50× compressed code transport
- Substrate-signed wire format (OMC-PROTOCOL v1)
- MCP server (`omnimcode-mcp`) exposing OMC as a runtime to LLM clients
- Package manager (`--install`, sha256-verified registry or arbitrary URL)
- Embedded CPython for ML interop (`py_import`, `py_call`, `py_callback`)
- WASM target with no LLVM/Python dependencies

---

## Use cases

Things people actually build with OMC:

- **Substrate-aware ML research.** Prometheus is a pure-OMC ML framework with built-in PyTorch parity for cross-validation. Substrate-K attention, S-MOD softmax, and substrate-V resample are production defaults. See `examples/lib/prometheus.omc` and `experiments/prometheus_parity/`.
- **Anomaly detection with structural signal.** `harmonic_anomaly` beats scikit-learn's IsolationForest on multi-dim credential-stuffing patterns (10/10 vs 7/10 at K=10). The substrate is a **structural detector**, not a primary computation replacement. See `examples/datascience/multidim_anomaly.omc`.
- **Code archaeology + dedupe.** `omc-grep` finds renamed-but-identical functions via canonical hash; on OMC's own examples tree, surfaced 31.7% redundancy that text-grep and ast-grep couldn't catch.
- **Multi-agent systems with cryptographic integrity but no PKI.** OMC-PROTOCOL v1 lets agents verify each other's messages by recomputing canonical hashes — no shared keys, no certificate authority. See `OMC-PROTOCOL.md` and `docs/SUBSTRATE_NATIVE_AGENT.md`.
- **Substrate-keyed compressed code transport.** `omc-codec` + `omc-kernel` together enable LLM-context-efficient code exchange: ship a 50-byte hash + lookup table reference instead of the full function body. Receiver recovers the original (alpha-rename-invariant) via library lookup.
- **Substrate-routed search on integer-keyed data.** When your data is already substrate-indexed (attractor-aligned IDs, Fibonacci-spaced keys), `substrate_search` uses fewer probes than binary search and the probe sequence carries substrate metadata.
- **Self-healing tooling.** `--check` reports diagnostics; `OMC_HEAL=1` auto-applies the fixes; `OMC_HEAL_RETRY=1` retries after runtime errors. Useful in CI and lint workflows.
- **Embedded scripting with Python interop.** Drive numpy, pandas, scikit-learn from inside OMC. The substrate primitives compose with `py_call` so substrate-routed pre/post-processing can wrap arbitrary Python workloads. See `examples/datascience/titanic.omc`.

---

## LLM integration

OMC is designed to be a runtime LLM clients can drive, not just an authoring target for humans:

- **MCP server** (`omnimcode-mcp`) — exposes OMC over the Model Context Protocol. An LLM client gets `omc_run`, `omc_eval`, `omc_check`, plus substrate primitives as MCP tools.
- **`did_you_mean` baked into runtime errors** — `Undefined function: fbi (did you mean: fib? — signature: fn fib(n) -> int)`. The error includes the suggestion AND its call shape so the LLM doesn't need a follow-up `omc_help` round-trip.
- **Substrate-bucketed typo lookup** — `~10×` faster than naive closest-name scan on projects with hundreds of names. Surfaces close matches even when the LLM produces a near-miss identifier.
- **Inline signature hints** in error messages reduce the "I generated wrong code → ask for signature → regenerate" loop to a single iteration.
- **Substrate codec** (`omc_codec_encode`) for compressed code context: when an LLM needs to reference a function it's seen before, the canonical hash is a 50-byte stand-in for the whole body.
- **Substrate-aware tokenizer** with 285+ builtins and 113 phrase-level dict entries. Tokens carry CRT-packed `(kind, vocab_id, position_class)` IDs, so the LLM can reason about token structure (`omc_token_distance` exposes the substrate metric).
- **`omc_explain_error`** — a curated catalog of 702 error patterns, each with a natural-language explanation and suggested fix.
- **`omc_find_by_signature`** + `omc_did_you_mean` — substring search over the builtin documentation surface.
- **LLM onboarding compression token** — a full-library codec dump as a single artifact, suitable for prepending to an LLM context window. See `docs/llm_onboarding.md`.

The MCP server + substrate codec are the entry points designed for LLM agents driving OMC programmatically; the heal pass and inline hints are what an LLM gets even when it's just authoring `.omc` files.

---

## Documentation map

| Doc | Subject |
|---|---|
| [`docs/jit_benchmark.md`](docs/jit_benchmark.md) | LLVM JIT measured speedups |
| [`docs/anomaly_detection.md`](docs/anomaly_detection.md) | Harmonic anomaly vs IsolationForest on real datasets |
| [`docs/heal_pass.md`](docs/heal_pass.md) | Heal classes, substrate-bucketed typo bench, per-class pragmas |
| [`docs/omc_kernel.md`](docs/omc_kernel.md) | Content-addressed code storage |
| [`docs/omc_grep.md`](docs/omc_grep.md) | Code archaeology via canonical hash |
| [`OMC-PROTOCOL.md`](OMC-PROTOCOL.md) | Substrate-signed wire format spec |
| [`omnimcode-core/src/prometheus/README.md`](omnimcode-core/src/prometheus/README.md) | Substrate-native ML framework |
| [`experiments/prometheus_parity/`](experiments/prometheus_parity/) | Substrate-attention findings (K, S-MOD, V), each with a `FINDING.md` |
| [`docs/SUBSTRATE_NATIVE_AGENT.md`](docs/SUBSTRATE_NATIVE_AGENT.md) | Two-agent demo composing every substrate primitive |
| [`CHANGELOG.md`](CHANGELOG.md) | Chapter-by-chapter project history (mirrors the release notes) |
| [`ROADMAP.md`](ROADMAP.md) | What's planned next |

---

## Reading the project's history

If you're trying to understand how OMC got here, **read the [GitHub Releases](https://github.com/RandomCoder-lab/OMC/releases) top-to-bottom**, or equivalently the [CHANGELOG](CHANGELOG.md). Each release is a chapter — `git show v0.X-name` (or click the linked Release page) gives a self-contained summary of what changed in that chapter, why it matters, and what's now possible that wasn't before.

| Tag | One-line |
|---|---|
| [V0.0.1](https://github.com/RandomCoder-lab/OMC/releases/tag/V0.0.1) | Genesis: circuit evolution engine + FFI bindings (pre-language) |
| [v0.0.2-language-core](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.2-language-core) | The language exists — parser, two-engine interpreter, HInt, self-hosting fixpoint |
| [v0.0.3-substrate-and-stdlib](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.3-substrate-and-stdlib) | Self-healing heal pass + substrate-routed search family + closures + `--check`/`--fmt` |
| [v0.0.4-jit-and-dual-band](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.4-jit-and-dual-band) | LLVM JIT, dual-band SSE2 codegen, harmony-gated branch elision |
| [v0.0.5-codec-kernel-protocol](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.5-codec-kernel-protocol) | Substrate codec, content-addressed `omc-kernel`, OMC-PROTOCOL v1 wire format |
| [v0.0.6-prometheus](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.0.6-prometheus) | Pure-OMC ML framework, multi-block transformer, first substrate-K (L1) wins |
| [v0.1-substrate-attention](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.1-substrate-attention) | Three substrate components (K, S-MOD, V) stack inside attention for −8.94% val |
| [v0.2-ergonomics](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.2-ergonomics) | OMC becomes forgiving: Python-idiom builtins, `+=`, traced errors, 11 heal classes |
| [v0.3-symbolic-prediction](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.3-symbolic-prediction) | Substrate-indexed code completion: `omc_predict_files` returns ranked provenance-tracked continuations |
| [v0.3.1-symbolic-compression](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.3.1-symbolic-compression) | `omc_predict` learns to compress: `format=hash` default is 3.8× smaller, with `omc_fetch_by_hash` for on-demand body recovery |
| [v0.4-substrate-context](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.4-substrate-context) | Symbolic compression end-to-end: `omc_compress_context` / `omc_decompress` + directory ingest + measured 2-3× LLM context-budget reduction |
| [v0.5-substrate-memory](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.5-substrate-memory) | Substrate-keyed conversation memory: `omc_memory_store` / `recall` / `list` / `stats` + filesystem persistence. **10.61× LLM context-budget reduction** on a 20-turn agent task. |
| [v0.6-fibtier-memory](https://github.com/RandomCoder-lab/OMC/releases/tag/v0.6-fibtier-memory) | Fibtier-bounded eviction for memory: cap the index at fibonacci-tier capacity (default 232); evicted entries still recoverable by hash. Memory now safe for arbitrarily long agent sessions. |

---

## Repository layout

| Path | What |
|---|---|
| `omnimcode-core/` | Parser, AST, tree-walk interpreter, bytecode VM, substrate (`phi_pi_fib`), HBit, harmonic types, ~500 builtins, substrate-routed heal pass |
| `omnimcode-codegen/` | LLVM-backed JIT, dual-band lowerer, L1.6 array bridges, harmonic-primitive intrinsics |
| `omnimcode-cli/` | Standalone binary (`omnimcode-standalone`) + `omc-bench` + `omc-grep` + `omc-kernel` |
| `omnimcode-mcp/` | MCP server exposing OMC to LLM clients |
| `omnimcode-wasm/` | WebAssembly target (no LLVM, no Python) |
| `omnimcode-lsp/` | LSP server + VS Code extension |
| `omnimcode-gdextension/` | Godot 4 GDExtension binding |
| `omnimcode-python/` | Python bindings via PyO3 |
| `experiments/prometheus_parity/` | Substrate-attention A/B harness — pure OMC vs PyTorch |
| `experiments/transformerless_lm/` | PyTorch CRT-PE vs sinusoidal training |
| `experiments/hybrid_llm/` | Per-component substrate substitution experiments |
| `experiments/substrate_primitives/` | Substrate vs native vs OMC search benchmarks |
| `examples/lib/` | `prometheus.omc`, `fibtier.omc`, `substrate.omc`, `harmonic_anomaly`, np/pd/sklearn/torch interop wrappers |
| `examples/tests/` | OMC test suite (1076 tests across 71 files) |
| `examples/datascience/` | Real-data demos: Titanic, NSL-KDD, multi-dim anomaly detection |
| `docs/` | Substrate audit, JIT benchmarks, anomaly comparisons, heal-pass docs |
| `registry/` | Central package registry (sha256-verified) |

---

## CLI reference

```bash
omnimcode-standalone FILE                 # run a program
omnimcode-standalone                      # REPL
omnimcode-standalone --init [DIR]         # scaffold a project
omnimcode-standalone --install [SPEC]     # package install
omnimcode-standalone --check FILE         # heal-pass diagnostics (no exec)
omnimcode-standalone --fmt FILE           # pretty-print canonical OMC
omnimcode-standalone --test FILE          # run fn test_*() suite
omnimcode-standalone --test-all DIR       # run every test file under DIR
omnimcode-standalone --bench FILE         # run fn bench_*() suite
omnimcode-standalone --audit FILE         # tree-walk vs VM divergence check
omnimcode-standalone --version            # version info
omnimcode-standalone --help               # all flags + env vars
```

Environment variables:

```
OMC_HBIT_JIT=1            # JIT-compile eligible user fns via omnimcode-codegen
OMC_HBIT_JIT_VERBOSE=1    # report which fns got JIT'd
OMC_HBIT_JIT_VERIFY=1     # LLVM module verification (debug)
OMC_HBIT_JIT_DUMP_IR=1    # dump LLVM IR for inspection
OMC_VM=1                  # use bytecode VM (default: tree-walk)
OMC_HEAL=1                # auto-heal AST iteratively before execution
OMC_HEAL_RETRY=1          # retry once with heal pass after a runtime error
OMC_NO_PYTHON=1           # skip embedded Python init
OMC_REGISTRY=<url>        # alternative package registry
OMC_KERNEL_ROOT=<dir>     # alternative omc-kernel storage root
```

### Package manager

```bash
omnimcode-standalone --install harmonic_anomaly                      # registry name (sha256-verified)
omnimcode-standalone --install                                       # everything in omc.toml
omnimcode-standalone --install https://example.com/raw/lib.omc       # arbitrary URL
omnimcode-standalone --list                                          # what's installed
```

`omc.toml` example:

```toml
[package]
name = "my-omc-project"
version = "0.1.0"

[dependencies]
np         = "np"
sklearn    = "sklearn"
substrate  = "substrate"
custom     = "https://example.com/raw/my_lib.omc"
```

Submit a package: PR an entry to [`registry/index.json`](registry/index.json).

---

## Status

Production-quality across the core surface:

- **Tests**: 213 Rust pass, 1073/1076 OMC end-to-end pass (3 pre-existing test_heal_pass.omc failures from `--test` bypassing heal)
- **Two-engine parity**: tree-walk and bytecode VM byte-identical, auditable via `--audit`
- **Self-hosting compiler**: `gen2 == gen3` byte-identical
- **JIT path**: 77 codegen tests pass, measured 272× factorial(12), 3.4× on real-world harmonic_anomaly (NSL-KDD 5000 rows)
- **Substrate-attention scoreboard**: three component swaps (K + S-MOD + V) stack for −8.94% val on TinyShakespeare; cross-validated in PyTorch
- **Substrate algorithms**: substrate_search wins on substrate-indexed data, ties or loses on uniform data — both code paths coexist so callers can pick
- **Anomaly detection**: wins multi-dim credential-stuffing 10/10 vs IsolationForest 7/10; loses on volumetric-dominated NSL-KDD K=500
- **Embedded CPython**: numpy/pandas/sklearn/torch all driveable from OMC
- **WASM + LSP + Godot + Python bindings**: all shipped

What's still open is documented per-chapter in the release notes (each chapter has a "what's now possible" section). The transformerless LLM as a top-to-bottom system isn't here yet — substrate-attention components win individually and stack inside attention, but a full harmonic-only architecture trained competitively at scale is the open work.

---

## Contributing

PRs welcome. Before submitting:

1. `cargo test --release` should pass cleanly (use `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` if your Python is newer than 3.13).
2. `omnimcode-standalone --test-all examples/tests/` should pass (modulo the 3 known `test_heal_pass.omc` failures).
3. For changes to the language surface, add a test in `examples/tests/` and a Rust unit test where relevant.
4. For substrate experiments, follow the `experiments/prometheus_parity/` template: `FINDING.md` describing the hypothesis, raw `results_*.json`, and a `torch_*.py` harness for cross-validation if applicable.

For substantial new features, consider whether they fit a new chapter — see [CHANGELOG.md](CHANGELOG.md) for the chapter-summary structure.

---

## License

MIT. See [LICENSE](LICENSE).

---

**Built around φ (1.6180339887…). The substrate is the architecture.**
