# OMNIcode

**A harmonic-substrate language built toward a transformerless LLM.** The same φ-math primitives that drive the anomaly detector and the dual-band JIT are the architectural pieces meant to compose into a model.

OMNIcode (OMC) is a small standalone runtime — one binary, two execution engines, optional LLVM-backed JIT — whose substrate is φ (the golden ratio) and the canonical Fibonacci attractor lattice rather than IEEE-754 + types. Harmonic operations are first-class language primitives: `fold(n)` snaps to the nearest attractor, `phi.res(n)` reads resonance, `harmony(x)` measures dual-band coherence inside JIT'd code.

The endpoint is a **transformerless LLM**: a model whose attention, positional encoding, OOD gating, and (where the experiments support it) computation itself are built from harmonic primitives instead of softmax + sinusoidal PE + MLPs. This isn't shipped — most of the substrate work is what makes it eventually possible. The current state is a research artifact with empirical pieces validated.

## Where the architecture stands today

OMC ships as one binary with these layers, ordered from foundation up:

1. **The substrate** — a single 40-entry Fibonacci attractor table reaching 63,245,986 routed through `phi_pi_fib::nearest_attractor_with_dist`. Every harmonic op (`fold`, `phi.res`, HInt resonance, the heal pass's literal-rewrite, the harmonic libraries' bucketing) goes through this. Substrate-correct from the language up.
2. **HBit dual-band** — values carry an α-band (classical value) and a β-band (harmonic shadow). Wired into the JIT as `<2 x i64>` packed lanes; `phi_shadow(x)` makes β diverge; `harmony(x)` reads the substrate-routed coherence. Branch elision in JIT'd code based on harmony is shipped.
3. **The JIT** — LLVM-18 backed, dual-band native code path. Pure-int / array / float OMC fns JIT through a 41-test pipeline at 250–1000× over tree-walk. `OMC_HBIT_JIT=1` enables it from the CLI.
4. **The harmonic libraries** — `harmonic_anomaly`, `harmonic_clustering`, `harmonic_recommend`. Beat IsolationForest 10/10 vs 7/10 on multi-dim credential stuffing (the structural-anomaly regime), tie or lose on volumetric data. Substrate-routed end-to-end.
5. **The hybrid LLM experiments** — 10 experiments in `experiments/hybrid_llm/` measuring where harmonic primitives win and lose vs transformer components. Headline: HBit cross-cutting tension is a reference-free OOD signal at AUROC 1.0; compression-gate models are 34× smaller than equivalent dense tables. Architecture step toward transformerless.
6. **The infrastructure** — self-hosting compiler with self-healing, package manager + registry, embedded CPython for hybrid workflows, two-engine parity, LSP, WASM. The plumbing that makes the substrate usable.

The work is an upward stack. The transformerless LLM goal sits at the top; everything below it is either substrate, executable form of substrate, or evidence that substrate operations have measurable utility.

## The transformerless LLM thesis (status: in progress)

A modern transformer has four components. The hybrid LLM experiments (`experiments/hybrid_llm/`) measured each against a harmonic alternative:

| Transformer piece | Harmonic alternative | Empirical status |
|---|---|---|
| Sinusoidal positional encoding | Multi-channel φ-fold PE | **Sinusoidal wins** at length distinctness past L≥16 (exp 3) |
| Softmax attention scoring | OmniWeight (`φ^(-|q-k|)`) | **Softmax wins** on perturbed-query recovery (exp 1) |
| Layer-norm + residual | `phi.fold(blend)` | Validated in `phi_field_llm_multilayer.omc` (no head-to-head bench yet) |
| L2-NN OOD detection | HBit cross-cutting tension | **Harmonic wins** AUROC 1.0 on scenario A; combined gate beats every individual gate on B (exp 5) |

Cumulative read from the experiments: the harmonic substrate is a **structural detector** that wins on OOD / off-manifold / structural-rarity signals, but loses to softmax + sinusoidal + L2 as drop-in replacements for the transformer's primary computation paths.

The transformerless thesis therefore needs harmonic primitives that **don't exist yet** in the experiments — primitives that win at the per-component level on real sequence tasks, not just as auxiliary detectors. The work below is what builds toward those primitives.

See [`experiments/hybrid_llm/README.md`](experiments/hybrid_llm/README.md) for the full empirical record.

---

## 30-second hello

```bash
git clone https://github.com/RandomCoder-lab/OMC.git
cd OMC
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release -p omnimcode-cli
./target/release/omnimcode-standalone --init
./target/release/omnimcode-standalone main.omc
```

`--init` creates `omc.toml` + a hello-world `main.omc`. Edit, run, you're going.

For the JIT path:

```bash
# Requires: sudo apt install llvm-18-dev libpolly-18-dev libzstd-dev
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 \
    cargo build --release -p omnimcode-cli --features llvm-jit
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 OMC_HBIT_JIT=1 OMC_HBIT_JIT_VERBOSE=1 \
    ./target/release/omnimcode-standalone main.omc
```

Eligible user fns get compiled to dual-band native code (`<2 x i64>` SSE2). 250–1000× speedup on pure-int hot paths; see `docs/jit_benchmark.md` for the numbers and break-even analysis.

---

## What's shipped, with proof

### Substrate
- `log_phi_pi_fibonacci(n)` and the canonical 40-entry FIBONACCI table reaching 63M
- Single-source attractor lookup via `phi_pi_fib::nearest_attractor_with_dist`
- 16 of 17 duplicate Fibonacci arrays deleted across the codebase (last one is `harmonic_split` chunk sizing — different semantic)
- All harmonic ops route through the canonical substrate. See [`SUBSTRATE_CHANGES.md`](SUBSTRATE_CHANGES.md) for the audit and migration log.

### HBit dual-band
- α/β representation as packed `<2 x i64>` LLVM vectors
- `phi_shadow(x)` and `harmony(x)` builtins, executable in JIT'd code
- Substrate-routed harmony formula: `1 / (1 + attractor_distance(|α-β|))`
- Branch elision based on harmony (the `@predict` mechanism), benched at 95.2% reduction on high-harmony inputs with 5–8% break-even fraction

### LLVM-backed JIT
- 41 codegen tests, all passing
- Cross-fn calls, recursion, locals, branches, loops, arrays (read + write), floats, comparisons
- `OMC_HBIT_JIT=1` env var on the CLI
- Bench: factorial(12) microbench shows 272× over tree-walk, 119× over the bytecode VM
- Honest gap: harmonic libraries as currently written use dicts and string concat that the JIT doesn't yet cover. See [`docs/jit_real_world.md`](docs/jit_real_world.md) for the empirical limit.

### Harmonic libraries (proof the substrate is useful for ML)

Beat scikit-learn's IsolationForest decisively on multi-dim structural anomaly detection:

| Workload | OMC harmonic | IsolationForest |
|---|:---:|:---:|
| **Multi-dim credential stuffing, K=10** | **10/10** | 7/10 |
| Multi-dim K=25 | **24/25** | 17/25 |
| Multi-dim K=50 | **49/50** | 40/50 |

Lose on volumetric-dominated data (NSL-KDD K=500: harmonic 302 vs IF 351). Tie on simple time-series (NAB 7/19 both). Full results in [`docs/anomaly_detection.md`](docs/anomaly_detection.md). The architectural finding from the LLM experiments — "harmonic is a structural detector, not a primary computation" — was first concretely demonstrated by these libraries on real data.

### Hybrid LLM experiments

10 experiments in [`experiments/hybrid_llm/`](experiments/hybrid_llm/) covering positional encoding, attention scoring, OOD gating, compression, and substrate search algorithms. Pure-OMC (no torch dependency) so they run inside the standalone binary. Full empirical record in the experiments README.

### Self-hosting + heal pass
- Self-hosting compiler V.9b — `gen2 == gen3` byte-identical
- 5-class self-healing pass (typo fix, off-attractor literal snap, divide-by-zero rescue, arity correction, parser recovery)
- Two-engine parity: 44 of 45 functional examples byte-identical between tree-walk and bytecode VM (the diverger is a benchmark file with timing-only output)

### Embedded CPython + package manager
- `py_import("numpy")`, `py_call`, `py_callback("omc_fn")` — full bidirectional bridge
- `omc --install <name>` from registry (sha256-verified) or URL
- 6 wrapper libraries: np, pd, sklearn, requests, sqlite, torch
- Lets the substrate work coexist with a real LLM today (use torch for the heavy lifting, replace pieces with harmonic primitives as the experiments validate them)

---

## What's NOT shipped (honest limits)

- **The transformerless LLM itself.** The architecture's top layer is research-in-progress. The experiments measure where harmonic primitives win or lose against transformer components in isolation; nobody has trained a harmonic-only model end-to-end yet.
- **Per-component harmonic wins on the primary computation path.** The current experiments showed harmonic substitutions for PE, attention, and OOD detection. PE and attention lose to their transformer baselines. OOD detection wins decisively (HBit AUROC 1.0). Building a transformerless LLM needs better-than-baseline harmonic alternatives for the computation paths, which the experiments haven't found yet.
- **JIT for the harmonic libraries.** The JIT works for pure-int/array/float fns. The harmonic libraries use dicts and string-keyed frequency tables that the JIT can't compile. Either extend codegen with dict + string support (~2-3 sessions), or rewrite the libs to use array-of-hashed-int (~half a session). See [`docs/jit_real_world.md`](docs/jit_real_world.md).
- **AVX-512 widening.** Dual-band uses `<2 x i64>` (SSE2). Wider lanes need array-processing OMC fns to actually fill them.
- **Float-typed Div / comparison.** OMC bytecode compiler doesn't yet emit `DivFloat` / `EqFloat`; plain `Op::Div` is treated as integer division on float bit-patterns. Compiler-side fix.

---

## Demos worth running

| File | Story |
|---|---|
| [`experiments/hybrid_llm/experiment_5_hbit_combined.omc`](experiments/hybrid_llm/experiment_5_hbit_combined.omc) | HBit cross-cutting tension as reference-free OOD: AUROC 1.0 |
| [`experiments/hybrid_llm/experiment_6_compression_gate.omc`](experiments/hybrid_llm/experiment_6_compression_gate.omc) | Compression-gate model: 34× smaller than dense; tolerates all 12 library deletions |
| [`examples/datascience/multidim_anomaly.omc`](examples/datascience/multidim_anomaly.omc) | Credential-stuffing detection: harmonic 10/10 vs IF 7/10 @ K=10 |
| [`examples/datascience/nsl_kdd_validation.omc`](examples/datascience/nsl_kdd_validation.omc) | Real network-intrusion data — honest mixed result |
| [`examples/self_hosting_v9b.omc`](examples/self_hosting_v9b.omc) | Self-hosting compiler, gen2 == gen3 byte-identical |
| [`examples/lisp.omc`](examples/lisp.omc) | Mini Scheme interpreter in OMC |
| [`examples/datascience/titanic.omc`](examples/datascience/titanic.omc) | Kaggle Titanic via embedded Python pipeline |

---

## Repo layout

| Path | What |
|---|---|
| `omnimcode-core/` | The language: parser, AST, interpreter, bytecode VM, substrate (`phi_pi_fib`), HBit, harmonic types |
| `omnimcode-codegen/` | LLVM-backed JIT, dual-band lowerer, intrinsics for `phi_shadow` / `harmony` |
| `omnimcode-cli/` | The standalone binary (`omnimcode-standalone`); also `omc-bench` for benchmarking |
| `omnimcode-wasm/` | WebAssembly target (no LLVM, no Python) |
| `omnimcode-lsp/` | LSP server for editor integration |
| `experiments/hybrid_llm/` | Empirical LLM-component substitution experiments |
| `examples/lib/` | Substrate-aligned libraries (harmonic_anomaly, harmonic_clustering, etc.) + Python wrappers (np, pd, sklearn, …) |
| `examples/datascience/` | Real-data demos with honest numbers |
| `docs/` | Substrate audit (`SUBSTRATE_CHANGES.md`), JIT benchmarks, anomaly-detection comparisons |
| `registry/` | Central package registry (sha256-verified) |

---

## Package manager

```bash
omnimcode-standalone --install harmonic_anomaly      # registry name (verified)
omnimcode-standalone --install                       # everything in omc.toml
omnimcode-standalone --install https://example.com/raw/lib.omc   # explicit URL
omnimcode-standalone --list                          # what's installed
```

Manifest:

```toml
[package]
name = "my-omc-project"
version = "0.1.0"

[dependencies]
np      = "np"
sklearn = "sklearn"
custom  = "https://example.com/raw/my_lib.omc"
```

Submit a package: PR an entry to [`registry/index.json`](registry/index.json).

---

## Quick reference

```bash
omnimcode-standalone FILE                 # run a program
omnimcode-standalone                      # REPL
omnimcode-standalone --init               # scaffold project
omnimcode-standalone --install [SPEC]     # package install
omnimcode-standalone --check FILE         # heal-pass diagnostics
omnimcode-standalone --fmt FILE           # pretty-print
omnimcode-standalone --test FILE          # run fn test_*() suite
omnimcode-standalone --bench FILE         # run fn bench_*() suite
omnimcode-standalone --audit FILE         # tree-walk vs VM divergence check
omnimcode-standalone --help               # all flags + env vars

OMC_HBIT_JIT=1         # JIT-compile eligible user fns through omnimcode-codegen
OMC_HBIT_JIT_VERBOSE=1 # report which fns got JIT'd
OMC_VM=1               # use bytecode VM
OMC_HEAL=1             # auto-heal AST iteratively
OMC_HEAL_RETRY=1       # retry after runtime errors
OMC_NO_PYTHON=1        # skip embedded Python init
OMC_REGISTRY=<url>     # alternative package registry
```

---

## Status

OMC is a **research artifact built toward a transformerless LLM**, with each piece below the LLM goal validated empirically:

| Layer | Status |
|---|---|
| Substrate (`log_phi_pi_fibonacci` everywhere) | shipped, audited |
| HBit dual-band executable | shipped (`OMC_HBIT_JIT=1`) |
| LLVM JIT for pure-int/array/float | shipped, 41 tests, 272× microbench |
| Harmonic libraries on real data | shipped, mixed-honest-results |
| Hybrid LLM experiments (detector/gate scope) | 10 experiments, 1 perfect AUROC, 2 negative findings, honest record |
| Per-component harmonic wins (PE / attention) | not yet — the experiments showed where simple substitutions lose |
| End-to-end transformerless LLM | not yet — the goal the rest of the work is building toward |

Build dependencies for the JIT path: `llvm-18-dev`, `libpolly-18-dev`, `libzstd-dev`. For the no-JIT build, just Rust + (optionally) Python 3.

---

License: MIT.

**Built around φ (1.618…). The substrate is the architecture.** The transformerless LLM is what the substrate is for.
