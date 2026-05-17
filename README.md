# OMNIcode (OMC)

**A harmonic-substrate programming language with first-class φ, dual-band execution, an LLVM-backed JIT, self-healing, and an O(log_φπfib N) algorithm family — built toward a transformerless LLM.**

OMC is not a thin layer over IEEE-754 and types. Its substrate is **φ** (the golden ratio) and the canonical 40-entry Fibonacci attractor table reaching 63,245,986. Every harmonic operation in the language — `fold(n)`, `phi.res(n)`, `harmony(x)`, `zeckendorf(n)`, `substrate_search(arr, target)`, the heal pass's literal-rewrite, the bucketing in the harmonic anomaly detector — routes through the same substrate.

It runs as one binary with two execution engines kept byte-identical, optional LLVM-18 JIT producing dual-band SSE2 code, embedded CPython for bidirectional interop, WASM and LSP targets, a self-hosting compiler that's gen2==gen3 byte-identical, a self-healing pass that fixes typos/off-attractor literals/divide-by-zero, and a registry-backed package manager.

The endpoint is a **transformerless LLM** — a model whose attention, positional encoding, and OOD gating are built from harmonic primitives instead of softmax + sinusoidal PE + L2. CRT-Fibonacci positional encoding **wins -19.9% (tiny scale) and -5.4% (TinyShakespeare scale) vs sinusoidal**. HBit cross-cutting tension is a reference-free OOD signal at AUROC 1.0. The architectural pieces are being built and measured one at a time.

---

## What's unique to OMC (nothing else has these)

These are concrete, present-in-the-code features, not aspirations:

- **The substrate is a primitive, not a library.** `HInt`, OMC's integer type, carries a φ-resonance and HIM score computed at construction. Every `Value::HInt(_)` ever created has been routed through `compute_resonance` and `nearest_attractor_with_dist`. This is at the type level, not at the user-code level.

- **Dual-band executable code.** OMC values have a classical α-band and a harmonic shadow β-band, packed into LLVM `<2 x i64>` SSE2 vectors inside JIT'd functions. `phi_shadow(x)` makes β diverge; `harmony(x)` reads the substrate-routed coherence between the bands. **Branch elision based on harmony** is shipped: high-coherence inputs skip entire conditional blocks at native code speed.

- **O(log_phi_pi_fibonacci N) algorithm family.** The `phi_pi_fib_search_v2` algorithm uses F(k)/φ^(π·k) split-points — each iteration shrinks the live range by **φ^π ≈ 4.534**, not 2. The substrate-canonical iteration bound is `log_phi_pi_fibonacci(n) ≈ 0.459 · log₂ n`. Exposed as a complete primitive family: `substrate_search`, `substrate_lower_bound`, `substrate_upper_bound`, `substrate_rank`, `substrate_count_range`, `substrate_slice_range`, `substrate_intersect`, `substrate_difference`, `substrate_insert`, `substrate_quantile`, `substrate_select_k`, `substrate_nearest`, `substrate_min_distance`, `substrate_hash`.

- **Zeckendorf as first-class integer encoding.** Every positive integer has a unique sum of non-consecutive Fibonaccis (Zeckendorf 1972). OMC exposes the canonical encoder/decoder: `zeckendorf(n) -> [indices]`, `from_zeckendorf(idxs) -> n`, plus `zeckendorf_weight`, `zeckendorf_bit`, `is_zeckendorf_valid`, and `substrate_hash` (Zeckendorf-mixed avalanche). The iteration count is bounded by `log_phi_pi_fibonacci(n)`.

- **CRT-Fibonacci positional encoding wins on a real LM training task.** Pairs of `(sin(2π·pos%m_i/m_i), cos(2π·pos%m_i/m_i))` with Fibonacci moduli `{5, 8, 13, 21, ...}`. Validation loss **−19.9% at toy scale (4/5 seeds) and −5.4% at TinyShakespeare scale (3/3 seeds)** vs Vaswani sinusoidal. See [`experiments/transformerless_lm/README.md`](experiments/transformerless_lm/README.md) for the full numbers.

- **Self-healing compiler.** A 5-class heal pass runs at the AST level: typo correction (Levenshtein over the symbol table), off-attractor literal snap, divide-by-zero rescue, arity correction, parser-error recovery. Enabled with `OMC_HEAL=1`.

- **Two-engine byte-identical parity.** A tree-walking interpreter AND a bytecode VM, kept lockstep. 44 of 45 functional examples produce byte-identical output between engines (the diverger is a timing-only benchmark). Verified by `--audit FILE`.

- **Self-hosting compiler V.9b.** Compiles itself; **gen2 == gen3 byte-identical**. See `examples/self_hosting_v9b.omc`.

- **`@harmony` and `@predict` JIT pragmas.** Mark a function or branch as harmony-eligible; the JIT compounds these with `@hbit` for layered speedups (270× alone; 95% additional branch reduction with `@harmony` + `@predict`).

- **Substrate-routed harmonic libraries.** `harmonic_anomaly` beats scikit-learn's IsolationForest **10/10 vs 7/10** on multi-dim credential-stuffing detection (the structural-anomaly regime).

- **`omc-grep`: alpha-rename-invariant duplicate finder.** A standalone CLI ([`docs/omc_grep.md`](docs/omc_grep.md)) that walks a tree, extracts every top-level fn, canonicalizes, and clusters by canonical hash. `--body-only` mode strips the fn signature so duplicates with *different names* surface — something text-grep, ast-grep, and tree-sitter queries can't do. On OMC's own examples tree (151 files / 2388 fns): **31.7%** redundancy with name-sensitive hashing, **33.0%** with body-only — surfacing renamed-but-identical fns like `_bucket_discrete` ≡ `endpoint_bucket` ≡ `status_bucket` that share no token in their names.

- **`omc-kernel`: content-addressed code DAG keyed by canonical hash.** A persistence layer for distributed-agent code exchange ([`docs/omc_kernel.md`](docs/omc_kernel.md)). Every fn gets stored at `~/.omc/kernel/store/<hex_hash>.omc`. `omc-kernel sign FILE` emits a substrate-signed wire message; `omc-kernel verify` (stdin) recovers the original canonical form from the store — proven end-to-end with alpha-rename: sender's `fn commit(handle)` recovers as store's `fn commit(conn)`. Code becomes a content-addressed Merkle DAG over substrate addresses; version it the way IPFS versions files, except the addressing is semantic.

- **Substrate-keyed code codec + compressed substrate-signed messaging.** `omc_codec_encode` produces a sampled-token payload addressed by the canonical AST hash (invariant under whitespace, comments, alpha-rename). `omc_codec_decode_lookup` returns the exact library entry on hash match. `omc_msg_sign_compressed` / `omc_msg_recover_compressed` carry the codec payload inside the substrate-signed wire format with lossless library recovery and full signature integrity. **Wire-byte sizing is honest**: token-count compression is ~N×, but wire-byte savings only appear at payloads ≳500 B with N≥8 (single-message). The always-on value is **library-lookup recovery** — alpha-rename invariant content addressing on the receiver, no shared key. 13 tests pass ([`test_codec.omc`](examples/tests/test_codec.omc), [`test_compressed_messaging.omc`](examples/tests/test_compressed_messaging.omc)). See [`experiments/seed_expansion/FINDINGS.md`](experiments/seed_expansion/FINDINGS.md).

---

## 30-second hello

```bash
git clone https://github.com/RandomCoder-lab/OMC.git
cd OMC
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release -p omnimcode-cli
./target/release/omnimcode-standalone --init
./target/release/omnimcode-standalone main.omc
```

For the JIT path (LLVM-backed, dual-band native code):

```bash
sudo apt install llvm-18-dev libpolly-18-dev libzstd-dev
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 \
    cargo build --release -p omnimcode-cli --features llvm-jit
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 OMC_HBIT_JIT=1 OMC_HBIT_JIT_VERBOSE=1 \
    ./target/release/omnimcode-standalone main.omc
```

Eligible user fns get compiled to dual-band native code. 272× over tree-walk, 119× over the bytecode VM on factorial(12); see [`docs/jit_benchmark.md`](docs/jit_benchmark.md).

---

## The architecture, bottom-up

OMC ships one binary. Six layers, each validated:

### 1. The substrate (`omnimcode-core/src/phi_pi_fib.rs`)

A 40-entry FIBONACCI table reaching 63,245,986 and a single canonical search algorithm:

- `nearest_attractor_with_dist(value)` — closest Fibonacci attractor + distance, `#[inline]`
- `phi_pi_fib_search_v2` — F(k)/φ^(π·k) split-point search with binary-search fallback when offset rounds to zero
- `log_phi_pi_fibonacci(n) = ln(n) / (π · ln(φ))` — the substrate's iteration-count bound
- `zeckendorf_indices` / `from_zeckendorf_indices` — canonical sparse decomposition
- `substrate_search_i64` / `substrate_lower_bound` / `substrate_upper_bound` — O(log_φπfib N) primitives backing the OMC builtins
- `attractor_bucket` — FIBONACCI-table index of nearest attractor; used for substrate-aligned hash bucketing

Every harmonic operation in the language routes through these. 16 of 17 duplicate Fibonacci arrays were deleted across the codebase; the substrate is single-source. See [`SUBSTRATE_CHANGES.md`](SUBSTRATE_CHANGES.md) for the audit.

### 2. HBit dual-band

Values carry α (classical) and β (harmonic shadow). Inside JIT'd code, they're packed into `<2 x i64>` LLVM vectors so harmonic ops execute as SIMD on both lanes.

- `phi_shadow(x)` — makes β diverge from α
- `harmony(x)` — substrate-routed coherence: `1 / (1 + attractor_distance(|α-β|))`
- `@hbit`, `@harmony`, `@predict` function-level pragmas
- Branch elision on harmony: 95.2% reduction on high-harmony inputs with 5–8% break-even fraction

### 3. The LLVM-backed JIT (`omnimcode-codegen`)

- LLVM 18 via inkwell, feature-gated as `llvm-jit`
- **77 codegen tests pass**: locals via allocas, CFG branches, loops, recursion, comparisons, floats, arrays (read + write), cross-fn calls, **L1.6 array bridges (both directions)**, **22 harmonic-primitive intrinsics**
- Dual-band lowerer produces packed `<2 x i64>` for `phi_shadow` and `harmony`
- Cascade-cleanup: failed-to-lower fns get `unreachable` trap stubs (not raw deletion), plus a fixpoint marks dependent fns as failed
- `OMC_HBIT_JIT_VERIFY=1`, `OMC_HBIT_JIT_DUMP_IR=1` for diagnostics
- Empirical: 272× on factorial(12), **115× on array-sum hot loop, 10.6× on substrate-heavy mixed workload**

**L1.6 Array↔JIT bridge** (both directions): `Value::Array(int_only)` marshals to a length-prefixed `Box<[i64]>` for arg-passing; the `@jit_returns_array_int` pragma triggers `omc_arr_heapify` so a JIT'd fn can return a `Value::Array` it built internally. Same layout both ways, no codegen changes needed at the lowerer.

**JIT'd harmonic primitives** (table-driven `HARMONIC_INTRINSICS` in `dual_band.rs`, 3 lines per new entry):
| Arity | Primitives |
|---|---|
| `i64 → i64` | `nth_fibonacci`, `is_attractor`, `attractor_distance`, `hbit_tension`, `fibonacci_index`, `attractor_bucket`, `substrate_hash`, `zeckendorf_weight`, `bit_count`, `bit_length`, `digit_sum`, `digit_count`, `harmonic_align`, `harmonic_unalign` |
| `i64, i64 → i64` | `gcd`, `lcm`, `safe_mod` |
| `i64, i64, i64 → i64` | `mod_pow` |
| `array_ptr → i64` | `arr_sum_int`, `arr_product`, `arr_min_int`, `arr_max_int` |
| `array_ptr, i64 → i64` | `int_binary_search`, `int_lower_bound`, `substrate_search` |

### 4. The O(log_phi_pi_fibonacci N) primitive family

A first-class API surface (50+ builtins this session alone):

| Substrate-routed (probe count: log_φπfib N) | Native baseline (probe count: log₂ N) |
|---|---|
| `substrate_search(arr, t)` | `int_binary_search(arr, t)` |
| `substrate_lower_bound(arr, t)` | `int_lower_bound(arr, t)` |
| `substrate_upper_bound(arr, t)` | `int_upper_bound(arr, t)` |
| `substrate_rank`, `substrate_count_range`, `substrate_slice_range` | — |
| `substrate_intersect`, `substrate_difference` | `sorted_merge`, `sorted_union`, `sorted_dedupe` |
| `substrate_insert`, `substrate_quantile`, `substrate_select_k` | — |
| `substrate_nearest`, `substrate_min_distance` | — |
| `substrate_hash`, `attractor_bucket` | `fnv1a_hash` |

Plus Zeckendorf encoding (`zeckendorf`, `from_zeckendorf`, `zeckendorf_weight`, `zeckendorf_bit`, `is_zeckendorf_valid`), substrate analytics (`harmonic_align`, `harmonic_unalign`, `harmonic_score`, `harmonic_resample`, `resonance_band_histogram`, `is_phi_resonant`, `phi_pi_log_distance`), and phi primitives (`phi_pow`, `phi_pi_pow`, `nth_fibonacci`, `attractor_table`, `fib_chunks`).

The architectural trade-off: substrate ops use fewer probes (7.3 vs 16 at N=65536) but each probe pays for F(k)/φ^(π·k) floating-point math. Native int-binary wins on raw throughput against uniform data; substrate ops win when probe sequence coherence matters (substrate-indexed data, attractor-aligned queries). Both paths coexist so callers pick. See [`experiments/substrate_primitives/bench_substrate_search.omc`](experiments/substrate_primitives/bench_substrate_search.omc).

### 5. The harmonic libraries (`examples/lib/`)

Substrate-routed end-to-end. `harmonic_anomaly` (+ **v2 with substrate-routed lookup**), `harmonic_clustering`, `harmonic_recommend`. Plus the high-level [`examples/lib/substrate.omc`](examples/lib/substrate.omc) wrapper exposing `s_*` (substrate-routed), `i_*` (int-binary), `h_*` (harmonic) naming.

Anomaly detection vs scikit-learn IsolationForest (full results in [`docs/anomaly_detection.md`](docs/anomaly_detection.md)):

| Workload | OMC harmonic | IsolationForest |
|---|:---:|:---:|
| **Multi-dim credential stuffing, K=10** | **10/10** | 7/10 |
| Multi-dim K=25 | **24/25** | 17/25 |
| Multi-dim K=50 | **49/50** | 40/50 |

OMC loses on volumetric-dominated data (NSL-KDD K=500: 302 vs 351). Ties on simple time-series. The pattern: harmonic substrate is a **structural detector**, not a primary computation replacement.

**JIT integration impact on NSL-KDD harmonic_anomaly fit (5000 rows, 6 dims):**

| Configuration | fit + score | Speedup vs tree-walk |
|---|--:|--:|
| Tree-walk | 363 ms | 1× |
| JIT pre-L1.6 (arrays in dispatch → tree-walk) | 363 ms | 1× (no JIT actually used) |
| JIT + L1.6 input bridge | 191 ms | 1.9× |
| JIT + L1.6 + harmonic-primitive intrinsics | 107 ms | 3.4× |
| **JIT + L1.6 + intrinsics + harmonic_anomaly_v2 (substrate_search)** | **271 ms total fit+score** | **substrate-routed lookup keeps recall byte-identical** ([`nsl_kdd_v1_vs_v2.omc`](examples/datascience/nsl_kdd_v1_vs_v2.omc)) |

### 6. Infrastructure

- Self-hosting compiler V.9b (gen2 == gen3 byte-identical)
- **Self-healing pass — 7 classes** of automatic correction (typo, arity-pad, arity-truncate, div-zero → safe_divide, mod-zero → safe_mod, harmonic-index snap, missing-return). **Substrate-routed typo lookup** uses 32-bucket `substrate_hash_name` index for ~10× speedup on projects with hundreds of names ([`docs/heal_pass.md`](docs/heal_pass.md) has the bench table). Per-class disable pragmas (`@no_heal_typo`, etc.) + per-pass heal budget.
- Two-engine parity verified by `--audit FILE`
- Embedded CPython via PyO3: `py_import`, `py_call`, `py_callback("omc_fn")` for callbacks
- WASM target (`omnimcode-wasm`, no LLVM/Python deps)
- LSP server (`omnimcode-lsp`) + VS Code extension
- Package manager (`--install` from registry, sha256-verified, or arbitrary URL)
- **161 OMC tests + 77 codegen tests + cargo unit tests** — all green

---

## The transformerless LLM thesis (live, empirically driven)

A modern transformer has four primitives. The hybrid LLM experiments measure each against a harmonic alternative:

| Transformer piece | Harmonic alternative | Empirical status |
|---|---|---|
| Sinusoidal PE | **CRT-Fibonacci PE** (pairwise-coprime moduli {5, 8, 13, 21, ...}) | **Harmonic wins:** −19.9% loss (tiny), **−5.4% on TinyShakespeare (3/3 seeds)** |
| Softmax attention | OmniWeight (`φ^(-|q-k|)`) | Softmax wins on perturbed-query recovery |
| Softmax-only attention | **Hybrid:** softmax × HBit-tension gate | **Harmonic wins on adversarial mixes** (experiment 12) |
| L2-NN OOD detection | **HBit cross-cutting tension** | **Harmonic wins:** AUROC 1.0 on scenario A |

CRT-PE is the first per-component substitution that beats the transformer baseline on a real LM training task, at two orders of magnitude in both model and data scale. The transformerless thesis is now testing whether the same substitution holds at modern transformer scale.

See [`experiments/hybrid_llm/README.md`](experiments/hybrid_llm/README.md) for the per-experiment record and [`experiments/transformerless_lm/README.md`](experiments/transformerless_lm/README.md) for the end-to-end LM results.

---

## A flavor of the language

```omc
# φ-resonance is built into the integer type. No imports needed.
h x = 89;                           # FIBONACCI[11], on-attractor
println(phi.res(x));                # high resonance

# Substrate-routed O(log_phi_pi_fibonacci N) search:
h sorted = arr_range(0, 1000000);
println(substrate_search(sorted, 524287));  # 524287

# Zeckendorf decomposition: every int is a unique sum of non-consecutive Fibonaccis
h z = zeckendorf(100);              # [11, 6, 4]  (89 + 8 + 3)
println(from_zeckendorf(z));        # 100  (round-trip)

# Substrate-coherence diagnostic — how Fibonacci-aligned is this data?
println(harmonic_score([89, 1, 1, 2, 3, 5, 8]));  # 1.0
println(harmonic_score([100, 7, 42, 99]));        # 0.0

# Self-healing: typo + off-attractor literal both auto-corrected by `--check`
fn near_attractor(x) {
    return fold(x);                 # snap to nearest substrate attractor
}

# Dual-band JIT: harmony(x) reads coherence at native code speed
@harmony
fn coherent_loop(n) {
    h i = 0;
    while i < n {
        if harmony(i) > 0.5 {       # branch eliminable on high-harmony inputs
            i = i + 1;
        } else {
            i = i + 2;
        }
    }
    return i;
}
```

---

## What's NOT shipped (honest limits)

- **The transformerless LLM itself.** CRT-PE wins at the per-component level; the hybrid attention gate as currently formulated lost the distractor-mix test 0/3 ([`experiments/transformerless_lm/distractor_mix_README.md`](experiments/transformerless_lm/distractor_mix_README.md)). Two concrete follow-on architectures documented (score-level gate, learned-threshold gate). Building a harmonic-only architecture top-to-bottom and training it competitively is the next step.
- **AVX-512 widening.** Dual-band uses `<2 x i64>` (SSE2). Wider lanes need array-processing OMC fns to fill them.
- **JIT for float-returning harmonic primitives.** `harmony_value` / `value_danger` shims exist as extern Rust fns; the dispatch boundary needs a `returns_float` flag mirroring `returns_array_int` to materialize their f64-bit-pattern returns correctly. Nothing in current hot paths needs it.
- **JIT for string/dict ops.** Pure JIT operates on i64 only; strings and dicts stay tree-walk by design. The harmonic libraries' L1 rewrite to array-of-hashed-int eliminated this constraint for the hot path (now 3.4× faster end-to-end).

---

## Demos worth running

| File | Story |
|---|---|
| [`experiments/transformerless_lm/`](experiments/transformerless_lm/) | CRT-PE wins on real LM training, two scales, 3-of-3 + 4-of-5 seeds |
| [`experiments/transformerless_lm/distractor_mix_README.md`](experiments/transformerless_lm/distractor_mix_README.md) | Adversarial-mix scaling test: CRT-PE generalizes, hybrid gate falsified |
| [`experiments/hybrid_llm/experiment_5_hbit_combined.omc`](experiments/hybrid_llm/experiment_5_hbit_combined.omc) | HBit cross-cutting tension as reference-free OOD: AUROC 1.0 |
| [`experiments/hybrid_llm/experiment_12_hybrid_attention.omc`](experiments/hybrid_llm/experiment_12_hybrid_attention.omc) | Hybrid softmax × HBit-gate attention beats softmax on adversarial mixes |
| [`experiments/substrate_primitives/bench_substrate_search.omc`](experiments/substrate_primitives/bench_substrate_search.omc) | 4-way bench: linear vs OMC binary vs substrate vs native int-binary |
| [`examples/datascience/nsl_kdd_v1_vs_v2.omc`](examples/datascience/nsl_kdd_v1_vs_v2.omc) | A/B: harmonic_anomaly v1 (linear) vs v2 (substrate_search) — 10.3% speedup, identical recall |
| [`examples/datascience/multidim_anomaly.omc`](examples/datascience/multidim_anomaly.omc) | Credential stuffing: harmonic 10/10 vs IsolationForest 7/10 @ K=10 |
| [`examples/datascience/nsl_kdd_validation.omc`](examples/datascience/nsl_kdd_validation.omc) | NSL-KDD intrusion detection — honest mixed result, JIT 3.4× via L1.6 + intrinsics |
| [`examples/self_hosting_v9b.omc`](examples/self_hosting_v9b.omc) | Self-hosting compiler, gen2 == gen3 byte-identical |
| [`examples/lisp.omc`](examples/lisp.omc) | Mini Scheme interpreter in OMC |
| [`examples/datascience/titanic.omc`](examples/datascience/titanic.omc) | Kaggle Titanic via embedded Python pipeline |
| [`examples/lib/substrate.omc`](examples/lib/substrate.omc) | High-level wrappers around the substrate primitives |
| [`examples/tests/test_heal_pass.omc`](examples/tests/test_heal_pass.omc) | 16 tests for the self-healing compiler's heal classes + per-class pragmas |
| [`examples/tests/test_codec.omc`](examples/tests/test_codec.omc) | 7 tests for `omc_codec_encode/decode_lookup` — alpha-rename invariant library recovery + inline error-hint UX check |
| [`examples/tests/test_compressed_messaging.omc`](examples/tests/test_compressed_messaging.omc) | 6 tests: substrate-signed wire payloads carrying codec output, alpha-equivalent recovery, JSON round-trip |
| [`examples/tests/test_codec_registry.omc`](examples/tests/test_codec_registry.omc) | 3 tests: `omc_registry_codec_library` + `omc_msg_recover_from_registry` graceful no-op when omc_modules/ doesn't exist |
| [`examples/demos/llm_tandem_registry.omc`](examples/demos/llm_tandem_registry.omc) | End-to-end: synthetic package in omc_modules/, signs renamed copy, recovers via registry — alpha-rename invariant |
| [`experiments/seed_expansion/FINDINGS.md`](experiments/seed_expansion/FINDINGS.md) | Empirical writeup: substrate-keyed codec works (lossless on in-library content); open-set ML stays data-budget bound at 40 samples — honest |

---

## Repo layout

| Path | What |
|---|---|
| `omnimcode-core/` | Parser, AST, interpreter, bytecode VM, substrate (`phi_pi_fib`), HBit, harmonic types, 50+ substrate builtins, substrate-routed heal pass |
| `omnimcode-codegen/` | LLVM-backed JIT, dual-band lowerer, L1.6 array bridges, 22 harmonic-primitive intrinsics (table-driven) |
| `omnimcode-cli/` | Standalone binary (`omnimcode-standalone`) + `omc-bench` + `omc-grep` + `omc-kernel` |
| `omnimcode-wasm/` | WebAssembly target (no LLVM, no Python) |
| `omnimcode-lsp/` | LSP server for editor integration |
| `omnimcode-gdextension/` | Godot 4 GDExtension binding |
| `experiments/transformerless_lm/` | PyTorch end-to-end CRT-PE vs sinusoidal training, two scales |
| `experiments/hybrid_llm/` | 12 pure-OMC per-component substitution experiments |
| `experiments/substrate_primitives/` | Empirical comparison of substrate vs native vs OMC search |
| `examples/lib/` | `substrate.omc`, `harmonic_anomaly`, `harmonic_clustering`, `harmonic_recommend`, np/pd/sklearn/torch/requests/sqlite |
| `examples/datascience/` | Real-data demos with honest numbers |
| `examples/tests/` | `test_substrate_primitives.omc` (57), `test_new_builtins.omc` (70), `test_harmonic_libs.omc` (18), `test_heal_pass.omc` (16), `test_codec.omc` (7), `test_compressed_messaging.omc` (6), `test_codec_registry.omc` (3) — **177 total** |
| `docs/` | Substrate audit, JIT benchmarks, anomaly-detection comparisons |
| `registry/` | Central package registry (sha256-verified) |

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

OMC_HBIT_JIT=1            # JIT-compile eligible user fns via omnimcode-codegen
OMC_HBIT_JIT_VERBOSE=1    # report which fns got JIT'd
OMC_HBIT_JIT_VERIFY=1     # LLVM module verification (debug)
OMC_HBIT_JIT_DUMP_IR=1    # dump LLVM IR for inspection
OMC_VM=1                  # use bytecode VM (default: tree-walk)
OMC_HEAL=1                # auto-heal AST iteratively
OMC_HEAL_RETRY=1          # retry after runtime errors
OMC_NO_PYTHON=1           # skip embedded Python init
OMC_REGISTRY=<url>        # alternative package registry
```

---

## Package manager

```bash
omnimcode-standalone --install harmonic_anomaly      # registry name (sha256-verified)
omnimcode-standalone --install                       # everything in omc.toml
omnimcode-standalone --install https://example.com/raw/lib.omc   # explicit URL
omnimcode-standalone --list                          # what's installed
```

`omc.toml` example:

```toml
[package]
name = "my-omc-project"
version = "0.1.0"

[dependencies]
np      = "np"
sklearn = "sklearn"
substrate = "substrate"
custom  = "https://example.com/raw/my_lib.omc"
```

Submit a package: PR an entry to [`registry/index.json`](registry/index.json).

---

## Status

| Layer | Status |
|---|---|
| Substrate (`log_phi_pi_fibonacci` everywhere) | shipped, audited |
| O(log_phi_pi_fibonacci N) primitive family | shipped, 50+ builtins, 57 tests |
| HBit dual-band executable | shipped (`OMC_HBIT_JIT=1`) |
| LLVM JIT for pure-int/array/float | shipped, **77 codegen tests**, 272× factorial(12), **3.4× harmonic_anomaly NSL-KDD** |
| **L1.6 Array↔JIT bridges (both directions)** | **shipped, 11 codegen tests; 115× synthetic, 1.9× real-world harmonic_anomaly** |
| **22 harmonic-primitive JIT intrinsics** | **shipped, table-driven, 10.6× substrate-heavy hot loop** |
| Zeckendorf encoding + substrate hash | shipped |
| **harmonic_anomaly v2 (substrate_search lookup)** | **shipped, 10.3% speedup, byte-identical recall to v1** |
| Harmonic libraries on real data | shipped, mixed-honest results |
| Hybrid LLM experiments (12 experiments) | shipped, 1 perfect AUROC, 1 architectural negative, 1 CRT-PE win |
| End-to-end transformerless LM (PyTorch) | CRT-PE wins -19.9% (tiny), **-5.4% (TinyShakespeare, 3/3 seeds), -2.9% (distractor mix, 3/3)** |
| Hybrid HBit-gate distractor-mix test | **falsified across THREE gate formulations** (0/3 wins each, +3–4% consistent loss): KEY-magnitude gate, SCORE-level gate, LEARNED-threshold gate. The architectural pivot per [`GATE_REFORMULATION_RESULTS.md`](experiments/transformerless_lm/GATE_REFORMULATION_RESULTS.md): substrate's home is positional + distributional, not as an attention-score shaper. |
| **Geodesic attention bias (substrate on positions, not activations)** | **WINS 3/3 seeds, −0.4% vs crt_only.** ALiBi-style additive bias `−α · geodesic(i, j)` using CRT-Fibonacci moduli. First attention-side substrate validation. Rule derived: *substrate metric applies to integer quantities only*. See [`GEODESIC_RESULT.md`](experiments/transformerless_lm/GEODESIC_RESULT.md). |
| **Prometheus: substrate-native ML framework** | **MVP shipped + 4 substrate-moat features verified** ([docs](omnimcode-core/src/prometheus/README.md)) — pure-OMC training (no PyTorch in the loop), content-addressed checkpoints, geodesic bias primitive, **harmonic SGD WINS 3/3 seeds at -13.2% vs vanilla SGD on tinyLM**, canonical-hash inference cache surviving model reload. |
| **Parameter-free substrate attention WINS 3/3 (−21.5%)** | Four-way A/B: standard QKV → substrate-K → substrate-K+Q → fully substrate. Monotonic improvement at every step *down* the substrate ladder; the variant with ZERO learnable attention params (CRT-PE as K and Q, identity V) beats standard learned attention by 21.5% on 3/3 seeds. See [`SUBSTRATE_ATTENTION_4WAY.md`](experiments/prometheus_parity/SUBSTRATE_ATTENTION_4WAY.md). |
| **Substrate-K attention WINS −8% val on TinyShakespeare** | The architectural sweet spot. K = CRT-Fibonacci positional table (no learnable K); Q and V stay learned. On 1.1MB corpus with 90/10 train/val split: L1 val=0.104 vs L0 val=0.113, **−8.0% with ~9% fewer params**. Fully-substrate (L3) catastrophically fails at scale; L1 is the architectural recommendation. See [`SUBSTRATE_K_FINDING.md`](experiments/prometheus_parity/SUBSTRATE_K_FINDING.md). |
| Self-hosting compiler V.9b | shipped, gen2 == gen3 byte-identical |
| **Self-healing pass (7 classes, substrate-routed typo)** | shipped, `OMC_HEAL=1`, **10× typo lookup**, 16 tests, per-class pragmas |
| **Substrate-keyed code codec + compressed messaging** | **shipped**, `omc_codec_encode/decode_lookup` + `omc_msg_sign_compressed/recover`, alpha-rename invariant, token-count ~N× (wire-byte breaks even at ≥500 B + N≥8); always-on win is library-lookup recovery; 13 tests, lossless on in-library content |
| **Inline error-fix hints** | **shipped**, `Undefined function` errors now carry the suggested fn's signature inline (eliminates a separate `omc_help` round-trip after a typo) |
| **`omc-grep`: alpha-rename-invariant code archaeology** | **shipped** ([docs/omc_grep.md](docs/omc_grep.md)) — standalone CLI; on OMC's examples: 31.7% redundancy (name-sensitive), 33.0% (body-only); surfaces renamed-but-identical fns that text-grep and ast-grep can't catch |
| **`omc-kernel`: content-addressed code DAG** | **shipped** ([docs/omc_kernel.md](docs/omc_kernel.md)) — store at ~/.omc/kernel/store/<hash>.omc; alpha-rename invariant sign/verify proven end-to-end; the persistence layer for the codec wire format |
| Two-engine parity (tree-walk + VM) | shipped, 44/45 byte-identical |
| Embedded CPython + callbacks | shipped, 6 wrapper libs |
| WASM + LSP + GDExtension targets | shipped |
| Package manager + registry | shipped |
| Per-component harmonic wins on PE | **shipped (CRT-PE), validated under noise** |
| Per-component harmonic wins on attention | hybrid only — clean softmax still wins; gate reformulation pending |
| End-to-end transformerless model | not yet — building blocks validated, integration is the open work |

Build dependencies for the JIT path: `llvm-18-dev`, `libpolly-18-dev`, `libzstd-dev`. For the no-JIT build, just Rust + (optionally) Python 3.

---

License: MIT.

**Built around φ (1.6180339887…). The substrate is the architecture.** The transformerless LLM is what the substrate is for.
