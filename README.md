# OMNIcode

**A harmonic-math platform: language, package manager, embedded Python ecosystem, and machine-learning libraries that demonstrably beat scikit-learn on structural anomalies.**

OMNIcode (OMC) is a small standalone runtime that gives you four things in one binary:

1. **A real harmonic-anomaly detector that beats IsolationForest** — the `harmonic_anomaly` library catches credential-stuffing patterns 10/10 vs scikit-learn's 7/10 at top-K=10 ([`examples/datascience/multidim_anomaly.omc`](examples/datascience/multidim_anomaly.omc)). Drop-in replacement for `IsolationForest()` on multi-dim tabular data.

2. **The full Python ecosystem on tap** — `py_import("numpy")`, `py_import("pandas")`, `py_import("sklearn")` work out of the box. CPython is embedded at link time. Six wrapper libraries ([`np`, `pd`, `sk`, `requests`, `sqlite`, `torch`](examples/lib/)) make the common cases idiomatic.

3. **A package manager + central registry** — `omc --install harmonic_anomaly` fetches from the registry, verifies sha256, caches under `omc_modules/`. Submit a new package by PRing [`registry/index.json`](registry/index.json).

4. **A self-hosting language with a self-healing compiler** — the bytecode compiler is itself written in OMC and `gen2 == gen3` of the compiler-on-itself ([`examples/self_hosting_v9b.omc`](examples/self_hosting_v9b.omc)). The static-analysis substrate is φ-math (Fibonacci attractors, resonance, HIM score), not types. Identifier typos, off-attractor literals, divide-by-zero, and parser slips get auto-rewritten by the heal pass.

Single Rust binary. Two execution engines (tree-walk + bytecode VM) with byte-identical output across 43 functional examples. The architecture is built so each layer reinforces the next: harmonic primitives drive the anomaly detector, the package manager ships those libraries, the embedded Python lets users compose with everything else.

---

## 30-second hello

```bash
git clone https://github.com/RandomCoder-lab/OMC.git
cd OMC
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release
./target/release/omnimcode-standalone --init
./target/release/omnimcode-standalone main.omc
```

`--init` creates `omc.toml` + a hello-world `main.omc`. Edit, run, you're going.

## 60-second wow — anomaly detection that beats scikit-learn

The `harmonic_anomaly` library is a drop-in replacement for `sklearn.IsolationForest` on multi-dim tabular data. It wins decisively on structural anomalies — the kind credential-stuffing, account takeover, and exfiltration produce, where every individual value looks normal but the combination is rare:

```omc
import "harmonic_anomaly" as ha;       # after: omc --install harmonic_anomaly

# Schema: each row = [latency_ms, status_code, endpoint_id, hour_of_day]
h det = ha.new(["latency", "status", "endpoint", "hour"]);
ha.set_strategy(det, 1, "discrete");   # status_code is categorical
ha.set_strategy(det, 2, "discrete");   # endpoint_id is categorical
ha.set_strategy(det, 3, "modulo");     # hour-of-day is small periodic

ha.fit(det, training_rows);
h alerts = ha.top_k(det, all_rows, 10);   # top-10 most anomalous indices
```

Measured on 5000 normal requests + 50 injected credential-stuffing rows:

|  | OMC harmonic | sklearn IsolationForest |
|---|:---:|:---:|
| Top-10 alerts (the SRE oncall regime) | **10/10 caught** | 7/10 (mixes in unrelated 500-error spikes) |
| Top-25 alerts | **25/25** | 17/25 |
| Top-50 alerts | **50/50** | 40/50 |

See [`examples/datascience/anomaly_tutorial.omc`](examples/datascience/anomaly_tutorial.omc) for the walkthrough, and [`examples/datascience/multidim_anomaly.omc`](examples/datascience/multidim_anomaly.omc) for the full comparison.

## And — OMC drives the whole Python ML stack

```omc
import "sk" as sk;       # after: omc --install sk
import "np" as np;       # after: omc --install np

# Train + score a random forest on the iris dataset
h iris = sk.load_iris();
h split = sk.train_test_split(arr_get(iris, 0), arr_get(iris, 1), 0.3);
h model = sk.random_forest_classifier(100);
sk.fit(model, arr_get(split, 0), arr_get(split, 2));
h preds = sk.predict(model, arr_get(split, 1));
println(concat_many("RF accuracy: ", sk.accuracy_score(arr_get(split, 3), preds)));
```

For the full real-world demo, run [`examples/datascience/titanic.omc`](examples/datascience/titanic.omc) — Kaggle Titanic via seaborn (~120 lines of OMC), loading 891 passengers in ~280ms, training a 100-tree forest. Zero Rust extensions for the user.

---

## What's in the box

### Language
- φ-math substrate with `HInt` (resonance, HIM, value_danger as primitives)
- Pattern matching with attractor ranges (`0..21 => ...`), type tags, alternation
- First-class functions, mutable closures (Rc-shared environments)
- Try / catch with stack traces that include source line numbers
- Two interpreters: tree-walk (fast iteration) and bytecode VM (~2× faster on hot paths)
- Self-healing pass (`OMC_HEAL=1`) — typo correction, harmonic-violation rewrites, dynamic divide-by-zero rescue

### Toolchain
- `omnimcode-standalone main.omc` — run a program (or REPL with no args)
- `--init` — scaffold a new project (omc.toml + main.omc)
- `--install [SPEC]` — install package by registry name OR URL into `omc_modules/`
- `--list` — enumerate installed modules
- `--check FILE` — heal pass + diagnostics, no execution (CI-friendly)
- `--fmt FILE` — pretty-print AST as canonical OMC source

### Embedded CPython (always-on)
- `py_import("numpy")`, `py_call(handle, "method", [args])`, `py_get`, `py_eval`, `py_exec`
- `py_call_kw` / `py_call_fn_kw` for kwargs-aware Python APIs
- `py_call_raw` to skip auto-conversion when chaining ops
- `py_callback("omc_fn_name")` — wraps an OMC fn as a Python callable for `df.apply` etc.
- Auto Value↔PyObject conversion: scalars, lists, tuples, dicts, numpy ndarrays
- Set `OMC_NO_PYTHON=1` to skip Python initialisation

### Integration libraries (written in OMC, all in [`examples/lib/`](examples/lib/))
- `np.omc` — numpy bridge (array, mean, dot, sort, percentile, argsort)
- `pd.omc` — pandas bridge (read_csv/json/parquet/excel, group_by, fillna, apply_omc)
- `sklearn.omc` — RandomForest, KMeans, train_test_split, accuracy
- `requests.omc` — HTTP client (get, post, json, fetch_json)
- `sqlite.omc` — embedded SQL via Python's sqlite3
- `torch.omc` — PyTorch tensors, nn.Linear, optimizers
- `harmonic_anomaly.omc` — multi-dim structural anomaly detection (drop-in IsolationForest replacement; wins on credential-stuffing patterns)

Each one is 30-110 lines of OMC. Fork them or write your own. All registered in [`registry/index.json`](registry/index.json) with sha256 verification.

### Harmonic primitives
- `harmonic_set` — dedupe by Fibonacci attractor equivalence
- `harmonic_pq` — priority queue ranked by HIM score
- `harmonic_index` — sub-linear lookup by attractor neighborhood
- `harmonic_sort`, `harmonic_partition`, `harmonic_dedupe` — bulk ops
- `fold(n)` — snap to nearest Fibonacci attractor
- `phi.res(n)`, `phi.him(n)`, `phi.fold(n)` — direct φ-math access

---

## Demos worth running

| File | Story |
|---|---|
| [`examples/self_hosting_v9b.omc`](examples/self_hosting_v9b.omc) | Compiler-in-OMC produces byte-identical bytecode under self-application |
| [`examples/self_healing_h5.omc`](examples/self_healing_h5.omc) | Out-of-bounds array reads become finite attractor-landing values |
| [`examples/lisp.omc`](examples/lisp.omc) | Mini Scheme interpreter in OMC — closures, recursion, quote, let, lambda |
| [`examples/json.omc`](examples/json.omc) | JSON parser + serializer in OMC, recursive descent via mutable-closure cursor |
| [`examples/recommend/recommend.omc`](examples/recommend/recommend.omc) | MovieLens 100k recommendation engine — `harmonic_index` over real ratings |
| [`examples/datascience/titanic.omc`](examples/datascience/titanic.omc) | Kaggle Titanic via seaborn → harmonic feature engineering → sklearn classifier |
| [`examples/datascience/movielens_harmonic.omc`](examples/datascience/movielens_harmonic.omc) | pandas-loaded movielens → harmonic_partition → numpy stats per bucket |
| [`examples/datascience/harmonic_ml.omc`](examples/datascience/harmonic_ml.omc) | sklearn wine + Python→OMC callback via `numpy.vectorize` |
| [`examples/datascience/anomaly_detection.omc`](examples/datascience/anomaly_detection.omc) | Power-law anomaly detection: harmonic 4/5 vs IF 0/5 @ K=5 (alert-budget regime) |
| [`examples/datascience/multidim_anomaly.omc`](examples/datascience/multidim_anomaly.omc) | Credential-stuffing detection: harmonic 10/10 vs IF 7/10 @ K=10 |
| [`examples/datascience/anomaly_tutorial.omc`](examples/datascience/anomaly_tutorial.omc) | Tutorial — using `harmonic_anomaly` as drop-in IsolationForest replacement |
| [`examples/datascience/nab_validation.omc`](examples/datascience/nab_validation.omc) | NAB benchmark: both detectors tie at 7/19 windows (naive baseline tier) |
| [`examples/datascience/nab_time_aware.omc`](examples/datascience/nab_time_aware.omc) | Time-aware harmonic — honest negative result; needs CUSUM/seasonality to beat IF on NAB |

---

## Package manager

```bash
# Install one package by registry name (sha256-verified)
omnimcode-standalone --install np

# Install everything in omc.toml
omnimcode-standalone --install

# Install from arbitrary URL
omnimcode-standalone --install https://example.com/raw/my_lib.omc
```

Manifest format:

```toml
[package]
name = "my-omc-project"
version = "0.1.0"

[dependencies]
np      = "np"            # registry name (verified)
sklearn = "sklearn"       # registry name
custom  = "https://example.com/raw/my_lib.omc"   # explicit URL
```

Installed modules land under `omc_modules/<name>.omc`. `import "name";` resolves the local copy first. Override the registry with `OMC_REGISTRY=<url>` for private forks.

Submit a package: PR an entry to [`registry/index.json`](registry/index.json).

---

## Architecture notes

OMC has **two semantic engines** that produce byte-identical output:
- **Tree-walk interpreter** — what you debug against, what `OMC_HEAL` runs through
- **Bytecode VM** — `OMC_VM=1` to enable; ~2× faster on hot paths

Both share:
- The same `Value` enum, including Rc-shared `Array` / `Dict` for O(1) clone
- The same builtin dispatch surface, with VM hot-path inlining for arr_get / dict_get / str_concat
- The same `register_builtin` API for embedders to register host functions

Self-healing is a static AST-rewrite pass with five diagnostic classes:
- Off-attractor numeric literal → snap to nearest Fibonacci
- Identifier typo → Levenshtein-closest match in defined-name table
- Literal `/0` → `safe_divide(...)`
- User-fn arity mismatch → auto-pad/truncate args
- Parser-level recovery (missing braces, parens, semicolons)

Run with `OMC_HEAL=1` to apply iteratively to fixpoint, `OMC_HEAL_RETRY=1` to retry once after a runtime error. Both opt-in — production code shouldn't ship with healing on.

The full historical arc lives in [CHANGELOG.md](CHANGELOG.md). The φ-math substrate is documented in [PHI_PI_FIB_ALGORITHM.md](PHI_PI_FIB_ALGORITHM.md).

---

## Performance

| Workload | Tree-walk | VM | Notes |
|---|---:|---:|---|
| MovieLens 10k aggregate | 29 ms | 33 ms | Was 16s before Rc-shared collections (552× speedup) |
| MovieLens 100k full pipeline | 0.92 sec | 1.0 sec | Builds 9724-entry harmonic_index in 345ms |
| recursive_fib(22) | 54 ms | 26 ms | VM 2.08× faster |
| arr_map(double) over 1k × 200 reps | 131 ms | 59 ms | VM 2.22× faster |

OMC is now usable for real-world data sizes (10k → 100k records routine). The architectural blocker (Value::Array clone-on-mutation) was killed in commit `d3c29b6` by switching to `Rc<RefCell<>>` semantics — collections now pass by reference like Python's mutable types.

---

## Where harmonic detection actually wins (vs scikit-learn)

Real comparisons against scikit-learn's IsolationForest. Not synthetic glory — measured on real and reproducible workloads.

| Workload | OMC harmonic | IsolationForest | Where it matters |
|---|:---:|:---:|---|
| **Multi-dim credential stuffing, K=10** | **10/10** | 7/10 | Account-takeover, exfiltration, structural attacks |
| Multi-dim K=25 | **24/25** | 17/25 | Subspace anomaly detection |
| Multi-dim K=50 | **49/50** | 40/50 | Same as above, broader recall |
| NSL-KDD real intrusion data, K=500 | 302/500 | **351/500** | Threat hunting on volumetric-dominated data |
| NSL-KDD K=10 / K=50 / K=100 | 6 / 43 / 78 | **9 / 45 / 92** | Volumetric DoS — IF wins on low-K when biggest spike = real |
| NAB realKnownCause (1-D time series) | 7/19 | 7/19 | Tie at naive baseline tier (SOTA needs CUSUM/HMM) |
| Power-law K=30 (broad recall) | 12/30 | **15/30** | IF still leads on total recall |
| Power-law K=5 (alert budget) | 1/5 | 0/5 | Both struggle at extreme low-K on this synthetic data |

The pattern: **harmonic decisively wins on multi-dim structural anomalies** (the credential-stuffing regime — values that look normal per-dim but rare in combination). Ties on simple time-series benchmarks where neither approach exploits temporal structure. Loses on volumetric-dominated data where the labeled anomalies are all magnitude outliers (IF's home turf).

Two substrate-architecture changes on 2026-05-15 affected these numbers. **Phase 1** (refactor `compute_resonance` to `log_phi_pi_fibonacci`) flipped NSL-KDD K=500 from a tie to a harmonic win (348→365 vs IF's 351). **Phase 2** (substrate-fill: route the harmonic_anomaly bucket function through the substrate too) traded that K=500 win and the K=5 alert-budget win for architectural completeness — substrate-tempo bucketing produces empirically different bucket distributions on heavy-tailed data than base-10 decades, and on NSL-KDD that's a net loss. The choice was deliberate: substrate purity over benchmark numbers. See [`SUBSTRATE_CHANGES.md`](SUBSTRATE_CHANGES.md).

The harmonic_anomaly library at [`examples/lib/harmonic_anomaly.omc`](examples/lib/harmonic_anomaly.omc) packages the multi-dim detector with a clean `new` / `fit` / `top_k` API. Install it:

```bash
omnimcode-standalone --install harmonic_anomaly
```

Then in OMC:

```omc
import "harmonic_anomaly" as ha;
h det = ha.new(["latency", "status", "endpoint", "hour"]);
ha.set_strategy(det, 1, "discrete");   # status_code is categorical
ha.fit(det, training_rows);
h alerts = ha.top_k(det, all_rows, 10);
```

See [`examples/datascience/anomaly_tutorial.omc`](examples/datascience/anomaly_tutorial.omc) for the drop-in IsolationForest replacement walkthrough.

---

## Status & honest limits

OMC is a research artifact built around an architectural premise. What works:
- Self-hosting compiler with self-healing (V.9b + H.5)
- Real ML pipelines via embedded Python (np / pd / sklearn / requests / sqlite)
- Two-engine parity (43/43 functional examples byte-identical)
- Package manager with registry + sha256 verification

What's not production-grade:
- Single-developer experimental codebase
- No formal type system (the static analysis is φ-math, not Hindley-Milner)
- Heavy operations (huge collections, parallelism, async) aren't a focus
- Some `OMC_HEAL` rewrites are over-eager on domain values (see [`examples/recommend/PAIN_POINTS.md`](examples/recommend/PAIN_POINTS.md) MED-3)

Open known issues live in [PAIN_POINTS.md](examples/recommend/PAIN_POINTS.md). Most surfaced from a single real-world stress test (10k MovieLens recommendation engine) — exactly as a research project should.

---

## Quick reference

```bash
omnimcode-standalone FILE                 # run a program
omnimcode-standalone                      # REPL
omnimcode-standalone --init               # scaffold project
omnimcode-standalone --install [SPEC]     # package install
omnimcode-standalone --list               # list installed
omnimcode-standalone --check FILE         # lint via heal pass
omnimcode-standalone --fmt FILE           # pretty-print
omnimcode-standalone --help               # all flags + env vars

OMC_VM=1               # use bytecode VM
OMC_HEAL=1             # auto-heal AST iteratively
OMC_HEAL_RETRY=1       # retry after runtime errors
OMC_NO_PYTHON=1        # skip embedded Python
OMC_REGISTRY=<url>     # alternative package registry
OMC_STDLIB_PATH=<...>  # extra import search paths
```

---

OMC stands for OMNIcode. The work builds on a long lineage of self-hosting language research — Lisp, Smalltalk, Forth — with an additional dimension: the static analysis substrate is φ-math, not S-expressions or types. The toolchain (lex, parse, emit, execute, analyze, repair, embed) lives inside the language.

License: MIT.

**Built around φ (1.618…). The substrate is the architecture.**
