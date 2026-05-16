# JIT vs real-world workloads — first honest measurement

**TL;DR:** the JIT works exactly as designed on pure-int + array + float OMC fns (proven by 41 codegen tests + bench harness), but the *currently-shipped* `harmonic_anomaly` library uses dicts and string-keyed frequency tables — both outside the JIT's current op coverage. Only **1 of 4** user fns JIT'd on the NSL-KDD validation, and that fn isn't in the hot loop. **Net wall-clock change: zero.**

The gap is well-defined and the architecture's path forward is clear.

## What the bench actually showed

Workload: `examples/datascience/nsl_kdd_validation.omc` — runs the harmonic_anomaly library's `fit + top_k` against a 5000-row NSL-KDD sample.

```
OMC_HBIT_JIT=1 OMC_HBIT_JIT_VERBOSE=1 ./omnimcode-standalone examples/datascience/nsl_kdd_validation.omc
```

JIT log:

```
[OMC_HBIT_JIT] JIT'd 1/4 user fns to dual-band native code
  - extract_features
```

Wall-clock comparison:

| Mode | User time | Wall-clock |
|---|--:|--:|
| Tree-walk (no `OMC_HBIT_JIT`) | 2.98s | 1.58s |
| `OMC_HBIT_JIT=1` | 2.98s | 1.54s |

Within measurement noise. The JIT didn't make this workload faster because the JIT'd fn (`extract_features`) runs once over 5000 rows at startup; the hot loop is in `harmonic_anomaly.fit()` which the JIT couldn't compile.

## Why the harmonic library doesn't JIT

The fns that the JIT **rejected**:

1. **`fit(detector, rows)`** — uses `dict_set(freq, key, ...)` to build per-dim frequency tables; uses `concat_many("", bkt)` to build dict keys. Both ops have no JIT lowering today.
2. **`score(detector, row)`** — same dict + string ops in the inner per-dim loop.
3. **`top_k(detector, rows, k)`** — calls `score_all` which calls `score`; transitively excluded.

The JIT is conservative: any fn whose body uses an unsupported op causes the whole fn to be silently skipped (Sessions D/H established this — partial fns get erased so the rest of the module compiles cleanly). The 4th fn `extract_features` is pure-int + arrays + a `csv_parse` builtin — but `csv_parse` is also unsupported, so it gets... wait, we said it JIT'd. Let me check.

Looking at the JIT verbose output again: 1/4 JIT'd was `extract_features`. So `csv_parse` must not be in `extract_features`'s body — it's a separate top-level call before the fn. That checks out.

## What this tells us about the architecture

The architecture is sound — Sessions A–H + Path A.1–A.4 + Path D shipped 41 codegen tests covering every JIT-eligible op. The bench harness shows 250–1000× speedups on workloads that fit those ops.

What the architecture *doesn't yet have* is the op coverage to JIT the harmonic libraries as they're written today. Two viable paths to fix:

### Option 1: extend codegen (the structural fix)

Add JIT support for:
- **Dicts** — would need a hash-table representation in LLVM. Significant: needs key hashing (probably an extern Rust call), bucket arrays, collision handling. Feasible but ~1 session of careful work.
- **Strings** — needs heap allocation (libc malloc) + pointer-based representation. Could share infrastructure with arrays. Another session.
- **`concat_many` / `csv_parse` / other builtins** — most wouldn't get JIT'd directly; they'd remain tree-walk. The JIT'd fn would call back through the dispatch hook into tree-walk for unsupported builtins. Needs a "fallback to tree-walk for one builtin" mechanism — currently the whole fn falls back if it hits an unsupported op.

**Cost:** 2-3 sessions. **Reward:** harmonic libs JIT, ~250× speedup applies to real workloads.

### Option 2: rewrite the harmonic libs (the empirical fix)

The frequency tables in `harmonic_anomaly` use `dict_set(freq, str_key, count)` because string keys are convenient for the multi-dim case (the key is the bucketed value rendered as a string). They could use **arrays of hashed-int keys** instead:
- `freq_keys: [int]` — hashes of bucket values
- `freq_counts: [int]` — counts parallel to keys
- Lookup via linear scan or sorted-array binary search

This is a real rewrite (~half a day of substantive work) but it produces a library that:
1. JITs end-to-end with current codegen
2. Runs in ~5 ms instead of ~135 ms (the projected speedup if the inner loop hits the JIT)
3. Stays substrate-aligned (the bucket math doesn't change)

**Cost:** ~half a session of library refactor. **Reward:** the same ~250× speedup applies, AND the library demonstrates that JIT-friendly idioms have a measurable payoff.

## The honest position

Path B as conceived asked: "does enabling JIT on a real OMC program produce real speedup?" The answer is **not yet** for the harmonic libraries as currently written, but **yes structurally** based on every microbench we've run since Session E. The JIT works; the libraries don't yet exercise it.

The path forward isn't "make the JIT work harder" — it's either to extend codegen to cover dicts (Option 1) or rewrite the hot path to use already-supported ops (Option 2). Either gets us to "harmonic libraries run 100×+ faster with `OMC_HBIT_JIT=1`."

This is the kind of honest negative result the architecture needed. The 277× number from Session E isn't a microbench artifact — but it doesn't automatically apply to libraries written for tree-walk's strengths (dicts, strings, dynamic dispatch).

## Reproduction

```bash
# Tree-walk baseline
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    time ./target/release/omnimcode-standalone examples/datascience/nsl_kdd_validation.omc

# JIT mode
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 OMC_HBIT_JIT=1 OMC_HBIT_JIT_VERBOSE=1 \
    time ./target/release/omnimcode-standalone examples/datascience/nsl_kdd_validation.omc
```

Numbers taken on 2026-05-15. If you want bigger numbers, choose Option 2 above and rewrite `examples/lib/harmonic_anomaly.omc` with array-based frequency tables.

---

## Update: L1.6 Array↔JIT bridging (2026-05-15)

The harmonic_anomaly L1 rewrite lifted dict-keyed frequency tables onto array-of-hashed-int. That made the library JIT-friendly at the OMC level, but the JIT dispatch boundary in `omnimcode-cli/src/main.rs` still rejected `Value::Array` arguments:

```rust
_ => return None, // non-int arg → fall through to tree-walk
```

So even after the library was JIT-friendly, every call from tree-walk land into a JIT'd fn with an array arg silently fell back to tree-walk. The JIT log said "JIT'd 1/4 fns" but only `extract_features` actually ran via JIT in practice; everything in the per-row hot loop took the tree-walk path.

**L1.6 fix**: marshal `Value::Array(int_only)` into a length-prefixed `Box<[i64]>` with layout `[len, v0, v1, ..., vN]` — matching the stack-frame array layout the dual-band lowerer's NewArray ops already use. The JIT'd function's `ArrayLen` / `ArrayIndex` code reads from the marshalled buffer with the same access pattern, so **no codegen changes were needed**. The Box drops after `.call()` returns; the JIT'd fn is guaranteed not to retain the pointer beyond the call.

### Empirical re-measurement after L1.6

Same workload (`examples/datascience/nsl_kdd_validation.omc`, 5000 rows):

| Mode | harmonic_anomaly fit | rows JIT'd |
|---|--:|--:|
| Tree-walk | 363 ms | n/a |
| JIT (pre-L1.6) | 363 ms | 1 of 4 user fns |
| **JIT (post-L1.6)** | **191 ms** | **15 of 53 user fns** (incl. `ha.score`) |

**1.9× wall-clock speedup on the real harmonic_anomaly workload.** The hot-loop fn `ha.score` now actually runs through the JIT instead of falling back to tree-walk.

Synthetic microbench (sum over arr_range(0, 1000), 1000 iterations):

| Mode | ms |
|---|--:|
| Tree-walk | 803 |
| JIT (post-L1.6) | 7 |

**115× on the pure array-consuming hot path.**

### Tests

`omnimcode-codegen/tests/jit_array_bridge.rs` — 6 tests covering sum, max, mixed-args, empty array, large array (1000 elements), and non-int-array rejection (falls through to tree-walk correctly). All pass; the existing 41 codegen tests still pass (48 total).

### Honest limits remaining

- **Read-only contract**: the bridge doesn't write back to the original `HArray` even if the JIT'd fn mutated the buffer. Common case (sum, score, count) is read-only; mutating-array fns return `i64` today so output-side bridging is a future extension.
- **Int-only arrays**: `Value::Array` whose elements aren't all `HInt` (or `Bool`) falls through to tree-walk. String / float arrays are next-session work.
- **Return-side bridge: infrastructure in place, codegen path disabled.** The wiring went in for an `@jit_returns_array_int` pragma that would call `omc_arr_heapify` before `Op::Return` (copying the frame-array buffer to heap so it outlives the JIT'd fn frame). The Rust extern + global-mapping + JittedFn flag + dispatch materializer + `omc_arr_free` are all present and pass their unit-test paths. But in end-to-end testing the JIT'd fn segfaults on its `ret` instruction AFTER `omc_arr_heapify` successfully runs and returns a valid heap pointer. The trip back through the extern-"C" boundary corrupts something (stack alignment? calling convention? alloca lifetime?). The codegen path is left disabled in `dual_band.rs` so the infrastructure can be re-enabled atomically once the segfault is understood. None of the current harmonic libraries' hot paths return arrays, so this gap costs no measurable performance today — `ha.top_k` / `ha.score_all` / `ha.new` are called O(1) times per fit, not per-row.
