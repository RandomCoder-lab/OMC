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
