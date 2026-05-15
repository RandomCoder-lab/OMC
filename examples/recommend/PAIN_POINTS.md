# OMC Pain Points — Real-World Stress Test (MovieLens 10k)

Live-captured findings from building a real movie recommendation
engine over MovieLens latest-small (10k rating subset of 100k full).
Each entry: severity, evidence, root cause, suggested fix.

Severity scale:
* **CRIT** — Wrong-output bug. Silent. Will bite users.
* **HIGH** — Performance cliff that prevents real-world use.
* **MED** — Ergonomic friction that adds 2× to development time.
* **LOW** — Polish; cosmetic or minor.

Sorted approximately by impact.

---

## CRIT-1: Float truncation in arr_get / dict_get arithmetic ✅ FIXED

**Symptom:** `arr_get(cur, 1) + rating` where `rating` is a float
silently truncates the result to int. Aggregating ratings 4.5 + 3.5
+ 5.0 returned 12 (or 13 depending on engine), not 13.0.

**Root cause:** The compiler's static return-type table at
`compiler.rs:140-141` claimed `arr_get / dict_get / arr_min / arr_max
/ arr_sum` always return int. They're polymorphic over element type.
The lie made the compiler emit `Op::AddInt` (typed fast-path), which
calls `.to_int()` on both operands → silent float→int truncation.

**Engine divergence:** Tree-walk and VM produced *different* wrong
answers (16 vs 12, 523 vs 782 hits) because tree-walk's eval_expr
uses runtime types and VM's bytecode uses compile-time types. Made
it look like a "VM bug" when both engines were affected.

**Fix:** Removed polymorphic builtins from the int-return table.
Commit `d792672`.

**Lesson:** Static type inference for collection accessors is unsound
unless the type system tracks element types — which OMC's doesn't.
Default to "no inference" for any builtin whose return depends on
input data; only inline fast-paths for builtins with truly fixed
return types (arr_len, str_len, fibonacci, etc).

---

## HIGH-1: Value::Dict / Value::Array clone on every read+write

**Symptom:** Aggregating 10k records into a dict that grows to 3218
entries takes 16 seconds. 100k records: hung — never completed in
several minutes. Same pattern affects `arr_push` on growing arrays
(0.4s for 10k integer pushes — should be ~10ms).

**Evidence (10k):**
```
load_csv:        9899 ms
aggregate:       16018 ms     ← THIS
agg_to_rows:     2125 ms
build_hidx:      4883 ms      ← AND THIS
linear scan:     1185 ms
```

**Root cause:** `Value::Dict(BTreeMap<...>)` and `Value::Array(...)`
both `derive(Clone)`. Every Op::DictSetNamed / arr_push / dict_get
invokes vm_get_var → vm_assign_var, each of which clones the entire
backing collection. For a dict growing to N entries, each iteration
of the loop costs O(N) — the whole thing is O(N²).

For 10k records building a 3k-entry dict:
- ~20k clones during the loop
- Avg clone size ~1.5k entries
- ~30M element copies total → ~16 seconds

**Suggested fix (architectural):** Wrap collections in
`Rc<RefCell<...>>`. clone becomes O(1) (Rc bump). Mutation through
vm_assign_var becomes a borrow_mut() into the shared backing.
Semantic implication: dicts/arrays become *shared by reference* like
closure environments, not pass-by-value. This matches Python's
reference semantics for dict/list and unblocks any algorithm that
builds collections in a loop.

**Cheaper interim fix:** Add a builtin-fused `dict_update(d, k,
fn)` that does in-place modify (one clone, not two), and a similar
`arr_extend` that does bulk-push. Speeds the common patterns ~2× but
doesn't escape the O(N²).

**Lesson:** "Pass-by-value" was a defensible early choice (matches
arrays' existing semantics). At 10k+ scale it stops working. The
architecture needs to choose: pass-by-value (O(N²) collections) or
pass-by-reference (sharing, mutation surprises). Document the
tradeoff and pick.

---

## HIGH-2: str_split per-line cost dominates CSV parsing

**Symptom:** `load_csv` is 10s for 10k lines; 6.4s of that is
calling `str_split(line, ",")` 10k times.

**Root cause:** Each `str_split` call goes through vm_call_builtin
(or vm_fast_dispatch) and allocates a fresh `Vec<Value::String>`.
40k Value::String allocations + 40k Value pushes through the VM
stack. The actual `s.split(",")` is fast; the wrapping overhead
isn't.

**Suggested fix:** Add a `csv_parse(text)` builtin that does the
entire parse in one call — returns `Array<Array<String>>` directly.
Eliminates 10k VM round-trips. Should bring 10k-line load under 100ms.

**Generalization:** Any "I'm doing the same VM-mediated thing 10k
times" pattern needs a vectorized builtin. Same applies to mapping
to_int across an array (could be `arr_to_int(strings)`).

---

## HIGH-3: VM is *slower* than tree-walk on dict-heavy code

**Evidence:**

| Workload | tree-walk | VM | ratio |
|---|---|---|---|
| 10k aggregate (dict-heavy) | 16s | 18-20s | 1.13× *slower* |
| 10k build_hidx | 4.9s | 6.7s | 1.37× *slower* |
| HOF arr_map (Phase 4 bench) | 131ms | 59ms | **2.22× faster** |

**Root cause:** The Op::DictSetNamed path goes vm_get_var → mutate
→ vm_assign_var. Both steps clone the dict. The tree-walk path
does the same number of clones, but its eval_expr tail-calls are
slightly cheaper than the VM's stack-machine bookkeeping when both
are bottlenecked on identical Rust-side work.

**Implication:** The Phase 4 win ("VM ≥ tree-walk on every
benchmark") is true for the Phase 4 benchmarks but doesn't
generalize. Anything that sits in the same dict-clone hot loop sees
no benefit from the bytecode VM — and pays its dispatch overhead.

**Fix:** Same as HIGH-1. Once dict mutation is O(1) (Rc-shared),
the VM's hot dispatch should win again because builtin calls amortize.

---

## MED-1: arr_push in a hot loop is silently O(N²)

Already covered under HIGH-1, but worth calling out:

**Symptom:** "Build an array of N records" takes O(N²) time.

**User-facing impact:** Anything that follows the standard pattern

```omc
h out = [];
while ... {
    arr_push(out, item);
}
return out;
```

stops working past ~5000 iterations. There's no syntactic
indication that this is the wrong pattern.

**Suggested fix:** Either fix HIGH-1 architecturally, OR teach
users an alternative pattern (`arr_new(N, default)` + index assign,
or a builder type). Either way, document the cliff.

---

## MED-2: harmonic_index hit count is misleading vs linear scan

**Evidence (10k):**
```
linear   (R≈4 ±0.1):  523 hits
harmonic (R≈4 by attractor):  1825 hits
```

The harmonic engine returns 3.5× as many hits as the linear scan
because the attractor bucket for `target * 100 = 400` folds to 377
and includes everything in roughly [277.5, 472.5] — a much wider
range than ±0.1.

**Not a bug** — it's the correct semantics of harmonic
neighborhood lookup. But it makes "compare engines" benchmarks
misleading.

**Suggested fix:** Document the intent more clearly. The harmonic
engine is a *coarse* index (sub-linear lookup → coarse bucket); the
linear scan is *fine* (exact distance → narrow bucket). For a
recommendation system this is great (more diversity for free), but
for an "exact lookup" it's wrong.

---

## MED-3: OMC_HEAL would silently rewrite domain values

**Evidence:**
```
$ OMC_HEAL=1 ./omc examples/recommend/recommend.omc
--- OMC_HEAL: 1 diagnostic(s) across 1 iteration(s) (converged) ---
  harmonic: 4 not Fibonacci → 3 (|Δ|=1)
--- end OMC_HEAL ---
```

The heal pass saw the literal `4` (used as `target = 4.0` in our
"recommend movies near rating 4") and helpfully rewrote it to `3`
(the nearest Fibonacci) — which would have meant the user query
"movies rated 4 stars" became "movies rated 3 stars." Coincidentally
the run completed identically, suggesting the heal didn't actually
trigger on the rating-4.0 expression (possibly due to it being a
float literal, not int), but the diagnostic firing on a 4 *somewhere*
in the file is concerning.

**Root cause:** The harmonic-rewrite rule fires on any int literal
within edit-distance 3 of a Fibonacci attractor, with no awareness
of *what the value means*.

**Suggested fix:** Heal pass should respect a `@no_heal` decoration
on functions/expressions, OR only fire when the literal appears in a
position where an attractor would make sense (e.g., array indexing,
not comparison RHS). For now, treat OMC_HEAL as opt-in per-file.

---

## MED-4: No way to import a single fn from another file

**Symptom:** I copy-pasted four `hidx_*` functions from
`examples/harmonic_collections.omc` into `examples/recommend/recommend.omc`
because OMC doesn't support `from "x" import y`. The full
`import "x" as alias` form imports *every* function and aliases the
whole module — too heavyweight when you want one helper.

**Suggested fix:** Either add `from "path" import name1, name2;` or
make `import "path" as alias` accept the alias as `*` or empty
("merge selected names into namespace").

---

## LOW-1: Float display drops trailing `.0` for whole numbers

**Evidence:** `println(3.0)` prints `3`. Same for `count=1 avg=4`
which suggested an int when avg was actually a float.

**Root cause:** Rust's `format!("{}", 3.0_f64)` produces `"3"`. We
inherit this in `Value::to_display_string`.

**Suggested fix:** `Value::HFloat(f)` display should always show a
decimal point. e.g., `format!("{:?}", f)` produces `"3.0"`. Trade:
all float output is slightly noisier; benefit: int-vs-float ambiguity
in user output disappears.

---

## LOW-2: Engine-divergence error reports are useless

When the float bug was active, both engines silently produced wrong
answers. The user has no way to know "this output is wrong" without
running both engines and diffing. We have a regression sweep in dev,
but a real user wouldn't.

**Suggested fix:** Add `--audit` mode that runs both engines on the
same input and flags ANY divergence in output. Lightweight CI tool.

---

## LOW-3: Performance reporting in user code is verbose

Compare: `now_ms()` paired with subtraction is the only timing tool.
Every benchmark stanza in the recommend.omc is 4 lines of boilerplate
for one timed step.

**Suggested fix:** A `time_block(label, fn)` builtin that wraps a
closure, runs it, prints `label: Xms`, returns the result. Saves 3
lines per timed step.

---

# Prioritized fix list

1. **HIGH-1** (Rc-shared collections) — unblocks 10k+ workloads
2. **HIGH-2** (`csv_parse` builtin) — unblocks loading large data
3. **CRIT-1** (float truncation) ✅ FIXED
4. **MED-3** (OMC_HEAL respects literal context) — silent semantic bugs
5. **MED-4** (selective imports) — every multi-file demo wants this
6. **HIGH-3** (VM dict perf) — automatic from HIGH-1
7. **LOW-1** through **LOW-3** — polish

# What surprised me

* The float bug had been latent forever and would have shipped to a
  real user the first time they aggregated floats through arr_get.
  Found in 5 minutes of running real code. **Real datasets are the
  test harness; toy demos miss everything.**

* Tree-walk and VM produce *different wrong answers* under the same
  bug. That's worse than both being wrong the same way — there's no
  ground truth to diff against.

* The harmonic_index *worked* on real data — buckets concentrated on
  the natural rating attractors (3.5, 4.0, 4.5). But the n² collection
  cost meant we couldn't even build it past 10k records. The
  language is the bottleneck, not the algorithm.

* `OMC_HEAL` actively makes things wrong on real data. It's a
  research-fun feature on isolated demos but unsafe-by-default for
  real programs. **Opt-in per-file decoration is the right move.**
