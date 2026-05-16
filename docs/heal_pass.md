# The OMC self-healing compiler

The heal pass is a substrate-routed AST rewriter that catches and silently fixes common bugs before they reach the interpreter or JIT. It's enabled via `OMC_HEAL=1` (or `--check FILE` for diagnostics-only).

## Heal classes

Each class detects one bug pattern and applies one rewrite. All run in a single AST traversal per pass; `heal_ast_until_fixpoint` loops until no more diagnostics fire.

| Class | Pattern | Rewrite | Counter |
|---|---|---|---|
| **typo** | call to unknown name `foo` within edit-distance 2 of a defined name | replace name | `typo` |
| **arity_pad** | user fn called with fewer args than declared | append `Number(0)` per missing arg | `arity_pad` |
| **arity_truncate** | user fn called with more args than declared | drop excess args | `arity_truncate` |
| **div_zero** | `expr / 0` (literal 0 on RHS) | rewrite to `safe_divide(expr, 0)` | `div_zero` |
| **mod_zero** | `expr % 0` (literal 0 on RHS) | rewrite to `safe_mod(expr, 0)` | `mod_zero` |
| **harmonic_index** | `arr[N]` where N is off-attractor and `|nearest - N| ≤ 3` | snap to nearest Fibonacci attractor | `harmonic_index` |
| **missing_return** | user fn body has NO `return` statement anywhere | append `return null;` | `missing_return` |

## Substrate-routed typo lookup

The typo class is the heaviest by default — naively comparing every call site to every defined name is `O(N · m · k)` where N is the symbol table size, m the call sites, k the average name length.

The substrate-routed implementation uses a two-phase scan:

1. **Phase 1 (full)**: scan the small `prefer` set (user-defined fns, project-bounded). User fn matches always beat builtin matches on ties — a typo is more likely meant for a user fn than a builtin.

2. **Phase 2 (substrate-bucketed)**: hash each builtin name into one of 32 buckets via `substrate_hash_name` (Zeckendorf-style avalanche). For a typo, probe only the target's bucket plus 2 neighbors. Expected speedup: ~10× for projects with hundreds of defined names.

Falls back to full `closest_name` if both phases miss — preserves correctness.

```rust
// closest_name_substrate() in src/interpreter.rs
//   Phase 1: full O(|prefer|) scan of user fns (correctness)
//   Phase 2: 3-bucket scan of remaining builtins (speed)
//   Fallback: full closest_name() if both miss
```

## Per-class disable pragmas

A function can opt out of any single heal class via a pragma without disabling the others:

```omc
@no_heal_typo
fn raw_typos_allowed() {
    foo();  # NOT corrected; will hit eval error
}

@no_heal_div
fn raw_div_allowed() {
    h x = 10 / 0;  # NOT wrapped in safe_divide; produces Singularity
}

@no_heal_index
fn raw_index_allowed() {
    h arr = [1, 2, 3, 4, 5];
    return arr[4];  # NOT snapped; uses literal index 4
}

@no_heal_return
fn explicit_no_return() {
    h x = 5;
    # No `return null;` appended
}

@no_heal       # disables ALL classes for this fn (legacy total-disable)
fn fully_opaque() {
    # nothing healed in this fn body
}
```

Available pragmas: `no_heal`, `no_heal_typo`, `no_heal_arity`, `no_heal_div`, `no_heal_mod`, `no_heal_index`, `no_heal_return`.

## Heal budget

Each `heal_ast` pass has a fixed budget of `HEAL_BUDGET_PER_PASS = 1024` rewrites. Once exhausted, further heals are silently skipped (the diagnostic still records the count but no AST mutation). Prevents runaway rewrites on adversarial inputs while comfortably above any legitimate project's heal count.

## Per-class diagnostic counts

Each pass populates a `HealClassCounts` struct accessible via `last_heal_counts()` (Rust API):

```rust
pub struct HealClassCounts {
    pub typo: u32,
    pub typo_substrate_hit: u32,    // bucketed pre-filter found a match
    pub typo_fallback: u32,         // bucketed missed → full scan was needed
    pub arity_pad: u32,
    pub arity_truncate: u32,
    pub div_zero: u32,
    pub mod_zero: u32,
    pub harmonic_index: u32,
    pub missing_return: u32,
    pub empty_index_safe: u32,
    pub reserved_var: u32,
    pub if_numeric: u32,
}
```

`typo_substrate_hit` / `typo_fallback` together tell you how often the bucketed pre-filter earned its keep — a high `typo_fallback` rate signals the substrate-routing isn't picking up enough matches and the symbol-table distribution is unusual.

## Safe-arithmetic family

The heal classes that involve numeric ops all rewrite to `safe_*` builtins which substrate-fold their inputs at runtime:

- `safe_divide(a, b)` — fold b to nearest non-zero attractor (1 if needed)
- `safe_mod(a, b)` — same, applied to modulus
- `safe_sqrt(x)` — returns 0 for x < 0 (singularity-tolerant)
- `safe_log(x)` — returns -1e308 for x ≤ 0
- `safe_arr_get(arr, i)` — substrate-folded index with `% len` bounds wrap
- `safe_arr_set(arr, i, v)` — same for writes

These can also be called explicitly when you want substrate-tolerant semantics without going through the heal pass.

## Iterative convergence

`heal_ast_until_fixpoint(stmts, max_iter)` loops the single-pass `heal_ast` until:
- **converged**: zero diagnostics in last pass (all bugs fixed)
- **stuck**: same diagnostic count two passes in a row (no further progress)
- **exhausted**: hit `max_iter` (default 8)

Most programs converge in 1 pass. Iteration helps when one heal exposes another (e.g. typo fix surfaces an arity mismatch on the corrected call).

## Tests

`examples/tests/test_heal_pass.omc` — 16 tests covering each class plus per-class pragmas. Run with:

```bash
OMC_HEAL=1 omnimcode-standalone --test examples/tests/test_heal_pass.omc
```
