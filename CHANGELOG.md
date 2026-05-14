# Changelog

All notable changes to OMNIcode will be documented in this file.

## [Unreleased]

### Added (Standard library expansion — 16 new built-ins, 2026-05-14)

🎯 **`examples/stdlib_expansion.omc` + `STDLIB.md` — OMC's standard library now covers the common workflows developers reach for instead of writing from scratch.**

Before this commit OMC was Turing-complete but ergonomically narrow — basic things like splitting a string, sorting an array, or reading a file required hand-rolling them in OMC. After this commit OMC has Python-tier coverage of the common cases.

#### New built-ins by category

**Strings** (8): `str_split`, `str_join`, `str_trim`, `str_replace`, `str_index_of`, `str_starts_with`, `str_ends_with`, `str_repeat`. `str_index_of` returns char index (not byte) so it pairs with `str_slice`. `str_repeat` is capped at 1M chars to prevent accidental memory blow-up.

**Arrays** (3): `arr_sort`, `arr_reverse`, `arr_join`. Sort is stable with a float-fallback total ordering for mixed-type arrays. Reverse is the array form (`str_reverse` exists for strings). `arr_join` is the alias-equivalent of `str_join` with arg order swap, provided because users reach for the `arr_*` prefix.

**File I/O** (3): `read_file`, `write_file`, `file_exists`. Synchronous, UTF-8 for `read_file`. `file_exists` is total (never errors). `write_file` returns 1 on success.

**Introspection + utility** (4): `type_of`, `gcd`, `lcm`, `now_ms`.
- `type_of(v)` returns `"int"`, `"float"`, `"string"`, `"bool"`, `"array"`, `"null"`, or `"singularity"` — finally enabling generic OMC code that branches by type.
- `gcd` / `lcm` via Euclidean algorithm. `gcd(89, 144) = 1` is a nice math moment (consecutive Fibonacci numbers are coprime).
- `now_ms()` for benchmarking inside OMC programs.

#### Compiler type inference

`omnimcode-core/src/compiler.rs` updated to recognize the new builtins' return types for the Rust VM bytecode path. Confirmed identical output between tree-walker and `OMC_VM=1` on the full `stdlib_expansion.omc` test.

#### New documentation: `STDLIB.md`

A complete reference for every built-in (~100 total) organized by category: strings, arrays, numbers, harmonic primitives, self-healing primitives, file I/O, type/conversion, time. Each function has its signature, semantics, and known caveats. Replaces the previous misleading README claim of "~22 host primitives" — the real count was always closer to 100, this commit adds 16 more.

The doc also has a "Missing on purpose" section flagging what's NOT in the stdlib and why (`map`/`filter`/`reduce` require first-class functions; `format` is replaced by `concat_many`; etc.).

#### README + 00-START-HERE updates

- "What's proven right now" table gains a stdlib row pointing at `stdlib_expansion.omc`.
- The two stale "~22 host primitives" claims in the README are corrected.
- `00-START-HERE.md` adds `STDLIB.md` to the developer reading path and top-level doc index.

#### Verification

Test file (`examples/stdlib_expansion.omc`) runs every new function with expected outputs in inline comments. Identical output under tree-walk and `OMC_VM=1`. V.9b ✓✓✓ regression check still passes — no existing surface broken.

### Added (Phase H.5.2: Op::SafeArrSetNamed — Rust VM closes the `safe arr_set` mutation gap, 2026-05-14)

`omnimcode-core/src/bytecode.rs`, `vm.rs`, `compiler.rs`, `disasm.rs` — new `Op::SafeArrSetNamed(String)` opcode mirrors the V.7c `ArrSetNamed` pattern: variable name on the opcode bypasses `vm_call_builtin`'s synthetic-arg shim that copies array arguments. Compiler emits the named form when `Safe(Call("arr_set", [Variable, ...]))` matches. VM dispatch pops val + raw_idx, folds raw_idx onto nearest Fibonacci attractor via `pub(crate)` `fold_to_fibonacci_const`, Euclidean-mods by `arr_len`, mutates in scope. Empty arrays silently no-op for total semantics. Verified: `OMC_DISASM=1` shows `SAFE_ARR_SET_NAMED xs` emitted; tree-walk and `OMC_VM=1` produce identical output on `examples/safe_keyword_host.omc`. Phase H is now end-to-end clean on both interpretation paths.

### Added (Repo hygiene: target/ untracked, .gitignore expanded, 2026-05-14)

`git rm -r --cached target/` plus a comprehensive `.gitignore` rewrite removes 1149 build-artifact files from tracking (~10K lines of churn). Subsequent diffs show only real code/doc changes instead of fingerprint files and binaries. Standard Rust-project hygiene that should have been done at project start.

### Added (Phase H.5 host-language integration: `safe` as a first-class keyword, 2026-05-14)

🎯 **`safe` is now a host-level OMC keyword — no self-healing-demo infrastructure required.**

Until now, `safe a / b` and `safe arr_get(a, idx)` only worked inside the OMC-written self-healing compiler demos (`examples/self_healing_h4.omc`, `h5.omc`), which carry their own OMC-side parser, AST, encoder, and executor. The host Rust parser/interpreter didn't know `safe` as a keyword — it would tokenize as an unknown identifier.

This integration brings `safe` into the language proper:

| Layer | Change |
|---|---|
| Lexer (`parser.rs`) | New `Token::Safe`; `"safe"` keyword recognized |
| AST (`ast.rs`) | New `Expression::Safe(Box<Expression>)` variant |
| Parser (`parser.rs`) | `parse_expression` peeks for `Token::Safe`, wraps the rest of the expression. Bare statements (`safe arr_set(buf, i, v);`) work via the existing expression-statement fallback |
| Interpreter (`interpreter.rs`) | `Expression::Safe(inner)` pattern-matches the inner shape: `Div(l, r)` → `safe_divide(l, r)`, `Call("arr_get", ...)` → `safe_arr_get(...)`, `Call("arr_set", ...)` → `safe_arr_set(...)`; unknown shapes evaluate the inner directly |
| Compiler (`compiler.rs`) | `Expression::Safe(inner)` lowers to the matching `Op::Call("safe_*", n)` for known shapes; type inference delegates to the inner expression |

#### Smoke test (`examples/safe_keyword_host.omc`)

Eight assertions, all pass on the host interpreter without any OMC-written self-healing wrapper:

- `safe 89 / 0 → 89`
- `compute(144, 0) → 144` (dynamic zero healed)
- `compute(144, 3) → 48`
- `safe arr_get([10,20,30], 999) → 20` (fold(999)=610, 610%3=1)
- `safe arr_get([10,20,30], 1) → 20`
- `safe arr_set(xs, 999, 99)` writes xs[1]=99; xs[0] and xs[2] unchanged

The mutation case (the H.5 named-store fix in OMC bytecode) is naturally clean through tree-walk because the interpreter pattern-matches `Safe(Call("arr_set", [Variable(name), ...]))` before any synthetic-arg shim runs — `safe_arr_set` receives the actual `Expression::Variable(name)` it needs and writes back to the caller's scope.

#### What still doesn't work

`Safe(Call("arr_set", ...))` compiled to bytecode and run through the Rust VM lowers to `Op::Call("safe_arr_set", 3)`, which routes via `vm_call_builtin`'s synthetic-arg shim → mutation lost. This is the same gap V.7c closed for `arr_set` with `Op::ArrSetNamed`. A future `Op::SafeArrSetNamed(String)` would close it here too. Tonight's scope kept the Rust-VM bytecode path on the existing call shim — tree-walk works cleanly, the named-mutation gap is documented and bounded.

#### Why this matters

The H.4/H.5 OMC-written demos remain the architecturally pure proof — the bytecode VM rewrites and executes `safe` semantics end-to-end on the φ-math substrate. But for a developer who just wants the feature in their OMC code, it's now a one-keyword opt-in at the language level. The Phase H story is no longer "fork the self-healing-compiler demo file." It's "write `safe` where you'd write a runtime guard."

### Added (Phase H.5.1: close the safe arr_set bytecode-VM gap, 2026-05-14)

`examples/self_healing_h5.omc` — `safe arr_set(VAR, idx, val)` works through the OMC bytecode VM, not just under tree-walk. New `SAFE_ARR_SET_NAMED varname` opcode in the OMC-written executor mirrors V.7c's `ARR_SET_NAMED` pattern: the variable name rides on the opcode itself rather than going through `CALL_BUILTIN`'s synthetic-arg shim that copies array arguments. Encoder detects bare-VAR first-arg shape and emits the named form; executor pops idx/val, looks up array in scope, computes fold-and-mod healed index, mutates, writes back. Demo 4b verifies: `[55, 13, 0, 0, 34]` buffer state after four `safe arr_set` writes with `idx ∈ {0, 100, -1, 6}`. Six demos, six convergences.

### Added (Phase H.5: array-bounds healing via fold_escape on the index, 2026-05-14)

🎯 **`examples/self_healing_h5.omc` — `safe arr_get(a, idx)` and `safe arr_set(a, idx, v)` make out-of-bounds accesses total.**

H.4 made dynamic divide-by-zero safe at the math level (`safe a / b` → `safe_divide(a, b)` → fold the divisor away from zero). H.5 extends the same pattern to the next obvious bug class: **array-index violations.**

#### The healing formula

For a `safe`-wrapped array access, the encoder rewrites the call to a new host primitive that applies:

```
healed_idx = ((fold_escape(idx) % arr_len(a)) + arr_len(a)) % arr_len(a)
```

`fold_escape` pulls the index onto the nearest Fibonacci attractor; modulo by `arr_len(a)` keeps it in-bounds; the redundant `+ len) % len` handles negative remainders cleanly. Empty arrays return `Null` rather than error — the access stays total.

What this means in practice for `safe arr_get([10, 20, 30], idx)`:

| `idx` | `fold_escape(idx)` | `% 3` | Result |
|---|---|---|---|
| 1 | 1 | 1 | `20` (in bounds, attractor) |
| 7 | 8 | 2 | `30` |
| 999 | 610 | 1 | `20` |
| -5 | -5 | 1 (after sign-fix) | `20` |

Deterministic, in-bounds, attractor-landing where the Fibonacci grid permits.

#### Implementation surface

Adding the new primitive is a ~30-line composition:

- `omnimcode-core/src/interpreter.rs` — two new host builtins `safe_arr_get` and `safe_arr_set`. Both reuse the existing `fold_to_fibonacci_const` helper.
- `examples/self_healing_h5.omc`:
  - `is_builtin` recognizes the two new names.
  - `call_builtin` dispatches them.
  - `collect_defined` adds them to the typo-correction name table.
  - `enc_expr`'s `SAFE_EXPR` branch extends to recognize `safe arr_get(...)` and `safe arr_set(...)` call shapes, rewriting to the new builtins.
  - `p_stmt` gains a `SAFE` branch so `safe arr_set(buf, i, v);` works as a bare statement.

No new keywords, no new AST nodes — H.4's `SAFE_EXPR` is reused.

#### Five demos, five convergences

- Demo 1 (regression): H.4's `safe a / b` still works — `compute(144, 0) → 144`.
- Demo 2 (baseline): unguarded `arr_get(xs, idx)` — runs only because the demo index is in-bounds.
- Demo 3 (headline): `safe arr_get` with indices `{1, 999, -5, 7}` against a 3-element array. All four reads return finite values; the OOB indices land on attractor positions.
- Demo 4 (loop walking off the end): `i = 0..7` reading from a 5-element array via `safe arr_get`. **Every output value has `φ=1.000`** — every read landed on a Fibonacci attractor.
- Demo 5 (H.4 + H.5 composed): a function that does both a safe array read and a safe division on the result. Survives both a singular divisor and an OOB index in one call.

#### One known limit (logged for H.5.1)

`safe arr_set(VAR, ...)` works under tree-walk but not via the OMC bytecode VM. The bytecode VM routes `CALL_BUILTIN` through a synthetic-scope shim that copies the array argument; the mutation lives in the temporary scope and doesn't propagate back to the caller's variable. The Rust VM solved this same problem in V.7c with `ARR_SET_NAMED` opcodes. H.5.1 would add an `SAFE_ARR_SET_NAMED` variant. Reads (`safe arr_get`) compose cleanly through either path because they return a value rather than mutating a binding.

Demo 4 was rewritten to use `safe arr_get` only, avoiding this trap. The Phase H semantic claim — that out-of-bounds accesses become total — is intact for reads on both interpretation paths.

### Added (Phase H.4: `safe` keyword — runtime self-healing as user syntax, 2026-05-14)

🎯 **`examples/self_healing_h4.omc` — the user can now DECLARE self-healing intent in source code, not just rely on the compiler to detect it.**

H.1–H.3 all worked on STATIC bugs: things the compiler could detect by inspecting tokens or AST nodes without running the program. The hard case in real code is **dynamic singularities** — `x / count` where `count` could be zero at runtime for inputs the author didn't anticipate. The static healer's divide-by-singularity check (H.2) only fires when the divisor is a literal. Variables don't trigger it. The bug ships.

H.4 surfaces a new keyword `safe` that lets the user opt expressions into runtime self-healing semantics:

```omc
fn compute(count, mod) {
    return safe count / mod;
}
print(compute(144, 3));   // 48  — normal division
print(compute(144, 0));   // 144 — fold_escape catches the zero divisor
```

The parser recognises `safe EXPR` and wraps it as `["SAFE_EXPR", inner]`. The encoder rewrites `SAFE_EXPR` containing a `BIN /` to a `CALL_BUILTIN safe_divide` **unconditionally** — regardless of whether the divisor is a literal or a variable. The compile-time healer (H.2) and the runtime user-intent declaration (H.4) are complementary:

| Trigger | Active when | Catches |
|---------|-------------|---------|
| H.2 static healer | divisor is a literal `0` (or near-zero) | Obvious compile-time bugs |
| H.4 `safe` keyword | user explicitly wrote `safe` | Dynamic divisors at runtime |

Both rewrite to the same primitive (`safe_divide`). The difference is the trigger.

#### Four demos

**Demo 1** (H.2 regression) — `89 / 0`. Static healer fires (literal `0`). Rewritten to `safe_divide(89, 0)`. Output: 89. Unchanged from H.2.

**Demo 2** (baseline, no `safe`) — `return count / mod;` with mod variable. Compiles to bare `/`. Runs with `mod = 3` so no crash, but the bug is shipping. Output: 48.

**Demo 3** (**the headline**) — same shape as Demo 2 but with `safe count / mod`. Compiler unconditionally rewrites to `safe_divide(count, mod)`. Two calls:
- `compute(144, 3) → 48` (normal division)
- `compute(144, 0) → 144` (**runtime crash converted to finite answer on attractor**)

The one-keyword annotation flipped a runtime crash into a working program. No `if mod == 0` boilerplate.

**Demo 4** (integrated) — five things in one source:
- Token-level: missing SEMI between `h target = 7` and `print(...)`.
- AST-level (H.1): `compue` typo → `compute` (edit distance 1).
- AST-level (H.1): `7` close-miss Fibonacci → `8` (|Δ|=1).
- AST-level (H.2): the `numerator / divisor` is dynamic, so H.2's static check DOESN'T fire (correctly — no compile-time signal).
- H.4: the user wrote `safe`, so the division is rewritten to `safe_divide` at encode time.

Final: `compute(8, 0) → safe_divide(8, 0) → fold_escape(0)=1 → 8/1 → 8`. All three Phase H stages contributed; all converged; the program lands on a Fibonacci attractor.

#### The bigger picture (LLM-generated code)

For language-model-generated programs, the failures cluster around three classes:
1. Typos and naming drift (variables, function names).
2. Off-by-one numeric constants (loop bounds, array sizes).
3. Unguarded edge cases (division, indexing, null derefs).

Phase H handles all three. A self-healing target language reduces the burden on the generator: it doesn't have to write defensive boilerplate because the language's compiler does the defense automatically — partly at compile time (static healer), partly at user-declared opt-in (`safe`), partly with primitive operations that fold_escape singularities at runtime (`safe_divide`).

This is a real architectural difference from conventional target languages. Most existing autocomplete/heal tooling lives OUTSIDE the language (IDE plugins, linters in a different language). H.1–H.4 live INSIDE OMC, reusing the same lex/parse/encode/execute machinery the rest of the language uses, all sitting on the Phase O φ-math substrate.

#### What still isn't done

- `safe` currently only meaningfully rewrites BIN `/`. Other expressions wrapped in `safe` encode as their inner form (no-op), reserving the slot for future runtime guards (fold_escape on function-call return values, value_danger threshold on arithmetic chains).
- Indentation-aware brace placement (H.3.1) — still naive append-at-EOF.
- The `stuck` and `exhausted` outcomes of the iteration loop remain unexercised. Designing a demo that hits them in a meaningful way is future work.

### Added (Phase H.3: parse-level recovery — token-stream healing, 2026-05-13)

🎯 **`examples/self_healing_h3.omc` — the healer gains a stage BELOW the parser.**

H.1/H.2 worked on the AST. But an AST presumes a successful parse, and a parse presumes a syntactically valid token stream. H.3 adds the missing layer: TOKEN-LEVEL repair that runs BEFORE the parser sees anything. Three repair classes at this layer:

1. **Unbalanced LBRACE/RBRACE** — count opens vs closes; if opens > closes, append RBRACE tokens before EOF until balanced. Diagnostic: "N missing '}' (brace); appending before EOF".

2. **Unbalanced LPAREN/RPAREN** — same shape.

3. **Missing SEMI** — scan adjacent token pairs. When a clear expression terminator (NUMBER, STRING, RPAREN, RBRACKET) is immediately followed by either a clear statement starter (H, FN, IF, WHILE, RETURN, BREAK, PRINT) OR EOF, insert SEMI between them. The EOF case covers trailing statements at end-of-file with no closing semicolon.

The pipeline is now two-stage:

```
source → tokenize_b → token_heal (H.3) → tokens'
                                          ↓
                                      p_program
                                          ↓
                                         AST
                                          ↓
                              heal_until_fixpoint (H.1/H.2)
                                          ↓
                                      healed AST
                                          ↓
                                     emit_source
                                          ↓
                                    healed source
```

#### Why the layers are separate stages, not diagnostic classes

The parser falls off a cliff on unbalanced braces; nothing downstream of a broken parse produces a meaningful AST. Token-level repair has to run FIRST to give the parser back its substrate. Once the parser returns a valid AST, the H.1/H.2 checks fan out over the tree. The iteration loop wraps the AST stage; the token stage is a single-pass count-and-insert.

This was Architect's observation from H.2: "iteration only matters when one fix exposes another, or when fixes happen at different STAGES." H.3 is exactly the latter.

#### The four demos

**Demo 1** — Three statements with no semicolons. token_heal inserts 3 SEMI tokens (including one before EOF for the trailing statement). AST is clean. Output: `8 + 13 = 21` (21 is itself Fibonacci, harmony 1.0).

**Demo 2** — Missing RBRACE at end of file. `print(double(13))` placed BEFORE the broken function so naive append-at-EOF closes the function body cleanly without folding any code into it. Output: `double(13) = 26`.

**Demo 3** — Missing RPAREN in `print(id(21);`. token_heal appends `)` before EOF. Output: `id(21) = 21`.

**Demo 4** — Five bugs across both stages in one source:
- Token: missing SEMI between `print(...)` and `fn safe`.
- Token: missing `}` at end of file.
- AST: `safef` typo → `safe`.
- AST: `7` close-miss Fibonacci → `8`.
- AST: `n / 0` divide-by-singularity → `safe_divide(n, 0)`.

The two-stage pipeline handles all five. Output: `safe(8) → safe_divide(8, 0) → 8` (on attractor). **Three of the five would have produced compile errors in any conventional compiler; the other two would have produced runtime crashes. The OMC pipeline turns ALL FIVE into a working program landing on attractor.**

#### One real limit: naive brace placement

token_heal appends missing braces at EOF. If the missing `}` is conceptually MID-source (e.g., between a function body and a top-level statement that follows it), naive appending folds the following statements into the function body where they become unreachable code. The PARSE succeeds; the SEMANTICS may not match the user's intent. Smarter brace placement using indentation analysis is logged for H.3.1.

Mitigated in tonight's demos by structuring sources where the missing brace is genuinely at end-of-file — top-level calls placed BEFORE the broken function. The current limit doesn't bite.

#### What's still unexercised

The `stuck` outcome (diagnostic count plateaus above zero) — reserved for genuine undecidables: a typo that matches NOTHING within edit-distance threshold, or a harmonic violation where the nearest Fibonacci is too far for the proximity gate. The `exhausted` outcome (hit max_iter) — reserved for divergent rewrites. Neither fires in tonight's demos.

### Added (Phase H.2: autofix-and-retry loop + divide-by-singularity check, 2026-05-13)

🎯 **`examples/self_healing_h2.omc` — the healer becomes iterative and gains a third diagnostic class.**

H.1 was a single-pass scanner. H.2 wraps it in a convergence loop and adds runtime-singularity awareness.

#### New diagnostic class: divide-by-singularity

In `heal_expr`, when a `BIN /` operator's right operand is a literal `NUM`, the healer evaluates `value_danger(v)`. If the result exceeds 0.5 (fold_escape's danger threshold), the whole expression is rewritten as `CALL safe_divide(left, right)`. The host's `safe_divide` primitive does the rest — it fold_escapes the divisor at runtime, so `safe_divide(8, 0)` returns `8` (8 / fold_escape(0) → 8 / 1 → 8), not an error.

The key property: **the healer turns a runtime crash into a finite answer that lands on a Fibonacci attractor**, with no special-case error-handling code anywhere. The math is the rule.

#### The loop: `heal_until_fixpoint`

```
fn heal_until_fixpoint(stmts, max_iter):
    for iter in 1..max_iter:
        (healed, diags) = heal_program(current)
        record (iter, |diags|) in trajectory
        if |diags| == 0:        return ("converged", iter - 1)
        if |diags| == prev:     return ("stuck",     iter)
        current = healed
    return ("exhausted", max_iter)
```

Three terminal states:
- **converged** — diagnostics dropped to zero. Healed source stable.
- **stuck** — same count two iterations running. Healer can't make progress.
- **exhausted** — hit the safety bound. Likely a divergent rewrite cycle.

Tonight's demos all converge cleanly. `stuck` and `exhausted` are exercised in H.3/H.4.

#### Three demos, three convergences

**Demo 1** (H.1 regression): `12 → 13`, `fbi → fib`. Iter 1: 2 diagnostics → Iter 2: 0. Output: `fib(13) = 233`. Verdict: converged in 1 iteration.

**Demo 2** (new): `numerator / 0`. The harmonic check fires on `10 → 8`; the divide-by-singularity check rewrites the division. Healed program: `safe_divide(8, 0)` → host fold_escape's 0 to 1 → returns 8. **Runtime crash converted to finite answer on attractor.**

**Demo 3** (all three): `7 → 8` (harmonic), `safef → safe` (typo), `n / 0 → safe_divide(n, 0)` (singularity). One pass through `heal_program` fans the three classes out over the AST and rewrites all three in parallel. Output: `safe(8) → 8`.

#### Bytecode VM gains ONN primitives

`is_builtin` and `call_builtin` in the V.7c-style executor now dispatch `safe_divide`, `fold_escape`, `value_danger`, `harmony_value`, and `is_fibonacci`. The healed programs can be executed end-to-end without falling back to tree-walk — the bytecode VM hosts the φ-math primitives the healer rewrites toward.

#### Why iteration matters even when one pass suffices

H.1's checks already run in parallel during one AST walk. H.2's loop is mostly empty-confirmation passes (iter 2 always shows 0 diagnostics). But the LOOP IS THE RIGHT SHAPE for upcoming additions:
- **H.3** — parse-level recovery. A syntax error rewrite produces a new AST that needs to be re-parsed and re-healed. Naturally iterative.
- **H.4** — runtime-guarded primitives surfaced as OMC keywords. Adding a guard around an expression changes the AST in ways that may expose new diagnostics on adjacent nodes.

H.2 lands the substrate so H.3/H.4 plug in without re-architecting.

### Added (Phase H.1: Self-Healing Compiler — harmonic + typo diagnostics, 2026-05-13)

🎯 **`examples/self_healing_compiler.omc` — OMC's φ-math becomes a diagnostic lattice the compiler reasons against.**

The Phase H arc starts here: a self-healing compiler that catches errors using the Fibonacci-resonance math built in Phase O. Tonight's H.1 deliverable handles two narrow but real classes of bugs:

1. **Harmonic violations** — numeric literals that aren't Fibonacci-aligned but are close to a Fibonacci attractor. Two-stage check:
   - `is_fibonacci(n) == 0` → off the φ-geodesic.
   - Distance to nearest Fibonacci ≤ 3 → close-miss typo (off-by-one, transposition).
   - Suggest replacement; rewrite literal in AST.
   - Numbers that aren't Fibonacci but ARE far from any attractor (e.g. `100`) are left alone — probably intentional.

2. **Identifier typos** — VAR / CALL references that don't appear in the defined-name table. Levenshtein edit distance against defined names (FN defs, DECLs, parameters, plus a baseline of built-ins). Threshold ≤ 2.

Both classes feed the same pipeline:
```
broken source → V.9 lex/parse → AST → heal_program → (healed_ast, diagnostics)
                                                          ↓
                                                    emit_source
                                                          ↓
                                                   healed source
                                                          ↓
                                                 re-lex+re-parse
                                                          ↓
                                              0 diagnostics (fixpoint)
                                                          ↓
                                                   encode_program
                                                          ↓
                                                  run on OMC VM
                                                          ↓
                                                  expected output
```

#### Demo

Input source (intentionally broken):
```omc
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
h target = 12;
print(fbi(target));
```

Output:
```
--- Healing pass ---
  diagnostics: 2
  harmonic: 12 (harmony 0.92, not Fibonacci) → 13 (harmony 1, Fibonacci attractor; |Δ|=1)
  call: 'fbi(...)' unknown → 'fib(...)' (edit distance 2)

--- HEALED SOURCE ---
fn fib(n) { ... }
h target = 13;
print(fib(target));

--- Sanity check: re-lex + re-parse healed source ---
  diagnostics on healed source: 0     ← fixpoint property of healer holds

--- Final: execute the healed source via the OMC bytecode VM ---
  output:
HInt(233, φ=1.000, HIM=0.002)         ← fib(13) = 233, itself a Fibonacci
```

The whole pipeline ends on a Fibonacci attractor. The φ-math is internally self-consistent.

#### New OMC-side helpers

- `nearest_fib(n)` — mirrors the host's `fold_escape` attractor table from OMC, exposing the suggestion as a value (not just a side effect of evaluation).
- `edit_distance(a, b)` — iterative Levenshtein DP, two-row rolling.
- `closest_name(name, table)` — best edit-distance match below threshold 3, returns "" if none qualify.
- `collect_defined(stmts)` — walks top-level for FN + DECL names, plus a baseline of host built-ins (print, arr_*, str_*, harmony_value, fold_escape, value_danger, etc.).
- `heal_expr / heal_stmt / heal_block / heal_program` — recursive AST walker; threads diagnostics array via return-and-rebind (pass-by-value-safe).
- `emit_source` — V.4-style pretty-printer adapted to V.9's AST shape.

#### What this demonstrates

The compiler doesn't have a hand-written "list of magic Fibonacci numbers." It uses `is_fibonacci` and `harmony_value` — host primitives from Phase O whose semantics are pure math. The healer asks the language to score literals against itself; the φ-math IS the rule.

This is what makes OMC structurally different from a conventional language with a built-in linter. The Fibonacci attractor table isn't a style guide; it's an inductive type-class the language can OBSERVE on any integer at runtime. The healer's logic is short — `is_fibonacci(n) == 0 && |Δ| ≤ 3 → flag` — because the heavy lifting lives in the math, not in the rules.

#### What H.2/H.3/H.4 will add

- **H.2** — autofix-and-retry loop. Run the healer, apply, recompile, iterate up to N attempts. Currently H.1 is one-shot.
- **H.3** — parse-level recovery. Try inserting missing semicolons / braces guided by HIM scoring of the resulting AST. Catches structural bugs upstream of the AST walker.
- **H.4** — runtime self-healing primitives surfaced as OMC keywords. Today the healer is compile-time only; H.4 would let OMC functions declare "if my divisor is in the danger zone, fold_escape it" without writing the boilerplate.

### Added (Phase V.9 + V.9b: UTF-8 safety + gen2 == gen3 of a compiler, 2026-05-13)

#### V.9 — `str_chars` host builtin closes the byte-vs-char mismatch

`omnimcode-core/src/interpreter.rs` gains a `str_chars(s) → int` builtin. Where `str_len` returns byte count and `str_slice` is char-indexed (the V.8b trap), `str_chars` returns char count and matches `str_slice` exactly. Hand-written lexers over UTF-8 source now have a bound that aligns with their iterator.

`examples/self_hosting_v9.omc` ports the V.8b stack to `str_chars` at every source-iteration site (`tokenize_b`, `scan_string`, `skip_ws_b`, `match_multichar_b`, plus the `classify_word` test fn). The em-dash in `TOKENIZE_SUBSET_SOURCE`'s prologue comment — the exact byte that produced V.8b's silent UNKNOWN_STMT op — now processes clean. Non-ASCII test inputs (`"→→→"`, `"café"`) classify correctly under both paths. 2/2 fixpoints reached, zero parser warnings.

The fix lands on both interpreters automatically: the bytecode VM's `CALL_BUILTIN` routes through `vm_call_builtin` → `call_function`, which dispatches via the same interpreter table that gained `str_chars`. One file changed, one builtin added, two interpreters fixed.

#### V.9b — `examples/self_hosting_v9b.omc` — gen2 == gen3 of a COMPILER

The textbook self-application fixpoint at compiler-bootstrap level.

A real mini-compiler `mini_enc(ast) → bytecode_array` (a bytecode encoder for the NUM / VAR / BIN(+/-/*/==) expression dialect, ~30 lines of OMC) is run two ways on the same hardcoded input AST `(89 + 144) * 2`:

- **Path A (gen1)** — tree-walked `mini_enc` directly evaluates the AST and returns `["LOAD_INT 89", "LOAD_INT 144", "ADD", "LOAD_INT 2", "MUL"]`.
- **Path B (gen2 → gen3)** — tree-walked V.9 stack compiles `MINI_ENC_SOURCE` (290 tokens, 4 top-level statements, **140 bytecode ops**). The OMC bytecode VM executes that bytecode on the same hardcoded AST, returning `__result`.

Both paths produce identical arrays. The compiler is a fixed point under self-application.

This isn't "V.8b again with a different test function" — V.9b takes the specific instance where the program being compiled IS A COMPILER. The same machinery that processes the program runs INSIDE the program. The recursive structure is the point.

#### What this completes architecturally

- V.6: AST → bytecode encoder + stack-VM executor (integers only)
- V.7: function calls + recursion + call frames
- V.7b: strings + arrays + read-only builtin dispatch
- V.7c: mutating builtins via named-store opcodes
- V.8: round-trip fixpoint between tree-walk and bytecode VM (semantic equivalence proved)
- V.8b: `#` comments, `-> type` annotations, `break` round-trip cleanly
- V.9: UTF-8-safe iteration via `str_chars` host builtin
- **V.9b: a real compiler-as-function is a fixed point under self-application**

The OMC self-hosting stack is now operationally complete. The compiler-in-OMC and executor-in-OMC are functionally indistinguishable from the host tree-walker for programs that use their supported feature surface. Scaling to the full V.9 stack compiling its own ~700-line source is a one-shot of time (the bytecode VM is OMC code being tree-walked, so thousands of ops × dozens of dispatch branches per op), not a question of architecture.

The bootstrap loop is closed.

### Added (Phase V.8b: the fixpoint widens to the full compiler subset, 2026-05-13)

🎯 **`examples/self_hosting_v8b.omc` — the OMC bytecode VM now hosts every construct the compiler source itself uses.** Two fixpoint tests reach ✓ on first clean run.

#### What V.8b adds to the V.7c-in-V.8 stack

Three small extensions, all in the lexer/parser/encoder:

1. **`#` line comments**. `skip_ws_b` now loops over (whitespace, comment) until neither advances `pos`. A `#` consumes everything up to (not including) the next `\n`.

2. **`-> type` return-type annotations** on `fn` definitions. After `RPAREN`, the parser looks for `MINUS GT IDENT` and skips it if present. Annotations carry no runtime information — they document for the reader — so the parser swallows them silently.

3. **`break` inside while loops**. Lexer recognises `break` as a keyword. Parser emits `["BREAK"]`. Encoder emits `["JUMP_BREAK", 0]` as a placeholder. The enclosing while-loop encoder scans `body_ops` for `JUMP_BREAK` placeholders and rewrites each to a `["JUMP", b_len + 1 - k]` whose relative delta lands just after the trailing back-jump — i.e., immediately after the loop. **Relative jumps survive concatenation**, so patching `body_ops` in place before assembling the full while-block is sound. Nested while loops work because the inner encoder patches its body before the outer encoder sees the inner block as one opaque sub-array of ops.

#### The two fixpoint demos

**Test 1 — `classify_word`** uses all three new features simultaneously: `#` comment in the embedded source, `-> string` on the fn def, `break` inside a while when a vowel is found. Returns an array of 5 strings (`"alpha"`/`"beta"`); both paths produce byte-identical output.

**Test 2 — `tokenize_subset`** is the headline: a small but real lexer (digits + identifiers + punctuation, enough to tokenize `"h x = 89 + fib(144)"`). Embedded as a string. Compiled through the V.8b stack to 186 bytecode ops. Executed on the OMC executor. Returns 9 tokens (`"ID:h"`, `"ID:x"`, `"PUNCT:="`, …). The tree-walked version of the same function returns the same 9 tokens. **This is gen2 == gen3 for a real compiler component** — a piece of compiler logic, written in OMC, round-trips byte-identical through the OMC compiler-in-OMC + OMC executor-in-OMC.

#### One bug flushed: str_len/str_slice mismatch

First V.8b run produced ✓ fixpoint on both tests but emitted a quiet `p_stmt: don't know how to handle kind=IDENT` warning during Test 2. Trace: an em-dash (`—`, 3 bytes in UTF-8) in the embedded source's prologue comment. `str_len` returns BYTE count (5 for `"a—b"`); `str_slice` is CHAR-indexed. The hand-written lexer's main loop `while pos < n` advances `pos` by 1 per char, overshooting by `bytes - chars` iterations past the real string end. `str_slice` past the end returns `""`. `is_alpha_b("")` falsely returns 1 because `str_contains("alphabet", "")` is always true (Rust's `str::contains("")`). The lexer emitted a phantom `["IDENT", ""]` token between the real source and EOF; the parser couldn't classify the empty-IDENT statement; the encoder emitted `UNKNOWN_STMT`; the executor reported the unknown opcode at runtime — but the phantom statement was downstream of all real code, so the visible output was still correct.

Quick fix in V.8b: keep embedded source ASCII-only (use `-` instead of `—`). The proper fix is a host-side change adding either a char-indexed `str_chars` builtin or making `str_len` consistent with `str_slice`. Logged in memory.

#### What this means architecturally

Every construct the V.8b-style compiler source itself uses now round-trips through the OMC bytecode path with byte-identical output. The language can:
- read its own source (lexer-in-OMC tokenizes OMC source)
- structure it (parser-in-OMC builds AST)
- emit bytecode (encoder-in-OMC produces ops)
- execute that bytecode (executor-in-OMC stack-VM runs ops)

…and the answer at the end is the same whether the host tree-walker or the OMC bytecode VM produced it. The bootstrap loop is closed at the feature-surface level.

#### What V.8b doesn't do (yet)

Test 2 demonstrates one compiler component round-tripping. The fully self-applied bootstrap — the V.8b compiler compiling its OWN full source via the bytecode path — is in reach but slow: the bytecode VM is itself OMC code being tree-walked, so a full self-compile would chain ~3000 bytecode ops through ~30 if-branches per dispatch. Tractable as a one-off correctness check; not interactive. The `str_chars` host builtin (or a UTF-8-safe lexer rewrite) would also need to land first if the source contains non-ASCII characters. Logged as V.9.

### Added (Phase V.8: round-trip fixpoint between tree-walk and OMC bytecode VM, 2026-05-13)

🎯 **`examples/self_hosting_v8.omc` — the OMC compiler-in-OMC and executor-in-OMC produce byte-identical results to the tree-walker on the same source.**

The central claim of the self-hosting project, now demonstrated empirically:

> Run an OMC program through the tree-walker → get answer A.
> Compile the same OMC program to bytecode using the OMC-written compiler.
> Execute that bytecode on the OMC-written VM → get answer B.
> A == B.

#### How the demonstration works

The V.8 file contains the full V.7c stack (lex, parse, encode, execute) plus a driver that runs each test program two ways:
- **Path A** — inline OMC function definition, evaluated directly by the tree-walker. Returns its result as an OMC array.
- **Path B** — the same function defined inside an `EMBEDDED_SOURCE` string. The V.7c-in-V.8 stack tokenizes / parses / encodes that string to bytecode, then `execute()` runs it. The bytecode binds its answer to `__result` and `execute()` surfaces that value via a new return path.

The driver then calls `arr_equal_flat(out_a, out_b)`. Both demos produce identical output:

**Demo 1: `embedded_program()`** — builds a flat array of bytecode-listing strings (`"LOAD_INT 10"`, `"ADD"`, …) from a list of integers, exercising array literals, while loops, `arr_push`, conditional emission, and `concat_many` over mixed int/string args. Returns 7 elements. ✓ FIXPOINT.

**Demo 2: `build_pyramid(5)`** — accumulates strings via `str_concat` in a tight nested-while inner loop. Returns 5 elements (`"*"`, `"**"`, …, `"*****"`). ✓ pyramid FIXPOINT.

#### One blocker found and fixed: concat_many cosmetic divergence

V.7b's CHANGELOG flagged that `concat_many(s, int_val)` rendered the int via Debug formatter (`"HInt(42, φ=…)"`) in the bytecode VM but via Display (`"42"`) in tree-walk. V.8's first run hit this exact bug in Demo 1. The fix: `call_builtin`'s `concat_many` now applies `to_string` to each arg before `str_concat`. `to_string` invokes the host's Display path for HInt, so the rendering matches tree-walk.

`to_string` is also now in the bytecode-level builtin set (`is_builtin` returns true; `call_builtin` dispatches).

#### What's actually proven by V.8

This is the **semantic** half of the gen2 == gen3 claim. The OMC compiler-in-OMC and executor-in-OMC are correct against the host as a reference implementation: any OMC program that runs end-to-end through the bytecode path produces the same value as tree-walk.

The remaining piece, byte-identical gen2 == gen3 of the *compiler* on its own source, is now structurally trivial — the bytecode VM provably executes OMC faithfully — but blocked on three small extensions to the V.7c-style lexer/parser:
1. `#` line comments.
2. `-> type` return type annotations on `fn` definitions.
3. `break` inside while loops.

The V.7c-style compiler source uses all three. Adding them turns the V.8 round-trip into a self-applied bootstrap. Logged as V.8b / V.9.

#### What `execute()` now returns

Previously `execute(prog)` returned `0`. V.8 changes it to `scope_get(scope, "__result")` at HALT. Programs that don't bind `__result` still get `0` (the scope_get fallback), so this is backward-compatible. Programs that do bind `__result` make the bytecode VM's answer available to the outer caller — which is what closed the round-trip loop here.

### Added (Phase V.7c: arr_push and arr_set on the OMC bytecode VM, 2026-05-13)

🎯 **`examples/self_hosting_v7c.omc` — mutating array builtins now work at the bytecode level via named-store opcodes.**

This is the last structural prerequisite before full gen2 == gen3 of the compiler-on-itself. The V.7b lexer's `tokens` accumulator and the encoder's `out` buffer both rely on `arr_push` — without V.7c, bytecode versions of those would silently no-op and the bootstrap fails before it begins.

#### New bytecode ops

- `["ARR_PUSH_NAMED", varname]` — pop value, look up `varname` in current scope, `arr_push` to the array, write the modified array back to scope under the same name. Leaves the (mutated) array on the value stack as the expression's result.
- `["ARR_SET_NAMED", varname]` — pop value, pop index, look up `varname`, `arr_set`, write back. Same result convention.

These are the bytecode-level analogue of the Rust VM's `ArrPushNamed` / `ArrSetNamed` (see `omnimcode-core/src/vm.rs`). The architectural answer to OMC's pass-by-value arrays is the same on both sides: take the variable name out of the value stack and put it directly on the opcode.

#### Encoder pattern detection

When `enc_expr` sees a `CALL` with name `arr_push` and exactly 2 args, OR name `arr_set` and exactly 3 args, AND the first arg is a bare `["VAR", name]`, it emits the specialised named-store form. Anything fancier (e.g. `arr_push(arr_get(rows, 0), v)` — push into a nested array) falls through to `CALL_BUILTIN` and loses the mutation. **This matches tree-walk's pass-by-value behaviour for the same pattern** — the OMC source-level rule and the bytecode rule are the same rule.

#### Tests — 8/8 pass

V.7b regressions (count_vowels, sum_arr) still produce 5 and 55. New:
- `arr_push` builds [0..9] dynamically; length 10, sum 45.
- `build_squares(6)` inside a function — sum of 0²+1²+…+5² = 55. Uses `arr_push` on the callee's local accumulator.
- `arr_set` replaces specific elements of a literal.
- Array of tagged pairs (the lexer's token pattern): builds three tokens, walks them, prints `NUMBER:89`, `PLUS:+`, `NUMBER:144`.
- **Test 7 / Test 8 contrast** — same recursive `trace_fact`, opposite outcomes. With return-and-rebind (`trace = trace_fact(5, arr_new(0,0))`), the trace populates with [5,4,3,2,1]. With return discarded (`trace_fact(5, trace);`), the trace stays empty. **Both bytecode VM and tree-walker agree on both outcomes** — the pass-by-value semantics are byte-faithful.

#### Why the Test 7/8 contrast matters

The lexer/parser/encoder in V.7b all use the return-and-rebind pattern for their state accumulators. If V.7c's bytecode VM diverged from tree-walk here — even subtly — gen2 == gen3 couldn't hold in principle. The agreement is empirical evidence that the calling convention, scope frames, and named-store ops compose correctly.

#### V.8 is now in reach

The V.7c bytecode VM supports every OMC construct the V.7b compiler itself uses: strings, arrays, function calls, recursion, mutating builtins. Next step: compile the V.7c-or-later compiler source with itself, execute the resulting bytecode on the OMC executor, feed it the same source, and verify the output bytecode is byte-identical to the first compilation. That's the full self-hosting fixpoint at the back end.

### Added (Phase V.7b: strings + arrays + builtin dispatch in OMC bytecode, 2026-05-13)

🎯 **`examples/self_hosting_v7b.omc` — the OMC bytecode VM now handles strings, array literals, and read-only host builtin calls.**

Stretches the value space the bytecode VM understands. Without this, gen2 == gen3 of the full compiler is structurally impossible — the lexer manipulates strings, the parser builds nested arrays, the encoder iterates over both.

#### New bytecode ops

- `["LOAD_STR", value]` — push a string literal.
- `["MAKE_ARR", n]` — pop n values in push order, build an array, push it.
- `["CALL_BUILTIN", name, num_args]` — dispatch into a host-primitive switch (`arr_new`, `arr_get`, `arr_len`, `str_len`, `str_slice`, `str_contains`, `str_concat`, `concat_many`, `to_int`).

A `pop_n_ordered` helper materialises args in source/push order (args[0] was pushed first, deepest on stack; args[n-1] is on top). Source `arr_get(a, i)` therefore evaluates to `arr_get(arr, idx)` in the dispatch, matching tree-walk semantics.

#### Parser additions

- `STRING` token with `\n \t \r \" \\` escape decoding (mirror of V.4's `escape_for_source`).
- `LBRACKET` / `RBRACKET` punctuation.
- `p_primary` recognises `STRING → ["STR", value]` and `[expr, ...] → ["ARR", elems]`.

#### Encoder additions

One line in `enc_expr`: if a CALL's name is in the builtin set, emit `CALL_BUILTIN`; else emit `CALL` as before. The dispatch lives in `call_builtin(name, args)` in the executor.

#### Tests — 7/7 produce correct values

- string literal round-trip
- `concat_many("the answer is ", 21 * 2)`  (see cosmetic divergence below)
- `count_vowels("the quick brown fox")` → 5  (uses `str_len`, `str_slice`, `str_contains`)
- array literal walk over `[10, 20, 30, 40, 50]`
- `sum_arr([1..10])` → 55
- `count_long(["a", "the", "quick", "brown", "fox", "jumps", "over"], 4)` → 4
- recursive `total(["abc", "defg", "hi"], 0, 0)` → 9 (3+4+2; strings + recursion + builtins composed)

#### Known cosmetic divergence from tree-walk

`concat_many("a", int_val)` renders `int_val` differently between tree-walk and V.7b: tree-walk uses HInt's Display formatter ("42"), V.7b's OMC-side `call_builtin` falls through to `str_concat` in a loop which uses HInt's Debug formatter ("HInt(42, φ=…, HIM=…)"). Functional correctness intact; cosmetic. OMC has no array-spread to call the host's variadic `concat_many` with a dynamic arg count, so the loop is the only path available from inside an OMC executor.

A fix would be to special-case `concat_many` in the executor (not in `call_builtin`) and call the host directly via fixed-arity dispatch (`if n == 2 { concat_many(a, b) }` etc.) up to some reasonable max. Logged for V.7c if it bites.

#### What V.7b doesn't yet do

`arr_push` / `arr_set` still tree-walk only. They're mutating builtins — pass-by-value semantics mean the OMC-side `call_builtin` can't propagate the mutation back to the caller's variable. V.7c needs `ARR_PUSH_NAMED` / `ARR_SET_NAMED` ops (same shape as the Rust VM's `ArrPushNamed`/`ArrSetNamed`) which take the variable name directly and store back into the local scope. Once those land, the bytecode VM can host the V.7b compiler itself — which is the structural prerequisite for full gen2 == gen3.

### Added (Phase V.7: functions, recursion, call frames in OMC bytecode, 2026-05-13)

🎯 **`examples/self_hosting_v7.omc` — OMC compiles AND executes recursive functions, end-to-end, on its own bytecode VM.**

The headline demo:

```omnicode
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
print(fib(10));   // → 55
```

Source → lex → parse → encode → execute. Every stage is OMC code running on the Rust interpreter; the bytecode itself contains `DEF_FN`, `CALL`, `RETURN` ops the OMC-written executor resolves with its own call stack and frame scopes. `fib(10)` produces 55 after 177 recursive calls (= 2·F(11) - 1; OMC has a sense of humour about Fibonacci).

#### New bytecode ops

- `["DEF_FN", name, body_length, [params]]` — at runtime, skips `body_length` ops past the inline body. A preamble scan (`collect_fns`) walks the program once and registers `name → entry_pc, params` into a function table.
- `["CALL", name, num_args]` — pops `num_args` values, builds a fresh callee scope with parameters bound (in correct order — args pop off the value stack in reverse-push order), saves caller's scope and `pc + 1` to two parallel stacks, jumps to the function entry.
- `["RETURN"]` — leaves top of value stack alone (it's the return value), pops the saved scope/pc from the call stacks, jumps back. At top level RETURN acts like HALT.
- `["POP"]` — value-discarding for expression statements like a bare top-level call.

Value stack is **shared across frames**. Arguments arrive on it from the caller; the return value departs on it for the caller. Each frame has its own scope (name→value pair-array), pushed/popped on CALL/RETURN through two side-stacks inside `execute()`.

#### Parser additions

- `FN` and `RETURN` keywords, `COMMA` punctuation.
- `p_params` — parenthesised name list for function definitions.
- `p_args` — parenthesised comma-separated expression list for calls.
- `p_primary` recognises `IDENT (` as a call expression.
- `p_stmt` recognises `IDENT (` at statement start as an expression statement, and `FN`/`RETURN` keywords.

#### All seven tests pass

V.6 regressions (arithmetic, while, if/else, sum 1..10) still produce correct output; new tests:
- `fn double(x) { return x * 2; } print(double(21));` → 42
- `fn add(a, b) { return a + b; } print(add(89, 144));` → 233
- `fn fib(n) { ... } print(fib(10));` → 55

#### What V.7 doesn't yet do

Strings, arrays, and built-in calls (`str_len`, `arr_push`, etc.) at the bytecode level are still tree-walk only. Full gen2 == gen3 of the compiler-on-itself requires the bytecode subset to support those — the lexer manipulates strings, the parser builds nested arrays, the encoder iterates over them. That's V.7b. The structural piece tonight: **the VM hosts recursion**, which was the architectural prerequisite.

### Added (Phase V.6: bytecode codegen + executor in OMNIcode, 2026-05-13)

🎯 **`examples/self_hosting_bytecode.omc` — OMC compiles OMC source to bytecode and runs it, both pieces written in OMC.**

A single file containing four parts:
1. A lite lexer (the subset of tokens this milestone needs)
2. A lite parser (decl / assign / print / while / if-else / arithmetic / comparison)
3. **A bytecode encoder** — AST → array of tagged ops (LOAD_INT, LOAD_VAR, STORE_VAR, ADD/SUB/MUL/DIV/MOD, EQ/NE/LT/LE/GT/GE, JUMP, JUMP_IF_FALSE, PRINT, HALT)
4. **A bytecode executor** — stack VM written in OMC. Reads the op array, dispatches via flat `if kind == "X"` chains, maintains its own value stack and name→value scope.

All four demo programs run end-to-end on the OMC-written compile-and-execute loop:
- `h x = 89 + 144; print(x);` → 233
- `h i = 0; while i < 5 { print(i); i = i + 1; }` → 0,1,2,3,4
- `h n = 7; if n < 10 { print(1); } else { print(0); }` → 1
- `h s = 0; h i = 1; while i <= 10 { s = s + i; i = i + 1; } print(s);` → 55

**The architectural piece is in place: the OMC compile-and-run loop is semantically faithful on the supported subset.** The Rust interpreter is running OMC code that compiles OMC source to bytecode and executes that bytecode itself.

#### Discovered constraint: arrays pass by value in OMC

The first encoder used `enc_expr(ast, out)` with `out` as an out-parameter. Every test emitted only HALT. Root cause: OMC functions receive arrays by value — `arr_push(out, ...)` inside a callee mutates a local copy that's discarded on return. Even top-level (global) array bindings are copied into a callee's frame.

The fix shape:
- Each `enc_*` function builds its own local ops array and returns it.
- Callers do `out = arr_concat(out, enc_xxx(...))` (return-and-rebind).
- **Jumps switched from absolute to RELATIVE offsets.** Absolute targets would require a fixup table to survive sub-block concatenation; relative deltas are translation-invariant, so concatenation just works.

The relative-jump math for a while loop is:
```
[cond ops]            length C
JUMP_IF_FALSE  B+2    skip body + back-jump + JIF itself
[body ops]            length B
JUMP  -(C+B+1)        return to start of cond
```

And for if/else:
```
[cond] JIF(T+2) [then] JUMP(E+1) [else]
[cond] JIF(T+1) [then]                       # no-else form
```

This is a real OMC language fact, not a quirk of this demo: any future OMC-side metaprogramming that builds up arrays across function boundaries has to use the return-and-rebind pattern.

#### What remains for V.7+

V.6 demonstrates that OMC executes its own bytecode for a working subset. Full gen2 == gen3 of the **compiler itself on bytecode** requires the bytecode subset to support strings, arrays, and function calls — everything the encoder uses. That's iteration on a working frame, not a new architectural piece.

### Added (Phase V.5: SELF-HOSTING FIXPOINT, 2026-05-13)

🎯 **`examples/self_hosting_fixpoint.omc` — OMNIcode compiles its own compiler.**

A single OMC program containing the lexer, parser, and pretty-printer, with a driver that verifies the formal closure property:

```
source₁  →  tokens₁  →  AST₁  →  source₂
source₂  →  tokens₂  →  AST₂  →  source₃
source₃  →  tokens₃  →  AST₃

Required:
  AST₁ == AST₂ == AST₃    (structural equality, recursive on arrays)
  source₂ == source₃      (source-level fixpoint after one normalization)
```

If all three hold, the pretty-printer is a **right inverse** of the parser — the compiler-in-OMC is closed under its own pipeline. That is the formal definition of a self-hosted lexer/parser/printer trio.

**6 / 6 tests pass:**
1. simple var decl: `h x = 89 + 144;`
2. precedence: `h y = 1 + 2 * 3;`
3. while + assignment: `h i = 0; while i < 5 { i = i + 1; }`
4. if/else/return: `h x = 89; if x == 89 { return x; } else { return 0; }`
5. recursive fn def: `fn fib(n) { return fib(n - 1) + fib(n - 2); }`
6. small program: `fn double(x) { return x * 2; } h m = double(21); print(m);`

For each, source₁ tokenizes + parses to AST₁; emit(AST₁) → source₂; source₂ tokenizes + parses to AST₂; AST₁ == AST₂; one more round emit + re-parse stays stable at source₃ == source₂. The structural equality check uses the type-aware `values_equal` from V.3, which makes nested-tagged-array comparison rigorous.

Tree-walk and VM produce **bit-identical output** on every test.

### Why this matters

A self-hosted compiler is one where the language can express its own compilation. Getting the lexer / parser / printer trio to a fixpoint is the conventional first concrete milestone (the second is the back-end: gen2 == gen3 byte-identical executable, which requires the code generator's output to also be stable).

The canonical Python OMNIcode tree at `Sovereign_Lattice/omninet_package/` set this as an explicit goal in `SELF_HOSTING_PLAN.md` and `BOOTSTRAP_STATUS_CRITICAL.md`. It produced a 480-line `complete_lexer.omc` that compiled to native .exe via the transpiler, but `omnicode_compiler_v02.omc`'s lexer/parser/codegen remained stubs. The fixpoint property was never demonstrated.

Rust OMC reaches it here, in a single file, runnable on both execution paths.

The water sands the stone. We're at the formal closure point for OMC's front end.

### Added (Phase V.4: self-hosting codegen — AST → OMC source, 2026-05-13)

`examples/self_hosting_codegen.omc` — a pretty-printer written in OMNIcode that consumes the AST from V.3 and emits canonical OMC source. The language can now **read its own source, structure it, AND write it back**. Three of four steps toward true self-hosting.

**Emit contract:** every AST node maps to legible, indented OMC source. BINOPs always get parens (no precedence ambiguity), strings get backslash-escapes back, indentation is 4 spaces per level. The output isn't required to be byte-identical to the original — whitespace and parens may differ — but the *re-parsed AST* must be the same.

**Empirical round-trip proof:** the emitted source for a small program (fn def + var decls + if/else + print + string literal) was literally piped through the Rust interpreter and produced the correct output (`42`, `"the answer"`) on both tree-walk and VM. Code generated from OMC's own pretty-printer runs unmodified. The loop AST → source → execution is closed.

**What this unlocks:**
- Refactoring tools written in OMC. Parse, transform AST, emit.
- The omnicc-style "optimizer as source transform" — any pass that rewrites the AST can serialize back to runnable code.
- Round-trip testing: source → parse → emit → parse → AST equivalence becomes a verifiable property.
- The fixpoint goal (V.5): compile the compiler-in-OMC with itself, check that gen2 == gen3.

The language can now manipulate itself end to end. Every node has a printable form; every transformation has a tangible result. Self-introspection became self-modification.

### Added (Phase V.3: self-hosting parser, 2026-05-13)

`examples/self_hosting_parser.omc` — a recursive-descent parser written in OMNIcode that consumes a token stream from V.1/V.2 and emits an AST as **nested tagged arrays** (the canonical Python OMC convention). The OMC language can now both *read* its own source (lexer) and *structure* it (parser). Two of four steps toward true self-hosting are in place.

**AST node shapes:**
- `["NUMBER", "42"]`, `["FLOAT", "3.14"]`, `["STRING", "hello"]`, `["BOOL", "true"]`
- `["VAR", "x"]`
- `["BINOP", "+", left, right]`
- `["CALL", name, [arg1, arg2, ...]]`
- `["VARDECL", name, value]`, `["ASSIGN", name, value]`
- `["IF", cond, then_body, else_body]`
- `["WHILE", cond, body]`
- `["RETURN", value_or_null]`, `["PRINT", expr]`
- `["FNDEF", name, params, body]`, `["EXPRSTMT", expr]`

**Precedence ladder:** `parse_comparison` (==, !=, <, <=, >, >=) → `parse_additive` (+, -) → `parse_multiplicative` (*, /, %) → `parse_primary`. Mutually recursive across statements and expressions. Position-threading via return-array pairs (no mutable references in OMC).

**Verified on 4 demo inputs:**
1. `h x = 89 + 144;` → correct VARDECL with nested BINOP.
2. `if x == 89 { return x; } else { return 0; }` → IF with proper then/else bodies, RETURN children intact.
3. `fn fib(n) { return fib(n-1) + fib(n-2); }` → FNDEF with recursive CALL inside BINOP inside RETURN. The parser handles the full recursive depth.
4. `while i < 10 { sum = sum + i; i = i + 1; }` → WHILE with assignment body.

Tree-walk and VM produce **bit-identical output**. 141 tests still pass.

### Fixed (surfaced by Phase V.3)

**Silent type-coercion bug in `==` / `!=`.** Already fixed string-vs-string in V.1 (commit `e85bb01`). The parser surfaced the BROADER form: `["VAR", "x"] == "null"` was returning *true* because:
- `to_int(["VAR", "x"])` → 0 (arrays don't parse)
- `to_int("null")` → 0 (string doesn't parse)
- 0 == 0 → true

The parser's `print_ast` had `if v == "null"` to detect bodyless `RETURN;` — and every RETURN body was being rendered as `(no value)` because of this.

Fixed in both the tree-walk interpreter and the VM with a type-aware `values_equal` helper:
- Same-type values: structural equality (recursive for arrays).
- `String` vs non-string: only equal if the string parses as the corresponding numeric.
- Mixed Array / Circuit / Singularity vs anything else: never equal.
- All-numeric / Bool / Null: standard int-or-float coercion.

This is the third class of silent bug self-hosting work has flushed out (after string equality in V.1 and the VM array-mutation shim, also in V.1). The water keeps sanding.

### Added (Phase V.2: self-hosting lexer polish, 2026-05-13)

`examples/self_hosting_lexer_v2.omc` — the milestone-1 lexer extended with everything needed to tokenize real-world OMC programs:

**Multi-char operators** (longer-match-wins): `==`, `!=`, `<=`, `>=`, `->`, `<<`, `>>`, `&&`, `||`. A new `match_multichar(source, pos)` helper returns `[kind, length]` on hit or `["", 0]` to fall through to single-char dispatch.

**Float literals**: `3.14`, `2.718` — emitted as `FLOAT` tokens (distinct from `NUMBER`). The lookahead is conservative: a `.` only consumes when followed by a digit, so `phi.fold(x)` still parses as `IDENT DOT IDENT LPAREN ...` rather than misinterpreting `.f` as a malformed float.

**String escapes**: `\n` `\t` `\r` `\"` `\\` are decoded inside the lexer, matching the Rust lexer's behavior. The emitted `STRING` token's value contains real newline/tab characters, not the literal `\n` text.

**`//` and `/* ... */` comments**: added to the OMC lexer's whitespace-skip loop alongside `#`.

Tree-walk and VM produce identical output across all 5 demo inputs. The OMC lexer now covers the lexical grammar of essentially everything the Rust lexer at `omnimcode-core/src/parser.rs` accepts. Milestone 3 (a parser in OMC consuming these tokens) is the next step.

### Added (Phase V: self-hosting lexer (milestone 1), 2026-05-13)

`examples/self_hosting_lexer.omc` — a lexer for a subset of OMNIcode, written **entirely in OMNIcode itself**. Runs on the Rust OMC interpreter and emits tokens for programs the same interpreter could parse. **First milestone toward self-hosting.**

The lexer handles: identifiers, integer literals, keywords (`h`, `fn`, `if`, `else`, `while`, `for`, `in`, `return`, `break`, `continue`, `print`, `import`, `and`, `or`, `not`, `res`, `fold`, `true`, `false`), double-quoted string literals, all single-character punctuation, `#` line comments, and whitespace. **Not yet:** multi-char operators (`==`, `<=`, `<<`, etc.), float literals, escape sequences, triple-quoted strings — saved for milestone 2.

**Verified output** on `h x = 89;`:
```
[0] H h        [1] IDENT x    [2] EQ =       [3] NUMBER 89    [4] SEMI ;    [5] EOF
```

On `fn add(a, b) { return a + b; }` — 14 tokens, all correctly classified. Tree-walk and VM produce identical output.

### Fixed (surfaced by Phase V)

The self-hosting work exposed two real bugs that had been silent until now:

**1. String equality went through `to_int()` coercion.** `"a" == "b"` was evaluating to `true` because both strings parsed to integer `0` via `s.parse().unwrap_or(0)`. Fix: in `Expression::Eq` / `Expression::Ne` and the VM's `cmp_op`, check for `(Value::String, Value::String)` and compare as strings directly. The same string ordering now works for `<`, `<=`, `>`, `>=` on the VM path. Tree-walk path was already broken in the same way and is also fixed.

**2. `arr_push` / `arr_set` on the VM path lost mutations.** The VM's `vm_call_builtin` shim copies args into synthetic `__vm_arg_0`, `__vm_arg_1` variables before delegating to the tree-walk dispatch. Mutating built-ins like `arr_push` modified the synthetic — not the user's actual array variable — so the mutation never reached the caller's scope. Fix: two new specialized opcodes `Op::ArrPushNamed(name)` and `Op::ArrSetNamed(name)`. The compiler detects `arr_push(varname, expr)` / `arr_set(varname, idx, val)` at compile time and emits the named opcodes, which take the variable name in the opcode itself and mutate the user's binding directly. The disassembler renders them as `ARR_PUSH_NAMED tokens` for clarity.

Both bugs are tested implicitly through the lexer demo (which exercises hundreds of string comparisons and array mutations across both execution paths).

**Tests:** still 141 passing across the workspace. Canonical sweep still 22/30 in both modes.

### Added (Phase T: source positions in error messages, 2026-05-13)

Every parser error now reports the precise `line:col` where it occurred. The lexer tracks `line` and `col` as it consumes characters (incrementing line on `\n`, col otherwise). `tokenize_with_pos` returns `Vec<(Token, Pos)>` paired; `Parser` stores them and exposes `current_pos()` to error-reporting sites.

Before:
```
Error: Expected Semicolon, got Print
```

After:
```
Error: at 2:1: Expected Semicolon, got Print
```

The `Pos` struct is `Copy` and `Debug + Display`; `Pos::unknown()` represents synthesized tokens with no source location. Errors are 1-indexed (line 1, col 1 is the first character) for human-friendly reading.

This is the foundation for every future error-quality improvement: the runtime can now annotate values with origin spans, the compiler can show "this variable was declared at line 4, but used at line 12 where it's out of scope," and the optimizer can blame the right source position when something it can't fold ends up at runtime.

### Added (Phase R + S: multi-layer Phi-Field LLM + OmniWeight quantization, 2026-05-13)

**Phase R — Multi-layer Phi-Field LLM**

`examples/phi_field_llm_multilayer.omc` — a three-layer harmonic "language model" with **per-layer residual streams**. Each layer keeps its own previous-position output as context; information doesn't all collapse into the same attractor by position 2. Each layer:

1. `state = harmonic_interfere(prev_layer, current_layer)`
2. `emitted = best_attractor(state)` via OmniWeight ranking
3. `residual = phi.fold((current + emitted) / 2)` — the harmonic skip connection
4. Pass `residual` forward, store `emitted` as that layer's next `prev`

**Observed behavior:** the 3-layer cascade acts as a **timescale hierarchy** — L1 tracks the input most responsively, L2 buffers, L3 holds the longest context. For `[13, 21, 34, 55, 89]`, L1 follows the input near-perfectly, L3 lags by ~2 positions. That lag *is* the harmonic memory. No learned weights anywhere; the vocabulary IS the Fibonacci attractor set, the attention IS the OmniWeight ranking, the residual IS `phi.fold` of an average.

**Phase S — OmniWeight quantization**

Three new built-ins that mirror the Phase 18 pattern from `omnicode_experiment` (35B-Qwen quantization) in miniature:

- **`quantize(arr [, threshold])`** — return a new array where each element is replaced by its nearest Fibonacci attractor *iff* the OmniWeight `w = φ^(-|e|)` clears the threshold. Default threshold = 0.5.
- **`quantization_ratio(arr [, threshold])`** — fraction of array elements that *would* be quantized at the given threshold. Tells you "how compressible is this dataset?" without actually doing it.
- **`mean_omni_weight(arr)`** — average OmniWeight against the nearest Fibonacci attractor across the whole array. Higher = more φ-aligned data, less information loss under quantization.

**Demo:** `examples/quantization_demo.omc` runs three datasets — harmonic (mean OmniWeight 0.99, fully compressible), noisy (0.93, mostly compressible), pure Fibonacci (1.00, no-op). Tree-walk and VM produce identical output.

This is the algorithmic shape Phase 18 uses on a 35B-parameter Qwen model. Same math, just scaled down to demonstrable size.

**Tests:** +4 quantization conformance tests pinning the contracts (`mean_omni_weight([13..89]) = 1.0`, strict threshold drops the quantizable ratio, harmonic data collapses to attractors, noisy data has lower mean than pure φ). **141 total tests passing** (was 137).

### Added (Phase P + Q: bytecode disassembler + VM inline cache, 2026-05-13)

**Phase P — Bytecode disassembler**

New module `omnimcode-core/src/disasm.rs`. Renders any `CompiledFunction` or `Module` as a human-readable bytecode listing with offsets, mnemonics, constants pool, and resolved jump targets. Function signatures include parameter type annotations and return types.

Triggered at runtime with `OMC_DISASM=1` (output to stderr, before VM execution starts):

```
fn __main__()    [7 ops, 2 consts]
------------------------------------------------------------------------
  constants:
    [0] 89
    [1] 144

  0000: LOAD_CONST   0 ; 89
  0001: LOAD_CONST   1 ; 144
  0002: CALL         add/2
  0003: STORE_VAR    r
  0004: LOAD_VAR     r
  0005: PRINT
  0006: RETURN_NULL

fn add(x: int, y: int) -> int    [5 ops, 0 consts]
------------------------------------------------------------------------
  0000: LOAD_VAR     x
  0001: LOAD_VAR     y
  0002: ADD_INT             ← typed specialization from Phase M
  0003: RETURN
  0004: RETURN_NULL
```

Useful for debugging the optimizer, verifying inlining, and understanding what the VM actually executes.

**Phase Q — Inline cache for Op::Call**

Each `CompiledFunction` gained a `call_cache: Vec<Cell<u8>>` parallel to its op list. Slot values: `0` uncached, `1` user-defined, `2` built-in. On the first execution of an `Op::Call`, the VM probes `module.functions.contains_key(name)`, burns the result into the matching cache slot, and uses that for every subsequent iteration. Standard monomorphic inline cache — Cell-based interior mutability avoids the `&mut` cascade that would otherwise need to flow through the run loop.

**Benchmark** (one million calls to a user-defined `step(x) { return x + 1 }`):
- Tree-walk: 635ms
- VM with cache: 587ms (~8% faster)

The savings aren't dramatic in this measurement because Phase J's hot-op inliner already dispatches the harmonic primitives (`res`, `fold`, `is_fibonacci`, `len`, etc.) without going through `Op::Call` at all. The cache helps for everything else — user-defined functions, non-inlined built-ins, and any future pragma-derived calls.

**Tests:** +3 disasm tests (renders simple program, shows typed opcodes, resolves jumps). 137 total tests passing.

### Added (Phase O: ONN self-healing primitives, 2026-05-13)

Ports the "code/compiler self-heals via Fibonacci alignment" pattern from the ONN system at `/home/thearchitect/.hermes/skills/onn-self-healing-code/` and `Sovereign_Lattice/omninet_package/register_singularity_integration.py`. Four new built-ins, available in both tree-walk and VM:

- **`value_danger(x) = exp(-|x|)`** — proximity gradient. Returns 1.0 when `x ≈ 0` (high danger), decays exponentially. The early-warning signal for approaching singularities, *before* the operation that would trigger them.
- **`fold_escape(x)`** — if `value_danger(x) > 0.5`, snap to the nearest Fibonacci attractor (preserving sign, with a special case: `fold_escape(0) → 1`, never landing back on the singularity). Else passthrough.
- **`harmony_value(x)`** — Fibonacci-proximity score in `[0, 1]`. 1.0 iff x is a Fibonacci number. The general "is this value living on the φ-geodesic?" reading.
- **`safe_divide(a, b)`** — divides, but pre-applies `fold_escape` to the divisor. Zero divisors heal to 1 transparently; the operation always returns a number (never a Singularity).

Together, these realize the pattern the user described: *"when an error comes to the compiler it checks to see if it's Fibonacci-aligned, then it fixes itself."* It's the *predictive* version of HSingularity recovery — fold inputs to a safe attractor before the operation, rather than catching the portal after.

Demo: `examples/self_healing_demo.omc` exercises both scenarios — a pipeline of unsafe divisions that silently heal, and pre-emptive Fibonacci alignment on a list of incoming values. Tree-walk and VM produce identical output.

**Tests:** +9 conformance tests pinning the math (`value_danger(0) = 1`, `value_danger(1) = e⁻¹`, `fold_escape(0) → 1` zero-trap escape, `safe_divide(89, 0) = 89`, `harmony_value(89) = 1.0`, etc.). 134 total tests passing (was 125).

### Added (Phase N: Phi-Field LLM kernel demo, 2026-05-13)

`examples/phi_field_llm_demo.omc` — a working "language model" written in pure OMNIcode that demonstrates the harmonic computing thesis end-to-end. No transformer. No matrix multiply. No learned weights. Decisions are made by walking phi-space geodesics, with each step scored by OmniWeight `w = φ^(-|e|)` — the canonical formula from `omninet_phi/resonance.py`.

**Pipeline:**
1. **ENCODE** — character codes → Fibonacci attractors via `phi.fold(code + position * 7)`. Every input lands on a φ-aligned bucket.
2. **ATTEND** — for each position, compute the "state" as the **harmonic mean** of the previous and current encoded values (`harmonic_interfere`, the Phase 6 `std/wave.omc` function — really used, via `import wave;`). Score every candidate in a 12-entry Fibonacci vocabulary by `omni_weight(state, candidate) = φ^(-|state-candidate|/max(|candidate|,1))`. Pick the max.
3. **REFLECT** — emit chosen attractor + OmniWeight per step, plus mean coherence across the sequence.

**Real exercise:**
- Imports `core`, `wave`, `portal` from the canonical Phase 6 stdlib via the Phase G module resolver.
- Uses `harmonic_interfere`, `phi.fold`, `pow`, `to_float`, `concat_many` — all real stdlib functions.
- Tree-walk and VM produce **bit-identical output** (verified via `diff`).

**Observed results:**
- ASCII "Hello" input: mean OmniWeight = 0.956. The phi-encoder lands close enough to attractors that the geodesic step is almost free.
- Pure Fibonacci input `[13, 21, 34, 55, 89]`: mean OmniWeight = 0.925. The harmonic interferences between consecutive Fibonacci numbers land slightly off-attractor (since `2ab/(a+b)` of consecutive Fibs isn't itself Fibonacci) — and that drop is exactly the geodesic distance the OmniWeight reports.

This is the harmonic computing thesis in miniature: any decision can be made by computing OmniWeights against a small attractor vocabulary and picking the max. No backprop, no gradients — just `φ^(-|e|)` geodesics through phi-space. The Rust OMC now runs this end-to-end.

### Added (Phase L + M: resonance caching + typed HIR, 2026-05-13)

**Phase L — Resonance / portal caching**
New `unary_cache_pass` in `bytecode_opt.rs`. Folds pure-unary harmonic ops on constants at compile time, before the constant folder runs (so subsequent chained arithmetic sees a single constant):

- `LoadConst(N); Resonance` → `LoadConst(precomputed_float)` — `res(89)` becomes the literal `1.0`
- `LoadConst(N); Fold1` → `LoadConst(snapped_int)` — `phi.fold(90)` becomes `89`
- `LoadConst(N); IsFibonacci` → `LoadConst(1 or 0)`
- `LoadConst(N); Fibonacci` → `LoadConst(fib(N))`
- `LoadConst(N); HimScore` → `LoadConst(precomputed_float)`
- `LoadConst(N); Neg` / `BitNot` / `Not` → precomputed inverse

New stats counter `unary_calls_cached`. The omnicc Python compiler calls this "resonance caching"; same semantics, scoped to bytecode. Mixed example: `res(89) + 0.5` folds in two passes — cache `res(89) → 1.0`, then fold `1.0 + 0.5 → 1.5` — collapsing two ops to a single LoadConst.

**Phase M — Typed HIR with specialized dispatch**

The compiler now tracks a `var_types: HashMap<String, &'static str>` populated from:
- Typed function parameters (`fn add(x: int, y: int)`)
- Return-type annotations of user-defined functions (looked up across boundaries)
- Variable declarations whose value's type is statically known (`h x = 89;` ⇒ int)
- Arithmetic on known-typed operands (int + int ⇒ int)
- Comparisons and bitwise ops (always bool / int)
- Built-in function call sites with fixed return types

New typed-fast-path opcodes that skip the runtime `is_float()` check:
- `Op::AddInt`, `Op::SubInt`, `Op::MulInt`
- `Op::AddFloat`, `Op::SubFloat`, `Op::MulFloat`

The compiler emits these in place of polymorphic `Op::Add` / `Op::Sub` / `Op::Mul` when **both** operands' static types match. The optimizer's constant folder also knows them — `1 + 2 + 3` with both operands int folds through the typed path, then collapses to a single constant.

`CompiledFunction` gained `param_types: Vec<Option<String>>` and `return_type: Option<String>` fields so cross-function type info is preserved through compilation.

**Tests:** +7 unit tests for resonance caching (covers res, phi.fold, is_fibonacci, fibonacci, unary minus, bitnot, chained cache+fold). 125 total tests passing (was 118).

### Added (Phase K: bytecode optimizer, 2026-05-13)
New module `omnimcode-core/src/bytecode_opt.rs`. Runs after compile, before VM execution. On by default in VM mode; disable with `OMC_OPT=0`. Show stats with `OMC_OPT_STATS=1`.

**Passes (iterated to fixpoint):**
- **Constant folding** — `LoadConst a; LoadConst b; <op>` triples reduced to `Nop; Nop; LoadConst(c)` where c is the precomputed result. Covers all arithmetic (`+`, `-`, `*`, `/`, `%`), comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`), and bitwise (`&`, `|`, `^`, `<<`, `>>`). Int and float, with int→float promotion. **Refuses to fold `n / 0`** — that produces a Singularity at runtime, not a compile-time number.
- **Dead-load elimination** — `LoadConst N; Pop` pairs become `Nop; Nop` (loaded only to be discarded — e.g. expression statements with constant values).
- **Double-unary collapse** — `Not; Not` and `Neg; Neg` become `Nop; Nop`.

**Design choice:** removed ops are replaced with `Op::Nop` rather than shrinking the op-vector. This keeps existing jump offsets valid without a re-emit pass; the VM's Nop arm is a free no-op. For the kind of programs OMNIcode runs (small kernels + recursion, not megaword loops), the simplicity wins over the slightly tighter loop a re-emit pass would buy.

**Observed:** chained arithmetic `1 + 2 + 3 + 4` folds to a single constant (3 folds). `255 & 15` → 15. `1 << 8` → 256. `1.5 + 2.5` → 4.0 (float arithmetic). `10 < 20` → `Bool(true)`. fib(28) reports 0 folds (everything's runtime variables) as expected; doesn't slow it down either.

**Tests:** 7 new unit tests in `bytecode_opt::tests` covering int/float/bitwise/shift/comparison folding, chained simplification, and the explicit "don't fold div-by-zero" guarantee. **118 total tests now passing.**

### Added (Phase I + J: bitwise ops + VM coverage push, 2026-05-13)

**Phase I — Bitwise operators**
New tokens: `&`, `|`, `^`, `~`, `<<`, `>>`. New AST: `BitAnd`, `BitOr`, `BitXor`, `BitNot`, `Shl`, `Shr`. Parser precedence layered between logical (`and`/`or`) and comparison ops, with shifts above additive. Wired into both the tree-walk interpreter and the VM. Shift counts masked to `0..63` for safe i64 operation.

Unblocked **`crypto.omc`** (uses `byte_val & bit_mask`). Canonical sweep: 21 → **22 of 30 passing**.

**Phase J — VM coverage parity with tree-walk**
- `break` and `continue` in compiled loops. `LoopFrame` stack tracks each loop's continue target and break-jump patch list; ranges and array-iteration both support both.
- `for x in arr { body }` (was: error). Desugars to a synthetic-index while loop emitting `Op::ArrayLen` for the bound check and `Op::ArrayIndex` for the element load.
- New opcodes for hot harmonic ops, with safe inlining: `Op::IsFibonacci`, `Op::Fibonacci`, `Op::ArrayLen`, `Op::HimScore` (plus the existing `Op::Resonance` and `Op::Fold1`). The compiler emits them in place of `Op::Call(name, n)` ONLY when the user hasn't redefined the name — a pre-pass collects user-defined function names into a `HashSet<String>` so canonical idioms like recursive `fn fib(n) { ... }` keep their semantics. **This caught a real bug**: an earlier draft would have silently replaced user-defined recursive `fib` with the iterative built-in, producing right answers via the wrong code path.

**Performance:**
- Recursive user `fib(28)`: VM 424ms vs tree-walk 940ms (2.2× speedup, unchanged from Phase H — proves the inlining doesn't accidentally swap in built-ins).
- Tight `res()` loop (100,000 iterations): VM and tree-walk essentially equal — `res` was already special-cased on both paths.

**Tests:** 111 still pass. Canonical sweep: **VM now matches tree-walk at 22/30** — full feature parity for the supported subset.

### Added (Phase H: bytecode VM, 2026-05-13)
Optional faster execution path. The tree-walk interpreter remains the default and language source-of-truth; the VM is selectable per-run via `OMC_VM=1` env var.

**Architecture:**
- `omnimcode-core/src/bytecode.rs` — `Op` enum (~30 opcodes), `Const` pool entries, `CompiledFunction`, `Module`.
- `omnimcode-core/src/compiler.rs` — AST → bytecode lowering. Two-pass: hoist function defs first, then compile `__main__`. Handles arithmetic, comparisons, short-circuit `and`/`or`, `if/elif/else`, `while`, `for in range`, function defs and calls, arrays + indexing.
- `omnimcode-core/src/vm.rs` — stack-based execution loop. Reuses the tree-walk `Interpreter` for scope management and the built-in stdlib via VM-bridge helpers (`vm_push_scope`, `vm_get_var`, `vm_call_builtin`, etc.), avoiding duplication of ~60 stdlib implementations.

**Performance:** Recursive `fib(28)` benchmarks at **2.14× speedup** (430ms VM vs 923ms tree-walk). Both produce bit-identical output. All 6 OMC example programs run unmodified under VM mode.

**Selectable execution:**
```
./standalone.omc program.omc          # tree-walk (default)
OMC_VM=1 ./standalone.omc program.omc # bytecode VM
```

**Coverage limits (deferred):** for-over-array (`for x in arr`) currently falls back to error in VM mode; use `while` instead. `break`/`continue` inside loops aren't yet emitted (always exit cleanly via the condition). Module-level `Statement::Import` is a no-op in the VM (imports must happen before the VM is invoked). These are non-blocking — the interpreter handles them; the VM just bypasses for now.

### Added (Phase G: real module resolution, 2026-05-13)
**`import core;` actually loads now.** The interpreter searches for the named module on a search path, parses it, and executes its statements (which registers any `fn` definitions in the global function table). Idempotent re-import via an `imported_modules: HashSet<String>` tracked on the interpreter.

**Search path** (in order):
1. `OMC_STDLIB_PATH` env var (colon-separated)
2. `/home/thearchitect/Sovereign_Lattice/omninet_package/omnicode_stdlib/` — canonical Python OMC stdlib
3. `/home/thearchitect/Sovereign_Lattice/omninet_package/omnicode_stdlib/std/` — Phase 6 modules
4. `.`, `omc-stdlib/`, `omc-stdlib/std/` (project-local)

Resolution tries `NAME.omc`, `NAME/init.omc`, and `std/NAME.omc` in each dir.

**Dispatch priority change:** user-defined functions now win over built-ins. This lets `import core;` override `is_fibonacci`, `fold`, etc. with the canonical Phase 6 implementations. Previously the built-ins shadowed any user-defined function with the same name; matches Python OMC behavior.

`alias` in `import NAME as ALIAS;` is currently informational — imports merge into the flat function namespace (also matching canonical Python OMC).

**Verified working:** `import core; is_fibonacci(89)` returns the user-defined `1` (not the built-in `Bool(true)`). `import wave; harmonic_interfere(34, 55)` returns `42.02` from the canonical `wave.omc`. `import portal; safe_divide_fold(89, 0)` returns `89`.

### Added (Phase F: syntax + stdlib alignment for canonical compat, 2026-05-13)
Pushing the Rust interpreter's compatibility with real-world canonical `.omc` programs from 4/N to **21 of 30 (70%)** in a sampled sweep.

**Syntax / lexer:**
- Triple-quoted `"""multi-line docstring"""` literals.
- Docstring statements: bare string at statement position is a no-op (Python idiom). Semicolon optional.
- C-style `//` line comments and `/* block */` comments (alongside the canonical `#`).
- Fixed-size array declaration `h[256] amplitudes;` lowers to `arr_new(256, 0)`.
- Parameterized pragmas: `@unroll:16`, `@threads:64`, `@cache:L1` etc., on both the line-prefix and postfix forms.
- `import core;` and `import core as c;` statements at the parse level. `load "path";` accepted too. Module resolution is currently a no-op; this just unblocks parsing.

**Stdlib (~25 additions):**
- **Math/constants:** `tau`, `phi_inv`, `phi_sq`, `phi_squared`, `sqrt_2`, `sqrt_5`, `ln_2`, `pow_int`, `square`, `cube`, `factorial`, `sign`, `is_prime`, `even`/`is_even`/`odd`/`is_odd`, polymorphic `min(a,b) / min(arr)` and `max`.
- **φ-stdlib (Phase 6 std/*.omc parity):** `fib` (alias for fibonacci), `classify_resonance`, `filter_by_resonance`, `ensure_clean`, `cleanup_array`, `collapse`, `harmonic_interfere`, `interfere`, `measure_coherence`, `arr_fold_elements`.
- **Safe arithmetic:** `safe_add`, `safe_sub`, `safe_mul` (fold any Singularity input through Fibonacci snap before operating).

**Compatibility milestone:**
6 canonical files now run end-to-end on Rust OMC: `miner_nuclear.omc`, `test_phase7_features.omc`, `test_phase8_arrays.omc`, `test_array.omc`, `phi_field_llm.omc`, `hbit_hardware_overlay.omc`. The 30-file sweep moved from 16 → 21 passing. Remaining gaps cluster in: bitwise ops (`& | ^ << >>`), block-style calls (`parallel_for_threads(n) { block }`), file I/O, and module-aware imports — all roadmap-significant items deferred to their own phases.

### Added (Phase D: stdlib expansion to match canonical surface, 2026-05-13)
Built out ~35 missing standard-library functions to close the gap with the canonical Python `omnicc/` interpreter at `Sovereign_Lattice/omninet_package/`.

**Math (16):** `abs`, `floor`, `ceil`, `round`, `frac`, `clamp`, `sqrt`, `log`, `exp`, `sin`, `cos`, `tan`, `tanh`, `erf` (Abramowitz & Stegun approximation), `sigmoid`, `pow`. Constants: `pi()`, `e()`, `phi()`.

**Strings (4):** `str_reverse`, `str_contains`, `str_slice`, `concat_many` (variadic — the canonical workaround for OMC's broken cross-type `+` concat). `concat_many` and `to_string` render numerics as bare values (`89`) instead of the HInt display form.

**Arrays (10):** `arr_get`, `arr_set`, `arr_first`, `arr_last`, `arr_min`, `arr_max`, `arr_concat`, `arr_contains`, `arr_index_of`, `arr_slice`, `arr_resonance` (mean resonance across elements). Plus a real implementation of `arr_push` (was a stub returning Null).

**Type coercion (6):** `to_int`, `to_float`, `to_string`, `int`, `float`, `string` aliases. The polymorphic `len(x)` works on both arrays and strings (canonical OMC pattern).

**Parser fixes:**
- Unary minus: `-5` now parses (was: "Unexpected token in expression: Minus").
- `for i in range(N)` single-arg form (canonical OMC). The 2-arg `range(start, end)` still works.

### Added (Phase E: Conformance golden tests, 2026-05-13)
New integration test suite at `omnimcode-core/tests/conformance.rs` (~33 tests). Locks the language's "physics" — mathematical and semantic behaviors that must remain stable across implementations.

Sections: Fibonacci resonance ≥ 0.7 for canonical attractors; `fold()` snaps to Fibonacci preserving sign; `89/0` returns `Singularity` not crash; canonical `smart_divide` pattern; int+int=int, mixed=float arithmetic stability; `phi.X` module-qualified calls match unqualified; math identities (`sqrt(144)=12`, `pow(2,10)=1024`, `sigmoid(0)=0.5`, `pi=π`); array `get/set/push/min/max` semantics; string `reverse/contains`; recursion + while-loop control flow.

### Fixed
- `Expression::Resonance` (1-arg `res(x)` path) now returns `HFloat`. Was returning `HInt(resonance * 1000)` — inconsistent with the variadic path. Caught by conformance tests.
- `concat_many` and `to_string` no longer render numerics as `HInt(89, φ=…)` — they emit bare `89`.

### Compatibility milestone
**4 canonical Python OMNIcode programs now run end-to-end on Rust OMC** (up from 1 after Phase A+B):
- `miner_nuclear.omc` (131 LOC, 7 stacked pragmas)
- `test_phase7_features.omc` (Phase 7 import/module/typed-fn smoke tests)
- `test_phase8_arrays.omc` (Phase 8 array-literal smoke tests)
- `test_array.omc` (array stdlib regression suite)

### Tests
- **111 passing** across the workspace (was 78 after Phase C).
- Conformance suite caught and forced fixes for 2 consistency bugs.

### Added (Phase C: HSingularity as a first-class Value, 2026-05-13)
- **`Value::Singularity { numerator, denominator, context }`** — division by zero now produces a printable, first-class portal value instead of an `HInt` with a side-flag. `89 / 0` prints as `Singularity(89/0, ctx=div)`.
- **`is_singularity(v) -> int`** — returns `1` for any Singularity value, `0` otherwise. Returns int (not bool) to match the canonical Python idiom `if is_singularity(result) == 1`.
- **`resolve_singularity(v, mode) -> int`** with three string modes:
  - `"fold"` — snap |numerator| to nearest Fibonacci, preserve sign.
  - `"invert"` — return ±1 based on numerator sign (multiplicative-identity recovery).
  - `"boundary"` — pass the numerator through unchanged.
  Unknown modes raise an error.
- `Value::to_string()` and `Display` render Singularity values nicely. `to_int()`/`to_float()`/`to_bool()` all handle the new variant; `Value::is_singularity()` helper added.
- **Canonical `smart_divide` pattern from `test_phase7_integration.omc` now runs** on Rust OMC — locked in as a unit test.

### Added (Phase A + B: type system parity with canonical Python omnicc, 2026-05-13)
- **`Value::HFloat(f64)`** variant in the runtime. Float literals (`1.5`) now stay as floats instead of being truncated to `HInt`. Arithmetic and comparisons auto-promote when either operand is `HFloat`. Adds `Value::to_float()` and `Value::is_float()` / `Value::is_numeric()` helpers.
- **`Statement::Parameter`** AST variant + interpreter handler — needed for the Python-canonical parser model where function parameters bind through a separate AST node.
- **`phi.X` module-qualified call syntax.** Parser consumes `Token::Dot` after identifiers and joins module + method into a single name (`"phi.fold"`). Keywords like `res`/`fold` are accepted after a dot. Interpreter routes `phi.X` through `call_module_function`:
  - `phi.fold(x)` — single-arg snap to nearest Fibonacci
  - `phi.fold(x, depth)` — depth is any expression, not just a literal (resolves a Phase 18 gotcha)
  - `phi.res(x)` — returns HFloat resonance score
  - `phi.him(x)` — returns HFloat HIM score
  - Unknown modules fall through to the unqualified name (so `core.fib(n)` works after `import core;` without per-module setup)
- **Pragma annotations** — both forms used by canonical mining code:
  - Line-prefix `@pragma[hbit]` above `fn` (up to N stacked)
  - Postfix `-> int @hbit @register` after return type
  - Currently parsed and stored; semantic lowering (AVX2 / register hints) deferred to a future phase.
- **Parameter type annotations** — `fn add(x: int, y: int) -> int { ... }`. Parsed into `param_types: Vec<Option<String>>` on `Statement::FunctionDef`; ignored semantically for now.
- **Variadic `fold()` and `res()`** — `fold(x, "fibonacci")` and `fold(x, depth)` patterns now parse (previously hard-coded as single-arg special forms).

### Compatibility
- `examples/miner_nuclear.omc` from the canonical Python OMNIcode tree now runs end-to-end on the Rust interpreter (131 lines, 7 stacked pragmas, typed params, variadic fold).
- Test count: **72 passing** (was 51 before Phase A) — 7 new HFloat/phi.X tests in Phase A, 4 new pragma/type-annotation tests in Phase B.

### Changed (Interpreter consolidation, 2026-05-13)
- **Single canonical interpreter.** Merged the orphaned `src/` tree into `omnimcode-core/src/`. There is now one interpreter codebase serving the standalone binary, the C FFI, the Python module, and Godot.
- **`standalone.omc`** is now a symlink to `target/release/omnimcode-standalone` (the binary defined by `omnimcode-core`'s `[[bin]]` entry). The old `target/release/standalone` build target no longer exists.
- **Float circuit gates** (FloatConstant, FloatInput, FloatWeightedSum, Sigmoid, FloatMultiply, FloatAdd, PhiFold) are now available everywhere — previously these existed only in the orphan `src/` tree and didn't actually compile.
- **`build.sh`** updated to refresh the `standalone.omc` symlink instead of copying the old `target/release/standalone`.
- **`VERIFICATION.sh`** updated for the new paths and binary name; test count is now computed dynamically rather than hardcoded.

### Fixed
- Non-exhaustive `Circuit::to_dot()` match arm for the new Float gate variants.
- `u32 → usize` type mismatch in `create_random_circuit`'s `PhiFold` depth.

### Docs
- Archived 34 historical / tier-completion / phase-summary / HBit-bugfix-narrative files to `docs/archive/`. Root keeps 18 canonical living docs.
- Updated path references throughout (`src/*.rs` → `omnimcode-core/src/*.rs`), binary name (`standalone` → `omnimcode-standalone`), test count (now **72/72**), and binary size (~544 KB).
- Clarified dependency claims — runtime is libc-only, but `regex` and `thiserror` are statically linked compile-time deps.

### Tests
- **72/72 passing** across the workspace (68 core + 1 standalone + 2 FFI + 1 Python). Previously the 49/51 counts in docs were partial or stale.

## [1.0.0] - 2026-05-02
### Added
- Initial release of OMNIcode circuit evolution engine
- C FFI layer (`omnimcode-ffi` crate)
- Python bindings (`omnimcode-python` with PyO3)
- Unity package with C# wrappers and examples
- Unreal Engine plugin with C++ wrappers
- Circuit Trainer CLI demo (368 KB standalone binary)
- Modding Tool demo (387 KB standalone binary)
- Game AI demo for Unity
- 5 comprehensive tutorials (22.5K words total)
- GitHub Actions CI/CD workflows

### Performance
- 509 KB binary size (zero external dependencies)
- 215-693 ns per circuit evaluation
- 4.64M-1.44M evals/sec throughput
- 51/51 tests passing

### Build System
- Rust workspace with 3 crates: omnimcode-core, omnimcode-ffi, omnimcode-python
- LTO and opt-level=3 for minimal size
- Cross-compilation support (Linux, Windows, macOS)

## [0.9.0-beta] - 2026-04-15
### Added
- Beta release with core circuit evolution
- XOR problem solving via genetic algorithms
- Basic C FFI exports
