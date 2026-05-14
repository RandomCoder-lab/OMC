# Changelog

All notable changes to OMNIcode will be documented in this file.

## [Unreleased]

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
