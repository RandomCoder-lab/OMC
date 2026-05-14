# Changelog

All notable changes to OMNIcode will be documented in this file.

## [Unreleased]

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
