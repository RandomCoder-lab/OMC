# Changelog

All notable changes to OMNIcode will be documented in this file.

## [Unreleased]

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
