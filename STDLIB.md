# OMC Standard Library Reference

Every built-in function available in OMNIcode, organized by category. Function signatures use `name(arg: type, ...) -> type` notation. Types: `int`, `float`, `string`, `bool`, `array`, `null`, `singularity`, `circuit`.

**HInt vs int.** All integer values in OMC carry harmonic metadata (`φ`-resonance, HIM score) — they're `HInt`s under the hood. The signatures below use `int` for readability; the metadata is computed automatically and surfaces only when you `print` a raw value.

**Where this lives.** Built-ins are implemented in `omnimcode-core/src/interpreter.rs`. The Rust VM's compile-time type inference is in `omnimcode-core/src/compiler.rs`. To add a new built-in, see `DEVELOPER.md`.

---

## Quick reference (alphabetical)

For "I know the name, what does it do" lookups. Skip to the category sections for "I want to do X, what should I reach for".

```
abs                  arr_filter           arr_set
arr_all              arr_find             arr_slice
arr_any              arr_first            arr_sort
arr_concat           arr_fold_elements    arr_sum
arr_contains         arr_from_range       arr_unique
arr_get              arr_index_of         arr_zip
arr_join             arr_last             boundary
arr_len              arr_map              ceil
arr_max              arr_min              clamp
arr_new              arr_push             classify_resonance
arr_reduce           arr_resonance        cleanup_array
arr_reverse          collapse             concat_many
cos                  cube                 e
ensure_clean         erf                  even
exp                  factorial            fib
fibonacci            file_exists          filter_by_resonance
float                floor                fold
fold_escape          frac                 gcd
harmonic_checksum    harmonic_interfere   harmonic_partition
harmonic_read_file   harmonic_sort        harmonic_split
harmonic_write_file  harmony_value        int
interfere            invert               is_even
is_fibonacci         is_odd               is_prime
is_singularity       lcm                  len
ln_2                 log                  max
mean_omni_weight     measure_coherence    min
now_ms               odd                  phi
phi_inv              phi_sq               phi_squared
pi                   pow                  pow_int
println              print_raw            quantization_ratio
quantize             random_float         random_int
random_seed          read_file            res
resolve_singularity  round                safe_add
safe_arr_get         safe_arr_set         safe_divide
safe_mul             safe_sub             sigmoid
sign                 sin                  sqrt
sqrt_2               sqrt_5               square
str_chars            str_concat           str_contains
str_ends_with        str_index_of         str_join
str_len              str_lowercase        str_pad_left
str_pad_right        str_repeat           str_replace
str_reverse          str_slice            str_split
str_starts_with      str_trim             str_uppercase
string               tan                  tanh
tau                  to_float             to_int
to_string            type_of              value_danger
write_file
```

Total: ~135 named builtins, plus `print` as a statement keyword.

---

## Strings

| Function | Signature | Notes |
|---|---|---|
| `str_len(s)` | `string -> int` | **Byte count** (not char count). For loop bounds, use `str_chars`. |
| `str_chars(s)` | `string -> int` | Char count (UTF-8 scalar values). Pairs with `str_slice`. |
| `str_slice(s, start, end)` | `string, int, int -> string` | **Char-indexed**. Out-of-range bounds clamp; never errors. |
| `str_concat(a, b)` | `string, string -> string` | Two-arg only; for more, use `concat_many`. |
| `concat_many(...)` | `... -> string` | Variadic, renders numerics as bare values. |
| `str_split(s, sep)` | `string, string -> array<string>` | Empty separator splits into individual chars. |
| `str_join(arr, sep)` | `array, string -> string` | Mixed-type elements stringify via Display. |
| `str_trim(s)` | `string -> string` | Strips both leading and trailing whitespace. |
| `str_replace(s, old, new)` | `string, string, string -> string` | Replaces all occurrences. Empty `old` returns original. |
| `str_index_of(s, needle)` | `string, string -> int` | **Char index**, not byte. Returns `-1` if not found. |
| `str_contains(s, needle)` | `string, string -> int` | Returns `1` or `0`. Empty needle returns `1`. |
| `str_starts_with(s, prefix)` | `string, string -> int` | Returns `1` or `0`. |
| `str_ends_with(s, suffix)` | `string, string -> int` | Returns `1` or `0`. |
| `str_repeat(s, n)` | `string, int -> string` | Capped at 1M chars to prevent accidental memory blow-up. |
| `str_reverse(s)` | `string -> string` | Char-aware reverse (not byte-reverse). |
| `str_uppercase(s)` | `string -> string` | Locale-independent. |
| `str_lowercase(s)` | `string -> string` | Locale-independent. |

---

## Arrays

| Function | Signature | Notes |
|---|---|---|
| `arr_new(size, default)` | `int, T -> array` | Pre-filled array. For empty, use `arr_new(0, 0)`. |
| `arr_from_range(start, end)` | `int, int -> array<int>` | Half-open: `[start, end)`. |
| `arr_len(arr)` | `array -> int` | Number of elements. |
| `arr_get(arr, idx)` | `array, int -> T` | Errors on out-of-bounds. Use `safe arr_get` for total semantics. |
| `arr_set(VAR, idx, val)` | `varname, int, T -> null` | Mutating; first arg must be a bare variable. |
| `arr_push(VAR, val)` | `varname, T -> null` | Mutating; first arg must be a bare variable. |
| `arr_first(arr)` | `array -> T` | Errors on empty. |
| `arr_last(arr)` | `array -> T` | Errors on empty. |
| `arr_slice(arr, start, end)` | `array, int, int -> array` | Half-open. Out-of-range bounds clamp. |
| `arr_concat(a, b)` | `array, array -> array` | New array; does not mutate inputs. |
| `arr_contains(arr, val)` | `array, T -> int` | Returns `1` or `0`. |
| `arr_index_of(arr, val)` | `array, T -> int` | Returns `-1` if not found. |
| `arr_sort(arr)` | `array -> array` | New array sorted ascending. Total ordering across types via float fallback. |
| `arr_reverse(arr)` | `array -> array` | New array; does not mutate input. For strings use `str_reverse`. |
| `arr_join(arr, sep)` | `array, string -> string` | Alias-equivalent to `str_join` with arg order swap. |
| `arr_min(arr)` | `array<numeric> -> int` | Errors on empty. |
| `arr_max(arr)` | `array<numeric> -> int` | Errors on empty. |
| `arr_sum(arr)` | `array<numeric> -> int` | Empty array sums to 0. |
| `arr_fold_elements(arr)` | `array<int> -> array<int>` | Maps `fold_escape` over every element. |
| `arr_resonance(arr)` | `array<int> -> float` | Mean φ-resonance of elements. |
| `filter_by_resonance(arr, threshold)` | `array<int>, float -> array<int>` | Keeps elements with resonance ≥ threshold. |
| `cleanup_array(arr)` | `array -> array` | Removes singularities; preserves valid values. |

---

## Numbers and math

### Basic

| Function | Signature | Notes |
|---|---|---|
| `abs(x)` | `numeric -> numeric` | Absolute value. |
| `min(a, b)` | `numeric, numeric -> numeric` | Two-arg form; for arrays use `arr_min`. |
| `max(a, b)` | `numeric, numeric -> numeric` | Two-arg form; for arrays use `arr_max`. |
| `sign(x)` | `numeric -> int` | -1, 0, or 1. |
| `floor(x)` | `float -> int` | |
| `ceil(x)` | `float -> int` | |
| `round(x)` | `float -> int` | Banker's rounding. |
| `frac(x)` | `float -> float` | Fractional part. |
| `gcd(a, b)` | `int, int -> int` | Greatest common divisor (Euclidean algorithm). |
| `lcm(a, b)` | `int, int -> int` | Least common multiple. |
| `square(x)` | `numeric -> numeric` | `x * x`. |
| `cube(x)` | `numeric -> numeric` | `x * x * x`. |
| `pow(base, exp)` | `numeric, numeric -> float` | Float exponent. |
| `pow_int(base, exp)` | `int, int -> int` | Integer-only. |
| `sqrt(x)` | `numeric -> float` | |
| `factorial(n)` | `int -> int` | Errors for `n > 20` (overflow). |

### Predicates

| Function | Signature | Notes |
|---|---|---|
| `is_even(n)` / `even(n)` | `int -> int` | Returns `1` or `0`. |
| `is_odd(n)` / `odd(n)` | `int -> int` | Returns `1` or `0`. |
| `is_prime(n)` | `int -> int` | Trial-division up to √n. |

### Transcendental

| Function | Signature |
|---|---|
| `sin(x)`, `cos(x)`, `tan(x)`, `tanh(x)` | `float -> float` |
| `exp(x)`, `log(x)` | `float -> float` |
| `erf(x)`, `sigmoid(x)` | `float -> float` |

### Constants

`pi`, `tau`, `e`, `phi`, `phi_inv`, `phi_sq`, `phi_squared`, `sqrt_2`, `sqrt_5`, `ln_2` — all return `float`.

---

## Harmonic primitives (φ-math substrate)

These are the building blocks the self-healing compiler reasons against. They're cheap to compute and pure.

| Function | Signature | Notes |
|---|---|---|
| `fib(n)` / `fibonacci(n)` | `int -> int` | The n-th Fibonacci number. |
| `is_fibonacci(n)` | `int -> int` | Returns `1` if `n` is in the Fibonacci sequence, `0` otherwise. **The decidable type-class.** |
| `harmony_value(n)` / `res(n)` | `int -> float` | φ-resonance (0..1). `1.0` for Fibonacci numbers; decays with relative distance. |
| `fold(n)` | `int -> int` | Snap to nearest Fibonacci attractor (unconditional). |
| `fold_escape(n)` | `int -> int` | Conditional fold: only snaps if `value_danger > 0.5`. |
| `value_danger(x)` | `numeric -> float` | `exp(-|x|)` — the danger curve. Approaches `1.0` near zero, vanishes for large magnitudes. |
| `classify_resonance(n)` | `int -> int` | Discretized resonance bucket (0..N). |
| `harmonic_interfere(a, b)` / `interfere(a, b)` | `int, int -> float` | Two-element resonance interference. |
| `measure_coherence(arr)` | `array<int> -> float` | Coherence score across an array. |
| `mean_omni_weight(arr)` | `array<int> -> float` | OmniWeight = `φ^(-|e|)` mean. The geodesic decision metric. |
| `boundary(n)` | `int -> float` | Distance to nearest Fibonacci attractor. |

---

## Self-healing primitives

Compose with the substrate above. These are what the `safe` keyword desugars to — see `examples/safe_keyword_host.omc`.

| Function | Signature | Notes |
|---|---|---|
| `safe_divide(a, b)` | `numeric, numeric -> numeric` | If `value_danger(b) > 0.5`, folds `b` away from zero first, then divides. Total: never produces a singularity. |
| `safe_arr_get(arr, idx)` | `array, int -> T` | `fold_escape(idx) % arr_len(arr)`. Out-of-bounds reads become attractor-landing finite values. |
| `safe_arr_set(VAR, idx, val)` | `varname, int, T -> null` | Same fold-and-mod; in-place write at the healed index. Empty arrays silently no-op. |
| `safe_add(a, b)` / `safe_sub(a, b)` / `safe_mul(a, b)` | `numeric, numeric -> numeric` | Reserved for harmonic-aware arithmetic. Currently delegate to ordinary operators. |
| `resolve_singularity(v, strategy)` | `singularity, string -> numeric` | Strategies: `"fold"`, `"zero"`, `"one"`. |
| `is_singularity(v)` | `T -> int` | Type-class predicate. |
| `ensure_clean(v)` | `T -> T` | Returns `v` if not a singularity; else folds to nearest Fibonacci. |
| `collapse(v)` | `T -> T` | Force-evaluate any pending singularity. |
| `invert(x)` | `numeric -> numeric` | `1/x` with singularity guard. |
| `quantize(x, q)` | `numeric, numeric -> numeric` | Snap to nearest multiple of `q`. |
| `quantization_ratio(arr)` | `array<numeric> -> float` | Coarseness metric. |

---

## File I/O

| Function | Signature | Notes |
|---|---|---|
| `read_file(path)` | `string -> string` | Reads the file as UTF-8. Errors if path doesn't exist or isn't readable. |
| `write_file(path, content)` | `string, string -> int` | Returns `1` on success. Overwrites existing files. Errors if path can't be written. |
| `file_exists(path)` | `string -> int` | Total; returns `1` or `0`. Never errors. |

---

## Type and conversion

| Function | Signature | Notes |
|---|---|---|
| `type_of(v)` | `T -> string` | Returns `"int"`, `"float"`, `"string"`, `"bool"`, `"array"`, `"null"`, `"singularity"`, or `"circuit"`. |
| `to_int(v)` / `int(v)` | `T -> int` | Parses strings; truncates floats. |
| `to_float(v)` / `float(v)` | `T -> float` | |
| `to_string(v)` / `string(v)` | `T -> string` | Display formatting; renders numerics as bare values (not `HInt(42, ...)`). |
| `len(v)` | `array \| string -> int` | Polymorphic length. |

---

## Time

| Function | Signature | Notes |
|---|---|---|
| `now_ms()` | `-> int` | Milliseconds since UNIX epoch. Useful for benchmarking inside OMC programs. |

---

## Random

xorshift64* PRNG seeded from system nanoseconds at interpreter construction. Not cryptographic. Use `random_seed(n)` for deterministic runs.

| Function | Signature | Notes |
|---|---|---|
| `random_int(lo, hi)` | `int, int -> int` | Inclusive on both ends. `hi <= lo` returns `lo` (graceful fallback). |
| `random_float()` | `-> float` | Uniform in `[0.0, 1.0)`. |
| `random_seed(s)` | `int -> int` | Deterministic seed; returns the seed value. `s == 0` substituted with the golden-ratio constant `0x9E3779B97F4A7C15`. |

---

## Higher-order array operations

These require first-class function values. Pass a function name as a bare identifier (preferred) or as a string literal:

```omc
fn double(x) { return x * 2; }
arr_map(xs, double)     # bare name → Value::Function
arr_map(xs, "double")   # string form also works
```

User-defined functions and built-ins both work. The captured function is its **definition**, not a closure over local scope — closures are future work.

| Function | Signature | Notes |
|---|---|---|
| `arr_map(arr, f)` | `array, function -> array` | Calls `f(elem)` per element; collects results. |
| `arr_filter(arr, pred)` | `array, function -> array` | Keeps elements where `pred(elem)` is truthy. |
| `arr_reduce(arr, f, init)` | `array, function, T -> T` | Left fold; `f(acc, elem) -> acc`. |
| `arr_any(arr, pred)` | `array, function -> int` | `1` if any element satisfies `pred`; short-circuits. |
| `arr_all(arr, pred)` | `array, function -> int` | `1` if every element satisfies `pred`; short-circuits. |
| `arr_find(arr, pred)` | `array, function -> T \| null` | First element where `pred(elem)` is truthy, else `null`. |

Polish-round additions:

| Function | Signature | Notes |
|---|---|---|
| `arr_zip(a, b)` | `array, array -> array` | Pairs elements positionally as `[a_i, b_i]`; shorter array sets length. |
| `arr_unique(arr)` | `array -> array` | Dedupe preserving first-occurrence order. Type-aware equality. |
| `str_pad_left(s, width, ch)` | `string, int, string -> string` | Pads `s` on the left to `width` chars using first char of `ch`. |
| `str_pad_right(s, width, ch)` | `string, int, string -> string` | Pads on the right. |
| `println(x)` | `T -> null` | Like `print` but uses Display formatting (no HInt scaffolding). |
| `print_raw(x)` | `T -> null` | Like `println` but no trailing newline. Pairs for progress lines. |

---

## OMNIcode harmonic variants

These take ordinary operations and route them through the φ-math substrate. Anyone can write a file; these write **harmonically** — aware of resonance, attractor geometry, harmonic checksum signatures.

| Function | Signature | Notes |
|---|---|---|
| `harmonic_checksum(s)` | `string -> float` | Resonance signature: sum over each char's codepoint resonance. Two strings with the same checksum are harmonically equivalent. |
| `harmonic_write_file(path, content)` | `string, string -> float` | Atomic write with a resonance gate. Computes the content's mean per-char resonance; commits via tmp+rename if score ≥ 0.5; rejects (returns negative score) below the gate. The original target is untouched on rejection. |
| `harmonic_read_file(path)` | `string -> array<string, float>` | Returns `[content, mean_resonance]` so callers can decide whether to trust low-coherence content. Errors on read failure (use `file_exists` first if uncertain). |
| `harmonic_sort(arr)` | `array -> array` | Sort by `harmony_value` of each element **descending**. Pure Fibonacci values lead; off-grid values sink. For strings, sorts by mean char-resonance. **Different from `arr_sort`**: that orders by NATURAL value (1<2<3), this by φ-alignment (89 outranks 100). |
| `harmonic_split(s)` | `string -> array<string>` | Split into chunks whose sizes are nearest-Fibonacci at word boundaries. For a 100-char string: chunk sizes from {89, 55+34, 89+8, ...} respecting whitespace. Useful for φ-aligned line wrapping and packet sizing. |
| `harmonic_partition(arr)` | `array -> array<array>` | Group elements by nearest Fibonacci attractor. Returns outer array of buckets (one per occupied attractor, in attractor order); inner arrays hold original elements. Use for distribution analysis along the φ-grid. |

---

---

## Statements (not functions)

These are language keywords, not functions, but bear mentioning here:

- **`print(x)`** — Writes to stdout. Renders numerics in `HInt(value, φ=…, HIM=…)` debug form by default; use `to_string` for clean rendering.
- **`safe <expr>`** — Wraps `<expr>` in self-healing semantics. See `examples/safe_keyword_host.omc`. Currently dispatches: `safe a / b → safe_divide`, `safe arr_get(...) → safe_arr_get`, `safe arr_set(...) → safe_arr_set`.
- **`h <name> = <expr>;`** — Harmonic variable declaration. Required; OMC has no implicit declarations.
- **`fn name(args) -> type? { body }`** — Function definition. Return type annotation is optional and informational only.
- **`if`, `else`, `while`, `for`, `return`, `break`, `continue`** — Standard control flow.
- **`import <name>` / `load <path>`** — Module imports.

---

## Missing on purpose

The following common builtins are **deliberately not in the standard library** today, in most cases because they conflict with the φ-math substrate or require language-level changes:

- **`map(f, arr)` / `filter(p, arr)` / `reduce(f, arr, init)`** — These exist as `arr_map` / `arr_filter` / `arr_reduce` (see *Higher-order array operations*). The standalone short names aren't aliased because they're too common to risk shadowing user-defined helpers.
- **`println(x)` and `print_raw(x)`** — Both now exist (see *Higher-order array operations* table). `println` uses Display formatting (no HInt scaffolding); `print_raw` is the same with no trailing newline. The original `print` is preserved for debug-format introspection.
- **`assert(cond)`** — Use `if cond == 0 { return; }` and check return values.
- **`format(fmt, ...)`** — Use `concat_many(...)` instead. The `concat_many` variadic handles type coercion.

If you reach for one of these and find it actually exists, this doc is stale — please update.

---

## Future-tense work

Categories under active design (see `OMC_STRATEGIC_PLAN.md`):

- **Closures over local scope.** First-class function references work today (named function passed as value); proper closures that capture local bindings are the next step.
- A bytecode-VM-fast subset of common primitives (currently the VM and tree-walker share the same primitive table; faster inlining is possible).
- Module system beyond the current `load`-by-path approach.
- More OMNIcode harmonic variants — natural next candidates: `harmonic_hash(s)` (collision-resistant resonance hash), `harmonic_diff(a, b)` (file diff weighted by resonance), `harmonic_dedupe(arr, threshold)` (cluster-then-collapse by resonance band).
