# OMC Standard Library Reference

Every built-in function available in OMNIcode, organized by category. Function signatures use `name(arg: type, ...) -> type` notation. Types: `int`, `float`, `string`, `bool`, `array`, `null`, `singularity`, `circuit`.

**HInt vs int.** All integer values in OMC carry harmonic metadata (`φ`-resonance, HIM score) — they're `HInt`s under the hood. The signatures below use `int` for readability; the metadata is computed automatically and surfaces only when you `print` a raw value.

**Where this lives.** Built-ins are implemented in `omnimcode-core/src/interpreter.rs`. The Rust VM's compile-time type inference is in `omnimcode-core/src/compiler.rs`. To add a new built-in, see `DEVELOPER.md`.

---

## Quick reference (alphabetical)

For "I know the name, what does it do" lookups. Skip to the category sections for "I want to do X, what should I reach for".

```
abs              arr_concat       arr_contains     arr_first
arr_fold_elements  arr_from_range  arr_get          arr_index_of
arr_join         arr_last         arr_len          arr_max
arr_min          arr_new          arr_push         arr_resonance
arr_reverse      arr_set          arr_slice        arr_sort
arr_sum          boundary         ceil             classify_resonance
cleanup_array    collapse         concat_many      cos
cube             e                ensure_clean     erf
even             exp              factorial        fib
fibonacci        file_exists      filter_by_resonance  float
floor            fold             fold_escape      frac
gcd              harmonic_interfere  harmony_value  int
interfere        invert           is_even          is_fibonacci
is_odd           is_prime         is_singularity   lcm
len              log              max              mean_omni_weight
measure_coherence  min            now_ms           odd
phi              phi_inv          phi_sq           phi_squared
pi               pow              pow_int          print
quantization_ratio  quantize      read_file        res
resolve_singularity  round        safe_add         safe_arr_get
safe_arr_set     safe_divide      safe_mul         safe_sub
sigmoid          sign             sin              sqrt
square           str_chars        str_concat       str_contains
str_ends_with    str_index_of     str_join         str_len
str_lowercase    str_repeat       str_replace      str_reverse
str_slice        str_split        str_starts_with  str_trim
str_uppercase    string           tan              tanh
tau              to_float         to_int           to_string
type_of          value_danger     write_file
```

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

- **`map(f, arr)` / `filter(p, arr)` / `reduce(f, arr, init)`** — Higher-order functions over arrays require first-class function values, which OMC doesn't have yet. Use `while` loops over `arr_len`.
- **`println(x)` / `print_no_newline(x)`** — `print` always emits a newline; for raw byte output use `write_file("/dev/stdout", x)`.
- **`assert(cond)`** — Use `if cond == 0 { return; }` and check return values.
- **`format(fmt, ...)`** — Use `concat_many(...)` instead. The `concat_many` variadic handles type coercion.

If you reach for one of these and find it actually exists, this doc is stale — please update.

---

## Future-tense work

Categories under active design (see `OMC_STRATEGIC_PLAN.md`):

- First-class functions and closures (would unlock `map`/`filter`/`reduce`).
- A bytecode-VM-fast subset of common primitives (currently the VM and tree-walker share the same primitive table; faster inlining is possible).
- Module system beyond the current `load`-by-path approach.
- A real `random()` family — there's a `fastrand` dependency but it's not yet exposed as built-ins.
