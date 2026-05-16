# OMC Builtin Reference

Auto-generated from `omnimcode-core/src/docs.rs`. Run `omc --gen-docs > OMC_REFERENCE.md` to regenerate.

**Total documented builtins**: 540

**OMC-unique**: 61 (no direct Python/NumPy equivalent — these are why you reach for OMC over numpy)

---

## Categories

- [core](#core) (124 builtins)
- [arrays](#arrays) (111 builtins)
- [linalg](#linalg) (4 builtins)
- [ml_kernels](#ml_kernels) (6 builtins)
- [substrate](#substrate) (39 builtins)
- [autograd](#autograd) (18 builtins)
- [duals](#duals) (21 builtins)
- [generators](#generators) (5 builtins)
- [strings](#strings) (33 builtins)
- [regex](#regex) (7 builtins)
- [json](#json) (2 builtins)
- [stdlib](#stdlib) (22 builtins)
- [exceptions](#exceptions) (2 builtins)
- [introspection](#introspection) (22 builtins)
- [tokenizer](#tokenizer) (16 builtins)
- [code_intel](#code_intel) (16 builtins)
- [math](#math) (58 builtins)
- [dicts](#dicts) (26 builtins)
- [test_runner](#test_runner) (8 builtins)

---

## core

### `print`

**Signature**: `(value) -> null`

Print value to stdout with newline.

```omc
print("hello");
```

### `to_string`

**Signature**: `(value) -> string`

Coerce any value to its display string.

```omc
to_string(42)  // "42"
```

### `type_of`

**Signature**: `(value) -> string`

Runtime type tag: int, float, string, bool, array, dict, function, null_t.

```omc
type_of([1,2,3])  // "array"
```

### `len`

**Signature**: `(string|array) -> int`

Length in bytes (string) or elements (array).

```omc
len([1,2,3])  // 3
```

### `attractor_table`

**Signature**: `(...) -> any`

`attractor_table`: see omc_explain or source for details. Auto-generated stub.

```omc
attractor_table(...)  // see omc_help
```

### `call`

**Signature**: `(...) -> any`

`call`: see omc_explain or source for details. Auto-generated stub.

```omc
call(...)  // see omc_help
```

### `classify_resonance`

**Signature**: `(...) -> any`

`classify_resonance`: see omc_explain or source for details. Auto-generated stub.

```omc
classify_resonance(...)  // see omc_help
```

### `collapse`

**Signature**: `(...) -> any`

`collapse`: see omc_explain or source for details. Auto-generated stub.

```omc
collapse(...)  // see omc_help
```

### `cube`

**Signature**: `(...) -> any`

`cube`: see omc_explain or source for details. Auto-generated stub.

```omc
cube(...)  // see omc_help
```

### `e`

**Signature**: `(...) -> any`

`e`: see omc_explain or source for details. Auto-generated stub.

```omc
e(...)  // see omc_help
```

### `ensure_clean`

**Signature**: `(...) -> any`

`ensure_clean`: see omc_explain or source for details. Auto-generated stub.

```omc
ensure_clean(...)  // see omc_help
```

### `erf`

**Signature**: `(...) -> any`

`erf`: see omc_explain or source for details. Auto-generated stub.

```omc
erf(...)  // see omc_help
```

### `even`

**Signature**: `(...) -> any`

`even`: see omc_explain or source for details. Auto-generated stub.

```omc
even(...)  // see omc_help
```

### `factorial`

**Signature**: `(...) -> any`

`factorial`: see omc_explain or source for details. Auto-generated stub.

```omc
factorial(...)  // see omc_help
```

### `fib`

**Signature**: `(...) -> any`

`fib`: see omc_explain or source for details. Auto-generated stub.

```omc
fib(...)  // see omc_help
```

### `fib_chunks`

**Signature**: `(...) -> any`

`fib_chunks`: see omc_explain or source for details. Auto-generated stub.

```omc
fib_chunks(...)  // see omc_help
```

### `fibonacci`

**Signature**: `(...) -> any`

`fibonacci`: see omc_explain or source for details. Auto-generated stub.

```omc
fibonacci(...)  // see omc_help
```

### `filter_by_resonance`

**Signature**: `(...) -> any`

`filter_by_resonance`: see omc_explain or source for details. Auto-generated stub.

```omc
filter_by_resonance(...)  // see omc_help
```

### `float`

**Signature**: `(...) -> any`

`float`: see omc_explain or source for details. Auto-generated stub.

```omc
float(...)  // see omc_help
```

### `fold`

**Signature**: `(...) -> any`

`fold`: see omc_explain or source for details. Auto-generated stub.

```omc
fold(...)  // see omc_help
```

### `fold_escape`

**Signature**: `(...) -> any`

`fold_escape`: see omc_explain or source for details. Auto-generated stub.

```omc
fold_escape(...)  // see omc_help
```

### `frac`

**Signature**: `(...) -> any`

`frac`: see omc_explain or source for details. Auto-generated stub.

```omc
frac(...)  // see omc_help
```

### `from_zeckendorf`

**Signature**: `(...) -> any`

`from_zeckendorf`: see omc_explain or source for details. Auto-generated stub.

```omc
from_zeckendorf(...)  // see omc_help
```

### `harmonic_align`

**Signature**: `(...) -> any`

`harmonic_align`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_align(...)  // see omc_help
```

### `harmonic_checksum`

**Signature**: `(...) -> any`

`harmonic_checksum`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_checksum(...)  // see omc_help
```

### `harmonic_interfere`

**Signature**: `(...) -> any`

`harmonic_interfere`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_interfere(...)  // see omc_help
```

### `harmonic_partition_3`

**Signature**: `(...) -> any`

`harmonic_partition_3`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_partition_3(...)  // see omc_help
```

### `harmonic_resample`

**Signature**: `(...) -> any`

`harmonic_resample`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_resample(...)  // see omc_help
```

### `harmonic_unalign`

**Signature**: `(...) -> any`

`harmonic_unalign`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_unalign(...)  // see omc_help
```

### `harmonic_write_file`

**Signature**: `(...) -> any`

`harmonic_write_file`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_write_file(...)  // see omc_help
```

### `harmony_value`

**Signature**: `(...) -> any`

`harmony_value`: see omc_explain or source for details. Auto-generated stub.

```omc
harmony_value(...)  // see omc_help
```

### `hypot`

**Signature**: `(...) -> any`

`hypot`: see omc_explain or source for details. Auto-generated stub.

```omc
hypot(...)  // see omc_help
```

### `int`

**Signature**: `(...) -> any`

`int`: see omc_explain or source for details. Auto-generated stub.

```omc
int(...)  // see omc_help
```

### `int_binary_search`

**Signature**: `(...) -> any`

`int_binary_search`: see omc_explain or source for details. Auto-generated stub.

```omc
int_binary_search(...)  // see omc_help
```

### `int_lower_bound`

**Signature**: `(...) -> any`

`int_lower_bound`: see omc_explain or source for details. Auto-generated stub.

```omc
int_lower_bound(...)  // see omc_help
```

### `int_upper_bound`

**Signature**: `(...) -> any`

`int_upper_bound`: see omc_explain or source for details. Auto-generated stub.

```omc
int_upper_bound(...)  // see omc_help
```

### `interfere`

**Signature**: `(...) -> any`

`interfere`: see omc_explain or source for details. Auto-generated stub.

```omc
interfere(...)  // see omc_help
```

### `is_even`

**Signature**: `(...) -> any`

`is_even`: see omc_explain or source for details. Auto-generated stub.

```omc
is_even(...)  // see omc_help
```

### `is_fibonacci`

**Signature**: `(...) -> any`

`is_fibonacci`: see omc_explain or source for details. Auto-generated stub.

```omc
is_fibonacci(...)  // see omc_help
```

### `is_odd`

**Signature**: `(...) -> any`

`is_odd`: see omc_explain or source for details. Auto-generated stub.

```omc
is_odd(...)  // see omc_help
```

### `is_phi_resonant`

**Signature**: `(...) -> any`

`is_phi_resonant`: see omc_explain or source for details. Auto-generated stub.

```omc
is_phi_resonant(...)  // see omc_help
```

### `is_prime`

**Signature**: `(...) -> any`

`is_prime`: see omc_explain or source for details. Auto-generated stub.

```omc
is_prime(...)  // see omc_help
```

### `is_singularity`

**Signature**: `(...) -> any`

`is_singularity`: see omc_explain or source for details. Auto-generated stub.

```omc
is_singularity(...)  // see omc_help
```

### `is_zeckendorf_valid`

**Signature**: `(...) -> any`

`is_zeckendorf_valid`: see omc_explain or source for details. Auto-generated stub.

```omc
is_zeckendorf_valid(...)  // see omc_help
```

### `lerp`

**Signature**: `(...) -> any`

`lerp`: see omc_explain or source for details. Auto-generated stub.

```omc
lerp(...)  // see omc_help
```

### `ln_2`

**Signature**: `(...) -> any`

`ln_2`: see omc_explain or source for details. Auto-generated stub.

```omc
ln_2(...)  // see omc_help
```

### `log_phi_pi_fibonacci`

**Signature**: `(...) -> any`

`log_phi_pi_fibonacci`: see omc_explain or source for details. Auto-generated stub.

```omc
log_phi_pi_fibonacci(...)  // see omc_help
```

### `mean_omni_weight`

**Signature**: `(...) -> any`

`mean_omni_weight`: see omc_explain or source for details. Auto-generated stub.

```omc
mean_omni_weight(...)  // see omc_help
```

### `measure_coherence`

**Signature**: `(...) -> any`

`measure_coherence`: see omc_explain or source for details. Auto-generated stub.

```omc
measure_coherence(...)  // see omc_help
```

### `nearest_attractor`

**Signature**: `(...) -> any`

`nearest_attractor`: see omc_explain or source for details. Auto-generated stub.

```omc
nearest_attractor(...)  // see omc_help
```

### `now_ms`

**Signature**: `(...) -> any`

`now_ms`: see omc_explain or source for details. Auto-generated stub.

```omc
now_ms(...)  // see omc_help
```

### `nth_fibonacci`

**Signature**: `(...) -> any`

`nth_fibonacci`: see omc_explain or source for details. Auto-generated stub.

```omc
nth_fibonacci(...)  // see omc_help
```

### `odd`

**Signature**: `(...) -> any`

`odd`: see omc_explain or source for details. Auto-generated stub.

```omc
odd(...)  // see omc_help
```

### `phi`

**Signature**: `(...) -> any`

`phi`: see omc_explain or source for details. Auto-generated stub.

```omc
phi(...)  // see omc_help
```

### `phi_inv`

**Signature**: `(...) -> any`

`phi_inv`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_inv(...)  // see omc_help
```

### `phi_pi_bin_search`

**Signature**: `(...) -> any`

`phi_pi_bin_search`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_bin_search(...)  // see omc_help
```

### `phi_pi_fib_nearest`

**Signature**: `(...) -> any`

`phi_pi_fib_nearest`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_nearest(...)  // see omc_help
```

### `phi_pi_fib_nearest_traced`

**Signature**: `(...) -> any`

`phi_pi_fib_nearest_traced`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_nearest_traced(...)  // see omc_help
```

### `phi_pi_fib_nearest_v2`

**Signature**: `(...) -> any`

`phi_pi_fib_nearest_v2`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_nearest_v2(...)  // see omc_help
```

### `phi_pi_fib_reset`

**Signature**: `(...) -> any`

`phi_pi_fib_reset`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_reset(...)  // see omc_help
```

### `phi_pi_fib_search`

**Signature**: `(...) -> any`

`phi_pi_fib_search`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_search(...)  // see omc_help
```

### `phi_pi_fib_search_traced`

**Signature**: `(...) -> any`

`phi_pi_fib_search_traced`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_search_traced(...)  // see omc_help
```

### `phi_pi_fib_search_v2`

**Signature**: `(...) -> any`

`phi_pi_fib_search_v2`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_search_v2(...)  // see omc_help
```

### `phi_pi_fib_stats`

**Signature**: `(...) -> any`

`phi_pi_fib_stats`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_stats(...)  // see omc_help
```

### `phi_pi_fib_stats_all`

**Signature**: `(...) -> any`

`phi_pi_fib_stats_all`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_stats_all(...)  // see omc_help
```

### `phi_pi_fib_stats_bg`

**Signature**: `(...) -> any`

`phi_pi_fib_stats_bg`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_fib_stats_bg(...)  // see omc_help
```

### `phi_pi_log_distance`

**Signature**: `(...) -> any`

`phi_pi_log_distance`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_log_distance(...)  // see omc_help
```

### `phi_pi_pow`

**Signature**: `(...) -> any`

`phi_pi_pow`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pi_pow(...)  // see omc_help
```

### `phi_pow`

**Signature**: `(...) -> any`

`phi_pow`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_pow(...)  // see omc_help
```

### `phi_sq`

**Signature**: `(...) -> any`

`phi_sq`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_sq(...)  // see omc_help
```

### `phi_squared`

**Signature**: `(...) -> any`

`phi_squared`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_squared(...)  // see omc_help
```

### `pi`

**Signature**: `(...) -> any`

`pi`: see omc_explain or source for details. Auto-generated stub.

```omc
pi(...)  // see omc_help
```

### `pow_int`

**Signature**: `(...) -> any`

`pow_int`: see omc_explain or source for details. Auto-generated stub.

```omc
pow_int(...)  // see omc_help
```

### `print_raw`

**Signature**: `(...) -> any`

`print_raw`: see omc_explain or source for details. Auto-generated stub.

```omc
print_raw(...)  // see omc_help
```

### `println`

**Signature**: `(...) -> any`

`println`: see omc_explain or source for details. Auto-generated stub.

```omc
println(...)  // see omc_help
```

### `quantization_ratio`

**Signature**: `(...) -> any`

`quantization_ratio`: see omc_explain or source for details. Auto-generated stub.

```omc
quantization_ratio(...)  // see omc_help
```

### `quantize`

**Signature**: `(...) -> any`

`quantize`: see omc_explain or source for details. Auto-generated stub.

```omc
quantize(...)  // see omc_help
```

### `random_float`

**Signature**: `(...) -> any`

`random_float`: see omc_explain or source for details. Auto-generated stub.

```omc
random_float(...)  // see omc_help
```

### `random_int`

**Signature**: `(...) -> any`

`random_int`: see omc_explain or source for details. Auto-generated stub.

```omc
random_int(...)  // see omc_help
```

### `random_seed`

**Signature**: `(...) -> any`

`random_seed`: see omc_explain or source for details. Auto-generated stub.

```omc
random_seed(...)  // see omc_help
```

### `resolve_singularity`

**Signature**: `(...) -> any`

`resolve_singularity`: see omc_explain or source for details. Auto-generated stub.

```omc
resolve_singularity(...)  // see omc_help
```

### `resonance_band`

**Signature**: `(...) -> any`

`resonance_band`: see omc_explain or source for details. Auto-generated stub.

```omc
resonance_band(...)  // see omc_help
```

### `resonance_band_histogram`

**Signature**: `(...) -> any`

`resonance_band_histogram`: see omc_explain or source for details. Auto-generated stub.

```omc
resonance_band_histogram(...)  // see omc_help
```

### `safe_add`

**Signature**: `(...) -> any`

`safe_add`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_add(...)  // see omc_help
```

### `safe_arr_get`

**Signature**: `(...) -> any`

`safe_arr_get`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_arr_get(...)  // see omc_help
```

### `safe_arr_set`

**Signature**: `(...) -> any`

`safe_arr_set`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_arr_set(...)  // see omc_help
```

### `safe_divide`

**Signature**: `(...) -> any`

`safe_divide`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_divide(...)  // see omc_help
```

### `safe_log`

**Signature**: `(...) -> any`

`safe_log`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_log(...)  // see omc_help
```

### `safe_mod`

**Signature**: `(...) -> any`

`safe_mod`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_mod(...)  // see omc_help
```

### `safe_mul`

**Signature**: `(...) -> any`

`safe_mul`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_mul(...)  // see omc_help
```

### `safe_sqrt`

**Signature**: `(...) -> any`

`safe_sqrt`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_sqrt(...)  // see omc_help
```

### `safe_sub`

**Signature**: `(...) -> any`

`safe_sub`: see omc_explain or source for details. Auto-generated stub.

```omc
safe_sub(...)  // see omc_help
```

### `sigmoid`

**Signature**: `(...) -> any`

`sigmoid`: see omc_explain or source for details. Auto-generated stub.

```omc
sigmoid(...)  // see omc_help
```

### `sorted_dedupe`

**Signature**: `(...) -> any`

`sorted_dedupe`: see omc_explain or source for details. Auto-generated stub.

```omc
sorted_dedupe(...)  // see omc_help
```

### `sorted_merge`

**Signature**: `(...) -> any`

`sorted_merge`: see omc_explain or source for details. Auto-generated stub.

```omc
sorted_merge(...)  // see omc_help
```

### `sorted_union`

**Signature**: `(...) -> any`

`sorted_union`: see omc_explain or source for details. Auto-generated stub.

```omc
sorted_union(...)  // see omc_help
```

### `sqrt_2`

**Signature**: `(...) -> any`

`sqrt_2`: see omc_explain or source for details. Auto-generated stub.

```omc
sqrt_2(...)  // see omc_help
```

### `sqrt_5`

**Signature**: `(...) -> any`

`sqrt_5`: see omc_explain or source for details. Auto-generated stub.

```omc
sqrt_5(...)  // see omc_help
```

### `square`

**Signature**: `(...) -> any`

`square`: see omc_explain or source for details. Auto-generated stub.

```omc
square(...)  // see omc_help
```

### `string`

**Signature**: `(...) -> any`

`string`: see omc_explain or source for details. Auto-generated stub.

```omc
string(...)  // see omc_help
```

### `substrate_count_range`

**Signature**: `(...) -> any`

`substrate_count_range`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_count_range(...)  // see omc_help
```

### `substrate_difference`

**Signature**: `(...) -> any`

`substrate_difference`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_difference(...)  // see omc_help
```

### `substrate_hash`

**Signature**: `(...) -> any`

`substrate_hash`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_hash(...)  // see omc_help
```

### `substrate_insert`

**Signature**: `(...) -> any`

`substrate_insert`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_insert(...)  // see omc_help
```

### `substrate_intersect`

**Signature**: `(...) -> any`

`substrate_intersect`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_intersect(...)  // see omc_help
```

### `substrate_lower_bound`

**Signature**: `(...) -> any`

`substrate_lower_bound`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_lower_bound(...)  // see omc_help
```

### `substrate_min_distance`

**Signature**: `(...) -> any`

`substrate_min_distance`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_min_distance(...)  // see omc_help
```

### `substrate_nearest`

**Signature**: `(...) -> any`

`substrate_nearest`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_nearest(...)  // see omc_help
```

### `substrate_quantile`

**Signature**: `(...) -> any`

`substrate_quantile`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_quantile(...)  // see omc_help
```

### `substrate_rank`

**Signature**: `(...) -> any`

`substrate_rank`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_rank(...)  // see omc_help
```

### `substrate_search`

**Signature**: `(...) -> any`

`substrate_search`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_search(...)  // see omc_help
```

### `substrate_select_k`

**Signature**: `(...) -> any`

`substrate_select_k`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_select_k(...)  // see omc_help
```

### `substrate_slice_range`

**Signature**: `(...) -> any`

`substrate_slice_range`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_slice_range(...)  // see omc_help
```

### `substrate_upper_bound`

**Signature**: `(...) -> any`

`substrate_upper_bound`: see omc_explain or source for details. Auto-generated stub.

```omc
substrate_upper_bound(...)  // see omc_help
```

### `tanh`

**Signature**: `(...) -> any`

`tanh`: see omc_explain or source for details. Auto-generated stub.

```omc
tanh(...)  // see omc_help
```

### `tau`

**Signature**: `(...) -> any`

`tau`: see omc_explain or source for details. Auto-generated stub.

```omc
tau(...)  // see omc_help
```

### `test_clear_failures`

**Signature**: `(...) -> any`

`test_clear_failures`: see omc_explain or source for details. Auto-generated stub.

```omc
test_clear_failures(...)  // see omc_help
```

### `test_get_current`

**Signature**: `(...) -> any`

`test_get_current`: see omc_explain or source for details. Auto-generated stub.

```omc
test_get_current(...)  // see omc_help
```

### `to_float`

**Signature**: `(...) -> any`

`to_float`: see omc_explain or source for details. Auto-generated stub.

```omc
to_float(...)  // see omc_help
```

### `to_int`

**Signature**: `(...) -> any`

`to_int`: see omc_explain or source for details. Auto-generated stub.

```omc
to_int(...)  // see omc_help
```

### `value_danger`

**Signature**: `(...) -> any`

`value_danger`: see omc_explain or source for details. Auto-generated stub.

```omc
value_danger(...)  // see omc_help
```

### `zeckendorf`

**Signature**: `(...) -> any`

`zeckendorf`: see omc_explain or source for details. Auto-generated stub.

```omc
zeckendorf(...)  // see omc_help
```

### `zeckendorf_bit`

**Signature**: `(...) -> any`

`zeckendorf_bit`: see omc_explain or source for details. Auto-generated stub.

```omc
zeckendorf_bit(...)  // see omc_help
```

### `zeckendorf_weight`

**Signature**: `(...) -> any`

`zeckendorf_weight`: see omc_explain or source for details. Auto-generated stub.

```omc
zeckendorf_weight(...)  // see omc_help
```

---

## arrays

### `arr_new`

**Signature**: `() -> array`

Create an empty mutable array.

```omc
arr_new()  // []
```

### `arr_push`

**Signature**: `(arr, value) -> array`

Append value to array in place.

```omc
arr_push(xs, 42);
```

### `arr_get`

**Signature**: `(arr, index) -> any`

Read element at index (0-based).

```omc
arr_get([10,20,30], 1)  // 20
```

### `arr_set`

**Signature**: `(arr, index, value) -> null`

Write element at index in place.

```omc
arr_set(xs, 0, 99);
```

### `arr_len`

**Signature**: `(arr) -> int`

Length of array.

```omc
arr_len([1,2,3])  // 3
```

### `arr_concat`

**Signature**: `(a, b) -> array`

Concatenate two arrays into a new one.

```omc
arr_concat([1,2], [3,4])  // [1,2,3,4]
```

### `arr_slice`

**Signature**: `(arr, start, end) -> array`

Half-open slice [start..end).

```omc
arr_slice([0,1,2,3,4], 1, 4)  // [1,2,3]
```

### `arr_map`

**Signature**: `(arr, fn) -> array`

Apply function to each element, returning new array.

```omc
arr_map([1,2,3], fn(x) { return x*x; })  // [1,4,9]
```

### `arr_filter`

**Signature**: `(arr, fn) -> array`

Keep elements where predicate returns truthy.

```omc
arr_filter([1,2,3,4], fn(x) { return x % 2 == 0; })  // [2,4]
```

### `arr_sort`

**Signature**: `(arr) -> array`

Ascending sort by numeric value.

```omc
arr_sort([3,1,2])  // [1,2,3]
```

### `arr_reverse`

**Signature**: `(arr) -> array`

Reverse a copy of the array.

```omc
arr_reverse([1,2,3])  // [3,2,1]
```

### `arr_sum_int`

**Signature**: `(arr) -> int`

Sum of integer elements.

```omc
arr_sum_int([1,2,3,4])  // 10
```

### `arr_mean`

**Signature**: `(arr) -> float`

Arithmetic mean.

```omc
arr_mean([1.0,2.0,3.0])  // 2.0
```

### `arr_variance`

**Signature**: `(arr) -> float`

Sample variance.

```omc
arr_variance([1.0,2.0,3.0,4.0,5.0])  // 2.5
```

### `arr_stddev`

**Signature**: `(arr) -> float`

Standard deviation.

```omc
arr_stddev([1.0,2.0,3.0,4.0,5.0])  // ~1.58
```

### `arr_dot`

**Signature**: `(a, b) -> float`

Dot product of two 1D arrays.

```omc
arr_dot([1.0,2.0], [3.0,4.0])  // 11.0
```

### `arr_min_int`

**Signature**: `(arr) -> int`

Minimum element (int).

```omc
arr_min_int([3,1,4,1,5])  // 1
```

### `arr_max_int`

**Signature**: `(arr) -> int`

Maximum element (int).

```omc
arr_max_int([3,1,4,1,5])  // 5
```

### `arr_argmax`

**Signature**: `(arr) -> int`

Index of largest element.

```omc
arr_argmax([3,1,4,1,5])  // 4
```

### `arr_argmin`

**Signature**: `(arr) -> int`

Index of smallest element.

```omc
arr_argmin([3,1,4,1,5])  // 1
```

### `arr_add`

**Signature**: `(a, b) -> array`

Elementwise add. Broadcasts scalar↔array and 2D↔1D row-vector.

```omc
arr_add([1,2,3], 10)  // [11,12,13]
```

### `arr_sub`

**Signature**: `(a, b) -> array`

Elementwise subtract, with broadcasting.

```omc
arr_sub([10,20,30], [1,2,3])  // [9,18,27]
```

### `arr_mul`

**Signature**: `(a, b) -> array`

Elementwise multiply, with broadcasting.

```omc
arr_mul([1,2,3], [10,10,10])  // [10,20,30]
```

### `arr_div_int`

**Signature**: `(a, b) -> array`

Elementwise integer division (div-by-0 → 0).

```omc
arr_div_int([10,20,30], [2,5,3])  // [5,4,10]
```

### `arr_neg`

**Signature**: `(arr) -> array`

Elementwise negation.

```omc
arr_neg([1,-2,3])  // [-1,2,-3]
```

### `arr_scale`

**Signature**: `(arr, scalar) -> array`

Multiply every element by a scalar.

```omc
arr_scale([1,2,3], 10)  // [10,20,30]
```

### `arr_all`

**Signature**: `(arr, val_or_pred) -> int`

`arr_all`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_all(...)  // see omc_help
```

### `arr_any`

**Signature**: `(arr, val_or_pred) -> int`

`arr_any`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_any(...)  // see omc_help
```

### `arr_avg_distance`

**Signature**: `(arr) -> float`

`arr_avg_distance`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_avg_distance(...)  // see omc_help
```

### `arr_chunk`

**Signature**: `(arr, ...) -> array`

`arr_chunk`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_chunk(...)  // see omc_help
```

### `arr_contains`

**Signature**: `(arr, val_or_pred) -> int`

`arr_contains`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_contains(...)  // see omc_help
```

### `arr_count`

**Signature**: `(arr) -> int`

`arr_count`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_count(...)  // see omc_help
```

### `arr_cumsum`

**Signature**: `(arr, ...) -> array`

`arr_cumsum`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_cumsum(...)  // see omc_help
```

### `arr_diff`

**Signature**: `(arr, ...) -> array`

`arr_diff`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_diff(...)  // see omc_help
```

### `arr_drop`

**Signature**: `(arr, ...) -> array`

`arr_drop`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_drop(...)  // see omc_help
```

### `arr_enumerate`

**Signature**: `(arr, ...) -> array`

`arr_enumerate`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_enumerate(...)  // see omc_help
```

### `arr_find`

**Signature**: `(arr) -> int`

`arr_find`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_find(...)  // see omc_help
```

### `arr_first`

**Signature**: `(arr) -> int`

`arr_first`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_first(...)  // see omc_help
```

### `arr_flatten`

**Signature**: `(arr, ...) -> array`

`arr_flatten`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_flatten(...)  // see omc_help
```

### `arr_fold_elements`

**Signature**: `(arr, ...) -> array`

`arr_fold_elements`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_fold_elements(...)  // see omc_help
```

### `arr_from_range`

**Signature**: `(arr, ...) -> array`

`arr_from_range`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_from_range(...)  // see omc_help
```

### `arr_gcd`

**Signature**: `(arr, ...) -> array`

`arr_gcd`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_gcd(...)  // see omc_help
```

### `arr_geometric_mean`

**Signature**: `(arr) -> float`

`arr_geometric_mean`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_geometric_mean(...)  // see omc_help
```

### `arr_harmonic_mean`

**Signature**: `(arr) -> float`

`arr_harmonic_mean`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_harmonic_mean(...)  // see omc_help
```

### `arr_index_of`

**Signature**: `(arr) -> int`

`arr_index_of`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_index_of(...)  // see omc_help
```

### `arr_is_sorted`

**Signature**: `(arr, val_or_pred) -> int`

`arr_is_sorted`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_is_sorted(...)  // see omc_help
```

### `arr_join`

**Signature**: `(arr, ...) -> array`

`arr_join`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_join(...)  // see omc_help
```

### `arr_last`

**Signature**: `(arr) -> int`

`arr_last`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_last(...)  // see omc_help
```

### `arr_max`

**Signature**: `(arr, ...) -> array`

`arr_max`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_max(...)  // see omc_help
```

### `arr_max_float`

**Signature**: `(arr) -> int`

`arr_max_float`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_max_float(...)  // see omc_help
```

### `arr_median`

**Signature**: `(arr) -> float`

`arr_median`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_median(...)  // see omc_help
```

### `arr_min`

**Signature**: `(arr, ...) -> array`

`arr_min`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_min(...)  // see omc_help
```

### `arr_min_float`

**Signature**: `(arr) -> int`

`arr_min_float`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_min_float(...)  // see omc_help
```

### `arr_norm`

**Signature**: `(arr) -> float`

`arr_norm`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_norm(...)  // see omc_help
```

### `arr_ones`

**Signature**: `(arr, ...) -> array`

`arr_ones`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_ones(...)  // see omc_help
```

### `arr_partition_by`

**Signature**: `(arr, ...) -> array`

`arr_partition_by`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_partition_by(...)  // see omc_help
```

### `arr_product`

**Signature**: `(arr, ...) -> array`

`arr_product`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_product(...)  // see omc_help
```

### `arr_range`

**Signature**: `(arr, ...) -> array`

`arr_range`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_range(...)  // see omc_help
```

### `arr_reduce`

**Signature**: `(arr, ...) -> array`

`arr_reduce`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_reduce(...)  // see omc_help
```

### `arr_repeat`

**Signature**: `(arr, ...) -> array`

`arr_repeat`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_repeat(...)  // see omc_help
```

### `arr_resonance`

**Signature**: `(arr, ...) -> array`

`arr_resonance`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_resonance(...)  // see omc_help
```

### `arr_sort_int`

**Signature**: `(arr) -> int`

`arr_sort_int`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_sort_int(...)  // see omc_help
```

### `arr_sum`

**Signature**: `(arr, ...) -> array`

`arr_sum`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_sum(...)  // see omc_help
```

### `arr_sum_sq`

**Signature**: `(arr) -> float`

`arr_sum_sq`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_sum_sq(...)  // see omc_help
```

### `arr_take`

**Signature**: `(arr, ...) -> array`

`arr_take`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_take(...)  // see omc_help
```

### `arr_unique`

**Signature**: `(arr, ...) -> array`

`arr_unique`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_unique(...)  // see omc_help
```

### `arr_unique_count`

**Signature**: `(arr) -> int`

`arr_unique_count`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_unique_count(...)  // see omc_help
```

### `arr_window`

**Signature**: `(arr, ...) -> array`

`arr_window`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_window(...)  // see omc_help
```

### `arr_zeros`

**Signature**: `(arr, ...) -> array`

`arr_zeros`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_zeros(...)  // see omc_help
```

### `arr_zip`

**Signature**: `(arr, ...) -> array`

`arr_zip`: see omc_explain or source for details. Auto-generated stub.

```omc
arr_zip(...)  // see omc_help
```

### `arr_all`

**Signature**: `(arr, pred_fn?) -> int`

1 if every element is truthy (or matches predicate).

```omc
arr_all([1,1,1])  // 1
```

### `arr_any`

**Signature**: `(arr, pred_fn?) -> int`

1 if any element is truthy (or matches predicate).

```omc
arr_any([0,0,1])  // 1
```

### `arr_avg_distance`

**Signature**: `(arr) -> float`

Average pairwise distance between elements.

```omc
arr_avg_distance([1,2,3,4])  // 1.0
```

### `arr_chunk`

**Signature**: `(arr, n: int) -> array[]`

Split into chunks of size n.

```omc
arr_chunk([1,2,3,4,5], 2)  // [[1,2],[3,4],[5]]
```

### `arr_contains`

**Signature**: `(arr, value) -> int`

1 if value appears in arr.

```omc
arr_contains([1,2,3], 2)  // 1
```

### `arr_count`

**Signature**: `(arr, value) -> int`

Number of times value appears.

```omc
arr_count([1,2,2,3], 2)  // 2
```

### `arr_cumsum`

**Signature**: `(arr) -> array`

Cumulative sum of elements.

```omc
arr_cumsum([1,2,3])  // [1,3,6]
```

### `arr_diff`

**Signature**: `(arr) -> array`

First differences (out[i] = arr[i+1] - arr[i]).

```omc
arr_diff([1,3,6,10])  // [2,3,4]
```

### `arr_drop`

**Signature**: `(arr, n: int) -> array`

Skip the first n elements.

```omc
arr_drop([1,2,3,4], 2)  // [3,4]
```

### `arr_enumerate`

**Signature**: `(arr) -> array`

Pairs of (index, value).

```omc
arr_enumerate(["a","b"])  // [[0,"a"],[1,"b"]]
```

### `arr_find`

**Signature**: `(arr, pred_fn) -> any`

First element matching predicate; null if none.

```omc
arr_find([1,2,3], fn(x){return x>1;})  // 2
```

### `arr_first`

**Signature**: `(arr) -> any`

First element, or null if empty.

```omc
arr_first([1,2,3])  // 1
```

### `arr_flatten`

**Signature**: `(arr_of_arrays) -> array`

One-level flatten.

```omc
arr_flatten([[1,2],[3,4]])  // [1,2,3,4]
```

### `arr_from_range`

**Signature**: `(start, end) -> int[]`

[start, start+1, ..., end-1].

```omc
arr_from_range(0, 5)  // [0,1,2,3,4]
```

### `arr_gcd`

**Signature**: `(arr: int[]) -> int`

GCD of all elements.

```omc
arr_gcd([12, 18, 24])  // 6
```

### `arr_geometric_mean`

**Signature**: `(arr) -> float`

n-th root of product.

```omc
arr_geometric_mean([1.0, 4.0])  // 2.0
```

### `arr_harmonic_mean`

**Signature**: `(arr) -> float`

n / sum(1/xi).

```omc
arr_harmonic_mean([1.0, 2.0])  // 1.333
```

### `arr_index_of`

**Signature**: `(arr, value) -> int`

Position of first occurrence; -1 if not found.

```omc
arr_index_of([1,2,3], 2)  // 1
```

### `arr_is_sorted`

**Signature**: `(arr) -> int`

1 if non-decreasing.

```omc
arr_is_sorted([1,2,3])  // 1
```

### `arr_join`

**Signature**: `(arr, sep: string) -> string`

Stringify and join with separator.

```omc
arr_join([1,2,3], ",")  // "1,2,3"
```

### `arr_last`

**Signature**: `(arr) -> any`

Last element, or null if empty.

```omc
arr_last([1,2,3])  // 3
```

### `arr_max`

**Signature**: `(arr) -> any`

Maximum element.

```omc
arr_max([3,1,4])  // 4
```

### `arr_max_float`

**Signature**: `(arr) -> float`

Maximum element (typed-float).

```omc
arr_max_float([1.0, 2.5, 0.5])  // 2.5
```

### `arr_median`

**Signature**: `(arr) -> float`

Median of values.

```omc
arr_median([1.0, 2.0, 3.0])  // 2.0
```

### `arr_min`

**Signature**: `(arr) -> any`

Minimum element.

```omc
arr_min([3,1,4])  // 1
```

### `arr_norm`

**Signature**: `(arr) -> float`

Euclidean norm (L2).

```omc
arr_norm([3.0, 4.0])  // 5.0
```

### `arr_ones`

**Signature**: `(n: int) -> int[]`

n-length array of ones.

```omc
arr_ones(3)  // [1,1,1]
```

### `arr_partition_by`

**Signature**: `(arr, pred_fn) -> [matching, rest]`

Two arrays split on predicate.

```omc
arr_partition_by([1,2,3,4], fn(x){return x>2;})  // [[3,4], [1,2]]
```

### `arr_product`

**Signature**: `(arr) -> int|float`

Product of elements.

```omc
arr_product([2,3,4])  // 24
```

### `arr_range`

**Signature**: `(start, end, step?) -> int[]`

Range with optional step.

```omc
arr_range(0, 10, 2)  // [0,2,4,6,8]
```

### `arr_reduce`

**Signature**: `(arr, fn, init) -> any`

Left fold with initial accumulator.

```omc
arr_reduce([1,2,3], fn(a,b){return a+b;}, 0)  // 6
```

### `arr_repeat`

**Signature**: `(value, n: int) -> array`

n-length array of value.

```omc
arr_repeat("x", 3)  // ["x","x","x"]
```

### `arr_sort_int`

**Signature**: `(arr) -> int[]`

Sort integer array ascending.

```omc
arr_sort_int([3,1,2])  // [1,2,3]
```

### `arr_sum`

**Signature**: `(arr) -> int|float`

Sum of elements.

```omc
arr_sum([1,2,3])  // 6
```

### `arr_sum_sq`

**Signature**: `(arr) -> float`

Sum of squares.

```omc
arr_sum_sq([3, 4])  // 25
```

### `arr_take`

**Signature**: `(arr, n: int) -> array`

Take the first n elements.

```omc
arr_take([1,2,3,4], 2)  // [1,2]
```

### `arr_unique`

**Signature**: `(arr) -> array`

Deduplicate preserving order.

```omc
arr_unique([1,2,2,3,1])  // [1,2,3]
```

### `arr_unique_count`

**Signature**: `(arr) -> int`

Number of distinct values.

```omc
arr_unique_count([1,2,2,3])  // 3
```

### `arr_window`

**Signature**: `(arr, size: int) -> array[]`

Sliding windows of given size.

```omc
arr_window([1,2,3,4], 2)  // [[1,2],[2,3],[3,4]]
```

### `arr_zeros`

**Signature**: `(n: int) -> int[]`

n-length array of zeros.

```omc
arr_zeros(3)  // [0,0,0]
```

### `arr_zip`

**Signature**: `(a, b) -> [a_i, b_i][]`

Zip two arrays into pairs.

```omc
arr_zip([1,2], [10,20])  // [[1,10],[2,20]]
```

---

## linalg

### `arr_matmul`

**Signature**: `(A, B) -> matrix`

Matrix multiplication A@B with cache-friendly ikj loop. Integer-in/integer-out preserves substrate metadata per cell.

```omc
arr_matmul([[1,2],[3,4]], [[5,6],[7,8]])  // [[19,22],[43,50]]
```

### `arr_transpose`

**Signature**: `(M) -> matrix`

Transpose 2D matrix.

```omc
arr_transpose([[1,2,3],[4,5,6]])  // [[1,4],[2,5],[3,6]]
```

### `arr_eye`

**Signature**: `(n) -> matrix`

n×n identity matrix.

```omc
arr_eye(3)  // [[1,0,0],[0,1,0],[0,0,1]]
```

### `arr_zeros_2d`

**Signature**: `(rows, cols) -> matrix`

rows×cols zero matrix.

```omc
arr_zeros_2d(2,3)  // [[0,0,0],[0,0,0]]
```

---

## ml_kernels

### `arr_softmax`

**Signature**: `(arr: float[]) -> float[]`

Numerically stable softmax (max-subtraction trick).

```omc
arr_softmax([1.0,2.0,3.0])  // ~[0.09,0.24,0.67]
```

### `arr_layer_norm`

**Signature**: `(arr, eps=1e-5) -> float[]`

LayerNorm: (x-mean)/sqrt(var+eps).

```omc
arr_layer_norm([1.0,2.0,3.0,4.0,5.0])  // zero-mean, unit-variance
```

### `arr_relu_vec`

**Signature**: `(arr: float[]) -> float[]`

Elementwise max(x, 0).

```omc
arr_relu_vec([-1.0,0.0,2.5])  // [0.0,0.0,2.5]
```

### `arr_sigmoid_vec`

**Signature**: `(arr: float[]) -> float[]`

Elementwise 1/(1+exp(-x)).

```omc
arr_sigmoid_vec([0.0])  // [0.5]
```

### `arr_conv1d`

**Signature**: `(input, kernel) -> float[]`

1D valid-mode convolution.

```omc
arr_conv1d([1,2,3,4,5], [1,1,1])  // [6,9,12]
```

### `arr_outer`

**Signature**: `(a, b) -> matrix`

Outer product: a[i]*b[j] for every (i,j).

```omc
arr_outer([1,2], [10,20])  // [[10,20],[20,40]]
```

---

## substrate

### `is_attractor` 🔱 *OMC-unique*

**Signature**: `(n: int) -> int`

1 iff n is a Fibonacci attractor (0,1,2,3,5,8,13,...).

```omc
is_attractor(8)  // 1 ; is_attractor(7)  // 0
```

### `attractor_distance` 🔱 *OMC-unique*

**Signature**: `(n: int) -> int`

Absolute distance to the nearest Fibonacci attractor.

```omc
attractor_distance(7)  // 1 (8 is nearest)
```

### `arr_resonance_vec` 🔱 *OMC-unique*

**Signature**: `(arr) -> float[]`

Per-element φ-resonance (∈[0,1], 1=on Fibonacci attractor).

```omc
arr_resonance_vec([8,13,21])  // [1.0,1.0,1.0]
```

### `arr_him_vec` 🔱 *OMC-unique*

**Signature**: `(arr) -> float[]`

Per-element HIM (Harmonic Interference Metric).

```omc
arr_him_vec([1,2,3,5])  // ~[<0.5 each]
```

### `arr_fold_all` 🔱 *OMC-unique*

**Signature**: `(arr) -> int[]`

Snap every element to its nearest Fibonacci attractor.

```omc
arr_fold_all([7,100,9])  // [8,89,8]
```

### `arr_substrate_attention` 🔱 *OMC-unique*

**Signature**: `(Q, K, V) -> matrix`

Attention scored by substrate distance (not dot product). Closer in Fibonacci-space = higher weight.

```omc
arr_substrate_attention(Q, K, V)  // (n_q × v_cols) output
```

### `arr_substrate_score_rows` 🔱 *OMC-unique*

**Signature**: `(matrix) -> float[]`

Per-row mean φ-resonance. Use as a substrate-coherence regularizer.

```omc
arr_substrate_score_rows([[1,2,3,5],[7,11,13,19]])  // [~1.0, lower]
```

### `crt_recover` 🔱 *OMC-unique*

**Signature**: `(remainders: int[], moduli: int[]) -> int`

Chinese Remainder Theorem recovery from per-modulus remainders.

```omc
crt_recover([2,3,2], [5,7,3])  // 23
```

### `fibonacci_index` 🔱 *OMC-unique*

**Signature**: `(n: int) -> int`

Position in Fibonacci sequence (-1 if not an attractor).

```omc
fibonacci_index(13)  // 7  ; fibonacci_index(14)  // -1
```

### `res` 🔱 *OMC-unique*

**Signature**: `(n: int) -> float`

φ-resonance of a single value (0..1, 1=on Fibonacci attractor).

```omc
res(8)  // 1.0  ; res(7)  // <1.0
```

### `harmony` 🔱 *OMC-unique*

**Signature**: `(n: int) -> float`

HBit harmony score derived from substrate alignment.

```omc
harmony(89)  // high (89 is Fibonacci)
```

### `attractor_bucket` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`attractor_bucket`: see omc_explain or source for details. Auto-generated stub.

```omc
attractor_bucket(...)  // see omc_help
```

### `crt_residues` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`crt_residues`: see omc_explain or source for details. Auto-generated stub.

```omc
crt_residues(...)  // see omc_help
```

### `harmonic_dedupe` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_dedupe`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_dedupe(...)  // see omc_help
```

### `harmonic_diff` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_diff`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_diff(...)  // see omc_help
```

### `harmonic_hash` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_hash`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_hash(...)  // see omc_help
```

### `harmonic_partition` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_partition`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_partition(...)  // see omc_help
```

### `harmonic_read_file` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_read_file`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_read_file(...)  // see omc_help
```

### `harmonic_score` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_score`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_score(...)  // see omc_help
```

### `harmonic_sort` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_sort`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_sort(...)  // see omc_help
```

### `harmonic_split` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`harmonic_split`: see omc_explain or source for details. Auto-generated stub.

```omc
harmonic_split(...)  // see omc_help
```

### `hbit_tension` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`hbit_tension`: see omc_explain or source for details. Auto-generated stub.

```omc
hbit_tension(...)  // see omc_help
```

### `largest_attractor_at_most` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`largest_attractor_at_most`: see omc_explain or source for details. Auto-generated stub.

```omc
largest_attractor_at_most(...)  // see omc_help
```

### `phi_shadow` 🔱 *OMC-unique*

**Signature**: `(...) -> any`

`phi_shadow`: see omc_explain or source for details. Auto-generated stub.

```omc
phi_shadow(...)  // see omc_help
```

### `attractor_bucket` 🔱 *OMC-unique*

**Signature**: `(n: int, k: int) -> int`

Bucket n into one of k Fibonacci-distance bands.

```omc
attractor_bucket(7, 5)  // 0..4
```

### `crt_residues` 🔱 *OMC-unique*

**Signature**: `(n: int, moduli: int[]) -> int[]`

Per-modulus remainders of n.

```omc
crt_residues(23, [5,7,3])  // [3,2,2]
```

### `harmonic_dedupe` 🔱 *OMC-unique*

**Signature**: `(arr) -> array`

Deduplicate by harmonic distance (close items merge).

```omc
harmonic_dedupe([1, 1, 100, 99])  // [1, 100]
```

### `harmonic_diff` 🔱 *OMC-unique*

**Signature**: `(a, b) -> float`

Difference in harmonic space.

```omc
harmonic_diff(8, 13)  // small
```

### `harmonic_hash` 🔱 *OMC-unique*

**Signature**: `(s: string) -> int`

Substrate-aware hash that maps to a Fibonacci attractor.

```omc
harmonic_hash("foo")  // attractor-aligned int
```

### `harmonic_partition` 🔱 *OMC-unique*

**Signature**: `(arr) -> [groups]`

Group elements by harmonic similarity.

```omc
harmonic_partition(xs)  // [[similar], [other]]
```

### `harmonic_read_file` 🔱 *OMC-unique*

**Signature**: `(path: string) -> array`

Read file, splitting on harmonic boundaries.

```omc
harmonic_read_file("log.txt")
```

### `harmonic_score` 🔱 *OMC-unique*

**Signature**: `(value) -> float`

Single-value harmonic coherence score.

```omc
harmonic_score(8)  // ~1.0
```

### `harmonic_sort` 🔱 *OMC-unique*

**Signature**: `(arr) -> array`

Sort by substrate-coherence rather than numeric value.

```omc
harmonic_sort([1, 7, 8, 100])
```

### `harmonic_split` 🔱 *OMC-unique*

**Signature**: `(s: string, sep: string) -> array`

Split with substrate-aware merging.

```omc
harmonic_split("x,y", ",")
```

### `is_singularity` 🔱 *OMC-unique*

**Signature**: `(value) -> int`

1 if value is the Singularity zero-division marker.

```omc
is_singularity(0/0)  // 1 in safe mode
```

### `largest_attractor_at_most` 🔱 *OMC-unique*

**Signature**: `(n: int) -> int`

Largest Fibonacci ≤ n.

```omc
largest_attractor_at_most(50)  // 34
```

### `phi_pi_fib_search` 🔱 *OMC-unique*

**Signature**: `(arr: int[], target: int) -> int`

O(log_phiπF |arr|) search.

```omc
phi_pi_fib_search([1,2,3,5,8,13], 5)  // 3
```

### `phi_shadow` 🔱 *OMC-unique*

**Signature**: `(a: int, b: int) -> int`

Divergent-band β computation.

```omc
phi_shadow(3, 5)
```

### `zeckendorf_weight` 🔱 *OMC-unique*

**Signature**: `(n: int) -> int`

Number of Fibonacci terms in n's Zeckendorf form.

```omc
zeckendorf_weight(10)  // 2
```

---

## autograd

### `tape_reset`

**Signature**: `() -> null`

Clear the autograd tape before starting a fresh forward pass.

```omc
tape_reset();
```

### `tape_var`

**Signature**: `(value) -> int`

Lift a value onto the tape as a leaf variable. Returns node id.

```omc
h x = tape_var(3.0);
```

### `tape_const`

**Signature**: `(value) -> int`

Lift a value as a constant (no gradient flows through).

```omc
h c = tape_const(2.0);
```

### `tape_value` 🔱 *OMC-unique*

**Signature**: `(node_id) -> any`

Read forward value at a node. Integral results come back as substrate-annotated HInt.

```omc
tape_value(y)  // current forward value at y
```

### `tape_grad`

**Signature**: `(node_id) -> any`

Read accumulated gradient at a node after tape_backward.

```omc
tape_grad(x)  // dL/dx
```

### `tape_add`

**Signature**: `(a_id, b_id) -> int`

Record a+b on the tape.

```omc
h s = tape_add(x, y);
```

### `tape_mul`

**Signature**: `(a_id, b_id) -> int`

Record a*b on the tape (elementwise/broadcast).

```omc
h p = tape_mul(x, x);  // x^2
```

### `tape_matmul`

**Signature**: `(A_id, B_id) -> int`

Record A@B on the tape. Backward: dA=dy@B^T, dB=A^T@dy.

```omc
h Y = tape_matmul(X, W);
```

### `tape_relu`

**Signature**: `(a_id) -> int`

Record max(a,0). Backward: pass gradient where a>0, else 0.

```omc
h h = tape_relu(z);
```

### `tape_sigmoid`

**Signature**: `(a_id) -> int`

Record sigmoid(a). Backward: y*(1-y).

```omc
h h = tape_sigmoid(z);
```

### `tape_sum`

**Signature**: `(a_id) -> int`

Record sum-of-cells reduction. Often used as the loss.

```omc
h L = tape_sum(Y);
```

### `tape_mean`

**Signature**: `(a_id) -> int`

Record mean reduction.

```omc
h L = tape_mean(Y);
```

### `tape_backward`

**Signature**: `(loss_id) -> null`

Walk the tape in reverse; populates grads on every node.

```omc
tape_backward(L);
```

### `tape_update`

**Signature**: `(var_id, lr) -> null`

In-place SGD step: value -= lr * grad.

```omc
tape_update(W, 0.01);
```

### `tape_neg`

**Signature**: `(...) -> int`

`tape_neg`: see omc_explain or source for details. Auto-generated stub.

```omc
tape_neg(...)  // see omc_help
```

### `tape_pow_int`

**Signature**: `(...) -> int`

`tape_pow_int`: see omc_explain or source for details. Auto-generated stub.

```omc
tape_pow_int(...)  // see omc_help
```

### `tape_neg`

**Signature**: `(a_id) -> int`

Record -a on the tape.

```omc
tape_neg(x)
```

### `tape_pow_int`

**Signature**: `(a_id, n: int) -> int`

Record a^n on the tape.

```omc
tape_pow_int(x, 3)
```

---

## duals

### `dual`

**Signature**: `(value, derivative) -> [v,d]`

Lift a scalar into a forward-mode dual number.

```omc
h x = dual(3.0, 1.0);
```

### `dual_mul`

**Signature**: `(a, b) -> [v,d]`

Multiply two dual numbers (scalars auto-lift to deriv=0).

```omc
h y = dual_mul(x, x);  // y is dual carrying x^2 + 2x*dx
```

### `dual_d`

**Signature**: `(dual) -> float`

Read the derivative component.

```omc
dual_d(y)  // current df/dx
```

### `dual_cos`

**Signature**: `(...) -> any`

`dual_cos`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_cos(...)  // see omc_help
```

### `dual_exp`

**Signature**: `(...) -> any`

`dual_exp`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_exp(...)  // see omc_help
```

### `dual_neg`

**Signature**: `(...) -> any`

`dual_neg`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_neg(...)  // see omc_help
```

### `dual_pow_int`

**Signature**: `(...) -> any`

`dual_pow_int`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_pow_int(...)  // see omc_help
```

### `dual_relu`

**Signature**: `(...) -> any`

`dual_relu`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_relu(...)  // see omc_help
```

### `dual_sigmoid`

**Signature**: `(...) -> any`

`dual_sigmoid`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_sigmoid(...)  // see omc_help
```

### `dual_sin`

**Signature**: `(...) -> any`

`dual_sin`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_sin(...)  // see omc_help
```

### `dual_tanh`

**Signature**: `(...) -> any`

`dual_tanh`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_tanh(...)  // see omc_help
```

### `dual_v`

**Signature**: `(...) -> any`

`dual_v`: see omc_explain or source for details. Auto-generated stub.

```omc
dual_v(...)  // see omc_help
```

### `dual_cos`

**Signature**: `(a) -> [v,d]`

cos(a).

```omc
dual_cos(x)
```

### `dual_exp`

**Signature**: `(a) -> [v,d]`

exp(a).

```omc
dual_exp(x)
```

### `dual_neg`

**Signature**: `(a) -> [v,d]`

Negate.

```omc
dual_neg(x)
```

### `dual_pow_int`

**Signature**: `(a, n: int) -> [v,d]`

a^n.

```omc
dual_pow_int(x, 3)
```

### `dual_relu`

**Signature**: `(a) -> [v,d]`

max(a, 0).

```omc
dual_relu(x)
```

### `dual_sigmoid`

**Signature**: `(a) -> [v,d]`

sigmoid(a).

```omc
dual_sigmoid(x)
```

### `dual_sin`

**Signature**: `(a) -> [v,d]`

sin(a).

```omc
dual_sin(x)
```

### `dual_tanh`

**Signature**: `(a) -> [v,d]`

tanh(a).

```omc
dual_tanh(x)
```

### `dual_v`

**Signature**: `(d) -> float`

Read value of dual.

```omc
dual_v(x)
```

---

## generators

### `gen_stream`

**Signature**: `(thunk, callback) -> int`

Run a generator with callback per yield. O(1) memory. Returns 1 if completed, 0 if shorted.

```omc
gen_stream(fn(){ return fib(1000000); }, fn(v){ return 1; });
```

### `gen_take`

**Signature**: `(thunk, n) -> array`

Pull the first n values from a lazy generator.

```omc
gen_take(fn(){ return count(); }, 5)  // [1,2,3,4,5]
```

### `gen_count`

**Signature**: `(thunk) -> int`

Count yields without storing them.

```omc
gen_count(fn(){ return count_to(100); })  // 100
```

### `gen_sum`

**Signature**: `(thunk) -> int`

Sum integer yields without storing them.

```omc
gen_sum(fn(){ return count_to(1000); })  // 500500
```

### `gen_substrate_fib` 🔱 *OMC-unique*

**Signature**: `(callback, max) -> int`

Native lazy Fibonacci stream up to max. Each value is on-attractor.

```omc
gen_substrate_fib(fn(v){ print(v); return 1; }, 100);
```

---

## strings

### `str_len`

**Signature**: `(s: string) -> int`

Byte length of string (NOT char count for non-ASCII).

```omc
str_len("hello")  // 5
```

### `str_split`

**Signature**: `(s, sep) -> string[]`

Split on separator.

```omc
str_split("a,b,c", ",")  // ["a","b","c"]
```

### `str_join`

**Signature**: `(arr, sep) -> string`

Join string array with separator.

```omc
str_join(["a","b"], "-")  // "a-b"
```

### `str_slice`

**Signature**: `(s, start, end) -> string`

Character-indexed substring [start..end).

```omc
str_slice("abcdef", 1, 4)  // "bcd"
```

### `concat_many`

**Signature**: `(...) -> string`

Concatenate any number of values as strings.

```omc
concat_many("x=", 42, " y=", 99)  // "x=42 y=99"
```

### `str_capitalize`

**Signature**: `(s, ...) -> string`

`str_capitalize`: see omc_explain or source for details. Auto-generated stub.

```omc
str_capitalize(...)  // see omc_help
```

### `str_chars`

**Signature**: `(s, ...) -> string`

`str_chars`: see omc_explain or source for details. Auto-generated stub.

```omc
str_chars(...)  // see omc_help
```

### `str_concat`

**Signature**: `(s, ...) -> string`

`str_concat`: see omc_explain or source for details. Auto-generated stub.

```omc
str_concat(...)  // see omc_help
```

### `str_contains`

**Signature**: `(s, ...) -> string`

`str_contains`: see omc_explain or source for details. Auto-generated stub.

```omc
str_contains(...)  // see omc_help
```

### `str_count`

**Signature**: `(s) -> int`

`str_count`: see omc_explain or source for details. Auto-generated stub.

```omc
str_count(...)  // see omc_help
```

### `str_ends_with`

**Signature**: `(s, ...) -> string`

`str_ends_with`: see omc_explain or source for details. Auto-generated stub.

```omc
str_ends_with(...)  // see omc_help
```

### `str_index_of`

**Signature**: `(s) -> int`

`str_index_of`: see omc_explain or source for details. Auto-generated stub.

```omc
str_index_of(...)  // see omc_help
```

### `str_is_empty`

**Signature**: `(s, ...) -> string`

`str_is_empty`: see omc_explain or source for details. Auto-generated stub.

```omc
str_is_empty(...)  // see omc_help
```

### `str_lowercase`

**Signature**: `(s, ...) -> string`

`str_lowercase`: see omc_explain or source for details. Auto-generated stub.

```omc
str_lowercase(...)  // see omc_help
```

### `str_pad_left`

**Signature**: `(s, ...) -> string`

`str_pad_left`: see omc_explain or source for details. Auto-generated stub.

```omc
str_pad_left(...)  // see omc_help
```

### `str_pad_right`

**Signature**: `(s, ...) -> string`

`str_pad_right`: see omc_explain or source for details. Auto-generated stub.

```omc
str_pad_right(...)  // see omc_help
```

### `str_repeat`

**Signature**: `(s, ...) -> string`

`str_repeat`: see omc_explain or source for details. Auto-generated stub.

```omc
str_repeat(...)  // see omc_help
```

### `str_replace`

**Signature**: `(s, ...) -> string`

`str_replace`: see omc_explain or source for details. Auto-generated stub.

```omc
str_replace(...)  // see omc_help
```

### `str_reverse`

**Signature**: `(s, ...) -> string`

`str_reverse`: see omc_explain or source for details. Auto-generated stub.

```omc
str_reverse(...)  // see omc_help
```

### `str_split_lines`

**Signature**: `(s, ...) -> string`

`str_split_lines`: see omc_explain or source for details. Auto-generated stub.

```omc
str_split_lines(...)  // see omc_help
```

### `str_starts_with`

**Signature**: `(s, ...) -> string`

`str_starts_with`: see omc_explain or source for details. Auto-generated stub.

```omc
str_starts_with(...)  // see omc_help
```

### `str_to_float`

**Signature**: `(s, ...) -> string`

`str_to_float`: see omc_explain or source for details. Auto-generated stub.

```omc
str_to_float(...)  // see omc_help
```

### `str_to_int`

**Signature**: `(s, ...) -> string`

`str_to_int`: see omc_explain or source for details. Auto-generated stub.

```omc
str_to_int(...)  // see omc_help
```

### `str_trim`

**Signature**: `(s, ...) -> string`

`str_trim`: see omc_explain or source for details. Auto-generated stub.

```omc
str_trim(...)  // see omc_help
```

### `str_uppercase`

**Signature**: `(s, ...) -> string`

`str_uppercase`: see omc_explain or source for details. Auto-generated stub.

```omc
str_uppercase(...)  // see omc_help
```

### `str_chars`

**Signature**: `(s) -> string[]`

Split into single-char strings.

```omc
str_chars("ab")  // ["a","b"]
```

### `str_count`

**Signature**: `(s, sub) -> int`

Non-overlapping occurrences.

```omc
str_count("banana", "a")  // 3
```

### `str_ends_with`

**Signature**: `(s, suffix) -> int`

1 if s ends with suffix.

```omc
str_ends_with("hello", "lo")  // 1
```

### `str_index_of`

**Signature**: `(s, sub) -> int`

Byte index of first occurrence; -1 if missing.

```omc
str_index_of("hello", "ll")  // 2
```

### `str_repeat`

**Signature**: `(s, n) -> string`

Repeat s n times.

```omc
str_repeat("ab", 3)  // "ababab"
```

### `str_replace`

**Signature**: `(s, find, replace) -> string`

Replace ALL occurrences.

```omc
str_replace("a.b", ".", "_")  // "a_b"
```

### `str_starts_with`

**Signature**: `(s, prefix) -> int`

1 if s begins with prefix.

```omc
str_starts_with("hello", "he")  // 1
```

### `str_trim`

**Signature**: `(s) -> string`

Strip leading/trailing whitespace.

```omc
str_trim("  x  ")  // "x"
```

---

## regex

### `re_match`

**Signature**: `(pattern, s) -> int`

1 if pattern matches anywhere in s, 0 otherwise.

```omc
re_match("^\\d+$", "123")  // 1
```

### `re_find_all`

**Signature**: `(pattern, s) -> string[]`

All non-overlapping matches.

```omc
re_find_all("\\d+", "a12 b34")  // ["12","34"]
```

### `re_replace`

**Signature**: `(pattern, s, replacement) -> string`

Replace all matches.

```omc
re_replace("\\d+", "a1b2", "X")  // "aXbX"
```

### `re_find`

**Signature**: `(pattern, s, ...) -> string|int|array`

`re_find`: see omc_explain or source for details. Auto-generated stub.

```omc
re_find(...)  // see omc_help
```

### `re_split`

**Signature**: `(pattern, s, ...) -> string|int|array`

`re_split`: see omc_explain or source for details. Auto-generated stub.

```omc
re_split(...)  // see omc_help
```

### `re_find`

**Signature**: `(pattern, s) -> string`

First match, or empty string.

```omc
re_find("\d+", "abc123")  // "123"
```

### `re_split`

**Signature**: `(pattern, s) -> string[]`

Split by regex.

```omc
re_split("\s+", "a b  c")  // ["a","b","c"]
```

---

## json

### `json_parse`

**Signature**: `(s: string) -> any`

Parse JSON into OMC value (object→dict, array→array).

```omc
json_parse("{\"x\":1}")  // dict
```

### `json_stringify`

**Signature**: `(value) -> string`

Serialize OMC value to JSON.

```omc
json_stringify([1,2,3])  // "[1,2,3]"
```

---

## stdlib

### `sha256`

**Signature**: `(s: string) -> string`

SHA-256 of input string, as 64-char hex.

```omc
sha256("hello")  // "2cf2..."
```

### `sha512`

**Signature**: `(s: string) -> string`

SHA-512 of input string, as 128-char hex.

```omc
sha512("x")  // 128 chars
```

### `base64_encode`

**Signature**: `(s: string) -> string`

Standard base64 encoding.

```omc
base64_encode("hi")  // "aGk="
```

### `base64_decode`

**Signature**: `(s: string) -> string`

Decode standard base64.

```omc
base64_decode("aGk=")  // "hi"
```

### `now_unix`

**Signature**: `() -> int`

Current Unix timestamp in seconds.

```omc
now_unix()  // 1747400000
```

### `now_iso`

**Signature**: `() -> string`

Current ISO-8601 UTC datetime string.

```omc
now_iso()  // "2026-05-16T12:34:56Z"
```

### `format_time`

**Signature**: `(unix_ts, fmt) -> string`

Format a unix timestamp via strftime-style fmt.

```omc
format_time(0, "%Y-%m-%d")  // "1970-01-01"
```

### `parse_time`

**Signature**: `(s, fmt) -> int`

Parse string via strftime fmt into unix timestamp.

```omc
parse_time("2026-05-16", "%Y-%m-%d")  // 1747353600
```

### `csv_parse`

**Signature**: `(...) -> any`

`csv_parse`: see omc_explain or source for details. Auto-generated stub.

```omc
csv_parse(...)  // see omc_help
```

### `file_exists`

**Signature**: `(...) -> any`

`file_exists`: see omc_explain or source for details. Auto-generated stub.

```omc
file_exists(...)  // see omc_help
```

### `read_file`

**Signature**: `(...) -> any`

`read_file`: see omc_explain or source for details. Auto-generated stub.

```omc
read_file(...)  // see omc_help
```

### `write_file`

**Signature**: `(...) -> any`

`write_file`: see omc_explain or source for details. Auto-generated stub.

```omc
write_file(...)  // see omc_help
```

### `cleanup_array`

**Signature**: `(arr) -> null`

Free internal slack capacity in an array.

```omc
cleanup_array(xs);
```

### `csv_parse`

**Signature**: `(text: string) -> string[][]`

Parse RFC-4180 CSV into rows of cells.

```omc
csv_parse("a,b
c,d")  // [["a","b"],["c","d"]]
```

### `defined_functions`

**Signature**: `() -> string[]`

All user + builtin function names currently in scope.

```omc
defined_functions()
```

### `error`

**Signature**: `(msg: string) -> null`

Raise a catchable error.

```omc
error("bad input");
```

### `file_exists`

**Signature**: `(path: string) -> int`

1 if file exists at path.

```omc
file_exists("data.txt")  // 1 or 0
```

### `random_float`

**Signature**: `() -> float`

Uniform random float in [0, 1).

```omc
random_float()
```

### `random_int`

**Signature**: `(lo, hi) -> int`

Random int in [lo, hi).

```omc
random_int(0, 10)
```

### `random_seed`

**Signature**: `(seed: int) -> null`

Set RNG seed for deterministic runs.

```omc
random_seed(42);
```

### `read_file`

**Signature**: `(path: string) -> string`

Read entire file as string.

```omc
read_file("data.txt")
```

### `write_file`

**Signature**: `(path: string, content: string) -> null`

Write content to file (overwrite).

```omc
write_file("out.txt", "hello");
```

---

## exceptions

### `is_instance`

**Signature**: `(value, class_name: string) -> int`

1 if value is a class instance whose __class__ matches OR inherits from class_name.

```omc
is_instance(HttpError(...), "AppError")  // 1 if HttpError extends AppError
```

### `error`

**Signature**: `(...) -> any`

`error`: see omc_explain or source for details. Auto-generated stub.

```omc
error(...)  // see omc_help
```

---

## introspection

### `omc_help`

**Signature**: `(name: string) -> dict`

Look up metadata for a builtin: signature, description, example.

```omc
omc_help("arr_softmax")  // {name, signature, description, example, ...}
```

### `omc_list_builtins`

**Signature**: `(category?: string) -> string[]`

List all documented builtins. Pass category to filter.

```omc
omc_list_builtins("substrate")  // [is_attractor, attractor_distance, ...]
```

### `omc_categories`

**Signature**: `() -> string[]`

List all builtin categories.

```omc
omc_categories()  // [core, arrays, linalg, ml_kernels, substrate, ...]
```

### `omc_did_you_mean`

**Signature**: `(name: string) -> string[]`

Closest known builtin names for `name` (edit distance ≤ 3).

```omc
omc_did_you_mean("arr_softmx")  // ["arr_softmax"]
```

### `omc_unique_builtins`

**Signature**: `() -> string[]`

Builtins flagged as unique to OMC (no clean Python equivalent).

```omc
omc_unique_builtins()  // [is_attractor, arr_substrate_attention, ...]
```

### `omc_explain_error`

**Signature**: `(msg: string) -> dict`

Pattern-match an error message against the curated catalog. Returns {matched, pattern, category, explanation, typical_cause, fix}.

```omc
try { arr_softmx([1.0]); } catch e { print(dict_get(omc_explain_error(e), "fix")); }
```

### `omc_error_categories`

**Signature**: `() -> string[]`

All distinct error categories in the catalog.

```omc
omc_error_categories()  // [dispatch, arrays, linalg, ...]
```

### `omc_error_count`

**Signature**: `() -> int`

Number of curated error patterns. The knowledge base size.

```omc
omc_error_count()  // 42+
```

### `omc_completion_hint`

**Signature**: `(prefix: string) -> string[]`

Documented builtin names starting with `prefix`. IDE-style autocomplete.

```omc
omc_completion_hint("arr_sub")  // [arr_sub, arr_substrate_attention, ...]
```

### `omc_categories_count`

**Signature**: `() -> int`

Number of distinct builtin categories.

```omc
omc_categories_count()  // 15+
```

### `omc_builtin_count`

**Signature**: `() -> int`

Total documented builtins.

```omc
omc_builtin_count()  // 390+
```

### `omc_unique_count`

**Signature**: `() -> int`

Count of OMC-unique builtins.

```omc
omc_unique_count()  // 15+
```

### `omc_remember` 🔱 *OMC-unique*

**Signature**: `(name: string, code: string) -> int`

Store the canonical hash of `code` under `name`. Returns the stored hash. Session-level memory for LLMs.

```omc
omc_remember("loss_v1", "fn loss(p, t){ ... }")
```

### `omc_recall`

**Signature**: `(name: string) -> int|null`

Get the hash stored under `name`, or null.

```omc
omc_recall("loss_v1")  // 1234567890 or null
```

### `omc_recall_matches` 🔱 *OMC-unique*

**Signature**: `(name: string, code: string) -> int`

1 if the current code's canonical hash matches what was remembered. 'Did this change?'

```omc
omc_recall_matches("loss_v1", current_source)  // 0 if edited
```

### `omc_memory_keys`

**Signature**: `() -> string[]`

All names currently in code-memory.

```omc
omc_memory_keys()  // ["loss_v1", "feature_pipeline", ...]
```

### `omc_memory_clear`

**Signature**: `() -> null`

Drop all stored hashes. Use between independent sessions.

```omc
omc_memory_clear();
```

### `omc_help_markdown`

**Signature**: `(name: string) -> string`

Help rendered as Markdown — easier for chat-window consumers.

```omc
omc_help_markdown("arr_softmax")  // ### `arr_softmax`...
```

### `omc_help_all_category`

**Signature**: `(category: string) -> dict[]`

All builtins in `category` returned as omc_help dicts. Bulk reference.

```omc
omc_help_all_category("substrate")  // array of help dicts
```

### `omc_search_builtins`

**Signature**: `(query: string) -> string[]`

Substring search across name + description. Find what you don't know the name of.

```omc
omc_search_builtins("softmax")  // ["arr_softmax"]
```

### `cleanup_array`

**Signature**: `(...) -> any`

`cleanup_array`: see omc_explain or source for details. Auto-generated stub.

```omc
cleanup_array(...)  // see omc_help
```

### `defined_functions`

**Signature**: `(...) -> any`

`defined_functions`: see omc_explain or source for details. Auto-generated stub.

```omc
defined_functions(...)  // see omc_help
```

---

## tokenizer

### `omc_token_encode` 🔱 *OMC-unique*

**Signature**: `(code: string) -> int[]`

Encode OMC source as substrate-typed token IDs. Common builtins land on small Fibonacci attractors; round-trips exactly via omc_token_decode.

```omc
omc_token_encode("arr_softmax([1.0])")  // short int array
```

### `omc_token_decode` 🔱 *OMC-unique*

**Signature**: `(ids: int[]) -> string`

Inverse of omc_token_encode — reconstructs the original source.

```omc
omc_token_decode([1, 3, 0, 98])  // recovers source
```

### `omc_token_distance` 🔱 *OMC-unique*

**Signature**: `(id_a: int, id_b: int) -> int`

Substrate distance between two token IDs (sum of attractor-distances + raw delta). Free 'semantic nearness' signal — Python tokenizers have no analogue.

```omc
omc_token_distance(3, 5)  // both on attractors → small
```

### `omc_token_vocab` 🔱 *OMC-unique*

**Signature**: `() -> string[]`

Full token dictionary (index = ID, value = canonical substring).

```omc
omc_token_vocab()  // ["<escape>", "h ", " = ", "arr_get", ...]
```

### `omc_token_vocab_size`

**Signature**: `() -> int`

Number of dictionary entries.

```omc
omc_token_vocab_size()  // 150+
```

### `omc_token_compression_ratio` 🔱 *OMC-unique*

**Signature**: `(code: string) -> float`

Raw bytes / encoded ints. >1 means the encoder is shrinking the input.

```omc
omc_token_compression_ratio("arr_softmax([1.0])")  // ~3-5×
```

### `omc_token_pack` 🔱 *OMC-unique*

**Signature**: `(streams: int[], moduli?: int[]) -> int`

CRT-pack a stream of remainders into a single i64. Default moduli pack (kind, vocab_id, position_class) for multi-stream tokens.

```omc
omc_token_pack([3, 42, 7])  // single packed int
```

### `omc_token_unpack` 🔱 *OMC-unique*

**Signature**: `(packed: int, moduli?: int[]) -> int[]`

Inverse of omc_token_pack.

```omc
omc_token_unpack(packed)  // [kind, vocab_id, position_class]
```

### `omc_code_hash` 🔱 *OMC-unique*

**Signature**: `(code: string) -> dict`

Hash a program's token stream and fold to nearest Fibonacci attractor. Equivalent programs land on the same attractor. Returns {raw, attractor, distance, resonance}.

```omc
omc_code_hash("arr_softmax([1])")  // {attractor: ..., resonance: ...}
```

### `omc_code_distance` 🔱 *OMC-unique*

**Signature**: `(code_a: string, code_b: string) -> int`

Substrate distance between two programs (|hash_a - hash_b|). Same code → 0; small edits → small distance.

```omc
omc_code_distance("return 1;", "return 2;")  // small
```

### `omc_code_canonical` 🔱 *OMC-unique*

**Signature**: `(code: string) -> string`

Parse + AST-canonicalize + re-emit. Output is invariant under whitespace/comments/local-var-names/param-names/loop-vars/catch-vars/lambda-params. Top-level fn/class names + globals preserved.

```omc
omc_code_canonical("fn f(x) { return x; }") == omc_code_canonical("fn f(a) { return a; }")
```

### `omc_code_equivalent` 🔱 *OMC-unique*

**Signature**: `(code_a: string, code_b: string) -> int`

1 iff the two programs canonicalize identically (semantic alpha-equivalence). LLMs use this as a memory-key check: 'is this still the same function I was editing?'

```omc
omc_code_equivalent("fn f(x) { return x; }", "fn f(a) { return a; }")  // 1
```

### `omc_token_lookup`

**Signature**: `(id: int) -> string`

Inverse of token-id-from-name. Get the substring expanded by a single ID.

```omc
omc_token_lookup(3)  // "arr_get"
```

### `omc_token_describe`

**Signature**: `(ids: int[]) -> string`

Pretty-print an encoded stream as id=N expand="..." lines for debugging.

```omc
omc_token_describe(omc_token_encode("h x = 1;"))  // multi-line
```

### `omc_token_byte_savings`

**Signature**: `(code: string) -> int`

raw_bytes - encoded_tokens. Positive = compression win.

```omc
omc_token_byte_savings("arr_softmax")  // 10 (11 bytes -> 1 token)
```

### `omc_token_compress_pct`

**Signature**: `(code: string) -> float`

% bytes saved by encoding. 100 * (1 - ids_len / raw_len).

```omc
omc_token_compress_pct("arr_softmax")  // ~90.9
```

---

## code_intel

### `omc_code_summary`

**Signature**: `(code: string) -> dict`

Structured summary: {functions, classes, imports, calls, stmt_count}. Each function: {name, params, body_stmts, canonical_hash}.

```omc
omc_code_summary("fn f(x){return x;}")  // .functions[0].name == "f"
```

### `omc_code_extract_fns`

**Signature**: `(code: string) -> string[]`

Just the top-level function names (Class methods come as Class.method).

```omc
omc_code_extract_fns("fn f(){} fn g(){}")  // ["f", "g"]
```

### `omc_code_dependencies`

**Signature**: `(code: string) -> string[]`

Every name this program calls — both builtins and user-defined. 'What does this need to run?'

```omc
omc_code_dependencies("fn f(x){return arr_softmax(x);}")  // includes arr_softmax
```

### `omc_code_complexity`

**Signature**: `(code: string) -> dict`

{complexity, ast_size, ast_depth}. Cyclomatic complexity = branch points + 1.

```omc
omc_code_complexity("fn f(x){if x>0{return 1;} return 0;}")  // complexity:2
```

### `omc_code_minify`

**Signature**: `(code: string) -> string`

Canonicalize + strip newlines. Single-line wire form.

```omc
omc_code_minify("fn f(x){\n  return x;\n}")  // single line
```

### `omc_code_similarity`

**Signature**: `(a: string, b: string) -> float`

Jaccard over canonical-token multisets. 1.0 = alpha-equivalent.

```omc
omc_code_similarity("x+1", "x+2")  // close to 1
```

### `omc_code_fingerprint` 🔱 *OMC-unique*

**Signature**: `(code: string) -> int`

CRT-packed fingerprint of (hash_attractor, ast_size, complexity). Same on equivalent code.

```omc
omc_code_fingerprint("fn f(x){return x;}")  // stable int
```

### `omc_code_signature`

**Signature**: `(code: string) -> string`

Public API: one `fn name(params)` per line.

```omc
omc_code_signature("fn add(x,y){return x+y;}")  // "fn add(x, y)"
```

### `omc_code_uses_python`

**Signature**: `(code: string) -> int`

1 if any py_* call appears. Quick sandboxing/safety check.

```omc
omc_code_uses_python("py_import(\"numpy\");")  // 1
```

### `omc_code_uses_substrate` 🔱 *OMC-unique*

**Signature**: `(code: string) -> int`

1 if any OMC-unique primitive is called. 'Does this code reach for OMC's differentiators?'

```omc
omc_code_uses_substrate("return arr_resonance_vec(xs);")  // 1
```

### `omc_canonical_hash` 🔱 *OMC-unique*

**Signature**: `(code: string) -> dict`

canonicalize + hash. The semantic memory key. {raw, attractor, distance, resonance}.

```omc
omc_canonical_hash("fn f(a){return a;}")  // matches the b-variant
```

### `omc_substrate_score` 🔱 *OMC-unique*

**Signature**: `(code: string) -> float`

Fraction of CANONICAL tokens whose ID is a Fibonacci attractor. 1.0 = perfectly substrate-aligned.

```omc
omc_substrate_score("h x = arr_get(xs, 0);")  // 0..1
```

### `omc_attractor_density` 🔱 *OMC-unique*

**Signature**: `(code: string) -> float`

Like omc_substrate_score but over RAW source (no canonicalize). Compare formatting styles.

```omc
omc_attractor_density("h x = 1;")  // 0..1
```

### `omc_hbit_hash` 🔱 *OMC-unique*

**Signature**: `(code: string) -> int`

Hash blended with substrate-resonance of the hash itself — OMC-only dual-band hashing.

```omc
omc_hbit_hash("h x = 1;")  // substrate-weighted int
```

### `omc_code_diff` 🔱 *OMC-unique*

**Signature**: `(a: string, b: string) -> dict`

Structural diff between two programs (after canonicalization). {added, removed, modified, unchanged} as function-name arrays.

```omc
omc_code_diff(old, new)  // {modified: ["loss"], ...}
```

### `omc_code_metrics`

**Signature**: `(code: string) -> dict`

Bulk metrics: {complexity, ast_size, ast_depth, source_bytes, token_count, compression_ratio}. One call instead of N.

```omc
omc_code_metrics(src)  // all stats at once
```

---

## math

### `abs`

**Signature**: `(n) -> int|float`

`abs`: see omc_explain or source for details. Auto-generated stub.

```omc
abs(...)  // see omc_help
```

### `acos`

**Signature**: `(...) -> any`

`acos`: see omc_explain or source for details. Auto-generated stub.

```omc
acos(...)  // see omc_help
```

### `asin`

**Signature**: `(...) -> any`

`asin`: see omc_explain or source for details. Auto-generated stub.

```omc
asin(...)  // see omc_help
```

### `atan`

**Signature**: `(...) -> any`

`atan`: see omc_explain or source for details. Auto-generated stub.

```omc
atan(...)  // see omc_help
```

### `atan2`

**Signature**: `(...) -> any`

`atan2`: see omc_explain or source for details. Auto-generated stub.

```omc
atan2(...)  // see omc_help
```

### `bit_count`

**Signature**: `(...) -> any`

`bit_count`: see omc_explain or source for details. Auto-generated stub.

```omc
bit_count(...)  // see omc_help
```

### `bit_length`

**Signature**: `(...) -> any`

`bit_length`: see omc_explain or source for details. Auto-generated stub.

```omc
bit_length(...)  // see omc_help
```

### `ceil`

**Signature**: `(n) -> int|float`

`ceil`: see omc_explain or source for details. Auto-generated stub.

```omc
ceil(...)  // see omc_help
```

### `clamp`

**Signature**: `(...) -> any`

`clamp`: see omc_explain or source for details. Auto-generated stub.

```omc
clamp(...)  // see omc_help
```

### `cos`

**Signature**: `(...) -> any`

`cos`: see omc_explain or source for details. Auto-generated stub.

```omc
cos(...)  // see omc_help
```

### `digit_count`

**Signature**: `(...) -> any`

`digit_count`: see omc_explain or source for details. Auto-generated stub.

```omc
digit_count(...)  // see omc_help
```

### `digit_sum`

**Signature**: `(...) -> any`

`digit_sum`: see omc_explain or source for details. Auto-generated stub.

```omc
digit_sum(...)  // see omc_help
```

### `exp`

**Signature**: `(...) -> any`

`exp`: see omc_explain or source for details. Auto-generated stub.

```omc
exp(...)  // see omc_help
```

### `floor`

**Signature**: `(n) -> int|float`

`floor`: see omc_explain or source for details. Auto-generated stub.

```omc
floor(...)  // see omc_help
```

### `fnv1a_hash`

**Signature**: `(...) -> any`

`fnv1a_hash`: see omc_explain or source for details. Auto-generated stub.

```omc
fnv1a_hash(...)  // see omc_help
```

### `gcd`

**Signature**: `(...) -> any`

`gcd`: see omc_explain or source for details. Auto-generated stub.

```omc
gcd(...)  // see omc_help
```

### `lcm`

**Signature**: `(...) -> any`

`lcm`: see omc_explain or source for details. Auto-generated stub.

```omc
lcm(...)  // see omc_help
```

### `log`

**Signature**: `(...) -> any`

`log`: see omc_explain or source for details. Auto-generated stub.

```omc
log(...)  // see omc_help
```

### `log10`

**Signature**: `(...) -> any`

`log10`: see omc_explain or source for details. Auto-generated stub.

```omc
log10(...)  // see omc_help
```

### `log2`

**Signature**: `(...) -> any`

`log2`: see omc_explain or source for details. Auto-generated stub.

```omc
log2(...)  // see omc_help
```

### `max`

**Signature**: `(...) -> any`

`max`: see omc_explain or source for details. Auto-generated stub.

```omc
max(...)  // see omc_help
```

### `min`

**Signature**: `(...) -> any`

`min`: see omc_explain or source for details. Auto-generated stub.

```omc
min(...)  // see omc_help
```

### `mod_pow`

**Signature**: `(...) -> any`

`mod_pow`: see omc_explain or source for details. Auto-generated stub.

```omc
mod_pow(...)  // see omc_help
```

### `pow`

**Signature**: `(...) -> any`

`pow`: see omc_explain or source for details. Auto-generated stub.

```omc
pow(...)  // see omc_help
```

### `round`

**Signature**: `(n) -> int|float`

`round`: see omc_explain or source for details. Auto-generated stub.

```omc
round(...)  // see omc_help
```

### `sign`

**Signature**: `(n) -> int|float`

`sign`: see omc_explain or source for details. Auto-generated stub.

```omc
sign(...)  // see omc_help
```

### `sin`

**Signature**: `(...) -> any`

`sin`: see omc_explain or source for details. Auto-generated stub.

```omc
sin(...)  // see omc_help
```

### `sqrt`

**Signature**: `(...) -> any`

`sqrt`: see omc_explain or source for details. Auto-generated stub.

```omc
sqrt(...)  // see omc_help
```

### `tan`

**Signature**: `(...) -> any`

`tan`: see omc_explain or source for details. Auto-generated stub.

```omc
tan(...)  // see omc_help
```

### `abs`

**Signature**: `(n) -> int|float`

Absolute value.

```omc
abs(-5)  // 5
```

### `acos`

**Signature**: `(x: float) -> float`

Arc-cosine (radians).

```omc
acos(0.0)  // π/2
```

### `asin`

**Signature**: `(x: float) -> float`

Arc-sine (radians).

```omc
asin(0.0)  // 0
```

### `atan`

**Signature**: `(x: float) -> float`

Arc-tangent (radians).

```omc
atan(1.0)  // π/4
```

### `atan2`

**Signature**: `(y, x) -> float`

Arc-tangent of y/x with quadrant handling.

```omc
atan2(1, 1)  // π/4
```

### `bit_count`

**Signature**: `(n: int) -> int`

Popcount: number of set bits.

```omc
bit_count(7)  // 3
```

### `bit_length`

**Signature**: `(n: int) -> int`

Highest set bit index + 1.

```omc
bit_length(8)  // 4
```

### `ceil`

**Signature**: `(x: float) -> int`

Round up to next integer.

```omc
ceil(1.2)  // 2
```

### `clamp`

**Signature**: `(x, lo, hi) -> any`

Clip x into [lo, hi].

```omc
clamp(15, 0, 10)  // 10
```

### `cos`

**Signature**: `(x) -> float`

Cosine.

```omc
cos(0)  // 1.0
```

### `digit_count`

**Signature**: `(n: int) -> int`

Count of decimal digits.

```omc
digit_count(1234)  // 4
```

### `digit_sum`

**Signature**: `(n: int) -> int`

Sum of decimal digits.

```omc
digit_sum(123)  // 6
```

### `exp`

**Signature**: `(x) -> float`

e^x.

```omc
exp(0)  // 1.0
```

### `floor`

**Signature**: `(x: float) -> int`

Round down to next integer.

```omc
floor(1.8)  // 1
```

### `fnv1a_hash`

**Signature**: `(s: string) -> int`

FNV-1a hash of a string. Fast non-cryptographic.

```omc
fnv1a_hash("foo")  // i64 hash
```

### `gcd`

**Signature**: `(a, b) -> int`

Greatest common divisor.

```omc
gcd(12, 18)  // 6
```

### `lcm`

**Signature**: `(a, b) -> int`

Least common multiple.

```omc
lcm(4, 6)  // 12
```

### `log`

**Signature**: `(x) -> float`

Natural log.

```omc
log(2.718281)  // ~1.0
```

### `log10`

**Signature**: `(x) -> float`

Base-10 log.

```omc
log10(1000)  // 3.0
```

### `log2`

**Signature**: `(x) -> float`

Base-2 log.

```omc
log2(8)  // 3.0
```

### `max`

**Signature**: `(a, b) -> any`

Larger of two numeric values.

```omc
max(3, 7)  // 7
```

### `min`

**Signature**: `(a, b) -> any`

Smaller of two numeric values.

```omc
min(3, 7)  // 3
```

### `mod_pow`

**Signature**: `(base, exp, mod) -> int`

Modular exponentiation.

```omc
mod_pow(2, 10, 1000)  // 24
```

### `pow`

**Signature**: `(base, exp) -> float`

base^exp (float).

```omc
pow(2, 10)  // 1024.0
```

### `round`

**Signature**: `(x: float) -> int`

Round to nearest integer.

```omc
round(1.5)  // 2
```

### `sign`

**Signature**: `(n) -> int`

Returns -1, 0, or 1 by sign.

```omc
sign(-3)  // -1
```

### `sin`

**Signature**: `(x) -> float`

Sine.

```omc
sin(0)  // 0.0
```

### `sqrt`

**Signature**: `(x) -> float`

Square root.

```omc
sqrt(16)  // 4.0
```

### `tan`

**Signature**: `(x) -> float`

Tangent.

```omc
tan(0)  // 0.0
```

---

## dicts

### `dict_clear`

**Signature**: `(dict, ...) -> any`

`dict_clear`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_clear(...)  // see omc_help
```

### `dict_del`

**Signature**: `(dict, ...) -> any`

`dict_del`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_del(...)  // see omc_help
```

### `dict_get`

**Signature**: `(dict, ...) -> any`

`dict_get`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_get(...)  // see omc_help
```

### `dict_get_or`

**Signature**: `(dict, ...) -> any`

`dict_get_or`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_get_or(...)  // see omc_help
```

### `dict_has`

**Signature**: `(dict, ...) -> int`

`dict_has`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_has(...)  // see omc_help
```

### `dict_items`

**Signature**: `(dict, ...) -> any`

`dict_items`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_items(...)  // see omc_help
```

### `dict_keys`

**Signature**: `(dict, ...) -> any`

`dict_keys`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_keys(...)  // see omc_help
```

### `dict_len`

**Signature**: `(dict, ...) -> int`

`dict_len`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_len(...)  // see omc_help
```

### `dict_merge`

**Signature**: `(dict, ...) -> any`

`dict_merge`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_merge(...)  // see omc_help
```

### `dict_new`

**Signature**: `(dict, ...) -> any`

`dict_new`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_new(...)  // see omc_help
```

### `dict_pop`

**Signature**: `(dict, ...) -> any`

`dict_pop`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_pop(...)  // see omc_help
```

### `dict_set`

**Signature**: `(dict, ...) -> any`

`dict_set`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_set(...)  // see omc_help
```

### `dict_size`

**Signature**: `(dict, ...) -> int`

`dict_size`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_size(...)  // see omc_help
```

### `dict_values`

**Signature**: `(dict, ...) -> any`

`dict_values`: see omc_explain or source for details. Auto-generated stub.

```omc
dict_values(...)  // see omc_help
```

### `dict_clear`

**Signature**: `(d) -> null`

Remove all entries.

```omc
dict_clear(d);
```

### `dict_del`

**Signature**: `(d, key) -> null`

Remove a key.

```omc
dict_del(d, "k");
```

### `dict_get_or`

**Signature**: `(d, key, default) -> any`

Get value or default if missing.

```omc
dict_get_or(d, "k", 0)
```

### `dict_has`

**Signature**: `(d, key) -> int`

1 if key present.

```omc
dict_has(d, "k")  // 1
```

### `dict_items`

**Signature**: `(d) -> [key, value][]`

Array of [key, value] pairs.

```omc
dict_items(d)
```

### `dict_keys`

**Signature**: `(d) -> string[]`

All keys.

```omc
dict_keys(d)
```

### `dict_len`

**Signature**: `(d) -> int`

Number of entries.

```omc
dict_len(d)
```

### `dict_merge`

**Signature**: `(a, b) -> dict`

Merge b into copy of a.

```omc
dict_merge(d1, d2)
```

### `dict_new`

**Signature**: `() -> dict`

Empty mutable dict.

```omc
h d = dict_new();
```

### `dict_pop`

**Signature**: `(d, key) -> any`

Remove and return value at key.

```omc
dict_pop(d, "k")
```

### `dict_size`

**Signature**: `(d) -> int`

Same as dict_len.

```omc
dict_size(d)
```

### `dict_values`

**Signature**: `(d) -> any[]`

All values.

```omc
dict_values(d)
```

---

## test_runner

### `test_failure_count`

**Signature**: `(...) -> any`

`test_failure_count`: see omc_explain or source for details. Auto-generated stub.

```omc
test_failure_count(...)  // see omc_help
```

### `test_get_failures`

**Signature**: `(...) -> any`

`test_get_failures`: see omc_explain or source for details. Auto-generated stub.

```omc
test_get_failures(...)  // see omc_help
```

### `test_record_failure`

**Signature**: `(...) -> any`

`test_record_failure`: see omc_explain or source for details. Auto-generated stub.

```omc
test_record_failure(...)  // see omc_help
```

### `test_set_current`

**Signature**: `(...) -> any`

`test_set_current`: see omc_explain or source for details. Auto-generated stub.

```omc
test_set_current(...)  // see omc_help
```

### `test_failure_count`

**Signature**: `() -> int`

Number of failures recorded.

```omc
test_failure_count()  // 0 if all pass
```

### `test_get_failures`

**Signature**: `() -> string[]`

All recorded failure messages.

```omc
test_get_failures()
```

### `test_record_failure`

**Signature**: `(msg: string) -> null`

Record a test failure with a message.

```omc
test_record_failure("fail");
```

### `test_set_current`

**Signature**: `(name: string) -> null`

Set the current test name for failure prefixing.

```omc
test_set_current("my_test");
```

---

