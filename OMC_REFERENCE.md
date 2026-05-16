# OMC Builtin Reference

Auto-generated from `omnimcode-core/src/docs.rs`. Run `omc --gen-docs > OMC_REFERENCE.md` to regenerate.

**Total documented builtins**: 97

**OMC-unique**: 13 (no direct Python/NumPy equivalent — these are why you reach for OMC over numpy)

---

## Categories

- [core](#core) (4 builtins)
- [arrays](#arrays) (26 builtins)
- [linalg](#linalg) (4 builtins)
- [ml_kernels](#ml_kernels) (6 builtins)
- [substrate](#substrate) (11 builtins)
- [autograd](#autograd) (14 builtins)
- [duals](#duals) (3 builtins)
- [generators](#generators) (5 builtins)
- [strings](#strings) (5 builtins)
- [regex](#regex) (3 builtins)
- [json](#json) (2 builtins)
- [stdlib](#stdlib) (8 builtins)
- [exceptions](#exceptions) (1 builtins)
- [introspection](#introspection) (5 builtins)

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

### `resonance` 🔱 *OMC-unique*

**Signature**: `(n: int) -> float`

φ-resonance of a single value.

```omc
resonance(8)  // 1.0  ; resonance(7)  // <1.0
```

### `harmony` 🔱 *OMC-unique*

**Signature**: `(n: int) -> float`

HBit harmony score derived from substrate alignment.

```omc
harmony(89)  // high (89 is Fibonacci)
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

---

## exceptions

### `is_instance`

**Signature**: `(value, class_name: string) -> int`

1 if value is a class instance whose __class__ matches OR inherits from class_name.

```omc
is_instance(HttpError(...), "AppError")  // 1 if HttpError extends AppError
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

---

