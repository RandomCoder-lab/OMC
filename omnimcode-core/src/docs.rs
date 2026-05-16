//! Builtin metadata registry.
//!
//! Every notable builtin gets an entry here so:
//!   - The `omc_help(name)` and `omc_list_builtins(category)` builtins
//!     can introspect the runtime from inside OMC code.
//!   - `omc --gen-docs` emits a stable Markdown reference.
//!   - Error paths can compute `did_you_mean` suggestions over the
//!     full known surface area.
//!
//! Adding a builtin to BUILTINS is the only thing required — the
//! introspection / docgen / suggester all read from this slice.
//!
//! Convention: `unique_to_omc: true` flags features that have no
//! direct Python/NumPy equivalent. These are the things an LLM
//! reaching for OMC over Python would actually want.

#[derive(Clone, Debug)]
pub struct BuiltinDoc {
    /// OMC-side name as called from user code.
    pub name: &'static str,
    /// Bucket for grouping (`arrays`, `substrate`, `autograd`, ...).
    pub category: &'static str,
    /// Pseudo-typed signature, written for human + LLM readers.
    /// Examples: `(arr: int[]) -> int`, `(a, b) -> array`.
    pub signature: &'static str,
    /// One-line description. Lead with the verb.
    pub description: &'static str,
    /// One worked example showing input/output or the typical pattern.
    pub example: &'static str,
    /// True when no clean Python equivalent exists — these are the
    /// reasons to pick OMC over numpy/jax.
    pub unique_to_omc: bool,
}

pub const BUILTINS: &[BuiltinDoc] = &[
    // ---- Core / IO ----
    BuiltinDoc {
        name: "print", category: "core",
        signature: "(value) -> null",
        description: "Print value to stdout with newline.",
        example: r#"print("hello");"#,
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "to_string", category: "core",
        signature: "(value) -> string",
        description: "Coerce any value to its display string.",
        example: "to_string(42)  // \"42\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "type_of", category: "core",
        signature: "(value) -> string",
        description: "Runtime type tag: int, float, string, bool, array, dict, function, null_t.",
        example: "type_of([1,2,3])  // \"array\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "len", category: "core",
        signature: "(string|array) -> int",
        description: "Length in bytes (string) or elements (array).",
        example: "len([1,2,3])  // 3",
        unique_to_omc: false,
    },

    // ---- 1D arrays ----
    BuiltinDoc {
        name: "arr_new", category: "arrays",
        signature: "() -> array",
        description: "Create an empty mutable array.",
        example: "arr_new()  // []",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_push", category: "arrays",
        signature: "(arr, value) -> array",
        description: "Append value to array in place.",
        example: "arr_push(xs, 42);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_get", category: "arrays",
        signature: "(arr, index) -> any",
        description: "Read element at index (0-based).",
        example: "arr_get([10,20,30], 1)  // 20",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_set", category: "arrays",
        signature: "(arr, index, value) -> null",
        description: "Write element at index in place.",
        example: "arr_set(xs, 0, 99);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_len", category: "arrays",
        signature: "(arr) -> int",
        description: "Length of array.",
        example: "arr_len([1,2,3])  // 3",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_concat", category: "arrays",
        signature: "(a, b) -> array",
        description: "Concatenate two arrays into a new one.",
        example: "arr_concat([1,2], [3,4])  // [1,2,3,4]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_slice", category: "arrays",
        signature: "(arr, start, end) -> array",
        description: "Half-open slice [start..end).",
        example: "arr_slice([0,1,2,3,4], 1, 4)  // [1,2,3]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_map", category: "arrays",
        signature: "(arr, fn) -> array",
        description: "Apply function to each element, returning new array.",
        example: "arr_map([1,2,3], fn(x) { return x*x; })  // [1,4,9]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_filter", category: "arrays",
        signature: "(arr, fn) -> array",
        description: "Keep elements where predicate returns truthy.",
        example: "arr_filter([1,2,3,4], fn(x) { return x % 2 == 0; })  // [2,4]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_sort", category: "arrays",
        signature: "(arr) -> array",
        description: "Ascending sort by numeric value.",
        example: "arr_sort([3,1,2])  // [1,2,3]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_reverse", category: "arrays",
        signature: "(arr) -> array",
        description: "Reverse a copy of the array.",
        example: "arr_reverse([1,2,3])  // [3,2,1]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_sum_int", category: "arrays",
        signature: "(arr) -> int",
        description: "Sum of integer elements.",
        example: "arr_sum_int([1,2,3,4])  // 10",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_mean", category: "arrays",
        signature: "(arr) -> float",
        description: "Arithmetic mean.",
        example: "arr_mean([1.0,2.0,3.0])  // 2.0",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_variance", category: "arrays",
        signature: "(arr) -> float",
        description: "Sample variance.",
        example: "arr_variance([1.0,2.0,3.0,4.0,5.0])  // 2.5",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_stddev", category: "arrays",
        signature: "(arr) -> float",
        description: "Standard deviation.",
        example: "arr_stddev([1.0,2.0,3.0,4.0,5.0])  // ~1.58",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_dot", category: "arrays",
        signature: "(a, b) -> float",
        description: "Dot product of two 1D arrays.",
        example: "arr_dot([1.0,2.0], [3.0,4.0])  // 11.0",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_min_int", category: "arrays",
        signature: "(arr) -> int",
        description: "Minimum element (int).",
        example: "arr_min_int([3,1,4,1,5])  // 1",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_max_int", category: "arrays",
        signature: "(arr) -> int",
        description: "Maximum element (int).",
        example: "arr_max_int([3,1,4,1,5])  // 5",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_argmax", category: "arrays",
        signature: "(arr) -> int",
        description: "Index of largest element.",
        example: "arr_argmax([3,1,4,1,5])  // 4",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_argmin", category: "arrays",
        signature: "(arr) -> int",
        description: "Index of smallest element.",
        example: "arr_argmin([3,1,4,1,5])  // 1",
        unique_to_omc: false,
    },

    // ---- Elementwise / broadcasting (2D-aware) ----
    BuiltinDoc {
        name: "arr_add", category: "arrays",
        signature: "(a, b) -> array",
        description: "Elementwise add. Broadcasts scalar↔array and 2D↔1D row-vector.",
        example: "arr_add([1,2,3], 10)  // [11,12,13]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_sub", category: "arrays",
        signature: "(a, b) -> array",
        description: "Elementwise subtract, with broadcasting.",
        example: "arr_sub([10,20,30], [1,2,3])  // [9,18,27]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_mul", category: "arrays",
        signature: "(a, b) -> array",
        description: "Elementwise multiply, with broadcasting.",
        example: "arr_mul([1,2,3], [10,10,10])  // [10,20,30]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_div_int", category: "arrays",
        signature: "(a, b) -> array",
        description: "Elementwise integer division (div-by-0 → 0).",
        example: "arr_div_int([10,20,30], [2,5,3])  // [5,4,10]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_neg", category: "arrays",
        signature: "(arr) -> array",
        description: "Elementwise negation.",
        example: "arr_neg([1,-2,3])  // [-1,2,-3]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_scale", category: "arrays",
        signature: "(arr, scalar) -> array",
        description: "Multiply every element by a scalar.",
        example: "arr_scale([1,2,3], 10)  // [10,20,30]",
        unique_to_omc: false,
    },

    // ---- 2D arrays / linear algebra ----
    BuiltinDoc {
        name: "arr_matmul", category: "linalg",
        signature: "(A, B) -> matrix",
        description: "Matrix multiplication A@B with cache-friendly ikj loop. Integer-in/integer-out preserves substrate metadata per cell.",
        example: "arr_matmul([[1,2],[3,4]], [[5,6],[7,8]])  // [[19,22],[43,50]]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_transpose", category: "linalg",
        signature: "(M) -> matrix",
        description: "Transpose 2D matrix.",
        example: "arr_transpose([[1,2,3],[4,5,6]])  // [[1,4],[2,5],[3,6]]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_eye", category: "linalg",
        signature: "(n) -> matrix",
        description: "n×n identity matrix.",
        example: "arr_eye(3)  // [[1,0,0],[0,1,0],[0,0,1]]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_zeros_2d", category: "linalg",
        signature: "(rows, cols) -> matrix",
        description: "rows×cols zero matrix.",
        example: "arr_zeros_2d(2,3)  // [[0,0,0],[0,0,0]]",
        unique_to_omc: false,
    },

    // ---- ML kernels (native Rust) ----
    BuiltinDoc {
        name: "arr_softmax", category: "ml_kernels",
        signature: "(arr: float[]) -> float[]",
        description: "Numerically stable softmax (max-subtraction trick).",
        example: "arr_softmax([1.0,2.0,3.0])  // ~[0.09,0.24,0.67]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_layer_norm", category: "ml_kernels",
        signature: "(arr, eps=1e-5) -> float[]",
        description: "LayerNorm: (x-mean)/sqrt(var+eps).",
        example: "arr_layer_norm([1.0,2.0,3.0,4.0,5.0])  // zero-mean, unit-variance",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_relu_vec", category: "ml_kernels",
        signature: "(arr: float[]) -> float[]",
        description: "Elementwise max(x, 0).",
        example: "arr_relu_vec([-1.0,0.0,2.5])  // [0.0,0.0,2.5]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_sigmoid_vec", category: "ml_kernels",
        signature: "(arr: float[]) -> float[]",
        description: "Elementwise 1/(1+exp(-x)).",
        example: "arr_sigmoid_vec([0.0])  // [0.5]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_conv1d", category: "ml_kernels",
        signature: "(input, kernel) -> float[]",
        description: "1D valid-mode convolution.",
        example: "arr_conv1d([1,2,3,4,5], [1,1,1])  // [6,9,12]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "arr_outer", category: "ml_kernels",
        signature: "(a, b) -> matrix",
        description: "Outer product: a[i]*b[j] for every (i,j).",
        example: "arr_outer([1,2], [10,20])  // [[10,20],[20,40]]",
        unique_to_omc: false,
    },

    // ---- Substrate primitives (THE OMC-ONLY STUFF) ----
    BuiltinDoc {
        name: "is_attractor", category: "substrate",
        signature: "(n: int) -> int",
        description: "1 iff n is a Fibonacci attractor (0,1,2,3,5,8,13,...).",
        example: "is_attractor(8)  // 1 ; is_attractor(7)  // 0",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "attractor_distance", category: "substrate",
        signature: "(n: int) -> int",
        description: "Absolute distance to the nearest Fibonacci attractor.",
        example: "attractor_distance(7)  // 1 (8 is nearest)",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "arr_resonance_vec", category: "substrate",
        signature: "(arr) -> float[]",
        description: "Per-element φ-resonance (∈[0,1], 1=on Fibonacci attractor).",
        example: "arr_resonance_vec([8,13,21])  // [1.0,1.0,1.0]",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "arr_him_vec", category: "substrate",
        signature: "(arr) -> float[]",
        description: "Per-element HIM (Harmonic Interference Metric).",
        example: "arr_him_vec([1,2,3,5])  // ~[<0.5 each]",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "arr_fold_all", category: "substrate",
        signature: "(arr) -> int[]",
        description: "Snap every element to its nearest Fibonacci attractor.",
        example: "arr_fold_all([7,100,9])  // [8,89,8]",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "arr_substrate_attention", category: "substrate",
        signature: "(Q, K, V) -> matrix",
        description: "Attention scored by substrate distance (not dot product). Closer in Fibonacci-space = higher weight.",
        example: "arr_substrate_attention(Q, K, V)  // (n_q × v_cols) output",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "arr_substrate_score_rows", category: "substrate",
        signature: "(matrix) -> float[]",
        description: "Per-row mean φ-resonance. Use as a substrate-coherence regularizer.",
        example: "arr_substrate_score_rows([[1,2,3,5],[7,11,13,19]])  // [~1.0, lower]",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "crt_recover", category: "substrate",
        signature: "(remainders: int[], moduli: int[]) -> int",
        description: "Chinese Remainder Theorem recovery from per-modulus remainders.",
        example: "crt_recover([2,3,2], [5,7,3])  // 23",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "fibonacci_index", category: "substrate",
        signature: "(n: int) -> int",
        description: "Position in Fibonacci sequence (-1 if not an attractor).",
        example: "fibonacci_index(13)  // 7  ; fibonacci_index(14)  // -1",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "res", category: "substrate",
        signature: "(n: int) -> float",
        description: "φ-resonance of a single value (0..1, 1=on Fibonacci attractor).",
        example: "res(8)  // 1.0  ; res(7)  // <1.0",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "harmony", category: "substrate",
        signature: "(n: int) -> float",
        description: "HBit harmony score derived from substrate alignment.",
        example: "harmony(89)  // high (89 is Fibonacci)",
        unique_to_omc: true,
    },

    // ---- Reverse-mode autograd ----
    BuiltinDoc {
        name: "tape_reset", category: "autograd",
        signature: "() -> null",
        description: "Clear the autograd tape before starting a fresh forward pass.",
        example: "tape_reset();",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_var", category: "autograd",
        signature: "(value) -> int",
        description: "Lift a value onto the tape as a leaf variable. Returns node id.",
        example: "h x = tape_var(3.0);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_const", category: "autograd",
        signature: "(value) -> int",
        description: "Lift a value as a constant (no gradient flows through).",
        example: "h c = tape_const(2.0);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_value", category: "autograd",
        signature: "(node_id) -> any",
        description: "Read forward value at a node. Integral results come back as substrate-annotated HInt.",
        example: "tape_value(y)  // current forward value at y",
        unique_to_omc: true,
    },
    BuiltinDoc {
        name: "tape_grad", category: "autograd",
        signature: "(node_id) -> any",
        description: "Read accumulated gradient at a node after tape_backward.",
        example: "tape_grad(x)  // dL/dx",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_add", category: "autograd",
        signature: "(a_id, b_id) -> int",
        description: "Record a+b on the tape.",
        example: "h s = tape_add(x, y);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_mul", category: "autograd",
        signature: "(a_id, b_id) -> int",
        description: "Record a*b on the tape (elementwise/broadcast).",
        example: "h p = tape_mul(x, x);  // x^2",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_matmul", category: "autograd",
        signature: "(A_id, B_id) -> int",
        description: "Record A@B on the tape. Backward: dA=dy@B^T, dB=A^T@dy.",
        example: "h Y = tape_matmul(X, W);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_relu", category: "autograd",
        signature: "(a_id) -> int",
        description: "Record max(a,0). Backward: pass gradient where a>0, else 0.",
        example: "h h = tape_relu(z);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_sigmoid", category: "autograd",
        signature: "(a_id) -> int",
        description: "Record sigmoid(a). Backward: y*(1-y).",
        example: "h h = tape_sigmoid(z);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_sum", category: "autograd",
        signature: "(a_id) -> int",
        description: "Record sum-of-cells reduction. Often used as the loss.",
        example: "h L = tape_sum(Y);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_mean", category: "autograd",
        signature: "(a_id) -> int",
        description: "Record mean reduction.",
        example: "h L = tape_mean(Y);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_backward", category: "autograd",
        signature: "(loss_id) -> null",
        description: "Walk the tape in reverse; populates grads on every node.",
        example: "tape_backward(L);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "tape_update", category: "autograd",
        signature: "(var_id, lr) -> null",
        description: "In-place SGD step: value -= lr * grad.",
        example: "tape_update(W, 0.01);",
        unique_to_omc: false,
    },

    // ---- Forward-mode duals (kept for cheap single-param grads) ----
    BuiltinDoc {
        name: "dual", category: "duals",
        signature: "(value, derivative) -> [v,d]",
        description: "Lift a scalar into a forward-mode dual number.",
        example: "h x = dual(3.0, 1.0);",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "dual_mul", category: "duals",
        signature: "(a, b) -> [v,d]",
        description: "Multiply two dual numbers (scalars auto-lift to deriv=0).",
        example: "h y = dual_mul(x, x);  // y is dual carrying x^2 + 2x*dx",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "dual_d", category: "duals",
        signature: "(dual) -> float",
        description: "Read the derivative component.",
        example: "dual_d(y)  // current df/dx",
        unique_to_omc: false,
    },

    // ---- Lazy generators ----
    BuiltinDoc {
        name: "gen_stream", category: "generators",
        signature: "(thunk, callback) -> int",
        description: "Run a generator with callback per yield. O(1) memory. Returns 1 if completed, 0 if shorted.",
        example: "gen_stream(fn(){ return fib(1000000); }, fn(v){ return 1; });",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "gen_take", category: "generators",
        signature: "(thunk, n) -> array",
        description: "Pull the first n values from a lazy generator.",
        example: "gen_take(fn(){ return count(); }, 5)  // [1,2,3,4,5]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "gen_count", category: "generators",
        signature: "(thunk) -> int",
        description: "Count yields without storing them.",
        example: "gen_count(fn(){ return count_to(100); })  // 100",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "gen_sum", category: "generators",
        signature: "(thunk) -> int",
        description: "Sum integer yields without storing them.",
        example: "gen_sum(fn(){ return count_to(1000); })  // 500500",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "gen_substrate_fib", category: "generators",
        signature: "(callback, max) -> int",
        description: "Native lazy Fibonacci stream up to max. Each value is on-attractor.",
        example: "gen_substrate_fib(fn(v){ print(v); return 1; }, 100);",
        unique_to_omc: true,
    },

    // ---- Strings ----
    BuiltinDoc {
        name: "str_len", category: "strings",
        signature: "(s: string) -> int",
        description: "Byte length of string (NOT char count for non-ASCII).",
        example: "str_len(\"hello\")  // 5",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "str_split", category: "strings",
        signature: "(s, sep) -> string[]",
        description: "Split on separator.",
        example: "str_split(\"a,b,c\", \",\")  // [\"a\",\"b\",\"c\"]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "str_join", category: "strings",
        signature: "(arr, sep) -> string",
        description: "Join string array with separator.",
        example: "str_join([\"a\",\"b\"], \"-\")  // \"a-b\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "str_slice", category: "strings",
        signature: "(s, start, end) -> string",
        description: "Character-indexed substring [start..end).",
        example: "str_slice(\"abcdef\", 1, 4)  // \"bcd\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "concat_many", category: "strings",
        signature: "(...) -> string",
        description: "Concatenate any number of values as strings.",
        example: "concat_many(\"x=\", 42, \" y=\", 99)  // \"x=42 y=99\"",
        unique_to_omc: false,
    },

    // ---- Regex ----
    BuiltinDoc {
        name: "re_match", category: "regex",
        signature: "(pattern, s) -> int",
        description: "1 if pattern matches anywhere in s, 0 otherwise.",
        example: "re_match(\"^\\\\d+$\", \"123\")  // 1",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "re_find_all", category: "regex",
        signature: "(pattern, s) -> string[]",
        description: "All non-overlapping matches.",
        example: "re_find_all(\"\\\\d+\", \"a12 b34\")  // [\"12\",\"34\"]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "re_replace", category: "regex",
        signature: "(pattern, s, replacement) -> string",
        description: "Replace all matches.",
        example: "re_replace(\"\\\\d+\", \"a1b2\", \"X\")  // \"aXbX\"",
        unique_to_omc: false,
    },

    // ---- JSON ----
    BuiltinDoc {
        name: "json_parse", category: "json",
        signature: "(s: string) -> any",
        description: "Parse JSON into OMC value (object→dict, array→array).",
        example: "json_parse(\"{\\\"x\\\":1}\")  // dict",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "json_stringify", category: "json",
        signature: "(value) -> string",
        description: "Serialize OMC value to JSON.",
        example: "json_stringify([1,2,3])  // \"[1,2,3]\"",
        unique_to_omc: false,
    },

    // ---- Stdlib expansion ----
    BuiltinDoc {
        name: "sha256", category: "stdlib",
        signature: "(s: string) -> string",
        description: "SHA-256 of input string, as 64-char hex.",
        example: "sha256(\"hello\")  // \"2cf2...\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "sha512", category: "stdlib",
        signature: "(s: string) -> string",
        description: "SHA-512 of input string, as 128-char hex.",
        example: "sha512(\"x\")  // 128 chars",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "base64_encode", category: "stdlib",
        signature: "(s: string) -> string",
        description: "Standard base64 encoding.",
        example: "base64_encode(\"hi\")  // \"aGk=\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "base64_decode", category: "stdlib",
        signature: "(s: string) -> string",
        description: "Decode standard base64.",
        example: "base64_decode(\"aGk=\")  // \"hi\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "now_unix", category: "stdlib",
        signature: "() -> int",
        description: "Current Unix timestamp in seconds.",
        example: "now_unix()  // 1747400000",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "now_iso", category: "stdlib",
        signature: "() -> string",
        description: "Current ISO-8601 UTC datetime string.",
        example: "now_iso()  // \"2026-05-16T12:34:56Z\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "format_time", category: "stdlib",
        signature: "(unix_ts, fmt) -> string",
        description: "Format a unix timestamp via strftime-style fmt.",
        example: "format_time(0, \"%Y-%m-%d\")  // \"1970-01-01\"",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "parse_time", category: "stdlib",
        signature: "(s, fmt) -> int",
        description: "Parse string via strftime fmt into unix timestamp.",
        example: "parse_time(\"2026-05-16\", \"%Y-%m-%d\")  // 1747353600",
        unique_to_omc: false,
    },

    // ---- Exception handling ----
    BuiltinDoc {
        name: "is_instance", category: "exceptions",
        signature: "(value, class_name: string) -> int",
        description: "1 if value is a class instance whose __class__ matches OR inherits from class_name.",
        example: "is_instance(HttpError(...), \"AppError\")  // 1 if HttpError extends AppError",
        unique_to_omc: false,
    },

    // ---- Introspection (THIS module's surface) ----
    BuiltinDoc {
        name: "omc_help", category: "introspection",
        signature: "(name: string) -> dict",
        description: "Look up metadata for a builtin: signature, description, example.",
        example: "omc_help(\"arr_softmax\")  // {name, signature, description, example, ...}",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_list_builtins", category: "introspection",
        signature: "(category?: string) -> string[]",
        description: "List all documented builtins. Pass category to filter.",
        example: "omc_list_builtins(\"substrate\")  // [is_attractor, attractor_distance, ...]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_categories", category: "introspection",
        signature: "() -> string[]",
        description: "List all builtin categories.",
        example: "omc_categories()  // [core, arrays, linalg, ml_kernels, substrate, ...]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_did_you_mean", category: "introspection",
        signature: "(name: string) -> string[]",
        description: "Closest known builtin names for `name` (edit distance ≤ 3).",
        example: "omc_did_you_mean(\"arr_softmx\")  // [\"arr_softmax\"]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_unique_builtins", category: "introspection",
        signature: "() -> string[]",
        description: "Builtins flagged as unique to OMC (no clean Python equivalent).",
        example: "omc_unique_builtins()  // [is_attractor, arr_substrate_attention, ...]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_explain_error", category: "introspection",
        signature: "(msg: string) -> dict",
        description: "Pattern-match an error message against the curated catalog. Returns {matched, pattern, category, explanation, typical_cause, fix}.",
        example: "try { arr_softmx([1.0]); } catch e { print(dict_get(omc_explain_error(e), \"fix\")); }",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_error_categories", category: "introspection",
        signature: "() -> string[]",
        description: "All distinct error categories in the catalog.",
        example: "omc_error_categories()  // [dispatch, arrays, linalg, ...]",
        unique_to_omc: false,
    },
    BuiltinDoc {
        name: "omc_error_count", category: "introspection",
        signature: "() -> int",
        description: "Number of curated error patterns. The knowledge base size.",
        example: "omc_error_count()  // 42+",
        unique_to_omc: false,
    },
];

/// Look up a builtin by name. Returns None when there's no docs entry
/// (which doesn't necessarily mean the builtin doesn't exist — just
/// that it's not yet in the registry).
pub fn lookup(name: &str) -> Option<&'static BuiltinDoc> {
    BUILTINS.iter().find(|b| b.name == name)
}

/// All distinct category names, in stable order.
pub fn categories() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for b in BUILTINS {
        if !out.contains(&b.category) {
            out.push(b.category);
        }
    }
    out
}

/// All names matching the given category, or all names when None.
pub fn names_in(category: Option<&str>) -> Vec<&'static str> {
    BUILTINS.iter()
        .filter(|b| category.map_or(true, |c| b.category == c))
        .map(|b| b.name)
        .collect()
}

/// Edit distance (Levenshtein) — used by did_you_mean. Small enough
/// that a manual implementation beats pulling another dep.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let n = a.chars().count();
    let m = b.chars().count();
    if n == 0 { return m; }
    if m == 0 { return n; }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Up to `limit` closest known names, sorted by ascending edit distance.
/// Bounded to distance ≤ 3 so we don't return wild suggestions.
pub fn did_you_mean(query: &str, limit: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = BUILTINS.iter()
        .map(|b| (edit_distance(query, b.name), b.name))
        .filter(|(d, _)| *d <= 3)
        .collect();
    scored.sort_by_key(|(d, n)| (*d, *n));
    scored.into_iter().take(limit).map(|(_, n)| n).collect()
}

/// Render a single builtin as a Markdown section. Used by docgen and
/// also by omc_help for human-readable output.
pub fn render_markdown(doc: &BuiltinDoc) -> String {
    let unique = if doc.unique_to_omc { " 🔱 *OMC-unique*" } else { "" };
    format!(
        "### `{}`{}\n\n**Signature**: `{}`\n\n{}\n\n```omc\n{}\n```\n",
        doc.name, unique, doc.signature, doc.description, doc.example
    )
}

/// Render the full reference as one Markdown doc.
pub fn render_full_reference() -> String {
    let mut out = String::new();
    out.push_str("# OMC Builtin Reference\n\n");
    out.push_str("Auto-generated from `omnimcode-core/src/docs.rs`. ");
    out.push_str("Run `omc --gen-docs > OMC_REFERENCE.md` to regenerate.\n\n");
    out.push_str(&format!("**Total documented builtins**: {}\n\n", BUILTINS.len()));
    let unique_count = BUILTINS.iter().filter(|b| b.unique_to_omc).count();
    out.push_str(&format!(
        "**OMC-unique**: {} (no direct Python/NumPy equivalent — these are why you reach for OMC over numpy)\n\n",
        unique_count
    ));
    out.push_str("---\n\n");
    out.push_str("## Categories\n\n");
    for cat in categories() {
        let n = BUILTINS.iter().filter(|b| b.category == cat).count();
        out.push_str(&format!("- [{}](#{}) ({} builtins)\n", cat, cat, n));
    }
    out.push_str("\n---\n\n");
    for cat in categories() {
        out.push_str(&format!("## {}\n\n", cat));
        for doc in BUILTINS.iter().filter(|b| b.category == cat) {
            out.push_str(&render_markdown(doc));
            out.push('\n');
        }
        out.push_str("---\n\n");
    }
    out
}
