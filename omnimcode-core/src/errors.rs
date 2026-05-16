//! Error-message knowledge base.
//!
//! Every common runtime/parser error gets an entry here with an
//! explanation, the typical cause, and a corrected example. The
//! runtime exposes this via `omc_explain_error(msg)`: an LLM catching
//! an OMC error can call that to get a structured explanation back,
//! often with a one-line fix.
//!
//! The patterns are matched substring-style (case-sensitive) against
//! the error message — order matters when multiple patterns could
//! apply. More specific patterns appear before more general ones.
//!
//! Add entries liberally: every "wait, what does this error mean?"
//! moment for a real user is a missing entry here.

#[derive(Clone, Debug)]
pub struct ErrorPattern {
    /// Substring matched against the error message.
    pub pattern: &'static str,
    /// Bucket for grouping in docs.
    pub category: &'static str,
    /// What the error means.
    pub explanation: &'static str,
    /// What the user typically did wrong.
    pub typical_cause: &'static str,
    /// One-line fix or corrected form (idiomatic OMC).
    pub fix: &'static str,
}

pub const ERROR_PATTERNS: &[ErrorPattern] = &[
    // ---- Function dispatch ----
    ErrorPattern {
        pattern: "Undefined function:",
        category: "dispatch",
        explanation: "The interpreter could not find a function or builtin with that name.",
        typical_cause: "Typo, or a Python/NumPy name (e.g. `numpy.dot`) used instead of the OMC equivalent (`arr_dot`).",
        fix: "Use `omc_did_you_mean(\"name\")` or `omc_list_builtins()` to find the correct name.",
    },
    ErrorPattern {
        pattern: "expects 3 arguments, got",
        category: "dispatch",
        explanation: "The function was called with the wrong number of arguments.",
        typical_cause: "Forgot an argument, or passed extras that the function doesn't accept.",
        fix: "Call `omc_help(\"<name>\")` and check the `signature` field.",
    },
    ErrorPattern {
        pattern: "expects 2 arguments, got",
        category: "dispatch",
        explanation: "Wrong number of arguments to a 2-arity function.",
        typical_cause: "Forgot the second argument or passed an extra one.",
        fix: "Check `omc_help(\"<name>\")` for the expected signature.",
    },
    ErrorPattern {
        pattern: "expects 1 arguments",
        category: "dispatch",
        explanation: "Wrong number of arguments to a 1-arity function.",
        typical_cause: "Passed extra arguments to a single-arg function.",
        fix: "Pass exactly one argument or check `omc_help(\"<name>\")`.",
    },
    ErrorPattern {
        pattern: "not a callable",
        category: "dispatch",
        explanation: "Tried to call a value that isn't a function or lambda.",
        typical_cause: "Passed a string/int where a `fn(...)` lambda was expected.",
        fix: "Pass a `fn(...) { ... }` literal, not the function's name as a string.",
    },

    // ---- Arrays ----
    ErrorPattern {
        pattern: "arr_get: index",
        category: "arrays",
        explanation: "Array index out of bounds.",
        typical_cause: "Off-by-one loop, or computing the index from data that exceeds the array length.",
        fix: "Guard with `if i < arr_len(xs) { ... }` before reading.",
    },
    ErrorPattern {
        pattern: "arr_get: first argument must be an array",
        category: "arrays",
        explanation: "Tried to index something that isn't an array (often a scalar or dict).",
        typical_cause: "Accidentally calling `arr_get` on the result of a builtin that returns a dict.",
        fix: "Use `dict_get` for dicts; check `type_of(value)` to confirm it's an array.",
    },
    ErrorPattern {
        pattern: "arr_set: index",
        category: "arrays",
        explanation: "Array index out of bounds on write.",
        typical_cause: "Writing past the end without first growing the array.",
        fix: "Use `arr_push(xs, v)` to append; `arr_set` only updates existing cells.",
    },
    ErrorPattern {
        pattern: "arr_set: first argument must be an array variable",
        category: "arrays",
        explanation: "arr_set's first argument must be a named variable, not an expression.",
        typical_cause: "Calling `arr_set(arr_get(xs, 0), 1, 99)` — the inner expression has no name.",
        fix: "Bind the inner array to a variable first: `h inner = arr_get(xs, 0); arr_set(inner, 1, 99);`.",
    },
    ErrorPattern {
        pattern: "length mismatch",
        category: "arrays",
        explanation: "Two arrays of incompatible length passed to an elementwise op.",
        typical_cause: "arr_add/sub/mul of arrays that aren't the same length and aren't 2D-broadcastable.",
        fix: "Check `arr_len(a)` and `arr_len(b)` match, or use scalar broadcasting.",
    },
    ErrorPattern {
        pattern: "ragged 2D array",
        category: "arrays",
        explanation: "A 2D array has rows of different lengths.",
        typical_cause: "Manually built a matrix with uneven row widths.",
        fix: "Ensure every inner array has the same length, or use `arr_zeros_2d(rows, cols)` to start fresh.",
    },
    ErrorPattern {
        pattern: "shape mismatch",
        category: "linalg",
        explanation: "Matrix dimensions don't match for matmul or elementwise op.",
        typical_cause: "Tried to compute A@B where A is (m,n) and B is (p,q) with n != p.",
        fix: "For A@B: A.cols must equal B.rows. Use `arr_transpose` to fix orientation.",
    },
    ErrorPattern {
        pattern: "row-broadcast length mismatch",
        category: "arrays",
        explanation: "Broadcast vector length doesn't match the matrix column count.",
        typical_cause: "Adding a 1D bias of length M to a matrix with N != M columns.",
        fix: "Make the bias vector length equal to the matrix's column count.",
    },
    ErrorPattern {
        pattern: "empty matrix",
        category: "linalg",
        explanation: "Matrix operation called on a matrix with zero rows.",
        typical_cause: "Forgot to populate the matrix, or filtered all rows out.",
        fix: "Check `arr_len(matrix) > 0` before passing to matmul/transpose.",
    },

    // ---- Dicts ----
    ErrorPattern {
        pattern: "dict_get: first argument must be a dict",
        category: "dicts",
        explanation: "Tried to look up a key on a value that isn't a dict.",
        typical_cause: "Confusing arrays and dicts — `arr_get(d, 0)` vs `dict_get(d, \"0\")`.",
        fix: "Check `type_of(value)`. Use `arr_get` for arrays, `dict_get` for dicts.",
    },
    ErrorPattern {
        pattern: "dict_set requires",
        category: "dicts",
        explanation: "dict_set wasn't given (dict, key, value).",
        typical_cause: "Missing argument or wrong order.",
        fix: "Call as `dict_set(d, \"key\", value);`.",
    },

    // ---- Type coercion ----
    ErrorPattern {
        pattern: "cannot lift",
        category: "types",
        explanation: "A type-conversion at a builtin boundary couldn't accept this value.",
        typical_cause: "Passed a function/closure/circuit where a number/array was expected.",
        fix: "Check `type_of(value)` and convert if needed.",
    },

    // ---- Substrate ----
    ErrorPattern {
        pattern: "is_attractor requires",
        category: "substrate",
        explanation: "is_attractor needs a single integer argument.",
        typical_cause: "Called with no args or an array.",
        fix: "Pass one integer: `is_attractor(8)` → 1.",
    },
    ErrorPattern {
        pattern: "attractor_distance requires",
        category: "substrate",
        explanation: "attractor_distance needs a single integer argument.",
        typical_cause: "Wrong arg count.",
        fix: "Pass one integer: `attractor_distance(7)` → 1 (8 is nearest Fibonacci).",
    },
    ErrorPattern {
        pattern: "arr_resonance_vec",
        category: "substrate",
        explanation: "arr_resonance_vec computes per-element φ-resonance — needs a 1D array.",
        typical_cause: "Passed a 2D matrix or a scalar.",
        fix: "Pass a 1D integer array. For a row of a matrix, do `arr_resonance_vec(arr_get(M, 0))`.",
    },
    ErrorPattern {
        pattern: "arr_substrate_attention",
        category: "substrate",
        explanation: "Substrate-aware attention needs three matrices: Q, K, V (sequence × dim).",
        typical_cause: "Wrong arg count, or passed 1D arrays.",
        fix: "Each input must be 2D: `arr_substrate_attention([[1,2]], [[1,2],[3,5]], [[10,20],[30,40]])`.",
    },

    // ---- Autograd ----
    ErrorPattern {
        pattern: "tape_value: id",
        category: "autograd",
        explanation: "Tried to read from a tape node that doesn't exist.",
        typical_cause: "Used a node id from a previous `tape_reset()`, or a stale variable.",
        fix: "Re-record after `tape_reset()` and use freshly returned ids.",
    },
    ErrorPattern {
        pattern: "tape_grad: id",
        category: "autograd",
        explanation: "Tried to read gradient at a tape node that doesn't exist.",
        typical_cause: "Node id became stale after tape_reset(), or you passed a non-int.",
        fix: "Hold node ids in variables and only read them in the same tape session.",
    },
    ErrorPattern {
        pattern: "tape_backward: id",
        category: "autograd",
        explanation: "Loss node id is out of tape range.",
        typical_cause: "Called tape_backward(loss) where loss is a stale id.",
        fix: "Build the loss with tape_* ops and pass the returned id immediately.",
    },
    ErrorPattern {
        pattern: "tape_matmul",
        category: "autograd",
        explanation: "Matrix multiply on the tape requires two 2D tape values.",
        typical_cause: "Passed scalar tape vars to tape_matmul.",
        fix: "Build with `tape_var([[1,2,3]])` (2D array literals).",
    },

    // ---- Duals (forward mode) ----
    ErrorPattern {
        pattern: "dual_mul requires",
        category: "duals",
        explanation: "Dual-number multiply needs two args (each scalar or dual).",
        typical_cause: "Wrong arg count.",
        fix: "Lift inputs first: `dual(3.0, 1.0)` then `dual_mul(x, x)`.",
    },
    ErrorPattern {
        pattern: "dual_d:",
        category: "duals",
        explanation: "Tried to read derivative from a malformed dual.",
        typical_cause: "Passed something that isn't a [value, derivative] 2-tuple.",
        fix: "Construct duals with `dual(v, d)` so the shape is correct.",
    },

    // ---- Lazy generators ----
    ErrorPattern {
        pattern: "gen_stream requires",
        category: "generators",
        explanation: "gen_stream needs (thunk, callback) — both are functions.",
        typical_cause: "Passed a direct generator call instead of a thunk.",
        fix: "Wrap in `fn() { return fib(N); }` so the generator doesn't start eagerly.",
    },
    ErrorPattern {
        pattern: "gen_take requires",
        category: "generators",
        explanation: "gen_take needs (thunk, n).",
        typical_cause: "Missing the n argument.",
        fix: "`gen_take(fn() { return count(); }, 5)`.",
    },
    ErrorPattern {
        pattern: "gen_substrate_fib requires",
        category: "generators",
        explanation: "Substrate Fibonacci stream needs (callback, max).",
        typical_cause: "Wrong arg count.",
        fix: "`gen_substrate_fib(fn(v) { return 1; }, 100);` — streams Fibs ≤ 100.",
    },

    // ---- Strings ----
    ErrorPattern {
        pattern: "str_split requires",
        category: "strings",
        explanation: "str_split needs (string, separator).",
        typical_cause: "Forgot the separator argument.",
        fix: "`str_split(\"a,b,c\", \",\")` → `[\"a\",\"b\",\"c\"]`.",
    },

    // ---- Regex ----
    ErrorPattern {
        pattern: "regex compile error",
        category: "regex",
        explanation: "The regex pattern is malformed.",
        typical_cause: "Unbalanced parens, invalid escape, unclosed character class.",
        fix: "Test the pattern in an external regex tool; escape `\\\\` in OMC strings.",
    },
    ErrorPattern {
        pattern: "re_match requires",
        category: "regex",
        explanation: "re_match needs (pattern, string).",
        typical_cause: "Wrong arg count.",
        fix: "`re_match(\"^[0-9]+$\", \"123\")` → 1.",
    },

    // ---- JSON ----
    ErrorPattern {
        pattern: "json_parse",
        category: "json",
        explanation: "JSON could not be parsed.",
        typical_cause: "Trailing comma, single quotes, unescaped string.",
        fix: "JSON is strict — use double quotes, no trailing commas.",
    },

    // ---- Stdlib ----
    ErrorPattern {
        pattern: "base64_decode",
        category: "stdlib",
        explanation: "base64 input couldn't be decoded.",
        typical_cause: "URL-safe base64 (using -_) passed to standard decoder.",
        fix: "Use only the standard alphabet (A-Z a-z 0-9 + /) with padding.",
    },
    ErrorPattern {
        pattern: "parse_time",
        category: "stdlib",
        explanation: "Time string didn't match the format spec.",
        typical_cause: "Format string doesn't match input — e.g. \"%Y-%m-%d\" vs \"05/16/2026\".",
        fix: "Make `fmt` match the input shape exactly; see strftime spec.",
    },

    // ---- Exceptions ----
    ErrorPattern {
        pattern: "is_instance requires",
        category: "exceptions",
        explanation: "is_instance needs (value, class_name_string).",
        typical_cause: "Forgot the class name.",
        fix: "`is_instance(err, \"AppError\")`.",
    },

    // ---- Generic / parser ----
    ErrorPattern {
        pattern: "Expected Semicolon",
        category: "parser",
        explanation: "OMC expected a `;` at the end of a statement.",
        typical_cause: "Statement on its own line without a trailing semicolon, or a class field declared without `;`.",
        fix: "Add `;` at the end. Class fields look like `fieldname;` not `fieldname` or `fieldname,`.",
    },
    ErrorPattern {
        pattern: "Expected identifier",
        category: "parser",
        explanation: "Parser expected a name where it found a keyword or symbol.",
        typical_cause: "Using a reserved word (`h`, `fn`, `if`, ...) as a variable name.",
        fix: "Rename the variable. `h` is reserved for harmonic-var declarations.",
    },
    ErrorPattern {
        pattern: "Expected ",
        category: "parser",
        explanation: "Parser expected a specific token and got something else.",
        typical_cause: "Mismatched braces, missing semicolons, or a typo where syntax meets expression.",
        fix: "Check the line/column. Common: missing `;` ends the previous statement and shifts everything after.",
    },
    ErrorPattern {
        pattern: "division by zero",
        category: "math",
        explanation: "Division (or mod) by zero.",
        typical_cause: "Dividing by a computed value that turned out to be 0.",
        fix: "Guard with `if denom != 0` before dividing.",
    },
    ErrorPattern {
        pattern: "stack overflow",
        category: "runtime",
        explanation: "Recursion depth exceeded.",
        typical_cause: "Recursive function without a base case, or a base case that isn't reachable.",
        fix: "Check the recursion's base case; convert to iteration if deeply nested.",
    },
    // ---- Auto-generated arity patterns (all `X requires (...)` errors) ----
    // 217 generated entries, one per builtin that asserts arity. The
    // hand-written entries above take precedence (matched first) for the
    // builtins with deeper guidance.
        ErrorPattern { pattern: "arr_add requires (", category: "arrays", explanation: "`arr_add` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_add\")` for the expected signature." },
    ErrorPattern { pattern: "arr_all requires (", category: "arrays", explanation: "`arr_all` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_all\")` for the expected signature." },
    ErrorPattern { pattern: "arr_any requires (", category: "arrays", explanation: "`arr_any` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_any\")` for the expected signature." },
    ErrorPattern { pattern: "arr_avg_distance requires (", category: "arrays", explanation: "`arr_avg_distance` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_avg_distance\")` for the expected signature." },
    ErrorPattern { pattern: "arr_chunk requires (", category: "arrays", explanation: "`arr_chunk` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_chunk\")` for the expected signature." },
    ErrorPattern { pattern: "arr_concat requires (", category: "arrays", explanation: "`arr_concat` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_concat\")` for the expected signature." },
    ErrorPattern { pattern: "arr_contains requires (", category: "arrays", explanation: "`arr_contains` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_contains\")` for the expected signature." },
    ErrorPattern { pattern: "arr_count requires (", category: "arrays", explanation: "`arr_count` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_count\")` for the expected signature." },
    ErrorPattern { pattern: "arr_div_int requires (", category: "arrays", explanation: "`arr_div_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_div_int\")` for the expected signature." },
    ErrorPattern { pattern: "arr_dot requires (", category: "arrays", explanation: "`arr_dot` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_dot\")` for the expected signature." },
    ErrorPattern { pattern: "arr_drop requires (", category: "arrays", explanation: "`arr_drop` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_drop\")` for the expected signature." },
    ErrorPattern { pattern: "arr_enumerate requires (", category: "arrays", explanation: "`arr_enumerate` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_enumerate\")` for the expected signature." },
    ErrorPattern { pattern: "arr_eye requires (", category: "arrays", explanation: "`arr_eye` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_eye\")` for the expected signature." },
    ErrorPattern { pattern: "arr_filter requires (", category: "arrays", explanation: "`arr_filter` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_filter\")` for the expected signature." },
    ErrorPattern { pattern: "arr_find requires (", category: "arrays", explanation: "`arr_find` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_find\")` for the expected signature." },
    ErrorPattern { pattern: "arr_flatten requires (", category: "arrays", explanation: "`arr_flatten` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_flatten\")` for the expected signature." },
    ErrorPattern { pattern: "arr_fold_all requires (", category: "arrays", explanation: "`arr_fold_all` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_fold_all\")` for the expected signature." },
    ErrorPattern { pattern: "arr_fold_elements requires (", category: "arrays", explanation: "`arr_fold_elements` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_fold_elements\")` for the expected signature." },
    ErrorPattern { pattern: "arr_get requires (", category: "arrays", explanation: "`arr_get` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_get\")` for the expected signature." },
    ErrorPattern { pattern: "arr_him_vec requires (", category: "arrays", explanation: "`arr_him_vec` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_him_vec\")` for the expected signature." },
    ErrorPattern { pattern: "arr_index_of requires (", category: "arrays", explanation: "`arr_index_of` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_index_of\")` for the expected signature." },
    ErrorPattern { pattern: "arr_is_sorted requires (", category: "arrays", explanation: "`arr_is_sorted` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_is_sorted\")` for the expected signature." },
    ErrorPattern { pattern: "arr_join requires (", category: "arrays", explanation: "`arr_join` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_join\")` for the expected signature." },
    ErrorPattern { pattern: "arr_layer_norm requires (", category: "arrays", explanation: "`arr_layer_norm` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_layer_norm\")` for the expected signature." },
    ErrorPattern { pattern: "arr_map requires (", category: "arrays", explanation: "`arr_map` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_map\")` for the expected signature." },
    ErrorPattern { pattern: "arr_matmul requires (", category: "arrays", explanation: "`arr_matmul` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_matmul\")` for the expected signature." },
    ErrorPattern { pattern: "arr_max_int requires (", category: "arrays", explanation: "`arr_max_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_max_int\")` for the expected signature." },
    ErrorPattern { pattern: "arr_min_int requires (", category: "arrays", explanation: "`arr_min_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_min_int\")` for the expected signature." },
    ErrorPattern { pattern: "arr_mul requires (", category: "arrays", explanation: "`arr_mul` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_mul\")` for the expected signature." },
    ErrorPattern { pattern: "arr_neg requires (", category: "arrays", explanation: "`arr_neg` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_neg\")` for the expected signature." },
    ErrorPattern { pattern: "arr_ones requires (", category: "arrays", explanation: "`arr_ones` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_ones\")` for the expected signature." },
    ErrorPattern { pattern: "arr_outer requires (", category: "arrays", explanation: "`arr_outer` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_outer\")` for the expected signature." },
    ErrorPattern { pattern: "arr_partition_by requires (", category: "arrays", explanation: "`arr_partition_by` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_partition_by\")` for the expected signature." },
    ErrorPattern { pattern: "arr_product requires (", category: "arrays", explanation: "`arr_product` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_product\")` for the expected signature." },
    ErrorPattern { pattern: "arr_push requires (", category: "arrays", explanation: "`arr_push` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_push\")` for the expected signature." },
    ErrorPattern { pattern: "arr_reduce requires (", category: "arrays", explanation: "`arr_reduce` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_reduce\")` for the expected signature." },
    ErrorPattern { pattern: "arr_relu_vec requires (", category: "arrays", explanation: "`arr_relu_vec` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_relu_vec\")` for the expected signature." },
    ErrorPattern { pattern: "arr_repeat requires (", category: "arrays", explanation: "`arr_repeat` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_repeat\")` for the expected signature." },
    ErrorPattern { pattern: "arr_resonance_vec requires (", category: "arrays", explanation: "`arr_resonance_vec` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_resonance_vec\")` for the expected signature." },
    ErrorPattern { pattern: "arr_scale requires (", category: "arrays", explanation: "`arr_scale` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_scale\")` for the expected signature." },
    ErrorPattern { pattern: "arr_set requires (", category: "arrays", explanation: "`arr_set` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_set\")` for the expected signature." },
    ErrorPattern { pattern: "arr_sigmoid_vec requires (", category: "arrays", explanation: "`arr_sigmoid_vec` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_sigmoid_vec\")` for the expected signature." },
    ErrorPattern { pattern: "arr_slice requires (", category: "arrays", explanation: "`arr_slice` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_slice\")` for the expected signature." },
    ErrorPattern { pattern: "arr_softmax requires (", category: "arrays", explanation: "`arr_softmax` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_softmax\")` for the expected signature." },
    ErrorPattern { pattern: "arr_sort_int requires (", category: "arrays", explanation: "`arr_sort_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_sort_int\")` for the expected signature." },
    ErrorPattern { pattern: "arr_sub requires (", category: "arrays", explanation: "`arr_sub` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_sub\")` for the expected signature." },
    ErrorPattern { pattern: "arr_substrate_attention requires (", category: "substrate", explanation: "`arr_substrate_attention` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_substrate_attention\")` for the expected signature." },
    ErrorPattern { pattern: "arr_substrate_score_rows requires (", category: "substrate", explanation: "`arr_substrate_score_rows` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_substrate_score_rows\")` for the expected signature." },
    ErrorPattern { pattern: "arr_sum_int requires (", category: "arrays", explanation: "`arr_sum_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_sum_int\")` for the expected signature." },
    ErrorPattern { pattern: "arr_take requires (", category: "arrays", explanation: "`arr_take` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_take\")` for the expected signature." },
    ErrorPattern { pattern: "arr_transpose requires (", category: "arrays", explanation: "`arr_transpose` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_transpose\")` for the expected signature." },
    ErrorPattern { pattern: "arr_unique requires (", category: "arrays", explanation: "`arr_unique` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_unique\")` for the expected signature." },
    ErrorPattern { pattern: "arr_window requires (", category: "arrays", explanation: "`arr_window` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_window\")` for the expected signature." },
    ErrorPattern { pattern: "arr_zeros requires (", category: "arrays", explanation: "`arr_zeros` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_zeros\")` for the expected signature." },
    ErrorPattern { pattern: "arr_zip requires (", category: "arrays", explanation: "`arr_zip` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"arr_zip\")` for the expected signature." },
    ErrorPattern { pattern: "attractor_bucket requires (", category: "substrate", explanation: "`attractor_bucket` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"attractor_bucket\")` for the expected signature." },
    ErrorPattern { pattern: "attractor_distance requires (", category: "substrate", explanation: "`attractor_distance` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"attractor_distance\")` for the expected signature." },
    ErrorPattern { pattern: "bit_count requires (", category: "core", explanation: "`bit_count` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"bit_count\")` for the expected signature." },
    ErrorPattern { pattern: "bit_length requires (", category: "core", explanation: "`bit_length` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"bit_length\")` for the expected signature." },
    ErrorPattern { pattern: "call requires (", category: "core", explanation: "`call` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"call\")` for the expected signature." },
    ErrorPattern { pattern: "clamp requires (", category: "core", explanation: "`clamp` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"clamp\")` for the expected signature." },
    ErrorPattern { pattern: "crt_recover requires (", category: "substrate", explanation: "`crt_recover` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"crt_recover\")` for the expected signature." },
    ErrorPattern { pattern: "crt_residues requires (", category: "substrate", explanation: "`crt_residues` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"crt_residues\")` for the expected signature." },
    ErrorPattern { pattern: "csv_parse requires (", category: "core", explanation: "`csv_parse` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"csv_parse\")` for the expected signature." },
    ErrorPattern { pattern: "dict_clear requires (", category: "dicts", explanation: "`dict_clear` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_clear\")` for the expected signature." },
    ErrorPattern { pattern: "dict_del requires (", category: "dicts", explanation: "`dict_del` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_del\")` for the expected signature." },
    ErrorPattern { pattern: "dict_get_or requires (", category: "dicts", explanation: "`dict_get_or` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_get_or\")` for the expected signature." },
    ErrorPattern { pattern: "dict_get requires (", category: "dicts", explanation: "`dict_get` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_get\")` for the expected signature." },
    ErrorPattern { pattern: "dict_has requires (", category: "dicts", explanation: "`dict_has` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_has\")` for the expected signature." },
    ErrorPattern { pattern: "dict_items requires (", category: "dicts", explanation: "`dict_items` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_items\")` for the expected signature." },
    ErrorPattern { pattern: "dict_keys requires (", category: "dicts", explanation: "`dict_keys` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_keys\")` for the expected signature." },
    ErrorPattern { pattern: "dict_len requires (", category: "dicts", explanation: "`dict_len` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_len\")` for the expected signature." },
    ErrorPattern { pattern: "dict_merge requires (", category: "dicts", explanation: "`dict_merge` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_merge\")` for the expected signature." },
    ErrorPattern { pattern: "dict_pop requires (", category: "dicts", explanation: "`dict_pop` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_pop\")` for the expected signature." },
    ErrorPattern { pattern: "dict_set requires (", category: "dicts", explanation: "`dict_set` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_set\")` for the expected signature." },
    ErrorPattern { pattern: "dict_size requires (", category: "dicts", explanation: "`dict_size` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_size\")` for the expected signature." },
    ErrorPattern { pattern: "dict_values requires (", category: "dicts", explanation: "`dict_values` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dict_values\")` for the expected signature." },
    ErrorPattern { pattern: "digit_count requires (", category: "core", explanation: "`digit_count` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"digit_count\")` for the expected signature." },
    ErrorPattern { pattern: "digit_sum requires (", category: "core", explanation: "`digit_sum` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"digit_sum\")` for the expected signature." },
    ErrorPattern { pattern: "dual_cos requires (", category: "duals", explanation: "`dual_cos` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_cos\")` for the expected signature." },
    ErrorPattern { pattern: "dual_d requires (", category: "duals", explanation: "`dual_d` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_d\")` for the expected signature." },
    ErrorPattern { pattern: "dual_exp requires (", category: "duals", explanation: "`dual_exp` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_exp\")` for the expected signature." },
    ErrorPattern { pattern: "dual_neg requires (", category: "duals", explanation: "`dual_neg` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_neg\")` for the expected signature." },
    ErrorPattern { pattern: "dual_pow_int requires (", category: "duals", explanation: "`dual_pow_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_pow_int\")` for the expected signature." },
    ErrorPattern { pattern: "dual_relu requires (", category: "duals", explanation: "`dual_relu` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_relu\")` for the expected signature." },
    ErrorPattern { pattern: "dual requires (", category: "core", explanation: "`dual` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual\")` for the expected signature." },
    ErrorPattern { pattern: "dual_sigmoid requires (", category: "duals", explanation: "`dual_sigmoid` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_sigmoid\")` for the expected signature." },
    ErrorPattern { pattern: "dual_sin requires (", category: "duals", explanation: "`dual_sin` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_sin\")` for the expected signature." },
    ErrorPattern { pattern: "dual_tanh requires (", category: "duals", explanation: "`dual_tanh` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_tanh\")` for the expected signature." },
    ErrorPattern { pattern: "dual_v requires (", category: "duals", explanation: "`dual_v` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"dual_v\")` for the expected signature." },
    ErrorPattern { pattern: "fib_chunks requires (", category: "core", explanation: "`fib_chunks` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"fib_chunks\")` for the expected signature." },
    ErrorPattern { pattern: "fibonacci_index requires (", category: "substrate", explanation: "`fibonacci_index` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"fibonacci_index\")` for the expected signature." },
    ErrorPattern { pattern: "file_exists requires (", category: "core", explanation: "`file_exists` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"file_exists\")` for the expected signature." },
    ErrorPattern { pattern: "filter_by_resonance requires (", category: "core", explanation: "`filter_by_resonance` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"filter_by_resonance\")` for the expected signature." },
    ErrorPattern { pattern: "format_time requires (", category: "stdlib", explanation: "`format_time` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"format_time\")` for the expected signature." },
    ErrorPattern { pattern: "from_zeckendorf requires (", category: "core", explanation: "`from_zeckendorf` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"from_zeckendorf\")` for the expected signature." },
    ErrorPattern { pattern: "gcd requires (", category: "core", explanation: "`gcd` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"gcd\")` for the expected signature." },
    ErrorPattern { pattern: "gen_count requires (", category: "generators", explanation: "`gen_count` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"gen_count\")` for the expected signature." },
    ErrorPattern { pattern: "gen_stream requires (", category: "generators", explanation: "`gen_stream` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"gen_stream\")` for the expected signature." },
    ErrorPattern { pattern: "gen_substrate_fib requires (", category: "generators", explanation: "`gen_substrate_fib` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"gen_substrate_fib\")` for the expected signature." },
    ErrorPattern { pattern: "gen_sum requires (", category: "generators", explanation: "`gen_sum` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"gen_sum\")` for the expected signature." },
    ErrorPattern { pattern: "gen_take requires (", category: "generators", explanation: "`gen_take` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"gen_take\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_align requires (", category: "core", explanation: "`harmonic_align` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_align\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_dedupe requires (", category: "core", explanation: "`harmonic_dedupe` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_dedupe\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_diff requires (", category: "core", explanation: "`harmonic_diff` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_diff\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_hash requires (", category: "core", explanation: "`harmonic_hash` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_hash\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_partition requires (", category: "core", explanation: "`harmonic_partition` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_partition\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_read_file requires (", category: "core", explanation: "`harmonic_read_file` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_read_file\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_resample requires (", category: "core", explanation: "`harmonic_resample` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_resample\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_score requires (", category: "core", explanation: "`harmonic_score` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_score\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_split requires (", category: "core", explanation: "`harmonic_split` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_split\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_unalign requires (", category: "core", explanation: "`harmonic_unalign` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_unalign\")` for the expected signature." },
    ErrorPattern { pattern: "harmonic_write_file requires (", category: "core", explanation: "`harmonic_write_file` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmonic_write_file\")` for the expected signature." },
    ErrorPattern { pattern: "harmony requires (", category: "substrate", explanation: "`harmony` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"harmony\")` for the expected signature." },
    ErrorPattern { pattern: "hbit_tension requires (", category: "core", explanation: "`hbit_tension` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"hbit_tension\")` for the expected signature." },
    ErrorPattern { pattern: "hypot requires (", category: "core", explanation: "`hypot` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"hypot\")` for the expected signature." },
    ErrorPattern { pattern: "int_binary_search requires (", category: "core", explanation: "`int_binary_search` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"int_binary_search\")` for the expected signature." },
    ErrorPattern { pattern: "interfere requires (", category: "core", explanation: "`interfere` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"interfere\")` for the expected signature." },
    ErrorPattern { pattern: "int_lower_bound requires (", category: "core", explanation: "`int_lower_bound` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"int_lower_bound\")` for the expected signature." },
    ErrorPattern { pattern: "int_upper_bound requires (", category: "core", explanation: "`int_upper_bound` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"int_upper_bound\")` for the expected signature." },
    ErrorPattern { pattern: "is_attractor requires (", category: "substrate", explanation: "`is_attractor` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"is_attractor\")` for the expected signature." },
    ErrorPattern { pattern: "is_instance requires (", category: "core", explanation: "`is_instance` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"is_instance\")` for the expected signature." },
    ErrorPattern { pattern: "is_phi_resonant requires (", category: "core", explanation: "`is_phi_resonant` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"is_phi_resonant\")` for the expected signature." },
    ErrorPattern { pattern: "is_zeckendorf_valid requires (", category: "core", explanation: "`is_zeckendorf_valid` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"is_zeckendorf_valid\")` for the expected signature." },
    ErrorPattern { pattern: "json_parse requires (", category: "stdlib", explanation: "`json_parse` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"json_parse\")` for the expected signature." },
    ErrorPattern { pattern: "json_stringify requires (", category: "stdlib", explanation: "`json_stringify` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"json_stringify\")` for the expected signature." },
    ErrorPattern { pattern: "largest_attractor_at_most requires (", category: "core", explanation: "`largest_attractor_at_most` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"largest_attractor_at_most\")` for the expected signature." },
    ErrorPattern { pattern: "lcm requires (", category: "core", explanation: "`lcm` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"lcm\")` for the expected signature." },
    ErrorPattern { pattern: "lerp requires (", category: "core", explanation: "`lerp` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"lerp\")` for the expected signature." },
    ErrorPattern { pattern: "log_phi_pi_fibonacci requires (", category: "core", explanation: "`log_phi_pi_fibonacci` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"log_phi_pi_fibonacci\")` for the expected signature." },
    ErrorPattern { pattern: "mean_omni_weight requires (", category: "core", explanation: "`mean_omni_weight` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"mean_omni_weight\")` for the expected signature." },
    ErrorPattern { pattern: "mod_pow requires (", category: "core", explanation: "`mod_pow` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"mod_pow\")` for the expected signature." },
    ErrorPattern { pattern: "nearest_attractor requires (", category: "core", explanation: "`nearest_attractor` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"nearest_attractor\")` for the expected signature." },
    ErrorPattern { pattern: "nth_fibonacci requires (", category: "core", explanation: "`nth_fibonacci` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"nth_fibonacci\")` for the expected signature." },
    ErrorPattern { pattern: "omc_did_you_mean requires (", category: "introspection", explanation: "`omc_did_you_mean` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"omc_did_you_mean\")` for the expected signature." },
    ErrorPattern { pattern: "omc_explain_error requires (", category: "introspection", explanation: "`omc_explain_error` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"omc_explain_error\")` for the expected signature." },
    ErrorPattern { pattern: "omc_help requires (", category: "introspection", explanation: "`omc_help` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"omc_help\")` for the expected signature." },
    ErrorPattern { pattern: "parse_time requires (", category: "stdlib", explanation: "`parse_time` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"parse_time\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_bin_search requires (", category: "core", explanation: "`phi_pi_bin_search` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_bin_search\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_fib_nearest requires (", category: "core", explanation: "`phi_pi_fib_nearest` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_fib_nearest\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_fib_nearest_traced requires (", category: "core", explanation: "`phi_pi_fib_nearest_traced` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_fib_nearest_traced\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_fib_search requires (", category: "core", explanation: "`phi_pi_fib_search` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_fib_search\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_fib_search_traced requires (", category: "core", explanation: "`phi_pi_fib_search_traced` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_fib_search_traced\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_log_distance requires (", category: "core", explanation: "`phi_pi_log_distance` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_log_distance\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pi_pow requires (", category: "core", explanation: "`phi_pi_pow` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pi_pow\")` for the expected signature." },
    ErrorPattern { pattern: "phi_pow requires (", category: "core", explanation: "`phi_pow` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_pow\")` for the expected signature." },
    ErrorPattern { pattern: "phi_shadow requires (", category: "core", explanation: "`phi_shadow` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"phi_shadow\")` for the expected signature." },
    ErrorPattern { pattern: "pow_int requires (", category: "core", explanation: "`pow_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"pow_int\")` for the expected signature." },
    ErrorPattern { pattern: "pow requires (", category: "core", explanation: "`pow` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"pow\")` for the expected signature." },
    ErrorPattern { pattern: "quantization_ratio requires (", category: "core", explanation: "`quantization_ratio` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"quantization_ratio\")` for the expected signature." },
    ErrorPattern { pattern: "quantize requires (", category: "core", explanation: "`quantize` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"quantize\")` for the expected signature." },
    ErrorPattern { pattern: "random_int requires (", category: "core", explanation: "`random_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"random_int\")` for the expected signature." },
    ErrorPattern { pattern: "random_seed requires (", category: "core", explanation: "`random_seed` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"random_seed\")` for the expected signature." },
    ErrorPattern { pattern: "read_file requires (", category: "core", explanation: "`read_file` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"read_file\")` for the expected signature." },
    ErrorPattern { pattern: "re_find_all requires (", category: "regex", explanation: "`re_find_all` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"re_find_all\")` for the expected signature." },
    ErrorPattern { pattern: "re_find requires (", category: "regex", explanation: "`re_find` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"re_find\")` for the expected signature." },
    ErrorPattern { pattern: "re_match requires (", category: "regex", explanation: "`re_match` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"re_match\")` for the expected signature." },
    ErrorPattern { pattern: "re_replace requires (", category: "regex", explanation: "`re_replace` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"re_replace\")` for the expected signature." },
    ErrorPattern { pattern: "resolve_singularity requires (", category: "core", explanation: "`resolve_singularity` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"resolve_singularity\")` for the expected signature." },
    ErrorPattern { pattern: "resonance_band_histogram requires (", category: "core", explanation: "`resonance_band_histogram` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"resonance_band_histogram\")` for the expected signature." },
    ErrorPattern { pattern: "resonance_band requires (", category: "core", explanation: "`resonance_band` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"resonance_band\")` for the expected signature." },
    ErrorPattern { pattern: "re_split requires (", category: "regex", explanation: "`re_split` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"re_split\")` for the expected signature." },
    ErrorPattern { pattern: "safe_arr_get requires (", category: "core", explanation: "`safe_arr_get` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"safe_arr_get\")` for the expected signature." },
    ErrorPattern { pattern: "safe_arr_set requires (", category: "core", explanation: "`safe_arr_set` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"safe_arr_set\")` for the expected signature." },
    ErrorPattern { pattern: "safe_divide requires (", category: "core", explanation: "`safe_divide` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"safe_divide\")` for the expected signature." },
    ErrorPattern { pattern: "safe_log requires (", category: "core", explanation: "`safe_log` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"safe_log\")` for the expected signature." },
    ErrorPattern { pattern: "safe_mod requires (", category: "core", explanation: "`safe_mod` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"safe_mod\")` for the expected signature." },
    ErrorPattern { pattern: "safe_sqrt requires (", category: "core", explanation: "`safe_sqrt` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"safe_sqrt\")` for the expected signature." },
    ErrorPattern { pattern: "sorted_dedupe requires (", category: "core", explanation: "`sorted_dedupe` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"sorted_dedupe\")` for the expected signature." },
    ErrorPattern { pattern: "sorted_merge requires (", category: "core", explanation: "`sorted_merge` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"sorted_merge\")` for the expected signature." },
    ErrorPattern { pattern: "sorted_union requires (", category: "core", explanation: "`sorted_union` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"sorted_union\")` for the expected signature." },
    ErrorPattern { pattern: "str_capitalize requires (", category: "strings", explanation: "`str_capitalize` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_capitalize\")` for the expected signature." },
    ErrorPattern { pattern: "str_contains requires (", category: "strings", explanation: "`str_contains` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_contains\")` for the expected signature." },
    ErrorPattern { pattern: "str_count requires (", category: "strings", explanation: "`str_count` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_count\")` for the expected signature." },
    ErrorPattern { pattern: "str_ends_with requires (", category: "strings", explanation: "`str_ends_with` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_ends_with\")` for the expected signature." },
    ErrorPattern { pattern: "str_index_of requires (", category: "strings", explanation: "`str_index_of` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_index_of\")` for the expected signature." },
    ErrorPattern { pattern: "str_is_empty requires (", category: "strings", explanation: "`str_is_empty` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_is_empty\")` for the expected signature." },
    ErrorPattern { pattern: "str_join requires (", category: "strings", explanation: "`str_join` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_join\")` for the expected signature." },
    ErrorPattern { pattern: "str_pad_left requires (", category: "strings", explanation: "`str_pad_left` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_pad_left\")` for the expected signature." },
    ErrorPattern { pattern: "str_pad_right requires (", category: "strings", explanation: "`str_pad_right` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_pad_right\")` for the expected signature." },
    ErrorPattern { pattern: "str_repeat requires (", category: "strings", explanation: "`str_repeat` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_repeat\")` for the expected signature." },
    ErrorPattern { pattern: "str_replace requires (", category: "strings", explanation: "`str_replace` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_replace\")` for the expected signature." },
    ErrorPattern { pattern: "str_slice requires (", category: "strings", explanation: "`str_slice` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_slice\")` for the expected signature." },
    ErrorPattern { pattern: "str_split_lines requires (", category: "strings", explanation: "`str_split_lines` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_split_lines\")` for the expected signature." },
    ErrorPattern { pattern: "str_split requires (", category: "strings", explanation: "`str_split` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_split\")` for the expected signature." },
    ErrorPattern { pattern: "str_starts_with requires (", category: "strings", explanation: "`str_starts_with` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_starts_with\")` for the expected signature." },
    ErrorPattern { pattern: "str_to_float requires (", category: "strings", explanation: "`str_to_float` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_to_float\")` for the expected signature." },
    ErrorPattern { pattern: "str_to_int requires (", category: "strings", explanation: "`str_to_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"str_to_int\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_count_range requires (", category: "core", explanation: "`substrate_count_range` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_count_range\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_difference requires (", category: "core", explanation: "`substrate_difference` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_difference\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_hash requires (", category: "core", explanation: "`substrate_hash` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_hash\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_insert requires (", category: "core", explanation: "`substrate_insert` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_insert\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_intersect requires (", category: "core", explanation: "`substrate_intersect` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_intersect\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_lower_bound requires (", category: "core", explanation: "`substrate_lower_bound` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_lower_bound\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_min_distance requires (", category: "core", explanation: "`substrate_min_distance` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_min_distance\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_nearest requires (", category: "core", explanation: "`substrate_nearest` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_nearest\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_quantile requires (", category: "core", explanation: "`substrate_quantile` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_quantile\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_rank requires (", category: "core", explanation: "`substrate_rank` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_rank\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_search requires (", category: "core", explanation: "`substrate_search` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_search\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_select_k requires (", category: "core", explanation: "`substrate_select_k` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_select_k\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_slice_range requires (", category: "core", explanation: "`substrate_slice_range` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_slice_range\")` for the expected signature." },
    ErrorPattern { pattern: "substrate_upper_bound requires (", category: "core", explanation: "`substrate_upper_bound` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"substrate_upper_bound\")` for the expected signature." },
    ErrorPattern { pattern: "tape_backward requires (", category: "autograd", explanation: "`tape_backward` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_backward\")` for the expected signature." },
    ErrorPattern { pattern: "tape_grad requires (", category: "autograd", explanation: "`tape_grad` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_grad\")` for the expected signature." },
    ErrorPattern { pattern: "tape_matmul requires (", category: "autograd", explanation: "`tape_matmul` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_matmul\")` for the expected signature." },
    ErrorPattern { pattern: "tape_mean requires (", category: "autograd", explanation: "`tape_mean` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_mean\")` for the expected signature." },
    ErrorPattern { pattern: "tape_neg requires (", category: "autograd", explanation: "`tape_neg` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_neg\")` for the expected signature." },
    ErrorPattern { pattern: "tape_pow_int requires (", category: "autograd", explanation: "`tape_pow_int` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_pow_int\")` for the expected signature." },
    ErrorPattern { pattern: "tape_sum requires (", category: "autograd", explanation: "`tape_sum` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_sum\")` for the expected signature." },
    ErrorPattern { pattern: "tape_update requires (", category: "autograd", explanation: "`tape_update` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_update\")` for the expected signature." },
    ErrorPattern { pattern: "tape_value requires (", category: "autograd", explanation: "`tape_value` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"tape_value\")` for the expected signature." },
    ErrorPattern { pattern: "test_record_failure requires (", category: "core", explanation: "`test_record_failure` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"test_record_failure\")` for the expected signature." },
    ErrorPattern { pattern: "test_set_current requires (", category: "core", explanation: "`test_set_current` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"test_set_current\")` for the expected signature." },
    ErrorPattern { pattern: "write_file requires (", category: "core", explanation: "`write_file` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"write_file\")` for the expected signature." },
    ErrorPattern { pattern: "zeckendorf_bit requires (", category: "core", explanation: "`zeckendorf_bit` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"zeckendorf_bit\")` for the expected signature." },
    ErrorPattern { pattern: "zeckendorf requires (", category: "core", explanation: "`zeckendorf` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"zeckendorf\")` for the expected signature." },
    ErrorPattern { pattern: "zeckendorf_weight requires (", category: "core", explanation: "`zeckendorf_weight` was called with the wrong number of arguments.", typical_cause: "Missing or extra argument(s).", fix: "Check `omc_help(\"zeckendorf_weight\")` for the expected signature." },
];

/// Best-matching pattern for an error message. Returns None if no
/// pattern matched — `omc_explain_error` then returns a "no match"
/// dict with did_you_mean suggestions over the catalog.
pub fn match_error(msg: &str) -> Option<&'static ErrorPattern> {
    // Patterns are kept in roughly most-specific-first order; the
    // first substring hit wins.
    ERROR_PATTERNS.iter().find(|p| msg.contains(p.pattern))
}

/// Distinct categories — used for cataloging.
pub fn error_categories() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for p in ERROR_PATTERNS {
        if !out.contains(&p.category) {
            out.push(p.category);
        }
    }
    out
}

/// Render a pattern as Markdown (used by --gen-docs).
pub fn render_pattern(p: &ErrorPattern) -> String {
    format!(
        "### `{}`\n\n**Category**: {}\n\n**Means**: {}\n\n**Cause**: {}\n\n**Fix**: {}\n",
        p.pattern, p.category, p.explanation, p.typical_cause, p.fix
    )
}

/// Render the full error catalog as Markdown.
pub fn render_full_errors() -> String {
    let mut out = String::new();
    out.push_str("# OMC Error Catalog\n\n");
    out.push_str(&format!("**Total patterns**: {}\n\n", ERROR_PATTERNS.len()));
    out.push_str("Pattern matching is substring-based. `omc_explain_error(msg)` runs the live runtime lookup.\n\n---\n\n");
    for cat in error_categories() {
        out.push_str(&format!("## {}\n\n", cat));
        for p in ERROR_PATTERNS.iter().filter(|p| p.category == cat) {
            out.push_str(&render_pattern(p));
            out.push('\n');
        }
        out.push_str("---\n\n");
    }
    out
}
