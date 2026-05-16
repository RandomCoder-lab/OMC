// omnimcode-core/src/compiler.rs — AST → bytecode lowering.

use crate::ast::*;
use crate::bytecode::*;

thread_local! {
    /// Monotonic counter for anonymous lambda names emitted by the
    /// compiler. Shared across all Compiler instances within a single
    /// compile_program call so closures get globally-unique names.
    static LAMBDA_SEQ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Loop tracking for `break` / `continue` patch-up.
struct LoopFrame {
    /// Instruction to resume on `continue`.
    continue_target: usize,
    /// Jump-op indices that need to be patched to the loop's exit (set on break).
    break_jumps: Vec<usize>,
}

/// Statically-known type for a variable or expression, used by Phase M's
/// HIR to specialize arithmetic opcodes. "int" / "float" / "string" / "bool"
/// / "array" map directly from the source-level annotations; `None` means
/// the type couldn't be proved statically and runtime polymorphism applies.
type TypeTag = Option<&'static str>;

pub struct Compiler {
    constants: Vec<Const>,
    ops: Vec<Op>,
    loop_stack: Vec<LoopFrame>,
    /// Names of user-defined functions. Used to suppress hot-path inlining
    /// at call sites where the user has redefined a built-in (e.g. a
    /// canonical recursive `fib`).
    user_fns: std::collections::HashSet<String>,
    /// Phase M: statically-tracked variable types, populated from parameter
    /// annotations and obvious-literal var decls.
    var_types: std::collections::HashMap<String, &'static str>,
    /// Phase M: declared return types of user-defined functions, looked up
    /// when inferring the type of a Call expression.
    fn_return_types: std::collections::HashMap<String, &'static str>,
    /// Lambda bodies compiled during this Compiler's run. Drained by
    /// compile_program after each top-level / per-function compile and
    /// inserted into module.functions so closure invocation can find them.
    pending_lambdas: Vec<CompiledFunction>,
    /// Lambda body AST forms — drained by compile_program and exposed
    /// via `compile_program`'s return so main.rs can register them
    /// into the interpreter's function table. Required because the
    /// existing call_first_class_function dispatches by name through
    /// the interpreter (tree-walk), not through module.functions.
    pending_lambda_asts: Vec<(String, Vec<String>, Vec<Statement>)>,
    /// Source position attached to each emitted op, indexed by op
    /// position. Built up alongside `ops`; finish() resizes to match
    /// the final op count, padding any missing tail with Pos::unknown().
    op_positions: Vec<crate::ast::Pos>,
}

impl Compiler {
    #[allow(dead_code)] // alternate constructor; primary compile entry points use seeded variants
    fn new() -> Self {
        Compiler {
            constants: Vec::new(),
            ops: Vec::new(),
            loop_stack: Vec::new(),
            user_fns: std::collections::HashSet::new(),
            var_types: std::collections::HashMap::new(),
            fn_return_types: std::collections::HashMap::new(),
            pending_lambdas: Vec::new(),
            pending_lambda_asts: Vec::new(),
            op_positions: Vec::new(),
        }
    }

    fn with_user_fns(user_fns: std::collections::HashSet<String>) -> Self {
        Compiler {
            constants: Vec::new(),
            ops: Vec::new(),
            loop_stack: Vec::new(),
            user_fns,
            var_types: std::collections::HashMap::new(),
            fn_return_types: std::collections::HashMap::new(),
            pending_lambdas: Vec::new(),
            pending_lambda_asts: Vec::new(),
            op_positions: Vec::new(),
        }
    }

    /// Statically infer the type of an Expression, returning Some(tag) when
    /// the type is provably one of "int" / "float" / "string" / "bool" /
    /// "array". Used by arithmetic emission to pick specialized opcodes.
    fn infer_type(&self, e: &Expression) -> TypeTag {
        match e {
            Expression::Number(_) => Some("int"),
            Expression::Float(_) => Some("float"),
            Expression::String(_) => Some("string"),
            Expression::Boolean(_) => Some("bool"),
            Expression::Array(_) => Some("array"),
            Expression::Dict(_) => Some("dict"),
            Expression::Variable(name) => self.var_types.get(name.as_str()).copied(),
            Expression::Add(l, r)
            | Expression::Sub(l, r)
            | Expression::Mul(l, r) => {
                match (self.infer_type(l), self.infer_type(r)) {
                    (Some("int"), Some("int")) => Some("int"),
                    (Some("float"), _) | (_, Some("float")) => Some("float"),
                    _ => None,
                }
            }
            Expression::Div(l, r) => {
                // Integer division of two ints stays int; mixed promotes to float.
                match (self.infer_type(l), self.infer_type(r)) {
                    (Some("int"), Some("int")) => Some("int"),
                    (Some("float"), _) | (_, Some("float")) => Some("float"),
                    _ => None,
                }
            }
            Expression::Mod(_, _) => Some("int"),
            Expression::Eq(_, _)
            | Expression::Ne(_, _)
            | Expression::Lt(_, _)
            | Expression::Le(_, _)
            | Expression::Gt(_, _)
            | Expression::Ge(_, _)
            | Expression::And(_, _)
            | Expression::Or(_, _)
            | Expression::Not(_) => Some("bool"),
            Expression::BitAnd(_, _)
            | Expression::BitOr(_, _)
            | Expression::BitXor(_, _)
            | Expression::BitNot(_)
            | Expression::Shl(_, _)
            | Expression::Shr(_, _) => Some("int"),
            Expression::Resonance(_) => Some("float"),
            Expression::Fold(_) => Some("int"),
            Expression::Call { name, .. } => {
                self.fn_return_types.get(name.as_str()).copied().or_else(|| {
                    // Built-ins whose return type is fixed.
                    match name.as_str() {
                        // Truly int-returning builtins. Polymorphic ones
                        // (arr_get, dict_get, arr_min/max/sum — return value
                        // depends on the element type) are deliberately
                        // EXCLUDED. Listing them here causes the compiler
                        // to emit Op::AddInt for `arr_get(...) + x`, which
                        // calls .to_int() on both operands and silently
                        // truncates floats. Caught by a real-world float-
                        // accumulator pattern in examples/recommend.
                        "fibonacci" | "fib" | "is_fibonacci" | "factorial"
                        | "abs" | "floor" | "ceil" | "round" | "is_prime"
                        | "even" | "odd" | "is_even" | "is_odd"
                        | "len" | "arr_len"
                        | "arr_index_of" | "arr_contains"
                        | "is_singularity" | "resolve_singularity"
                        | "pow_int" | "square" | "cube" | "sign" | "to_int"
                        | "int" | "classify_resonance" | "safe_add" | "safe_sub"
                        | "safe_mul"
                        // Substrate primitives — int returns
                        | "attractor_distance" | "nearest_attractor"
                        | "largest_attractor_at_most" | "hbit_tension"
                        | "is_attractor" | "resonance_band"
                        | "crt_recover" | "fibonacci_index"
                        // Substrate-routed O(log_phi_pi_fibonacci N) search ints
                        | "substrate_search" | "substrate_lower_bound"
                        | "substrate_upper_bound" | "substrate_rank"
                        | "substrate_count_range" | "from_zeckendorf"
                        | "zeckendorf_weight" | "zeckendorf_bit" | "substrate_hash"
                        | "attractor_bucket" | "substrate_insert"
                        | "harmonic_align" | "harmonic_unalign"
                        | "substrate_select_k"
                        | "int_binary_search" | "int_lower_bound" | "int_upper_bound"
                        | "nth_fibonacci" | "is_zeckendorf_valid"
                        | "substrate_min_distance" | "substrate_nearest"
                        | "arr_sum_int" | "arr_product" | "arr_is_sorted"
                        | "arr_min_int" | "arr_max_int"
                        | "is_phi_resonant"
                        // 2026-05-14 stdlib expansion (ints)
                        | "str_index_of" | "str_starts_with" | "str_ends_with"
                        | "file_exists" | "write_file" | "gcd" | "lcm"
                        | "now_ms"
                        // polish round (ints)
                        | "random_int" | "random_seed"
                        // test runner ints
                        | "test_failure_count" | "test_record_failure"
                        // String → int parsers + counters + size queries
                        | "str_to_int" | "str_count" | "str_is_empty"
                        | "dict_size" | "dict_pop"
                        // Regex predicate returns int
                        | "re_match"
                        // Datetime ints
                        | "now_unix" | "parse_time"
                        // Array index/aggregate ints
                        | "arr_argmax" | "arr_argmin"
                        // Bit/digit/modular int returns
                        | "mod_pow" | "bit_count" | "bit_length"
                        | "digit_sum" | "digit_count"
                        | "arr_unique_count" | "arr_gcd" | "fnv1a_hash"
                        | "is_instance" | "omc_error_count"
                        // Substrate-token adapter: token IDs + distance + pack
                        | "omc_token_distance" | "omc_token_vocab_size"
                        | "omc_token_pack" | "omc_code_distance"
                        | "omc_code_equivalent"
                        | "omc_code_uses_python" | "omc_code_uses_substrate"
                        | "omc_code_fingerprint"
                        | "omc_categories_count" | "omc_builtin_count"
                        | "omc_unique_count"
                        // tape_* op constructors return node IDs (int)
                        | "tape_var" | "tape_const"
                        | "tape_add" | "tape_sub" | "tape_mul" | "tape_div"
                        | "tape_neg" | "tape_pow_int"
                        | "tape_exp" | "tape_sin" | "tape_cos"
                        | "tape_relu" | "tape_sigmoid" | "tape_tanh"
                        | "tape_matmul" | "tape_sum" | "tape_mean"
                        // Lazy generators: gen_stream/gen_count/gen_sum
                        // return int (success/count/accumulator).
                        | "gen_stream" | "gen_count" | "gen_sum"
                        | "gen_substrate_fib" => Some("int"),
                        "pow" | "sqrt" | "log" | "log2" | "log10"
                        | "exp" | "sin" | "cos" | "tan" | "asin" | "acos"
                        | "atan" | "atan2" | "hypot" | "lerp"
                        | "tanh" | "erf" | "sigmoid" | "frac" | "clamp"
                        | "arr_mean" | "arr_variance" | "arr_stddev"
                        | "arr_median" | "arr_harmonic_mean"
                        | "arr_geometric_mean" | "arr_sum_sq"
                        | "arr_norm" | "arr_dot"
                        | "pi" | "e" | "phi" | "tau" | "phi_inv" | "phi_sq"
                        | "phi_squared" | "sqrt_2" | "sqrt_5" | "ln_2"
                        | "to_float" | "float" | "interfere"
                        | "harmonic_interfere" | "measure_coherence"
                        | "arr_resonance" | "collapse" | "res" | "phi.res"
                        | "phi.fold" | "phi.him"
                        // L1: substrate-routed log; returns float in
                        // [0, ~10] for typical input. Without this entry
                        // the compiler emits Op::Mul (untyped) for
                        // `log_phi_pi_fibonacci(x) * 50` which JIT
                        // treats as int mul of float-bit-pattern,
                        // producing garbage.
                        | "log_phi_pi_fibonacci"
                        // polish round (floats)
                        | "random_float"
                        // String → float parser
                        | "str_to_float"
                        // Float-preserving array reductions
                        | "arr_min_float" | "arr_max_float"
                        // Substrate-canonical distance metric
                        | "phi_pi_log_distance"
                        // Substrate growth rates
                        | "phi_pow" | "phi_pi_pow"
                        // Substrate-coherence + array-stat float returns
                        | "harmonic_score" | "arr_avg_distance" => Some("float"),
                        "to_string" | "string" | "str_concat"
                        | "str_uppercase" | "str_lowercase" | "str_reverse"
                        | "str_slice" | "concat_many"
                        // 2026-05-14 stdlib expansion (strings)
                        | "str_trim" | "str_replace" | "str_repeat"
                        | "str_join" | "arr_join" | "read_file" | "type_of"
                        // polish round (strings)
                        | "str_pad_left" | "str_pad_right"
                        // test runner: get_current returns the current test name
                        | "test_get_current" => Some("string"),
                        // Float returns
                        "harmonic_checksum" | "harmonic_write_file"
                        | "harmonic_hash" | "harmonic_diff" => Some("float"),
                        "arr_new" | "arr_from_range" | "arr_concat"
                        | "arr_slice" | "cleanup_array"
                        | "filter_by_resonance"
                        // 2026-05-14 stdlib expansion (arrays)
                        | "str_split" | "arr_sort" | "arr_reverse"
                        // First-class higher-order returns array of mapped items
                        | "arr_map" | "arr_filter"
                        // Harmonic variants returning arrays
                        | "harmonic_read_file" | "harmonic_sort"
                        | "harmonic_split" | "harmonic_partition"
                        | "harmonic_dedupe"
                        // polish round (arrays)
                        | "arr_zip" | "arr_unique"
                        // 2D array primitives (Track 2 — 2026-05-16)
                        | "arr_matmul" | "arr_transpose"
                        | "arr_eye" | "arr_zeros_2d"
                        // Native ML primitives (Track 3 — 2026-05-16)
                        | "arr_softmax" | "arr_layer_norm"
                        | "arr_relu_vec" | "arr_sigmoid_vec"
                        | "arr_conv1d" | "arr_outer"
                        // Substrate-native acceleration (OMC-unique)
                        | "arr_substrate_attention"
                        | "arr_substrate_score_rows"
                        // Lazy generator collector: returns array
                        | "gen_take"
                        // Introspection / discoverability surface
                        | "omc_list_builtins" | "omc_categories"
                        | "omc_did_you_mean" | "omc_unique_builtins"
                        | "omc_error_categories"
                        // Substrate-token adapter returns int array / string array
                        | "omc_token_encode" | "omc_token_unpack"
                        | "omc_token_vocab"
                        // Code intel returns arrays of names
                        | "omc_code_extract_fns" | "omc_code_dependencies"
                        | "omc_completion_hint"
                        // Forward-mode autograd duals (Track 2 — 2026-05-16)
                        | "dual" | "dual_add" | "dual_sub"
                        | "dual_mul" | "dual_div" | "dual_neg"
                        | "dual_pow_int" | "dual_exp"
                        | "dual_sin" | "dual_cos"
                        | "dual_relu" | "dual_sigmoid" | "dual_tanh"
                        // introspection
                        | "defined_functions"
                        // test runner: get_failures returns array of strings
                        | "test_get_failures" => Some("array"),
                        _ => None,
                    }
                })
            }
            Expression::Index { .. } => None,
            // H.5: `safe <expr>` evaluates to the same type as the inner
            // expression after self-healing dispatch. For Div the result is
            // int-or-float same as Div itself; for arr_get/arr_set the
            // result mirrors the wrapped call. Delegating to the inner
            // gives the right answer in every supported shape.
            Expression::Safe(inner) => self.infer_type(inner),
            // Lambdas evaluate to a function value at runtime. Type
            // inference can't see across the call boundary statically,
            // so we don't claim a return-type tag here.
            Expression::Lambda { .. } => None,
        }
    }

    fn add_const(&mut self, c: Const) -> usize {
        let idx = self.constants.len();
        self.constants.push(c);
        idx
    }

    fn emit(&mut self, op: Op) -> usize {
        self.emit_at(op, crate::ast::Pos::unknown())
    }

    /// Emit an op with an attached source position. Used by Op::Call
    /// emission so VM-thrown errors can point back at the call site
    /// in the original source.
    fn emit_at(&mut self, op: Op, pos: crate::ast::Pos) -> usize {
        let idx = self.ops.len();
        self.ops.push(op);
        // Keep op_positions in lockstep so the VM can index either.
        self.op_positions.push(pos);
        idx
    }

    fn patch_jump(&mut self, jump_idx: usize, target: usize) {
        // jumps are relative to the instruction AFTER the jump op.
        let offset = (target as i32) - (jump_idx as i32) - 1;
        match &mut self.ops[jump_idx] {
            Op::Jump(o) | Op::JumpIfFalse(o) | Op::JumpIfTrue(o) => *o = offset,
            _ => panic!("patch_jump on non-jump op at {}", jump_idx),
        }
    }

    fn compile_expr(&mut self, e: &Expression) -> Result<(), String> {
        match e {
            Expression::Number(n) => {
                let idx = self.add_const(Const::Int(*n));
                self.emit(Op::LoadConst(idx));
            }
            Expression::Float(f) => {
                let idx = self.add_const(Const::Float(*f));
                self.emit(Op::LoadConst(idx));
            }
            Expression::String(s) => {
                let idx = self.add_const(Const::Str(s.clone()));
                self.emit(Op::LoadConst(idx));
            }
            Expression::Boolean(b) => {
                let idx = self.add_const(Const::Bool(*b));
                self.emit(Op::LoadConst(idx));
            }
            Expression::Variable(name) => {
                self.emit(Op::LoadVar(name.clone()));
            }
            Expression::Index { name, index } => {
                self.emit(Op::LoadVar(name.clone()));
                self.compile_expr(index)?;
                self.emit(Op::ArrayIndex);
            }
            Expression::Array(items) => {
                for item in items {
                    self.compile_expr(item)?;
                }
                self.emit(Op::NewArray(items.len()));
            }
            Expression::Dict(pairs) => {
                for (k, v) in pairs {
                    self.compile_expr(k)?;
                    self.compile_expr(v)?;
                }
                self.emit(Op::NewDict(pairs.len()));
            }
            Expression::Add(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("int"), Some("int")) => self.emit(Op::AddInt),
                    (Some("float"), Some("float")) => self.emit(Op::AddFloat),
                    _ => self.emit(Op::Add),
                };
            }
            Expression::Sub(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("int"), Some("int")) => self.emit(Op::SubInt),
                    (Some("float"), Some("float")) => self.emit(Op::SubFloat),
                    _ => self.emit(Op::Sub),
                };
            }
            Expression::Mul(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("int"), Some("int")) => self.emit(Op::MulInt),
                    (Some("float"), Some("float")) => self.emit(Op::MulFloat),
                    _ => self.emit(Op::Mul),
                };
            }
            Expression::Div(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::DivFloat),
                    _ => self.emit(Op::Div),
                };
            }
            Expression::Mod(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Mod);
            }
            Expression::Eq(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::EqFloat),
                    _ => self.emit(Op::Eq),
                };
            }
            Expression::Ne(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::NeFloat),
                    _ => self.emit(Op::Ne),
                };
            }
            Expression::Lt(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::LtFloat),
                    _ => self.emit(Op::Lt),
                };
            }
            Expression::Le(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::LeFloat),
                    _ => self.emit(Op::Le),
                };
            }
            Expression::Gt(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::GtFloat),
                    _ => self.emit(Op::Gt),
                };
            }
            Expression::Ge(l, r) => {
                let lt = self.infer_type(l);
                let rt = self.infer_type(r);
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                match (lt, rt) {
                    (Some("float"), Some("float")) => self.emit(Op::GeFloat),
                    _ => self.emit(Op::Ge),
                };
            }
            Expression::And(l, r) => {
                // Short-circuit: eval l; if false, push false and skip r.
                self.compile_expr(l)?;
                // Duplicate top, so we can branch and keep one copy.
                // Simpler: branch on negation, otherwise pop and eval r.
                let jump = self.emit(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                self.compile_expr(r)?;
                let end = self.ops.len();
                self.patch_jump(jump, end);
            }
            Expression::Or(l, r) => {
                self.compile_expr(l)?;
                let jump = self.emit(Op::JumpIfTrue(0));
                self.emit(Op::Pop);
                self.compile_expr(r)?;
                let end = self.ops.len();
                self.patch_jump(jump, end);
            }
            Expression::Not(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Not);
            }
            Expression::BitAnd(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::BitAnd);
            }
            Expression::BitOr(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::BitOr);
            }
            Expression::BitXor(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::BitXor);
            }
            Expression::BitNot(e) => {
                self.compile_expr(e)?;
                self.emit(Op::BitNot);
            }
            Expression::Shl(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Shl);
            }
            Expression::Shr(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Shr);
            }
            Expression::Resonance(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Resonance);
            }
            Expression::Fold(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Fold1);
            }
            Expression::Call { name, args, pos } => {
                // Capture the call-site position for the bytecode emit
                // so VM-thrown errors can show the same "(line:col)" the
                // tree-walk side does. Stored on Op::Call directly.
                let _site_pos = *pos;
                // Mutating built-ins must be specialized so the VM doesn't
                // route them through vm_call_builtin's synthetic-arg shim
                // (which would otherwise lose the mutation — the shim
                // copies args into __vm_arg_N variables and the built-in
                // mutates the COPY).
                if !self.user_fns.contains(name) {
                    if name == "arr_push" && args.len() == 2 {
                        if let Expression::Variable(arr_name) = &args[0] {
                            // value first → on stack; then the named push.
                            self.compile_expr(&args[1])?;
                            self.emit(Op::ArrPushNamed(arr_name.clone()));
                            return Ok(());
                        }
                    }
                    if name == "arr_set" && args.len() == 3 {
                        if let Expression::Variable(arr_name) = &args[0] {
                            // value, then index → stack top is index, then value
                            self.compile_expr(&args[1])?; // index
                            self.compile_expr(&args[2])?; // value
                            self.emit(Op::ArrSetNamed(arr_name.clone()));
                            return Ok(());
                        }
                    }
                    if name == "dict_set" && args.len() == 3 {
                        if let Expression::Variable(d_name) = &args[0] {
                            // key then value → stack top is value, beneath it is key
                            self.compile_expr(&args[1])?; // key
                            self.compile_expr(&args[2])?; // value
                            self.emit(Op::DictSetNamed(d_name.clone()));
                            // dict_set returns Null in tree-walk; mirror that
                            // so the stack stays balanced for the caller.
                            let null_idx = self.add_const(Const::Null);
                            self.emit(Op::LoadConst(null_idx));
                            return Ok(());
                        }
                    }
                    if name == "dict_del" && args.len() == 2 {
                        if let Expression::Variable(d_name) = &args[0] {
                            self.compile_expr(&args[1])?; // key
                            self.emit(Op::DictDelNamed(d_name.clone()));
                            let null_idx = self.add_const(Const::Null);
                            self.emit(Op::LoadConst(null_idx));
                            return Ok(());
                        }
                    }
                }
                // Fast-path inline for hot harmonic ops — avoids the Call -> bridge
                // -> stdlib lookup overhead. Only inline when the user HASN'T
                // redefined the name (preserves recursion-by-shadowing).
                let can_inline = !self.user_fns.contains(name);
                if can_inline {
                    match (name.as_str(), args.len()) {
                        // `phi.X` module-qualified calls are always built-ins —
                        // the dot disambiguates so inlining is safe.
                        ("phi.res", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::Resonance);
                            return Ok(());
                        }
                        ("phi.fold", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::Fold1);
                            return Ok(());
                        }
                        ("phi.him", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::HimScore);
                            return Ok(());
                        }
                        // Bare names — inline only when not user-redefined.
                        ("res", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::Resonance);
                            return Ok(());
                        }
                        ("fold", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::Fold1);
                            return Ok(());
                        }
                        ("is_fibonacci", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::IsFibonacci);
                            return Ok(());
                        }
                        ("fibonacci", 1) | ("fib", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::Fibonacci);
                            return Ok(());
                        }
                        ("arr_len", 1) | ("len", 1) => {
                            self.compile_expr(&args[0])?;
                            self.emit(Op::ArrayLen);
                            return Ok(());
                        }
                        // arr_get(arr, idx) is the hottest array call —
                        // inline to ArrayIndex so we skip vm_call_builtin's
                        // synthetic-arg shim. This is the same dispatch
                        // `arr[idx]` already uses; aligning fn syntax with
                        // bracket syntax was the gap that made the
                        // arr_push+arr_get benchmark slower under VM than
                        // tree-walk. ArrayIndex is polymorphic over arrays
                        // and dicts, so dict_get(d, k) inlines too.
                        ("arr_get", 2) | ("dict_get", 2) => {
                            self.compile_expr(&args[0])?;
                            self.compile_expr(&args[1])?;
                            self.emit(Op::ArrayIndex);
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                for arg in args {
                    self.compile_expr(arg)?;
                }
                // emit_at attaches the call-site pos so VM-thrown
                // errors surface a line number in stack traces.
                self.emit_at(Op::Call(name.clone(), args.len()), _site_pos);
            }
            Expression::Safe(inner) => {
                // H.5 host-level: lower `safe <expr>` to the matching
                // ONN primitive call. The host primitives (safe_divide,
                // safe_arr_get, safe_arr_set) handle the fold-and-mod /
                // fold-escape logic at runtime. For shapes we don't have
                // a primitive for, just compile the inner directly.
                //
                // KNOWN GAP: Safe(arr_set(VAR, ...)) goes through Op::Call
                // which routes via the vm_call_builtin shim — the mutation
                // is lost when run through the Rust VM. Tree-walk works
                // fine because the interpreter pattern-matches Safe before
                // any shim. A future Op::SafeArrSetNamed would close this
                // gap (same shape as Op::ArrSetNamed in the existing VM).
                match inner.as_ref() {
                    Expression::Div(l, r) => {
                        self.compile_expr(l)?;
                        self.compile_expr(r)?;
                        self.emit(Op::Call("safe_divide".to_string(), 2));
                    }
                    Expression::Call { name, args, .. } if name == "arr_get" && args.len() == 2 => {
                        for arg in args {
                            self.compile_expr(arg)?;
                        }
                        self.emit(Op::Call("safe_arr_get".to_string(), 2));
                    }
                    Expression::Call { name, args, .. } if name == "arr_set" && args.len() == 3 => {
                        // H.5.2: bare-VAR first arg → emit SafeArrSetNamed
                        // so the mutation propagates back through VM scope.
                        // Non-VAR shapes (e.g. nested array) fall through
                        // to the synthetic-arg call shim, which loses the
                        // mutation (same semantics as plain arr_set on a
                        // non-VAR).
                        if let Expression::Variable(arr_name) = &args[0] {
                            self.compile_expr(&args[1])?; // index
                            self.compile_expr(&args[2])?; // value
                            self.emit(Op::SafeArrSetNamed(arr_name.clone()));
                        } else {
                            for arg in args {
                                self.compile_expr(arg)?;
                            }
                            self.emit(Op::Call("safe_arr_set".to_string(), 3));
                        }
                    }
                    _ => self.compile_expr(inner)?,
                }
            }
            Expression::Lambda { params, body } => {
                // Generate a unique anonymous name so it doesn't collide
                // with anything in module.functions. The counter is per-
                // Compiler — main.rs creates one Compiler for the top
                // level + one per user fn, so the namespace `__lambda_*`
                // is shared across them but globally unique due to the
                // module-level lambda_seq counter.
                let lambda_seq = LAMBDA_SEQ.with(|c| {
                    let v = c.get();
                    c.set(v + 1);
                    v
                });
                let fn_name = format!("__lambda_{}", lambda_seq);
                // Stash the AST body too — call_first_class_function
                // dispatches by name through the interpreter (tree-walk),
                // not through module.functions, so we need the original
                // AST registered there as well.
                self.pending_lambda_asts.push((
                    fn_name.clone(),
                    params.clone(),
                    body.clone(),
                ));
                // Compile the body. We use a fresh Compiler with the
                // outer user_fns set so the body sees the same names.
                let mut fc = Compiler::with_user_fns(self.user_fns.clone());
                fc.fn_return_types = self.fn_return_types.clone();
                for s in body {
                    fc.compile_stmt(s)?;
                }
                fc.emit(Op::ReturnNull);
                // Drain nested lambdas BEFORE finish (which consumes fc).
                let nested = std::mem::take(&mut fc.pending_lambdas);
                let nested_asts = std::mem::take(&mut fc.pending_lambda_asts);
                let func = fc.finish(
                    fn_name.clone(),
                    params.clone(),
                    vec![None; params.len()],
                    None,
                    Vec::new(), // lambdas don't carry pragmas
                );
                self.pending_lambdas.push(func);
                for nf in nested {
                    self.pending_lambdas.push(nf);
                }
                for na in nested_asts {
                    self.pending_lambda_asts.push(na);
                }
                // Emit the runtime op that creates Value::Function with
                // captured = current scope. Sibling closures in the same
                // scope share the captured Rc.
                self.emit(Op::Lambda(fn_name));
            }
        }
        Ok(())
    }

    fn compile_stmt(&mut self, s: &Statement) -> Result<(), String> {
        match s {
            Statement::Print(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Print);
            }
            Statement::Expression(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Pop);
            }
            Statement::VarDecl { name, value, .. } | Statement::Parameter { name, value } => {
                // Phase M: remember statically-known type before lowering the
                // value, so any subsequent uses in expressions can specialize.
                if let Some(t) = self.infer_type(value) {
                    self.var_types.insert(name.clone(), t);
                }
                self.compile_expr(value)?;
                self.emit(Op::StoreVar(name.clone()));
            }
            Statement::Assignment { name, value } => {
                if let Some(t) = self.infer_type(value) {
                    self.var_types.insert(name.clone(), t);
                }
                self.compile_expr(value)?;
                self.emit(Op::AssignVar(name.clone()));
            }
            Statement::IndexAssignment { name, index, value } => {
                self.compile_expr(value)?;
                self.compile_expr(index)?;
                self.emit(Op::ArrayIndexAssign(name.clone()));
            }
            Statement::If {
                condition,
                then_body,
                elif_parts,
                else_body,
            } => {
                // if / elif / else chain
                let mut end_jumps: Vec<usize> = Vec::new();

                self.compile_expr(condition)?;
                let mut last_skip = self.emit(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                for stmt in then_body {
                    self.compile_stmt(stmt)?;
                }
                end_jumps.push(self.emit(Op::Jump(0)));

                for (elif_cond, elif_body) in elif_parts {
                    let here = self.ops.len();
                    self.patch_jump(last_skip, here);
                    self.emit(Op::Pop); // pop the false condition value
                    self.compile_expr(elif_cond)?;
                    last_skip = self.emit(Op::JumpIfFalse(0));
                    self.emit(Op::Pop);
                    for stmt in elif_body {
                        self.compile_stmt(stmt)?;
                    }
                    end_jumps.push(self.emit(Op::Jump(0)));
                }

                let else_start = self.ops.len();
                self.patch_jump(last_skip, else_start);
                self.emit(Op::Pop);
                if let Some(body) = else_body {
                    for stmt in body {
                        self.compile_stmt(stmt)?;
                    }
                }
                let end = self.ops.len();
                for j in end_jumps {
                    self.patch_jump(j, end);
                }
            }
            Statement::While { condition, body } => {
                let loop_start = self.ops.len();
                self.loop_stack.push(LoopFrame {
                    continue_target: loop_start,
                    break_jumps: Vec::new(),
                });
                self.compile_expr(condition)?;
                let exit_jump = self.emit(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                // Unconditional jump back to start.
                let back = self.emit(Op::Jump(0));
                let offset = (loop_start as i32) - (back as i32) - 1;
                if let Op::Jump(o) = &mut self.ops[back] {
                    *o = offset;
                }
                let exit = self.ops.len();
                self.patch_jump(exit_jump, exit);
                self.emit(Op::Pop); // pop the false condition

                // Patch any `break` jumps inside this loop to the exit.
                let frame = self.loop_stack.pop().unwrap();
                let after_exit = self.ops.len();
                for j in frame.break_jumps {
                    self.patch_jump(j, after_exit);
                }
            }
            Statement::For { var, iterable, body } => {
                match iterable {
                    ForIterable::Range { start, end } => {
                        // for var in start..end:  var = start; while var < end { body; var += 1 }
                        self.compile_expr(start)?;
                        self.emit(Op::StoreVar(var.clone()));

                        let loop_start = self.ops.len();
                        self.loop_stack.push(LoopFrame {
                            continue_target: 0, // patched below
                            break_jumps: Vec::new(),
                        });
                        self.emit(Op::LoadVar(var.clone()));
                        self.compile_expr(end)?;
                        self.emit(Op::Lt);
                        let exit_jump = self.emit(Op::JumpIfFalse(0));
                        self.emit(Op::Pop);

                        for stmt in body {
                            self.compile_stmt(stmt)?;
                        }
                        // continue lands HERE — at the increment
                        let cont_target = self.ops.len();
                        self.loop_stack.last_mut().unwrap().continue_target = cont_target;
                        self.emit(Op::LoadVar(var.clone()));
                        let one = self.add_const(Const::Int(1));
                        self.emit(Op::LoadConst(one));
                        self.emit(Op::Add);
                        self.emit(Op::StoreVar(var.clone()));

                        let back = self.emit(Op::Jump(0));
                        let offset = (loop_start as i32) - (back as i32) - 1;
                        if let Op::Jump(o) = &mut self.ops[back] {
                            *o = offset;
                        }
                        let exit = self.ops.len();
                        self.patch_jump(exit_jump, exit);
                        self.emit(Op::Pop);

                        let frame = self.loop_stack.pop().unwrap();
                        let after_exit = self.ops.len();
                        for j in frame.break_jumps {
                            self.patch_jump(j, after_exit);
                        }
                    }
                    ForIterable::Expr(arr_expr) => {
                        // for var in arr:
                        //   __it = 0; __n = len(arr);
                        //   while __it < __n { var = arr[__it]; body; __it += 1 }
                        // Uses a unique-ish index name to avoid collisions.
                        let idx_var = format!("__for_idx_{}", self.ops.len());
                        let arr_var = format!("__for_arr_{}", self.ops.len());

                        // __arr = arr_expr; __it = 0;
                        self.compile_expr(arr_expr)?;
                        self.emit(Op::StoreVar(arr_var.clone()));
                        let zero = self.add_const(Const::Int(0));
                        self.emit(Op::LoadConst(zero));
                        self.emit(Op::StoreVar(idx_var.clone()));

                        let loop_start = self.ops.len();
                        self.loop_stack.push(LoopFrame {
                            continue_target: 0, // patched below
                            break_jumps: Vec::new(),
                        });
                        // condition: __it < len(__arr)
                        self.emit(Op::LoadVar(idx_var.clone()));
                        self.emit(Op::LoadVar(arr_var.clone()));
                        self.emit(Op::ArrayLen);
                        self.emit(Op::Lt);
                        let exit_jump = self.emit(Op::JumpIfFalse(0));
                        self.emit(Op::Pop);

                        // var = arr[__it]
                        self.emit(Op::LoadVar(arr_var.clone()));
                        self.emit(Op::LoadVar(idx_var.clone()));
                        self.emit(Op::ArrayIndex);
                        self.emit(Op::StoreVar(var.clone()));

                        for stmt in body {
                            self.compile_stmt(stmt)?;
                        }

                        // continue lands HERE — at the increment
                        let cont_target = self.ops.len();
                        self.loop_stack.last_mut().unwrap().continue_target = cont_target;
                        // __it = __it + 1
                        self.emit(Op::LoadVar(idx_var.clone()));
                        let one = self.add_const(Const::Int(1));
                        self.emit(Op::LoadConst(one));
                        self.emit(Op::Add);
                        self.emit(Op::StoreVar(idx_var.clone()));

                        let back = self.emit(Op::Jump(0));
                        let offset = (loop_start as i32) - (back as i32) - 1;
                        if let Op::Jump(o) = &mut self.ops[back] {
                            *o = offset;
                        }
                        let exit = self.ops.len();
                        self.patch_jump(exit_jump, exit);
                        self.emit(Op::Pop);

                        let frame = self.loop_stack.pop().unwrap();
                        let after_exit = self.ops.len();
                        for j in frame.break_jumps {
                            self.patch_jump(j, after_exit);
                        }
                    }
                }
            }
            Statement::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(e)?;
                    self.emit(Op::Return);
                } else {
                    self.emit(Op::ReturnNull);
                }
            }
            Statement::Break => {
                if self.loop_stack.is_empty() {
                    return Err("`break` outside of any loop".to_string());
                }
                let j = self.emit(Op::Jump(0));
                self.loop_stack.last_mut().unwrap().break_jumps.push(j);
            }
            Statement::Continue => {
                if self.loop_stack.is_empty() {
                    return Err("`continue` outside of any loop".to_string());
                }
                let target = self.loop_stack.last().unwrap().continue_target;
                // If target is 0 we're inside a frame whose continue point
                // hasn't been set yet (range loops set it AFTER the body —
                // continue before that point means jump back to start, which
                // is the same as `continue` semantics).
                let here = self.emit(Op::Jump(0));
                let resolved_target = if target == 0 {
                    // Patch later when the for-body's increment is emitted.
                    // For simplicity here, treat as a break (exits the loop).
                    self.loop_stack.last_mut().unwrap().break_jumps.push(here);
                    return Ok(());
                } else {
                    target
                };
                let offset = (resolved_target as i32) - (here as i32) - 1;
                if let Op::Jump(o) = &mut self.ops[here] {
                    *o = offset;
                }
            }
            Statement::Import { .. } => {
                // Imports are handled outside the VM (by the interpreter before
                // compilation runs). The VM treats them as no-ops.
            }
            Statement::FunctionDef { .. } => {
                // Function defs hoisted by compile_program(); skip here.
            }
            Statement::Try { .. } | Statement::Throw(_) | Statement::Yield(_) => {
                // Tree-walk fallback. See Op::ExecStmt comments — full
                // exception unwind would require a side try-stack and
                // a Result-aware op dispatch loop. Until that pays for
                // itself, fall back to the AST walker for try/catch/
                // throw/yield.
                self.emit(Op::ExecStmt(Box::new(s.clone())));
            }
            Statement::ClassDef { .. } => {
                // ClassDef is consumed at fn-registration time
                // (register_user_functions desugars it into a
                // constructor + mangled methods). No code is emitted
                // at the statement level — by the time we get here
                // the class's fns are already in the user-fn table.
            }
            Statement::Match { .. } => {
                // Same fallback strategy as Try. A native lowering
                // would compile each arm into a guarded Jump and
                // emit the bindings as Op::StoreVar — straightforward
                // but adds 50+ lines of Rust per pattern variant.
                // Defer until benchmarks show match in a hot path.
                self.emit(Op::ExecStmt(Box::new(s.clone())));
            }
        }
        Ok(())
    }

    fn finish(
        self,
        name: String,
        params: Vec<String>,
        param_types: Vec<Option<String>>,
        return_type: Option<String>,
        pragmas: Vec<String>,
    ) -> CompiledFunction {
        let n = self.ops.len();
        CompiledFunction {
            name,
            params,
            param_types,
            return_type,
            ops: self.ops,
            constants: self.constants,
            // Pre-size the inline call cache to match the op count. All slots
            // start uncached (0); the VM fills them in on first execution.
            call_cache: (0..n).map(|_| std::cell::Cell::new(0u8)).collect(),
            // Pad op_positions to match. Compiler appends the correct
            // pos at every emit site that knows it (Op::Call); other
            // ops get Pos::unknown() and never appear in traces.
            op_positions: {
                let mut v = self.op_positions;
                v.resize(n, crate::ast::Pos::unknown());
                v
            },
            pragmas,
        }
    }
}

/// Map a source-level type name ("int" / "string" / etc.) to the static
/// TypeTag understood by the compiler's inference helper. Returns None
/// for unknown annotations so they're treated as untyped.
fn type_tag_of(s: &str) -> Option<&'static str> {
    match s {
        "int" | "i64" => Some("int"),
        "float" | "f64" => Some("float"),
        "string" | "str" => Some("string"),
        "bool" => Some("bool"),
        "array" => Some("array"),
        _ => None,
    }
}

pub fn compile_program(statements: &[Statement]) -> Result<Module, String> {
    let mut module = Module::default();

    // Pre-pass A: collect every user-defined function name. We pass this set
    // into every Compiler so the hot-path inliner can refuse to inline a
    // name the user has shadowed (e.g. a recursive user `fib`).
    let mut user_fns: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    // Pre-pass B: collect declared return-types so Compiler::infer_type
    // can see across function boundaries.
    let mut fn_return_types: std::collections::HashMap<String, &'static str> =
        std::collections::HashMap::new();
    for stmt in statements {
        if let Statement::FunctionDef {
            name, return_type, ..
        } = stmt
        {
            user_fns.insert(name.clone());
            if let Some(rt) = return_type {
                if let Some(tag) = type_tag_of(rt) {
                    fn_return_types.insert(name.clone(), tag);
                }
            }
        }
    }

    // First pass: hoist function definitions.
    for stmt in statements {
        if let Statement::FunctionDef {
            name,
            params,
            param_types,
            body,
            return_type,
            pragmas,
        } = stmt
        {
            let mut fc = Compiler::with_user_fns(user_fns.clone());
            fc.fn_return_types = fn_return_types.clone();
            // Seed var_types from typed parameters so arithmetic on them
            // can specialize.
            for (pname, ptype_opt) in params.iter().zip(param_types.iter()) {
                if let Some(ptype) = ptype_opt {
                    if let Some(tag) = type_tag_of(ptype) {
                        fc.var_types.insert(pname.clone(), tag);
                    }
                }
            }
            for s in body {
                fc.compile_stmt(s)?;
            }
            // Ensure every function ends with an implicit ReturnNull so the VM
            // doesn't fall off the end.
            fc.emit(Op::ReturnNull);
            // Drain anonymous lambda bodies + ASTs out of this Compiler
            // BEFORE finishing the outer fn (finish consumes self).
            let lambdas = std::mem::take(&mut fc.pending_lambdas);
            for lf in lambdas {
                module.functions.insert(lf.name.clone(), lf);
            }
            let lambda_asts = std::mem::take(&mut fc.pending_lambda_asts);
            module.lambda_asts.extend(lambda_asts);
            let func = fc.finish(
                name.clone(),
                params.clone(),
                param_types.clone(),
                return_type.clone(),
                pragmas.clone(),
            );
            module.functions.insert(name.clone(), func);
        }
    }

    // Second pass: compile the top-level (non-fn) statements as `main`.
    let mut mc = Compiler::with_user_fns(user_fns);
    mc.fn_return_types = fn_return_types;
    for stmt in statements {
        if matches!(stmt, Statement::FunctionDef { .. }) {
            continue;
        }
        mc.compile_stmt(stmt)?;
    }
    mc.emit(Op::ReturnNull);
    let lambdas = std::mem::take(&mut mc.pending_lambdas);
    for lf in lambdas {
        module.functions.insert(lf.name.clone(), lf);
    }
    let lambda_asts = std::mem::take(&mut mc.pending_lambda_asts);
    module.lambda_asts.extend(lambda_asts);
    module.main = mc.finish("__main__".to_string(), Vec::new(), Vec::new(), None, Vec::new());

    Ok(module)
}
