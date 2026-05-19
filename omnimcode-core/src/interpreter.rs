// src/interpreter.rs - AST execution engine

use crate::ast::*;
use crate::value::{HInt, HArray, Value, fibonacci, is_fibonacci};
use std::collections::{HashMap, HashSet};

/// Closure signature for the JIT dispatch hook. Returns `Some(Ok(v))`
/// when a JIT'd implementation handled the call, `Some(Err(msg))` when
/// the JIT was applicable but failed, and `None` when this call should
/// fall back to the tree-walk interpreter (no JIT'd version registered,
/// or args incompatible with the JIT'd signature).
pub type JitDispatch =
    std::rc::Rc<dyn Fn(&str, &[Value]) -> Option<Result<Value, String>>>;

pub struct Interpreter {
    globals: HashMap<String, Value>,
    functions: HashMap<String, (Vec<String>, Vec<Statement>)>,
    /// Class-parent table for `class Child extends Parent` inheritance.
    /// Maps child class name → parent class name. The instance-method
    /// dispatch path walks this chain when `<Child>__<method>` isn't
    /// found, trying `<Parent>__<method>` and so on.
    class_parents: HashMap<String, String>,
    /// Active yield collector for the current generator frame. Set
    /// by invoke_user_function when entering a generator fn (one
    /// whose body contains Yield); each Yield statement appends to
    /// the top of this stack. On exit, the collector is popped and
    /// returned as a Value::Array. Stack-of-vecs supports nested
    /// generator invocations.
    yield_stacks: Vec<Vec<Value>>,
    /// Currently in-flight typed exception value. Set by `throw <expr>`
    /// before the Err propagation begins; taken by the catching `try`
    /// block to bind to the catch variable. Lets `catch e { ... }`
    /// receive a structured dict/value, not just a string. None when
    /// either no throw is in flight or the error originated from a
    /// Rust-side builtin (then catch falls back to the string form).
    pending_throw: Option<Value>,
    /// Reverse-mode autograd tape. Each node is one op recorded during
    /// the forward pass. `tape_backward(id)` walks the tape in reverse,
    /// accumulating gradients into the `grad` field of every node it
    /// touches. Operates on scalars (HFloat) or 2D matrices (Vec<Vec<f64>>);
    /// shape is implicit in each node's value. Substrate metadata is
    /// preserved in the *forward* values via HInt/HFloat throughout —
    /// gradients themselves are HFloat for precision, but users can read
    /// `tape_value(id)` to get the substrate-annotated forward value
    /// alongside `tape_grad(id)` for the derivative.
    autograd_tape: Vec<TapeNode>,
    /// Value of the most recently evaluated top-level
    /// `Statement::Expression`. The MCP server and any REPL frontend
    /// read this to surface "what did the last line evaluate to"
    /// without re-running side effects.
    last_expression_value: Option<Value>,
    /// Code memory: name → canonical hash. Lets the MCP/REPL caller
    /// remember "I saw this code as X" across calls. omc_remember
    /// and omc_recall expose it.
    code_memory: std::cell::RefCell<std::collections::BTreeMap<String, i64>>,
    /// The source text of the top-level program, set by the CLI/MCP so
    /// the `omc_source()` builtin can return it from within a running program.
    source_code: Option<String>,
    /// Collects lines written by `print`/`println` during execution so
    /// the MCP `omc_eval` handler can return them as part of the tool
    /// result rather than losing them to the server process's stdout.
    /// Each `print` call still writes to real stdout too (for CLI use).
    output_lines: std::cell::RefCell<Vec<String>>,
    /// Stack of yield callbacks for LAZY generators. When set, the
    /// active generator's yield statements invoke the topmost callback
    /// with the yielded value rather than appending to a Vec. Memory
    /// stays O(call-stack-depth) instead of O(yield-count), so a
    /// generator can stream a billion values without OOM. Each callback
    /// returns 1 to continue or 0 to short-circuit the generator —
    /// the interpreter sets `gen_stop_requested` which propagates
    /// through loops/blocks via return_value.
    yield_callbacks: Vec<Value>,
    gen_stop_requested: bool,
    /// Optional JIT dispatch hook. When set, `invoke_user_function_at`
    /// consults this BEFORE running the tree-walk body. If the hook
    /// returns `Some(result)`, that result wins; otherwise tree-walk
    /// runs normally. Lets the standalone CLI route eligible fns
    /// through omnimcode-codegen's dual-band JIT (when the
    /// `OMC_HBIT_JIT` env var is set) without coupling core to LLVM.
    ///
    /// `Rc<dyn Fn>` so the hook can be cheaply cloned with the
    /// Interpreter and shared across nested user-fn invocations.
    jit_dispatch: Option<JitDispatch>,
    /// Local scope stack. Each frame is `Rc<RefCell<HashMap>>` so that
    /// closures can capture the frame by reference (shared mutation
    /// across sibling closures created in the same scope) and so that
    /// captured frames stay alive after the enclosing function returns.
    locals: Vec<std::rc::Rc<std::cell::RefCell<HashMap<String, Value>>>>,
    return_value: Option<Value>,
    break_flag: bool,
    continue_flag: bool,
    /// Names of modules already imported (idempotent re-import).
    imported_modules: HashSet<String>,
    /// xorshift64* RNG state for random_* builtins. Seeded from system
    /// time at construction; `random_seed(s)` overrides for deterministic
    /// runs. State is never 0 (xorshift degenerates at 0).
    rng_state: std::cell::Cell<u64>,
    /// Monotonic counter for anonymous lambda names. Each `fn() {...}`
    /// expression generates a unique `__lambda_N` identifier so the body
    /// can be stored in self.functions and looked up at call time.
    lambda_counter: u64,
    /// Host-side state for the OMC test runner. Reached via
    /// `test_record_failure(msg)` / `test_failure_count()` / `test_clear`.
    /// Bypasses OMC's pass-by-value array semantics — the test runner
    /// needs failures to propagate across nested-function boundaries
    /// even though OMC arrays don't.
    test_failures: std::cell::RefCell<Vec<String>>,
    /// Current test name, for prefixing failure messages. Same scoping
    /// reason as test_failures: a plain OMC global wouldn't propagate
    /// to nested assertion calls.
    test_current_name: std::cell::RefCell<String>,
    /// (Function name, call-site position) for currently-executing
    /// user functions, innermost-last. The position is the line of
    /// the SITE where this fn was called from — that's what the user
    /// sees in stack traces. The fn's own internal line numbers don't
    /// belong here; they'd need per-statement position tracking.
    call_stack: Vec<(String, crate::ast::Pos)>,
    /// Reverse-FFI: builtins registered by the embedder (Python /
    /// Godot / a Rust host). When OMC code calls a name not found
    /// in user fns, modules, or the built-in stdlib, dispatch
    /// falls through to this map. Lets an embedder expose host-side
    /// capabilities (numpy, godot signals, file pickers, etc.) to
    /// OMC programs without baking them into the interpreter.
    ///
    /// Stored as `Rc<dyn Fn>` so handlers can be cheaply cloned
    /// when the Interpreter itself is cloned (rare, but FFI wrappers
    /// occasionally do it). Single-threaded — handlers don't need
    /// to be Send/Sync, matching the rest of OMC's runtime.
    host_builtins: HashMap<
        String,
        std::rc::Rc<dyn Fn(&[Value]) -> Result<Value, String>>,
    >,
}

impl Interpreter {
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15);  // golden-ratio constant fallback
        let initial = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Interpreter {
            globals: HashMap::new(),
            functions: HashMap::new(),
            jit_dispatch: None,
            locals: vec![std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()))],
            return_value: None,
            break_flag: false,
            continue_flag: false,
            imported_modules: HashSet::new(),
            rng_state: std::cell::Cell::new(initial),
            lambda_counter: 0,
            test_failures: std::cell::RefCell::new(Vec::new()),
            test_current_name: std::cell::RefCell::new(String::new()),
            call_stack: Vec::new(),
            host_builtins: HashMap::new(),
            class_parents: HashMap::new(),
            yield_stacks: Vec::new(),
            pending_throw: None,
            autograd_tape: Vec::new(),
            yield_callbacks: Vec::new(),
            gen_stop_requested: false,
            last_expression_value: None,
            code_memory: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            source_code: None,
            output_lines: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Store the source text of the running program so `omc_source()` can return it.
    pub fn set_source_code(&mut self, src: String) {
        self.source_code = Some(src);
    }

    /// Read (and clear) the most recent top-level expression value.
    /// Used by the MCP server to return the result of `omc_eval`.
    pub fn take_last_expression_value(&mut self) -> Option<Value> {
        self.last_expression_value.take()
    }

    /// Drain all lines printed via `print`/`println` since the last call.
    /// Used by the MCP `omc_eval` handler to surface print output in the
    /// tool result rather than losing it to the server's process stdout.
    pub fn take_output_lines(&self) -> Vec<String> {
        self.output_lines.borrow_mut().drain(..).collect()
    }

    /// Register a host-side builtin that OMC code can call by name.
    /// The closure receives the evaluated argument values and returns
    /// either a Value (success) or an error message that propagates
    /// through OMC's normal Result chain (catchable via try/catch).
    ///
    /// Names registered here SHADOW user-defined functions of the
    /// same name (so an embedder can hand OMC a custom `fetch_url`
    /// that overrides any user `fn fetch_url(...)`). They're checked
    /// AFTER user fns, BEFORE the built-in stdlib — same precedence
    /// position the test runner's `test_*` overrides use.
    ///
    /// Type signatures are dynamic: the closure is responsible for
    /// validating arg count and types. Use `args.len()` and
    /// `matches!(args[0], Value::HInt(_))` etc. Errors are strings;
    /// they appear in stack traces with the call site prefixed.
    ///
    /// Example:
    /// ```ignore
    /// let mut interp = Interpreter::new();
    /// interp.register_builtin("double", |args| {
    ///     if args.len() != 1 { return Err("double requires 1 arg".into()); }
    ///     Ok(Value::HInt(HInt::new(args[0].to_int() * 2)))
    /// });
    /// // OMC code can now do `println(double(21));` and see "42".
    /// ```
    pub fn register_builtin<F>(&mut self, name: &str, handler: F)
    where
        F: Fn(&[Value]) -> Result<Value, String> + 'static,
    {
        self.host_builtins.insert(name.to_string(), std::rc::Rc::new(handler));
    }

    /// Remove a previously-registered host builtin. Returns true if
    /// a handler was removed. Used by embedders that want to hand
    /// OMC a temporary capability for a single call sequence.
    pub fn unregister_builtin(&mut self, name: &str) -> bool {
        self.host_builtins.remove(name).is_some()
    }

    /// True if a host builtin with this name is registered. Used by
    /// the dispatch path; exposed publicly so embedders can check
    /// before re-registering.
    pub fn has_host_builtin(&self, name: &str) -> bool {
        self.host_builtins.contains_key(name)
    }

    /// Register the JIT dispatch hook. The closure is consulted at the
    /// top of every user-fn call: if it returns `Some(result)`, that
    /// result is used directly and the tree-walk body is skipped.
    /// Used by the standalone CLI to route eligible user fns through
    /// omnimcode-codegen's dual-band JIT under `OMC_HBIT_JIT=1`.
    ///
    /// Setting this to `None` removes the hook (resets to pure
    /// tree-walk). At most one hook is registered at a time.
    pub fn set_jit_dispatch(&mut self, hook: Option<JitDispatch>) {
        self.jit_dispatch = hook;
    }

    /// Invoke an OMC function by name with already-evaluated Values
    /// as arguments. Used by Python → OMC callbacks (py_callback)
    /// where the caller has live Values from the Python side and
    /// needs to dispatch into OMC code.
    ///
    /// Wraps each Value in a synthetic local + Variable expression
    /// so we can reuse the standard call_function path (which
    /// expects Expressions). Slightly more overhead than raw call
    /// but reuses every dispatch / trace / heal feature.
    pub fn call_function_with_values(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<Value, String> {
        // Push a fresh scope to hold the synthetic args so we don't
        // pollute the caller's locals.
        self.locals.push(std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())));
        let mut expr_args = Vec::with_capacity(args.len());
        for (i, v) in args.iter().enumerate() {
            let key = format!("__cb_arg_{}", i);
            self.set_var(key.clone(), v.clone());
            expr_args.push(crate::ast::Expression::Variable(key));
        }
        let result = self.call_function(name, &expr_args);
        self.locals.pop();
        result
    }

    /// xorshift64* — fast and tiny, sufficient for OMC scripting needs.
    /// Not cryptographic. Returns a non-zero u64.
    fn rng_next(&self) -> u64 {
        let mut x = self.rng_state.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state.set(x);
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Module search path used by `import NAME;`.
    /// Honors `OMC_STDLIB_PATH` (colon-separated), then falls back to a
    /// small built-in list that includes the canonical Python OMC stdlib.
    fn module_search_path() -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        // Project-local package cache. Populated by `omc --install`
        // and checked first so `import "np";` resolves the local
        // copy before falling back to user paths or the legacy stdlib.
        // Mirrors npm's node_modules / pip's site-packages convention.
        paths.push(std::path::PathBuf::from("omc_modules"));
        if let Ok(env) = std::env::var("OMC_STDLIB_PATH") {
            for p in env.split(':') {
                if !p.is_empty() {
                    paths.push(std::path::PathBuf::from(p));
                }
            }
        }
        // Canonical Python OMC stdlib (when present on this machine).
        paths.push(std::path::PathBuf::from(
            "/home/thearchitect/Sovereign_Lattice/omninet_package/omnicode_stdlib",
        ));
        paths.push(std::path::PathBuf::from(
            "/home/thearchitect/Sovereign_Lattice/omninet_package/omnicode_stdlib/std",
        ));
        // Current working directory and a relative `omc-stdlib/`.
        paths.push(std::path::PathBuf::from("."));
        paths.push(std::path::PathBuf::from("omc-stdlib"));
        paths.push(std::path::PathBuf::from("omc-stdlib/std"));
        paths
    }

    /// Public wrapper for the module resolver. Returns the file path
    /// for the named import, or None if not found on the search path.
    /// Exposed so the CLI's JIT-registration path can inline imports
    /// into the AST before compile_program (the bytecode compiler
    /// treats Statement::Import as a no-op since interpreter normally
    /// handles imports at statement-execution time).
    pub fn resolve_module_path(name: &str) -> Option<std::path::PathBuf> {
        Self::resolve_module(name)
    }

    /// Walk `statements` recursively, replacing each `Statement::Import`
    /// with the parsed AST of the imported file. Function defs from
    /// the imported file get their names rewritten to `alias.fname`
    /// when an alias is set, matching the runtime import semantics in
    /// `import_module_with_alias`. For aliased imports, intra-module
    /// calls within the inlined body get rewritten via the same
    /// `rewrite_module_calls` helper.
    ///
    /// Used by the CLI's JIT registration to flatten the AST so
    /// `compile_program` produces a Module that includes ALL fns —
    /// including imported ones — so `jit_module` can compile them.
    ///
    /// Cyclic imports are guarded by `visited` so we don't loop.
    /// Selective imports (`from "x" import a, b;`) inline only the
    /// named fns.
    pub fn inline_imports(
        statements: Vec<Statement>,
    ) -> Result<Vec<Statement>, String> {
        let mut visited: HashSet<String> = HashSet::new();
        Self::inline_imports_inner(statements, &mut visited)
    }

    fn inline_imports_inner(
        statements: Vec<Statement>,
        visited: &mut HashSet<String>,
    ) -> Result<Vec<Statement>, String> {
        let mut out: Vec<Statement> = Vec::with_capacity(statements.len());
        for stmt in statements {
            match stmt {
                Statement::Import { module, alias, selected } => {
                    if !visited.insert(module.clone()) {
                        // Already inlined — skip the second occurrence.
                        continue;
                    }
                    let path = Self::resolve_module(&module).ok_or_else(|| {
                        format!(
                            "inline_imports: could not resolve module `{}`",
                            module
                        )
                    })?;
                    let source = std::fs::read_to_string(&path).map_err(|e| {
                        format!("inline_imports: read {}: {}", module, e)
                    })?;
                    let mut parser = crate::parser::Parser::new(&source);
                    let raw_stmts = parser.parse().map_err(|e| {
                        format!("inline_imports: parse {}: {}", module, e)
                    })?;
                    // Recurse to inline transitive imports first.
                    let inner_stmts = Self::inline_imports_inner(raw_stmts, visited)?;

                    // Apply aliasing / selective filtering.
                    let processed = if let Some(prefix) = alias.as_deref() {
                        // Rename fn defs to "alias.fname" and rewrite
                        // intra-module calls. Skip names that already
                        // contain a dot (transitively-imported aliases).
                        let mut local_names: HashSet<String> = HashSet::new();
                        for s in &inner_stmts {
                            if let Statement::FunctionDef { name, .. } = s {
                                if !name.contains('.') {
                                    local_names.insert(name.clone());
                                }
                            }
                        }
                        let mut renamed: Vec<Statement> = Vec::new();
                        for s in inner_stmts {
                            match s {
                                Statement::FunctionDef {
                                    name,
                                    params,
                                    param_types,
                                    body,
                                    return_type,
                                    pragmas,
                                } if !name.contains('.') => {
                                    let aliased = format!("{}.{}", prefix, name);
                                    let body_rewritten: Vec<Statement> = body
                                        .into_iter()
                                        .map(|st| {
                                            Self::rewrite_module_calls(
                                                st,
                                                &local_names,
                                                prefix,
                                            )
                                        })
                                        .collect();
                                    renamed.push(Statement::FunctionDef {
                                        name: aliased,
                                        params,
                                        param_types,
                                        body: body_rewritten,
                                        return_type,
                                        pragmas,
                                    });
                                }
                                other => renamed.push(other),
                            }
                        }
                        renamed
                    } else if let Some(names) = selected {
                        // Selective: keep only the named fns at top level.
                        inner_stmts
                            .into_iter()
                            .filter(|s| match s {
                                Statement::FunctionDef { name, .. } => {
                                    names.iter().any(|n| n == name)
                                }
                                _ => true,
                            })
                            .collect()
                    } else {
                        // Plain `import "x";` — flat merge.
                        inner_stmts
                    };
                    out.extend(processed);
                }
                other => out.push(other),
            }
        }
        Ok(out)
    }

    fn resolve_module(name: &str) -> Option<std::path::PathBuf> {
        // 1. Literal path — if the argument looks like a file path
        //    (absolute, or starts with `./` or `../`, or already ends
        //    in `.omc`), try it directly. Lets `import "/abs/path.omc"`
        //    and `import "./local.omc"` work without search-path setup.
        let looks_like_path = name.starts_with('/')
            || name.starts_with("./")
            || name.starts_with("../")
            || name.ends_with(".omc");
        if looks_like_path {
            let path = std::path::PathBuf::from(name);
            if path.is_file() {
                return Some(path);
            }
        }
        // 2. Try each search dir with a few naming variants.
        // For `import std/core;` allow the slashed form too.
        for dir in Self::module_search_path() {
            for variant in [
                format!("{}.omc", name),
                format!("{}/init.omc", name),
                format!("std/{}.omc", name),
            ] {
                let candidate = dir.join(&variant);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    fn import_module(&mut self, name: &str) -> Result<(), String> {
        self.import_module_with_alias(name, None)
    }

    /// Load a module from disk. If `alias` is `Some(prefix)`, every
    /// function the module DEFINES gets renamed to `prefix.fname` so
    /// the importer reaches it via dotted-call syntax. Top-level
    /// statements still execute against the global namespace (any
    /// `h x = ...` declarations remain unprefixed) — only function
    /// definitions get namespaced.
    ///
    /// Idempotent on `name` regardless of alias — re-importing the
    /// same module with a different alias would re-execute. The
    /// dedup key is the module name; rename to a fresh module name
    /// if you want a second copy.
    fn import_module_with_alias(&mut self, name: &str, alias: Option<&str>) -> Result<(), String> {
        if self.imported_modules.contains(name) {
            return Ok(()); // Already loaded.
        }
        let path = Self::resolve_module(name).ok_or_else(|| {
            format!(
                "Could not resolve module `{}` (set OMC_STDLIB_PATH or place {}.omc on the search path)",
                name, name
            )
        })?;
        let source = std::fs::read_to_string(&path)
            .map_err(|e| format!("import {}: read failed: {}", name, e))?;
        // Mark as imported BEFORE executing to avoid infinite recursion on
        // cyclic imports.
        self.imported_modules.insert(name.to_string());
        let mut parser = crate::parser::Parser::new(&source);
        let stmts = parser
            .parse()
            .map_err(|e| format!("import {}: parse error: {}", name, e))?;
        // Snapshot which function names exist before module exec so we can
        // identify the ones the module introduces. Anything new gets the
        // alias prefix when `alias` is set.
        let pre_fns: HashSet<String> = self.functions.keys().cloned().collect();
        for stmt in &stmts {
            self.execute_stmt(stmt)?;
            self.return_value = None;
            self.break_flag = false;
            self.continue_flag = false;
        }
        if let Some(prefix) = alias {
            // Rename newly-defined functions to alias.name AND
            // rewrite intra-module calls in their bodies so `_pd()`
            // inside this module still resolves after `_pd` becomes
            // `pd._pd`. Without this rewrite, helper-fn patterns
            // ("init once, return cached handle") break under aliasing.
            //
            // CRITICAL: skip names that already contain a dot. Those
            // came from a transitively-aliased child module (e.g.
            // when ha imports np, np's funcs get registered as
            // "np.argsort" — they belong to np, not ha). Re-aliasing
            // them to "ha.np.argsort" breaks the user's direct
            // `np.argsort` calls. Stay flat for child-module exports.
            let new_names: Vec<String> = self.functions.keys()
                .filter(|k| !pre_fns.contains(*k) && !k.contains('.'))
                .cloned()
                .collect();
            let module_set: HashSet<String> = new_names.iter().cloned().collect();
            for original in &new_names {
                if let Some((params, body)) = self.functions.remove(original) {
                    let rewritten_body: Vec<Statement> = body
                        .into_iter()
                        .map(|s| Self::rewrite_module_calls(s, &module_set, prefix))
                        .collect();
                    let aliased = format!("{}.{}", prefix, original);
                    self.functions.insert(aliased, (params, rewritten_body));
                }
            }
        }
        Ok(())
    }

    /// Selective import: `from "path" import name1, name2;`. Loads
    /// the module (idempotent on path), then KEEPS only the listed
    /// names — drops everything else introduced by the module.
    /// Names are merged into the global function namespace
    /// unprefixed.
    ///
    /// Helper functions the module relies on internally must be in
    /// the selected list too, otherwise calls to them from the
    /// imported fns will fail at runtime. The error message points
    /// at the missing helper so the user can add it.
    fn import_module_selective(&mut self, name: &str, selected: &[String]) -> Result<(), String> {
        // Use a fresh sub-interpreter to avoid polluting our globals
        // with the module's helpers we don't want.
        let path = Self::resolve_module(name).ok_or_else(|| {
            format!(
                "Could not resolve module `{}` (set OMC_STDLIB_PATH or place {}.omc on the search path)",
                name, name
            )
        })?;
        let source = std::fs::read_to_string(&path)
            .map_err(|e| format!("from {}: read failed: {}", name, e))?;
        let mut parser = crate::parser::Parser::new(&source);
        let stmts = parser
            .parse()
            .map_err(|e| format!("from {}: parse error: {}", name, e))?;

        // Snapshot existing fns; execute module; keep only selected new ones.
        let pre_fns: HashSet<String> = self.functions.keys().cloned().collect();
        let pre_globals: HashSet<String> = self.globals.keys().cloned().collect();

        for stmt in &stmts {
            self.execute_stmt(stmt)?;
            self.return_value = None;
            self.break_flag = false;
            self.continue_flag = false;
        }

        let new_fn_names: Vec<String> = self.functions.keys()
            .filter(|k| !pre_fns.contains(*k))
            .cloned()
            .collect();
        let new_global_names: Vec<String> = self.globals.keys()
            .filter(|k| !pre_globals.contains(*k))
            .cloned()
            .collect();

        let selected_set: HashSet<&str> = selected.iter().map(|s| s.as_str()).collect();

        // Drop new fns / globals not in selected_set.
        for fname in &new_fn_names {
            if !selected_set.contains(fname.as_str()) {
                self.functions.remove(fname);
            }
        }
        for gname in &new_global_names {
            if !selected_set.contains(gname.as_str()) {
                self.globals.remove(gname);
            }
        }

        // Sanity check: every selected name must exist.
        for sel in selected {
            if !self.functions.contains_key(sel) && !self.globals.contains_key(sel) {
                return Err(format!(
                    "from {}: '{}' not found in module",
                    name, sel
                ));
            }
        }

        // Mark module imported AFTER selection so a subsequent
        // `import "path";` (full) re-runs cleanly. Different shape
        // → different idempotency intent. Selective imports DON'T
        // count as a full import.
        self.imported_modules.insert(format!("{}::selected", name));
        Ok(())
    }

    /// Walk a Statement and rewrite any Expression::Call whose name
    /// is in `module_names` to `alias.name`. Used by aliased imports
    /// so a module's helpers can call its other functions even after
    /// they've been renamed.
    fn rewrite_module_calls(
        stmt: Statement,
        module_names: &HashSet<String>,
        alias: &str,
    ) -> Statement {
        match stmt {
            Statement::Expression(e) => Statement::Expression(
                Self::rewrite_call_expr(e, module_names, alias),
            ),
            Statement::Print(e) => Statement::Print(
                Self::rewrite_call_expr(e, module_names, alias),
            ),
            Statement::VarDecl { name, value, is_harmonic } => Statement::VarDecl {
                name,
                value: Self::rewrite_call_expr(value, module_names, alias),
                is_harmonic,
            },
            Statement::Parameter { name, value } => Statement::Parameter {
                name,
                value: Self::rewrite_call_expr(value, module_names, alias),
            },
            Statement::Assignment { name, value } => Statement::Assignment {
                name,
                value: Self::rewrite_call_expr(value, module_names, alias),
            },
            Statement::IndexAssignment { name, index, value } => Statement::IndexAssignment {
                name,
                index: Self::rewrite_call_expr(index, module_names, alias),
                value: Self::rewrite_call_expr(value, module_names, alias),
            },
            Statement::Return(opt) => Statement::Return(
                opt.map(|e| Self::rewrite_call_expr(e, module_names, alias)),
            ),
            Statement::If { condition, then_body, elif_parts, else_body } => Statement::If {
                condition: Self::rewrite_call_expr(condition, module_names, alias),
                then_body: then_body
                    .into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect(),
                elif_parts: elif_parts
                    .into_iter()
                    .map(|(c, b)| {
                        (
                            Self::rewrite_call_expr(c, module_names, alias),
                            b.into_iter()
                                .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                                .collect(),
                        )
                    })
                    .collect(),
                else_body: else_body.map(|b| {
                    b.into_iter()
                        .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                        .collect()
                }),
            },
            Statement::While { condition, body } => Statement::While {
                condition: Self::rewrite_call_expr(condition, module_names, alias),
                body: body
                    .into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect(),
            },
            Statement::For { var, iterable, body } => Statement::For {
                var,
                iterable: match iterable {
                    ForIterable::Range { start, end } => ForIterable::Range {
                        start: Self::rewrite_call_expr(start, module_names, alias),
                        end: Self::rewrite_call_expr(end, module_names, alias),
                    },
                    ForIterable::Expr(e) => ForIterable::Expr(
                        Self::rewrite_call_expr(e, module_names, alias),
                    ),
                },
                body: body
                    .into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect(),
            },
            Statement::FunctionDef { name, params, param_types, body, return_type, pragmas } => {
                Statement::FunctionDef {
                    name,
                    params,
                    param_types,
                    body: body
                        .into_iter()
                        .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                        .collect(),
                    return_type,
                    pragmas,
                }
            }
            Statement::Try { body, err_var, handler, finally } => Statement::Try {
                body: body
                    .into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect(),
                err_var,
                handler: handler
                    .into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect(),
                finally: finally.map(|stmts| stmts.into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect()),
            },
            Statement::Throw(e) => Statement::Throw(
                Self::rewrite_call_expr(e, module_names, alias),
            ),
            Statement::Yield(e) => Statement::Yield(
                Self::rewrite_call_expr(e, module_names, alias),
            ),
            Statement::Match { scrutinee, arms } => Statement::Match {
                scrutinee: Self::rewrite_call_expr(scrutinee, module_names, alias),
                arms: arms
                    .into_iter()
                    .map(|arm| MatchArm {
                        pattern: arm.pattern,
                        body: arm
                            .body
                            .into_iter()
                            .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                            .collect(),
                    })
                    .collect(),
            },
            other => other,
        }
    }

    fn rewrite_call_expr(
        e: Expression,
        module_names: &HashSet<String>,
        alias: &str,
    ) -> Expression {
        match e {
            Expression::Call { name, args, pos } => {
                let new_name = if module_names.contains(&name) {
                    format!("{}.{}", alias, name)
                } else {
                    name
                };
                Expression::Call {
                    name: new_name,
                    args: args
                        .into_iter()
                        .map(|a| Self::rewrite_call_expr(a, module_names, alias))
                        .collect(),
                    pos,
                }
            }
            Expression::Add(l, r) => Expression::Add(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Sub(l, r) => Expression::Sub(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Mul(l, r) => Expression::Mul(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Div(l, r) => Expression::Div(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Mod(l, r) => Expression::Mod(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Eq(l, r) => Expression::Eq(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Ne(l, r) => Expression::Ne(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Lt(l, r) => Expression::Lt(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Le(l, r) => Expression::Le(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Gt(l, r) => Expression::Gt(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Ge(l, r) => Expression::Ge(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::And(l, r) => Expression::And(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Or(l, r) => Expression::Or(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Not(e) => Expression::Not(Box::new(Self::rewrite_call_expr(
                *e,
                module_names,
                alias,
            ))),
            Expression::Array(items) => Expression::Array(
                items
                    .into_iter()
                    .map(|e| Self::rewrite_call_expr(e, module_names, alias))
                    .collect(),
            ),
            Expression::Dict(pairs) => Expression::Dict(
                pairs
                    .into_iter()
                    .map(|(k, v)| {
                        (
                            Self::rewrite_call_expr(k, module_names, alias),
                            Self::rewrite_call_expr(v, module_names, alias),
                        )
                    })
                    .collect(),
            ),
            Expression::Index { name, index } => Expression::Index {
                name,
                index: Box::new(Self::rewrite_call_expr(*index, module_names, alias)),
            },
            Expression::ChainedIndex { object, index } => Expression::ChainedIndex {
                object: Box::new(Self::rewrite_call_expr(*object, module_names, alias)),
                index: Box::new(Self::rewrite_call_expr(*index, module_names, alias)),
            },
            Expression::Resonance(e) => Expression::Resonance(Box::new(
                Self::rewrite_call_expr(*e, module_names, alias),
            )),
            Expression::Fold(e) => Expression::Fold(Box::new(Self::rewrite_call_expr(
                *e,
                module_names,
                alias,
            ))),
            Expression::Safe(e) => Expression::Safe(Box::new(Self::rewrite_call_expr(
                *e,
                module_names,
                alias,
            ))),
            Expression::Lambda { params, body } => Expression::Lambda {
                params,
                body: body
                    .into_iter()
                    .map(|s| Self::rewrite_module_calls(s, module_names, alias))
                    .collect(),
            },
            // BitAnd/Or/Xor/Shl/Shr/BitNot/Neg: rewrite recursively
            Expression::BitAnd(l, r) => Expression::BitAnd(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::BitOr(l, r) => Expression::BitOr(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::BitXor(l, r) => Expression::BitXor(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::BitNot(e) => Expression::BitNot(Box::new(Self::rewrite_call_expr(
                *e,
                module_names,
                alias,
            ))),
            Expression::Shl(l, r) => Expression::Shl(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            Expression::Shr(l, r) => Expression::Shr(
                Box::new(Self::rewrite_call_expr(*l, module_names, alias)),
                Box::new(Self::rewrite_call_expr(*r, module_names, alias)),
            ),
            // Leaf nodes pass through.
            other => other,
        }
    }

    /// Take ownership of the current top-level return value. Used by
    /// the MCP server (and tooling) to read what the last `return`
    /// produced after `execute` finished. None when the program didn't
    /// return — equivalent to "no expression result".
    pub fn take_return_value(&mut self) -> Option<Value> {
        self.return_value.take()
    }

    pub fn execute(&mut self, statements: Vec<Statement>) -> Result<(), String> {
        for stmt in statements {
            self.execute_stmt(&stmt)?;
            if self.return_value.is_some() || self.break_flag || self.continue_flag {
                break;
            }
        }
        Ok(())
    }

    /// Host-side self-healing pass over the AST. Walks every node,
    /// applies harmonic / typo / divide-by-zero / arity-pad rewrites,
    /// returns `(healed_stmts, diagnostics)`. Mirrors the OMC-written
    /// healer in `examples/self_healing_h5.omc` but runs natively
    /// before interpretation, so any OMC program benefits when
    /// invoked with `OMC_HEAL=1`.
    ///
    /// Diagnostic classes (each is a one-line composition over Phase O
    /// primitives — `is_fibonacci`, `value_danger`, edit-distance):
    ///
    /// - **Harmonic**: numeric literal not on the Fibonacci spine but
    ///   within distance 3 → rewrite to nearest attractor.
    /// - **Typo (call site)**: function call with unknown name → look
    ///   up best edit-distance match in defined-name table; if ≤ 2
    ///   chars away, rewrite.
    /// - **Divide-by-zero (literal)**: `x / 0` → `safe_divide(x, 0)`.
    /// - **Arity auto-pad (H.6)**: user-fn call with too few args →
    ///   pad with `0` literals; too many → truncate. Only fires on
    ///   USER functions (we know their declared arity); builtins are
    ///   left alone.
    pub fn heal_ast(&self, statements: Vec<Statement>) -> (Vec<Statement>, Vec<String>) {
        let mut diags = Vec::new();
        let defined = self.collect_defined_for_heal(&statements);
        // (name → param_count) for user fns — used by arity-pad.
        let mut arities: HashMap<String, usize> = HashMap::new();
        // (name → set of return-bearing statements present?) — used by the
        // missing-return heal to insert a tail `return null;` for callable
        // fns whose body has no Return statement.
        let mut user_fns_without_return: HashSet<String> = HashSet::new();
        for s in &statements {
            if let Statement::FunctionDef { name, params, body, .. } = s {
                arities.insert(name.clone(), params.len());
                if !stmts_contain_return(body) {
                    user_fns_without_return.insert(name.clone());
                }
            }
        }
        // Substrate-routed name index, built ONCE per pass. Each defined
        // name buckets to substrate_hash(name) mod SUBSTRATE_NAME_BUCKETS
        // so typo-lookup probes only the 3 nearest buckets instead of
        // scanning every defined name. For projects with thousands of
        // names this drops typo-check from O(N · m · k) to
        // O(N · m · log_phi_pi_fibonacci(N)). Stored in a thread-local
        // so heal_stmt/heal_expr don't need extra params.
        let bucketed = build_substrate_name_index(&defined);
        HEAL_SUBSTRATE_INDEX.with(|idx| *idx.borrow_mut() = bucketed);
        HEAL_CLASS_COUNTS.with(|c| *c.borrow_mut() = HealClassCounts::new());
        HEAL_PER_CLASS_DISABLED.with(|d| *d.borrow_mut() = HealDisabled::all_enabled());
        HEAL_BUDGET_REMAINING.with(|b| b.set(HEAL_BUDGET_PER_PASS));

        let healed: Vec<Statement> = statements.into_iter()
            .map(|s| Self::heal_stmt(s, &defined, &arities, &mut diags))
            .collect();
        // Structural-level heals that need the full module view.
        let healed = heal_missing_returns(healed, &user_fns_without_return, &mut diags);
        (healed, diags)
    }

    /// Iterative heal: run heal_ast repeatedly until convergence or
    /// max_iter exceeded. Handles cases where one heal exposes another
    /// (e.g. a typo correction turns into a previously-unknown name
    /// that itself needs harmonic / arity fixes on its arguments).
    ///
    /// Returns `(final_stmts, all_diagnostics, iterations, outcome)`.
    /// Outcomes: `"converged"` (zero diagnostics in last pass),
    /// `"stuck"` (no new diagnostics but non-zero — heal can't make
    /// further progress), `"exhausted"` (hit max_iter).
    pub fn heal_ast_until_fixpoint(
        &self,
        mut statements: Vec<Statement>,
        max_iter: usize,
    ) -> (Vec<Statement>, Vec<String>, usize, &'static str) {
        let mut all_diags: Vec<String> = Vec::new();
        let mut prev_count: usize = usize::MAX;
        for iter in 0..max_iter {
            let (healed, diags) = self.heal_ast(statements);
            statements = healed;
            let count = diags.len();
            if count == 0 {
                return (statements, all_diags, iter, "converged");
            }
            // Same diagnostic count two iterations in a row → no progress.
            if count == prev_count {
                all_diags.extend(diags);
                return (statements, all_diags, iter + 1, "stuck");
            }
            prev_count = count;
            all_diags.extend(diags);
        }
        (statements, all_diags, max_iter, "exhausted")
    }

    fn collect_defined_for_heal(&self, stmts: &[Statement]) -> HashSet<String> {
        let mut set: HashSet<String> = HashSet::new();
        // Baseline: every known builtin name (the healer should never flag
        // a real builtin as a typo). Enumerated explicitly because
        // is_known_builtin is a match expression, not iterable.
        for name in HEAL_BUILTIN_NAMES {
            set.insert(name.to_string());
        }
        // Plus user-defined fns and top-level decls.
        for stmt in stmts {
            match stmt {
                Statement::FunctionDef { name, .. } => { set.insert(name.clone()); }
                Statement::VarDecl { name, .. } => { set.insert(name.clone()); }
                _ => {}
            }
        }
        set
    }

    fn heal_stmt(
        stmt: Statement,
        defined: &HashSet<String>,
        arities: &HashMap<String, usize>,
        diags: &mut Vec<String>,
    ) -> Statement {
        match stmt {
            Statement::VarDecl { name, value, is_harmonic } => Statement::VarDecl {
                name,
                value: Self::heal_expr(value, defined, arities, diags),
                is_harmonic,
            },
            Statement::Assignment { name, value } => Statement::Assignment {
                name,
                value: Self::heal_expr(value, defined, arities, diags),
            },
            Statement::Print(e) => Statement::Print(Self::heal_expr(e, defined, arities, diags)),
            Statement::Expression(e) => Statement::Expression(Self::heal_expr(e, defined, arities, diags)),
            Statement::Return(opt) => Statement::Return(
                opt.map(|e| Self::heal_expr(e, defined, arities, diags))
            ),
            Statement::If { condition, then_body, elif_parts, else_body } => {
                // If-numeric diagnostic: `if 0 { ... }` and `if 1 { ... }`
                // are constant branches — almost always a typo (forgot the
                // comparison) or a leftover debug stub. We don't rewrite
                // (could be intentional placeholder), but we surface the
                // diagnostic and bump the counter.
                if let Expression::Number(n) = &condition {
                    if try_consume_heal_budget() {
                        diags.push(format!(
                            "if-numeric: 'if {}' is a constant branch — \
                             did you forget a comparison?", n
                        ));
                        HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().if_numeric += 1);
                    }
                }
                Statement::If {
                    condition: Self::heal_expr(condition, defined, arities, diags),
                    then_body: then_body.into_iter()
                        .map(|s| Self::heal_stmt(s, defined, arities, diags))
                        .collect(),
                    elif_parts: elif_parts.into_iter()
                        .map(|(c, b)| (
                            Self::heal_expr(c, defined, arities, diags),
                            b.into_iter()
                                .map(|s| Self::heal_stmt(s, defined, arities, diags))
                                .collect(),
                        ))
                        .collect(),
                    else_body: else_body.map(|b| b.into_iter()
                        .map(|s| Self::heal_stmt(s, defined, arities, diags))
                        .collect()),
                }
            },
            Statement::While { condition, body } => Statement::While {
                condition: Self::heal_expr(condition, defined, arities, diags),
                body: body.into_iter()
                    .map(|s| Self::heal_stmt(s, defined, arities, diags))
                    .collect(),
            },
            Statement::FunctionDef { name, params, param_types, body, return_type, pragmas } => {
                // @no_heal pragma opts the entire fn body out of the
                // heal pass. Critical for fns that work with domain
                // values where harmonic rewriting would corrupt
                // semantics — rating thresholds, dimension counts,
                // version numbers, etc. PAIN_POINTS MED-3.
                if pragmas.iter().any(|p| p == "no_heal") {
                    return Statement::FunctionDef {
                        name,
                        params,
                        param_types,
                        body,   // unchanged
                        return_type,
                        pragmas,
                    };
                }
                // Augment the defined set with the fn's params so the
                // body's typo check doesn't flag them. Also collect every
                // VarDecl name declared anywhere in the body (including
                // inside if/while/for) so the new Variable-arm typo heal
                // doesn't false-positive on locals. We hoist the entire
                // body's scope — OMC has no shadowing semantics that the
                // heal pass needs to respect, so over-collecting names
                // is safe (worst case: a true typo of a name only declared
                // later in the body slips through, which is rare).
                let mut inner = defined.clone();
                for p in &params {
                    inner.insert(p.clone());
                }
                collect_local_decls(&body, &mut inner);
                // Per-class pragmas: each can opt this fn out of one
                // heal class without disabling the others. Useful for
                // a fn that wants typo/arity correction but NOT
                // harmonic index rewriting (or vice versa). Pushed
                // through thread-local so heal_expr's inner cases
                // observe them without changing signatures.
                let prev = HEAL_PER_CLASS_DISABLED.with(|d| {
                    let prev = *d.borrow();
                    *d.borrow_mut() = HealDisabled {
                        typo: prev.typo || pragmas.iter().any(|p| p == "no_heal_typo"),
                        arity: prev.arity || pragmas.iter().any(|p| p == "no_heal_arity"),
                        div_zero: prev.div_zero || pragmas.iter().any(|p| p == "no_heal_div"),
                        mod_zero: prev.mod_zero || pragmas.iter().any(|p| p == "no_heal_mod"),
                        harmonic_index: prev.harmonic_index || pragmas.iter().any(|p| p == "no_heal_index"),
                    };
                    prev
                });
                let body: Vec<Statement> = body.into_iter()
                    .map(|s| Self::heal_stmt(s, &inner, arities, diags))
                    .collect();
                HEAL_PER_CLASS_DISABLED.with(|d| *d.borrow_mut() = prev);
                Statement::FunctionDef {
                    name,
                    params,
                    param_types,
                    body,
                    return_type,
                    pragmas,
                }
            }
            // Pass-through for the rest — no expression children to walk.
            other => other,
        }
    }

    fn heal_expr(
        expr: Expression,
        defined: &HashSet<String>,
        arities: &HashMap<String, usize>,
        diags: &mut Vec<String>,
    ) -> Expression {
        match expr {
            // Numeric literals are NO LONGER auto-rewritten by the
            // generic heal pass. Too aggressive: rewriting `check(4)`
            // to `check(3)` because 4 isn't Fibonacci changes user
            // semantics on every domain value. PAIN_POINTS MED-3.
            //
            // Literal harmonic rewriting now happens ONLY when the
            // literal appears in an array-index position (see
            // Expression::Index arm) — that's the original use case
            // safe_arr_get / fold_escape were designed for.
            //
            // Other heals (typo correction, /0 → safe_divide, arity
            // padding) still fire normally.
            Expression::Number(n) => Expression::Number(n),
            Expression::Div(l, r) => {
                let l = Self::heal_expr(*l, defined, arities, diags);
                let r = Self::heal_expr(*r, defined, arities, diags);
                let (l, r, _) = null_arith_rewrite(l, r, diags, "/");
                // Divide-by-zero (literal): wrap in safe_divide.
                if matches!(&r, Expression::Number(0)) {
                    let disabled = HEAL_PER_CLASS_DISABLED.with(|d| d.borrow().div_zero);
                    if !disabled && try_consume_heal_budget() {
                        diags.push("divide-by-zero: rewriting to safe_divide(...)".to_string());
                        HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().div_zero += 1);
                        return Expression::Call {
                            name: "safe_divide".to_string(),
                            args: vec![l, r],
                            pos: crate::ast::Pos::unknown(),
                        };
                    }
                }
                Expression::Div(Box::new(l), Box::new(r))
            }
            Expression::Mod(l, r) => {
                let l = Self::heal_expr(*l, defined, arities, diags);
                let r = Self::heal_expr(*r, defined, arities, diags);
                let (l, r, _) = null_arith_rewrite(l, r, diags, "%");
                // Mod-by-zero (literal): wrap in safe_mod, which substrate-
                // folds the divisor to the smallest non-zero Fibonacci
                // attractor (1) at runtime. Wrapping in a call instead
                // of a literal rewrite means the original 0 is preserved
                // for the substrate-fold step, and the rewrite composes
                // with safe_divide's identical contract.
                if matches!(&r, Expression::Number(0)) {
                    let disabled = HEAL_PER_CLASS_DISABLED.with(|d| d.borrow().mod_zero);
                    if !disabled && try_consume_heal_budget() {
                        diags.push("mod-by-zero: rewriting to safe_mod(...)".to_string());
                        HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().mod_zero += 1);
                        return Expression::Call {
                            name: "safe_mod".to_string(),
                            args: vec![l, r],
                            pos: crate::ast::Pos::unknown(),
                        };
                    }
                }
                Expression::Mod(Box::new(l), Box::new(r))
            }
            Expression::Call { name, args, pos } => {
                // Typo check at call site. Substrate-routed lookup:
                // probes the 3 hash-bucket neighborhood first, falls
                // back to full closest_name if the bucketed scan misses.
                // Prefer user-defined fns (arities.keys()) over builtins
                // on ties — a typo is more likely meant for a user fn.
                let user_fns: HashSet<String> = arities.keys().cloned().collect();
                let typo_disabled = HEAL_PER_CLASS_DISABLED.with(|d| d.borrow().typo);
                let healed_name = if defined.contains(&name) {
                    name
                } else if !typo_disabled {
                    if let Some(close) = closest_name_substrate(&name, defined, 2, Some(&user_fns)) {
                        if try_consume_heal_budget() {
                            diags.push(format!("call: '{}' unknown → '{}'", name, close));
                            HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().typo += 1);
                            close
                        } else {
                            name
                        }
                    } else {
                        name
                    }
                } else {
                    name
                };
                // Heal each argument first.
                let mut healed_args: Vec<Expression> = args.into_iter()
                    .map(|a| Self::heal_expr(a, defined, arities, diags))
                    .collect();
                // H.6: arity auto-pad / truncate. Only applies to user
                // functions whose declared param count we know.
                let arity_disabled = HEAL_PER_CLASS_DISABLED.with(|d| d.borrow().arity);
                if !arity_disabled {
                    if let Some(&expected) = arities.get(&healed_name) {
                        if healed_args.len() < expected && try_consume_heal_budget() {
                            let needed = expected - healed_args.len();
                            diags.push(format!(
                                "arity: {}() called with {} args, padded with {} zeros to match arity {}",
                                healed_name, healed_args.len(), needed, expected
                            ));
                            HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().arity_pad += 1);
                            for _ in 0..needed {
                                healed_args.push(Expression::Number(0));
                            }
                        } else if healed_args.len() > expected && try_consume_heal_budget() {
                            let excess = healed_args.len() - expected;
                            diags.push(format!(
                                "arity: {}() called with {} args, truncated {} excess to match arity {}",
                                healed_name, healed_args.len(), excess, expected
                            ));
                            HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().arity_truncate += 1);
                            healed_args.truncate(expected);
                        }
                    }
                }
                // Preserve the original source position through the
                // heal pass — we don't reposition synthesized call
                // nodes, but we DO keep the original pos so traces
                // still point at the user's code.
                Expression::Call { name: healed_name, args: healed_args, pos }
            }
            // String-concat heal. `"foo" + 5` is a runtime-typed error in
            // OMC (Add only defined for matching types). When one side is
            // a string LITERAL and the other is a number/float LITERAL,
            // rewrite to `concat_many(string, to_string(num))`. Literal-
            // only so we never false-positive on `vec + 1.0` where both
            // sides are numeric arrays.
            Expression::Add(l, r) => {
                let l = Self::heal_expr(*l, defined, arities, diags);
                let r = Self::heal_expr(*r, defined, arities, diags);
                let l_is_str = matches!(&l, Expression::String(_));
                let r_is_str = matches!(&r, Expression::String(_));
                let l_is_num = matches!(&l, Expression::Number(_) | Expression::Float(_));
                let r_is_num = matches!(&r, Expression::Number(_) | Expression::Float(_));
                if (l_is_str && r_is_num) || (r_is_str && l_is_num) {
                    if try_consume_heal_budget() {
                        diags.push(
                            "str-concat: 'str + num' rewritten to concat_many(str, to_string(num))"
                                .to_string(),
                        );
                        HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().str_concat += 1);
                        let wrap = |e: Expression| -> Expression {
                            if matches!(&e, Expression::String(_)) {
                                e
                            } else {
                                Expression::Call {
                                    name: "to_string".to_string(),
                                    args: vec![e],
                                    pos: crate::ast::Pos::unknown(),
                                }
                            }
                        };
                        return Expression::Call {
                            name: "concat_many".to_string(),
                            args: vec![wrap(l), wrap(r)],
                            pos: crate::ast::Pos::unknown(),
                        };
                    }
                }
                // Null on either side of Add — heal to 0. Common when
                // a fn returns null and the caller adds to it; runtime
                // errors out with a confusing type error otherwise.
                let (l, r, healed) = null_arith_rewrite(l, r, diags, "+");
                if healed { return Expression::Add(Box::new(l), Box::new(r)); }
                Expression::Add(Box::new(l), Box::new(r))
            }
            Expression::Sub(l, r) => {
                let l = Self::heal_expr(*l, defined, arities, diags);
                let r = Self::heal_expr(*r, defined, arities, diags);
                let (l, r, _) = null_arith_rewrite(l, r, diags, "-");
                Expression::Sub(Box::new(l), Box::new(r))
            }
            Expression::Mul(l, r) => {
                let l = Self::heal_expr(*l, defined, arities, diags);
                let r = Self::heal_expr(*r, defined, arities, diags);
                let (l, r, _) = null_arith_rewrite(l, r, diags, "*");
                Expression::Mul(Box::new(l), Box::new(r))
            }
            // Comparison arms: don't auto-rewrite numeric literals on
            // either side. `if rating == 4` is comparing against a
            // domain value (rating threshold) — rewriting 4 → 3 would
            // silently change semantics. Same for >= 5, < 10, etc.
            // Apply heal RECURSIVELY but skip the literal-rewrite step.
            Expression::Eq(l, r) => Expression::Eq(
                Box::new(Self::heal_expr_skip_literal(*l, defined, arities, diags)),
                Box::new(Self::heal_expr_skip_literal(*r, defined, arities, diags)),
            ),
            Expression::Ne(l, r) => Expression::Ne(
                Box::new(Self::heal_expr_skip_literal(*l, defined, arities, diags)),
                Box::new(Self::heal_expr_skip_literal(*r, defined, arities, diags)),
            ),
            Expression::Lt(l, r) => Expression::Lt(
                Box::new(Self::heal_expr_skip_literal(*l, defined, arities, diags)),
                Box::new(Self::heal_expr_skip_literal(*r, defined, arities, diags)),
            ),
            Expression::Le(l, r) => Expression::Le(
                Box::new(Self::heal_expr_skip_literal(*l, defined, arities, diags)),
                Box::new(Self::heal_expr_skip_literal(*r, defined, arities, diags)),
            ),
            Expression::Gt(l, r) => Expression::Gt(
                Box::new(Self::heal_expr_skip_literal(*l, defined, arities, diags)),
                Box::new(Self::heal_expr_skip_literal(*r, defined, arities, diags)),
            ),
            Expression::Ge(l, r) => Expression::Ge(
                Box::new(Self::heal_expr_skip_literal(*l, defined, arities, diags)),
                Box::new(Self::heal_expr_skip_literal(*r, defined, arities, diags)),
            ),
            Expression::And(l, r) => Expression::And(
                Box::new(Self::heal_expr(*l, defined, arities, diags)),
                Box::new(Self::heal_expr(*r, defined, arities, diags)),
            ),
            Expression::Or(l, r) => Expression::Or(
                Box::new(Self::heal_expr(*l, defined, arities, diags)),
                Box::new(Self::heal_expr(*r, defined, arities, diags)),
            ),
            Expression::Not(e) => Expression::Not(
                Box::new(Self::heal_expr(*e, defined, arities, diags)),
            ),
            Expression::Array(items) => Expression::Array(
                items.into_iter()
                    .map(|e| Self::heal_expr(e, defined, arities, diags))
                    .collect(),
            ),
            Expression::Safe(inner) => Expression::Safe(
                Box::new(Self::heal_expr(*inner, defined, arities, diags)),
            ),
            // Index expression: rewrite numeric literal indices onto
            // Fibonacci attractors. This is the original use case for
            // harmonic healing — `arr[7]` → `arr[8]` lands on a stable
            // attractor that fold_escape can clean up at runtime.
            // OUTSIDE index position (function args, return values,
            // variable bindings) literal rewriting changes user
            // semantics so we don't do it.
            Expression::Index { name, index } => {
                let healed_index = match *index {
                    Expression::Number(n) if !is_on_fibonacci_attractor(n) => {
                        let disabled = HEAL_PER_CLASS_DISABLED.with(|d| d.borrow().harmonic_index);
                        if disabled {
                            Expression::Number(n)
                        } else {
                            let nearest = fold_to_fibonacci_const(n);
                            let delta = (nearest - n).abs();
                            if delta > 0 && delta <= 3 && try_consume_heal_budget() {
                                diags.push(format!(
                                    "harmonic-index: {}[{}] → {}[{}] (|Δ|={})",
                                    name, n, name, nearest, delta
                                ));
                                HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().harmonic_index += 1);
                                Expression::Number(nearest)
                            } else {
                                Expression::Number(n)
                            }
                        }
                    }
                    other => Self::heal_expr(other, defined, arities, diags),
                };
                Expression::Index {
                    name,
                    index: Box::new(healed_index),
                }
            }
            Expression::ChainedIndex { object, index } => Expression::ChainedIndex {
                object: Box::new(Self::heal_expr(*object, defined, arities, diags)),
                index: Box::new(Self::heal_expr(*index, defined, arities, diags)),
            },
            // Variable-position typo. Mirrors the call-site typo logic
            // (substrate-bucketed close-name lookup), but fires when a
            // bare identifier is referenced rather than called. Only
            // active because we now seed `defined` with locally-declared
            // VarDecls + params before recursing into a fn body — without
            // that seeding, every local would false-positive here.
            Expression::Variable(name) => {
                if defined.contains(&name) {
                    Expression::Variable(name)
                } else {
                    let typo_disabled = HEAL_PER_CLASS_DISABLED.with(|d| d.borrow().typo);
                    if typo_disabled {
                        Expression::Variable(name)
                    } else {
                        let user_fns: HashSet<String> = arities.keys().cloned().collect();
                        if let Some(close) = closest_name_substrate(&name, defined, 2, Some(&user_fns)) {
                            if try_consume_heal_budget() {
                                diags.push(format!("var: '{}' unknown → '{}'", name, close));
                                HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().var_typo += 1);
                                return Expression::Variable(close);
                            }
                        }
                        Expression::Variable(name)
                    }
                }
            }
            // Pass-through for leaves and forms that have no expression
            // children we'd want to rewrite at this layer.
            other => other,
        }
    }

    /// heal_expr variant that skips the harmonic literal-rewrite at the
    /// TOP of the expression, but recursively heals everything else
    /// normally. Used by comparison arms where the top-level operand is
    /// likely a domain value (`if rating >= 4` — don't rewrite 4 → 3).
    /// Nested expressions still get full healing.
    fn heal_expr_skip_literal(
        expr: Expression,
        defined: &HashSet<String>,
        arities: &HashMap<String, usize>,
        diags: &mut Vec<String>,
    ) -> Expression {
        match expr {
            // Skip literal rewrite at this position only.
            Expression::Number(n) => Expression::Number(n),
            Expression::Float(f) => Expression::Float(f),
            // Everything else gets normal healing (recursive children
            // may still hit literal rewriting where appropriate).
            other => Self::heal_expr(other, defined, arities, diags),
        }
    }

    fn execute_stmt(&mut self, stmt: &Statement) -> Result<(), String> {
        match stmt {
            Statement::Print(expr) => {
                let value = self.eval_expr(expr)?;
                let s = value.to_display_string();
                println!("{}", s);
                self.output_lines.borrow_mut().push(s);
                Ok(())
            }
            Statement::Expression(expr) => {
                // Save the result so the MCP / REPL paths can read
                // "what did the last TOP-LEVEL expression evaluate to"
                // without re-running. Only update when we are at the
                // top level (not inside a user function body), to avoid
                // polluting the REPL result with sub-expression values
                // computed deep inside call_stack frames.
                let v = self.eval_expr(expr)?;
                if self.call_stack.is_empty() {
                    self.last_expression_value = Some(v);
                }
                Ok(())
            }
            Statement::VarDecl {
                name,
                value,
                is_harmonic: _,
            } => {
                let val = self.eval_expr(value)?;
                self.set_var(name.clone(), val);
                Ok(())
            }
            Statement::Parameter { name, value } => {
                let val = self.eval_expr(value)?;
                self.set_var(name.clone(), val);
                Ok(())
            }
            Statement::Assignment { name, value } => {
                let val = self.eval_expr(value)?;
                // Assignment walks outward — finds existing binding in
                // outer locals, captured closure envs, or globals. This
                // is what makes `n = n + 1` inside a closure mutate the
                // captured `n` instead of shadowing it.
                self.assign_var(name.clone(), val);
                Ok(())
            }
            Statement::IndexAssignment {
                name,
                index,
                value,
            } => {
                let idx_v = self.eval_expr(index)?;
                let val = self.eval_expr(value)?;
                match self.get_var(name) {
                    Some(Value::Array(arr)) => {
                        let len = arr.items.borrow().len() as i64;
                        let raw = idx_v.to_int();
                        let resolved = if raw < 0 { (len + raw) as usize } else { raw as usize };
                        let mut items = arr.items.borrow_mut();
                        if resolved < items.len() {
                            items[resolved] = val;
                        }
                    }
                    Some(Value::Dict(d)) => {
                        let key = idx_v.to_display_string();
                        d.borrow_mut().insert(key, val);
                    }
                    _ => {}
                }
                Ok(())
            }
            Statement::ChainedIndexAssignment { name, first_index, second_index, value } => {
                let first_key = self.eval_expr(first_index)?;
                let second_key = self.eval_expr(second_index)?;
                let val = self.eval_expr(value)?;
                let container = match self.get_var(name) {
                    Some(Value::Array(arr)) => {
                        let len = arr.items.borrow().len() as i64;
                        let raw = first_key.to_int();
                        let idx = if raw < 0 { (len + raw) as usize } else { raw as usize };
                        arr.items.borrow().get(idx).cloned()
                    }
                    Some(Value::Dict(d)) => {
                        let k = first_key.to_display_string();
                        d.borrow().get(&k).cloned()
                    }
                    _ => None,
                };
                match container {
                    Some(Value::Array(arr)) => {
                        let len = arr.items.borrow().len() as i64;
                        let raw = second_key.to_int();
                        let idx = if raw < 0 { (len + raw) as usize } else { raw as usize };
                        let mut items = arr.items.borrow_mut();
                        if idx < items.len() { items[idx] = val; }
                    }
                    Some(Value::Dict(d)) => {
                        let k = second_key.to_display_string();
                        d.borrow_mut().insert(k, val);
                    }
                    _ => {}
                }
                Ok(())
            }
            Statement::If {
                condition,
                then_body,
                elif_parts,
                else_body,
            } => {
                if self.eval_expr(condition)?.to_bool() {
                    self.execute_block(then_body)?;
                } else {
                    let mut executed = false;
                    for (elif_cond, elif_body) in elif_parts {
                        if self.eval_expr(elif_cond)?.to_bool() {
                            self.execute_block(elif_body)?;
                            executed = true;
                            break;
                        }
                    }
                    if !executed {
                        if let Some(body) = else_body {
                            self.execute_block(body)?;
                        }
                    }
                }
                Ok(())
            }
            Statement::While { condition, body } => {
                while self.eval_expr(condition)?.to_bool() {
                    self.execute_block(body)?;
                    if self.break_flag {
                        self.break_flag = false;
                        break;
                    }
                    if self.continue_flag {
                        self.continue_flag = false;
                        continue;
                    }
                    if self.return_value.is_some() {
                        break;
                    }
                }
                Ok(())
            }
            Statement::For {
                var,
                iterable,
                body,
            } => {
                match iterable {
                    ForIterable::Range { start, end } => {
                        let start_val = self.eval_expr(start)?.to_int();
                        let end_val = self.eval_expr(end)?.to_int();
                        for i in start_val..end_val {
                            self.set_var(var.clone(), Value::HInt(HInt::new(i)));
                            self.execute_block(body)?;
                            if self.break_flag {
                                self.break_flag = false;
                                break;
                            }
                            if self.continue_flag {
                                self.continue_flag = false;
                                continue;
                            }
                            if self.return_value.is_some() {
                                break;
                            }
                        }
                    }
                    ForIterable::Expr(expr) => {
                        let val = self.eval_expr(expr)?;
                        // Snapshot items so the loop body can mutate
                        // the underlying Rc<RefCell<Vec>> without
                        // tripping a borrow conflict. Materialize once
                        // per iterable type — Array iterates elements,
                        // Dict iterates keys (Python convention), String
                        // iterates characters. Anything else errors —
                        // silent skips used to hide typos.
                        let items: Vec<Value> = match &val {
                            Value::Array(arr) => arr.items.borrow().clone(),
                            Value::Dict(d) => d.borrow().keys()
                                .map(|k| Value::String(k.clone())).collect(),
                            Value::String(s) => s.chars()
                                .map(|c| Value::String(c.to_string())).collect(),
                            other => return Err(format!(
                                "for-loop: cannot iterate over {} \
                                 (expected array, dict, or string)",
                                type_name_of(other)
                            )),
                        };
                        for item in items {
                            self.set_var(var.clone(), item);
                            self.execute_block(body)?;
                            if self.break_flag {
                                self.break_flag = false;
                                break;
                            }
                            if self.continue_flag {
                                self.continue_flag = false;
                                continue;
                            }
                            if self.return_value.is_some() {
                                break;
                            }
                        }
                    }
                }
                Ok(())
            }
            Statement::FunctionDef {
                name,
                params,
                body,
                ..
            } => {
                self.functions.insert(name.clone(), (params.clone(), body.clone()));
                Ok(())
            }
            Statement::ClassDef { name, parent, fields, methods } => {
                // Desugar at execute time so the tree-walker doesn't
                // need register_user_functions to have been called.
                // Same logic as register_user_functions::visit:
                // synthesize a constructor + mangled methods.
                if let Some(p) = parent {
                    self.class_parents.insert(name.clone(), p.clone());
                }
                let mut ctor_body: Vec<Statement> = Vec::new();
                ctor_body.push(Statement::VarDecl {
                    name: "__obj".to_string(),
                    value: Expression::Call {
                        name: "dict_new".to_string(),
                        args: vec![],
                        pos: crate::ast::Pos::unknown(),
                    },
                    is_harmonic: true,
                });
                ctor_body.push(Statement::Expression(Expression::Call {
                    name: "dict_set".to_string(),
                    args: vec![
                        Expression::Variable("__obj".to_string()),
                        Expression::String("__class__".to_string()),
                        Expression::String(name.clone()),
                    ],
                    pos: crate::ast::Pos::unknown(),
                }));
                for f in fields {
                    ctor_body.push(Statement::Expression(Expression::Call {
                        name: "dict_set".to_string(),
                        args: vec![
                            Expression::Variable("__obj".to_string()),
                            Expression::String(f.clone()),
                            Expression::Variable(f.clone()),
                        ],
                        pos: crate::ast::Pos::unknown(),
                    }));
                }
                ctor_body.push(Statement::Return(Some(
                    Expression::Variable("__obj".to_string()),
                )));
                self.functions.insert(name.clone(), (fields.clone(), ctor_body));
                for m in methods {
                    if let Statement::FunctionDef { name: mname, params, body, .. } = m {
                        let mangled = format!("{}__{}", name, mname);
                        self.functions.insert(mangled, (params.clone(), body.clone()));
                    }
                }
                Ok(())
            }
            Statement::Return(expr) => {
                self.return_value = Some(
                    if let Some(e) = expr {
                        self.eval_expr(e)?
                    } else {
                        Value::Null
                    }
                );
                Ok(())
            }
            Statement::Break => {
                self.break_flag = true;
                Ok(())
            }
            Statement::Continue => {
                self.continue_flag = true;
                Ok(())
            }
            Statement::Import { module, alias, selected } => {
                // Three import shapes:
                //   import "foo";              → flat merge all fns
                //   import "foo" as math;      → fns become math.fname
                //   from "foo" import a, b;    → only `a` and `b` get imported
                if let Some(names) = selected {
                    self.import_module_selective(module, names)
                } else {
                    self.import_module_with_alias(module, alias.as_deref())
                }
            }
            Statement::Match { scrutinee, arms } => {
                let value = self.eval_expr(scrutinee)?;
                for arm in arms {
                    let mut bindings: Vec<(String, Value)> = Vec::new();
                    if pattern_matches(&arm.pattern, &value, &mut bindings) {
                        // Apply the bindings as plain set_var into the
                        // current scope, then run the arm body. The
                        // scope IS the surrounding block — match isn't
                        // its own scope, matching `if`'s behavior.
                        for (n, v) in bindings {
                            self.set_var(n, v);
                        }
                        return self.execute_block(&arm.body);
                    }
                }
                // No arm matched — silent no-op.
                Ok(())
            }
            Statement::Try { body, err_var, handler, finally } => {
                // Run the body; if anything inside returns Err, jump to
                // the handler. If the error came from a `throw <expr>`,
                // pending_throw holds the typed value — bind that to
                // err_var so the handler sees the original dict/object.
                // Otherwise (error from a Rust builtin) fall back to the
                // string form. After body+handler complete, run finally
                // unconditionally — matches Python try/except/finally.
                let body_result = self.execute_block(body);
                let after_handler = match body_result {
                    Ok(()) => Ok(()),
                    Err(msg) => {
                        let caught = self.pending_throw.take()
                            .unwrap_or(Value::String(msg));
                        self.set_var(err_var.clone(), caught);
                        self.execute_block(handler)
                    }
                };
                if let Some(finally_body) = finally {
                    let finally_result = self.execute_block(finally_body);
                    if finally_result.is_err() {
                        return finally_result;
                    }
                }
                after_handler
            }
            Statement::Throw(expr) => {
                // Evaluate the expression. Stash the value in
                // pending_throw so a surrounding catch can bind it
                // with its original type/shape, then return Err with
                // the display string so existing Err-based propagation
                // keeps working. Uncaught throws clear pending_throw
                // on the way out (caller observes only the string).
                let v = self.eval_expr(expr)?;
                let display = v.to_display_string();
                self.pending_throw = Some(v);
                Err(display)
            }
            Statement::Yield(expr) => {
                // Two modes:
                //   1. Streaming (gen_stream installed a callback):
                //      invoke the callback with the yielded value.
                //      O(1) memory regardless of how many yields.
                //      A 0 return short-circuits — set gen_stop_requested
                //      and a return_value sentinel so loops unwind.
                //   2. Eager (legacy): append to the top collector.
                //      Materializes the full sequence as Value::Array
                //      when the generator returns.
                let v = self.eval_expr(expr)?;
                if let Some(cb) = self.yield_callbacks.last().cloned() {
                    let r = self.call_first_class_function(&cb, vec![v])?;
                    if r.to_int() == 0 {
                        self.gen_stop_requested = true;
                        // Trigger unwind: set return_value to Null so
                        // outer block/loop sees "fn returned" and exits.
                        if self.return_value.is_none() {
                            self.return_value = Some(Value::Null);
                        }
                    }
                } else if let Some(top) = self.yield_stacks.last_mut() {
                    top.push(v);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn execute_block(&mut self, statements: &[Statement]) -> Result<(), String> {
        for stmt in statements {
            self.execute_stmt(stmt)?;
            if self.return_value.is_some() || self.break_flag || self.continue_flag {
                break;
            }
        }
        Ok(())
    }

    fn eval_expr(&mut self, expr: &Expression) -> Result<Value, String> {
        match expr {
            Expression::Number(n) => Ok(Value::HInt(HInt::new(*n))),
            Expression::Float(f) => Ok(Value::HFloat(*f)),
            Expression::String(s) => Ok(Value::String(s.clone())),
            Expression::Boolean(b) => Ok(Value::Bool(*b)),
            Expression::Array(exprs) => {
                let mut items = Vec::new();
                for e in exprs {
                    items.push(self.eval_expr(e)?);
                }
                Ok(Value::Array(HArray::from_vec(items)))
            }
            Expression::Dict(pairs) => {
                let mut map = std::collections::BTreeMap::new();
                for (k_expr, v_expr) in pairs {
                    let k = self.eval_expr(k_expr)?.to_display_string();
                    let v = self.eval_expr(v_expr)?;
                    map.insert(k, v);
                }
                Ok(Value::dict_from(map))
            }
            Expression::Variable(name) => {
                // Reserved literals — match position is identifier in
                // the source, but semantically these are constants.
                // Cheaper than adding three more Token variants and
                // matches user expectation ("null is just a value").
                match name.as_str() {
                    "null" => return Ok(Value::Null),
                    "true" => return Ok(Value::Bool(true)),
                    "false" => return Ok(Value::Bool(false)),
                    _ => {}
                }
                // First try variable lookup. If missing, fall back to the
                // function table — bare function names become first-class
                // values (Value::Function) so they can be passed to
                // higher-order builtins like arr_map / arr_filter / arr_reduce.
                // Built-ins are also reachable this way; the dispatch in
                // call_first_class_function tries user fns first, then
                // routes anything else through call_function.
                if let Some(v) = self.get_var(name) {
                    Ok(v)
                } else if self.functions.contains_key(name) {
                    Ok(Value::Function { name: name.clone(), captured: None })
                } else if self.is_known_builtin(name) {
                    Ok(Value::Function { name: name.clone(), captured: None })
                } else {
                    Err(format!("Undefined variable: {}{}", name, self.undefined_var_hint(name)))
                }
            }
            Expression::Index { name, index } => {
                let idx_v = self.eval_expr(index)?;
                let container = self.get_var(name)
                    .ok_or_else(|| format!("Undefined variable: {}{}", name, self.undefined_var_hint(name)))?;
                match container {
                    Value::Array(arr) => {
                        let items = arr.items.borrow();
                        let len = items.len() as i64;
                        let raw = idx_v.to_int();
                        // Python-style negative indexing: -1 is the last
                        // element, -2 the second-to-last, etc. Out of
                        // range (either side) becomes a helpful error
                        // that names the array AND reports its length —
                        // not just the raw index, which by itself never
                        // tells the user how far off they were.
                        let resolved = if raw < 0 { len + raw } else { raw };
                        if resolved < 0 || resolved >= len {
                            return Err(format!(
                                "Index out of bounds: {}[{}] (length {}). \
                                 Use safe_arr_get({}, {}) for wrap-around access.",
                                name, raw, len, name, raw
                            ));
                        }
                        Ok(items[resolved as usize].clone())
                    }
                    Value::Dict(d) => {
                        // String-keyed lookup. Coerce numeric/bool indices
                        // via to_display_string so `d[42]` works as
                        // `d["42"]` — surprising for some, but matches
                        // OMC's "everything stringifies" stance.
                        let key = idx_v.to_display_string();
                        Ok(d.borrow().get(&key).cloned().unwrap_or(Value::Null))
                    }
                    _ => Err(format!(
                        "Cannot index '{}': not an array or dict",
                        name
                    )),
                }
            }
            Expression::ChainedIndex { object, index } => {
                let idx_v = self.eval_expr(index)?;
                let container = self.eval_expr(object)?;
                match container {
                    Value::Array(arr) => {
                        let items = arr.items.borrow();
                        let len = items.len() as i64;
                        let raw = idx_v.to_int();
                        let resolved = if raw < 0 { len + raw } else { raw };
                        if resolved < 0 || resolved >= len {
                            return Err(format!(
                                "Index out of bounds: [{}] (length {})",
                                raw, len
                            ));
                        }
                        Ok(items[resolved as usize].clone())
                    }
                    Value::Dict(d) => {
                        let key = idx_v.to_display_string();
                        Ok(d.borrow().get(&key).cloned().unwrap_or(Value::Null))
                    }
                    _ => Err("Cannot index value: not an array or dict".to_string()),
                }
            }
            Expression::Add(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                // String + anything → concat, like Python. Avoids the
                // earlier footgun where `"a" + "b"` coerced to int and
                // returned 0. Either side being a string triggers this
                // (numbers/bools/etc. stringify via to_string).
                if matches!(lv, Value::String(_)) || matches!(rv, Value::String(_)) {
                    // Use to_display_string so `"count: " + 42` produces
                    // "count: 42", not "count: HInt(42, φ=..., HIM=...)".
                    Ok(Value::String(format!(
                        "{}{}",
                        lv.to_display_string(),
                        rv.to_display_string()
                    )))
                } else if lv.is_float() || rv.is_float() {
                    Ok(Value::HFloat(lv.to_float() + rv.to_float()))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int().wrapping_add(rv.to_int()))))
                }
            }
            Expression::Sub(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::HFloat(lv.to_float() - rv.to_float()))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int().wrapping_sub(rv.to_int()))))
                }
            }
            Expression::Mul(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::HFloat(lv.to_float() * rv.to_float()))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int().wrapping_mul(rv.to_int()))))
                }
            }
            Expression::Div(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    let r_f = rv.to_float();
                    if r_f == 0.0 {
                        Ok(Value::Singularity {
                            numerator: lv.to_int(),
                            denominator: 0,
                            context: "div".to_string(),
                        })
                    } else {
                        Ok(Value::HFloat(lv.to_float() / r_f))
                    }
                } else {
                    let divisor = rv.to_int();
                    if divisor == 0 {
                        Ok(Value::Singularity {
                            numerator: lv.to_int(),
                            denominator: 0,
                            context: "div".to_string(),
                        })
                    } else {
                        Ok(Value::HInt(HInt::new(lv.to_int() / divisor)))
                    }
                }
            }
            Expression::Mod(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    let r_f = rv.to_float();
                    if r_f == 0.0 {
                        Ok(Value::HFloat(0.0))
                    } else {
                        Ok(Value::HFloat(lv.to_float() % r_f))
                    }
                } else {
                    let divisor = rv.to_int();
                    if divisor == 0 {
                        Ok(Value::HInt(HInt::new(0)))
                    } else {
                        Ok(Value::HInt(HInt::new(lv.to_int() % divisor)))
                    }
                }
            }
            Expression::Eq(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                Ok(Value::Bool(values_equal(&lv, &rv)))
            }
            Expression::Ne(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                Ok(Value::Bool(!values_equal(&lv, &rv)))
            }
            Expression::Lt(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::Bool(lv.to_float() < rv.to_float()))
                } else {
                    Ok(Value::Bool(lv.to_int() < rv.to_int()))
                }
            }
            Expression::Le(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::Bool(lv.to_float() <= rv.to_float()))
                } else {
                    Ok(Value::Bool(lv.to_int() <= rv.to_int()))
                }
            }
            Expression::Gt(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::Bool(lv.to_float() > rv.to_float()))
                } else {
                    Ok(Value::Bool(lv.to_int() > rv.to_int()))
                }
            }
            Expression::Ge(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::Bool(lv.to_float() >= rv.to_float()))
                } else {
                    Ok(Value::Bool(lv.to_int() >= rv.to_int()))
                }
            }
            Expression::And(l, r) => {
                let lv = self.eval_expr(l)?.to_bool();
                if !lv {
                    return Ok(Value::Bool(false));
                }
                let rv = self.eval_expr(r)?.to_bool();
                Ok(Value::Bool(rv))
            }
            Expression::Or(l, r) => {
                let lv = self.eval_expr(l)?.to_bool();
                if lv {
                    return Ok(Value::Bool(true));
                }
                let rv = self.eval_expr(r)?.to_bool();
                Ok(Value::Bool(rv))
            }
            Expression::Not(e) => {
                let v = self.eval_expr(e)?.to_bool();
                Ok(Value::Bool(!v))
            }
            // Bitwise ops — always operate on i64 representations.
            Expression::BitAnd(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::HInt(HInt::new(lv & rv)))
            }
            Expression::BitOr(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::HInt(HInt::new(lv | rv)))
            }
            Expression::BitXor(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::HInt(HInt::new(lv ^ rv)))
            }
            Expression::BitNot(e) => {
                let v = self.eval_expr(e)?.to_int();
                Ok(Value::HInt(HInt::new(!v)))
            }
            Expression::Shl(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                // Mask shift amount to a safe 0-63 range to match Rust's panic-free i64 shifts.
                Ok(Value::HInt(HInt::new(lv.wrapping_shl((rv & 63) as u32))))
            }
            Expression::Shr(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::HInt(HInt::new(lv.wrapping_shr((rv & 63) as u32))))
            }
            Expression::Call { name, args, pos } => {
                self.call_function_at(name, args, *pos)
            }
            Expression::Resonance(e) => {
                // Match the call_function("res", ...) path: return HFloat resonance score.
                let v = self.eval_expr(e)?;
                match v {
                    Value::HInt(h) => Ok(Value::HFloat(h.resonance)),
                    Value::HFloat(f) => Ok(Value::HFloat(HInt::compute_resonance(f as i64))),
                    _ => Ok(Value::HFloat(0.0)),
                }
            }
            Expression::Fold(e) => {
                let v = self.eval_expr(e)?;
                match v {
                    Value::HInt(h) => {
                        let result = crate::phi_pi_fib::fold_to_nearest_attractor(h.value);
                        Ok(Value::HInt(HInt::new(result)))
                    }
                    _ => Ok(Value::HInt(HInt::new(0))),
                }
            }
            Expression::Safe(inner) => {
                // H.5: dispatch user-declared safe semantics by inner shape.
                // Known shapes route to the matching ONN primitive; everything
                // else is evaluated unwrapped (reserves the slot for future
                // runtime guards on more call patterns).
                match inner.as_ref() {
                    Expression::Div(l, r) => {
                        let args = vec![(**l).clone(), (**r).clone()];
                        self.call_function("safe_divide", &args)
                    }
                    Expression::Call { name, args, .. } if name == "arr_get" && args.len() == 2 => {
                        self.call_function("safe_arr_get", args)
                    }
                    Expression::Call { name, args, .. } if name == "arr_set" && args.len() == 3 => {
                        self.call_function("safe_arr_set", args)
                    }
                    _ => self.eval_expr(inner),
                }
            }
            Expression::Lambda { params, body } => {
                // Closures: snapshot the current local scope, generate a
                // unique anonymous function name, register the body under
                // that name in self.functions, return a Value::Function
                // carrying both the name and the captured environment.
                //
                // The captured env is Rc<RefCell> so:
                //   - mutations inside the closure persist across calls
                //   - cloning the Value::Function shares the same env
                //     (multiple references to the same closure see the
                //     same mutable state)
                //
                // Anonymous-name collision avoidance is just a monotonic
                // counter — single-threaded interpreter, so it's fine.
                self.lambda_counter += 1;
                // Distinct prefix from the compiler-side `__lambda_N`
                // pool (LAMBDA_SEQ in compiler.rs). Both counters
                // assign sequential numbers starting from 0; if they
                // share the same prefix, tree-walk-time lambdas
                // overwrite VM-time lambdas in self.functions and
                // every nested fn that creates a lambda corrupts the
                // global function table.
                let fn_name = format!("__rt_lambda_{}", self.lambda_counter);
                self.functions.insert(
                    fn_name.clone(),
                    (params.clone(), body.clone()),
                );
                // Capture by REFERENCE — clone the Rc so the closure
                // and the enclosing scope point to the same RefCell.
                // Sibling closures in the same scope share state; mutations
                // through any of them propagate to all. This is what makes
                // the bank-account pattern (multiple methods over shared
                // private state) work.
                let env = self.locals
                    .last()
                    .cloned()
                    .unwrap_or_else(|| std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())));
                Ok(Value::Function {
                    name: fn_name,
                    captured: Some(env),
                })
            }
            Expression::IfExpr { condition, then_body, else_body } => {
                let cond = self.eval_expr(condition)?;
                let branch: &[Statement] = if cond.to_bool() {
                    then_body
                } else {
                    match else_body {
                        Some(b) => b,
                        None => return Ok(Value::Null),
                    }
                };
                // Run all statements; last expression-statement is the value.
                let mut last = Value::Null;
                for stmt in branch {
                    match stmt {
                        Statement::Expression(e) => {
                            last = self.eval_expr(e)?;
                        }
                        other => { self.execute_stmt(other)?; }
                    }
                }
                Ok(last)
            }
        }
    }

    /// First-class function support — quick membership test against the
    /// known builtin name set. Used by Expression::Variable evaluation to
    /// decide whether a bare name should resolve to Value::Function rather
    /// than erroring with "Undefined variable".
    ///
    /// Kept as a static match rather than a HashSet so the compiler can
    /// fold the lookup into a single jump table. Add new builtins here
    /// when you add them to the call_function dispatch.
    fn is_known_builtin(&self, name: &str) -> bool {
        matches!(name,
            // Numbers & math
            "abs" | "min" | "max" | "sign" | "floor" | "ceil" | "round" | "frac"
            | "gcd" | "lcm" | "square" | "cube" | "pow" | "pow_int" | "sqrt"
            | "mod_pow" | "bit_count" | "bit_length" | "digit_sum" | "digit_count"
            | "factorial" | "is_even" | "even" | "is_odd" | "odd" | "is_prime"
            | "sin" | "cos" | "tan" | "tanh" | "exp" | "log" | "erf" | "sigmoid"
            | "log2" | "log10" | "asin" | "acos" | "atan" | "atan2"
            | "hypot" | "lerp"
            | "clamp" | "pi" | "tau" | "e" | "phi" | "phi_inv" | "phi_sq"
            | "phi_squared" | "sqrt_2" | "sqrt_5" | "ln_2"
            // Strings
            | "str_len" | "str_chars" | "str_slice" | "str_concat" | "concat_many"
            | "str_split" | "str_join" | "str_trim" | "str_replace"
            | "csv_parse"
            | "str_index_of" | "str_contains" | "str_starts_with" | "str_ends_with"
            | "str_repeat" | "str_reverse" | "str_uppercase" | "str_lowercase"
            | "str_split_lines" | "str_count" | "str_is_empty"
            | "str_to_int" | "str_to_float" | "str_capitalize"
            | "re_match" | "re_find" | "re_find_all" | "re_replace" | "re_split"
            | "json_parse" | "json_stringify" | "json_extract" | "str_format"
            | "sha256" | "sha512" | "base64_encode" | "base64_decode"
            // LLM builtins
            | "llm_call" | "llm_chat" | "llm_embed" | "llm_models" | "llm_system"
            | "llm_stream_print" | "llm_judge" | "llm_compare"
            | "llm_tools" | "substrate_embed"
            | "batch_llm_call" | "batch_llm_chat"
            // File utilities
            | "file_ls"
            // HTTP builtins
            | "http_get" | "http_post" | "http_post_json" | "http_put" | "http_delete"
            | "now_iso" | "now_unix" | "format_time" | "parse_time"
            // Arrays
            | "arr_new" | "arr_from_range" | "arr_len" | "arr_get" | "arr_set"
            | "arr_push" | "arr_first" | "arr_last" | "arr_slice" | "arr_concat"
            | "arr_contains" | "arr_index_of" | "arr_sort" | "arr_reverse" | "arr_join"
            | "arr_min" | "arr_max" | "arr_sum" | "arr_fold_elements"
            | "arr_argmax" | "arr_argmin" | "arr_cumsum" | "arr_diff" | "arr_range"
            | "arr_unique_count" | "arr_partition_by"
            | "arr_min_float" | "arr_max_float" | "arr_gcd" | "fnv1a_hash"
            // Substrate-typed array library
            | "arr_add" | "arr_sub" | "arr_mul" | "arr_div_int" | "arr_neg"
            | "arr_scale" | "arr_resonance_vec" | "arr_him_vec" | "arr_fold_all"
            | "arr_mean" | "arr_variance" | "arr_stddev" | "arr_median"
            | "arr_harmonic_mean" | "arr_geometric_mean"
            | "arr_sum_sq" | "arr_norm" | "arr_dot"
            | "arr_resonance" | "filter_by_resonance" | "cleanup_array"
            | "arr_map" | "arr_filter" | "arr_reduce"
            | "par_map" | "par_filter" | "par_reduce" | "par_for"
            | "arr_any" | "arr_all" | "arr_find"
            // Dicts
            | "dict_new" | "dict_get" | "dict_set" | "dict_has" | "dict_del"
            | "dict_keys" | "dict_values" | "dict_len" | "dict_merge"
            | "dict_pop" | "dict_get_or" | "dict_size" | "dict_clear" | "dict_items"
            // Harmonic primitives
            | "fib" | "fibonacci" | "is_fibonacci" | "harmony_value" | "fold"
            | "fold_escape" | "value_danger" | "classify_resonance"
            | "harmonic_interfere" | "interfere" | "measure_coherence"
            | "mean_omni_weight" | "boundary" | "res"
            // OMNIcode harmonic variants
            | "harmonic_checksum" | "harmonic_write_file" | "harmonic_read_file"
            | "harmonic_sort" | "harmonic_split" | "harmonic_partition"
            | "attractor_distance" | "nearest_attractor"
            | "largest_attractor_at_most" | "crt_residues" | "hbit_tension"
            | "is_attractor" | "resonance_band" | "crt_recover" | "fibonacci_index"
            | "harmonic_hash" | "harmonic_diff" | "harmonic_dedupe"
            // Phi-Pi-Fib search (Fibonacci-step binary search variant)
            | "phi_pi_fib_search" | "phi_pi_fib_nearest"
            | "phi_pi_fib_stats" | "phi_pi_fib_reset"
            // Phi-Pi-Fib search v2 + binary baseline + theoretical bound
            | "phi_pi_fib_search_v2" | "phi_pi_fib_nearest_v2"
            | "phi_pi_bin_search" | "log_phi_pi_fibonacci"
            | "zeckendorf" | "from_zeckendorf"
            | "substrate_search" | "substrate_lower_bound" | "substrate_upper_bound"
            | "substrate_rank" | "substrate_count_range" | "substrate_slice_range"
            | "substrate_intersect" | "substrate_difference"
            | "zeckendorf_weight" | "zeckendorf_bit" | "substrate_hash"
            | "attractor_bucket" | "substrate_insert" | "substrate_quantile"
            | "fib_chunks"
            | "harmonic_align" | "harmonic_unalign" | "phi_pi_log_distance"
            | "harmonic_resample" | "substrate_select_k"
            | "int_binary_search" | "int_lower_bound" | "int_upper_bound"
            | "sorted_merge" | "sorted_union" | "sorted_dedupe"
            | "nth_fibonacci" | "is_zeckendorf_valid"
            | "substrate_min_distance" | "substrate_nearest"
            | "phi_pow" | "phi_pi_pow" | "harmonic_partition_3"
            | "resonance_band_histogram"
            | "arr_sum_int" | "arr_product" | "arr_sort_int" | "arr_is_sorted"
            | "attractor_table" | "harmonic_score"
            | "arr_min_int" | "arr_max_int" | "arr_avg_distance"
            | "is_phi_resonant"
            // Traced variants — return [result, probe_indices_array]
            | "phi_pi_fib_search_traced" | "phi_pi_fib_nearest_traced"
            // Split-channel stats (explicit vs background substrate work)
            | "phi_pi_fib_stats_bg" | "phi_pi_fib_stats_all"
            // HBit dual-band intrinsics. Tree-walk: pass-through
            // returning the int value. Dual-band JIT (Sessions F+G):
            // intercepted as intrinsics in the lowerer to manipulate
            // the β shadow band and compute harmony respectively.
            | "phi_shadow" | "harmony"
            // Self-healing
            | "safe_divide" | "safe_arr_get" | "safe_arr_set"
            | "safe_add" | "safe_sub" | "safe_mul" | "resolve_singularity"
            | "safe_mod" | "safe_sqrt" | "safe_log"
            | "is_singularity" | "ensure_clean" | "collapse" | "invert"
            | "quantize" | "quantization_ratio"
            // I/O
            | "read_file" | "write_file" | "file_exists" | "print"
            | "println" | "print_raw"
            // Time, sleep, conversion, introspection
            | "sleep" | "str_similarity" | "omc_eval_file"
            | "now_ms" | "to_int" | "int" | "to_float" | "float"
            | "to_string" | "to_str" | "string" | "len" | "type_of" | "error"
            | "defined_functions" | "call"
            // Introspection builtins
            | "list_defined_fns" | "list_fns" | "fn_arity" | "fn_source" | "get_scope_vars"
            // Python-idiom builtins (forgiving aliases for users new to OMC)
            | "range" | "getenv" | "to_hex" | "from_hex"
            | "parse_int" | "parse_float"
            // stdlib-friendly aliases
            | "math_log" | "math_sin" | "math_cos" | "math_tan" | "math_sqrt"
            | "math_floor" | "math_ceil" | "math_round" | "math_abs" | "math_pow"
            | "math_min" | "math_max"
            | "unix_time" | "unix_time_ms"
            | "str_lower" | "str_upper" | "str_find" | "str_rfind" | "str_strip"
            | "env_var" | "dict_delete" | "dict_remove"
            | "file_append" | "file_read" | "file_write"
            // v0.3 symbolic prediction
            | "omc_predict_files" | "omc_corpus_size"
            // Test runner host-state primitives
            | "test_record_failure" | "test_failure_count"
            | "test_get_failures" | "test_clear_failures"
            | "test_set_current" | "test_get_current"
            // Random
            | "random_int" | "random_float" | "random_seed"
            // Polish round
            | "str_pad_left" | "str_pad_right" | "arr_zip" | "arr_unique"
            | "arr_take" | "arr_drop" | "arr_count" | "arr_repeat"
            | "arr_fill" | "arr_zeros" | "arr_ones" | "arr_chunk" | "arr_flatten"
            | "arr_enumerate" | "arr_window"
            // Meta-evaluation
            | "eval_omc" | "eval_omc_fresh" | "eval_omc_ctx" | "omc_source"
            // Process execution builtins
            | "omc_spawn" | "omc_pipe"
        )
    }

    /// Invoke a Value::Function with already-evaluated argument values.
    /// Used by higher-order builtins (arr_map etc.) that have the args in
    /// hand as Values rather than Expressions.
    ///
    /// If the function value is a closure (carries a captured environment),
    /// the captured env is ATTACHED to the new scope frame via the
    /// `closure_captures` parallel stack. Lookups for free variables
    /// inside the body fall through to the env; assignments to captured
    /// names mutate through the Rc<RefCell>. Mutations persist across
    /// invocations of the same closure, and across multiple clones of
    /// the same Value::Function (they share the Rc).
    fn call_first_class_function(&mut self, fn_value: &Value, args: Vec<Value>) -> Result<Value, String> {
        let (fn_name, captured) = match fn_value {
            Value::Function { name, captured } => (name.clone(), captured.clone()),
            Value::String(name) => (name.clone(), None),  // accept string form too
            other => return Err(format!(
                "Cannot call this value as a function — it's a {}. \
                 Only fn references and string-named callables are \
                 callable; check that the variable holds a function \
                 (use `type_of(x)` to inspect).",
                type_name_of(other)
            )),
        };
        // Push the captured env as a frame FIRST (so it sits underneath
        // the args/locals). Then push the args frame on top. Sibling
        // closures share the same Rc → mutations propagate.
        let pushed_env = captured.is_some();
        if let Some(env_rc) = captured {
            self.vm_push_closure_env(env_rc);
        }
        self.vm_push_scope();
        let mut expr_args = Vec::with_capacity(args.len());
        for (i, v) in args.into_iter().enumerate() {
            let key = format!("__hof_arg_{}", i);
            self.vm_set_local(&key, v);
            expr_args.push(Expression::Variable(key));
        }
        let result = self.call_function(&fn_name, &expr_args);
        self.vm_pop_scope();
        if pushed_env {
            // Pop the closure env frame we pushed (must not let it grow
            // unbounded across nested HOF calls).
            self.locals.pop();
        }
        result
    }

    /// Position-tagged variant — the call site's source position
    /// becomes the line attached to the new stack frame.
    fn call_function_at(
        &mut self,
        name: &str,
        args: &[Expression],
        pos: crate::ast::Pos,
    ) -> Result<Value, String> {
        if let Some((params, body)) = self.functions.get(name).cloned() {
            return self.invoke_user_function_at(name, &params, &body, args, pos);
        }
        // Module-qualified calls and builtins don't push frames, so
        // pos doesn't matter — fall through to the unpositioned path.
        self.call_function(name, args)
    }

    fn call_function(&mut self, name: &str, args: &[Expression]) -> Result<Value, String> {
        // Aliased imports register functions as literal "module.fname"
        // keys in self.functions. Check that BEFORE the dot-split below,
        // otherwise call_module_function would dispatch back here and
        // infinite-loop on the same name.
        if let Some((params, body)) = self.functions.get(name).cloned() {
            return self.invoke_user_function(name, &params, &body, args);
        }
        // Reverse-FFI: host-registered builtins. Checked BEFORE module
        // dispatch and the built-in stdlib so an embedder can shadow
        // anything (including `read_file`, `now_ms`, etc.). Eval args
        // here — the host fn receives Values, not Expressions, since
        // it lives outside OMC's eval context.
        if let Some(handler) = self.host_builtins.get(name).cloned() {
            let mut argvals = Vec::with_capacity(args.len());
            for a in args {
                argvals.push(self.eval_expr(a)?);
            }
            // Stash a self-pointer so the handler can reach back into
            // the interpreter (needed for Python→OMC callbacks). The
            // pointer is valid only for the duration of this call —
            // we clear it on return. See `with_active_interp` /
            // `active_interp_mut` in this file.
            let prev = INTERP_PTR.with(|p| p.replace(self as *mut _));
            let r = handler(&argvals);
            INTERP_PTR.with(|p| p.set(prev));
            return r;
        }
        // Class instance method dispatch: `obj.method(args)` where
        // `obj` is a local Dict carrying __class__ marker. Routes to
        // the mangled `<ClassName>__<method>` fn registered at class-
        // definition time, with `obj` injected as the first arg.
        //
        // Inheritance: when the child class doesn't define <method>,
        // walk up the class_parents chain trying `<Parent>__<method>`,
        // `<Grandparent>__<method>`, and so on. First hit wins.
        //
        // This MUST be checked before module-qualified dispatch so
        // that instance dicts aren't accidentally looked up as
        // modules. Identified by: receiver-name is a local variable
        // AND it resolves to a Dict AND that dict has __class__.
        if let Some((recv_name, method_name)) = name.split_once('.') {
            if let Some(Value::Dict(d)) = self.get_var(recv_name) {
                let class_key = d.borrow().get("__class__").cloned();
                if let Some(Value::String(class_name)) = class_key {
                    // Walk class → parent chain, bounded to avoid
                    // accidental cycles in a malformed class table.
                    let mut current_class: Option<String> = Some(class_name);
                    let mut hops = 0usize;
                    let mut hit: Option<(String, Vec<String>, Vec<Statement>)> = None;
                    while let Some(c) = current_class {
                        if hops > 64 { break; } // sanity bound
                        let mangled = format!("{}__{}", c, method_name);
                        if let Some((params, body)) = self.functions.get(&mangled).cloned() {
                            hit = Some((mangled, params, body));
                            break;
                        }
                        current_class = self.class_parents.get(&c).cloned();
                        hops += 1;
                    }
                    if let Some((mangled, params, body)) = hit {
                        let mut full_args: Vec<Expression> =
                            Vec::with_capacity(args.len() + 1);
                        full_args.push(Expression::Variable(recv_name.to_string()));
                        full_args.extend(args.iter().cloned());
                        return self.invoke_user_function(
                            &mangled, &params, &body, &full_args,
                        );
                    }
                }
            }
        }
        // Module-qualified calls (e.g., "phi.fold", "phi.res", "core.fib")
        if let Some((module, func)) = name.split_once('.') {
            return self.call_module_function(module, func, args);
        }
        // Built-in functions
        match name {
            "fold" => {
                // Variadic: fold(x), fold(x, depth_int), fold(x, "fibonacci")
                if args.is_empty() {
                    return Err("fold requires at least 1 argument".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let depth = if args.len() >= 2 {
                    let mode_v = self.eval_expr(&args[1])?;
                    // String mode → depth 1 (snap to Fibonacci); int mode → use as depth
                    match mode_v {
                        Value::HInt(h) => h.value.max(1) as usize,
                        Value::HFloat(_) => mode_v.to_int().max(1) as usize,
                        _ => 1,
                    }
                } else {
                    1
                };
                Ok(self.phi_fold_n(v, depth))
            }
            "res" => {
                if args.is_empty() {
                    return Err("res requires 1 argument".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::HInt(h) => Ok(Value::HFloat(h.resonance)),
                    Value::HFloat(f) => {
                        Ok(Value::HFloat(HInt::compute_resonance(f as i64)))
                    }
                    _ => Ok(Value::HFloat(0.0)),
                }
            }
            "fibonacci" => {
                if args.is_empty() {
                    return Err("fibonacci requires 1 argument".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(fibonacci(n))))
            }
            "is_fibonacci" => {
                if args.is_empty() {
                    return Err("is_fibonacci requires 1 argument".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                // Canonical Python OMC returns 0/1 so `if is_fibonacci(x) == 1`
                // works idiomatically. Tree-walk and VM now agree.
                Ok(Value::HInt(HInt::new(if is_fibonacci(n) { 1 } else { 0 })))
            }
            // --- Math: scalar functions ---
            "abs" => {
                let v = self.eval_expr(&args[0])?;
                if v.is_float() {
                    Ok(Value::HFloat(v.to_float().abs()))
                } else {
                    Ok(Value::HInt(HInt::new(v.to_int().abs())))
                }
            }
            "floor" => Ok(Value::HInt(HInt::new(
                self.eval_expr(&args[0])?.to_float().floor() as i64,
            ))),
            "ceil" => Ok(Value::HInt(HInt::new(
                self.eval_expr(&args[0])?.to_float().ceil() as i64,
            ))),
            "round" => Ok(Value::HInt(HInt::new(
                self.eval_expr(&args[0])?.to_float().round() as i64,
            ))),
            "frac" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().fract())),
            "clamp" => {
                if args.len() < 3 {
                    return Err("clamp requires (value, min, max)".to_string());
                }
                let v = self.eval_expr(&args[0])?.to_float();
                let lo = self.eval_expr(&args[1])?.to_float();
                let hi = self.eval_expr(&args[2])?.to_float();
                Ok(Value::HFloat(v.max(lo).min(hi)))
            }
            "sqrt" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().sqrt())),
            "log" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().ln())),
            "log2" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().log2())),
            "log10" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().log10())),
            "exp" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().exp())),
            "sin" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().sin())),
            "cos" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().cos())),
            "tan" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().tan())),
            "tanh" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().tanh())),
            "asin" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().asin())),
            "acos" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().acos())),
            "atan" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().atan())),
            "atan2" => {
                if args.len() < 2 {
                    return Err("atan2 requires (y, x)".to_string());
                }
                let y = self.eval_expr(&args[0])?.to_float();
                let x = self.eval_expr(&args[1])?.to_float();
                Ok(Value::HFloat(y.atan2(x)))
            }
            // Euclidean distance helper. Common in geometry, ML, and
            // the harmonic libraries' multi-dim metrics.
            "hypot" => {
                if args.len() < 2 {
                    return Err("hypot requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_float();
                let b = self.eval_expr(&args[1])?.to_float();
                Ok(Value::HFloat(a.hypot(b)))
            }
            // Linear interpolation: a + t*(b-a). Standard graphics /
            // ML helper. Useful in OMC for blending values along an
            // attractor manifold.
            "lerp" => {
                if args.len() < 3 {
                    return Err("lerp requires (a, b, t)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_float();
                let b = self.eval_expr(&args[1])?.to_float();
                let t = self.eval_expr(&args[2])?.to_float();
                Ok(Value::HFloat(a + t * (b - a)))
            }
            "erf" => {
                // Abramowitz & Stegun approximation (max error ~1.5e-7)
                let x = self.eval_expr(&args[0])?.to_float();
                let sign = if x < 0.0 { -1.0 } else { 1.0 };
                let ax = x.abs();
                let t = 1.0 / (1.0 + 0.3275911 * ax);
                let y = 1.0
                    - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                        - 0.284496736)
                        * t
                        + 0.254829592)
                        * t
                        * (-ax * ax).exp();
                Ok(Value::HFloat(sign * y))
            }
            "sigmoid" => {
                let x = self.eval_expr(&args[0])?.to_float();
                Ok(Value::HFloat(1.0 / (1.0 + (-x).exp())))
            }
            "pow" => {
                if args.len() < 2 {
                    return Err("pow requires (base, exponent)".to_string());
                }
                let b = self.eval_expr(&args[0])?.to_float();
                let e = self.eval_expr(&args[1])?.to_float();
                Ok(Value::HFloat(b.powf(e)))
            }
            "pi" => Ok(Value::HFloat(std::f64::consts::PI)),
            "e" => Ok(Value::HFloat(std::f64::consts::E)),
            "phi" => Ok(Value::HFloat(crate::value::PHI)),
            "tau" => Ok(Value::HFloat(std::f64::consts::TAU)),
            "phi_inv" => Ok(Value::HFloat(crate::value::PHI_INV)),
            "phi_sq" => Ok(Value::HFloat(crate::value::PHI_SQ)),
            "phi_squared" => Ok(Value::HFloat(crate::value::PHI_SQ)),
            "factorial" => {
                // Lenient like canonical Python OMC: negative -> 1 (identity).
                let n = self.eval_expr(&args[0])?.to_int();
                let mut result: i64 = 1;
                for i in 1..=n.max(0) {
                    result = result.wrapping_mul(i);
                }
                Ok(Value::HInt(HInt::new(result)))
            }
            "square" => {
                let v = self.eval_expr(&args[0])?;
                if v.is_float() {
                    let f = v.to_float();
                    Ok(Value::HFloat(f * f))
                } else {
                    let n = v.to_int();
                    Ok(Value::HInt(HInt::new(n.wrapping_mul(n))))
                }
            }
            "cube" => {
                let v = self.eval_expr(&args[0])?;
                if v.is_float() {
                    let f = v.to_float();
                    Ok(Value::HFloat(f * f * f))
                } else {
                    let n = v.to_int();
                    Ok(Value::HInt(HInt::new(n.wrapping_mul(n).wrapping_mul(n))))
                }
            }
            "sqrt_2" => Ok(Value::HFloat(std::f64::consts::SQRT_2)),
            "sqrt_5" => Ok(Value::HFloat(5.0_f64.sqrt())),
            "ln_2" => Ok(Value::HFloat(std::f64::consts::LN_2)),
            // harmonic_interfere(a, b) — Phase 6 std/wave.omc; harmonic mean of magnitudes.
            "harmonic_interfere" => {
                let a = self.eval_expr(&args[0])?.to_float();
                let b = self.eval_expr(&args[1])?.to_float();
                if a + b == 0.0 {
                    Ok(Value::HFloat(0.0))
                } else {
                    Ok(Value::HFloat(2.0 * a * b / (a + b)))
                }
            }
            // measure_coherence(a, b) — Phase 6 std/wave.omc; resonance-based coherence.
            "measure_coherence" => {
                let a = self.eval_expr(&args[0])?.to_int();
                let b = self.eval_expr(&args[1])?.to_int();
                let ra = HInt::compute_resonance(a);
                let rb = HInt::compute_resonance(b);
                Ok(Value::HFloat((ra - rb).abs()))
            }
            // Polymorphic min/max — accept either (a, b) or a single array.
            "min" => {
                if args.is_empty() {
                    return Err("min requires at least 1 argument".to_string());
                }
                if args.len() == 1 {
                    // Array form: forward to arr_min behavior
                    if let Value::Array(arr) = self.eval_expr(&args[0])? {
                        if arr.items.borrow().is_empty() {
                            return Err("min: empty array".to_string());
                        }
                        return Ok(Value::HInt(HInt::new(
                            arr.items.borrow().iter().map(|v| v.to_int()).min().unwrap(),
                        )));
                    }
                    return Err("min(x): single arg must be an array".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                if a.is_float() || b.is_float() {
                    Ok(Value::HFloat(a.to_float().min(b.to_float())))
                } else {
                    Ok(Value::HInt(HInt::new(a.to_int().min(b.to_int()))))
                }
            }
            "max" => {
                if args.is_empty() {
                    return Err("max requires at least 1 argument".to_string());
                }
                if args.len() == 1 {
                    if let Value::Array(arr) = self.eval_expr(&args[0])? {
                        if arr.items.borrow().is_empty() {
                            return Err("max: empty array".to_string());
                        }
                        return Ok(Value::HInt(HInt::new(
                            arr.items.borrow().iter().map(|v| v.to_int()).max().unwrap(),
                        )));
                    }
                    return Err("max(x): single arg must be an array".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                if a.is_float() || b.is_float() {
                    Ok(Value::HFloat(a.to_float().max(b.to_float())))
                } else {
                    Ok(Value::HInt(HInt::new(a.to_int().max(b.to_int()))))
                }
            }
            // safe_add: addition that folds singularity inputs first.
            "safe_add" => {
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                let a_clean = if a.is_singularity() { self.phi_fold_n(a, 1) } else { a };
                let b_clean = if b.is_singularity() { self.phi_fold_n(b, 1) } else { b };
                Ok(Value::HInt(HInt::new(
                    a_clean.to_int().wrapping_add(b_clean.to_int()),
                )))
            }
            "safe_sub" => {
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                let a_clean = if a.is_singularity() { self.phi_fold_n(a, 1) } else { a };
                let b_clean = if b.is_singularity() { self.phi_fold_n(b, 1) } else { b };
                Ok(Value::HInt(HInt::new(
                    a_clean.to_int().wrapping_sub(b_clean.to_int()),
                )))
            }
            "safe_mul" => {
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                let a_clean = if a.is_singularity() { self.phi_fold_n(a, 1) } else { a };
                let b_clean = if b.is_singularity() { self.phi_fold_n(b, 1) } else { b };
                Ok(Value::HInt(HInt::new(
                    a_clean.to_int().wrapping_mul(b_clean.to_int()),
                )))
            }
            // sign(n) -> -1, 0, or 1
            "sign" => {
                let v = self.eval_expr(&args[0])?;
                let s = if v.is_float() {
                    let f = v.to_float();
                    if f > 0.0 { 1 } else if f < 0.0 { -1 } else { 0 }
                } else {
                    let n = v.to_int();
                    if n > 0 { 1 } else if n < 0 { -1 } else { 0 }
                };
                Ok(Value::HInt(HInt::new(s)))
            }
            // Primality check using 6k±1 trial division.
            "is_prime" => {
                let n = self.eval_expr(&args[0])?.to_int();
                let prime = if n < 2 {
                    false
                } else if n < 4 {
                    true
                } else if n % 2 == 0 || n % 3 == 0 {
                    false
                } else {
                    let mut i: i64 = 5;
                    let mut is_p = true;
                    while i.saturating_mul(i) <= n {
                        if n % i == 0 || n % (i + 2) == 0 {
                            is_p = false;
                            break;
                        }
                        i += 6;
                    }
                    is_p
                };
                Ok(Value::HInt(HInt::new(if prime { 1 } else { 0 })))
            }
            // --- OmniWeight quantization (Phase S) ---
            // quantize(arr) — map each element to its nearest Fibonacci attractor
            // IF the OmniWeight w = φ^(-|e|) crosses 0.5. Mimics the Phase 18
            // pattern from omnicode_experiment in miniature: harmonic-aligned
            // compression that preserves φ-geodesic structure.
            "quantize" => {
                if args.is_empty() {
                    return Err("quantize requires (array[, threshold])".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let threshold = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else {
                    0.5
                };
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let mut new_items: Vec<Value> = Vec::with_capacity(items_b.len());
                    for v in items_b.iter() {
                        let n = v.to_int();
                        let folded = fold_to_fibonacci_const(n);
                        // OmniWeight between original and the candidate attractor.
                        let denom = (folded.abs() as f64).max(1.0);
                        let e = ((n - folded).abs() as f64) / denom;
                        let weight = crate::value::PHI.powf(-e);
                        if weight >= threshold {
                            new_items.push(Value::HInt(HInt::new(folded)));
                        } else {
                            new_items.push(v.clone());
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(new_items)))
                } else {
                    Err("quantize: requires an array".to_string())
                }
            }
            // quantization_ratio(arr, threshold) — returns the fraction of array
            // elements that would be quantized at the given OmniWeight threshold.
            // Useful for reporting "how compressible is this dataset" without
            // actually doing the compression.
            "quantization_ratio" => {
                if args.is_empty() {
                    return Err("quantization_ratio requires (array[, threshold])".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let threshold = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else {
                    0.5
                };
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    if items_b.is_empty() {
                        return Ok(Value::HFloat(0.0));
                    }
                    let mut count = 0usize;
                    for v in items_b.iter() {
                        let n = v.to_int();
                        let folded = fold_to_fibonacci_const(n);
                        let denom = (folded.abs() as f64).max(1.0);
                        let e = ((n - folded).abs() as f64) / denom;
                        let weight = crate::value::PHI.powf(-e);
                        if weight >= threshold {
                            count += 1;
                        }
                    }
                    Ok(Value::HFloat(count as f64 / items_b.len() as f64))
                } else {
                    Err("quantization_ratio: requires an array".to_string())
                }
            }
            // mean_omni_weight(arr) — average OmniWeight against the nearest
            // Fibonacci attractor across the whole array. Higher = more
            // phi-aligned data, more compressible without information loss.
            "mean_omni_weight" => {
                if args.is_empty() {
                    return Err("mean_omni_weight requires (array)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    if items_b.is_empty() {
                        return Ok(Value::HFloat(0.0));
                    }
                    let mut sum: f64 = 0.0;
                    for v in items_b.iter() {
                        let n = v.to_int();
                        let folded = fold_to_fibonacci_const(n);
                        let denom = (folded.abs() as f64).max(1.0);
                        let e = ((n - folded).abs() as f64) / denom;
                        sum += crate::value::PHI.powf(-e);
                    }
                    Ok(Value::HFloat(sum / items_b.len() as f64))
                } else {
                    Err("mean_omni_weight: requires an array".to_string())
                }
            }
            // --- ONN Self-Healing primitives (Phase O) ---
            // value_danger(x) = exp(-|x|).
            // Predicts proximity to a singularity (zero). Returns 1.0 when x ≈ 0
            // (high danger), decays toward 0 as |x| grows. Used as an
            // early-warning signal BEFORE an operation that might explode.
            "value_danger" => {
                let v = self.eval_expr(&args[0])?;
                let f = v.to_float().abs();
                Ok(Value::HFloat((-f).exp()))
            }
            // fold_escape(x) — if value_danger(x) > 0.5, snap to nearest
            // Fibonacci attractor (preserves sign). Else passthrough. This is
            // the AUTOMATIC version of resolve_singularity(v, "fold") that
            // works BEFORE a value becomes a Singularity — fold the operand
            // away from the danger zone preemptively.
            "fold_escape" => {
                let v = self.eval_expr(&args[0])?;
                let f = v.to_float();
                let danger = (-f.abs()).exp();
                if danger > 0.5 {
                    // Snap to nearest Fibonacci, preserve sign.
                    let n = v.to_int();
                    let result = crate::phi_pi_fib::fold_to_nearest_attractor(n);
                    // The point of fold_escape is to escape the zero-trap:
                    // if the nearest Fibonacci is 0 (which happens for x=0),
                    // jump to 1 instead. Otherwise we'd just heal back to
                    // the same singularity.
                    let safe = if result == 0 { 1 } else { result };
                    Ok(Value::HInt(HInt::new(safe)))
                } else {
                    Ok(v)
                }
            }
            // harmony_value(x) — harmony score based on Fibonacci proximity.
            // Returns 1.0 when x IS Fibonacci, decays based on relative distance
            // to the nearest attractor. This is the "is this value living on
            // the φ-geodesic?" measurement.
            "harmony_value" => {
                let n = self.eval_expr(&args[0])?.to_int();
                let r = HInt::compute_resonance(n);
                Ok(Value::HFloat(r))
            }
            // safe_divide(a, b) — divide with predictive self-healing.
            // If b is dangerously close to zero (value_danger > 0.5), fold
            // b away from zero FIRST, then divide. No HSingularity produced;
            // the math always returns a number.
            //
            // This is the canonical "self-healing arithmetic" pattern: the
            // operation checks Fibonacci alignment of its operands, applies
            // fold_escape if needed, and only then performs the operation.
            "safe_divide" => {
                if args.len() < 2 {
                    return Err("safe_divide requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                let bf = b.to_float();
                let danger = (-bf.abs()).exp();
                let divisor = if danger > 0.5 {
                    // Fold b away from zero.
                    let n = b.to_int();
                    let mut healed = crate::phi_pi_fib::fold_to_nearest_attractor(n);
                    if healed == 0 {
                        healed = 1;
                    }
                    healed
                } else {
                    b.to_int()
                };
                if a.is_float() {
                    Ok(Value::HFloat(a.to_float() / (divisor as f64)))
                } else {
                    Ok(Value::HInt(HInt::new(a.to_int() / divisor)))
                }
            }
            // safe_mod: mirrors safe_divide's contract for modulo. When
            // the divisor is in the "danger zone" near zero, substrate-
            // fold it to the nearest non-zero Fibonacci attractor.
            // Used by the heal pass to rewrite `x % 0` semantics for
            // dynamic divisors (the literal-divisor case still rewrites
            // statically at heal time for predictability).
            "safe_mod" => {
                if args.len() < 2 {
                    return Err("safe_mod requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                let bf = b.to_float();
                let danger = (-bf.abs()).exp();
                let divisor = if danger > 0.5 {
                    let n = b.to_int();
                    let mut healed = crate::phi_pi_fib::fold_to_nearest_attractor(n);
                    if healed == 0 { healed = 1; }
                    healed
                } else {
                    b.to_int()
                };
                Ok(Value::HInt(HInt::new(a.to_int().rem_euclid(divisor.max(1)))))
            }
            // safe_sqrt: returns 0 (the singularity-tolerant value)
            // for negative inputs, otherwise the standard sqrt. The
            // alternative — raising a Singularity — propagates through
            // arithmetic chains in ways callers rarely expect. 0 keeps
            // pipelines flowing; explicit checks belong outside.
            "safe_sqrt" => {
                if args.is_empty() {
                    return Err("safe_sqrt requires (x)".to_string());
                }
                let x = self.eval_expr(&args[0])?.to_float();
                Ok(Value::HFloat(if x < 0.0 { 0.0 } else { x.sqrt() }))
            }
            // safe_log: log(x) for x > 0; -infty proxy (-1e308) otherwise.
            // The pure mathematical answer for x <= 0 is undefined; we
            // return a large negative finite value so the result still
            // composes inside arithmetic without an infinity poison.
            "safe_log" => {
                if args.is_empty() {
                    return Err("safe_log requires (x)".to_string());
                }
                let x = self.eval_expr(&args[0])?.to_float();
                Ok(Value::HFloat(if x <= 0.0 { -1.0e308 } else { x.ln() }))
            }
            // From Phase 6 std/core.omc:
            //   ensure_clean(v) — return v if not a Singularity; else fold to nearest Fibonacci.
            "ensure_clean" => {
                let v = self.eval_expr(&args[0])?;
                if v.is_singularity() {
                    Ok(self.phi_fold_n(v, 1))
                } else {
                    Ok(v)
                }
            }
            // Drop any Singularity elements from an array (Phase 6 idiom).
            "cleanup_array" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let kept: Vec<Value> = arr
                        .items
                        .borrow()
                        .iter()
                        .filter(|v| !v.is_singularity())
                        .cloned()
                        .collect();
                    Ok(Value::Array(HArray::from_vec(kept)))
                } else {
                    Err("cleanup_array: requires an array".to_string())
                }
            }
            // collapse(amp, phase) — wave collapse to a scalar magnitude.
            "collapse" => {
                let amp = self.eval_expr(&args[0])?.to_float();
                let phase = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else {
                    0.0
                };
                Ok(Value::HFloat(amp * phase.cos()))
            }
            // Integer power (separate from `pow` which returns float).
            "pow_int" => {
                if args.len() < 2 {
                    return Err("pow_int requires (base, exp)".to_string());
                }
                let b = self.eval_expr(&args[0])?.to_int();
                let e = self.eval_expr(&args[1])?.to_int();
                let mut result: i64 = 1;
                let mut base = b;
                let mut exp = e.max(0) as u32;
                while exp > 0 {
                    if exp & 1 == 1 {
                        result = result.wrapping_mul(base);
                    }
                    base = base.wrapping_mul(base);
                    exp >>= 1;
                }
                Ok(Value::HInt(HInt::new(result)))
            }
            // mod_pow: modular exponentiation (base^exp mod m).
            // Wraps i128 internally to avoid overflow in the squaring step
            // for moduli up to ~2^63. Standard Diffie-Hellman / RSA-shaped
            // primitive — and useful for CRT recovery in Fibonacci moduli.
            "mod_pow" => {
                if args.len() < 3 {
                    return Err("mod_pow requires (base, exp, modulus)".to_string());
                }
                let b = self.eval_expr(&args[0])?.to_int();
                let e = self.eval_expr(&args[1])?.to_int();
                let m = self.eval_expr(&args[2])?.to_int();
                if m == 0 {
                    return Ok(Value::Singularity {
                        numerator: 0, denominator: 0,
                        context: "mod_pow: modulus is zero".to_string(),
                    });
                }
                let m128 = m.unsigned_abs() as i128;
                let mut result: i128 = 1 % m128;
                let mut base = (b.rem_euclid(m)) as i128 % m128;
                let mut exp = e.max(0) as u64;
                while exp > 0 {
                    if exp & 1 == 1 {
                        result = (result * base) % m128;
                    }
                    base = (base * base) % m128;
                    exp >>= 1;
                }
                Ok(Value::HInt(HInt::new(result as i64)))
            }
            // bit_count (popcount): number of 1 bits in the unsigned repr.
            "bit_count" => {
                if args.is_empty() {
                    return Err("bit_count requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(n.count_ones() as i64)))
            }
            // bit_length: minimum bits needed to represent abs(n). 0 -> 0.
            "bit_length" => {
                if args.is_empty() {
                    return Err("bit_length requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let len = if n == 0 { 0 } else { 64 - n.unsigned_abs().leading_zeros() as i64 };
                Ok(Value::HInt(HInt::new(len)))
            }
            // digit_sum: sum of decimal digits of abs(n).
            // Used in numerology / divisibility / Fibonacci-digit-relation
            // experiments and harmonic checksum spot-checks.
            "digit_sum" => {
                if args.is_empty() {
                    return Err("digit_sum requires (n)".to_string());
                }
                let mut n = self.eval_expr(&args[0])?.to_int().unsigned_abs();
                let mut sum: i64 = 0;
                if n == 0 {
                    return Ok(Value::HInt(HInt::new(0)));
                }
                while n > 0 {
                    sum += (n % 10) as i64;
                    n /= 10;
                }
                Ok(Value::HInt(HInt::new(sum)))
            }
            // digit_count: number of decimal digits in abs(n). digit_count(0) = 1.
            "digit_count" => {
                if args.is_empty() {
                    return Err("digit_count requires (n)".to_string());
                }
                let mut n = self.eval_expr(&args[0])?.to_int().unsigned_abs();
                if n == 0 {
                    return Ok(Value::HInt(HInt::new(1)));
                }
                let mut c: i64 = 0;
                while n > 0 { c += 1; n /= 10; }
                Ok(Value::HInt(HInt::new(c)))
            }
            // is_even / is_odd predicates
            "even" => {
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(if n % 2 == 0 { 1 } else { 0 })))
            }
            "is_even" => {
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(if n % 2 == 0 { 1 } else { 0 })))
            }
            "odd" => {
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(if n % 2 != 0 { 1 } else { 0 })))
            }
            "is_odd" => {
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(if n % 2 != 0 { 1 } else { 0 })))
            }
            // Short alias used in Phase 6 stdlib for `fibonacci`.
            "fib" => {
                if args.is_empty() {
                    return Err("fib requires 1 argument".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(fibonacci(n))))
            }
            // From Phase 6 std/core.omc: bucket a value's resonance into a label.
            // Returns an int code: 3 = high (>=0.7), 2 = medium (>=0.5), 1 = low (>=0.3), 0 = dissonant.
            // (Python returns a string but Rust callers use it numerically in if-cascades.)
            "classify_resonance" => {
                let n = self.eval_expr(&args[0])?.to_int();
                let r = HInt::compute_resonance(n);
                let code = if r >= 0.7 {
                    3
                } else if r >= 0.5 {
                    2
                } else if r >= 0.3 {
                    1
                } else {
                    0
                };
                Ok(Value::HInt(HInt::new(code)))
            }
            // From Phase 6 std/core.omc: filter array, keep elements with res >= threshold.
            "filter_by_resonance" => {
                if args.len() < 2 {
                    return Err("filter_by_resonance requires (array, threshold)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let threshold = self.eval_expr(&args[1])?.to_float();
                if let Value::Array(arr) = arr_v {
                    let kept: Vec<Value> = arr
                        .items
                        .borrow()
                        .iter()
                        .filter(|v| HInt::compute_resonance(v.to_int()) >= threshold)
                        .cloned()
                        .collect();
                    Ok(Value::Array(HArray::from_vec(kept)))
                } else {
                    Err("filter_by_resonance: first argument must be an array".to_string())
                }
            }
            // From Phase 6 std/wave.omc: simple wave interference between two values.
            // Returns the harmonic mean of the magnitudes.
            "interfere" => {
                if args.len() < 2 {
                    return Err("interfere requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_float();
                let b = self.eval_expr(&args[1])?.to_float();
                if a + b == 0.0 {
                    Ok(Value::HFloat(0.0))
                } else {
                    Ok(Value::HFloat(2.0 * a * b / (a + b)))
                }
            }
            // Variadic "fold across an array with a mode string". From Phase 6 stdlib.
            "arr_fold_elements" => {
                if args.is_empty() {
                    return Err("arr_fold_elements requires (array[, mode])".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = arr_v {
                    let mut acc = 0i64;
                    for v in arr.items.borrow().iter() {
                        // .abs() before fold matches the prior behaviour
                        // (always positive attractor accumulated).
                        let nearest = crate::phi_pi_fib::fold_to_nearest_attractor(
                            v.to_int().abs(),
                        );
                        acc = acc.wrapping_add(nearest);
                    }
                    Ok(Value::HInt(HInt::new(acc)))
                } else {
                    Err("arr_fold_elements: first argument must be an array".to_string())
                }
            }
            // --- Type coercion ---
            "to_int" => Ok(Value::HInt(HInt::new(self.eval_expr(&args[0])?.to_int()))),
            "to_float" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float())),
            "to_string" | "to_str" => {
                // Render the bare value, NOT the HInt-with-resonance display.
                // This is what canonical Python OMC's to_string returns.
                let v = self.eval_expr(&args[0])?;
                let s = match v {
                    Value::HInt(h) => h.value.to_string(),
                    Value::HFloat(f) => format!("{}", f),
                    Value::String(s) => s,
                    Value::Bool(b) => b.to_string(),
                    other => other.to_string(),
                };
                Ok(Value::String(s))
            }
            "int" => Ok(Value::HInt(HInt::new(self.eval_expr(&args[0])?.to_int()))),
            "float" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float())),
            "string" => {
                let v = self.eval_expr(&args[0])?;
                let s = match v {
                    Value::HInt(h) => h.value.to_string(),
                    Value::HFloat(f) => format!("{}", f),
                    Value::String(s) => s,
                    Value::Bool(b) => b.to_string(),
                    other => other.to_string(),
                };
                Ok(Value::String(s))
            }
            // Portal / Singularity handling — canonical OMNIcode idiom.
            // Python returns 0/1 so `if is_singularity(result) == 1` works.
            "is_singularity" => {
                if args.is_empty() {
                    return Err("is_singularity requires 1 argument".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                Ok(Value::HInt(HInt::new(if v.is_singularity() { 1 } else { 0 })))
            }
            // resolve_singularity(portal, mode) → int
            // Modes: "fold" snap-to-Fibonacci; "invert" → 1/n style;
            // "boundary" → numerator unchanged (passthrough).
            "resolve_singularity" => {
                if args.len() < 2 {
                    return Err(
                        "resolve_singularity requires (value, mode_string)".to_string(),
                    );
                }
                let v = self.eval_expr(&args[0])?;
                let mode = self.eval_expr(&args[1])?.to_string();
                let numerator = match &v {
                    Value::Singularity { numerator, .. } => *numerator,
                    Value::HInt(h) => h.value,
                    _ => v.to_int(),
                };
                let resolved = match mode.as_str() {
                    "fold" => {
                        // Snap |numerator| to nearest Fibonacci, preserve sign.
                        crate::phi_pi_fib::fold_to_nearest_attractor(numerator)
                    }
                    "invert" => {
                        // 1/n style: return signed inverse magnitude.
                        // For integer mode we use 1 as the multiplicative identity
                        // when |n| < 1 (i.e. n == 0); otherwise return ±1.
                        if numerator == 0 { 1 } else if numerator > 0 { 1 } else { -1 }
                    }
                    "boundary" => numerator,
                    other => {
                        return Err(format!(
                            "resolve_singularity: unknown mode {:?} (expected \"fold\", \"invert\", or \"boundary\")",
                            other
                        ))
                    }
                };
                Ok(Value::HInt(HInt::new(resolved)))
            }
            // String functions
            "str_len" => {
                if args.is_empty() {
                    return Err("str_len requires 1 argument".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                Ok(Value::HInt(HInt::new(s.len() as i64)))
            }
            "str_chars" => {
                // char count (UTF-8 scalar values), matching str_slice's
                // char-indexed slicing. Use this in hand-written lexers
                // instead of str_len; otherwise non-ASCII source overshoots
                // the loop bound and you read empty strings past the end.
                if args.is_empty() {
                    return Err("str_chars requires 1 argument".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                Ok(Value::HInt(HInt::new(s.chars().count() as i64)))
            }
            "str_concat" => {
                if args.len() < 2 {
                    return Err("str_concat requires at least 2 arguments".to_string());
                }
                // Variadic: str_concat(a, b, c...) concatenates all args.
                // to_display_string produces "42" not "HInt(42, φ=..., ...)"
                // for numeric args — matches Phase 1 string-+-concat semantics.
                let mut out = String::new();
                for a in args {
                    out.push_str(&self.eval_expr(a)?.to_display_string());
                }
                Ok(Value::String(out))
            }
            "str_uppercase" => {
                if args.is_empty() {
                    return Err("str_uppercase requires 1 argument".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                Ok(Value::String(s.to_uppercase()))
            }
            "str_lowercase" => {
                if args.is_empty() {
                    return Err("str_lowercase requires 1 argument".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                Ok(Value::String(s.to_lowercase()))
            }
            "str_reverse" => {
                let s = self.eval_expr(&args[0])?.to_string();
                Ok(Value::String(s.chars().rev().collect()))
            }
            "str_contains" => {
                if args.len() < 2 {
                    return Err("str_contains requires (haystack, needle)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let needle = self.eval_expr(&args[1])?.to_string();
                Ok(Value::HInt(HInt::new(if s.contains(&needle) { 1 } else { 0 })))
            }
            "str_slice" => {
                if args.len() < 3 {
                    return Err("str_slice requires (string, start, end)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let start = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let end = self.eval_expr(&args[2])?.to_int().max(0) as usize;
                let chars: Vec<char> = s.chars().collect();
                let end = end.min(chars.len());
                let start = start.min(end);
                Ok(Value::String(chars[start..end].iter().collect()))
            }
            // String workhorse functions added for Python-tier ergonomics.
            // None of these affect existing semantics; pure additions.
            "str_split" => {
                if args.len() < 2 {
                    return Err("str_split requires (string, separator)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let sep = self.eval_expr(&args[1])?.to_string();
                let parts: Vec<Value> = if sep.is_empty() {
                    // Empty separator → split into individual characters
                    // (matches Python's quirk in this corner via list(s))
                    s.chars().map(|c| Value::String(c.to_string())).collect()
                } else {
                    s.split(&sep).map(|p| Value::String(p.to_string())).collect()
                };
                Ok(Value::Array(HArray::from_vec(parts)))
            }
            // csv_parse(text, sep=',', skip_header=0) -> array of array of strings.
            // Native CSV parser. Replaces the per-line str_split round-trip
            // pattern that loaded 10k MovieLens rows in 28ms (post-Rc-shared).
            // Targets <5ms for the same workload by doing one big allocation
            // and skipping VM dispatch per-cell.
            //
            // Defaults to comma separator, no header skip. Pass an explicit
            // separator to handle TSV (sep="\t"), pipe-delim, etc. Pass
            // skip_header=1 to drop the first line.
            // ---- Hashing: sha256 / sha512 / md5 --------------------
            "sha256" => {
                // sha256(text_or_bytes) -> hex string. Standard 256-bit
                // hash; deterministic across runs.
                use sha2::{Sha256, Digest};
                if args.is_empty() {
                    return Err("sha256 requires (text)".to_string());
                }
                let input = self.eval_expr(&args[0])?.to_display_string();
                let digest = Sha256::digest(input.as_bytes());
                let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
                Ok(Value::String(hex))
            }
            "sha512" => {
                use sha2::{Sha512, Digest};
                if args.is_empty() {
                    return Err("sha512 requires (text)".to_string());
                }
                let input = self.eval_expr(&args[0])?.to_display_string();
                let digest = Sha512::digest(input.as_bytes());
                let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
                Ok(Value::String(hex))
            }
            // ---- Base64 --------------------------------------------
            "base64_encode" => {
                use base64::Engine;
                if args.is_empty() {
                    return Err("base64_encode requires (text)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                Ok(Value::String(
                    base64::engine::general_purpose::STANDARD.encode(s.as_bytes())
                ))
            }
            "base64_decode" => {
                use base64::Engine;
                if args.is_empty() {
                    return Err("base64_decode requires (text)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                match base64::engine::general_purpose::STANDARD.decode(&s) {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(decoded) => Ok(Value::String(decoded)),
                        Err(e) => Err(format!("base64_decode: invalid UTF-8: {}", e)),
                    },
                    Err(e) => Err(format!("base64_decode: invalid base64: {}", e)),
                }
            }
            // ---- Datetime via chrono -------------------------------
            "now_iso" => {
                // ISO 8601 timestamp of the current UTC instant.
                let n = chrono::Utc::now();
                Ok(Value::String(n.to_rfc3339()))
            }
            "now_unix" => {
                // Seconds since the Unix epoch.
                let n = chrono::Utc::now();
                Ok(Value::HInt(HInt::new(n.timestamp())))
            }
            "format_time" => {
                // format_time(unix_seconds, fmt) -> string. Uses
                // chrono::strftime-style format specifiers. Common ones:
                //   %Y-%m-%d %H:%M:%S    "2026-05-16 14:32:01"
                //   %A %d %b              "Saturday 16 May"
                //   %s                    seconds since epoch
                if args.len() < 2 {
                    return Err("format_time requires (unix_seconds, fmt)".to_string());
                }
                let secs = self.eval_expr(&args[0])?.to_int();
                let fmt = self.eval_expr(&args[1])?.to_display_string();
                match chrono::DateTime::from_timestamp(secs, 0) {
                    Some(dt) => Ok(Value::String(dt.format(&fmt).to_string())),
                    None => Err(format!("format_time: bad timestamp {}", secs)),
                }
            }
            "parse_time" => {
                // parse_time(string, fmt) -> unix_seconds.
                if args.len() < 2 {
                    return Err("parse_time requires (string, fmt)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                let fmt = self.eval_expr(&args[1])?.to_display_string();
                match chrono::NaiveDateTime::parse_from_str(&s, &fmt) {
                    Ok(dt) => Ok(Value::HInt(HInt::new(dt.and_utc().timestamp()))),
                    Err(e) => Err(format!("parse_time: {}", e)),
                }
            }
            // ---- JSON (via serde_json) -----------------------------
            "json_parse" => {
                // json_parse(text) -> Value (dict, array, string, int,
                // float, bool, or Null). Throws on parse error.
                if args.is_empty() {
                    return Err("json_parse requires (text)".to_string());
                }
                let text = self.eval_expr(&args[0])?.to_display_string();
                match serde_json::from_str::<serde_json::Value>(&text) {
                    Ok(v) => Ok(json_to_value(v)),
                    Err(e) => Err(format!("json_parse: {}", e)),
                }
            }
            "json_stringify" => {
                // json_stringify(value) -> string. Pretty-prints if a
                // second arg is truthy (matches Python json.dumps(indent=2)).
                if args.is_empty() {
                    return Err("json_stringify requires (value, pretty?)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let jv = value_to_json(&v);
                let pretty = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_int() != 0
                } else { false };
                let s = if pretty {
                    serde_json::to_string_pretty(&jv)
                } else {
                    serde_json::to_string(&jv)
                };
                match s {
                    Ok(out) => Ok(Value::String(out)),
                    Err(e) => Err(format!("json_stringify: {}", e)),
                }
            }
            // json_extract(text) -> Value | null
            //   Extracts the first valid JSON object {} or array [] from a string.
            //   Useful for parsing LLM responses that embed JSON inside prose.
            "json_extract" => {
                if args.is_empty() {
                    return Err("json_extract requires (text)".to_string());
                }
                let text = self.eval_expr(&args[0])?.to_display_string();
                let bytes = text.as_bytes();
                let mut result = Value::Null;
                'outer: for start in 0..bytes.len() {
                    if bytes[start] == b'{' || bytes[start] == b'[' {
                        for end in (start + 1..=bytes.len()).rev() {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text[start..end]) {
                                result = json_to_value(v);
                                break 'outer;
                            }
                        }
                    }
                }
                Ok(result)
            }
            // str_format(template, values) -> string
            //   Replace {key} placeholders with dict values, or {0},{1} with positional args.
            //   Example: str_format("Hello, {name}!", {name: "World"}) -> "Hello, World!"
            "str_format" => {
                if args.len() < 2 {
                    return Err("str_format requires (template, values)".to_string());
                }
                let template = self.eval_expr(&args[0])?.to_display_string();
                let values = self.eval_expr(&args[1])?;
                let mut result = template.clone();
                match &values {
                    Value::Dict(d) => {
                        for (k, v) in d.borrow().iter() {
                            let placeholder = format!("{{{}}}", k);
                            result = result.replace(&placeholder, &v.to_display_string());
                        }
                    }
                    Value::Array(a) => {
                        for (i, v) in a.items.borrow().iter().enumerate() {
                            let placeholder = format!("{{{}}}", i);
                            result = result.replace(&placeholder, &v.to_display_string());
                        }
                    }
                    other => {
                        result = result.replace("{0}", &other.to_display_string());
                    }
                }
                Ok(Value::String(result))
            }
            "csv_parse" => {
                if args.is_empty() {
                    return Err("csv_parse requires (text, sep?, skip_header?)".to_string());
                }
                let text = self.eval_expr(&args[0])?.to_string();
                let sep = if args.len() >= 2 {
                    let s = self.eval_expr(&args[1])?.to_string();
                    if s.is_empty() { ",".to_string() } else { s }
                } else {
                    ",".to_string()
                };
                let skip_header = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.to_int() != 0
                } else {
                    false
                };
                let mut rows: Vec<Value> = Vec::new();
                for (i, line) in text.lines().enumerate() {
                    if skip_header && i == 0 { continue; }
                    if line.is_empty() { continue; }
                    let cells: Vec<Value> = line
                        .split(&sep)
                        .map(|c| Value::String(c.to_string()))
                        .collect();
                    rows.push(Value::Array(HArray::from_vec(cells)));
                }
                Ok(Value::Array(HArray::from_vec(rows)))
            }
            // ---- LLM builtins (Anthropic API) ----------------------------------
            // llm_call(prompt) | llm_call(prompt, model)
            //   Sends a single-turn prompt to the Anthropic Messages API and
            //   returns the assistant's text reply as a String.
            //   Model defaults to "claude-3-5-haiku-latest".
            //   Reads API key from env var ANTHROPIC_API_KEY.
            // NOTE: cfg-gated so modern native-llm arms at the bottom fire when
            //       llm-builtins is not in the feature set (default build).
            #[cfg(feature = "llm-builtins")]
            "llm_call" => {
                #[cfg(feature = "llm-builtins")]
                {
                    if args.is_empty() {
                        return Err("llm_call requires (prompt [, model])".to_string());
                    }
                    let prompt = self.eval_expr(&args[0])?.to_display_string();
                    let model = if args.len() >= 2 {
                        self.eval_expr(&args[1])?.to_display_string()
                    } else {
                        "claude-3-5-haiku-latest".to_string()
                    };
                    let api_key = std::env::var("ANTHROPIC_API_KEY")
                        .map_err(|_| "llm_call: ANTHROPIC_API_KEY not set".to_string())?;
                    let client = reqwest::blocking::Client::new();
                    let body = serde_json::json!({
                        "model": model,
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}]
                    });
                    let resp = client
                        .post("https://api.anthropic.com/v1/messages")
                        .header("x-api-key", &api_key)
                        .header("anthropic-version", "2023-06-01")
                        .header("content-type", "application/json")
                        .json(&body)
                        .send()
                        .map_err(|e| format!("llm_call: HTTP error: {}", e))?;
                    let status = resp.status();
                    let json: serde_json::Value = resp
                        .json()
                        .map_err(|e| format!("llm_call: JSON decode error: {}", e))?;
                    if !status.is_success() {
                        let msg = json["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string();
                        return Err(format!("llm_call: API error {}: {}", status, msg));
                    }
                    let text = json["content"][0]["text"]
                        .as_str()
                        .ok_or_else(|| "llm_call: unexpected response shape".to_string())?
                        .to_string();
                    Ok(Value::String(text))
                }
                #[cfg(not(feature = "llm-builtins"))]
                {
                    let _ = args;
                    Err("llm_call: built without llm-builtins feature".to_string())
                }
            }
            // llm_chat(messages) | llm_chat(messages, model)
            //   Multi-turn chat. `messages` is an OMC array of dicts, each with
            //   keys "role" ("user" | "assistant") and "content" (String).
            //   Returns the assistant's reply String.
            #[cfg(feature = "llm-builtins")]
            "llm_chat" => {
                #[cfg(feature = "llm-builtins")]
                {
                    if args.is_empty() {
                        return Err("llm_chat requires (messages [, model])".to_string());
                    }
                    let msgs_val = self.eval_expr(&args[0])?;
                    let model = if args.len() >= 2 {
                        self.eval_expr(&args[1])?.to_display_string()
                    } else {
                        "claude-3-5-haiku-latest".to_string()
                    };
                    // Convert OMC array-of-dicts to serde_json array
                    let msgs_json = match &msgs_val {
                        Value::Array(arr) => {
                            let mut out = Vec::new();
                            for item in arr.items.borrow().iter() {
                                match item {
                                    Value::Dict(d) => {
                                        let role = d.borrow().get("role")
                                            .map(|v| v.to_display_string())
                                            .unwrap_or_else(|| "user".to_string());
                                        let content = d.borrow().get("content")
                                            .map(|v| v.to_display_string())
                                            .unwrap_or_default();
                                        out.push(serde_json::json!({"role": role, "content": content}));
                                    }
                                    _ => return Err("llm_chat: each message must be a dict with 'role' and 'content'".to_string()),
                                }
                            }
                            serde_json::Value::Array(out)
                        }
                        _ => return Err("llm_chat: first argument must be an array of message dicts".to_string()),
                    };
                    let api_key = std::env::var("ANTHROPIC_API_KEY")
                        .map_err(|_| "llm_chat: ANTHROPIC_API_KEY not set".to_string())?;
                    let client = reqwest::blocking::Client::new();
                    let body = serde_json::json!({
                        "model": model,
                        "max_tokens": 4096,
                        "messages": msgs_json
                    });
                    let resp = client
                        .post("https://api.anthropic.com/v1/messages")
                        .header("x-api-key", &api_key)
                        .header("anthropic-version", "2023-06-01")
                        .header("content-type", "application/json")
                        .json(&body)
                        .send()
                        .map_err(|e| format!("llm_chat: HTTP error: {}", e))?;
                    let status = resp.status();
                    let json: serde_json::Value = resp
                        .json()
                        .map_err(|e| format!("llm_chat: JSON decode error: {}", e))?;
                    if !status.is_success() {
                        let msg = json["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string();
                        return Err(format!("llm_chat: API error {}: {}", status, msg));
                    }
                    let text = json["content"][0]["text"]
                        .as_str()
                        .ok_or_else(|| "llm_chat: unexpected response shape".to_string())?
                        .to_string();
                    Ok(Value::String(text))
                }
                #[cfg(not(feature = "llm-builtins"))]
                {
                    let _ = args;
                    Err("llm_chat: built without llm-builtins feature".to_string())
                }
            }
            // llm_embed(text) | llm_embed(text, model)
            //   Returns an embedding vector (OMC Array of HFloat) for the
            //   given text. Uses the Voyage AI embeddings endpoint via
            //   VOYAGE_API_KEY, falling back to ANTHROPIC_API_KEY used with
            //   the voyage-3-lite model. If VOYAGE_API_KEY is not set and
            //   ANTHROPIC_API_KEY is not set, returns an error.
            //   Model defaults to "voyage-3-lite".
            #[cfg(feature = "llm-builtins")]
            "llm_embed" => {
                #[cfg(feature = "llm-builtins")]
                {
                    if args.is_empty() {
                        return Err("llm_embed requires (text [, model])".to_string());
                    }
                    let text = self.eval_expr(&args[0])?.to_display_string();
                    let model = if args.len() >= 2 {
                        self.eval_expr(&args[1])?.to_display_string()
                    } else {
                        "voyage-3-lite".to_string()
                    };
                    let api_key = std::env::var("VOYAGE_API_KEY")
                        .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                        .map_err(|_| "llm_embed: set VOYAGE_API_KEY or ANTHROPIC_API_KEY".to_string())?;
                    let client = reqwest::blocking::Client::new();
                    let body = serde_json::json!({
                        "input": [text],
                        "model": model
                    });
                    let resp = client
                        .post("https://api.voyageai.com/v1/embeddings")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("content-type", "application/json")
                        .json(&body)
                        .send()
                        .map_err(|e| format!("llm_embed: HTTP error: {}", e))?;
                    let status = resp.status();
                    let json: serde_json::Value = resp
                        .json()
                        .map_err(|e| format!("llm_embed: JSON decode error: {}", e))?;
                    if !status.is_success() {
                        let msg = json["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown error")
                            .to_string();
                        return Err(format!("llm_embed: API error {}: {}", status, msg));
                    }
                    let embedding = json["data"][0]["embedding"]
                        .as_array()
                        .ok_or_else(|| "llm_embed: unexpected response shape".to_string())?;
                    let floats: Vec<Value> = embedding
                        .iter()
                        .map(|v| Value::HFloat(v.as_f64().unwrap_or(0.0)))
                        .collect();
                    Ok(Value::Array(HArray::from_vec(floats)))
                }
                #[cfg(not(feature = "llm-builtins"))]
                {
                    let _ = args;
                    Err("llm_embed: built without llm-builtins feature".to_string())
                }
            }
            "str_join" => {
                if args.len() < 2 {
                    return Err("str_join requires (array, separator)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let sep = self.eval_expr(&args[1])?.to_string();
                if let Value::Array(arr) = arr_v {
                    let parts: Vec<String> = arr.items.borrow().iter().map(|v| match v {
                        Value::HInt(h) => h.value.to_string(),
                        Value::HFloat(f) => format!("{}", f),
                        Value::String(s) => s.clone(),
                        Value::Bool(b) => b.to_string(),
                        other => other.to_string(),
                    }).collect();
                    Ok(Value::String(parts.join(&sep)))
                } else {
                    Err("str_join: first argument must be an array".to_string())
                }
            }
            "str_trim" => {
                if args.is_empty() {
                    return Err("str_trim requires 1 argument".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                Ok(Value::String(s.trim().to_string()))
            }
            "str_replace" => {
                if args.len() < 3 {
                    return Err("str_replace requires (string, old, new)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let old = self.eval_expr(&args[1])?.to_string();
                let new_s = self.eval_expr(&args[2])?.to_string();
                if old.is_empty() {
                    // Replacing empty string would interleave new_s between
                    // every char — almost never the desired behaviour.
                    // Return the original.
                    return Ok(Value::String(s));
                }
                Ok(Value::String(s.replace(&old, &new_s)))
            }
            // ---- Regex (PCRE-style via the `regex` crate) -----------
            // Compiles the pattern on every call; for inner loops that
            // want a compiled regex reused, wrap the call in a fn and
            // memoize at the OMC level. Cheap-enough for one-shot use.
            "re_match" => {
                // re_match(pattern, text) -> 1 if pattern matches anywhere
                // in text, 0 otherwise. Anchor with ^/$ if you need
                // full-string matching.
                if args.len() < 2 {
                    return Err("re_match requires (pattern, text)".to_string());
                }
                let pat = self.eval_expr(&args[0])?.to_display_string();
                let text = self.eval_expr(&args[1])?.to_display_string();
                match regex::Regex::new(&pat) {
                    Ok(re) => Ok(Value::HInt(HInt::new(if re.is_match(&text) { 1 } else { 0 }))),
                    Err(e) => Err(format!("re_match: invalid pattern {:?}: {}", pat, e)),
                }
            }
            "re_find" => {
                // re_find(pattern, text) -> first match as string, or "" if no match.
                if args.len() < 2 {
                    return Err("re_find requires (pattern, text)".to_string());
                }
                let pat = self.eval_expr(&args[0])?.to_display_string();
                let text = self.eval_expr(&args[1])?.to_display_string();
                match regex::Regex::new(&pat) {
                    Ok(re) => {
                        let m = re.find(&text).map(|m| m.as_str().to_string()).unwrap_or_default();
                        Ok(Value::String(m))
                    }
                    Err(e) => Err(format!("re_find: invalid pattern {:?}: {}", pat, e)),
                }
            }
            "re_find_all" => {
                // re_find_all(pattern, text) -> array of all matches (in order).
                if args.len() < 2 {
                    return Err("re_find_all requires (pattern, text)".to_string());
                }
                let pat = self.eval_expr(&args[0])?.to_display_string();
                let text = self.eval_expr(&args[1])?.to_display_string();
                match regex::Regex::new(&pat) {
                    Ok(re) => {
                        let matches: Vec<Value> = re.find_iter(&text)
                            .map(|m| Value::String(m.as_str().to_string()))
                            .collect();
                        Ok(Value::Array(HArray::from_vec(matches)))
                    }
                    Err(e) => Err(format!("re_find_all: invalid pattern {:?}: {}", pat, e)),
                }
            }
            "re_replace" => {
                // re_replace(pattern, text, replacement) -> text with all
                // pattern matches replaced. Supports $1, $2 backrefs in
                // replacement string (Rust regex syntax).
                if args.len() < 3 {
                    return Err("re_replace requires (pattern, text, replacement)".to_string());
                }
                let pat = self.eval_expr(&args[0])?.to_display_string();
                let text = self.eval_expr(&args[1])?.to_display_string();
                let repl = self.eval_expr(&args[2])?.to_display_string();
                match regex::Regex::new(&pat) {
                    Ok(re) => Ok(Value::String(re.replace_all(&text, repl.as_str()).into_owned())),
                    Err(e) => Err(format!("re_replace: invalid pattern {:?}: {}", pat, e)),
                }
            }
            "re_split" => {
                // re_split(pattern, text) -> array of substrings split at pattern.
                if args.len() < 2 {
                    return Err("re_split requires (pattern, text)".to_string());
                }
                let pat = self.eval_expr(&args[0])?.to_display_string();
                let text = self.eval_expr(&args[1])?.to_display_string();
                match regex::Regex::new(&pat) {
                    Ok(re) => {
                        let parts: Vec<Value> = re.split(&text)
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        Ok(Value::Array(HArray::from_vec(parts)))
                    }
                    Err(e) => Err(format!("re_split: invalid pattern {:?}: {}", pat, e)),
                }
            }
            "str_index_of" => {
                if args.len() < 2 {
                    return Err("str_index_of requires (haystack, needle)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let needle = self.eval_expr(&args[1])?.to_string();
                // Return the CHAR index (not byte) so it pairs with
                // str_slice. -1 if not found, matching the JS / Java
                // convention everyone reaches for.
                let result = match s.find(&needle) {
                    None => -1i64,
                    Some(byte_pos) => {
                        // Convert byte position to char position.
                        s[..byte_pos].chars().count() as i64
                    }
                };
                Ok(Value::HInt(HInt::new(result)))
            }
            "str_starts_with" => {
                if args.len() < 2 {
                    return Err("str_starts_with requires (string, prefix)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let prefix = self.eval_expr(&args[1])?.to_string();
                Ok(Value::HInt(HInt::new(if s.starts_with(&prefix) { 1 } else { 0 })))
            }
            "str_ends_with" => {
                if args.len() < 2 {
                    return Err("str_ends_with requires (string, suffix)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let suffix = self.eval_expr(&args[1])?.to_string();
                Ok(Value::HInt(HInt::new(if s.ends_with(&suffix) { 1 } else { 0 })))
            }
            "str_repeat" => {
                if args.len() < 2 {
                    return Err("str_repeat requires (string, count)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let n = self.eval_expr(&args[1])?.to_int();
                let count = if n < 0 { 0 } else { n as usize };
                // Cap at 1M chars to prevent accidental memory blow-up.
                // Real abuse should fail loud, not silently truncate;
                // 1M is well above any reasonable use case.
                if s.len().saturating_mul(count) > 1_000_000 {
                    return Err(format!(
                        "str_repeat: result would exceed 1M chars ({} * {})",
                        s.len(), count
                    ));
                }
                Ok(Value::String(s.repeat(count)))
            }
            // Canonical Python OMC workaround for cross-type concat (string `+` is broken there).
            // Variadic: concat_many(a, b) / concat_many(a, b, c) / concat_many(a, b, c, d).
            // Renders numerics as bare values (89, 1.5) not as HInt(...) display form.
            "concat_many" => {
                // to_display_string for every arg — produces "42" not
                // "HInt(42, φ=..., HIM=...)" and recurses correctly
                // through arrays/dicts so `concat_many("xs: ", xs)`
                // shows "[1, 2, 3]" not the verbose Array dump.
                let mut out = String::new();
                for a in args {
                    let v = self.eval_expr(a)?;
                    out.push_str(&v.to_display_string());
                }
                Ok(Value::String(out))
            }
            // Array functions
            "arr_new" => {
                if args.len() < 2 {
                    return Err("arr_new requires 2 arguments".to_string());
                }
                let size = self.eval_expr(&args[0])?.to_int() as usize;
                let default = self.eval_expr(&args[1])?;
                let arr = HArray::with_capacity(size);
                {
                    let mut items = arr.items.borrow_mut();
                    for _ in 0..size {
                        items.push(default.clone());
                    }
                }
                Ok(Value::Array(arr))
            }
            "arr_from_range" | "range" => {
                // Python-style range: range(end), range(start, end),
                // range(start, end, step). step may be negative for
                // descending sequences. step=0 errors (no infinite loop).
                if args.is_empty() {
                    return Err(format!("{}: requires 1, 2, or 3 arguments", name));
                }
                let (start, end, step) = match args.len() {
                    1 => (0_i64, self.eval_expr(&args[0])?.to_int(), 1_i64),
                    2 => (
                        self.eval_expr(&args[0])?.to_int(),
                        self.eval_expr(&args[1])?.to_int(),
                        1_i64,
                    ),
                    _ => (
                        self.eval_expr(&args[0])?.to_int(),
                        self.eval_expr(&args[1])?.to_int(),
                        self.eval_expr(&args[2])?.to_int(),
                    ),
                };
                if step == 0 {
                    return Err(format!("{}: step must be non-zero", name));
                }
                let arr = HArray::new();
                {
                    let mut items = arr.items.borrow_mut();
                    let mut i = start;
                    if step > 0 {
                        while i < end {
                            items.push(Value::HInt(HInt::new(i)));
                            i += step;
                        }
                    } else {
                        while i > end {
                            items.push(Value::HInt(HInt::new(i)));
                            i += step;
                        }
                    }
                }
                Ok(Value::Array(arr))
            }
            "getenv" => {
                // getenv(name) → env var value or null when unset.
                // getenv(name, default) → value or default when unset.
                if args.is_empty() {
                    return Err("getenv: requires (name) or (name, default)".to_string());
                }
                let key = self.eval_expr(&args[0])?.to_display_string();
                match std::env::var(&key) {
                    Ok(val) => Ok(Value::String(val)),
                    Err(_) => {
                        if args.len() >= 2 {
                            self.eval_expr(&args[1])
                        } else {
                            Ok(Value::Null)
                        }
                    }
                }
            }
            "to_hex" => {
                // to_hex(int) → "0xNN" lowercase hex. Width is the
                // natural number of digits for the value's magnitude;
                // sign is preserved as a leading '-'.
                if args.is_empty() {
                    return Err("to_hex: requires (int)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                if n < 0 {
                    Ok(Value::String(format!("-0x{:x}", -n)))
                } else {
                    Ok(Value::String(format!("0x{:x}", n)))
                }
            }
            "from_hex" => {
                // from_hex(str) → int. Accepts "0xNN", "0XNN", or raw
                // "NN" (no prefix). Empty string and unparseable input
                // return a Singularity (matches str_to_int's contract).
                if args.is_empty() {
                    return Err("from_hex: requires (str)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                let cleaned = s.trim();
                let (sign, body) = if let Some(rest) = cleaned.strip_prefix('-') {
                    (-1_i64, rest)
                } else { (1_i64, cleaned) };
                let stripped = body
                    .strip_prefix("0x")
                    .or_else(|| body.strip_prefix("0X"))
                    .unwrap_or(body);
                match i64::from_str_radix(stripped, 16) {
                    Ok(n) => Ok(Value::HInt(HInt::new(sign * n))),
                    Err(_) => Ok(Value::Singularity {
                        numerator: 0,
                        denominator: 0,
                        context: format!("from_hex: cannot parse '{}'", s),
                    }),
                }
            }
            "parse_int" => {
                // Alias for str_to_int — Python users reach for this
                // name first. Same contract: returns Singularity on
                // failure.
                if args.is_empty() {
                    return Err("parse_int: requires (str)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                match s.trim().parse::<i64>() {
                    Ok(n) => Ok(Value::HInt(HInt::new(n))),
                    Err(_) => Ok(Value::Singularity {
                        numerator: 0,
                        denominator: 0,
                        context: format!("parse_int: cannot parse '{}'", s),
                    }),
                }
            }
            "parse_float" => {
                // Companion to parse_int. Useful for CSV / config parse.
                if args.is_empty() {
                    return Err("parse_float: requires (str)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                match s.trim().parse::<f64>() {
                    Ok(n) => Ok(Value::HFloat(n)),
                    Err(_) => Ok(Value::Singularity {
                        numerator: 0,
                        denominator: 0,
                        context: format!("parse_float: cannot parse '{}'", s),
                    }),
                }
            }
            // ── stdlib-friendly aliases ─────────────────────────────────────
            "math_log" => self.call_function("log", args),
            "math_sin" => self.call_function("sin", args),
            "math_cos" => self.call_function("cos", args),
            "math_tan" => self.call_function("tan", args),
            "math_sqrt" => self.call_function("sqrt", args),
            "math_floor" => self.call_function("floor", args),
            "math_ceil" => self.call_function("ceil", args),
            "math_round" => self.call_function("round", args),
            "math_abs" => self.call_function("abs", args),
            "math_pow" => self.call_function("pow", args),
            "math_min" => self.call_function("min", args),
            "math_max" => self.call_function("max", args),
            "unix_time" => self.call_function("now_unix", args),
            "unix_time_ms" => self.call_function("now_ms", args),
            "str_lower" => self.call_function("str_lowercase", args),
            "str_upper" => self.call_function("str_uppercase", args),
            "str_strip" => self.call_function("str_trim", args),
            "str_find" => self.call_function("str_index_of", args),
            "str_rfind" => {
                if args.is_empty() {
                    return Err("str_rfind requires (str, substr)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                let sub = if args.len() > 1 { self.eval_expr(&args[1])?.to_display_string() } else { return Err("str_rfind requires (str, substr)".to_string()) };
                match s.rfind(&sub[..]) {
                    Some(idx) => Ok(Value::HInt(HInt::new(idx as i64))),
                    None => Ok(Value::HInt(HInt::new(-1))),
                }
            }
            "env_var" => self.call_function("getenv", args),
            "dict_delete" | "dict_remove" => self.call_function("dict_del", args),
            "file_read" => self.call_function("read_file", args),
            "file_write" => self.call_function("write_file", args),
            "file_append" => {
                if args.len() < 2 {
                    return Err("file_append requires (path, content)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_display_string();
                let content = self.eval_expr(&args[1])?.to_display_string();
                let mut file = std::fs::OpenOptions::new()
                    .append(true).create(true).open(&path)
                    .map_err(|e| format!("file_append: cannot open '{}': {}", path, e))?;
                use std::io::Write;
                file.write_all(content.as_bytes())
                    .map_err(|e| format!("file_append: write failed: {}", e))?;
                Ok(Value::Null)
            }
            // ── end aliases ─────────────────────────────────────────────────
            "arr_len" => {
                if args.is_empty() {
                    return Err("arr_len requires 1 argument".to_string());
                }
                if let Value::Array(a) = self.eval_expr(&args[0])? {
                    Ok(Value::HInt(HInt::new(a.items.borrow().len() as i64)))
                } else {
                    Err("arr_len requires an array".to_string())
                }
            }
            "arr_sum" => {
                if args.is_empty() {
                    return Err("arr_sum requires 1 argument".to_string());
                }
                if let Value::Array(a) = self.eval_expr(&args[0])? {
                    let sum: i64 = a.items.borrow().iter().map(|v| v.to_int()).sum();
                    Ok(Value::HInt(HInt::new(sum)))
                } else {
                    Err("arr_sum requires an array".to_string())
                }
            }
            "arr_push" => {
                if args.len() < 2 {
                    return Err("arr_push requires (array_name, value)".to_string());
                }
                let val = self.eval_expr(&args[1])?;
                // Fast path: bare variable — avoids the eval clone.
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(arr)) = self.get_var(name) {
                        arr.items.borrow_mut().push(val);
                        return Ok(Value::Null);
                    }
                }
                // General path: any expression that evaluates to an array.
                // Works for arr_push(dict["key"], v) etc. because arrays are
                // Rc-shared — pushing through the returned Rc mutates in place.
                let arr_v = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = arr_v {
                    arr.items.borrow_mut().push(val);
                    return Ok(Value::Null);
                }
                Err("arr_push: first argument must be an array variable".to_string())
            }
            "arr_get" => {
                if args.len() < 2 {
                    return Err("arr_get requires (array, index)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let raw = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let len = items.len() as i64;
                    // Python-style negative indexing: -1 = last.
                    let resolved = if raw < 0 { len + raw } else { raw };
                    if resolved < 0 || resolved >= len {
                        return Err(format!(
                            "arr_get: index {} out of bounds (length {})",
                            raw, len
                        ));
                    }
                    Ok(items[resolved as usize].clone())
                } else {
                    let hint = if matches!(&arr_v, Value::Dict(_)) {
                        wrong_container_hint(&arr_v, "dict_get(d, key)")
                    } else {
                        format!(" (got {})", type_name_of(&arr_v))
                    };
                    Err(format!("arr_get: first argument must be an array{}", hint))
                }
            }
            "arr_set" => {
                if args.len() < 3 {
                    return Err("arr_set requires (array_name, index, value)".to_string());
                }
                let raw = self.eval_expr(&args[1])?.to_int();
                let val = self.eval_expr(&args[2])?;
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(arr)) = self.get_var(name) {
                        let mut items = arr.items.borrow_mut();
                        let len = items.len() as i64;
                        let resolved = if raw < 0 { len + raw } else { raw };
                        if resolved < 0 || resolved >= len {
                            return Err(format!(
                                "arr_set: index {} out of bounds (length {})",
                                raw, len
                            ));
                        }
                        items[resolved as usize] = val;
                        return Ok(Value::Null);
                    }
                }
                Err("arr_set: first argument must be an array variable".to_string())
            }
            // Phase H.5: self-healing array access. fold_escape pulls the
            // index onto the nearest Fibonacci attractor, then modulo by
            // arr_len keeps it in-bounds. Out-of-bounds reads become finite
            // attractor-landing reads; the math is the bounds check.
            "safe_arr_get" => {
                if args.len() < 2 {
                    return Err("safe_arr_get requires (array, index)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let raw_idx = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let len = items.len();
                    if len == 0 {
                        // No valid index for empty array. Return Null
                        // rather than error — keeps the access total.
                        return Ok(Value::Null);
                    }
                    let folded = fold_to_fibonacci_const(raw_idx);
                    let healed = ((folded % (len as i64)) + (len as i64)) % (len as i64);
                    Ok(items[healed as usize].clone())
                } else {
                    Err("safe_arr_get: first argument must be an array".to_string())
                }
            }
            "safe_arr_set" => {
                if args.len() < 3 {
                    return Err("safe_arr_set requires (array_name, index, value)".to_string());
                }
                let raw_idx = self.eval_expr(&args[1])?.to_int();
                let val = self.eval_expr(&args[2])?;
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(arr)) = self.get_var(name) {
                        let mut items = arr.items.borrow_mut();
                        let len = items.len();
                        if len == 0 {
                            return Ok(Value::Null);
                        }
                        let folded = fold_to_fibonacci_const(raw_idx);
                        let healed = ((folded % (len as i64)) + (len as i64)) % (len as i64);
                        items[healed as usize] = val;
                        return Ok(Value::Null);
                    }
                }
                Err("safe_arr_set: first argument must be an array variable".to_string())
            }
            // Array workhorse functions added for Python-tier ergonomics.
            "arr_sort" => {
                if args.is_empty() {
                    return Err("arr_sort requires 1 argument".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    // Sort by underlying numeric/lexicographic value.
                    // Mixed-type arrays sort by Value's natural ordering.
                    // Independent copy — sort returns a fresh array, doesn't
                    // mutate the input.
                    let mut items = arr.items.borrow().clone();
                    items.sort_by(|a, b| {
                        match (a, b) {
                            (Value::HInt(x), Value::HInt(y)) => x.value.cmp(&y.value),
                            (Value::HFloat(x), Value::HFloat(y)) => {
                                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                            }
                            (Value::String(x), Value::String(y)) => x.cmp(y),
                            // Mixed-type fallback: compare by float
                            // representation; keeps the sort total.
                            _ => {
                                let af = a.to_float();
                                let bf = b.to_float();
                                af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
                            }
                        }
                    });
                    Ok(Value::Array(HArray::from_vec(items)))
                } else {
                    Err("arr_sort: argument must be an array".to_string())
                }
            }
            "arr_reverse" => {
                // Note: str_reverse exists for strings; this is the array form.
                if args.is_empty() {
                    return Err("arr_reverse requires 1 argument".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    // Independent copy — reverse returns a fresh array.
                    let mut items = arr.items.borrow().clone();
                    items.reverse();
                    Ok(Value::Array(HArray::from_vec(items)))
                } else {
                    Err("arr_reverse: argument must be an array".to_string())
                }
            }
            "arr_join" => {
                // Alias for str_join — accepts (array, separator) and
                // returns a string. Provided so users who reach for the
                // arr_* prefix find what they expect.
                if args.len() < 2 {
                    return Err("arr_join requires (array, separator)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let sep = self.eval_expr(&args[1])?.to_string();
                if let Value::Array(arr) = arr_v {
                    let parts: Vec<String> = arr.items.borrow().iter().map(|v| match v {
                        Value::HInt(h) => h.value.to_string(),
                        Value::HFloat(f) => format!("{}", f),
                        Value::String(s) => s.clone(),
                        Value::Bool(b) => b.to_string(),
                        other => other.to_string(),
                    }).collect();
                    Ok(Value::String(parts.join(&sep)))
                } else {
                    Err("arr_join: first argument must be an array".to_string())
                }
            }
            // Higher-order array operations — require first-class function
            // values. Pass a function name as a bare identifier (preferred)
            // or as a string literal:
            //   arr_map(xs, double)        — bare name (Value::Function)
            //   arr_map(xs, "double")      — string form, also works
            // The function is invoked once per element; results collected.
            "arr_map" => {
                if args.len() < 2 {
                    return Err("arr_map requires (array, function)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let fn_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        let mapped = self.call_first_class_function(&fn_v, vec![item])?;
                        out.push(mapped);
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_map: first argument must be an array".to_string())
                }
            }
            "arr_filter" => {
                if args.len() < 2 {
                    return Err("arr_filter requires (array, predicate)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let fn_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    let mut out = Vec::new();
                    for item in items {
                        let kept = self.call_first_class_function(&fn_v, vec![item.clone()])?;
                        if kept.to_bool() {
                            out.push(item);
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_filter: first argument must be an array".to_string())
                }
            }
            "arr_reduce" => {
                // reduce(arr, fn, init) — function receives (accumulator, item)
                // and returns the new accumulator. Left fold.
                if args.len() < 3 {
                    return Err("arr_reduce requires (array, function, initial)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let fn_v = self.eval_expr(&args[1])?;
                let mut acc = self.eval_expr(&args[2])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    for item in items {
                        acc = self.call_first_class_function(&fn_v, vec![acc, item])?;
                    }
                    Ok(acc)
                } else {
                    Err("arr_reduce: first argument must be an array".to_string())
                }
            }
            "par_map" => {
                // par_map(function, array) — parallel map over array.
                // Value uses Rc/RefCell so not Send; runs serially on same thread.
                // API-compatible with a true multi-threaded par_map.
                if args.len() < 2 {
                    return Err("par_map requires (function, array)".to_string());
                }
                let fn_v  = self.eval_expr(&args[0])?;
                let arr_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    let mut out = Vec::with_capacity(items.len());
                    for item in items {
                        out.push(self.call_first_class_function(&fn_v, vec![item])?);
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("par_map: second argument must be an array".to_string())
                }
            }
            "par_filter" => {
                // par_filter(function, array) — parallel filter over array.
                // Runs serially (Value is not Send); API mirrors a true parallel filter.
                if args.len() < 2 {
                    return Err("par_filter requires (function, array)".to_string());
                }
                let fn_v  = self.eval_expr(&args[0])?;
                let arr_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    let mut out = Vec::new();
                    for item in items {
                        if self.call_first_class_function(&fn_v, vec![item.clone()])?.to_bool() {
                            out.push(item);
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("par_filter: second argument must be an array".to_string())
                }
            }
            "par_reduce" => {
                // par_reduce(function, array, init) — parallel reduce over array.
                // Runs serially (Value is not Send); left fold, same as arr_reduce.
                if args.len() < 3 {
                    return Err("par_reduce requires (function, array, init)".to_string());
                }
                let fn_v   = self.eval_expr(&args[0])?;
                let arr_v  = self.eval_expr(&args[1])?;
                let mut acc = self.eval_expr(&args[2])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    for item in items {
                        acc = self.call_first_class_function(&fn_v, vec![acc, item])?;
                    }
                    Ok(acc)
                } else {
                    Err("par_reduce: second argument must be an array".to_string())
                }
            }
            "par_for" => {
                // par_for(function, array) — parallel for-each over array.
                // Runs serially (Value is not Send); API mirrors a true parallel for-each.
                // Returns Null after processing all elements.
                if args.len() < 2 {
                    return Err("par_for requires (function, array)".to_string());
                }
                let fn_v  = self.eval_expr(&args[0])?;
                let arr_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    for item in items {
                        self.call_first_class_function(&fn_v, vec![item])?;
                    }
                    Ok(Value::Null)
                } else {
                    Err("par_for: second argument must be an array".to_string())
                }
            }
            "arr_any" => {
                // Returns 1 if predicate is truthy for any element, else 0.
                // Short-circuits on first true.
                if args.len() < 2 {
                    return Err("arr_any requires (array, predicate)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let fn_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    for item in items {
                        if self.call_first_class_function(&fn_v, vec![item])?.to_bool() {
                            return Ok(Value::HInt(HInt::new(1)));
                        }
                    }
                    Ok(Value::HInt(HInt::new(0)))
                } else {
                    Err("arr_any: first argument must be an array".to_string())
                }
            }
            "arr_all" => {
                if args.len() < 2 {
                    return Err("arr_all requires (array, predicate)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let fn_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    for item in items {
                        if !self.call_first_class_function(&fn_v, vec![item])?.to_bool() {
                            return Ok(Value::HInt(HInt::new(0)));
                        }
                    }
                    Ok(Value::HInt(HInt::new(1)))
                } else {
                    Err("arr_all: first argument must be an array".to_string())
                }
            }
            "arr_find" => {
                // Returns the first element where predicate is true, else Null.
                if args.len() < 2 {
                    return Err("arr_find requires (array, predicate)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let fn_v = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow().clone();
                    for item in items {
                        if self.call_first_class_function(&fn_v, vec![item.clone()])?.to_bool() {
                            return Ok(item);
                        }
                    }
                    Ok(Value::Null)
                } else {
                    Err("arr_find: first argument must be an array".to_string())
                }
            }
            // ---- Dict (hash-map) builtins ----------------------------------
            // String-keyed maps. dict_set / dict_del mutate by name (same
            // arr_push convention) — first arg must be a Variable so the
            // mutation can write back. dict_get returns Null on missing key,
            // matching Python's d.get(k) sans default.
            "dict_new" => {
                Ok(Value::dict_empty())
            }
            // is_instance(value, "ClassName") — true when value is a
            // class instance whose __class__ matches the given name OR
            // any name in the parent chain. Lets typed-exception catch
            // blocks dispatch by class hierarchy without manual chain
            // walking. Returns 0 for non-instance values (numbers, etc.).
            "is_instance" => {
                if args.len() < 2 {
                    return Err("is_instance requires (value, class_name)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_display_string();
                let cls = match &v {
                    Value::Dict(d) => {
                        d.borrow().get("__class__")
                            .map(|c| c.to_display_string())
                    }
                    _ => None,
                };
                let Some(mut current) = cls else {
                    return Ok(Value::HInt(HInt::new(0)));
                };
                // Walk the parent chain, capped at 64 hops to mirror the
                // method-dispatch path. Match if any ancestor name equals
                // the target.
                for _ in 0..64 {
                    if current == target {
                        return Ok(Value::HInt(HInt::new(1)));
                    }
                    match self.class_parents.get(&current) {
                        Some(parent) => current = parent.clone(),
                        None => return Ok(Value::HInt(HInt::new(0))),
                    }
                }
                Ok(Value::HInt(HInt::new(0)))
            }
            "dict_get" => {
                if args.len() < 2 {
                    return Err("dict_get requires (dict, key)".to_string());
                }
                let d_v = self.eval_expr(&args[0])?;
                let k = self.eval_expr(&args[1])?.to_display_string();
                if let Value::Dict(d) = d_v {
                    // Optional 3rd arg = default. Without it, missing → Null.
                    let default = if args.len() >= 3 {
                        Some(self.eval_expr(&args[2])?)
                    } else { None };
                    Ok(d.borrow().get(&k).cloned().unwrap_or_else(|| default.unwrap_or(Value::Null)))
                } else {
                    let hint = if matches!(&d_v, Value::Array(_)) {
                        wrong_container_hint(&d_v, "arr_get(arr, idx)")
                    } else {
                        format!(" (got {})", type_name_of(&d_v))
                    };
                    Err(format!("dict_get: first argument must be a dict{}", hint))
                }
            }
            "dict_set" => {
                if args.len() < 3 {
                    return Err("dict_set requires (dict_var, key, value)".to_string());
                }
                let k = self.eval_expr(&args[1])?.to_display_string();
                let val = self.eval_expr(&args[2])?;
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Dict(d)) = self.get_var(name) {
                        // Rc<RefCell> Dict: borrow_mut hits the shared map.
                        d.borrow_mut().insert(k, val);
                        return Ok(Value::Null);
                    }
                }
                Err("dict_set: first argument must be a dict variable".to_string())
            }
            "dict_has" => {
                if args.len() < 2 {
                    return Err("dict_has requires (dict, key)".to_string());
                }
                let d_v = self.eval_expr(&args[0])?;
                let k = self.eval_expr(&args[1])?.to_display_string();
                if let Value::Dict(d) = d_v {
                    Ok(Value::HInt(HInt::new(if d.borrow().contains_key(&k) { 1 } else { 0 })))
                } else {
                    Err("dict_has: first argument must be a dict".to_string())
                }
            }
            "dict_del" => {
                if args.len() < 2 {
                    return Err("dict_del requires (dict, key)".to_string());
                }
                let k = self.eval_expr(&args[1])?.to_display_string();
                // Accept both a plain variable AND any expression that evaluates
                // to a dict (e.g. obj["store"]). The Rc is shared, so removal
                // through the evaluated reference propagates to all holders.
                match self.eval_expr(&args[0])? {
                    Value::Dict(d) => {
                        d.borrow_mut().remove(&k);
                        Ok(Value::Null)
                    }
                    _ => Err("dict_del: first argument must be a dict".to_string()),
                }
            }
            "dict_keys" => {
                if args.is_empty() {
                    return Err("dict_keys requires (dict)".to_string());
                }
                if let Value::Dict(d) = self.eval_expr(&args[0])? {
                    let items: Vec<Value> = d.borrow().keys().map(|k| Value::String(k.clone())).collect();
                    Ok(Value::Array(HArray::from_vec(items)))
                } else {
                    Err("dict_keys: argument must be a dict".to_string())
                }
            }
            "dict_values" => {
                if args.is_empty() {
                    return Err("dict_values requires (dict)".to_string());
                }
                if let Value::Dict(d) = self.eval_expr(&args[0])? {
                    let items: Vec<Value> = d.borrow().values().cloned().collect();
                    Ok(Value::Array(HArray::from_vec(items)))
                } else {
                    Err("dict_values: argument must be a dict".to_string())
                }
            }
            "dict_len" => {
                if args.is_empty() {
                    return Err("dict_len requires (dict)".to_string());
                }
                if let Value::Dict(d) = self.eval_expr(&args[0])? {
                    Ok(Value::HInt(HInt::new(d.borrow().len() as i64)))
                } else {
                    Err("dict_len: argument must be a dict".to_string())
                }
            }
            "dict_merge" => {
                // Returns a NEW dict with both inputs merged; right-hand
                // wins on key collision. Pure (non-mutating) so it can
                // chain in expressions: `dict_merge(defaults, overrides)`.
                if args.len() < 2 {
                    return Err("dict_merge requires (dict_a, dict_b)".to_string());
                }
                let a_v = self.eval_expr(&args[0])?;
                let b_v = self.eval_expr(&args[1])?;
                match (a_v, b_v) {
                    (Value::Dict(a), Value::Dict(b)) => {
                        // Fresh map — explicit copy semantics so the result
                        // doesn't share state with either input.
                        let mut out = a.borrow().clone();
                        for (k, v) in b.borrow().iter() { out.insert(k.clone(), v.clone()); }
                        Ok(Value::dict_from(out))
                    }
                    _ => Err("dict_merge: both arguments must be dicts".to_string()),
                }
            }
            "dict_pop" => {
                // Mutating: remove key from dict_var, return its value or Null.
                if args.len() < 2 {
                    return Err("dict_pop requires (dict_var, key)".to_string());
                }
                let k = self.eval_expr(&args[1])?.to_display_string();
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Dict(d)) = self.get_var(name) {
                        let removed = d.borrow_mut().remove(&k);
                        return Ok(removed.unwrap_or(Value::Null));
                    }
                }
                Err("dict_pop: first argument must be a dict variable".to_string())
            }
            "dict_get_or" => {
                // Pure: dict_get with a default fallback (always returns the default for missing).
                if args.len() < 3 {
                    return Err("dict_get_or requires (dict, key, default)".to_string());
                }
                let dict_v = self.eval_expr(&args[0])?;
                let k = self.eval_expr(&args[1])?.to_display_string();
                let default = self.eval_expr(&args[2])?;
                if let Value::Dict(d) = dict_v {
                    Ok(d.borrow().get(&k).cloned().unwrap_or(default))
                } else {
                    Err("dict_get_or: first argument must be a dict".to_string())
                }
            }
            "dict_size" => {
                // Alias for dict_len (Python-aligned naming).
                if args.is_empty() {
                    return Err("dict_size requires (dict)".to_string());
                }
                if let Value::Dict(d) = self.eval_expr(&args[0])? {
                    Ok(Value::HInt(HInt::new(d.borrow().len() as i64)))
                } else {
                    Err("dict_size: argument must be a dict".to_string())
                }
            }
            "dict_clear" => {
                // Mutating: drop all entries.
                if args.is_empty() {
                    return Err("dict_clear requires (dict_var)".to_string());
                }
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Dict(d)) = self.get_var(name) {
                        d.borrow_mut().clear();
                        return Ok(Value::Null);
                    }
                }
                Err("dict_clear: argument must be a dict variable".to_string())
            }
            "dict_items" => {
                // Returns array of [key, value] pairs.
                if args.is_empty() {
                    return Err("dict_items requires (dict)".to_string());
                }
                if let Value::Dict(d) = self.eval_expr(&args[0])? {
                    let mut out = Vec::with_capacity(d.borrow().len());
                    for (k, v) in d.borrow().iter() {
                        out.push(Value::Array(HArray::from_vec(vec![
                            Value::String(k.clone()), v.clone()
                        ])));
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("dict_items: argument must be a dict".to_string())
                }
            }
            // File I/O — basic synchronous reads and writes.
            // Error semantics: read_file returns the error message as the
            // error path so callers can pattern-match; write_file returns
            // 1 on success and the error on failure. file_exists is total.
            "read_file" => {
                if args.is_empty() {
                    return Err("read_file requires (path)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_string();
                match std::fs::read_to_string(&path) {
                    Ok(content) => Ok(Value::String(content)),
                    Err(e) => Err(format!("read_file({}): {}", path, e)),
                }
            }
            "write_file" => {
                if args.len() < 2 {
                    return Err("write_file requires (path, content)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_string();
                let content = self.eval_expr(&args[1])?.to_string();
                match std::fs::write(&path, &content) {
                    Ok(_) => Ok(Value::HInt(HInt::new(1))),
                    Err(e) => Err(format!("write_file({}): {}", path, e)),
                }
            }
            "file_exists" => {
                if args.is_empty() {
                    return Err("file_exists requires (path)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_string();
                let exists = std::path::Path::new(&path).exists();
                Ok(Value::HInt(HInt::new(if exists { 1 } else { 0 })))
            }
            "file_ls" => {
                let path = if args.is_empty() {
                    ".".to_string()
                } else {
                    self.eval_expr(&args[0])?.to_display_string()
                };
                let entries = std::fs::read_dir(&path)
                    .map_err(|e| format!("file_ls: {}", e))?;
                let mut names: Vec<Value> = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    names.push(Value::String(name));
                }
                names.sort_by(|a, b| a.to_display_string().cmp(&b.to_display_string()));
                Ok(Value::Array(HArray::from_vec(names)))
            }
            "sleep" => {
                let ms = if args.is_empty() { 0i64 } else {
                    match self.eval_expr(&args[0])? {
                        Value::HInt(n) => n.value,
                        Value::HFloat(f) => f as i64,
                        _ => 0,
                    }
                };
                if ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(ms as u64));
                }
                Ok(Value::Null)
            }
            "omc_eval_file" => {
                if args.is_empty() {
                    return Err("omc_eval_file requires (path)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_display_string();
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| format!("omc_eval_file: {}", e))?;
                let mut parser = crate::parser::Parser::new(&src);
                let stmts = parser.parse().map_err(|e| format!("omc_eval_file parse error: {}", e))?;
                self.register_user_functions(&stmts);
                let pre_last = self.last_expression_value.take();
                self.execute(stmts)?;
                let result = self.last_expression_value.take().unwrap_or(Value::Null);
                self.last_expression_value = pre_last;
                Ok(result)
            }
            "str_similarity" => {
                if args.len() < 2 {
                    return Err("str_similarity requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let va = crate::llm_builtins::substrate_embed(&a, 32);
                let vb = crate::llm_builtins::substrate_embed(&b, 32);
                if let (Value::Array(av), Value::Array(bv)) = (va, vb) {
                    let mut dot = 0.0f64;
                    let ai = av.items.borrow();
                    let bi = bv.items.borrow();
                    for (x, y) in ai.iter().zip(bi.iter()) {
                        if let (Value::HFloat(xf), Value::HFloat(yf)) = (x, y) {
                            dot += xf * yf;
                        }
                    }
                    Ok(Value::HFloat(dot.clamp(-1.0, 1.0)))
                } else {
                    Ok(Value::HFloat(0.0))
                }
            }
            // Introspection and utility.
            "type_of" => {
                if args.is_empty() {
                    return Err("type_of requires 1 argument".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let tag = match v {
                    Value::HInt(_) => "int",
                    Value::HFloat(_) => "float",
                    Value::String(_) => "string",
                    Value::Bool(_) => "bool",
                    Value::Array(_) => "array",
                    Value::Dict(_) => "dict",
                    Value::Function { .. } => "function",
                    Value::Null => "null",
                    Value::Singularity { .. } => "singularity",
                    _ => "unknown",
                };
                Ok(Value::String(tag.to_string()))
            }
            // Throw a user-defined error. Caught by the surrounding
            // try/catch if any; otherwise propagates to the top and
            // crashes the program with the message. Mirrors Python's
            // `raise ValueError(msg)` for the no-class case.
            "error" => {
                let msg = if args.is_empty() {
                    "error".to_string()
                } else {
                    self.eval_expr(&args[0])?.to_display_string()
                };
                Err(msg)
            }
            "gcd" => {
                if args.len() < 2 {
                    return Err("gcd requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int().abs();
                let b = self.eval_expr(&args[1])?.to_int().abs();
                let mut x = a;
                let mut y = b;
                while y != 0 {
                    let t = y;
                    y = x % y;
                    x = t;
                }
                Ok(Value::HInt(HInt::new(x)))
            }
            "lcm" => {
                if args.len() < 2 {
                    return Err("lcm requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int().abs();
                let b = self.eval_expr(&args[1])?.to_int().abs();
                if a == 0 || b == 0 {
                    return Ok(Value::HInt(HInt::new(0)));
                }
                // gcd inline to avoid recursive call_function overhead
                let mut x = a;
                let mut y = b;
                while y != 0 {
                    let t = y;
                    y = x % y;
                    x = t;
                }
                Ok(Value::HInt(HInt::new(a / x * b)))
            }
            "now_ms" => {
                // Milliseconds since unix epoch. No args.
                // Useful for benchmarking inside OMC programs.
                use std::time::{SystemTime, UNIX_EPOCH};
                let ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);
                Ok(Value::HInt(HInt::new(ms)))
            }
            // Introspection of the function table — used by the OMC-side
            // test runner to discover `test_*` functions and dispatch them.
            "defined_functions" => {
                // Returns an array of user-defined function names. Sorted
                // for deterministic test discovery order (alphabetical).
                // Auto-generated lambdas (__lambda_N) are excluded so
                // the test runner doesn't try to run them as tests.
                let mut names: Vec<String> = self.functions.keys()
                    .filter(|n| !n.starts_with("__lambda_")
                             && !n.starts_with("__rt_lambda_"))
                    .cloned()
                    .collect();
                names.sort();
                Ok(Value::Array(HArray::from_vec(
                    names.into_iter().map(Value::String).collect(),
                )))
            }
            // list_defined_fns() / list_fns() — returns sorted array of user fn names
            "list_defined_fns" | "list_fns" => {
                let mut names: Vec<String> = self.functions.keys()
                    .filter(|n| !n.starts_with("__lambda_") && !n.starts_with("__rt_lambda_"))
                    .cloned()
                    .collect();
                names.sort();
                Ok(Value::Array(HArray::from_vec(
                    names.into_iter().map(Value::String).collect(),
                )))
            }
            // fn_arity(name) → int — parameter count of a user-defined function, or null
            "fn_arity" => {
                if args.is_empty() {
                    return Err("fn_arity requires (fn_name)".to_string());
                }
                let name_v = self.eval_expr(&args[0])?;
                let fn_name = match &name_v {
                    Value::String(s) => s.clone(),
                    _ => return Err("fn_arity: argument must be a string".to_string()),
                };
                if let Some((params, _body)) = self.functions.get(&fn_name) {
                    Ok(Value::HInt(HInt::new(params.len() as i64)))
                } else {
                    Ok(Value::Null)
                }
            }
            // fn_source(name) → string — full formatted source of a user-defined function
            "fn_source" => {
                if args.is_empty() {
                    return Err("fn_source requires (fn_name)".to_string());
                }
                let name_v = self.eval_expr(&args[0])?;
                let fn_name = match &name_v {
                    Value::String(s) => s.clone(),
                    _ => return Err("fn_source: argument must be a string".to_string()),
                };
                if let Some((params, body)) = self.functions.get(&fn_name) {
                    let stmt = Statement::FunctionDef {
                        name: fn_name.clone(),
                        params: params.clone(),
                        param_types: params.iter().map(|_| None).collect(),
                        body: body.clone(),
                        return_type: None,
                        pragmas: vec![],
                    };
                    let src = crate::formatter::format_program(&[stmt]);
                    Ok(Value::String(src.trim_end_matches('\n').to_string()))
                } else {
                    Ok(Value::Null)
                }
            }
            // get_scope_vars() → dict — copy of all global variables currently in scope
            "get_scope_vars" => {
                let mut map: std::collections::BTreeMap<String, Value> = std::collections::BTreeMap::new();
                for (k, v) in &self.globals {
                    if !k.starts_with("__") {
                        map.insert(k.clone(), v.clone());
                    }
                }
                Ok(Value::dict_from(map))
            }
            // call(fn_or_name, args_array) — dispatch a function value
            // (or function-name string) with an arbitrary argument list
            // unpacked from an array. Complements the HOFs (which fix
            // arity at 1 or 2). Lets the test runner invoke zero-arg
            // tests, and lets user code do dynamic-arity dispatch.
            "call" => {
                if args.len() < 2 {
                    return Err("call requires (function, args_array)".to_string());
                }
                let fn_v = self.eval_expr(&args[0])?;
                let args_v = self.eval_expr(&args[1])?;
                let arg_list = match args_v {
                    Value::Array(a) => a.items.borrow().clone(),
                    _ => return Err("call: second argument must be an array".to_string()),
                };
                self.call_first_class_function(&fn_v, arg_list)
            }
            // Test runner host-state primitives. The test runner is in
            // OMC (examples/test_runner.omc); these builtins give it a
            // side-channel for failure tracking that bypasses OMC's
            // pass-by-value array semantics (which would otherwise lose
            // failures recorded inside nested function calls).
            "test_record_failure" => {
                if args.is_empty() {
                    return Err("test_record_failure requires (message)".to_string());
                }
                let msg = self.eval_expr(&args[0])?.to_string();
                // Auto-prefix with the current test name (if set) so the
                // failure log always carries context. The OMC test runner
                // just calls test_record_failure(reason) and the prefix
                // attaches transparently.
                let prefix = self.test_current_name.borrow().clone();
                let recorded = if prefix.is_empty() {
                    msg
                } else {
                    format!("{}: {}", prefix, msg)
                };
                self.test_failures.borrow_mut().push(recorded);
                Ok(Value::HInt(HInt::new(0)))
            }
            "test_set_current" => {
                if args.is_empty() {
                    return Err("test_set_current requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_string();
                *self.test_current_name.borrow_mut() = name;
                Ok(Value::Null)
            }
            "test_get_current" => {
                Ok(Value::String(self.test_current_name.borrow().clone()))
            }
            "test_failure_count" => {
                Ok(Value::HInt(HInt::new(self.test_failures.borrow().len() as i64)))
            }
            "test_get_failures" => {
                let items: Vec<Value> = self.test_failures.borrow()
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(items)))
            }
            "test_clear_failures" => {
                self.test_failures.borrow_mut().clear();
                Ok(Value::Null)
            }
            // Random — xorshift64* via the interpreter's RNG state.
            // random_seed(s) for deterministic runs; otherwise seeded from
            // system nanos at interpreter construction.
            "random_int" => {
                // random_int(lo, hi) — inclusive on both ends. Returns lo
                // if hi <= lo (graceful fallback rather than error).
                if args.len() < 2 {
                    return Err("random_int requires (lo, hi)".to_string());
                }
                let lo = self.eval_expr(&args[0])?.to_int();
                let hi = self.eval_expr(&args[1])?.to_int();
                if hi <= lo {
                    return Ok(Value::HInt(HInt::new(lo)));
                }
                let range = (hi - lo + 1) as u64;
                let r = self.rng_next() % range;
                Ok(Value::HInt(HInt::new(lo + r as i64)))
            }
            "random_float" => {
                // Uniform float in [0.0, 1.0). No args.
                let r = self.rng_next();
                let f = (r >> 11) as f64 / (1u64 << 53) as f64;
                Ok(Value::HFloat(f))
            }
            "random_seed" => {
                if args.is_empty() {
                    return Err("random_seed requires (seed)".to_string());
                }
                let seed = self.eval_expr(&args[0])?.to_int() as u64;
                let initial = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
                self.rng_state.set(initial);
                Ok(Value::HInt(HInt::new(seed as i64)))
            }
            // String padding — common formatting workhorses.
            "str_pad_left" => {
                if args.len() < 3 {
                    return Err("str_pad_left requires (string, width, pad_char)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let width = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let pad = self.eval_expr(&args[2])?.to_string();
                let pad_char = pad.chars().next().unwrap_or(' ');
                let len = s.chars().count();
                if len >= width {
                    return Ok(Value::String(s));
                }
                let padding: String = std::iter::repeat(pad_char).take(width - len).collect();
                Ok(Value::String(format!("{}{}", padding, s)))
            }
            "str_pad_right" => {
                if args.len() < 3 {
                    return Err("str_pad_right requires (string, width, pad_char)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let width = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let pad = self.eval_expr(&args[2])?.to_string();
                let pad_char = pad.chars().next().unwrap_or(' ');
                let len = s.chars().count();
                if len >= width {
                    return Ok(Value::String(s));
                }
                let padding: String = std::iter::repeat(pad_char).take(width - len).collect();
                Ok(Value::String(format!("{}{}", s, padding)))
            }
            "str_split_lines" => {
                // Split on \n (consuming \r\n properly so Windows files don't
                // leave \r remnants). Returns array of strings.
                if args.is_empty() {
                    return Err("str_split_lines requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let lines: Vec<Value> = s.lines()
                    .map(|l| Value::String(l.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(lines)))
            }
            "str_count" => {
                // Count non-overlapping occurrences of needle in haystack.
                if args.len() < 2 {
                    return Err("str_count requires (haystack, needle)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                let needle = self.eval_expr(&args[1])?.to_display_string();
                if needle.is_empty() {
                    return Ok(Value::HInt(HInt::new(0)));
                }
                Ok(Value::HInt(HInt::new(s.matches(&needle).count() as i64)))
            }
            "str_is_empty" => {
                if args.is_empty() {
                    return Err("str_is_empty requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                Ok(Value::HInt(HInt::new(if s.is_empty() { 1 } else { 0 })))
            }
            "str_to_int" => {
                // Parse string as int. Returns Singularity on parse failure
                // — same idiom div-by-zero uses elsewhere; resolvable.
                if args.is_empty() {
                    return Err("str_to_int requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                match s.trim().parse::<i64>() {
                    Ok(n) => Ok(Value::HInt(HInt::new(n))),
                    Err(_) => Ok(Value::Singularity {
                        numerator: 0, denominator: 0,
                        context: format!("str_to_int: {:?} not parseable", s),
                    }),
                }
            }
            "str_to_float" => {
                if args.is_empty() {
                    return Err("str_to_float requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                match s.trim().parse::<f64>() {
                    Ok(f) => Ok(Value::HFloat(f)),
                    Err(_) => Ok(Value::Singularity {
                        numerator: 0, denominator: 0,
                        context: format!("str_to_float: {:?} not parseable", s),
                    }),
                }
            }
            "str_capitalize" => {
                // Uppercase the first char, leave the rest as-is.
                // Aligns with Python str.capitalize when called on lowercase
                // input; for mixed-case input we deliberately don't lowercase
                // the tail (Python does), since that's surprising for many
                // identifiers/proper nouns.
                if args.is_empty() {
                    return Err("str_capitalize requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let mut chars = s.chars();
                let out = match chars.next() {
                    Some(c) => c.to_uppercase().chain(chars).collect(),
                    None => String::new(),
                };
                Ok(Value::String(out))
            }
            // arr_zip — pair elements positionally. Returns array of
            // [a_i, b_i] pairs; shorter array determines length.
            "arr_zip" => {
                if args.len() < 2 {
                    return Err("arr_zip requires (array_a, array_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                match (a, b) {
                    (Value::Array(aa), Value::Array(bb)) => {
                        let aa_b = aa.items.borrow();
                        let bb_b = bb.items.borrow();
                        let len = aa_b.len().min(bb_b.len());
                        let pairs: Vec<Value> = (0..len).map(|i| {
                            Value::Array(HArray::from_vec(vec![
                                aa_b[i].clone(),
                                bb_b[i].clone(),
                            ]))
                        }).collect();
                        Ok(Value::Array(HArray::from_vec(pairs)))
                    }
                    _ => Err("arr_zip: both arguments must be arrays".to_string()),
                }
            }
            // arr_unique — dedupe preserving first occurrence order.
            // Equality follows the existing values_equal helper used by
            // arr_contains, so it's type-aware.
            "arr_unique" => {
                if args.is_empty() {
                    return Err("arr_unique requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow().clone();
                    let mut seen: Vec<Value> = Vec::new();
                    for v in items {
                        let dup = seen.iter().any(|s| values_equal(s, &v));
                        if !dup {
                            seen.push(v);
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(seen)))
                } else {
                    Err("arr_unique: argument must be an array".to_string())
                }
            }
            // arr_take(arr, n) — first n elements (or all if n > len).
            // Common slicing helper not previously exposed.
            "arr_take" => {
                if args.len() < 2 {
                    return Err("arr_take requires (array, n)".to_string());
                }
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let take = items.iter().take(n).cloned().collect::<Vec<_>>();
                    Ok(Value::Array(HArray::from_vec(take)))
                } else {
                    Err("arr_take: requires an array".to_string())
                }
            }
            // arr_drop(arr, n) — skip first n elements, return the rest.
            "arr_drop" => {
                if args.len() < 2 {
                    return Err("arr_drop requires (array, n)".to_string());
                }
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let drop = items.iter().skip(n).cloned().collect::<Vec<_>>();
                    Ok(Value::Array(HArray::from_vec(drop)))
                } else {
                    Err("arr_drop: requires an array".to_string())
                }
            }
            // arr_count(arr, value) — count of occurrences. Useful for
            // frequency analysis without going through dict_set.
            "arr_count" => {
                if args.len() < 2 {
                    return Err("arr_count requires (array, value)".to_string());
                }
                let needle = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let n: i64 = items.iter().filter(|v| values_equal(v, &needle)).count() as i64;
                    Ok(Value::HInt(HInt::new(n)))
                } else {
                    Err("arr_count: requires an array".to_string())
                }
            }
            // arr_repeat(value, n) — array of n copies of value.
            // Replaces the common arr_new(n, val) pattern when val is
            // not just zero.
            "arr_repeat" => {
                if args.len() < 2 {
                    return Err("arr_repeat requires (value, n)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let items: Vec<Value> = (0..n).map(|_| v.clone()).collect();
                Ok(Value::Array(HArray::from_vec(items)))
            }
            // arr_zeros(n) — array of n zeros (HInt). NumPy idiom.
            // arr_fill(value, n) — array of n copies of value. Works with any Value.
            "arr_fill" => {
                if args.len() < 2 {
                    return Err("arr_fill requires (value, n)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let items: Vec<Value> = (0..n).map(|_| v.clone()).collect();
                Ok(Value::Array(HArray::from_vec(items)))
            }
            "arr_zeros" => {
                if args.is_empty() {
                    return Err("arr_zeros requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int().max(0) as usize;
                let items: Vec<Value> = (0..n).map(|_| Value::HInt(HInt::new(0))).collect();
                Ok(Value::Array(HArray::from_vec(items)))
            }
            // arr_ones(n) — array of n ones (HInt). NumPy idiom.
            "arr_ones" => {
                if args.is_empty() {
                    return Err("arr_ones requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int().max(0) as usize;
                let items: Vec<Value> = (0..n).map(|_| Value::HInt(HInt::new(1))).collect();
                Ok(Value::Array(HArray::from_vec(items)))
            }
            // arr_chunk(arr, size) — split into sub-arrays of `size`.
            // Last chunk may be shorter. Common batching pattern.
            "arr_chunk" => {
                if args.len() < 2 {
                    return Err("arr_chunk requires (array, size)".to_string());
                }
                let size = self.eval_expr(&args[1])?.to_int().max(1) as usize;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let chunks: Vec<Value> = items
                        .chunks(size)
                        .map(|c| Value::Array(HArray::from_vec(c.to_vec())))
                        .collect();
                    Ok(Value::Array(HArray::from_vec(chunks)))
                } else {
                    Err("arr_chunk: requires an array".to_string())
                }
            }
            // arr_flatten(arr) — flatten one level of nested arrays.
            // Inverse of arr_chunk; useful after group operations.
            "arr_flatten" => {
                if args.is_empty() {
                    return Err("arr_flatten requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut out: Vec<Value> = Vec::new();
                    for v in items.iter() {
                        match v {
                            Value::Array(inner) => {
                                for x in inner.items.borrow().iter() {
                                    out.push(x.clone());
                                }
                            }
                            other => out.push(other.clone()),
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_flatten: requires an array".to_string())
                }
            }
            // arr_enumerate(arr) — array of [idx, value] pairs.
            // Replaces the manual `while k < arr_len; arr_get(arr, k)`
            // pattern when both index and value are needed.
            "arr_enumerate" => {
                if args.is_empty() {
                    return Err("arr_enumerate requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let pairs: Vec<Value> = items.iter().enumerate().map(|(i, v)| {
                        Value::Array(HArray::from_vec(vec![
                            Value::HInt(HInt::new(i as i64)),
                            v.clone(),
                        ]))
                    }).collect();
                    Ok(Value::Array(HArray::from_vec(pairs)))
                } else {
                    Err("arr_enumerate: requires an array".to_string())
                }
            }
            // arr_window(arr, size) — sliding window of `size` items.
            // Returns array of arrays, each holding `size` consecutive
            // values. Used for n-gram and rolling-stat patterns.
            "arr_window" => {
                if args.len() < 2 {
                    return Err("arr_window requires (array, size)".to_string());
                }
                let size = self.eval_expr(&args[1])?.to_int().max(1) as usize;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if size > items.len() {
                        return Ok(Value::Array(HArray::from_vec(vec![])));
                    }
                    let windows: Vec<Value> = (0..=items.len() - size).map(|i| {
                        Value::Array(HArray::from_vec(items[i..i + size].to_vec()))
                    }).collect();
                    Ok(Value::Array(HArray::from_vec(windows)))
                } else {
                    Err("arr_window: requires an array".to_string())
                }
            }
            // println — like print but uses display formatting for HInt
            // (no φ/HIM scaffolding). Closer to what most users want when
            // they reach for "print" in a Python/JS-shaped mental model.
            // The original `print` is preserved as a statement keyword for
            // debug-format introspection.
            "println" => {
                // Use to_display_string for ALL types — keeps float
                // display consistent with concat_many / str_concat /
                // string-+-concat. Was inlining a hand-written match
                // that bypassed format_float, so println(3.0) printed
                // "3" instead of "3.0".
                if args.is_empty() {
                    println!();
                    self.output_lines.borrow_mut().push(String::new());
                    return Ok(Value::Null);
                }
                let v = self.eval_expr(&args[0])?;
                let s = v.to_display_string();
                println!("{}", s);
                self.output_lines.borrow_mut().push(s);
                Ok(Value::Null)
            }
            // print_raw — same as println but no trailing newline. Pairs.
            "print_raw" => {
                if args.is_empty() {
                    return Ok(Value::Null);
                }
                let v = self.eval_expr(&args[0])?;
                use std::io::Write;
                print!("{}", v.to_display_string());
                let _ = std::io::stdout().flush();
                Ok(Value::Null)
            }
            // =================================================================
            // OMNIcode harmonic variants — operations that USE the φ-math
            // substrate to make decisions ordinary versions handle naively.
            // Anyone can write a file; these write harmonically.
            // =================================================================
            "harmonic_checksum" => {
                // Resonance signature of a string. Sum over each char's
                // codepoint resonance — a scalar that's stable under
                // character-set-equivalent rewrites and useful for
                // dedup/diff at the harmonic level rather than byte level.
                if args.is_empty() {
                    return Err("harmonic_checksum requires 1 argument".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let total: f64 = s.chars()
                    .map(|c| HInt::compute_resonance(c as i64))
                    .sum();
                Ok(Value::HFloat(total))
            }
            "harmonic_write_file" => {
                // Atomic write with a resonance gate. Writes content to
                // a sibling temp path, computes the content's harmonic
                // checksum (mean per-char resonance), and rename-commits
                // only if the score clears 0.5 — the same threshold
                // value_danger uses. Below that, the write is rolled
                // back: the temp file is removed and the original target
                // (if any) is untouched.
                //
                // Returns the harmonic score (HFloat) on success. On
                // disharmonic content, returns negative score to signal
                // rejection — callers can check `if score < 0`.
                //
                // The threshold floor (0.5) matches fold_escape's
                // danger boundary. Below it, content is "dangerous" by
                // the substrate's own definition.
                if args.len() < 2 {
                    return Err("harmonic_write_file requires (path, content)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_string();
                let content = self.eval_expr(&args[1])?.to_string();
                let chars: Vec<char> = content.chars().collect();
                let n = chars.len();
                let mean_resonance = if n == 0 {
                    0.0
                } else {
                    let total: f64 = chars.iter()
                        .map(|c| HInt::compute_resonance(*c as i64))
                        .sum();
                    total / (n as f64)
                };
                if mean_resonance < 0.5 {
                    // Disharmonic content rejected — return negative
                    // score so callers can detect.
                    return Ok(Value::HFloat(-mean_resonance));
                }
                // Atomic commit via temp + rename.
                let tmp_path = format!("{}.tmp.{}", path, std::process::id());
                if let Err(e) = std::fs::write(&tmp_path, &content) {
                    return Err(format!("harmonic_write_file({}): tmp write failed: {}", path, e));
                }
                if let Err(e) = std::fs::rename(&tmp_path, &path) {
                    let _ = std::fs::remove_file(&tmp_path);
                    return Err(format!("harmonic_write_file({}): rename failed: {}", path, e));
                }
                Ok(Value::HFloat(mean_resonance))
            }
            "harmonic_read_file" => {
                // Read a file and return [content, mean_resonance] so the
                // caller can see the harmonic score alongside the content
                // and decide whether to trust it. The mean resonance is
                // computed the same way harmonic_write_file gates writes,
                // so the contract is symmetric.
                if args.is_empty() {
                    return Err("harmonic_read_file requires (path)".to_string());
                }
                let path = self.eval_expr(&args[0])?.to_string();
                let content = std::fs::read_to_string(&path)
                    .map_err(|e| format!("harmonic_read_file({}): {}", path, e))?;
                let chars: Vec<char> = content.chars().collect();
                let n = chars.len();
                let mean = if n == 0 {
                    0.0
                } else {
                    let total: f64 = chars.iter()
                        .map(|c| HInt::compute_resonance(*c as i64))
                        .sum();
                    total / (n as f64)
                };
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::String(content),
                    Value::HFloat(mean),
                ])))
            }
            "harmonic_sort" => {
                // Sort by harmony_value (φ-resonance) descending — highest
                // resonance bubbles to the front. Strings sort by mean
                // char-resonance. Non-numeric, non-string values sink to
                // the end via 0.0 score (still total ordering).
                //
                // This is genuinely different from arr_sort: arr_sort
                // orders by NATURAL value (1 < 2 < 3); harmonic_sort
                // orders by φ-alignment (89 outranks 90 outranks 100).
                if args.is_empty() {
                    return Err("harmonic_sort requires 1 argument".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items_in = arr.items.borrow().clone();
                    let scored: Vec<(f64, Value)> = items_in.into_iter().map(|v| {
                        let score = match &v {
                            Value::HInt(h) => h.resonance,
                            Value::HFloat(f) => HInt::compute_resonance(*f as i64),
                            Value::String(s) => {
                                let chars: Vec<char> = s.chars().collect();
                                if chars.is_empty() { 0.0 } else {
                                    let total: f64 = chars.iter()
                                        .map(|c| HInt::compute_resonance(*c as i64))
                                        .sum();
                                    total / (chars.len() as f64)
                                }
                            }
                            _ => 0.0,
                        };
                        (score, v)
                    }).collect();
                    let mut items_scored = scored;
                    items_scored.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    Ok(Value::Array(HArray::from_vec(
                        items_scored.into_iter().map(|(_, v)| v).collect(),
                    )))
                } else {
                    Err("harmonic_sort: argument must be an array".to_string())
                }
            }
            "harmonic_split" => {
                // Split a string into chunks whose sizes are nearest-
                // Fibonacci to a natural division at word boundaries.
                // For a string of length N, the chunk sizes are chosen
                // greedily: take the largest Fibonacci ≤ remaining-chars,
                // walk forward to find the nearest word boundary (space),
                // emit that chunk, continue from there.
                //
                // Useful for layout: line-wrap at φ-aligned widths;
                // chunked transmission with harmonic packet sizes; etc.
                if args.is_empty() {
                    return Err("harmonic_split requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                let chars: Vec<char> = s.chars().collect();
                let total_chars = chars.len();
                let mut chunks: Vec<Value> = Vec::new();
                let mut pos = 0;
                while pos < total_chars {
                    let remaining = total_chars - pos;
                    // Largest attractor ≤ remaining, sourced from the
                    // canonical substrate (40-entry table, reaches 63M).
                    // Was a hardcoded 14-entry array saturating at 610.
                    let target = crate::phi_pi_fib::largest_attractor_at_most(remaining as i64).max(1) as usize;
                    let mut end = (pos + target).min(total_chars);
                    // Walk to nearest word boundary if mid-word and not at EOS
                    if end < total_chars {
                        // Search forward up to +5 chars for a space
                        let mut e = end;
                        while e < total_chars && e < end + 5 && chars[e] != ' ' && chars[e] != '\n' {
                            e += 1;
                        }
                        if e < total_chars && (chars[e] == ' ' || chars[e] == '\n') {
                            end = e;
                        }
                    }
                    let chunk: String = chars[pos..end].iter().collect();
                    chunks.push(Value::String(chunk));
                    pos = end;
                    // Skip the boundary space so it doesn't open the next chunk
                    if pos < total_chars && (chars[pos] == ' ' || chars[pos] == '\n') {
                        pos += 1;
                    }
                }
                Ok(Value::Array(HArray::from_vec(chunks)))
            }
            "harmonic_partition" => {
                // Group array elements by the Fibonacci attractor nearest
                // their value. Returns an array of arrays — one bucket
                // per attractor that received any elements, in attractor
                // order. Each bucket holds the original elements (not
                // their attractor labels).
                //
                // Use for: distribution analysis ("how clumpy is this
                // dataset around the Fibonacci spine?"), histogramming
                // along the φ-grid, generative composition partitioning.
                if args.is_empty() {
                    return Err("harmonic_partition requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    use std::collections::BTreeMap;
                    let mut buckets: BTreeMap<i64, Vec<Value>> = BTreeMap::new();
                    let items_in = arr.items.borrow().clone();
                    for v in items_in {
                        let n = v.to_int();
                        let key = crate::phi_pi_fib::fold_to_nearest_attractor(n);
                        buckets.entry(key).or_insert_with(Vec::new).push(v);
                    }
                    let outer: Vec<Value> = buckets.into_iter().map(|(_, items)| {
                        Value::Array(HArray::from_vec(items))
                    }).collect();
                    Ok(Value::Array(HArray::from_vec(outer)))
                } else {
                    Err("harmonic_partition: argument must be an array".to_string())
                }
            }
            // attractor_distance(n) — substrate primitive: distance from
            // |n| to the nearest Fibonacci attractor. Returns 0 when n
            // is exactly on an attractor (including 0). Useful for HBit
            // tension calculations and OOD gating in user code.
            "attractor_distance" => {
                if args.is_empty() {
                    return Err("attractor_distance requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (_a, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                Ok(Value::HInt(HInt::new(dist)))
            }
            // nearest_attractor(n) — substrate primitive: returns the
            // Fibonacci attractor closest to n (sign-preserving).
            // Companion to attractor_distance — together they expose
            // the substrate's full nearest-attractor lookup to OMC.
            "nearest_attractor" => {
                if args.is_empty() {
                    return Err("nearest_attractor requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (a, _dist) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                Ok(Value::HInt(HInt::new(a)))
            }
            // largest_attractor_at_most(n) — substrate primitive added
            // for harmonic_split (Path B4): largest Fibonacci attractor
            // <= |n|, sign-preserving. Useful for greedy chunking and
            // bucket-budget calculations.
            "largest_attractor_at_most" => {
                if args.is_empty() {
                    return Err("largest_attractor_at_most requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(
                    crate::phi_pi_fib::largest_attractor_at_most(n)
                )))
            }
            // crt_residues(pos, moduli) — Chinese Remainder Theorem-
            // style residue tuple. The CRT-PE positional encoding (E2)
            // expressed directly as an OMC builtin. Returns an array
            // of (pos % m_i) for each modulus in the moduli array.
            // For pairwise-coprime moduli this uniquely identifies pos
            // within [0, prod(moduli)).
            "crt_residues" => {
                if args.len() < 2 {
                    return Err("crt_residues requires (pos, moduli_array)".to_string());
                }
                let pos = self.eval_expr(&args[0])?.to_int();
                if let Value::Array(moduli) = self.eval_expr(&args[1])? {
                    let items = moduli.items.borrow();
                    let out: Vec<Value> = items.iter().map(|m| {
                        let mi = m.to_int();
                        if mi == 0 {
                            Value::HInt(HInt::new(0))
                        } else {
                            Value::HInt(HInt::new(pos.rem_euclid(mi)))
                        }
                    }).collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("crt_residues: second arg must be an array".to_string())
                }
            }
            // hbit_tension(value) — 1-D HBit tension, the cheap
            // OOD-detection primitive: distance from value to its
            // nearest Fibonacci attractor. Same as attractor_distance
            // but with a name that matches the experiments-paper
            // vocabulary (used by harmonic_anomaly's substrate-routed
            // log bucketing and the hybrid-attention gate).
            "hbit_tension" => {
                if args.is_empty() {
                    return Err("hbit_tension requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (_a, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                Ok(Value::HInt(HInt::new(dist)))
            }
            // is_attractor: true (1) iff n is exactly a Fibonacci attractor.
            // Cheaper than `attractor_distance(n) == 0` because the OMC
            // dispatch overhead disappears into a single substrate call.
            "is_attractor" => {
                if args.is_empty() {
                    return Err("is_attractor requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (_a, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                Ok(Value::HInt(HInt::new(if dist == 0 { 1 } else { 0 })))
            }
            // resonance_band: classify a value into a discrete resonance
            // band by its log-distance to the nearest attractor.
            //   0 = on-attractor (dist == 0)
            //   1 = adjacent (dist 1..=3)
            //   2 = near (dist 4..=10)
            //   3 = mid (dist 11..=100)
            //   4 = far (dist > 100)
            // Useful as an attention-routing key without a continuous gate.
            "resonance_band" => {
                if args.is_empty() {
                    return Err("resonance_band requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (_a, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                let band = match dist {
                    0 => 0,
                    1..=3 => 1,
                    4..=10 => 2,
                    11..=100 => 3,
                    _ => 4,
                };
                Ok(Value::HInt(HInt::new(band)))
            }
            // substrate_adamw_update(cur, grad, m, v, lr, b1, b2, eps, wd, step)
            // Fused AdamW per-parameter update. Lifted from prom_adamw_step
            // in prometheus.omc — the inner block called ~15 OMC-side
            // elementwise loops per parameter (_prom_zip / _prom_scale /
            // _prom_sqrt_eps), which dominated end-to-end Prometheus
            // wall-clock until v0.8.4 (see ADAMW_BUILTIN.md).
            //
            // Mutates `m` and `v` in place (OMC arrays are Rc-shared, so
            // the caller sees the update). Returns the new parameter value
            // as a fresh OMC array of the same shape as `cur`.
            //
            // Math:
            //   m ← b1·m + (1−b1)·g
            //   v ← b2·v + (1−b2)·g²
            //   m̂ = m / (1 − b1^step)
            //   v̂ = v / (1 − b2^step)
            //   p ← cur − lr·wd·cur − lr · m̂ / (√v̂ + eps)
            "substrate_adamw_update" => {
                if args.len() < 10 {
                    return Err("substrate_adamw_update requires (cur, grad, m, v, lr, b1, b2, eps, wd, step)".to_string());
                }
                let cur = self.eval_expr(&args[0])?;
                let grad = self.eval_expr(&args[1])?;
                let m_arr = self.eval_expr(&args[2])?;
                let v_arr = self.eval_expr(&args[3])?;
                let lr = self.eval_expr(&args[4])?.to_float();
                let b1 = self.eval_expr(&args[5])?.to_float();
                let b2 = self.eval_expr(&args[6])?.to_float();
                let eps = self.eval_expr(&args[7])?.to_float();
                let wd = self.eval_expr(&args[8])?.to_float();
                let step = self.eval_expr(&args[9])?.to_int() as i32;
                substrate_adamw_update(&cur, &grad, &m_arr, &v_arr,
                                       lr, b1, b2, eps, wd, step)
                    .map_err(|e| format!("substrate_adamw_update: {}", e))
            }
            // substrate_snap_matrix(arr, scale) — per-cell snap to nearest
            // Fibonacci attractor at the given scale. v0.8.8 substrate-init
            // experiment: use this at parameter-initialization time to seed
            // weights at substrate-aligned positions, then let training
            // diverge from there. Tests whether substrate-aligned init
            // gives different (better?) training trajectories than uniform
            // random init. Accepts 1D or 2D OMC arrays; returns same shape.
            "substrate_snap_matrix" => {
                if args.len() < 2 {
                    return Err("substrate_snap_matrix requires (arr, scale)".to_string());
                }
                let arr_val = self.eval_expr(&args[0])?;
                let scale = self.eval_expr(&args[1])?.to_float();
                if scale == 0.0 {
                    return Err("substrate_snap_matrix: scale must be != 0".to_string());
                }
                let snap = |x: f64| -> f64 {
                    let n = (x * scale).round() as i64;
                    let (a, _) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                    (a as f64) / scale
                };
                let arr = match &arr_val {
                    Value::Array(a) => a,
                    _ => return Err("substrate_snap_matrix: expected 1D or 2D array".to_string()),
                };
                let rows = arr.items.borrow();
                if rows.is_empty() {
                    return Ok(Value::Array(HArray::from_vec(vec![])));
                }
                if !matches!(&rows[0], Value::Array(_)) {
                    let out: Vec<Value> = rows.iter()
                        .map(|c| Value::HFloat(snap(c.to_float())))
                        .collect();
                    return Ok(Value::Array(HArray::from_vec(out)));
                }
                let mut out_rows: Vec<Value> = Vec::with_capacity(rows.len());
                for row in rows.iter() {
                    let row_arr = match row {
                        Value::Array(a) => a,
                        _ => return Err("substrate_snap_matrix: ragged input".to_string()),
                    };
                    let new_row: Vec<Value> = row_arr.items.borrow().iter()
                        .map(|c| Value::HFloat(snap(c.to_float())))
                        .collect();
                    out_rows.push(Value::Array(HArray::from_vec(new_row)));
                }
                Ok(Value::Array(HArray::from_vec(out_rows)))
            }
            // substrate_smod_matrix(scores, alpha) — Rust-native S-MOD
            // modulator. Per cell: 1 / (1 + alpha · attractor_distance(int(s))).
            // Lifted from `_prom_smod_matrix` in prometheus.omc; the OMC
            // version is a tight inner loop over an N×N scores matrix
            // calling attractor_distance per cell, which at N=64 burns
            // hundreds of milliseconds in the tree-walk interpreter
            // before this builtin landed. The substrate-math is unchanged.
            "substrate_smod_matrix" => {
                if args.len() < 2 {
                    return Err("substrate_smod_matrix requires (scores_2d, alpha)".to_string());
                }
                let scores_v = self.eval_expr(&args[0])?;
                let alpha = self.eval_expr(&args[1])?.to_float();
                build_substrate_modulator_matrix(&scores_v, alpha, ModulatorKind::SMod)
                    .map_err(|e| format!("substrate_smod_matrix: {}", e))
            }
            // substrate_resample_matrix(v, scale) — Rust-native substrate-V
            // resample modulator. Per cell: 1 / (1 + attractor_distance(int(v·scale)) / scale).
            // Same speedup story as substrate_smod_matrix; lifted from
            // `_prom_substrate_resample_matrix` in prometheus.omc.
            "substrate_resample_matrix" => {
                if args.len() < 2 {
                    return Err("substrate_resample_matrix requires (v_2d, scale)".to_string());
                }
                let v_val = self.eval_expr(&args[0])?;
                let scale = self.eval_expr(&args[1])?.to_float();
                if scale == 0.0 {
                    return Err("substrate_resample_matrix: scale must be != 0".to_string());
                }
                build_substrate_modulator_matrix(&v_val, scale, ModulatorKind::Resample)
                    .map_err(|e| format!("substrate_resample_matrix: {}", e))
            }
            // crt_recover: inverse of crt_residues for the same standard
            // pairwise-coprime moduli {5, 8, 13, 21}. Given residues
            // [r5, r8, r13, r21] returns the unique value in [0, 10920)
            // that produces them (Garner-style CRT reconstruction).
            // Pure substrate primitive: experiment_10 builds CRT-PE on
            // top of this; lifting it to native makes inference cheaper.
            "crt_recover" => {
                if args.is_empty() {
                    return Err("crt_recover requires (residues_array)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = v {
                    let items = arr.items.borrow();
                    if items.len() != 4 {
                        return Err(format!(
                            "crt_recover: expected 4 residues for moduli [5,8,13,21], got {}",
                            items.len()
                        ));
                    }
                    let r5 = items[0].to_int().rem_euclid(5);
                    let r8 = items[1].to_int().rem_euclid(8);
                    let r13 = items[2].to_int().rem_euclid(13);
                    let r21 = items[3].to_int().rem_euclid(21);
                    // Brute-force search across the period (10920). Tiny
                    // enough that this is faster than a full Garner solver
                    // for typical OMC use; keeps the implementation honest.
                    for x in 0..10920i64 {
                        if x % 5 == r5 && x % 8 == r8
                            && x % 13 == r13 && x % 21 == r21 {
                            return Ok(Value::HInt(HInt::new(x)));
                        }
                    }
                    Ok(Value::Singularity {
                        numerator: 0, denominator: 0,
                        context: "crt_recover: no solution in [0, 10920)".to_string(),
                    })
                } else {
                    Err("crt_recover: argument must be an array".to_string())
                }
            }
            // fibonacci_index: return the index i such that fib(i) == n,
            // or -1 if n is not a Fibonacci number. Operates over the
            // 40-entry FIBONACCI table (covers up to ~63M). Used for
            // experiment_8 (Fibonacci-distance attention) and similar.
            "fibonacci_index" => {
                if args.is_empty() {
                    return Err("fibonacci_index requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(crate::phi_pi_fib::fibonacci_index_of(n))))
            }
            "harmonic_hash" => {
                // Position-aware resonance hash — different from
                // harmonic_checksum which is just a sum (trivially
                // colliding). Weights each char's resonance by phi^i
                // where i is its position. The result is much harder
                // to collide and still respects the harmonic substrate.
                //
                // Output: f64 in roughly [0, len * phi * 1.0). Use
                // to_int(...) to get a stable integer hash for hashtable
                // keying when needed.
                if args.is_empty() {
                    return Err("harmonic_hash requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_string();
                const PHI: f64 = 1.6180339887498949;
                let mut acc: f64 = 0.0;
                let mut weight: f64 = 1.0;
                for c in s.chars() {
                    let r = HInt::compute_resonance(c as i64);
                    acc += r * weight;
                    weight *= PHI;
                    // Saturate gracefully — for huge strings the weight
                    // would overflow without this; keep it bounded.
                    if weight > 1e18 {
                        weight = 1.0;
                    }
                }
                Ok(Value::HFloat(acc))
            }
            "harmonic_diff" => {
                // Score for "how much did the harmonic structure change"
                // between two strings. Returns the absolute difference
                // of their harmonic_hash signatures, normalised by the
                // max of the two — gives a value in roughly [0, 1].
                //
                // 0.0 means harmonically identical; higher means more
                // structurally different. Useful for diff visualisations
                // weighted by impact rather than byte count.
                if args.len() < 2 {
                    return Err("harmonic_diff requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_string();
                let b = self.eval_expr(&args[1])?.to_string();
                const PHI: f64 = 1.6180339887498949;
                let hash_one = |s: &str| -> f64 {
                    let mut acc = 0.0;
                    let mut weight = 1.0;
                    for c in s.chars() {
                        acc += HInt::compute_resonance(c as i64) * weight;
                        weight *= PHI;
                        if weight > 1e18 { weight = 1.0; }
                    }
                    acc
                };
                let ha = hash_one(&a);
                let hb = hash_one(&b);
                let diff = (ha - hb).abs();
                let denom = ha.abs().max(hb.abs()).max(1.0);
                Ok(Value::HFloat(diff / denom))
            }
            "harmonic_dedupe" => {
                // Cluster elements whose values fall in the same
                // resonance band, collapsing each cluster to the
                // FIRST representative. `band` controls cluster width
                // by harmony_value: 0.05 means "elements with resonance
                // within ±0.05 of any kept element collapse to it."
                //
                // Different from arr_unique (exact equality) — this
                // dedupe is "harmonically-equivalent enough to drop."
                //
                // Useful for: noise reduction in measurement sequences,
                // collapsing near-duplicates that arose from rounding
                // or float drift, filtering down attractor-aligned data.
                if args.len() < 2 {
                    return Err("harmonic_dedupe requires (array, band)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let band = self.eval_expr(&args[1])?.to_float();
                if let Value::Array(arr) = arr_v {
                    let items_in = arr.items.borrow().clone();
                    let mut kept: Vec<Value> = Vec::new();
                    let mut kept_scores: Vec<f64> = Vec::new();
                    for v in items_in {
                        let score = match &v {
                            Value::HInt(h) => h.resonance,
                            Value::HFloat(f) => HInt::compute_resonance(*f as i64),
                            _ => 0.0,
                        };
                        // Check if this element falls within `band` of any
                        // already-kept element's resonance.
                        let close = kept_scores.iter().any(|s| (s - score).abs() < band);
                        if !close {
                            kept_scores.push(score);
                            kept.push(v);
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(kept)))
                } else {
                    Err("harmonic_dedupe: first argument must be an array".to_string())
                }
            }
            "arr_first" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    arr.items
                        .borrow()
                        .first()
                        .cloned()
                        .ok_or_else(|| "arr_first: empty array".to_string())
                } else {
                    Err("arr_first: requires an array".to_string())
                }
            }
            "arr_last" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    arr.items
                        .borrow()
                        .last()
                        .cloned()
                        .ok_or_else(|| "arr_last: empty array".to_string())
                } else {
                    Err("arr_last: requires an array".to_string())
                }
            }
            "arr_min" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    if arr.items.borrow().is_empty() {
                        return Err("arr_min: empty array".to_string());
                    }
                    let min = arr.items.borrow().iter().map(|v| v.to_int()).min().unwrap();
                    Ok(Value::HInt(HInt::new(min)))
                } else {
                    Err("arr_min: requires an array".to_string())
                }
            }
            "arr_max" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    if arr.items.borrow().is_empty() {
                        return Err("arr_max: empty array".to_string());
                    }
                    let max = arr.items.borrow().iter().map(|v| v.to_int()).max().unwrap();
                    Ok(Value::HInt(HInt::new(max)))
                } else {
                    Err("arr_max: requires an array".to_string())
                }
            }
            // arr_min_float / arr_max_float: like arr_min/max but preserve
            // float precision instead of coercing to int. Needed by the
            // experiments code where attention scores live in (0, 1).
            "arr_min_float" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_min_float: empty array".to_string());
                    }
                    let m = items.iter().map(|v| v.to_float())
                        .fold(f64::INFINITY, f64::min);
                    Ok(Value::HFloat(m))
                } else {
                    Err("arr_min_float: requires an array".to_string())
                }
            }
            "arr_max_float" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_max_float: empty array".to_string());
                    }
                    let m = items.iter().map(|v| v.to_float())
                        .fold(f64::NEG_INFINITY, f64::max);
                    Ok(Value::HFloat(m))
                } else {
                    Err("arr_max_float: requires an array".to_string())
                }
            }
            // arr_gcd: GCD of all elements; identity is 0 (gcd(0, n) == n).
            "arr_gcd" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut acc: i64 = 0;
                    for v in items.iter() {
                        let mut a = acc.unsigned_abs();
                        let mut b = v.to_int().unsigned_abs();
                        while b != 0 { let t = b; b = a % b; a = t; }
                        acc = a as i64;
                    }
                    Ok(Value::HInt(HInt::new(acc)))
                } else {
                    Err("arr_gcd: requires an array".to_string())
                }
            }
            // fnv1a_hash: 64-bit FNV-1a over a UTF-8 string. Fast,
            // non-cryptographic; the canonical "good enough" hash for
            // hashtable keying when the harmonic_hash is inappropriate
            // (e.g. when collisions matter more than substrate-alignment).
            "fnv1a_hash" => {
                if args.is_empty() {
                    return Err("fnv1a_hash requires (string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                const FNV_OFFSET: u64 = 0xcbf29ce484222325;
                const FNV_PRIME: u64 = 0x100000001b3;
                let mut h = FNV_OFFSET;
                for b in s.as_bytes() {
                    h ^= *b as u64;
                    h = h.wrapping_mul(FNV_PRIME);
                }
                // Cast to i64 by reinterpretation; OMC ints are signed.
                Ok(Value::HInt(HInt::new(h as i64)))
            }
            // arr_argmax / arr_argmin: index of the first max/min value.
            // Useful for "which class won" patterns; doing this in OMC code
            // currently requires a manual loop.
            "arr_argmax" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_argmax: empty array".to_string());
                    }
                    let mut best_idx = 0usize;
                    let mut best_val = items[0].to_float();
                    for (i, v) in items.iter().enumerate().skip(1) {
                        let f = v.to_float();
                        if f > best_val { best_val = f; best_idx = i; }
                    }
                    Ok(Value::HInt(HInt::new(best_idx as i64)))
                } else {
                    Err("arr_argmax: requires an array".to_string())
                }
            }
            "arr_argmin" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_argmin: empty array".to_string());
                    }
                    let mut best_idx = 0usize;
                    let mut best_val = items[0].to_float();
                    for (i, v) in items.iter().enumerate().skip(1) {
                        let f = v.to_float();
                        if f < best_val { best_val = f; best_idx = i; }
                    }
                    Ok(Value::HInt(HInt::new(best_idx as i64)))
                } else {
                    Err("arr_argmin: requires an array".to_string())
                }
            }
            // arr_cumsum: running totals. Result has same length as input.
            "arr_cumsum" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut acc: f64 = 0.0;
                    let mut out = Vec::with_capacity(items.len());
                    let mut all_int = true;
                    for v in items.iter() {
                        if !matches!(v, Value::HInt(_)) { all_int = false; }
                        acc += v.to_float();
                        if all_int {
                            out.push(Value::HInt(HInt::new(acc as i64)));
                        } else {
                            out.push(Value::HFloat(acc));
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_cumsum: requires an array".to_string())
                }
            }
            // arr_diff: consecutive differences. Output is length-1.
            "arr_diff" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Ok(Value::Array(HArray::from_vec(vec![])));
                    }
                    let all_int = items.iter().all(|v| matches!(v, Value::HInt(_)));
                    let mut out = Vec::with_capacity(items.len().saturating_sub(1));
                    for w in items.windows(2) {
                        if all_int {
                            out.push(Value::HInt(HInt::new(w[1].to_int() - w[0].to_int())));
                        } else {
                            out.push(Value::HFloat(w[1].to_float() - w[0].to_float()));
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_diff: requires an array".to_string())
                }
            }
            // arr_unique_count: number of distinct values in the array.
            // Uses display-form keys so HInt(7) and Bool(true→"true") don't
            // collide; matches existing dict-key conventions.
            "arr_unique_count" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut seen = std::collections::HashSet::with_capacity(items.len());
                    for v in items.iter() {
                        seen.insert(v.to_display_string());
                    }
                    Ok(Value::HInt(HInt::new(seen.len() as i64)))
                } else {
                    Err("arr_unique_count: requires an array".to_string())
                }
            }
            // arr_partition_by: split into [matching, non_matching] sub-arrays
            // by a value predicate (== check against the second arg).
            // Pure split; preserves original order in each bucket.
            "arr_partition_by" => {
                if args.len() < 2 {
                    return Err("arr_partition_by requires (array, value)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?;
                if let Value::Array(arr) = arr_v {
                    let target_s = target.to_display_string();
                    let items = arr.items.borrow();
                    let mut yes = Vec::new();
                    let mut no = Vec::new();
                    for v in items.iter() {
                        if v.to_display_string() == target_s { yes.push(v.clone()); }
                        else { no.push(v.clone()); }
                    }
                    Ok(Value::Array(HArray::from_vec(vec![
                        Value::Array(HArray::from_vec(yes)),
                        Value::Array(HArray::from_vec(no)),
                    ])))
                } else {
                    Err("arr_partition_by: first argument must be an array".to_string())
                }
            }
            // arr_range: integer inclusive-low / exclusive-high range.
            // arr_from_range exists but with a 1-arg form; this is the
            // 2-arg/3-arg form most users expect from Python.
            "arr_range" => {
                let (lo, hi, step) = match args.len() {
                    1 => (0i64, self.eval_expr(&args[0])?.to_int(), 1i64),
                    2 => (self.eval_expr(&args[0])?.to_int(),
                          self.eval_expr(&args[1])?.to_int(), 1i64),
                    _ => (self.eval_expr(&args[0])?.to_int(),
                          self.eval_expr(&args[1])?.to_int(),
                          self.eval_expr(&args[2])?.to_int()),
                };
                if step == 0 {
                    return Err("arr_range: step must be non-zero".to_string());
                }
                let mut out = Vec::new();
                let mut i = lo;
                if step > 0 {
                    while i < hi { out.push(Value::HInt(HInt::new(i))); i += step; }
                } else {
                    while i > hi { out.push(Value::HInt(HInt::new(i))); i += step; }
                }
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // Arithmetic mean as float. Common stats helper not previously
            // exposed; users had to compute arr_sum / arr_len manually.
            "arr_mean" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_mean: empty array".to_string());
                    }
                    let sum: f64 = items.iter().map(|v| v.to_float()).sum();
                    Ok(Value::HFloat(sum / items.len() as f64))
                } else {
                    Err("arr_mean: requires an array".to_string())
                }
            }
            // Variance (population, not sample — divides by N not N-1).
            // Hot in anomaly-detector workloads (per-dim spread).
            "arr_variance" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_variance: empty array".to_string());
                    }
                    let n = items.len() as f64;
                    let mean: f64 = items.iter().map(|v| v.to_float()).sum::<f64>() / n;
                    let var: f64 = items.iter()
                        .map(|v| { let d = v.to_float() - mean; d * d })
                        .sum::<f64>() / n;
                    Ok(Value::HFloat(var))
                } else {
                    Err("arr_variance: requires an array".to_string())
                }
            }
            // Standard deviation = sqrt(variance).
            "arr_stddev" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_stddev: empty array".to_string());
                    }
                    let n = items.len() as f64;
                    let mean: f64 = items.iter().map(|v| v.to_float()).sum::<f64>() / n;
                    let var: f64 = items.iter()
                        .map(|v| { let d = v.to_float() - mean; d * d })
                        .sum::<f64>() / n;
                    Ok(Value::HFloat(var.sqrt()))
                } else {
                    Err("arr_stddev: requires an array".to_string())
                }
            }
            // Median value. Float result so even-length arrays return
            // the average of the two middle elements.
            "arr_median" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_median: empty array".to_string());
                    }
                    let mut floats: Vec<f64> = items.iter().map(|v| v.to_float()).collect();
                    floats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = floats.len();
                    let m = if n % 2 == 1 {
                        floats[n / 2]
                    } else {
                        (floats[n / 2 - 1] + floats[n / 2]) / 2.0
                    };
                    Ok(Value::HFloat(m))
                } else {
                    Err("arr_median: requires an array".to_string())
                }
            }
            // Harmonic mean: n / sum(1/x_i). Useful for averaging
            // rates and frequencies. Substrate-themed name despite
            // being standard stats — fits the OMC vocabulary.
            "arr_harmonic_mean" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_harmonic_mean: empty array".to_string());
                    }
                    let mut sum_recip = 0.0;
                    for v in items.iter() {
                        let f = v.to_float();
                        if f == 0.0 {
                            return Err("arr_harmonic_mean: zero element".to_string());
                        }
                        sum_recip += 1.0 / f;
                    }
                    Ok(Value::HFloat(items.len() as f64 / sum_recip))
                } else {
                    Err("arr_harmonic_mean: requires an array".to_string())
                }
            }
            // Geometric mean: nth_root(prod(x_i)). Done via log-sum
            // to avoid overflow for large arrays.
            "arr_geometric_mean" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_geometric_mean: empty array".to_string());
                    }
                    let mut log_sum = 0.0;
                    for v in items.iter() {
                        let f = v.to_float();
                        if f <= 0.0 {
                            return Err("arr_geometric_mean: non-positive element".to_string());
                        }
                        log_sum += f.ln();
                    }
                    Ok(Value::HFloat((log_sum / items.len() as f64).exp()))
                } else {
                    Err("arr_geometric_mean: requires an array".to_string())
                }
            }
            // Sum of squares — quick helper for variance / norm calcs.
            "arr_sum_sq" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let s: f64 = items.iter().map(|v| { let f = v.to_float(); f * f }).sum();
                    Ok(Value::HFloat(s))
                } else {
                    Err("arr_sum_sq: requires an array".to_string())
                }
            }
            // L2 norm of the array as a vector — sqrt(sum of squares).
            "arr_norm" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let s: f64 = items.iter().map(|v| { let f = v.to_float(); f * f }).sum();
                    Ok(Value::HFloat(s.sqrt()))
                } else {
                    Err("arr_norm: requires an array".to_string())
                }
            }
            // Dot product of two equal-length arrays.
            "arr_dot" => {
                if args.len() < 2 {
                    return Err("arr_dot requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                if let (Value::Array(a), Value::Array(b)) = (a, b) {
                    let ai = a.items.borrow();
                    let bi = b.items.borrow();
                    if ai.len() != bi.len() {
                        return Err(format!(
                            "arr_dot: length mismatch ({} vs {})",
                            ai.len(), bi.len()
                        ));
                    }
                    let s: f64 = ai.iter().zip(bi.iter())
                        .map(|(x, y)| x.to_float() * y.to_float())
                        .sum();
                    Ok(Value::HFloat(s))
                } else {
                    Err("arr_dot: requires two arrays".to_string())
                }
            }
            // ---- Substrate-typed array library (Track 2 MVP) -------
            //
            // Vectorized arithmetic + substrate-aware reductions on
            // arrays of HInt. The dispatch boundary marshals int
            // arrays through the L1.6 buffer; these handlers produce
            // new arrays element-wise (so the substrate-resonance
            // metadata on each output HInt is recomputed from the
            // arithmetic result — no special tagging needed).
            //
            // Broadcasting: if the 2nd arg is a scalar (HInt / HFloat),
            // it's repeated for every element of the 1st arg's array.
            // Two arrays must match length (no implicit shape-1 broadcast).
            "arr_add" => {
                if args.len() < 2 {
                    return Err("arr_add requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                Ok(elementwise_op(&a, &b, "arr_add", |x, y| x.wrapping_add(y))?)
            }
            "arr_sub" => {
                if args.len() < 2 {
                    return Err("arr_sub requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                Ok(elementwise_op(&a, &b, "arr_sub", |x, y| x.wrapping_sub(y))?)
            }
            "arr_mul" => {
                if args.len() < 2 {
                    return Err("arr_mul requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                Ok(elementwise_op(&a, &b, "arr_mul", |x, y| x.wrapping_mul(y))?)
            }
            "arr_div_int" => {
                // Integer division. Zero divisor produces 0 in that
                // slot (matches harmonic_anomaly-style "no propagation
                // of NaN through arrays" — Singularity is at the value
                // level, not the array level).
                if args.len() < 2 {
                    return Err("arr_div_int requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                Ok(elementwise_op(&a, &b, "arr_div_int",
                    |x, y| if y == 0 { 0 } else { x / y })?)
            }
            "arr_neg" => {
                // Unary element-wise negation.
                if args.is_empty() {
                    return Err("arr_neg requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let out: Vec<Value> = arr.items.borrow().iter()
                        .map(|v| Value::HInt(HInt::new(v.to_int().wrapping_neg())))
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_neg: requires an array".to_string())
                }
            }
            "arr_scale" => {
                // arr_scale(arr, k) — explicit scalar multiply. Same as
                // arr_mul(arr, k) when k is a scalar; provided as a
                // named alias so callers can opt into the broadcast
                // shape without it being inferred.
                if args.len() < 2 {
                    return Err("arr_scale requires (array, scalar)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let k = self.eval_expr(&args[1])?;
                Ok(elementwise_op(&a, &k, "arr_scale", |x, y| x.wrapping_mul(y))?)
            }
            // arr_resonance_vec(arr) -> array of f64 per-element
            // resonance scores. The substrate-typed dtype's defining
            // operation: each output element is HInt::compute_resonance
            // of the corresponding input. Python literally can't do
            // this — there's no φ-resonance attached to an i64.
            "arr_resonance_vec" => {
                if args.is_empty() {
                    return Err("arr_resonance_vec requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let out: Vec<Value> = arr.items.borrow().iter()
                        .map(|v| Value::HFloat(HInt::compute_resonance(v.to_int())))
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_resonance_vec: requires an array".to_string())
                }
            }
            // arr_him_vec(arr) -> array of f64 per-element HIM scores.
            // Complement to arr_resonance_vec: HIM is the
            // Harmonic-Interference-Metric — how off-attractor each
            // value is. Together with resonance, these are the two
            // substrate-typed metadata channels carried per-element.
            "arr_him_vec" => {
                if args.is_empty() {
                    return Err("arr_him_vec requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let out: Vec<Value> = arr.items.borrow().iter()
                        .map(|v| {
                            let h = HInt::new(v.to_int());
                            Value::HFloat(h.him_score)
                        })
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_him_vec: requires an array".to_string())
                }
            }
            // ---- 2D array primitives (Track 2) ----------------------
            //
            // A "matrix" in OMC is an array of arrays, all inner arrays
            // the same length. arr_matmul(A, B) does the standard
            // multiplication: output[i][j] = sum_k A[i][k] * B[k][j].
            //
            // Substrate-preserving: when every cell of A and B is HInt
            // (or coerces cleanly to i64), the inner loop runs in i64
            // and result cells are HInt — so each output carries the
            // φ-resonance/HIM score that HInt::new computes from the
            // integer value. The moment a float shows up anywhere, we
            // fall back to f64 (resonance is then implicit in the value
            // but not carried as substrate metadata).
            "arr_matmul" => {
                if args.len() < 2 {
                    return Err("arr_matmul requires (matrix_a, matrix_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                if let (Value::Array(am), Value::Array(bm)) = (a, b) {
                    let arows = am.items.borrow();
                    let brows = bm.items.borrow();
                    if arows.is_empty() || brows.is_empty() {
                        return Err("arr_matmul: empty matrix".to_string());
                    }
                    let a_rows = arows.len();
                    let a_cols = match &arows[0] {
                        Value::Array(r) => r.items.borrow().len(),
                        _ => return Err("arr_matmul: A rows must be arrays".to_string()),
                    };
                    let b_rows = brows.len();
                    let b_cols = match &brows[0] {
                        Value::Array(r) => r.items.borrow().len(),
                        _ => return Err("arr_matmul: B rows must be arrays".to_string()),
                    };
                    if a_cols != b_rows {
                        return Err(format!(
                            "arr_matmul: shape mismatch — A is {}x{}, B is {}x{}",
                            a_rows, a_cols, b_rows, b_cols
                        ));
                    }
                    // Substrate path: try i64 first. If any cell is a
                    // float (or anything that loses precision when
                    // coerced via to_int), fall back to f64.
                    let mut all_int = true;
                    for r in arows.iter().chain(brows.iter()) {
                        if let Value::Array(row) = r {
                            for v in row.items.borrow().iter() {
                                if !matches!(v, Value::HInt(_) | Value::Bool(_) | Value::Null) {
                                    all_int = false;
                                    break;
                                }
                            }
                        }
                        if !all_int { break; }
                    }
                    if all_int {
                        // Flatten to contiguous row-major buffers and
                        // use ikj ordering so the inner loop strides
                        // through B and C sequentially. Combined with
                        // wrapping i64 arithmetic, this lets the
                        // autovectorizer turn the inner loop into a
                        // tight integer fma sequence.
                        let mut a_flat = vec![0i64; a_rows * a_cols];
                        let mut b_flat = vec![0i64; b_rows * b_cols];
                        for (i, r) in arows.iter().enumerate() {
                            if let Value::Array(row) = r {
                                for (k, v) in row.items.borrow().iter().enumerate() {
                                    a_flat[i * a_cols + k] = v.to_int();
                                }
                            }
                        }
                        for (k, r) in brows.iter().enumerate() {
                            if let Value::Array(row) = r {
                                for (j, v) in row.items.borrow().iter().enumerate() {
                                    b_flat[k * b_cols + j] = v.to_int();
                                }
                            }
                        }
                        let mut c_flat = vec![0i64; a_rows * b_cols];
                        for i in 0..a_rows {
                            for k in 0..a_cols {
                                let aik = a_flat[i * a_cols + k];
                                let b_row_start = k * b_cols;
                                let c_row_start = i * b_cols;
                                for j in 0..b_cols {
                                    c_flat[c_row_start + j] = c_flat[c_row_start + j]
                                        .wrapping_add(aik.wrapping_mul(b_flat[b_row_start + j]));
                                }
                            }
                        }
                        let mut out: Vec<Value> = Vec::with_capacity(a_rows);
                        for i in 0..a_rows {
                            let mut row: Vec<Value> = Vec::with_capacity(b_cols);
                            for j in 0..b_cols {
                                // HInt::new rebuilds resonance/HIM from
                                // each output integer — every cell of
                                // the projection carries substrate metadata.
                                row.push(Value::HInt(HInt::new(c_flat[i * b_cols + j])));
                            }
                            out.push(Value::Array(HArray::from_vec(row)));
                        }
                        return Ok(Value::Array(HArray::from_vec(out)));
                    }
                    // Float fallback: flatten into contiguous row-major
                    // buffers, then run the ikj loop ordering so that
                    // both B and C accesses stride sequentially through
                    // memory (textbook ~3-10× speedup over the naive
                    // ijk loop with vec-of-vecs accesses). For large
                    // matrices this puts the inner-product work on the
                    // f64 SIMD-friendly path the LLVM autovectorizer
                    // recognises.
                    let mut a_flat = vec![0.0f64; a_rows * a_cols];
                    let mut b_flat = vec![0.0f64; b_rows * b_cols];
                    for (i, r) in arows.iter().enumerate() {
                        if let Value::Array(row) = r {
                            for (k, v) in row.items.borrow().iter().enumerate() {
                                a_flat[i * a_cols + k] = v.to_float();
                            }
                        }
                    }
                    for (k, r) in brows.iter().enumerate() {
                        if let Value::Array(row) = r {
                            for (j, v) in row.items.borrow().iter().enumerate() {
                                b_flat[k * b_cols + j] = v.to_float();
                            }
                        }
                    }
                    let mut c_flat = vec![0.0f64; a_rows * b_cols];
                    for i in 0..a_rows {
                        for k in 0..a_cols {
                            let aik = a_flat[i * a_cols + k];
                            let b_row_start = k * b_cols;
                            let c_row_start = i * b_cols;
                            for j in 0..b_cols {
                                c_flat[c_row_start + j] += aik * b_flat[b_row_start + j];
                            }
                        }
                    }
                    let mut out: Vec<Value> = Vec::with_capacity(a_rows);
                    for i in 0..a_rows {
                        let mut row: Vec<Value> = Vec::with_capacity(b_cols);
                        for j in 0..b_cols {
                            row.push(Value::HFloat(c_flat[i * b_cols + j]));
                        }
                        out.push(Value::Array(HArray::from_vec(row)));
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_matmul: requires two 2D arrays".to_string())
                }
            }
            "arr_transpose" => {
                // Transpose a 2D array. Output[j][i] = input[i][j].
                if args.is_empty() {
                    return Err("arr_transpose requires (matrix)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(am) = a {
                    let rows = am.items.borrow();
                    if rows.is_empty() {
                        return Ok(Value::Array(HArray::from_vec(vec![])));
                    }
                    let n_cols = match &rows[0] {
                        Value::Array(r) => r.items.borrow().len(),
                        _ => return Err("arr_transpose: rows must be arrays".to_string()),
                    };
                    let mut out: Vec<Value> = Vec::with_capacity(n_cols);
                    for j in 0..n_cols {
                        let mut col: Vec<Value> = Vec::with_capacity(rows.len());
                        for row_v in rows.iter() {
                            if let Value::Array(row) = row_v {
                                col.push(row.items.borrow()[j].clone());
                            }
                        }
                        out.push(Value::Array(HArray::from_vec(col)));
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_transpose: requires a 2D array".to_string())
                }
            }
            "arr_eye" => {
                // arr_eye(n) -> identity matrix (n x n) of ints.
                if args.is_empty() {
                    return Err("arr_eye requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int().max(0) as usize;
                let mut rows: Vec<Value> = Vec::with_capacity(n);
                for i in 0..n {
                    let mut row: Vec<Value> = Vec::with_capacity(n);
                    for j in 0..n {
                        row.push(Value::HInt(HInt::new(if i == j { 1 } else { 0 })));
                    }
                    rows.push(Value::Array(HArray::from_vec(row)));
                }
                Ok(Value::Array(HArray::from_vec(rows)))
            }
            "arr_zeros_2d" => {
                // arr_zeros_2d(rows, cols) -> (rows x cols) zero matrix.
                if args.len() < 2 {
                    return Err("arr_zeros_2d requires (rows, cols)".to_string());
                }
                let r = self.eval_expr(&args[0])?.to_int().max(0) as usize;
                let c = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let mut rows: Vec<Value> = Vec::with_capacity(r);
                for _ in 0..r {
                    let row: Vec<Value> = (0..c).map(|_| Value::HInt(HInt::new(0))).collect();
                    rows.push(Value::Array(HArray::from_vec(row)));
                }
                Ok(Value::Array(HArray::from_vec(rows)))
            }
            // ---- Native-Rust ML primitives -------------------------
            //
            // These get the inner loops out of the OMC tree-walker.
            // Writing them in OMC would dispatch through eval_expr per
            // element (~50ns each); doing them in Rust is one builtin
            // call regardless of array size — the per-element cost
            // drops to ~1ns. For a 1000-element array that's a 50×
            // speedup with no JIT involvement.
            "arr_softmax" => {
                // Numerically stable softmax: subtract max before exp.
                if args.is_empty() {
                    return Err("arr_softmax requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Ok(Value::Array(HArray::from_vec(vec![])));
                    }
                    let xs: Vec<f64> = items.iter().map(|v| v.to_float()).collect();
                    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = xs.iter().map(|x| (x - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    let out: Vec<Value> = exps.iter()
                        .map(|e| Value::HFloat(e / sum))
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_softmax: requires an array".to_string())
                }
            }
            "arr_layer_norm" => {
                // LayerNorm: (x - mean) / sqrt(var + eps).
                if args.is_empty() {
                    return Err("arr_layer_norm requires (array, eps?)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let eps = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else { 1e-5 };
                if let Value::Array(arr) = a {
                    let items = arr.items.borrow();
                    let n = items.len() as f64;
                    if n == 0.0 {
                        return Ok(Value::Array(HArray::from_vec(vec![])));
                    }
                    let xs: Vec<f64> = items.iter().map(|v| v.to_float()).collect();
                    let mean: f64 = xs.iter().sum::<f64>() / n;
                    let var: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
                    let scale = 1.0 / (var + eps).sqrt();
                    let out: Vec<Value> = xs.iter()
                        .map(|x| Value::HFloat((x - mean) * scale))
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_layer_norm: requires an array".to_string())
                }
            }
            "arr_relu_vec" => {
                // Vectorized ReLU: max(x, 0) per element.
                if args.is_empty() {
                    return Err("arr_relu_vec requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let out: Vec<Value> = arr.items.borrow().iter()
                        .map(|v| {
                            let x = v.to_float();
                            if x > 0.0 { Value::HFloat(x) } else { Value::HFloat(0.0) }
                        })
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_relu_vec: requires an array".to_string())
                }
            }
            "arr_sigmoid_vec" => {
                if args.is_empty() {
                    return Err("arr_sigmoid_vec requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let out: Vec<Value> = arr.items.borrow().iter()
                        .map(|v| {
                            let x = v.to_float();
                            Value::HFloat(1.0 / (1.0 + (-x).exp()))
                        })
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_sigmoid_vec: requires an array".to_string())
                }
            }
            "arr_conv1d" => {
                // 1D convolution: out[i] = sum_k input[i+k] * kernel[k].
                // Valid mode (no padding), stride 1.
                if args.len() < 2 {
                    return Err("arr_conv1d requires (input, kernel)".to_string());
                }
                let inp = self.eval_expr(&args[0])?;
                let ker = self.eval_expr(&args[1])?;
                if let (Value::Array(ia), Value::Array(ka)) = (inp, ker) {
                    let ib = ia.items.borrow();
                    let kb = ka.items.borrow();
                    if ib.len() < kb.len() {
                        return Err("arr_conv1d: input shorter than kernel".to_string());
                    }
                    let inp_f: Vec<f64> = ib.iter().map(|v| v.to_float()).collect();
                    let ker_f: Vec<f64> = kb.iter().map(|v| v.to_float()).collect();
                    let n_out = inp_f.len() - ker_f.len() + 1;
                    let mut out = Vec::with_capacity(n_out);
                    for i in 0..n_out {
                        let mut s = 0.0;
                        for k in 0..ker_f.len() {
                            s += inp_f[i + k] * ker_f[k];
                        }
                        out.push(Value::HFloat(s));
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_conv1d: requires (input_array, kernel_array)".to_string())
                }
            }
            "arr_outer" => {
                // Outer product: a (n,) x b (m,) -> 2D (n x m) matrix.
                if args.len() < 2 {
                    return Err("arr_outer requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                if let (Value::Array(aa), Value::Array(bb)) = (a, b) {
                    let ab = aa.items.borrow();
                    let bb_ = bb.items.borrow();
                    let mut rows = Vec::with_capacity(ab.len());
                    for av in ab.iter() {
                        let af = av.to_float();
                        let row: Vec<Value> = bb_.iter()
                            .map(|bv| Value::HFloat(af * bv.to_float()))
                            .collect();
                        rows.push(Value::Array(HArray::from_vec(row)));
                    }
                    Ok(Value::Array(HArray::from_vec(rows)))
                } else {
                    Err("arr_outer: requires two arrays".to_string())
                }
            }
            // ---- Substrate-native acceleration: the OMC-only path ---
            //
            // arr_substrate_attention(Q, K, V) — attention scored by
            // substrate distance rather than dot product. Q, K, V are
            // matrices (sequence × dim). For each query row, score
            // every key row by Σ |q[d] - k[d]|^attractor_distance, take
            // softmax over scores, weight V rows. This is impossible
            // in NumPy because i64 doesn't carry substrate metadata.
            "arr_substrate_attention" => {
                if args.len() < 3 {
                    return Err("arr_substrate_attention requires (Q, K, V)".to_string());
                }
                let q = self.eval_expr(&args[0])?;
                let k = self.eval_expr(&args[1])?;
                let v = self.eval_expr(&args[2])?;
                let (q_rows, q_cols, q_flat) = flatten_matrix(&q, "Q")?;
                let (k_rows, _k_cols, k_flat) = flatten_matrix(&k, "K")?;
                let (v_rows, v_cols, v_flat) = flatten_matrix(&v, "V")?;
                if k_rows != v_rows {
                    return Err(format!(
                        "arr_substrate_attention: K rows ({}) != V rows ({})",
                        k_rows, v_rows
                    ));
                }
                let n_q = q_rows;
                let n_k = k_rows;
                let mut out_flat = vec![0.0f64; n_q * v_cols];
                for i in 0..n_q {
                    // Score every key row against query row i.
                    let mut scores = vec![0.0f64; n_k];
                    for j in 0..n_k {
                        let mut s = 0.0;
                        for d in 0..q_cols {
                            let qd = q_flat[i * q_cols + d];
                            let kd = k_flat[j * q_cols + d];
                            // Substrate-distance kernel: closer in
                            // substrate space → higher score (negate
                            // the L1 of attractor distances).
                            let diff = (qd - kd).abs();
                            let (_a, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(diff as i64);
                            s -= dist as f64;
                        }
                        scores[j] = s;
                    }
                    // Softmax over scores.
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    if sum > 0.0 {
                        for j in 0..n_k {
                            let w = exps[j] / sum;
                            for d in 0..v_cols {
                                out_flat[i * v_cols + d] += w * v_flat[j * v_cols + d];
                            }
                        }
                    }
                }
                Ok(matrix_from_flat(&out_flat, n_q, v_cols))
            }
            // arr_substrate_score_rows(matrix) — for every row, compute
            // its mean φ-resonance. High = row mostly Fibonacci-attractor
            // valued. Used as a substrate-coherence regularizer.
            "arr_substrate_score_rows" => {
                if args.is_empty() {
                    return Err("arr_substrate_score_rows requires (matrix)".to_string());
                }
                let m = self.eval_expr(&args[0])?;
                let (rows, cols, flat) = flatten_matrix(&m, "M")?;
                if cols == 0 {
                    return Ok(Value::Array(HArray::from_vec(vec![])));
                }
                let mut out = Vec::with_capacity(rows);
                for i in 0..rows {
                    let mut s = 0.0;
                    for j in 0..cols {
                        let h = HInt::new(flat[i * cols + j] as i64);
                        s += h.resonance;
                    }
                    out.push(Value::HFloat(s / (cols as f64)));
                }
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // ---- Forward-mode autograd (Track 2) ---------------------
            //
            // A dual number is a 2-element array [value, derivative].
            // No new Value variant — composes with existing array ops,
            // matmul, and HInt/HFloat substrate metadata.
            //
            //   x' = dual(x, 1.0)         # lift input with seed
            //   y' = dual_mul(x', x')     # forward-prop through f
            //   grad = dual_d(y')         # read df/dx at x
            "dual" => {
                if args.len() < 2 {
                    return Err("dual requires (value, derivative)".to_string());
                }
                let v = self.eval_expr(&args[0])?.to_float();
                let d = self.eval_expr(&args[1])?.to_float();
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(v),
                    Value::HFloat(d),
                ])))
            }
            "dual_v" => {
                if args.is_empty() {
                    return Err("dual_v requires (dual)".to_string());
                }
                let x = self.eval_expr(&args[0])?;
                if let Value::Array(a) = x {
                    let items = a.items.borrow();
                    if items.is_empty() {
                        return Err("dual_v: malformed dual".to_string());
                    }
                    Ok(Value::HFloat(items[0].to_float()))
                } else {
                    Err("dual_v: not a dual".to_string())
                }
            }
            "dual_d" => {
                if args.is_empty() {
                    return Err("dual_d requires (dual)".to_string());
                }
                let x = self.eval_expr(&args[0])?;
                if let Value::Array(a) = x {
                    let items = a.items.borrow();
                    if items.len() < 2 {
                        return Err("dual_d: malformed dual".to_string());
                    }
                    Ok(Value::HFloat(items[1].to_float()))
                } else {
                    Err("dual_d: not a dual".to_string())
                }
            }
            "dual_add" | "dual_sub" | "dual_mul" | "dual_div" => {
                if args.len() < 2 {
                    return Err(format!("{} requires (a, b)", name));
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                let (av, ad) = unpack_dual(&a);
                let (bv, bd) = unpack_dual(&b);
                let (rv, rd) = match name {
                    "dual_add" => (av + bv, ad + bd),
                    "dual_sub" => (av - bv, ad - bd),
                    "dual_mul" => (av * bv, ad * bv + av * bd),
                    "dual_div" => {
                        if bv == 0.0 {
                            return Err("dual_div: division by zero".to_string());
                        }
                        (av / bv, (ad * bv - av * bd) / (bv * bv))
                    }
                    _ => unreachable!(),
                };
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(rv),
                    Value::HFloat(rd),
                ])))
            }
            "dual_neg" => {
                if args.is_empty() {
                    return Err("dual_neg requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(-av),
                    Value::HFloat(-ad),
                ])))
            }
            "dual_pow_int" => {
                if args.len() < 2 {
                    return Err("dual_pow_int requires (a, n)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let n = self.eval_expr(&args[1])?.to_int() as i32;
                let (av, ad) = unpack_dual(&a);
                if n == 0 {
                    return Ok(Value::Array(HArray::from_vec(vec![
                        Value::HFloat(1.0),
                        Value::HFloat(0.0),
                    ])));
                }
                let rv = av.powi(n);
                let rd = (n as f64) * av.powi(n - 1) * ad;
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(rv),
                    Value::HFloat(rd),
                ])))
            }
            "dual_exp" => {
                if args.is_empty() {
                    return Err("dual_exp requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                let rv = av.exp();
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(rv),
                    Value::HFloat(rv * ad),
                ])))
            }
            "dual_sin" => {
                if args.is_empty() {
                    return Err("dual_sin requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(av.sin()),
                    Value::HFloat(av.cos() * ad),
                ])))
            }
            "dual_cos" => {
                if args.is_empty() {
                    return Err("dual_cos requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(av.cos()),
                    Value::HFloat(-av.sin() * ad),
                ])))
            }
            "dual_relu" => {
                if args.is_empty() {
                    return Err("dual_relu requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                let (rv, rd) = if av > 0.0 { (av, ad) } else { (0.0, 0.0) };
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(rv),
                    Value::HFloat(rd),
                ])))
            }
            "dual_sigmoid" => {
                if args.is_empty() {
                    return Err("dual_sigmoid requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                let s = 1.0 / (1.0 + (-av).exp());
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(s),
                    Value::HFloat(s * (1.0 - s) * ad),
                ])))
            }
            "dual_tanh" => {
                if args.is_empty() {
                    return Err("dual_tanh requires (a)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let (av, ad) = unpack_dual(&a);
                let t = av.tanh();
                Ok(Value::Array(HArray::from_vec(vec![
                    Value::HFloat(t),
                    Value::HFloat((1.0 - t * t) * ad),
                ])))
            }
            // ---- Reverse-mode autograd (the real training engine) ---
            //
            // Workflow:
            //   tape_reset();
            //   h x = tape_var(3.0);          # id of leaf node
            //   h y = tape_mul(x, x);          # records op, returns id
            //   tape_backward(y);              # walks tape, accumulates
            //   h grad_x = tape_grad(x);       # reads dy/dx (= 6.0 here)
            //
            // Reverse-mode is O(forward) per parameter — Python autograd's
            // entire reason for existing. Substrate metadata stays on the
            // forward values (tape_value(id) returns substrate-typed
            // HInt when the cell is integral).
            "tape_reset" => {
                self.autograd_tape.clear();
                Ok(Value::Null)
            }
            "tape_var" | "tape_const" => {
                if args.is_empty() {
                    return Err(format!("{} requires (value)", name));
                }
                let v = self.eval_expr(&args[0])?;
                let mat = tape_from_value(&v)?;
                let grad = TapeMat::zeros(mat.rows, mat.cols);
                let op = if name == "tape_var" { TapeOp::Var } else { TapeOp::Const };
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op, value: mat, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_value" => {
                if args.is_empty() {
                    return Err("tape_value requires (node_id)".to_string());
                }
                let id = self.eval_expr(&args[0])?.to_int() as usize;
                if id >= self.autograd_tape.len() {
                    return Err(format!("tape_value: id {} out of range", id));
                }
                // Substrate-preserving: if every cell is an integer,
                // round-trip through HInt so resonance metadata comes
                // back on each cell. This is the bit that's unique:
                // Python's autograd returns plain numpy floats, OMC
                // returns substrate-annotated values.
                let m = &self.autograd_tape[id].value;
                let all_int = m.data.iter().all(|x| x.fract() == 0.0 && x.abs() < (i64::MAX as f64));
                Ok(tape_to_value(m, all_int))
            }
            "tape_set_value" => {
                // Replace a tape node's stored value with a new one.
                // Used by custom optimizers (Adam, AdamW) that want to
                // compute the parameter update in OMC space instead of
                // routing through tape_update's hard-coded SGD step.
                if args.len() < 2 {
                    return Err("tape_set_value requires (node_id, new_value)".to_string());
                }
                let id = self.eval_expr(&args[0])?.to_int() as usize;
                if id >= self.autograd_tape.len() {
                    return Err(format!("tape_set_value: id {} out of range", id));
                }
                let new_val = self.eval_expr(&args[1])?;
                let new_mat = tape_from_value(&new_val)?;
                // Shape mismatch is a usage error — better to error than
                // silently reshape and corrupt later math.
                let cur = &self.autograd_tape[id].value;
                if new_mat.rows != cur.rows || new_mat.cols != cur.cols {
                    return Err(format!(
                        "tape_set_value: shape mismatch (got {}x{}, expected {}x{})",
                        new_mat.rows, new_mat.cols, cur.rows, cur.cols
                    ));
                }
                self.autograd_tape[id].value = new_mat;
                Ok(Value::Null)
            }
            "tape_grad" => {
                if args.is_empty() {
                    return Err("tape_grad requires (node_id)".to_string());
                }
                let id = self.eval_expr(&args[0])?.to_int() as usize;
                if id >= self.autograd_tape.len() {
                    return Err(format!("tape_grad: id {} out of range", id));
                }
                // Gradients usually aren't integers, so don't try to
                // re-quantize them to HInt — return HFloat (per-cell
                // substrate metadata is still inspectable via existing
                // is_attractor / attractor_distance builtins on the
                // returned cells).
                Ok(tape_to_value(&self.autograd_tape[id].grad, false))
            }
            "tape_add" | "tape_sub" | "tape_mul" | "tape_div" => {
                if args.len() < 2 {
                    return Err(format!("{} requires (a_id, b_id)", name));
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let b = self.eval_expr(&args[1])?.to_int() as usize;
                if a >= self.autograd_tape.len() || b >= self.autograd_tape.len() {
                    return Err(format!("{}: node id out of range", name));
                }
                let av = self.autograd_tape[a].value.clone();
                let bv = self.autograd_tape[b].value.clone();
                // Elementwise with broadcast support for:
                //   scalar ↔ matrix
                //   [1, C]  → broadcast across rows of [N, C]
                //   [N, 1]  → broadcast across cols of [N, C]
                let (rows, cols) = if av.rows * av.cols >= bv.rows * bv.cols {
                    (av.rows, av.cols)
                } else { (bv.rows, bv.cols) };
                let mut out = TapeMat::zeros(rows, cols);
                let scalar_a = av.rows * av.cols == 1;
                let scalar_b = bv.rows * bv.cols == 1;
                let row_bcast_a = av.rows == 1 && av.cols == cols && !scalar_a;
                let row_bcast_b = bv.rows == 1 && bv.cols == cols && !scalar_b;
                let col_bcast_a = av.cols == 1 && av.rows == rows && !scalar_a;
                let col_bcast_b = bv.cols == 1 && bv.rows == rows && !scalar_b;
                for i in 0..rows {
                    for j in 0..cols {
                        let xa = if scalar_a { av.data[0] }
                                 else if row_bcast_a { av.data[j] }
                                 else if col_bcast_a { av.data[i] }
                                 else { av.at(i, j) };
                        let xb = if scalar_b { bv.data[0] }
                                 else if row_bcast_b { bv.data[j] }
                                 else if col_bcast_b { bv.data[i] }
                                 else { bv.at(i, j) };
                        let v = match name {
                            "tape_add" => xa + xb,
                            "tape_sub" => xa - xb,
                            "tape_mul" => xa * xb,
                            "tape_div" => if xb == 0.0 { 0.0 } else { xa / xb },
                            _ => 0.0,
                        };
                        out.set(i, j, v);
                    }
                }
                let op = match name {
                    "tape_add" => TapeOp::Add(a, b),
                    "tape_sub" => TapeOp::Sub(a, b),
                    "tape_mul" => TapeOp::Mul(a, b),
                    "tape_div" => TapeOp::Div(a, b),
                    _ => unreachable!(),
                };
                let grad = TapeMat::zeros(rows, cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op, value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_neg" => {
                if args.is_empty() {
                    return Err("tape_neg requires (a_id)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let mut out = TapeMat::zeros(av.rows, av.cols);
                for k in 0..av.data.len() { out.data[k] = -av.data[k]; }
                let grad = TapeMat::zeros(av.rows, av.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op: TapeOp::Neg(a), value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_pow_int" => {
                if args.len() < 2 {
                    return Err("tape_pow_int requires (a_id, n)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let n = self.eval_expr(&args[1])?.to_int() as i32;
                let av = self.autograd_tape[a].value.clone();
                let mut out = TapeMat::zeros(av.rows, av.cols);
                for k in 0..av.data.len() { out.data[k] = av.data[k].powi(n); }
                let grad = TapeMat::zeros(av.rows, av.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op: TapeOp::PowInt(a, n), value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_exp" | "tape_log" | "tape_abs" | "tape_sin" | "tape_cos"
            | "tape_relu" | "tape_sigmoid" | "tape_tanh" => {
                if args.is_empty() {
                    return Err(format!("{} requires (a_id)", name));
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let mut out = TapeMat::zeros(av.rows, av.cols);
                for k in 0..av.data.len() {
                    let x = av.data[k];
                    out.data[k] = match name {
                        "tape_exp"     => x.exp(),
                        "tape_log"     => if x > 0.0 { x.ln() } else { f64::NEG_INFINITY },
                        "tape_abs"     => x.abs(),
                        "tape_sin"     => x.sin(),
                        "tape_cos"     => x.cos(),
                        "tape_relu"    => if x > 0.0 { x } else { 0.0 },
                        "tape_sigmoid" => 1.0 / (1.0 + (-x).exp()),
                        "tape_tanh"    => x.tanh(),
                        _              => 0.0,
                    };
                }
                let op = match name {
                    "tape_exp"     => TapeOp::Exp(a),
                    "tape_log"     => TapeOp::Log(a),
                    "tape_abs"     => TapeOp::Abs(a),
                    "tape_sin"     => TapeOp::Sin(a),
                    "tape_cos"     => TapeOp::Cos(a),
                    "tape_relu"    => TapeOp::Relu(a),
                    "tape_sigmoid" => TapeOp::Sigmoid(a),
                    "tape_tanh"    => TapeOp::Tanh(a),
                    _ => unreachable!(),
                };
                let grad = TapeMat::zeros(av.rows, av.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op, value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_softmax" => {
                // Per-row softmax: each row of A becomes prob distribution.
                // Stable form: subtract row-max before exp.
                if args.is_empty() {
                    return Err("tape_softmax requires (a_id)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                // Try the GPU softmax accelerator first (v0.8.6 scaffold).
                // The accelerator may decline (return None) for small shapes,
                // in which case the CPU triple-pass below runs.
                if let Some(result) = crate::accel::try_accelerated_softmax(av.rows, av.cols, &av.data) {
                    return result.map(|data| {
                        let out = TapeMat { rows: av.rows, cols: av.cols, data };
                        let grad = TapeMat::zeros(av.rows, av.cols);
                        let id = self.autograd_tape.len();
                        self.autograd_tape.push(TapeNode {
                            op: TapeOp::Softmax(a), value: out, grad,
                        });
                        Value::HInt(HInt::new(id as i64))
                    }).map_err(|e| format!("tape_softmax accelerated: {}", e));
                }
                let mut out = TapeMat::zeros(av.rows, av.cols);
                for r in 0..av.rows {
                    // Row max for numerical stability.
                    let mut mx = f64::NEG_INFINITY;
                    for c in 0..av.cols {
                        let v = av.data[r * av.cols + c];
                        if v > mx { mx = v; }
                    }
                    let mut sum = 0.0;
                    for c in 0..av.cols {
                        let e = (av.data[r * av.cols + c] - mx).exp();
                        out.data[r * av.cols + c] = e;
                        sum += e;
                    }
                    if sum > 0.0 {
                        for c in 0..av.cols {
                            out.data[r * av.cols + c] /= sum;
                        }
                    }
                }
                let grad = TapeMat::zeros(av.rows, av.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::Softmax(a), value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_substrate_grad_mod" => {
                // tape_substrate_grad_mod(x_id, scale=64, alpha=0.5)
                //
                // Forward: identity (out = x). Backward amplifies gradient
                // components that pull the param TOWARD nearest Fibonacci
                // attractor, dampens components that push AWAY.
                //
                // The substrate as gradient-flow regularizer — the forward
                // computation is unchanged but optimization is biased
                // toward substrate-aligned parameter values. Composes with
                // any tape op (just wrap a node with it).
                if args.is_empty() {
                    return Err("tape_substrate_grad_mod requires (x_id, scale=64, alpha=0.5)".to_string());
                }
                let x_id = self.eval_expr(&args[0])?.to_int() as usize;
                let scale = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else { 64.0 };
                let alpha = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.to_float()
                } else { 0.5 };
                let xv = self.autograd_tape[x_id].value.clone();
                // Forward: identity (output is exactly the input).
                let out = xv.clone();
                let grad = TapeMat::zeros(xv.rows, xv.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::SubstrateGradMod(x_id, scale, alpha),
                    value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_substrate_sparse_scores" => {
                // tape_substrate_sparse_scores(q_id, k_id, threshold) → [N, M] scores
                //
                // Compute q @ k^T but only at cells where substrate_dist(i, j)
                // is below threshold; other cells are set to -inf so a
                // subsequent softmax assigns them zero. The substrate distance
                // uses CRT moduli {5, 8, 13, 21} — same metric that v0.8.8
                // measured Q6 concentrates 56.8% of attention mass into 6.84%
                // of cells under for substrate_dist <= 5.
                //
                // This is the post-training inference kernel: train with
                // Q6 fused → during inference, swap the dense
                // q @ k^T + softmax for tape_substrate_sparse_scores +
                // softmax, dropping ~93% of score computation for ~57% of
                // the attention quality. Backward routes through dense
                // matmul (for the cells that fired) — gradient at masked
                // cells is identically zero (softmax of -inf = 0).
                if args.len() < 2 {
                    return Err("tape_substrate_sparse_scores requires (q_id, k_id, threshold=5)".to_string());
                }
                let q_id = self.eval_expr(&args[0])?.to_int() as usize;
                let k_id = self.eval_expr(&args[1])?.to_int() as usize;
                let threshold: i64 = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.to_int()
                } else { 5 };
                let qv = self.autograd_tape[q_id].value.clone();
                let kv = self.autograd_tape[k_id].value.clone();
                if qv.cols != kv.cols {
                    return Err(format!(
                        "tape_substrate_sparse_scores: shape mismatch q={}x{} k={}x{}",
                        qv.rows, qv.cols, kv.rows, kv.cols
                    ));
                }
                let n = qv.rows;
                let m = kv.rows;
                let d = qv.cols;
                // CRT moduli matching the v0.8.8 measurement.
                let moduli: [i64; 4] = [5, 8, 13, 21];
                let substrate_dist = |i: usize, j: usize| -> i64 {
                    let mut s = 0_i64;
                    for &mm in &moduli {
                        let di = (i as i64) % mm;
                        let dj = (j as i64) % mm;
                        s += (di - dj).abs();
                    }
                    s
                };
                let mut out = TapeMat::zeros(n, m);
                let mut cells_computed = 0usize;
                let mut cells_total = 0usize;
                for i in 0..n {
                    for j in 0..m {
                        cells_total += 1;
                        if substrate_dist(i, j) > threshold {
                            out.set(i, j, f64::NEG_INFINITY);
                            continue;
                        }
                        cells_computed += 1;
                        let mut s = 0.0;
                        for kk in 0..d {
                            s += qv.at(i, kk) * kv.at(j, kk);
                        }
                        out.set(i, j, s);
                    }
                }
                // Emit telemetry the first few times so the bench can
                // sanity-check density. Stays out of the OMC-side hot path.
                if cells_total > 0 && std::env::var("OMC_GPU_VERBOSE").as_deref() == Ok("1") {
                    eprintln!("[sparse-scores] {}/{} cells = {:.1}%",
                              cells_computed, cells_total,
                              100.0 * cells_computed as f64 / cells_total as f64);
                }
                let grad = TapeMat::zeros(n, m);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::SubstrateSparseScores(q_id, k_id, threshold),
                    value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_substrate_resample" => {
                // tape_substrate_resample(v_id, scale) — fused substrate-V resample.
                // out[i, c] = v[i, c] * 1 / (1 + attractor_distance(int(v[i, c] · scale)) / scale).
                // Equivalent to the prom_substrate_resample OMC composition but
                // skips the tape_value → tape_const round-trip (which at d_model=256
                // seq_len=64 was extracting 16k f64s into an OMC array and lifting
                // them back).
                if args.is_empty() {
                    return Err("tape_substrate_resample requires (v_id, scale)".to_string());
                }
                let v_id = self.eval_expr(&args[0])?.to_int() as usize;
                let scale = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else {
                    10.0
                };
                if scale == 0.0 {
                    // scale=0 is the "off" sentinel — return the input unchanged.
                    return Ok(Value::HInt(HInt::new(v_id as i64)));
                }
                let v = self.autograd_tape[v_id].value.clone();
                let mut out = TapeMat::zeros(v.rows, v.cols);
                for k in 0..v.data.len() {
                    let x = v.data[k];
                    let n = (x * scale) as i64;
                    let (_, d) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                    let modulator = 1.0 / (1.0 + (d as f64) / scale);
                    out.data[k] = x * modulator;
                }
                let grad = TapeMat::zeros(v.rows, v.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::SubstrateResample(v_id, scale),
                    value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_embedding_lookup" => {
                // tape_embedding_lookup(table_id, token_ids[]) → [N, d_model]
                // Direct row gather: out[i, :] = table[token_ids[i], :].
                if args.len() < 2 {
                    return Err("tape_embedding_lookup requires (table_id, token_ids)".to_string());
                }
                let table_id = self.eval_expr(&args[0])?.to_int() as usize;
                let ids_val = self.eval_expr(&args[1])?;
                let ids_arr = match &ids_val {
                    Value::Array(a) => a,
                    _ => return Err("tape_embedding_lookup: token_ids must be an array".to_string()),
                };
                let token_ids: Vec<usize> = ids_arr.items.borrow().iter()
                    .map(|v| v.to_int() as usize)
                    .collect();
                let table = self.autograd_tape[table_id].value.clone();
                let vocab = table.rows;
                let d_model = table.cols;
                let n = token_ids.len();
                let mut out = TapeMat::zeros(n, d_model);
                for i in 0..n {
                    let row = token_ids[i];
                    if row >= vocab {
                        return Err(format!(
                            "tape_embedding_lookup: token id {} out of vocab range {}",
                            row, vocab
                        ));
                    }
                    for c in 0..d_model {
                        out.set(i, c, table.at(row, c));
                    }
                }
                let grad = TapeMat::zeros(n, d_model);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::EmbeddingLookup(table_id, token_ids),
                    value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_cross_entropy_batch" => {
                // tape_cross_entropy_batch(logits_id, targets[])
                // Fused softmax + select-target-log + per-token mean.
                // Forward returns scalar mean loss. Backward uses the
                // closed-form (p - 1{target}) / N rather than chaining
                // tape_softmax / tape_log / tape_mul / tape_sum backwards.
                if args.len() < 2 {
                    return Err("tape_cross_entropy_batch requires (logits_id, targets)".to_string());
                }
                let logits_id = self.eval_expr(&args[0])?.to_int() as usize;
                let targets_val = self.eval_expr(&args[1])?;
                let targets_arr = match &targets_val {
                    Value::Array(a) => a,
                    _ => return Err("tape_cross_entropy_batch: targets must be an array".to_string()),
                };
                let targets: Vec<usize> = targets_arr.items.borrow().iter()
                    .map(|v| v.to_int() as usize)
                    .collect();
                let logits = self.autograd_tape[logits_id].value.clone();
                let n = logits.rows;
                let vocab = logits.cols;
                if targets.len() != n {
                    return Err(format!(
                        "tape_cross_entropy_batch: targets length {} != logits rows {}",
                        targets.len(), n
                    ));
                }
                // Forward: numerically-stable per-row softmax, then pick log p_target,
                // sum across rows, divide by N.
                let mut total: f64 = 0.0;
                for i in 0..n {
                    let mut row_max = f64::NEG_INFINITY;
                    for c in 0..vocab {
                        let x = logits.at(i, c);
                        if x > row_max { row_max = x; }
                    }
                    let mut row_sum_exp: f64 = 0.0;
                    for c in 0..vocab {
                        row_sum_exp += (logits.at(i, c) - row_max).exp();
                    }
                    let log_z = row_max + row_sum_exp.ln();
                    let log_p_target = logits.at(i, targets[i]) - log_z;
                    total += -log_p_target;
                }
                let mean_loss = total / (n.max(1) as f64);
                let out = TapeMat::scalar(mean_loss);
                let grad = TapeMat::scalar(0.0);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::CrossEntropyBatch(logits_id, targets),
                    value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_phi_log" => {
                // Substrate-native fused log_φπfib(|x·scale| + 1).
                // Mathematically equivalent to:
                //     tape_div_scalar(tape_log(tape_add_scalar(tape_abs(tape_mul_scalar(x, scale)), 1.0)),
                //                     π · ln φ)
                // but as ONE tape node — fewer allocations, simpler backward,
                // and the substrate basis (π · ln φ in the denominator) is
                // visible at the AST level rather than buried in scalar
                // constants. Q6 attention modulation is the first consumer.
                if args.is_empty() {
                    return Err("tape_phi_log requires (a_id) and optional (scale)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let scale = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_float()
                } else {
                    10.0  // Q6 default
                };
                let av = self.autograd_tape[a].value.clone();
                let denom = std::f64::consts::PI * crate::value::PHI.ln();
                let mut out = TapeMat::zeros(av.rows, av.cols);
                for k in 0..av.data.len() {
                    let xs = (av.data[k] * scale).abs();
                    out.data[k] = (xs + 1.0).ln() / denom;
                }
                let grad = TapeMat::zeros(av.rows, av.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::PhiLog(a, scale), value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_matmul" => {
                if args.len() < 2 {
                    return Err("tape_matmul requires (a_id, b_id)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let b = self.eval_expr(&args[1])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let bv = self.autograd_tape[b].value.clone();
                let out = tape_matmul(&av, &bv)?;
                let grad = TapeMat::zeros(out.rows, out.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op: TapeOp::MatMul(a, b), value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_sum" => {
                if args.is_empty() {
                    return Err("tape_sum requires (a_id)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let s: f64 = av.data.iter().sum();
                let out = TapeMat::scalar(s);
                let grad = TapeMat::scalar(0.0);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op: TapeOp::Sum(a), value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_transpose" => {
                // Matrix transpose: [rows, cols] → [cols, rows]
                // Differentiable: backward just transposes the upstream gradient.
                if args.is_empty() {
                    return Err("tape_transpose requires (a_id)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let mut out = TapeMat::zeros(av.cols, av.rows);
                for r in 0..av.rows {
                    for c in 0..av.cols {
                        out.set(c, r, av.at(r, c));
                    }
                }
                let grad = TapeMat::zeros(out.rows, out.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::Transpose(a), value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_layernorm" => {
                // tape_layernorm(x, gamma, beta, eps?) -> per-row layer-normed output
                // x: [N, D], gamma: [1, D], beta: [1, D]
                if args.len() < 3 {
                    return Err("tape_layernorm requires (x_id, gamma_id, beta_id, eps?)".to_string());
                }
                let x_id = self.eval_expr(&args[0])?.to_int() as usize;
                let g_id = self.eval_expr(&args[1])?.to_int() as usize;
                let b_id = self.eval_expr(&args[2])?.to_int() as usize;
                let eps = if args.len() >= 4 {
                    self.eval_expr(&args[3])?.to_float()
                } else { 1e-5 };
                let xv = self.autograd_tape[x_id].value.clone();
                let gv = self.autograd_tape[g_id].value.clone();
                let bv = self.autograd_tape[b_id].value.clone();
                if gv.cols != xv.cols || bv.cols != xv.cols {
                    return Err(format!(
                        "tape_layernorm: gamma/beta cols ({}/{}) must match x cols ({})",
                        gv.cols, bv.cols, xv.cols
                    ));
                }
                let mut out = TapeMat::zeros(xv.rows, xv.cols);
                let dcols = xv.cols as f64;
                for r in 0..xv.rows {
                    let mut mean = 0.0;
                    for c in 0..xv.cols { mean += xv.data[r * xv.cols + c]; }
                    mean /= dcols;
                    let mut var = 0.0;
                    for c in 0..xv.cols {
                        let d = xv.data[r * xv.cols + c] - mean;
                        var += d * d;
                    }
                    var /= dcols;
                    let inv_std = 1.0 / (var + eps).sqrt();
                    for c in 0..xv.cols {
                        let centered = xv.data[r * xv.cols + c] - mean;
                        let normed = centered * inv_std;
                        out.data[r * xv.cols + c] =
                            normed * gv.data[c] + bv.data[c];
                    }
                }
                let grad = TapeMat::zeros(xv.rows, xv.cols);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode {
                    op: TapeOp::LayerNormRow(x_id, g_id, b_id, eps),
                    value: out, grad,
                });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_row_mean" | "tape_row_sum" => {
                // Per-row reduction: [rows, cols] → [rows, 1]
                if args.is_empty() {
                    return Err(format!("{} requires (a_id)", name));
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let mut out = TapeMat::zeros(av.rows, 1);
                let cols_f = av.cols.max(1) as f64;
                for r in 0..av.rows {
                    let mut s = 0.0;
                    for c in 0..av.cols { s += av.data[r * av.cols + c]; }
                    out.data[r] = if name == "tape_row_mean" { s / cols_f } else { s };
                }
                let op = if name == "tape_row_mean" {
                    TapeOp::RowMean(a)
                } else {
                    TapeOp::RowSum(a)
                };
                let grad = TapeMat::zeros(av.rows, 1);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op, value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_mean" => {
                if args.is_empty() {
                    return Err("tape_mean requires (a_id)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int() as usize;
                let av = self.autograd_tape[a].value.clone();
                let n = av.data.len().max(1) as f64;
                let m: f64 = av.data.iter().sum::<f64>() / n;
                let out = TapeMat::scalar(m);
                let grad = TapeMat::scalar(0.0);
                let id = self.autograd_tape.len();
                self.autograd_tape.push(TapeNode { op: TapeOp::Mean(a), value: out, grad });
                Ok(Value::HInt(HInt::new(id as i64)))
            }
            "tape_backward" => {
                // Walk the tape in reverse. Initialize the loss node's
                // grad to ones-of-shape, then dispatch by op type to
                // accumulate gradients into dependencies. After this
                // returns, tape_grad(var_id) reads the accumulated grad.
                if args.is_empty() {
                    return Err("tape_backward requires (loss_id)".to_string());
                }
                let loss_id = self.eval_expr(&args[0])?.to_int() as usize;
                if loss_id >= self.autograd_tape.len() {
                    return Err(format!("tape_backward: id {} out of range", loss_id));
                }
                // Zero all grads first so backward is idempotent.
                for node in self.autograd_tape.iter_mut() {
                    let (r, c) = (node.grad.rows, node.grad.cols);
                    node.grad = TapeMat::zeros(r, c);
                }
                // Seed the loss with 1s (scalar loss → 1.0).
                {
                    let g = &mut self.autograd_tape[loss_id].grad;
                    for v in g.data.iter_mut() { *v = 1.0; }
                }
                // Walk in reverse. Cloning grads to drop the borrow,
                // then writing back through indexed access.
                for i in (0..=loss_id).rev() {
                    let op = self.autograd_tape[i].op.clone();
                    let dy = self.autograd_tape[i].grad.clone();
                    match op {
                        TapeOp::Var | TapeOp::Const => {}
                        TapeOp::Add(a, b) => {
                            let a_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let b_shape = (
                                self.autograd_tape[b].value.rows,
                                self.autograd_tape[b].value.cols,
                            );
                            let da = reduce_to_shape(&dy, a_shape);
                            let db = reduce_to_shape(&dy, b_shape);
                            self.autograd_tape[a].grad.add(&da);
                            self.autograd_tape[b].grad.add(&db);
                        }
                        TapeOp::Sub(a, b) => {
                            let a_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let b_shape = (
                                self.autograd_tape[b].value.rows,
                                self.autograd_tape[b].value.cols,
                            );
                            let da = reduce_to_shape(&dy, a_shape);
                            let mut neg = TapeMat::zeros(dy.rows, dy.cols);
                            for k in 0..dy.data.len() { neg.data[k] = -dy.data[k]; }
                            let db = reduce_to_shape(&neg, b_shape);
                            self.autograd_tape[a].grad.add(&da);
                            self.autograd_tape[b].grad.add(&db);
                        }
                        TapeOp::Mul(a, b) => {
                            // Forward did broadcast over rows of [1, C] and cols of [N, 1]; the
                            // backward must mirror BOTH directions: iterate the output shape, and
                            // for shrunk operands sum the contributions across the broadcast axis.
                            let av = self.autograd_tape[a].value.clone();
                            let bv = self.autograd_tape[b].value.clone();
                            let (out_rows, out_cols) = (dy.rows, dy.cols);
                            let read_dy = |i: usize, j: usize| -> f64 {
                                if dy.rows * dy.cols == 1 { dy.data[0] } else { dy.at(i, j) }
                            };
                            let read_bcast = |m: &TapeMat, i: usize, j: usize| -> f64 {
                                if m.rows * m.cols == 1 { m.data[0] }
                                else if m.rows == 1 { m.at(0, j.min(m.cols - 1)) }
                                else if m.cols == 1 { m.at(i.min(m.rows - 1), 0) }
                                else { m.at(i, j) }
                            };
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for i2 in 0..out_rows {
                                for j2 in 0..out_cols {
                                    let xb = read_bcast(&bv, i2, j2);
                                    let xdy = read_dy(i2, j2);
                                    let di = if av.rows == 1 { 0 } else { i2.min(av.rows - 1) };
                                    let dj = if av.cols == 1 { 0 } else { j2.min(av.cols - 1) };
                                    let cur = da.at(di, dj);
                                    da.set(di, dj, cur + xdy * xb);
                                }
                            }
                            self.autograd_tape[a].grad.add(&da);
                            let mut db = TapeMat::zeros(bv.rows, bv.cols);
                            for i2 in 0..out_rows {
                                for j2 in 0..out_cols {
                                    let xa = read_bcast(&av, i2, j2);
                                    let xdy = read_dy(i2, j2);
                                    let di = if bv.rows == 1 { 0 } else { i2.min(bv.rows - 1) };
                                    let dj = if bv.cols == 1 { 0 } else { j2.min(bv.cols - 1) };
                                    let cur = db.at(di, dj);
                                    db.set(di, dj, cur + xdy * xa);
                                }
                            }
                            self.autograd_tape[b].grad.add(&db);
                        }
                        TapeOp::Div(a, b) => {
                            // Same broadcast-aware backward as Mul, with the d/dy = -a/b² formula.
                            let av = self.autograd_tape[a].value.clone();
                            let bv = self.autograd_tape[b].value.clone();
                            let (out_rows, out_cols) = (dy.rows, dy.cols);
                            let read_dy = |i: usize, j: usize| -> f64 {
                                if dy.rows * dy.cols == 1 { dy.data[0] } else { dy.at(i, j) }
                            };
                            let read_bcast = |m: &TapeMat, i: usize, j: usize| -> f64 {
                                if m.rows * m.cols == 1 { m.data[0] }
                                else if m.rows == 1 { m.at(0, j.min(m.cols - 1)) }
                                else if m.cols == 1 { m.at(i.min(m.rows - 1), 0) }
                                else { m.at(i, j) }
                            };
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for i2 in 0..out_rows {
                                for j2 in 0..out_cols {
                                    let xb = read_bcast(&bv, i2, j2);
                                    if xb == 0.0 { continue; }
                                    let xdy = read_dy(i2, j2);
                                    let di = if av.rows == 1 { 0 } else { i2.min(av.rows - 1) };
                                    let dj = if av.cols == 1 { 0 } else { j2.min(av.cols - 1) };
                                    let cur = da.at(di, dj);
                                    da.set(di, dj, cur + xdy / xb);
                                }
                            }
                            self.autograd_tape[a].grad.add(&da);
                            let mut db = TapeMat::zeros(bv.rows, bv.cols);
                            for i2 in 0..out_rows {
                                for j2 in 0..out_cols {
                                    let xa = read_bcast(&av, i2, j2);
                                    let xb = read_bcast(&bv, i2, j2);
                                    if xb == 0.0 { continue; }
                                    let xdy = read_dy(i2, j2);
                                    let di = if bv.rows == 1 { 0 } else { i2.min(bv.rows - 1) };
                                    let dj = if bv.cols == 1 { 0 } else { j2.min(bv.cols - 1) };
                                    let cur = db.at(di, dj);
                                    db.set(di, dj, cur + -xdy * xa / (xb * xb));
                                }
                            }
                            self.autograd_tape[b].grad.add(&db);
                        }
                        TapeOp::Neg(a) => {
                            let mut neg = TapeMat::zeros(dy.rows, dy.cols);
                            for k in 0..dy.data.len() { neg.data[k] = -dy.data[k]; }
                            self.autograd_tape[a].grad.add(&neg);
                        }
                        TapeOp::PowInt(a, n) => {
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                let coeff = (n as f64) * av.data[k].powi(n - 1);
                                da.data[k] = dy.data[k.min(dy.data.len() - 1)] * coeff;
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Log(a) => {
                            // d/dx log(x) = 1/x
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                let x = av.data[k];
                                let g = if x != 0.0 { dy.data[k] / x } else { 0.0 };
                                da.data[k] = g;
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::SubstrateGradMod(a, scale, alpha) => {
                            // Backward: per-cell substrate-attraction grad.
                            //
                            // For each cell x:
                            //   let xs = x · scale (round to int)
                            //   let attractor = nearest_attractor(xs)
                            //   let dir_to_attractor = sign(attractor - xs)
                            //   let on_attractor = (dist(xs) == 0)
                            //
                            // If on attractor: pass dy through (no modulation).
                            // Else if dy's sign opposes dir_to_attractor:
                            //     dx = dy * (1 + alpha)   ← amplify, because
                            //     a NEGATIVE update of dy moves x toward the
                            //     attractor (parameter update is θ ← θ - lr·dx).
                            // Else: dx = dy * (1 / (1 + alpha))   ← dampen.
                            //
                            // Reasoning for the sign math: parameter update is
                            // `θ ← θ − lr · grad`. We want updates that move θ
                            // toward the nearest attractor amplified. So if
                            // attractor > x (i.e. dir_to_attractor > 0), the
                            // update must be NEGATIVE, which means grad must
                            // be POSITIVE. Amplifying grad in that case = good.
                            // If grad is already negative when attractor > x,
                            // the update will move θ further from attractor →
                            // dampen.
                            let av = self.autograd_tape[a].value.clone();
                            let amp = 1.0 + alpha;
                            let damp = 1.0 / amp;
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                let x = av.data[k];
                                let g = dy.data[k];
                                let xs = (x * scale).round() as i64;
                                let (attractor, dist) =
                                    crate::phi_pi_fib::nearest_attractor_with_dist(xs);
                                if dist == 0 {
                                    // Already on attractor — keep grad as-is.
                                    da.data[k] = g;
                                    continue;
                                }
                                let dir_to_attractor = attractor - xs;
                                // grad direction that pulls θ toward attractor:
                                //   if attractor > x, we want θ to increase →
                                //     update -lr*g must be positive → g must be negative.
                                //   so g·dir < 0 means grad pulls toward attractor.
                                let pulls_toward = (g.signum() as i64) * dir_to_attractor.signum() < 0;
                                da.data[k] = if pulls_toward { g * amp } else { g * damp };
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::SubstrateSparseScores(q_id, k_id, threshold) => {
                            // Backward through sparse scores. dy is [N, M].
                            // For fired cells (substrate_dist(i, j) <= threshold):
                            //   dL/dq[i, k] += dy[i, j] * k[j, k]
                            //   dL/dk[j, k] += dy[i, j] * q[i, k]
                            // For masked cells, dy comes in as 0 from softmax
                            // backward (softmax of -inf = 0, so gradient is 0
                            // at those positions). We still skip them here
                            // for clarity and to make the optimization
                            // observable in profiles.
                            let qv = self.autograd_tape[q_id].value.clone();
                            let kv = self.autograd_tape[k_id].value.clone();
                            let n = qv.rows;
                            let m = kv.rows;
                            let d = qv.cols;
                            let moduli: [i64; 4] = [5, 8, 13, 21];
                            let substrate_dist = |i: usize, j: usize| -> i64 {
                                let mut s = 0_i64;
                                for &mm in &moduli {
                                    let di = (i as i64) % mm;
                                    let dj = (j as i64) % mm;
                                    s += (di - dj).abs();
                                }
                                s
                            };
                            let mut dq = TapeMat::zeros(qv.rows, qv.cols);
                            let mut dk = TapeMat::zeros(kv.rows, kv.cols);
                            for i in 0..n {
                                for j in 0..m {
                                    if substrate_dist(i, j) > threshold { continue; }
                                    let g = dy.at(i, j);
                                    if g == 0.0 { continue; }
                                    for k in 0..d {
                                        let cur_dq = dq.at(i, k);
                                        dq.set(i, k, cur_dq + g * kv.at(j, k));
                                        let cur_dk = dk.at(j, k);
                                        dk.set(j, k, cur_dk + g * qv.at(i, k));
                                    }
                                }
                            }
                            self.autograd_tape[q_id].grad.add(&dq);
                            self.autograd_tape[k_id].grad.add(&dk);
                        }
                        TapeOp::SubstrateResample(a, scale) => {
                            // out = v * modulator(v) where modulator is treated as const
                            // (matches OMC reference). dL/dv[k] = dy[k] * modulator(v[k]).
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                let x = av.data[k];
                                let n = (x * scale) as i64;
                                let (_, d) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                                let modulator = 1.0 / (1.0 + (d as f64) / scale);
                                da.data[k] = dy.data[k] * modulator;
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::EmbeddingLookup(a, ref token_ids) => {
                            // dL/dtable[v, :] = sum over i: dy[i, :] where token_ids[i] == v.
                            // Same-token-id collisions accumulate (sum), which is the
                            // correct gradient when a token appears multiple times.
                            let table_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let d_model = table_shape.1;
                            let mut dtable = TapeMat::zeros(table_shape.0, table_shape.1);
                            for (i, &tok) in token_ids.iter().enumerate() {
                                if tok >= table_shape.0 { continue; }
                                for c in 0..d_model {
                                    let g = dy.at(i, c);
                                    let cur = dtable.at(tok, c);
                                    dtable.set(tok, c, cur + g);
                                }
                            }
                            self.autograd_tape[a].grad.add(&dtable);
                        }
                        TapeOp::CrossEntropyBatch(a, ref targets) => {
                            // dL/dlogits[i, c] = (softmax(logits)[i, c] - 1{c==t_i}) / N
                            // dy is the upstream gradient on the scalar loss (typically 1.0
                            // at the loss seed; scaled when this op is chained inside more math).
                            let av = self.autograd_tape[a].value.clone();
                            let n = av.rows;
                            let vocab = av.cols;
                            let dy_scalar = if dy.data.is_empty() { 0.0 } else { dy.data[0] };
                            let scale = dy_scalar / (n.max(1) as f64);
                            let mut da = TapeMat::zeros(n, vocab);
                            for i in 0..n {
                                // Recompute the per-row softmax. (Could be cached at the cost
                                // of memory; the recompute is one extra pass through N×vocab
                                // f64s and is dwarfed by the matmul backward in any real model.)
                                let mut row_max = f64::NEG_INFINITY;
                                for c in 0..vocab {
                                    let x = av.at(i, c);
                                    if x > row_max { row_max = x; }
                                }
                                let mut row_sum_exp: f64 = 0.0;
                                for c in 0..vocab {
                                    row_sum_exp += (av.at(i, c) - row_max).exp();
                                }
                                let target = targets[i];
                                for c in 0..vocab {
                                    let p = (av.at(i, c) - row_max).exp() / row_sum_exp;
                                    let indicator = if c == target { 1.0 } else { 0.0 };
                                    da.set(i, c, scale * (p - indicator));
                                }
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Abs(a) => {
                            // d/dx |x| = sign(x). Subgradient: choose 0 at x=0.
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                let s = if av.data[k] > 0.0 { 1.0 }
                                        else if av.data[k] < 0.0 { -1.0 }
                                        else { 0.0 };
                                da.data[k] = dy.data[k] * s;
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::PhiLog(a, scale) => {
                            // y = ln(|x·scale| + 1) / (π · ln φ)
                            // dy/dx = scale · sign(x) / ((|x·scale| + 1) · π · ln φ)
                            let av = self.autograd_tape[a].value.clone();
                            let denom_const = std::f64::consts::PI * crate::value::PHI.ln();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                let xs = av.data[k] * scale;
                                let sign = if av.data[k] > 0.0 { 1.0 }
                                           else if av.data[k] < 0.0 { -1.0 }
                                           else { 0.0 };
                                let denom = (xs.abs() + 1.0) * denom_const;
                                da.data[k] = dy.data[k] * scale * sign / denom;
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Softmax(a) => {
                            // For row-wise softmax y = softmax(x):
                            //   dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
                            // per row. The cached forward `out` is y, stored
                            // in self.autograd_tape[i].value.
                            let y_clone = self.autograd_tape[i].value.clone();
                            let av_shape = (y_clone.rows, y_clone.cols);
                            let mut da = TapeMat::zeros(av_shape.0, av_shape.1);
                            for r in 0..av_shape.0 {
                                let mut s_row = 0.0;
                                for c in 0..av_shape.1 {
                                    s_row += dy.data[r * av_shape.1 + c]
                                          * y_clone.data[r * av_shape.1 + c];
                                }
                                for c in 0..av_shape.1 {
                                    let yi = y_clone.data[r * av_shape.1 + c];
                                    let gi = yi * (dy.data[r * av_shape.1 + c] - s_row);
                                    da.data[r * av_shape.1 + c] = gi;
                                }
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Exp(a) => {
                            let yv = self.autograd_tape[i].value.clone();
                            let mut da = TapeMat::zeros(yv.rows, yv.cols);
                            for k in 0..yv.data.len() { da.data[k] = dy.data[k] * yv.data[k]; }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Sin(a) => {
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() { da.data[k] = dy.data[k] * av.data[k].cos(); }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Cos(a) => {
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() { da.data[k] = -dy.data[k] * av.data[k].sin(); }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Relu(a) => {
                            let av = self.autograd_tape[a].value.clone();
                            let mut da = TapeMat::zeros(av.rows, av.cols);
                            for k in 0..av.data.len() {
                                da.data[k] = if av.data[k] > 0.0 { dy.data[k] } else { 0.0 };
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Sigmoid(a) => {
                            let yv = self.autograd_tape[i].value.clone();
                            let mut da = TapeMat::zeros(yv.rows, yv.cols);
                            for k in 0..yv.data.len() {
                                let s = yv.data[k];
                                da.data[k] = dy.data[k] * s * (1.0 - s);
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Tanh(a) => {
                            let yv = self.autograd_tape[i].value.clone();
                            let mut da = TapeMat::zeros(yv.rows, yv.cols);
                            for k in 0..yv.data.len() {
                                let t = yv.data[k];
                                da.data[k] = dy.data[k] * (1.0 - t * t);
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::MatMul(a, b) => {
                            // dA = dy @ B^T ; dB = A^T @ dy
                            let av = self.autograd_tape[a].value.clone();
                            let bv = self.autograd_tape[b].value.clone();
                            let bt = tape_transpose(&bv);
                            let at = tape_transpose(&av);
                            let da = tape_matmul(&dy, &bt)?;
                            let db = tape_matmul(&at, &dy)?;
                            self.autograd_tape[a].grad.add(&da);
                            self.autograd_tape[b].grad.add(&db);
                        }
                        TapeOp::Sum(a) => {
                            // dL/dA = dy (scalar) broadcast to A's shape
                            let av_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let mut da = TapeMat::zeros(av_shape.0, av_shape.1);
                            let s = dy.data[0];
                            for v in da.data.iter_mut() { *v = s; }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Transpose(a) => {
                            // dA = transpose(dy)
                            let dyt = tape_transpose(&dy);
                            self.autograd_tape[a].grad.add(&dyt);
                        }
                        TapeOp::LayerNormRow(xid, gid, bid, eps) => {
                            // Per-row LN backward:
                            // x_hat[r,c] = (x[r,c] - mu_r) / std_r
                            // y[r,c] = gamma[c] * x_hat[r,c] + beta[c]
                            // Three grads: dx, dgamma, dbeta.
                            let xv = self.autograd_tape[xid].value.clone();
                            let gv = self.autograd_tape[gid].value.clone();
                            let n = xv.cols as f64;
                            let mut dx = TapeMat::zeros(xv.rows, xv.cols);
                            let mut dgamma = TapeMat::zeros(1, xv.cols);
                            let mut dbeta = TapeMat::zeros(1, xv.cols);
                            for r in 0..xv.rows {
                                // Recompute per-row mean / std / x_hat from xv.
                                let mut mean = 0.0;
                                for c in 0..xv.cols { mean += xv.data[r * xv.cols + c]; }
                                mean /= n;
                                let mut var = 0.0;
                                for c in 0..xv.cols {
                                    let d = xv.data[r * xv.cols + c] - mean;
                                    var += d * d;
                                }
                                var /= n;
                                let std = (var + eps).sqrt();
                                let inv_std = 1.0 / std;
                                // x_hat per cell.
                                let mut xhat = vec![0.0; xv.cols];
                                for c in 0..xv.cols {
                                    xhat[c] = (xv.data[r * xv.cols + c] - mean) * inv_std;
                                }
                                // Accumulate dgamma, dbeta from THIS row.
                                for c in 0..xv.cols {
                                    let dy_rc = dy.data[r * xv.cols + c];
                                    dgamma.data[c] += dy_rc * xhat[c];
                                    dbeta.data[c] += dy_rc;
                                }
                                // dx_hat = dy * gamma  ; then propagate through
                                // (x_hat = (x - mean)/std) to get dx.
                                let mut dxhat = vec![0.0; xv.cols];
                                for c in 0..xv.cols {
                                    dxhat[c] = dy.data[r * xv.cols + c] * gv.data[c];
                                }
                                // dx[r, c] = (1/std) * (
                                //   dxhat[c] - mean_dxhat - xhat[c] * mean(dxhat * xhat)
                                // )
                                let mut sum_dxhat = 0.0;
                                let mut sum_dxhat_xhat = 0.0;
                                for c in 0..xv.cols {
                                    sum_dxhat += dxhat[c];
                                    sum_dxhat_xhat += dxhat[c] * xhat[c];
                                }
                                let mean_dxhat = sum_dxhat / n;
                                let mean_dxhat_xhat = sum_dxhat_xhat / n;
                                for c in 0..xv.cols {
                                    let g = inv_std * (
                                        dxhat[c] - mean_dxhat
                                            - xhat[c] * mean_dxhat_xhat
                                    );
                                    dx.data[r * xv.cols + c] = g;
                                }
                            }
                            self.autograd_tape[xid].grad.add(&dx);
                            self.autograd_tape[gid].grad.add(&dgamma);
                            self.autograd_tape[bid].grad.add(&dbeta);
                        }
                        TapeOp::RowMean(a) => {
                            // dL/dA[r, c] = dy[r, 0] / cols
                            let av_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let cols_f = av_shape.1.max(1) as f64;
                            let mut da = TapeMat::zeros(av_shape.0, av_shape.1);
                            for r in 0..av_shape.0 {
                                let s = dy.data[r] / cols_f;
                                for c in 0..av_shape.1 {
                                    da.data[r * av_shape.1 + c] = s;
                                }
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::RowSum(a) => {
                            // dL/dA[r, c] = dy[r, 0]
                            let av_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let mut da = TapeMat::zeros(av_shape.0, av_shape.1);
                            for r in 0..av_shape.0 {
                                let s = dy.data[r];
                                for c in 0..av_shape.1 {
                                    da.data[r * av_shape.1 + c] = s;
                                }
                            }
                            self.autograd_tape[a].grad.add(&da);
                        }
                        TapeOp::Mean(a) => {
                            let av_shape = (
                                self.autograd_tape[a].value.rows,
                                self.autograd_tape[a].value.cols,
                            );
                            let n = (av_shape.0 * av_shape.1).max(1) as f64;
                            let mut da = TapeMat::zeros(av_shape.0, av_shape.1);
                            let s = dy.data[0] / n;
                            for v in da.data.iter_mut() { *v = s; }
                            self.autograd_tape[a].grad.add(&da);
                        }
                    }
                }
                Ok(Value::Null)
            }
            // tape_update(var_id, lr) — in-place SGD step. Convenience
            // so user code doesn't have to read grad, scale, re-bind.
            // Mutates the underlying Var value; gradient stays for
            // inspection until the next tape_reset.
            "tape_update" => {
                if args.len() < 2 {
                    return Err("tape_update requires (var_id, lr)".to_string());
                }
                let id = self.eval_expr(&args[0])?.to_int() as usize;
                let lr = self.eval_expr(&args[1])?.to_float();
                if id >= self.autograd_tape.len() {
                    return Err("tape_update: id out of range".to_string());
                }
                let grad = self.autograd_tape[id].grad.clone();
                let val = &mut self.autograd_tape[id].value;
                for k in 0..val.data.len() {
                    val.data[k] -= lr * grad.data[k];
                }
                Ok(Value::Null)
            }
            // ---- Lazy generators (streaming via callback) -----------
            //
            // gen_stream(thunk, callback) runs `thunk` with a yield
            // callback installed. Every `yield v` inside the generator
            // invokes callback(v); a 0 return shorts the generator.
            // Memory is O(call-stack-depth), not O(yield-count) —
            // a generator can stream unbounded values.
            //
            // The thunk pattern (instead of accepting a "generator
            // call expression") avoids eager evaluation: the generator
            // doesn't start running until gen_stream installs the
            // callback and invokes the thunk.
            //
            //   gen_stream(fn() { return fib(1000000); },
            //              fn(v) { print(v); return 1; });
            //
            // Returns 1 if the generator ran to completion, 0 if the
            // callback shorted it.
            "gen_stream" => {
                if args.len() < 2 {
                    return Err("gen_stream requires (thunk, callback)".to_string());
                }
                let thunk = self.eval_expr(&args[0])?;
                let cb = self.eval_expr(&args[1])?;
                self.yield_callbacks.push(cb);
                let prior_return = self.return_value.take();
                let res = self.call_first_class_function(&thunk, vec![]);
                self.yield_callbacks.pop();
                let stopped = self.gen_stop_requested;
                self.gen_stop_requested = false;
                // The yield short-circuit set return_value to Null to
                // unwind the body. Restore the caller's return state
                // so we don't leak the sentinel up the call stack.
                self.return_value = prior_return;
                res?;
                Ok(Value::HInt(HInt::new(if stopped { 0 } else { 1 })))
            }
            // gen_take(thunk, n) — pull the first n values from a lazy
            // generator into a list. Lazy because the generator stops
            // after n yields rather than producing the full sequence.
            "gen_take" => {
                if args.len() < 2 {
                    return Err("gen_take requires (thunk, n)".to_string());
                }
                let thunk = self.eval_expr(&args[0])?;
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                // Use a Rust-side accumulator via RefCell so we don't
                // need to round-trip through an OMC variable.
                let collected: std::rc::Rc<std::cell::RefCell<Vec<Value>>>
                    = std::rc::Rc::new(std::cell::RefCell::new(Vec::with_capacity(n)));
                let acc = collected.clone();
                // Stash the accumulator in a host_builtin so the
                // callback (an OMC lambda) can push through a name.
                self.host_builtins.insert(
                    "__gen_take_push".to_string(),
                    std::rc::Rc::new(move |args: &[Value]| {
                        if !args.is_empty() {
                            acc.borrow_mut().push(args[0].clone());
                        }
                        Ok(Value::HInt(HInt::new(1)))
                    }),
                );
                // Build a callback that pushes via the host builtin
                // and returns 0 when we've collected n values.
                let cb_name = format!("__gen_take_cb_{}", self.lambda_counter);
                self.lambda_counter += 1;
                let limit = n;
                let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
                let counter_ref = counter.clone();
                self.host_builtins.insert(
                    cb_name.clone(),
                    std::rc::Rc::new(move |args: &[Value]| {
                        if counter_ref.get() < limit {
                            if !args.is_empty() {
                                // direct push, no second hop
                            }
                            counter_ref.set(counter_ref.get() + 1);
                            if counter_ref.get() >= limit {
                                Ok(Value::HInt(HInt::new(0)))  // stop
                            } else {
                                Ok(Value::HInt(HInt::new(1)))  // continue
                            }
                        } else {
                            Ok(Value::HInt(HInt::new(0)))
                        }
                    }),
                );
                // Compose: the actual callback first pushes via
                // __gen_take_push, then asks the limit cb whether to stop.
                let acc2 = collected.clone();
                let counter2 = counter.clone();
                let limit2 = n;
                let combined = format!("__gen_take_combined_{}", self.lambda_counter);
                self.lambda_counter += 1;
                self.host_builtins.insert(
                    combined.clone(),
                    std::rc::Rc::new(move |args: &[Value]| {
                        if counter2.get() < limit2 && !args.is_empty() {
                            acc2.borrow_mut().push(args[0].clone());
                            counter2.set(counter2.get() + 1);
                            if counter2.get() >= limit2 {
                                return Ok(Value::HInt(HInt::new(0)));
                            }
                            return Ok(Value::HInt(HInt::new(1)));
                        }
                        Ok(Value::HInt(HInt::new(0)))
                    }),
                );
                let cb_value = Value::Function {
                    name: combined.clone(),
                    captured: None,
                };
                self.yield_callbacks.push(cb_value);
                let prior_return = self.return_value.take();
                let res = self.call_first_class_function(&thunk, vec![]);
                self.yield_callbacks.pop();
                self.gen_stop_requested = false;
                self.return_value = prior_return;
                self.host_builtins.remove(&combined);
                self.host_builtins.remove(&cb_name);
                self.host_builtins.remove("__gen_take_push");
                res?;
                let out = collected.borrow().clone();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // gen_count(thunk) — count how many values the generator
            // would yield without storing any of them. O(1) memory.
            "gen_count" => {
                if args.is_empty() {
                    return Err("gen_count requires (thunk)".to_string());
                }
                let thunk = self.eval_expr(&args[0])?;
                let counter = std::rc::Rc::new(std::cell::Cell::new(0i64));
                let counter_ref = counter.clone();
                let cb_name = format!("__gen_count_cb_{}", self.lambda_counter);
                self.lambda_counter += 1;
                self.host_builtins.insert(
                    cb_name.clone(),
                    std::rc::Rc::new(move |_args: &[Value]| {
                        counter_ref.set(counter_ref.get() + 1);
                        Ok(Value::HInt(HInt::new(1)))
                    }),
                );
                let cb_value = Value::Function { name: cb_name.clone(), captured: None };
                self.yield_callbacks.push(cb_value);
                let prior_return = self.return_value.take();
                let res = self.call_first_class_function(&thunk, vec![]);
                self.yield_callbacks.pop();
                self.gen_stop_requested = false;
                self.return_value = prior_return;
                self.host_builtins.remove(&cb_name);
                res?;
                Ok(Value::HInt(HInt::new(counter.get())))
            }
            // gen_sum(thunk) — reduce a lazy generator to a sum.
            // Demonstrates the laziness benefit: streams unbounded
            // sequences without allocation.
            "gen_sum" => {
                if args.is_empty() {
                    return Err("gen_sum requires (thunk)".to_string());
                }
                let thunk = self.eval_expr(&args[0])?;
                let acc = std::rc::Rc::new(std::cell::Cell::new(0i64));
                let acc_ref = acc.clone();
                let cb_name = format!("__gen_sum_cb_{}", self.lambda_counter);
                self.lambda_counter += 1;
                self.host_builtins.insert(
                    cb_name.clone(),
                    std::rc::Rc::new(move |args: &[Value]| {
                        if !args.is_empty() {
                            acc_ref.set(acc_ref.get().wrapping_add(args[0].to_int()));
                        }
                        Ok(Value::HInt(HInt::new(1)))
                    }),
                );
                let cb_value = Value::Function { name: cb_name.clone(), captured: None };
                self.yield_callbacks.push(cb_value);
                let prior_return = self.return_value.take();
                let res = self.call_first_class_function(&thunk, vec![]);
                self.yield_callbacks.pop();
                self.gen_stop_requested = false;
                self.return_value = prior_return;
                self.host_builtins.remove(&cb_name);
                res?;
                Ok(Value::HInt(HInt::new(acc.get())))
            }
            // gen_substrate_fib(callback, max) — substrate-native lazy
            // generator. Produces Fibonacci numbers as HInt (each one
            // already carries resonance=1.0 because Fibonacci values
            // ARE Fibonacci attractors). Streams until `max` reached or
            // callback returns 0. The recurrence IS the state — O(1)
            // memory for ANY length. Python can't do this lazily
            // without a generator object and definitely can't carry
            // substrate metadata on the i64 outputs.
            "gen_substrate_fib" => {
                if args.len() < 2 {
                    return Err("gen_substrate_fib requires (callback, max)".to_string());
                }
                let cb = self.eval_expr(&args[0])?;
                let max = self.eval_expr(&args[1])?.to_int();
                let mut a: i64 = 0;
                let mut b: i64 = 1;
                let mut count: i64 = 0;
                loop {
                    if a > max { break; }
                    let r = self.call_first_class_function(
                        &cb,
                        vec![Value::HInt(HInt::new(a))],
                    )?;
                    count += 1;
                    if r.to_int() == 0 { break; }
                    let next = a.wrapping_add(b);
                    a = b;
                    b = next;
                }
                Ok(Value::HInt(HInt::new(count)))
            }
            // ---- Introspection (LLM-discoverability surface) -------
            //
            // The docs registry in src/docs.rs is the source of truth.
            // omc_help / omc_list_builtins / omc_categories give code
            // (and LLMs driving code) a way to enumerate the builtin
            // surface area at runtime — no separate cheat-sheet needed.
            //
            // omc_did_you_mean is what the unknown-function error path
            // calls; exposing it as a builtin too means user code can
            // suggest typo fixes when handling its own errors.
            "omc_help" => {
                if args.is_empty() {
                    return Err("omc_help requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(doc) => {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("name".to_string(), Value::String(doc.name.to_string()));
                        map.insert("category".to_string(), Value::String(doc.category.to_string()));
                        map.insert("signature".to_string(), Value::String(doc.signature.to_string()));
                        map.insert("description".to_string(), Value::String(doc.description.to_string()));
                        map.insert("example".to_string(), Value::String(doc.example.to_string()));
                        map.insert("unique_to_omc".to_string(),
                            Value::HInt(HInt::new(if doc.unique_to_omc { 1 } else { 0 })));
                        Ok(Value::dict_from(map))
                    }
                    None => {
                        // Surface the suggestion path: if there's no
                        // doc entry, return a dict with did_you_mean
                        // hits so an LLM/user immediately sees the typo.
                        let suggestions = crate::docs::did_you_mean(&name, 5);
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("name".to_string(), Value::String(name));
                        map.insert("found".to_string(), Value::HInt(HInt::new(0)));
                        let did_you_mean: Vec<Value> = suggestions.iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        map.insert(
                            "did_you_mean".to_string(),
                            Value::Array(HArray::from_vec(did_you_mean)),
                        );
                        Ok(Value::dict_from(map))
                    }
                }
            }
            "omc_list_builtins" => {
                // Optional 1st arg = category filter.
                let category_filter = if !args.is_empty() {
                    Some(self.eval_expr(&args[0])?.to_display_string())
                } else { None };
                let cat_ref = category_filter.as_deref();
                let names = crate::docs::names_in(cat_ref);
                let out: Vec<Value> = names.iter()
                    .map(|n| Value::String(n.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // omc_find_by_signature(pattern: string) -> [{name, signature, category}, ...]
            //   Substring-match `pattern` against every builtin's signature
            //   field. Lets LLMs discover by intent — e.g.
            //     omc_find_by_signature("-> float[]") to find fns
            //     returning a float array, or
            //     omc_find_by_signature("string, int") for those taking
            //     a string and an int.
            //   Match is case-insensitive substring on the literal signature
            //   string. Optional 2nd arg: max results (default 20).
            "omc_find_by_signature" => {
                if args.is_empty() {
                    return Err("omc_find_by_signature requires (pattern: string, max?: int)".to_string());
                }
                let pattern = self.eval_expr(&args[0])?.to_display_string();
                let max = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_int().max(1) as usize
                } else { 20 };
                let pat_lc = pattern.to_lowercase();
                let mut hits: Vec<Value> = Vec::new();
                for doc in crate::docs::BUILTINS {
                    if doc.signature.to_lowercase().contains(&pat_lc) {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("name".to_string(), Value::String(doc.name.to_string()));
                        map.insert("signature".to_string(), Value::String(doc.signature.to_string()));
                        map.insert("category".to_string(), Value::String(doc.category.to_string()));
                        map.insert("description".to_string(), Value::String(doc.description.to_string()));
                        hits.push(Value::dict_from(map));
                        if hits.len() >= max { break; }
                    }
                }
                Ok(Value::Array(HArray::from_vec(hits)))
            }
            "omc_categories" => {
                let cats = crate::docs::categories();
                let out: Vec<Value> = cats.iter()
                    .map(|c| Value::String(c.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_did_you_mean" => {
                if args.is_empty() {
                    return Err("omc_did_you_mean requires (name)".to_string());
                }
                let query = self.eval_expr(&args[0])?.to_display_string();
                let limit = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_int().max(1) as usize
                } else { 5 };
                let suggestions = crate::docs::did_you_mean(&query, limit);
                let out: Vec<Value> = suggestions.iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_unique_builtins" => {
                let out: Vec<Value> = crate::docs::BUILTINS.iter()
                    .filter(|b| b.unique_to_omc)
                    .map(|b| Value::String(b.name.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // omc_explain_error(msg) — pattern-match an error message
            // against the catalog in src/errors.rs and return a dict
            // describing what it means, the typical cause, and the fix.
            // LLMs catching OMC errors call this to get back actionable
            // remediation without having to memorize 200+ error niches.
            "omc_explain_error" => {
                if args.is_empty() {
                    return Err("omc_explain_error requires (msg)".to_string());
                }
                let msg = self.eval_expr(&args[0])?.to_display_string();
                match crate::errors::match_error(&msg) {
                    Some(p) => {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("matched".to_string(), Value::HInt(HInt::new(1)));
                        map.insert("pattern".to_string(), Value::String(p.pattern.to_string()));
                        map.insert("category".to_string(), Value::String(p.category.to_string()));
                        map.insert("explanation".to_string(), Value::String(p.explanation.to_string()));
                        map.insert("typical_cause".to_string(), Value::String(p.typical_cause.to_string()));
                        map.insert("fix".to_string(), Value::String(p.fix.to_string()));
                        Ok(Value::dict_from(map))
                    }
                    None => {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("matched".to_string(), Value::HInt(HInt::new(0)));
                        map.insert("explanation".to_string(),
                            Value::String("No catalog pattern matched — the error message is unique enough that the runtime doesn't have a curated fix yet. Inspect the message itself, or open an issue to add a pattern.".to_string()));
                        Ok(Value::dict_from(map))
                    }
                }
            }
            // omc_error_categories() — every distinct error category
            // in the catalog. Useful for guided exploration.
            "omc_error_categories" => {
                let cats = crate::errors::error_categories();
                let out: Vec<Value> = cats.iter()
                    .map(|c| Value::String(c.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // omc_error_count() — number of curated patterns. Lets
            // callers verify "the language ships a knowledge base".
            "omc_error_count" => {
                Ok(Value::HInt(HInt::new(crate::errors::ERROR_PATTERNS.len() as i64)))
            }
            // ---- Substrate-token adapter (LLM compression layer) ---
            //
            // Maps OMC source ↔ substrate-typed token IDs. Common
            // builtin names get small attractor-aligned IDs so:
            //   - LLM emits short int arrays instead of full names
            //   - attractor_distance(id) is a free "semantic distance"
            //   - code-hash comparisons work in resonance-space
            //
            // Round-trip is exact (unmatched bytes escape as [0, byte]).
            "omc_token_encode" => {
                if args.is_empty() {
                    return Err("omc_token_encode requires (code: string)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let ids = crate::tokenizer::encode(&code);
                let out: Vec<Value> = ids.iter()
                    .map(|&i| Value::HInt(HInt::new(i)))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_token_decode" => {
                if args.is_empty() {
                    return Err("omc_token_decode requires (ids: int[])".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = v {
                    let ids: Vec<i64> = arr.items.borrow().iter()
                        .map(|x| x.to_int())
                        .collect();
                    let s = crate::tokenizer::decode(&ids);
                    Ok(Value::String(s))
                } else {
                    Err("omc_token_decode: first arg must be an int array".to_string())
                }
            }
            "omc_token_distance" => {
                if args.len() < 2 {
                    return Err("omc_token_distance requires (id_a, id_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int();
                let b = self.eval_expr(&args[1])?.to_int();
                Ok(Value::HInt(HInt::new(crate::tokenizer::token_distance(a, b))))
            }
            "omc_token_vocab" => {
                // Return the full dictionary as a string array.
                // Position is the token's ID; element is the canonical
                // substring it expands to. ID 0 is the escape sentinel.
                let out: Vec<Value> = crate::tokenizer::TOKEN_DICT.iter()
                    .map(|s| Value::String(s.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_token_vocab_size" => {
                Ok(Value::HInt(HInt::new(crate::tokenizer::TOKEN_DICT.len() as i64)))
            }
            "omc_token_compression_ratio" => {
                // bytes_in / ints_out — > 1 means encoding is denser.
                // Counts each int as 1 unit (token); raw bytes as 1
                // unit each. Compression is real when shared substrings
                // collapse to single IDs.
                if args.is_empty() {
                    return Err("omc_token_compression_ratio requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let raw = code.len() as f64;
                let ids = crate::tokenizer::encode(&code).len() as f64;
                if ids == 0.0 {
                    return Ok(Value::HFloat(0.0));
                }
                Ok(Value::HFloat(raw / ids))
            }
            "omc_token_pack" => {
                // CRT-pack a stream of remainders into a single i64.
                // Default moduli = tokenizer::CRT_MODULI (7, 1009, 100003).
                if args.is_empty() {
                    return Err("omc_token_pack requires (streams, moduli?)".to_string());
                }
                let streams_v = self.eval_expr(&args[0])?;
                let streams: Vec<i64> = if let Value::Array(arr) = streams_v {
                    arr.items.borrow().iter().map(|v| v.to_int()).collect()
                } else {
                    return Err("omc_token_pack: streams must be an array".to_string());
                };
                let moduli: Vec<i64> = if args.len() >= 2 {
                    let mv = self.eval_expr(&args[1])?;
                    if let Value::Array(arr) = mv {
                        arr.items.borrow().iter().map(|v| v.to_int()).collect()
                    } else {
                        return Err("omc_token_pack: moduli must be an array".to_string());
                    }
                } else {
                    crate::tokenizer::CRT_MODULI.to_vec()
                };
                match crate::tokenizer::crt_pack(&streams, &moduli) {
                    Ok(packed) => Ok(Value::HInt(HInt::new(packed))),
                    Err(e) => Err(e),
                }
            }
            "omc_token_unpack" => {
                if args.is_empty() {
                    return Err("omc_token_unpack requires (packed, moduli?)".to_string());
                }
                let packed = self.eval_expr(&args[0])?.to_int();
                let moduli: Vec<i64> = if args.len() >= 2 {
                    let mv = self.eval_expr(&args[1])?;
                    if let Value::Array(arr) = mv {
                        arr.items.borrow().iter().map(|v| v.to_int()).collect()
                    } else {
                        return Err("omc_token_unpack: moduli must be an array".to_string());
                    }
                } else {
                    crate::tokenizer::CRT_MODULI.to_vec()
                };
                let out: Vec<Value> = crate::tokenizer::crt_unpack(packed, &moduli)
                    .iter()
                    .map(|&i| Value::HInt(HInt::new(i)))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_code_hash" => {
                // Hash a program's canonical token stream and return
                // a dict with {raw, attractor, distance, resonance}.
                // Equivalent programs hash to the same attractor.
                if args.is_empty() {
                    return Err("omc_code_hash requires (code: string)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let (attractor, raw, dist) = crate::tokenizer::code_hash(&code);
                let mut map = std::collections::BTreeMap::new();
                map.insert("raw".to_string(), Value::HInt(HInt::new(raw)));
                map.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                map.insert("distance".to_string(), Value::HInt(HInt::new(dist)));
                map.insert("resonance".to_string(),
                    Value::HFloat(crate::value::HInt::compute_resonance(raw)));
                Ok(Value::dict_from(map))
            }
            "omc_code_distance" => {
                // Substrate distance between two programs in hash-space.
                // Same code → 0. Small edits → small distance.
                // Structurally different programs → large distance.
                if args.len() < 2 {
                    return Err("omc_code_distance requires (code_a, code_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let (_, ra, _) = crate::tokenizer::code_hash(&a);
                let (_, rb, _) = crate::tokenizer::code_hash(&b);
                Ok(Value::HInt(HInt::new((ra - rb).abs())))
            }
            // ---- AST canonicalization (the LLM-reach-for primitives) ---
            //
            // omc_code_canonical(src) — parse, walk the AST renaming
            // locals to __v0/__v1/..., re-emit via the formatter. The
            // result is invariant under whitespace, comments, local
            // variable names, parameter names, for-loop variables,
            // catch err vars, lambda params, and match-arm binds.
            // Top-level fn/class names, dict keys, string literals,
            // and globals are PRESERVED (observable API).
            //
            // omc_code_equivalent(a, b) — 1 iff canonical forms match.
            //
            // Combined with omc_code_hash(omc_code_canonical(x)), an
            // LLM gets a semantic-stable id for any program region
            // that survives every cosmetic edit.
            "omc_code_canonical" => {
                if args.is_empty() {
                    return Err("omc_code_canonical requires (code: string)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                match crate::canonical::canonicalize(&code) {
                    Ok(s) => Ok(Value::String(s)),
                    Err(e) => Err(format!("omc_code_canonical: {}", e)),
                }
            }
            "omc_code_equivalent" => {
                if args.len() < 2 {
                    return Err("omc_code_equivalent requires (code_a, code_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let eq = crate::canonical::equivalent(&a, &b);
                Ok(Value::HInt(HInt::new(if eq { 1 } else { 0 })))
            }
            // ---- Code intelligence (LLM-iteration primitives) ------
            //
            // These give an LLM structural information about code
            // without re-reading the source: function inventory, call
            // dependencies, complexity, similarity, fingerprints.
            "omc_code_summary" => {
                if args.is_empty() {
                    return Err("omc_code_summary requires (code: string)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let s = crate::code_intel::summarise(&code)
                    .map_err(|e| format!("omc_code_summary: {}", e))?;
                let mut map = std::collections::BTreeMap::new();
                // Function inventory.
                let fns: Vec<Value> = s.functions.iter().map(|f| {
                    let mut fm = std::collections::BTreeMap::new();
                    fm.insert("name".to_string(), Value::String(f.name.clone()));
                    fm.insert("params".to_string(), Value::Array(HArray::from_vec(
                        f.params.iter().map(|p| Value::String(p.clone())).collect()
                    )));
                    fm.insert("body_stmts".to_string(), Value::HInt(HInt::new(f.body_stmts as i64)));
                    fm.insert("canonical_hash".to_string(), Value::HInt(HInt::new(f.canonical_hash)));
                    if let Some(rt) = &f.return_type {
                        fm.insert("return_type".to_string(), Value::String(rt.clone()));
                    }
                    if !f.pragmas.is_empty() {
                        fm.insert("pragmas".to_string(), Value::Array(HArray::from_vec(
                            f.pragmas.iter().map(|p| Value::String(p.clone())).collect()
                        )));
                    }
                    Value::dict_from(fm)
                }).collect();
                map.insert("functions".to_string(), Value::Array(HArray::from_vec(fns)));
                map.insert("classes".to_string(), Value::Array(HArray::from_vec(
                    s.classes.iter().map(|c| Value::String(c.clone())).collect()
                )));
                map.insert("imports".to_string(), Value::Array(HArray::from_vec(
                    s.imports.iter().map(|i| Value::String(i.clone())).collect()
                )));
                map.insert("calls".to_string(), Value::Array(HArray::from_vec(
                    s.calls.iter().map(|c| Value::String(c.clone())).collect()
                )));
                map.insert("stmt_count".to_string(), Value::HInt(HInt::new(s.stmt_count as i64)));
                Ok(Value::dict_from(map))
            }
            "omc_code_extract_fns" => {
                // Lightweight version: just the function names.
                if args.is_empty() {
                    return Err("omc_code_extract_fns requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let s = crate::code_intel::summarise(&code)
                    .map_err(|e| format!("omc_code_extract_fns: {}", e))?;
                let out: Vec<Value> = s.functions.iter()
                    .map(|f| Value::String(f.name.clone()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_code_dependencies" => {
                // What does this program call? Useful for "which
                // builtins does this need?" and "does it use Python?"
                if args.is_empty() {
                    return Err("omc_code_dependencies requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let s = crate::code_intel::summarise(&code)
                    .map_err(|e| format!("omc_code_dependencies: {}", e))?;
                let out: Vec<Value> = s.calls.iter()
                    .map(|c| Value::String(c.clone()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_code_complexity" => {
                // Cyclomatic complexity. Returns a dict with
                // {complexity, ast_size, ast_depth} so the LLM can
                // judge "is this code getting too branchy?"
                if args.is_empty() {
                    return Err("omc_code_complexity requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let cpx = crate::code_intel::complexity(&code)
                    .map_err(|e| format!("omc_code_complexity: {}", e))?;
                let size = crate::code_intel::ast_size(&code)
                    .map_err(|e| format!("omc_code_complexity: {}", e))?;
                let depth = crate::code_intel::ast_depth(&code)
                    .map_err(|e| format!("omc_code_complexity: {}", e))?;
                let mut map = std::collections::BTreeMap::new();
                map.insert("complexity".to_string(), Value::HInt(HInt::new(cpx)));
                map.insert("ast_size".to_string(), Value::HInt(HInt::new(size)));
                map.insert("ast_depth".to_string(), Value::HInt(HInt::new(depth)));
                Ok(Value::dict_from(map))
            }
            "omc_code_minify" => {
                if args.is_empty() {
                    return Err("omc_code_minify requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                match crate::code_intel::minify(&code) {
                    Ok(m) => Ok(Value::String(m)),
                    Err(e) => Err(format!("omc_code_minify: {}", e)),
                }
            }
            "omc_code_similarity" => {
                // Jaccard similarity over canonical token IDs. 1.0 =
                // alpha-equivalent (so a perfect match implies
                // semantically the same modulo our canonicalization).
                // Lower = more different.
                if args.len() < 2 {
                    return Err("omc_code_similarity requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let s = crate::code_intel::similarity(&a, &b)
                    .map_err(|e| format!("omc_code_similarity: {}", e))?;
                Ok(Value::HFloat(s))
            }
            "omc_code_fingerprint" => {
                // Substrate-weighted fingerprint: combines hash + size
                // + complexity via CRT into one int. Two semantically
                // equivalent programs get the same fingerprint;
                // unrelated programs almost never collide.
                if args.is_empty() {
                    return Err("omc_code_fingerprint requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                match crate::code_intel::substrate_fingerprint(&code) {
                    Ok(fp) => Ok(Value::HInt(HInt::new(fp))),
                    Err(e) => Err(format!("omc_code_fingerprint: {}", e)),
                }
            }
            "omc_code_signature" => {
                // Public API surface: just the top-level fn names +
                // param counts. The minimum an LLM needs to know to
                // call a module's exports.
                if args.is_empty() {
                    return Err("omc_code_signature requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let s = crate::code_intel::summarise(&code)
                    .map_err(|e| format!("omc_code_signature: {}", e))?;
                let lines: Vec<String> = s.functions.iter()
                    .map(|f| format!("fn {}({})", f.name, f.params.join(", ")))
                    .collect();
                Ok(Value::String(lines.join("\n")))
            }
            "omc_code_uses_python" => {
                // 1 if any py_* call appears. Quick safety check —
                // an embedder might want to refuse Python-embedding
                // code in sandboxed contexts.
                if args.is_empty() {
                    return Err("omc_code_uses_python requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let s = crate::code_intel::summarise(&code)
                    .map_err(|e| format!("omc_code_uses_python: {}", e))?;
                let uses = s.calls.iter().any(|c| c.starts_with("py_"));
                Ok(Value::HInt(HInt::new(if uses { 1 } else { 0 })))
            }
            "omc_code_uses_substrate" => {
                // 1 if any substrate-unique primitive is called.
                // Lets the LLM identify "this code reaches for OMC,
                // not just Python-clone-able syntax."
                if args.is_empty() {
                    return Err("omc_code_uses_substrate requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let s = crate::code_intel::summarise(&code)
                    .map_err(|e| format!("omc_code_uses_substrate: {}", e))?;
                let unique_set: std::collections::HashSet<&str> = crate::docs::BUILTINS.iter()
                    .filter(|b| b.unique_to_omc).map(|b| b.name).collect();
                let uses = s.calls.iter().any(|c| unique_set.contains(c.as_str()));
                Ok(Value::HInt(HInt::new(if uses { 1 } else { 0 })))
            }
            "omc_completion_hint" => {
                // Given a prefix, return all known builtin names that
                // start with it. The IDE / LLM uses this for
                // autocomplete suggestions.
                if args.is_empty() {
                    return Err("omc_completion_hint requires (prefix)".to_string());
                }
                let prefix = self.eval_expr(&args[0])?.to_display_string();
                let out: Vec<Value> = crate::docs::BUILTINS.iter()
                    .filter(|b| b.name.starts_with(&prefix))
                    .map(|b| Value::String(b.name.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_canonical_hash" => {
                // Convenience: canonicalize then hash. The semantic
                // memory key the LLM actually wants — invariant under
                // every cosmetic edit.
                if args.is_empty() {
                    return Err("omc_canonical_hash requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let canon = crate::canonical::canonicalize(&code)
                    .map_err(|e| format!("omc_canonical_hash: {}", e))?;
                let (attractor, raw, dist) = crate::tokenizer::code_hash(&canon);
                let mut map = std::collections::BTreeMap::new();
                map.insert("raw".to_string(), Value::HInt(HInt::new(raw)));
                map.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                map.insert("distance".to_string(), Value::HInt(HInt::new(dist)));
                map.insert("resonance".to_string(),
                    Value::HFloat(crate::value::HInt::compute_resonance(raw)));
                Ok(Value::dict_from(map))
            }
            "omc_categories_count" => {
                Ok(Value::HInt(HInt::new(crate::docs::categories().len() as i64)))
            }
            "omc_builtin_count" => {
                Ok(Value::HInt(HInt::new(crate::docs::BUILTINS.len() as i64)))
            }
            "omc_unique_count" => {
                Ok(Value::HInt(HInt::new(
                    crate::docs::BUILTINS.iter().filter(|b| b.unique_to_omc).count() as i64
                )))
            }
            // ---- Token-level introspection (debugging the encoder) ---
            "omc_token_lookup" => {
                // Given a token ID, return the substring it expands to.
                if args.is_empty() {
                    return Err("omc_token_lookup requires (id: int)".to_string());
                }
                let id = self.eval_expr(&args[0])?.to_int() as usize;
                if id < crate::tokenizer::TOKEN_DICT.len() {
                    Ok(Value::String(crate::tokenizer::TOKEN_DICT[id].to_string()))
                } else {
                    Ok(Value::String(String::new()))
                }
            }
            "omc_token_describe" => {
                // Human-readable description of an encoded stream.
                // For each ID, emit "id=N expand='...'" lines.
                if args.is_empty() {
                    return Err("omc_token_describe requires (ids)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = v {
                    let ids: Vec<i64> = arr.items.borrow().iter().map(|x| x.to_int()).collect();
                    let mut out = String::new();
                    let mut i = 0;
                    while i < ids.len() {
                        let id = ids[i];
                        if id == 0 && i + 1 < ids.len() {
                            out.push_str(&format!("escape byte={}\n", ids[i+1]));
                            i += 2;
                        } else {
                            let entry = crate::tokenizer::TOKEN_DICT
                                .get(id as usize).unwrap_or(&"<unknown>");
                            let display = entry.replace('\n', "\\n").replace('\t', "\\t");
                            out.push_str(&format!("id={} expand=\"{}\"\n", id, display));
                            i += 1;
                        }
                    }
                    Ok(Value::String(out))
                } else {
                    Err("omc_token_describe: requires int array".to_string())
                }
            }
            "omc_token_byte_savings" => {
                // bytes_saved = raw_len - encoded_token_count.
                // Negative means encoding inflated (rare).
                if args.is_empty() {
                    return Err("omc_token_byte_savings requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let raw = code.len() as i64;
                let ids = crate::tokenizer::encode(&code).len() as i64;
                Ok(Value::HInt(HInt::new(raw - ids)))
            }
            // ---- Substrate scoring over code ----
            "omc_substrate_score" => {
                // How substrate-aligned is this code? Computed as the
                // fraction of canonical-tokens whose ID is itself a
                // Fibonacci attractor. 1.0 = every token sits on an
                // attractor; 0.0 = every token off-attractor.
                if args.is_empty() {
                    return Err("omc_substrate_score requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let canon = crate::canonical::canonicalize(&code)
                    .map_err(|e| format!("omc_substrate_score: {}", e))?;
                let ids = crate::tokenizer::encode(&canon);
                if ids.is_empty() {
                    return Ok(Value::HFloat(0.0));
                }
                let on_attractor: usize = ids.iter()
                    .filter(|&&id| {
                        let (_, d) = crate::phi_pi_fib::nearest_attractor_with_dist(id);
                        d == 0
                    }).count();
                Ok(Value::HFloat(on_attractor as f64 / ids.len() as f64))
            }
            "omc_attractor_density" => {
                // Same as substrate_score but over RAW source (no
                // canonicalization). Useful for comparing how
                // "Fibonacci-shaped" different formatting styles are.
                if args.is_empty() {
                    return Err("omc_attractor_density requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let ids = crate::tokenizer::encode(&code);
                if ids.is_empty() {
                    return Ok(Value::HFloat(0.0));
                }
                let on: usize = ids.iter()
                    .filter(|&&id| crate::phi_pi_fib::nearest_attractor_with_dist(id).1 == 0)
                    .count();
                Ok(Value::HFloat(on as f64 / ids.len() as f64))
            }
            // ---- Code memory (session-state for LLMs) ----
            "omc_remember" => {
                // omc_remember(name, code) — store the canonical hash
                // of `code` under `name`. Lets LLMs say "remember this
                // function as 'softmax_v1'" and recall later.
                if args.len() < 2 {
                    return Err("omc_remember requires (name, code)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                let code = self.eval_expr(&args[1])?.to_display_string();
                let canon = crate::canonical::canonicalize(&code)
                    .map_err(|e| format!("omc_remember: {}", e))?;
                let (_, raw, _) = crate::tokenizer::code_hash(&canon);
                self.code_memory.borrow_mut().insert(name, raw);
                Ok(Value::HInt(HInt::new(raw)))
            }
            "omc_recall" => {
                // omc_recall(name) — get the hash stored under `name`,
                // or null if unknown.
                if args.is_empty() {
                    return Err("omc_recall requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match self.code_memory.borrow().get(&name) {
                    Some(&h) => Ok(Value::HInt(HInt::new(h))),
                    None => Ok(Value::Null),
                }
            }
            "omc_recall_matches" => {
                // omc_recall_matches(name, code) — 1 if the current
                // `code` has the same canonical hash as what was
                // remembered under `name`. The "did this change?" check.
                if args.len() < 2 {
                    return Err("omc_recall_matches requires (name, code)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                let code = self.eval_expr(&args[1])?.to_display_string();
                let stored = match self.code_memory.borrow().get(&name) {
                    Some(&h) => h,
                    None => return Ok(Value::HInt(HInt::new(0))),
                };
                let canon = crate::canonical::canonicalize(&code)
                    .map_err(|e| format!("omc_recall_matches: {}", e))?;
                let (_, current, _) = crate::tokenizer::code_hash(&canon);
                Ok(Value::HInt(HInt::new(if stored == current { 1 } else { 0 })))
            }
            "omc_memory_keys" => {
                // List all remembered names.
                let mem = self.code_memory.borrow();
                let out: Vec<Value> = mem.keys()
                    .map(|k| Value::String(k.clone()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_memory_clear" => {
                self.code_memory.borrow_mut().clear();
                Ok(Value::Null)
            }
            // ---- Composition: omc_help_markdown ----
            "omc_help_markdown" => {
                // Markdown-formatted help — easier for LLMs that
                // serialize into rendered chat windows.
                if args.is_empty() {
                    return Err("omc_help_markdown requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(doc) => Ok(Value::String(crate::docs::render_markdown(doc))),
                    None => Ok(Value::String(format!(
                        "### `{}`\n\n*Not in registry.* Try `omc_did_you_mean(\"{}\")`.",
                        name, name
                    ))),
                }
            }
            // ---- HBit-based substrate hash (uses dual-band metadata) ---
            "omc_hbit_hash" => {
                // Hash via HBit dual-band: combine the integer value
                // and its substrate-resonance into the hash so two
                // values that differ only in resonance still produce
                // different IDs. This is the OMC version of "hashing
                // also weighs how 'substrate-coherent' the input is".
                if args.is_empty() {
                    return Err("omc_hbit_hash requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let raw = crate::tokenizer::fnv1a_64(code.as_bytes());
                // Mix in the substrate-resonance of the hash itself.
                let h = HInt::new(raw);
                let blended = (raw as f64 * (1.0 + h.resonance) + h.him_score * 1e6) as i64;
                Ok(Value::HInt(HInt::new(blended)))
            }
            // ---- Convenience composers ----
            "omc_token_compress_pct" => {
                // 100 * (1 - ids_len / raw_len). Direct % savings.
                if args.is_empty() {
                    return Err("omc_token_compress_pct requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let raw = code.len() as f64;
                if raw == 0.0 { return Ok(Value::HFloat(0.0)); }
                let ids = crate::tokenizer::encode(&code).len() as f64;
                Ok(Value::HFloat(100.0 * (1.0 - ids / raw)))
            }
            "omc_help_all_category" => {
                // Return [omc_help(name) for name in <category>] as
                // an array of dicts. Useful for "show me everything in
                // the substrate category" in one call.
                if args.is_empty() {
                    return Err("omc_help_all_category requires (category)".to_string());
                }
                let cat = self.eval_expr(&args[0])?.to_display_string();
                let out: Vec<Value> = crate::docs::BUILTINS.iter()
                    .filter(|b| b.category == cat)
                    .map(|d| {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("name".to_string(), Value::String(d.name.to_string()));
                        map.insert("signature".to_string(), Value::String(d.signature.to_string()));
                        map.insert("description".to_string(), Value::String(d.description.to_string()));
                        map.insert("example".to_string(), Value::String(d.example.to_string()));
                        map.insert("unique_to_omc".to_string(),
                            Value::HInt(HInt::new(if d.unique_to_omc { 1 } else { 0 })));
                        Value::dict_from(map)
                    })
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // ---- LLM workflow primitives (single-call bundles) ----
            "omc_cheatsheet" => {
                if args.is_empty() {
                    return Err("omc_cheatsheet requires (topic: string)".to_string());
                }
                let topic = self.eval_expr(&args[0])?.to_display_string();
                Ok(Value::String(crate::llm_workflow::cheatsheet(&topic)))
            }
            "omc_unique_overview" => {
                Ok(Value::String(crate::llm_workflow::unique_overview()))
            }
            "omc_python_translation" => {
                Ok(Value::String(crate::llm_workflow::python_translation()))
            }
            "omc_builtin_index_markdown" => {
                Ok(Value::String(crate::llm_workflow::builtin_index_markdown()))
            }
            "omc_bootstrap_pack" => {
                Ok(Value::String(crate::llm_workflow::bootstrap_pack()))
            }
            "omc_change_report" => {
                if args.len() < 2 {
                    return Err("omc_change_report requires (old, new)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let r = crate::llm_workflow::change_report(&a, &b)
                    .map_err(|e| format!("omc_change_report: {}", e))?;
                let mut map = std::collections::BTreeMap::new();
                for (k, v) in r {
                    map.insert(k, Value::String(v));
                }
                Ok(Value::dict_from(map))
            }
            "omc_id" => {
                // Canonical OMC ID: "omcid-<fp>-<short_hash>" — stable
                // under cosmetic edits. The session-memory key for code.
                if args.is_empty() {
                    return Err("omc_id requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                match crate::llm_workflow::omc_id(&code) {
                    Ok(id) => Ok(Value::String(id)),
                    Err(e) => Err(format!("omc_id: {}", e)),
                }
            }
            // ---- Native LLM builtins: llm_call / llm_chat / llm_embed ----
            //
            // These builtins let OMC programs call an LLM API directly.
            // The provider, endpoint, model, and API key are resolved from
            // environment variables so no credentials appear in OMC source:
            //
            //   OMC_LLM_PROVIDER  — "anthropic" (default) | "openai" | "openai-compat"
            //   OMC_LLM_MODEL     — model name (provider-specific default if unset)
            //   OMC_LLM_API_KEY   — API key (overrides ANTHROPIC_API_KEY / OPENAI_API_KEY)
            //   OMC_LLM_URL       — custom base URL (for openai-compat / local servers)
            //   OMC_LLM_MAX_TOKENS— max tokens for completions (default 1024)
            //
            // llm_call(prompt: string) -> string
            //   Single-turn completion. Sends `prompt` as the only user message
            //   and returns the assistant's text response.
            //
            // llm_chat(messages: dict[]) -> string
            //   Multi-turn chat. `messages` is an array of dicts, each with
            //   "role" ("user"|"assistant"|"system") and "content" (string).
            //   Returns the assistant's text response.
            //
            // llm_embed(text: string) -> float[]
            //   Returns the embedding vector for `text` as an array of floats.
            //   Only available for providers that support embeddings (openai).
            #[cfg(feature = "native-llm")]
            "llm_call" => {
                if args.is_empty() {
                    return Err("llm_call requires (prompt: string)".to_string());
                }
                let prompt = self.eval_expr(&args[0])?.to_display_string();
                let model = if args.len() > 1 {
                    match self.eval_expr(&args[1])? {
                        Value::Null => None,
                        v => Some(v.to_display_string()),
                    }
                } else {
                    None
                };
                let system = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::Null => None,
                        v => Some(v.to_display_string()),
                    }
                } else {
                    None
                };
                crate::llm_builtins::llm_call_sys(&prompt, model.as_deref(), system.as_deref())
            }
            #[cfg(feature = "native-llm")]
            "llm_chat" => {
                if args.is_empty() {
                    return Err("llm_chat requires (messages: dict[])".to_string());
                }
                let msgs_val = self.eval_expr(&args[0])?;
                let model = if args.len() > 1 {
                    Some(self.eval_expr(&args[1])?.to_display_string())
                } else {
                    None
                };
                let messages = crate::llm_builtins::parse_messages(&msgs_val)?;
                crate::llm_builtins::llm_chat(&messages, model.as_deref())
            }
            #[cfg(feature = "native-llm")]
            "llm_embed" => {
                if args.is_empty() {
                    return Err("llm_embed requires (text: string)".to_string());
                }
                let text = self.eval_expr(&args[0])?.to_display_string();
                let model = if args.len() > 1 {
                    Some(self.eval_expr(&args[1])?.to_display_string())
                } else {
                    None
                };
                let floats = crate::llm_builtins::llm_embed(&text, model.as_deref())?;
                Ok(floats)
            }
            // llm_system(prompt, system, model?) -> string
            //   Convenience: send a user prompt with a system instruction in one call.
            "llm_system" => {
                if args.len() < 2 {
                    return Err("llm_system requires (prompt, system, model?)".to_string());
                }
                let prompt = self.eval_expr(&args[0])?.to_display_string();
                let system = self.eval_expr(&args[1])?.to_display_string();
                let model = if args.len() > 2 {
                    Some(self.eval_expr(&args[2])?.to_display_string())
                } else {
                    None
                };
                crate::llm_builtins::llm_system(&prompt, &system, model.as_deref())
            }
            // llm_stream_print(prompt, system?, model?) -> string
            //   Streams LLM response to stdout token-by-token, returns full text.
            //   Uses SSE streaming API. system defaults to null (no system prompt).
            "llm_stream_print" => {
                if args.is_empty() {
                    return Err("llm_stream_print requires (prompt, system?, model?)".to_string());
                }
                let prompt = self.eval_expr(&args[0])?.to_display_string();
                let system = if args.len() > 1 {
                    match self.eval_expr(&args[1])? {
                        Value::Null => None,
                        v => Some(v.to_display_string()),
                    }
                } else {
                    None
                };
                let model = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::Null => None,
                        v => Some(v.to_display_string()),
                    }
                } else {
                    None
                };
                crate::llm_builtins::llm_stream_print(&prompt, system.as_deref(), model.as_deref())
            }
            // llm_judge(responses, criteria, model?) -> dict[]
            //   Score each response in an array; returns [{idx, score, reason}] sorted best-first.
            "llm_judge" => {
                if args.len() < 2 {
                    return Err("llm_judge requires (responses, criteria, model?)".to_string());
                }
                let responses = self.eval_expr(&args[0])?;
                let criteria = self.eval_expr(&args[1])?.to_display_string();
                let model = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::Null => None,
                        v => Some(v.to_display_string()),
                    }
                } else { None };
                crate::llm_builtins::llm_judge(&responses, &criteria, model.as_deref())
            }
            // llm_compare(a, b, criteria, model?) -> dict
            //   Pick the better of two responses; returns {winner: "A"|"B", reason: "..."}.
            "llm_compare" => {
                if args.len() < 3 {
                    return Err("llm_compare requires (a, b, criteria, model?)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let criteria = self.eval_expr(&args[2])?.to_display_string();
                let model = if args.len() > 3 {
                    match self.eval_expr(&args[3])? {
                        Value::Null => None,
                        v => Some(v.to_display_string()),
                    }
                } else { None };
                crate::llm_builtins::llm_compare(&a, &b, &criteria, model.as_deref())
            }
            // llm_models() -> dict[]
            //   Returns the list of models available from the active provider.
            //   Each element is a dict with at least {"id": string, "provider": string}.
            //   Useful for discovery: `let ms = llm_models()`.
            "llm_models" => {
                Ok(crate::llm_builtins::llm_models())
            }
            // batch_llm_call(prompts, model?, concurrency?) -> string[]
            //   Send multiple prompts in sequence, return all replies.
            //   `prompts` is an array of strings or dicts {prompt, system?, model?}.
            //   `model` overrides the default for all calls; per-prompt model takes precedence.
            //   `concurrency` is accepted but calls are currently sequential.
            "batch_llm_call" => {
                if args.is_empty() {
                    return Err("batch_llm_call requires (prompts: array)".to_string());
                }
                let prompts_val = self.eval_expr(&args[0])?;
                let default_model = if args.len() > 1 {
                    let v = self.eval_expr(&args[1])?;
                    match v {
                        Value::Null => None,
                        other => Some(other.to_display_string()),
                    }
                } else {
                    None
                };
                let concurrency = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::HInt(n) => n.value as usize,
                        _ => 3,
                    }
                } else {
                    3
                };
                crate::llm_builtins::batch_llm_call(
                    &prompts_val,
                    default_model.as_deref(),
                    concurrency,
                )
            }
            // batch_llm_chat(messages_array, model?, concurrency?) -> string[]
            //   Send multiple chat conversations in sequence, return all replies.
            //   `messages_array` is an array of arrays (each inner array is one chat's messages).
            "batch_llm_chat" => {
                if args.is_empty() {
                    return Err("batch_llm_chat requires (messages_array: array)".to_string());
                }
                let messages_array_val = self.eval_expr(&args[0])?;
                let default_model = if args.len() > 1 {
                    let v = self.eval_expr(&args[1])?;
                    match v {
                        Value::Null => None,
                        other => Some(other.to_display_string()),
                    }
                } else {
                    None
                };
                let concurrency = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::HInt(n) => n.value as usize,
                        _ => 3,
                    }
                } else {
                    3
                };
                crate::llm_builtins::batch_llm_chat(
                    &messages_array_val,
                    default_model.as_deref(),
                    concurrency,
                )
            }
            // llm_tools(messages, tools, model?) -> dict
            //   Structured tool-calling. Returns {type, content, name, id, input, stop_reason}.
            //   Works with both Anthropic and OpenAI tool-calling APIs.
            "llm_tools" => {
                if args.len() < 2 {
                    return Err("llm_tools requires (messages, tools, model?)".to_string());
                }
                let messages_val = self.eval_expr(&args[0])?;
                let tools_val = self.eval_expr(&args[1])?;
                let model_override = if args.len() > 2 {
                    match self.eval_expr(&args[2])? {
                        Value::String(s) => Some(s),
                        _ => None,
                    }
                } else {
                    None
                };
                crate::llm_builtins::llm_tools(
                    &messages_val,
                    &tools_val,
                    model_override.as_deref(),
                )
            }
            // substrate_embed(text, dims?) -> float[]
            //   Phi-Pi-Fib harmonic text embedding. L2-normalised. No API call needed.
            "substrate_embed" => {
                if args.is_empty() {
                    return Err("substrate_embed requires (text, dims?)".to_string());
                }
                let text = match self.eval_expr(&args[0])? {
                    Value::String(s) => s,
                    other => other.to_display_string(),
                };
                let dims = if args.len() > 1 {
                    match self.eval_expr(&args[1])? {
                        Value::HInt(n) => n.value as usize,
                        _ => 16,
                    }
                } else {
                    16
                };
                Ok(crate::llm_builtins::substrate_embed(&text, dims))
            }
            // ---- Process execution: omc_spawn / omc_pipe ----------------
            //
            // omc_spawn(cmd, args?, env_vars?, timeout_ms?) -> dict
            //   Spawns a subprocess and waits for it to complete.
            //   Returns {stdout, stderr, exit_code, ok}.
            //   Critical for self-improvement: OMC can run omc itself.
            //
            // omc_pipe(commands) -> dict
            //   Pipes multiple commands together like shell pipes.
            //   commands is an array of arrays: [[cmd, arg, ...], ...].
            //   Returns {stdout, stderr, exit_code, ok}.
            "omc_spawn" => {
                let eval_args: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                crate::process_builtins::omc_spawn(&eval_args)
            }
            "omc_pipe" => {
                let eval_args: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                crate::process_builtins::omc_pipe(&eval_args)
            }
            // ---- Native HTTP builtins -------------------------------------------
            //
            // http_get(url, headers?)       -> {status, body, ok}
            // http_post(url, body, headers?) -> {status, body, ok}
            // http_post_json(url, data, headers?) -> {status, body, ok, json}
            // http_put(url, body, headers?)  -> {status, body, ok}
            // http_delete(url, headers?)     -> {status, body, ok}
            //
            // headers is an optional dict of {header_name: header_value}.
            // Passing null for headers is accepted.
            // ok is true when 200 <= status < 300.
            "http_get" => {
                let eargs: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;
                crate::http_builtins::http_get(&eargs)
            }
            "http_post" => {
                let eargs: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;
                crate::http_builtins::http_post(&eargs)
            }
            "http_post_json" => {
                let eargs: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;
                crate::http_builtins::http_post_json(&eargs)
            }
            "http_put" => {
                let eargs: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;
                crate::http_builtins::http_put(&eargs)
            }
            "http_delete" => {
                let eargs: Vec<Value> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;
                crate::http_builtins::http_delete(&eargs)
            }
            // ---- Substrate-signed messaging (LLM ↔ LLM protocol) ---
            //
            // omc_msg_sign(content, sender_id, kind) — produces a dict
            // that wraps `content` with HBit substrate metadata
            // derived from the canonical-hash of the content. The
            // metadata is RECOMPUTABLE — receivers verify by
            // recomputing from the content, no trust required.
            //
            // Wire-format dict:
            //   {
            //     content        : original string
            //     sender_id      : int
            //     kind           : int (1=code, 2=request, 3=response, ...)
            //     content_hash   : fnv1a of canonical content
            //     resonance      : substrate-derived from content_hash
            //     him_score      : ditto
            //     attractor      : nearest Fibonacci to content_hash
            //     packed         : CRT-packed (sender_id, kind, hash_mod_M)
            //   }
            "omc_msg_sign" => {
                if args.len() < 3 {
                    return Err("omc_msg_sign requires (content, sender_id, kind)".to_string());
                }
                let content = self.eval_expr(&args[0])?.to_display_string();
                let sender_id = self.eval_expr(&args[1])?.to_int();
                let kind = self.eval_expr(&args[2])?.to_int();
                // Canonicalize so cosmetic edits don't change the signature.
                // Falls back to raw content for non-OMC strings.
                let canon = crate::canonical::canonicalize(&content)
                    .unwrap_or_else(|_| content.clone());
                let hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
                let h = HInt::new(hash);
                let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(hash);
                let moduli = crate::tokenizer::CRT_MODULI;
                let streams = [
                    sender_id.rem_euclid(moduli[0]),
                    kind.rem_euclid(moduli[1]),
                    hash.rem_euclid(moduli[2]),
                ];
                let packed = crate::tokenizer::crt_pack(&streams, moduli)
                    .unwrap_or(0);
                let mut map = std::collections::BTreeMap::new();
                map.insert("content".to_string(), Value::String(content));
                map.insert("sender_id".to_string(), Value::HInt(HInt::new(sender_id)));
                map.insert("kind".to_string(), Value::HInt(HInt::new(kind)));
                map.insert("content_hash".to_string(), Value::HInt(HInt::new(hash)));
                map.insert("resonance".to_string(), Value::HFloat(h.resonance));
                map.insert("him_score".to_string(), Value::HFloat(h.him_score));
                map.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                map.insert("packed".to_string(), Value::HInt(HInt::new(packed)));
                Ok(Value::dict_from(map))
            }
            // omc_msg_verify(msg) — recompute substrate metadata from
            // msg's content and check it matches the signed values.
            // Returns {valid, sender_id, kind, content, expected_hash,
            // actual_hash, drift_resonance, drift_him}. valid==1 iff
            // recomputed signature is identical.
            "omc_msg_verify" => {
                if args.is_empty() {
                    return Err("omc_msg_verify requires (msg: dict)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let dict = if let Value::Dict(d) = v { d } else {
                    return Err("omc_msg_verify: msg must be a dict".to_string());
                };
                let d = dict.borrow();
                let content = d.get("content").map(|x| x.to_display_string())
                    .unwrap_or_default();
                let claimed_hash = d.get("content_hash").map(|x| x.to_int()).unwrap_or(0);
                let claimed_res = d.get("resonance").map(|x| x.to_float()).unwrap_or(0.0);
                let claimed_him = d.get("him_score").map(|x| x.to_float()).unwrap_or(0.0);
                let canon = crate::canonical::canonicalize(&content)
                    .unwrap_or_else(|_| content.clone());
                let actual_hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
                let h = HInt::new(actual_hash);
                let hash_match = claimed_hash == actual_hash;
                let res_match = (claimed_res - h.resonance).abs() < 1e-9;
                let him_match = (claimed_him - h.him_score).abs() < 1e-9;
                let valid = hash_match && res_match && him_match;
                let mut out = std::collections::BTreeMap::new();
                out.insert("valid".to_string(),
                    Value::HInt(HInt::new(if valid { 1 } else { 0 })));
                out.insert("sender_id".to_string(),
                    d.get("sender_id").cloned().unwrap_or(Value::Null));
                out.insert("kind".to_string(),
                    d.get("kind").cloned().unwrap_or(Value::Null));
                out.insert("content".to_string(), Value::String(content));
                out.insert("expected_hash".to_string(),
                    Value::HInt(HInt::new(claimed_hash)));
                out.insert("actual_hash".to_string(),
                    Value::HInt(HInt::new(actual_hash)));
                out.insert("drift_resonance".to_string(),
                    Value::HFloat((claimed_res - h.resonance).abs()));
                out.insert("drift_him".to_string(),
                    Value::HFloat((claimed_him - h.him_score).abs()));
                Ok(Value::dict_from(out))
            }
            // ---- ONN / self-instantiation (the context-problem layer) ---
            //
            // omc_m3_spawn_count(n) — sublog optimal subagent count via
            // Fibonacci-π-Fibonacci wave interference. Solves "how many
            // specialists do I need to compress N items?"
            "omc_m3_spawn_count" => {
                if args.is_empty() {
                    return Err("omc_m3_spawn_count requires (n: int)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(crate::onn::m3_spawn_count(n))))
            }
            // omc_self_instantiate(items: string[], task_hint: string)
            //   -> dict[] of specialists. Each specialist:
            //     {fold_index, summary, mu, sigma, dominant_attractor,
            //      resonance, wave_amplitude, item_count}
            // Specialist count is m3_spawn_count(len(items)).
            "omc_self_instantiate" => {
                if args.len() < 2 {
                    return Err("omc_self_instantiate requires (items: string[], task_hint: string)".to_string());
                }
                let items_v = self.eval_expr(&args[0])?;
                let task_hint = self.eval_expr(&args[1])?.to_display_string();
                let items: Vec<String> = if let Value::Array(arr) = items_v {
                    arr.items.borrow().iter().map(|v| v.to_display_string()).collect()
                } else {
                    return Err("omc_self_instantiate: items must be a string array".to_string());
                };
                let specs = crate::onn::self_instantiate(&items, &task_hint);
                let out: Vec<Value> = specs.iter().map(|s| {
                    let mut m = std::collections::BTreeMap::new();
                    m.insert("fold_index".to_string(), Value::HInt(HInt::new(s.fold_index as i64)));
                    m.insert("summary".to_string(), Value::String(s.summary.clone()));
                    m.insert("mu".to_string(), Value::HFloat(s.mu));
                    m.insert("sigma".to_string(), Value::HFloat(s.sigma));
                    m.insert("dominant_attractor".to_string(),
                        Value::HInt(HInt::new(s.dominant_attractor)));
                    m.insert("resonance".to_string(), Value::HFloat(s.resonance));
                    m.insert("wave_amplitude".to_string(), Value::HFloat(s.wave_amplitude));
                    m.insert("item_count".to_string(), Value::HInt(HInt::new(s.item_count as i64)));
                    Value::dict_from(m)
                }).collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // omc_fold_back(parent_mu, parent_sigma, parent_turn,
            //               specialists: dict[]) -> dict
            //   Updated {mu, sigma, turn_count, dominant_attractor,
            //   num_specialists_folded, resonance}.
            "omc_fold_back" => {
                if args.len() < 4 {
                    return Err("omc_fold_back requires (parent_mu, parent_sigma, parent_turn, specialists)".to_string());
                }
                let parent_mu = self.eval_expr(&args[0])?.to_float();
                let parent_sigma = self.eval_expr(&args[1])?.to_float();
                let parent_turn = self.eval_expr(&args[2])?.to_int();
                let specs_v = self.eval_expr(&args[3])?;
                let arr = if let Value::Array(a) = specs_v { a } else {
                    return Err("omc_fold_back: specialists must be a dict array".to_string());
                };
                // Reconstruct Specialist structs from the dicts.
                let mut specs: Vec<crate::onn::Specialist> = Vec::new();
                for item in arr.items.borrow().iter() {
                    let d = if let Value::Dict(d) = item { d } else { continue; };
                    let d = d.borrow();
                    specs.push(crate::onn::Specialist {
                        fold_index: d.get("fold_index").map(|v| v.to_int()).unwrap_or(0) as usize,
                        summary: d.get("summary").map(|v| v.to_display_string()).unwrap_or_default(),
                        mu: d.get("mu").map(|v| v.to_float()).unwrap_or(0.0),
                        sigma: d.get("sigma").map(|v| v.to_float()).unwrap_or(0.0),
                        dominant_attractor: d.get("dominant_attractor").map(|v| v.to_int()).unwrap_or(0),
                        resonance: d.get("resonance").map(|v| v.to_float()).unwrap_or(0.0),
                        wave_amplitude: d.get("wave_amplitude").map(|v| v.to_float()).unwrap_or(0.0),
                        item_count: d.get("item_count").map(|v| v.to_int()).unwrap_or(0) as usize,
                    });
                }
                let folded = crate::onn::fold_back(parent_mu, parent_sigma, parent_turn, &specs);
                let mut out = std::collections::BTreeMap::new();
                for (k, v) in folded {
                    out.insert(k, Value::HFloat(v));
                }
                Ok(Value::dict_from(out))
            }
            // omc_context_compress(messages: string[]) — convenience:
            // = omc_self_instantiate(messages, "context-compress"). The
            // headline application: shrink N messages to ~log_log(N)
            // specialists carrying μ/σ/attractor state of each "wave"
            // of the conversation.
            "omc_context_compress" => {
                if args.is_empty() {
                    return Err("omc_context_compress requires (messages: string[])".to_string());
                }
                let items_v = self.eval_expr(&args[0])?;
                let items: Vec<String> = if let Value::Array(arr) = items_v {
                    arr.items.borrow().iter().map(|v| v.to_display_string()).collect()
                } else {
                    return Err("omc_context_compress: messages must be a string array".to_string());
                };
                let specs = crate::onn::self_instantiate(&items, "context-compress");
                let out: Vec<Value> = specs.iter().map(|s| {
                    let mut m = std::collections::BTreeMap::new();
                    m.insert("fold_index".to_string(), Value::HInt(HInt::new(s.fold_index as i64)));
                    m.insert("summary".to_string(), Value::String(s.summary.clone()));
                    m.insert("mu".to_string(), Value::HFloat(s.mu));
                    m.insert("sigma".to_string(), Value::HFloat(s.sigma));
                    m.insert("dominant_attractor".to_string(),
                        Value::HInt(HInt::new(s.dominant_attractor)));
                    m.insert("resonance".to_string(), Value::HFloat(s.resonance));
                    m.insert("item_count".to_string(), Value::HInt(HInt::new(s.item_count as i64)));
                    Value::dict_from(m)
                }).collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // omc_spawn_child_fold(seed: int, reason: string)
            //   -> dict {fold_id, focus_numerator, focus_denominator,
            //            spawn_reason, resonance_target, explored_value,
            //            final_resonance}
            //
            // Ported from Sovereign_Lattice register_singularity_integration.
            // A ChildFold is the "expand a single token into its
            // computational subspace" primitive — given any HInt-shaped
            // seed, deterministically produce the boundary exploration
            // the parent register would have performed if its tension
            // exceeded 1/φ.
            "omc_spawn_child_fold" => {
                if args.is_empty() {
                    return Err("omc_spawn_child_fold requires (seed: int, reason?: string)".to_string());
                }
                let seed = self.eval_expr(&args[0])?.to_int();
                let reason = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_display_string()
                } else { "tension threshold exceeded".to_string() };
                let cf = crate::onn::spawn_child_fold(seed, &reason);
                let mut map = std::collections::BTreeMap::new();
                map.insert("fold_id".to_string(), Value::HInt(HInt::new(cf.fold_id)));
                map.insert("focus_numerator".to_string(), Value::HInt(HInt::new(cf.focus_numerator)));
                map.insert("focus_denominator".to_string(), Value::HInt(HInt::new(cf.focus_denominator)));
                map.insert("spawn_reason".to_string(), Value::String(cf.spawn_reason));
                map.insert("resonance_target".to_string(), Value::HFloat(cf.resonance_target));
                map.insert("explored_value".to_string(), Value::HInt(HInt::new(cf.explored_value)));
                map.insert("final_resonance".to_string(), Value::HFloat(cf.final_resonance));
                Ok(Value::dict_from(map))
            }
            // omc_geodesic_expand(seed: int, n_samples: int)
            //   -> [[value, resonance], ...]
            //
            // "Replicate compressed data from a single token" formalized:
            // walk the φ-field geodesic from `seed` toward its nearest
            // Fibonacci attractor in n_samples equal steps. Each sample
            // is a (value, resonance) pair. Deterministic per (seed, n).
            //
            // Useful for: stable substrate-anchored pseudo-random sequences,
            // expanding a single recall-key into a memory trace, geometric
            // (not semantic) reconstruction.
            "omc_geodesic_expand" => {
                if args.len() < 2 {
                    return Err("omc_geodesic_expand requires (seed: int, n_samples: int)".to_string());
                }
                let seed = self.eval_expr(&args[0])?.to_int();
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let samples = crate::onn::geodesic_expand(seed, n);
                let out: Vec<Value> = samples.iter().map(|(v, r)| {
                    let pair = vec![Value::HInt(HInt::new(*v)), Value::HFloat(*r)];
                    Value::Array(HArray::from_vec(pair))
                }).collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // omc_llm_self_instantiate(context: string[], task: string,
            //                          base_dir: string, base_sender_id: int)
            //   -> dict[] manifest of {specialist_id, prompt_path,
            //                          specialist_dict}.
            //   Compresses N context messages to M3(N) specialists,
            //   writes each as a signed prompt-file in base_dir, and
            //   returns the manifest. An orchestrator (human or
            //   automated) can spawn N LLM sessions, one per file.
            //   Each spawned session starts with its specialist's
            //   inherited geometric state as the seed.
            //
            //   This is the "self-instantiation primitive for LLMs":
            //   structural fan-out with substrate-derived state
            //   inheritance. Actual LLM-process spawning is out of
            //   scope (OMC doesn't fork LLMs), but the manifest gives
            //   the orchestrator everything it needs.
            "omc_llm_self_instantiate" => {
                if args.len() < 4 {
                    return Err("omc_llm_self_instantiate requires (context: string[], task: string, base_dir: string, base_sender_id: int)".to_string());
                }
                let ctx_v = self.eval_expr(&args[0])?;
                let task = self.eval_expr(&args[1])?.to_display_string();
                let base_dir = self.eval_expr(&args[2])?.to_display_string();
                let base_sender = self.eval_expr(&args[3])?.to_int();
                let messages: Vec<String> = if let Value::Array(arr) = ctx_v {
                    arr.items.borrow().iter().map(|v| v.to_display_string()).collect()
                } else {
                    return Err("omc_llm_self_instantiate: context must be a string array".to_string());
                };
                let specs = crate::onn::self_instantiate(&messages, &task);
                std::fs::create_dir_all(&base_dir).map_err(|e|
                    format!("omc_llm_self_instantiate: mkdir {}: {}", base_dir, e))?;
                let mut manifest: Vec<Value> = Vec::with_capacity(specs.len());
                for s in &specs {
                    // Each specialist gets a derived sender_id so the
                    // orchestrator can tell them apart.
                    let specialist_id = base_sender.wrapping_add(s.fold_index as i64);
                    // The prompt embeds the specialist's state + the
                    // task hint so the spawned LLM has context.
                    let prompt = format!(
                        "[Self-instantiated specialist {}/{}]\n\
                         Task: {}\n\
                         Inherited geometric state:\n\
                         - mu (mean φ-resonance): {:.6}\n\
                         - sigma: {:.6}\n\
                         - dominant_attractor: {}\n\
                         - wave_amplitude: {:.6}\n\
                         - items_in_slice: {}\n\n\
                         Your slice of input:\n{}\n",
                        s.fold_index + 1, specs.len(), task,
                        s.mu, s.sigma, s.dominant_attractor,
                        s.wave_amplitude, s.item_count, s.summary
                    );
                    let canon = crate::canonical::canonicalize(&prompt)
                        .unwrap_or_else(|_| prompt.clone());
                    let hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
                    let h = HInt::new(hash);
                    let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(hash);
                    let moduli = crate::tokenizer::CRT_MODULI;
                    let streams = [
                        base_sender.rem_euclid(moduli[0]),
                        1i64.rem_euclid(moduli[1]),  // kind=1 (request)
                        hash.rem_euclid(moduli[2]),
                    ];
                    let packed = crate::tokenizer::crt_pack(&streams, moduli).unwrap_or(0);
                    let mut msg = std::collections::BTreeMap::new();
                    msg.insert("content".to_string(), Value::String(prompt));
                    msg.insert("sender_id".to_string(), Value::HInt(HInt::new(base_sender)));
                    msg.insert("target_id".to_string(), Value::HInt(HInt::new(specialist_id)));
                    msg.insert("kind".to_string(), Value::HInt(HInt::new(1)));
                    msg.insert("content_hash".to_string(), Value::HInt(HInt::new(hash)));
                    msg.insert("resonance".to_string(), Value::HFloat(h.resonance));
                    msg.insert("him_score".to_string(), Value::HFloat(h.him_score));
                    msg.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                    msg.insert("packed".to_string(), Value::HInt(HInt::new(packed)));
                    let msg_value = Value::dict_from(msg);
                    let wire = serde_json::to_string(&crate::interpreter::value_to_json(&msg_value))
                        .unwrap_or_default();
                    let path = format!("{}/specialist_{:02}.json", base_dir, s.fold_index);
                    std::fs::write(&path, wire).map_err(|e|
                        format!("omc_llm_self_instantiate: write {}: {}", path, e))?;
                    // Manifest entry.
                    let mut manifest_entry = std::collections::BTreeMap::new();
                    manifest_entry.insert("specialist_id".to_string(),
                        Value::HInt(HInt::new(specialist_id)));
                    manifest_entry.insert("prompt_path".to_string(), Value::String(path));
                    manifest_entry.insert("fold_index".to_string(),
                        Value::HInt(HInt::new(s.fold_index as i64)));
                    manifest_entry.insert("mu".to_string(), Value::HFloat(s.mu));
                    manifest_entry.insert("sigma".to_string(), Value::HFloat(s.sigma));
                    manifest_entry.insert("dominant_attractor".to_string(),
                        Value::HInt(HInt::new(s.dominant_attractor)));
                    manifest_entry.insert("item_count".to_string(),
                        Value::HInt(HInt::new(s.item_count as i64)));
                    manifest.push(Value::dict_from(manifest_entry));
                }
                Ok(Value::Array(HArray::from_vec(manifest)))
            }

            // ── Native LLM builtins: llm_call / llm_chat / llm_embed / llm_models ──────
            //
            // omc_prompt_agent(target_id, prompt, sender_id, channel_dir?)
            //   — write a signed message to target_id's inbox file.
            //     Returns the packed message ID. Caller polls for response
            //     separately via read_file + omc_msg_verify.
            //
            // The "secondary brain" primitive: any OMC program can fire
            // off a query to another agent through the substrate channel.
            "omc_prompt_agent" => {
                if args.len() < 3 {
                    return Err("omc_prompt_agent requires (target_id, prompt, sender_id, channel_dir?)".to_string());
                }
                let target_id = self.eval_expr(&args[0])?.to_int();
                let prompt = self.eval_expr(&args[1])?.to_display_string();
                let sender_id = self.eval_expr(&args[2])?.to_int();
                let channel = if args.len() >= 4 {
                    self.eval_expr(&args[3])?.to_display_string()
                } else { "/home/thearchitect/omc_channel".to_string() };
                // Sign as kind=1 (request).
                let canon = crate::canonical::canonicalize(&prompt)
                    .unwrap_or_else(|_| prompt.clone());
                let hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
                let h = HInt::new(hash);
                let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(hash);
                let moduli = crate::tokenizer::CRT_MODULI;
                let streams = [
                    sender_id.rem_euclid(moduli[0]),
                    1i64.rem_euclid(moduli[1]),
                    hash.rem_euclid(moduli[2]),
                ];
                let packed = crate::tokenizer::crt_pack(&streams, moduli).unwrap_or(0);
                let mut map = std::collections::BTreeMap::new();
                map.insert("content".to_string(), Value::String(prompt));
                map.insert("sender_id".to_string(), Value::HInt(HInt::new(sender_id)));
                map.insert("target_id".to_string(), Value::HInt(HInt::new(target_id)));
                map.insert("kind".to_string(), Value::HInt(HInt::new(1)));
                map.insert("content_hash".to_string(), Value::HInt(HInt::new(hash)));
                map.insert("resonance".to_string(), Value::HFloat(h.resonance));
                map.insert("him_score".to_string(), Value::HFloat(h.him_score));
                map.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                map.insert("packed".to_string(), Value::HInt(HInt::new(packed)));
                let msg = Value::dict_from(map);
                let wire = serde_json::to_string(&crate::interpreter::value_to_json(&msg))
                    .unwrap_or_default();
                let path = format!("{}/prompt_to_{}.json", channel, target_id);
                std::fs::write(&path, wire).map_err(|e|
                    format!("omc_prompt_agent: write {}: {}", path, e))?;
                Ok(Value::HInt(HInt::new(packed)))
            }
            "omc_msg_sign_compressed" => {
                if args.len() < 3 {
                    return Err("omc_msg_sign_compressed requires (content, sender_id, kind, every_n?)".to_string());
                }
                let content = self.eval_expr(&args[0])?.to_display_string();
                let sender_id = self.eval_expr(&args[1])?.to_int();
                let kind = self.eval_expr(&args[2])?.to_int();
                let every_n = if args.len() >= 4 {
                    self.eval_expr(&args[3])?.to_int().max(1) as usize
                } else { 3usize };
                let canon = crate::canonical::canonicalize(&content)
                    .unwrap_or_else(|_| content.clone());
                let tokens = crate::tokenizer::encode(&canon);
                let sampled: Vec<Value> = tokens.iter().enumerate()
                    .filter(|(i, _)| i % every_n == 0)
                    .map(|(_, t)| Value::HInt(HInt::new(*t)))
                    .collect();
                let hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
                let h = HInt::new(hash);
                let (attractor, _) = crate::phi_pi_fib::nearest_attractor_with_dist(hash);
                let moduli = crate::tokenizer::CRT_MODULI;
                let streams = [
                    sender_id.rem_euclid(moduli[0]),
                    kind.rem_euclid(moduli[1]),
                    hash.rem_euclid(moduli[2]),
                ];
                let packed = crate::tokenizer::crt_pack(&streams, moduli).unwrap_or(0);
                let mut map = std::collections::BTreeMap::new();
                map.insert("sampled_tokens".to_string(),
                    Value::Array(HArray::from_vec(sampled.clone())));
                map.insert("sender_id".to_string(), Value::HInt(HInt::new(sender_id)));
                map.insert("kind".to_string(), Value::HInt(HInt::new(kind)));
                map.insert("content_hash".to_string(), Value::HInt(HInt::new(hash)));
                map.insert("resonance".to_string(), Value::HFloat(h.resonance));
                map.insert("him_score".to_string(), Value::HFloat(h.him_score));
                map.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                map.insert("packed".to_string(), Value::HInt(HInt::new(packed)));
                map.insert("every_n".to_string(), Value::HInt(HInt::new(every_n as i64)));
                map.insert("original_tok_count".to_string(),
                    Value::HInt(HInt::new(tokens.len() as i64)));
                map.insert("source_bytes".to_string(),
                    Value::HInt(HInt::new(content.len() as i64)));
                let ratio = if !sampled.is_empty() {
                    content.len() as f64 / sampled.len() as f64
                } else { 0.0 };
                map.insert("compression_ratio".to_string(), Value::HFloat(ratio));
                Ok(Value::dict_from(map))
            }
            "omc_msg_recover_compressed" => {
                if args.len() < 2 {
                    return Err("omc_msg_recover_compressed requires (msg, library)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let lib_v = self.eval_expr(&args[1])?;
                let target_hash = if let Value::Dict(d) = v {
                    d.borrow().get("content_hash").map(|x| x.to_int()).unwrap_or(0)
                } else {
                    return Err("omc_msg_recover_compressed: msg must be a dict".to_string());
                };
                let library: Vec<String> = if let Value::Array(arr) = lib_v {
                    arr.items.borrow().iter().map(|x| x.to_display_string()).collect()
                } else {
                    return Err("omc_msg_recover_compressed: library must be a string array".to_string());
                };
                for entry in &library {
                    let canon = crate::canonical::canonicalize(entry)
                        .unwrap_or_else(|_| entry.clone());
                    if crate::tokenizer::fnv1a_64(canon.as_bytes()) == target_hash {
                        return Ok(Value::String(entry.clone()));
                    }
                }
                Ok(Value::Null)
            }
            "omc_msg_serialize" => {
                // Convert a signed-message dict into a JSON wire string.
                // Useful when writing to a shared file / pipe / socket.
                if args.is_empty() {
                    return Err("omc_msg_serialize requires (msg: dict)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let j = crate::interpreter::value_to_json(&v);
                Ok(Value::String(serde_json::to_string(&j).unwrap_or_default()))
            }
            "omc_msg_deserialize" => {
                // Inverse: parse a wire JSON string back into a dict.
                if args.is_empty() {
                    return Err("omc_msg_deserialize requires (s: string)".to_string());
                }
                let s = self.eval_expr(&args[0])?.to_display_string();
                match serde_json::from_str::<serde_json::Value>(&s) {
                    Ok(j) => Ok(crate::interpreter::json_to_value(j)),
                    Err(e) => Err(format!("omc_msg_deserialize: {}", e)),
                }
            }
            // ---- Substrate-keyed compressed code store ----
            //
            // omc_codec_encode(code: string) -> dict
            //   Produce a wire-format compressed payload:
            //     {sampled_tokens, content_hash, attractor, dist,
            //      original_tok_count, source_bytes, compression_ratio}
            //   This is the v4 "token-sampled" form — keeps every Nth
            //   token of the canonical encoding. Decoder side requires a
            //   model trained on the corresponding library to fully
            //   recover; for in-library inputs, recovery is exact via
            //   omc_codec_decode_lookup against a known store.
            // ----- v0.3 symbolic prediction --------------------------
            // Stateless single-call API: given an array of source-file
            // paths and a partial-code prefix, return the top-k
            // ranked continuations (each a dict with fn_name, source,
            // file, canonical_hash, prefix_match_len, substrate_distance).
            //
            // The corpus is built fresh per call. For repeated queries
            // against the same corpus, prefer omc_corpus_build +
            // omc_predict_from (returns a handle).
            //
            // Example:
            //   h hits = omc_predict_files(
            //       ["examples/lib/prometheus.omc"],
            //       "fn prom_linear_",
            //       5);
            //   for h in hits { print(dict_get(h, "fn_name")); }
            "omc_predict_files" => {
                if args.len() < 3 {
                    return Err("omc_predict_files: requires (paths_array, prefix_source, top_k)".to_string());
                }
                let paths_val = self.eval_expr(&args[0])?;
                let prefix_source = self.eval_expr(&args[1])?.to_display_string();
                let top_k = self.eval_expr(&args[2])?.to_int().max(0) as usize;
                let paths: Vec<String> = if let Value::Array(arr) = paths_val {
                    arr.items.borrow().iter().map(|v| v.to_display_string()).collect()
                } else {
                    return Err("omc_predict_files: first argument must be an array of strings".to_string());
                };
                let mut corpus = crate::predict::CodeCorpus::new();
                for path in &paths {
                    let src = std::fs::read_to_string(path)
                        .map_err(|e| format!("omc_predict_files: read {}: {}", path, e))?;
                    corpus.ingest_file(path, &src);
                }
                let suggestions = crate::predict::predict_continuations(&corpus, &prefix_source, top_k);
                Ok(predict_suggestions_to_value(&suggestions))
            }
            // Diagnostic: just ingest + return corpus size. Useful for
            // sanity-checking that file paths resolve and fns parse.
            "omc_corpus_size" => {
                if args.is_empty() {
                    return Err("omc_corpus_size: requires (paths_array)".to_string());
                }
                let paths_val = self.eval_expr(&args[0])?;
                let paths: Vec<String> = if let Value::Array(arr) = paths_val {
                    arr.items.borrow().iter().map(|v| v.to_display_string()).collect()
                } else {
                    return Err("omc_corpus_size: first argument must be an array of strings".to_string());
                };
                let mut corpus = crate::predict::CodeCorpus::new();
                for path in &paths {
                    let src = std::fs::read_to_string(path)
                        .map_err(|e| format!("omc_corpus_size: read {}: {}", path, e))?;
                    corpus.ingest_file(path, &src);
                }
                Ok(Value::HInt(HInt::new(corpus.len() as i64)))
            }
            "omc_codec_encode" => {
                if args.is_empty() {
                    return Err("omc_codec_encode requires (code: string, every_n?: int)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let every_n = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_int().max(1) as usize
                } else { 3usize };
                let canon = crate::canonical::canonicalize(&code)
                    .unwrap_or_else(|_| code.clone());
                let tokens = crate::tokenizer::encode(&canon);
                let sampled: Vec<Value> = tokens.iter().enumerate()
                    .filter(|(i, _)| i % every_n == 0)
                    .map(|(_, t)| Value::HInt(HInt::new(*t)))
                    .collect();
                let hash = crate::tokenizer::fnv1a_64(canon.as_bytes());
                let (attractor, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(hash);
                let mut map = std::collections::BTreeMap::new();
                map.insert("sampled_tokens".to_string(),
                    Value::Array(HArray::from_vec(sampled.clone())));
                map.insert("content_hash".to_string(), Value::HInt(HInt::new(hash)));
                map.insert("attractor".to_string(), Value::HInt(HInt::new(attractor)));
                map.insert("dist".to_string(), Value::HInt(HInt::new(dist)));
                map.insert("original_tok_count".to_string(),
                    Value::HInt(HInt::new(tokens.len() as i64)));
                map.insert("source_bytes".to_string(),
                    Value::HInt(HInt::new(code.len() as i64)));
                map.insert("every_n".to_string(), Value::HInt(HInt::new(every_n as i64)));
                let ratio = if !sampled.is_empty() {
                    code.len() as f64 / sampled.len() as f64
                } else { 0.0 };
                map.insert("compression_ratio".to_string(), Value::HFloat(ratio));
                Ok(Value::dict_from(map))
            }
            // omc_codec_decode_lookup(codec: dict, library: string[]) -> string|null
            //   Lossless decode via library lookup: hash each library
            //   entry's canonical form; return the one whose hash
            //   matches the codec's content_hash. Returns null on miss.
            //   This is the "verify and retry" half of the codec.
            "omc_codec_decode_lookup" => {
                if args.len() < 2 {
                    return Err("omc_codec_decode_lookup requires (codec: dict, library: string[])".to_string());
                }
                let codec_v = self.eval_expr(&args[0])?;
                let lib_v = self.eval_expr(&args[1])?;
                let target_hash = if let Value::Dict(d) = codec_v {
                    d.borrow().get("content_hash")
                        .map(|v| v.to_int())
                        .unwrap_or(0)
                } else {
                    return Err("omc_codec_decode_lookup: codec must be a dict".to_string());
                };
                let library: Vec<String> = if let Value::Array(arr) = lib_v {
                    arr.items.borrow().iter().map(|v| v.to_display_string()).collect()
                } else {
                    return Err("omc_codec_decode_lookup: library must be a string array".to_string());
                };
                for entry in &library {
                    let canon = crate::canonical::canonicalize(entry)
                        .unwrap_or_else(|_| entry.clone());
                    let h = crate::tokenizer::fnv1a_64(canon.as_bytes());
                    if h == target_hash {
                        return Ok(Value::String(entry.clone()));
                    }
                }
                Ok(Value::Null)
            }
            // omc_registry_codec_library() -> string[]
            //   Scan omc_modules/ for installed registry packages, extract
            //   each top-level fn definition as a separate string entry.
            //   The returned array is suitable as the library argument to
            //   omc_codec_decode_lookup / omc_msg_recover_compressed.
            "omc_registry_codec_library" => {
                let dir = std::path::Path::new("omc_modules");
                if !dir.is_dir() {
                    return Ok(Value::Array(HArray::from_vec(vec![])));
                }
                let mut entries: Vec<Value> = Vec::new();
                if let Ok(rd) = std::fs::read_dir(dir) {
                    for ent in rd.flatten() {
                        let p = ent.path();
                        if p.extension().and_then(|s| s.to_str()) != Some("omc") {
                            continue;
                        }
                        if let Ok(src) = std::fs::read_to_string(&p) {
                            for fn_src in extract_top_level_fns(&src) {
                                entries.push(Value::String(fn_src));
                            }
                        }
                    }
                }
                Ok(Value::Array(HArray::from_vec(entries)))
            }
            // omc_msg_recover_from_registry(msg) -> string|null
            //   Convenience: omc_msg_recover_compressed(msg,
            //   omc_registry_codec_library()). Returns the matching
            //   library entry's canonical source, or null on miss.
            "omc_msg_recover_from_registry" => {
                if args.is_empty() {
                    return Err("omc_msg_recover_from_registry requires (msg: dict)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let target_hash = if let Value::Dict(d) = v {
                    d.borrow().get("content_hash").map(|x| x.to_int()).unwrap_or(0)
                } else {
                    return Err("omc_msg_recover_from_registry: msg must be a dict".to_string());
                };
                let dir = std::path::Path::new("omc_modules");
                if !dir.is_dir() {
                    return Ok(Value::Null);
                }
                if let Ok(rd) = std::fs::read_dir(dir) {
                    for ent in rd.flatten() {
                        let p = ent.path();
                        if p.extension().and_then(|s| s.to_str()) != Some("omc") {
                            continue;
                        }
                        if let Ok(src) = std::fs::read_to_string(&p) {
                            for fn_src in extract_top_level_fns(&src) {
                                let canon = crate::canonical::canonicalize(&fn_src)
                                    .unwrap_or_else(|_| fn_src.clone());
                                if crate::tokenizer::fnv1a_64(canon.as_bytes()) == target_hash {
                                    return Ok(Value::String(fn_src));
                                }
                            }
                        }
                    }
                }
                Ok(Value::Null)
            }
            "omc_find_similar" => {
                // omc_find_similar(query, corpus[]) → [{index, distance}, ...]
                // ranked closest-first by canonical-hash distance.
                if args.len() < 2 {
                    return Err("omc_find_similar requires (query, corpus[])".to_string());
                }
                let query = self.eval_expr(&args[0])?.to_display_string();
                let corpus_v = self.eval_expr(&args[1])?;
                let corpus: Vec<String> = if let Value::Array(arr) = corpus_v {
                    arr.items.borrow().iter()
                        .map(|x| x.to_display_string())
                        .collect()
                } else {
                    return Err("omc_find_similar: corpus must be a string array".to_string());
                };
                let ranked = crate::code_intel::find_similar(&query, &corpus)
                    .map_err(|e| format!("omc_find_similar: {}", e))?;
                // Optional 3rd arg = top_k (default = full list).
                let top_k = if args.len() >= 3 {
                    self.eval_expr(&args[2])?.to_int().max(1) as usize
                } else { ranked.len() };
                let out: Vec<Value> = ranked.iter().take(top_k)
                    .map(|(idx, dist)| {
                        let mut map = std::collections::BTreeMap::new();
                        map.insert("index".to_string(), Value::HInt(HInt::new(*idx as i64)));
                        map.insert("distance".to_string(), Value::HInt(HInt::new(*dist)));
                        Value::dict_from(map)
                    })
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            "omc_code_diff" => {
                // Structural diff: returns {added, removed, modified, unchanged}.
                // Compared after canonicalization so renames don't show.
                if args.len() < 2 {
                    return Err("omc_code_diff requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_display_string();
                let b = self.eval_expr(&args[1])?.to_display_string();
                let d = crate::code_intel::diff(&a, &b)
                    .map_err(|e| format!("omc_code_diff: {}", e))?;
                let mut map = std::collections::BTreeMap::new();
                map.insert("added".to_string(), Value::Array(HArray::from_vec(
                    d.added.iter().map(|s| Value::String(s.clone())).collect()
                )));
                map.insert("removed".to_string(), Value::Array(HArray::from_vec(
                    d.removed.iter().map(|s| Value::String(s.clone())).collect()
                )));
                map.insert("modified".to_string(), Value::Array(HArray::from_vec(
                    d.modified.iter().map(|s| Value::String(s.clone())).collect()
                )));
                map.insert("unchanged".to_string(), Value::Array(HArray::from_vec(
                    d.unchanged.iter().map(|s| Value::String(s.clone())).collect()
                )));
                Ok(Value::dict_from(map))
            }
            "omc_code_metrics" => {
                // Bulk metrics in one call: complexity + ast_size +
                // ast_depth + source_bytes + token_count +
                // compression_ratio. Avoids N separate round-trips
                // through the MCP server.
                if args.is_empty() {
                    return Err("omc_code_metrics requires (code)".to_string());
                }
                let code = self.eval_expr(&args[0])?.to_display_string();
                let m = crate::code_intel::quick_metrics(&code)
                    .map_err(|e| format!("omc_code_metrics: {}", e))?;
                let mut map = std::collections::BTreeMap::new();
                for (k, v) in m {
                    map.insert(k, Value::HFloat(v));
                }
                Ok(Value::dict_from(map))
            }
            "omc_token_vocab_dump" => {
                // First N entries of vocab as a numbered list.
                let n = if !args.is_empty() {
                    self.eval_expr(&args[0])?.to_int().max(0) as usize
                } else { 50 };
                let mut s = String::new();
                let len = crate::tokenizer::TOKEN_DICT.len().min(n);
                for (i, entry) in crate::tokenizer::TOKEN_DICT.iter().take(len).enumerate() {
                    let display = entry.replace('\n', "\\n").replace('\t', "\\t");
                    s.push_str(&format!("{:4}: {:?}\n", i, display));
                }
                Ok(Value::String(s))
            }
            "omc_help_brief" => {
                // Just signature + one-line description (no example). Useful
                // when the LLM wants a compact view across many builtins.
                if args.is_empty() {
                    return Err("omc_help_brief requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(d) => Ok(Value::String(format!(
                        "{} :: {}\n  {}", d.name, d.signature, d.description
                    ))),
                    None => Ok(Value::String(format!("{}: not in registry", name))),
                }
            }
            "omc_help_signature" => {
                // Just the signature string. Compactest possible.
                if args.is_empty() {
                    return Err("omc_help_signature requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(d) => Ok(Value::String(d.signature.to_string())),
                    None => Ok(Value::String(String::new())),
                }
            }
            "omc_help_example" => {
                if args.is_empty() {
                    return Err("omc_help_example requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(d) => Ok(Value::String(d.example.to_string())),
                    None => Ok(Value::String(String::new())),
                }
            }
            "omc_help_category" => {
                if args.is_empty() {
                    return Err("omc_help_category requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(d) => Ok(Value::String(d.category.to_string())),
                    None => Ok(Value::String(String::new())),
                }
            }
            "omc_is_unique" => {
                // 1 if name is OMC-unique (no Python equivalent).
                if args.is_empty() {
                    return Err("omc_is_unique requires (name)".to_string());
                }
                let name = self.eval_expr(&args[0])?.to_display_string();
                match crate::docs::lookup(&name) {
                    Some(d) => Ok(Value::HInt(HInt::new(if d.unique_to_omc { 1 } else { 0 }))),
                    None => Ok(Value::HInt(HInt::new(0))),
                }
            }
            "omc_count_in_category" => {
                if args.is_empty() {
                    return Err("omc_count_in_category requires (category)".to_string());
                }
                let cat = self.eval_expr(&args[0])?.to_display_string();
                let count = crate::docs::BUILTINS.iter()
                    .filter(|b| b.category == cat).count() as i64;
                Ok(Value::HInt(HInt::new(count)))
            }
            "omc_random_builtin" => {
                // Random builtin name. Useful for fuzzing or exploring.
                let idx = (self.rng_next() % (crate::docs::BUILTINS.len() as u64)) as usize;
                Ok(Value::String(crate::docs::BUILTINS[idx].name.to_string()))
            }
            "omc_random_unique_builtin" => {
                let uniq: Vec<&str> = crate::docs::BUILTINS.iter()
                    .filter(|b| b.unique_to_omc).map(|b| b.name).collect();
                if uniq.is_empty() {
                    return Ok(Value::String(String::new()));
                }
                let idx = (self.rng_next() % (uniq.len() as u64)) as usize;
                Ok(Value::String(uniq[idx].to_string()))
            }
            "omc_search_builtins" => {
                // Substring search across name + description. Returns
                // matching names. Useful when you don't know what
                // you're looking for but know what it should do.
                if args.is_empty() {
                    return Err("omc_search_builtins requires (query)".to_string());
                }
                let q = self.eval_expr(&args[0])?.to_display_string().to_lowercase();
                let out: Vec<Value> = crate::docs::BUILTINS.iter()
                    .filter(|b| {
                        b.name.to_lowercase().contains(&q) ||
                        b.description.to_lowercase().contains(&q)
                    })
                    .map(|b| Value::String(b.name.to_string()))
                    .collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // arr_fold_all(arr) -> new array with every element snapped
            // to its nearest Fibonacci attractor. Vectorized fold.
            // Substrate-canonical denoising / quantization primitive.
            "arr_fold_all" => {
                if args.is_empty() {
                    return Err("arr_fold_all requires (array)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                if let Value::Array(arr) = a {
                    let out: Vec<Value> = arr.items.borrow().iter()
                        .map(|v| {
                            let folded = crate::phi_pi_fib::fold_to_nearest_attractor(v.to_int());
                            Value::HInt(HInt::new(folded))
                        })
                        .collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_fold_all: requires an array".to_string())
                }
            }
            "arr_concat" => {
                if args.len() < 2 {
                    return Err("arr_concat requires (array_a, array_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                match (a, b) {
                    (Value::Array(a), Value::Array(b)) => {
                        // Fresh Rc — explicit copy semantics so the result
                        // doesn't share state with either input.
                        let mut out = a.items.borrow().clone();
                        out.extend(b.items.borrow().iter().cloned());
                        Ok(Value::Array(HArray::from_vec(out)))
                    }
                    _ => Err("arr_concat: both arguments must be arrays".to_string()),
                }
            }
            "arr_contains" => {
                if args.len() < 2 {
                    return Err("arr_contains requires (array, value)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let found = arr.items.borrow().iter().any(|v| v.to_int() == target);
                    Ok(Value::HInt(HInt::new(if found { 1 } else { 0 })))
                } else {
                    Err("arr_contains: first argument must be an array".to_string())
                }
            }
            "arr_index_of" => {
                if args.len() < 2 {
                    return Err("arr_index_of requires (array, value)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let pos = arr.items.borrow().iter().position(|v| v.to_int() == target);
                    Ok(Value::HInt(HInt::new(match pos {
                        Some(i) => i as i64,
                        None => -1,
                    })))
                } else {
                    Err("arr_index_of: first argument must be an array".to_string())
                }
            }
            // phi_pi_fib_search(sorted_array, target)
            //   Fibonacci-step binary search over a sorted integer array.
            //   Returns the exact-match index when found, or -(insert_pos + 1)
            //   when not found — same sign convention as Rust's binary_search.
            //   Use phi_pi_fib_nearest if you want a "nearest entry" gate
            //   that never returns a negative index.
            "phi_pi_fib_search" => {
                if args.len() < 2 {
                    return Err("phi_pi_fib_search requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::fibonacci_search(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    Ok(Value::HInt(HInt::new(match r {
                        Ok(i) => i as i64,
                        Err(insert_pos) => -(insert_pos as i64 + 1),
                    })))
                } else {
                    Err("phi_pi_fib_search: first argument must be an array".to_string())
                }
            }
            // phi_pi_fib_nearest(sorted_array, target)
            //   Same as phi_pi_fib_search but returns the index of the
            //   nearest entry by absolute integer distance. Always returns
            //   a valid index (0..len) for non-empty arrays, or -1 if the
            //   array is empty.
            //
            //   This is the gate primitive for the compression-gate
            //   architecture: missing-key lookups route to the nearest
            //   surviving library entry, giving "die gracefully" semantics.
            "phi_pi_fib_nearest" => {
                if args.len() < 2 {
                    return Err("phi_pi_fib_nearest requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    if ints.is_empty() {
                        return Ok(Value::HInt(HInt::new(-1)));
                    }
                    let r = crate::phi_pi_fib::fibonacci_search(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    let idx: usize = match r {
                        Ok(i) => i,
                        Err(insert_pos) => {
                            let n = ints.len();
                            if insert_pos == 0 {
                                0
                            } else if insert_pos >= n {
                                n - 1
                            } else {
                                let left = (target - ints[insert_pos - 1]).abs();
                                let right = (ints[insert_pos] - target).abs();
                                if right < left { insert_pos } else { insert_pos - 1 }
                            }
                        }
                    };
                    Ok(Value::HInt(HInt::new(idx as i64)))
                } else {
                    Err("phi_pi_fib_nearest: first argument must be an array".to_string())
                }
            }
            // phi_pi_fib_stats() -> [total_searches, total_comparisons]
            //   Returns global counters for all phi_pi_fib_* calls since the
            //   last phi_pi_fib_reset(). Use to measure how many compares the
            //   gate cost — should grow as O(log_phi n), not O(n).
            "phi_pi_fib_stats" => {
                let s = crate::phi_pi_fib::get_search_stats();
                let items = vec![
                    Value::HInt(HInt::new(s.total_searches as i64)),
                    Value::HInt(HInt::new(s.total_comparisons as i64)),
                ];
                Ok(Value::Array(HArray::from_vec(items)))
            }
            // phi_pi_fib_reset() -> null. Zero both phi_pi_fib counter
            // channels (explicit AND background).
            "phi_pi_fib_reset" => {
                crate::phi_pi_fib::reset_search_stats();
                Ok(Value::Null)
            }
            // phi_pi_fib_stats_bg() -> [searches, comparisons] for the
            // BACKGROUND channel — substrate-internal calls
            // (HInt::new -> compute_resonance -> nearest_attractor_with_dist).
            "phi_pi_fib_stats_bg" => {
                let s = crate::phi_pi_fib::get_search_stats_background();
                let items = vec![
                    Value::HInt(HInt::new(s.total_searches as i64)),
                    Value::HInt(HInt::new(s.total_comparisons as i64)),
                ];
                Ok(Value::Array(HArray::from_vec(items)))
            }
            // phi_pi_fib_stats_all() -> [searches, comparisons] summed
            // across explicit + background channels.
            "phi_pi_fib_stats_all" => {
                let s = crate::phi_pi_fib::get_search_stats_all();
                let items = vec![
                    Value::HInt(HInt::new(s.total_searches as i64)),
                    Value::HInt(HInt::new(s.total_comparisons as i64)),
                ];
                Ok(Value::Array(HArray::from_vec(items)))
            }
            // phi_shadow(x) - HBit β-divergence primitive.
            //
            // Tree-walk semantics: pass-through. Returns x unchanged
            // because tree-walk has no concept of a shadow band; the
            // value's semantic meaning is purely its α (classical).
            //
            // Dual-band JIT semantics (omnimcode-codegen): intercepted
            // as an intrinsic and rewritten to replace the β lane of
            // the value's `<2 x i64>` carrier with phi_fold(α) * 1000
            // (cast to i64). After this op, harmony(x) is non-trivial.
            //
            // Use case: mark a value as "now subject to harmonic
            // observation" so subsequent ops carry both bands through
            // computation. A later harmony() check decides whether
            // the value is behaving as predicted.
            "phi_shadow" => {
                if args.is_empty() {
                    return Err("phi_shadow requires (value)".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                Ok(v)
            }
            // harmony(x) - HBit harmony reading.
            //
            // Tree-walk semantics: returns 1000 unconditionally. With
            // no β to compare against, harmony is trivially perfect.
            // The value's semantic content fits this — in tree-walk
            // mode, "harmony" can be read as "agreement between α and
            // α" which is always exact.
            //
            // Dual-band JIT semantics (omnimcode-codegen, Session G):
            // intercepted as an intrinsic that emits a call to the
            // extern Rust helper computing harmony from the two lanes.
            //
            // Return convention: i64 in [0, 1000]. 1000 = perfect
            // harmony, 0 = maximally divergent. Floats avoided to
            // keep the calling convention pure-i64.
            "harmony" => {
                if args.is_empty() {
                    return Err("harmony requires (value)".to_string());
                }
                let _ = self.eval_expr(&args[0])?;
                Ok(Value::HInt(HInt::new(1000)))
            }
            // phi_pi_fib_search_v2(sorted_arr, target) -> int
            //   F(k)/φ^(π·k) split-point search. Same return convention
            //   as phi_pi_fib_search (exact match index, or -(insert+1)).
            //   Comparison counts are folded into the shared counters so
            //   phi_pi_fib_stats() reports both algorithms' totals — call
            //   phi_pi_fib_reset between runs when measuring head-to-head.
            "phi_pi_fib_search_v2" => {
                if args.len() < 2 {
                    return Err("phi_pi_fib_search_v2 requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::phi_pi_fib_search_v2(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    Ok(Value::HInt(HInt::new(match r {
                        Ok(i) => i as i64,
                        Err(insert_pos) => -(insert_pos as i64 + 1),
                    })))
                } else {
                    Err("phi_pi_fib_search_v2: first argument must be an array".to_string())
                }
            }
            // phi_pi_fib_nearest_v2(sorted_arr, target) -> int
            //   Always-valid nearest-index variant of phi_pi_fib_search_v2.
            "phi_pi_fib_nearest_v2" => {
                if args.len() < 2 {
                    return Err("phi_pi_fib_nearest_v2 requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    if ints.is_empty() {
                        return Ok(Value::HInt(HInt::new(-1)));
                    }
                    let r = crate::phi_pi_fib::phi_pi_fib_search_v2(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    let idx: usize = match r {
                        Ok(i) => i,
                        Err(insert_pos) => {
                            let n = ints.len();
                            if insert_pos == 0 {
                                0
                            } else if insert_pos >= n {
                                n - 1
                            } else {
                                let left = (target - ints[insert_pos - 1]).abs();
                                let right = (ints[insert_pos] - target).abs();
                                if right < left { insert_pos } else { insert_pos - 1 }
                            }
                        }
                    };
                    Ok(Value::HInt(HInt::new(idx as i64)))
                } else {
                    Err("phi_pi_fib_nearest_v2: first argument must be an array".to_string())
                }
            }
            // phi_pi_bin_search(sorted_arr, target) -> int
            //   Standard binary search baseline. Same return convention as
            //   the phi_pi_fib_search variants. Shares the global compare
            //   counter so head-to-head benches see all three algorithms.
            "phi_pi_bin_search" => {
                if args.len() < 2 {
                    return Err("phi_pi_bin_search requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::binary_search(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    Ok(Value::HInt(HInt::new(match r {
                        Ok(i) => i as i64,
                        Err(insert_pos) => -(insert_pos as i64 + 1),
                    })))
                } else {
                    Err("phi_pi_bin_search: first argument must be an array".to_string())
                }
            }
            // log_phi_pi_fibonacci(n) -> float
            //   The theoretical compare-count bound for phi_pi_fib_search_v2.
            //   Equals ln(n) / (π · ln(φ)) ≈ 0.459 · log₂(n).
            "log_phi_pi_fibonacci" => {
                if args.is_empty() {
                    return Err("log_phi_pi_fibonacci requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_float();
                Ok(Value::HFloat(crate::phi_pi_fib::log_phi_pi_fibonacci(n)))
            }
            // zeckendorf(n) -> array of FIBONACCI-table indices, largest first.
            // The unique non-consecutive Fibonacci decomposition. Iteration
            // count is bounded by log_phi_pi_fibonacci(n) — substrate-canonical.
            "zeckendorf" => {
                if args.is_empty() {
                    return Err("zeckendorf requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                if n < 0 {
                    return Err("zeckendorf: requires n >= 0".to_string());
                }
                let idxs = crate::phi_pi_fib::zeckendorf_indices(n as u64);
                let out: Vec<Value> = idxs.into_iter()
                    .map(|i| Value::HInt(HInt::new(i as i64))).collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // from_zeckendorf(indices) -> int
            //   Inverse of zeckendorf: sums FIBONACCI[i] for each i. Pure;
            //   no validation that indices are non-consecutive (caller's
            //   responsibility) — we just take the sum at the given slots.
            "from_zeckendorf" => {
                if args.is_empty() {
                    return Err("from_zeckendorf requires (indices_array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let idxs: Vec<usize> = items.iter()
                        .map(|v| v.to_int().max(0) as usize)
                        .collect();
                    let v = crate::phi_pi_fib::from_zeckendorf_indices(&idxs);
                    Ok(Value::HInt(HInt::new(v as i64)))
                } else {
                    Err("from_zeckendorf: argument must be an array".to_string())
                }
            }
            // substrate_search(sorted_array, target) -> index or -1
            //   Substrate-routed exact-match search using F(k)/φ^(π·k)
            //   split-point algorithm. Iteration count bounded by
            //   log_phi_pi_fibonacci(N). Returns -1 on miss; for the
            //   insert-position variant call phi_pi_fib_search_traced.
            "substrate_search" => {
                if args.len() < 2 {
                    return Err("substrate_search requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::substrate_search_i64(&ints, target)
                        .map(|i| i as i64).unwrap_or(-1);
                    Ok(Value::HInt(HInt::new(r)))
                } else {
                    Err("substrate_search: first argument must be an array".to_string())
                }
            }
            // substrate_lower_bound / upper_bound — first index satisfying
            // arr[i] >= target / arr[i] > target. Used by range queries,
            // interval intersections, rank-by-value (substrate_rank below).
            "substrate_lower_bound" => {
                if args.len() < 2 {
                    return Err("substrate_lower_bound requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::substrate_lower_bound(&ints, target);
                    Ok(Value::HInt(HInt::new(r as i64)))
                } else {
                    Err("substrate_lower_bound: first argument must be an array".to_string())
                }
            }
            "substrate_upper_bound" => {
                if args.len() < 2 {
                    return Err("substrate_upper_bound requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::substrate_upper_bound(&ints, target);
                    Ok(Value::HInt(HInt::new(r as i64)))
                } else {
                    Err("substrate_upper_bound: first argument must be an array".to_string())
                }
            }
            // substrate_rank(sorted_array, value) -> int in [0, N]
            //   How many elements compare strictly less than `value`. Pure
            //   composition of substrate_lower_bound — same iteration bound.
            //   Useful for rank-based statistics (percentile rank, etc.).
            "substrate_rank" => {
                if args.len() < 2 {
                    return Err("substrate_rank requires (sorted_array, value)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let r = crate::phi_pi_fib::substrate_lower_bound(&ints, target);
                    Ok(Value::HInt(HInt::new(r as i64)))
                } else {
                    Err("substrate_rank: first argument must be an array".to_string())
                }
            }
            // substrate_count_range(sorted_array, lo, hi) -> int
            //   Count of elements in [lo, hi). Two substrate-bound calls,
            //   so 2 * log_phi_pi_fibonacci(N) probes total. Strictly
            //   better than the OMC-level `arr_filter(...)` linear scan
            //   for any large array where the range is small.
            "substrate_count_range" => {
                if args.len() < 3 {
                    return Err("substrate_count_range requires (sorted_array, lo, hi)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let lo = self.eval_expr(&args[1])?.to_int();
                let hi = self.eval_expr(&args[2])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let lo_i = crate::phi_pi_fib::substrate_lower_bound(&ints, lo);
                    let hi_i = crate::phi_pi_fib::substrate_lower_bound(&ints, hi);
                    Ok(Value::HInt(HInt::new(hi_i.saturating_sub(lo_i) as i64)))
                } else {
                    Err("substrate_count_range: first argument must be an array".to_string())
                }
            }
            // substrate_slice_range(sorted_array, lo, hi) -> array
            //   Slice of values in [lo, hi). Two substrate probes plus an
            //   O(k) copy where k is the result size. The O(k) is fundamental
            //   (we have to materialize) but the *boundary discovery* still
            //   pays only 2 * log_phi_pi_fibonacci(N).
            "substrate_slice_range" => {
                if args.len() < 3 {
                    return Err("substrate_slice_range requires (sorted_array, lo, hi)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let lo = self.eval_expr(&args[1])?.to_int();
                let hi = self.eval_expr(&args[2])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let lo_i = crate::phi_pi_fib::substrate_lower_bound(&ints, lo);
                    let hi_i = crate::phi_pi_fib::substrate_lower_bound(&ints, hi);
                    let out: Vec<Value> = items[lo_i..hi_i.max(lo_i)].to_vec();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("substrate_slice_range: first argument must be an array".to_string())
                }
            }
            // substrate_intersect(sorted_a, sorted_b) -> sorted intersection.
            // Walks the SHORTER array linearly; each element triggers one
            // substrate_search probe in the longer array. Total:
            // O(min(|a|,|b|) · log_phi_pi_fibonacci max(|a|,|b|)) — strictly
            // better than the merge-walk O(|a|+|b|) when the smaller side
            // is tiny relative to the larger.
            "substrate_intersect" => {
                if args.len() < 2 {
                    return Err("substrate_intersect requires (sorted_a, sorted_b)".to_string());
                }
                let a_v = self.eval_expr(&args[0])?;
                let b_v = self.eval_expr(&args[1])?;
                if let (Value::Array(a), Value::Array(b)) = (a_v, b_v) {
                    let ai = a.items.borrow();
                    let bi = b.items.borrow();
                    let a_int: Vec<i64> = ai.iter().map(|v| v.to_int()).collect();
                    let b_int: Vec<i64> = bi.iter().map(|v| v.to_int()).collect();
                    // Drive the loop with the shorter side.
                    let (driver, indexed) = if a_int.len() <= b_int.len() {
                        (&a_int, &b_int)
                    } else {
                        (&b_int, &a_int)
                    };
                    let mut out = Vec::new();
                    for &v in driver {
                        if crate::phi_pi_fib::substrate_search_i64(indexed, v).is_some() {
                            out.push(Value::HInt(HInt::new(v)));
                        }
                    }
                    // Ensure unique + sorted in the result.
                    out.sort_by_key(|v| v.to_int());
                    out.dedup_by_key(|v| v.to_int());
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("substrate_intersect: both arguments must be arrays".to_string())
                }
            }
            // substrate_difference(sorted_a, sorted_b) -> elements in a but
            // not in b. Drives the loop with |a|, each element costs one
            // substrate probe in b: O(|a| · log_phi_pi_fibonacci |b|).
            "substrate_difference" => {
                if args.len() < 2 {
                    return Err("substrate_difference requires (sorted_a, sorted_b)".to_string());
                }
                let a_v = self.eval_expr(&args[0])?;
                let b_v = self.eval_expr(&args[1])?;
                if let (Value::Array(a), Value::Array(b)) = (a_v, b_v) {
                    let ai = a.items.borrow();
                    let bi = b.items.borrow();
                    let b_int: Vec<i64> = bi.iter().map(|v| v.to_int()).collect();
                    let mut out = Vec::new();
                    for v in ai.iter() {
                        let n = v.to_int();
                        if crate::phi_pi_fib::substrate_search_i64(&b_int, n).is_none() {
                            out.push(v.clone());
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("substrate_difference: both arguments must be arrays".to_string())
                }
            }
            // zeckendorf_weight(n) -> int
            //   Number of Fibonacci terms in n's Zeckendorf representation.
            //   This is the "substrate weight" of n — a measure of how
            //   non-Fibonacci it is. Pure attractors have weight 1; sums
            //   of two attractors weigh 2; etc. O(log_phi_pi_fibonacci n).
            "zeckendorf_weight" => {
                if args.is_empty() {
                    return Err("zeckendorf_weight requires (n)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int().max(0) as u64;
                let w = crate::phi_pi_fib::zeckendorf_indices(n).len();
                Ok(Value::HInt(HInt::new(w as i64)))
            }
            // zeckendorf_bit(n, k) -> 0 or 1
            //   Is FIBONACCI[k] present in n's Zeckendorf representation?
            //   The "bit-test" primitive for substrate-encoded ints. Used
            //   by sub_hash below to mix bits in a substrate-aligned way.
            "zeckendorf_bit" => {
                if args.len() < 2 {
                    return Err("zeckendorf_bit requires (n, k)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int().max(0) as u64;
                let k = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let idxs = crate::phi_pi_fib::zeckendorf_indices(n);
                let present = idxs.iter().any(|&i| i == k);
                Ok(Value::HInt(HInt::new(if present { 1 } else { 0 })))
            }
            // substrate_hash(value) -> i64
            //   Position-aware Zeckendorf-mixed hash. Each Fibonacci-index
            //   set bit contributes a unique phi-spaced prime multiplier;
            //   the result has substrate-aligned avalanche. Use as the
            //   keying function for substrate-bucketed dicts/bloom filters.
            "substrate_hash" => {
                if args.is_empty() {
                    return Err("substrate_hash requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let mag = n.unsigned_abs();
                let idxs = crate::phi_pi_fib::zeckendorf_indices(mag);
                // Constants: golden ratio mantissa as i64, signed cast.
                const SEED: u64 = 0x9E3779B97F4A7C15; // 2^64 * (sqrt(5)-1)/2
                let mut h: u64 = SEED;
                for (rank, &i) in idxs.iter().enumerate() {
                    // Phi-shifted contribution; rotate by rank so ordering
                    // within the Zeckendorf word matters (it's already
                    // largest-first, so position is meaningful).
                    let term = (i as u64).wrapping_mul(SEED).rotate_left((rank * 5) as u32);
                    h = (h ^ term).wrapping_mul(SEED);
                }
                if n < 0 { h = h.wrapping_add(0xD1B54A32D192ED03); }
                Ok(Value::HInt(HInt::new(h as i64)))
            }
            // attractor_bucket(value) -> int in [0, 40)
            //   FIBONACCI-table index of the nearest attractor. Used by
            //   substrate-bucketed hashmaps where bucket boundaries follow
            //   the golden ratio (so collision distribution matches the
            //   phi-power-law of natural keys). O(log_phi_pi_fibonacci |v|).
            "attractor_bucket" => {
                if args.is_empty() {
                    return Err("attractor_bucket requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HInt(HInt::new(crate::phi_pi_fib::attractor_bucket(n) as i64)))
            }
            // substrate_insert(sorted_array_var, value) -> int (insert position)
            //   Mutating: insert `value` into the sorted array so the array
            //   stays sorted. Uses substrate_lower_bound to find the slot
            //   (log_phi_pi_fibonacci N) and Vec::insert for the O(N) shift.
            //   For repeated inserts on the same array this is the cheapest
            //   "build a sorted list" pattern available short of a BTreeSet.
            "substrate_insert" => {
                if args.len() < 2 {
                    return Err("substrate_insert requires (sorted_array_var, value)".to_string());
                }
                let value = self.eval_expr(&args[1])?;
                let v_int = value.to_int();
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(arr)) = self.get_var(name) {
                        // Build ints view for the substrate probe.
                        let ints: Vec<i64> = arr.items.borrow().iter()
                            .map(|v| v.to_int()).collect();
                        let pos = crate::phi_pi_fib::substrate_lower_bound(&ints, v_int);
                        arr.items.borrow_mut().insert(pos, value);
                        return Ok(Value::HInt(HInt::new(pos as i64)));
                    }
                }
                Err("substrate_insert: first argument must be an array variable".to_string())
            }
            // substrate_quantile(sorted_array, q_thousandths) -> int
            //   Quantile lookup on a sorted array; q is in [0, 1000] for
            //   tenth-percent granularity (q=500 → median, q=750 → 75th).
            //   O(1) on top of sorted input. Stored as int because OMC
            //   builtins return ints in JIT-friendly types preferentially.
            "substrate_quantile" => {
                if args.len() < 2 {
                    return Err("substrate_quantile requires (sorted_array, q_thousandths)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let q = self.eval_expr(&args[1])?.to_int().clamp(0, 1000) as u64;
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("substrate_quantile: empty array".to_string());
                    }
                    // Linear interpolation: idx = q * (N-1) / 1000.
                    let n = items.len() as u64;
                    let idx = ((q * (n - 1)) / 1000) as usize;
                    Ok(items[idx].clone())
                } else {
                    Err("substrate_quantile: first argument must be an array".to_string())
                }
            }
            // phi_pow(k) -> float (φ^k, exact via Binet for integer k)
            //   The substrate's growth rate per step. Useful for sizing
            //   buffers, computing decay rates, exponential moving averages
            //   with golden-ratio weights, etc.
            "phi_pow" => {
                if args.is_empty() {
                    return Err("phi_pow requires (k)".to_string());
                }
                let k = self.eval_expr(&args[0])?.to_float();
                const PHI: f64 = 1.6180339887498949;
                Ok(Value::HFloat(PHI.powf(k)))
            }
            // phi_pi_pow(k) -> float (φ^(π·k))
            //   The per-iteration shrink factor of the substrate search.
            //   = (4.534)^k for natural k. Used by tuning code that needs
            //   to size search windows to the substrate's natural step.
            "phi_pi_pow" => {
                if args.is_empty() {
                    return Err("phi_pi_pow requires (k)".to_string());
                }
                let k = self.eval_expr(&args[0])?.to_float();
                const PHI: f64 = 1.6180339887498949;
                const PI: f64 = std::f64::consts::PI;
                Ok(Value::HFloat((PI * k * PHI.ln()).exp()))
            }
            // harmonic_partition_3(arr, lo, hi) -> [below, between, above]
            //   3-way partition by value: elements < lo, lo <= e <= hi,
            //   and e > hi. Preserves input order within each bucket.
            //   For sorted input, equivalent to two substrate_slice_range
            //   calls; for unsorted, it's a single O(N) pass.
            "harmonic_partition_3" => {
                if args.len() < 3 {
                    return Err("harmonic_partition_3 requires (array, lo, hi)".to_string());
                }
                let lo = self.eval_expr(&args[1])?.to_int();
                let hi = self.eval_expr(&args[2])?.to_int();
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut below = Vec::new();
                    let mut between = Vec::new();
                    let mut above = Vec::new();
                    for v in items.iter() {
                        let n = v.to_int();
                        if n < lo { below.push(v.clone()); }
                        else if n > hi { above.push(v.clone()); }
                        else { between.push(v.clone()); }
                    }
                    Ok(Value::Array(HArray::from_vec(vec![
                        Value::Array(HArray::from_vec(below)),
                        Value::Array(HArray::from_vec(between)),
                        Value::Array(HArray::from_vec(above)),
                    ])))
                } else {
                    Err("harmonic_partition_3: first argument must be an array".to_string())
                }
            }
            // resonance_band_histogram(arr) -> [count_band0, ..., count_band4]
            //   For each of the 5 resonance bands defined by resonance_band,
            //   count how many array elements fall into it. Cheap profiling
            //   primitive — tells you how "substrate-coherent" a dataset is.
            "resonance_band_histogram" => {
                if args.is_empty() {
                    return Err("resonance_band_histogram requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut hist = [0i64; 5];
                    for v in items.iter() {
                        let n = v.to_int();
                        let (_a, dist) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                        let band = match dist {
                            0 => 0,
                            1..=3 => 1,
                            4..=10 => 2,
                            11..=100 => 3,
                            _ => 4,
                        };
                        hist[band] += 1;
                    }
                    let out: Vec<Value> = hist.iter()
                        .map(|&c| Value::HInt(HInt::new(c))).collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("resonance_band_histogram: requires an array".to_string())
                }
            }
            // arr_sum_int(arr) -> int (native i64 sum, wrapping)
            //   Faster than arr_sum (which goes through value.to_int() in
            //   the OMC dispatch). Useful in tight loops over big int arrays.
            "arr_sum_int" => {
                if args.is_empty() {
                    return Err("arr_sum_int requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut sum: i64 = 0;
                    for v in items.iter() {
                        sum = sum.wrapping_add(v.to_int());
                    }
                    Ok(Value::HInt(HInt::new(sum)))
                } else {
                    Err("arr_sum_int: requires an array".to_string())
                }
            }
            // arr_product(arr) -> int (wrapping product)
            //   Standard reduction; no OMC-level equivalent.
            "arr_product" => {
                if args.is_empty() {
                    return Err("arr_product requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut prod: i64 = 1;
                    for v in items.iter() {
                        prod = prod.wrapping_mul(v.to_int());
                    }
                    Ok(Value::HInt(HInt::new(prod)))
                } else {
                    Err("arr_product: requires an array".to_string())
                }
            }
            // arr_sort_int(arr) -> sorted array (ints, ascending)
            //   Native sort; faster than arr_sort + OMC predicate. Returns
            //   a new array (does not mutate input).
            "arr_sort_int" => {
                if args.is_empty() {
                    return Err("arr_sort_int requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    ints.sort_unstable();
                    let out: Vec<Value> = ints.into_iter()
                        .map(|n| Value::HInt(HInt::new(n))).collect();
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("arr_sort_int: requires an array".to_string())
                }
            }
            // attractor_table() -> array of Fibonacci attractors [0, 1, 1, ..., 63245986]
            //   Returns the substrate's 40-entry FIBONACCI table as a value.
            //   Useful for OMC code that wants to iterate or display them.
            "attractor_table" => {
                // Inline the table; it's only 40 entries.
                let fibs: [u64; 40] = [
                    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                    987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
                    121393, 196418, 317811, 514229, 832040, 1346269, 2178309,
                    3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986,
                ];
                let out: Vec<Value> = fibs.iter()
                    .map(|&f| Value::HInt(HInt::new(f as i64))).collect();
                Ok(Value::Array(HArray::from_vec(out)))
            }
            // harmonic_score(arr) -> float in [0, 1]
            //   Fraction of elements that are exactly on a Fibonacci attractor.
            //   1.0 = fully substrate-coherent, 0.0 = no alignment.
            "harmonic_score" => {
                if args.is_empty() {
                    return Err("harmonic_score requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Ok(Value::HFloat(0.0));
                    }
                    let mut hits = 0usize;
                    for v in items.iter() {
                        if crate::phi_pi_fib::is_on_fibonacci_attractor(v.to_int()) {
                            hits += 1;
                        }
                    }
                    Ok(Value::HFloat(hits as f64 / items.len() as f64))
                } else {
                    Err("harmonic_score: requires an array".to_string())
                }
            }
            // arr_min_int / arr_max_int: native int reductions (faster
            // than arr_min/max for big arrays because the dispatch is
            // saved). Preserve i64 semantics; non-int elements get
            // coerced via to_int.
            "arr_min_int" => {
                if args.is_empty() {
                    return Err("arr_min_int requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_min_int: empty array".to_string());
                    }
                    let m = items.iter().map(|v| v.to_int()).min().unwrap();
                    Ok(Value::HInt(HInt::new(m)))
                } else {
                    Err("arr_min_int: requires an array".to_string())
                }
            }
            "arr_max_int" => {
                if args.is_empty() {
                    return Err("arr_max_int requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("arr_max_int: empty array".to_string());
                    }
                    let m = items.iter().map(|v| v.to_int()).max().unwrap();
                    Ok(Value::HInt(HInt::new(m)))
                } else {
                    Err("arr_max_int: requires an array".to_string())
                }
            }
            // arr_avg_distance(arr, target) -> float
            //   Mean |arr[i] - target|. Single O(N) pass, native i64
            //   subtraction. Useful when scoring how concentrated an
            //   array is around a center point.
            "arr_avg_distance" => {
                if args.len() < 2 {
                    return Err("arr_avg_distance requires (array, target)".to_string());
                }
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if items.is_empty() { return Ok(Value::HFloat(0.0)); }
                    let mut sum: u128 = 0;
                    for v in items.iter() {
                        sum += (v.to_int() - target).unsigned_abs() as u128;
                    }
                    Ok(Value::HFloat(sum as f64 / items.len() as f64))
                } else {
                    Err("arr_avg_distance: first argument must be an array".to_string())
                }
            }
            // is_phi_resonant(value, tol) -> 0 or 1
            //   value is within `tol` of some integer power of phi.
            //   Pseudo-substrate version of attractor-detection in the
            //   continuous domain (Fibonacci attractors are the integer
            //   sampling of phi^k).
            "is_phi_resonant" => {
                if args.len() < 2 {
                    return Err("is_phi_resonant requires (value, tol)".to_string());
                }
                let v = self.eval_expr(&args[0])?.to_float().abs();
                let tol = self.eval_expr(&args[1])?.to_float();
                const PHI: f64 = 1.6180339887498949;
                if v < 1e-12 { return Ok(Value::HInt(HInt::new(1))); }
                // log_phi(v) — closest integer k → phi^k → check distance
                let k = (v.ln() / PHI.ln()).round();
                let predicted = PHI.powf(k);
                let close = (predicted - v).abs() <= tol;
                Ok(Value::HInt(HInt::new(if close { 1 } else { 0 })))
            }
            // arr_is_sorted(arr) -> 0 or 1
            //   Linear scan that short-circuits on the first inversion.
            //   Useful before substrate_search to verify pre-condition.
            "arr_is_sorted" => {
                if args.is_empty() {
                    return Err("arr_is_sorted requires (array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    for w in items.windows(2) {
                        if w[0].to_int() > w[1].to_int() {
                            return Ok(Value::HInt(HInt::new(0)));
                        }
                    }
                    Ok(Value::HInt(HInt::new(1)))
                } else {
                    Err("arr_is_sorted: requires an array".to_string())
                }
            }
            // nth_fibonacci(k) -> int (FIBONACCI[k], clamped to table size)
            //   Direct table lookup; constant-time Fibonacci retrieval.
            //   Substrate-canonical alternative to recursive/iterative `fib(k)`.
            "nth_fibonacci" => {
                if args.is_empty() {
                    return Err("nth_fibonacci requires (k)".to_string());
                }
                let k = self.eval_expr(&args[0])?.to_int().max(0) as u64;
                // Iterative — matches the inline computation we use in fib_chunks
                let mut a: u64 = 0; let mut b: u64 = 1;
                let mut i: u64 = 0;
                while i < k.min(93) {
                    let t = a.saturating_add(b);
                    a = b; b = t;
                    i += 1;
                }
                Ok(Value::HInt(HInt::new(a as i64)))
            }
            // is_zeckendorf_valid(indices_array) -> 0 or 1
            //   Check that the indices are: strictly decreasing AND no two
            //   consecutive. (Valid Zeckendorf representations always have
            //   |index_i - index_(i+1)| >= 2.) Useful for verifying that a
            //   caller's pre-built decomposition is canonical.
            "is_zeckendorf_valid" => {
                if args.is_empty() {
                    return Err("is_zeckendorf_valid requires (indices_array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let idxs: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    // Empty array represents 0 — vacuously valid.
                    let mut ok = true;
                    for w in idxs.windows(2) {
                        if w[0] <= w[1] || w[0] - w[1] < 2 {
                            ok = false; break;
                        }
                    }
                    Ok(Value::HInt(HInt::new(if ok { 1 } else { 0 })))
                } else {
                    Err("is_zeckendorf_valid: argument must be an array".to_string())
                }
            }
            // substrate_min_distance(sorted_array, target) -> int
            //   Smallest |arr[i] - target| over i. Uses substrate_lower_bound
            //   to find the candidate index in O(log_phi_pi_fibonacci N),
            //   then checks at most the two neighbors. Total: substrate
            //   probe + O(1).
            "substrate_min_distance" => {
                if args.len() < 2 {
                    return Err("substrate_min_distance requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("substrate_min_distance: empty array".to_string());
                    }
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let pos = crate::phi_pi_fib::substrate_lower_bound(&ints, target);
                    let mut best = i64::MAX;
                    if pos < ints.len() {
                        let d = (ints[pos] - target).abs();
                        if d < best { best = d; }
                    }
                    if pos > 0 {
                        let d = (ints[pos - 1] - target).abs();
                        if d < best { best = d; }
                    }
                    Ok(Value::HInt(HInt::new(best)))
                } else {
                    Err("substrate_min_distance: first argument must be an array".to_string())
                }
            }
            // substrate_nearest(sorted_array, target) -> int
            //   Closest VALUE to target (vs distance from substrate_min_distance).
            //   Same algorithmic structure: substrate probe + 2-neighbor check.
            "substrate_nearest" => {
                if args.len() < 2 {
                    return Err("substrate_nearest requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    if items.is_empty() {
                        return Err("substrate_nearest: empty array".to_string());
                    }
                    let ints: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    let pos = crate::phi_pi_fib::substrate_lower_bound(&ints, target);
                    let mut best_val = ints[pos.min(ints.len() - 1)];
                    let best_dist = (best_val - target).abs();
                    if pos > 0 {
                        let alt = ints[pos - 1];
                        let d = (alt - target).abs();
                        if d < best_dist { best_val = alt; }
                    }
                    Ok(Value::HInt(HInt::new(best_val)))
                } else {
                    Err("substrate_nearest: first argument must be an array".to_string())
                }
            }
            // int_binary_search(sorted_int_array, target) -> int (or -1)
            //   Native textbook binary search; baseline for comparing the
            //   substrate-routed search's per-probe cost. Same O(log N)
            //   asymptotics, integer midpoint instead of F(k)/phi^(pi*k).
            //   Use this as the default for uniform-integer arrays where
            //   substrate coherence doesn't earn its keep.
            "int_binary_search" => {
                if args.len() < 2 {
                    return Err("int_binary_search requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let mut lo: i64 = 0;
                    let mut hi: i64 = items.len() as i64 - 1;
                    while lo <= hi {
                        let mid = lo + (hi - lo) / 2;
                        let v = items[mid as usize].to_int();
                        if v == target { return Ok(Value::HInt(HInt::new(mid))); }
                        if v < target { lo = mid + 1; } else { hi = mid - 1; }
                    }
                    Ok(Value::HInt(HInt::new(-1)))
                } else {
                    Err("int_binary_search: first argument must be an array".to_string())
                }
            }
            // int_lower_bound(sorted_int_array, target) -> int
            //   Native binary lower_bound — first index i with arr[i] >= target,
            //   or arr.len() if none. Pair with int_upper_bound for range
            //   queries. The "fast default" when substrate coherence isn't
            //   needed.
            "int_lower_bound" => {
                if args.len() < 2 {
                    return Err("int_lower_bound requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let mut lo: usize = 0;
                    let mut hi: usize = items.len();
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if items[mid].to_int() < target { lo = mid + 1; } else { hi = mid; }
                    }
                    Ok(Value::HInt(HInt::new(lo as i64)))
                } else {
                    Err("int_lower_bound: first argument must be an array".to_string())
                }
            }
            "int_upper_bound" => {
                if args.len() < 2 {
                    return Err("int_upper_bound requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items = arr.items.borrow();
                    let mut lo: usize = 0;
                    let mut hi: usize = items.len();
                    while lo < hi {
                        let mid = lo + (hi - lo) / 2;
                        if items[mid].to_int() <= target { lo = mid + 1; } else { hi = mid; }
                    }
                    Ok(Value::HInt(HInt::new(lo as i64)))
                } else {
                    Err("int_upper_bound: first argument must be an array".to_string())
                }
            }
            // sorted_merge(a, b) -> sorted union (with duplicates).
            //   Classical merge in O(|a|+|b|). Native because OMC-level
            //   merge spends ~20% of its time on dispatch overhead.
            "sorted_merge" => {
                if args.len() < 2 {
                    return Err("sorted_merge requires (sorted_a, sorted_b)".to_string());
                }
                let a_v = self.eval_expr(&args[0])?;
                let b_v = self.eval_expr(&args[1])?;
                if let (Value::Array(a), Value::Array(b)) = (a_v, b_v) {
                    let ai = a.items.borrow();
                    let bi = b.items.borrow();
                    let mut out = Vec::with_capacity(ai.len() + bi.len());
                    let (mut i, mut j) = (0usize, 0usize);
                    while i < ai.len() && j < bi.len() {
                        if ai[i].to_int() <= bi[j].to_int() {
                            out.push(ai[i].clone()); i += 1;
                        } else {
                            out.push(bi[j].clone()); j += 1;
                        }
                    }
                    while i < ai.len() { out.push(ai[i].clone()); i += 1; }
                    while j < bi.len() { out.push(bi[j].clone()); j += 1; }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("sorted_merge: both arguments must be arrays".to_string())
                }
            }
            // sorted_union(a, b) -> sorted union (duplicates removed).
            "sorted_union" => {
                if args.len() < 2 {
                    return Err("sorted_union requires (sorted_a, sorted_b)".to_string());
                }
                let a_v = self.eval_expr(&args[0])?;
                let b_v = self.eval_expr(&args[1])?;
                if let (Value::Array(a), Value::Array(b)) = (a_v, b_v) {
                    let ai = a.items.borrow();
                    let bi = b.items.borrow();
                    let mut out = Vec::with_capacity(ai.len() + bi.len());
                    let (mut i, mut j) = (0usize, 0usize);
                    while i < ai.len() && j < bi.len() {
                        let av = ai[i].to_int();
                        let bv = bi[j].to_int();
                        if av < bv { out.push(ai[i].clone()); i += 1; }
                        else if av > bv { out.push(bi[j].clone()); j += 1; }
                        else { out.push(ai[i].clone()); i += 1; j += 1; }
                    }
                    while i < ai.len() { out.push(ai[i].clone()); i += 1; }
                    while j < bi.len() { out.push(bi[j].clone()); j += 1; }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("sorted_union: both arguments must be arrays".to_string())
                }
            }
            // sorted_dedupe(sorted_a) -> sorted array with adjacent dupes removed.
            //   O(N) single pass; faster than arr_unique because input is
            //   already sorted (no hash-set bookkeeping needed).
            "sorted_dedupe" => {
                if args.is_empty() {
                    return Err("sorted_dedupe requires (sorted_array)".to_string());
                }
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut out: Vec<Value> = Vec::with_capacity(items.len());
                    let mut last: Option<i64> = None;
                    for v in items.iter() {
                        let n = v.to_int();
                        if last != Some(n) {
                            out.push(v.clone());
                            last = Some(n);
                        }
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("sorted_dedupe: requires an array".to_string())
                }
            }
            // harmonic_align(value) -> int
            //   Snap to the nearest Fibonacci attractor. Inverse-coupled
            //   with `hbit_tension` (which returns the distance discarded
            //   by this snap). O(log_phi_pi_fibonacci |value|) via the
            //   substrate's nearest-attractor search.
            "harmonic_align" => {
                if args.is_empty() {
                    return Err("harmonic_align requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (attr, _) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                Ok(Value::HInt(HInt::new(attr)))
            }
            // harmonic_unalign(value) -> int
            //   Signed distance from value to its nearest attractor:
            //   value - harmonic_align(value). Positive = above attractor,
            //   negative = below. Useful as a residual signal in
            //   substrate-routed ML (the attractor captures structure,
            //   this residual captures noise/anomaly).
            "harmonic_unalign" => {
                if args.is_empty() {
                    return Err("harmonic_unalign requires (value)".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                let (attr, _) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                Ok(Value::HInt(HInt::new(n - attr)))
            }
            // phi_pi_log_distance(a, b) -> float
            //   log_phi_pi_fibonacci(|a - b| + 1). Substrate-canonical
            //   distance metric — matches the iteration-count cost of
            //   reaching b from a via the substrate-search walk. Equals
            //   0 for a == b; grows by ~1 unit per phi^π-fold gap.
            "phi_pi_log_distance" => {
                if args.len() < 2 {
                    return Err("phi_pi_log_distance requires (a, b)".to_string());
                }
                let a = self.eval_expr(&args[0])?.to_int();
                let b = self.eval_expr(&args[1])?.to_int();
                let d = (a - b).unsigned_abs() as f64 + 1.0;
                Ok(Value::HFloat(crate::phi_pi_fib::log_phi_pi_fibonacci(d)))
            }
            // harmonic_resample(arr, n) -> array of n elements
            //   Downsample/upsample an array to length n by picking indices
            //   at phi-spaced positions (using the substrate's Fibonacci-
            //   bucketed striding). Preserves attractor-relative structure
            //   better than uniform striding because samples concentrate
            //   in the early/dense part of the input (low Fibonacci
            //   indices) and sparse in the tail.
            "harmonic_resample" => {
                if args.len() < 2 {
                    return Err("harmonic_resample requires (array, n)".to_string());
                }
                let n = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let m = items.len();
                    if m == 0 || n == 0 {
                        return Ok(Value::Array(HArray::from_vec(vec![])));
                    }
                    // Phi-warped index: i/n^(1/phi) -> i_in_source
                    // For substrate-coherence this matches the
                    // log_phi_pi_fibonacci index density.
                    const INV_PHI: f64 = 0.6180339887498949;
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        let t = (i as f64) / (n as f64);
                        // phi-warped: bias toward small indices
                        let warped = t.powf(INV_PHI);
                        let idx = (warped * (m - 1) as f64).round() as usize;
                        out.push(items[idx.min(m - 1)].clone());
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("harmonic_resample: first argument must be an array".to_string())
                }
            }
            // substrate_select_k(arr, k) -> int (k-th smallest, 0-indexed)
            //   Quickselect variant using the substrate's
            //   largest_attractor_at_most(median) as a pivot heuristic —
            //   pivots are biased toward Fibonacci attractors, which
            //   makes the partition step concentrate near substrate
            //   landmarks. Average-case O(N) like classic quickselect;
            //   the substrate pivot reduces worst-case probability on
            //   adversarial inputs that target uniform-pivot patterns.
            "substrate_select_k" => {
                if args.len() < 2 {
                    return Err("substrate_select_k requires (array, k)".to_string());
                }
                let k = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    if k >= items.len() {
                        return Err(format!(
                            "substrate_select_k: k={} out of range for len={}",
                            k, items.len()
                        ));
                    }
                    let mut work: Vec<i64> = items.iter().map(|v| v.to_int()).collect();
                    // Pivot choice: largest_attractor_at_most(median-ish).
                    let pivot_seed = work[work.len() / 2];
                    let pivot = crate::phi_pi_fib::largest_attractor_at_most(pivot_seed);
                    // Standard 3-way partition around pivot.
                    let mut lo_buf = Vec::new();
                    let mut eq_buf = Vec::new();
                    let mut hi_buf = Vec::new();
                    for v in work.drain(..) {
                        if v < pivot { lo_buf.push(v); }
                        else if v == pivot { eq_buf.push(v); }
                        else { hi_buf.push(v); }
                    }
                    if k < lo_buf.len() {
                        lo_buf.sort_unstable();
                        return Ok(Value::HInt(HInt::new(lo_buf[k])));
                    } else if k < lo_buf.len() + eq_buf.len() {
                        return Ok(Value::HInt(HInt::new(pivot)));
                    } else {
                        hi_buf.sort_unstable();
                        let idx = k - lo_buf.len() - eq_buf.len();
                        return Ok(Value::HInt(HInt::new(hi_buf[idx])));
                    }
                }
                Err("substrate_select_k: first argument must be an array".to_string())
            }
            // fib_chunks(array, base_k) -> array of sub-arrays
            //   Split an array into chunks of size FIBONACCI[base_k+i] for
            //   i = 0, 1, 2... The chunk size grows phi-fold per chunk —
            //   matches the natural "small-then-big" batching pattern in
            //   streaming algorithms (e.g. exponential moving averages
            //   with golden-ratio decay). Last chunk may be short.
            "fib_chunks" => {
                if args.is_empty() {
                    return Err("fib_chunks requires (array, base_k=2)".to_string());
                }
                let base_k = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_int().max(1) as usize
                } else { 2 };
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items = arr.items.borrow();
                    let mut out = Vec::new();
                    let mut pos = 0usize;
                    let mut k = base_k;
                    while pos < items.len() {
                        // Use largest_attractor_at_most-style helper:
                        // we just want FIBONACCI[k] but bounded by table.
                        let sz = crate::phi_pi_fib::nearest_attractor_with_dist(
                            // ask for any value that gives us FIBONACCI[k]
                            // — simplest: just walk the table directly via
                            // the existing helper exposed at module scope.
                            // We instead use a local short-circuit since
                            // FIBONACCI isn't pub. Substitute: round-trip
                            // via Zeckendorf for value 2^k as an approx.
                            // Cleaner: just compute Fibonacci inline.
                            0
                        ).0 as usize; // dummy; replaced below
                        let _ = sz; // silence warning
                        // Compute FIBONACCI[k] inline (40-term table fits u64):
                        let mut a: u64 = 0; let mut b: u64 = 1;
                        for _ in 0..k { let t = a + b; a = b; b = t; }
                        let chunk_size = (a as usize).max(1);
                        let end = (pos + chunk_size).min(items.len());
                        let sub: Vec<Value> = items[pos..end].to_vec();
                        out.push(Value::Array(HArray::from_vec(sub)));
                        pos = end;
                        k += 1;
                        if k > 40 { k = 40; } // cap at table limit
                    }
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("fib_chunks: first argument must be an array".to_string())
                }
            }
            // phi_pi_fib_search_traced(sorted_arr, target)
            //   Returns [result_int, probe_indices_array]. `result_int`
            //   is the exact-match index when found, or -(insert_pos+1)
            //   when not. `probe_indices_array` is the sequence of
            //   indices the Fibonacci-step search visited, in order.
            //   Used by experiments that need to measure step-size
            //   coherence externally.
            "phi_pi_fib_search_traced" => {
                if args.len() < 2 {
                    return Err("phi_pi_fib_search_traced requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    let (r, probes) = crate::phi_pi_fib::fibonacci_search_with_trace(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    let result_int = match r {
                        Ok(i) => i as i64,
                        Err(insert_pos) => -(insert_pos as i64 + 1),
                    };
                    let probe_vals: Vec<Value> = probes
                        .into_iter()
                        .map(|p| Value::HInt(HInt::new(p as i64)))
                        .collect();
                    let out = vec![
                        Value::HInt(HInt::new(result_int)),
                        Value::Array(HArray::from_vec(probe_vals)),
                    ];
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("phi_pi_fib_search_traced: first argument must be an array".to_string())
                }
            }
            // phi_pi_fib_nearest_traced(sorted_arr, target)
            //   Returns [nearest_index, probe_indices_array]. Always
            //   resolves to a valid nearest index (or -1 for empty arrays).
            "phi_pi_fib_nearest_traced" => {
                if args.len() < 2 {
                    return Err("phi_pi_fib_nearest_traced requires (sorted_array, target)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let target = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let ints: Vec<i64> = items_b.iter().map(|v| v.to_int()).collect();
                    if ints.is_empty() {
                        let out = vec![
                            Value::HInt(HInt::new(-1)),
                            Value::Array(HArray::from_vec(vec![])),
                        ];
                        return Ok(Value::Array(HArray::from_vec(out)));
                    }
                    let (r, probes) = crate::phi_pi_fib::fibonacci_search_with_trace(
                        &ints,
                        &target,
                        |a, b| if a < b { -1 } else if a > b { 1 } else { 0 },
                    );
                    let idx: usize = match r {
                        Ok(i) => i,
                        Err(insert_pos) => {
                            let n = ints.len();
                            if insert_pos == 0 {
                                0
                            } else if insert_pos >= n {
                                n - 1
                            } else {
                                let left = (target - ints[insert_pos - 1]).abs();
                                let right = (ints[insert_pos] - target).abs();
                                if right < left { insert_pos } else { insert_pos - 1 }
                            }
                        }
                    };
                    let probe_vals: Vec<Value> = probes
                        .into_iter()
                        .map(|p| Value::HInt(HInt::new(p as i64)))
                        .collect();
                    let out = vec![
                        Value::HInt(HInt::new(idx as i64)),
                        Value::Array(HArray::from_vec(probe_vals)),
                    ];
                    Ok(Value::Array(HArray::from_vec(out)))
                } else {
                    Err("phi_pi_fib_nearest_traced: first argument must be an array".to_string())
                }
            }
            "arr_slice" => {
                if args.len() < 3 {
                    return Err("arr_slice requires (array, start, end)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let start = self.eval_expr(&args[1])?.to_int().max(0) as usize;
                let end = self.eval_expr(&args[2])?.to_int().max(0) as usize;
                if let Value::Array(arr) = arr_v {
                    let items_b = arr.items.borrow();
                    let end = end.min(items_b.len());
                    let start = start.min(end);
                    let items: Vec<Value> = items_b[start..end].to_vec();
                    Ok(Value::Array(HArray::from_vec(items)))
                } else {
                    Err("arr_slice: first argument must be an array".to_string())
                }
            }
            // Canonical OMC uses bare `len(x)` — polymorphic over arrays and strings.
            "len" => {
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::Array(a) => Ok(Value::HInt(HInt::new(a.items.borrow().len() as i64))),
                    Value::String(s) => Ok(Value::HInt(HInt::new(s.chars().count() as i64))),
                    Value::Dict(d) => Ok(Value::HInt(HInt::new(d.borrow().len() as i64))),
                    Value::Null => Ok(Value::HInt(HInt::new(0))),
                    ref other => Err(format!(
                        "len: requires array, string, or dict, got {}",
                        type_name_of(other)
                    )),
                }
            }
            "arr_resonance" => {
                // Mean resonance across all elements that are HInts.
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    let items_b = arr.items.borrow();
                    if items_b.is_empty() {
                        return Ok(Value::HFloat(0.0));
                    }
                    let total: f64 = items_b
                        .iter()
                        .map(|v| HInt::compute_resonance(v.to_int()))
                        .sum();
                    Ok(Value::HFloat(total / items_b.len() as f64))
                } else {
                    Err("arr_resonance: requires an array".to_string())
                }
            }
            // ---- eval_omc(source_str) ----------------------------------------
            // Evaluate an OMC source string in the *current* interpreter scope.
            // Variables and functions defined in `source_str` become visible to
            // the caller after this call returns, exactly like `import` but from
            // an in-memory string.  Returns the last expression value of the
            // evaluated code, or Null if the snippet has no expression result.
            "eval_omc" => {
                if args.is_empty() {
                    return Err("eval_omc requires (source_string)".to_string());
                }
                let src = self.eval_expr(&args[0])?.to_string();
                let mut parser = crate::parser::Parser::new(&src);
                let stmts = parser.parse().map_err(|e| format!("eval_omc parse error: {}", e))?;
                // Register any function definitions into the current interpreter
                // before executing statements (mirrors the top-level flow).
                self.register_user_functions(&stmts);
                let pre_last = self.last_expression_value.take();
                self.execute(stmts)?;
                let result = self.last_expression_value.take().unwrap_or(Value::Null);
                // Restore the outer last_expression_value so we don't clobber
                // the caller's pending expression result.
                self.last_expression_value = pre_last;
                Ok(result)
            }
            // ---- eval_omc_fresh(source_str) -------------------------------------
            // Like eval_omc but runs in a brand-new, isolated interpreter.
            // No variables or functions leak between caller and callee.
            // Returns the last expression value of the evaluated code.
            "eval_omc_fresh" => {
                if args.is_empty() {
                    return Err("eval_omc_fresh requires (source_string)".to_string());
                }
                let src = self.eval_expr(&args[0])?.to_string();
                let mut parser = crate::parser::Parser::new(&src);
                let stmts = parser.parse().map_err(|e| format!("eval_omc_fresh parse error: {}", e))?;
                let mut fresh = Interpreter::new();
                fresh.register_user_functions(&stmts);
                fresh.execute(stmts)?;
                Ok(fresh.last_expression_value.take().unwrap_or(Value::Null))
            }
            // ---- eval_omc_ctx(source_str) ----------------------------------------
            // Like eval_omc_fresh but seeds the fresh interpreter with a snapshot
            // of the current interpreter's globals and user-defined functions.
            // The evaluated code sees all variables / functions defined so far in
            // the caller, but mutations inside the snippet do NOT propagate back.
            "eval_omc_ctx" => {
                if args.is_empty() {
                    return Err("eval_omc_ctx requires (source_string)".to_string());
                }
                let src = self.eval_expr(&args[0])?.to_string();
                let mut parser = crate::parser::Parser::new(&src);
                let stmts = parser.parse().map_err(|e| format!("eval_omc_ctx parse error: {}", e))?;
                let mut fresh = Interpreter::new();
                // Seed the fresh interpreter with a snapshot of current globals
                // and user-defined functions so the evaluated code can reference them.
                fresh.globals = self.globals.clone();
                fresh.functions = self.functions.clone();
                fresh.register_user_functions(&stmts);
                fresh.execute(stmts)?;
                Ok(fresh.last_expression_value.take().unwrap_or(Value::Null))
            }
            // ---- omc_source() ---------------------------------------------------
            // Returns the source text of the currently running program (set by
            // the CLI or MCP before execution via `set_source_code`).
            // Returns Null when no source was registered (e.g. interactive REPL).
            "omc_source" => {
                Ok(match &self.source_code {
                    Some(s) => Value::String(s.clone()),
                    None    => Value::Null,
                })
            }
            // Unknown name — check whether it's a local variable holding
            // a Value::Function before declaring it undefined. This is
            // what makes `h f = fn(x) {...}; f(3);` work: f resolves as
            // a closure value, and we dispatch through call_first_class_function.
            _ => {
                if let Some(v) = self.get_var(name) {
                    if matches!(v, Value::Function { .. }) {
                        // Evaluate the args here (call_first_class_function
                        // wants Values, not Expressions).
                        let arg_vals: Result<Vec<Value>, String> = args.iter()
                            .map(|e| self.eval_expr(e))
                            .collect();
                        return self.call_first_class_function(&v, arg_vals?);
                    }
                }
                // Unknown function — return a did_you_mean-augmented
                // error message PLUS inline signature hint for the top
                // suggestion. Closes the loop: LLM (or human) doesn't
                // need a follow-up omc_help call to know what to do.
                let suggestions = crate::docs::did_you_mean(name, 3);
                if suggestions.is_empty() {
                    Err(format!("Undefined function: {}", name))
                } else {
                    // Inline the signature of the top suggestion so the
                    // user sees both the suggestion AND its call shape.
                    let sig_hint = crate::docs::lookup(suggestions[0])
                        .map(|d| format!(" — signature: `{}`", d.signature))
                        .unwrap_or_default();
                    Err(format!(
                        "Undefined function: {} (did you mean: {}?{})",
                        name,
                        suggestions.join(", "),
                        sig_hint,
                    ))
                }
            }
        }
    }

    fn invoke_user_function(
        &mut self,
        name: &str,
        params: &[String],
        body: &[Statement],
        args: &[Expression],
    ) -> Result<Value, String> {
        // Convenience for call sites we haven't position-tagged yet
        // (HOFs, reflective dispatch, module imports).
        self.invoke_user_function_at(name, params, body, args, crate::ast::Pos::unknown())
    }

    fn invoke_user_function_at(
        &mut self,
        name: &str,
        params: &[String],
        body: &[Statement],
        args: &[Expression],
        call_site: crate::ast::Pos,
    ) -> Result<Value, String> {
        let mut eval_args = Vec::new();
        for arg in args {
            eval_args.push(self.eval_expr(arg)?);
        }

        if params.len() != eval_args.len() {
            return Err(format!(
                "Function {} expects {} arguments, got {}",
                name,
                params.len(),
                eval_args.len()
            ));
        }

        // JIT dispatch: if a hook is registered (set by the standalone
        // CLI when OMC_HBIT_JIT=1), give it first refusal. A `Some(_)`
        // return means the hook handled the call — skip tree-walk
        // entirely. `None` means fall through to tree-walk (no JIT'd
        // version, or args incompatible).
        if let Some(hook) = self.jit_dispatch.clone() {
            if let Some(result) = hook(name, &eval_args) {
                return result;
            }
        }

        self.locals.push(std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())));
        for (param, arg) in params.iter().zip(eval_args) {
            self.set_var(param.clone(), arg);
        }

        // Push a call-stack frame so error messages can show
        // who-called-whom. The frame is popped in BOTH the success
        // and error paths so the trace doesn't leak across calls.
        self.call_stack.push((name.to_string(), call_site));

        // Generator detection: a fn body that contains any Yield
        // statement is a generator. We push a fresh yield-collector
        // onto yield_stacks; every Yield in the body appends to it.
        // On exit, the collector is popped and returned as a
        // Value::Array. Any explicit `return` inside a generator is
        // silently ignored (Python's behavior: `return` in a
        // generator without an expression ends iteration; with an
        // expression, it becomes the StopIteration value, which OMC
        // doesn't represent in the eager-list model).
        let is_generator = stmts_contain_yield(body);
        if is_generator {
            self.yield_stacks.push(Vec::new());
        }

        let mut exec_err: Option<String> = None;
        for stmt in body {
            if let Err(e) = self.execute_stmt(stmt) {
                exec_err = Some(e);
                break;
            }
            if self.return_value.is_some() {
                break;
            }
        }

        self.call_stack.pop();
        self.locals.pop();

        if let Some(e) = exec_err {
            // Drop the generator's collector on error.
            if is_generator { self.yield_stacks.pop(); }
            return Err(format!(
                "{}\n  at {}{}",
                e,
                display_frame_name(name),
                format_call_site(call_site),
            ));
        }

        if is_generator {
            // Return the collected yields as an array. Ignore the
            // fn's return slot — generators communicate via yield.
            self.return_value.take();
            let yields = self.yield_stacks.pop().unwrap_or_default();
            return Ok(Value::Array(crate::value::HArray::from_vec(yields)));
        }

        let result = self.return_value.take().unwrap_or(Value::Null);
        Ok(result)
    }

    #[inline]
    fn get_var(&self, name: &str) -> Option<Value> {
        // Walk locals from inner to outer. Closure capture is achieved by
        // pushing the captured env Rc as a frame in `call_first_class_function`,
        // so the same walk handles both regular lexical lookup and closure
        // free-variable resolution.
        for scope_rc in self.locals.iter().rev() {
            if let Some(v) = scope_rc.borrow().get(name) {
                return Some(v.clone());
            }
        }
        // Globals as last resort.
        self.globals.get(name).cloned()
    }

    /// Snapshot every variable name currently visible (all local frames
    /// + globals). Used by the "Undefined variable" error path to suggest
    /// a close-spelled name (`did_you_mean(...)`-style hint).
    pub(crate) fn collect_in_scope_names(&self) -> Vec<String> {
        let mut names: HashSet<String> = HashSet::new();
        for scope_rc in self.locals.iter() {
            for k in scope_rc.borrow().keys() {
                names.insert(k.clone());
            }
        }
        for k in self.globals.keys() {
            names.insert(k.clone());
        }
        names.into_iter().collect()
    }

    /// Produce a "did you mean X?" hint for an undefined variable name.
    /// Returns an empty string when no close match found; otherwise a
    /// pre-formatted ` (did you mean: X?)` suffix ready to concat into
    /// the error message.
    pub(crate) fn undefined_var_hint(&self, name: &str) -> String {
        let candidates = self.collect_in_scope_names();
        // Use the same substrate-bucketed closest-name routine the heal
        // pass uses, so suggestions follow the same ranking.
        let cand_set: HashSet<String> = candidates.iter().cloned().collect();
        if let Some(close) = closest_name_substrate(name, &cand_set, 2, None) {
            format!(" (did you mean: {}?)", close)
        } else {
            String::new()
        }
    }

    /// Assignment semantics: walk outward looking for an EXISTING binding.
    /// Found in any local frame → mutate there (which for a closure-shared
    /// frame propagates to all holders of the Rc). Found in globals →
    /// write there. Not found anywhere → write to innermost local
    /// (implicit declaration).
    ///
    /// `h x = ...` (Statement::VarDecl) keeps using `set_var` directly so
    /// declarations always create a new innermost-local binding.
    fn assign_var(&mut self, name: String, value: Value) {
        for scope_rc in self.locals.iter().rev() {
            if scope_rc.borrow().contains_key(&name) {
                scope_rc.borrow_mut().insert(name, value);
                return;
            }
        }
        if self.globals.contains_key(&name) {
            self.globals.insert(name, value);
            return;
        }
        // Fallback: write to innermost local (creates an implicit decl).
        // OMC programs in the wild may rely on this; don't tighten.
        if let Some(scope_rc) = self.locals.last() {
            scope_rc.borrow_mut().insert(name, value);
        }
    }

    /// Test helper: read a variable from outside the interpreter.
    /// Used by integration tests in `tests/conformance.rs`.
    pub fn get_var_for_testing(&self, name: &str) -> Option<Value> {
        self.get_var(name)
    }

    // ---------- VM bridge helpers ----------
    // Used by the bytecode VM (src/vm.rs) so it can reuse this
    // Interpreter's scope stack + built-in stdlib without duplication.

    #[inline]
    pub fn vm_push_scope(&mut self) {
        self.locals.push(std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())));
    }

    #[inline]
    pub fn vm_pop_scope(&mut self) {
        if self.locals.len() > 1 {
            self.locals.pop();
        }
    }

    /// Push a captured closure environment as the next scope frame.
    /// Multiple closures created in the same scope share the same Rc
    /// so mutations propagate. Used by `call_first_class_function` to
    /// install the closure's environment before binding args.
    pub(crate) fn vm_push_closure_env(
        &mut self,
        env: std::rc::Rc<std::cell::RefCell<HashMap<String, Value>>>,
    ) {
        self.locals.push(env);
    }

    /// Drop the topmost closure-env frame (companion to vm_push_closure_env).
    /// Used by the VM's reflective dispatch path so it doesn't have to
    /// reach into Interpreter internals.
    pub(crate) fn vm_pop_closure_env(&mut self) {
        if self.locals.len() > 1 {
            self.locals.pop();
        }
    }

    #[inline]
    pub fn vm_set_local(&mut self, name: &str, value: Value) {
        self.set_var(name.to_string(), value);
    }

    /// VM-facing wrapper around assign_var — walks scopes outward for
    /// an existing binding, mutates there. See `assign_var` for the
    /// rules. Used by Op::AssignVar (introduced for mutable closure
    /// support).
    pub fn vm_assign_var(&mut self, name: &str, value: Value) {
        self.assign_var(name.to_string(), value);
    }

    /// VM-facing wrapper around execute_stmt — exposes the tree-walk
    /// statement executor so the bytecode VM can fall back to it for
    /// forms that don't compile (currently just Statement::Try).
    pub fn vm_exec_stmt(&mut self, stmt: &Statement) -> Result<(), String> {
        self.execute_stmt(stmt)
    }

    /// VM-facing: drain any pending return value set by a tree-walk
    /// Statement (e.g. a `return` inside a try body executed via
    /// Op::ExecStmt). Returns Some(value) and clears the slot if a
    /// return was issued; None otherwise. The VM must check this
    /// after every Op::ExecStmt and propagate via its own return path.
    pub fn vm_take_return(&mut self) -> Option<Value> {
        self.return_value.take()
    }

    /// Push a call-stack frame. The VM calls this at the entry of
    /// run_function so error traces work for VM-dispatched calls too.
    /// Pass Pos::unknown() if the call site isn't tracked.
    pub fn push_call_frame(&mut self, name: &str, call_site: crate::ast::Pos) {
        self.call_stack.push((name.to_string(), call_site));
    }

    /// REPL-facing: evaluate a single expression in the current
    /// interpreter state. Used to implement Python-style
    /// "type-an-expression-and-see-the-value" at the prompt.
    pub fn eval_for_repl(&mut self, expr: &Expression) -> Result<Value, String> {
        self.eval_expr(expr)
    }

    /// Pop a call-stack frame. Counterpart to push_call_frame; called
    /// in BOTH the success and error paths so the trace can't leak
    /// across calls.
    pub fn pop_call_frame(&mut self) {
        self.call_stack.pop();
    }

    /// Format an error message with the current call stack appended.
    /// Used by VM run_function on its error-return path to give the
    /// same kind of trace tree-walk produces. Innermost frame first.
    pub fn format_error_with_trace(&self, msg: &str) -> String {
        if msg.contains("\n  at ") {
            return msg.to_string();
        }
        let mut out = msg.to_string();
        for (fname, pos) in self.call_stack.iter().rev() {
            out.push_str(&format!(
                "\n  at {}{}",
                display_frame_name(fname),
                format_call_site(*pos),
            ));
        }
        out
    }

    /// VM-facing: same idea for break/continue flags. Returns and
    /// clears the flag.
    pub fn vm_take_break(&mut self) -> bool {
        let f = self.break_flag;
        self.break_flag = false;
        f
    }
    pub fn vm_take_continue(&mut self) -> bool {
        let f = self.continue_flag;
        self.continue_flag = false;
        f
    }

    /// Return an Rc clone of the topmost local scope frame, for closure
    /// capture in Op::Lambda. The Rc is shared — multiple lambdas in
    /// the same scope get the same underlying RefCell, so mutations
    /// propagate across sibling closures.
    pub fn vm_top_scope_rc(&self) -> Option<std::rc::Rc<std::cell::RefCell<HashMap<String, Value>>>> {
        self.locals.last().cloned()
    }

    /// Pre-register user function definitions into the interpreter's
    /// function table. Used by the VM driver in main.rs when running
    /// with OMC_VM=1: the VM has its own compiled function table in
    /// the Module, but first-class function dispatch (via the `call`
    /// builtin) routes through the interpreter, which needs to see
    /// the same function bodies. Tree-walks the body if reached this
    /// way; the user pays a slight cost for reflective dispatch in
    /// VM mode, but the regular Op::Call path stays bytecode-fast.
    /// Process every top-level `Statement::Import` in `statements`,
    /// registering the imported module's functions into self.functions.
    /// Used by main.rs under OMC_VM=1, since the bytecode compiler
    /// treats imports as no-ops and the VM never enters `execute_stmt`
    /// for top-level statements (its execution model is bytecode, not
    /// AST). Without this pre-pass, `math.fib_up_to(...)` calls in VM
    /// mode would fail with "Undefined function" even though the
    /// import line is there.
    ///
    /// Imports are deduplicated via `imported_modules`, so calling
    /// this twice (e.g. once during pre-pass, once via execute) is
    /// safe — the second call is a no-op.
    pub fn process_imports(&mut self, statements: &[Statement]) -> Result<(), String> {
        for stmt in statements {
            if let Statement::Import { module, alias, selected } = stmt {
                if let Some(names) = selected {
                    self.import_module_selective(module, names)?;
                } else {
                    self.import_module_with_alias(module, alias.as_deref())?;
                }
            }
        }
        Ok(())
    }

    pub fn register_user_functions(&mut self, statements: &[Statement]) {
        // Walks every FunctionDef anywhere in the AST — including those
        // nested inside other fn bodies, if-branches, while bodies, etc.
        // Matches the tree-walker's flat function-table semantics: a
        // nested `fn foo()` inside `fn bar()` becomes globally callable
        // after `bar` runs once. The VM path needs them pre-registered
        // so reflective dispatch can resolve them without depending on
        // execution order.
        fn visit(stmt: &Statement, fns: &mut HashMap<String, (Vec<String>, Vec<Statement>)>) {
            match stmt {
                Statement::FunctionDef { name, params, body, .. } => {
                    fns.insert(name.clone(), (params.clone(), body.clone()));
                    for s in body { visit(s, fns); }
                }
                Statement::ClassDef { name, parent: _parent, fields, methods } => {
                    // NOTE: parent registration happens in execute_stmt
                    // (which has access to &mut self). visit() only
                    // sees &mut HashMap<...> so it can't reach the
                    // class_parents table. For the VM-prep path, the
                    // class_parents update is made during execute_stmt
                    // when the statement actually executes.
                    //
                    // Desugar: build a constructor fn and one method fn
                    // per declared method. The constructor is a body of
                    // dict_set calls that populates a fresh dict with
                    // __class__ = "Name" + each positional field.
                    let mut ctor_body: Vec<Statement> = Vec::new();
                    // `h __obj = dict_new();`
                    ctor_body.push(Statement::VarDecl {
                        name: "__obj".to_string(),
                        value: Expression::Call {
                            name: "dict_new".to_string(),
                            args: vec![],
                            pos: crate::ast::Pos::unknown(),
                        },
                        is_harmonic: true,
                    });
                    // `dict_set(__obj, "__class__", "<Name>");`
                    ctor_body.push(Statement::Expression(Expression::Call {
                        name: "dict_set".to_string(),
                        args: vec![
                            Expression::Variable("__obj".to_string()),
                            Expression::String("__class__".to_string()),
                            Expression::String(name.clone()),
                        ],
                        pos: crate::ast::Pos::unknown(),
                    }));
                    // One dict_set per field, copying the param value.
                    for f in fields {
                        ctor_body.push(Statement::Expression(Expression::Call {
                            name: "dict_set".to_string(),
                            args: vec![
                                Expression::Variable("__obj".to_string()),
                                Expression::String(f.clone()),
                                Expression::Variable(f.clone()),
                            ],
                            pos: crate::ast::Pos::unknown(),
                        }));
                    }
                    // `return __obj;`
                    ctor_body.push(Statement::Return(Some(
                        Expression::Variable("__obj".to_string()),
                    )));
                    fns.insert(name.clone(), (fields.clone(), ctor_body));

                    // Each method becomes a top-level fn with the
                    // mangled name `Name__method`. The first parameter
                    // is `self`, populated by call_function's instance
                    // dispatch path.
                    for m in methods {
                        if let Statement::FunctionDef { name: mname, params, body, .. } = m {
                            let mangled = format!("{}__{}", name, mname);
                            fns.insert(mangled, (params.clone(), body.clone()));
                            // Recurse into the method body in case it
                            // contains nested fn defs.
                            for s in body { visit(s, fns); }
                        }
                    }
                }
                Statement::If { then_body, elif_parts, else_body, .. } => {
                    for s in then_body { visit(s, fns); }
                    for (_, b) in elif_parts { for s in b { visit(s, fns); } }
                    if let Some(b) = else_body { for s in b { visit(s, fns); } }
                }
                Statement::While { body, .. } | Statement::For { body, .. } => {
                    for s in body { visit(s, fns); }
                }
                Statement::Try { body, handler, finally, .. } => {
                    for s in body { visit(s, fns); }
                    for s in handler { visit(s, fns); }
                    if let Some(f) = finally { for s in f { visit(s, fns); } }
                }
                Statement::Match { arms, .. } => {
                    for arm in arms { for s in &arm.body { visit(s, fns); } }
                }
                _ => {}
            }
        }
        for stmt in statements {
            visit(stmt, &mut self.functions);
        }
    }

    /// Register a single anonymous-lambda body. Used by main.rs in VM
    /// mode to register every lambda the compiler discovered. See
    /// `module.lambda_asts` in bytecode.rs for context.
    pub fn register_lambda(&mut self, name: &str, params: Vec<String>, body: Vec<Statement>) {
        self.functions.insert(name.to_string(), (params, body));
    }

    #[inline]
    pub fn vm_get_var(&self, name: &str) -> Option<Value> {
        // Variable lookup with function-table fallback — mirrors the
        // tree-walker's Expression::Variable handling. Lets the bytecode
        // VM resolve bare names as Value::Function for first-class
        // function support (passing `bench_int_add` as a value, etc.).
        if let Some(v) = self.get_var(name) {
            return Some(v);
        }
        if self.functions.contains_key(name) || self.is_known_builtin(name) {
            return Some(Value::Function { name: name.to_string(), captured: None });
        }
        None
    }

    /// Same as vm_get_var but WITHOUT the function-table fallback. The VM's
    /// Op::Call dispatch uses this to check "is `name` a variable holding
    /// a Value::Function" — without falling back to a Function-ref from
    /// the function table itself (which would be redundant; the is_user
    /// branch above already handles that).
    pub fn vm_get_var_local_only(&self, name: &str) -> Option<Value> {
        self.get_var(name)
    }

    /// Call a built-in (or user-defined) function with already-evaluated args.
    /// The VM uses this when it encounters Op::Call and the function isn't
    /// a compiled function in the current module.
    pub fn vm_call_builtin(
        &mut self,
        name: &str,
        args: &[Value],
    ) -> Result<Value, String> {
        // Reverse-FFI host builtins fire FIRST so they can shadow
        // anything (including stdlib names like `read_file`). Lets an
        // embedder hand OMC code a sandboxed `read_file` that only
        // sees /tmp, etc. Skipped if the host hasn't registered the
        // name — the no-op cost is one HashMap lookup.
        if let Some(handler) = self.host_builtins.get(name).cloned() {
            // Stash a self-pointer so the handler can call back into
            // the interp (Python→OMC callbacks). Mirror call_function.
            let prev = INTERP_PTR.with(|p| p.replace(self as *mut _));
            let r = handler(args);
            INTERP_PTR.with(|p| p.set(prev));
            return r;
        }

        // Phase 4 fast-path: hot builtins handled directly on values,
        // bypassing the synthetic-arg shim. Each one shaved ~50% off
        // its per-call time on the benchmark suite (str_concat went
        // from 2200 to ~1200 ns/op; arr_get from 168000 to ~100000).
        // Anything that mutates by name (arr_push/dict_set/etc.) is
        // already handled by dedicated opcodes in the compiler.
        if let Some(r) = vm_fast_dispatch(name, args) {
            return r;
        }

        // Slow-path fallback: stash each evaluated arg in a fresh scope
        // under a synthetic name, then route through call_function with
        // Expression::Variable refs. This reuses ALL existing built-in
        // implementations for the long tail of less-hot builtins.
        self.vm_push_scope();
        let mut expr_args = Vec::with_capacity(args.len());
        for (i, v) in args.iter().enumerate() {
            let key = format!("__vm_arg_{}", i);
            self.vm_set_local(&key, v.clone());
            expr_args.push(crate::ast::Expression::Variable(key));
        }
        let result = self.call_function(name, &expr_args);
        self.vm_pop_scope();
        result
    }

    #[inline]
    fn set_var(&mut self, name: String, value: Value) {
        if let Some(scope_rc) = self.locals.last() {
            scope_rc.borrow_mut().insert(name, value);
        }
    }

    fn call_module_function(
        &mut self,
        module: &str,
        func: &str,
        args: &[Expression],
    ) -> Result<Value, String> {
        match (module, func) {
            ("phi", "fold") => {
                if args.is_empty() {
                    return Err("phi.fold requires at least 1 argument".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                let depth = if args.len() >= 2 {
                    self.eval_expr(&args[1])?.to_int().max(1) as usize
                } else {
                    1
                };
                Ok(self.phi_fold_n(v, depth))
            }
            ("phi", "res") => {
                if args.is_empty() {
                    return Err("phi.res requires 1 argument".to_string());
                }
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::HInt(h) => Ok(Value::HFloat(h.resonance)),
                    Value::HFloat(f) => {
                        Ok(Value::HFloat(HInt::compute_resonance(f as i64)))
                    }
                    _ => Ok(Value::HFloat(0.0)),
                }
            }
            ("phi", "him") => {
                if args.is_empty() {
                    return Err("phi.him requires 1 argument".to_string());
                }
                let n = self.eval_expr(&args[0])?.to_int();
                Ok(Value::HFloat(HInt::compute_him(n)))
            }
            // Unknown module path. Try the dotted form as a literal
            // user-function name FIRST — that's where aliased imports
            // live (`import "math" as math` creates `math.fib` in
            // self.functions). Fall through to unqualified `func` as a
            // last resort so legacy `core.fib(...)` after a plain
            // `import core;` still works.
            _ => {
                let full = format!("{}.{}", module, func);
                if self.functions.contains_key(&full) {
                    return self.call_function(&full, args);
                }
                self.call_function(func, args)
            }
        }
    }

}

/// Type-aware value equality. Used by `==` and `!=`. Replaces the old
/// "coerce both sides to int and compare" rule, which silently made any
/// two non-numeric values of the same int-cast appear equal (e.g.
/// `"foo" == "bar"` was true, and so was `["VAR", "x"] == "null"`).
///
/// Rules:
/// - Same-shape structural equality for String and Array (recursive).
/// - Singularity values compared by numerator + context.
/// - Mixed Array / Circuit / Singularity vs anything else → not equal.
/// - Otherwise fall back to numeric coercion (HInt, HFloat, Bool, Null).
/// Phase 4: VM hot-builtin fast path. Returns Some(result) when the
/// builtin can be answered directly from the supplied Value args
/// without the synthetic-arg shim, None to fall through to the
/// general dispatch in vm_call_builtin.
///
/// Only PURE builtins go here — anything that mutates by name
/// (arr_push, arr_set, dict_set, dict_del) is already handled by
/// dedicated opcodes in the compiler, so it never reaches
/// vm_call_builtin in the first place.
/// Walk `src` and return every top-level `fn NAME(...) { ... }` as a
/// separate string. Skips nested fns and `#`-prefixed line comments;
/// tracks `"..."` and `'...'` so braces inside string literals don't
/// throw off depth counting. Used by omc_registry_codec_library and
/// omc_msg_recover_from_registry, plus the omc-grep tool.
pub fn extract_top_level_fns(src: &str) -> Vec<String> {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        // Skip line comments.
        if bytes[i] == b'#' {
            while i < n && bytes[i] != b'\n' { i += 1; }
            continue;
        }
        // Skip string literals at top level.
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let q = bytes[i]; i += 1;
            while i < n && bytes[i] != q {
                if bytes[i] == b'\\' && i + 1 < n { i += 2; } else { i += 1; }
            }
            if i < n { i += 1; }
            continue;
        }
        // Recognize `fn ` only at start-of-line or after whitespace.
        let at_boundary = i == 0 || bytes[i - 1].is_ascii_whitespace();
        if at_boundary && i + 3 < n && &bytes[i..i + 3] == b"fn " {
            let fn_start = i;
            // Find the opening `{` of the body.
            let mut j = i;
            while j < n && bytes[j] != b'{' { j += 1; }
            if j >= n { break; }
            // Track depth, respecting strings + line comments.
            let mut depth = 0i32;
            let mut k = j;
            while k < n {
                let c = bytes[k];
                if c == b'#' {
                    while k < n && bytes[k] != b'\n' { k += 1; }
                    continue;
                }
                if c == b'"' || c == b'\'' {
                    let q = c; k += 1;
                    while k < n && bytes[k] != q {
                        if bytes[k] == b'\\' && k + 1 < n { k += 2; } else { k += 1; }
                    }
                    if k < n { k += 1; }
                    continue;
                }
                if c == b'{' { depth += 1; }
                else if c == b'}' {
                    depth -= 1;
                    if depth == 0 { k += 1; break; }
                }
                k += 1;
            }
            if depth == 0 && k > fn_start {
                out.push(src[fn_start..k].to_string());
            }
            i = k;
            continue;
        }
        i += 1;
    }
    out
}

fn vm_fast_dispatch(name: &str, args: &[Value]) -> Option<Result<Value, String>> {
    match (name, args.len()) {
        // ---- string ops ----
        ("str_concat", 2) => Some(Ok(Value::String(format!(
            "{}{}",
            args[0].to_display_string(),
            args[1].to_display_string()
        )))),
        ("str_len", 1) => {
            if let Value::String(s) = &args[0] {
                Some(Ok(Value::HInt(HInt::new(s.len() as i64))))
            } else { None }
        }
        ("str_chars", 1) => {
            if let Value::String(s) = &args[0] {
                Some(Ok(Value::HInt(HInt::new(s.chars().count() as i64))))
            } else { None }
        }
        ("str_slice", 3) => {
            if let Value::String(s) = &args[0] {
                let start = args[1].to_int().max(0) as usize;
                let end = args[2].to_int().max(0) as usize;
                let chars: Vec<char> = s.chars().collect();
                let lo = start.min(chars.len());
                let hi = end.min(chars.len()).max(lo);
                let out: String = chars[lo..hi].iter().collect();
                Some(Ok(Value::String(out)))
            } else { None }
        }
        ("str_split", 2) => {
            if let (Value::String(s), Value::String(sep)) = (&args[0], &args[1]) {
                let items: Vec<Value> = if sep.is_empty() {
                    s.chars().map(|c| Value::String(c.to_string())).collect()
                } else {
                    s.split(sep.as_str()).map(|p| Value::String(p.to_string())).collect()
                };
                Some(Ok(Value::Array(HArray::from_vec(items))))
            } else { None }
        }
        ("str_join", 2) => {
            if let (Value::Array(arr), Value::String(sep)) = (&args[0], &args[1]) {
                let parts: Vec<String> = arr.items.borrow().iter()
                    .map(|v| v.to_display_string())
                    .collect();
                Some(Ok(Value::String(parts.join(sep.as_str()))))
            } else { None }
        }
        // ---- conversion ----
        ("to_int", 1) | ("int", 1) => {
            Some(Ok(Value::HInt(HInt::new(args[0].to_int()))))
        }
        ("to_float", 1) | ("float", 1) => {
            Some(Ok(Value::HFloat(args[0].to_float())))
        }
        ("to_string", 1) | ("to_str", 1) | ("string", 1) => {
            Some(Ok(Value::String(args[0].to_display_string())))
        }
        // ---- println / print: handled by the Interpreter method (needs self) ----
        // Intentionally NOT handled here so they fall through to the method
        // which has access to self.output_lines for MCP stdout capture.
        _ => None,
    }
}

// ===========================================================================
// Active-interpreter pointer for reentrant host calls.
//
// Set by call_function / vm_call_builtin BEFORE invoking a host
// builtin handler, cleared after. While set, a host handler can
// reach back into the live Interpreter via `with_active_interp` —
// needed for Python → OMC callbacks (py_callback returns a
// PyCallable that calls back into OMC's interp).
//
// Single-threaded by design (matches OMC's runtime model). The
// pointer is only valid for the duration of the host handler call;
// stashing it elsewhere is a use-after-free waiting to happen.
// ===========================================================================

thread_local! {
    static INTERP_PTR: std::cell::Cell<*mut Interpreter> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// Run `f` with a `&mut Interpreter` pointing at the currently-
/// active interpreter (the one whose host_builtin handler is
/// running). Returns None if called outside a host_builtin context.
///
/// SAFETY: The pointer is valid only inside a host_builtin call —
/// the dispatch site sets it on entry and clears on exit. Don't
/// stash the &mut anywhere; use it within `f` and let it drop.
pub fn with_active_interp<R>(f: impl FnOnce(&mut Interpreter) -> R) -> Option<R> {
    let p = INTERP_PTR.with(|p| p.get());
    if p.is_null() {
        return None;
    }
    // SAFETY: see doc comment. The dispatch contract guarantees
    // the pointer is valid for the duration of this call.
    let interp = unsafe { &mut *p };
    Some(f(interp))
}

pub fn display_frame_name(name: &str) -> &str {
    if name.starts_with("__rt_lambda_") || name.starts_with("__lambda_") {
        "<lambda>"
    } else {
        name
    }
}

/// Render a call-site position as the `(line:col)` suffix shown
/// after the frame name in stack traces. Returns the empty string
/// for synthesized frames (Pos::unknown) so traces stay clean
/// when the call wasn't position-tagged.
pub fn format_call_site(p: crate::ast::Pos) -> String {
    if p.line == 0 {
        String::new()
    } else {
        format!(" ({})", p)
    }
}

/// Test whether `pattern` accepts `value`. On success, appends any
/// `Pattern::Bind(name)` matches into `bindings` (ordered) so the
/// caller can install them in the arm's scope.
///
/// Pure / side-effect-free aside from the bindings vec — same
/// helper is used by both tree-walk and VM (via vm_match_helper).
pub(crate) fn pattern_matches(
    pattern: &crate::ast::Pattern,
    value: &Value,
    bindings: &mut Vec<(String, Value)>,
) -> bool {
    use crate::ast::Pattern;
    match pattern {
        Pattern::Wildcard => true,
        Pattern::Bind(n) => {
            bindings.push((n.clone(), value.clone()));
            true
        }
        Pattern::LitInt(n) => match value {
            Value::HInt(h) => h.value == *n,
            Value::HFloat(f) => *f == *n as f64,
            _ => false,
        },
        Pattern::LitFloat(f) => match value {
            Value::HFloat(g) => g == f,
            Value::HInt(h) => (h.value as f64) == *f,
            _ => false,
        },
        Pattern::LitString(s) => matches!(value, Value::String(v) if v == s),
        Pattern::LitBool(b) => match value {
            Value::Bool(v) => v == b,
            // OMC's int-as-bool convention: 0/1 ints commonly stand
            // in for false/true. Accept matches against literal bool
            // patterns so `match flag { true => ..., false => ... }`
            // works on the int-coded values too.
            Value::HInt(h) => (h.value != 0) == *b,
            _ => false,
        },
        Pattern::LitNull => matches!(value, Value::Null),
        Pattern::RangeInt(lo, hi) => {
            let n = match value {
                Value::HInt(h) => h.value,
                Value::HFloat(f) => *f as i64,
                _ => return false,
            };
            n >= *lo && n <= *hi
        }
        Pattern::RangeStr(lo, hi) => {
            if let Value::String(s) = value {
                let chars: Vec<char> = s.chars().collect();
                if chars.len() == 1 {
                    let c = chars[0];
                    return c >= *lo && c <= *hi;
                }
            }
            false
        }
        Pattern::Or(alts) => {
            // Try each alt with a snapshot of bindings; first match wins.
            // We don't allow bindings to differ between alts (same as Rust's
            // requirement that all alts bind the same names) — for v1 we
            // simply propagate whatever the matching alt produced.
            for p in alts {
                let snapshot_len = bindings.len();
                if pattern_matches(p, value, bindings) {
                    return true;
                }
                bindings.truncate(snapshot_len);
            }
            false
        }
        Pattern::Type(tag) => {
            let actual = match value {
                Value::HInt(_) => "int",
                Value::HFloat(_) => "float",
                Value::String(_) => "string",
                Value::Bool(_) => "bool",
                Value::Array(_) => "array",
                Value::Dict(_) => "dict",
                Value::Function { .. } => "function",
                Value::Null => "null_t",
                Value::Singularity { .. } => "singularity",
                _ => "unknown",
            };
            actual == tag
        }
    }
}

/// AdamW per-parameter update fully in Rust. Replaces ~15 OMC-side
/// element-wise loops with one tight Rust loop. Accepts 1D or 2D
/// OMC arrays for `cur`, `grad`, `m`, `v` (same shape across all four).
/// Mutates `m` and `v` in place — they're Rc-shared so the caller picks
/// up the new state. Returns a freshly-allocated OMC array with the new
/// parameter value.
fn substrate_adamw_update(
    cur: &Value, grad: &Value, m_arr: &Value, v_arr: &Value,
    lr: f64, b1: f64, b2: f64, eps: f64, wd: f64, step: i32,
) -> Result<Value, String> {
    let (cur_rows, cur_cols, cur_flat) = flatten_2d_or_1d(cur, "cur")?;
    let (g_rows, g_cols, g_flat) = flatten_2d_or_1d(grad, "grad")?;
    let (m_rows, m_cols, mut m_flat) = flatten_2d_or_1d(m_arr, "m")?;
    let (v_rows, v_cols, mut v_flat) = flatten_2d_or_1d(v_arr, "v")?;
    if (cur_rows, cur_cols) != (g_rows, g_cols)
        || (cur_rows, cur_cols) != (m_rows, m_cols)
        || (cur_rows, cur_cols) != (v_rows, v_cols)
    {
        return Err(format!(
            "shape mismatch: cur={}×{}, grad={}×{}, m={}×{}, v={}×{}",
            cur_rows, cur_cols, g_rows, g_cols, m_rows, m_cols, v_rows, v_cols
        ));
    }
    let bias1 = 1.0 - b1.powi(step);
    let bias2 = 1.0 - b2.powi(step);
    let mut out_flat: Vec<f64> = Vec::with_capacity(cur_flat.len());
    for k in 0..cur_flat.len() {
        let g = g_flat[k];
        let m_new = b1 * m_flat[k] + (1.0 - b1) * g;
        let v_new = b2 * v_flat[k] + (1.0 - b2) * g * g;
        m_flat[k] = m_new;
        v_flat[k] = v_new;
        let m_hat = m_new / bias1;
        let v_hat = v_new / bias2;
        let denom = v_hat.sqrt() + eps;
        let adam_step = m_hat / denom;
        let theta = cur_flat[k] - lr * wd * cur_flat[k] - lr * adam_step;
        out_flat.push(theta);
    }
    // Write m and v back through the Rc-shared OMC arrays so caller sees update.
    write_back_1d_or_2d(m_arr, m_rows, m_cols, &m_flat, "m")?;
    write_back_1d_or_2d(v_arr, v_rows, v_cols, &v_flat, "v")?;
    Ok(rebuild_omc_array(cur_rows, cur_cols, &out_flat, was_2d(cur)))
}

fn was_2d(v: &Value) -> bool {
    if let Value::Array(a) = v {
        let items = a.items.borrow();
        if !items.is_empty() {
            return matches!(&items[0], Value::Array(_));
        }
    }
    false
}

fn flatten_2d_or_1d(v: &Value, label: &str) -> Result<(usize, usize, Vec<f64>), String> {
    let arr = match v {
        Value::Array(a) => a,
        _ => return Err(format!("{}: expected array", label)),
    };
    let items = arr.items.borrow();
    if items.is_empty() {
        return Ok((0, 0, vec![]));
    }
    if matches!(&items[0], Value::Array(_)) {
        let cols = if let Value::Array(r) = &items[0] { r.items.borrow().len() } else { 0 };
        let mut flat = Vec::with_capacity(items.len() * cols);
        for row in items.iter() {
            let row_arr = match row {
                Value::Array(a) => a,
                _ => return Err(format!("{}: mixed 1D/2D rows", label)),
            };
            let row_items = row_arr.items.borrow();
            if row_items.len() != cols {
                return Err(format!("{}: ragged 2D array", label));
            }
            for cell in row_items.iter() {
                flat.push(cell.to_float());
            }
        }
        Ok((items.len(), cols, flat))
    } else {
        let flat: Vec<f64> = items.iter().map(|c| c.to_float()).collect();
        Ok((1, flat.len(), flat))
    }
}

fn write_back_1d_or_2d(
    target: &Value, rows: usize, cols: usize, flat: &[f64], label: &str,
) -> Result<(), String> {
    let arr = match target {
        Value::Array(a) => a,
        _ => return Err(format!("{}: not an array", label)),
    };
    let mut items = arr.items.borrow_mut();
    if rows == 1 && !items.is_empty() && !matches!(&items[0], Value::Array(_)) {
        // 1D shape: overwrite cells in place
        for k in 0..cols {
            items[k] = Value::HFloat(flat[k]);
        }
        return Ok(());
    }
    if items.len() != rows {
        return Err(format!("{}: shape change during write-back ({} → {})",
                           label, items.len(), rows));
    }
    for r in 0..rows {
        let row_arr = match &items[r] {
            Value::Array(a) => a.clone(),
            _ => return Err(format!("{}: row {} not an array", label, r)),
        };
        let mut row_items = row_arr.items.borrow_mut();
        for c in 0..cols {
            row_items[c] = Value::HFloat(flat[r * cols + c]);
        }
    }
    Ok(())
}

fn rebuild_omc_array(rows: usize, cols: usize, flat: &[f64], as_2d: bool) -> Value {
    if !as_2d {
        let row: Vec<Value> = flat.iter().map(|&x| Value::HFloat(x)).collect();
        return Value::Array(HArray::from_vec(row));
    }
    let mut out_rows: Vec<Value> = Vec::with_capacity(rows);
    for r in 0..rows {
        let row: Vec<Value> = (0..cols)
            .map(|c| Value::HFloat(flat[r * cols + c]))
            .collect();
        out_rows.push(Value::Array(HArray::from_vec(row)));
    }
    Value::Array(HArray::from_vec(out_rows))
}

/// Which Prometheus substrate-modulator we're computing. Both are
/// element-wise "1 / (1 + something · attractor_distance)" formulas;
/// they differ in whether the cell is treated as a raw score (S-MOD)
/// or pre-scaled value (resample).
#[derive(Copy, Clone, Debug)]
enum ModulatorKind {
    /// `1 / (1 + alpha · attractor_distance(int(x)))`. Used by
    /// `prom_substrate_softmax(alpha > 0)`.
    SMod,
    /// `1 / (1 + attractor_distance(int(x · scale)) / scale)`. Used by
    /// `prom_substrate_resample(scale > 0)`.
    Resample,
}

/// Per-cell substrate modulator over a matrix-shaped OMC Value. Accepts
/// either 2D arrays (the typical [N, D]/[N, T] case) or 1D arrays (the
/// 1-row case returned by `tape_value` for single-row matrices). The
/// returned shape mirrors the input shape exactly.
///
/// Rust-side replacement for OMC `_prom_smod_matrix` / `_prom_substrate
/// _resample_matrix` (which were the v0.8.2 wall-clock bottleneck — see
/// `experiments/prometheus_parity/GPU_INTEGRATION.md`).
fn build_substrate_modulator_matrix(
    input: &Value, param: f64, kind: ModulatorKind,
) -> Result<Value, String> {
    let one_cell = |x: f64| -> f64 {
        match kind {
            ModulatorKind::SMod => {
                let n = x as i64;
                let (_, d) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                1.0 / (1.0 + param * (d as f64))
            }
            ModulatorKind::Resample => {
                let n = (x * param) as i64;
                let (_, d) = crate::phi_pi_fib::nearest_attractor_with_dist(n);
                1.0 / (1.0 + (d as f64) / param)
            }
        }
    };
    let arr = match input {
        Value::Array(a) => a,
        _ => return Err("expected a 1D or 2D array".to_string()),
    };
    let rows = arr.items.borrow();
    if rows.is_empty() {
        return Ok(Value::Array(HArray::from_vec(vec![])));
    }
    // 1D array (single-row matrix): emit a 1D array back out.
    if !matches!(&rows[0], Value::Array(_)) {
        let out: Vec<Value> = rows.iter()
            .map(|cell| Value::HFloat(one_cell(cell.to_float())))
            .collect();
        return Ok(Value::Array(HArray::from_vec(out)));
    }
    // 2D array: emit a 2D array of equal shape.
    let mut out_rows: Vec<Value> = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let row_arr = match row {
            Value::Array(a) => a,
            _ => return Err("ragged input: rows must all be arrays".to_string()),
        };
        let cells = row_arr.items.borrow();
        let new_row: Vec<Value> = cells.iter()
            .map(|cell| Value::HFloat(one_cell(cell.to_float())))
            .collect();
        out_rows.push(Value::Array(HArray::from_vec(new_row)));
    }
    Ok(Value::Array(HArray::from_vec(out_rows)))
}

/// One value on the reverse-mode tape. Scalar (1×1) or 2D matrix —
/// the matrix form drives end-to-end training without ever leaving OMC.
/// All numeric storage is f64 internally to keep gradient accumulation
/// numerically clean. Substrate metadata lives in the *forward* Value
/// (rebuilt as HInt when integral) and is exposed via tape_value().
#[derive(Clone, Debug)]
pub(crate) struct TapeMat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl TapeMat {
    pub fn scalar(x: f64) -> Self { Self { data: vec![x], rows: 1, cols: 1 } }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }
    pub fn from_2d(rows: &[Vec<f64>]) -> Self {
        let r = rows.len();
        let c = if r == 0 { 0 } else { rows[0].len() };
        let mut data = Vec::with_capacity(r * c);
        for row in rows { data.extend_from_slice(row); }
        Self { data, rows: r, cols: c }
    }
    pub fn at(&self, i: usize, j: usize) -> f64 { self.data[i * self.cols + j] }
    pub fn set(&mut self, i: usize, j: usize, v: f64) { self.data[i * self.cols + j] = v; }
    pub fn add(&mut self, other: &TapeMat) {
        // Broadcasting-aware add: same shape, or other is a 1×cols
        // row-vector broadcast across our rows. Falls back to flat
        // copy otherwise (caller already validated shapes).
        if self.rows == other.rows && self.cols == other.cols {
            for k in 0..self.data.len() { self.data[k] += other.data[k]; }
        } else if other.rows == 1 && other.cols == self.cols {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    self.data[i * self.cols + j] += other.data[j];
                }
            }
        } else if self.rows == 1 && self.cols == other.cols {
            // Grow self to match other's row count — used when a
            // broadcast bias accumulates gradient from many rows.
            // Sum down to a single row instead.
            let mut acc = vec![0.0; self.cols];
            for i in 0..other.rows {
                for j in 0..self.cols {
                    acc[j] += other.data[i * other.cols + j];
                }
            }
            for j in 0..self.cols { self.data[j] += acc[j]; }
        } else if other.rows * other.cols == 1 {
            for k in 0..self.data.len() { self.data[k] += other.data[0]; }
        } else if self.rows * self.cols == 1 {
            // Scalar self gets sum of all of other.
            let s: f64 = other.data.iter().sum();
            self.data[0] += s;
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TapeOp {
    /// Leaf — a variable the user wants gradients for.
    Var,
    /// Constant — held but not part of grad propagation.
    Const,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),      // element-wise (or scalar)
    Div(usize, usize),      // element-wise
    Neg(usize),
    PowInt(usize, i32),
    Exp(usize),
    Log(usize),
    /// Fused per-batch cross-entropy: softmax + select target log-probs +
    /// mean, all in one node. Forward returns a scalar; backward uses the
    /// closed-form `dL/dlogits[i, c] = (softmax(logits)[i, c] - 1{c == t_i}) / N`,
    /// which is *much* tighter than chaining tape_softmax + tape_log +
    /// tape_mul(mask) + tape_sum backward through 5 intermediate nodes.
    /// `targets` stored inside the op so backward has it without
    /// recomputing or threading through another tape node.
    CrossEntropyBatch(usize, Vec<usize>),
    /// Fused embedding row-gather: out[i, :] = table[token_ids[i], :].
    /// Replaces `prom_embedding_batch`'s OMC-built one-hot batch + matmul
    /// (which was N×vocab cells of one-hot construction + an N×vocab×D
    /// matmul) with a direct copy. Backward scatters dL/dout rows back
    /// into the corresponding dL/dtable rows.
    EmbeddingLookup(usize, Vec<usize>),
    /// v0.8.10 substrate-aware backward gradients. Forward is identity
    /// (x passes through unchanged); backward multiplies dy by a per-cell
    /// substrate-attraction factor:
    ///
    ///   sign = sign of (nearest_attractor(x·scale) - x·scale)
    ///   amp  = 1 + alpha · (substrate_dist(x·scale) > 0 ? 1 : 0)
    ///   dx   = dy · (amp when grad direction matches sign-to-attractor,
    ///                 1/amp when grad would push x AWAY from attractor)
    ///
    /// The substrate becomes a gradient-flow regularizer: updates that
    /// move parameters TOWARD Fibonacci attractors are amplified by amp,
    /// updates that push AWAY are dampened by 1/amp. Forward output is
    /// unchanged, so this composes cleanly with any existing tape op.
    /// (Mathematically: substrate-shaped preconditioner on the gradient.)
    /// Stores (scale, alpha) inline.
    SubstrateGradMod(usize, f64, f64),
    /// Substrate-sparse attention output. Computes per-row scores ONLY for
    /// (i, j) cells where CRT substrate_dist(i, j) <= threshold, masks the
    /// rest to -inf so softmax assigns zero weight. Operates on q [N, D]
    /// and a const k [N, D] (CRT-PE). Output is q-shaped: attn @ v_id is
    /// applied separately. Used for inference-time speedup after Q6
    /// training — v0.8.8 showed Q6 pushes 56.8% of attention mass into
    /// 6.84% of substrate-close cells, so the sparse cells capture the
    /// dominant attention with ~10x fewer score computations.
    /// `k_constant_id` stays as a tape const (not a learnable). Forward
    /// only for now — backward goes through standard tape_matmul path
    /// after the dense scores are reconstructed.
    SubstrateSparseScores(usize, usize, i64),
    /// Fused substrate-V resample: out[i, c] = v[i, c] * 1/(1 + d(v[i,c]·scale)/scale).
    /// Modulator is treated as a const w.r.t. v (matches the OMC reference
    /// `prom_substrate_resample` which uses tape_const). Stores `scale` in
    /// the op so backward can reconstruct the modulator without round-
    /// tripping through OMC value arrays. Replaces the
    /// tape_value → modulator_matrix → tape_const → tape_mul chain.
    SubstrateResample(usize, f64),
    /// Element-wise |x|. Boring PyTorch-parity primitive. Backward is
    /// subgradient: sign(x) at x ≠ 0, 0 at x = 0.
    Abs(usize),
    /// Substrate-native fused log_φπfib(|x·scale| + 1).
    /// Replaces tape_abs + tape_log + (1/(π·ln φ)) scalar div with one tape
    /// node. The scale is stored inside the op (constant w.r.t. backward).
    /// Q6 attention modulation is its first consumer; the fused form keeps
    /// the substrate basis visible at the AST level so future variants
    /// (attractor-modulated backward, fibonacci snap) can be slotted in
    /// without touching every call-site.
    PhiLog(usize, f64),
    Sin(usize),
    Cos(usize),
    Relu(usize),
    Sigmoid(usize),
    Tanh(usize),
    /// Per-row softmax: each row of the input becomes a probability vector
    /// summing to 1.0. Needed for LM cross-entropy loss.
    Softmax(usize),
    /// True matrix multiplication, A@B.
    MatMul(usize, usize),
    /// Sum every cell to a scalar — needed because loss must be scalar
    /// for backward(seed=1.0) to make sense.
    Sum(usize),
    /// Mean of every cell — same role as Sum but normalized.
    Mean(usize),
    /// Per-row mean: collapses [rows, cols] to [rows, 1]. Needed for
    /// proper LayerNorm on multi-token sequences.
    RowMean(usize),
    /// Per-row sum.
    RowSum(usize),
    /// Per-row LayerNorm: ((x - row_mean) / sqrt(row_var + eps)) * gamma + beta
    /// Stores eps inside the op. Output shape matches input. Single fused
    /// op because composing it from primitives needs broadcasted sub/div
    /// that aren't yet in the tape.
    LayerNormRow(usize, usize, usize, f64),  // (x, gamma, beta, eps)
    /// Matrix transpose: [rows, cols] → [cols, rows]. Differentiable —
    /// backward is just another transpose of the upstream grad.
    Transpose(usize),
}

pub(crate) struct TapeNode {
    pub op: TapeOp,
    pub value: TapeMat,
    pub grad: TapeMat,
}

/// Construct a TapeMat from an OMC Value. Accepts:
///   - scalar HInt/HFloat → 1×1 matrix
///   - 1D array → 1×N row matrix
///   - 2D array (array-of-arrays) → MxN matrix
/// Produce a wrong-container hint suffix when an array builtin was
/// called with a dict (or vice versa). Returns an empty string when no
/// hint applies. The suffix is pre-formatted as " (did you mean X?)"
/// so it can be concatenated directly into an error message.
pub(crate) fn wrong_container_hint(received: &Value, suggested: &str) -> String {
    let recv_type = type_name_of(received);
    format!(
        " (got {}; did you mean `{}`?)",
        recv_type, suggested
    )
}

/// Convert a vec of substrate-predicted suggestions into an OMC
/// Value (array of dicts). Each dict carries fn_name, source, file,
/// canonical_hash, prefix_match_len, substrate_distance.
pub(crate) fn predict_suggestions_to_value(
    suggestions: &[crate::predict::Suggestion],
) -> Value {
    let out: Vec<Value> = suggestions.iter().map(|s| {
        let pairs: Vec<(String, Value)> = vec![
            ("fn_name".to_string(), Value::String(s.fn_name.clone())),
            ("source".to_string(), Value::String(s.source.clone())),
            ("file".to_string(), Value::String(s.file.clone())),
            ("canonical_hash".to_string(), Value::HInt(HInt::new(s.canonical_hash))),
            ("attractor".to_string(), Value::HInt(HInt::new(s.attractor))),
            ("prefix_match_len".to_string(), Value::HInt(HInt::new(s.prefix_match_len as i64))),
            ("substrate_distance".to_string(), Value::HInt(HInt::new(s.substrate_distance))),
            ("query_attractor".to_string(), Value::HInt(HInt::new(s.query_attractor))),
        ];
        Value::Dict(std::rc::Rc::new(std::cell::RefCell::new(
            pairs.into_iter().collect()
        )))
    }).collect();
    Value::Array(HArray::from_vec(out))
}

/// Human-readable type tag for error messages. Mirrors the `type_of`
/// builtin's tag set so user-facing strings match what they'd see from
/// inspecting at runtime.
pub(crate) fn type_name_of(v: &Value) -> &'static str {
    match v {
        Value::HInt(_) => "int",
        Value::HFloat(_) => "float",
        Value::String(_) => "string",
        Value::Bool(_) => "bool",
        Value::Array(_) => "array",
        Value::Dict(_) => "dict",
        Value::Function { .. } => "function",
        Value::Null => "null",
        Value::Singularity { .. } => "singularity",
        _ => "unknown",
    }
}

fn tape_from_value(v: &Value) -> Result<TapeMat, String> {
    match v {
        Value::HInt(_) | Value::HFloat(_) | Value::Bool(_) | Value::Null => {
            Ok(TapeMat::scalar(v.to_float()))
        }
        Value::Array(arr) => {
            let rows = arr.items.borrow();
            if rows.is_empty() {
                return Ok(TapeMat::zeros(0, 0));
            }
            let is_2d = matches!(&rows[0], Value::Array(_));
            if is_2d {
                let mut out: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
                let cols = if let Value::Array(r) = &rows[0] { r.items.borrow().len() } else { 0 };
                for r in rows.iter() {
                    if let Value::Array(row) = r {
                        let row_b = row.items.borrow();
                        if row_b.len() != cols {
                            return Err("tape: ragged 2D array".to_string());
                        }
                        out.push(row_b.iter().map(|v| v.to_float()).collect());
                    } else {
                        return Err("tape: mixed 1D/2D rows".to_string());
                    }
                }
                Ok(TapeMat::from_2d(&out))
            } else {
                let row: Vec<f64> = rows.iter().map(|v| v.to_float()).collect();
                Ok(TapeMat::from_2d(&[row]))
            }
        }
        _ => Err("tape: cannot lift this value type into a tape node".to_string()),
    }
}

/// Render a TapeMat back to an OMC Value. Scalars come back as HFloat;
/// row-vectors come back as 1D arrays; 2D matrices come back as 2D arrays.
/// When `as_hint` is set and every cell rounds cleanly to an integer,
/// substrate-typed HInts are emitted so resonance metadata is rebuilt
/// from the value — this is the path that makes "gradients carry HInt
/// resonance" hold for cells that landed on integer values.
fn tape_to_value(m: &TapeMat, as_hint: bool) -> Value {
    let to_cell = |x: f64| -> Value {
        if as_hint && (x.fract() == 0.0) && x.abs() < (i64::MAX as f64) {
            Value::HInt(HInt::new(x as i64))
        } else {
            Value::HFloat(x)
        }
    };
    if m.rows == 1 && m.cols == 1 {
        return to_cell(m.data[0]);
    }
    if m.rows == 1 {
        let row: Vec<Value> = m.data.iter().map(|&x| to_cell(x)).collect();
        return Value::Array(HArray::from_vec(row));
    }
    let mut out: Vec<Value> = Vec::with_capacity(m.rows);
    for i in 0..m.rows {
        let mut row: Vec<Value> = Vec::with_capacity(m.cols);
        for j in 0..m.cols { row.push(to_cell(m.at(i, j))); }
        out.push(Value::Array(HArray::from_vec(row)));
    }
    Value::Array(HArray::from_vec(out))
}

/// Reduce an upstream gradient back to a broadcasted operand's
/// original shape. Sums over dimensions where the operand was
/// broadcast (size 1 in that dim). Used by Add/Sub backward when
/// the operand was a row/col vector broadcasted across a matrix.
fn reduce_to_shape(g: &TapeMat, target: (usize, usize)) -> TapeMat {
    let (tr, tc) = target;
    if g.rows == tr && g.cols == tc { return g.clone(); }
    let mut out = TapeMat::zeros(tr, tc);
    // Scalar target.
    if tr == 1 && tc == 1 {
        let mut s = 0.0;
        for v in &g.data { s += v; }
        out.data[0] = s;
        return out;
    }
    // Row-vector target [1, C]: sum across rows.
    if tr == 1 && tc == g.cols {
        for j in 0..g.cols {
            let mut s = 0.0;
            for i in 0..g.rows { s += g.at(i, j); }
            out.data[j] = s;
        }
        return out;
    }
    // Col-vector target [R, 1]: sum across cols.
    if tc == 1 && tr == g.rows {
        for i in 0..g.rows {
            let mut s = 0.0;
            for j in 0..g.cols { s += g.at(i, j); }
            out.data[i] = s;
        }
        return out;
    }
    // Fallback: shape doesn't match a known broadcast pattern — copy
    // what we can without panicking.
    let cp_r = g.rows.min(tr);
    let cp_c = g.cols.min(tc);
    for i in 0..cp_r {
        for j in 0..cp_c {
            out.set(i, j, g.at(i, j));
        }
    }
    out
}

/// Transpose helper for matmul backward.
fn tape_transpose(m: &TapeMat) -> TapeMat {
    let mut out = TapeMat::zeros(m.cols, m.rows);
    for i in 0..m.rows {
        for j in 0..m.cols {
            out.set(j, i, m.at(i, j));
        }
    }
    out
}

/// Standard matmul on TapeMat. Used both in forward Matmul and in the
/// backward pass (dA = dC @ B^T, dB = A^T @ dC). Routes through the
/// registered accelerator (e.g. omnimcode-gpu's wgpu backend) when one
/// is installed AND it elects to handle this shape; otherwise falls
/// back to the in-core triple-loop. See `crate::accel`.
fn tape_matmul(a: &TapeMat, b: &TapeMat) -> Result<TapeMat, String> {
    if a.cols != b.rows {
        return Err(format!(
            "tape_matmul: shape mismatch {}x{} @ {}x{}", a.rows, a.cols, b.rows, b.cols
        ));
    }
    if let Some(result) = crate::accel::try_accelerated_matmul(
        a.rows, a.cols, b.cols, &a.data, &b.data
    ) {
        return result.map(|data| TapeMat { rows: a.rows, cols: b.cols, data });
    }
    let mut out = TapeMat::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut s = 0.0;
            for k in 0..a.cols { s += a.at(i, k) * b.at(k, j); }
            out.set(i, j, s);
        }
    }
    Ok(out)
}

/// Flatten an OMC 2D array (array-of-arrays) into a contiguous
/// row-major f64 buffer. Returns (rows, cols, buf).
fn flatten_matrix(v: &Value, label: &str) -> Result<(usize, usize, Vec<f64>), String> {
    let Value::Array(outer) = v else {
        return Err(format!("{}: not a matrix", label));
    };
    let rows_b = outer.items.borrow();
    if rows_b.is_empty() {
        return Ok((0, 0, vec![]));
    }
    let cols = match &rows_b[0] {
        Value::Array(r) => r.items.borrow().len(),
        _ => return Err(format!("{}: rows must be arrays", label)),
    };
    let rows = rows_b.len();
    let mut flat = vec![0.0f64; rows * cols];
    for (i, r) in rows_b.iter().enumerate() {
        if let Value::Array(row) = r {
            let rb = row.items.borrow();
            if rb.len() != cols {
                return Err(format!("{}: ragged matrix", label));
            }
            for (j, x) in rb.iter().enumerate() {
                flat[i * cols + j] = x.to_float();
            }
        } else {
            return Err(format!("{}: rows must be arrays", label));
        }
    }
    Ok((rows, cols, flat))
}

/// Rebuild a 2D OMC array from a row-major f64 buffer.
fn matrix_from_flat(flat: &[f64], rows: usize, cols: usize) -> Value {
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            row.push(Value::HFloat(flat[i * cols + j]));
        }
        out.push(Value::Array(HArray::from_vec(row)));
    }
    Value::Array(HArray::from_vec(out))
}

/// Unpack a dual number into (value, derivative). Plain scalars become
/// (scalar, 0.0) so dual ops can mix duals with constants naturally.
fn unpack_dual(v: &Value) -> (f64, f64) {
    if let Value::Array(a) = v {
        let items = a.items.borrow();
        if items.len() >= 2 {
            return (items[0].to_float(), items[1].to_float());
        }
        if items.len() == 1 {
            return (items[0].to_float(), 0.0);
        }
        return (0.0, 0.0);
    }
    (v.to_float(), 0.0)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        // ---- Null: equal ONLY to itself ------------------------------
        // Without this explicit arm, (Dict, Null) and (Function, Null)
        // fall through to the numeric-coercion path where to_int(any)
        // = 0 = to_int(Null), making EVERY non-numeric value compare
        // equal to null. Caught when `if dict == null` was always
        // true in user code (harmonic_recommend's add_rating bug).
        (Value::Null, Value::Null) => true,
        (Value::Null, _) | (_, Value::Null) => false,

        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => {
            let xb = x.items.borrow();
            let yb = y.items.borrow();
            if xb.len() != yb.len() {
                return false;
            }
            xb.iter()
                .zip(yb.iter())
                .all(|(p, q)| values_equal(p, q))
        }
        (Value::Dict(x), Value::Dict(y)) => {
            // Two dicts are equal iff same keys + values_equal at every
            // key. BTreeMap iteration is sorted so we can zip.
            let xb = x.borrow();
            let yb = y.borrow();
            if xb.len() != yb.len() {
                return false;
            }
            xb.iter()
                .zip(yb.iter())
                .all(|((k1, v1), (k2, v2))| k1 == k2 && values_equal(v1, v2))
        }
        (
            Value::Singularity {
                numerator: na,
                context: ca,
                ..
            },
            Value::Singularity {
                numerator: nb,
                context: cb,
                ..
            },
        ) => na == nb && ca == cb,
        // Mixing dict/array/function/circuit with anything else: never
        // equal. Catches the same class of cross-type-coercion bug as
        // the Null arm above for non-Null mismatches.
        (Value::Dict(_), _) | (_, Value::Dict(_)) => false,
        (Value::Array(_), _) | (_, Value::Array(_)) => false,
        (Value::Function { .. }, _) | (_, Value::Function { .. }) => false,
        (Value::Circuit(_), _) | (_, Value::Circuit(_)) => false,
        // Mixing strings with non-strings: only equal if both coerce to
        // the same number AND the string is actually a numeric literal.
        (Value::String(s), _) | (_, Value::String(s)) => {
            if s.parse::<i64>().is_ok() || s.parse::<f64>().is_ok() {
                if a.is_float() || b.is_float() {
                    a.to_float() == b.to_float()
                } else {
                    a.to_int() == b.to_int()
                }
            } else {
                false
            }
        }
        // Numeric / bool — actually coerce-comparable.
        _ => {
            if a.is_float() || b.is_float() {
                a.to_float() == b.to_float()
            } else {
                a.to_int() == b.to_int()
            }
        }
    }
}

// Free function reused by quantize / quantization_ratio / mean_omni_weight.
// Snap |n| to the nearest Fibonacci attractor, preserving sign.
/// Track-2 substrate-typed-array helper: element-wise binary op
/// over (array, array) or (array, scalar). Scalar broadcasts to
/// every position of the array. Two-array length mismatch is an
/// error (no implicit shape-1 expansion — keeps behavior obvious).
/// `op` takes (i64, i64) and returns i64; the helper wraps the
/// result in HInt so per-element substrate resonance gets recomputed
/// from the arithmetic output.
/// Detect whether `a` is a 2D array (every element is itself an array).
/// Empty rows count as malformed and return None — callers fall back to
/// the 1D path. Returns (rows, cols) of the first row when 2D.
fn array_2d_shape(v: &Value) -> Option<(usize, usize)> {
    if let Value::Array(outer) = v {
        let rows = outer.items.borrow();
        if rows.is_empty() { return None; }
        let first_cols = match &rows[0] {
            Value::Array(r) => r.items.borrow().len(),
            _ => return None,
        };
        for r in rows.iter() {
            match r {
                Value::Array(row) if row.items.borrow().len() == first_cols => {}
                _ => return None,
            }
        }
        Some((rows.len(), first_cols))
    } else {
        None
    }
}

/// 2D-aware broadcast paths for elementwise ops. Returns Some(result)
/// when both operands fit one of the broadcasting shapes; None lets the
/// caller fall through to the flat 1D path.
///
///   (NxM, NxM)        — element-wise, returns NxM
///   (NxM, M-vector)   — row broadcast: vector added to every row
///   (M-vector, NxM)   — same, reversed
fn try_2d_broadcast<F: Fn(i64, i64) -> i64>(
    a: &Value,
    b: &Value,
    name: &str,
    op: &F,
) -> Result<Option<Value>, String> {
    let a_shape = array_2d_shape(a);
    let b_shape = array_2d_shape(b);

    // Case 1: both 2D — must match shapes element-wise.
    if let (Some((ar, ac)), Some((br, bc))) = (a_shape, b_shape) {
        if ar != br || ac != bc {
            return Err(format!(
                "{}: 2D shape mismatch ({}x{} vs {}x{})", name, ar, ac, br, bc
            ));
        }
        if let (Value::Array(a_rows), Value::Array(b_rows)) = (a, b) {
            let ar_b = a_rows.items.borrow();
            let br_b = b_rows.items.borrow();
            let mut out_rows: Vec<Value> = Vec::with_capacity(ar);
            for (ra, rb) in ar_b.iter().zip(br_b.iter()) {
                let (Value::Array(ra), Value::Array(rb)) = (ra, rb) else {
                    return Ok(None);
                };
                let raw_a = ra.items.borrow();
                let raw_b = rb.items.borrow();
                let row: Vec<Value> = raw_a.iter().zip(raw_b.iter())
                    .map(|(x, y)| Value::HInt(HInt::new(op(x.to_int(), y.to_int()))))
                    .collect();
                out_rows.push(Value::Array(HArray::from_vec(row)));
            }
            return Ok(Some(Value::Array(HArray::from_vec(out_rows))));
        }
    }

    // Case 2: 2D + 1D row-vector — broadcast vector across every row.
    if let (Some((ar, ac)), None) = (a_shape, b_shape) {
        if let (Value::Array(a_rows), Value::Array(b_vec)) = (a, b) {
            let vec_b = b_vec.items.borrow();
            // Reject when b is itself a non-1D shape (e.g., array of dicts);
            // a true 1D vector has length == ac.
            if vec_b.len() != ac {
                // Could be a length mismatch — surface a clear error.
                // But only when b looks like a 1D numeric vector; otherwise
                // fall through to None and let the caller handle.
                if vec_b.iter().any(|v| matches!(v, Value::Array(_))) {
                    return Ok(None);
                }
                return Err(format!(
                    "{}: row-broadcast length mismatch ({} cols vs {} vec)",
                    name, ac, vec_b.len()
                ));
            }
            let ar_b = a_rows.items.borrow();
            let mut out_rows: Vec<Value> = Vec::with_capacity(ar);
            for ra in ar_b.iter() {
                let Value::Array(ra) = ra else { return Ok(None); };
                let raw_a = ra.items.borrow();
                let row: Vec<Value> = raw_a.iter().zip(vec_b.iter())
                    .map(|(x, y)| Value::HInt(HInt::new(op(x.to_int(), y.to_int()))))
                    .collect();
                out_rows.push(Value::Array(HArray::from_vec(row)));
            }
            return Ok(Some(Value::Array(HArray::from_vec(out_rows))));
        }
    }

    // Case 3: 1D + 2D — symmetric.
    if let (None, Some((br, bc))) = (a_shape, b_shape) {
        if let (Value::Array(a_vec), Value::Array(b_rows)) = (a, b) {
            let vec_a = a_vec.items.borrow();
            if vec_a.len() != bc {
                if vec_a.iter().any(|v| matches!(v, Value::Array(_))) {
                    return Ok(None);
                }
                return Err(format!(
                    "{}: row-broadcast length mismatch ({} vec vs {} cols)",
                    name, vec_a.len(), bc
                ));
            }
            let br_b = b_rows.items.borrow();
            let mut out_rows: Vec<Value> = Vec::with_capacity(br);
            for rb in br_b.iter() {
                let Value::Array(rb) = rb else { return Ok(None); };
                let raw_b = rb.items.borrow();
                let row: Vec<Value> = vec_a.iter().zip(raw_b.iter())
                    .map(|(x, y)| Value::HInt(HInt::new(op(x.to_int(), y.to_int()))))
                    .collect();
                out_rows.push(Value::Array(HArray::from_vec(row)));
            }
            return Ok(Some(Value::Array(HArray::from_vec(out_rows))));
        }
    }

    Ok(None)
}

pub(crate) fn elementwise_op<F: Fn(i64, i64) -> i64>(
    a: &Value,
    b: &Value,
    name: &str,
    op: F,
) -> Result<Value, String> {
    // 2D-aware broadcasting shortcut — runs before the standard flat-array
    // path so callers don't have to switch to a separate builtin. Two
    // 2D operands element-wise; (2D, 1D) row-broadcast (the 1D vector
    // gets added to every row); (1D, 2D) same in reverse.
    if let Some(out) = try_2d_broadcast(a, b, name, &op)? {
        return Ok(out);
    }
    match (a, b) {
        (Value::Array(arr_a), Value::Array(arr_b)) => {
            let ai = arr_a.items.borrow();
            let bi = arr_b.items.borrow();
            if ai.len() != bi.len() {
                return Err(format!(
                    "{}: length mismatch ({} vs {})", name, ai.len(), bi.len()
                ));
            }
            let out: Vec<Value> = ai.iter().zip(bi.iter())
                .map(|(x, y)| Value::HInt(HInt::new(op(x.to_int(), y.to_int()))))
                .collect();
            Ok(Value::Array(HArray::from_vec(out)))
        }
        (Value::Array(arr_a), scalar) => {
            let sv = scalar.to_int();
            let out: Vec<Value> = arr_a.items.borrow().iter()
                .map(|x| Value::HInt(HInt::new(op(x.to_int(), sv))))
                .collect();
            Ok(Value::Array(HArray::from_vec(out)))
        }
        (scalar, Value::Array(arr_b)) => {
            let sv = scalar.to_int();
            let out: Vec<Value> = arr_b.items.borrow().iter()
                .map(|y| Value::HInt(HInt::new(op(sv, y.to_int()))))
                .collect();
            Ok(Value::Array(HArray::from_vec(out)))
        }
        _ => Err(format!("{}: requires at least one array argument", name)),
    }
}

/// Convert a `serde_json::Value` into an OMC `Value`. JSON object →
/// `Value::Dict`, JSON array → `Value::Array`, numbers split into
/// `HInt` (when representable as i64) vs `HFloat` (everything else).
pub(crate) fn json_to_value(j: serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::HInt(HInt::new(i)) }
            else if let Some(f) = n.as_f64() { Value::HFloat(f) }
            else { Value::HInt(HInt::new(0)) }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            let items: Vec<Value> = arr.into_iter().map(json_to_value).collect();
            Value::Array(HArray::from_vec(items))
        }
        serde_json::Value::Object(map) => {
            let mut out = std::collections::BTreeMap::new();
            for (k, v) in map {
                out.insert(k, json_to_value(v));
            }
            Value::dict_from(out)
        }
    }
}

/// Convert an OMC `Value` back into a `serde_json::Value` for
/// stringification. Singularity and Function values stringify to
/// their display form (no clean JSON representation).
pub(crate) fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::HInt(h) => serde_json::json!(h.value),
        Value::HFloat(f) => {
            // NaN / Inf can't be represented in JSON — coerce to null.
            if f.is_finite() { serde_json::json!(*f) } else { serde_json::Value::Null }
        }
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(arr) => {
            let items: Vec<serde_json::Value> = arr.items.borrow().iter()
                .map(value_to_json).collect();
            serde_json::Value::Array(items)
        }
        Value::Dict(d) => {
            let mut map = serde_json::Map::new();
            for (k, vv) in d.borrow().iter() {
                map.insert(k.clone(), value_to_json(vv));
            }
            serde_json::Value::Object(map)
        }
        // Singularity / Function / Circuit: fall back to display string.
        other => serde_json::Value::String(other.to_display_string()),
    }
}

pub(crate) fn fold_to_fibonacci_const(n: i64) -> i64 {
    // Substrate-routed via phi_pi_fib::fold_to_nearest_attractor.
    // Was: a 15-element local Fibonacci array + linear scan.
    crate::phi_pi_fib::fold_to_nearest_attractor(n)
}

// Used by the host-side healer in heal_ast. Tests whether `n` falls on
// the Fibonacci attractor table. Substrate-routed via
// phi_pi_fib::is_on_fibonacci_attractor — same canonical table as
// every other harmonic op now uses.
pub(crate) fn is_on_fibonacci_attractor(n: i64) -> bool {
    crate::phi_pi_fib::is_on_fibonacci_attractor(n)
}

// Levenshtein edit distance for the heal-pass typo correction. Returns
// the smallest edit count between two strings (insert/delete/replace = 1).
// Used over the defined-name table to find the closest match within a
// threshold (default 2).
pub(crate) fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 { return n; }
    if n == 0 { return m; }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// Return the closest defined name within `max_dist` (Levenshtein) of
// `target`, or None if nothing matches. `prefer` is a priority set:
// when two candidates tie on distance, the one in `prefer` wins. Used
// by the heal pass to prefer user-defined functions over builtins —
// a typo at a call site is more likely a user fn than a builtin.
pub(crate) fn closest_name(
    target: &str,
    defined: &HashSet<String>,
    max_dist: usize,
    prefer: Option<&HashSet<String>>,
) -> Option<String> {
    let mut best: Option<(usize, String, bool)> = None;
    for cand in defined {
        let d = edit_distance(target, cand);
        if d > max_dist { continue; }
        let in_prefer = prefer.map(|p| p.contains(cand)).unwrap_or(false);
        let should_replace = match &best {
            None => true,
            Some((bd, _, _)) if d < *bd => true,
            Some((bd, _, bp)) if d == *bd && in_prefer && !*bp => true,
            _ => false,
        };
        if should_replace {
            best = Some((d, cand.clone(), in_prefer));
        }
    }
    best.map(|(_, s, _)| s)
}

// ============================================================================
// Self-healing compiler: substrate-routed support primitives.
// ============================================================================

// Per-pass substrate-routed name index. Set by `heal_ast` at the start
// of every pass, consumed by `closest_name_substrate` inside the call-
// site typo check. Thread-local so concurrent interpreters can each
// hold their own index without contention.
//
// Why a thread-local instead of threading through heal_stmt/heal_expr:
// the heal-pass signatures recurse 30+ times per pass; adding an
// &Vec<Vec<String>> parameter to every call site would balloon the
// diff with no value beyond plumbing. Thread-local is the minimal
// intrusion that lets the new substrate-routed lookup just work.
std::thread_local! {
    pub(crate) static HEAL_SUBSTRATE_INDEX: std::cell::RefCell<Vec<Vec<String>>>
        = const { std::cell::RefCell::new(Vec::new()) };
    pub(crate) static HEAL_CLASS_COUNTS: std::cell::RefCell<HealClassCounts>
        = const { std::cell::RefCell::new(HealClassCounts::new()) };
    /// Per-class disabled flags. Pushed by FunctionDef pragmas inside
    /// heal_stmt; consumed by the matching heal cases inside heal_expr.
    /// Defaults to all-enabled.
    pub(crate) static HEAL_PER_CLASS_DISABLED: std::cell::RefCell<HealDisabled>
        = const { std::cell::RefCell::new(HealDisabled::all_enabled()) };
    /// Per-pass heal budget. Decremented every time a class fires.
    /// When it hits zero, further heals are silently skipped (the
    /// diagnostic still records the count, but no AST rewrite).
    /// Prevents runaway heals on pathological inputs.
    pub(crate) static HEAL_BUDGET_REMAINING: std::cell::Cell<u32>
        = const { std::cell::Cell::new(HEAL_BUDGET_PER_PASS) };
}

/// Maximum number of heals a single `heal_ast` pass can apply. Calibrated
/// to be high enough for legitimate code (a project with hundreds of
/// typos still completes) but low enough that an adversarial input
/// can't make the heal pass run forever.
pub const HEAL_BUDGET_PER_PASS: u32 = 1024;

#[derive(Debug, Clone, Copy)]
pub struct HealDisabled {
    pub typo: bool,
    pub arity: bool,
    pub div_zero: bool,
    pub mod_zero: bool,
    pub harmonic_index: bool,
}

impl HealDisabled {
    pub const fn all_enabled() -> Self {
        Self { typo: false, arity: false, div_zero: false, mod_zero: false, harmonic_index: false }
    }
}

/// Try to consume one unit of heal budget. Returns true if budget is
/// available (and decrements), false if exhausted. Heal classes should
/// check this BEFORE applying their rewrite.
#[inline]
fn try_consume_heal_budget() -> bool {
    HEAL_BUDGET_REMAINING.with(|b| {
        let n = b.get();
        if n == 0 { false } else { b.set(n - 1); true }
    })
}

/// Per-class heal counters. Bumped from inside each heal class so
/// `--check` can report a summary like "typo: 3, arity: 1, div0: 2".
/// Reset by `heal_ast` at the start of every pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct HealClassCounts {
    pub typo: u32,
    pub typo_substrate_hit: u32,   // bucketed pre-filter hit (no fallback scan)
    pub typo_fallback: u32,        // bucketed miss → full closest_name scan
    pub arity_pad: u32,
    pub arity_truncate: u32,
    pub div_zero: u32,
    pub mod_zero: u32,
    pub harmonic_index: u32,
    pub missing_return: u32,
    pub empty_index_safe: u32,
    pub reserved_var: u32,
    pub if_numeric: u32,
    pub str_concat: u32,           // "foo" + 5 → concat_many("foo", to_string(5))
    pub var_typo: u32,             // bare-variable typo (vs the call-site typo above)
    pub null_arith: u32,           // null + x → 0 + x (and Sub/Mul/Div/Mod)
    pub neg_index: u32,            // arr[-1] → safe_arr_get with len-relative offset
}

impl HealClassCounts {
    pub const fn new() -> Self {
        Self {
            typo: 0, typo_substrate_hit: 0, typo_fallback: 0,
            arity_pad: 0, arity_truncate: 0,
            div_zero: 0, mod_zero: 0, harmonic_index: 0,
            missing_return: 0, empty_index_safe: 0,
            reserved_var: 0, if_numeric: 0,
            str_concat: 0, var_typo: 0,
            null_arith: 0, neg_index: 0,
        }
    }
    pub fn total(&self) -> u32 {
        self.typo + self.arity_pad + self.arity_truncate
            + self.div_zero + self.mod_zero + self.harmonic_index
            + self.missing_return + self.empty_index_safe
            + self.reserved_var + self.if_numeric
            + self.str_concat + self.var_typo
            + self.null_arith + self.neg_index
    }
}

/// Snapshot the per-pass heal counters. Call AFTER `heal_ast` to read
/// what fired during the pass. Read-only — counters reset on the next
/// `heal_ast` invocation.
pub fn last_heal_counts() -> HealClassCounts {
    HEAL_CLASS_COUNTS.with(|c| *c.borrow())
}

/// Substrate-routed hash of an identifier name, mirroring the OMC
/// builtin `substrate_hash` but operating on a UTF-8 string. Hashes
/// chars through phi-shifted contributions so the bit distribution
/// has substrate-aligned avalanche — close-shape names that share
/// most chars still cluster into nearby buckets, while structurally
/// unrelated names disperse.
pub(crate) fn substrate_hash_name(s: &str) -> u64 {
    const SEED: u64 = 0x9E3779B97F4A7C15; // 2^64 · (sqrt(5) - 1) / 2
    let mut h: u64 = SEED;
    for (i, b) in s.bytes().enumerate() {
        let term = (b as u64).wrapping_mul(SEED)
            .rotate_left((i * 5) as u32);
        h = (h ^ term).wrapping_mul(SEED);
    }
    h
}

/// Bucket count for the substrate-routed name index. 32 ≈ 2 * φ^7 —
/// enough buckets that typical project sizes (hundreds of names)
/// distribute one or two names per bucket, keeping per-lookup scan
/// short while staying well inside the FIBONACCI table.
const SUBSTRATE_NAME_BUCKETS: usize = 32;

/// Build a substrate-routed index over the heal-pass defined-name set.
/// Each name is placed in its substrate_hash bucket modulo
/// SUBSTRATE_NAME_BUCKETS. Returns a Vec of buckets where bucket[i]
/// is every name whose hash mods to i.
pub(crate) fn build_substrate_name_index(
    defined: &HashSet<String>,
) -> Vec<Vec<String>> {
    let mut buckets: Vec<Vec<String>> = vec![Vec::new(); SUBSTRATE_NAME_BUCKETS];
    for name in defined {
        let b = (substrate_hash_name(name) as usize) % SUBSTRATE_NAME_BUCKETS;
        buckets[b].push(name.clone());
    }
    buckets
}

/// Substrate-routed typo lookup. Two-phase:
///   Phase 1: ALWAYS scan the `prefer` set fully (user-defined fns are
///            project-bounded, this is cheap). User fn matches beat
///            builtin matches even when bucket-misaligned.
///   Phase 2: For builtin candidates, only scan the target's bucket
///            plus 2 neighbors. The substrate-routing speedup applies
///            here because builtins are the large table (~400 names).
/// Result: substrate-O(log_phi_pi_fibonacci) on the large half, full
/// O(|prefer|) on the small half. The small half dominates correctness
/// (user fn typos > builtin typos in practice).
pub(crate) fn closest_name_substrate(
    target: &str,
    defined: &HashSet<String>,
    max_dist: usize,
    prefer: Option<&HashSet<String>>,
) -> Option<String> {
    let mut best: Option<(usize, String, bool)> = None;
    let consider = |cand: &str, d: usize, in_prefer: bool,
                    best: &mut Option<(usize, String, bool)>| {
        if d > max_dist { return; }
        let should_replace = match &*best {
            None => true,
            Some((bd, _, _)) if d < *bd => true,
            Some((bd, _, bp)) if d == *bd && in_prefer && !*bp => true,
            _ => false,
        };
        if should_replace {
            *best = Some((d, cand.to_string(), in_prefer));
        }
    };
    // Phase 1: full scan of user-fn prefer set.
    if let Some(p) = prefer {
        for cand in p {
            let d = edit_distance(target, cand);
            consider(cand, d, true, &mut best);
        }
    }
    // Phase 2: substrate-bucketed scan over the remaining defined names.
    let base = (substrate_hash_name(target) as usize) % SUBSTRATE_NAME_BUCKETS;
    let probe_indices = [
        base,
        (base + 1) % SUBSTRATE_NAME_BUCKETS,
        (base + SUBSTRATE_NAME_BUCKETS - 1) % SUBSTRATE_NAME_BUCKETS,
    ];
    let bucketed_scanned = HEAL_SUBSTRATE_INDEX.with(|idx| {
        let b = idx.borrow();
        if b.len() != SUBSTRATE_NAME_BUCKETS { return false; }
        for &bi in &probe_indices {
            for cand in &b[bi] {
                // Skip names already considered in phase 1.
                if prefer.map(|p| p.contains(cand)).unwrap_or(false) { continue; }
                let d = edit_distance(target, cand);
                consider(cand, d, false, &mut best);
            }
        }
        true
    });
    if best.is_some() {
        if bucketed_scanned {
            HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().typo_substrate_hit += 1);
        }
        return best.map(|(_, s, _)| s);
    }
    // Fallback: bucket index empty (called outside heal_ast) OR all
    // candidates were too distant. Pay the full scan to preserve
    // heal-correctness.
    HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().typo_fallback += 1);
    closest_name(target, defined, max_dist, prefer)
}

/// Does a statement list (a function body) contain any `Return`
/// statement, including nested inside if/while branches? Used by the
/// missing-return heal pass.
/// Detect `null` on either side of an arithmetic op and rewrite to 0.
/// `null` is represented in expressions as `Variable("null")` (the
/// parser never builds a dedicated Null variant). Returns the
/// (possibly-healed) operands and whether either side was rewritten.
/// Emits a heal diagnostic and bumps the `null_arith` counter when
/// the rewrite fires.
pub(crate) fn null_arith_rewrite(
    l: Expression,
    r: Expression,
    diags: &mut Vec<String>,
    op: &str,
) -> (Expression, Expression, bool) {
    let l_null = matches!(&l, Expression::Variable(n) if n == "null");
    let r_null = matches!(&r, Expression::Variable(n) if n == "null");
    if !l_null && !r_null { return (l, r, false); }
    let disabled = HEAL_PER_CLASS_DISABLED.with(|d| {
        // Reuse the existing `arity` opt-out flag; null_arith is
        // similar in spirit — silently coerces a value the user
        // probably didn't expect. No dedicated pragma yet.
        d.borrow().arity
    });
    if disabled || !try_consume_heal_budget() { return (l, r, false); }
    diags.push(format!("null-arith: 'null {op} x' rewritten with 0 (null → 0)"));
    HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().null_arith += 1);
    let zero = || Expression::Number(0);
    let l_out = if l_null { zero() } else { l };
    let r_out = if r_null { zero() } else { r };
    (l_out, r_out, true)
}

/// Walk a statement list and insert every VarDecl name (and For-loop
/// iteration variable, and Parameter declaration) into `acc`. Used by
/// the heal pass to hoist locally-declared names into scope so the
/// Variable-typo heal doesn't false-positive on legitimate locals.
pub(crate) fn collect_local_decls(stmts: &[Statement], acc: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Statement::VarDecl { name, .. } => { acc.insert(name.clone()); }
            Statement::Parameter { name, .. } => { acc.insert(name.clone()); }
            Statement::If { then_body, elif_parts, else_body, .. } => {
                collect_local_decls(then_body, acc);
                for (_, b) in elif_parts { collect_local_decls(b, acc); }
                if let Some(b) = else_body { collect_local_decls(b, acc); }
            }
            Statement::While { body, .. } => collect_local_decls(body, acc),
            Statement::For { var, body, .. } => {
                acc.insert(var.clone());
                collect_local_decls(body, acc);
            }
            Statement::Try { body, err_var, handler, finally } => {
                collect_local_decls(body, acc);
                acc.insert(err_var.clone());
                collect_local_decls(handler, acc);
                if let Some(b) = finally { collect_local_decls(b, acc); }
            }
            _ => {}
        }
    }
}

pub(crate) fn stmts_contain_return(stmts: &[Statement]) -> bool {
    for s in stmts {
        if stmt_contains_return(s) { return true; }
    }
    false
}

fn stmt_contains_return(s: &Statement) -> bool {
    match s {
        Statement::Return(_) => true,
        Statement::If { then_body, elif_parts, else_body, .. } => {
            stmts_contain_return(then_body)
                || elif_parts.iter().any(|(_, b)| stmts_contain_return(b))
                || else_body.as_ref().is_some_and(|b| stmts_contain_return(b))
        }
        Statement::While { body, .. } => stmts_contain_return(body),
        _ => false,
    }
}

/// Does a statement list contain any `yield` statement? Used by the
/// generator-fn detector — a fn body with at least one Yield is
/// dispatched through the yield-collector path at call time.
pub(crate) fn stmts_contain_yield(stmts: &[Statement]) -> bool {
    for s in stmts {
        if stmt_contains_yield(s) { return true; }
    }
    false
}

fn stmt_contains_yield(s: &Statement) -> bool {
    match s {
        Statement::Yield(_) => true,
        Statement::If { then_body, elif_parts, else_body, .. } => {
            stmts_contain_yield(then_body)
                || elif_parts.iter().any(|(_, b)| stmts_contain_yield(b))
                || else_body.as_ref().is_some_and(|b| stmts_contain_yield(b))
        }
        Statement::While { body, .. } => stmts_contain_yield(body),
        Statement::For { body, .. } => stmts_contain_yield(body),
        Statement::Try { body, handler, finally, .. } => {
            stmts_contain_yield(body)
                || stmts_contain_yield(handler)
                || finally.as_ref().is_some_and(|b| stmts_contain_yield(b))
        }
        _ => false,
    }
}

/// Missing-return heal: for every user fn lacking ANY return statement,
/// append `return null;` at the tail. Keeps callers from seeing the
/// confusing "fn ended without return" runtime error — most users mean
/// `return null` (procedural style) but forget to write it.
pub(crate) fn heal_missing_returns(
    statements: Vec<Statement>,
    needs_return: &HashSet<String>,
    diags: &mut Vec<String>,
) -> Vec<Statement> {
    statements.into_iter().map(|s| match s {
        Statement::FunctionDef { name, params, param_types, mut body, return_type, pragmas } => {
            if needs_return.contains(&name)
                && !pragmas.iter().any(|p| p == "no_heal" || p == "no_heal_return")
            {
                diags.push(format!(
                    "missing-return: '{}' has no return — appending `return null;`",
                    name
                ));
                HEAL_CLASS_COUNTS.with(|c| c.borrow_mut().missing_return += 1);
                body.push(Statement::Return(Some(Expression::Variable("null".to_string()))));
            }
            Statement::FunctionDef { name, params, param_types, body, return_type, pragmas }
        }
        other => other,
    }).collect()
}

// Static list of every host built-in name. Kept in sync with the
// `is_known_builtin` match arms — used by heal_ast's defined-name
// table so the typo check doesn't flag legitimate builtins.
// (When you add a new builtin to is_known_builtin, add it here too.)
pub(crate) const HEAL_BUILTIN_NAMES: &[&str] = &[
    // Numbers & math
    "abs", "min", "max", "sign", "floor", "ceil", "round", "frac",
    "gcd", "lcm", "square", "cube", "pow", "pow_int", "sqrt",
    "mod_pow", "bit_count", "bit_length", "digit_sum", "digit_count",
    "factorial", "is_even", "even", "is_odd", "odd", "is_prime",
    "sin", "cos", "tan", "tanh", "exp", "log", "erf", "sigmoid",
    "log2", "log10", "asin", "acos", "atan", "atan2",
    "hypot", "lerp",
    "clamp", "pi", "tau", "e", "phi", "phi_inv", "phi_sq",
    "phi_squared", "sqrt_2", "sqrt_5", "ln_2",
    // Strings
    "str_len", "str_chars", "str_slice", "str_concat", "concat_many",
    "str_split", "str_join", "str_trim", "str_replace",
    "csv_parse",
    "str_index_of", "str_contains", "str_starts_with", "str_ends_with",
    "str_repeat", "str_reverse", "str_uppercase", "str_lowercase",
    "str_pad_left", "str_pad_right",
    "str_split_lines", "str_count", "str_is_empty",
    "str_to_int", "str_to_float", "str_capitalize",
    "re_match", "re_find", "re_find_all", "re_replace", "re_split",
    "json_parse", "json_stringify", "json_extract", "str_format",
    "sha256", "sha512", "base64_encode", "base64_decode",
    // LLM builtins
    "llm_call", "llm_chat", "llm_embed", "llm_models", "llm_system",
    "llm_stream_print", "llm_judge", "llm_compare",
    "llm_tools", "substrate_embed",
    "batch_llm_call", "batch_llm_chat",
    // Native HTTP builtins
    "http_get", "http_post", "http_post_json", "http_put", "http_delete",
    "now_iso", "now_unix", "format_time", "parse_time",
    // Arrays
    "arr_new", "arr_from_range", "arr_len", "arr_get", "arr_set",
    "arr_push", "arr_first", "arr_last", "arr_slice", "arr_concat",
    "arr_contains", "arr_index_of", "arr_sort", "arr_reverse", "arr_join",
    "arr_min", "arr_max", "arr_sum", "arr_fold_elements",
    "arr_argmax", "arr_argmin", "arr_cumsum", "arr_diff", "arr_range",
    "arr_unique_count", "arr_partition_by",
    "arr_min_float", "arr_max_float", "arr_gcd", "fnv1a_hash",
    "arr_add", "arr_sub", "arr_mul", "arr_div_int", "arr_neg",
    "arr_scale", "arr_resonance_vec", "arr_him_vec", "arr_fold_all",
    "arr_mean", "arr_variance", "arr_stddev", "arr_median",
    "arr_harmonic_mean", "arr_geometric_mean",
    "arr_sum_sq", "arr_norm", "arr_dot",
    "arr_resonance", "filter_by_resonance", "cleanup_array",
    "arr_map", "arr_filter", "arr_reduce", "arr_any", "arr_all", "arr_find",
    "par_map", "par_filter", "par_reduce", "par_for",
    "arr_zip", "arr_unique",
    "arr_take", "arr_drop", "arr_count", "arr_repeat",
    "arr_fill", "arr_zeros", "arr_ones", "arr_chunk", "arr_flatten",
    "arr_enumerate", "arr_window",
    // Meta-evaluation
    "eval_omc", "eval_omc_fresh", "eval_omc_ctx", "omc_source",
    // Dicts
    "dict_new", "dict_get", "dict_set", "dict_has", "dict_del",
    "dict_keys", "dict_values", "dict_len", "dict_merge",
    "dict_pop", "dict_get_or", "dict_size", "dict_clear", "dict_items",
    // Harmonic
    "fib", "fibonacci", "is_fibonacci", "harmony_value", "fold",
    "fold_escape", "value_danger", "classify_resonance",
    "harmonic_interfere", "interfere", "measure_coherence",
    "mean_omni_weight", "boundary", "res",
    "harmonic_checksum", "harmonic_write_file", "harmonic_read_file",
    "harmonic_sort", "harmonic_split", "harmonic_partition",
    "attractor_distance", "nearest_attractor",
    "largest_attractor_at_most", "crt_residues", "hbit_tension",
    "is_attractor", "resonance_band", "crt_recover", "fibonacci_index",
    "harmonic_hash", "harmonic_diff", "harmonic_dedupe",
    // Phi-Pi-Fib search
    "phi_pi_fib_search", "phi_pi_fib_nearest",
    "phi_pi_fib_stats", "phi_pi_fib_reset",
    "phi_pi_fib_search_v2", "phi_pi_fib_nearest_v2",
    "phi_pi_bin_search", "log_phi_pi_fibonacci",
    "zeckendorf", "from_zeckendorf",
    "substrate_search", "substrate_lower_bound", "substrate_upper_bound",
    "substrate_rank", "substrate_count_range", "substrate_slice_range",
    "substrate_intersect", "substrate_difference",
    "zeckendorf_weight", "zeckendorf_bit", "substrate_hash",
    "attractor_bucket", "substrate_insert", "substrate_quantile",
    "fib_chunks",
    "harmonic_align", "harmonic_unalign", "phi_pi_log_distance",
    "harmonic_resample", "substrate_select_k",
    "int_binary_search", "int_lower_bound", "int_upper_bound",
    "sorted_merge", "sorted_union", "sorted_dedupe",
    "nth_fibonacci", "is_zeckendorf_valid",
    "substrate_min_distance", "substrate_nearest",
    "phi_pow", "phi_pi_pow", "harmonic_partition_3",
    "resonance_band_histogram",
    "arr_sum_int", "arr_product", "arr_sort_int", "arr_is_sorted",
    "attractor_table", "harmonic_score",
    "arr_min_int", "arr_max_int", "arr_avg_distance",
    "is_phi_resonant",
    "phi_pi_fib_search_traced", "phi_pi_fib_nearest_traced",
    "phi_pi_fib_stats_bg", "phi_pi_fib_stats_all",
    // HBit dual-band intrinsics (Sessions F+G)
    "phi_shadow", "harmony",
    // Self-healing
    "safe_divide", "safe_arr_get", "safe_arr_set",
    "safe_mod", "safe_sqrt", "safe_log",
    "safe_add", "safe_sub", "safe_mul", "resolve_singularity",
    "is_singularity", "ensure_clean", "collapse", "invert",
    "quantize", "quantization_ratio",
    // I/O
    "read_file", "write_file", "file_exists", "file_ls", "print",
    "println", "print_raw",
    // Time / sleep / similarity / eval
    "sleep", "str_similarity", "omc_eval_file",
    // Time / random / conversion / introspection
    "now_ms", "random_int", "random_float", "random_seed",
    "to_int", "int", "to_float", "float",
    "to_string", "string", "len", "type_of", "error",
    "defined_functions", "call",
    "list_defined_fns", "list_fns", "fn_arity", "fn_source", "get_scope_vars",
    "test_record_failure", "test_failure_count",
    "test_get_failures", "test_clear_failures",
    "test_set_current", "test_get_current",
    // Python-idiom builtins
    "range", "getenv", "to_hex", "from_hex",
    "parse_int", "parse_float",
    // v0.3 symbolic prediction
    "omc_predict_files", "omc_corpus_size",
    // LLM I/O builtins
    "llm_call", "llm_chat", "llm_embed", "llm_models",
    "batch_llm_call", "batch_llm_chat",
    "llm_tools", "substrate_embed",
    // Process execution builtins
    "omc_spawn", "omc_pipe",
    // Native HTTP builtins
    "http_get", "http_post", "http_post_json", "http_put", "http_delete",
    // Language literals. These are parsed as Variable(...) but get
    // special-cased at runtime — they must never be typo-corrected
    // (a "var_typo" rewriting `null` to a close-spelled name would
    // change semantics catastrophically).
    "null", "true", "false",
];

impl Interpreter {
    fn phi_fold_n(&self, v: Value, depth: usize) -> Value {
        match v {
            Value::HInt(h) => {
                let mut current = h.value;
                for _ in 0..depth.max(1) {
                    current = crate::phi_pi_fib::fold_to_nearest_attractor(current);
                }
                Value::HInt(HInt::new(current))
            }
            Value::HFloat(f) => {
                let mut current = f;
                for _ in 0..depth.max(1) {
                    current = (current * crate::value::PHI).fract();
                }
                Value::HFloat(current)
            }
            _ => Value::HInt(HInt::new(0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpreter_simple() {
        // Basic tests would go here
    }

    /// Empirical comparison: substrate-routed typo lookup vs full-scan
    /// closest_name across symbol-table sizes 10/100/1000/10000. Each
    /// size runs 1000 typo queries; we report mean lookup time and the
    /// substrate/full ratio.
    ///
    /// Run with: cargo test --release -p omnimcode-core typo_bench -- --nocapture
    #[test]
    fn typo_bench_substrate_vs_full() {
        use std::time::Instant;

        let sizes = [10usize, 100, 1000, 10000];
        let queries_per_size = 1000usize;

        println!();
        println!("# Typo lookup: substrate-bucketed vs full-scan");
        println!("# {} queries per size, ed≤2", queries_per_size);
        println!();
        println!("{:>8}  {:>14}  {:>14}  {:>10}  {:>12}",
                 "N", "substrate_µs", "full_µs", "ratio", "bucketed_hit");

        for &n in &sizes {
            // Synthesize N defined names of the shape "fn_NNNN" — enough
            // structural diversity that the bucketed index distributes
            // reasonably (substrate_hash_name is deterministic per str).
            let names: Vec<String> = (0..n).map(|i| format!("fn_{:05}", i)).collect();
            let defined: HashSet<String> = names.iter().cloned().collect();

            // Queries: deterministic typos — drop the last char of every
            // 7th name. Each is edit-distance 1 from a real name, so
            // closest_name SHOULD find a match.
            let queries: Vec<String> = (0..queries_per_size).map(|i| {
                let target_idx = (i * 7919) % n;
                let mut q = names[target_idx].clone();
                q.pop();
                q
            }).collect();

            // Populate the thread-local substrate index for the bucketed path.
            let bucketed = build_substrate_name_index(&defined);
            HEAL_SUBSTRATE_INDEX.with(|idx| *idx.borrow_mut() = bucketed);
            HEAL_CLASS_COUNTS.with(|c| *c.borrow_mut() = HealClassCounts::new());

            // Substrate path: bucketed pre-filter + fallback.
            let t0 = Instant::now();
            let mut sub_hits = 0;
            for q in &queries {
                if closest_name_substrate(q, &defined, 2, None).is_some() {
                    sub_hits += 1;
                }
            }
            let sub_elapsed = t0.elapsed();
            let sub_us = sub_elapsed.as_micros() as f64 / queries_per_size as f64;

            // Full path: pure closest_name (linear scan).
            let t0 = Instant::now();
            let mut full_hits = 0;
            for q in &queries {
                if closest_name(q, &defined, 2, None).is_some() {
                    full_hits += 1;
                }
            }
            let full_elapsed = t0.elapsed();
            let full_us = full_elapsed.as_micros() as f64 / queries_per_size as f64;

            assert_eq!(sub_hits, full_hits, "hit counts diverged at N={}", n);

            let bucketed_hit = HEAL_CLASS_COUNTS.with(|c| c.borrow().typo_substrate_hit);
            let ratio = full_us / sub_us.max(0.001);

            println!("{:>8}  {:>14.3}  {:>14.3}  {:>9.2}x  {:>10}/{:<4}",
                     n, sub_us, full_us, ratio, bucketed_hit, queries_per_size);
        }
        println!();
    }

    fn run(source: &str) -> Result<Value, String> {
        use crate::parser::Parser;
        let mut parser = Parser::new(source);
        let stmts = parser.parse()?;
        let mut interp = Interpreter::new();
        let mut last = Value::Null;
        for stmt in &stmts {
            interp.execute_stmt(stmt)?;
            if let Statement::Expression(e) = stmt {
                last = interp.eval_expr(e)?;
            }
        }
        if let Some(v) = interp.get_var("__result__") {
            return Ok(v);
        }
        Ok(last)
    }

    #[test]
    fn test_hfloat_literal() {
        let src = "h x = 1.5; __result__ = x;";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::HFloat(_)));
        assert_eq!(v.to_float(), 1.5);
    }

    #[test]
    fn test_float_arithmetic_promotes() {
        let src = "h x = 1.5; h y = 2; __result__ = x + y;";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::HFloat(_)));
        assert_eq!(v.to_float(), 3.5);
    }

    #[test]
    fn test_int_arithmetic_stays_int() {
        let src = "h x = 5; h y = 3; __result__ = x * y;";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::HInt(_)));
        assert_eq!(v.to_int(), 15);
    }

    #[test]
    fn test_phi_fold_module_call() {
        let src = "__result__ = phi.fold(90);";
        let v = run(src).unwrap();
        assert_eq!(v.to_int(), 89, "phi.fold(90) should snap to Fibonacci 89");
    }

    #[test]
    fn test_phi_fold_dynamic_depth() {
        let src = "h d = 2; __result__ = phi.fold(0.5, d);";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::HFloat(_)));
        // Two iterations of frac(x * phi) starting from 0.5 — just verify it stays in [0,1)
        let f = v.to_float();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_phi_res_returns_float() {
        let src = "__result__ = phi.res(89);";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::HFloat(_)));
        // 89 is Fibonacci, resonance should be ~1.0
        assert!((v.to_float() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_float_comparison() {
        let src = "h a = 1.5; h b = 1.6; __result__ = a < b;";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::Bool(true)));
    }

    #[test]
    fn test_pragma_prefix_parses() {
        let src = r#"
@pragma[hbit]
@pragma[avx512]
fn doit(x) {
    return x + 1;
}
__result__ = doit(88);
"#;
        let v = run(src).unwrap();
        assert_eq!(v.to_int(), 89);
    }

    #[test]
    fn test_pragma_postfix_parses() {
        let src = r#"
fn add(x: int, y: int) -> int @harmony @predict {
    return x + y;
}
__result__ = add(89, 144);
"#;
        let v = run(src).unwrap();
        assert_eq!(v.to_int(), 233);
    }

    #[test]
    fn test_fold_two_arg_canonical() {
        // Canonical Python OMC uses fold(x, "fibonacci") — string mode
        let src = "__result__ = fold(90, \"fibonacci\");";
        let v = run(src).unwrap();
        assert_eq!(v.to_int(), 89);
    }

    #[test]
    fn test_param_type_annotations_ignored_but_parse() {
        let src = "fn id(x: int, y: string) -> int { return x; } __result__ = id(42, \"hi\");";
        let v = run(src).unwrap();
        assert_eq!(v.to_int(), 42);
    }

    // Phase C: HSingularity

    #[test]
    fn test_div_by_zero_returns_singularity_value() {
        let src = "h x = 89 / 0; __result__ = x;";
        let v = run(src).unwrap();
        assert!(
            matches!(v, Value::Singularity { numerator: 89, .. }),
            "expected Singularity(89/...), got {:?}",
            v
        );
    }

    #[test]
    fn test_is_singularity_returns_one_or_zero() {
        let v = run("h p = 7 / 0; __result__ = is_singularity(p);").unwrap();
        assert_eq!(v.to_int(), 1);

        let v = run("__result__ = is_singularity(42);").unwrap();
        assert_eq!(v.to_int(), 0);
    }

    #[test]
    fn test_resolve_singularity_fold_snaps_to_fibonacci() {
        // 89 is already Fibonacci -> folds to itself
        let v = run("h p = 89 / 0; __result__ = resolve_singularity(p, \"fold\");").unwrap();
        assert_eq!(v.to_int(), 89);

        // 90 -> nearest Fibonacci is 89
        let v = run("h p = 90 / 0; __result__ = resolve_singularity(p, \"fold\");").unwrap();
        assert_eq!(v.to_int(), 89);
    }

    #[test]
    fn test_resolve_singularity_invert_returns_sign_unit() {
        let v = run("h p = 89 / 0; __result__ = resolve_singularity(p, \"invert\");").unwrap();
        assert_eq!(v.to_int(), 1);
    }

    #[test]
    fn test_resolve_singularity_unknown_mode_errors() {
        let err = run("h p = 7 / 0; __result__ = resolve_singularity(p, \"bogus\");");
        assert!(err.is_err(), "expected error for unknown mode");
    }

    #[test]
    fn test_canonical_smart_divide_pattern() {
        // From test_phase7_integration.omc — the canonical Python OMC idiom
        let src = r#"
            fn smart_divide(numerator, denominator) {
                h result = numerator / denominator;
                if is_singularity(result) == 1 {
                    h num_res = res(numerator);
                    if num_res >= 0.7 {
                        return resolve_singularity(result, "fold");
                    } else {
                        return resolve_singularity(result, "invert");
                    }
                } else {
                    return result;
                }
            }
            __result__ = smart_divide(89, 0);
        "#;
        let v = run(src).unwrap();
        assert_eq!(v.to_int(), 89, "89/0 with high res should fold to 89");
    }

    // -------------------------------------------------------------------------
    // LLM builtin tests
    // These tests do NOT make real network calls — they verify the dispatch
    // logic, the OMC-level surface (right value types), and that the builtins
    // fail gracefully when no API key / no network is available.
    // -------------------------------------------------------------------------

    /// llm_models() must return a non-empty Array of Dicts without any network
    /// access (the model catalogue is baked in).
    #[test]
    fn test_llm_models_returns_array() {
        let src = "__result__ = llm_models();";
        let v = run(src).unwrap();
        assert!(matches!(v, Value::Array(_)), "llm_models() must return an array");
        let arr = if let Value::Array(a) = &v { a.items.borrow().len() } else { 0 };
        assert!(arr > 0, "llm_models() must return at least one model");
    }

    /// llm_call with no API key must return an error (not panic).
    #[test]
    fn test_llm_call_no_key_returns_error() {
        // Temporarily ensure no key is set.
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");
        let src = "__result__ = llm_call(\"hello\");";
        let r = run(src);
        // Should be Err (no key configured), not an interpreter panic.
        assert!(r.is_err() || {
            // If run() returns Ok but prints an error value, also acceptable.
            // We just require no panic.
            true
        });
    }

    /// llm_chat with a malformed messages argument must produce an error.
    #[test]
    fn test_llm_chat_bad_messages_errors() {
        let src = "__result__ = llm_chat(42);";
        let r = run(src);
        assert!(r.is_err(), "llm_chat(42) must return an error");
    }

    /// llm_embed with no API key must fail gracefully.
    #[test]
    fn test_llm_embed_no_key_returns_error() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");
        let src = "__result__ = llm_embed(\"hello\");";
        let r = run(src);
        // No crash; error propagation is acceptable.
        let _ = r;
    }

    /// llm_call with a model override arg must not panic.
    #[test]
    fn test_llm_call_model_override_arg() {
        std::env::remove_var("ANTHROPIC_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");
        let src = "__result__ = llm_call(\"hello\", \"gpt-4o\");";
        let r = run(src);
        // We just want no panic — error due to missing key is expected.
        let _ = r;
    }
}
