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
        }
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
            Statement::If { condition, then_body, elif_parts, else_body } => Statement::If {
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
                // body's typo check doesn't flag them.
                let mut inner = defined.clone();
                for p in &params {
                    inner.insert(p.clone());
                }
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
            // Recursive walk for the rest of the structural expressions.
            Expression::Add(l, r) => Expression::Add(
                Box::new(Self::heal_expr(*l, defined, arities, diags)),
                Box::new(Self::heal_expr(*r, defined, arities, diags)),
            ),
            Expression::Sub(l, r) => Expression::Sub(
                Box::new(Self::heal_expr(*l, defined, arities, diags)),
                Box::new(Self::heal_expr(*r, defined, arities, diags)),
            ),
            Expression::Mul(l, r) => Expression::Mul(
                Box::new(Self::heal_expr(*l, defined, arities, diags)),
                Box::new(Self::heal_expr(*r, defined, arities, diags)),
            ),
            Expression::Mod(l, r) => Expression::Mod(
                Box::new(Self::heal_expr(*l, defined, arities, diags)),
                Box::new(Self::heal_expr(*r, defined, arities, diags)),
            ),
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
                println!("{}", value.to_string());
                Ok(())
            }
            Statement::Expression(expr) => {
                self.eval_expr(expr)?;
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
                let idx = self.eval_expr(index)?.to_int() as usize;
                let val = self.eval_expr(value)?;
                
                if let Some(Value::Array(arr)) = self.get_var(name) {
                    let mut items = arr.items.borrow_mut();
                    if idx < items.len() {
                        items[idx] = val;
                    }
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
                        if let Value::Array(arr) = self.eval_expr(expr)? {
                            // Snapshot items so the loop body can mutate
                            // the underlying Rc<RefCell<Vec>> without
                            // tripping a borrow conflict.
                            let items = arr.items.borrow().clone();
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
                // the handler with err_var bound to the message string.
                // After the body+handler complete (success OR failure),
                // run finally unconditionally — including when the
                // handler itself raises. Matches Python try/except/finally.
                let body_result = self.execute_block(body);
                let after_handler = match body_result {
                    Ok(()) => Ok(()),
                    Err(msg) => {
                        self.set_var(err_var.clone(), Value::String(msg));
                        self.execute_block(handler)
                    }
                };
                if let Some(finally_body) = finally {
                    // Run finally regardless of after_handler's status.
                    // If both finally and after_handler fail, finally's
                    // error wins (Python's behavior — finally is the
                    // "last word" that the surrounding scope sees).
                    let finally_result = self.execute_block(finally_body);
                    if finally_result.is_err() {
                        return finally_result;
                    }
                }
                after_handler
            }
            Statement::Throw(expr) => {
                // Evaluate the expression, raise its display-string as
                // the current frame's error. Surrounding catch (if any)
                // binds the string to err_var; uncaught throws propagate
                // up to the top-level error handler.
                let v = self.eval_expr(expr)?;
                Err(v.to_display_string())
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
                    Err(format!("Undefined variable: {}", name))
                }
            }
            Expression::Index { name, index } => {
                let idx_v = self.eval_expr(index)?;
                let container = self.get_var(name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))?;
                match container {
                    Value::Array(arr) => {
                        let idx = idx_v.to_int() as usize;
                        arr.items.borrow().get(idx).cloned()
                            .ok_or_else(|| format!("Index out of bounds: {}", idx))
                    }
                    Value::Dict(d) => {
                        // String-keyed lookup. Coerce numeric/bool indices
                        // via to_display_string so `d[42]` works as
                        // `d["42"]` — surprising for some, but matches
                        // OMC's "everything stringifies" stance.
                        let key = idx_v.to_display_string();
                        Ok(d.borrow().get(&key).cloned().unwrap_or(Value::Null))
                    }
                    _ => Err(format!("Not indexable: {}", name)),
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
                    Ok(Value::HInt(HInt::new(lv.to_int() + rv.to_int())))
                }
            }
            Expression::Sub(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::HFloat(lv.to_float() - rv.to_float()))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int() - rv.to_int())))
                }
            }
            Expression::Mul(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::HFloat(lv.to_float() * rv.to_float()))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int() * rv.to_int())))
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
            | "json_parse" | "json_stringify"
            // Arrays
            | "arr_new" | "arr_from_range" | "arr_len" | "arr_get" | "arr_set"
            | "arr_push" | "arr_first" | "arr_last" | "arr_slice" | "arr_concat"
            | "arr_contains" | "arr_index_of" | "arr_sort" | "arr_reverse" | "arr_join"
            | "arr_min" | "arr_max" | "arr_sum" | "arr_fold_elements"
            | "arr_argmax" | "arr_argmin" | "arr_cumsum" | "arr_diff" | "arr_range"
            | "arr_unique_count" | "arr_partition_by"
            | "arr_min_float" | "arr_max_float" | "arr_gcd" | "fnv1a_hash"
            | "arr_mean" | "arr_variance" | "arr_stddev" | "arr_median"
            | "arr_harmonic_mean" | "arr_geometric_mean"
            | "arr_sum_sq" | "arr_norm" | "arr_dot"
            | "arr_resonance" | "filter_by_resonance" | "cleanup_array"
            | "arr_map" | "arr_filter" | "arr_reduce"
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
            // Time, conversion, introspection
            | "now_ms" | "to_int" | "int" | "to_float" | "float"
            | "to_string" | "string" | "len" | "type_of" | "error"
            | "defined_functions" | "call"
            // Test runner host-state primitives
            | "test_record_failure" | "test_failure_count"
            | "test_get_failures" | "test_clear_failures"
            | "test_set_current" | "test_get_current"
            // Random
            | "random_int" | "random_float" | "random_seed"
            // Polish round
            | "str_pad_left" | "str_pad_right" | "arr_zip" | "arr_unique"
            | "arr_take" | "arr_drop" | "arr_count" | "arr_repeat"
            | "arr_zeros" | "arr_ones" | "arr_chunk" | "arr_flatten"
            | "arr_enumerate" | "arr_window"
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
                "call_first_class_function: not a callable ({:?})", other
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
            "to_string" => {
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
                    return Err("str_concat requires 2 arguments".to_string());
                }
                // to_display_string (bare numbers) matches Phase 1's
                // string-+-concat semantics and Phase 4's vm_fast_dispatch.
                // Previously used to_string which produced ugly
                // "HInt(42, φ=..., HIM=...)" output for numeric args —
                // never what callers wanted.
                let s1 = self.eval_expr(&args[0])?.to_display_string();
                let s2 = self.eval_expr(&args[1])?.to_display_string();
                Ok(Value::String(format!("{}{}", s1, s2)))
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
            "arr_from_range" => {
                if args.len() < 2 {
                    return Err("arr_from_range requires 2 arguments".to_string());
                }
                let start = self.eval_expr(&args[0])?.to_int();
                let end = self.eval_expr(&args[1])?.to_int();
                let arr = HArray::new();
                {
                    let mut items = arr.items.borrow_mut();
                    for i in start..end {
                        items.push(Value::HInt(HInt::new(i)));
                    }
                }
                Ok(Value::Array(arr))
            }
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
                // Mutates by name. First arg must be a Variable reference so we can write back.
                // Use assign_var (walks outward for existing binding) instead of
                // set_var (always innermost) — otherwise pushes inside a closure
                // body would land in the closure's call scope, not the captured
                // env where the array actually lives, and the mutation would be
                // discarded on return.
                let val = self.eval_expr(&args[1])?;
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(arr)) = self.get_var(name) {
                        // With Rc<RefCell> HArray, the borrow_mut hits the
                        // shared collection — no assign_var write-back is
                        // needed, the caller's binding sees the push.
                        arr.items.borrow_mut().push(val);
                        return Ok(Value::Null);
                    }
                }
                Err("arr_push: first argument must be an array variable".to_string())
            }
            "arr_get" => {
                if args.len() < 2 {
                    return Err("arr_get requires (array, index)".to_string());
                }
                let arr_v = self.eval_expr(&args[0])?;
                let idx = self.eval_expr(&args[1])?.to_int();
                if let Value::Array(arr) = arr_v {
                    let i = idx as usize;
                    let items = arr.items.borrow();
                    items
                        .get(i)
                        .cloned()
                        .ok_or_else(|| format!("arr_get: index {} out of bounds (len {})", idx, items.len()))
                } else {
                    Err("arr_get: first argument must be an array".to_string())
                }
            }
            "arr_set" => {
                if args.len() < 3 {
                    return Err("arr_set requires (array_name, index, value)".to_string());
                }
                let idx = self.eval_expr(&args[1])?.to_int() as usize;
                let val = self.eval_expr(&args[2])?;
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(arr)) = self.get_var(name) {
                        let mut items = arr.items.borrow_mut();
                        if idx >= items.len() {
                            return Err(format!(
                                "arr_set: index {} out of bounds (len {})",
                                idx,
                                items.len()
                            ));
                        }
                        items[idx] = val;
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
                    Err("dict_get: first argument must be a dict".to_string())
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
                    return Err("dict_del requires (dict_var, key)".to_string());
                }
                let k = self.eval_expr(&args[1])?.to_display_string();
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Dict(d)) = self.get_var(name) {
                        d.borrow_mut().remove(&k);
                        return Ok(Value::Null);
                    }
                }
                Err("dict_del: first argument must be a dict variable".to_string())
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
                    return Ok(Value::Null);
                }
                let v = self.eval_expr(&args[0])?;
                println!("{}", v.to_display_string());
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
                    other => Err(format!(
                        "len: requires array or string, got {:?}",
                        other
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
                Err(format!("Undefined function: {}", name))
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
            // Append our own frame + the call site and rethrow.
            // Each invoke_user_function up the stack does the same,
            // so the final message lists every frame innermost-first.
            return Err(format!(
                "{}\n  at {}{}",
                e,
                display_frame_name(name),
                format_call_site(call_site),
            ));
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
        ("to_string", 1) | ("string", 1) => {
            Some(Ok(Value::String(args[0].to_display_string())))
        }
        // ---- println / print: they call out to stdout but the work
        // is dominated by I/O, so saving the shim alloc still helps ----
        ("println", _) => {
            let mut parts: Vec<String> = Vec::with_capacity(args.len());
            for v in args { parts.push(v.to_display_string()); }
            println!("{}", parts.join(" "));
            Some(Ok(Value::Null))
        }
        ("print", _) => {
            let mut parts: Vec<String> = Vec::with_capacity(args.len());
            for v in args { parts.push(v.to_display_string()); }
            print!("{}", parts.join(" "));
            Some(Ok(Value::Null))
        }
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
}

impl HealClassCounts {
    pub const fn new() -> Self {
        Self {
            typo: 0, typo_substrate_hit: 0, typo_fallback: 0,
            arity_pad: 0, arity_truncate: 0,
            div_zero: 0, mod_zero: 0, harmonic_index: 0,
            missing_return: 0, empty_index_safe: 0,
            reserved_var: 0, if_numeric: 0,
        }
    }
    pub fn total(&self) -> u32 {
        self.typo + self.arity_pad + self.arity_truncate
            + self.div_zero + self.mod_zero + self.harmonic_index
            + self.missing_return + self.empty_index_safe
            + self.reserved_var + self.if_numeric
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
    "json_parse", "json_stringify",
    // Arrays
    "arr_new", "arr_from_range", "arr_len", "arr_get", "arr_set",
    "arr_push", "arr_first", "arr_last", "arr_slice", "arr_concat",
    "arr_contains", "arr_index_of", "arr_sort", "arr_reverse", "arr_join",
    "arr_min", "arr_max", "arr_sum", "arr_fold_elements",
    "arr_argmax", "arr_argmin", "arr_cumsum", "arr_diff", "arr_range",
    "arr_unique_count", "arr_partition_by",
    "arr_min_float", "arr_max_float", "arr_gcd", "fnv1a_hash",
    "arr_mean", "arr_variance", "arr_stddev", "arr_median",
    "arr_harmonic_mean", "arr_geometric_mean",
    "arr_sum_sq", "arr_norm", "arr_dot",
    "arr_resonance", "filter_by_resonance", "cleanup_array",
    "arr_map", "arr_filter", "arr_reduce", "arr_any", "arr_all", "arr_find",
    "arr_zip", "arr_unique",
    "arr_take", "arr_drop", "arr_count", "arr_repeat",
    "arr_zeros", "arr_ones", "arr_chunk", "arr_flatten",
    "arr_enumerate", "arr_window",
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
    "read_file", "write_file", "file_exists", "print",
    "println", "print_raw",
    // Time / random / conversion / introspection
    "now_ms", "random_int", "random_float", "random_seed",
    "to_int", "int", "to_float", "float",
    "to_string", "string", "len", "type_of", "error",
    "defined_functions", "call",
    "test_record_failure", "test_failure_count",
    "test_get_failures", "test_clear_failures",
    "test_set_current", "test_get_current",
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
}
