//! OMCJ — OMC-to-JavaScript transpiler core.
//!
//! Walk the `Statement` / `Expression` AST produced by
//! `omnimcode_core::parser::Parser` and emit readable ES2020 JavaScript
//! that imports all OMC built-ins from the runtime module.
//!
//! # Usage
//! ```no_run
//! let js = omnimcode_js::transpile(r#"
//!   fn greet(name)
//!     print("Hello, " .. name)
//! "#, "./omc-runtime.js").unwrap();
//! println!("{}", js);
//! ```

use omnimcode_core::ast::{Expression, ForIterable, MatchArm, Pattern, Statement};
use omnimcode_core::parser::Parser;

/// Default import path for the runtime module.
pub const DEFAULT_RUNTIME_PATH: &str = "./omc-runtime.js";

/// Transpile OMC source to an ES2020 module string.
///
/// `runtime_path` is the import specifier used for the OMC built-in runtime
/// (e.g. `"./omc-runtime.js"` or `"https://cdn.example.com/omc-runtime.js"`).
pub fn transpile(src: &str, runtime_path: &str) -> Result<String, String> {
    let mut parser = Parser::new(src);
    let stmts = parser.parse()?;
    let mut t = Transpiler::new();
    Ok(t.emit_module(&stmts, runtime_path))
}

// ── known OMC built-in function names ─────────────────────────────────────────
//
// Calls to any of these are emitted as `omc.<name>(args)` so the runtime
// module provides them transparently.  Everything else is treated as a
// user-defined function and emitted verbatim.
const OMC_BUILTINS: &[&str] = &[
    // I/O
    "print", "println", "input",
    // type conversions
    "str", "int", "float", "bool", "type_of",
    // core collections
    "len", "range", "append", "push", "pop", "sort", "reverse",
    "keys", "values", "items", "has_key",
    // arrays
    "arr_push", "arr_pop", "arr_len", "arr_sort", "arr_reverse",
    "arr_append", "arr_extend", "arr_index", "arr_slice",
    "arr_map", "arr_filter", "arr_reduce", "arr_sum",
    "arr_min", "arr_max", "arr_flatten", "arr_unique",
    "arr_zip", "arr_enumerate", "arr_head", "arr_tail",
    // dicts
    "dict_get", "dict_set", "dict_keys", "dict_values",
    "dict_items", "dict_has", "dict_delete", "dict_merge",
    "dict_from_pairs",
    // strings
    "str_len", "str_upper", "str_lower", "str_trim", "str_split",
    "str_join", "str_replace", "str_starts", "str_ends",
    "str_contains", "str_find", "str_format", "str_reverse",
    "concat", "join", "split", "trim", "upper", "lower",
    "starts_with", "ends_with", "contains", "replace", "format",
    "parse_int", "parse_float",
    // maths
    "sqrt", "abs", "floor", "ceil", "round",
    "sin", "cos", "tan", "log", "log2", "exp", "pow",
    "min", "max", "sum",
    "math_sqrt", "math_abs", "math_floor", "math_ceil",
    "math_round", "math_sin", "math_cos", "math_tan",
    "math_log", "math_log2", "math_exp", "math_pow",
    "math_min", "math_max",
    // functional
    "map", "filter", "reduce", "zip", "enumerate", "collect",
    // OMC-specific harmonics
    "phi", "fib", "resonance", "fold", "safe",
    "phi_res", "phi_fold", "phi_him", "phi_shadow", "harmony",
    // iterators
    "iter", "next",
    // error handling
    "error", "assert", "panic",
    // JSON/CSV helpers
    "json_parse", "json_str", "csv_parse",
];

// ── known math module method mappings to JS Math ─────────────────────────────
fn math_method_to_js(method: &str) -> Option<&'static str> {
    match method {
        "sqrt"  => Some("Math.sqrt"),
        "abs"   => Some("Math.abs"),
        "floor" => Some("Math.floor"),
        "ceil"  => Some("Math.ceil"),
        "round" => Some("Math.round"),
        "sin"   => Some("Math.sin"),
        "cos"   => Some("Math.cos"),
        "tan"   => Some("Math.tan"),
        "log"   => Some("Math.log"),
        "log2"  => Some("Math.log2"),
        "log10" => Some("Math.log10"),
        "exp"   => Some("Math.exp"),
        "pow"   => Some("Math.pow"),
        "min"   => Some("Math.min"),
        "max"   => Some("Math.max"),
        _       => None,
    }
}

// ── transpiler state ──────────────────────────────────────────────────────────

struct Transpiler {
    indent: usize,
    /// Names defined by the user (fn / class) — these shadow any omc.* builtin.
    user_defs: std::collections::HashSet<String>,
}

impl Transpiler {
    fn new() -> Self {
        Self { indent: 0, user_defs: Default::default() }
    }

    fn pad(&self) -> String {
        "  ".repeat(self.indent)
    }

    // ── top-level ─────────────────────────────────────────────────────────────

    fn emit_module(&mut self, stmts: &[Statement], runtime_path: &str) -> String {
        // Pre-pass: collect user-defined function and class names so they
        // shadow OMC builtins (e.g. a user fn named `fib` must stay `fib`,
        // not be rewritten to `omc.fib`).
        for stmt in stmts {
            match stmt {
                Statement::FunctionDef { name, .. } => { self.user_defs.insert(name.clone()); }
                Statement::ClassDef    { name, .. } => { self.user_defs.insert(name.clone()); }
                _ => {}
            }
        }

        let mut out = String::new();
        out.push_str("// Generated by omcj — OMC to JavaScript transpiler\n");
        out.push_str(&format!("import * as omc from '{}';\n\n", runtime_path));
        out.push_str(&self.emit_stmts(stmts));
        out
    }

    fn emit_stmts(&mut self, stmts: &[Statement]) -> String {
        stmts
            .iter()
            .map(|s| self.emit_stmt(s))
            .collect::<Vec<_>>()
            .join("")
    }

    fn emit_block(&mut self, stmts: &[Statement]) -> String {
        self.indent += 1;
        let body = self.emit_stmts(stmts);
        self.indent -= 1;
        body
    }

    // ── statements ────────────────────────────────────────────────────────────

    fn emit_stmt(&mut self, s: &Statement) -> String {
        let pad = self.pad();
        match s {
            // print(expr)
            Statement::Print(e) => {
                format!("{}omc.print({});\n", pad, self.emit_expr(e))
            }

            // expression-statement
            Statement::Expression(e) => {
                format!("{}{};\n", pad, self.emit_expr(e))
            }

            // variable declaration  (let x = value)
            Statement::VarDecl { name, value, .. } => {
                format!("{}let {} = {};\n", pad, name, self.emit_expr(value))
            }

            // parameter (always has a value in the AST — acts as "let name = value;")
            Statement::Parameter { name, value } => {
                format!("{}let {} = {};\n", pad, name, self.emit_expr(value))
            }

            // assignment  x = value
            Statement::Assignment { name, value } => {
                format!("{}{} = {};\n", pad, name, self.emit_expr(value))
            }

            // index assignment  x[idx] = value
            Statement::IndexAssignment { name, index, value } => {
                format!(
                    "{}{}[{}] = {};\n",
                    pad,
                    name,
                    self.emit_expr(index),
                    self.emit_expr(value)
                )
            }

            // chained index assignment  x[a][b] = value
            Statement::ChainedIndexAssignment { name, first_index, second_index, value } => {
                format!(
                    "{}{}[{}][{}] = {};\n",
                    pad,
                    name,
                    self.emit_expr(first_index),
                    self.emit_expr(second_index),
                    self.emit_expr(value)
                )
            }

            // if / else if / else
            Statement::If {
                condition,
                then_body,
                elif_parts,
                else_body,
            } => {
                let mut out =
                    format!("{}if ({}) {{\n", pad, self.emit_expr(condition));
                out.push_str(&self.emit_block(then_body));
                out.push_str(&format!("{}}}", pad));

                for (elif_cond, elif_body) in elif_parts {
                    out.push_str(&format!(
                        " else if ({}) {{\n",
                        self.emit_expr(elif_cond)
                    ));
                    out.push_str(&self.emit_block(elif_body));
                    out.push_str(&format!("{}}}", pad));
                }

                if let Some(eb) = else_body {
                    out.push_str(" else {\n");
                    out.push_str(&self.emit_block(eb));
                    out.push_str(&format!("{}}}", pad));
                }

                out.push('\n');
                out
            }

            // while loop
            Statement::While { condition, body } => {
                let mut out =
                    format!("{}while ({}) {{\n", pad, self.emit_expr(condition));
                out.push_str(&self.emit_block(body));
                out.push_str(&format!("{}}}\n", pad));
                out
            }

            // for loop — range or iterable
            Statement::For { var, iterable, body } => {
                let header = match iterable {
                    ForIterable::Range { start, end } => {
                        let s = self.emit_expr(start);
                        let e = self.emit_expr(end);
                        format!(
                            "for (let {} = {}; {} < {}; {}++)",
                            var, s, var, e, var
                        )
                    }
                    ForIterable::Expr(e) => {
                        format!(
                            "for (const {} of omc.iter({}))",
                            var,
                            self.emit_expr(e)
                        )
                    }
                };
                let mut out = format!("{}{} {{\n", pad, header);
                out.push_str(&self.emit_block(body));
                out.push_str(&format!("{}}}\n", pad));
                out
            }

            // function definition
            Statement::FunctionDef {
                name,
                params,
                body,
                pragmas,
                ..
            } => {
                let async_kw = if pragmas.iter().any(|p| p == "async") {
                    "async "
                } else {
                    ""
                };
                let params_str = params.join(", ");
                let mut out = format!(
                    "{}{}function {}({}) {{\n",
                    pad, async_kw, name, params_str
                );
                out.push_str(&self.emit_block(body));
                out.push_str(&format!("{}}}\n", pad));
                out
            }

            Statement::Return(e) => match e {
                Some(v) => format!("{}return {};\n", pad, self.emit_expr(v)),
                None    => format!("{}return;\n", pad),
            },

            Statement::Break    => format!("{}break;\n", pad),
            Statement::Continue => format!("{}continue;\n", pad),

            // imports — kept as a comment so the generated file is self-contained;
            // users can adapt to their bundler's module resolution as needed.
            Statement::Import { module, alias, selected } => {
                let sel = selected.as_deref().unwrap_or(&[]);
                if sel.is_empty() {
                    match alias {
                        Some(a) => format!("// import * as {} from '{}';\n", a, module),
                        None    => format!("// import '{}';\n", module),
                    }
                } else {
                    let names = sel.join(", ");
                    match alias {
                        Some(a) => format!(
                            "// import {{ {} }} from '{}';  // alias: {}\n",
                            names, module, a
                        ),
                        None => format!("// import {{ {} }} from '{}';\n", names, module),
                    }
                }
            }

            // try / catch / finally
            Statement::Try {
                body,
                err_var,
                handler,
                finally,
            } => {
                let mut out = format!("{}try {{\n", pad);
                out.push_str(&self.emit_block(body));
                out.push_str(&format!("{}}} catch ({}) {{\n", pad, err_var));
                out.push_str(&self.emit_block(handler));
                out.push_str(&format!("{}}}", pad));
                if let Some(fin) = finally {
                    out.push_str(" finally {\n");
                    out.push_str(&self.emit_block(fin));
                    out.push_str(&format!("{}}}", pad));
                }
                out.push('\n');
                out
            }

            // throw new Error(...)
            Statement::Throw(e) => {
                format!(
                    "{}throw new Error(omc.str({}));\n",
                    pad,
                    self.emit_expr(e)
                )
            }

            Statement::Yield(e) => {
                format!("{}yield {};\n", pad, self.emit_expr(e))
            }

            // class definition
            Statement::ClassDef {
                name,
                parent,
                methods,
                ..
            } => {
                let extends = parent
                    .as_deref()
                    .map(|p| format!(" extends {}", p))
                    .unwrap_or_default();
                let mut out = format!("{}class {}{} {{\n", pad, name, extends);
                self.indent += 1;
                for method_stmt in methods {
                    if let Statement::FunctionDef {
                        name: mname,
                        params,
                        body,
                        pragmas,
                        ..
                    } = method_stmt
                    {
                        let async_kw = if pragmas.iter().any(|p| p == "async") {
                            "async "
                        } else {
                            ""
                        };
                        let mpad = self.pad();
                        let params_str = params.join(", ");
                        out.push_str(&format!(
                            "{}{}{}({}) {{\n",
                            mpad, async_kw, mname, params_str
                        ));
                        out.push_str(&self.emit_block(body));
                        out.push_str(&format!("{}}}\n", mpad));
                    }
                }
                self.indent -= 1;
                out.push_str(&format!("{}}}\n", pad));
                out
            }

            // match — emit as an if / else-if / else chain
            Statement::Match { scrutinee, arms } => {
                self.emit_match(scrutinee, arms)
            }
        }
    }

    // ── match helpers ─────────────────────────────────────────────────────────

    fn emit_match(&mut self, scrutinee: &Expression, arms: &[MatchArm]) -> String {
        let pad = self.pad();
        // Use a temp variable to avoid re-evaluating the scrutinee.
        let tmp = "__m__";
        let mut out = format!("{}const {} = {};\n", pad, tmp, self.emit_expr(scrutinee));

        for (i, arm) in arms.iter().enumerate() {
            let kw = if i == 0 { "if" } else { " else if" };

            match &arm.pattern {
                Pattern::Wildcard => {
                    // default arm — close the if chain and emit as else
                    if i == 0 {
                        out.push_str(&format!("{}if (true) {{\n", pad));
                    } else {
                        out.push_str(&format!(" else {{\n"));
                    }
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                    out.push('\n');
                    return out; // nothing can match after wildcard
                }

                Pattern::Bind(v) => {
                    // bind v = __m__ unconditionally
                    out.push_str(&format!("{}{}(true) {{\n", pad, kw));
                    self.indent += 1;
                    out.push_str(&format!(
                        "{}const {} = {};\n",
                        self.pad(),
                        v,
                        tmp
                    ));
                    out.push_str(&self.emit_stmts(&arm.body));
                    self.indent -= 1;
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::LitInt(n) => {
                    out.push_str(&format!(
                        "{}{}({} === {}) {{\n",
                        pad, kw, tmp, n
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::LitFloat(f) => {
                    out.push_str(&format!(
                        "{}{}({} === {}) {{\n",
                        pad, kw, tmp, f
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::LitString(s) => {
                    out.push_str(&format!(
                        "{}{}({} === {}) {{\n",
                        pad,
                        kw,
                        tmp,
                        js_string_lit(s)
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::LitBool(b) => {
                    out.push_str(&format!(
                        "{}{}({} === {}) {{\n",
                        pad,
                        kw,
                        tmp,
                        if *b { "true" } else { "false" }
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::LitNull => {
                    out.push_str(&format!(
                        "{}{}({} == null) {{\n",
                        pad, kw, tmp
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::RangeInt(lo, hi) => {
                    out.push_str(&format!(
                        "{}{}({} >= {} && {} <= {}) {{\n",
                        pad, kw, tmp, lo, tmp, hi
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::RangeStr(lo, hi) => {
                    out.push_str(&format!(
                        "{}{}({} >= {:?} && {} <= {:?}) {{\n",
                        pad, kw, tmp, lo.to_string(), tmp, hi.to_string()
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::Or(pats) => {
                    let conds: Vec<String> = pats
                        .iter()
                        .map(|p| self.pattern_cond(tmp, p))
                        .collect();
                    out.push_str(&format!(
                        "{}{}({}) {{\n",
                        pad,
                        kw,
                        conds.join(" || ")
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }

                Pattern::Type(ty) => {
                    out.push_str(&format!(
                        "{}{}(omc.type_of({}) === {}) {{\n",
                        pad,
                        kw,
                        tmp,
                        js_string_lit(ty)
                    ));
                    out.push_str(&self.emit_block(&arm.body));
                    out.push_str(&format!("{}}}", pad));
                }
            }
        }
        out.push('\n');
        out
    }

    fn pattern_cond(&self, tmp: &str, p: &Pattern) -> String {
        match p {
            Pattern::LitInt(n)    => format!("{} === {}", tmp, n),
            Pattern::LitFloat(f)  => format!("{} === {}", tmp, f),
            Pattern::LitString(s) => format!("{} === {}", tmp, js_string_lit(s)),
            Pattern::LitBool(b)   => format!("{} === {}", tmp, b),
            Pattern::LitNull      => format!("{} == null", tmp),
            Pattern::Wildcard | Pattern::Bind(_) => "true".to_string(),
            Pattern::RangeInt(lo, hi) => {
                format!("{} >= {} && {} <= {}", tmp, lo, tmp, hi)
            }
            Pattern::RangeStr(lo, hi) => {
                format!(
                    "{} >= {:?} && {} <= {:?}",
                    tmp,
                    lo.to_string(),
                    tmp,
                    hi.to_string()
                )
            }
            Pattern::Or(ps) => ps
                .iter()
                .map(|q| self.pattern_cond(tmp, q))
                .collect::<Vec<_>>()
                .join(" || "),
            Pattern::Type(ty) => {
                format!("omc.type_of({}) === {}", tmp, js_string_lit(ty))
            }
        }
    }

    // ── expressions ───────────────────────────────────────────────────────────

    fn emit_expr(&self, e: &Expression) -> String {
        match e {
            Expression::Number(n)  => n.to_string(),
            Expression::Float(f)   => format_float(*f),
            Expression::String(s)  => js_string_lit(s),
            Expression::Boolean(b) => if *b { "true".to_owned() } else { "false".to_owned() },

            Expression::Array(elems) => {
                let parts: Vec<String> = elems.iter().map(|x| self.emit_expr(x)).collect();
                format!("[{}]", parts.join(", "))
            }

            Expression::Dict(pairs) => {
                let parts: Vec<String> = pairs
                    .iter()
                    .map(|(k, v)| {
                        let key = match k {
                            Expression::String(s) => js_bare_key(s),
                            other => format!("[{}]", self.emit_expr(other)),
                        };
                        format!("{}: {}", key, self.emit_expr(v))
                    })
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }

            Expression::Variable(name) => name.clone(),

            Expression::Index { name, index } => {
                format!("{}[{}]", name, self.emit_expr(index))
            }
            Expression::ChainedIndex { object, index } => {
                format!("{}[{}]", self.emit_expr(object), self.emit_expr(index))
            }

            // arithmetic
            Expression::Add(l, r) => format!("({} + {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::Sub(l, r) => format!("({} - {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::Mul(l, r) => format!("({} * {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::Div(l, r) => format!("({} / {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::Mod(l, r) => format!("({} % {})",  self.emit_expr(l), self.emit_expr(r)),

            // comparisons
            Expression::Eq(l, r)  => format!("({} === {})", self.emit_expr(l), self.emit_expr(r)),
            Expression::Ne(l, r)  => format!("({} !== {})", self.emit_expr(l), self.emit_expr(r)),
            Expression::Lt(l, r)  => format!("({} < {})",   self.emit_expr(l), self.emit_expr(r)),
            Expression::Le(l, r)  => format!("({} <= {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::Gt(l, r)  => format!("({} > {})",   self.emit_expr(l), self.emit_expr(r)),
            Expression::Ge(l, r)  => format!("({} >= {})",  self.emit_expr(l), self.emit_expr(r)),

            // logical
            Expression::And(l, r) => format!("({} && {})", self.emit_expr(l), self.emit_expr(r)),
            Expression::Or(l, r)  => format!("({} || {})", self.emit_expr(l), self.emit_expr(r)),
            Expression::Not(e)    => format!("(!({}))",   self.emit_expr(e)),

            // bitwise
            Expression::BitAnd(l, r) => format!("({} & {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::BitOr(l, r)  => format!("({} | {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::BitXor(l, r) => format!("({} ^ {})",  self.emit_expr(l), self.emit_expr(r)),
            Expression::BitNot(e)    => format!("(~({}))",    self.emit_expr(e)),
            Expression::Shl(l, r)   => format!("({} << {})", self.emit_expr(l), self.emit_expr(r)),
            Expression::Shr(l, r)   => format!("({} >> {})", self.emit_expr(l), self.emit_expr(r)),

            // function call (including dotted names like "phi.fold" or "arr.push")
            Expression::Call { name, args, .. } => self.emit_call(name, args),

            // OMC-specific primitives
            Expression::Resonance(e) => format!("omc.resonance({})", self.emit_expr(e)),
            Expression::Fold(e)      => format!("omc.fold({})",      self.emit_expr(e)),
            Expression::Safe(e)      => {
                // Safe expression — wraps a call that might throw; returns null on error.
                format!("omc.safe(() => {})", self.emit_expr(e))
            }

            // lambda / closure
            Expression::Lambda { params, body } => {
                self.emit_lambda(params, body)
            }

            // if-expression: `if cond { stmts } else { stmts }` used as a value.
            // Emitted as IIFE so it can appear in expression position.
            Expression::IfExpr { condition, then_body, else_body } => {
                let cond_js = self.emit_expr(condition);
                let mut sub = Transpiler { indent: self.indent + 1, user_defs: self.user_defs.clone() };
                let then_js = sub.emit_stmts(then_body);
                let else_js = else_body.as_ref()
                    .map(|b| {
                        let mut s = Transpiler { indent: self.indent + 1, user_defs: self.user_defs.clone() };
                        s.emit_stmts(b)
                    })
                    .unwrap_or_default();
                format!(
                    "(() => {{ if ({}) {{ {}return null; }} else {{ {}return null; }} }})()",
                    cond_js, then_js, else_js
                )
            }
            Expression::CallExpr { callee, args, .. } => {
                let callee_js = self.emit_expr(callee);
                let args_js: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
                format!("({})({})", callee_js, args_js.join(", "))
            }
        }
    }

    // ── call dispatch ─────────────────────────────────────────────────────────

    fn emit_call(&self, name: &str, args: &[Expression]) -> String {
        let args_js: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
        let args_str = args_js.join(", ");

        // Dotted name: "phi.fold", "math.sqrt", "arr.push", ...
        if let Some(dot) = name.find('.') {
            let obj    = &name[..dot];
            let method = &name[dot + 1..];

            match obj {
                // OMC harmonic module → omc.phi_<method>
                "phi" | "core" | "fib" => {
                    return format!("omc.{}_{}({})", obj, method, args_str);
                }
                // math module → prefer native JS Math, fall back to omc.math_*
                "math" => {
                    if let Some(js_fn) = math_method_to_js(method) {
                        return format!("{}({})", js_fn, args_str);
                    }
                    return format!("omc.math_{}({})", method, args_str);
                }
                // everything else is a user variable: obj.method(args)
                _ => {
                    return format!("{}.{}({})", obj, method, args_str);
                }
            }
        }

        // User-defined function always wins over builtin name collision
        if self.user_defs.contains(name) {
            return format!("{}({})", name, args_str);
        }
        // Known OMC built-in → prefix with omc.
        if OMC_BUILTINS.contains(&name) {
            format!("omc.{}({})", name, args_str)
        } else {
            // User-defined function
            format!("{}({})", name, args_str)
        }
    }

    // ── lambda ────────────────────────────────────────────────────────────────

    fn emit_lambda(&self, params: &[String], body: &[Statement]) -> String {
        let params_str = params.join(", ");

        // Single expression / return statement → compact arrow form
        if body.len() == 1 {
            let compact = match &body[0] {
                Statement::Return(Some(e)) => Some(self.emit_expr(e)),
                Statement::Expression(e)   => Some(self.emit_expr(e)),
                _                          => None,
            };
            if let Some(expr_js) = compact {
                return format!("(({}) => {})", params_str, expr_js);
            }
        }

        // Multi-statement body → arrow function with block
        let mut sub = Transpiler { indent: self.indent + 1, user_defs: self.user_defs.clone() };
        let body_js = sub.emit_stmts(body);
        format!(
            "(({}) => {{\n{}{}}})",
            params_str,
            body_js,
            "  ".repeat(self.indent)
        )
    }
}

// ── formatting helpers ────────────────────────────────────────────────────────

/// Emit a float literal that round-trips cleanly.
fn format_float(f: f64) -> String {
    if f.is_nan()              { return "Number.NaN".to_owned(); }
    if f.is_infinite()         {
        return if f > 0.0 { "Infinity".to_owned() } else { "-Infinity".to_owned() };
    }
    if f.fract() == 0.0       { format!("{:.1}", f) }
    else                       { format!("{}", f) }
}

/// Emit a JS string literal with proper escaping.
fn js_string_lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"'  => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c    => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Emit a dict key: bare identifier if valid JS ident, otherwise quoted.
fn js_bare_key(s: &str) -> String {
    let valid = !s.is_empty()
        && s.chars()
            .enumerate()
            .all(|(i, c)| {
                if i == 0 { c.is_alphabetic() || c == '_' || c == '$' }
                else      { c.is_alphanumeric() || c == '_' || c == '$' }
            });
    if valid { s.to_owned() } else { js_string_lit(s) }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn snip(src: &str) -> String {
        // Transpile and strip the ESM header for brevity.
        transpile(src, "./rt.js")
            .unwrap()
            .lines()
            .skip(2)   // skip comment + import line
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_owned()
    }

    #[test]
    fn test_print() {
        assert!(snip(r#"print("hi")"#).contains("omc.print"));
    }

    #[test]
    fn test_var_decl() {
        let out = snip("var x = 42");
        assert!(out.contains("let x = 42"));
    }

    #[test]
    fn test_if_else() {
        let out = snip("if x > 0\n  print(x)\nelse\n  print(0)");
        assert!(out.contains("if ("));
        assert!(out.contains("else"));
    }

    #[test]
    fn test_function() {
        let out = snip("fn add(a, b)\n  return a + b");
        assert!(out.contains("function add(a, b)"));
        assert!(out.contains("return"));
    }

    #[test]
    fn test_for_range() {
        let out = snip("for i in 0..10\n  print(i)");
        assert!(out.contains("for (let i = 0;"));
    }

    #[test]
    fn test_builtin_call() {
        let out = snip("len(arr)");
        assert!(out.contains("omc.len(arr)"));
    }

    #[test]
    fn test_math_module() {
        let out = snip("math.sqrt(x)");
        assert!(out.contains("Math.sqrt(x)"));
    }

    #[test]
    fn test_phi_module() {
        let out = snip("phi.fold(x)");
        assert!(out.contains("omc.phi_fold(x)"));
    }
}
