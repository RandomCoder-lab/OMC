// src/interpreter.rs - AST execution engine

use crate::ast::*;
use crate::value::{HInt, HArray, Value, fibonacci, is_fibonacci};
use std::collections::{HashMap, HashSet};

pub struct Interpreter {
    globals: HashMap<String, Value>,
    functions: HashMap<String, (Vec<String>, Vec<Statement>)>,
    locals: Vec<HashMap<String, Value>>,
    return_value: Option<Value>,
    break_flag: bool,
    continue_flag: bool,
    /// Names of modules already imported (idempotent re-import).
    imported_modules: HashSet<String>,
}

impl Interpreter {
    pub fn new() -> Self {
        Interpreter {
            globals: HashMap::new(),
            functions: HashMap::new(),
            locals: vec![HashMap::new()],
            return_value: None,
            break_flag: false,
            continue_flag: false,
            imported_modules: HashSet::new(),
        }
    }

    /// Module search path used by `import NAME;`.
    /// Honors `OMC_STDLIB_PATH` (colon-separated), then falls back to a
    /// small built-in list that includes the canonical Python OMC stdlib.
    fn module_search_path() -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
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

    fn resolve_module(name: &str) -> Option<std::path::PathBuf> {
        // Try each search dir with a few naming variants.
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

    fn import_module(&mut self, name: &str) -> Result<(), String> {
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
        for stmt in &stmts {
            self.execute_stmt(stmt)?;
            // Don't propagate `return` / `break` / `continue` past module boundary.
            self.return_value = None;
            self.break_flag = false;
            self.continue_flag = false;
        }
        Ok(())
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
                self.set_var(name.clone(), val);
                Ok(())
            }
            Statement::IndexAssignment {
                name,
                index,
                value,
            } => {
                let idx = self.eval_expr(index)?.to_int() as usize;
                let val = self.eval_expr(value)?;
                
                if let Some(Value::Array(mut arr)) = self.get_var(name) {
                    if idx < arr.items.len() {
                        arr.items[idx] = val;
                        self.set_var(name.clone(), Value::Array(arr));
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
                            for item in arr.items {
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
            Statement::Import { module, alias: _ } => {
                // Alias is currently informational only — imports merge into a
                // flat function namespace (matching canonical Python OMC).
                self.import_module(module)
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
                Ok(Value::Array(HArray { items }))
            }
            Expression::Variable(name) => self.get_var(name).ok_or_else(|| format!("Undefined variable: {}", name)),
            Expression::Index { name, index } => {
                let idx = self.eval_expr(index)?.to_int() as usize;
                if let Some(Value::Array(arr)) = self.get_var(name) {
                    arr.items.get(idx).cloned().ok_or_else(|| "Index out of bounds".to_string())
                } else {
                    Err(format!("Not an array: {}", name))
                }
            }
            Expression::Add(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
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
                if lv.is_float() || rv.is_float() {
                    Ok(Value::Bool(lv.to_float() == rv.to_float()))
                } else {
                    Ok(Value::Bool(lv.to_int() == rv.to_int()))
                }
            }
            Expression::Ne(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                if lv.is_float() || rv.is_float() {
                    Ok(Value::Bool(lv.to_float() != rv.to_float()))
                } else {
                    Ok(Value::Bool(lv.to_int() != rv.to_int()))
                }
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
            Expression::Call { name, args } => {
                self.call_function(name, args)
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
                        let fibs: [i64; 15] = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610];
                        let abs_val = h.value.abs();
                        let mut nearest = fibs[0];
                        let mut min_dist = abs_val;
                        for &fib in &fibs {
                            let d = (fib - abs_val).abs();
                            if d < min_dist {
                                min_dist = d;
                                nearest = fib;
                            }
                        }
                        let result = if h.value < 0 { -nearest } else { nearest };
                        Ok(Value::HInt(HInt::new(result)))
                    }
                    _ => Ok(Value::HInt(HInt::new(0))),
                }
            }
        }
    }

    fn call_function(&mut self, name: &str, args: &[Expression]) -> Result<Value, String> {
        // Module-qualified calls (e.g., "phi.fold", "phi.res", "core.fib")
        if let Some((module, func)) = name.split_once('.') {
            return self.call_module_function(module, func, args);
        }
        // User-defined functions win over built-ins so that `import core;`
        // can override built-in implementations with the canonical Phase 6
        // versions. Match Python OMC behavior.
        if let Some((params, body)) = self.functions.get(name).cloned() {
            return self.invoke_user_function(name, &params, &body, args);
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
            "exp" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().exp())),
            "sin" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().sin())),
            "cos" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().cos())),
            "tan" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().tan())),
            "tanh" => Ok(Value::HFloat(self.eval_expr(&args[0])?.to_float().tanh())),
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
                        if arr.items.is_empty() {
                            return Err("min: empty array".to_string());
                        }
                        return Ok(Value::HInt(HInt::new(
                            arr.items.iter().map(|v| v.to_int()).min().unwrap(),
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
                        if arr.items.is_empty() {
                            return Err("max: empty array".to_string());
                        }
                        return Ok(Value::HInt(HInt::new(
                            arr.items.iter().map(|v| v.to_int()).max().unwrap(),
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
                    let kept: Vec<Value> =
                        arr.items.into_iter().filter(|v| !v.is_singularity()).collect();
                    Ok(Value::Array(HArray { items: kept }))
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
                        .into_iter()
                        .filter(|v| HInt::compute_resonance(v.to_int()) >= threshold)
                        .collect();
                    Ok(Value::Array(HArray { items: kept }))
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
                    for v in &arr.items {
                        let n = v.to_int().abs();
                        let fibs: [i64; 15] = [
                            0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                        ];
                        let mut nearest = fibs[0];
                        let mut min_dist = n;
                        for &f in &fibs {
                            let d = (f - n).abs();
                            if d < min_dist {
                                min_dist = d;
                                nearest = f;
                            }
                        }
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
                        let fibs: [i64; 15] = [
                            0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                        ];
                        let abs_n = numerator.abs();
                        let mut nearest = fibs[0];
                        let mut min_dist = abs_n;
                        for &fib in &fibs {
                            let d = (fib - abs_n).abs();
                            if d < min_dist {
                                min_dist = d;
                                nearest = fib;
                            }
                        }
                        if numerator < 0 { -nearest } else { nearest }
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
            "str_concat" => {
                if args.len() < 2 {
                    return Err("str_concat requires 2 arguments".to_string());
                }
                let s1 = self.eval_expr(&args[0])?.to_string();
                let s2 = self.eval_expr(&args[1])?.to_string();
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
            // Canonical Python OMC workaround for cross-type concat (string `+` is broken there).
            // Variadic: concat_many(a, b) / concat_many(a, b, c) / concat_many(a, b, c, d).
            // Renders numerics as bare values (89, 1.5) not as HInt(...) display form.
            "concat_many" => {
                let mut out = String::new();
                for a in args {
                    let v = self.eval_expr(a)?;
                    let s = match v {
                        Value::HInt(h) => h.value.to_string(),
                        Value::HFloat(f) => format!("{}", f),
                        Value::String(s) => s,
                        Value::Bool(b) => b.to_string(),
                        other => other.to_string(),
                    };
                    out.push_str(&s);
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
                let mut arr = HArray::with_capacity(size);
                for _ in 0..size {
                    arr.items.push(default.clone());
                }
                Ok(Value::Array(arr))
            }
            "arr_from_range" => {
                if args.len() < 2 {
                    return Err("arr_from_range requires 2 arguments".to_string());
                }
                let start = self.eval_expr(&args[0])?.to_int();
                let end = self.eval_expr(&args[1])?.to_int();
                let mut arr = HArray::new();
                for i in start..end {
                    arr.items.push(Value::HInt(HInt::new(i)));
                }
                Ok(Value::Array(arr))
            }
            "arr_len" => {
                if args.is_empty() {
                    return Err("arr_len requires 1 argument".to_string());
                }
                if let Value::Array(a) = self.eval_expr(&args[0])? {
                    Ok(Value::HInt(HInt::new(a.items.len() as i64)))
                } else {
                    Err("arr_len requires an array".to_string())
                }
            }
            "arr_sum" => {
                if args.is_empty() {
                    return Err("arr_sum requires 1 argument".to_string());
                }
                if let Value::Array(a) = self.eval_expr(&args[0])? {
                    let sum: i64 = a.items.iter().map(|v| v.to_int()).sum();
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
                let val = self.eval_expr(&args[1])?;
                if let Expression::Variable(name) = &args[0] {
                    if let Some(Value::Array(mut arr)) = self.get_var(name) {
                        arr.items.push(val);
                        self.set_var(name.clone(), Value::Array(arr));
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
                    arr.items
                        .get(i)
                        .cloned()
                        .ok_or_else(|| format!("arr_get: index {} out of bounds (len {})", idx, arr.items.len()))
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
                    if let Some(Value::Array(mut arr)) = self.get_var(name) {
                        if idx >= arr.items.len() {
                            return Err(format!(
                                "arr_set: index {} out of bounds (len {})",
                                idx,
                                arr.items.len()
                            ));
                        }
                        arr.items[idx] = val;
                        self.set_var(name.clone(), Value::Array(arr));
                        return Ok(Value::Null);
                    }
                }
                Err("arr_set: first argument must be an array variable".to_string())
            }
            "arr_first" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    arr.items
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
                        .last()
                        .cloned()
                        .ok_or_else(|| "arr_last: empty array".to_string())
                } else {
                    Err("arr_last: requires an array".to_string())
                }
            }
            "arr_min" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    if arr.items.is_empty() {
                        return Err("arr_min: empty array".to_string());
                    }
                    let min = arr.items.iter().map(|v| v.to_int()).min().unwrap();
                    Ok(Value::HInt(HInt::new(min)))
                } else {
                    Err("arr_min: requires an array".to_string())
                }
            }
            "arr_max" => {
                if let Value::Array(arr) = self.eval_expr(&args[0])? {
                    if arr.items.is_empty() {
                        return Err("arr_max: empty array".to_string());
                    }
                    let max = arr.items.iter().map(|v| v.to_int()).max().unwrap();
                    Ok(Value::HInt(HInt::new(max)))
                } else {
                    Err("arr_max: requires an array".to_string())
                }
            }
            "arr_concat" => {
                if args.len() < 2 {
                    return Err("arr_concat requires (array_a, array_b)".to_string());
                }
                let a = self.eval_expr(&args[0])?;
                let b = self.eval_expr(&args[1])?;
                match (a, b) {
                    (Value::Array(mut a), Value::Array(b)) => {
                        a.items.extend(b.items);
                        Ok(Value::Array(a))
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
                    let found = arr.items.iter().any(|v| v.to_int() == target);
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
                    let pos = arr.items.iter().position(|v| v.to_int() == target);
                    Ok(Value::HInt(HInt::new(match pos {
                        Some(i) => i as i64,
                        None => -1,
                    })))
                } else {
                    Err("arr_index_of: first argument must be an array".to_string())
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
                    let end = end.min(arr.items.len());
                    let start = start.min(end);
                    let items: Vec<Value> = arr.items[start..end].to_vec();
                    Ok(Value::Array(HArray { items }))
                } else {
                    Err("arr_slice: first argument must be an array".to_string())
                }
            }
            // Canonical OMC uses bare `len(x)` — polymorphic over arrays and strings.
            "len" => {
                let v = self.eval_expr(&args[0])?;
                match v {
                    Value::Array(a) => Ok(Value::HInt(HInt::new(a.items.len() as i64))),
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
                    if arr.items.is_empty() {
                        return Ok(Value::HFloat(0.0));
                    }
                    let total: f64 = arr
                        .items
                        .iter()
                        .map(|v| HInt::compute_resonance(v.to_int()))
                        .sum();
                    Ok(Value::HFloat(total / arr.items.len() as f64))
                } else {
                    Err("arr_resonance: requires an array".to_string())
                }
            }
            // Unknown name — already checked user-defined functions at the top.
            _ => Err(format!("Undefined function: {}", name)),
        }
    }

    fn invoke_user_function(
        &mut self,
        name: &str,
        params: &[String],
        body: &[Statement],
        args: &[Expression],
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

        self.locals.push(HashMap::new());
        for (param, arg) in params.iter().zip(eval_args) {
            self.set_var(param.clone(), arg);
        }

        for stmt in body {
            self.execute_stmt(stmt)?;
            if self.return_value.is_some() {
                break;
            }
        }

        let result = self.return_value.take().unwrap_or(Value::Null);
        self.locals.pop();
        Ok(result)
    }

    fn get_var(&self, name: &str) -> Option<Value> {
        for scope in self.locals.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        self.globals.get(name).cloned()
    }

    /// Test helper: read a variable from outside the interpreter.
    /// Used by integration tests in `tests/conformance.rs`.
    pub fn get_var_for_testing(&self, name: &str) -> Option<Value> {
        self.get_var(name)
    }

    // ---------- VM bridge helpers ----------
    // Used by the bytecode VM (src/vm.rs) so it can reuse this
    // Interpreter's scope stack + built-in stdlib without duplication.

    pub fn vm_push_scope(&mut self) {
        self.locals.push(HashMap::new());
    }

    pub fn vm_pop_scope(&mut self) {
        if self.locals.len() > 1 {
            self.locals.pop();
        }
    }

    pub fn vm_set_local(&mut self, name: &str, value: Value) {
        self.set_var(name.to_string(), value);
    }

    pub fn vm_get_var(&self, name: &str) -> Option<Value> {
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
        // Stash each evaluated arg in a fresh scope under a synthetic name,
        // then route through call_function with Expression::Variable refs.
        // This reuses ALL existing built-in implementations.
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

    fn set_var(&mut self, name: String, value: Value) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, value);
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
            // Unknown module: fall through to the unqualified name.
            // Lets `core.fib(n)` work after `import core;` without explicit module setup.
            _ => self.call_function(func, args),
        }
    }

    fn phi_fold_n(&self, v: Value, depth: usize) -> Value {
        match v {
            Value::HInt(h) => {
                let mut current = h.value;
                let fibs: [i64; 15] = [
                    0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
                ];
                for _ in 0..depth.max(1) {
                    let abs_val = current.abs();
                    let mut nearest = fibs[0];
                    let mut min_dist = abs_val;
                    for &fib in &fibs {
                        let d = (fib - abs_val).abs();
                        if d < min_dist {
                            min_dist = d;
                            nearest = fib;
                        }
                    }
                    current = if current < 0 { -nearest } else { nearest };
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
