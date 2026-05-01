// src/interpreter.rs - AST execution engine

use crate::ast::*;
use crate::value::{HInt, HArray, Value, fibonacci, is_fibonacci};
use std::collections::HashMap;

pub struct Interpreter {
    globals: HashMap<String, Value>,
    functions: HashMap<String, (Vec<String>, Vec<Statement>)>,
    locals: Vec<HashMap<String, Value>>,
    return_value: Option<Value>,
    break_flag: bool,
    continue_flag: bool,
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
            Expression::Float(f) => Ok(Value::HInt(HInt::new(*f as i64))),
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
                Ok(Value::HInt(HInt::new(lv.to_int() + rv.to_int())))
            }
            Expression::Sub(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                Ok(Value::HInt(HInt::new(lv.to_int() - rv.to_int())))
            }
            Expression::Mul(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                Ok(Value::HInt(HInt::new(lv.to_int() * rv.to_int())))
            }
            Expression::Div(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                let divisor = rv.to_int();
                if divisor == 0 {
                    Ok(Value::HInt(HInt::singularity()))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int() / divisor)))
                }
            }
            Expression::Mod(l, r) => {
                let lv = self.eval_expr(l)?;
                let rv = self.eval_expr(r)?;
                let divisor = rv.to_int();
                if divisor == 0 {
                    Ok(Value::HInt(HInt::new(0)))
                } else {
                    Ok(Value::HInt(HInt::new(lv.to_int() % divisor)))
                }
            }
            Expression::Eq(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::Bool(lv == rv))
            }
            Expression::Ne(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::Bool(lv != rv))
            }
            Expression::Lt(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::Bool(lv < rv))
            }
            Expression::Le(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::Bool(lv <= rv))
            }
            Expression::Gt(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::Bool(lv > rv))
            }
            Expression::Ge(l, r) => {
                let lv = self.eval_expr(l)?.to_int();
                let rv = self.eval_expr(r)?.to_int();
                Ok(Value::Bool(lv >= rv))
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
            Expression::Call { name, args } => {
                self.call_function(name, args)
            }
            Expression::Resonance(e) => {
                let v = self.eval_expr(e)?;
                match v {
                    Value::HInt(h) => Ok(Value::HInt(HInt::new((h.resonance * 1000.0) as i64))),
                    _ => Ok(Value::HInt(HInt::new(0))),
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
        // Built-in functions
        match name {
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
                Ok(Value::Bool(is_fibonacci(n)))
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
                    return Err("arr_push requires 2 arguments".to_string());
                }
                // This would modify the array in place - simplified
                Ok(Value::Null)
            }
            // User-defined functions
            _ => {
                if let Some((params, body)) = self.functions.get(name).cloned() {
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

                    // Create new scope
                    self.locals.push(HashMap::new());
                    for (param, arg) in params.iter().zip(eval_args) {
                        self.set_var(param.clone(), arg);
                    }

                    // Execute function body
                    for stmt in &body {
                        self.execute_stmt(stmt)?;
                        if self.return_value.is_some() {
                            break;
                        }
                    }

                    let result = self.return_value.take().unwrap_or(Value::Null);

                    // Restore scope
                    self.locals.pop();

                    Ok(result)
                } else {
                    Err(format!("Undefined function: {}", name))
                }
            }
        }
    }

    fn get_var(&self, name: &str) -> Option<Value> {
        for scope in self.locals.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        self.globals.get(name).cloned()
    }

    fn set_var(&mut self, name: String, value: Value) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, value);
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
}
