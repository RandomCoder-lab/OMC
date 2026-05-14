// omnimcode-core/src/vm.rs — Stack-based VM for OMNIcode bytecode.
//
// Reuses the tree-walk Interpreter's built-in stdlib via a shim:
// when the VM encounters an Op::Call(name, argc) for a built-in,
// it constructs synthetic AST args from the values on its stack
// and delegates to the existing call_function. This avoids
// duplicating ~60 stdlib implementations.

use crate::bytecode::*;
use crate::interpreter::Interpreter;
use crate::value::{HInt, HArray, Value};

pub struct Vm {
    /// Reuses tree-walk Interpreter for built-in stdlib + module imports
    /// + Value handling. The VM only takes over the hot dispatch path.
    interp: Interpreter,
}

impl Vm {
    pub fn new() -> Self {
        Vm {
            interp: Interpreter::new(),
        }
    }

    /// Mutable access to the internal Interpreter — used by main.rs to
    /// pre-register user function definitions before the VM runs, so
    /// first-class function dispatch can resolve them.
    pub fn interp_mut(&mut self) -> &mut Interpreter {
        &mut self.interp
    }

    pub fn run_module(&mut self, module: &Module) -> Result<Value, String> {
        self.run_function(&module.main, &[], module)
    }

    fn run_function(
        &mut self,
        func: &CompiledFunction,
        args: &[Value],
        module: &Module,
    ) -> Result<Value, String> {
        let mut stack: Vec<Value> = Vec::with_capacity(32);
        let mut ip: usize = 0;
        let ops = &func.ops;

        // Push a fresh local scope for this frame; bind parameters.
        self.interp.vm_push_scope();
        for (i, param) in func.params.iter().enumerate() {
            let v = args
                .get(i)
                .cloned()
                .unwrap_or(Value::Null);
            self.interp.vm_set_local(param, v);
        }

        while ip < ops.len() {
            let op = &ops[ip];
            ip += 1;
            match op {
                Op::Nop => {}
                Op::LoadConst(idx) => {
                    stack.push(func.constants[*idx].to_value());
                }
                Op::Pop => {
                    stack.pop();
                }
                Op::LoadVar(name) => {
                    // First try the interpreter's variable lookup (with
                    // its own function-table fallback). If nothing found,
                    // check the Module's function table — that's where
                    // VM-compiled user functions live, and supporting
                    // first-class function values means they need to
                    // resolve as Value::Function here too.
                    let v = if let Some(v) = self.interp.vm_get_var(name) {
                        v
                    } else if module.functions.contains_key(name) {
                        Value::Function { name: name.clone(), captured: None }
                    } else {
                        return Err(format!("Undefined variable: {}", name));
                    };
                    stack.push(v);
                }
                Op::StoreVar(name) => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    self.interp.vm_set_local(name, v);
                }
                Op::LoadParam(_) => {
                    // params are stored as locals; LoadVar is equivalent.
                    return Err("LoadParam not yet implemented".to_string());
                }
                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod => {
                    let r = stack.pop().ok_or("stack underflow")?;
                    let l = stack.pop().ok_or("stack underflow")?;
                    let result = match op {
                        Op::Add => arith_add(&l, &r),
                        Op::Sub => arith_sub(&l, &r),
                        Op::Mul => arith_mul(&l, &r),
                        Op::Div => arith_div(&l, &r),
                        Op::Mod => arith_mod(&l, &r),
                        _ => unreachable!(),
                    };
                    stack.push(result);
                }
                // Typed fast-path arithmetic (Phase M). Skip the runtime
                // is_float() check when the compiler proved both sides have
                // a single concrete type.
                Op::AddInt => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l.wrapping_add(r))));
                }
                Op::SubInt => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l.wrapping_sub(r))));
                }
                Op::MulInt => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l.wrapping_mul(r))));
                }
                Op::AddFloat => {
                    let r = stack.pop().ok_or("stack underflow")?.to_float();
                    let l = stack.pop().ok_or("stack underflow")?.to_float();
                    stack.push(Value::HFloat(l + r));
                }
                Op::SubFloat => {
                    let r = stack.pop().ok_or("stack underflow")?.to_float();
                    let l = stack.pop().ok_or("stack underflow")?.to_float();
                    stack.push(Value::HFloat(l - r));
                }
                Op::MulFloat => {
                    let r = stack.pop().ok_or("stack underflow")?.to_float();
                    let l = stack.pop().ok_or("stack underflow")?.to_float();
                    stack.push(Value::HFloat(l * r));
                }
                Op::Neg => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    if v.is_float() {
                        stack.push(Value::HFloat(-v.to_float()));
                    } else {
                        stack.push(Value::HInt(HInt::new(-v.to_int())));
                    }
                }
                Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                    let r = stack.pop().ok_or("stack underflow")?;
                    let l = stack.pop().ok_or("stack underflow")?;
                    let cmp = cmp_op(&l, &r, op);
                    stack.push(Value::Bool(cmp));
                }
                Op::And => {
                    let r = stack.pop().ok_or("stack underflow")?;
                    let l = stack.pop().ok_or("stack underflow")?;
                    stack.push(Value::Bool(l.to_bool() && r.to_bool()));
                }
                Op::Or => {
                    let r = stack.pop().ok_or("stack underflow")?;
                    let l = stack.pop().ok_or("stack underflow")?;
                    stack.push(Value::Bool(l.to_bool() || r.to_bool()));
                }
                Op::Not => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    stack.push(Value::Bool(!v.to_bool()));
                }
                Op::BitAnd => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l & r)));
                }
                Op::BitOr => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l | r)));
                }
                Op::BitXor => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l ^ r)));
                }
                Op::BitNot => {
                    let v = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(!v)));
                }
                Op::Shl => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l.wrapping_shl((r & 63) as u32))));
                }
                Op::Shr => {
                    let r = stack.pop().ok_or("stack underflow")?.to_int();
                    let l = stack.pop().ok_or("stack underflow")?.to_int();
                    stack.push(Value::HInt(HInt::new(l.wrapping_shr((r & 63) as u32))));
                }
                Op::Jump(offset) => {
                    ip = ((ip as i32) + offset) as usize;
                }
                Op::JumpIfFalse(offset) => {
                    let v = stack.last().ok_or("stack underflow")?;
                    if !v.to_bool() {
                        ip = ((ip as i32) + offset) as usize;
                    }
                }
                Op::JumpIfTrue(offset) => {
                    let v = stack.last().ok_or("stack underflow")?;
                    if v.to_bool() {
                        ip = ((ip as i32) + offset) as usize;
                    }
                }
                Op::Call(name, argc) => {
                    // Pop argc values into a vec (preserving order).
                    let mut argvals: Vec<Value> = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        argvals.push(stack.pop().ok_or("stack underflow")?);
                    }
                    argvals.reverse();

                    // Phase Q: inline call cache. `ip` has been incremented
                    // past the current op, so the cache slot is at `ip - 1`.
                    let cache_ip = ip - 1;
                    let cached = func.call_cache.get(cache_ip).map(|c| c.get()).unwrap_or(0);
                    let is_user = match cached {
                        1 => true,
                        2 => false,
                        _ => {
                            // First execution at this site — probe the function
                            // table and burn the result into the cache.
                            let resolved = module.functions.contains_key(name);
                            if let Some(c) = func.call_cache.get(cache_ip) {
                                c.set(if resolved { 1 } else { 2 });
                            }
                            resolved
                        }
                    };

                    let result = if is_user {
                        // Safe: we already proved this key exists.
                        let callee = module.functions.get(name).expect("inline cache lied");
                        self.run_function(callee, &argvals, module)?
                    } else if name == "call" && argvals.len() == 2 {
                        // VM-native dispatch for reflective `call(fn, args)`.
                        // Without this special case, every reflective call
                        // routes through vm_call_builtin → tree-walk and
                        // loses the bytecode-VM hot-path advantage. With
                        // it, `call(test_name, args)` in the test runner
                        // and `call(fn, args)` everywhere else execute the
                        // body via run_function. Real ~2.4× speedup on
                        // call-heavy workloads (verified on recursive fib
                        // dispatched through `call`).
                        //
                        // Falls through to vm_call_builtin if the target
                        // isn't a VM-compiled function (e.g. tree-walk
                        // builtin called via reflection — rare but valid).
                        let fn_v = &argvals[0];
                        let args_v = &argvals[1];
                        let target_name = match fn_v {
                            Value::Function { name, .. } => Some(name.clone()),
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        };
                        let unpacked_args = match args_v {
                            Value::Array(a) => Some(a.items.clone()),
                            _ => None,
                        };
                        match (target_name, unpacked_args) {
                            (Some(tname), Some(arg_list)) if module.functions.contains_key(&tname) => {
                                // VM-native dispatch path.
                                let captured = if let Value::Function { captured, .. } = fn_v {
                                    captured.clone()
                                } else {
                                    None
                                };
                                let pushed = captured.is_some();
                                if let Some(env) = captured {
                                    self.interp.vm_push_closure_env(env);
                                }
                                let callee = module.functions.get(&tname).expect("checked above");
                                let r = self.run_function(callee, &arg_list, module);
                                if pushed {
                                    // Drop the captured env frame we pushed.
                                    // Use a small helper rather than poking
                                    // interp.locals directly so the VM
                                    // doesn't reach into Interpreter internals.
                                    self.interp.vm_pop_closure_env();
                                }
                                r?
                            }
                            _ => self.interp.vm_call_builtin(name, &argvals)?,
                        }
                    } else {
                        self.interp.vm_call_builtin(name, &argvals)?
                    };
                    stack.push(result);
                }
                Op::Return => {
                    let v = stack.pop().unwrap_or(Value::Null);
                    self.interp.vm_pop_scope();
                    return Ok(v);
                }
                Op::ReturnNull => {
                    self.interp.vm_pop_scope();
                    return Ok(Value::Null);
                }
                Op::NewArray(n) => {
                    let mut items = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        items.push(stack.pop().ok_or("stack underflow")?);
                    }
                    items.reverse();
                    stack.push(Value::Array(HArray { items }));
                }
                Op::ArrayIndex => {
                    let idx = stack.pop().ok_or("stack underflow")?.to_int() as usize;
                    let arr = stack.pop().ok_or("stack underflow")?;
                    if let Value::Array(a) = arr {
                        let v = a
                            .items
                            .get(idx)
                            .cloned()
                            .ok_or_else(|| format!("array index {} out of bounds", idx))?;
                        stack.push(v);
                    } else {
                        return Err("ArrayIndex: not an array".to_string());
                    }
                }
                Op::ArrPushNamed(name) => {
                    let val = stack.pop().ok_or("stack underflow")?;
                    if let Some(Value::Array(mut a)) = self.interp.vm_get_var(name) {
                        a.items.push(val);
                        self.interp.vm_set_local(name, Value::Array(a));
                    } else {
                        return Err(format!(
                            "ArrPushNamed: {} is not an array variable",
                            name
                        ));
                    }
                }
                Op::ArrSetNamed(name) => {
                    let val = stack.pop().ok_or("stack underflow")?;
                    let idx = stack.pop().ok_or("stack underflow")?.to_int() as usize;
                    if let Some(Value::Array(mut a)) = self.interp.vm_get_var(name) {
                        if idx >= a.items.len() {
                            return Err(format!(
                                "ArrSetNamed: index {} out of bounds (len {})",
                                idx,
                                a.items.len()
                            ));
                        }
                        a.items[idx] = val;
                        self.interp.vm_set_local(name, Value::Array(a));
                    } else {
                        return Err(format!(
                            "ArrSetNamed: {} is not an array variable",
                            name
                        ));
                    }
                }
                Op::Lambda(name) => {
                    // Closure creation: push Value::Function with the
                    // current top scope frame as captured env. Sibling
                    // lambdas in the same scope share the same Rc so
                    // mutations propagate (matches tree-walk semantics).
                    // Actual body execution still routes through tree-walk
                    // via call_first_class_function; fast VM-native body
                    // execution is future work.
                    let captured = self.interp.vm_top_scope_rc();
                    stack.push(Value::Function {
                        name: name.clone(),
                        captured,
                    });
                }
                Op::SafeArrSetNamed(name) => {
                    let val = stack.pop().ok_or("stack underflow")?;
                    let raw_idx = stack.pop().ok_or("stack underflow")?.to_int();
                    if let Some(Value::Array(mut a)) = self.interp.vm_get_var(name) {
                        let len = a.items.len();
                        if len > 0 {
                            // Fold onto nearest Fibonacci attractor, then
                            // Euclidean mod by len.
                            let folded = crate::interpreter::fold_to_fibonacci_const(raw_idx);
                            let len_i = len as i64;
                            let mut healed = folded % len_i;
                            if healed < 0 {
                                healed += len_i;
                            }
                            a.items[healed as usize] = val;
                            self.interp.vm_set_local(name, Value::Array(a));
                        }
                        // Empty arrays: silently drop the write (total
                        // semantics — never errors).
                    } else {
                        return Err(format!(
                            "SafeArrSetNamed: {} is not an array variable",
                            name
                        ));
                    }
                }
                Op::ArrayIndexAssign(name) => {
                    let idx = stack.pop().ok_or("stack underflow")?.to_int() as usize;
                    let val = stack.pop().ok_or("stack underflow")?;
                    if let Some(Value::Array(mut a)) = self.interp.vm_get_var(name) {
                        if idx < a.items.len() {
                            a.items[idx] = val;
                            self.interp.vm_set_local(name, Value::Array(a));
                        } else {
                            return Err(format!("array {} index {} out of bounds", name, idx));
                        }
                    } else {
                        return Err(format!("{} is not an array", name));
                    }
                }
                Op::Resonance => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    let r = match v {
                        Value::HInt(h) => h.resonance,
                        Value::HFloat(f) => HInt::compute_resonance(f as i64),
                        _ => 0.0,
                    };
                    stack.push(Value::HFloat(r));
                }
                Op::Fold1 => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    let folded = match v {
                        Value::HInt(h) => fold_to_fibonacci(h.value),
                        Value::HFloat(f) => fold_to_fibonacci(f as i64),
                        _ => 0,
                    };
                    stack.push(Value::HInt(HInt::new(folded)));
                }
                Op::IsFibonacci => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    let n = v.to_int();
                    let is_fib = crate::value::is_fibonacci(n);
                    stack.push(Value::HInt(HInt::new(if is_fib { 1 } else { 0 })));
                }
                Op::Fibonacci => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    let n = v.to_int();
                    stack.push(Value::HInt(HInt::new(crate::value::fibonacci(n))));
                }
                Op::ArrayLen => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    let n = match v {
                        Value::Array(a) => a.items.len() as i64,
                        Value::String(s) => s.chars().count() as i64,
                        _ => 0,
                    };
                    stack.push(Value::HInt(HInt::new(n)));
                }
                Op::HimScore => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    stack.push(Value::HFloat(HInt::compute_him(v.to_int())));
                }
                Op::Print => {
                    let v = stack.pop().ok_or("stack underflow")?;
                    println!("{}", v.to_string());
                }
            }
        }
        self.interp.vm_pop_scope();
        Ok(stack.pop().unwrap_or(Value::Null))
    }
}

// ---------- helpers ----------

fn arith_add(l: &Value, r: &Value) -> Value {
    if l.is_float() || r.is_float() {
        Value::HFloat(l.to_float() + r.to_float())
    } else {
        Value::HInt(HInt::new(l.to_int().wrapping_add(r.to_int())))
    }
}
fn arith_sub(l: &Value, r: &Value) -> Value {
    if l.is_float() || r.is_float() {
        Value::HFloat(l.to_float() - r.to_float())
    } else {
        Value::HInt(HInt::new(l.to_int().wrapping_sub(r.to_int())))
    }
}
fn arith_mul(l: &Value, r: &Value) -> Value {
    if l.is_float() || r.is_float() {
        Value::HFloat(l.to_float() * r.to_float())
    } else {
        Value::HInt(HInt::new(l.to_int().wrapping_mul(r.to_int())))
    }
}
fn arith_div(l: &Value, r: &Value) -> Value {
    if l.is_float() || r.is_float() {
        let r_f = r.to_float();
        if r_f == 0.0 {
            Value::Singularity {
                numerator: l.to_int(),
                denominator: 0,
                context: "div".to_string(),
            }
        } else {
            Value::HFloat(l.to_float() / r_f)
        }
    } else {
        let divisor = r.to_int();
        if divisor == 0 {
            Value::Singularity {
                numerator: l.to_int(),
                denominator: 0,
                context: "div".to_string(),
            }
        } else {
            Value::HInt(HInt::new(l.to_int() / divisor))
        }
    }
}
fn arith_mod(l: &Value, r: &Value) -> Value {
    let divisor = r.to_int();
    if divisor == 0 {
        Value::HInt(HInt::new(0))
    } else {
        Value::HInt(HInt::new(l.to_int() % divisor))
    }
}
fn cmp_op(l: &Value, r: &Value, op: &Op) -> bool {
    // For == and != use the same type-aware equality the tree-walk
    // interpreter does (handles array==string, etc. correctly).
    if matches!(op, Op::Eq) {
        return values_equal_vm(l, r);
    }
    if matches!(op, Op::Ne) {
        return !values_equal_vm(l, r);
    }
    // Ordering on strings is lexicographic.
    if let (Value::String(a), Value::String(b)) = (l, r) {
        return match op {
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
            _ => unreachable!(),
        };
    }
    if l.is_float() || r.is_float() {
        let lf = l.to_float();
        let rf = r.to_float();
        match op {
            Op::Eq => lf == rf,
            Op::Ne => lf != rf,
            Op::Lt => lf < rf,
            Op::Le => lf <= rf,
            Op::Gt => lf > rf,
            Op::Ge => lf >= rf,
            _ => unreachable!(),
        }
    } else {
        let li = l.to_int();
        let ri = r.to_int();
        match op {
            Op::Eq => li == ri,
            Op::Ne => li != ri,
            Op::Lt => li < ri,
            Op::Le => li <= ri,
            Op::Gt => li > ri,
            Op::Ge => li >= ri,
            _ => unreachable!(),
        }
    }
}
/// VM-side analogue of the interpreter's values_equal. Same rules — kept
/// duplicated rather than pub-exported to keep the VM self-contained.
fn values_equal_vm(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Array(x), Value::Array(y)) => {
            if x.items.len() != y.items.len() {
                return false;
            }
            x.items
                .iter()
                .zip(y.items.iter())
                .all(|(p, q)| values_equal_vm(p, q))
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
        (Value::Circuit(_), _) | (_, Value::Circuit(_)) => false,
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
        (Value::Array(_), _) | (_, Value::Array(_)) => false,
        _ => {
            if a.is_float() || b.is_float() {
                a.to_float() == b.to_float()
            } else {
                a.to_int() == b.to_int()
            }
        }
    }
}

fn fold_to_fibonacci(n: i64) -> i64 {
    let fibs: [i64; 15] = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610];
    let abs_val = n.abs();
    let mut nearest = fibs[0];
    let mut min_dist = abs_val;
    for &f in &fibs {
        let d = (f - abs_val).abs();
        if d < min_dist {
            min_dist = d;
            nearest = f;
        }
    }
    if n < 0 { -nearest } else { nearest }
}
