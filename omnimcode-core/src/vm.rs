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
                    let v = self
                        .interp
                        .vm_get_var(name)
                        .ok_or_else(|| format!("Undefined variable: {}", name))?;
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

                    // Prefer user-defined (compiled in this module) over
                    // built-ins, to match the tree-walk interpreter's priority.
                    let result = if let Some(callee) = module.functions.get(name) {
                        self.run_function(callee, &argvals, module)?
                    } else {
                        // Delegate to the tree-walk interpreter's stdlib.
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
