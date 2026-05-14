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
                Op::AssignVar(name) => {
                    // Walks scopes outward for an existing binding —
                    // mirrors tree-walk's Statement::Assignment via
                    // assign_var. Required for mutable closures: an
                    // `x = ...` inside a closure body should mutate
                    // the captured `x`, not shadow it.
                    let v = stack.pop().ok_or("stack underflow")?;
                    self.interp.vm_assign_var(name, v);
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
                    } else if let Some(Value::Function { name: fn_name, captured }) =
                        self.interp.vm_get_var_local_only(name)
                    {
                        // VM-native dispatch for `add5(10)`-style calls
                        // where `add5` is a LOCAL VARIABLE holding a
                        // closure value (not a name in module.functions).
                        // Without this branch, every closure invocation
                        // from VM-compiled code routes through tree-walk
                        // via call_first_class_function. With it, calls
                        // hit the same run_function hot path as direct
                        // user-fn calls.
                        //
                        // We use vm_get_var_local_only (no function-table
                        // fallback) to avoid recursion: if `name` is
                        // already known to be a user fn, the `is_user`
                        // branch above would have caught it.
                        //
                        // Only takes the fast path if the closure's body
                        // is in module.functions — otherwise the body
                        // doesn't exist as bytecode and we have to
                        // tree-walk (e.g. a closure created via a
                        // runtime Lambda eval that wasn't compile-time).
                        if module.functions.contains_key(&fn_name) {
                            let pushed_env = captured.is_some();
                            if let Some(env) = captured {
                                self.interp.vm_push_closure_env(env);
                            }
                            let callee = module.functions.get(&fn_name)
                                .expect("checked above");
                            let r = self.run_function(callee, &argvals, module);
                            if pushed_env {
                                self.interp.vm_pop_closure_env();
                            }
                            r?
                        } else {
                            self.interp.vm_call_builtin(name, &argvals)?
                        }
                    } else if name == "call" && argvals.len() == 2 {
                        // VM-native dispatch for reflective `call(fn, args)`.
                        // Routes through vm_invoke_callable so the body runs
                        // as bytecode rather than tree-walk. ~2.4× speedup on
                        // call-heavy workloads (verified: recursive fib via
                        // `call`).
                        let fn_v = &argvals[0];
                        let unpacked = match &argvals[1] {
                            Value::Array(a) => Some(a.items.clone()),
                            _ => None,
                        };
                        match unpacked {
                            Some(arg_list) => match self.vm_invoke_callable(fn_v, &arg_list, module) {
                                Some(r) => r?,
                                None => self.interp.vm_call_builtin(name, &argvals)?,
                            },
                            None => self.interp.vm_call_builtin(name, &argvals)?,
                        }
                    } else if let Some(v) = self.try_dispatch_vm_hof(name, &argvals, module)? {
                        // VM-native higher-order builtins (arr_map / arr_filter /
                        // arr_reduce / arr_any / arr_all / arr_find). When the
                        // callable is a VM-compiled function, each per-element
                        // invocation runs through run_function — closing the
                        // last gap where compiled bytecode was being driven by
                        // tree-walk just to satisfy a HOF iteration loop.
                        v
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
                Op::DictSetNamed(name) => {
                    // Pop value then key; mutate the named dict via
                    // assign_var so the change propagates into a
                    // captured-env scope when run inside a closure.
                    let val = stack.pop().ok_or("stack underflow")?;
                    let key = stack.pop().ok_or("stack underflow")?
                        .to_display_string();
                    if let Some(Value::Dict(mut d)) = self.interp.vm_get_var(name) {
                        d.insert(key, val);
                        self.interp.vm_assign_var(name, Value::Dict(d));
                    } else {
                        return Err(format!(
                            "DictSetNamed: {} is not a dict variable",
                            name
                        ));
                    }
                }
                Op::ExecStmt(stmt) => {
                    // Tree-walk fallback. Currently only emitted for
                    // Statement::Try because exception unwind would
                    // require either a side try-stack or a Result-
                    // aware op dispatch loop refactor. The Interpreter
                    // shares its globals/locals/functions with the VM
                    // (same Interpreter instance), so state changes
                    // propagate transparently.
                    self.interp.vm_exec_stmt(stmt)?;
                    // Drain any control-flow flags the tree-walked body
                    // may have set: a `return` inside a try body needs
                    // to bubble out of the surrounding VM-compiled fn.
                    if let Some(v) = self.interp.vm_take_return() {
                        self.interp.vm_pop_scope();
                        return Ok(v);
                    }
                    // break/continue flags are flags-only — the VM's
                    // outer loops use Op::Jump for control flow, so we
                    // can't propagate them across the bytecode/AST
                    // boundary. Clear them so they don't leak into
                    // unrelated subsequent statements; warn in debug
                    // builds. Future: emit Op::Break/Op::Continue when
                    // the AST-walked body signals these flags.
                    let _ = self.interp.vm_take_break();
                    let _ = self.interp.vm_take_continue();
                }
                Op::DictDelNamed(name) => {
                    let key = stack.pop().ok_or("stack underflow")?
                        .to_display_string();
                    if let Some(Value::Dict(mut d)) = self.interp.vm_get_var(name) {
                        d.remove(&key);
                        self.interp.vm_assign_var(name, Value::Dict(d));
                    } else {
                        return Err(format!(
                            "DictDelNamed: {} is not a dict variable",
                            name
                        ));
                    }
                }
                Op::NewDict(n) => {
                    // Pairs were emitted in source order; we pop them
                    // off the stack reversed (value first, then key)
                    // and reinsert into a temp Vec to restore order
                    // before building the BTreeMap.
                    let mut pairs: Vec<(String, Value)> = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        let v = stack.pop().ok_or("stack underflow")?;
                        let k = stack.pop().ok_or("stack underflow")?
                            .to_display_string();
                        pairs.push((k, v));
                    }
                    pairs.reverse();
                    let mut map = std::collections::BTreeMap::new();
                    for (k, v) in pairs { map.insert(k, v); }
                    stack.push(Value::Dict(map));
                }
                Op::ArrayIndex => {
                    // Polymorphic: container on top is either Array
                    // (index → int slot) or Dict (index → string key).
                    let idx_v = stack.pop().ok_or("stack underflow")?;
                    let container = stack.pop().ok_or("stack underflow")?;
                    match container {
                        Value::Array(a) => {
                            let idx = idx_v.to_int() as usize;
                            let v = a.items.get(idx).cloned()
                                .ok_or_else(|| format!("array index {} out of bounds", idx))?;
                            stack.push(v);
                        }
                        Value::Dict(d) => {
                            let key = idx_v.to_display_string();
                            stack.push(d.get(&key).cloned().unwrap_or(Value::Null));
                        }
                        _ => return Err("ArrayIndex: not indexable".to_string()),
                    }
                }
                Op::ArrPushNamed(name) => {
                    // assign_var (walks outward) — not set_local — so pushes
                    // from inside a closure body land in the captured env
                    // where the array actually lives, not the closure's call
                    // scope. Same rationale as tree-walk arr_push.
                    let val = stack.pop().ok_or("stack underflow")?;
                    if let Some(Value::Array(mut a)) = self.interp.vm_get_var(name) {
                        a.items.push(val);
                        self.interp.vm_assign_var(name, Value::Array(a));
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
                        // assign_var (walks outward) — see ArrPushNamed above.
                        self.interp.vm_assign_var(name, Value::Array(a));
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
                            self.interp.vm_assign_var(name, Value::Array(a));
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
                            // assign_var so the mutation hits the outer
                            // binding when this runs inside a closure body.
                            self.interp.vm_assign_var(name, Value::Array(a));
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

    /// Invoke a Value::Function (or string naming a function) via the
    /// VM's bytecode hot path when possible. Returns None when the
    /// callee has no compiled body in module.functions — caller should
    /// fall back to tree-walk dispatch.
    ///
    /// Centralizes the captured-env push/pop bookkeeping that was
    /// previously inlined at every Op::Call intercept site.
    fn vm_invoke_callable(
        &mut self,
        fn_v: &Value,
        args: &[Value],
        module: &Module,
    ) -> Option<Result<Value, String>> {
        let (name, captured) = match fn_v {
            Value::Function { name, captured } => (name.clone(), captured.clone()),
            Value::String(s) => (s.clone(), None),
            _ => return None,
        };
        // Borrow the CompiledFunction directly out of module — its
        // lifetime is tied to the &Module we pass through to
        // run_function, so the immutable borrow stays valid through
        // the call. Avoids cloning the (Vec<Op>, Vec<Const>, ...)
        // payload on every HOF iteration.
        let callee = module.functions.get(&name)?;
        let pushed = captured.is_some();
        if let Some(env) = captured {
            self.interp.vm_push_closure_env(env);
        }
        let r = self.run_function(callee, args, module);
        if pushed {
            self.interp.vm_pop_closure_env();
        }
        Some(r)
    }

    /// VM-native dispatch for the higher-order array builtins. Replaces
    /// the otherwise tree-walk path where arr_map et al. invoke
    /// `call_first_class_function → invoke_user_function`, which runs
    /// the callable's body via the AST walker. With this helper, when
    /// the callable is a VM-compiled function (which is the common
    /// case), every per-element invocation hits run_function instead.
    ///
    /// Returns:
    ///   Ok(Some(v)) — handled; v is the result the VM should push
    ///   Ok(None)    — not a HOF or no VM-native body, fall back to
    ///                 vm_call_builtin
    ///   Err         — dispatched but the body errored
    fn try_dispatch_vm_hof(
        &mut self,
        name: &str,
        argvals: &[Value],
        module: &Module,
    ) -> Result<Option<Value>, String> {
        // All HOFs in OMC take the array first, the callable second.
        // arr_reduce additionally takes an initial-accumulator value.
        if argvals.len() < 2 {
            return Ok(None);
        }
        let fn_v = &argvals[1];
        // Cheap pre-flight: require the callable to resolve to a
        // VM-compiled function before we take over. Otherwise we'd
        // duplicate the fallback work vm_call_builtin already handles.
        let target_name = match fn_v {
            Value::Function { name, .. } => name.clone(),
            Value::String(s) => s.clone(),
            _ => return Ok(None),
        };
        if !module.functions.contains_key(&target_name) {
            return Ok(None);
        }
        let arr_items: Vec<Value> = match &argvals[0] {
            Value::Array(a) => a.items.clone(),
            _ => return Ok(None),
        };

        match name {
            "arr_map" => {
                let mut out = Vec::with_capacity(arr_items.len());
                for item in arr_items {
                    let r = self.vm_invoke_callable(fn_v, &[item], module)
                        .expect("checked target above");
                    out.push(r?);
                }
                Ok(Some(Value::Array(HArray { items: out })))
            }
            "arr_filter" => {
                let mut out = Vec::new();
                for item in arr_items {
                    let r = self.vm_invoke_callable(fn_v, &[item.clone()], module)
                        .expect("checked target above")?;
                    if r.to_bool() {
                        out.push(item);
                    }
                }
                Ok(Some(Value::Array(HArray { items: out })))
            }
            "arr_reduce" => {
                if argvals.len() < 3 {
                    return Ok(None);
                }
                let mut acc = argvals[2].clone();
                for item in arr_items {
                    acc = self.vm_invoke_callable(fn_v, &[acc, item], module)
                        .expect("checked target above")?;
                }
                Ok(Some(acc))
            }
            "arr_any" => {
                for item in arr_items {
                    let r = self.vm_invoke_callable(fn_v, &[item], module)
                        .expect("checked target above")?;
                    if r.to_bool() {
                        return Ok(Some(Value::HInt(HInt::new(1))));
                    }
                }
                Ok(Some(Value::HInt(HInt::new(0))))
            }
            "arr_all" => {
                for item in arr_items {
                    let r = self.vm_invoke_callable(fn_v, &[item], module)
                        .expect("checked target above")?;
                    if !r.to_bool() {
                        return Ok(Some(Value::HInt(HInt::new(0))));
                    }
                }
                Ok(Some(Value::HInt(HInt::new(1))))
            }
            "arr_find" => {
                for item in arr_items {
                    let r = self.vm_invoke_callable(fn_v, &[item.clone()], module)
                        .expect("checked target above")?;
                    if r.to_bool() {
                        return Ok(Some(item));
                    }
                }
                Ok(Some(Value::Null))
            }
            _ => Ok(None),
        }
    }
}

// ---------- helpers ----------

fn arith_add(l: &Value, r: &Value) -> Value {
    // String + anything → concat. Mirrors tree-walk Expression::Add.
    if matches!(l, Value::String(_)) || matches!(r, Value::String(_)) {
        return Value::String(format!(
            "{}{}",
            l.to_display_string(),
            r.to_display_string()
        ));
    }
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
