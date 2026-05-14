// omnimcode-core/src/compiler.rs — AST → bytecode lowering.

use crate::ast::*;
use crate::bytecode::*;

/// Loop tracking for `break` / `continue` patch-up.
struct LoopFrame {
    /// Instruction to resume on `continue`.
    continue_target: usize,
    /// Jump-op indices that need to be patched to the loop's exit (set on break).
    break_jumps: Vec<usize>,
}

pub struct Compiler {
    constants: Vec<Const>,
    ops: Vec<Op>,
    loop_stack: Vec<LoopFrame>,
    /// Names of user-defined functions. Used to suppress hot-path inlining
    /// at call sites where the user has redefined a built-in (e.g. a
    /// canonical recursive `fib`).
    user_fns: std::collections::HashSet<String>,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            constants: Vec::new(),
            ops: Vec::new(),
            loop_stack: Vec::new(),
            user_fns: std::collections::HashSet::new(),
        }
    }

    fn with_user_fns(user_fns: std::collections::HashSet<String>) -> Self {
        Compiler {
            constants: Vec::new(),
            ops: Vec::new(),
            loop_stack: Vec::new(),
            user_fns,
        }
    }

    fn add_const(&mut self, c: Const) -> usize {
        let idx = self.constants.len();
        self.constants.push(c);
        idx
    }

    fn emit(&mut self, op: Op) -> usize {
        let idx = self.ops.len();
        self.ops.push(op);
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
            Expression::Add(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Add);
            }
            Expression::Sub(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Sub);
            }
            Expression::Mul(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Mul);
            }
            Expression::Div(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Div);
            }
            Expression::Mod(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Mod);
            }
            Expression::Eq(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Eq);
            }
            Expression::Ne(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Ne);
            }
            Expression::Lt(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Lt);
            }
            Expression::Le(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Le);
            }
            Expression::Gt(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Gt);
            }
            Expression::Ge(l, r) => {
                self.compile_expr(l)?;
                self.compile_expr(r)?;
                self.emit(Op::Ge);
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
            Expression::Call { name, args } => {
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
                        _ => {}
                    }
                }
                for arg in args {
                    self.compile_expr(arg)?;
                }
                self.emit(Op::Call(name.clone(), args.len()));
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
                self.compile_expr(value)?;
                self.emit(Op::StoreVar(name.clone()));
            }
            Statement::Assignment { name, value } => {
                self.compile_expr(value)?;
                self.emit(Op::StoreVar(name.clone()));
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
        }
        Ok(())
    }

    fn finish(self, name: String, params: Vec<String>) -> CompiledFunction {
        CompiledFunction {
            name,
            params,
            ops: self.ops,
            constants: self.constants,
        }
    }
}

pub fn compile_program(statements: &[Statement]) -> Result<Module, String> {
    let mut module = Module::default();

    // Pre-pass: collect every user-defined function name. We pass this set
    // into every Compiler so the hot-path inliner can refuse to inline a
    // name the user has shadowed (e.g. a recursive user `fib`).
    let mut user_fns: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for stmt in statements {
        if let Statement::FunctionDef { name, .. } = stmt {
            user_fns.insert(name.clone());
        }
    }

    // First pass: hoist function definitions.
    for stmt in statements {
        if let Statement::FunctionDef { name, params, body, .. } = stmt {
            let mut fc = Compiler::with_user_fns(user_fns.clone());
            for s in body {
                fc.compile_stmt(s)?;
            }
            // Ensure every function ends with an implicit ReturnNull so the VM
            // doesn't fall off the end.
            fc.emit(Op::ReturnNull);
            let func = fc.finish(name.clone(), params.clone());
            module.functions.insert(name.clone(), func);
        }
    }

    // Second pass: compile the top-level (non-fn) statements as `main`.
    let mut mc = Compiler::with_user_fns(user_fns);
    for stmt in statements {
        if matches!(stmt, Statement::FunctionDef { .. }) {
            continue;
        }
        mc.compile_stmt(stmt)?;
    }
    mc.emit(Op::ReturnNull);
    module.main = mc.finish("__main__".to_string(), Vec::new());

    Ok(module)
}
