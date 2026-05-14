// omnimcode-core/src/compiler.rs — AST → bytecode lowering.

use crate::ast::*;
use crate::bytecode::*;

pub struct Compiler {
    constants: Vec<Const>,
    ops: Vec<Op>,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            constants: Vec::new(),
            ops: Vec::new(),
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
            Expression::Resonance(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Resonance);
            }
            Expression::Fold(e) => {
                self.compile_expr(e)?;
                self.emit(Op::Fold1);
            }
            Expression::Call { name, args } => {
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
                self.compile_expr(condition)?;
                let exit_jump = self.emit(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                // Unconditional jump back to start
                let back = self.emit(Op::Jump(0));
                let offset = (loop_start as i32) - (back as i32) - 1;
                if let Op::Jump(o) = &mut self.ops[back] {
                    *o = offset;
                }
                let exit = self.ops.len();
                self.patch_jump(exit_jump, exit);
                self.emit(Op::Pop); // pop the false condition
            }
            Statement::For { var, iterable, body } => {
                // Desugar `for v in start..end { body }` to:
                //   v = start;
                //   while v < end { body; v = v + 1; }
                // We don't compile iterable-from-array form (rare in canonical).
                let (start_expr, end_expr) = match iterable {
                    ForIterable::Range { start, end } => (start.clone(), end.clone()),
                    ForIterable::Expr(_) => {
                        return Err(
                            "VM: for-over-array not yet supported, use while".to_string()
                        )
                    }
                };
                // var = start;
                self.compile_expr(&start_expr)?;
                self.emit(Op::StoreVar(var.clone()));

                let loop_start = self.ops.len();
                // condition: var < end
                self.emit(Op::LoadVar(var.clone()));
                self.compile_expr(&end_expr)?;
                self.emit(Op::Lt);
                let exit_jump = self.emit(Op::JumpIfFalse(0));
                self.emit(Op::Pop);

                for stmt in body {
                    self.compile_stmt(stmt)?;
                }
                // var = var + 1;
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
            }
            Statement::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(e)?;
                    self.emit(Op::Return);
                } else {
                    self.emit(Op::ReturnNull);
                }
            }
            // Skip break/continue/import/functiondef in expression compiler;
            // function defs handled at module level.
            Statement::Break | Statement::Continue | Statement::Import { .. } => {
                // No bytecode emitted. Break/continue are tricky inside compiled
                // loops — for now the VM only supports clean while/for exits.
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

    // First pass: hoist function definitions.
    for stmt in statements {
        if let Statement::FunctionDef { name, params, body, .. } = stmt {
            let mut fc = Compiler::new();
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
    let mut mc = Compiler::new();
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
