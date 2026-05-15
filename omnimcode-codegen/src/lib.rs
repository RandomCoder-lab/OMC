//! OMNIcode native codegen — LLVM-backed JIT/AOT for hot paths.
//!
//! Session A scope (shipped): lower a `CompiledFunction` whose ops are a
//! pure subset of i64-arithmetic into LLVM IR and JIT it.
//!
//! Session B scope (this file): broaden the bytecode coverage so any
//! pure-i64 OMC fn with locals, comparisons, branches, loops, and
//! recursive self-calls JITs correctly. Specifically supported now:
//!
//! - Stack: LoadConst(Int), Pop
//! - Locals: LoadParam, LoadVar, StoreVar, AssignVar (via entry-block allocas)
//! - Arithmetic: Add/AddInt, Sub/SubInt, Mul/MulInt, Div, Mod, Neg
//! - Bitwise: BitAnd, BitOr, BitXor, BitNot, Shl, Shr
//! - Comparison: Eq, Ne, Lt, Le, Gt, Ge (return i64 0/1)
//! - Logical: And, Or, Not (eager, non-short-circuiting — matches the
//!   bytecode compiler's emission)
//! - Control flow: Jump, JumpIfFalse, JumpIfTrue, Return, ReturnNull
//! - Calls: Op::Call for recursive self-calls (target name == current fn name)
//!
//! Session B does NOT yet handle:
//! - HBit dual-band — Session C
//! - Floats, strings, arrays, dicts, builtins — Session D
//! - Cross-fn calls — Session D
//! - Closures, exception handling, match — much later
//!
//! Why JIT-first: `@hbit` functions need to be cheap to specialize.
//! AOT requires linker integration and shipped-binary changes; JIT
//! gives us "compile on first call, cache the native fn pointer" which
//! is the right shape for a per-fn pragma like `@hbit`.

#![cfg(feature = "llvm-jit")]

use std::collections::HashMap;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module as LlvmModule;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{IntPredicate, OptimizationLevel};

use omnimcode_core::bytecode::{CompiledFunction, Const, Op};

/// JITted-OMC function wrapper. Holds the LLVM ExecutionEngine alive
/// for the lifetime of the compiled code — when this is dropped, the
/// native function pointer becomes invalid.
pub struct JitContext<'ctx> {
    pub context: &'ctx Context,
    pub module: LlvmModule<'ctx>,
    pub engine: ExecutionEngine<'ctx>,
}

/// Error type for codegen failures. Keeps it simple — just a String.
pub type CodegenError = String;

impl<'ctx> JitContext<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self, CodegenError> {
        let module = context.create_module("omc_jit");
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| format!("failed to create JIT engine: {}", e))?;
        Ok(JitContext {
            context,
            module,
            engine,
        })
    }

    /// Lower one CompiledFunction into LLVM IR. Returns the
    /// `FunctionValue` so callers can verify it.
    ///
    /// Session B constraints:
    /// - All params and the return type are `i64`.
    /// - Only the int-flavored op subset listed in the crate docs.
    /// - `Op::Call(name, _)` must target the function being lowered
    ///   (recursion); cross-fn calls are Session D.
    pub fn lower_function(
        &self,
        f: &CompiledFunction,
    ) -> Result<FunctionValue<'ctx>, CodegenError> {
        let lowerer = FunctionLowerer::prepare(self.context, &self.module, f)?;
        lowerer.lower()
    }

    /// JIT-lookup helper for single-arg i64 functions.
    pub unsafe fn get_i64_i64(
        &self,
        name: &str,
    ) -> Result<JitFunction<'_, unsafe extern "C" fn(i64) -> i64>, CodegenError> {
        self.engine
            .get_function(name)
            .map_err(|e| format!("get_function({}): {:?}", name, e))
    }

    /// Two-arg variant — `fn(i64, i64) -> i64`.
    pub unsafe fn get_i64_i64_i64(
        &self,
        name: &str,
    ) -> Result<JitFunction<'_, unsafe extern "C" fn(i64, i64) -> i64>, CodegenError> {
        self.engine
            .get_function(name)
            .map_err(|e| format!("get_function({}): {:?}", name, e))
    }
}

/// Per-function lowering driver. Pulled into its own struct because
/// the body has enough state (block table, var slots, the stack
/// machine, the builder) that threading it all as args to free
/// functions would be noisy.
struct FunctionLowerer<'ctx, 'a> {
    ctx: &'ctx Context,
    builder: Builder<'ctx>,
    function: FunctionValue<'ctx>,
    f: &'a CompiledFunction,

    /// One LLVM basic block per op-index leader, plus the entry block.
    /// Map: bytecode op-index -> the LLVM block whose body begins there.
    blocks: HashMap<usize, BasicBlock<'ctx>>,

    /// Per-local-name stack slot (alloca). Populated lazily as we see
    /// StoreVar / AssignVar / LoadVar. Each slot is `alloca i64`.
    var_slots: HashMap<String, PointerValue<'ctx>>,

    /// `Pop` op-indices we should treat as no-ops because they're the
    /// "cleanup pop" that the bytecode compiler emits after each
    /// JumpIfFalse / JumpIfTrue. The condition value is peeked rather
    /// than popped by the branch ops; the compiler then emits a Pop
    /// in BOTH the fall-through and the branch-target so the operand
    /// stack stays balanced. We model the branches as consume-and-jump
    /// instead, so those cleanup Pops become redundant.
    cleanup_pops: std::collections::HashSet<usize>,
}

impl<'ctx, 'a> FunctionLowerer<'ctx, 'a> {
    fn prepare(
        ctx: &'ctx Context,
        module: &'a LlvmModule<'ctx>,
        f: &'a CompiledFunction,
    ) -> Result<Self, CodegenError> {
        let i64_type = ctx.i64_type();
        let param_types: Vec<_> = f.params.iter().map(|_| i64_type.into()).collect();
        let fn_type = i64_type.fn_type(&param_types, false);
        let function = module.add_function(&f.name, fn_type, None);
        let builder = ctx.create_builder();

        Ok(FunctionLowerer {
            ctx,
            builder,
            function,
            f,
            blocks: HashMap::new(),
            var_slots: HashMap::new(),
            cleanup_pops: std::collections::HashSet::new(),
        })
    }

    /// Two-pass lower: scan for leaders, then emit per-block.
    fn lower(mut self) -> Result<FunctionValue<'ctx>, CodegenError> {
        let entry = self.ctx.append_basic_block(self.function, "entry");
        self.builder.position_at_end(entry);
        self.blocks.insert(0, entry);

        self.collect_leaders()?;
        self.collect_cleanup_pops();
        self.emit_body()?;
        Ok(self.function)
    }

    /// First pass: find op-indices that begin a new basic block. An
    /// op-index is a leader if:
    /// - it's 0 (entry)
    /// - it's the target of a Jump / JumpIfFalse / JumpIfTrue
    /// - it's the op immediately following a terminator (Jump,
    ///   JumpIfFalse, JumpIfTrue, Return, ReturnNull)
    fn collect_leaders(&mut self) -> Result<(), CodegenError> {
        let mut leaders: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        leaders.insert(0);

        for (i, op) in self.f.ops.iter().enumerate() {
            match op {
                Op::Jump(off) | Op::JumpIfFalse(off) | Op::JumpIfTrue(off) => {
                    let target = ((i as i32) + 1 + off) as usize;
                    if target <= self.f.ops.len() {
                        leaders.insert(target);
                    }
                    // Op after a branch starts a new block (fall-through
                    // for conditional jumps, dead-code for unconditional).
                    if i + 1 < self.f.ops.len() {
                        leaders.insert(i + 1);
                    }
                }
                Op::Return | Op::ReturnNull => {
                    if i + 1 < self.f.ops.len() {
                        leaders.insert(i + 1);
                    }
                }
                _ => {}
            }
        }

        // Materialize a BasicBlock for every leader other than 0
        // (which is already the entry block).
        for &leader_idx in &leaders {
            if leader_idx == 0 {
                continue;
            }
            let name = format!("op{}", leader_idx);
            let block = self.ctx.append_basic_block(self.function, &name);
            self.blocks.insert(leader_idx, block);
        }

        Ok(())
    }

    /// Identify which Op::Pop indices are the "cleanup pop" idiom that
    /// the compiler emits after JumpIfFalse / JumpIfTrue. There are two
    /// per branch: one for the fall-through (immediately after the
    /// branch op) and one at the jump target.
    fn collect_cleanup_pops(&mut self) {
        for (i, op) in self.f.ops.iter().enumerate() {
            if let Op::JumpIfFalse(off) | Op::JumpIfTrue(off) = op {
                // Fall-through cleanup: Pop right after the branch op.
                let next = i + 1;
                if matches!(self.f.ops.get(next), Some(Op::Pop)) {
                    self.cleanup_pops.insert(next);
                }
                // Target cleanup: Pop at the branch target.
                let target = ((i as i32) + 1 + off) as usize;
                if matches!(self.f.ops.get(target), Some(Op::Pop)) {
                    self.cleanup_pops.insert(target);
                }
            }
        }
    }

    /// Second pass: walk ops, emit LLVM IR. Stack state is per-block;
    /// we don't propagate values across blocks via phi nodes, which
    /// works because OMC's bytecode-from-statements produces empty-
    /// stack block boundaries (modulo the JumpIfFalse cleanup-Pop
    /// idiom we handle explicitly).
    fn emit_body(&mut self) -> Result<(), CodegenError> {
        let i64_type = self.ctx.i64_type();

        let mut stack: Vec<IntValue<'ctx>> = Vec::new();
        let mut block_terminated = false;

        for i in 0..self.f.ops.len() {
            // Block-leader transitions: if i is a leader (other than 0),
            // close the current block (unless already terminated) with
            // an unconditional branch to the leader's block, then switch
            // to the new block and reset stack.
            if i != 0 {
                if let Some(&new_block) = self.blocks.get(&i) {
                    if !block_terminated {
                        self.builder
                            .build_unconditional_branch(new_block)
                            .map_err(|e| format!("br at op{}: {}", i, e))?;
                    }
                    self.builder.position_at_end(new_block);
                    stack.clear();
                    block_terminated = false;
                }
            }

            let op = &self.f.ops[i];
            match op {
                Op::Nop => {}
                Op::Pop => {
                    if self.cleanup_pops.contains(&i) {
                        // Suppressed cleanup pop — the corresponding
                        // branch op already consumed top-of-stack.
                    } else {
                        stack
                            .pop()
                            .ok_or_else(|| format!("Pop with empty stack at op{}", i))?;
                    }
                }
                Op::LoadConst(idx) => {
                    let c = self.f.constants.get(*idx).ok_or_else(|| {
                        format!("LoadConst out of range at op{}: idx={}", i, idx)
                    })?;
                    let v = match c {
                        Const::Int(n) => i64_type.const_int(*n as u64, true),
                        Const::Bool(b) => i64_type.const_int(*b as u64, false),
                        _ => {
                            return Err(format!(
                                "Session B only supports Const::Int and Const::Bool, got {:?} at op{}",
                                c, i
                            ));
                        }
                    };
                    stack.push(v);
                }
                Op::LoadParam(slot) => {
                    let param = self
                        .function
                        .get_nth_param(*slot as u32)
                        .ok_or_else(|| format!("LoadParam slot={} at op{}", slot, i))?;
                    match param {
                        BasicValueEnum::IntValue(iv) => stack.push(iv),
                        other => {
                            return Err(format!(
                                "non-int param {} at op{}: got {:?}",
                                slot, i, other
                            ));
                        }
                    }
                }
                Op::LoadVar(name) => {
                    let slot = self.get_or_create_slot(name)?;
                    let v = self
                        .builder
                        .build_load(i64_type, slot, &format!("{}_load", name))
                        .map_err(|e| format!("load {} at op{}: {}", name, i, e))?;
                    if let BasicValueEnum::IntValue(iv) = v {
                        stack.push(iv);
                    } else {
                        return Err(format!("load of {} not int at op{}", name, i));
                    }
                }
                Op::StoreVar(name) | Op::AssignVar(name) => {
                    let v = pop(&mut stack, i, "StoreVar/AssignVar")?;
                    let slot = self.get_or_create_slot(name)?;
                    self.builder
                        .build_store(slot, v)
                        .map_err(|e| format!("store {} at op{}: {}", name, i, e))?;
                }
                Op::Add | Op::AddInt => self.bin_int(&mut stack, i, |b, l, r| b.build_int_add(l, r, "add"))?,
                Op::Sub | Op::SubInt => self.bin_int(&mut stack, i, |b, l, r| b.build_int_sub(l, r, "sub"))?,
                Op::Mul | Op::MulInt => self.bin_int(&mut stack, i, |b, l, r| b.build_int_mul(l, r, "mul"))?,
                Op::Div => self.bin_int(&mut stack, i, |b, l, r| b.build_int_signed_div(l, r, "div"))?,
                Op::Mod => self.bin_int(&mut stack, i, |b, l, r| b.build_int_signed_rem(l, r, "rem"))?,
                Op::Neg => {
                    let v = pop(&mut stack, i, "Neg")?;
                    let zero = i64_type.const_int(0, false);
                    let n = self
                        .builder
                        .build_int_sub(zero, v, "neg")
                        .map_err(|e| format!("neg at op{}: {}", i, e))?;
                    stack.push(n);
                }
                Op::BitAnd => self.bin_int(&mut stack, i, |b, l, r| b.build_and(l, r, "and"))?,
                Op::BitOr => self.bin_int(&mut stack, i, |b, l, r| b.build_or(l, r, "or"))?,
                Op::BitXor => self.bin_int(&mut stack, i, |b, l, r| b.build_xor(l, r, "xor"))?,
                Op::BitNot => {
                    let v = pop(&mut stack, i, "BitNot")?;
                    let all_ones = i64_type.const_int(u64::MAX, false);
                    let n = self
                        .builder
                        .build_xor(v, all_ones, "not")
                        .map_err(|e| format!("bitnot at op{}: {}", i, e))?;
                    stack.push(n);
                }
                Op::Shl => self.bin_int(&mut stack, i, |b, l, r| b.build_left_shift(l, r, "shl"))?,
                Op::Shr => self.bin_int(&mut stack, i, |b, l, r| b.build_right_shift(l, r, true, "shr"))?,

                Op::Eq => self.cmp_op(&mut stack, i, IntPredicate::EQ)?,
                Op::Ne => self.cmp_op(&mut stack, i, IntPredicate::NE)?,
                Op::Lt => self.cmp_op(&mut stack, i, IntPredicate::SLT)?,
                Op::Le => self.cmp_op(&mut stack, i, IntPredicate::SLE)?,
                Op::Gt => self.cmp_op(&mut stack, i, IntPredicate::SGT)?,
                Op::Ge => self.cmp_op(&mut stack, i, IntPredicate::SGE)?,

                Op::And => {
                    // Non-short-circuit: pop both, treat zero as false,
                    // non-zero as true. Result is i64 0/1.
                    let r = pop(&mut stack, i, "And rhs")?;
                    let l = pop(&mut stack, i, "And lhs")?;
                    let zero = i64_type.const_int(0, false);
                    let l_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, l, zero, "lb")
                        .map_err(|e| format!("And lhs cmp at op{}: {}", i, e))?;
                    let r_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, r, zero, "rb")
                        .map_err(|e| format!("And rhs cmp at op{}: {}", i, e))?;
                    let combined = self
                        .builder
                        .build_and(l_bool, r_bool, "and")
                        .map_err(|e| format!("And combine at op{}: {}", i, e))?;
                    let extended = self
                        .builder
                        .build_int_z_extend(combined, i64_type, "andi64")
                        .map_err(|e| format!("And extend at op{}: {}", i, e))?;
                    stack.push(extended);
                }
                Op::Or => {
                    let r = pop(&mut stack, i, "Or rhs")?;
                    let l = pop(&mut stack, i, "Or lhs")?;
                    let zero = i64_type.const_int(0, false);
                    let l_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, l, zero, "lb")
                        .map_err(|e| format!("Or lhs cmp at op{}: {}", i, e))?;
                    let r_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, r, zero, "rb")
                        .map_err(|e| format!("Or rhs cmp at op{}: {}", i, e))?;
                    let combined = self
                        .builder
                        .build_or(l_bool, r_bool, "or")
                        .map_err(|e| format!("Or combine at op{}: {}", i, e))?;
                    let extended = self
                        .builder
                        .build_int_z_extend(combined, i64_type, "ori64")
                        .map_err(|e| format!("Or extend at op{}: {}", i, e))?;
                    stack.push(extended);
                }
                Op::Not => {
                    let v = pop(&mut stack, i, "Not")?;
                    let zero = i64_type.const_int(0, false);
                    let is_zero = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, v, zero, "iszero")
                        .map_err(|e| format!("Not cmp at op{}: {}", i, e))?;
                    let extended = self
                        .builder
                        .build_int_z_extend(is_zero, i64_type, "noti64")
                        .map_err(|e| format!("Not extend at op{}: {}", i, e))?;
                    stack.push(extended);
                }

                Op::Jump(off) => {
                    let target = ((i as i32) + 1 + off) as usize;
                    let target_bb = self.blocks.get(&target).copied().ok_or_else(|| {
                        format!("Jump target op{} has no block (idx {})", target, i)
                    })?;
                    self.builder
                        .build_unconditional_branch(target_bb)
                        .map_err(|e| format!("Jump br at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::JumpIfFalse(off) => {
                    let cond_i64 = pop(&mut stack, i, "JumpIfFalse")?;
                    let zero = i64_type.const_int(0, false);
                    let cond_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, cond_i64, zero, "jifcond")
                        .map_err(|e| format!("JumpIfFalse cmp at op{}: {}", i, e))?;
                    let target = ((i as i32) + 1 + off) as usize;
                    let then_bb = self.blocks.get(&(i + 1)).copied().ok_or_else(|| {
                        format!("JumpIfFalse fall-through missing at op{}", i)
                    })?;
                    let else_bb = self.blocks.get(&target).copied().ok_or_else(|| {
                        format!("JumpIfFalse target op{} has no block", target)
                    })?;
                    self.builder
                        .build_conditional_branch(cond_bool, then_bb, else_bb)
                        .map_err(|e| format!("JumpIfFalse br at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::JumpIfTrue(off) => {
                    let cond_i64 = pop(&mut stack, i, "JumpIfTrue")?;
                    let zero = i64_type.const_int(0, false);
                    let cond_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, cond_i64, zero, "jitcond")
                        .map_err(|e| format!("JumpIfTrue cmp at op{}: {}", i, e))?;
                    let target = ((i as i32) + 1 + off) as usize;
                    let then_bb = self.blocks.get(&target).copied().ok_or_else(|| {
                        format!("JumpIfTrue target op{} has no block", target)
                    })?;
                    let else_bb = self.blocks.get(&(i + 1)).copied().ok_or_else(|| {
                        format!("JumpIfTrue fall-through missing at op{}", i)
                    })?;
                    self.builder
                        .build_conditional_branch(cond_bool, then_bb, else_bb)
                        .map_err(|e| format!("JumpIfTrue br at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::Return => {
                    let v = pop(&mut stack, i, "Return")?;
                    self.builder
                        .build_return(Some(&v))
                        .map_err(|e| format!("ret at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::ReturnNull => {
                    let zero = i64_type.const_int(0, false);
                    self.builder
                        .build_return(Some(&zero))
                        .map_err(|e| format!("retnull at op{}: {}", i, e))?;
                    block_terminated = true;
                }

                Op::Call(name, argc) => {
                    // Session B: only recursive self-calls. Cross-fn
                    // calls (Session D) need a callable-resolution
                    // strategy — currently routed through tree-walk's
                    // self.functions map, which codegen can't see.
                    if name != &self.f.name {
                        return Err(format!(
                            "Session B Call only supports recursive self-call; got call to {} at op{}",
                            name, i
                        ));
                    }
                    let mut args: Vec<IntValue<'ctx>> = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(pop(&mut stack, i, "Call arg")?);
                    }
                    args.reverse();
                    let metadata_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                        args.iter().map(|v| (*v).into()).collect();
                    let call = self
                        .builder
                        .build_call(self.function, &metadata_args, "callret")
                        .map_err(|e| format!("Call at op{}: {}", i, e))?;
                    let ret = call
                        .try_as_basic_value()
                        .left()
                        .ok_or_else(|| format!("Call ret at op{} had no value", i))?;
                    if let BasicValueEnum::IntValue(iv) = ret {
                        stack.push(iv);
                    } else {
                        return Err(format!("Call ret not int at op{}", i));
                    }
                }

                other => {
                    return Err(format!(
                        "Session B doesn't yet lower op: {:?} at op{}",
                        other, i
                    ));
                }
            }
        }

        // If we fell off the end of the bytecode without an explicit
        // Return, emit one returning 0. (The compiler doesn't always
        // emit ReturnNull on every path; many functions terminate
        // naturally on the last Op::Return.)
        if !block_terminated {
            let zero = i64_type.const_int(0, false);
            self.builder
                .build_return(Some(&zero))
                .map_err(|e| format!("implicit ret: {}", e))?;
        }

        Ok(())
    }

    /// Get or create the alloca slot for a local. All allocas go in
    /// the entry block per LLVM's standard SSA mem-to-reg pattern.
    fn get_or_create_slot(
        &mut self,
        name: &str,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        if let Some(&p) = self.var_slots.get(name) {
            return Ok(p);
        }
        // Save current position, jump to entry, alloca, restore.
        let current_block = self
            .builder
            .get_insert_block()
            .ok_or_else(|| format!("no insert block when allocating {}", name))?;
        let entry = self.function.get_first_basic_block().unwrap();
        // Position at the start of the entry block, before existing
        // instructions, so the alloca dominates all uses.
        match entry.get_first_instruction() {
            Some(first) => self.builder.position_before(&first),
            None => self.builder.position_at_end(entry),
        }
        let i64_type = self.ctx.i64_type();
        let slot = self
            .builder
            .build_alloca(i64_type, &format!("{}_slot", name))
            .map_err(|e| format!("alloca {}: {}", name, e))?;
        self.builder.position_at_end(current_block);
        self.var_slots.insert(name.to_string(), slot);
        Ok(slot)
    }

    fn bin_int<F>(
        &self,
        stack: &mut Vec<IntValue<'ctx>>,
        op_idx: usize,
        f: F,
    ) -> Result<(), CodegenError>
    where
        F: FnOnce(
            &Builder<'ctx>,
            IntValue<'ctx>,
            IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, inkwell::builder::BuilderError>,
    {
        let rhs = pop(stack, op_idx, "bin rhs")?;
        let lhs = pop(stack, op_idx, "bin lhs")?;
        let v = f(&self.builder, lhs, rhs).map_err(|e| format!("binop at op{}: {}", op_idx, e))?;
        stack.push(v);
        Ok(())
    }

    fn cmp_op(
        &self,
        stack: &mut Vec<IntValue<'ctx>>,
        op_idx: usize,
        pred: IntPredicate,
    ) -> Result<(), CodegenError> {
        let rhs = pop(stack, op_idx, "cmp rhs")?;
        let lhs = pop(stack, op_idx, "cmp lhs")?;
        let i64_type = self.ctx.i64_type();
        let i1 = self
            .builder
            .build_int_compare(pred, lhs, rhs, "cmp")
            .map_err(|e| format!("cmp at op{}: {}", op_idx, e))?;
        let i64v = self
            .builder
            .build_int_z_extend(i1, i64_type, "cmpi64")
            .map_err(|e| format!("cmp ext at op{}: {}", op_idx, e))?;
        stack.push(i64v);
        Ok(())
    }
}

fn pop<'ctx>(
    stack: &mut Vec<IntValue<'ctx>>,
    op_idx: usize,
    context: &str,
) -> Result<IntValue<'ctx>, CodegenError> {
    stack
        .pop()
        .ok_or_else(|| format!("stack underflow at op{} ({})", op_idx, context))
}
