//! Session C — HBit dual-band code generation.
//!
//! Every i64-typed value at the bytecode level becomes a `<2 x i64>`
//! LLVM vector value here. Element 0 is the α band (the user-visible
//! "classical" value); element 1 is the β band (the harmonic shadow).
//! Operations apply to both lanes in parallel, which LLVM lowers to
//! 128-bit SSE2 vector instructions on x86-64. Later sessions widen
//! the carrier to `<8 x i64>` for AVX-512 packed dispatch.
//!
//! Caller-facing API is still scalar:
//! - Params come in as i64 and get splatted into `<α=p, β=p>` at fn
//!   entry, so JIT-lookup with `get_i64_i64` etc still works.
//! - Return value extracts the α lane back to i64.
//!
//! What this proves architecturally: the dual-band representation
//! flows correctly through the JIT pipeline. Every arithmetic op
//! emits a packed vector instruction visible in the LLVM IR. The
//! mechanism is in place for β to carry semantic information
//! distinct from α (Session D adds the explicit shadow ops that
//! make the bands diverge).
//!
//! Session C does NOT yet:
//! - Make α and β diverge automatically (a "PhiShadow" op or builtin
//!   that sets β = phi_fold(α) is Session D).
//! - Emit AVX-512 intrinsics for >128-bit vectors (Session E).
//! - Plumb the @hbit pragma through compile_module to dispatch
//!   tagged fns through this lowerer (Session D).

use std::collections::HashMap;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module as LlvmModule;
use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue, VectorValue};
use inkwell::IntPredicate;

use omnimcode_core::bytecode::{CompiledFunction, Const, Op};

use crate::CodegenError;

/// Per-function dual-band lowering driver. Mirrors `FunctionLowerer`
/// from `lib.rs` but with `<2 x i64>` as the carrier type throughout.
pub(crate) struct DualBandLowerer<'ctx, 'a> {
    ctx: &'ctx Context,
    builder: Builder<'ctx>,
    function: FunctionValue<'ctx>,
    f: &'a CompiledFunction,

    /// `<2 x i64>` vector type. Cached because every op references it.
    v2i64: VectorType<'ctx>,

    /// One LLVM basic block per op-index leader.
    blocks: HashMap<usize, BasicBlock<'ctx>>,

    /// Per-local-name stack slot. Each slot is `alloca <2 x i64>` so
    /// reads/writes are vector-typed throughout.
    var_slots: HashMap<String, PointerValue<'ctx>>,

    /// Same cleanup-Pop idiom as the scalar lowerer — JumpIfFalse /
    /// JumpIfTrue peek rather than pop in the bytecode VM, but we
    /// model them as consume-and-jump, so the cleanup Pops emitted
    /// by the compiler become redundant.
    cleanup_pops: std::collections::HashSet<usize>,
}

impl<'ctx, 'a> DualBandLowerer<'ctx, 'a> {
    pub(crate) fn prepare(
        ctx: &'ctx Context,
        module: &'a LlvmModule<'ctx>,
        f: &'a CompiledFunction,
    ) -> Result<Self, CodegenError> {
        let i64_type = ctx.i64_type();
        let v2i64 = i64_type.vec_type(2);
        // Caller-facing signature is still scalar i64. We splat params
        // to vectors at fn entry and extract α at return.
        let param_types: Vec<_> = f.params.iter().map(|_| i64_type.into()).collect();
        // Mark the dual-band fn with a "_hbit" suffix so it doesn't
        // collide with the scalar version in the same module if both
        // are present (e.g., for parity testing).
        let name = format!("{}_hbit", f.name);
        let fn_type = i64_type.fn_type(&param_types, false);
        let function = module.add_function(&name, fn_type, None);
        let builder = ctx.create_builder();

        Ok(DualBandLowerer {
            ctx,
            builder,
            function,
            f,
            v2i64,
            blocks: HashMap::new(),
            var_slots: HashMap::new(),
            cleanup_pops: std::collections::HashSet::new(),
        })
    }

    pub(crate) fn lower(mut self) -> Result<FunctionValue<'ctx>, CodegenError> {
        let entry = self.ctx.append_basic_block(self.function, "entry");
        self.builder.position_at_end(entry);
        self.blocks.insert(0, entry);

        self.collect_leaders()?;
        self.collect_cleanup_pops();
        self.emit_body()?;
        Ok(self.function)
    }

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
        for &leader_idx in &leaders {
            if leader_idx == 0 {
                continue;
            }
            let block = self
                .ctx
                .append_basic_block(self.function, &format!("op{}", leader_idx));
            self.blocks.insert(leader_idx, block);
        }
        Ok(())
    }

    fn collect_cleanup_pops(&mut self) {
        for (i, op) in self.f.ops.iter().enumerate() {
            if let Op::JumpIfFalse(off) | Op::JumpIfTrue(off) = op {
                let next = i + 1;
                if matches!(self.f.ops.get(next), Some(Op::Pop)) {
                    self.cleanup_pops.insert(next);
                }
                let target = ((i as i32) + 1 + off) as usize;
                if matches!(self.f.ops.get(target), Some(Op::Pop)) {
                    self.cleanup_pops.insert(target);
                }
            }
        }
    }

    fn emit_body(&mut self) -> Result<(), CodegenError> {
        let i64_type = self.ctx.i64_type();

        let mut stack: Vec<VectorValue<'ctx>> = Vec::new();
        let mut block_terminated = false;

        for i in 0..self.f.ops.len() {
            if i != 0 {
                if let Some(&new_block) = self.blocks.get(&i) {
                    if !block_terminated {
                        self.builder
                            .build_unconditional_branch(new_block)
                            .map_err(|e| format!("hbit br at op{}: {}", i, e))?;
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
                    if !self.cleanup_pops.contains(&i) {
                        stack
                            .pop()
                            .ok_or_else(|| format!("hbit Pop empty at op{}", i))?;
                    }
                }
                Op::LoadConst(idx) => {
                    let c = self.f.constants.get(*idx).ok_or_else(|| {
                        format!("hbit LoadConst out of range at op{}: idx={}", i, idx)
                    })?;
                    let alpha = match c {
                        Const::Int(n) => i64_type.const_int(*n as u64, true),
                        Const::Bool(b) => i64_type.const_int(*b as u64, false),
                        _ => {
                            return Err(format!(
                                "Session C only supports Const::Int/Const::Bool, got {:?} at op{}",
                                c, i
                            ));
                        }
                    };
                    // Matched-band entry: β = α. (Session D will add
                    // explicit phi-shadow ops that diverge β.)
                    let v = self.splat(alpha, &format!("const{}_v", idx))?;
                    stack.push(v);
                }
                Op::LoadParam(slot) => {
                    let param = self
                        .function
                        .get_nth_param(*slot as u32)
                        .ok_or_else(|| format!("hbit LoadParam slot={} at op{}", slot, i))?;
                    let iv = match param {
                        BasicValueEnum::IntValue(v) => v,
                        other => {
                            return Err(format!(
                                "hbit non-int param {} at op{}: {:?}",
                                slot, i, other
                            ));
                        }
                    };
                    let v = self.splat(iv, &format!("param{}_v", slot))?;
                    stack.push(v);
                }
                Op::LoadVar(name) => {
                    let slot = self.get_or_create_slot(name)?;
                    let raw = self
                        .builder
                        .build_load(self.v2i64, slot, &format!("{}_load", name))
                        .map_err(|e| format!("hbit load {} at op{}: {}", name, i, e))?;
                    let vv = match raw {
                        BasicValueEnum::VectorValue(vv) => vv,
                        _ => return Err(format!("hbit load of {} not vector at op{}", name, i)),
                    };
                    stack.push(vv);
                }
                Op::StoreVar(name) | Op::AssignVar(name) => {
                    let v = self.pop(&mut stack, i, "Store/AssignVar")?;
                    let slot = self.get_or_create_slot(name)?;
                    self.builder
                        .build_store(slot, v)
                        .map_err(|e| format!("hbit store {} at op{}: {}", name, i, e))?;
                }

                Op::Add | Op::AddInt => self.bin_vec(&mut stack, i, |b, l, r| b.build_int_add(l, r, "add"))?,
                Op::Sub | Op::SubInt => self.bin_vec(&mut stack, i, |b, l, r| b.build_int_sub(l, r, "sub"))?,
                Op::Mul | Op::MulInt => self.bin_vec(&mut stack, i, |b, l, r| b.build_int_mul(l, r, "mul"))?,
                Op::Div => self.bin_vec(&mut stack, i, |b, l, r| b.build_int_signed_div(l, r, "div"))?,
                Op::Mod => self.bin_vec(&mut stack, i, |b, l, r| b.build_int_signed_rem(l, r, "rem"))?,
                Op::Neg => {
                    let v = self.pop(&mut stack, i, "Neg")?;
                    let zero_v = self.v2i64.const_zero();
                    let n = self
                        .builder
                        .build_int_sub(zero_v, v, "neg")
                        .map_err(|e| format!("hbit neg at op{}: {}", i, e))?;
                    stack.push(n);
                }
                Op::BitAnd => self.bin_vec(&mut stack, i, |b, l, r| b.build_and(l, r, "and"))?,
                Op::BitOr => self.bin_vec(&mut stack, i, |b, l, r| b.build_or(l, r, "or"))?,
                Op::BitXor => self.bin_vec(&mut stack, i, |b, l, r| b.build_xor(l, r, "xor"))?,
                Op::BitNot => {
                    let v = self.pop(&mut stack, i, "BitNot")?;
                    let all_ones = i64_type.const_int(u64::MAX, false);
                    let all_ones_v = self.splat(all_ones, "ones_v")?;
                    let n = self
                        .builder
                        .build_xor(v, all_ones_v, "not")
                        .map_err(|e| format!("hbit bitnot at op{}: {}", i, e))?;
                    stack.push(n);
                }
                Op::Shl => self.bin_vec(&mut stack, i, |b, l, r| b.build_left_shift(l, r, "shl"))?,
                Op::Shr => self.bin_vec(&mut stack, i, |b, l, r| b.build_right_shift(l, r, true, "shr"))?,

                Op::Eq => self.cmp_vec(&mut stack, i, IntPredicate::EQ)?,
                Op::Ne => self.cmp_vec(&mut stack, i, IntPredicate::NE)?,
                Op::Lt => self.cmp_vec(&mut stack, i, IntPredicate::SLT)?,
                Op::Le => self.cmp_vec(&mut stack, i, IntPredicate::SLE)?,
                Op::Gt => self.cmp_vec(&mut stack, i, IntPredicate::SGT)?,
                Op::Ge => self.cmp_vec(&mut stack, i, IntPredicate::SGE)?,

                Op::And => self.logical_vec(&mut stack, i, true)?,
                Op::Or => self.logical_vec(&mut stack, i, false)?,
                Op::Not => {
                    let v = self.pop(&mut stack, i, "Not")?;
                    let zero_v = self.v2i64.const_zero();
                    let is_zero = self
                        .builder
                        .build_int_compare(IntPredicate::EQ, v, zero_v, "iszero")
                        .map_err(|e| format!("hbit Not cmp at op{}: {}", i, e))?;
                    let i64v = self
                        .builder
                        .build_int_z_extend(is_zero, self.v2i64, "noti64")
                        .map_err(|e| format!("hbit Not extend at op{}: {}", i, e))?;
                    stack.push(i64v);
                }

                Op::Jump(off) => {
                    let target = ((i as i32) + 1 + off) as usize;
                    let target_bb = self.blocks.get(&target).copied().ok_or_else(|| {
                        format!("hbit Jump target op{} has no block (idx {})", target, i)
                    })?;
                    self.builder
                        .build_unconditional_branch(target_bb)
                        .map_err(|e| format!("hbit Jump br at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::JumpIfFalse(off) => {
                    let cond_v = self.pop(&mut stack, i, "JumpIfFalse")?;
                    // Branch on the α lane only — control flow is
                    // determined by the user-visible value. β is
                    // semantic/observation; it doesn't drive branches.
                    let alpha = self
                        .builder
                        .build_extract_element(cond_v, i64_type.const_int(0, false), "alpha")
                        .map_err(|e| format!("hbit alpha extract at op{}: {}", i, e))?;
                    let alpha_iv = match alpha {
                        BasicValueEnum::IntValue(iv) => iv,
                        _ => return Err(format!("hbit alpha not int at op{}", i)),
                    };
                    let zero = i64_type.const_int(0, false);
                    let cond_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, alpha_iv, zero, "jifcond")
                        .map_err(|e| format!("hbit JumpIfFalse cmp at op{}: {}", i, e))?;
                    let target = ((i as i32) + 1 + off) as usize;
                    let then_bb = self.blocks.get(&(i + 1)).copied().ok_or_else(|| {
                        format!("hbit JumpIfFalse fall-through missing at op{}", i)
                    })?;
                    let else_bb = self.blocks.get(&target).copied().ok_or_else(|| {
                        format!("hbit JumpIfFalse target op{} has no block", target)
                    })?;
                    self.builder
                        .build_conditional_branch(cond_bool, then_bb, else_bb)
                        .map_err(|e| format!("hbit JumpIfFalse br at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::JumpIfTrue(off) => {
                    let cond_v = self.pop(&mut stack, i, "JumpIfTrue")?;
                    let alpha = self
                        .builder
                        .build_extract_element(cond_v, i64_type.const_int(0, false), "alpha")
                        .map_err(|e| format!("hbit alpha extract at op{}: {}", i, e))?;
                    let alpha_iv = match alpha {
                        BasicValueEnum::IntValue(iv) => iv,
                        _ => return Err(format!("hbit alpha not int at op{}", i)),
                    };
                    let zero = i64_type.const_int(0, false);
                    let cond_bool = self
                        .builder
                        .build_int_compare(IntPredicate::NE, alpha_iv, zero, "jitcond")
                        .map_err(|e| format!("hbit JumpIfTrue cmp at op{}: {}", i, e))?;
                    let target = ((i as i32) + 1 + off) as usize;
                    let then_bb = self.blocks.get(&target).copied().ok_or_else(|| {
                        format!("hbit JumpIfTrue target op{} has no block", target)
                    })?;
                    let else_bb = self.blocks.get(&(i + 1)).copied().ok_or_else(|| {
                        format!("hbit JumpIfTrue fall-through missing at op{}", i)
                    })?;
                    self.builder
                        .build_conditional_branch(cond_bool, then_bb, else_bb)
                        .map_err(|e| format!("hbit JumpIfTrue br at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::Return => {
                    let v = self.pop(&mut stack, i, "Return")?;
                    // Return α — the user-visible classical value.
                    let alpha = self
                        .builder
                        .build_extract_element(v, i64_type.const_int(0, false), "ret_alpha")
                        .map_err(|e| format!("hbit ret extract at op{}: {}", i, e))?;
                    let alpha_iv = match alpha {
                        BasicValueEnum::IntValue(iv) => iv,
                        _ => return Err(format!("hbit ret alpha not int at op{}", i)),
                    };
                    self.builder
                        .build_return(Some(&alpha_iv))
                        .map_err(|e| format!("hbit ret at op{}: {}", i, e))?;
                    block_terminated = true;
                }
                Op::ReturnNull => {
                    let zero = i64_type.const_int(0, false);
                    self.builder
                        .build_return(Some(&zero))
                        .map_err(|e| format!("hbit retnull at op{}: {}", i, e))?;
                    block_terminated = true;
                }

                Op::Call(name, argc) => {
                    if name != &self.f.name {
                        return Err(format!(
                            "Session C hbit Call only supports recursive self-call; got call to {} at op{}",
                            name, i
                        ));
                    }
                    // The recursive self-call wants scalar i64 args
                    // (because the caller-facing fn signature is scalar).
                    // Extract α from each vector arg, pass as i64,
                    // splat the scalar return back to a vector.
                    let mut vec_args: Vec<VectorValue<'ctx>> = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        vec_args.push(self.pop(&mut stack, i, "Call arg")?);
                    }
                    vec_args.reverse();
                    let mut scalar_args: Vec<inkwell::values::BasicMetadataValueEnum> =
                        Vec::with_capacity(*argc);
                    for (k, va) in vec_args.iter().enumerate() {
                        let a = self
                            .builder
                            .build_extract_element(
                                *va,
                                i64_type.const_int(0, false),
                                &format!("arg{}_alpha", k),
                            )
                            .map_err(|e| format!("hbit call arg extract at op{}: {}", i, e))?;
                        let a_iv = match a {
                            BasicValueEnum::IntValue(iv) => iv,
                            _ => return Err(format!("hbit call arg not int at op{}", i)),
                        };
                        scalar_args.push(a_iv.into());
                    }
                    let call = self
                        .builder
                        .build_call(self.function, &scalar_args, "callret")
                        .map_err(|e| format!("hbit Call at op{}: {}", i, e))?;
                    let ret = call
                        .try_as_basic_value()
                        .left()
                        .ok_or_else(|| format!("hbit Call ret at op{} had no value", i))?;
                    let ret_iv = match ret {
                        BasicValueEnum::IntValue(iv) => iv,
                        _ => return Err(format!("hbit Call ret not int at op{}", i)),
                    };
                    let v = self.splat(ret_iv, "callret_v")?;
                    stack.push(v);
                }

                other => {
                    return Err(format!(
                        "Session C hbit doesn't yet lower op: {:?} at op{}",
                        other, i
                    ));
                }
            }
        }

        if !block_terminated {
            let zero = i64_type.const_int(0, false);
            self.builder
                .build_return(Some(&zero))
                .map_err(|e| format!("hbit implicit ret: {}", e))?;
        }
        Ok(())
    }

    fn splat(&self, scalar: IntValue<'ctx>, name: &str) -> Result<VectorValue<'ctx>, CodegenError> {
        let i64_type = self.ctx.i64_type();
        let undef = self.v2i64.get_undef();
        let with_alpha = self
            .builder
            .build_insert_element(
                undef,
                scalar,
                i64_type.const_int(0, false),
                &format!("{}_a", name),
            )
            .map_err(|e| format!("splat insert α: {}", e))?;
        let full = self
            .builder
            .build_insert_element(
                with_alpha,
                scalar,
                i64_type.const_int(1, false),
                &format!("{}_b", name),
            )
            .map_err(|e| format!("splat insert β: {}", e))?;
        Ok(full)
    }

    fn get_or_create_slot(
        &mut self,
        name: &str,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        if let Some(&p) = self.var_slots.get(name) {
            return Ok(p);
        }
        let current_block = self
            .builder
            .get_insert_block()
            .ok_or_else(|| format!("hbit no insert block when allocating {}", name))?;
        let entry = self.function.get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(first) => self.builder.position_before(&first),
            None => self.builder.position_at_end(entry),
        }
        let slot = self
            .builder
            .build_alloca(self.v2i64, &format!("{}_slot", name))
            .map_err(|e| format!("hbit alloca {}: {}", name, e))?;
        self.builder.position_at_end(current_block);
        self.var_slots.insert(name.to_string(), slot);
        Ok(slot)
    }

    fn pop(
        &self,
        stack: &mut Vec<VectorValue<'ctx>>,
        op_idx: usize,
        context: &str,
    ) -> Result<VectorValue<'ctx>, CodegenError> {
        stack
            .pop()
            .ok_or_else(|| format!("hbit stack underflow at op{} ({})", op_idx, context))
    }

    fn bin_vec<F>(
        &self,
        stack: &mut Vec<VectorValue<'ctx>>,
        op_idx: usize,
        f: F,
    ) -> Result<(), CodegenError>
    where
        F: FnOnce(
            &Builder<'ctx>,
            VectorValue<'ctx>,
            VectorValue<'ctx>,
        ) -> Result<VectorValue<'ctx>, inkwell::builder::BuilderError>,
    {
        let rhs = self.pop(stack, op_idx, "bin rhs")?;
        let lhs = self.pop(stack, op_idx, "bin lhs")?;
        let v = f(&self.builder, lhs, rhs)
            .map_err(|e| format!("hbit binop at op{}: {}", op_idx, e))?;
        stack.push(v);
        Ok(())
    }

    fn cmp_vec(
        &self,
        stack: &mut Vec<VectorValue<'ctx>>,
        op_idx: usize,
        pred: IntPredicate,
    ) -> Result<(), CodegenError> {
        let rhs = self.pop(stack, op_idx, "cmp rhs")?;
        let lhs = self.pop(stack, op_idx, "cmp lhs")?;
        let cmp_i1 = self
            .builder
            .build_int_compare(pred, lhs, rhs, "cmp")
            .map_err(|e| format!("hbit cmp at op{}: {}", op_idx, e))?;
        let cmp_i64 = self
            .builder
            .build_int_z_extend(cmp_i1, self.v2i64, "cmpi64")
            .map_err(|e| format!("hbit cmp extend at op{}: {}", op_idx, e))?;
        stack.push(cmp_i64);
        Ok(())
    }

    fn logical_vec(
        &self,
        stack: &mut Vec<VectorValue<'ctx>>,
        op_idx: usize,
        is_and: bool,
    ) -> Result<(), CodegenError> {
        let r = self.pop(stack, op_idx, "log rhs")?;
        let l = self.pop(stack, op_idx, "log lhs")?;
        let zero_v = self.v2i64.const_zero();
        let l_bool = self
            .builder
            .build_int_compare(IntPredicate::NE, l, zero_v, "lb")
            .map_err(|e| format!("hbit log lhs at op{}: {}", op_idx, e))?;
        let r_bool = self
            .builder
            .build_int_compare(IntPredicate::NE, r, zero_v, "rb")
            .map_err(|e| format!("hbit log rhs at op{}: {}", op_idx, e))?;
        let combined = if is_and {
            self.builder
                .build_and(l_bool, r_bool, "logand")
                .map_err(|e| format!("hbit log and at op{}: {}", op_idx, e))?
        } else {
            self.builder
                .build_or(l_bool, r_bool, "logor")
                .map_err(|e| format!("hbit log or at op{}: {}", op_idx, e))?
        };
        let extended = self
            .builder
            .build_int_z_extend(combined, self.v2i64, "logi64")
            .map_err(|e| format!("hbit log extend at op{}: {}", op_idx, e))?;
        stack.push(extended);
        Ok(())
    }
}
