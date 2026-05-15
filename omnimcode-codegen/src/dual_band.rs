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
    /// The LLVM module emit-target. Held so intrinsics that need to
    /// look up/declare external helper fns (llvm.floor.f64, harmony
    /// callback, etc.) can do so without going through transmute.
    module: &'a LlvmModule<'ctx>,
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
            module,
            builder,
            function,
            f,
            v2i64,
            blocks: HashMap::new(),
            var_slots: HashMap::new(),
            cleanup_pops: std::collections::HashSet::new(),
        })
    }

    /// Variant of `prepare` that reuses an already-declared
    /// FunctionValue from the module instead of declaring a new one.
    /// Used by `JitContext::jit_module`'s phase-2 body lowering, which
    /// needs the declarations populated up-front so cross-fn calls
    /// (Session H) can find their targets by name.
    pub(crate) fn prepare_existing(
        ctx: &'ctx Context,
        module: &'a LlvmModule<'ctx>,
        f: &'a CompiledFunction,
    ) -> Result<Self, CodegenError> {
        let i64_type = ctx.i64_type();
        let v2i64 = i64_type.vec_type(2);
        let suffixed = format!("{}_hbit", f.name);
        let function = module
            .get_function(&suffixed)
            .ok_or_else(|| format!("prepare_existing: {} not declared", suffixed))?;
        let builder = ctx.create_builder();
        Ok(DualBandLowerer {
            ctx,
            module,
            builder,
            function,
            f,
            v2i64,
            blocks: HashMap::new(),
            var_slots: HashMap::new(),
            cleanup_pops: std::collections::HashSet::new(),
        })
    }

    /// Convenience wrapper used by `JitContext::jit_module` —
    /// `prepare_existing` then `lower`.
    pub(crate) fn lower_existing(
        ctx: &'ctx Context,
        module: &'a LlvmModule<'ctx>,
        f: &'a CompiledFunction,
    ) -> Result<FunctionValue<'ctx>, CodegenError> {
        Self::prepare_existing(ctx, module, f)?.lower()
    }

    pub(crate) fn lower(mut self) -> Result<FunctionValue<'ctx>, CodegenError> {
        let entry = self.ctx.append_basic_block(self.function, "entry");
        self.builder.position_at_end(entry);
        self.blocks.insert(0, entry);

        self.collect_leaders()?;
        self.collect_cleanup_pops();
        self.bind_params_into_locals()?;
        self.emit_body()?;
        Ok(self.function)
    }

    /// Bind each fn parameter into a named local-variable slot. The
    /// OMC bytecode compiler emits `LoadVar("x")` for parameter access
    /// in fn bodies; we mirror what the bytecode VM does at fn entry
    /// and pre-populate each parameter into a `<2 x i64>` alloca slot
    /// keyed by the parameter name. β = α at entry (matched bands);
    /// later sessions add explicit phi-shadow ops that diverge β.
    fn bind_params_into_locals(&mut self) -> Result<(), CodegenError> {
        for (i, pname) in self.f.params.clone().iter().enumerate() {
            let param = self
                .function
                .get_nth_param(i as u32)
                .ok_or_else(|| format!("hbit bind_params: no param at slot {}", i))?;
            let iv = match param {
                BasicValueEnum::IntValue(iv) => iv,
                _ => {
                    return Err(format!(
                        "hbit bind_params: non-int param at slot {}",
                        i
                    ))
                }
            };
            let v = self.splat(iv, &format!("{}_init", pname))?;
            let slot = self.get_or_create_slot(pname)?;
            self.builder
                .build_store(slot, v)
                .map_err(|e| format!("hbit bind_params store {}: {}", pname, e))?;
        }
        Ok(())
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
                        // Path A.2: floats live on the i64 stack via
                        // bitcast IEEE-754 bit pattern. Float-typed
                        // ops bitcast back to f64 at the boundary.
                        Const::Float(f) => i64_type.const_int(f.to_bits(), false),
                        _ => {
                            return Err(format!(
                                "dual-band lowerer doesn't support {:?} at op{}",
                                c, i
                            ));
                        }
                    };
                    // Matched-band entry: β = α. (Session F adds
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
                // Path A.2: float arithmetic in dual-band mode.
                // <2 x i64> bitcasts to <2 x f64> directly (same total
                // bit-width); both lanes get the float op in parallel.
                // β tracks α through float math the same way it does
                // through int math (matched-band semantics until an
                // explicit phi_shadow re-derives β).
                Op::AddFloat => self.bin_vec_float(&mut stack, i, |b, l, r| b.build_float_add(l, r, "fadd"))?,
                Op::SubFloat => self.bin_vec_float(&mut stack, i, |b, l, r| b.build_float_sub(l, r, "fsub"))?,
                Op::MulFloat => self.bin_vec_float(&mut stack, i, |b, l, r| b.build_float_mul(l, r, "fmul"))?,
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

                // Path A.4: read-only array support.
                //
                // Layout: `alloca [N+1 x i64]`. Slot 0 holds the
                // length; slots 1..=N hold the elements. Self-describing
                // so ArrayLen needs no side-channel.
                //
                // Operand-stack convention: arrays live as
                // pointer-cast-to-i64 on the stack. ptrtoint at push;
                // inttoptr at use. The bit pattern survives storage in
                // user-level h-variables (which are <2 x i64> in
                // dual-band) because lane 0 carries the pointer and
                // matches what ArrayIndex / ArrayLen extract.
                //
                // Arrays live in the fn's stack frame. ArrayIndexAssign
                // (mutable writes) and dynamic resize are out of scope
                // for Path A.4 MVP — see Sessions later for those.
                Op::NewArray(n_elems) => {
                    let v = self.emit_new_array(&mut stack, i, *n_elems)?;
                    stack.push(v);
                }
                Op::ArrayLen => {
                    let arr_v = self.pop(&mut stack, i, "ArrayLen ptr")?;
                    let len = self.emit_array_len(arr_v, i)?;
                    stack.push(self.splat(len, "alen_v")?);
                }
                Op::ArrayIndex => {
                    let idx_v = self.pop(&mut stack, i, "ArrayIndex idx")?;
                    let arr_v = self.pop(&mut stack, i, "ArrayIndex ptr")?;
                    let val = self.emit_array_index(arr_v, idx_v, i)?;
                    stack.push(self.splat(val, "aidx_v")?);
                }

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
                    // HBit intrinsics — intercepted before the generic
                    // user-fn-call path. Pattern-match on (name, argc).
                    if name == "phi_shadow" && *argc == 1 {
                        // Session F: replace β with phi_fold(α) * 1000.
                        // α stays untouched (the user-visible value is
                        // unchanged), β becomes the harmonic shadow.
                        let v = self.pop(&mut stack, i, "phi_shadow arg")?;
                        let new_v = self.emit_phi_shadow(v, i)?;
                        stack.push(new_v);
                        continue;
                    }
                    if name == "harmony" && *argc == 1 {
                        // Session G: harmony() calls the extern Rust
                        // helper `omc_harmony(α, β) -> i64` which
                        // computes the substrate-routed harmony in
                        // [0, 1000]. Pre-declared in JitContext::new
                        // and bound via global mapping.
                        let v = self.pop(&mut stack, i, "harmony arg")?;
                        let h_scalar = self.emit_harmony_call(v, i)?;
                        let h_v = self.splat(h_scalar, "harmony_ret_v")?;
                        stack.push(h_v);
                        continue;
                    }
                    // Path A.2: int↔float boundary intrinsics. The
                    // dual-band carrier is <2 x i64>; we operate on
                    // the α lane only (β is the harmonic shadow,
                    // which doesn't follow the user-visible value
                    // through int↔float conversions).
                    if name == "to_float" && *argc == 1 {
                        let v_v = self.pop(&mut stack, i, "to_float arg")?;
                        let f64_type = self.ctx.f64_type();
                        let alpha = self
                            .builder
                            .build_extract_element(v_v, i64_type.const_int(0, false), "tof_a")
                            .map_err(|e| format!("hbit to_float extract at op{}: {}", i, e))?;
                        let alpha_iv = match alpha {
                            BasicValueEnum::IntValue(iv) => iv,
                            _ => return Err(format!("hbit to_float not int at op{}", i)),
                        };
                        let f = self
                            .builder
                            .build_signed_int_to_float(alpha_iv, f64_type, "tof")
                            .map_err(|e| format!("hbit to_float sitofp at op{}: {}", i, e))?;
                        let ri = self
                            .builder
                            .build_bit_cast(f, i64_type, "tof_i")
                            .map_err(|e| format!("hbit to_float bitcast at op{}: {}", i, e))?
                            .into_int_value();
                        let new_v = self.splat(ri, "tof_v")?;
                        stack.push(new_v);
                        continue;
                    }
                    if name == "to_int" && *argc == 1 {
                        let v_v = self.pop(&mut stack, i, "to_int arg")?;
                        let f64_type = self.ctx.f64_type();
                        let alpha = self
                            .builder
                            .build_extract_element(v_v, i64_type.const_int(0, false), "toi_a")
                            .map_err(|e| format!("hbit to_int extract at op{}: {}", i, e))?;
                        let alpha_iv = match alpha {
                            BasicValueEnum::IntValue(iv) => iv,
                            _ => return Err(format!("hbit to_int not int at op{}", i)),
                        };
                        let v_f = self
                            .builder
                            .build_bit_cast(alpha_iv, f64_type, "toi_f")
                            .map_err(|e| format!("hbit to_int bitcast at op{}: {}", i, e))?
                            .into_float_value();
                        let ri = self
                            .builder
                            .build_float_to_signed_int(v_f, i64_type, "toi")
                            .map_err(|e| format!("hbit to_int fptosi at op{}: {}", i, e))?;
                        let new_v = self.splat(ri, "toi_v")?;
                        stack.push(new_v);
                        continue;
                    }
                    // Resolve the call target. Self-recursion uses
                    // self.function directly. Cross-fn calls (Session
                    // H) look up `<name>_hbit` in the module's symbol
                    // table — populated by jit_module's phase-1
                    // declaration pass before any body emission.
                    let target_fn = if name == &self.f.name {
                        self.function
                    } else {
                        let suffixed = format!("{}_hbit", name);
                        match self.module.get_function(&suffixed) {
                            Some(f) => f,
                            None => {
                                return Err(format!(
                                    "hbit Call target {} not declared (not JIT-eligible) at op{}",
                                    suffixed, i
                                ));
                            }
                        }
                    };
                    // Args: extract α from each vector, pass scalars
                    // (the called fn's caller-facing signature is
                    // scalar i64; it splats internally).
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
                        .build_call(target_fn, &scalar_args, "callret")
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

    /// Session G intrinsic: read α and β out of the vector and call
    /// the extern Rust helper `omc_harmony(α, β) -> i64` which
    /// computes the substrate-routed harmony scaled to [0, 1000].
    /// Returns the i64 result as a scalar — the caller is expected
    /// to splat it back into a vector if needed.
    fn emit_harmony_call(
        &self,
        v: VectorValue<'ctx>,
        op_idx: usize,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        let i64_type = self.ctx.i64_type();
        let alpha = self
            .builder
            .build_extract_element(v, i64_type.const_int(0, false), "harmony_alpha")
            .map_err(|e| format!("harmony extract α at op{}: {}", op_idx, e))?;
        let beta = self
            .builder
            .build_extract_element(v, i64_type.const_int(1, false), "harmony_beta")
            .map_err(|e| format!("harmony extract β at op{}: {}", op_idx, e))?;
        let alpha_iv = match alpha {
            BasicValueEnum::IntValue(iv) => iv,
            _ => return Err(format!("harmony: α not int at op{}", op_idx)),
        };
        let beta_iv = match beta {
            BasicValueEnum::IntValue(iv) => iv,
            _ => return Err(format!("harmony: β not int at op{}", op_idx)),
        };
        // omc_harmony is pre-declared in the module by JitContext::new
        // and bound via add_global_mapping. Look it up by name.
        let harmony_fn = self
            .module
            .get_function("omc_harmony")
            .ok_or_else(|| format!("harmony: omc_harmony not declared at op{}", op_idx))?;
        let call = self
            .builder
            .build_call(
                harmony_fn,
                &[alpha_iv.into(), beta_iv.into()],
                "harmony_call",
            )
            .map_err(|e| format!("harmony call at op{}: {}", op_idx, e))?;
        let ret = call
            .try_as_basic_value()
            .left()
            .ok_or_else(|| format!("harmony call no value at op{}", op_idx))?;
        match ret {
            BasicValueEnum::IntValue(iv) => Ok(iv),
            _ => Err(format!("harmony call ret not int at op{}", op_idx)),
        }
    }

    /// Path A.4: NewArray — pop N values from the operand stack, build
    /// a length-prefixed `[N+1 x i64]` alloca in the entry block, store
    /// the popped values into slots 1..=N (in source order — bytecode
    /// pushes elements left-to-right so popping gives reverse order),
    /// store length N at slot 0, and return the pointer as a splat'd
    /// `<2 x i64>` (lane 0 = ptr-as-i64, lane 1 = same).
    fn emit_new_array(
        &mut self,
        stack: &mut Vec<VectorValue<'ctx>>,
        op_idx: usize,
        n: usize,
    ) -> Result<VectorValue<'ctx>, CodegenError> {
        let i64_type = self.ctx.i64_type();
        // Pop N values (each is a <2 x i64>; we extract α as the
        // user-visible scalar). Reverse to get source order.
        let mut elems: Vec<IntValue<'ctx>> = Vec::with_capacity(n);
        for k in 0..n {
            let v_v = self
                .pop(stack, op_idx, &format!("NewArray elem {}", k))?;
            let alpha = self
                .builder
                .build_extract_element(v_v, i64_type.const_int(0, false), "narr_a")
                .map_err(|e| format!("NewArray extract α at op{}: {}", op_idx, e))?;
            let alpha_iv = match alpha {
                BasicValueEnum::IntValue(iv) => iv,
                _ => return Err(format!("NewArray elem {} not int at op{}", k, op_idx)),
            };
            elems.push(alpha_iv);
        }
        elems.reverse();

        // Allocate [N+1 x i64] in the entry block so the alloca
        // dominates all uses, regardless of which CFG block the
        // NewArray op was emitted from.
        let arr_ty = i64_type.array_type((n as u32) + 1);
        let current_block = self
            .builder
            .get_insert_block()
            .ok_or_else(|| format!("NewArray no insert block at op{}", op_idx))?;
        let entry = self.function.get_first_basic_block().unwrap();
        match entry.get_first_instruction() {
            Some(first) => self.builder.position_before(&first),
            None => self.builder.position_at_end(entry),
        }
        let arr_ptr = self
            .builder
            .build_alloca(arr_ty, &format!("arr_op{}", op_idx))
            .map_err(|e| format!("NewArray alloca at op{}: {}", op_idx, e))?;
        self.builder.position_at_end(current_block);

        // Store length at slot 0.
        let zero32 = self.ctx.i32_type().const_int(0, false);
        let len_gep = unsafe {
            self.builder
                .build_in_bounds_gep(arr_ty, arr_ptr, &[zero32, zero32], "narr_len_gep")
                .map_err(|e| format!("NewArray len gep at op{}: {}", op_idx, e))?
        };
        self.builder
            .build_store(len_gep, i64_type.const_int(n as u64, false))
            .map_err(|e| format!("NewArray len store at op{}: {}", op_idx, e))?;

        // Store elements at slots 1..=N.
        for (k, val) in elems.iter().enumerate() {
            let idx32 = self.ctx.i32_type().const_int((k + 1) as u64, false);
            let elem_gep = unsafe {
                self.builder
                    .build_in_bounds_gep(arr_ty, arr_ptr, &[zero32, idx32], "narr_e_gep")
                    .map_err(|e| format!("NewArray elem{} gep at op{}: {}", k, op_idx, e))?
            };
            self.builder
                .build_store(elem_gep, *val)
                .map_err(|e| format!("NewArray elem{} store at op{}: {}", k, op_idx, e))?;
        }

        // Cast the pointer to i64 and splat into <2 x i64>.
        let ptr_as_i64 = self
            .builder
            .build_ptr_to_int(arr_ptr, i64_type, "narr_ptr_i64")
            .map_err(|e| format!("NewArray ptrtoint at op{}: {}", op_idx, e))?;
        self.splat(ptr_as_i64, "narr_v")
    }

    /// Path A.4: ArrayLen — extract α (pointer-as-i64) from the
    /// vector, inttoptr to a [N+1 x i64] pointer, GEP slot 0, load.
    /// Returns the length as a scalar i64 (caller will splat it).
    fn emit_array_len(
        &self,
        arr_v: VectorValue<'ctx>,
        op_idx: usize,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        let i64_type = self.ctx.i64_type();
        let alpha = self
            .builder
            .build_extract_element(arr_v, i64_type.const_int(0, false), "alen_a")
            .map_err(|e| format!("ArrayLen extract α at op{}: {}", op_idx, e))?;
        let alpha_iv = match alpha {
            BasicValueEnum::IntValue(iv) => iv,
            _ => return Err(format!("ArrayLen ptr not int at op{}", op_idx)),
        };
        // For opaque pointers, GEP needs the element type. We use a
        // single-element pointee `[1 x i64]` to GEP slot 0; the load
        // returns the length we wrote at NewArray time.
        let one_i64 = i64_type.array_type(1);
        let ptr_ty = self.ctx.ptr_type(inkwell::AddressSpace::default());
        let ptr = self
            .builder
            .build_int_to_ptr(alpha_iv, ptr_ty, "alen_ptr")
            .map_err(|e| format!("ArrayLen inttoptr at op{}: {}", op_idx, e))?;
        let zero32 = self.ctx.i32_type().const_int(0, false);
        let len_gep = unsafe {
            self.builder
                .build_in_bounds_gep(one_i64, ptr, &[zero32, zero32], "alen_gep")
                .map_err(|e| format!("ArrayLen gep at op{}: {}", op_idx, e))?
        };
        let len = self
            .builder
            .build_load(i64_type, len_gep, "alen_load")
            .map_err(|e| format!("ArrayLen load at op{}: {}", op_idx, e))?;
        match len {
            BasicValueEnum::IntValue(iv) => Ok(iv),
            _ => Err(format!("ArrayLen load not int at op{}", op_idx)),
        }
    }

    /// Path A.4: ArrayIndex — extract α (pointer) and the user-given
    /// scalar index, GEP to slot `idx + 1` (skipping the length
    /// prefix), load the element. Returns the element as a scalar i64.
    fn emit_array_index(
        &self,
        arr_v: VectorValue<'ctx>,
        idx_v: VectorValue<'ctx>,
        op_idx: usize,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        let i64_type = self.ctx.i64_type();
        let arr_alpha = self
            .builder
            .build_extract_element(arr_v, i64_type.const_int(0, false), "aidx_aptr")
            .map_err(|e| format!("ArrayIndex extract α at op{}: {}", op_idx, e))?;
        let idx_alpha = self
            .builder
            .build_extract_element(idx_v, i64_type.const_int(0, false), "aidx_aix")
            .map_err(|e| format!("ArrayIndex extract idx α at op{}: {}", op_idx, e))?;
        let arr_iv = match arr_alpha {
            BasicValueEnum::IntValue(iv) => iv,
            _ => return Err(format!("ArrayIndex ptr not int at op{}", op_idx)),
        };
        let idx_iv = match idx_alpha {
            BasicValueEnum::IntValue(iv) => iv,
            _ => return Err(format!("ArrayIndex idx not int at op{}", op_idx)),
        };
        let ptr_ty = self.ctx.ptr_type(inkwell::AddressSpace::default());
        let ptr = self
            .builder
            .build_int_to_ptr(arr_iv, ptr_ty, "aidx_ptr")
            .map_err(|e| format!("ArrayIndex inttoptr at op{}: {}", op_idx, e))?;
        // Compute slot index = user_idx + 1 (skip the length prefix).
        let one = i64_type.const_int(1, false);
        let slot = self
            .builder
            .build_int_add(idx_iv, one, "aidx_slot")
            .map_err(|e| format!("ArrayIndex slot calc at op{}: {}", op_idx, e))?;
        // Use `i64` as the GEP element type — equivalent to "i64*"
        // arithmetic. Each step is sizeof(i64) = 8 bytes.
        let elem_gep = unsafe {
            self.builder
                .build_in_bounds_gep(i64_type, ptr, &[slot], "aidx_gep")
                .map_err(|e| format!("ArrayIndex gep at op{}: {}", op_idx, e))?
        };
        let val = self
            .builder
            .build_load(i64_type, elem_gep, "aidx_load")
            .map_err(|e| format!("ArrayIndex load at op{}: {}", op_idx, e))?;
        match val {
            BasicValueEnum::IntValue(iv) => Ok(iv),
            _ => Err(format!("ArrayIndex load not int at op{}", op_idx)),
        }
    }

    /// Session F intrinsic: replace the β lane of a `<2 x i64>`
    /// vector value with the phi-shadow of α.
    ///
    /// phi_fold(α) = frac(α * PHI) — the fractional part of α scaled
    /// by the golden ratio, in [0, 1). We multiply by 1000 to get an
    /// integer-friendly range, then cast back to i64. This matches
    /// the existing `HBitProcessor::phi_fold` semantics used by tree-
    /// walk callers when they want a divergent β.
    ///
    /// After this op, harmony(α, β) is non-trivial: β depends on α
    /// in a way that's stable under matched-band operations (Add a
    /// constant to both → diff preserved → harmony unchanged) and
    /// breaks under operations that touch only one band.
    fn emit_phi_shadow(
        &self,
        v: VectorValue<'ctx>,
        op_idx: usize,
    ) -> Result<VectorValue<'ctx>, CodegenError> {
        let i64_type = self.ctx.i64_type();
        let f64_type = self.ctx.f64_type();
        // Extract α from lane 0.
        let alpha = self
            .builder
            .build_extract_element(v, i64_type.const_int(0, false), "shadow_alpha")
            .map_err(|e| format!("phi_shadow extract α at op{}: {}", op_idx, e))?;
        let alpha_iv = match alpha {
            BasicValueEnum::IntValue(iv) => iv,
            _ => return Err(format!("phi_shadow: α not int at op{}", op_idx)),
        };
        // α_d = (double) α
        let alpha_d = self
            .builder
            .build_signed_int_to_float(alpha_iv, f64_type, "alpha_d")
            .map_err(|e| format!("phi_shadow sitofp at op{}: {}", op_idx, e))?;
        // α_phi = α_d * PHI
        let phi_const = f64_type.const_float(crate::PHI);
        let alpha_phi = self
            .builder
            .build_float_mul(alpha_d, phi_const, "alpha_phi")
            .map_err(|e| format!("phi_shadow mul PHI at op{}: {}", op_idx, e))?;
        // floor(α_phi) via llvm.floor.f64 intrinsic
        let floor_fn = match self.module.get_function("llvm.floor.f64") {
            Some(f) => f,
            None => {
                let ft = f64_type.fn_type(&[f64_type.into()], false);
                self.module.add_function("llvm.floor.f64", ft, None)
            }
        };
        let floor_call = self
            .builder
            .build_call(floor_fn, &[alpha_phi.into()], "alpha_phi_floor")
            .map_err(|e| format!("phi_shadow floor at op{}: {}", op_idx, e))?;
        let floor_val = floor_call
            .try_as_basic_value()
            .left()
            .ok_or_else(|| format!("phi_shadow floor no value at op{}", op_idx))?;
        let floor_f = match floor_val {
            BasicValueEnum::FloatValue(fv) => fv,
            _ => return Err(format!("phi_shadow floor not float at op{}", op_idx)),
        };
        // frac = α_phi - floor(α_phi)  ∈ [0, 1)
        let frac = self
            .builder
            .build_float_sub(alpha_phi, floor_f, "alpha_frac")
            .map_err(|e| format!("phi_shadow sub at op{}: {}", op_idx, e))?;
        // β_d = frac * 1000.0
        let one_thousand = f64_type.const_float(1000.0);
        let beta_d = self
            .builder
            .build_float_mul(frac, one_thousand, "beta_d")
            .map_err(|e| format!("phi_shadow mul1000 at op{}: {}", op_idx, e))?;
        // β = (i64) β_d
        let beta_iv = self
            .builder
            .build_float_to_signed_int(beta_d, i64_type, "beta_i64")
            .map_err(|e| format!("phi_shadow fptosi at op{}: {}", op_idx, e))?;
        // Replace lane 1 of v with β. α (lane 0) is preserved.
        let new_v = self
            .builder
            .build_insert_element(v, beta_iv, i64_type.const_int(1, false), "shadow_v")
            .map_err(|e| format!("phi_shadow insert β at op{}: {}", op_idx, e))?;
        Ok(new_v)
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

    /// Path A.2: float-arithmetic binop on the dual-band vector.
    /// `<2 x i64>` bitcasts to `<2 x f64>` (same 128-bit width); both
    /// lanes get the float op in parallel; result bitcasts back to
    /// `<2 x i64>` for stack storage. Bytecode compiler enforces
    /// type discipline; the JIT just trusts the typed op.
    fn bin_vec_float<F>(
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
        let f64_type = self.ctx.f64_type();
        let v2f64 = f64_type.vec_type(2);
        let rhs = self.pop(stack, op_idx, "fbin rhs")?;
        let lhs = self.pop(stack, op_idx, "fbin lhs")?;
        let lhs_f = self
            .builder
            .build_bit_cast(lhs, v2f64, "fbin_lf")
            .map_err(|e| format!("hbit fbin lhs cast at op{}: {}", op_idx, e))?
            .into_vector_value();
        let rhs_f = self
            .builder
            .build_bit_cast(rhs, v2f64, "fbin_rf")
            .map_err(|e| format!("hbit fbin rhs cast at op{}: {}", op_idx, e))?
            .into_vector_value();
        let r_f = f(&self.builder, lhs_f, rhs_f)
            .map_err(|e| format!("hbit fbinop at op{}: {}", op_idx, e))?;
        let r_i = self
            .builder
            .build_bit_cast(r_f, self.v2i64, "fbin_ri")
            .map_err(|e| format!("hbit fbin ret cast at op{}: {}", op_idx, e))?
            .into_vector_value();
        stack.push(r_i);
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
