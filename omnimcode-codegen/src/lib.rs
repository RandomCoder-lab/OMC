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

mod dual_band;

/// φ — the golden ratio constant. Same value as `omnimcode_core::value::PHI`
/// but kept locally so the dual-band lowerer can use it without leaking
/// a core-private type. Synchronizing the value with core's constant
/// is enforced by the test in `dual_band::tests` (TODO).
pub(crate) const PHI: f64 = 1.6180339887498948482045868343656;

/// Session G runtime helper: compute HBit harmony from raw band
/// values. Exposed with `#[no_mangle] extern "C"` so JIT'd code can
/// call it via a global-mapping binding installed in
/// `JitContext::new`. Returns harmony scaled to `[0, 1000]` integer
/// range (1000 = perfect, 0 = maximally divergent) so the JIT side
/// stays pure-i64 without float-passing-convention concerns.
#[no_mangle]
pub extern "C" fn omc_harmony(alpha: i64, beta: i64) -> i64 {
    let h = omnimcode_core::value::HBit::harmony(alpha, beta);
    (h * 1000.0).round() as i64
}

/// Path L1 runtime helper: call into the substrate's
/// `log_phi_pi_fibonacci` from JIT'd code. Argument and return are
/// passed as raw f64 bit patterns (i64 on the wire) to keep the
/// calling convention pure-i64. The JIT bitcasts at the boundary.
///
/// Without this extern, OMC fns that call `log_phi_pi_fibonacci(x)`
/// (the substrate-routed log) couldn't JIT — including the bucket
/// fn at the heart of harmonic_anomaly.
#[no_mangle]
pub extern "C" fn omc_log_phi_pi_fibonacci(arg_bits: i64) -> i64 {
    let x = f64::from_bits(arg_bits as u64);
    let r = omnimcode_core::phi_pi_fib::log_phi_pi_fibonacci(x);
    r.to_bits() as i64
}

/// Path L1 runtime helper: call into the substrate's
/// `fold_to_nearest_attractor` from JIT'd code. Pure i64 in / out.
#[no_mangle]
pub extern "C" fn omc_fold(value: i64) -> i64 {
    omnimcode_core::phi_pi_fib::fold_to_nearest_attractor(value)
}

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

/// A successfully JIT'd OMC function, presented as an arity-tagged
/// raw function pointer. Callable via `JittedFn::call(args)` for
/// the supported arities (0..=4); larger arities should be folded
/// down via a future uniform-arg-array calling convention.
///
/// SAFETY: the underlying machine code is owned by the
/// `JitContext`/`ExecutionEngine` that produced this struct. Calling
/// after that JitContext is dropped is undefined behavior. In the
/// current Session D design, the main CLI keeps the JitContext
/// alive for the entire program duration (Box::leak), so the
/// invariant holds for normal use.
#[derive(Clone, Copy, Debug)]
pub struct JittedFn {
    pub arity: usize,
    /// Erased fn pointer. Cast to the right `unsafe extern "C" fn`
    /// signature at call time based on `arity`.
    pub fn_ptr: *const (),
}

// SAFETY: a raw function pointer is `Send + Sync` — it's plain data.
// The LLVM-generated machine code is read-only and re-entrant.
unsafe impl Send for JittedFn {}
unsafe impl Sync for JittedFn {}

impl JittedFn {
    /// Call this JITted fn with i64 args. Returns `Some(result)` when
    /// arity matches a supported overload, `None` otherwise. Caller is
    /// responsible for keeping the producing JitContext alive — that's
    /// the unsafe invariant this method enforces minimally (it's
    /// "safe" because we trust the pointer, but a use-after-free of
    /// the JitContext would crash here).
    pub fn call(&self, args: &[i64]) -> Option<i64> {
        if args.len() != self.arity {
            return None;
        }
        unsafe {
            match self.arity {
                0 => {
                    let f: unsafe extern "C" fn() -> i64 = std::mem::transmute(self.fn_ptr);
                    Some(f())
                }
                1 => {
                    let f: unsafe extern "C" fn(i64) -> i64 = std::mem::transmute(self.fn_ptr);
                    Some(f(args[0]))
                }
                2 => {
                    let f: unsafe extern "C" fn(i64, i64) -> i64 = std::mem::transmute(self.fn_ptr);
                    Some(f(args[0], args[1]))
                }
                3 => {
                    let f: unsafe extern "C" fn(i64, i64, i64) -> i64 =
                        std::mem::transmute(self.fn_ptr);
                    Some(f(args[0], args[1], args[2]))
                }
                4 => {
                    let f: unsafe extern "C" fn(i64, i64, i64, i64) -> i64 =
                        std::mem::transmute(self.fn_ptr);
                    Some(f(args[0], args[1], args[2], args[3]))
                }
                _ => None,
            }
        }
    }
}

impl<'ctx> JitContext<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self, CodegenError> {
        let module = context.create_module("omc_jit");
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| format!("failed to create JIT engine: {}", e))?;
        // Pre-declare `omc_harmony` and bind it to the runtime helper
        // so JIT'd code (Session G harmony intrinsic) can call into
        // omnimcode_core::value::HBit::harmony without a per-fn
        // declaration dance. External linkage + global mapping is
        // inkwell's idiom for "Rust fn callable from JIT".
        let i64_type = context.i64_type();
        let harmony_ty = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
        let harmony_fn = module.add_function(
            "omc_harmony",
            harmony_ty,
            Some(inkwell::module::Linkage::External),
        );
        engine.add_global_mapping(&harmony_fn, omc_harmony as *const () as usize);
        // Path L1 helpers: substrate primitives callable from JIT'd
        // code. Same global-mapping idiom as omc_harmony.
        let log_ty = i64_type.fn_type(&[i64_type.into()], false);
        let log_fn = module.add_function(
            "omc_log_phi_pi_fibonacci",
            log_ty,
            Some(inkwell::module::Linkage::External),
        );
        engine.add_global_mapping(
            &log_fn,
            omc_log_phi_pi_fibonacci as *const () as usize,
        );
        let fold_ty = i64_type.fn_type(&[i64_type.into()], false);
        let fold_fn = module.add_function(
            "omc_fold",
            fold_ty,
            Some(inkwell::module::Linkage::External),
        );
        engine.add_global_mapping(&fold_fn, omc_fold as *const () as usize);
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

    /// Lower one CompiledFunction in HBit dual-band mode. The emitted
    /// LLVM IR uses `<2 x i64>` as the carrier for every bytecode-level
    /// i64 value — element 0 is the α band (the classical value the
    /// caller sees), element 1 is the β band (the harmonic shadow).
    /// All ops apply to both lanes in parallel; on x86-64 this lowers
    /// to 128-bit SSE2 vector instructions.
    ///
    /// The emitted function is named `<original_name>_hbit` so a
    /// scalar version (from `lower_function`) and a dual-band version
    /// can coexist in the same module for parity testing.
    ///
    /// Caller-facing signature is still scalar — params come in as
    /// i64 and get splatted to `<α=p, β=p>` at fn entry; the return
    /// extracts the α lane back to i64.
    pub fn lower_function_dual_band(
        &self,
        f: &CompiledFunction,
    ) -> Result<FunctionValue<'ctx>, CodegenError> {
        let lowerer = dual_band::DualBandLowerer::prepare(self.context, &self.module, f)?;
        lowerer.lower()
    }

    /// Try to JIT every user function in a bytecode `Module` in dual-band
    /// mode. Functions whose bodies use ops the codegen layer doesn't
    /// yet support (strings, dicts, builtins, cross-fn calls, etc.)
    /// are silently skipped — they stay routed through the tree-walk
    /// interpreter at runtime.
    ///
    /// Returns a map of `fn_name -> JittedFn` for every fn that did
    /// lower successfully. The native function pointers inside
    /// `JittedFn` are owned by `self` (the underlying ExecutionEngine);
    /// callers must not invoke the returned fns after `self` is dropped.
    ///
    /// The returned name uses the ORIGINAL (un-suffixed) bytecode-side
    /// fn name; under the hood the LLVM module sees `<name>_hbit` per
    /// the dual-band lowerer's naming convention.
    ///
    /// Session D scope: every user fn is attempted. Sessions later
    /// add explicit `@hbit` pragma filtering so non-tagged fns aren't
    /// JIT'd even if they could be.
    pub fn jit_module(
        &self,
        module: &omnimcode_core::bytecode::Module,
    ) -> Result<HashMap<String, JittedFn>, CodegenError> {
        // Three-phase orchestration:
        //
        //   1. DECLARE every user fn in the LLVM module with its
        //      signature (i64 in, i64 out). No body, just the
        //      FunctionValue handle. This must happen before any
        //      body is emitted so the dual-band lowerer can find
        //      cross-fn call targets by name (Session H).
        //
        //   2. LOWER each declared fn's body. The lowerer locates
        //      its own FunctionValue by the suffixed name and emits
        //      blocks/ops into it. Cross-fn calls resolve via the
        //      module's symbol table populated in phase 1.
        //
        //   3. EXTRACT raw fn pointers via typed get_function. This
        //      triggers JIT finalization on a now-complete module,
        //      so cross-fn references resolve correctly.
        //
        // This replaces the two-phase order from Session D, which
        // worked for self-recursion but couldn't handle cross-fn
        // calls because targets weren't declared when their callers
        // tried to reference them.
        let i64_type = self.context.i64_type();

        // Phase 1: declare.
        for (name, cf) in &module.functions {
            let suffixed = format!("{}_hbit", name);
            // Skip if already declared (e.g. omc_harmony from
            // JitContext::new). New names get a fresh declaration.
            if self.module.get_function(&suffixed).is_none() {
                let param_types: Vec<_> =
                    cf.params.iter().map(|_| i64_type.into()).collect();
                let fn_type = i64_type.fn_type(&param_types, false);
                self.module.add_function(&suffixed, fn_type, None);
            }
        }

        // Phase 2: lower bodies. Track names that succeeded and
        // names that failed so we can do dependency cleanup below.
        let mut succeeded: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut failed: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for (name, cf) in &module.functions {
            let suffixed = format!("{}_hbit", name);
            match dual_band::DualBandLowerer::lower_existing(self.context, &self.module, cf) {
                Ok(_) => {
                    succeeded.insert(name.clone());
                }
                Err(_) => {
                    failed.insert(name.clone());
                    // Erase the broken declaration entirely. Any
                    // already-lowered caller that referenced this
                    // symbol will now have a dangling reference,
                    // which the dependency-cleanup pass below
                    // detects and erases. This is the fix for the
                    // L1 silent-wrong-answer bug: previously failed
                    // fns got a "broken stub returns 0" body that
                    // produced wrong results in callers; now the
                    // entire dependency chain is correctly removed
                    // from the JIT registry.
                    if let Some(partial) = self.module.get_function(&suffixed) {
                        unsafe { partial.delete() };
                    }
                }
            }
        }

        // Phase 2b: dependency-cleanup fixpoint. A fn that
        // successfully lowered but whose body calls a `failed` fn is
        // itself broken (its IR contains a dangling reference). Walk
        // each succeeded fn's bytecode, look for Op::Call to failed
        // targets, mark caller as failed too. Iterate until no new
        // failures. Skip intrinsics / builtins (handled inline by the
        // lowerer, not via cross-fn references).
        let intrinsics: std::collections::HashSet<&'static str> = [
            "phi_shadow",
            "harmony",
            "to_int",
            "to_float",
            "harmony_value",
            // L1: substrate primitives lowered as extern Rust calls,
            // not user-fn references.
            "log_phi_pi_fibonacci",
        ]
        .iter()
        .copied()
        .collect();
        loop {
            let mut newly_failed: Vec<String> = Vec::new();
            for name in succeeded.iter() {
                if let Some(cf) = module.functions.get(name) {
                    for op in &cf.ops {
                        if let omnimcode_core::bytecode::Op::Call(target, _argc) = op {
                            if intrinsics.contains(target.as_str()) {
                                continue;
                            }
                            // Self-recursion is fine.
                            if target == name {
                                continue;
                            }
                            if failed.contains(target) {
                                newly_failed.push(name.clone());
                                break;
                            }
                            // Target isn't in the user-fn set at all
                            // (probably a builtin). Trust the
                            // lowerer's intrinsic handling — if it
                            // didn't error during phase 2, the
                            // builtin is supported.
                            if !module.functions.contains_key(target) {
                                continue;
                            }
                            // Target is a user fn but not in
                            // succeeded — it failed. Cascade.
                            if !succeeded.contains(target) {
                                newly_failed.push(name.clone());
                                break;
                            }
                        }
                    }
                }
            }
            if newly_failed.is_empty() {
                break;
            }
            for name in newly_failed {
                let suffixed = format!("{}_hbit", name);
                if let Some(broken) = self.module.get_function(&suffixed) {
                    unsafe { broken.delete() };
                }
                succeeded.remove(&name);
                failed.insert(name);
            }
        }

        // Phase 3: extract fn pointers for everything that survived
        // both lowering and dependency cleanup.
        let mut out: HashMap<String, JittedFn> = HashMap::new();
        for name in &succeeded {
            let suffixed = format!("{}_hbit", name);
            let arity = module.functions.get(name).map(|cf| cf.params.len()).unwrap_or(0);
            match unsafe { self.extract_raw_fn_ptr(&suffixed, arity) } {
                Ok(fn_ptr) => {
                    out.insert(name.clone(), JittedFn { arity, fn_ptr });
                }
                Err(_) => {
                    // Extraction failure → skip; tree-walk handles it.
                }
            }
        }
        Ok(out)
    }

    /// Erase a typed JitFunction down to a `*const ()` pointer for
    /// arity-tagged storage in `JittedFn`. Internal helper for
    /// `jit_module`; the caller is responsible for not invoking the
    /// returned pointer after `self` is dropped.
    unsafe fn extract_raw_fn_ptr(
        &self,
        name: &str,
        arity: usize,
    ) -> Result<*const (), CodegenError> {
        macro_rules! by_arity {
            ($t:ty) => {{
                let jf: JitFunction<'ctx, $t> = self
                    .engine
                    .get_function(name)
                    .map_err(|e| format!("get_function({}): {:?}", name, e))?;
                jf.into_raw() as *const ()
            }};
        }
        let ptr = match arity {
            0 => by_arity!(unsafe extern "C" fn() -> i64),
            1 => by_arity!(unsafe extern "C" fn(i64) -> i64),
            2 => by_arity!(unsafe extern "C" fn(i64, i64) -> i64),
            3 => by_arity!(unsafe extern "C" fn(i64, i64, i64) -> i64),
            4 => by_arity!(unsafe extern "C" fn(i64, i64, i64, i64) -> i64),
            _ => return Err(format!("arity {} not supported in Session D jit_module", arity)),
        };
        Ok(ptr)
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
        self.bind_params_into_locals()?;
        self.emit_body()?;
        Ok(self.function)
    }

    /// Bind each fn parameter into a named local-variable slot.
    /// The OMC bytecode compiler emits `Op::LoadVar("x")` for parameter
    /// access in the body (treating params as locals already in scope).
    /// The bytecode VM and tree-walk interpreter both pre-populate
    /// these bindings before executing the body; we mirror that here
    /// so LoadVar resolves to the actual parameter value rather than
    /// reading from an uninitialized alloca.
    fn bind_params_into_locals(&mut self) -> Result<(), CodegenError> {
        for (i, pname) in self.f.params.clone().iter().enumerate() {
            let param = self
                .function
                .get_nth_param(i as u32)
                .ok_or_else(|| format!("bind_params: no param at slot {}", i))?;
            let iv = match param {
                BasicValueEnum::IntValue(iv) => iv,
                _ => return Err(format!("bind_params: non-int param at slot {}", i)),
            };
            let slot = self.get_or_create_slot(pname)?;
            self.builder
                .build_store(slot, iv)
                .map_err(|e| format!("bind_params store {}: {}", pname, e))?;
        }
        Ok(())
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
                        Const::Float(f) => {
                            // Path A.2: floats live on the i64 stack as
                            // bitcast-i64. const_int(bits) gives the
                            // raw IEEE-754 bit pattern stored as i64;
                            // float-typed ops bitcast it back via
                            // bin_float when consuming.
                            i64_type.const_int(f.to_bits(), false)
                        }
                        _ => {
                            return Err(format!(
                                "scalar lowerer doesn't support {:?} at op{}",
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
                // Float arithmetic — Path A.2.
                //
                // Floats live on the stack as bitcast-i64 (the slot
                // type is uniform i64 throughout the lowerer; floats
                // are interpreted via bitcast at the float-op boundary
                // and bitcast back to i64 for storage). The bytecode
                // compiler only emits the Float-typed ops when it has
                // statically-typed-float operands, so the bitcast
                // assumption is sound at the bytecode level.
                Op::AddFloat => self.bin_float(&mut stack, i, |b, l, r| b.build_float_add(l, r, "fadd"))?,
                Op::SubFloat => self.bin_float(&mut stack, i, |b, l, r| b.build_float_sub(l, r, "fsub"))?,
                Op::MulFloat => self.bin_float(&mut stack, i, |b, l, r| b.build_float_mul(l, r, "fmul"))?,
                Op::DivFloat => self.bin_float(&mut stack, i, |b, l, r| b.build_float_div(l, r, "fdiv"))?,
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
                // J4: float-typed comparisons. Bitcast i64 stack
                // operands to f64, compare with FloatPredicate, zext
                // result back to i64 for stack storage. OEQ/ONE/etc
                // are "ordered" predicates — return false on NaN
                // operands, matching standard float comparison semantics.
                Op::EqFloat => self.cmp_op_float(&mut stack, i, inkwell::FloatPredicate::OEQ)?,
                Op::NeFloat => self.cmp_op_float(&mut stack, i, inkwell::FloatPredicate::ONE)?,
                Op::LtFloat => self.cmp_op_float(&mut stack, i, inkwell::FloatPredicate::OLT)?,
                Op::LeFloat => self.cmp_op_float(&mut stack, i, inkwell::FloatPredicate::OLE)?,
                Op::GtFloat => self.cmp_op_float(&mut stack, i, inkwell::FloatPredicate::OGT)?,
                Op::GeFloat => self.cmp_op_float(&mut stack, i, inkwell::FloatPredicate::OGE)?,

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
                    // Path A.2 intrinsics: int↔float boundary.
                    if name == "to_float" && *argc == 1 {
                        let v = pop(&mut stack, i, "to_float arg")?;
                        let f64_type = self.ctx.f64_type();
                        let f = self
                            .builder
                            .build_signed_int_to_float(v, f64_type, "tof")
                            .map_err(|e| format!("to_float sitofp at op{}: {}", i, e))?;
                        let ri = self
                            .builder
                            .build_bit_cast(f, i64_type, "tof_i")
                            .map_err(|e| format!("to_float bitcast at op{}: {}", i, e))?
                            .into_int_value();
                        stack.push(ri);
                        continue;
                    }
                    if name == "to_int" && *argc == 1 {
                        let v_i = pop(&mut stack, i, "to_int arg")?;
                        let f64_type = self.ctx.f64_type();
                        let v_f = self
                            .builder
                            .build_bit_cast(v_i, f64_type, "toi_f")
                            .map_err(|e| format!("to_int bitcast at op{}: {}", i, e))?
                            .into_float_value();
                        let ri = self
                            .builder
                            .build_float_to_signed_int(v_f, i64_type, "toi")
                            .map_err(|e| format!("to_int fptosi at op{}: {}", i, e))?;
                        stack.push(ri);
                        continue;
                    }
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

    /// Path A.2: float-arithmetic binop. The stack holds i64s; the
    /// operands are interpreted as f64 via bitcast. Result is bitcast
    /// back to i64 for storage. Caller is responsible for ensuring
    /// the operands actually contain float bit-patterns (the bytecode
    /// compiler enforces this via its typed AddFloat/SubFloat/MulFloat
    /// emission; the JIT just trusts the typed op).
    fn bin_float<F>(
        &self,
        stack: &mut Vec<inkwell::values::IntValue<'ctx>>,
        op_idx: usize,
        f: F,
    ) -> Result<(), CodegenError>
    where
        F: FnOnce(
            &Builder<'ctx>,
            inkwell::values::FloatValue<'ctx>,
            inkwell::values::FloatValue<'ctx>,
        ) -> Result<
            inkwell::values::FloatValue<'ctx>,
            inkwell::builder::BuilderError,
        >,
    {
        let f64_type = self.ctx.f64_type();
        let i64_type = self.ctx.i64_type();
        let rhs_i = pop(stack, op_idx, "fbin rhs")?;
        let lhs_i = pop(stack, op_idx, "fbin lhs")?;
        let rhs_f = self
            .builder
            .build_bit_cast(rhs_i, f64_type, "fbin_rf")
            .map_err(|e| format!("fbin rhs cast at op{}: {}", op_idx, e))?
            .into_float_value();
        let lhs_f = self
            .builder
            .build_bit_cast(lhs_i, f64_type, "fbin_lf")
            .map_err(|e| format!("fbin lhs cast at op{}: {}", op_idx, e))?
            .into_float_value();
        let r_f = f(&self.builder, lhs_f, rhs_f)
            .map_err(|e| format!("fbinop at op{}: {}", op_idx, e))?;
        let r_i = self
            .builder
            .build_bit_cast(r_f, i64_type, "fbin_ri")
            .map_err(|e| format!("fbin ret cast at op{}: {}", op_idx, e))?
            .into_int_value();
        stack.push(r_i);
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

    /// J4: float comparison. Bitcast i64 stack operands back to f64,
    /// compare with FloatPredicate (ordered: O*), zext result to i64.
    /// Symmetric to bin_float — operands live as bitcast-i64 on the
    /// stack; we cast at the boundary.
    fn cmp_op_float(
        &self,
        stack: &mut Vec<IntValue<'ctx>>,
        op_idx: usize,
        pred: inkwell::FloatPredicate,
    ) -> Result<(), CodegenError> {
        let rhs_i = pop(stack, op_idx, "fcmp rhs")?;
        let lhs_i = pop(stack, op_idx, "fcmp lhs")?;
        let f64_type = self.ctx.f64_type();
        let i64_type = self.ctx.i64_type();
        let lhs_f = self
            .builder
            .build_bit_cast(lhs_i, f64_type, "fcmp_lf")
            .map_err(|e| format!("fcmp lhs cast at op{}: {}", op_idx, e))?
            .into_float_value();
        let rhs_f = self
            .builder
            .build_bit_cast(rhs_i, f64_type, "fcmp_rf")
            .map_err(|e| format!("fcmp rhs cast at op{}: {}", op_idx, e))?
            .into_float_value();
        let i1 = self
            .builder
            .build_float_compare(pred, lhs_f, rhs_f, "fcmp")
            .map_err(|e| format!("fcmp at op{}: {}", op_idx, e))?;
        let i64v = self
            .builder
            .build_int_z_extend(i1, i64_type, "fcmp_i64")
            .map_err(|e| format!("fcmp ext at op{}: {}", op_idx, e))?;
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
