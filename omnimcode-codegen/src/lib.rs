//! OMNIcode native codegen — LLVM-backed JIT/AOT for hot paths.
//!
//! Session A scope: prove the end-to-end pipeline on the smallest possible
//! viable target. This file lowers a `CompiledFunction` whose ops are a
//! pure subset of i64-arithmetic (LoadConst Int, LoadParam, AddInt / Add
//! / SubInt / Sub / MulInt / Mul, Return) into LLVM IR and JIT-compiles
//! it into a native `extern "C" fn(i64, ...) -> i64`.
//!
//! What this is NOT (yet):
//! - HBit dual-band emission. Single-band i64 only. Session C adds
//!   the (α, β) packed pair representation.
//! - General OMC support. Strings, arrays, dicts, branches, calls all
//!   route through tree-walk / VM. Session B extends bytecode coverage.
//! - AOT. JIT only, MCJIT-style, function lookup by name.
//!
//! Why JIT-first: `@hbit` functions need to be cheap to specialize.
//! AOT requires linker integration and shipped-binary changes; JIT
//! gives us "compile on first call, cache the native fn pointer" which
//! is the right shape for a per-fn pragma like `@hbit`.

#![cfg(feature = "llvm-jit")]

use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module as LlvmModule;
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue};
use inkwell::OptimizationLevel;

use omnimcode_core::bytecode::{CompiledFunction, Const, Op};

/// JITted-OMC function wrapper. Holds the LLVM ExecutionEngine alive
/// for the lifetime of the compiled code — when this is dropped, the
/// native function pointer becomes invalid.
///
/// Generic over the function arity for type safety. Session A ships
/// only `extern "C" fn(i64) -> i64` and `extern "C" fn(i64, i64) -> i64`.
/// Subsequent sessions add more arities or a uniform stack-frame
/// calling convention.
pub struct JitContext<'ctx> {
    pub context: &'ctx Context,
    pub module: LlvmModule<'ctx>,
    pub engine: ExecutionEngine<'ctx>,
}

/// Error type for codegen failures. Keeps it simple — just a String.
/// All inkwell builder errors and our own "unsupported op" cases
/// flow through this.
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

    /// Lower one CompiledFunction into LLVM IR. Returns the `FunctionValue`
    /// so callers can verify it or pass it to the engine.
    ///
    /// Session A constraints:
    /// - All params and the return type are `i64`.
    /// - The body must terminate with `Op::Return`.
    /// - Only the int-flavored arithmetic ops are accepted.
    pub fn lower_function(
        &self,
        f: &CompiledFunction,
    ) -> Result<FunctionValue<'ctx>, CodegenError> {
        let i64_type = self.context.i64_type();
        let param_types: Vec<_> = f
            .params
            .iter()
            .map(|_| i64_type.into())
            .collect();
        let fn_type = i64_type.fn_type(&param_types, false);
        let function = self.module.add_function(&f.name, fn_type, None);

        let entry = self.context.append_basic_block(function, "entry");
        let builder = self.context.create_builder();
        builder.position_at_end(entry);

        // Stack-machine state during lowering. Bytecode is stack-based;
        // we maintain a Vec<IntValue> that mirrors the runtime operand
        // stack, pushing/popping LLVM SSA values as we walk ops.
        let mut stack: Vec<IntValue<'ctx>> = Vec::new();
        let pop = |s: &mut Vec<IntValue<'ctx>>| -> Result<IntValue<'ctx>, CodegenError> {
            s.pop().ok_or_else(|| "stack underflow during lowering".to_string())
        };

        for op in &f.ops {
            match op {
                Op::LoadConst(idx) => {
                    let c = f.constants.get(*idx).ok_or_else(|| {
                        format!("LoadConst out of range: idx={}", idx)
                    })?;
                    let v = match c {
                        Const::Int(n) => i64_type.const_int(*n as u64, true),
                        _ => {
                            return Err(format!(
                                "Session A only supports Const::Int, got {:?}",
                                c
                            ));
                        }
                    };
                    stack.push(v);
                }
                Op::LoadParam(slot) => {
                    let param = function.get_nth_param(*slot as u32).ok_or_else(|| {
                        format!("LoadParam out of range: slot={}", slot)
                    })?;
                    let v: IntValue = match param {
                        BasicValueEnum::IntValue(iv) => iv,
                        _ => {
                            return Err("non-int param in Session A".to_string());
                        }
                    };
                    stack.push(v);
                }
                Op::Add | Op::AddInt => {
                    let rhs = pop(&mut stack)?;
                    let lhs = pop(&mut stack)?;
                    let v = builder
                        .build_int_add(lhs, rhs, "addtmp")
                        .map_err(|e| format!("build_int_add: {}", e))?;
                    stack.push(v);
                }
                Op::Sub | Op::SubInt => {
                    let rhs = pop(&mut stack)?;
                    let lhs = pop(&mut stack)?;
                    let v = builder
                        .build_int_sub(lhs, rhs, "subtmp")
                        .map_err(|e| format!("build_int_sub: {}", e))?;
                    stack.push(v);
                }
                Op::Mul | Op::MulInt => {
                    let rhs = pop(&mut stack)?;
                    let lhs = pop(&mut stack)?;
                    let v = builder
                        .build_int_mul(lhs, rhs, "multmp")
                        .map_err(|e| format!("build_int_mul: {}", e))?;
                    stack.push(v);
                }
                Op::Return => {
                    let ret = pop(&mut stack)?;
                    builder
                        .build_return(Some(&ret))
                        .map_err(|e| format!("build_return: {}", e))?;
                    return Ok(function);
                }
                other => {
                    return Err(format!(
                        "Session A doesn't yet lower op: {:?}",
                        other
                    ));
                }
            }
        }

        // Body fell through without an explicit Return.
        Err(format!(
            "function `{}` ended without Op::Return",
            f.name
        ))
    }
}

impl<'ctx> JitContext<'ctx> {
    /// JIT-lookup helper for single-arg i64 functions. Returned
    /// `JitFunction` borrows from `self`, so it can't outlive the
    /// `JitContext` — drop-order is enforced by the borrow checker.
    ///
    /// inkwell's `JitFunction<'ctx, F>` would naively suggest the
    /// returned fn lives as long as the Context, but the actual
    /// invariant is "lives as long as the ExecutionEngine"; tying
    /// it to `&self` here is the right constraint.
    ///
    /// SAFETY: caller must not call the returned fn after `self`
    /// is dropped, and must call it only with appropriate i64 args
    /// (no aliasing / TOCTOU guarantees beyond what LLVM provides).
    pub unsafe fn get_i64_i64(
        &self,
        name: &str,
    ) -> Result<JitFunction<'_, unsafe extern "C" fn(i64) -> i64>, CodegenError> {
        self.engine
            .get_function(name)
            .map_err(|e| format!("get_function({}): {:?}", name, e))
    }

    /// Two-arg variant — `fn(i64, i64) -> i64`. Same drop-order
    /// guarantees as get_i64_i64.
    pub unsafe fn get_i64_i64_i64(
        &self,
        name: &str,
    ) -> Result<JitFunction<'_, unsafe extern "C" fn(i64, i64) -> i64>, CodegenError> {
        self.engine
            .get_function(name)
            .map_err(|e| format!("get_function({}): {:?}", name, e))
    }
}
