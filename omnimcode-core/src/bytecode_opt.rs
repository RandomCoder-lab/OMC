// omnimcode-core/src/bytecode_opt.rs — Peephole + constant-folding passes
// over compiled OMNIcode bytecode.
//
// Design: every pass that removes an op replaces it with `Op::Nop`
// instead of actually shrinking the Vec, so already-computed jump
// offsets stay valid. The VM treats Nop as a free no-op. Worth ~3
// cycles per Nop in the hot loop, but simpler to maintain than a
// full re-emit pass that would have to walk all jumps and recompute
// offsets. For the kind of programs OMNIcode runs (small kernels +
// recursion, not megaword loops), the simplicity wins.

use crate::bytecode::*;

#[derive(Debug, Default, Clone)]
pub struct OptStats {
    pub constants_folded: usize,
    pub dead_loads_removed: usize,
    pub double_nots_collapsed: usize,
    pub double_negs_collapsed: usize,
    /// Pure-unary ops on constants folded: res(89), phi.fold(N), fibonacci(N),
    /// is_fibonacci(N), HimScore(N), -N, !N, ~N, etc.
    pub unary_calls_cached: usize,
    /// Nop holes removed after all peephole passes + jump offsets rewritten.
    pub nops_compacted: usize,
}

impl OptStats {
    pub fn total(&self) -> usize {
        self.constants_folded
            + self.dead_loads_removed
            + self.double_nots_collapsed
            + self.double_negs_collapsed
            + self.unary_calls_cached
            + self.nops_compacted
    }
}

/// Optimize a single function in place. Returns the stats from this run.
pub fn optimize_function(func: &mut CompiledFunction) -> OptStats {
    let mut stats = OptStats::default();
    // Run passes until a fixpoint is reached. In practice 2-3 iterations.
    loop {
        let before = stats.total();
        // Resonance caching FIRST — turns `LoadConst(89); Resonance` into a
        // single constant, which the constant folder can then absorb into
        // surrounding arithmetic.
        unary_cache_pass(func, &mut stats);
        constant_fold_pass(func, &mut stats);
        dead_load_pass(func, &mut stats);
        double_unary_pass(func, &mut stats);
        if stats.total() == before {
            break;
        }
    }
    // Compact Nop holes once, after the peephole fixpoint.
    // This rewrites jump offsets so the VM traverses a dense bytecode
    // slice with no wasted iterations. Must run after all Nop-emitting
    // passes are done; running it inside the loop would only help if a
    // compacted bytecode exposed new constant-fold opportunities (it
    // doesn't — the folded constants are already in func.constants).
    stats.nops_compacted = compact_nops(func);
    stats
}

/// Fold `LoadConst a; LoadConst b; <op>` into `Nop; Nop; LoadConst c`.
/// The arithmetic and comparison ops are pure functions of the operand
/// pair, so this is safe regardless of surrounding control flow as
/// long as we don't disturb the jump-offset count (we don't — Nops
/// preserve indices).
fn constant_fold_pass(func: &mut CompiledFunction, stats: &mut OptStats) {
    let n = func.ops.len();
    if n < 3 {
        return;
    }
    for i in 0..(n - 2) {
        let (a, b, op) = match (&func.ops[i], &func.ops[i + 1], &func.ops[i + 2]) {
            (Op::LoadConst(a_idx), Op::LoadConst(b_idx), op) => {
                (*a_idx, *b_idx, op.clone())
            }
            _ => continue,
        };
        let a_val = match func.constants.get(a) {
            Some(c) => c.clone(),
            None => continue,
        };
        let b_val = match func.constants.get(b) {
            Some(c) => c.clone(),
            None => continue,
        };
        let folded = match fold_binary(&a_val, &b_val, &op) {
            Some(v) => v,
            None => continue,
        };
        let new_idx = func.constants.len();
        func.constants.push(folded);
        func.ops[i] = Op::Nop;
        func.ops[i + 1] = Op::Nop;
        func.ops[i + 2] = Op::LoadConst(new_idx);
        stats.constants_folded += 1;
    }
}

/// Remove `LoadConst N; Pop` pairs — the constant is loaded only to be
/// discarded. Both become Nops.
fn dead_load_pass(func: &mut CompiledFunction, stats: &mut OptStats) {
    let n = func.ops.len();
    if n < 2 {
        return;
    }
    for i in 0..(n - 1) {
        if matches!(func.ops[i], Op::LoadConst(_)) && matches!(func.ops[i + 1], Op::Pop) {
            func.ops[i] = Op::Nop;
            func.ops[i + 1] = Op::Nop;
            stats.dead_loads_removed += 1;
        }
    }
}

/// Cache pure-unary harmonic ops on constants:
///   LoadConst(N); Resonance   → LoadConst(precomputed_float)
///   LoadConst(N); Fold1       → LoadConst(snapped_int)
///   LoadConst(N); IsFibonacci → LoadConst(1 or 0)
///   LoadConst(N); Fibonacci   → LoadConst(fib(N))
///   LoadConst(N); HimScore    → LoadConst(precomputed_float)
///   LoadConst(N); Neg         → LoadConst(-N)
///   LoadConst(N); BitNot      → LoadConst(!N)
///   LoadConst(B); Not         → LoadConst(!B)
///
/// These are pure functions of a single constant — they cannot fail and
/// cannot observe runtime state. The omnicc Python compiler calls this
/// "resonance caching"; same idea, scoped to bytecode.
fn unary_cache_pass(func: &mut CompiledFunction, stats: &mut OptStats) {
    let n = func.ops.len();
    if n < 2 {
        return;
    }
    for i in 0..(n - 1) {
        let const_idx = match &func.ops[i] {
            Op::LoadConst(idx) => *idx,
            _ => continue,
        };
        let c = match func.constants.get(const_idx) {
            Some(c) => c.clone(),
            None => continue,
        };
        let result = match (&func.ops[i + 1], &c) {
            (Op::Resonance, Const::Int(n)) => {
                Some(Const::Float(crate::value::HInt::compute_resonance(*n)))
            }
            (Op::Resonance, Const::Float(f)) => Some(Const::Float(
                crate::value::HInt::compute_resonance(*f as i64),
            )),
            (Op::Fold1, Const::Int(n)) => Some(Const::Int(fold_to_fib_const(*n))),
            (Op::Fold1, Const::Float(f)) => Some(Const::Int(fold_to_fib_const(*f as i64))),
            (Op::IsFibonacci, Const::Int(n)) => {
                Some(Const::Int(if crate::value::is_fibonacci(*n) { 1 } else { 0 }))
            }
            (Op::Fibonacci, Const::Int(n)) => {
                Some(Const::Int(crate::value::fibonacci(*n)))
            }
            (Op::HimScore, Const::Int(n)) => {
                Some(Const::Float(crate::value::HInt::compute_him(*n)))
            }
            (Op::Neg, Const::Int(n)) => Some(Const::Int(-*n)),
            (Op::Neg, Const::Float(f)) => Some(Const::Float(-*f)),
            (Op::BitNot, Const::Int(n)) => Some(Const::Int(!*n)),
            (Op::Not, Const::Bool(b)) => Some(Const::Bool(!*b)),
            (Op::Not, Const::Int(n)) => Some(Const::Bool(*n == 0)),
            _ => None,
        };
        if let Some(folded) = result {
            let new_idx = func.constants.len();
            func.constants.push(folded);
            func.ops[i] = Op::Nop;
            func.ops[i + 1] = Op::LoadConst(new_idx);
            stats.unary_calls_cached += 1;
        }
    }
}

fn fold_to_fib_const(n: i64) -> i64 {
    // Substrate-routed. Was: 15-element local Fibonacci array + linear scan.
    crate::phi_pi_fib::fold_to_nearest_attractor(n)
}

/// Collapse `Not; Not` (and similar double-unary ops) to no-op.
fn double_unary_pass(func: &mut CompiledFunction, stats: &mut OptStats) {
    let n = func.ops.len();
    if n < 2 {
        return;
    }
    for i in 0..(n - 1) {
        match (&func.ops[i], &func.ops[i + 1]) {
            (Op::Not, Op::Not) => {
                func.ops[i] = Op::Nop;
                func.ops[i + 1] = Op::Nop;
                stats.double_nots_collapsed += 1;
            }
            (Op::Neg, Op::Neg) => {
                func.ops[i] = Op::Nop;
                func.ops[i + 1] = Op::Nop;
                stats.double_negs_collapsed += 1;
            }
            _ => {}
        }
    }
}

/// Apply a binary op to two constants. Returns None if the op isn't
/// foldable (e.g. it's a control-flow op, or the constants are
/// incompatible).
fn fold_binary(a: &Const, b: &Const, op: &Op) -> Option<Const> {
    // Promote to float if either is float.
    let any_float = matches!(a, Const::Float(_)) || matches!(b, Const::Float(_));
    if any_float {
        let af = const_to_float(a)?;
        let bf = const_to_float(b)?;
        return match op {
            Op::Add | Op::AddFloat => Some(Const::Float(af + bf)),
            Op::Sub | Op::SubFloat => Some(Const::Float(af - bf)),
            Op::Mul | Op::MulFloat => Some(Const::Float(af * bf)),
            Op::Div => {
                if bf == 0.0 {
                    None // can't fold div-by-zero (produces Singularity)
                } else {
                    Some(Const::Float(af / bf))
                }
            }
            Op::Eq => Some(Const::Bool(af == bf)),
            Op::Ne => Some(Const::Bool(af != bf)),
            Op::Lt => Some(Const::Bool(af < bf)),
            Op::Le => Some(Const::Bool(af <= bf)),
            Op::Gt => Some(Const::Bool(af > bf)),
            Op::Ge => Some(Const::Bool(af >= bf)),
            _ => None,
        };
    }
    let ai = const_to_int(a)?;
    let bi = const_to_int(b)?;
    match op {
        Op::Add | Op::AddInt => Some(Const::Int(ai.wrapping_add(bi))),
        Op::Sub | Op::SubInt => Some(Const::Int(ai.wrapping_sub(bi))),
        Op::Mul | Op::MulInt => Some(Const::Int(ai.wrapping_mul(bi))),
        Op::Div => {
            if bi == 0 {
                None
            } else {
                Some(Const::Int(ai / bi))
            }
        }
        Op::Mod => {
            if bi == 0 {
                None
            } else {
                Some(Const::Int(ai % bi))
            }
        }
        Op::Eq => Some(Const::Bool(ai == bi)),
        Op::Ne => Some(Const::Bool(ai != bi)),
        Op::Lt => Some(Const::Bool(ai < bi)),
        Op::Le => Some(Const::Bool(ai <= bi)),
        Op::Gt => Some(Const::Bool(ai > bi)),
        Op::Ge => Some(Const::Bool(ai >= bi)),
        Op::BitAnd => Some(Const::Int(ai & bi)),
        Op::BitOr => Some(Const::Int(ai | bi)),
        Op::BitXor => Some(Const::Int(ai ^ bi)),
        Op::Shl => Some(Const::Int(ai.wrapping_shl((bi & 63) as u32))),
        Op::Shr => Some(Const::Int(ai.wrapping_shr((bi & 63) as u32))),
        _ => None,
    }
}

fn const_to_int(c: &Const) -> Option<i64> {
    match c {
        Const::Int(n) => Some(*n),
        Const::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

fn const_to_float(c: &Const) -> Option<f64> {
    match c {
        Const::Int(n) => Some(*n as f64),
        Const::Float(f) => Some(*f),
        Const::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Remove all `Op::Nop` holes from a function's bytecode and rewrite
/// the relative jump offsets (Jump/JumpIfFalse/JumpIfTrue) so they
/// point to the same logical targets in the compacted slice.
///
/// Targets that pointed INTO a run of Nops are forwarded to the first
/// non-Nop after them (or to end-of-bytecode if the tail is all Nops).
///
/// Returns the number of Nops removed.
fn compact_nops(func: &mut CompiledFunction) -> usize {
    let n = func.ops.len();

    // Map old_index -> new_index. None = this op is a Nop (removed).
    let mut old_to_new: Vec<Option<usize>> = vec![None; n];
    let mut new_count = 0usize;
    for i in 0..n {
        if !matches!(func.ops[i], Op::Nop) {
            old_to_new[i] = Some(new_count);
            new_count += 1;
        }
    }
    let removed = n - new_count;
    if removed == 0 {
        return 0;
    }

    // For a jump target at old absolute index `abs`, find the new index.
    // If `abs` lands on a Nop, skip forward to the first non-Nop.
    // If everything remaining is Nops, return `new_count` (past end).
    let resolve = |abs: usize| -> usize {
        let mut t = abs;
        while t < n {
            if let Some(nidx) = old_to_new[t] {
                return nidx;
            }
            t += 1;
        }
        new_count
    };

    let mut new_ops: Vec<Op> = Vec::with_capacity(new_count);
    for i in 0..n {
        match &func.ops[i] {
            Op::Nop => {}
            Op::Jump(off) => {
                let abs_old = ((i as i32) + 1 + off) as usize;
                let abs_new = resolve(abs_old);
                let my_new = old_to_new[i].unwrap() as i32;
                new_ops.push(Op::Jump((abs_new as i32) - my_new - 1));
            }
            Op::JumpIfFalse(off) => {
                let abs_old = ((i as i32) + 1 + off) as usize;
                let abs_new = resolve(abs_old);
                let my_new = old_to_new[i].unwrap() as i32;
                new_ops.push(Op::JumpIfFalse((abs_new as i32) - my_new - 1));
            }
            Op::JumpIfTrue(off) => {
                let abs_old = ((i as i32) + 1 + off) as usize;
                let abs_new = resolve(abs_old);
                let my_new = old_to_new[i].unwrap() as i32;
                new_ops.push(Op::JumpIfTrue((abs_new as i32) - my_new - 1));
            }
            other => new_ops.push(other.clone()),
        }
    }

    func.ops = new_ops;
    removed
}

pub fn optimize_module(module: &mut Module) -> OptStats {
    let mut total = OptStats::default();
    accumulate(&mut total, optimize_function(&mut module.main));
    for (_, func) in module.functions.iter_mut() {
        accumulate(&mut total, optimize_function(func));
    }
    total
}

fn accumulate(total: &mut OptStats, s: OptStats) {
    total.constants_folded += s.constants_folded;
    total.dead_loads_removed += s.dead_loads_removed;
    total.double_nots_collapsed += s.double_nots_collapsed;
    total.double_negs_collapsed += s.double_negs_collapsed;
    total.unary_calls_cached += s.unary_calls_cached;
    total.nops_compacted += s.nops_compacted;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile_program;
    use crate::parser::Parser;

    fn compile_and_opt(src: &str) -> (Module, OptStats) {
        let mut parser = Parser::new(src);
        let stmts = parser.parse().unwrap();
        let mut module = compile_program(&stmts).unwrap();
        let stats = optimize_module(&mut module);
        (module, stats)
    }

    #[test]
    fn folds_simple_int_add() {
        let (_, stats) = compile_and_opt("h x = 2 + 3;");
        assert!(stats.constants_folded >= 1);
    }

    #[test]
    fn chained_arithmetic_folds_to_one_constant() {
        let (m, stats) = compile_and_opt("h x = 1 + 2 + 3 + 4;");
        assert!(stats.constants_folded >= 3, "expected >=3 folds, got {}", stats.constants_folded);
        // After folding, main should contain a single LoadConst(10) plus
        // StoreVar plus a return — at least one of the constants is 10.
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(10))),
        );
    }

    #[test]
    fn folds_bitwise() {
        let (m, stats) = compile_and_opt("h x = 255 & 15;");
        assert!(stats.constants_folded >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(15))),
        );
    }

    #[test]
    fn folds_shift() {
        let (m, stats) = compile_and_opt("h x = 1 << 8;");
        assert!(stats.constants_folded >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(256))),
        );
    }

    #[test]
    fn does_not_fold_div_by_zero() {
        // 10 / 0 must NOT be pre-folded — at runtime it produces a Singularity.
        let (_, stats) = compile_and_opt("h x = 10 / 0;");
        assert_eq!(stats.constants_folded, 0, "must preserve div-by-zero semantics");
    }

    #[test]
    fn folds_float_arithmetic() {
        let (m, stats) = compile_and_opt("h x = 1.5 + 2.5;");
        assert!(stats.constants_folded >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Float(f) if (f - 4.0).abs() < 1e-9)),
        );
    }

    #[test]
    fn folds_comparison() {
        let (m, stats) = compile_and_opt("h x = 10 < 20;");
        assert!(stats.constants_folded >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Bool(true))),
        );
    }

    // ----- Phase L: resonance / portal caching -----

    #[test]
    fn caches_resonance_of_constant() {
        // res(89) on a constant — 89 is Fibonacci so resonance = 1.0
        let (m, stats) = compile_and_opt("h x = res(89);");
        assert!(stats.unary_calls_cached >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Float(f) if (f - 1.0).abs() < 1e-9)));
    }

    #[test]
    fn caches_phi_fold_of_constant() {
        // phi.fold(90) → 89 (snap to nearest Fibonacci)
        let (m, stats) = compile_and_opt("h x = phi.fold(90);");
        assert!(stats.unary_calls_cached >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(89))));
    }

    #[test]
    fn caches_fibonacci_of_constant() {
        let (m, stats) = compile_and_opt("h x = fibonacci(10);");
        assert!(stats.unary_calls_cached >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(55))));
    }

    #[test]
    fn caches_is_fibonacci_of_constant() {
        let (m, stats) = compile_and_opt("h x = is_fibonacci(89);");
        assert!(stats.unary_calls_cached >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(1))));

        let (m2, stats2) = compile_and_opt("h x = is_fibonacci(90);");
        assert!(stats2.unary_calls_cached >= 1);
        assert!(m2
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(0))));
    }

    #[test]
    fn caches_unary_minus_of_constant() {
        let (m, stats) = compile_and_opt("h x = -42;");
        assert!(stats.unary_calls_cached >= 1 || stats.constants_folded >= 1);
        // -42 should appear as a constant after folding (the parser desugars
        // unary minus to `0 - 42`, which the constant folder reduces).
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(-42))));
    }

    #[test]
    fn caches_bitnot_of_constant() {
        let (m, stats) = compile_and_opt("h x = ~0;");
        assert!(stats.unary_calls_cached >= 1);
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Int(-1))));
    }

    #[test]
    fn chains_unary_cache_then_constant_fold() {
        // res(89) folds to 1.0, then `1.0 + 0.5` folds to 1.5.
        let (m, stats) = compile_and_opt("h x = res(89) + 0.5;");
        assert!(stats.unary_calls_cached >= 1);
        assert!(stats.constants_folded >= 1, "should fold the chained add");
        assert!(m
            .main
            .constants
            .iter()
            .any(|c| matches!(c, Const::Float(f) if (f - 1.5).abs() < 1e-9)));
    }

    // ── compact_nops tests ─────────────────────────────────────────────────

    #[test]
    fn compaction_removes_nops_from_constant_folded_code() {
        // `2 + 3` constant-folds: two LoadConsts become Nop;Nop, then
        // compaction removes them.
        let (m, stats) = compile_and_opt("h x = 2 + 3;");
        assert!(stats.nops_compacted >= 2, "expected >=2 Nops removed, got {}", stats.nops_compacted);
        // No Nops should remain after compaction.
        assert!(
            !m.main.ops.iter().any(|op| matches!(op, Op::Nop)),
            "Nop found after compaction"
        );
    }

    #[test]
    fn compaction_preserves_branch_semantics() {
        // A while-loop with runtime variable condition emits JumpIfFalse +
        // Jump. After constant-folding dead loads and compacting Nops, all
        // remaining jump offsets must stay in-bounds.
        let src = "h x = 10; while x > 0 { x = x - 1 }";
        let (m, _stats) = compile_and_opt(src);
        let n = m.main.ops.len();
        for (i, op) in m.main.ops.iter().enumerate() {
            let off = match op {
                Op::Jump(o) => Some(*o),
                Op::JumpIfFalse(o) => Some(*o),
                Op::JumpIfTrue(o) => Some(*o),
                _ => None,
            };
            if let Some(off) = off {
                let target = (i as i32 + 1 + off) as usize;
                assert!(
                    target <= n,
                    "jump at op{} target {} out of range (len={})", i, target, n
                );
            }
        }
    }

    #[test]
    fn compaction_preserves_loop_semantics() {
        // A while loop has a backward jump; compaction must not break it.
        let src = "h s = 0; h i = 0; while i < 5 { s += i; i += 1 }";
        let (m, _stats) = compile_and_opt(src);
        let n = m.main.ops.len();
        for (i, op) in m.main.ops.iter().enumerate() {
            let off = match op {
                Op::Jump(o) => Some(*o),
                Op::JumpIfFalse(o) => Some(*o),
                Op::JumpIfTrue(o) => Some(*o),
                _ => None,
            };
            if let Some(off) = off {
                let target = (i as i32 + 1 + off) as usize;
                assert!(
                    target <= n,
                    "loop jump at op{} target {} out of range (len={})", i, target, n
                );
            }
        }
        assert!(!m.main.ops.iter().any(|op| matches!(op, Op::Nop)),
            "Nop found after compaction");
    }

    fn make_func(ops: Vec<Op>, constants: Vec<Const>) -> CompiledFunction {
        use crate::ast::Pos;
        let n = ops.len();
        CompiledFunction {
            name: "test".to_string(),
            params: vec![],
            param_types: vec![],
            return_type: None,
            ops,
            constants,
            call_cache: vec![std::cell::Cell::new(0); n],
            op_positions: vec![Pos::unknown(); n],
            pragmas: vec![],
        }
    }

    #[test]
    fn compact_nops_direct_no_nops() {
        let mut func = make_func(vec![Op::LoadConst(0), Op::Return], vec![Const::Int(42)]);
        let removed = compact_nops(&mut func);
        assert_eq!(removed, 0);
        assert_eq!(func.ops.len(), 2);
        assert!(!func.ops.iter().any(|op| matches!(op, Op::Nop)));
    }

    #[test]
    fn compact_nops_direct_removes_nops() {
        // Nop; Nop; LoadConst(0); Nop; Return  →  LoadConst(0); Return
        let mut func = make_func(
            vec![Op::Nop, Op::Nop, Op::LoadConst(0), Op::Nop, Op::Return],
            vec![Const::Int(1)],
        );
        let removed = compact_nops(&mut func);
        assert_eq!(removed, 3);
        assert_eq!(func.ops.len(), 2);
        assert!(!func.ops.iter().any(|op| matches!(op, Op::Nop)));
    }
}
