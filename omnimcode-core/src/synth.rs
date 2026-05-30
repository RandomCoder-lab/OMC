//! Correct-by-construction OMC generator (Phase 4) — synthesis as a language service.
//!
//! A standard LM emits tokens and *hopes* they parse (the FibRec net managed ~10% valid OMC). This
//! generator emits only grammar-legal structure, so every program PARSES by construction, and it
//! tracks declared variables + guards division + bounds loops, so (almost) every program also RUNS.
//! Verified by the real parser/interpreter (see the in-module test: parse-rate 1.00).
//!
//! Ported from `experiments/transformerless_lm/grammar_gen.py`. Covers the executable core
//! (FunctionDef, VarDecl, Assignment, If, While, Return, arithmetic/comparison, call). Constructs
//! the generator does NOT yet emit (For/Try/Match/ClassDef/…) are the honest residual — closing
//! that to full coverage by auto-synthesizing emitters from the parser is Phase 4.2.

use std::collections::HashSet;

/// Arithmetic operators — mirror the AST's binary `Expression` variants (Add/Sub/Mul/Div/Mod).
/// (The in-core counterpart of `derive_grammar.py`; full auto-derivation from the parser is 4.2.)
const ARITH: &[&str] = &["+", "-", "*", "/", "%"];

struct Gen {
    rng: u64,
    counter: u32,
    vars: Vec<String>,
    protected: HashSet<String>,
}

impl Gen {
    fn new(seed: u64) -> Self {
        let s = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x2545_f491_4f6c_dd1d)
            | 1;
        Gen { rng: s, counter: 0, vars: Vec::new(), protected: HashSet::new() }
    }
    fn rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.rand() % n as u64) as usize
    }
    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}{}", prefix, self.counter)
    }

    /// An int-valued atom: a declared variable or a small literal.
    fn atom(&mut self) -> String {
        if !self.vars.is_empty() && self.pick(2) == 0 {
            let i = self.pick(self.vars.len());
            self.vars[i].clone()
        } else {
            format!("{}", self.pick(10))
        }
    }

    /// A run-safe int expression: division/modulo always by a nonzero literal.
    fn int_expr(&mut self, depth: u32) -> String {
        if depth == 0 || self.pick(2) == 0 {
            return self.atom();
        }
        let op = ARITH[self.pick(ARITH.len())];
        let lhs = self.int_expr(depth - 1);
        if op == "/" || op == "%" {
            format!("({} {} {})", lhs, op, 1 + self.pick(9))
        } else {
            let rhs = self.int_expr(depth - 1);
            format!("({} {} {})", lhs, op, rhs)
        }
    }

    fn cond(&mut self) -> String {
        format!("{} < {}", self.atom(), self.pick(10))
    }

    /// A block body of plain assignments to ALREADY-declared, non-protected vars (no new decls
    /// inside blocks → no scoping hazard; no protected-counter writes → loops terminate).
    fn block_assigns(&mut self, out: &mut Vec<String>, indent: &str, k: usize) {
        let assignable: Vec<String> =
            self.vars.iter().filter(|v| !self.protected.contains(*v)).cloned().collect();
        if assignable.is_empty() {
            out.push(format!("{}a = a;", indent)); // a/b params are always in scope
            return;
        }
        for _ in 0..k {
            let v = assignable[self.pick(assignable.len())].clone();
            let op = ["=", "+=", "-="][self.pick(3)];
            out.push(format!("{}{} {} {};", indent, v, op, self.int_expr(1)));
        }
    }

    /// One top-level statement (function-body scope → declarations allowed here). Control bodies
    /// are flat assignment blocks (no nested declarations) so nothing ever goes out of scope.
    fn top_stmt(&mut self, out: &mut Vec<String>, indent: &str) {
        let inner = format!("{}    ", indent);
        match self.pick(6) {
            // declare a fresh int
            0 => {
                let v = self.fresh("v");
                let e = self.int_expr(2);
                out.push(format!("{}h {} = {};", indent, v, e));
                self.vars.push(v);
            }
            // assign to an existing var
            1 => self.block_assigns(out, indent, 1),
            // bounded while with a protected counter
            2 => {
                let c = self.fresh("c");
                let bound = 1 + self.pick(4);
                out.push(format!("{}h {} = 0;", indent, c));
                out.push(format!("{}while {} < {} {{", indent, c, bound));
                self.protected.insert(c.clone());
                let k = 1 + self.pick(2);
                self.block_assigns(out, &inner, k);
                out.push(format!("{}    {} += 1;", indent, c));
                out.push(format!("{}}}", indent));
                self.protected.remove(&c);
                self.vars.push(c);
            }
            // bounded for over a range (loop var unused in body → always safe + terminating)
            3 => {
                let i = self.fresh("i");
                let bound = 1 + self.pick(4);
                out.push(format!("{}for {} in range(0, {}) {{", indent, i, bound));
                let k = 1 + self.pick(2);
                self.block_assigns(out, &inner, k);
                out.push(format!("{}}}", indent));
            }
            // if / else with flat bodies
            4 => {
                let cnd = self.cond();
                out.push(format!("{}if {} {{", indent, cnd));
                let k1 = 1 + self.pick(2);
                self.block_assigns(out, &inner, k1);
                out.push(format!("{}}} else {{", indent));
                let k2 = 1 + self.pick(2);
                self.block_assigns(out, &inner, k2);
                out.push(format!("{}}}", indent));
            }
            // print an expression
            _ => {
                let e = self.int_expr(1);
                out.push(format!("{}print({});", indent, e));
            }
        }
    }

    fn program(&mut self) -> String {
        self.vars = vec!["a".to_string(), "b".to_string()];
        self.protected.clear();
        self.counter = 0;
        let mut out = vec!["fn g(a, b) {".to_string()];
        // a couple of guaranteed declarations first so later statements have vars to use
        for _ in 0..2 {
            let v = self.fresh("v");
            let e = self.int_expr(2);
            out.push(format!("    h {} = {};", v, e));
            self.vars.push(v);
        }
        for _ in 0..(2 + self.pick(4)) {
            self.top_stmt(&mut out, "    ");
        }
        let ret = self.int_expr(2);
        out.push(format!("    return {};", ret));
        out.push("}".to_string());
        out.push("g(3, 4);".to_string());
        out.join("\n")
    }
}

/// Generate a valid-by-construction OMC program for `seed` (deterministic).
pub fn gen_program(seed: u64) -> String {
    Gen::new(seed).program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_programs_are_valid_by_construction() {
        // The correct-by-construction guarantee, checked by the REAL parser + interpreter.
        let n = 300;
        let mut parsed = 0;
        let mut ran = 0;
        for seed in 0..n {
            let prog = gen_program(seed);
            let mut p = crate::parser::Parser::new(&prog);
            match p.parse() {
                Ok(stmts) => {
                    parsed += 1;
                    let mut interp = crate::interpreter::Interpreter::new();
                    interp.register_user_functions(&stmts);
                    if interp.execute(stmts).is_ok() {
                        ran += 1;
                    }
                }
                Err(_) => {}
            }
        }
        let (pr, rr) = (parsed as f64 / n as f64, ran as f64 / n as f64);
        println!("[synth] gen_omc over {n} seeds: parse-rate={pr:.3} run-rate={rr:.3}");
        assert_eq!(parsed, n, "correct-by-construction VIOLATED: only {parsed}/{n} parsed");
        assert!(rr > 0.95, "run-rate unexpectedly low: {rr}");
    }
}
