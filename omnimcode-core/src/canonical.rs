//! AST canonicalization for semantic-equivalence detection.
//!
//! `canonicalize(source)` produces a string such that two semantically
//! equivalent OMC programs map to the same output, regardless of:
//!   - whitespace / indentation / blank lines
//!   - comments
//!   - choice of local variable names (alpha-equivalence)
//!   - choice of function parameter names
//!   - for-loop iterator names
//!   - try/catch error-variable names
//!   - lambda parameter names
//!
//! Top-level function names, class names, dict keys, string literals,
//! and global variables are PRESERVED — those are observable API.
//!
//! Pipeline: parse → walk AST renaming locals → re-emit via formatter.
//! The formatter already strips whitespace and comments and inserts
//! canonical operator parens, so combining the two passes gives us
//! the full canonical form.
//!
//! Use cases:
//!   omc_code_canonical(code)        → canonical string
//!   omc_code_equivalent(a, b)       → 1 if canonicals match
//!   omc_code_hash(omc_code_canonical(x))  → semantic hash (LLM-stable id)

use std::collections::HashMap;

use crate::ast::{Expression, ForIterable, Pattern, Statement};
use crate::formatter::format_program;
use crate::parser::Parser;

/// Parse + canonicalize + re-emit. Returns the canonical source.
pub fn canonicalize(source: &str) -> Result<String, String> {
    let mut p = Parser::new(source);
    let stmts = p.parse().map_err(|e| format!("parse error: {}", e))?;
    let renamed = canonicalize_program(&stmts);
    Ok(format_program(&renamed))
}

/// True when two sources canonicalize identically. Both sources must
/// parse — a parse error in either propagates as `false` (rather than
/// claiming equivalence we can't verify).
pub fn equivalent(a: &str, b: &str) -> bool {
    match (canonicalize(a), canonicalize(b)) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// Walk the program, rewriting each function body's locals to a
/// canonical naming scheme. Top-level statements outside any function
/// are NOT renamed — they're observable program state.
fn canonicalize_program(stmts: &[Statement]) -> Vec<Statement> {
    stmts.iter().map(canonicalize_top_stmt).collect()
}

/// Top-level statement rewrite. Function and class definitions get
/// their bodies canonicalized; everything else passes through with
/// only expression-level rewriting (renames inside lambdas).
fn canonicalize_top_stmt(stmt: &Statement) -> Statement {
    match stmt {
        Statement::FunctionDef {
            name, params, param_types, body, return_type, pragmas,
        } => {
            let (new_params, new_body) = rename_function(params, body);
            Statement::FunctionDef {
                name: name.clone(),
                params: new_params,
                param_types: param_types.clone(),
                body: new_body,
                return_type: return_type.clone(),
                pragmas: pragmas.clone(),
            }
        }
        Statement::ClassDef { name, parent, fields, methods } => {
            // Each method is itself a FunctionDef — canonicalize independently.
            let new_methods: Vec<Statement> = methods.iter()
                .map(canonicalize_top_stmt)
                .collect();
            Statement::ClassDef {
                name: name.clone(),
                parent: parent.clone(),
                fields: fields.clone(),
                methods: new_methods,
            }
        }
        _ => {
            // Top-level non-function: only rewrite expression-internal
            // lambdas. Locals at top level stay as written.
            let mut scope = Scope::empty();
            rename_stmt(stmt, &mut scope)
        }
    }
}

/// Local rename scope: name → canonical name. Sibling scopes do not
/// share; nested scopes inherit + extend.
struct Scope {
    /// Parent scope's mappings (for inheritance), copied at construction.
    /// Cheaper than a linked list for the depths OMC programs use.
    map: HashMap<String, String>,
    /// Next ID to assign for a new local. Reset per function/lambda.
    next: usize,
}

impl Scope {
    fn empty() -> Self {
        Self { map: HashMap::new(), next: 0 }
    }

    fn fresh() -> Self {
        Self { map: HashMap::new(), next: 0 }
    }

    fn child(&self) -> Self {
        // Inherit parent bindings so a nested block can still reference
        // outer locals. New locals shadow.
        Self { map: self.map.clone(), next: self.next }
    }

    /// Introduce a new local, returning its canonical name.
    fn introduce(&mut self, original: &str) -> String {
        let canon = format!("__v{}", self.next);
        self.next += 1;
        self.map.insert(original.to_string(), canon.clone());
        canon
    }

    /// Resolve a name. Returns the canonical form when known, otherwise
    /// the original (preserves globals + builtin calls + top-level fns).
    fn resolve(&self, name: &str) -> String {
        self.map.get(name).cloned().unwrap_or_else(|| name.to_string())
    }
}

fn rename_function(params: &[String], body: &[Statement]) -> (Vec<String>, Vec<Statement>) {
    let mut scope = Scope::fresh();
    let mut new_params = Vec::with_capacity(params.len());
    for p in params {
        new_params.push(scope.introduce(p));
    }
    let new_body: Vec<Statement> = body.iter()
        .map(|s| rename_stmt(s, &mut scope))
        .collect();
    (new_params, new_body)
}

fn rename_stmt(stmt: &Statement, scope: &mut Scope) -> Statement {
    match stmt {
        Statement::Print(e) => Statement::Print(rename_expr(e, scope)),
        Statement::Expression(e) => Statement::Expression(rename_expr(e, scope)),
        Statement::VarDecl { name, value, is_harmonic } => {
            // Evaluate value with the OLD scope, then introduce the new name.
            let new_value = rename_expr(value, scope);
            let new_name = scope.introduce(name);
            Statement::VarDecl {
                name: new_name,
                value: new_value,
                is_harmonic: *is_harmonic,
            }
        }
        Statement::Parameter { name, value } => {
            let new_value = rename_expr(value, scope);
            let new_name = scope.introduce(name);
            Statement::Parameter { name: new_name, value: new_value }
        }
        Statement::Assignment { name, value } => {
            let new_value = rename_expr(value, scope);
            let new_name = scope.resolve(name);
            Statement::Assignment { name: new_name, value: new_value }
        }
        Statement::IndexAssignment { name, index, value } => {
            let new_index = rename_expr(index, scope);
            let new_value = rename_expr(value, scope);
            let new_name = scope.resolve(name);
            Statement::IndexAssignment {
                name: new_name,
                index: new_index,
                value: new_value,
            }
        }
        Statement::ChainedIndexAssignment { name, first_index, second_index, value } => {
            Statement::ChainedIndexAssignment {
                name: scope.resolve(name),
                first_index: rename_expr(first_index, scope),
                second_index: rename_expr(second_index, scope),
                value: rename_expr(value, scope),
            }
        }
        Statement::If { condition, then_body, elif_parts, else_body } => {
            let new_cond = rename_expr(condition, scope);
            // Each branch gets its own scope so a var declared in one
            // branch doesn't leak into the next. Use child() so outer
            // names are still visible.
            let new_then = {
                let mut s = scope.child();
                then_body.iter().map(|st| rename_stmt(st, &mut s)).collect()
            };
            let new_elifs: Vec<(Expression, Vec<Statement>)> = elif_parts.iter()
                .map(|(c, b)| {
                    let nc = rename_expr(c, scope);
                    let mut s = scope.child();
                    let nb: Vec<Statement> = b.iter().map(|st| rename_stmt(st, &mut s)).collect();
                    (nc, nb)
                }).collect();
            let new_else = else_body.as_ref().map(|b| {
                let mut s = scope.child();
                b.iter().map(|st| rename_stmt(st, &mut s)).collect()
            });
            Statement::If {
                condition: new_cond,
                then_body: new_then,
                elif_parts: new_elifs,
                else_body: new_else,
            }
        }
        Statement::While { condition, body } => {
            let new_cond = rename_expr(condition, scope);
            let mut s = scope.child();
            let new_body: Vec<Statement> = body.iter()
                .map(|st| rename_stmt(st, &mut s))
                .collect();
            Statement::While { condition: new_cond, body: new_body }
        }
        Statement::For { var, iterable, body } => {
            // For-loop variable is local to the loop body.
            let new_iter = rename_for_iterable(iterable, scope);
            let mut s = scope.child();
            let new_var = s.introduce(var);
            let new_body: Vec<Statement> = body.iter()
                .map(|st| rename_stmt(st, &mut s))
                .collect();
            Statement::For {
                var: new_var,
                iterable: new_iter,
                body: new_body,
            }
        }
        Statement::FunctionDef { name, params, param_types, body, return_type, pragmas } => {
            // Nested function defs (rare but legal) get a FRESH scope —
            // they don't inherit the enclosing function's locals.
            let (new_params, new_body) = rename_function(params, body);
            Statement::FunctionDef {
                name: name.clone(),
                params: new_params,
                param_types: param_types.clone(),
                body: new_body,
                return_type: return_type.clone(),
                pragmas: pragmas.clone(),
            }
        }
        Statement::Return(e) => Statement::Return(e.as_ref().map(|x| rename_expr(x, scope))),
        Statement::Break => Statement::Break,
        Statement::Continue => Statement::Continue,
        Statement::Import { .. } => stmt.clone(),
        Statement::Try { body, err_var, handler, finally } => {
            let mut try_scope = scope.child();
            let new_body: Vec<Statement> = body.iter()
                .map(|st| rename_stmt(st, &mut try_scope))
                .collect();
            let mut catch_scope = scope.child();
            let new_err = catch_scope.introduce(err_var);
            let new_handler: Vec<Statement> = handler.iter()
                .map(|st| rename_stmt(st, &mut catch_scope))
                .collect();
            let new_finally = finally.as_ref().map(|f| {
                let mut s = scope.child();
                f.iter().map(|st| rename_stmt(st, &mut s)).collect()
            });
            Statement::Try {
                body: new_body,
                err_var: new_err,
                handler: new_handler,
                finally: new_finally,
            }
        }
        Statement::Throw(e) => Statement::Throw(rename_expr(e, scope)),
        Statement::Yield(e) => Statement::Yield(rename_expr(e, scope)),
        Statement::ClassDef { name, parent, fields, methods } => {
            // Class defs nested in functions: canonicalize each method.
            let new_methods: Vec<Statement> = methods.iter()
                .map(canonicalize_top_stmt)
                .collect();
            Statement::ClassDef {
                name: name.clone(),
                parent: parent.clone(),
                fields: fields.clone(),
                methods: new_methods,
            }
        }
        Statement::Match { scrutinee, arms } => {
            let new_scrutinee = rename_expr(scrutinee, scope);
            let new_arms: Vec<crate::ast::MatchArm> = arms.iter().map(|arm| {
                let mut arm_scope = scope.child();
                let new_pattern = rename_pattern(&arm.pattern, &mut arm_scope);
                let new_body: Vec<Statement> = arm.body.iter()
                    .map(|st| rename_stmt(st, &mut arm_scope))
                    .collect();
                crate::ast::MatchArm { pattern: new_pattern, body: new_body }
            }).collect();
            Statement::Match { scrutinee: new_scrutinee, arms: new_arms }
        }
    }
}

fn rename_pattern(pat: &Pattern, scope: &mut Scope) -> Pattern {
    match pat {
        // Bind introduces a new local name in the arm body.
        Pattern::Bind(name) => Pattern::Bind(scope.introduce(name)),
        Pattern::Or(alts) => Pattern::Or(alts.iter().map(|p| rename_pattern(p, scope)).collect()),
        // Everything else has no variable-binding semantics.
        _ => pat.clone(),
    }
}

fn rename_for_iterable(it: &ForIterable, scope: &Scope) -> ForIterable {
    match it {
        ForIterable::Range { start, end } => ForIterable::Range {
            start: rename_expr(start, scope),
            end: rename_expr(end, scope),
        },
        ForIterable::Expr(e) => ForIterable::Expr(rename_expr(e, scope)),
    }
}

fn rename_expr(expr: &Expression, scope: &Scope) -> Expression {
    match expr {
        Expression::Number(_)
        | Expression::Float(_)
        | Expression::String(_)
        | Expression::Boolean(_) => expr.clone(),

        Expression::Variable(name) => Expression::Variable(scope.resolve(name)),
        Expression::Index { name, index } => Expression::Index {
            name: scope.resolve(name),
            index: Box::new(rename_expr(index, scope)),
        },
        Expression::ChainedIndex { object, index } => Expression::ChainedIndex {
            object: Box::new(rename_expr(object, scope)),
            index: Box::new(rename_expr(index, scope)),
        },

        Expression::Array(items) => Expression::Array(
            items.iter().map(|e| rename_expr(e, scope)).collect(),
        ),
        Expression::Dict(pairs) => Expression::Dict(
            pairs.iter()
                .map(|(k, v)| (rename_expr(k, scope), rename_expr(v, scope)))
                .collect(),
        ),

        Expression::Add(a, b) => Expression::add(rename_expr(a, scope), rename_expr(b, scope)),
        Expression::Sub(a, b) => Expression::sub(rename_expr(a, scope), rename_expr(b, scope)),
        Expression::Mul(a, b) => Expression::mul(rename_expr(a, scope), rename_expr(b, scope)),
        Expression::Div(a, b) => Expression::div(rename_expr(a, scope), rename_expr(b, scope)),
        Expression::Mod(a, b) => Expression::Mod(
            Box::new(rename_expr(a, scope)),
            Box::new(rename_expr(b, scope)),
        ),
        Expression::Eq(a, b) => Expression::Eq(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Ne(a, b) => Expression::Ne(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Lt(a, b) => Expression::Lt(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Le(a, b) => Expression::Le(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Gt(a, b) => Expression::Gt(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Ge(a, b) => Expression::Ge(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::And(a, b) => Expression::And(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Or(a, b) => Expression::Or(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Not(e) => Expression::Not(Box::new(rename_expr(e, scope))),
        Expression::BitAnd(a, b) => Expression::BitAnd(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::BitOr(a, b) => Expression::BitOr(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::BitXor(a, b) => Expression::BitXor(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::BitNot(e) => Expression::BitNot(Box::new(rename_expr(e, scope))),
        Expression::Shl(a, b) => Expression::Shl(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),
        Expression::Shr(a, b) => Expression::Shr(Box::new(rename_expr(a, scope)), Box::new(rename_expr(b, scope))),

        Expression::Call { name, args, pos } => Expression::Call {
            // Function names at call sites: pass through. They could be
            // either top-level fn names (preserve) or higher-order
            // closure-variable lookups via the resolver — try to resolve
            // and fall back to the original name when nothing matches.
            name: scope.resolve(name),
            args: args.iter().map(|a| rename_expr(a, scope)).collect(),
            pos: *pos,
        },
        Expression::Resonance(e) => Expression::Resonance(Box::new(rename_expr(e, scope))),
        Expression::Fold(e) => Expression::Fold(Box::new(rename_expr(e, scope))),
        Expression::Safe(e) => Expression::Safe(Box::new(rename_expr(e, scope))),

        Expression::Lambda { params, body } => {
            // Lambdas open a fresh scope. Captures via outer names still
            // work because Lambda values capture by value at runtime; for
            // canonicalization we just rename params + body internally.
            let mut lambda_scope = scope.child();
            let new_params: Vec<String> = params.iter()
                .map(|p| lambda_scope.introduce(p))
                .collect();
            let new_body: Vec<Statement> = body.iter()
                .map(|s| rename_stmt(s, &mut lambda_scope))
                .collect();
            Expression::Lambda { params: new_params, body: new_body }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_invariant() {
        let a = "fn add(x, y) { return x + y; }";
        let b = "fn   add(x,y){return x+y;}";
        assert_eq!(canonicalize(a).unwrap(), canonicalize(b).unwrap());
    }

    #[test]
    fn comment_invariant() {
        let a = "fn f(x) { return x; }";
        let b = "fn f(x) {\n  # the doc\n  return x;\n}";
        assert_eq!(canonicalize(a).unwrap(), canonicalize(b).unwrap());
    }

    #[test]
    fn alpha_equivalence() {
        let a = "fn add(x, y) { return x + y; }";
        let b = "fn add(a, b) { return a + b; }";
        assert_eq!(canonicalize(a).unwrap(), canonicalize(b).unwrap());
    }

    #[test]
    fn top_level_fn_names_preserved() {
        let a = "fn add(x, y) { return x + y; }";
        let b = "fn sub(x, y) { return x + y; }";
        assert_ne!(canonicalize(a).unwrap(), canonicalize(b).unwrap());
    }

    #[test]
    fn local_var_alpha_equivalence() {
        let a = "fn f(x) { h tmp = x * 2; return tmp; }";
        let b = "fn f(x) { h other = x * 2; return other; }";
        assert_eq!(canonicalize(a).unwrap(), canonicalize(b).unwrap());
    }

    #[test]
    fn structurally_different_not_equivalent() {
        let a = "fn f(x) { return x; }";
        let b = "fn f(x) { return x + 1; }";
        assert_ne!(canonicalize(a).unwrap(), canonicalize(b).unwrap());
    }

    #[test]
    fn equivalent_returns_true_for_equivalents() {
        assert!(equivalent(
            "fn f(x) { return x * 2; }",
            "fn f(a) { return a * 2; }",
        ));
    }

    #[test]
    fn equivalent_returns_false_for_different() {
        assert!(!equivalent(
            "fn f(x) { return x; }",
            "fn f(x) { return x + 1; }",
        ));
    }
}
