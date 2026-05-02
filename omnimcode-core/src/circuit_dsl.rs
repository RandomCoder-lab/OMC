// src/circuit_dsl.rs - Circuit DSL and transpiler
// Handles circuit notation parsing and macro expansion

use crate::circuits::{Circuit, Gate, GateId};
use std::collections::HashMap;

/// Circuit expression AST for DSL
#[derive(Clone, Debug)]
pub enum CircuitExpr {
    /// Direct gate: Input reference or Constant
    Atom(AtomExpr),
    /// Binary operation: AND, OR, XOR, etc.
    BinOp {
        op: CircuitOp,
        left: Box<CircuitExpr>,
        right: Box<CircuitExpr>,
    },
    /// Unary operation: NOT
    UnaryOp {
        op: UnaryOp,
        arg: Box<CircuitExpr>,
    },
    /// Conditional: IF-THEN-ELSE
    IfExpr {
        condition: Box<CircuitExpr>,
        then_expr: Box<CircuitExpr>,
        else_expr: Box<CircuitExpr>,
    },
    /// Macro call: @name(args)
    MacroCall {
        name: String,
        args: Vec<CircuitExpr>,
    },
    /// Variable reference
    Var(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum CircuitOp {
    And,     // &
    Or,      // | (XOR semantics)
    Xor,     // ^ (explicit XOR)
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOp {
    Not,     // !
}

#[derive(Clone, Debug)]
pub enum AtomExpr {
    Input(usize),           // i0, i1, ...
    Constant(bool),         // true, false
    Const(i64),            // Converts to bool
}

/// Macro definition
#[derive(Clone, Debug)]
pub struct MacroDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: CircuitExpr,
}

/// Linting issue
#[derive(Clone, Debug)]
pub struct LintIssue {
    pub level: LintLevel,
    pub code: String,
    pub message: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LintLevel {
    Warning,
    Error,
}

/// Circuit DSL transpiler
pub struct CircuitTranspiler {
    macros: HashMap<String, MacroDef>,
    var_map: HashMap<String, GateId>,
    num_inputs: usize,
    issues: Vec<LintIssue>,
}

impl CircuitTranspiler {
    pub fn new(num_inputs: usize) -> Self {
        Self {
            macros: HashMap::new(),
            var_map: HashMap::new(),
            num_inputs,
            issues: Vec::new(),
        }
    }

    /// Register a macro definition
    pub fn define_macro(&mut self, macro_def: MacroDef) -> Result<(), String> {
        if self.macros.contains_key(&macro_def.name) {
            return Err(format!("Macro '{}' already defined", macro_def.name));
        }
        self.macros.insert(macro_def.name.clone(), macro_def);
        Ok(())
    }

    /// Transpile circuit expression to native Circuit
    pub fn transpile(&mut self, expr: CircuitExpr) -> Result<Circuit, String> {
        let mut circuit = Circuit::new(self.num_inputs);
        self.var_map.clear();

        // Pre-populate input gates
        for i in 0..self.num_inputs {
            let gate_id = circuit.add_gate(Gate::Input { index: i });
            self.var_map.insert(format!("i{}", i), gate_id);
        }

        // Transpile expression
        let output_id = self.transpile_expr(&mut circuit, expr)?;
        circuit.output = output_id;
        
        // Validate result
        circuit.validate()?;
        
        Ok(circuit)
    }

    /// Transpile a circuit expression
    fn transpile_expr(
        &mut self,
        circuit: &mut Circuit,
        expr: CircuitExpr,
    ) -> Result<GateId, String> {
        match expr {
            CircuitExpr::Atom(atom) => self.transpile_atom(circuit, atom),
            CircuitExpr::BinOp { op, left, right } => {
                let left_id = self.transpile_expr(circuit, *left)?;
                let right_id = self.transpile_expr(circuit, *right)?;
                
                let gate = match op {
                    CircuitOp::And => Gate::XAnd {
                        inputs: vec![left_id, right_id],
                    },
                    CircuitOp::Or | CircuitOp::Xor => Gate::XOr {
                        inputs: vec![left_id, right_id],
                    },
                };
                
                Ok(circuit.add_gate(gate))
            }
            CircuitExpr::UnaryOp { op, arg } => {
                let arg_id = self.transpile_expr(circuit, *arg)?;
                
                match op {
                    UnaryOp::Not => {
                        Ok(circuit.add_gate(Gate::Not { input: arg_id }))
                    }
                }
            }
            CircuitExpr::IfExpr {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond_id = self.transpile_expr(circuit, *condition)?;
                let then_id = self.transpile_expr(circuit, *then_expr)?;
                let else_id = self.transpile_expr(circuit, *else_expr)?;
                
                Ok(circuit.add_gate(Gate::XIf {
                    condition: cond_id,
                    then_gate: then_id,
                    else_gate: else_id,
                }))
            }
            CircuitExpr::MacroCall { name, args } => {
                self.expand_macro(circuit, &name, args)
            }
            CircuitExpr::Var(name) => {
                self.var_map
                    .get(&name)
                    .copied()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }
        }
    }

    fn transpile_atom(
        &mut self,
        circuit: &mut Circuit,
        atom: AtomExpr,
    ) -> Result<GateId, String> {
        match atom {
            AtomExpr::Input(idx) => {
                if idx >= self.num_inputs {
                    return Err(format!(
                        "Input index {} out of range (max: {})",
                        idx,
                        self.num_inputs - 1
                    ));
                }
                self.var_map
                    .get(&format!("i{}", idx))
                    .copied()
                    .ok_or_else(|| "Input not initialized".into())
            }
            AtomExpr::Constant(val) => {
                Ok(circuit.add_gate(Gate::Constant { value: val }))
            }
            AtomExpr::Const(val) => {
                let bool_val = val != 0;
                Ok(circuit.add_gate(Gate::Constant { value: bool_val }))
            }
        }
    }

    fn expand_macro(
        &mut self,
        circuit: &mut Circuit,
        name: &str,
        args: Vec<CircuitExpr>,
    ) -> Result<GateId, String> {
        let macro_def = self
            .macros
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Undefined macro: {}", name))?;

        if args.len() != macro_def.params.len() {
            return Err(format!(
                "Macro '{}' expects {} arguments, got {}",
                name,
                macro_def.params.len(),
                args.len()
            ));
        }

        // Save current var map
        let saved_vars = self.var_map.clone();

        // Bind macro arguments to parameters
        for (param, arg_expr) in macro_def.params.iter().zip(args.into_iter()) {
            let arg_id = self.transpile_expr(circuit, arg_expr)?;
            self.var_map.insert(param.clone(), arg_id);
        }

        // Expand macro body
        let result = self.transpile_expr(circuit, macro_def.body)?;

        // Restore var map
        self.var_map = saved_vars;

        Ok(result)
    }

    /// Lint a circuit expression
    pub fn lint(&mut self, expr: &CircuitExpr) -> Vec<LintIssue> {
        self.issues.clear();
        self.lint_expr(expr);
        self.issues.clone()
    }

    fn lint_expr(&mut self, expr: &CircuitExpr) {
        match expr {
            CircuitExpr::Atom(_) => {
                // Atoms are always OK
            }
            CircuitExpr::BinOp { left, right, op } => {
                // Lint both sides
                self.lint_expr(left);
                self.lint_expr(right);

                // Warn about redundant operations
                match op {
                    CircuitOp::And => {
                        if self.is_same_expr(left, right) {
                            self.issues.push(LintIssue {
                                level: LintLevel::Warning,
                                code: "W001".to_string(),
                                message: "Redundant AND: a & a is always a".to_string(),
                                line: 0,
                                column: 0,
                            });
                        }
                    }
                    CircuitOp::Or | CircuitOp::Xor => {
                        if self.is_same_expr(left, right) {
                            self.issues.push(LintIssue {
                                level: LintLevel::Warning,
                                code: "W002".to_string(),
                                message: "Redundant XOR: a | a is always 0".to_string(),
                                line: 0,
                                column: 0,
                            });
                        }
                    }
                }
            }
            CircuitExpr::UnaryOp { arg, .. } => {
                self.lint_expr(arg);
            }
            CircuitExpr::IfExpr {
                condition,
                then_expr,
                else_expr,
            } => {
                self.lint_expr(condition);
                self.lint_expr(then_expr);
                self.lint_expr(else_expr);
            }
            CircuitExpr::MacroCall { args, .. } => {
                for arg in args {
                    self.lint_expr(arg);
                }
            }
            CircuitExpr::Var(_) => {
                // Variable lint happens during expansion
            }
        }
    }

    fn is_same_expr(&self, a: &CircuitExpr, b: &CircuitExpr) -> bool {
        format!("{:?}", a) == format!("{:?}", b)
    }

    /// Get all linting issues
    pub fn get_issues(&self) -> &[LintIssue] {
        &self.issues
    }
}

/// Parser for circuit DSL
pub struct CircuitParser {
    tokens: Vec<String>,
    pos: usize,
}

impl CircuitParser {
    pub fn new(input: &str) -> Self {
        let tokens = Self::tokenize(input);
        Self { tokens, pos: 0 }
    }

    fn tokenize(input: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in input.chars() {
            match ch {
                '&' | '|' | '!' | '(' | ')' | ',' | '^' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                    tokens.push(ch.to_string());
                }
                ' ' | '\t' | '\n' | '\r' => {
                    if !current.is_empty() {
                        tokens.push(current.clone());
                        current.clear();
                    }
                }
                _ => current.push(ch),
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Parse circuit expression
    pub fn parse(&mut self) -> Result<CircuitExpr, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<CircuitExpr, String> {
        let mut left = self.parse_and()?;

        while self.current() == Some("|") {
            self.consume();
            let right = self.parse_and()?;
            left = CircuitExpr::BinOp {
                op: CircuitOp::Or,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> Result<CircuitExpr, String> {
        let mut left = self.parse_not()?;

        while self.current() == Some("&") {
            self.consume();
            let right = self.parse_not()?;
            left = CircuitExpr::BinOp {
                op: CircuitOp::And,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_not(&mut self) -> Result<CircuitExpr, String> {
        if self.current() == Some("!") {
            self.consume();
            let arg = self.parse_not()?;
            Ok(CircuitExpr::UnaryOp {
                op: UnaryOp::Not,
                arg: Box::new(arg),
            })
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<CircuitExpr, String> {
        match self.current() {
            Some("(") => {
                self.consume();
                let expr = self.parse_or()?;
                if self.current() != Some(")") {
                    return Err("Expected ')'".into());
                }
                self.consume();
                Ok(expr)
            }
            Some(tok) if tok.starts_with("i") => {
                let idx = tok[1..]
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid input reference: {}", tok))?;
                self.consume();
                Ok(CircuitExpr::Atom(AtomExpr::Input(idx)))
            }
            Some("true") => {
                self.consume();
                Ok(CircuitExpr::Atom(AtomExpr::Constant(true)))
            }
            Some("false") => {
                self.consume();
                Ok(CircuitExpr::Atom(AtomExpr::Constant(false)))
            }
            Some(tok) if tok.parse::<i64>().is_ok() => {
                let val = tok.parse::<i64>().unwrap();
                self.consume();
                Ok(CircuitExpr::Atom(AtomExpr::Const(val)))
            }
            Some(tok) => {
                let var = tok.to_string();
                self.consume();
                Ok(CircuitExpr::Var(var))
            }
            None => Err("Unexpected end of input".into()),
        }
    }

    fn current(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn consume(&mut self) {
        self.pos += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and() {
        let mut parser = CircuitParser::new("i0 & i1");
        let expr = parser.parse().unwrap();
        match expr {
            CircuitExpr::BinOp { op: CircuitOp::And, .. } => {}
            _ => panic!("Expected AND operation"),
        }
    }

    #[test]
    fn test_parse_or() {
        let mut parser = CircuitParser::new("i0 | i1");
        let expr = parser.parse().unwrap();
        match expr {
            CircuitExpr::BinOp { op: CircuitOp::Or, .. } => {}
            _ => panic!("Expected OR operation"),
        }
    }

    #[test]
    fn test_parse_not() {
        let mut parser = CircuitParser::new("!i0");
        let expr = parser.parse().unwrap();
        match expr {
            CircuitExpr::UnaryOp { op: UnaryOp::Not, .. } => {}
            _ => panic!("Expected NOT operation"),
        }
    }

    #[test]
    fn test_parse_complex() {
        let mut parser = CircuitParser::new("(i0 & i1) | (!i2)");
        let expr = parser.parse().unwrap();
        match expr {
            CircuitExpr::BinOp { op: CircuitOp::Or, .. } => {}
            _ => panic!("Expected OR at top level"),
        }
    }

    #[test]
    fn test_transpile_simple() {
        let mut transpiler = CircuitTranspiler::new(2);
        let expr = CircuitExpr::BinOp {
            op: CircuitOp::And,
            left: Box::new(CircuitExpr::Atom(AtomExpr::Input(0))),
            right: Box::new(CircuitExpr::Atom(AtomExpr::Input(1))),
        };
        let circuit = transpiler.transpile(expr).unwrap();
        assert_eq!(circuit.num_inputs, 2);
    }

    #[test]
    fn test_macro_definition() {
        let mut transpiler = CircuitTranspiler::new(2);
        let macro_def = MacroDef {
            name: "xor".to_string(),
            params: vec!["a".to_string(), "b".to_string()],
            body: CircuitExpr::BinOp {
                op: CircuitOp::Xor,
                left: Box::new(CircuitExpr::Var("a".to_string())),
                right: Box::new(CircuitExpr::Var("b".to_string())),
            },
        };
        assert!(transpiler.define_macro(macro_def).is_ok());
    }

    #[test]
    fn test_lint_redundant() {
        let mut transpiler = CircuitTranspiler::new(1);
        let expr = CircuitExpr::BinOp {
            op: CircuitOp::And,
            left: Box::new(CircuitExpr::Atom(AtomExpr::Input(0))),
            right: Box::new(CircuitExpr::Atom(AtomExpr::Input(0))),
        };
        let issues = transpiler.lint(&expr);
        assert!(!issues.is_empty());
        assert_eq!(issues[0].level, LintLevel::Warning);
    }
}
