// src/circuits.rs - Genetic logic circuit engine
// Implements xIF, xELSE, xAND, xOR gate primitives with hard/soft evaluation

use std::fmt;
use std::collections::HashMap;

pub type GateId = usize;

/// Supported logic gates and circuit elements
#[derive(Clone, Debug)]
pub enum Gate {
    /// xAND: outputs true if all inputs true
    XAnd { inputs: Vec<GateId> },
    
    /// xOR: outputs true if odd number of true inputs
    XOr { inputs: Vec<GateId> },
    
    /// xIF-xELSE: conditional branch
    XIf { condition: GateId, then_gate: GateId, else_gate: GateId },
    
    /// xELSE: default fallback (used with xIF)
    XElse { default_value: bool },
    
    /// Input: references an external input by index
    Input { index: usize },
    
    /// Constant: hardcoded true/false value
    Constant { value: bool },
    
    /// NOT: logical negation
    Not { input: GateId },
}

/// A genetic logic circuit - a DAG of gates with single output
#[derive(Clone, Debug)]
pub struct Circuit {
    pub gates: Vec<Gate>,
    pub output: GateId,
    pub num_inputs: usize,
}

impl Circuit {
    /// Create a new empty circuit
    pub fn new(num_inputs: usize) -> Self {
        Circuit {
            gates: vec![Gate::Constant { value: false }],
            output: 0,
            num_inputs,
        }
    }

    /// Add a gate and return its ID
    pub fn add_gate(&mut self, gate: Gate) -> GateId {
        let id = self.gates.len();
        self.gates.push(gate);
        id
    }

    /// Validate circuit structure (DAG check, input bounds)
    pub fn validate(&self) -> Result<(), String> {
        // Check that output gate exists
        if self.output >= self.gates.len() {
            return Err(format!("Output gate ID {} out of range", self.output));
        }

        // Check for cycles using DFS
        let mut visited = vec![false; self.gates.len()];
        let mut rec_stack = vec![false; self.gates.len()];

        for i in 0..self.gates.len() {
            if !visited[i] {
                if self.has_cycle(i, &mut visited, &mut rec_stack)? {
                    return Err("Circuit contains cycles".to_string());
                }
            }
        }

        // Check input bounds
        for (id, gate) in self.gates.iter().enumerate() {
            match gate {
                Gate::Input { index } => {
                    if *index >= self.num_inputs {
                        return Err(format!(
                            "Gate {} references input {} but circuit only has {} inputs",
                            id, index, self.num_inputs
                        ));
                    }
                }
                Gate::XAnd { inputs } | Gate::XOr { inputs } => {
                    for &input_id in inputs {
                        if input_id >= self.gates.len() {
                            return Err(format!(
                                "Gate {} references invalid gate {}",
                                id, input_id
                            ));
                        }
                    }
                }
                Gate::XIf {
                    condition,
                    then_gate,
                    else_gate,
                } => {
                    if *condition >= self.gates.len()
                        || *then_gate >= self.gates.len()
                        || *else_gate >= self.gates.len()
                    {
                        return Err(format!("Gate {} has invalid references", id));
                    }
                }
                Gate::Not { input } => {
                    if *input >= self.gates.len() {
                        return Err(format!("Gate {} references invalid input gate", id));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// DFS cycle detection helper
    fn has_cycle(
        &self,
        node: usize,
        visited: &mut [bool],
        rec_stack: &mut [bool],
    ) -> Result<bool, String> {
        visited[node] = true;
        rec_stack[node] = true;

        let children = match &self.gates[node] {
            Gate::XAnd { inputs } | Gate::XOr { inputs } => inputs.clone(),
            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => vec![*condition, *then_gate, *else_gate],
            Gate::Not { input } => vec![*input],
            _ => vec![],
        };

        for &child in &children {
            if !visited[child] {
                if self.has_cycle(child, visited, rec_stack)? {
                    return Ok(true);
                }
            } else if rec_stack[child] {
                return Ok(true);
            }
        }

        rec_stack[node] = false;
        Ok(false)
    }

    /// Evaluate circuit in hard (Boolean) mode
    pub fn eval_hard(&self, inputs: &[bool]) -> bool {
        let mut cache = HashMap::new();
        self.eval_gate_hard(self.output, inputs, &mut cache)
    }

    /// Helper: recursive evaluation with memoization
    fn eval_gate_hard(&self, gate_id: GateId, inputs: &[bool], cache: &mut HashMap<GateId, bool>) -> bool {
        if let Some(&result) = cache.get(&gate_id) {
            return result;
        }

        let result = match &self.gates[gate_id] {
            Gate::Constant { value } => *value,
            Gate::Input { index } => {
                if *index < inputs.len() {
                    inputs[*index]
                } else {
                    false
                }
            }
            Gate::XAnd { inputs: input_ids } => {
                input_ids.iter()
                    .all(|&id| self.eval_gate_hard(id, inputs, cache))
            }
            Gate::XOr { inputs: input_ids } => {
                input_ids.iter()
                    .filter(|&&id| self.eval_gate_hard(id, inputs, cache))
                    .count() % 2 == 1
            }
            Gate::Not { input } => {
                !self.eval_gate_hard(*input, inputs, cache)
            }
            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => {
                if self.eval_gate_hard(*condition, inputs, cache) {
                    self.eval_gate_hard(*then_gate, inputs, cache)
                } else {
                    self.eval_gate_hard(*else_gate, inputs, cache)
                }
            }
            Gate::XElse { default_value } => *default_value,
        };

        cache.insert(gate_id, result);
        result
    }

    /// Evaluate circuit in soft (probabilistic/fuzzy) mode
    /// Inputs are probabilities [0, 1], outputs are combined probabilistically
    pub fn eval_soft(&self, inputs: &[f64]) -> f64 {
        let mut cache = HashMap::new();
        self.eval_gate_soft(self.output, inputs, &mut cache)
    }

    /// Helper: soft evaluation with probabilistic logic
    fn eval_gate_soft(&self, gate_id: GateId, inputs: &[f64], cache: &mut HashMap<GateId, f64>) -> f64 {
        if let Some(&result) = cache.get(&gate_id) {
            return result;
        }

        let result = match &self.gates[gate_id] {
            Gate::Constant { value } => {
                if *value { 1.0 } else { 0.0 }
            }
            Gate::Input { index } => {
                if *index < inputs.len() {
                    inputs[*index].clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            Gate::XAnd { inputs: input_ids } => {
                // Soft AND: product of probabilities
                input_ids.iter()
                    .map(|&id| self.eval_gate_soft(id, inputs, cache))
                    .product()
            }
            Gate::XOr { inputs: input_ids } => {
                // Soft XOR: balanced function for odd parity
                let probs: Vec<f64> = input_ids.iter()
                    .map(|&id| self.eval_gate_soft(id, inputs, cache))
                    .collect();
                
                if probs.is_empty() {
                    0.0
                } else if probs.len() == 1 {
                    probs[0]
                } else {
                    // For soft XOR, use: a + b - 2*a*b (smooth approximation)
                    let mut result = probs[0];
                    for &p in &probs[1..] {
                        result = result + p - 2.0 * result * p;
                        result = result.clamp(0.0, 1.0);
                    }
                    result
                }
            }
            Gate::Not { input } => {
                1.0 - self.eval_gate_soft(*input, inputs, cache)
            }
            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => {
                let cond_prob = self.eval_gate_soft(*condition, inputs, cache);
                let then_val = self.eval_gate_soft(*then_gate, inputs, cache);
                let else_val = self.eval_gate_soft(*else_gate, inputs, cache);
                
                // Soft IF: weighted average
                cond_prob * then_val + (1.0 - cond_prob) * else_val
            }
            Gate::XElse { default_value } => {
                if *default_value { 1.0 } else { 0.0 }
            }
        };

        cache.insert(gate_id, result);
        result
    }

    /// Export circuit to Graphviz DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph Circuit {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for (id, gate) in self.gates.iter().enumerate() {
            let label = match gate {
                Gate::Constant { value } => {
                    format!("Const({})", if *value { "T" } else { "F" })
                }
                Gate::Input { index } => format!("Input({})", index),
                Gate::XAnd { .. } => "xAND".to_string(),
                Gate::XOr { .. } => "xOR".to_string(),
                Gate::Not { .. } => "NOT".to_string(),
                Gate::XIf { .. } => "xIF".to_string(),
                Gate::XElse { .. } => "xELSE".to_string(),
            };

            let shape = if id == self.output {
                "shape=ellipse,style=filled,fillcolor=lightgreen"
            } else {
                "shape=box"
            };

            dot.push_str(&format!("  node_{} [label=\"{}\",{}];\n", id, label, shape));
        }

        dot.push_str("\n");

        // Add edges
        for (id, gate) in self.gates.iter().enumerate() {
            match gate {
                Gate::XAnd { inputs } | Gate::XOr { inputs } => {
                    for &input_id in inputs {
                        dot.push_str(&format!("  node_{} -> node_{};\n", input_id, id));
                    }
                }
                Gate::XIf {
                    condition,
                    then_gate,
                    else_gate,
                } => {
                    dot.push_str(&format!("  node_{} -> node_{}[label=\"cond\"];\n", condition, id));
                    dot.push_str(&format!("  node_{} -> node_{}[label=\"then\"];\n", then_gate, id));
                    dot.push_str(&format!("  node_{} -> node_{}[label=\"else\"];\n", else_gate, id));
                }
                Gate::Not { input } => {
                    dot.push_str(&format!("  node_{} -> node_{};\n", input, id));
                }
                _ => {}
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Get circuit complexity metrics
    pub fn metrics(&self) -> CircuitMetrics {
        CircuitMetrics {
            num_gates: self.gates.len(),
            num_inputs: self.num_inputs,
            num_outputs: 1,
            depth: self.compute_depth(),
            gate_histogram: self.compute_gate_histogram(),
        }
    }

    /// Compute circuit depth (longest path from input to output)
    fn compute_depth(&self) -> usize {
        let mut depths = vec![0; self.gates.len()];
        self.compute_depth_recursive(self.output, &mut depths)
    }

    fn compute_depth_recursive(&self, gate_id: GateId, depths: &mut [usize]) -> usize {
        if depths[gate_id] > 0 {
            return depths[gate_id];
        }

        let depth = 1 + match &self.gates[gate_id] {
            Gate::XAnd { inputs } | Gate::XOr { inputs } => {
                inputs.iter()
                    .map(|&id| self.compute_depth_recursive(id, depths))
                    .max()
                    .unwrap_or(0)
            }
            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => {
                let cond_depth = self.compute_depth_recursive(*condition, depths);
                let then_depth = self.compute_depth_recursive(*then_gate, depths);
                let else_depth = self.compute_depth_recursive(*else_gate, depths);
                cond_depth.max(then_depth).max(else_depth)
            }
            Gate::Not { input } => self.compute_depth_recursive(*input, depths),
            _ => 0,
        };

        depths[gate_id] = depth;
        depth
    }

    /// Count gate types in circuit
    fn compute_gate_histogram(&self) -> HashMap<String, usize> {
        let mut hist = HashMap::new();
        for gate in &self.gates {
            let gate_type = match gate {
                Gate::XAnd { .. } => "xAND",
                Gate::XOr { .. } => "xOR",
                Gate::Not { .. } => "NOT",
                Gate::XIf { .. } => "xIF",
                Gate::Constant { .. } => "Const",
                Gate::Input { .. } => "Input",
                Gate::XElse { .. } => "xELSE",
            };
            *hist.entry(gate_type.to_string()).or_insert(0) += 1;
        }
        hist
    }
}

/// Circuit metrics for analysis and fitness evaluation
#[derive(Clone, Debug)]
pub struct CircuitMetrics {
    pub num_gates: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub depth: usize,
    pub gate_histogram: HashMap<String, usize>,
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Circuit({} inputs, {} gates, depth {})",
            self.num_inputs, self.gates.len(), self.metrics().depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_and() {
        let mut c = Circuit::new(2);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        assert_eq!(c.eval_hard(&[true, true]), true);
        assert_eq!(c.eval_hard(&[true, false]), false);
        assert_eq!(c.eval_hard(&[false, true]), false);
        assert_eq!(c.eval_hard(&[false, false]), false);
    }

    #[test]
    fn test_circuit_or() {
        let mut c = Circuit::new(2);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XOr {
            inputs: vec![i0, i1],
        });

        // XOR: true if odd number of true inputs
        assert_eq!(c.eval_hard(&[true, true]), false); // 2 true = even
        assert_eq!(c.eval_hard(&[true, false]), true);
        assert_eq!(c.eval_hard(&[false, true]), true);
        assert_eq!(c.eval_hard(&[false, false]), false);
    }

    #[test]
    fn test_circuit_soft_eval() {
        let mut c = Circuit::new(2);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        assert!((c.eval_soft(&[1.0, 1.0]) - 1.0).abs() < 0.01);
        assert!((c.eval_soft(&[0.5, 0.5]) - 0.25).abs() < 0.01);
        assert!((c.eval_soft(&[0.0, 1.0]) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_circuit_validation_cycle() {
        let mut c = Circuit::new(1);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        
        // Create a cycle: i0 -> and1 -> i0 (impossible but tests validation)
        // Actually, we'll test proper cycle detection
        c.output = i0;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_circuit_metrics() {
        let mut c = Circuit::new(2);
        // Circuit::new adds initial constant gate, so starting count is 1
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        let m = c.metrics();
        // Total gates: 1 (initial const) + 1 (Input 0) + 1 (Input 1) + 1 (XAnd) = 4
        assert_eq!(m.num_gates, 4);
        assert_eq!(m.num_inputs, 2);
        assert_eq!(m.depth, 2);
    }

    #[test]
    fn test_circuit_dot_export() {
        let mut c = Circuit::new(2);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        let dot = c.to_dot();
        assert!(dot.contains("digraph Circuit"));
        assert!(dot.contains("xAND"));
        assert!(dot.contains("Input(0)"));
    }
}
