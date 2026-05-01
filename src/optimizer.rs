// src/optimizer.rs - Circuit Optimization Engine
// Tier 3: constant folding, algebraic simplification, dead code elimination

use crate::circuits::{Circuit, Gate, GateId};
use std::collections::{HashMap, HashSet};

/// Optimization statistics
#[derive(Clone, Debug, Default)]
pub struct OptimizationStats {
    pub gates_removed: usize,
    pub constant_folds: usize,
    pub algebraic_simplifications: usize,
    pub dead_code_eliminated: usize,
    pub original_gate_count: usize,
    pub optimized_gate_count: usize,
}

impl OptimizationStats {
    /// Calculate improvement percentage
    pub fn improvement_percent(&self) -> f64 {
        if self.original_gate_count == 0 {
            0.0
        } else {
            ((self.original_gate_count - self.optimized_gate_count) as f64
                / self.original_gate_count as f64)
                * 100.0
        }
    }

    /// Calculate speedup estimate
    pub fn estimated_speedup(&self) -> f64 {
        if self.optimized_gate_count == 0 {
            1.0
        } else {
            self.original_gate_count as f64 / self.optimized_gate_count as f64
        }
    }
}

/// Circuit optimizer
pub struct CircuitOptimizer {
    stats: OptimizationStats,
    gate_map: HashMap<GateId, GateId>, // Maps old gate IDs to new gate IDs
}

impl CircuitOptimizer {
    pub fn new() -> Self {
        Self {
            stats: OptimizationStats::default(),
            gate_map: HashMap::new(),
        }
    }

    /// Optimize a circuit (all passes)
    pub fn optimize(&mut self, circuit: &Circuit) -> (Circuit, OptimizationStats) {
        let mut optimized = circuit.clone();
        self.stats.original_gate_count = optimized.gates.len();

        // Pass 1: Constant folding
        optimized = self.constant_fold_pass(&optimized);

        // Pass 2: Algebraic simplification
        optimized = self.algebraic_simplify_pass(&optimized);

        // Pass 3: Dead code elimination
        optimized = self.dead_code_elimination_pass(&optimized);

        // Repeat passes until convergence
        let mut iterations = 0;
        let max_iterations = 5;
        let mut prev_count = optimized.gates.len();

        while iterations < max_iterations {
            optimized = self.constant_fold_pass(&optimized);
            optimized = self.algebraic_simplify_pass(&optimized);
            optimized = self.dead_code_elimination_pass(&optimized);

            if optimized.gates.len() == prev_count {
                break; // Converged
            }
            prev_count = optimized.gates.len();
            iterations += 1;
        }

        self.stats.optimized_gate_count = optimized.gates.len();
        self.stats.gates_removed =
            self.stats.original_gate_count.saturating_sub(self.stats.optimized_gate_count);

        (optimized, self.stats.clone())
    }

    /// Constant folding: evaluate constant expressions at compile time
    fn constant_fold_pass(&mut self, circuit: &Circuit) -> Circuit {
        let mut optimized = Circuit::new(circuit.num_inputs);

        // Track original to optimized gate mapping
        let mut gate_map: HashMap<GateId, GateId> = HashMap::new();

        // Pre-populate input mappings
        for i in 0..circuit.num_inputs {
            let gate_id = optimized.add_gate(Gate::Input { index: i });
            gate_map.insert(i, gate_id);
        }

        // Process each gate
        for (orig_id, gate) in circuit.gates.iter().enumerate() {
            if orig_id < circuit.num_inputs {
                continue; // Skip inputs
            }

            let folded_result = self.try_fold_gate(gate, &gate_map, circuit);

            if let Some(constant_val) = folded_result {
                // Gate folded to constant
                let new_id = optimized.add_gate(Gate::Constant { value: constant_val });
                gate_map.insert(orig_id, new_id);
                self.stats.constant_folds += 1;
            } else {
                // Gate couldn't be folded, remap inputs and add
                let new_gate = self.remap_gate_inputs(gate, &gate_map);
                let new_id = optimized.add_gate(new_gate);
                gate_map.insert(orig_id, new_id);
            }
        }

        // Remap output
        optimized.output = gate_map
            .get(&circuit.output)
            .copied()
            .unwrap_or(circuit.output);

        optimized
    }

    /// Try to fold a gate to a constant
    fn try_fold_gate(
        &self,
        gate: &Gate,
        gate_map: &HashMap<GateId, GateId>,
        circuit: &Circuit,
    ) -> Option<bool> {
        match gate {
            Gate::XAnd { inputs } => {
                let values: Option<Vec<bool>> = inputs
                    .iter()
                    .map(|&id| self.get_gate_constant_value(id, gate_map, circuit))
                    .collect();

                values.map(|vals| vals.iter().all(|&v| v))
            }

            Gate::XOr { inputs } => {
                let values: Option<Vec<bool>> = inputs
                    .iter()
                    .map(|&id| self.get_gate_constant_value(id, gate_map, circuit))
                    .collect();

                values.map(|vals| vals.iter().filter(|&&v| v).count() % 2 == 1)
            }

            Gate::Not { input } => self
                .get_gate_constant_value(*input, gate_map, circuit)
                .map(|v| !v),

            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => {
                let cond_val = self.get_gate_constant_value(*condition, gate_map, circuit);
                if let Some(c) = cond_val {
                    if c {
                        self.get_gate_constant_value(*then_gate, gate_map, circuit)
                    } else {
                        self.get_gate_constant_value(*else_gate, gate_map, circuit)
                    }
                } else {
                    None
                }
            }

            Gate::Constant { value } => Some(*value),
            _ => None,
        }
    }

    /// Get constant value of a gate if it's constant
    fn get_gate_constant_value(
        &self,
        gate_id: GateId,
        gate_map: &HashMap<GateId, GateId>,
        circuit: &Circuit,
    ) -> Option<bool> {
        if gate_id >= circuit.gates.len() {
            return None;
        }

        if let Gate::Constant { value } = &circuit.gates[gate_id] {
            return Some(*value);
        }

        None
    }

    /// Remap gate inputs according to gate_map
    fn remap_gate_inputs(&self, gate: &Gate, gate_map: &HashMap<GateId, GateId>) -> Gate {
        match gate {
            Gate::XAnd { inputs } => {
                let new_inputs = inputs
                    .iter()
                    .map(|&id| gate_map.get(&id).copied().unwrap_or(id))
                    .collect();
                Gate::XAnd { inputs: new_inputs }
            }
            Gate::XOr { inputs } => {
                let new_inputs = inputs
                    .iter()
                    .map(|&id| gate_map.get(&id).copied().unwrap_or(id))
                    .collect();
                Gate::XOr { inputs: new_inputs }
            }
            Gate::Not { input } => {
                let new_input = gate_map.get(input).copied().unwrap_or(*input);
                Gate::Not { input: new_input }
            }
            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => {
                let new_cond = gate_map.get(condition).copied().unwrap_or(*condition);
                let new_then = gate_map.get(then_gate).copied().unwrap_or(*then_gate);
                let new_else = gate_map.get(else_gate).copied().unwrap_or(*else_gate);
                Gate::XIf {
                    condition: new_cond,
                    then_gate: new_then,
                    else_gate: new_else,
                }
            }
            other => other.clone(),
        }
    }

    /// Algebraic simplification: apply identities like a & true → a
    fn algebraic_simplify_pass(&mut self, circuit: &Circuit) -> Circuit {
        let mut optimized = Circuit::new(circuit.num_inputs);
        let mut gate_map: HashMap<GateId, GateId> = HashMap::new();

        // Pre-populate inputs
        for i in 0..circuit.num_inputs {
            let gate_id = optimized.add_gate(Gate::Input { index: i });
            gate_map.insert(i, gate_id);
        }

        for (orig_id, gate) in circuit.gates.iter().enumerate() {
            if orig_id < circuit.num_inputs {
                continue;
            }

            if let Some(simplified) = self.try_simplify_gate(gate, &gate_map, circuit) {
                match simplified {
                    SimplifyResult::Constant(val) => {
                        let new_id = optimized.add_gate(Gate::Constant { value: val });
                        gate_map.insert(orig_id, new_id);
                        self.stats.algebraic_simplifications += 1;
                    }
                    SimplifyResult::Gate(new_gate) => {
                        let new_id = optimized.add_gate(new_gate);
                        gate_map.insert(orig_id, new_id);
                        self.stats.algebraic_simplifications += 1;
                    }
                    SimplifyResult::Reference(ref_id) => {
                        gate_map.insert(
                            orig_id,
                            gate_map.get(&ref_id).copied().unwrap_or(ref_id),
                        );
                        self.stats.algebraic_simplifications += 1;
                    }
                    SimplifyResult::None => {
                        let new_gate = self.remap_gate_inputs(gate, &gate_map);
                        let new_id = optimized.add_gate(new_gate);
                        gate_map.insert(orig_id, new_id);
                    }
                }
            } else {
                let new_gate = self.remap_gate_inputs(gate, &gate_map);
                let new_id = optimized.add_gate(new_gate);
                gate_map.insert(orig_id, new_id);
            }
        }

        optimized.output = gate_map
            .get(&circuit.output)
            .copied()
            .unwrap_or(circuit.output);

        optimized
    }

    /// Try to simplify a gate using algebraic identities
    fn try_simplify_gate(
        &self,
        gate: &Gate,
        gate_map: &HashMap<GateId, GateId>,
        circuit: &Circuit,
    ) -> Option<SimplifyResult> {
        match gate {
            // AND identities
            Gate::XAnd { inputs } => {
                // a & true → a (identity)
                if inputs.len() == 2 {
                    let a = inputs[0];
                    let b = inputs[1];

                    // Check for a & true
                    if let Gate::Constant { value: true } = &circuit.gates[b] {
                        return Some(SimplifyResult::Reference(a));
                    }
                    // Check for true & a
                    if let Gate::Constant { value: true } = &circuit.gates[a] {
                        return Some(SimplifyResult::Reference(b));
                    }

                    // a & false → false (annihilation)
                    if let Gate::Constant { value: false } = &circuit.gates[b] {
                        return Some(SimplifyResult::Constant(false));
                    }
                    if let Gate::Constant { value: false } = &circuit.gates[a] {
                        return Some(SimplifyResult::Constant(false));
                    }

                    // a & !a → false (contradiction)
                    if let Gate::Not { input: neg_inner } = &circuit.gates[b] {
                        if *neg_inner == a {
                            return Some(SimplifyResult::Constant(false));
                        }
                    }
                    if let Gate::Not { input: neg_inner } = &circuit.gates[a] {
                        if *neg_inner == b {
                            return Some(SimplifyResult::Constant(false));
                        }
                    }
                }

                // All inputs same: a & a → a
                if inputs.len() > 1 && inputs.iter().all(|&id| id == inputs[0]) {
                    return Some(SimplifyResult::Reference(inputs[0]));
                }

                None
            }

            // OR/XOR identities
            Gate::XOr { inputs } => {
                // a | false → a (identity)
                if inputs.len() == 2 {
                    let a = inputs[0];
                    let b = inputs[1];

                    // Check for a | false
                    if let Gate::Constant { value: false } = &circuit.gates[b] {
                        return Some(SimplifyResult::Reference(a));
                    }
                    // Check for false | a
                    if let Gate::Constant { value: false } = &circuit.gates[a] {
                        return Some(SimplifyResult::Reference(b));
                    }

                    // a | true → true (domination)
                    if let Gate::Constant { value: true } = &circuit.gates[b] {
                        return Some(SimplifyResult::Constant(true));
                    }
                    if let Gate::Constant { value: true } = &circuit.gates[a] {
                        return Some(SimplifyResult::Constant(true));
                    }

                    // a | a → false (XOR: odd parity)
                    if a == b {
                        return Some(SimplifyResult::Constant(false));
                    }

                    // a | !a → true (tautology for XOR with single NOT)
                    if let Gate::Not { input: neg_inner } = &circuit.gates[b] {
                        if *neg_inner == a {
                            return Some(SimplifyResult::Constant(true));
                        }
                    }
                    if let Gate::Not { input: neg_inner } = &circuit.gates[a] {
                        if *neg_inner == b {
                            return Some(SimplifyResult::Constant(true));
                        }
                    }
                }

                None
            }

            // Double negation: !!a → a
            Gate::Not { input } => {
                if let Gate::Not { input: inner } = &circuit.gates[*input] {
                    return Some(SimplifyResult::Reference(*inner));
                }

                // !true → false
                if let Gate::Constant { value: true } = &circuit.gates[*input] {
                    return Some(SimplifyResult::Constant(false));
                }

                // !false → true
                if let Gate::Constant { value: false } = &circuit.gates[*input] {
                    return Some(SimplifyResult::Constant(true));
                }

                None
            }

            // IF simplification
            Gate::XIf {
                condition,
                then_gate,
                else_gate,
            } => {
                // if true then a else b → a
                if let Gate::Constant { value: true } = &circuit.gates[*condition] {
                    return Some(SimplifyResult::Reference(*then_gate));
                }

                // if false then a else b → b
                if let Gate::Constant { value: false } = &circuit.gates[*condition] {
                    return Some(SimplifyResult::Reference(*else_gate));
                }

                // if a then a else false → a (idempotent)
                if then_gate == condition {
                    if let Gate::Constant { value: false } = &circuit.gates[*else_gate] {
                        return Some(SimplifyResult::Reference(*condition));
                    }
                }

                // if a then true else false → a
                if let Gate::Constant { value: true } = &circuit.gates[*then_gate] {
                    if let Gate::Constant { value: false } = &circuit.gates[*else_gate] {
                        return Some(SimplifyResult::Reference(*condition));
                    }
                }

                // if a then false else true → !a
                if let Gate::Constant { value: false } = &circuit.gates[*then_gate] {
                    if let Gate::Constant { value: true } = &circuit.gates[*else_gate] {
                        let not_gate = Gate::Not {
                            input: *condition,
                        };
                        return Some(SimplifyResult::Gate(not_gate));
                    }
                }

                None
            }

            _ => None,
        }
    }

    /// Dead code elimination: remove unreachable gates
    fn dead_code_elimination_pass(&mut self, circuit: &Circuit) -> Circuit {
        // Mark reachable gates
        let mut reachable = HashSet::new();
        self.mark_reachable(circuit.output, circuit, &mut reachable);

        // Mark all inputs as reachable
        for i in 0..circuit.num_inputs {
            reachable.insert(i);
        }

        // Build mapping from old IDs to new IDs (only for reachable gates)
        let mut gate_map: HashMap<GateId, GateId> = HashMap::new();
        let mut new_circuit = Circuit::new(circuit.num_inputs);

        // Add inputs
        for i in 0..circuit.num_inputs {
            let gate_id = new_circuit.add_gate(Gate::Input { index: i });
            gate_map.insert(i, gate_id);
        }

        // Add reachable gates in order
        for (old_id, gate) in circuit.gates.iter().enumerate() {
            if reachable.contains(&old_id) {
                let new_gate = self.remap_gate_inputs(gate, &gate_map);
                let new_id = new_circuit.add_gate(new_gate);
                gate_map.insert(old_id, new_id);
            } else {
                self.stats.dead_code_eliminated += 1;
            }
        }

        new_circuit.output = gate_map
            .get(&circuit.output)
            .copied()
            .unwrap_or(circuit.output);

        new_circuit
    }

    /// Mark reachable gates by walking backward from output
    fn mark_reachable(&self, gate_id: GateId, circuit: &Circuit, reachable: &mut HashSet<GateId>) {
        if gate_id >= circuit.gates.len() || reachable.contains(&gate_id) {
            return;
        }

        reachable.insert(gate_id);

        if let Some(gate) = circuit.gates.get(gate_id) {
            match gate {
                Gate::XAnd { inputs } | Gate::XOr { inputs } => {
                    for &input_id in inputs {
                        self.mark_reachable(input_id, circuit, reachable);
                    }
                }
                Gate::Not { input } => {
                    self.mark_reachable(*input, circuit, reachable);
                }
                Gate::XIf {
                    condition,
                    then_gate,
                    else_gate,
                } => {
                    self.mark_reachable(*condition, circuit, reachable);
                    self.mark_reachable(*then_gate, circuit, reachable);
                    self.mark_reachable(*else_gate, circuit, reachable);
                }
                _ => {}
            }
        }
    }

    pub fn get_stats(&self) -> OptimizationStats {
        self.stats.clone()
    }
}

/// Simplification result
enum SimplifyResult {
    Constant(bool),
    Gate(Gate),
    Reference(GateId),
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding() {
        let mut circuit = Circuit::new(1);
        let i0 = circuit.add_gate(Gate::Input { index: 0 });
        let t = circuit.add_gate(Gate::Constant { value: true });
        let f = circuit.add_gate(Gate::Constant { value: false });

        // a & true & false → false
        let and1 = circuit.add_gate(Gate::XAnd {
            inputs: vec![i0, t],
        });
        let and2 = circuit.add_gate(Gate::XAnd {
            inputs: vec![and1, f],
        });
        circuit.output = and2;

        let mut optimizer = CircuitOptimizer::new();
        let (opt, stats) = optimizer.optimize(&circuit);

        // Should fold to constant false
        assert!(stats.constant_folds > 0);
        assert!(opt.gates.len() < circuit.gates.len());
    }

    #[test]
    fn test_algebraic_simplification_and_identity() {
        let mut circuit = Circuit::new(1);
        let i0 = circuit.add_gate(Gate::Input { index: 0 });
        let t = circuit.add_gate(Gate::Constant { value: true });

        // a & true → a
        let and_gate = circuit.add_gate(Gate::XAnd {
            inputs: vec![i0, t],
        });
        circuit.output = and_gate;

        let mut optimizer = CircuitOptimizer::new();
        let (opt, stats) = optimizer.optimize(&circuit);

        assert!(stats.algebraic_simplifications > 0);
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut circuit = Circuit::new(2);
        let i0 = circuit.add_gate(Gate::Input { index: 0 });
        let i1 = circuit.add_gate(Gate::Input { index: 1 });

        // Dead code: this output is never used
        let _dead = circuit.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        // Real output: just i0
        let output = circuit.add_gate(Gate::Constant { value: false });
        circuit.output = output;

        let mut optimizer = CircuitOptimizer::new();
        let (opt, stats) = optimizer.optimize(&circuit);

        assert!(stats.dead_code_eliminated > 0);
    }

    #[test]
    fn test_double_negation() {
        let mut circuit = Circuit::new(1);
        let i0 = circuit.add_gate(Gate::Input { index: 0 });
        let not1 = circuit.add_gate(Gate::Not { input: i0 });
        let not2 = circuit.add_gate(Gate::Not { input: not1 });
        circuit.output = not2;

        let mut optimizer = CircuitOptimizer::new();
        let (opt, stats) = optimizer.optimize(&circuit);

        assert!(stats.algebraic_simplifications > 0);
        // Should simplify to i0 (or close to it)
        assert!(opt.gates.len() <= circuit.gates.len());
    }

    #[test]
    fn test_speedup_calculation() {
        let mut stats = OptimizationStats {
            original_gate_count: 10,
            optimized_gate_count: 5,
            ..Default::default()
        };

        assert!((stats.improvement_percent() - 50.0).abs() < 0.1);
        assert!((stats.estimated_speedup() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_convergence() {
        let mut circuit = Circuit::new(1);
        let i0 = circuit.add_gate(Gate::Input { index: 0 });
        let t = circuit.add_gate(Gate::Constant { value: true });
        let f = circuit.add_gate(Gate::Constant { value: false });

        let and1 = circuit.add_gate(Gate::XAnd {
            inputs: vec![i0, t],
        });
        let and2 = circuit.add_gate(Gate::XAnd {
            inputs: vec![and1, f],
        });
        circuit.output = and2;

        let mut optimizer = CircuitOptimizer::new();
        let (opt, stats) = optimizer.optimize(&circuit);

        // Multiple passes should converge to minimal circuit
        assert!(opt.gates.len() < circuit.gates.len());
        assert!(stats.improvement_percent() > 0.0);
    }
}
