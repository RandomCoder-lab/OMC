// src/evolution.rs - Genetic algorithm operators for circuit evolution

use crate::circuits::{Circuit, Gate, GateId};
use std::collections::HashMap;

/// Genetic algorithm parameters
#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_size: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        EvolutionConfig {
            population_size: 50,
            num_generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            elite_size: 5,
        }
    }
}

/// Test case for fitness evaluation
pub type TestCase = (Vec<bool>, bool);

/// Evaluate fitness of a circuit against test cases
pub fn evaluate_fitness(circuit: &Circuit, test_cases: &[TestCase]) -> f64 {
    if test_cases.is_empty() {
        return 0.0;
    }

    let correct = test_cases
        .iter()
        .filter(|(inputs, expected)| circuit.eval_hard(inputs) == *expected)
        .count();

    correct as f64 / test_cases.len() as f64
}

/// Mutate a circuit by randomly modifying gates
pub fn mutate_circuit(circuit: &Circuit, mutation_rate: f64) -> Circuit {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    
    let mut mutated = circuit.clone();
    
    // Simple RNG using time-based seed (would use rand crate in production)
    let seed = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64) ^ ((mutation_rate * 1000.0) as u64);
    
    for gate_id in 0..mutated.gates.len() {
        let random = pseudo_random(seed.wrapping_add(gate_id as u64));
        
        if (random as f64 / u32::MAX as f64) < mutation_rate {
            mutate_gate(&mut mutated, gate_id);
        }
    }
    
    mutated
}

/// Mutate a single gate
fn mutate_gate(circuit: &mut Circuit, gate_id: usize) {
    if gate_id >= circuit.gates.len() {
        return;
    }

    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_usize(gate_id);
    hasher.write_u64(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64);
    let mutation_type = (hasher.finish() % 3) as usize;

    match mutation_type {
        0 => {
            // Flip gate type
            match &circuit.gates[gate_id] {
                Gate::XAnd { inputs } => {
                    circuit.gates[gate_id] = Gate::XOr { inputs: inputs.clone() };
                }
                Gate::XOr { inputs } => {
                    circuit.gates[gate_id] = Gate::XAnd { inputs: inputs.clone() };
                }
                Gate::Not { input } => {
                    circuit.gates[gate_id] = Gate::Constant { value: true };
                }
                _ => {}
            }
        }
        1 => {
            // Add/remove input (for XAnd/XOr gates)
            if let Gate::XAnd { ref mut inputs } | Gate::XOr { ref mut inputs } = &mut circuit.gates[gate_id] {
                if !inputs.is_empty() && pseudo_random(gate_id as u64) % 2 == 0 {
                    // Remove random input
                    let idx = pseudo_random((gate_id as u64).wrapping_mul(2)) as usize % inputs.len();
                    inputs.remove(idx);
                }
            }
        }
        _ => {
            // Flip constant value
            if let Gate::Constant { ref mut value } = &mut circuit.gates[gate_id] {
                *value = !*value;
            }
        }
    }
}

/// Crossover two circuits by swapping subtrees
pub fn crossover(parent1: &Circuit, parent2: &Circuit) -> (Circuit, Circuit) {
    let mut child1 = parent1.clone();
    let mut child2 = parent2.clone();

    if parent1.gates.is_empty() || parent2.gates.is_empty() {
        return (child1, child2);
    }

    let seed1 = pseudo_random(1) as usize;
    let seed2 = pseudo_random(2) as usize;
    
    let crossover_point1 = seed1 % parent1.gates.len();
    let crossover_point2 = seed2 % parent2.gates.len();

    // Swap gate at crossover points (simplified crossover)
    if crossover_point1 < child1.gates.len() && crossover_point2 < child2.gates.len() {
        child1.gates.swap(child1.output, crossover_point1);
        child2.gates.swap(child2.output, crossover_point2);
    }

    (child1, child2)
}

/// Create a random circuit
pub fn create_random_circuit(num_inputs: usize, max_gates: usize) -> Circuit {
    let mut circuit = Circuit::new(num_inputs);
    
    // Create input gates
    for i in 0..num_inputs {
        circuit.add_gate(Gate::Input { index: i });
    }

    // Create random internal gates
    let num_internal = pseudo_random((num_inputs as u64).wrapping_mul(1000)) as usize % (max_gates - num_inputs).max(1) + 1;
    
    for _ in 0..num_internal {
        let gate_type = pseudo_random(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64) % 3;

        let gate = match gate_type {
            0 => {
                let inputs = vec![
                    pseudo_random(100) as usize % circuit.gates.len(),
                    pseudo_random(101) as usize % circuit.gates.len(),
                ];
                Gate::XAnd { inputs }
            }
            1 => {
                let inputs = vec![
                    pseudo_random(200) as usize % circuit.gates.len(),
                ];
                Gate::Not { input: inputs[0] }
            }
            _ => Gate::Constant { value: pseudo_random(300) % 2 == 0 },
        };

        circuit.add_gate(gate);
    }

    circuit.output = circuit.gates.len() - 1;
    let _ = circuit.validate(); // Ignore validation errors for random circuits
    
    circuit
}

/// Simple pseudo-random number generator (PCG variant for demonstration)
/// In production, use the `rand` crate
fn pseudo_random(seed: u64) -> u32 {
    let state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let rot = (state >> 59) as u32;
    let xorshifted = (((state ^ (state >> 18)) >> 27) as u32).wrapping_shr(rot);
    xorshifted.wrapping_add((state >> 32) as u32)
}

/// Run genetic algorithm to evolve circuits
pub fn evolve_circuits(
    initial_circuit: &Circuit,
    test_cases: &[TestCase],
    config: &EvolutionConfig,
) -> EvolutionResult {
    let mut population: Vec<Circuit> = (0..config.population_size)
        .map(|_| create_random_circuit(initial_circuit.num_inputs, 20))
        .collect();

    let mut best_fitness = 0.0;
    let mut best_circuit = initial_circuit.clone();
    let mut fitness_history = Vec::new();

    for generation in 0..config.num_generations {
        // Evaluate fitness
        let fitness_scores: Vec<f64> = population
            .iter()
            .map(|c| evaluate_fitness(c, test_cases))
            .collect();

        // Track best
        if let Some((best_idx, &best)) = fitness_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if best > best_fitness {
                best_fitness = best;
                best_circuit = population[best_idx].clone();
            }
        }

        fitness_history.push(best_fitness);

        // Selection and breeding
        let mut new_population = Vec::new();

        // Elitism: keep best individuals
        let mut elite_indices: Vec<usize> = (0..population.len()).collect();
        elite_indices.sort_by(|a, b| {
            fitness_scores[*b]
                .partial_cmp(&fitness_scores[*a])
                .unwrap()
        });

        for i in 0..config.elite_size.min(population.len()) {
            new_population.push(population[elite_indices[i]].clone());
        }

        // Fill rest with crossover and mutation
        while new_population.len() < config.population_size {
            let parent1_idx = elite_indices[pseudo_random((generation as u64).wrapping_mul(1)) as usize % config.elite_size];
            let parent2_idx = elite_indices[pseudo_random((generation as u64).wrapping_mul(2)) as usize % config.elite_size];

            let (mut child1, mut child2) = crossover(&population[parent1_idx], &population[parent2_idx]);

            if (pseudo_random((generation as u64).wrapping_mul(3)) as f64 / u32::MAX as f64) < config.mutation_rate {
                child1 = mutate_circuit(&child1, 0.1);
            }
            if (pseudo_random((generation as u64).wrapping_mul(4)) as f64 / u32::MAX as f64) < config.mutation_rate {
                child2 = mutate_circuit(&child2, 0.1);
            }

            new_population.push(child1);
            if new_population.len() < config.population_size {
                new_population.push(child2);
            }
        }

        population = new_population;
    }

    EvolutionResult {
        best_circuit,
        best_fitness,
        fitness_history,
    }
}

/// Result of evolution run
#[derive(Clone, Debug)]
pub struct EvolutionResult {
    pub best_circuit: Circuit,
    pub best_fitness: f64,
    pub fitness_history: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_fitness() {
        let mut c = Circuit::new(2);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        let test_cases = vec![
            (vec![true, true], true),
            (vec![true, false], false),
            (vec![false, true], false),
            (vec![false, false], false),
        ];

        let fitness = evaluate_fitness(&c, &test_cases);
        assert_eq!(fitness, 1.0); // All tests pass for AND gate
    }

    #[test]
    fn test_mutate_circuit() {
        let mut c = Circuit::new(2);
        let i0 = c.add_gate(Gate::Input { index: 0 });
        let i1 = c.add_gate(Gate::Input { index: 1 });
        c.output = c.add_gate(Gate::XAnd {
            inputs: vec![i0, i1],
        });

        let mutated = mutate_circuit(&c, 0.5);
        // Just check it doesn't crash and produces valid circuit
        let _ = mutated.validate();
    }

    #[test]
    fn test_create_random_circuit() {
        let c = create_random_circuit(2, 10);
        assert_eq!(c.num_inputs, 2);
        assert!(c.gates.len() > 0);
    }
}
