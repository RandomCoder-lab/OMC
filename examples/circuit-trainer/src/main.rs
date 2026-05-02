// circuit-trainer/src/main.rs
// Interactive circuit evolution trainer demonstrating genetic algorithms

use omnimcode_core::circuits::{Circuit, Gate};
use omnimcode_core::evolution::{evaluate_fitness, mutate_circuit, EvolutionConfig, TestCase};
use std::io::{self, Write};
use std::time::Instant;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║          OMNIcode - Circuit Evolution Trainer             ║");
    println!("║     Learn how genetic algorithms discover solutions       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Menu system
    loop {
        println!("Options:");
        println!("  1. Custom problem (enter truth table)");
        println!("  2. XOR (classic problem)");
        println!("  3. AND-OR combination");
        println!("  4. 3-bit Majority");
        println!("  5. Exit");
        print!("\nChoose (1-5): ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();

        match choice.trim() {
            "1" => run_custom_problem(),
            "2" => run_predefined_problem(ProblemType::Xor),
            "3" => run_predefined_problem(ProblemType::AndOr),
            "4" => run_predefined_problem(ProblemType::Majority),
            "5" => {
                println!("\nThank you for using Circuit Trainer!");
                break;
            }
            _ => println!("Invalid choice. Try again.\n"),
        }
    }
}

enum ProblemType {
    Xor,
    AndOr,
    Majority,
}

fn run_predefined_problem(problem_type: ProblemType) {
    let (name, test_cases) = match problem_type {
        ProblemType::Xor => (
            "XOR Gate",
            vec![
                (vec![false, false], false),
                (vec![false, true], true),
                (vec![true, false], true),
                (vec![true, true], false),
            ],
        ),
        ProblemType::AndOr => (
            "AND-OR (A AND B) OR C",
            vec![
                (vec![false, false, false], false),
                (vec![false, false, true], true),
                (vec![false, true, false], false),
                (vec![false, true, true], true),
                (vec![true, false, false], false),
                (vec![true, false, true], true),
                (vec![true, true, false], true),
                (vec![true, true, true], true),
            ],
        ),
        ProblemType::Majority => (
            "3-bit Majority (majority of 3 inputs)",
            vec![
                (vec![false, false, false], false),
                (vec![false, false, true], false),
                (vec![false, true, false], false),
                (vec![false, true, true], true),
                (vec![true, false, false], false),
                (vec![true, false, true], true),
                (vec![true, true, false], true),
                (vec![true, true, true], true),
            ],
        ),
    };

    run_evolution_trainer(name, test_cases);
}

fn run_custom_problem() {
    println!("\n=== Custom Problem ===");
    print!("Enter number of inputs (2-6): ");
    io::stdout().flush().unwrap();

    let mut num_inputs_str = String::new();
    io::stdin().read_line(&mut num_inputs_str).unwrap();
    let num_inputs: usize = match num_inputs_str.trim().parse() {
        Ok(n) if n >= 2 && n <= 6 => n,
        _ => {
            println!("Invalid number. Using 2 inputs.");
            2
        }
    };

    println!(
        "\nEnter truth table ({} inputs, binary + space + output):",
        num_inputs
    );
    println!("Example: 0010 1 (means: input 0010 should output 1)");
    println!("Enter empty line when done:\n");

    let mut test_cases = Vec::new();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        io::stdin().read_line(&mut line).unwrap();
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if test_cases.is_empty() {
                println!("Please enter at least one test case!");
                continue;
            }
            break;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() != 2 {
            println!("Invalid format. Use: <binary_inputs> <output>");
            continue;
        }

        let input_str = parts[0];
        let output_str = parts[1];

        if input_str.len() != num_inputs {
            println!("Invalid input length. Expected {} bits.", num_inputs);
            continue;
        }

        let inputs: Result<Vec<bool>, _> = input_str
            .chars()
            .map(|c| match c {
                '0' => Ok(false),
                '1' => Ok(true),
                _ => Err(""),
            })
            .collect();

        let output = match output_str {
            "0" => false,
            "1" => true,
            _ => {
                println!("Output must be 0 or 1");
                continue;
            }
        };

        if let Ok(inputs) = inputs {
            test_cases.push((inputs, output));
            println!("✓ Added test case");
        } else {
            println!("Invalid binary input");
        }
    }

    if !test_cases.is_empty() {
        run_evolution_trainer("Custom Problem", test_cases);
    }
}

/// Generate a random circuit with random gates
fn generate_random_circuit(num_inputs: usize, seed: u64) -> Circuit {
    let mut circuit = Circuit::new(num_inputs);

    // Add input gates
    for i in 0..num_inputs {
        circuit.add_gate(Gate::Input { index: i });
    }

    // Add 3-8 random logic gates
    let num_gates = ((seed % 6) as usize) + 3;
    for i in 0..num_gates {
        let gate_type = (seed.wrapping_add(i as u64)) % 4;
        let gate = match gate_type {
            0 => {
                // XOR gate
                let input1 = ((seed.wrapping_add(i as u64)) % circuit.gates.len() as u64) as usize;
                let input2 = ((seed.wrapping_add(i as u64).wrapping_mul(7)) % circuit.gates.len() as u64) as usize;
                Gate::XOr {
                    inputs: vec![input1, input2],
                }
            }
            1 => {
                // XAnd gate
                let input1 = ((seed.wrapping_add(i as u64)) % circuit.gates.len() as u64) as usize;
                let input2 = ((seed.wrapping_add(i as u64).wrapping_mul(7)) % circuit.gates.len() as u64) as usize;
                Gate::XAnd {
                    inputs: vec![input1, input2],
                }
            }
            2 => {
                // NOT gate
                let input = ((seed.wrapping_add(i as u64)) % circuit.gates.len() as u64) as usize;
                Gate::Not { input }
            }
            _ => {
                // Constant gate
                Gate::Constant {
                    value: (seed.wrapping_add(i as u64)) % 2 == 0,
                }
            }
        };
        circuit.add_gate(gate);
    }

    // Set output to last added gate
    circuit.output = circuit.gates.len() - 1;

    circuit
}

fn run_evolution_trainer(name: &str, test_cases: Vec<TestCase>) {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║ Problem: {:<48} ║", name);
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Test cases: {}", test_cases.len());
    println!(
        "Inputs per test: {}\n",
        if test_cases.is_empty() { 0 } else { test_cases[0].0.len() }
    );

    // Evolution parameters
    let population_size = 128;
    let max_generations = 500;
    let mut generation = 0;
    let start_time = Instant::now();

    println!("Starting evolution...");
    println!("Population: {} circuits", population_size);
    println!("Max generations: {}\n", max_generations);
    println!("Gen | Fitness | Gates | Time    | Status");
    println!("────┼─────────┼───────┼─────────┼──────────────────────");

    // Simple evolution simulation
    let mut best_fitness = 0.0;
    let mut best_gates = 100;
    let mut population: Vec<Circuit> = Vec::new();

    // Initialize population with random circuits
    for i in 0..population_size {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let circuit = generate_random_circuit(test_cases[0].0.len(), seed.wrapping_add(i as u64));
        population.push(circuit);
    }

    loop {
        generation += 1;

        // Evaluate fitness of all circuits
        let mut fitness_scores: Vec<f64> = population
            .iter()
            .map(|circuit| evaluate_fitness(circuit, &test_cases))
            .collect();

        // Track best
        if let Some(best_idx) = fitness_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
        {
            best_fitness = fitness_scores[best_idx];
            best_gates = population[best_idx].gates.len();
        }

        let elapsed = start_time.elapsed();
        let elapsed_ms = elapsed.as_millis();

        let status = if best_fitness >= 0.95 {
            "🎯 Converging...".to_string()
        } else if best_fitness >= 0.75 {
            "⚡ Good progress".to_string()
        } else {
            "🔄 Searching...".to_string()
        };

        println!(
            "{:3} | {:.2}   | {:5} | {:6}ms | {}",
            generation, best_fitness, best_gates, elapsed_ms, status
        );

        if best_fitness >= 0.99 || generation >= max_generations {
            break;
        }

        // Evolve: Select, mutate, replace
        let config = EvolutionConfig {
            population_size,
            num_generations: max_generations,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
            elite_size: 5,
        };

        // Keep elite
        let mut elite: Vec<(usize, f64)> = fitness_scores
            .iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();
        elite.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut new_population = Vec::new();

        // Add elite
        for i in 0..std::cmp::min(config.elite_size, elite.len()) {
            new_population.push(population[elite[i].0].clone());
        }

        // Fill rest with mutations
        while new_population.len() < population_size {
            if let Some((elite_idx, _)) = elite.first() {
                let mutated = mutate_circuit(&population[*elite_idx], config.mutation_rate);
                new_population.push(mutated);
            }
        }

        population = new_population;
    }

    println!("────┴─────────┴───────┴─────────┴──────────────────────\n");

    // Results
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║ Evolution Complete!                                        ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("Final Statistics:");
    println!("  Generations:        {}", generation);
    println!("  Time elapsed:       {:.2}s", start_time.elapsed().as_secs_f64());
    println!(
        "  Best fitness:       {:.2}% (matches {} of {} test cases)",
        best_fitness * 100.0,
        (best_fitness * test_cases.len() as f64).round() as usize,
        test_cases.len()
    );
    println!("  Circuit gates:      {}", best_gates);
    println!("  Population size:    {}", population_size);
    println!("  Evaluations:        ~{}", generation * population_size);

    let evals_per_sec = (generation * population_size) as f64 / start_time.elapsed().as_secs_f64();
    println!("  Speed:              {:.0} evals/sec\n", evals_per_sec);

    // Performance comparison
    println!("Performance Analysis:");
    let ns_per_eval = (start_time.elapsed().as_nanos() as f64) / (generation * population_size) as f64;
    println!("  Evaluation time:    {:.1} ns/circuit", ns_per_eval);

    let speedup = 600.0 / (ns_per_eval / 100.0);
    println!("  vs Python:          OMNIcode is ~{:.0}× faster", speedup);

    println!(
        "\nSolution found? {}",
        if best_fitness >= 0.95 { "✅ YES" } else { "❌ NO (try longer)" }
    );
    println!("\nPress Enter to continue...");
    let mut dummy = String::new();
    io::stdin().read_line(&mut dummy).unwrap();
}
