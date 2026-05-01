use criterion::{black_box, criterion_group, criterion_main, Criterion};
use omnimcode::circuits::{Circuit, Gate};
use omnimcode::evolution::{evaluate_fitness, TestCase};

/// Generate XOR test cases (2 inputs, 1 output)
fn xor_test_cases() -> Vec<TestCase> {
    vec![
        (vec![false, false], false),
        (vec![false, true], true),
        (vec![true, false], true),
        (vec![true, true], false),
    ]
}

/// Generate 1-bit adder test cases 
fn adder_test_cases() -> Vec<TestCase> {
    vec![
        (vec![false, false, false], false),
        (vec![false, false, true], true),
        (vec![false, true, false], true),
        (vec![false, true, true], false),
        (vec![true, false, false], true),
        (vec![true, false, true], false),
        (vec![true, true, false], false),
        (vec![true, true, true], true),
    ]
}

fn benchmark_fitness_xor_gate(c: &mut Criterion) {
    // Create an AND gate (simple circuit)
    let mut circuit = Circuit::new(2);
    let i0 = circuit.add_gate(Gate::Input { index: 0 });
    let i1 = circuit.add_gate(Gate::Input { index: 1 });
    circuit.output = circuit.add_gate(Gate::XAnd { inputs: vec![i0, i1] });
    
    let test_cases = black_box(xor_test_cases());
    
    c.bench_function("fitness_eval_and_vs_xor_4cases", |b| {
        b.iter(|| evaluate_fitness(&circuit, &test_cases))
    });
}

fn benchmark_fitness_adder_circuit(c: &mut Criterion) {
    // Create a more complex circuit: (a OR b) XOR c
    let mut circuit = Circuit::new(3);
    let i0 = circuit.add_gate(Gate::Input { index: 0 });
    let i1 = circuit.add_gate(Gate::Input { index: 1 });
    let i2 = circuit.add_gate(Gate::Input { index: 2 });
    
    let or_gate = circuit.add_gate(Gate::XOr { inputs: vec![i0, i1] });
    circuit.output = circuit.add_gate(Gate::XOr { inputs: vec![or_gate, i2] });
    
    let test_cases = black_box(adder_test_cases());
    
    c.bench_function("fitness_eval_xor_xor_vs_adder_8cases", |b| {
        b.iter(|| evaluate_fitness(&circuit, &test_cases))
    });
}

fn benchmark_circuit_eval_deep(c: &mut Criterion) {
    // Create a deeper circuit (5 gates)
    let mut circuit = Circuit::new(2);
    let i0 = circuit.add_gate(Gate::Input { index: 0 });
    let i1 = circuit.add_gate(Gate::Input { index: 1 });
    
    let c1 = circuit.add_gate(Gate::XAnd { inputs: vec![i0, i1] });
    let c2 = circuit.add_gate(Gate::XOr { inputs: vec![i0, i1] });
    let c3 = circuit.add_gate(Gate::Not { input: i0 });
    let c4 = circuit.add_gate(Gate::XAnd { inputs: vec![c1, c2] });
    circuit.output = circuit.add_gate(Gate::XOr { inputs: vec![c4, c3] });
    
    let test_cases = black_box(xor_test_cases());
    
    c.bench_function("fitness_eval_deep_circuit_4cases", |b| {
        b.iter(|| evaluate_fitness(&circuit, &test_cases))
    });
}

criterion_group!(
    benches,
    benchmark_fitness_xor_gate,
    benchmark_fitness_adder_circuit,
    benchmark_circuit_eval_deep
);
criterion_main!(benches);
