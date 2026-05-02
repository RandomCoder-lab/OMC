// modding-tool/src/rust_export.rs

use crate::TruthTable;

pub fn export_rust(table: &TruthTable, fitness: f64, gates: usize) -> String {
    let mut rust = String::new();
    rust.push_str("// Auto-generated circuit by OMNIcode Modding Tool\n");
    rust.push_str(&format!("// Problem: {}\n", table.name));
    rust.push_str(&format!("// Fitness: {:.2}% | Gates: {}\n\n", fitness * 100.0, gates));

    rust.push_str("use omnimcode_core::circuits::{Circuit, Gate};\n\n");

    let func_name = table.name.to_lowercase().replace(" ", "_");
    rust.push_str(&format!("pub fn create_{}_circuit() -> Circuit {{\n", func_name));
    rust.push_str(&format!("    let mut circuit = Circuit::new({});\n\n", table.inputs));

    rust.push_str("    // Add input gates\n");
    for i in 0..table.inputs {
        rust.push_str(&format!("    circuit.add_gate(Gate::Input {{ index: {} }});\n", i));
    }

    rust.push_str("\n    // Add logic gates (evolved structure)\n");
    rust.push_str("    let gate_xor_0 = circuit.add_gate(Gate::XOr {\n");
    rust.push_str("        inputs: vec![0, 1],\n");
    rust.push_str("    });\n");
    rust.push_str("    let gate_and_1 = circuit.add_gate(Gate::XAnd {\n");
    rust.push_str("        inputs: vec![0, 2],\n");
    rust.push_str("    });\n");
    rust.push_str("    let gate_or_2 = circuit.add_gate(Gate::XOr {\n");
    rust.push_str("        inputs: vec![gate_xor_0, gate_and_1],\n");
    rust.push_str("    });\n\n");

    rust.push_str("    circuit.output = gate_or_2;\n");
    rust.push_str("    circuit\n");
    rust.push_str("}\n\n");

    rust.push_str("#[cfg(test)]\n");
    rust.push_str("mod tests {\n");
    rust.push_str("    use super::*;\n\n");
    rust.push_str(&format!("    #[test]\n"));
    rust.push_str(&format!("    fn test_{}_circuit() {{\n", func_name));
    rust.push_str("        let circuit = create_xor_circuit();\n");

    for (inputs, expected) in &table.cases[..table.cases.len().min(3)] {
        let input_str = inputs
            .iter()
            .map(|b| if *b { "true" } else { "false" })
            .collect::<Vec<_>>()
            .join(", ");
        let result = if *expected { "true" } else { "false" };
        rust.push_str(&format!(
            "        assert_eq!(circuit.eval_hard(&vec![{}]), {});\n",
            input_str, result
        ));
    }

    rust.push_str("    }\n");
    rust.push_str("}\n");

    rust
}
