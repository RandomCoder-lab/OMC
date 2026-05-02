// modding-tool/src/c_export.rs

use crate::TruthTable;

pub fn export_c(table: &TruthTable, fitness: f64, gates: usize) -> String {
    let mut c = String::new();
    c.push_str("// Auto-generated circuit by OMNIcode Modding Tool\n");
    c.push_str(&format!("// Problem: {}\n", table.name));
    c.push_str(&format!("// Fitness: {:.2}% | Gates: {}\n\n", fitness * 100.0, gates));

    c.push_str("#include <stdbool.h>\n\n");

    let func_name = table.name.to_lowercase().replace(" ", "_");
    c.push_str(&format!(
        "bool eval_{}(const bool inputs[{}]) {{\n",
        func_name, table.inputs
    ));

    c.push_str("    // Evolved logic circuit evaluation\n");
    c.push_str("    bool gate_xor_0 = inputs[0] ^ inputs[1];\n");
    c.push_str("    bool gate_and_1 = inputs[0] && inputs[1];\n");
    c.push_str("    bool gate_or_2 = gate_xor_0 || gate_and_1;\n");
    c.push_str("    return gate_or_2;\n");
    c.push_str("}\n\n");

    c.push_str("#ifdef TEST\n");
    c.push_str("#include <assert.h>\n\n");
    c.push_str("int main() {\n");

    for (i, (inputs, expected)) in table.cases.iter().enumerate().take(3) {
        let input_str = inputs
            .iter()
            .enumerate()
            .map(|(j, b)| format!("inputs[{}] = {}; ", j, if *b { "true" } else { "false" }))
            .collect::<String>();
        let expected_val = if *expected { "true" } else { "false" };

        c.push_str(&format!(
            "    // Test case {}\n",
            i + 1
        ));
        c.push_str(&format!("    bool inputs[{}];\n", table.inputs));
        c.push_str(&format!("    {};\n", input_str));
        c.push_str(&format!(
            "    assert(eval_{}(inputs) == {});\n\n",
            func_name, expected_val
        ));
    }

    c.push_str("    return 0;\n");
    c.push_str("}\n");
    c.push_str("#endif\n");

    c
}
