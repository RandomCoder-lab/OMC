// modding-tool/src/json_export.rs

use crate::TruthTable;

pub fn export_json(table: &TruthTable, fitness: f64, gates: usize) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"name\": \"{}\",\n", table.name));
    json.push_str(&format!("  \"inputs\": {},\n", table.inputs));
    json.push_str(&format!("  \"fitness\": {:.2},\n", fitness));
    json.push_str(&format!("  \"gates\": {},\n", gates));
    json.push_str("  \"test_cases\": [\n");

    for (i, (inputs, output)) in table.cases.iter().enumerate() {
        let input_str = inputs
            .iter()
            .map(|b| if *b { "1" } else { "0" })
            .collect::<String>();
        json.push_str(&format!(
            "    {{\"input\": \"{}\", \"output\": {}}}{}\n",
            input_str,
            if *output { "1" } else { "0" },
            if i < table.cases.len() - 1 { "," } else { "" }
        ));
    }

    json.push_str("  ]\n");
    json.push_str("}\n");
    json
}
