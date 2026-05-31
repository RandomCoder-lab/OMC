//! End-to-end test of the in-core addressed-memory builtins, driven through real OMC source
//! (parser + interpreter). The unit tests in src/addressed_memory.rs cover the datastore logic;
//! THIS asserts the language wiring: amem_new / amem_write / amem_index / amem_search / amem_len
//! parse, dispatch, and marshal arrays<->values correctly. If one fails: fix the regression.

use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::value::Value;

fn run(source: &str) -> Result<Value, String> {
    let mut parser = Parser::new(source);
    let stmts = parser.parse()?;
    let mut interp = Interpreter::new();
    interp.execute(stmts)?;
    interp.get_var_for_testing("__result__").ok_or_else(|| "no __result__ variable".to_string())
}

fn int(src: &str) -> i64 {
    run(src).unwrap().to_int()
}

#[test]
fn amem_write_then_search_returns_nearest_cluster_value() {
    // two well-separated clusters; the query sits in cluster 200, so the nearest value is 200.
    let src = r#"
        h m = amem_new();
        amem_write(m, [0.0, 0.0], 100);
        amem_write(m, [0.1, 0.0], 100);
        amem_write(m, [0.0, 0.1], 100);
        amem_write(m, [10.0, 10.0], 200);
        amem_write(m, [10.1, 9.9], 200);
        amem_write(m, [9.9, 10.1], 200);
        __result__ = amem_search(m, [9.95, 10.0], 1, 1)[0];
    "#;
    assert_eq!(int(src), 200);
}

#[test]
fn amem_search_after_ivf_index_is_consistent() {
    // building the IVF index must not change the answer (it only changes WHERE we look).
    let src = r#"
        h m = amem_new();
        amem_write(m, [0.0, 0.0], 1);
        amem_write(m, [0.2, 0.1], 1);
        amem_write(m, [5.0, 5.0], 2);
        amem_write(m, [5.1, 4.9], 2);
        h cells = amem_index(m, 2, 8, 0);
        __result__ = amem_search(m, [5.05, 5.0], 1, 2)[0];
    "#;
    assert_eq!(int(src), 2);
}

#[test]
fn amem_len_counts_entries() {
    let src = r#"
        h m = amem_new();
        amem_write(m, [1.0, 1.0], 1);
        amem_write(m, [2.0, 2.0], 2);
        amem_write(m, [3.0, 3.0], 3);
        __result__ = amem_len(m);
    "#;
    assert_eq!(int(src), 3);
}

#[test]
fn amem_handles_are_independent() {
    // two stores on the same thread must not share entries.
    let src = r#"
        h a = amem_new();
        h b = amem_new();
        amem_write(a, [1.0], 7);
        amem_write(a, [2.0], 7);
        amem_write(b, [9.0], 9);
        __result__ = amem_len(a) * 100 + amem_len(b);
    "#;
    assert_eq!(int(src), 201); // 2 entries in a, 1 in b
}
