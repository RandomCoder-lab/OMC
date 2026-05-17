//! End-to-end MCP protocol tests.
//!
//! Spawns the binary, talks JSON-RPC over stdio, asserts on the
//! responses. Covers the full request → handler → response path
//! including JSON parsing and protocol-level errors.
//!
//! Why integration rather than unit tests: the crate is bin-only, so
//! handler functions aren't reachable from a unit-test module. This
//! also exercises the actual protocol path a real LLM client would use.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

/// Find the built `omnimcode-mcp` binary relative to the test
/// executable's path (target/release/deps/integration-XXX or
/// target/debug/deps/integration-XXX → target/{profile}/omnimcode-mcp).
fn find_binary() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    // exe is in target/<profile>/deps/integration-<hash>
    // walk up to target/<profile>/
    let target_profile_dir = exe.parent().unwrap().parent().unwrap();
    let bin = target_profile_dir.join("omnimcode-mcp");
    assert!(
        bin.exists(),
        "binary not found at {} — rebuild with `cargo build -p omnimcode-mcp`",
        bin.display()
    );
    bin
}

/// Find the OMC repo root so test fixtures (`examples/lib/prometheus.omc`)
/// can be referenced by relative path. CARGO_MANIFEST_DIR points at the
/// crate dir; the repo root is one up.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Send a sequence of JSON-RPC request strings to the binary, return
/// the parsed response Values in order. Runs the binary fresh, sets cwd
/// to the OMC repo root so file-path arguments resolve.
fn rpc_exchange(requests: &[Value]) -> Vec<Value> {
    let bin = find_binary();
    let mut child = Command::new(bin)
        .current_dir(repo_root())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mcp server");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    for r in requests {
        writeln!(stdin, "{}", r).expect("write");
    }
    drop(stdin); // closes the server's stdin → it'll exit after replying
    let reader = BufReader::new(stdout);
    let mut responses = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read");
        if line.trim().is_empty() { continue; }
        let v: Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("parse {}: {}", line, e));
        responses.push(v);
    }
    let _ = child.wait();
    responses
}

#[test]
fn initialize_returns_server_info() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
    ]);
    assert_eq!(responses.len(), 1);
    let r = &responses[0];
    assert_eq!(r["id"], 1);
    assert_eq!(r["result"]["serverInfo"]["name"], "omnimcode-mcp");
}

#[test]
fn tools_list_includes_predict_tools() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
    ]);
    let tools = &responses[1]["result"]["tools"];
    let names: Vec<&str> = tools.as_array().unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"omc_predict"), "predict tool present: {:?}", names);
    assert!(names.contains(&"omc_corpus_size"), "corpus_size present: {:?}", names);
    // Pre-existing tools still there too.
    assert!(names.contains(&"omc_eval"));
    assert!(names.contains(&"omc_help"));
}

#[test]
fn omc_corpus_size_ingests_prometheus() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_corpus_size",
            "arguments":{"paths":["examples/lib/prometheus.omc"]}
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], false, "should not be an error: {}", r);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    // Prometheus has ~70 fns currently; lower bound is the only stable assertion.
    let n = payload["fn_count"].as_i64().unwrap();
    assert!(n > 30, "expected >30 fns, got {}", n);
}

#[test]
fn omc_predict_ranks_prom_linear_prefix() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_",
                "top_k":5
            }
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], false, "should not be an error: {}", r);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["prefix"], "fn prom_linear_");
    let suggestions = payload["suggestions"].as_array().unwrap();
    assert!(suggestions.len() >= 3, "should have at least 3 hits for fn prom_linear_, got {}", suggestions.len());
    let names: Vec<&str> = suggestions.iter()
        .map(|s| s["fn_name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"prom_linear_new"), "missing prom_linear_new in {:?}", names);
    assert!(names.contains(&"prom_linear_forward"), "missing prom_linear_forward in {:?}", names);
    assert!(names.contains(&"prom_linear_params"), "missing prom_linear_params in {:?}", names);
    // Each suggestion carries provenance fields.
    let first = &suggestions[0];
    assert!(first["source"].is_string(), "source field");
    assert_eq!(first["file"], "examples/lib/prometheus.omc");
    assert!(first["canonical_hash"].is_i64(), "canonical_hash field");
    assert!(first["prefix_match_len"].as_i64().unwrap() > 0, "prefix matched some tokens");
    assert!(first["substrate_distance"].as_i64().unwrap() >= 0);
}

#[test]
fn omc_predict_top_k_caps_results() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_",
                "top_k":2
            }
        }}),
    ]);
    let text = responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    let suggestions = payload["suggestions"].as_array().unwrap();
    assert!(suggestions.len() <= 2, "top_k=2 capped at 2, got {}", suggestions.len());
}

#[test]
fn omc_predict_missing_paths_is_a_friendly_error() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{"prefix":"fn anything","top_k":3}
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], true);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("missing 'paths'"), "error mentions missing paths: {}", text);
}

#[test]
fn omc_predict_unreadable_path_is_friendly() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["/nonexistent/path/does/not/exist.omc"],
                "prefix":"fn foo"
            }
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], true);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("read") && text.contains("nonexistent"),
            "names the bad path: {}", text);
}

#[test]
fn unknown_tool_returns_error_text() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_does_not_exist","arguments":{}
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], true);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Unknown tool"), "error mentions unknown tool: {}", text);
}
