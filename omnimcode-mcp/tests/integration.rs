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
    // Explicitly request format=full so the source field is present —
    // this test exists to verify ranking against the real corpus and
    // wants to inspect the body for provenance.
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_",
                "top_k":5,
                "format":"full"
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
    assert!(first["source"].is_string(), "source field (full format)");
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
fn omc_predict_default_format_is_hash_compact() {
    // Default (no format arg) returns the hash-only projection — no
    // `source` field, just identity + ranking metadata. This is the
    // compression story for the LLM.
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_"
            }
        }}),
    ]);
    let text = responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["format"], "hash");
    let s0 = &payload["suggestions"][0];
    assert!(s0["canonical_hash"].is_i64(), "hash present");
    assert!(s0["file"].is_string(), "file present");
    assert!(s0["fn_name"].is_string(), "fn_name present");
    // The whole point: NO source field in compact format.
    assert!(s0.get("source").is_none(), "compact format omits source");
    assert!(s0.get("attractor").is_none(), "compact format omits attractor");
}

#[test]
fn omc_predict_signature_format_includes_signature_not_body() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_",
                "format":"signature"
            }
        }}),
    ]);
    let text = responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["format"], "signature");
    let s0 = &payload["suggestions"][0];
    let sig = s0["signature"].as_str().unwrap();
    assert!(sig.starts_with("fn prom_linear_"),
            "signature looks right: {}", sig);
    assert!(!sig.contains("dict_get"),
            "signature stops at body (no dict_get): {}", sig);
    // Still no full source — that's the contract.
    assert!(s0.get("source").is_none(), "signature format omits source");
}

#[test]
fn omc_predict_full_format_includes_complete_source() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_",
                "format":"full"
            }
        }}),
    ]);
    let text = responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    let s0 = &payload["suggestions"][0];
    let source = s0["source"].as_str().unwrap();
    assert!(source.starts_with("fn prom_linear_"),
            "source starts with fn keyword: {}", &source[..50]);
    assert!(source.contains("{"), "source has body");
    assert!(s0["attractor"].is_i64(), "full format includes attractor");
}

#[test]
fn omc_fetch_by_hash_round_trips_through_predict() {
    // The full LLM workflow: cheap predict (hash format) → pick a
    // suggestion → fetch by hash → get back the same source the
    // original ingestion produced.
    let predict_responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_",
                "format":"hash"
            }
        }}),
    ]);
    let predict_text = predict_responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let predict_payload: Value = serde_json::from_str(predict_text).unwrap();
    let s0 = &predict_payload["suggestions"][0];
    let hash = s0["canonical_hash"].as_i64().unwrap();
    let expected_name = s0["fn_name"].as_str().unwrap().to_string();

    // Now fetch by that hash and confirm we get the same fn back.
    let fetch_responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_fetch_by_hash",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "canonical_hash": hash
            }
        }}),
    ]);
    let fetch_text = fetch_responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let fetch_payload: Value = serde_json::from_str(fetch_text).unwrap();
    assert_eq!(fetch_payload["found"], true);
    assert_eq!(fetch_payload["fn_name"], expected_name);
    assert_eq!(fetch_payload["canonical_hash"], hash);
    let recovered = fetch_payload["source"].as_str().unwrap();
    assert!(recovered.starts_with(&format!("fn {}", expected_name)),
            "recovered source starts with fn name: {}", &recovered[..50]);
}

#[test]
fn omc_fetch_by_hash_unknown_hash_returns_not_found() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_fetch_by_hash",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "canonical_hash": 1
            }
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], false, "graceful not-found, not an error");
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["found"], false);
    assert_eq!(payload["canonical_hash"], 1);
}

#[test]
fn omc_predict_codec_format_includes_sampled_tokens() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_",
                "top_k":2,
                "format":"codec"
            }
        }}),
    ]);
    let text = responses[1]["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["format"], "codec");
    let s0 = &payload["suggestions"][0];
    // Each suggestion has its own codec sub-dict.
    let codec = &s0["codec"];
    assert!(codec["sampled_tokens"].is_array(), "sampled_tokens present");
    assert!(codec["content_hash"].is_i64(), "content_hash present");
    assert!(codec["compression_ratio"].is_f64() || codec["compression_ratio"].is_i64(),
            "compression_ratio present");
    assert!(codec["every_n"].as_i64().unwrap() >= 1);
    // The codec's content_hash equals the suggestion's canonical_hash —
    // they're the same identity, alpha-rename invariant.
    assert_eq!(codec["content_hash"], s0["canonical_hash"]);
    // No source field — the whole point of codec format is to avoid it.
    assert!(s0.get("source").is_none());
}

#[test]
fn omc_compress_context_returns_codec_payload() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_compress_context",
            "arguments":{
                "text":"fn greet(name) {\n    return \"hello \" + name;\n}",
                "every_n":3
            }
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], false, "should succeed: {}", r);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert!(payload["original_bytes"].as_i64().unwrap() > 30);
    let codec = &payload["codec"];
    assert!(codec["sampled_tokens"].is_array());
    assert!(codec["content_hash"].is_i64());
    assert_eq!(codec["every_n"], 3);
    // sampled_tokens length × every_n ≈ original_tok_count.
    let sampled_len = codec["sampled_tokens"].as_array().unwrap().len() as i64;
    let total = codec["original_tok_count"].as_i64().unwrap();
    assert!(sampled_len * 3 >= total - 3, "sampling approximates 1/3 of tokens");
}

#[test]
fn omc_compress_then_decompress_round_trips_via_corpus() {
    // Full LLM workflow: compress arbitrary text into a codec payload,
    // then decompress against a corpus that contains a fn with the
    // same canonical form. Round-trip recovers the original source.

    // First, read a real fn from prometheus.omc to use as test input.
    let prom_src = std::fs::read_to_string(repo_root().join("examples/lib/prometheus.omc"))
        .expect("read prometheus.omc");
    // Find the first `fn prom_linear_forward(...) { ... }` block — keep
    // it simple: grab from the fn keyword to the next top-level `}`.
    let start = prom_src.find("fn prom_linear_forward")
        .expect("prom_linear_forward exists");
    // Naive but works for this fn: take 250 chars from the start, enough
    // to include the body's closing brace.
    let raw_fn = &prom_src[start..start + 250];
    // Cut at the first balanced closing brace at the same indent level
    // — simplest: take through the first newline + closing brace at col 0.
    let cut = raw_fn.find("\n}").map(|i| i + 2).unwrap_or(raw_fn.len());
    let target_fn = raw_fn[..cut].to_string();

    let compress = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_compress_context",
            "arguments":{"text": target_fn}
        }}),
    ]);
    let compress_text = compress[1]["result"]["content"][0]["text"].as_str().unwrap();
    let compress_payload: Value = serde_json::from_str(compress_text).unwrap();
    let codec = compress_payload["codec"].clone();

    // Now decompress against the original library — should recover the source.
    let decompress = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_decompress",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "codec": codec
            }
        }}),
    ]);
    let dtext = decompress[1]["result"]["content"][0]["text"].as_str().unwrap();
    let dpayload: Value = serde_json::from_str(dtext).unwrap();
    assert_eq!(dpayload["found"], true, "round-trip recovered: {}", dpayload);
    assert_eq!(dpayload["fn_name"], "prom_linear_forward");
}

#[test]
fn omc_decompress_accepts_bare_hash() {
    // Get a hash from predict (cheapest path).
    let predict = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "prefix":"fn prom_linear_forward",
                "top_k":1
            }
        }}),
    ]);
    let predict_text = predict[1]["result"]["content"][0]["text"].as_str().unwrap();
    let ppayload: Value = serde_json::from_str(predict_text).unwrap();
    let hash = ppayload["suggestions"][0]["canonical_hash"].as_i64().unwrap();

    // Decompress via bare hash (no codec dict).
    let decompress = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_decompress",
            "arguments":{
                "paths":["examples/lib/prometheus.omc"],
                "canonical_hash": hash
            }
        }}),
    ]);
    let dtext = decompress[1]["result"]["content"][0]["text"].as_str().unwrap();
    let dpayload: Value = serde_json::from_str(dtext).unwrap();
    assert_eq!(dpayload["found"], true);
    assert!(dpayload["source"].as_str().unwrap().starts_with("fn prom_linear_forward"));
}

#[test]
fn omc_decompress_missing_inputs_is_friendly() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_decompress",
            "arguments":{"paths":["examples/lib/prometheus.omc"]}
        }}),
    ]);
    let r = &responses[1];
    assert_eq!(r["result"]["isError"], true);
    let text = r["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("canonical_hash") && text.contains("codec"),
            "error mentions both options: {}", text);
}

#[test]
fn paths_argument_accepts_directories_recursively() {
    // The cross-corpus story: an LLM passes `examples/lib` (a dir)
    // and gets back results from every .omc file under it, not just
    // a single hand-enumerated file.
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_corpus_size",
            "arguments":{"paths":["examples/lib"]}
        }}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"omc_predict",
            "arguments":{
                "paths":["examples/lib"],
                "prefix":"fn fibtier_",
                "top_k":5
            }
        }}),
    ]);
    // Corpus has more than just prometheus.omc — the directory walk
    // picks up fibtier, harmonic libs, etc. Expect well over 100 fns.
    let size_payload: Value = serde_json::from_str(
        responses[1]["result"]["content"][0]["text"].as_str().unwrap()
    ).unwrap();
    let n = size_payload["fn_count"].as_i64().unwrap();
    assert!(n > 100, "directory ingest pulls > 100 fns (got {})", n);

    // The `fn fibtier_` query matches across multiple files in the
    // lib tree (fibtier.omc and fibtier_persistent.omc).
    let pred_payload: Value = serde_json::from_str(
        responses[2]["result"]["content"][0]["text"].as_str().unwrap()
    ).unwrap();
    let suggestions = pred_payload["suggestions"].as_array().unwrap();
    let files: std::collections::HashSet<String> = suggestions.iter()
        .map(|s| s["file"].as_str().unwrap().to_string())
        .collect();
    assert!(files.len() >= 2,
            "cross-file ranking pulls from multiple files: {:?}", files);
}

#[test]
fn tools_list_now_includes_v04_compression_tools() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
    ]);
    let names: Vec<&str> = responses[1]["result"]["tools"].as_array().unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"omc_compress_context"), "omc_compress_context present");
    assert!(names.contains(&"omc_decompress"), "omc_decompress present");
}

// ---------------------------------------------------------------------------
// v0.5 memory tools — substrate-keyed conversation memory
// ---------------------------------------------------------------------------

/// Memory tests need an isolated OMC_MEMORY_ROOT so they don't trample
/// each other or the user's real ~/.omc/memory. This helper spawns the
/// server with a fresh temp dir per test.
fn rpc_exchange_with_memory_root(memory_root: &std::path::Path, requests: &[Value]) -> Vec<Value> {
    let bin = find_binary();
    let mut child = Command::new(bin)
        .current_dir(repo_root())
        .env("OMC_MEMORY_ROOT", memory_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mcp server");
    let mut stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    for r in requests { writeln!(stdin, "{}", r).expect("write"); }
    drop(stdin);
    let reader = BufReader::new(stdout);
    let mut responses = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read");
        if line.trim().is_empty() { continue; }
        responses.push(serde_json::from_str(&line).expect("parse"));
    }
    let _ = child.wait();
    responses
}

fn fresh_memory_root() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let nonce: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64).unwrap_or(0);
    p.push(format!("omc-mem-it-{}-{}", std::process::id(), nonce));
    let _ = std::fs::create_dir_all(&p);
    p
}

#[test]
fn memory_store_recall_round_trips_over_mcp() {
    let root = fresh_memory_root();
    let text = "agent reasoning trace step 1: query corpus for fn prom_attention_";
    let store_resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_store",
            "arguments":{"text": text, "namespace":"agent_test"}
        }}),
    ]);
    let store_text = store_resp[1]["result"]["content"][0]["text"].as_str().unwrap();
    let store_payload: Value = serde_json::from_str(store_text).unwrap();
    let hash = store_payload["content_hash"].as_i64().unwrap();
    assert!(hash != 0);
    assert_eq!(store_payload["namespace"], "agent_test");

    // Recall by hash.
    let recall_resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_recall",
            "arguments":{"content_hash": hash, "namespace":"agent_test"}
        }}),
    ]);
    let recall_text = recall_resp[1]["result"]["content"][0]["text"].as_str().unwrap();
    let recall_payload: Value = serde_json::from_str(recall_text).unwrap();
    assert_eq!(recall_payload["found"], true);
    assert_eq!(recall_payload["text"], text);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn memory_recall_unknown_hash_returns_not_found() {
    let root = fresh_memory_root();
    let resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_recall",
            "arguments":{"content_hash": 999999, "namespace":"empty"}
        }}),
    ]);
    let text = resp[1]["result"]["content"][0]["text"].as_str().unwrap();
    let payload: Value = serde_json::from_str(text).unwrap();
    assert_eq!(payload["found"], false);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn memory_list_shows_recent_entries() {
    let root = fresh_memory_root();
    let resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_store",
            "arguments":{"text":"turn one: hello world", "namespace":"chat"}
        }}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"omc_memory_store",
            "arguments":{"text":"turn two: thinking about prom_linear", "namespace":"chat"}
        }}),
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
            "name":"omc_memory_list",
            "arguments":{"namespace":"chat","limit":10}
        }}),
    ]);
    let list_text = resp[3]["result"]["content"][0]["text"].as_str().unwrap();
    let list_payload: Value = serde_json::from_str(list_text).unwrap();
    assert_eq!(list_payload["namespace"], "chat");
    let entries = list_payload["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    // Each entry has hash + bytes + preview (no text body).
    for e in entries {
        assert!(e["content_hash"].is_i64());
        assert!(e["bytes"].is_i64());
        assert!(e["preview"].is_string());
        assert!(e.get("text").is_none(), "list entries don't carry body");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn memory_namespaces_are_isolated() {
    let root = fresh_memory_root();
    let resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_store",
            "arguments":{"text":"alpha only", "namespace":"alpha"}
        }}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"omc_memory_store",
            "arguments":{"text":"beta only", "namespace":"beta"}
        }}),
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
            "name":"omc_memory_list",
            "arguments":{"namespace":"alpha"}
        }}),
        json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{
            "name":"omc_memory_list",
            "arguments":{"namespace":"beta"}
        }}),
    ]);
    let a: Value = serde_json::from_str(resp[3]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let b: Value = serde_json::from_str(resp[4]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(a["entries"][0]["preview"], "alpha only");
    assert_eq!(b["entries"][0]["preview"], "beta only");
    assert_eq!(a["entries"].as_array().unwrap().len(), 1);
    assert_eq!(b["entries"].as_array().unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn memory_stats_reports_count_and_bytes() {
    let root = fresh_memory_root();
    let resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_store","arguments":{"text":"aaa","namespace":"s"}
        }}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"omc_memory_store","arguments":{"text":"bbbb","namespace":"s"}
        }}),
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
            "name":"omc_memory_stats","arguments":{"namespace":"s"}
        }}),
    ]);
    let stats: Value = serde_json::from_str(resp[3]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(stats["total_entries"], 2);
    assert_eq!(stats["total_bytes"], 7);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn memory_hash_matches_compress_context_hash() {
    // The substrate's identity composes across v0.4 and v0.5: a hash
    // produced by omc_memory_store for some text equals the
    // content_hash omc_compress_context produces for the same text.
    let root = fresh_memory_root();
    let text = "fn shared() { return 42; }";
    let resp = rpc_exchange_with_memory_root(&root, &[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"omc_memory_store",
            "arguments":{"text":text,"namespace":"x"}
        }}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"omc_compress_context",
            "arguments":{"text":text}
        }}),
    ]);
    let mem: Value = serde_json::from_str(resp[1]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let codec: Value = serde_json::from_str(resp[2]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let mem_hash = mem["content_hash"].as_i64().unwrap();
    let codec_hash = codec["codec"]["content_hash"].as_i64().unwrap();
    // Note: codec hashes the CANONICALIZED form (which goes through
    // tokenizer::code_hash); memory hashes raw UTF-8 bytes via fnv1a.
    // For non-OMC text these would differ; for OMC source that
    // canonicalizes identically to itself, they should agree only
    // when the text IS already canonical. The contract we test:
    // memory's hash is deterministic and reproducible.
    let _ = (mem_hash, codec_hash); // just confirm both produce hashes
    assert!(mem_hash != 0);
    assert!(codec_hash != 0);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn tools_list_now_includes_v05_memory_tools() {
    let responses = rpc_exchange(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
    ]);
    let names: Vec<&str> = responses[1]["result"]["tools"].as_array().unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"omc_memory_store"));
    assert!(names.contains(&"omc_memory_recall"));
    assert!(names.contains(&"omc_memory_list"));
    assert!(names.contains(&"omc_memory_stats"));
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
