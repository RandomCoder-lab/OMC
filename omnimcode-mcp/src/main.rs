//! MCP server for OMC.
//!
//! Implements just enough of the Model Context Protocol over stdio
//! JSON-RPC for an LLM client (Claude Desktop, Cursor, etc.) to:
//!   - eval OMC code
//!   - introspect the builtin surface (help / list / categories)
//!   - explain runtime errors against the curated catalog
//!   - enumerate OMC-unique primitives so the LLM knows what's
//!     worth reaching for OMC instead of NumPy
//!
//! Protocol: line-delimited JSON-RPC 2.0 over stdin/stdout. The
//! handshake (initialize → initialized notification → tools/list →
//! tools/call) follows MCP. We keep the surface minimal — no
//! resources, no prompts, no sampling, just tools.
//!
//! Configure in Claude Desktop:
//!   {
//!     "mcpServers": {
//!       "omc": { "command": "/path/to/omnimcode-mcp" }
//!     }
//!   }

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use std::io::{self, BufRead, Write};

use omnimcode_core::docs;
use omnimcode_core::errors;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use omnimcode_core::predict::{CodeCorpus, predict_continuations};
use omnimcode_core::value::Value;

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    id: Option<Json>,
    method: String,
    #[serde(default)]
    params: Json,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Json,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut interp = Interpreter::new();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req): Result<RpcRequest, _> = serde_json::from_str(&line) else {
            // Garbage on the line — skip it. MCP clients sometimes
            // send junk during startup.
            continue;
        };
        if req.jsonrpc != "2.0" {
            continue;
        }
        // Notifications (no id field) don't get a response.
        let Some(id) = req.id.clone() else {
            // initialized, etc. — acknowledge implicitly.
            continue;
        };

        let response = handle(&mut interp, &req.method, &req.params, id);
        let s = serde_json::to_string(&response).unwrap();
        let _ = writeln!(out, "{}", s);
        let _ = out.flush();
    }
}

fn handle(interp: &mut Interpreter, method: &str, params: &Json, id: Json) -> RpcResponse {
    match method {
        "initialize" => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "omnimcode-mcp",
                    "version": "1.0.0"
                }
            })),
            error: None,
        },
        "tools/list" => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "tools": list_tools() })),
            error: None,
        },
        "tools/call" => {
            let name = params.get("name").and_then(Json::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match dispatch_tool(interp, name, &args) {
                Ok(text) => RpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{ "type": "text", "text": text }],
                        "isError": false
                    })),
                    error: None,
                },
                Err(msg) => RpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{ "type": "text", "text": msg }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        }
        _ => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code: -32601,
                message: format!("Method not found: {}", method),
            }),
        },
    }
}

/// Tool catalog exposed to MCP clients. Keep descriptions punchy —
/// the LLM uses them to decide which tool to call.
fn list_tools() -> Vec<Json> {
    vec![
        json!({
            "name": "omc_eval",
            "description": "Evaluate OMC source code and return stdout. Use this to run OMC programs, test snippets, or compute results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "OMC source code to evaluate." }
                },
                "required": ["code"]
            }
        }),
        json!({
            "name": "omc_help",
            "description": "Look up signature + description + example for an OMC builtin. Returns 'did you mean' suggestions on miss.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Builtin name, e.g. arr_softmax" }
                },
                "required": ["name"]
            }
        }),
        json!({
            "name": "omc_list_builtins",
            "description": "List all documented OMC builtins, optionally filtered by category (substrate, ml_kernels, autograd, generators, ...).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": { "type": "string", "description": "Optional category filter." }
                }
            }
        }),
        json!({
            "name": "omc_categories",
            "description": "List all builtin categories. Use this before omc_list_builtins to see what's available.",
            "inputSchema": { "type": "object" }
        }),
        json!({
            "name": "omc_unique_builtins",
            "description": "List OMC-unique builtins with NO Python/NumPy equivalent. These are the reason to reach for OMC over numpy: substrate-aware primitives, harmonic ops, native lazy generators.",
            "inputSchema": { "type": "object" }
        }),
        json!({
            "name": "omc_explain_error",
            "description": "Given an OMC error message, return a structured explanation: what it means, typical cause, one-line fix.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The OMC error message." }
                },
                "required": ["message"]
            }
        }),
        json!({
            "name": "omc_did_you_mean",
            "description": "Closest known builtin names for a typo. Useful when you've guessed a name that doesn't exist.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The (probably wrong) name." }
                },
                "required": ["name"]
            }
        }),
        json!({
            "name": "omc_predict",
            "description": "Substrate-indexed code completion. Given a partial OMC code prefix \
                            (e.g. `fn prom_linear_`), return the top-k ranked continuations from \
                            a content-addressed corpus of OMC files. Each result is a viable \
                            branch.\n\
                            \n\
                            The `format` arg controls how much context each suggestion costs:\n\
                            - `hash` (default, ~50 bytes/suggestion): fn_name + file + \
                              canonical_hash + substrate_distance. Use this for browsing — \
                              cheap context. Fetch the body on demand with omc_fetch_by_hash.\n\
                            - `signature` (~100 bytes/suggestion): adds the fn signature line. \
                              Enough for an LLM to know the call shape.\n\
                            - `full`: includes the complete source. Use only when you'll \
                              actually edit/adapt the body.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths to ingest. Top-level fns from each file are added to the corpus."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Partial OMC source (e.g. `fn prom_linear_`). May be incomplete."
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "description": "Number of ranked continuations to return."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["hash", "signature", "full"],
                        "default": "hash",
                        "description": "Response detail level. See tool description."
                    }
                },
                "required": ["paths", "prefix"]
            }
        }),
        json!({
            "name": "omc_corpus_size",
            "description": "Diagnostic: report how many top-level fns are ingested across a list \
                            of OMC source paths. Useful for verifying paths resolve before \
                            building a larger predict query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths to ingest."
                    }
                },
                "required": ["paths"]
            }
        }),
        json!({
            "name": "omc_fetch_by_hash",
            "description": "Recover a function body by its canonical hash. The companion to \
                            omc_predict with format=hash: the LLM browses cheaply via hash \
                            digests, then fetches the actual source only when ready to use \
                            it. Walks the same paths corpus as omc_predict; returns the full \
                            source of the matching fn, or notFound:true if no fn in the \
                            corpus has that hash.\n\
                            \n\
                            The canonical_hash is alpha-rename invariant — a fn that's been \
                            renamed still recovers from the same hash.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths to search."
                    },
                    "canonical_hash": {
                        "type": "integer",
                        "description": "The canonical_hash returned by a previous omc_predict call."
                    }
                },
                "required": ["paths", "canonical_hash"]
            }
        }),
    ]
}

fn dispatch_tool(interp: &mut Interpreter, name: &str, args: &Json) -> Result<String, String> {
    match name {
        "omc_eval" => {
            let code = args.get("code").and_then(Json::as_str)
                .ok_or_else(|| "omc_eval: missing 'code' arg".to_string())?;
            eval_program(interp, code)
        }
        "omc_help" => {
            let name = args.get("name").and_then(Json::as_str)
                .ok_or_else(|| "omc_help: missing 'name' arg".to_string())?;
            match docs::lookup(name) {
                Some(d) => Ok(serde_json::to_string_pretty(&json!({
                    "name": d.name,
                    "category": d.category,
                    "signature": d.signature,
                    "description": d.description,
                    "example": d.example,
                    "unique_to_omc": d.unique_to_omc
                })).unwrap()),
                None => {
                    let suggestions = docs::did_you_mean(name, 5);
                    Ok(serde_json::to_string_pretty(&json!({
                        "found": false,
                        "name": name,
                        "did_you_mean": suggestions
                    })).unwrap())
                }
            }
        }
        "omc_list_builtins" => {
            let cat = args.get("category").and_then(Json::as_str);
            let names = docs::names_in(cat);
            Ok(serde_json::to_string_pretty(&json!(names)).unwrap())
        }
        "omc_categories" => {
            let cats = docs::categories();
            Ok(serde_json::to_string_pretty(&json!(cats)).unwrap())
        }
        "omc_unique_builtins" => {
            let names: Vec<&str> = docs::BUILTINS.iter()
                .filter(|b| b.unique_to_omc)
                .map(|b| b.name)
                .collect();
            Ok(serde_json::to_string_pretty(&json!(names)).unwrap())
        }
        "omc_explain_error" => {
            let msg = args.get("message").and_then(Json::as_str)
                .ok_or_else(|| "omc_explain_error: missing 'message' arg".to_string())?;
            match errors::match_error(msg) {
                Some(p) => Ok(serde_json::to_string_pretty(&json!({
                    "matched": true,
                    "pattern": p.pattern,
                    "category": p.category,
                    "explanation": p.explanation,
                    "typical_cause": p.typical_cause,
                    "fix": p.fix
                })).unwrap()),
                None => Ok(serde_json::to_string_pretty(&json!({
                    "matched": false,
                    "explanation": "No catalog pattern matched. Try `omc_did_you_mean` if it looks like a typo."
                })).unwrap()),
            }
        }
        "omc_did_you_mean" => {
            let name = args.get("name").and_then(Json::as_str)
                .ok_or_else(|| "omc_did_you_mean: missing 'name' arg".to_string())?;
            let suggestions = docs::did_you_mean(name, 5);
            Ok(serde_json::to_string_pretty(&json!(suggestions)).unwrap())
        }
        "omc_predict" => {
            let paths = parse_paths_arg(args, "omc_predict")?;
            let prefix = args.get("prefix").and_then(Json::as_str)
                .ok_or_else(|| "omc_predict: missing 'prefix' arg".to_string())?;
            // top_k optional, defaults to 5. Clamp to [1, 50] so a
            // misconfigured client can't ask for the entire corpus.
            let top_k = args.get("top_k").and_then(Json::as_i64)
                .unwrap_or(5)
                .clamp(1, 50) as usize;
            let format = args.get("format")
                .and_then(Json::as_str)
                .unwrap_or("hash");
            let corpus = build_corpus(&paths)?;
            let suggestions = predict_continuations(&corpus, prefix, top_k);
            let suggestion_jsons: Vec<Json> = suggestions.iter()
                .map(|s| project_suggestion(s, format))
                .collect();
            let payload = json!({
                "prefix": prefix,
                "corpus_size": corpus.len(),
                "top_k": top_k,
                "format": format,
                "suggestions": suggestion_jsons,
            });
            Ok(serde_json::to_string_pretty(&payload).unwrap())
        }
        "omc_corpus_size" => {
            let paths = parse_paths_arg(args, "omc_corpus_size")?;
            let corpus = build_corpus(&paths)?;
            let payload = json!({
                "paths": paths,
                "fn_count": corpus.len(),
            });
            Ok(serde_json::to_string_pretty(&payload).unwrap())
        }
        "omc_fetch_by_hash" => {
            let paths = parse_paths_arg(args, "omc_fetch_by_hash")?;
            let target = args.get("canonical_hash").and_then(Json::as_i64)
                .ok_or_else(|| "omc_fetch_by_hash: missing 'canonical_hash' (i64) arg".to_string())?;
            let corpus = build_corpus(&paths)?;
            match corpus.entries.iter().find(|e| e.canonical_hash == target) {
                Some(entry) => {
                    let payload = json!({
                        "found": true,
                        "canonical_hash": entry.canonical_hash,
                        "fn_name": entry.fn_name,
                        "file": entry.file,
                        "source": entry.source,
                    });
                    Ok(serde_json::to_string_pretty(&payload).unwrap())
                }
                None => {
                    let payload = json!({
                        "found": false,
                        "canonical_hash": target,
                        "searched_paths": paths,
                        "corpus_size": corpus.len(),
                    });
                    Ok(serde_json::to_string_pretty(&payload).unwrap())
                }
            }
        }
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

/// Compact one Suggestion into the requested response format.
///
/// - `hash` (~50 bytes): identity only. The LLM uses it to remember a
///   match it might fetch later via omc_fetch_by_hash.
/// - `signature` (~100 bytes): adds the fn signature line so the LLM
///   knows the call shape without paying for the body.
/// - `full`: everything including the body. Use when the LLM intends
///   to read or adapt the implementation.
///
/// `prefix_match_len` and `substrate_distance` are included at every
/// level — they're the ranking explanation and cost essentially nothing.
fn project_suggestion(s: &omnimcode_core::predict::Suggestion, format: &str) -> Json {
    match format {
        "full" => json!({
            "fn_name": s.fn_name,
            "source": s.source,
            "file": s.file,
            "canonical_hash": s.canonical_hash,
            "attractor": s.attractor,
            "prefix_match_len": s.prefix_match_len,
            "substrate_distance": s.substrate_distance,
            "query_attractor": s.query_attractor,
        }),
        "signature" => json!({
            "fn_name": s.fn_name,
            "signature": extract_signature(&s.source),
            "file": s.file,
            "canonical_hash": s.canonical_hash,
            "prefix_match_len": s.prefix_match_len,
            "substrate_distance": s.substrate_distance,
        }),
        // "hash" is the default and the most compressed form.
        _ => json!({
            "fn_name": s.fn_name,
            "file": s.file,
            "canonical_hash": s.canonical_hash,
            "prefix_match_len": s.prefix_match_len,
            "substrate_distance": s.substrate_distance,
        }),
    }
}

/// Extract the function signature line from a fn body's source. The
/// signature is everything from `fn` through the closing paren of the
/// argument list, plus any `-> ReturnType` annotation. Stops at the
/// opening `{` of the body.
///
/// Robust to multi-line signatures (joins lines, collapses whitespace).
fn extract_signature(source: &str) -> String {
    // Join everything before the first `{` then collapse whitespace.
    let head = source.split_once('{').map(|(h, _)| h).unwrap_or(source);
    let cleaned: String = head.split_whitespace().collect::<Vec<_>>().join(" ");
    cleaned.trim().to_string()
}

/// Extract a `paths` array argument from a tool's JSON args. Used by
/// both omc_predict and omc_corpus_size — same shape, same validation.
fn parse_paths_arg(args: &Json, tool: &str) -> Result<Vec<String>, String> {
    let paths_val = args.get("paths")
        .ok_or_else(|| format!("{}: missing 'paths' arg", tool))?;
    let arr = paths_val.as_array()
        .ok_or_else(|| format!("{}: 'paths' must be an array of strings", tool))?;
    arr.iter()
        .map(|v| v.as_str()
            .ok_or_else(|| format!("{}: every 'paths' entry must be a string", tool))
            .map(|s| s.to_string()))
        .collect()
}

/// Build a CodeCorpus by reading + ingesting every file in `paths`.
/// Surface I/O errors as MCP-style strings so the client sees a clean
/// `isError: true` text instead of a panic.
fn build_corpus(paths: &[String]) -> Result<CodeCorpus, String> {
    let mut corpus = CodeCorpus::new();
    for path in paths {
        let src = std::fs::read_to_string(path)
            .map_err(|e| format!("omc_predict: read {}: {}", path, e))?;
        corpus.ingest_file(path, &src);
    }
    Ok(corpus)
}

/// Evaluate an OMC program. Errors come back as structured strings
/// (the MCP client sees isError=true alongside the text). Each
/// tools/call uses a fresh interpreter to avoid state bleed.
///
/// Returns the display string of the final statement's value, or
/// "null" if the program ends on a non-expression. This matches the
/// REPL convention LLMs expect when iterating quickly.
fn eval_program(_interp: &mut Interpreter, code: &str) -> Result<String, String> {
    let mut parser = Parser::new(code);
    let stmts = parser.parse()
        .map_err(|e| format!("parse error: {}", e))?;
    // Fresh Interpreter per call: keeps the MCP server stateless,
    // which is what most LLM clients expect. Tooling can layer
    // session state on top if needed.
    let mut fresh = Interpreter::new();
    fresh.execute(stmts).map_err(|e| format!("runtime error: {}", e))?;
    // Prefer the last top-level expression value, then fall back to
    // any function-level return value (e.g. `return 42;` at top level).
    let v = fresh.take_last_expression_value()
        .or_else(|| fresh.take_return_value());
    Ok(match v {
        Some(v) => display_value(&v),
        None => "null".to_string(),
    })
}

fn display_value(v: &Value) -> String {
    // Compact, LLM-friendly rendering. HInt shows value + substrate
    // metadata so the LLM sees the resonance/HIM that distinguishes
    // OMC from numpy. Arrays unwrap their RefCell wrapper visually
    // — the inner Debug format leaks Rust internals that aren't useful.
    match v {
        Value::HInt(h) => format!(
            "HInt {{ value: {}, resonance: {:.3}, him: {:.3} }}",
            h.value, h.resonance, h.him_score
        ),
        Value::HFloat(f) => format!("{}", f),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => format!("{}", b),
        Value::Null => "null".to_string(),
        Value::Array(arr) => {
            let items = arr.items.borrow();
            let parts: Vec<String> = items.iter().map(display_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Dict(d) => {
            let d = d.borrow();
            let parts: Vec<String> = d.iter()
                .map(|(k, v)| format!("\"{}\": {}", k, display_value(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Function { name, .. } => format!("<fn {}>", name),
        _ => format!("{:?}", v),
    }
}
