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
        _ => Err(format!("Unknown tool: {}", name)),
    }
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
