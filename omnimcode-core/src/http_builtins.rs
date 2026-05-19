//! Native HTTP builtins: `http_get`, `http_post`, `http_post_json`, `http_put`, `http_delete`.
//!
//! These builtins give OMC programs direct access to HTTP without needing LLM
//! credentials.  They are built on the same `ureq` crate already used by
//! `llm_builtins` and are gated behind the same `native-llm` Cargo feature
//! (which controls `dep:ureq`).
//!
//! ## Return value
//!
//! Every function returns a dict with at least:
//! - `status`  — HTTP status code as int
//! - `body`    — response body as string
//! - `ok`      — bool, true when 200 <= status < 300
//!
//! `http_post_json` additionally includes:
//! - `json`    — parsed JSON body as OMC value, or null on parse failure

use crate::value::Value;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Public entry points (called from interpreter.rs dispatch)
// ---------------------------------------------------------------------------

/// `http_get(url: string, headers?: dict) -> dict`
#[cfg(feature = "native-llm")]
pub fn http_get(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("http_get requires (url: string, headers?: dict)".to_string());
    }
    let url = args[0].to_display_string();
    let headers = extract_headers(args.get(1))?;

    let mut req = ureq::get(&url);
    for (k, v) in &headers {
        req = req.set(k, v);
    }

    let (status, body) = send_request(req)?;
    Ok(make_response_dict(status, body))
}

/// `http_post(url: string, body: string, headers?: dict) -> dict`
#[cfg(feature = "native-llm")]
pub fn http_post(args: &[Value]) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("http_post requires (url: string, body: string, headers?: dict)".to_string());
    }
    let url = args[0].to_display_string();
    let body_str = args[1].to_display_string();
    let headers = extract_headers(args.get(2))?;

    let mut req = ureq::post(&url);
    for (k, v) in &headers {
        req = req.set(k, v);
    }

    let resp = req
        .send_string(&body_str)
        .map_err(|e| format!("http_post failed: {e}"))?;
    let status = resp.status();
    let body = resp
        .into_string()
        .map_err(|e| format!("http_post: read body failed: {e}"))?;
    Ok(make_response_dict(status, body))
}

/// `http_post_json(url: string, data: dict|array, headers?: dict) -> dict`
///
/// Serialises `data` to JSON, sends with `Content-Type: application/json`,
/// and additionally attempts to parse the response body as JSON, returning it
/// under the `json` key (null on parse failure).
#[cfg(feature = "native-llm")]
pub fn http_post_json(args: &[Value]) -> Result<Value, String> {
    if args.len() < 2 {
        return Err(
            "http_post_json requires (url: string, data: dict|array, headers?: dict)".to_string(),
        );
    }
    let url = args[0].to_display_string();
    let json_body = crate::interpreter::value_to_json(&args[1]);
    let json_str = serde_json::to_string(&json_body)
        .map_err(|e| format!("http_post_json: JSON serialisation failed: {e}"))?;
    let headers = extract_headers(args.get(2))?;

    let mut req = ureq::post(&url).set("Content-Type", "application/json");
    for (k, v) in &headers {
        req = req.set(k, v);
    }

    let resp = req
        .send_string(&json_str)
        .map_err(|e| format!("http_post_json failed: {e}"))?;
    let status = resp.status();
    let body = resp
        .into_string()
        .map_err(|e| format!("http_post_json: read body failed: {e}"))?;

    Ok(make_json_response_dict(status, body))
}

/// `http_put(url: string, body: string, headers?: dict) -> dict`
#[cfg(feature = "native-llm")]
pub fn http_put(args: &[Value]) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("http_put requires (url: string, body: string, headers?: dict)".to_string());
    }
    let url = args[0].to_display_string();
    let body_str = args[1].to_display_string();
    let headers = extract_headers(args.get(2))?;

    let mut req = ureq::put(&url);
    for (k, v) in &headers {
        req = req.set(k, v);
    }

    let resp = req
        .send_string(&body_str)
        .map_err(|e| format!("http_put failed: {e}"))?;
    let status = resp.status();
    let body = resp
        .into_string()
        .map_err(|e| format!("http_put: read body failed: {e}"))?;
    Ok(make_response_dict(status, body))
}

/// `http_delete(url: string, headers?: dict) -> dict`
#[cfg(feature = "native-llm")]
pub fn http_delete(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("http_delete requires (url: string, headers?: dict)".to_string());
    }
    let url = args[0].to_display_string();
    let headers = extract_headers(args.get(1))?;

    let mut req = ureq::delete(&url);
    for (k, v) in &headers {
        req = req.set(k, v);
    }

    let (status, body) = send_request(req)?;
    Ok(make_response_dict(status, body))
}

// ---------------------------------------------------------------------------
// Stubs for non-native builds
// ---------------------------------------------------------------------------

#[cfg(not(feature = "native-llm"))]
pub fn http_get(_args: &[Value]) -> Result<Value, String> {
    Err("http_get: recompile with --features native-llm".to_string())
}

#[cfg(not(feature = "native-llm"))]
pub fn http_post(_args: &[Value]) -> Result<Value, String> {
    Err("http_post: recompile with --features native-llm".to_string())
}

#[cfg(not(feature = "native-llm"))]
pub fn http_post_json(_args: &[Value]) -> Result<Value, String> {
    Err("http_post_json: recompile with --features native-llm".to_string())
}

#[cfg(not(feature = "native-llm"))]
pub fn http_put(_args: &[Value]) -> Result<Value, String> {
    Err("http_put: recompile with --features native-llm".to_string())
}

#[cfg(not(feature = "native-llm"))]
pub fn http_delete(_args: &[Value]) -> Result<Value, String> {
    Err("http_delete: recompile with --features native-llm".to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract optional header dict (Value::Dict) into a Vec<(String, String)>.
/// Accepts null / missing arg gracefully.
fn extract_headers(v: Option<&Value>) -> Result<Vec<(String, String)>, String> {
    match v {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Dict(d)) => {
            let map = d.borrow();
            let mut out = Vec::with_capacity(map.len());
            for (k, val) in map.iter() {
                out.push((k.clone(), val.to_display_string()));
            }
            Ok(out)
        }
        Some(other) => Err(format!(
            "http headers must be a dict or null, got {}",
            other.to_display_string()
        )),
    }
}

/// Fire a GET/DELETE-style request and return (status, body).
#[cfg(feature = "native-llm")]
fn send_request(req: ureq::Request) -> Result<(u16, String), String> {
    let resp = req.call().map_err(|e| format!("HTTP request failed: {e}"))?;
    let status = resp.status();
    let body = resp
        .into_string()
        .map_err(|e| format!("read body failed: {e}"))?;
    Ok((status, body))
}

/// Build the standard {status, body, ok} response dict.
fn make_response_dict(status: u16, body: String) -> Value {
    let mut map = BTreeMap::new();
    map.insert("status".to_string(), Value::HInt(crate::value::HInt::new(status as i64)));
    map.insert("body".to_string(), Value::String(body));
    map.insert("ok".to_string(), Value::Bool(status >= 200 && status < 300));
    Value::dict_from(map)
}

/// Build the {status, body, ok, json} response dict used by http_post_json.
fn make_json_response_dict(status: u16, body: String) -> Value {
    let parsed_json = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .map(crate::interpreter::json_to_value)
        .unwrap_or(Value::Null);

    let mut map = BTreeMap::new();
    map.insert("status".to_string(), Value::HInt(crate::value::HInt::new(status as i64)));
    map.insert("body".to_string(), Value::String(body));
    map.insert("ok".to_string(), Value::Bool(status >= 200 && status < 300));
    map.insert("json".to_string(), parsed_json);
    Value::dict_from(map)
}
