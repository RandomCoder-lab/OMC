//! omnimcode-apiproxy — substrate-rewriting reverse proxy for
//! api.anthropic.com.
//!
//! Sits between an MCP client (Claude Code, anything pointing at the
//! Anthropic API) and api.anthropic.com. On each /v1/messages POST it:
//!
//!   1. Parses the request body
//!   2. Walks `messages[].content[]` for text blocks bigger than the
//!      threshold (default 4096 bytes), replaces each one with a
//!      `<omc:ref hash_str="..." preview="..." bytes=N/>` marker. The
//!      original text is cached in the MemoryStore so the marker can
//!      be expanded losslessly on demand.
//!   3. Injects a single `omc_proxy_expand_ref` tool into the request's
//!      `tools` array so the LLM has a way to retrieve any marker's
//!      full content if the preview isn't enough for its reasoning.
//!   4. Forwards the rewritten request to the real upstream
//!   5. Returns the response unmodified (v0.14.0-alpha — response-side
//!      rewriting is a follow-up that requires walking assistant content
//!      and persisting the cache across turns)
//!
//! Hard limits in this MVP:
//!   - No streaming (`stream: true` requests pass through untouched)
//!   - No image / tool_use_block / citation rewriting
//!   - No request batching
//!   - Auth header is forwarded as-is; we never read/log it
//!
//! Honest scope: this saves LLM context tokens to the extent that
//!   (a) prior assistant turns or large text inputs (file pastes,
//!       Read-tool output) re-appear in the user's next turn, AND
//!   (b) the LLM doesn't immediately expand the marker again.
//! For tool-heavy, repetitive sessions: expect 30-60% reduction on the
//! input-token bill. Not 10-50× — that was overpromised in the design
//! conversation.

use anyhow::Result;
use axum::{
    body::Bytes,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{any, post},
    Router,
};
use clap::Parser;
use omnimcode_core::memory::MemoryStore;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, info, warn};

const PROXY_CACHE_NAMESPACE: &str = "_apiproxy_cache";
const EXPAND_TOOL_NAME: &str = "omc_proxy_expand_ref";

#[derive(Parser, Debug, Clone)]
#[command(name = "omnimcode-apiproxy", version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Bind address (default 127.0.0.1:8088 — localhost-only by design,
    /// since this proxy sees the full LLM conversation in cleartext).
    #[arg(long, default_value = "127.0.0.1:8088")]
    bind: String,

    /// Upstream API base URL.
    #[arg(long, default_value = "https://api.anthropic.com")]
    upstream: String,

    /// Threshold above which a text block in a message gets rewritten
    /// to a `<omc:ref/>` marker. Smaller blocks pass through unchanged
    /// because the marker framing (~80 bytes) would cost more than
    /// inlining the original.
    #[arg(long, default_value_t = 4096)]
    rewrite_threshold: usize,

    /// Number of bytes to keep as a human-readable preview alongside the
    /// hash inside the marker. The LLM uses this to decide whether the
    /// preview alone is enough or it needs to expand.
    #[arg(long, default_value_t = 200)]
    preview_bytes: usize,
}

#[derive(Default, Debug, Clone)]
struct RewriteStats {
    requests: u64,
    bytes_in: u64,
    bytes_out: u64,
    blocks_rewritten: u64,
    bytes_saved_messages: u64,
    bytes_saved_tool_result: u64,
    bytes_saved_system: u64,
    bytes_saved_tool_use_input: u64,
}

#[derive(Clone)]
struct AppState {
    upstream: String,
    rewrite_threshold: usize,
    preview_bytes: usize,
    http: reqwest::Client,
    store: Arc<MemoryStore>,
    stats: Arc<std::sync::Mutex<RewriteStats>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "omnimcode_apiproxy=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();
    info!(
        "omnimcode-apiproxy v{} starting — bind={} upstream={} threshold={}B preview={}B",
        env!("CARGO_PKG_VERSION"),
        args.bind, args.upstream, args.rewrite_threshold, args.preview_bytes,
    );
    info!(
        "this proxy sees the full LLM conversation. localhost-only bind unless you change --bind."
    );

    let state = AppState {
        upstream: args.upstream.clone(),
        rewrite_threshold: args.rewrite_threshold,
        preview_bytes: args.preview_bytes,
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?,
        store: Arc::new(MemoryStore::from_env()),
        stats: Arc::new(std::sync::Mutex::new(RewriteStats::default())),
    };

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/_stats", axum::routing::get(stats_endpoint))
        .fallback(any(passthrough))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!("listening on {}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Rewrite-and-forward the /v1/messages POST. After receiving the upstream
/// response, if the assistant emitted a sole tool_use for
/// `omc_proxy_expand_ref`, the proxy resolves it locally from the cache and
/// issues a follow-up upstream request — the client never sees the
/// expand-tool round-trip. Mixed tool_use (expand + other) passes through.
async fn handle_messages(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_REQUEST,
            &format!("read request body: {}", e)),
    };

    let is_streaming = is_streaming_request(&body_bytes);
    let model_name = serde_json::from_slice::<Value>(&body_bytes)
        .ok().and_then(|v| v.get("model").and_then(Value::as_str).map(String::from))
        .unwrap_or_else(|| "?".into());
    info!("/v1/messages received: {} bytes, model={}, streaming={}",
        body_bytes.len(), model_name, is_streaming);

    // The REQUEST body is synchronous JSON even when the response will be streamed.
    // We can always rewrite the body. The streaming flag only affects how the
    // RESPONSE is delivered (SSE chunks). For streaming responses we skip the
    // expand-tool-use interception loop (which requires parsing the full response)
    // and just pass the SSE chunks straight through.
    let rewritten = match rewrite_request_body(&body_bytes, &state) {
        Ok((b, outcome)) => {
            if outcome.any() {
                info!("rewrote request: {} → {} bytes ({:+} bytes saved across {} blocks) | \
                       sys={}B msg={}B tool_result={}B tool_use_input={}B",
                    body_bytes.len(), b.len(), -((body_bytes.len() - b.len()) as i64),
                    outcome.rewritten_count,
                    outcome.bytes_system, outcome.bytes_messages_text,
                    outcome.bytes_tool_result, outcome.bytes_tool_use_input);
            }
            // Update cumulative stats
            {
                let mut s = state.stats.lock().unwrap();
                s.requests += 1;
                s.bytes_in += body_bytes.len() as u64;
                s.bytes_out += b.len() as u64;
                s.blocks_rewritten += outcome.rewritten_count as u64;
                s.bytes_saved_messages += outcome.bytes_messages_text as u64;
                s.bytes_saved_tool_result += outcome.bytes_tool_result as u64;
                s.bytes_saved_system += outcome.bytes_system as u64;
                s.bytes_saved_tool_use_input += outcome.bytes_tool_use_input as u64;
            }
            b
        }
        Err(e) => {
            warn!("rewrite failed, passing original through: {}", e);
            body_bytes.clone()
        }
    };

    let _saved_unused = body_bytes.len() as i64 - rewritten.len() as i64;

    if is_streaming {
        // SSE response: just pass through. The LLM can still emit the expand
        // tool_use in the stream; the client will surface it. We accept this
        // sharp edge in exchange for getting request-side compression on
        // streaming sessions (the common case for Claude Code).
        forward_to_upstream(&state, &parts.headers, rewritten).await
    } else {
        handle_with_expand_loop(&state, &parts.headers, rewritten).await
    }
}

/// Upstream call + expand-tool auto-resolution loop. If the upstream's
/// response contains a sole `tool_use` for `omc_proxy_expand_ref`, look
/// up the hash in the cache, build a follow-up request with the
/// tool_result synthetically appended, and re-call upstream. Bounded to
/// MAX_EXPAND_ROUNDS to prevent runaway loops if the LLM keeps asking
/// to expand.
async fn handle_with_expand_loop(
    state: &AppState, headers: &HeaderMap, initial_body: Bytes,
) -> Response {
    const MAX_EXPAND_ROUNDS: usize = 8;
    let mut current_body = initial_body;
    for round in 0..MAX_EXPAND_ROUNDS {
        // Forward to upstream
        let url = format!("{}/v1/messages",
            state.upstream.trim_end_matches('/'));
        let mut req = state.http.post(&url).body(current_body.to_vec());
        for (k, v) in headers.iter() {
            if k != "host" && k != "content-length" { req = req.header(k, v); }
        }
        let upstream_resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return error_response(StatusCode::BAD_GATEWAY,
                &format!("upstream: {}", e)),
        };
        let status = upstream_resp.status();
        let resp_headers = upstream_resp.headers().clone();
        let resp_body = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => return error_response(StatusCode::BAD_GATEWAY,
                &format!("read upstream: {}", e)),
        };
        // Only intercept successful, parseable responses
        if !status.is_success() {
            return rebuild_response(status, &resp_headers, resp_body);
        }
        let resp_json: Value = match serde_json::from_slice(&resp_body) {
            Ok(v) => v,
            Err(_) => return rebuild_response(status, &resp_headers, resp_body),
        };

        // Look for an exclusive expand tool_use
        let expand_calls = collect_sole_expand_tool_uses(&resp_json);
        if expand_calls.is_empty() {
            return rebuild_response(status, &resp_headers, resp_body);
        }
        info!("round {}: auto-resolving {} expand tool_use(s)",
            round + 1, expand_calls.len());

        // Build follow-up request: previous messages + assistant response
        // (rewritten through marker logic) + new user turn with tool_result
        let mut next_req: Value = match serde_json::from_slice(&current_body) {
            Ok(v) => v,
            Err(_) => return rebuild_response(status, &resp_headers, resp_body),
        };
        let messages = next_req.get_mut("messages")
            .and_then(Value::as_array_mut);
        let Some(messages) = messages else {
            return rebuild_response(status, &resp_headers, resp_body);
        };
        // Append the assistant turn (the upstream's response) verbatim
        if let Some(asst_content) = resp_json.get("content").cloned() {
            messages.push(json!({"role": "assistant", "content": asst_content}));
        }
        // Append a user turn with one tool_result per expand call
        let mut tool_results: Vec<Value> = Vec::new();
        for (tool_use_id, hash_str) in &expand_calls {
            let body_text = lookup_expand(&hash_str, &state).unwrap_or_else(|e|
                format!("[apiproxy: expand cache miss for {}: {}]", hash_str, e));
            tool_results.push(json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": body_text,
            }));
        }
        messages.push(json!({"role": "user", "content": tool_results}));

        current_body = Bytes::from(serde_json::to_vec(&next_req).unwrap());
    }
    warn!("expand loop exceeded {} rounds, returning error", MAX_EXPAND_ROUNDS);
    error_response(StatusCode::BAD_GATEWAY,
        "apiproxy: expand loop limit exceeded")
}

/// If the response's `content` array contains exactly one tool_use AND it
/// is for `omc_proxy_expand_ref`, return its (id, hash_str). Returning
/// multiple results means there were multiple expand calls in a row, which
/// also auto-resolves. Returns empty Vec for mixed tool_use (skip
/// interception, let client handle) or no tool_use at all.
fn collect_sole_expand_tool_uses(resp: &Value) -> Vec<(String, String)> {
    let Some(content) = resp.get("content").and_then(Value::as_array) else {
        return vec![];
    };
    let mut expand = Vec::new();
    let mut has_other_tool_use = false;
    for block in content {
        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
            let name = block.get("name").and_then(Value::as_str).unwrap_or("");
            if name == EXPAND_TOOL_NAME {
                let id = block.get("id").and_then(Value::as_str)
                    .unwrap_or("").to_string();
                let hash = block.get("input")
                    .and_then(|i| i.get("hash_str"))
                    .and_then(Value::as_str).unwrap_or("").to_string();
                if !id.is_empty() && !hash.is_empty() {
                    expand.push((id, hash));
                }
            } else {
                has_other_tool_use = true;
            }
        }
    }
    if has_other_tool_use { vec![] } else { expand }
}

fn lookup_expand(hash_str: &str, state: &AppState) -> Result<String> {
    let hash: i64 = hash_str.parse()
        .map_err(|e| anyhow::anyhow!("hash_str parse: {}", e))?;
    let body = state.store.recall(Some(PROXY_CACHE_NAMESPACE), hash)
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| anyhow::anyhow!("not in cache"))?;
    Ok(body)
}

fn rebuild_response(status: StatusCode, headers: &HeaderMap, body: Bytes) -> Response {
    let mut resp = Response::builder().status(status);
    for (k, v) in headers.iter() {
        if k != "transfer-encoding" && k != "connection" && k != "content-length" {
            resp = resp.header(k, v);
        }
    }
    resp.body(axum::body::Body::from(body)).unwrap()
}

/// Forward anything else (model list, batches, etc.) unmodified.
async fn passthrough(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_REQUEST,
            &format!("read request body: {}", e)),
    };
    let path = parts.uri.path().to_string();
    debug!("passthrough: {} {}", parts.method, path);
    let url = format!("{}{}", state.upstream.trim_end_matches('/'), path);
    let mut req = state.http.request(parts.method, &url).body(body_bytes.to_vec());
    for (k, v) in parts.headers.iter() {
        if k != "host" && k != "content-length" {
            req = req.header(k, v);
        }
    }
    match req.send().await {
        Ok(r) => {
            let status = r.status();
            let h = r.headers().clone();
            match r.bytes().await {
                Ok(b) => rebuild_response(status, &h, b),
                Err(e) => error_response(StatusCode::BAD_GATEWAY,
                    &format!("read upstream: {}", e)),
            }
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY,
            &format!("upstream: {}", e)),
    }
}

/// Used by the streaming-passthrough path in handle_messages and by the
/// catch-all passthrough route. Bytes-in, bytes-out, no rewriting.
async fn forward_to_upstream(
    state: &AppState, headers: &HeaderMap, body: Bytes,
) -> Response {
    let url = format!("{}/v1/messages", state.upstream.trim_end_matches('/'));
    let mut req = state.http.post(&url).body(body.to_vec());
    for (k, v) in headers.iter() {
        if k != "host" && k != "content-length" { req = req.header(k, v); }
    }
    match req.send().await {
        Ok(r) => {
            let status = r.status();
            let h = r.headers().clone();
            match r.bytes().await {
                Ok(b) => rebuild_response(status, &h, b),
                Err(e) => error_response(StatusCode::BAD_GATEWAY,
                    &format!("read upstream: {}", e)),
            }
        }
        Err(e) => error_response(StatusCode::BAD_GATEWAY,
            &format!("upstream: {}", e)),
    }
}

fn error_response(code: StatusCode, msg: &str) -> Response {
    (code, [(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
     json!({"error": {"type": "apiproxy_error", "message": msg}}).to_string())
        .into_response()
}

fn is_streaming_request(body: &[u8]) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(Value::as_bool))
        .unwrap_or(false)
}

/// Per-request rewrite outcome — what was compressed and by how much, broken
/// down by source so the operator can see at a glance whether system prompts,
/// historical tool_results, or LLM tool_use inputs are the dominant savings.
#[derive(Default, Debug)]
struct RewriteOutcome {
    rewritten_count: usize,
    bytes_messages_text: usize,
    bytes_tool_result: usize,
    bytes_system: usize,
    bytes_tool_use_input: usize,
}

impl RewriteOutcome {
    fn total_saved(&self) -> usize {
        self.bytes_messages_text + self.bytes_tool_result
            + self.bytes_system + self.bytes_tool_use_input
    }
    fn any(&self) -> bool { self.rewritten_count > 0 }
}

/// v0.14.3 — live cumulative-stats endpoint. `curl http://localhost:8090/_stats`
async fn stats_endpoint(State(state): State<AppState>) -> Response {
    let s = state.stats.lock().unwrap().clone();
    let ratio = if s.bytes_out > 0 {
        s.bytes_in as f64 / s.bytes_out as f64
    } else { 0.0 };
    let total_saved = s.bytes_saved_messages + s.bytes_saved_tool_result
        + s.bytes_saved_system + s.bytes_saved_tool_use_input;
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "requests_processed": s.requests,
        "bytes_in_total":  s.bytes_in,
        "bytes_out_total": s.bytes_out,
        "bytes_saved_total": total_saved,
        "compression_ratio": ratio,
        "blocks_rewritten": s.blocks_rewritten,
        "bytes_saved_by_source": {
            "messages_text": s.bytes_saved_messages,
            "tool_result": s.bytes_saved_tool_result,
            "system_prompt": s.bytes_saved_system,
            "tool_use_input": s.bytes_saved_tool_use_input,
        }
    })).unwrap();
    (StatusCode::OK,
     [(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
     json).into_response()
}

/// Walk the request body and rewrite every eligible large block.
///
/// What gets rewritten (each independently):
///   - `messages[].content` — string form or array-of-blocks form, except
///     the LAST user message (kept intact so the LLM sees the current ask)
///   - `messages[].content[]` of type `tool_result` — the `content` field
///   - `messages[].content[]` of type `tool_use` — the JSON-serialized
///     `input` field when its serialized form exceeds threshold; this
///     catches the LLM's own large tool arguments (e.g., Write file content)
///   - `system` (top-level): if a string, rewrites it as a single block; if
///     an array, walks each `{type: "text", text: ...}` element. Critically
///     PRESERVES the `cache_control` field on each element so Anthropic's
///     prompt-cache layer still works on the rewritten form.
///
/// Safety rule: the LAST user message is never rewritten — that's the
/// user's current intent.
fn rewrite_request_body(body: &[u8], state: &AppState) -> Result<(Bytes, RewriteOutcome)> {
    let mut v: Value = serde_json::from_slice(body)?;
    let mut out = RewriteOutcome::default();

    // ---- system prompt (top-level field) ----
    if let Some(system) = v.get_mut("system") {
        match system {
            Value::String(s) => {
                if s.len() >= state.rewrite_threshold {
                    if let Ok(marker) = make_marker(s, state) {
                        out.bytes_system += s.len();
                        out.rewritten_count += 1;
                        *system = Value::String(marker);
                    }
                }
            }
            Value::Array(blocks) => {
                for block in blocks.iter_mut() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        let Some(text) = block.get("text").and_then(Value::as_str) else { continue };
                        if text.len() < state.rewrite_threshold { continue; }
                        let Ok(marker) = make_marker(text, state) else { continue };
                        out.bytes_system += text.len();
                        out.rewritten_count += 1;
                        // Mutate ONLY the `text` field; preserve cache_control + everything else
                        block["text"] = Value::String(marker);
                    }
                }
            }
            _ => {}
        }
    }

    // ---- messages array ----
    let Some(messages) = v.get_mut("messages").and_then(Value::as_array_mut) else {
        // No messages? Just system rewriting may have happened — return what we have.
        let bytes = Bytes::from(serde_json::to_vec(&v)?);
        return Ok((bytes, out));
    };
    let last_user_idx = messages.iter().enumerate().rev()
        .find(|(_, m)| m.get("role").and_then(Value::as_str) == Some("user"))
        .map(|(i, _)| i);

    for (idx, msg) in messages.iter_mut().enumerate() {
        if Some(idx) == last_user_idx { continue; }
        let Some(content) = msg.get_mut("content") else { continue };
        match content {
            Value::String(s) => {
                if s.len() >= state.rewrite_threshold {
                    if let Ok(marker) = make_marker(s, state) {
                        out.bytes_messages_text += s.len();
                        out.rewritten_count += 1;
                        *content = Value::String(marker);
                    }
                }
            }
            Value::Array(blocks) => {
                for block in blocks.iter_mut() {
                    let block_type = block.get("type").and_then(Value::as_str).unwrap_or("");
                    match block_type {
                        "text" => {
                            let Some(text) = block.get("text").and_then(Value::as_str) else { continue };
                            if text.len() < state.rewrite_threshold { continue; }
                            let Ok(marker) = make_marker(text, state) else { continue };
                            out.bytes_messages_text += text.len();
                            out.rewritten_count += 1;
                            block["text"] = Value::String(marker);
                        }
                        "tool_result" => {
                            if let Some(inner) = block.get_mut("content") {
                                rewrite_tool_result_content(inner, state, &mut out);
                            }
                        }
                        "tool_use" => {
                            // Compress big string values INSIDE the input dict.
                            // Crucially, preserve the original key names so the
                            // LLM doesn't see (and thus copy) a fake field name
                            // when generating fresh tool calls in later turns.
                            if let Some(input) = block.get_mut("input") {
                                rewrite_strings_recursive(input, state, &mut out);
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if out.any() {
        inject_expand_tool(&mut v);
    }
    let bytes = Bytes::from(serde_json::to_vec(&v)?);
    Ok((bytes, out))
}

/// v0.14.4 — walk a JSON value and replace any large STRING values in place,
/// preserving all key names so the LLM doesn't see (and copy) a fake field
/// name when generating new tool calls. Used for `tool_use.input`.
///
/// Two layers of value-rewriting:
///   1. A top-level string longer than threshold → marker.
///   2. Any string FIELD inside an object whose value exceeds threshold →
///      marker (e.g. `{"content": "...big..."} → {"content": "<omc:ref ...>"}`).
///   3. Array elements that are strings → same rule, in place.
fn rewrite_strings_recursive(
    val: &mut Value, state: &AppState, out: &mut RewriteOutcome,
) {
    match val {
        Value::String(s) => {
            if s.len() >= state.rewrite_threshold {
                if let Ok(marker) = make_marker(s, state) {
                    out.bytes_tool_use_input += s.len();
                    out.rewritten_count += 1;
                    *val = Value::String(marker);
                }
            }
        }
        Value::Object(map) => {
            for (_k, v) in map.iter_mut() {
                rewrite_strings_recursive(v, state, out);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                rewrite_strings_recursive(v, state, out);
            }
        }
        _ => {}
    }
}

fn rewrite_tool_result_content(
    inner: &mut Value, state: &AppState, out: &mut RewriteOutcome,
) {
    match inner {
        Value::String(s) => {
            if s.len() >= state.rewrite_threshold {
                if let Ok(marker) = make_marker(s, state) {
                    out.bytes_tool_result += s.len();
                    out.rewritten_count += 1;
                    *inner = Value::String(marker);
                }
            }
        }
        Value::Array(parts) => {
            for part in parts.iter_mut() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    let Some(text) = part.get("text").and_then(Value::as_str) else { continue };
                    if text.len() < state.rewrite_threshold { continue; }
                    let Ok(marker) = make_marker(text, state) else { continue };
                    out.bytes_tool_result += text.len();
                    out.rewritten_count += 1;
                    part["text"] = Value::String(marker);
                }
            }
        }
        _ => {}
    }
}

fn make_marker(text: &str, state: &AppState) -> Result<String> {
    let hash = state.store.store(PROXY_CACHE_NAMESPACE, text)
        .map_err(anyhow::Error::msg)?;
    // For very large blocks the LLM almost certainly wants either:
    // (a) the full content (expand via tool), or (b) to move on.
    // The preview adds no decision-quality. Drop it past 8 KB.
    if text.len() >= 8192 {
        return Ok(format!("<omc:ref h=\"{}\" b=\"{}\"/>", hash, text.len()));
    }
    let preview: String = text.chars()
        .filter(|c| !c.is_control())
        .take(state.preview_bytes)
        .collect();
    Ok(format!(
        "<omc:ref hash_str=\"{}\" bytes=\"{}\" preview={:?}/>",
        hash, text.len(), preview
    ))
}

/// Add the omc_proxy_expand_ref tool to the request's tools array so the
/// LLM has a way to retrieve full bytes for any marker it cares about.
fn inject_expand_tool(req: &mut Value) {
    let tool = json!({
        "name": EXPAND_TOOL_NAME,
        "description": "Expand an <omc:ref/> marker back to its full text. \
                        The proxy replaced large content blocks in your context \
                        with these markers to save tokens. Call this ONLY when \
                        the preview isn't enough for your reasoning; in most \
                        cases the preview is sufficient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hash_str": {
                    "type": "string",
                    "description": "The hash_str attribute from the <omc:ref/> marker."
                }
            },
            "required": ["hash_str"]
        }
    });
    match req.get_mut("tools") {
        Some(Value::Array(tools)) => {
            // Don't double-inject if a previous turn already added it.
            let exists = tools.iter().any(|t|
                t.get("name").and_then(Value::as_str) == Some(EXPAND_TOOL_NAME));
            if !exists { tools.push(tool); }
        }
        _ => {
            req["tools"] = Value::Array(vec![tool]);
        }
    }
}
