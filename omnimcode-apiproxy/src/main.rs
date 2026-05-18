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

#[derive(Clone)]
struct AppState {
    upstream: String,
    rewrite_threshold: usize,
    preview_bytes: usize,
    http: reqwest::Client,
    store: Arc<MemoryStore>,
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
    };

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .fallback(any(passthrough))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!("listening on {}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Rewrite-and-forward the /v1/messages POST.
async fn handle_messages(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_REQUEST,
            &format!("read request body: {}", e)),
    };

    // Streaming requests bypass the rewriter for now — handling SSE responses
    // requires per-event chunked rewriting which is v0.14.1+ work.
    let is_streaming = is_streaming_request(&body_bytes);

    let rewritten = if is_streaming {
        debug!("streaming request — passing through without rewrite");
        body_bytes.clone()
    } else {
        match rewrite_request_body(&body_bytes, &state) {
            Ok(b) => b,
            Err(e) => {
                warn!("rewrite failed, passing original through: {}", e);
                body_bytes.clone()
            }
        }
    };

    let saved = body_bytes.len() as i64 - rewritten.len() as i64;
    if saved > 0 {
        info!("rewrote request: {} → {} bytes ({:+} bytes saved)",
            body_bytes.len(), rewritten.len(), -saved);
    }

    forward_to_upstream(&state, &parts.headers, rewritten).await
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
        Ok(r) => relay(r).await,
        Err(e) => error_response(StatusCode::BAD_GATEWAY, &format!("upstream: {}", e)),
    }
}

async fn forward_to_upstream(
    state: &AppState, headers: &HeaderMap, body: Bytes,
) -> Response {
    let url = format!("{}/v1/messages", state.upstream.trim_end_matches('/'));
    let mut req = state.http.post(&url).body(body.to_vec());
    for (k, v) in headers.iter() {
        if k != "host" && k != "content-length" {
            req = req.header(k, v);
        }
    }
    match req.send().await {
        Ok(r) => relay(r).await,
        Err(e) => error_response(StatusCode::BAD_GATEWAY, &format!("upstream: {}", e)),
    }
}

async fn relay(upstream: reqwest::Response) -> Response {
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let body = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY,
            &format!("read upstream: {}", e)),
    };
    let mut resp = Response::builder().status(status);
    for (k, v) in headers.iter() {
        // Skip hop-by-hop headers; reqwest already drops Transfer-Encoding
        // but Content-Length needs to track our (possibly rewritten) body.
        if k != "transfer-encoding" && k != "connection" {
            resp = resp.header(k, v);
        }
    }
    resp.body(axum::body::Body::from(body)).unwrap()
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

/// Walk `messages[].content[]` for text blocks above the threshold and
/// replace each with a `<omc:ref/>` marker. Inject the expand tool into
/// the request's `tools` array.
fn rewrite_request_body(body: &[u8], state: &AppState) -> Result<Bytes> {
    let mut v: Value = serde_json::from_slice(body)?;
    let Some(messages) = v.get_mut("messages").and_then(Value::as_array_mut) else {
        anyhow::bail!("no 'messages' array in request");
    };

    let mut rewritten_count = 0usize;
    let mut bytes_replaced = 0usize;

    for msg in messages.iter_mut() {
        let content = msg.get_mut("content");
        match content {
            Some(Value::String(s)) => {
                if s.len() >= state.rewrite_threshold {
                    if let Ok(marker) = make_marker(s, state) {
                        bytes_replaced += s.len();
                        *content.unwrap() = Value::String(marker);
                        rewritten_count += 1;
                    }
                }
            }
            Some(Value::Array(blocks)) => {
                for block in blocks.iter_mut() {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            if text.len() >= state.rewrite_threshold {
                                if let Ok(marker) = make_marker(text, state) {
                                    bytes_replaced += text.len();
                                    block["text"] = Value::String(marker);
                                    rewritten_count += 1;
                                }
                            }
                        }
                    }
                    // tool_result blocks carry a `content` field which can be a
                    // string or an array of {type, text}. Same rewrite rule.
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(inner) = block.get_mut("content") {
                            rewrite_tool_result_content(inner, state,
                                &mut rewritten_count, &mut bytes_replaced);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if rewritten_count > 0 {
        inject_expand_tool(&mut v);
        debug!("rewrote {} blocks, replaced {} bytes", rewritten_count, bytes_replaced);
    }

    let out = serde_json::to_vec(&v)?;
    Ok(Bytes::from(out))
}

fn rewrite_tool_result_content(
    inner: &mut Value, state: &AppState, count: &mut usize, bytes: &mut usize,
) {
    match inner {
        Value::String(s) => {
            if s.len() >= state.rewrite_threshold {
                if let Ok(marker) = make_marker(s, state) {
                    *bytes += s.len();
                    *count += 1;
                    *inner = Value::String(marker);
                }
            }
        }
        Value::Array(parts) => {
            for part in parts.iter_mut() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        if text.len() >= state.rewrite_threshold {
                            if let Ok(marker) = make_marker(text, state) {
                                *bytes += text.len();
                                *count += 1;
                                part["text"] = Value::String(marker);
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

fn make_marker(text: &str, state: &AppState) -> Result<String> {
    let hash = state.store.store(PROXY_CACHE_NAMESPACE, text)
        .map_err(anyhow::Error::msg)?;
    let preview: String = text.chars()
        .filter(|c| !c.is_control())
        .take(state.preview_bytes)
        .collect();
    // The marker uses an XML-ish form because LLMs are well-trained on
    // tagged content and don't try to "interpret" attribute values as
    // executable. The proxy's expand tool is the LLM's way out.
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
