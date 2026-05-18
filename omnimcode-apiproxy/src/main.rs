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
//!   - Response-side rewriting skipped for streaming sessions (the LLM can
//!     still expand markers in a streamed response via the tool call path)
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
const EXPAND_TOOL_NAME:   &str = "omc_proxy_expand_ref";
const REMEMBER_TOOL_NAME:  &str = "omc_proxy_remember";
const RECALL_TOOL_NAME:    &str = "omc_proxy_recall";
const LIST_REFS_TOOL_NAME: &str = "omc_proxy_list_refs";
const FORWARD_TOOL_NAME:   &str = "omc_proxy_forward";
const DIFF_TOOL_NAME:      &str = "omc_proxy_diff";

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

    /// Minimum content size (bytes) for a block to be eligible for delta
    /// (OMCD/Axis-5) storage. Blocks shorter than this are always stored
    /// plain — no diff attempted. Lower = more delta attempts, more disk I/O.
    #[arg(long, default_value_t = 1024)]
    delta_min_bytes: usize,
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
    bytes_saved_tool_definitions: u64,
    cache_control_inserted: u64,
    conversation_count: u64,
    delta_stores_attempted: u64,
    /// Streaming requests pass through rewritten (request side) but
    /// the response is piped directly rather than buffered.
    streaming_passthrough: u64,
    /// omc_proxy_remember calls resolved by the proxy.
    remember_calls: u64,
    /// omc_proxy_recall calls resolved by the proxy.
    recall_calls: u64,
    /// Bytes saved by replacing repeated base64 image blocks with text markers.
    bytes_saved_images: u64,
    /// omc_proxy_forward calls resolved by the proxy.
    forward_calls: u64,
}

/// Per-conversation state the proxy remembers across turns. Key is a stable
/// `conversation_id` (hash of system + tools + first user message). Value is
/// the set of prefix hashes we've seen this conversation, so on each new turn
/// we can identify which prefix is "stable" (seen before) and mark it for
/// Anthropic's prompt cache.
#[derive(Default)]
struct ConversationState {
    /// Largest message-array length we've seen for this conversation. Anthropic
    /// has already processed messages[0..max_prior_len-1] in a prior request, so
    /// those tokens are eligible for prompt-cache. The block at
    /// messages[max_prior_len-1] is where we should set cache_control.
    max_prior_len: usize,
    /// Total turns observed in this conversation, for diagnostics.
    turn_count: u64,
    /// When we last saw this conversation, for eviction.
    last_seen_unix: i64,
}

#[derive(Clone)]
struct AppState {
    upstream: String,
    rewrite_threshold: usize,
    preview_bytes: usize,
    delta_min_bytes: usize,
    http: reqwest::Client,
    store: Arc<MemoryStore>,
    /// Named key→hash index for omc_proxy_remember / omc_proxy_recall.
    named_refs: Arc<std::sync::Mutex<std::collections::HashMap<String, i64>>>,
    /// Hashes of base64 image blocks seen in previous turns; repeated images
    /// are replaced with a compact text marker instead of re-sending the full
    /// base64 payload (which can be hundreds of KB per image).
    image_hashes: Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
    stats: Arc<std::sync::Mutex<RewriteStats>>,
    /// v0.14.6: per-conversation state, keyed by `conversation_id` (hash of
    /// system + tools + first user message). Bounded to ~256 conversations
    /// before the oldest are evicted to keep proxy memory steady.
    conversations: Arc<std::sync::Mutex<
        std::collections::HashMap<i64, ConversationState>
    >>,
    /// v0.14.8-I: prefix index for fast near-cache-hit lookup. Maps
    /// fnv1a(first 256 bytes of content) → content_hash. When a new block
    /// arrives, we check if its prefix matches anything indexed; if yes,
    /// we compare full text and might emit a differential marker.
    /// Bounded to ~4096 entries with LRU eviction.
    prefix_index: Arc<std::sync::Mutex<
        std::collections::HashMap<u64, i64>
    >>,
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
        "omnimcode-apiproxy v{} starting — bind={} upstream={} threshold={}B preview={}B delta_min={}B",
        env!("CARGO_PKG_VERSION"),
        args.bind, args.upstream, args.rewrite_threshold, args.preview_bytes, args.delta_min_bytes,
    );
    info!(
        "this proxy sees the full LLM conversation. localhost-only bind unless you change --bind."
    );

    let state = AppState {
        upstream: args.upstream.clone(),
        rewrite_threshold: args.rewrite_threshold,
        preview_bytes: args.preview_bytes,
        delta_min_bytes: args.delta_min_bytes,
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?,
        store: Arc::new(MemoryStore::from_env()),
        stats: Arc::new(std::sync::Mutex::new(RewriteStats::default())),
        named_refs: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        image_hashes: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
        conversations: Arc::new(std::sync::Mutex::new(
            std::collections::HashMap::new())),
        prefix_index: Arc::new(std::sync::Mutex::new(
            std::collections::HashMap::new())),
    };

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/_stats", axum::routing::get(stats_endpoint))
        .route("/_version", axum::routing::get(version_endpoint))
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
                       sys={}B msg={}B tool_result={}B tool_use_input={}B tool_defs={}B",
                    body_bytes.len(), b.len(), -((body_bytes.len() - b.len()) as i64),
                    outcome.rewritten_count,
                    outcome.bytes_system, outcome.bytes_messages_text,
                    outcome.bytes_tool_result, outcome.bytes_tool_use_input,
                    outcome.bytes_tool_definitions);
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
                s.bytes_saved_tool_definitions += outcome.bytes_tool_definitions as u64;
                s.bytes_saved_images += outcome.bytes_images as u64;
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
        // Streaming response: pipe SSE chunks in real time. The request body
        // was still fully rewritten (compression happens), and the response
        // is forwarded chunk-by-chunk so the client sees tokens immediately.
        // The auto-expand loop is skipped for streaming; the LLM can still
        // call omc_proxy_expand_ref and the client surfaces it normally.
        state.stats.lock().unwrap().streaming_passthrough += 1;
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
/// Dispatch an `omc_proxy_forward` call: send `message` (optionally with
/// expanded `context_refs`) to `endpoint`/v1/messages and return the
/// sub-LLM's reply text (compressed to a marker if large).
async fn dispatch_forward(
    id: &str, endpoint: &str, message: &str, model: &str,
    context_refs: &[String], headers: &HeaderMap, state: &AppState,
) -> (String, String) {
    // Resolve context_refs from MemoryStore
    let mut parts = vec![message.to_string()];
    for hash_str in context_refs {
        if let Ok(h) = hash_str.parse::<i64>() {
            if let Ok(Some(text)) = state.store.recall(Some(PROXY_CACHE_NAMESPACE), h) {
                parts.push(format!("---\nContext ref {}:\n{}", hash_str, text));
            }
        }
    }
    let user_content = parts.join("\n\n");

    let target_url = if endpoint.is_empty() || endpoint == "self" {
        format!("{}/v1/messages", state.upstream.trim_end_matches('/'))
    } else {
        format!("{}/v1/messages", endpoint.trim_end_matches('/'))
    };

    let effective_model = if model.is_empty() { "claude-haiku-4-5" } else { model };
    let req_body = serde_json::json!({
        "model": effective_model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user_content}]
    });

    let mut req = state.http.post(&target_url).json(&req_body);
    for (k, v) in headers.iter() {
        if k == "x-api-key" || k == "authorization" || k == "anthropic-version" {
            req = req.header(k, v);
        }
    }

    match req.send().await {
        Ok(resp) => {
            match resp.json::<Value>().await {
                Ok(body) => {
                    let reply = body["content"][0]["text"]
                        .as_str().unwrap_or("").to_string();
                    let result = if reply.len() > state.rewrite_threshold {
                        let hash = state.store
                            .store(PROXY_CACHE_NAMESPACE, &reply)
                            .unwrap_or(-1);
                        let preview: String = reply.chars()
                            .take(state.preview_bytes).collect();
                        format!("<omc:ref h=\"{}\" b=\"{}\" preview={:?}/>",
                            hash, reply.len(), preview)
                    } else {
                        reply
                    };
                    state.stats.lock().unwrap().forward_calls += 1;
                    info!("omc_proxy_forward: endpoint={:?} model={:?} → {} bytes",
                        endpoint, effective_model, result.len());
                    (id.to_string(), result)
                }
                Err(e) => (id.to_string(),
                    format!("[omc_proxy_forward: decode error: {}]", e))
            }
        }
        Err(e) => (id.to_string(),
            format!("[omc_proxy_forward: network error: {}]", e))
    }
}

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

        // Resolve any proxy-managed tool calls transparently
        let proxy_calls = collect_proxy_tool_calls(&resp_json);
        if proxy_calls.is_empty() {
            return rebuild_response(status, &resp_headers, resp_body);
        }
        info!("round {}: auto-resolving {} proxy tool_call(s)",
            round + 1, proxy_calls.len());

        let mut next_req: Value = match serde_json::from_slice(&current_body) {
            Ok(v) => v,
            Err(_) => return rebuild_response(status, &resp_headers, resp_body),
        };
        let messages = next_req.get_mut("messages")
            .and_then(Value::as_array_mut);
        let Some(messages) = messages else {
            return rebuild_response(status, &resp_headers, resp_body);
        };
        if let Some(asst_content) = resp_json.get("content").cloned() {
            messages.push(json!({"role": "assistant", "content": asst_content}));
        }

        let mut tool_results: Vec<Value> = Vec::new();
        for call in &proxy_calls {
            let (tool_use_id, result_text) = match call {
                ProxyCall::ExpandRef { id, hash_str } => {
                    let body_text = lookup_expand(hash_str, &state).unwrap_or_else(|e|
                        format!("[apiproxy: expand miss for {}: {}]", hash_str, e));
                    (id.clone(), body_text)
                }
                ProxyCall::Remember { id, key, value } => {
                    let hash = state.store
                        .store(PROXY_CACHE_NAMESPACE, value)
                        .unwrap_or(-1);
                    state.named_refs.lock().unwrap().insert(key.clone(), hash);
                    state.stats.lock().unwrap().remember_calls += 1;
                    info!("omc_proxy_remember: stored key={:?} hash={}", key, hash);
                    (id.clone(), format!("Stored under key {:?} (hash {}).", key, hash))
                }
                ProxyCall::Recall { id, key } => {
                    let result = {
                        let map = state.named_refs.lock().unwrap();
                        map.get(key.as_str()).copied()
                    };
                    let text = match result {
                        Some(h) => state.store.recall(Some(PROXY_CACHE_NAMESPACE), h)
                            .ok().flatten()
                            .unwrap_or_else(|| format!("[apiproxy: recall cache miss for key {:?}]", key)),
                        None => format!("[apiproxy: no memory stored under key {:?}]", key),
                    };
                    state.stats.lock().unwrap().recall_calls += 1;
                    info!("omc_proxy_recall: key={:?}", key);
                    (id.clone(), text)
                }
                ProxyCall::ListRefs { id } => {
                    let req_val: Value = serde_json::from_slice(&current_body)
                        .unwrap_or_default();
                    let markers = find_markers_in_value(&req_val);
                    info!("omc_proxy_list_refs: found {} markers", markers.len());
                    (id.clone(),
                     serde_json::to_string_pretty(&markers).unwrap_or_default())
                }
                ProxyCall::Forward { id, endpoint, message, model, context_refs } => {
                    dispatch_forward(id, endpoint, message, model, context_refs,
                                     headers, state).await
                }
                ProxyCall::Diff { id, hash_a, hash_b } => {
                    let result = diff_marker_contents(hash_a, hash_b, state);
                    info!("omc_proxy_diff: {} vs {}", hash_a, hash_b);
                    (id.clone(), result)
                }
            };
            tool_results.push(json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_text,
            }));
        }
        messages.push(json!({"role": "user", "content": tool_results}));

        current_body = Bytes::from(serde_json::to_vec(&next_req).unwrap());
    }
    warn!("expand loop exceeded {} rounds, returning error", MAX_EXPAND_ROUNDS);
    error_response(StatusCode::BAD_GATEWAY,
        "apiproxy: expand loop limit exceeded")
}

/// If the response's `content` array contains only proxy-managed tool_uses
/// (expand_ref / remember / recall), return them all. If there are any
/// non-proxy tool_uses the client must handle, return empty vec so the
/// response passes through unchanged.
#[derive(Debug)]
enum ProxyCall {
    ExpandRef { id: String, hash_str: String },
    Remember  { id: String, key: String, value: String },
    Recall    { id: String, key: String },
    ListRefs  { id: String },
    /// Route a message to another LLM endpoint and return its compressed reply.
    Forward   { id: String, endpoint: String, message: String, model: String,
                 context_refs: Vec<String> },
    /// Return a unified diff between two stored hashes.
    Diff      { id: String, hash_a: String, hash_b: String },
}

fn collect_proxy_tool_calls(resp: &Value) -> Vec<ProxyCall> {
    let Some(content) = resp.get("content").and_then(Value::as_array) else {
        return vec![];
    };
    let mut calls = Vec::new();
    for block in content {
        if block.get("type").and_then(Value::as_str) != Some("tool_use") { continue; }
        let name = block.get("name").and_then(Value::as_str).unwrap_or("");
        let id   = block.get("id").and_then(Value::as_str).unwrap_or("").to_string();
        let inp  = block.get("input").cloned().unwrap_or(Value::Null);
        match name {
            n if n == EXPAND_TOOL_NAME => {
                let hash_str = inp.get("hash_str").and_then(Value::as_str)
                    .unwrap_or("").to_string();
                if !id.is_empty() && !hash_str.is_empty() {
                    calls.push(ProxyCall::ExpandRef { id, hash_str });
                } else { return vec![]; }
            }
            n if n == REMEMBER_TOOL_NAME => {
                let key   = inp.get("key").and_then(Value::as_str).unwrap_or("").to_string();
                let value = inp.get("value").and_then(Value::as_str).unwrap_or("").to_string();
                if !id.is_empty() && !key.is_empty() {
                    calls.push(ProxyCall::Remember { id, key, value });
                } else { return vec![]; }
            }
            n if n == RECALL_TOOL_NAME => {
                let key = inp.get("key").and_then(Value::as_str).unwrap_or("").to_string();
                if !id.is_empty() && !key.is_empty() {
                    calls.push(ProxyCall::Recall { id, key });
                } else { return vec![]; }
            }
            n if n == LIST_REFS_TOOL_NAME => {
                if !id.is_empty() {
                    calls.push(ProxyCall::ListRefs { id });
                }
            }
            n if n == FORWARD_TOOL_NAME => {
                let endpoint = inp.get("endpoint")
                    .and_then(Value::as_str).unwrap_or("").to_string();
                let message = inp.get("message")
                    .and_then(Value::as_str).unwrap_or("").to_string();
                let model = inp.get("model")
                    .and_then(Value::as_str).unwrap_or("claude-opus-4-5").to_string();
                let context_refs = inp.get("context_refs")
                    .and_then(Value::as_array)
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect())
                    .unwrap_or_default();
                if !id.is_empty() && !message.is_empty() {
                    calls.push(ProxyCall::Forward { id, endpoint, message, model, context_refs });
                }
            }
            _ => return vec![], // non-proxy tool → client must handle
        }
    }
    calls
}

fn lookup_expand(hash_str: &str, state: &AppState) -> Result<String> {
    let hash: i64 = hash_str.parse()
        .map_err(|e| anyhow::anyhow!("hash_str parse: {}", e))?;
    let body = state.store.recall(Some(PROXY_CACHE_NAMESPACE), hash)
        .map_err(anyhow::Error::msg)?
        .ok_or_else(|| anyhow::anyhow!("not in cache"))?;
    Ok(body)
}

/// Extract a double-quoted attribute value from an `<omc:ref .../>` marker string.
fn extract_marker_attr<'a>(marker: &'a str, attr: &str) -> Option<&'a str> {
    let pat = format!(" {}=\"", attr);
    let start = marker.find(pat.as_str())? + pat.len();
    let end  = marker[start..].find('"')? + start;
    Some(&marker[start..end])
}

/// Walk a JSON Value tree and collect every `<omc:ref/>` marker (deduplicated).
/// Returns `[{"hash": "...", "bytes": N}, ...]`.
fn find_markers_in_value(val: &Value) -> Vec<Value> {
    let mut seen    = std::collections::HashSet::<String>::new();
    let mut results = Vec::<Value>::new();
    find_markers_rec(val, &mut seen, &mut results);
    results
}

fn find_markers_rec(
    val: &Value,
    seen: &mut std::collections::HashSet<String>,
    out:  &mut Vec<Value>,
) {
    match val {
        Value::String(s) => {
            let mut pos = 0usize;
            while let Some(rel) = s[pos..].find("<omc:ref") {
                let abs = pos + rel;
                let end = s[abs..].find("/>").map(|e| abs + e + 2).unwrap_or(s.len());
                let marker = &s[abs..end];
                let hash = extract_marker_attr(marker, "hash_str")
                    .or_else(|| extract_marker_attr(marker, "h"))
                    .map(|h| h.to_string());
                let bytes: Option<u64> = extract_marker_attr(marker, "bytes")
                    .or_else(|| extract_marker_attr(marker, "b"))
                    .and_then(|b| b.parse().ok());
                if let Some(h) = hash {
                    if seen.insert(h.clone()) {
                        out.push(json!({ "hash": h, "bytes": bytes }));
                    }
                }
                pos = end;
            }
        }
        Value::Array(arr) => { for v in arr { find_markers_rec(v, seen, out); } }
        Value::Object(map) => { for (_, v) in map { find_markers_rec(v, seen, out); } }
        _ => {}
    }
}

/// Compute a simple line-level unified diff between two stored entries.
fn diff_marker_contents(hash_a: &str, hash_b: &str, state: &AppState) -> String {
    let parse = |h: &str| -> Option<String> {
        let n: i64 = h.parse().ok()?;
        state.store.recall(Some(PROXY_CACHE_NAMESPACE), n).ok()?.filter(|s| !s.is_empty())
    };
    let a = match parse(hash_a) { Some(s) => s, None => return format!("[diff: hash_a {} not found]", hash_a) };
    let b = match parse(hash_b) { Some(s) => s, None => return format!("[diff: hash_b {} not found]", hash_b) };
    if a == b { return "[diff: contents are identical]".into(); }
    let lines_a: Vec<&str> = a.lines().collect();
    let lines_b: Vec<&str> = b.lines().collect();
    let mut out = format!("--- hash:{} ({} lines)\n+++ hash:{} ({} lines)\n",
        hash_a, lines_a.len(), hash_b, lines_b.len());
    // Simple line diff: removals then additions (no LCS for now — sufficient for context).
    let only_a: Vec<&str> = lines_a.iter().filter(|l| !lines_b.contains(l)).copied().collect();
    let only_b: Vec<&str> = lines_b.iter().filter(|l| !lines_a.contains(l)).copied().collect();
    for l in &only_a { out.push('-'); out.push_str(l); out.push('\n'); }
    for l in &only_b { out.push('+'); out.push_str(l); out.push('\n'); }
    if out.len() > 4096 { out.truncate(4096); out.push_str("\n...[truncated]"); }
    out
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
/// Forward the rewritten request to upstream and stream the response back
/// to the client as chunks arrive — works for both:
///   • JSON (non-streaming) responses: the body is small, single chunk
///   • SSE (`text/event-stream`) responses: chunks are forwarded in real-time
///     instead of buffering the entire stream before returning
async fn forward_to_upstream(
    state: &AppState, headers: &HeaderMap, body: Bytes,
) -> Response {
    let url = format!("{}/v1/messages", state.upstream.trim_end_matches('/'));
    let mut req = state.http.post(&url).body(body.to_vec());
    for (k, v) in headers.iter() {
        if k != "host" && k != "content-length" { req = req.header(k, v); }
    }
    let upstream = match req.send().await {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY,
            &format!("upstream: {}", e)),
    };
    let status = upstream.status();
    let resp_headers = upstream.headers().clone();
    // Pipe upstream bytes through to the client as they arrive.
    // Hop-by-hop headers must not be forwarded; content-length is unknown
    // for streamed bodies and is omitted (axum uses chunked encoding).
    let stream = upstream.bytes_stream();
    let mut resp = Response::builder().status(status);
    for (k, v) in resp_headers.iter() {
        if k == "content-length" || k == "transfer-encoding" || k == "connection" { continue; }
        resp = resp.header(k, v);
    }
    resp.body(axum::body::Body::from_stream(stream))
        .unwrap_or_else(|e| error_response(StatusCode::INTERNAL_SERVER_ERROR,
            &format!("response build: {}", e)))
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
    bytes_tool_definitions: usize,
    bytes_images: usize,
}

impl RewriteOutcome {
    fn any(&self) -> bool { self.rewritten_count > 0 }
}

/// v0.14.3 — live cumulative-stats endpoint. `curl http://localhost:8090/_stats`
async fn stats_endpoint(State(state): State<AppState>) -> Response {
    let s = state.stats.lock().unwrap().clone();
    let ratio = if s.bytes_out > 0 {
        s.bytes_in as f64 / s.bytes_out as f64
    } else { 0.0 };
    let total_saved = s.bytes_saved_messages + s.bytes_saved_tool_result
        + s.bytes_saved_system + s.bytes_saved_tool_use_input
        + s.bytes_saved_tool_definitions + s.bytes_saved_images;
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
            "tool_definitions": s.bytes_saved_tool_definitions,
            "images": s.bytes_saved_images,
        },
        "cache_control_inserted_count": s.cache_control_inserted,
        "conversations_seen": s.conversation_count,
        "delta_stores_attempted": s.delta_stores_attempted,
        "streaming_passthrough": s.streaming_passthrough,
        "remember_calls": s.remember_calls,
        "recall_calls": s.recall_calls,
        "forward_calls": s.forward_calls,
    })).unwrap();
    (StatusCode::OK,
     [(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
     json).into_response()
}

/// GET `/_version` — lightweight health-check endpoint.
///
/// Returns a JSON object with the proxy binary version and name. Useful for
/// systemd ExecStartPost health probes, monitoring scripts, and CI smoke tests.
///
/// ```text
/// curl http://localhost:8088/_version
/// {"name":"omnimcode-apiproxy","version":"1.0.0"}
/// ```
async fn version_endpoint() -> Response {
    let json = serde_json::json!({
        "name": "omnimcode-apiproxy",
        "version": env!("CARGO_PKG_VERSION"),
    });
    (StatusCode::OK,
     [(axum::http::header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
     json.to_string()).into_response()
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
    // v0.14.7-L: track hashes already seen this request so duplicates can
    // emit the bare-minimum `<omc:ref h="..."/>` form.
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();

    // ---- system prompt (top-level field) ----
    if let Some(system) = v.get_mut("system") {
        match system {
            Value::String(s) => {
                if s.len() >= state.rewrite_threshold {
                    if let Ok(marker) = make_marker_with_dedup(
                        s, state, MarkerKind::HistoricalText, Some(&mut seen)) {
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
                        let Ok(marker) = make_marker_with_dedup(
                            text, state, MarkerKind::HistoricalText, Some(&mut seen))
                            else { continue };
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
        let is_last_user = Some(idx) == last_user_idx;
        let Some(content) = msg.get_mut("content") else { continue };

        if is_last_user {
            // The last user message contains the human's current question — we
            // never rewrite text blocks there so the LLM sees it verbatim.
            // However, tool_result blocks in the same message (bash output, file
            // reads, etc.) can be megabytes of already-executed output that the
            // LLM doesn't need to re-read in full to answer. Compress those.
            if let Value::Array(blocks) = content {
                for block in blocks.iter_mut() {
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(inner) = block.get_mut("content") {
                            rewrite_tool_result_content(inner, state, &mut out, &mut seen);
                        }
                    }
                    // text blocks in the last user message: pass through verbatim.
                }
            }
            // String content (plain question): skip entirely.
            continue;
        }
        match content {
            Value::String(s) => {
                if s.len() >= state.rewrite_threshold {
                    if let Ok(marker) = make_marker_with_dedup(
                        s, state, MarkerKind::HistoricalText, Some(&mut seen)) {
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
                            let Ok(marker) = make_marker_with_dedup(
                                text, state, MarkerKind::HistoricalText, Some(&mut seen))
                                else { continue };
                            out.bytes_messages_text += text.len();
                            out.rewritten_count += 1;
                            block["text"] = Value::String(marker);
                        }
                        "tool_result" => {
                            if let Some(inner) = block.get_mut("content") {
                                rewrite_tool_result_content(inner, state, &mut out, &mut seen);
                            }
                        }
                        "tool_use" => {
                            // Compress big string values INSIDE the input dict.
                            // Crucially, preserve the original key names so the
                            // LLM doesn't see (and thus copy) a fake field name
                            // when generating fresh tool calls in later turns.
                            if let Some(input) = block.get_mut("input") {
                                rewrite_strings_recursive(input, state, &mut out, &mut seen);
                            }
                        }
                        "image" => {
                            // Repeated base64 images (same screenshot across turns) can be
                            // hundreds of KB each. After the first occurrence — which the LLM
                            // must see to understand the image — replace subsequent occurrences
                            // with a compact text note. The LLM has already seen and processed
                            // the image; the marker conveys that this slot was an image.
                            if let Some(src) = block.get("source") {
                                if src.get("type").and_then(Value::as_str) == Some("base64") {
                                    let data = src.get("data").and_then(Value::as_str).unwrap_or("");
                                    let media_type = src.get("media_type")
                                        .and_then(Value::as_str).unwrap_or("image");
                                    let byte_len = data.len();
                                    // hash just the first 256 bytes of data (fast, collision-resistant enough)
                                    let hash_key = omnimcode_core::tokenizer::fnv1a_64(
                                        data.as_bytes().get(..256).unwrap_or(data.as_bytes())) as u64;
                                    let already_seen = {
                                        let mut set = state.image_hashes.lock().unwrap();
                                        !set.insert(hash_key)
                                    };
                                    if already_seen {
                                        // Replace the whole image block with a text note.
                                        let note = format!(
                                            "[image repeated from prior turn — {}, {} bytes, hash={:x}]",
                                            media_type, byte_len, hash_key
                                        );
                                        out.bytes_images += byte_len;
                                        out.rewritten_count += 1;
                                        *block = json!({ "type": "text", "text": note });
                                    }
                                    // First occurrence: pass through so the LLM can see the image.
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // ---- tool definitions (rewrite BEFORE injecting our expand tool,
    // so we don't compress + re-emit the expand tool we just added) ----
    if let Some(tools) = v.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools.iter_mut() {
            if let Some(desc) = tool.get_mut("description") {
                if let Value::String(s) = desc {
                    if s.len() >= state.rewrite_threshold {
                        if let Ok(marker) = make_marker_with_dedup(
                            s, state, MarkerKind::HistoricalText, Some(&mut seen)) {
                            out.bytes_tool_definitions += s.len();
                            out.rewritten_count += 1;
                            *desc = Value::String(marker);
                        }
                    }
                }
            }
            // input_schema is a JSON Schema dict; walk it for big strings in
            // property descriptions, enums, etc. — preserves schema structure.
            if let Some(schema) = tool.get_mut("input_schema") {
                let before_count = out.rewritten_count;
                rewrite_schema_strings(schema, state, &mut out, &mut seen);
                if out.rewritten_count > before_count {
                    // already counted via rewrite_schema_strings into the
                    // bytes_tool_definitions field
                }
            }
        }
    }

    // ---- v0.14.6: auto-insert cache_control on stable prefix ----
    // This compounds with marker compression: we compress the bytes we send,
    // AND we get Anthropic's 90% prompt-cache discount on the bytes that
    // still go through. On steady-state long sessions this can push effective
    // savings past 95%.
    if maybe_insert_cache_control(&mut v, state) {
        out.rewritten_count += 1;  // Count it as a "block" for stats purposes.
        // We don't add to any byte-savings counter — the savings happen at
        // Anthropic's server, not in our wire size.
    }

    if out.any() {
        inject_expand_tool(&mut v);
    }
    let bytes = Bytes::from(serde_json::to_vec(&v)?);
    Ok((bytes, out))
}

/// v0.14.6: identify a conversation by hashing its stable prefix (system +
/// tools + first user message). This is the same across all turns of one
/// conversation, so we can use it as a key into the per-conversation cache.
fn conversation_id(req: &Value) -> i64 {
    let mut buf = String::new();
    if let Some(s) = req.get("system") {
        buf.push_str(&serde_json::to_string(s).unwrap_or_default());
    }
    if let Some(t) = req.get("tools") {
        buf.push_str(&serde_json::to_string(t).unwrap_or_default());
    }
    if let Some(m) = req.get("messages").and_then(Value::as_array).and_then(|a| a.first()) {
        buf.push_str(&serde_json::to_string(m).unwrap_or_default());
    }
    omnimcode_core::tokenizer::fnv1a_64(buf.as_bytes())
}

/// If this looks like a continuing conversation (we've seen its prefix before),
/// auto-insert `cache_control: ephemeral` on the LAST stable block so Anthropic's
/// prompt-cache layer caches the prefix. Returns `true` if it inserted a hint.
///
/// "Stable block" = the last item BEFORE the current user's turn. The user's
/// current message is the only block that changed; everything before it is
/// what we want cached.
///
/// Idempotent: if the user already set `cache_control` somewhere, we don't
/// touch it. If we already inserted one this request, we don't double-insert.
fn maybe_insert_cache_control(v: &mut Value, state: &AppState) -> bool {
    let current_len = v.get("messages").and_then(Value::as_array)
        .map(|m| m.len()).unwrap_or(0);
    // Need at least 3 messages: [user_q1, assistant_a1, user_q2]. With fewer,
    // there's no stable block worth caching (turn 1 is brand new, turn 2 has
    // only one prior turn which gets cached after we see another one).
    if current_len < 3 { return false; }

    // Track the conversation so /_stats has something interesting. The cache
    // placement itself doesn't need state — we always cache the last stable
    // block, which is messages[current_len - 2] (everything before the
    // current user turn).
    let conv_id = conversation_id(v);
    {
        let mut convs = state.conversations.lock().unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64).unwrap_or(0);
        if convs.len() > 256 {
            let cutoff = now - 3600;
            convs.retain(|_, c| c.last_seen_unix >= cutoff);
        }
        let entry = convs.entry(conv_id).or_default();
        let first_time = entry.turn_count == 0;
        entry.turn_count += 1;
        entry.last_seen_unix = now;
        entry.max_prior_len = entry.max_prior_len.max(current_len);
        if first_time {
            let mut s = state.stats.lock().unwrap();
            s.conversation_count += 1;
        }
    }

    let cache_idx = current_len - 2;  // last stable block (before current user msg)
    let messages_mut = v.get_mut("messages").and_then(Value::as_array_mut).unwrap();
    let target = &mut messages_mut[cache_idx];

    // Idempotent: respect any cache_control the upstream client already set.
    if message_has_cache_control(target) {
        return false;
    }
    let inserted = insert_cache_control_on_last_block(target);
    if inserted {
        let mut s = state.stats.lock().unwrap();
        s.cache_control_inserted += 1;
        debug!("auto-inserted cache_control on conv_id={} at messages[{}]",
               conv_id, cache_idx);
    }
    inserted
}

fn message_has_cache_control(msg: &Value) -> bool {
    match msg.get("content") {
        Some(Value::Array(blocks)) => blocks.iter().any(|b|
            b.get("cache_control").is_some()),
        _ => false,
    }
}

fn insert_cache_control_on_last_block(msg: &mut Value) -> bool {
    let Some(content) = msg.get_mut("content") else { return false };
    match content {
        Value::String(s) => {
            // Convert string-form content to array-form with cache_control hint.
            let text = std::mem::take(s);
            *content = json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }]);
            true
        }
        Value::Array(blocks) => {
            if let Some(last) = blocks.last_mut() {
                if let Value::Object(map) = last {
                    map.insert("cache_control".into(),
                               json!({"type": "ephemeral"}));
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// Walk a JSON-Schema-shaped tool input_schema and marker-rewrite any large
/// string VALUES while preserving structure. Schema dicts contain `description`
/// fields, `enum` arrays of strings, and nested `properties` — all candidates.
fn rewrite_schema_strings(
    val: &mut Value, state: &AppState, out: &mut RewriteOutcome,
    seen: &mut std::collections::HashSet<i64>,
) {
    match val {
        Value::String(s) => {
            if s.len() >= state.rewrite_threshold {
                if let Ok(marker) = make_marker_with_dedup(
                    s, state, MarkerKind::HistoricalText, Some(seen)) {
                    out.bytes_tool_definitions += s.len();
                    out.rewritten_count += 1;
                    *val = Value::String(marker);
                }
            }
        }
        Value::Object(map) => {
            for (_k, v) in map.iter_mut() { rewrite_schema_strings(v, state, out, seen); }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() { rewrite_schema_strings(v, state, out, seen); }
        }
        _ => {}
    }
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
    seen: &mut std::collections::HashSet<i64>,
) {
    match val {
        Value::String(s) => {
            if s.len() >= state.rewrite_threshold {
                if let Ok(marker) = make_marker_with_dedup(
                    s, state, MarkerKind::ToolUseInput, Some(seen)) {
                    out.bytes_tool_use_input += s.len();
                    out.rewritten_count += 1;
                    *val = Value::String(marker);
                }
            }
        }
        Value::Object(map) => {
            for (_k, v) in map.iter_mut() {
                rewrite_strings_recursive(v, state, out, seen);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                rewrite_strings_recursive(v, state, out, seen);
            }
        }
        _ => {}
    }
}

fn rewrite_tool_result_content(
    inner: &mut Value, state: &AppState, out: &mut RewriteOutcome,
    seen: &mut std::collections::HashSet<i64>,
) {
    match inner {
        Value::String(s) => {
            if s.len() >= state.rewrite_threshold {
                if let Ok(marker) = make_marker_with_dedup(
                    s, state, MarkerKind::ToolResult, Some(seen)) {
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
                    let Ok(marker) = make_marker_with_dedup(
                        text, state, MarkerKind::ToolResult, Some(seen))
                        else { continue };
                    out.bytes_tool_result += text.len();
                    out.rewritten_count += 1;
                    part["text"] = Value::String(marker);
                }
            }
        }
        _ => {}
    }
}

/// What category of content is being compressed. Drives whether the marker
/// gets a `preview=` attribute (helpful for tool_result, wasted bytes for
/// historical assistant text or tool_use inputs).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MarkerKind {
    /// `tool_result.content` — LLM benefits from preview to know if it
    /// needs to expand. Keep the full marker.
    ToolResult,
    /// Historical assistant or user text block, system prompt, tool def.
    /// LLM has already "seen" this in a prior turn; preview is wasted.
    HistoricalText,
    /// `tool_use.input` field value — LLM emitted this itself, doesn't
    /// need to re-read its own output. Preview is wasted.
    ToolUseInput,
}

#[cfg(test)]
fn make_marker(text: &str, state: &AppState, kind: MarkerKind) -> Result<String> {
    make_marker_with_dedup(text, state, kind, None)
}

/// v0.14.7-L: intra-request dedup. `seen_hashes` is `Some(set)` when we want
/// to track repeated content within a single request — first occurrence
/// emits the full marker, subsequent emit the bare `<omc:ref h="..."/>`
/// form (~30 bytes instead of ~150 for the duplicates).
fn make_marker_with_dedup(
    text: &str, state: &AppState, kind: MarkerKind,
    seen_hashes: Option<&mut std::collections::HashSet<i64>>,
) -> Result<String> {
    // v0.14.8-I: route cache writes through Axis 5 (OMCD delta) when we
    // detect a near-edit of a previously-cached body. The base-hash lookup
    // is O(1) via prefix_index. If a base is found, store_as_delta stores
    // a tiny delta on disk instead of duplicating the full body.
    //
    // IMPORTANT: this is a DISK-side optimization, not a wire-side one.
    // The wire marker is the same compact `<omc:ref h="..." b="N"/>` form.
    // We tried emitting `<omc:diff base="..." pre="N" suf="..."/>` markers
    // on the wire, but honest accounting showed they're LARGER than the
    // 50-byte slim ref marker the recall path already produces. So the win
    // is purely disk-resident: future store-side dedup, not request-time
    // bytes.
    let hash = try_delta_store(text, state)
        .or_else(|| state.store.store(PROXY_CACHE_NAMESPACE, text).ok())
        .ok_or_else(|| anyhow::anyhow!("cache write failed"))?;
    // Index this body's prefix so the NEXT near-edit can find it as base.
    if text.len() >= state.delta_min_bytes { register_prefix(text, hash, state); }

    // v0.14.7-L: if we've already emitted a full marker for this hash this
    // request, the subsequent ones can be the bare-minimum form.
    if let Some(set) = seen_hashes {
        if !set.insert(hash) {
            // already present
            return Ok(format!("<omc:ref h=\"{}\"/>", hash));
        }
    }

    // v0.14.7-K: drop preview for content the LLM doesn't benefit from
    // previewing. Saves ~150 bytes/marker × 100s of markers per turn.
    match kind {
        MarkerKind::HistoricalText | MarkerKind::ToolUseInput => {
            Ok(format!("<omc:ref h=\"{}\" b=\"{}\"/>", hash, text.len()))
        }
        MarkerKind::ToolResult => {
            // Keep the preview only when content is large enough that the
            // LLM might want to decide whether to expand.
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
    }
}

/// v0.14.8-I: index a body's first-256-byte prefix → content_hash so the next
/// call can try a near-cache-hit lookup.
fn register_prefix(text: &str, hash: i64, state: &AppState) {
    let prefix = &text.as_bytes()[..text.len().min(256)];
    let prefix_hash = omnimcode_core::tokenizer::fnv1a_64(prefix) as u64;
    let mut idx = state.prefix_index.lock().unwrap();
    if idx.len() > 4096 {
        // Crude eviction: clear when we hit the cap. Not LRU, but the
        // MemoryStore is the source of truth so a cleared index just means
        // future near-edits fall back to plain store (no data loss).
        idx.clear();
    }
    idx.insert(prefix_hash, hash);
}

/// v0.14.8-I: try to store `text` as a delta against a prefix-near cached
/// body. Returns `Some(hash_of_text)` if delta was viable, `None` otherwise.
/// The hash returned is still the hash of the FULL text (so the marker / recall
/// path is unchanged for the LLM).
fn try_delta_store(text: &str, state: &AppState) -> Option<i64> {
    if text.len() < state.delta_min_bytes { return None; }
    let prefix = &text.as_bytes()[..text.len().min(256)];
    let prefix_hash = omnimcode_core::tokenizer::fnv1a_64(prefix) as u64;
    let base_hash = {
        let idx = state.prefix_index.lock().unwrap();
        *idx.get(&prefix_hash)?
    };
    // store_as_delta handles the "is the prefix actually long enough?" check
    // itself (need ≥64 bytes shared) and falls back to plain store if not.
    // Either way we get a valid content-hash for `text`.
    let result = state.store.store_as_delta(PROXY_CACHE_NAMESPACE, text, base_hash).ok()?;
    {
        let mut s = state.stats.lock().unwrap();
        s.delta_stores_attempted += 1;
    }
    Some(result)
}

/// Inject all proxy-owned tools into the request's tools array.
/// Currently: omc_proxy_expand_ref, omc_proxy_remember, omc_proxy_recall.
/// We avoid double-injection by checking for expand_ref (the sentinel tool).
fn inject_proxy_tools(req: &mut Value) {
    let tools_arr = match req.get_mut("tools") {
        Some(Value::Array(a)) => {
            if a.iter().any(|t| t.get("name").and_then(Value::as_str) == Some(EXPAND_TOOL_NAME)) {
                return; // already injected
            }
            a
        }
        _ => {
            req["tools"] = Value::Array(vec![]);
            req["tools"].as_array_mut().unwrap()
        }
    };

    // ── omc_proxy_expand_ref ──────────────────────────────────────────────
    tools_arr.push(json!({
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
    }));

    // ── omc_proxy_remember ────────────────────────────────────────────────
    tools_arr.push(json!({
        "name": REMEMBER_TOOL_NAME,
        "description": "Durably store an arbitrary string under a human-readable \
                        key. The proxy persists it and you can retrieve it in any \
                        future turn (including across sessions) via omc_proxy_recall. \
                        Use this to save findings, summaries, or shared state that \
                        another agent or future-you will need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key":   { "type": "string",
                           "description": "Human-readable name (e.g. 'plan_v2', 'user_prefs')." },
                "value": { "type": "string",
                           "description": "Content to store (any UTF-8 text)." }
            },
            "required": ["key", "value"]
        }
    }));

    // ── omc_proxy_recall ──────────────────────────────────────────────────
    tools_arr.push(json!({
        "name": RECALL_TOOL_NAME,
        "description": "Retrieve content previously stored with omc_proxy_remember. \
                        Returns the stored text, or an error if the key has never \
                        been set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": { "type": "string",
                         "description": "The key passed to a prior omc_proxy_remember call." }
            },
            "required": ["key"]
        }
    }));

    // ── omc_proxy_list_refs ─────────────────────────────────────────────────
    tools_arr.push(json!({
        "name": DIFF_TOOL_NAME,
        "description": "Compute a line-level diff between two stored <omc:ref/> entries. \
                        Returns a unified diff string. Useful for understanding what changed \
                        between two near-edit versions without expanding both in full.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hash_a": { "type": "string", "description": "Hash of the base content." },
                "hash_b": { "type": "string", "description": "Hash of the changed content." }
            },
            "required": ["hash_a", "hash_b"]
        }
    }));
    tools_arr.push(json!({
        "name": LIST_REFS_TOOL_NAME,
        "description": "Return a JSON array describing every <omc:ref/> marker \
                        currently present in the conversation context. Each entry \
                        has {\"hash\", \"bytes\"} so you can decide which to expand. \
                        Takes no arguments.",
        "input_schema": { "type": "object", "properties": {}, "required": [] }
    }));

    // ── omc_proxy_forward ────────────────────────────────────────────────
    tools_arr.push(json!({
        "name": FORWARD_TOOL_NAME,
        "description": "Route a message to another LLM and return its compressed \
                        reply as an <omc:ref/> marker. Use `endpoint` to target a \
                        different proxy instance (default: same upstream). \
                        Optionally pass `context_refs` (list of hash strings from \
                        omc_proxy_list_refs) to expand relevant prior context into \
                        the sub-request — sharing memory costs O(marker) not O(content).",
        "input_schema": {
            "type": "object",
            "properties": {
                "message":      { "type": "string",
                                  "description": "The message to send to the target LLM." },
                "model":        { "type": "string",
                                  "description": "Model to use (default: claude-opus-4-5)." },
                "endpoint":     { "type": "string",
                                  "description": "Target proxy/API base URL (omit or 'self' for same upstream)." },
                "context_refs": { "type": "array", "items": { "type": "string" },
                                  "description": "Hash strings of context markers to include in the sub-request." }
            },
            "required": ["message"]
        }
    }));
}

/// Compatibility shim -- callers that used inject_expand_tool still work.
#[allow(dead_code)]
fn inject_expand_tool(req: &mut Value) { inject_proxy_tools(req); }


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build an AppState pointing at a tempdir-scoped MemoryStore so tests
    /// don't share cache state with each other or the real user store.
    fn test_state(threshold: usize) -> AppState {
        let tmpdir = std::env::temp_dir().join(format!(
            "omc-apiproxy-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&tmpdir).unwrap();
        std::env::set_var("OMC_MEMORY_ROOT", &tmpdir);
        AppState {
            upstream: "http://127.0.0.1:0".into(),
            rewrite_threshold: threshold,
            preview_bytes: 80,
            delta_min_bytes: 1024,
            http: reqwest::Client::new(),
            store: Arc::new(MemoryStore::from_env()),
            stats: Arc::new(std::sync::Mutex::new(RewriteStats::default())),
            named_refs: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            image_hashes: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
            conversations: Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new())),
            prefix_index: Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new())),
        }
    }

    /// v0.14.4 regression: tool_use.input keys MUST be preserved. The whole
    /// point of the v0.14.4 hotfix was to stop replacing the input dict with
    /// `{"_omc_compressed_input_marker": "..."}` which the LLM then learned
    /// to copy. Verify the rewritten input still has its original keys.
    #[test]
    fn tool_use_input_preserves_keys() {
        let state = test_state(256);
        let big = "X".repeat(1000);
        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "Write",
                     "input": {"file_path": "/tmp/x.txt", "content": big}}
                ]},
                {"role": "user", "content": "summarize"}
            ]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, outcome) = rewrite_request_body(&body, &state).unwrap();
        assert!(outcome.rewritten_count > 0, "expected at least 1 rewrite");
        let v: Value = serde_json::from_slice(&out).unwrap();
        let input = &v["messages"][0]["content"][0]["input"];
        // Real keys must still be there. NO `_omc_compressed_input_marker` allowed.
        assert!(input.get("file_path").is_some(), "lost file_path key");
        assert!(input.get("content").is_some(), "lost content key");
        assert!(input.get("_omc_compressed_input_marker").is_none(),
                "v0.14.3 schema-poisoning regression — fake key reintroduced");
        // The big string was marker-replaced, file_path is small enough to stay.
        let content_str = input["content"].as_str().unwrap();
        assert!(content_str.starts_with("<omc:ref"),
                "expected content to become an <omc:ref/> marker, got: {}",
                &content_str[..50.min(content_str.len())]);
        assert_eq!(input["file_path"].as_str().unwrap(), "/tmp/x.txt",
                   "small file_path should remain untouched");
    }

    /// The LAST user message is the user's current intent — it must NEVER
    /// be marker-replaced, or the LLM would have to round-trip just to know
    /// what was asked.
    #[test]
    fn last_user_message_never_rewritten() {
        // Text content (the human's actual question) in the last user message
        // must always pass through verbatim — the LLM needs to see it to respond.
        let state = test_state(256);
        let big_question = "Please analyze: ".to_string() + &"Q".repeat(1000);
        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "old turn"},
                {"role": "assistant", "content": "old reply"},
                {"role": "user", "content": big_question.clone()}
            ]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, _) = rewrite_request_body(&body, &state).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        let last = v["messages"][2]["content"].as_str().unwrap();
        assert_eq!(last, big_question,
                   "last user message text content must be byte-identical to input");
    }

    /// tool_result blocks in the last user message ARE compressed even though
    /// the wrapping message is "last user".  In agentic workflows the last
    /// message is nearly always an array of tool_results (bash output, file
    /// reads, …) — often megabytes — and the LLM can expand via
    /// omc_proxy_expand_ref if it needs the full content.
    #[test]
    fn last_user_tool_results_are_compressed() {
        let threshold = 256;
        let state = test_state(threshold);

        // A large tool_result body — clearly above threshold.
        let big_output = "line: data output\n".repeat(100); // ~1.8KB
        assert!(big_output.len() > threshold, "pre-condition: must exceed threshold");

        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "user",  "content": "run the script"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "bash",
                     "input": {"command": "echo hello"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": big_output.clone()},
                    {"type": "text", "text": "What does this mean?"}
                ]}
            ]
        });

        let body = serde_json::to_vec(&req).unwrap();
        let (out, outcome) = rewrite_request_body(&body, &state).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();

        // tool_result block must have been rewritten to a marker
        let last_msg_content = v["messages"][2]["content"].as_array().unwrap();
        let tool_result_block = &last_msg_content[0];
        let tr_content = tool_result_block["content"].as_str().unwrap();
        assert!(tr_content.starts_with("<omc:ref"),
            "tool_result in last user message must be compressed; got: {}", &tr_content[..80]);

        // text block must pass through verbatim
        let text_block = &last_msg_content[1];
        assert_eq!(text_block["text"].as_str().unwrap(), "What does this mean?",
            "text block in last user message must be verbatim");

        // stat must have ticked
        assert!(outcome.any(), "rewrite outcome must report at least one rewritten block");
    }

    /// Marker round-trip: any text we compress must come back IDENTICAL via
    /// the cache lookup path that the expand-tool uses.
    #[test]
    fn marker_round_trip_lossless() {
        let state = test_state(256);
        let original = "abc🎯 ñ é 漢字\nline2\n\tindented\n".repeat(50);  // multi-byte, control chars
        let marker = make_marker(&original, &state, MarkerKind::ToolResult).unwrap();
        // Extract hash_str from the marker
        let hash_attr = marker.split("hash_str=\"").nth(1).unwrap();
        let hash_str = hash_attr.split('"').next().unwrap();
        let hash: i64 = hash_str.parse().unwrap();
        let recovered = state.store.recall(Some(PROXY_CACHE_NAMESPACE), hash)
            .unwrap().expect("must be in cache");
        assert_eq!(recovered, original, "byte-identical round-trip required");
    }

    /// Small blocks under threshold pass through unmodified.
    #[test]
    fn small_blocks_untouched() {
        let state = test_state(1024);
        let small = "short content";
        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": small},
                {"role": "user", "content": "ask"}
            ]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, outcome) = rewrite_request_body(&body, &state).unwrap();
        assert_eq!(outcome.rewritten_count, 0);
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["messages"][0]["content"].as_str().unwrap(), small);
    }

    /// System prompt with cache_control hints — the hint MUST survive the rewrite
    /// so Anthropic's 90% prompt-cache discount keeps working.
    #[test]
    fn system_prompt_preserves_cache_control() {
        let state = test_state(256);
        let big_sys = "You are an expert. ".repeat(100);
        let req = json!({
            "model": "test", "max_tokens": 10,
            "system": [
                {"type": "text", "text": big_sys,
                 "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, outcome) = rewrite_request_body(&body, &state).unwrap();
        assert!(outcome.bytes_system > 0, "system prompt should have been compressed");
        let v: Value = serde_json::from_slice(&out).unwrap();
        let cc = &v["system"][0]["cache_control"];
        assert_eq!(cc["type"].as_str().unwrap(), "ephemeral",
                   "cache_control hint lost — would break Anthropic prompt-cache");
    }

    /// v0.14.5b: tool definitions (`tools[].description` + nested input_schema
    /// strings) get compressed. The injected `omc_proxy_expand_ref` tool MUST
    /// not itself be compressed (it was just added by us in this same pass).
    #[test]
    fn tool_definitions_compressed_but_expand_tool_preserved() {
        let state = test_state(256);
        let long_desc = "This tool does X. It accepts Y. Returns Z. ".repeat(50);
        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [{"role": "user", "content": "use the tool"}],
            "tools": [
                {
                    "name": "BigTool",
                    "description": long_desc.clone(),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "arg": {
                                "type": "string",
                                "description": "A long arg description. ".repeat(50)
                            }
                        }
                    }
                }
            ]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, outcome) = rewrite_request_body(&body, &state).unwrap();
        assert!(outcome.bytes_tool_definitions > 0,
                "expected tool definition bytes to be compressed");
        let v: Value = serde_json::from_slice(&out).unwrap();
        let tools = v["tools"].as_array().unwrap();
        // Original tool still has its name + shape, but description is a marker
        let big = tools.iter().find(|t| t["name"] == "BigTool").unwrap();
        let desc = big["description"].as_str().unwrap();
        assert!(desc.starts_with("<omc:ref"),
                "expected description to be marker, got: {}", &desc[..50.min(desc.len())]);
        assert_eq!(big["input_schema"]["type"].as_str().unwrap(), "object",
                   "schema structure must be preserved");
        // The injected expand tool MUST exist and MUST have its uncompressed
        // description (otherwise the LLM can't tell what it does).
        let expand = tools.iter().find(|t| t["name"] == EXPAND_TOOL_NAME)
            .expect("expand tool must be injected");
        let expand_desc = expand["description"].as_str().unwrap();
        assert!(!expand_desc.starts_with("<omc:ref"),
                "expand tool's own description must not be compressed");
    }

    /// v0.14.6: cache_control auto-insertion fires whenever messages.len() >= 3,
    /// placing the hint on the LAST stable block (messages[len-2]) so Anthropic
    /// caches everything up through it. First two turns lack a stable block.
    #[test]
    fn cache_control_inserted_on_third_turn() {
        let state = test_state(256);
        let sys = json!([{"type":"text","text":"You are a helpful assistant."}]);
        let tools = json!([{"name":"x","description":"x","input_schema":{"type":"object"}}]);

        // Turn 1: one user message. Nothing to cache.
        let t1 = json!({
            "model": "test", "max_tokens": 10, "system": sys, "tools": tools,
            "messages": [{"role": "user", "content": "first ask"}]
        });
        let (out1, _) = rewrite_request_body(&serde_json::to_vec(&t1).unwrap(), &state).unwrap();
        let v1: Value = serde_json::from_slice(&out1).unwrap();
        assert!(!message_has_cache_control(&v1["messages"][0]),
                "turn 1 has no stable block, no cache_control");

        // Turn 2: 2 messages [user, assistant]. Wait — what does Claude Code
        // actually send on turn 2? It sends [user_q1, assistant_a1, user_q2]
        // which is 3 messages. The "turn count" from the proxy POV is the
        // number of requests, but each request grows messages by 2 (one
        // assistant response, one user follow-up). So messages.len() goes
        // 1, 3, 5, 7, ... Turn 1 = 1 message, turn 2 = 3, turn 3 = 5.
        // With current_len >= 3 guard: turn 2 onward fires.
        let t2 = json!({
            "model": "test", "max_tokens": 10, "system": sys, "tools": tools,
            "messages": [
                {"role": "user", "content": "first ask"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second ask"}
            ]
        });
        let (out2, _) = rewrite_request_body(&serde_json::to_vec(&t2).unwrap(), &state).unwrap();
        let v2: Value = serde_json::from_slice(&out2).unwrap();
        // Stable block is messages[1] (the assistant reply). Should have cc now.
        assert!(message_has_cache_control(&v2["messages"][1]),
                "turn 2 should cache assistant_a1");
        // Current user turn MUST NOT have cache_control.
        assert!(!message_has_cache_control(&v2["messages"][2]),
                "current user turn must not have cache_control");

        // Turn 3: 5 messages. Stable block = messages[3] (the latest assistant).
        let t3 = json!({
            "model": "test", "max_tokens": 10, "system": sys, "tools": tools,
            "messages": [
                {"role": "user", "content": "first ask"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second ask"},
                {"role": "assistant", "content": "second reply"},
                {"role": "user", "content": "third ask"}
            ]
        });
        let (out3, _) = rewrite_request_body(&serde_json::to_vec(&t3).unwrap(), &state).unwrap();
        let v3: Value = serde_json::from_slice(&out3).unwrap();
        assert!(message_has_cache_control(&v3["messages"][3]),
                "turn 3 should cache assistant_a2 (the new stable block)");
        assert!(!message_has_cache_control(&v3["messages"][4]),
                "current user turn must not have cache_control");
    }

    /// v0.14.6: if the user (or upstream client) already set cache_control,
    /// respect it. Don't add a duplicate or override their placement.
    #[test]
    fn cache_control_respects_user_provided() {
        let state = test_state(256);
        // Prime the conversation cache so we'd normally insert.
        let primer = json!({
            "model": "test", "max_tokens": 10,
            "system": "sys", "tools": [],
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"}
            ]
        });
        let _ = rewrite_request_body(&serde_json::to_vec(&primer).unwrap(), &state);

        // Now request with user-supplied cache_control:
        let req = json!({
            "model": "test", "max_tokens": 10,
            "system": "sys", "tools": [],
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "a",
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"}
            ]
        });
        let (out, _) = rewrite_request_body(&serde_json::to_vec(&req).unwrap(), &state).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        // Original cache_control on messages[1].content[0] is preserved.
        assert_eq!(v["messages"][1]["content"][0]["cache_control"]["type"]
                   .as_str().unwrap(), "ephemeral");
        // We did NOT insert one on messages[3] because we found one upstream.
        // (Actually we check the LAST stable message which is messages[3];
        // it has no cache_control, but messages[1] does. The check should
        // see the existing one and skip.)
        // The test: messages[3] should NOT have cache_control because the
        // overall conversation already had one set.
        // ... wait: our check is per-message, not per-conversation. So this
        // test only validates that we don't insert ANOTHER cache_control on
        // the LAST stable block if it already has one. Let's verify that.
        // Re-run with the LAST stable block (messages[3]) already having cc:
        let req2 = json!({
            "model": "test", "max_tokens": 10,
            "system": "sys2", "tools": [],
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "a2",
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": "q3"}
            ]
        });
        // Prime, then run twice so the prefix is seen.
        let _ = rewrite_request_body(&serde_json::to_vec(&req2).unwrap(), &state);
        let (out2, _) = rewrite_request_body(&serde_json::to_vec(&req2).unwrap(), &state).unwrap();
        let v2: Value = serde_json::from_slice(&out2).unwrap();
        // messages[3].content[0] should still have EXACTLY ONE cache_control.
        let last = &v2["messages"][3]["content"];
        let blocks = last.as_array().unwrap();
        assert_eq!(blocks.len(), 1, "should not have added new block");
        assert!(blocks[0].get("cache_control").is_some(),
                "user's cache_control preserved");
    }

    /// v0.14.7-K: markers for HistoricalText/ToolUseInput drop the
    /// preview attribute (saves ~150 bytes/marker).
    #[test]
    fn slim_markers_drop_preview_for_historical_text() {
        let state = test_state(256);
        let big = "X".repeat(1000);
        // Historical text in messages[]
        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": big.clone()},
                {"role": "user", "content": "ask"}
            ]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, _) = rewrite_request_body(&body, &state).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        let marker = v["messages"][0]["content"].as_str().unwrap();
        // HistoricalText markers should be the slim form: <omc:ref h="..." b="N"/>
        assert!(marker.contains(" h=\""), "slim marker should use h= not hash_str=");
        assert!(marker.contains(" b=\""), "slim marker should use b= not bytes=");
        assert!(!marker.contains("preview="),
                "slim marker for HistoricalText must not have preview");
        // tool_result markers, by contrast, keep preview (for content < 8KB)
        let req2 = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "x", "content": big.clone()}
                ]},
                {"role": "user", "content": "ask"}
            ]
        });
        let (out2, _) = rewrite_request_body(&serde_json::to_vec(&req2).unwrap(), &state).unwrap();
        let v2: Value = serde_json::from_slice(&out2).unwrap();
        let tr_marker = v2["messages"][0]["content"][0]["content"].as_str().unwrap();
        assert!(tr_marker.contains("preview="),
                "tool_result marker should keep preview (LLM needs it to decide expansion)");
    }

    /// v0.14.7-L: when the same content appears twice in one request,
    /// the second occurrence collapses to bare `<omc:ref h="..."/>`.
    #[test]
    fn intra_request_dedup_collapses_repeats() {
        let state = test_state(256);
        let big = "REPEATING CONTENT BLOCK ".repeat(50); // ~1200 bytes
        let req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": big.clone()},
                {"role": "user", "content": big.clone()},
                {"role": "assistant", "content": big.clone()},
                {"role": "user", "content": "current"}
            ]
        });
        let body = serde_json::to_vec(&req).unwrap();
        let (out, _) = rewrite_request_body(&body, &state).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        // Helper: pull marker text from a message regardless of whether
        // it's string-form or array-form (cache_control insertion can
        // convert string-form to array-form mid-pass).
        let extract = |idx: usize| -> String {
            let c = &v["messages"][idx]["content"];
            if let Some(s) = c.as_str() { return s.to_string(); }
            if let Some(arr) = c.as_array() {
                if let Some(first) = arr.first() {
                    if let Some(t) = first.get("text").and_then(Value::as_str) {
                        return t.to_string();
                    }
                }
            }
            panic!("could not extract marker from messages[{}]: {}", idx, c)
        };
        let m0 = extract(0);
        let m1 = extract(1);
        let m2 = extract(2);
        // First occurrence: full slim marker with b=
        assert!(m0.contains(" b=\""), "first marker should be full: {}", m0);
        // Second + third: bare form (no b= attr)
        assert!(!m1.contains(" b=\""),
                "second occurrence should be bare ref: {}", m1);
        assert!(!m2.contains(" b=\""),
                "third occurrence should be bare ref: {}", m2);
        // All three reference the same hash
        let extract_h = |m: &str| -> String {
            m.split(" h=\"").nth(1).unwrap().split('"').next().unwrap().to_string()
        };
        assert_eq!(extract_h(&m0), extract_h(&m1));
        assert_eq!(extract_h(&m0), extract_h(&m2));
    }

    /// v0.14.8-I: when a content body is a near-edit of a previously-cached
    /// body, the disk-side store should route through Axis 5 (OMCD delta).
    /// We verify by checking that delta_stores_attempted ticks up AND that
    /// recall still returns the correct full text byte-for-byte.
    #[test]
    fn near_edit_routes_through_delta_store() {
        let state = test_state(256);
        // Base body. Large enough to be eligible for prefix indexing.
        let base = "Common prefix.\n".repeat(80); // ~1200 bytes
        // First request stores `base`. No delta possible (nothing prior).
        let req1 = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": base.clone()},
                {"role": "user", "content": "first"}
            ]
        });
        let _ = rewrite_request_body(&serde_json::to_vec(&req1).unwrap(), &state).unwrap();
        let delta_attempts_before = state.stats.lock().unwrap().delta_stores_attempted;

        // Now a near-edit: same content + a small suffix. Should trigger delta.
        let near_edit = format!("{}APPENDED MORE CONTENT TO THE END", base);
        let req2 = json!({
            "model": "test", "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": near_edit.clone()},
                {"role": "user", "content": "second"}
            ]
        });
        let (out2, _) = rewrite_request_body(&serde_json::to_vec(&req2).unwrap(), &state).unwrap();
        let delta_attempts_after = state.stats.lock().unwrap().delta_stores_attempted;
        assert!(delta_attempts_after > delta_attempts_before,
                "expected delta_stores_attempted to increment for near-edit");

        // Extract the marker that was emitted for near_edit, then recall via
        // the hash inside it. Should reconstruct byte-identical original.
        let v: Value = serde_json::from_slice(&out2).unwrap();
        let marker_holder = &v["messages"][0]["content"];
        let marker_str = if let Some(s) = marker_holder.as_str() {
            s.to_string()
        } else if let Some(arr) = marker_holder.as_array() {
            // cache_control insertion may have moved it into array form
            arr.first().and_then(|b| b.get("text"))
                .and_then(Value::as_str).unwrap().to_string()
        } else {
            panic!("couldn't extract marker")
        };
        // Slim marker form: <omc:ref h="N" b="M"/>
        let h = marker_str.split(" h=\"").nth(1).unwrap()
            .split('"').next().unwrap().parse::<i64>().unwrap();
        let recovered = state.store.recall(Some(PROXY_CACHE_NAMESPACE), h)
            .unwrap().expect("must be recoverable");
        assert_eq!(recovered, near_edit,
                   "delta-stored body must round-trip byte-identical");
    }

    /// Multi-turn dogfood simulation: walk a conversation, verify each turn's
    /// rewrite preserves the LLM-emitted shape AND the markers expand cleanly
    /// to the original bytes via the cache.
    #[test]
    fn five_turn_conversation_no_drift() {
        let state = test_state(256);
        let mut messages: Vec<Value> = Vec::new();
        let mut originals: Vec<(i64, String)> = Vec::new();

        for turn in 0..5 {
            // User turn
            messages.push(json!({
                "role": "user",
                "content": format!("turn {} ask", turn)
            }));
            // Build the request with this conversation so far
            let req = json!({
                "model": "test", "max_tokens": 10,
                "messages": messages.clone()
            });
            let body = serde_json::to_vec(&req).unwrap();
            let (out, _) = rewrite_request_body(&body, &state).unwrap();
            let v: Value = serde_json::from_slice(&out).unwrap();

            // Assert last user message is uncompressed every turn
            let last_idx = v["messages"].as_array().unwrap().len() - 1;
            let last_text = v["messages"][last_idx]["content"].as_str().unwrap();
            assert_eq!(last_text, format!("turn {} ask", turn),
                "turn {}: last user msg got rewritten", turn);

            // Now LLM emits an assistant reply with a big tool result
            let big_output = format!("LARGE OUTPUT FOR TURN {} ", turn).repeat(50);
            let h = state.store.store(PROXY_CACHE_NAMESPACE, &big_output).unwrap();
            originals.push((h, big_output.clone()));
            messages.push(json!({
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": format!("tu_{}", turn),
                     "name": "Write", "input": {
                         "file_path": format!("/tmp/{}.txt", turn),
                         "content": big_output
                     }}
                ]
            }));
            messages.push(json!({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": format!("tu_{}", turn),
                     "content": format!("wrote turn {}", turn)}
                ]
            }));
        }

        // After 5 turns, all stored originals must round-trip from cache.
        for (h, expected) in originals {
            let got = state.store.recall(Some(PROXY_CACHE_NAMESPACE), h)
                .unwrap().expect("must still be in cache");
            assert_eq!(got, expected);
        }
    }

    // ── Feature: streaming detection ──────────────────────────────────────────

    /// `is_streaming_request` must return true only when the top-level
    /// `stream` field is the boolean `true`.
    #[test]
    fn streaming_flag_detected() {
        let to_bytes = |v: &serde_json::Value| serde_json::to_vec(v).unwrap();

        let streaming    = json!({"model":"claude-3","stream":true, "messages":[]});
        let not_streaming = json!({"model":"claude-3","stream":false,"messages":[]});
        let no_field     = json!({"model":"claude-3","messages":[]});
        let null_field   = json!({"model":"claude-3","stream":null,"messages":[]});

        assert!( is_streaming_request(&to_bytes(&streaming)),    "stream:true should be detected");
        assert!(!is_streaming_request(&to_bytes(&not_streaming)), "stream:false must not be detected");
        assert!(!is_streaming_request(&to_bytes(&no_field)),     "missing field must not be detected");
        assert!(!is_streaming_request(&to_bytes(&null_field)),   "null field must not be detected");
    }

    // ── Feature: configurable delta_min_bytes ─────────────────────────────────

    /// With a very high `delta_min_bytes`, near-edits no longer attempt
    /// delta storage — delta_stores_attempted stays zero.
    #[test]
    fn high_delta_min_bytes_skips_delta() {
        let mut state = test_state(200);
        state.delta_min_bytes = 999_999; // nothing will ever be this long in tests

        let base = "a".repeat(500);
        let edit = format!("{}modified", &base[..400]);

        // Store base, then attempt a near-edit — both should be plain stores.
        let body1 = json!({"model":"m","messages":[
            {"role":"user","content": base.clone()}
        ]});
        let body2 = json!({"model":"m","messages":[
            {"role":"assistant","content": base.clone()},
            {"role":"user","content": edit.clone()}
        ]});
        let _ = rewrite_request_body(
            &serde_json::to_vec(&body1).unwrap(), &state);
        let _ = rewrite_request_body(
            &serde_json::to_vec(&body2).unwrap(), &state);

        assert_eq!(state.stats.lock().unwrap().delta_stores_attempted, 0,
            "delta_stores_attempted must stay 0 when delta_min_bytes is very large");
    }

    /// With a low `delta_min_bytes`, even short near-edits route through the
    /// delta store (delta_stores_attempted increments).
    #[test]
    fn low_delta_min_bytes_triggers_delta() {
        let mut state = test_state(64);
        state.delta_min_bytes = 64; // very low — any block ≥64 bytes attempts delta

        let base = "x".repeat(200);

        // First turn: assistant reply with base content — assistant messages are
        // always eligible for rewriting, so the prefix gets registered.
        let body1 = json!({"model":"m","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":[{"type":"text","text": base.clone()}]}
        ]});
        let _ = rewrite_request_body(&serde_json::to_vec(&body1).unwrap(), &state);

        // Second turn: same assistant content appears again as history.
        // The prefix_index now has an entry for it, so try_delta_store fires.
        let body2 = json!({"model":"m","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":[{"type":"text","text": base.clone()}]},
            {"role":"user","content":"follow-up question?"},
        ]});
        let _ = rewrite_request_body(&serde_json::to_vec(&body2).unwrap(), &state);

        assert!(state.stats.lock().unwrap().delta_stores_attempted > 0,
            "expected delta_stores_attempted > 0 with low delta_min_bytes");
    }

    // ── Feature: proxy tool call parsing (remember / recall) ─────────────────

    /// collect_proxy_tool_calls parses remember and recall tool_use blocks
    /// from a synthetic LLM response and returns the right ProxyCall variants.
    #[test]
    fn collect_proxy_calls_parses_remember_and_recall() {
        let resp = json!({
            "type": "message", "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "omc_proxy_remember",
                 "input": {"key": "mykey", "value": "stored content"}},
                {"type": "tool_use", "id": "tu2", "name": "omc_proxy_recall",
                 "input": {"key": "mykey"}},
            ]
        });
        let calls = collect_proxy_tool_calls(&resp);
        assert_eq!(calls.len(), 2, "must parse both calls");
        assert!(matches!(&calls[0], ProxyCall::Remember { key, .. } if key == "mykey"));
        assert!(matches!(&calls[1], ProxyCall::Recall { key, .. } if key == "mykey"));
    }

    /// Non-proxy tool_uses (e.g. bash) cause collect_proxy_tool_calls to
    /// return an empty vec so the response passes through to the client.
    #[test]
    fn non_proxy_tools_return_empty_vec() {
        let resp = json!({
            "type": "message", "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "bash",
                 "input": {"command": "ls"}},
            ]
        });
        assert!(collect_proxy_tool_calls(&resp).is_empty(),
            "non-proxy tool must not be intercepted");
    }

    /// Mixed proxy + non-proxy: the non-proxy call poisons the batch so the
    /// proxy passes the response through unchanged.
    #[test]
    fn mixed_proxy_and_non_proxy_tools_return_empty_vec() {
        let resp = json!({
            "type": "message", "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "tu1", "name": EXPAND_TOOL_NAME,
                 "input": {"hash_str": "12345"}},
                {"type": "tool_use", "id": "tu2", "name": "bash",
                 "input": {"command": "ls"}},
            ]
        });
        assert!(collect_proxy_tool_calls(&resp).is_empty(),
            "mixed batch must pass through so the client handles bash");
    }

    /// All three proxy tools are injected into outbound requests.
    #[test]
    fn all_proxy_tools_are_injected() {
        let mut req = json!({
            "model": "test", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        });
        inject_proxy_tools(&mut req);
        let tools = req["tools"].as_array().expect("tools array must exist");
        let names: Vec<&str> = tools.iter()
            .filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&EXPAND_TOOL_NAME),    "expand_ref must be injected");
        assert!(names.contains(&REMEMBER_TOOL_NAME),  "remember must be injected");
        assert!(names.contains(&RECALL_TOOL_NAME),    "recall must be injected");
        assert!(names.contains(&LIST_REFS_TOOL_NAME), "list_refs must be injected");
        assert!(names.contains(&FORWARD_TOOL_NAME),   "forward must be injected");
    }

    /// find_markers_in_value correctly discovers all <omc:ref> markers in a
    /// JSON tree (including inside nested arrays/objects) and deduplicates.
    #[test]
    fn list_refs_finds_markers_in_value() {
        let text_with_marker = format!(
            "Here is a compressed block: <omc:ref h=\"1234567\" b=\"4096\"/> and that's it.");
        let text_with_two = format!(
            "<omc:ref h=\"1234567\" b=\"4096\"/> and again <omc:ref h=\"9999999\" b=\"512\"/>");
        let val = json!({
            "messages": [
                {"role": "user", "content": text_with_marker.clone()},
                {"role": "assistant", "content": [
                    {"type": "text", "text": text_with_two.clone()}
                ]},
                // Duplicate of first marker — should be deduped
                {"role": "user", "content": text_with_marker.clone()},
            ]
        });
        let markers = find_markers_in_value(&val);
        assert_eq!(markers.len(), 2, "must find exactly 2 distinct hashes");
        let hashes: Vec<&str> = markers.iter()
            .filter_map(|m| m["hash"].as_str()).collect();
        assert!(hashes.contains(&"1234567"));
        assert!(hashes.contains(&"9999999"));
    }

    /// Repeated base64 image blocks in historical turns are replaced with a
    /// compact text note. The first occurrence passes through untouched.
    #[test]
    fn image_dedup_second_occurrence_compressed() {
        let state = test_state(64);
        let img_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ".repeat(20);
        let img_block = json!({
            "type": "image",
            "source": { "type": "base64", "media_type": "image/png", "data": img_data }
        });

        // Turn 1: single assistant message with the image — first occurrence passes through.
        let body1 = json!({"model":"m","messages":[
            {"role":"user","content":"look at this"},
            {"role":"assistant","content":[img_block.clone()]}
        ]});
        let (out1, _) = rewrite_request_body(&serde_json::to_vec(&body1).unwrap(), &state).unwrap();
        let v1: serde_json::Value = serde_json::from_slice(&out1).unwrap();
        let first_block = &v1["messages"][1]["content"][0];
        assert_eq!(first_block["type"].as_str().unwrap(), "image",
            "first occurrence must pass through as image block");

        // Turn 2: same image recurs in history — must be replaced with a text note.
        let body2 = json!({"model":"m","messages":[
            {"role":"user","content":"look at this"},
            {"role":"assistant","content":[img_block.clone()]},
            {"role":"user","content":"and now?"},
        ]});
        let (out2, outcome) = rewrite_request_body(&serde_json::to_vec(&body2).unwrap(), &state).unwrap();
        let v2: serde_json::Value = serde_json::from_slice(&out2).unwrap();
        let second_block = &v2["messages"][1]["content"][0];
        assert_eq!(second_block["type"].as_str().unwrap(), "text",
            "second occurrence must be replaced with text marker");
        assert!(second_block["text"].as_str().unwrap().contains("image repeated"),
            "text marker must mention 'image repeated'");
        assert!(outcome.bytes_images > 0,
            "bytes_images outcome must be positive");
    }
}
