//! Native LLM builtins: `llm_call`, `llm_chat`, `llm_embed`.
//!
//! These builtins close the self-improvement loop: any OMC program can call
//! Claude / GPT / Gemini (or any OpenAI-compatible endpoint) directly from
//! OMC source without leaving the interpreter.
//!
//! ## Environment variables
//!
//! | Var                  | Purpose                                      | Default           |
//! |----------------------|----------------------------------------------|-------------------|
//! | `LLM_API_KEY`        | API key (falls back to `ANTHROPIC_API_KEY`,  | — (required)      |
//! |                      |  `OPENAI_API_KEY`)                           |                   |
//! | `LLM_BASE_URL`       | Base URL for the completions endpoint        | see below         |
//! | `LLM_MODEL`          | Model identifier                             | see below         |
//! | `LLM_PROVIDER`       | `"anthropic"` \| `"openai"` (default)        | `"openai"`        |
//! | `LLM_EMBED_URL`      | Override just the embeddings endpoint        | —                 |
//! | `LLM_EMBED_MODEL`    | Override just the embedding model            | `"text-embedding-3-small"` |
//!
//! When `LLM_PROVIDER=anthropic`, `LLM_BASE_URL` defaults to
//! `https://api.anthropic.com/v1/messages` and the model to
//! `claude-3-5-haiku-20241022`.
//!
//! When `LLM_PROVIDER=openai` (or unset), `LLM_BASE_URL` defaults to
//! `https://api.openai.com/v1/chat/completions` and the model to
//! `gpt-4o-mini`.
//!
//! Point `LLM_BASE_URL` at `http://localhost:8088/v1/chat/completions` to
//! route through the OMC apiproxy for substrate-compressed context windows.
//!
//! ## Availability
//!
//! These functions compile only when the `native-llm` Cargo feature is
//! enabled (the default for all native targets).  WASM builds automatically
//! exclude this module via `--no-default-features`.

use crate::value::{HArray, HInt, Value};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Public entry points (called from interpreter.rs dispatch)
// ---------------------------------------------------------------------------

/// `llm_call(prompt: string, model?: string) -> string`
///
/// Send a single-turn prompt and return the assistant reply text.
/// `model` overrides `LLM_MODEL` for this call only.
#[cfg(feature = "native-llm")]
pub fn llm_call(prompt: &str, model_override: Option<&str>) -> Result<Value, String> {
    llm_call_sys(prompt, model_override, None)
}

#[cfg(feature = "native-llm")]
pub fn llm_call_sys(
    prompt: &str,
    model_override: Option<&str>,
    system: Option<&str>,
) -> Result<Value, String> {
    let cfg = Config::from_env()?;
    let model = model_override.unwrap_or(&cfg.model).to_string();
    let reply = if let Some(sys) = system {
        let msgs = vec![
            ChatMessage { role: "system".to_string(), content: sys.to_string() },
            ChatMessage { role: "user".to_string(), content: prompt.to_string() },
        ];
        cfg.provider.complete_chat(&cfg, &model, &msgs)?
    } else {
        cfg.provider.complete_single(&cfg, &model, prompt)?
    };
    Ok(Value::String(reply))
}

/// `llm_chat(messages: dict[], model?: string) -> string`
///
/// Multi-turn chat.  Each element of `messages` is a dict with at least
/// `"role"` (`"system"` | `"user"` | `"assistant"`) and `"content"` (string).
/// Returns the assistant reply text.
#[cfg(feature = "native-llm")]
pub fn llm_chat(messages: &[ChatMessage], model_override: Option<&str>) -> Result<Value, String> {
    let cfg = Config::from_env()?;
    let model = model_override.unwrap_or(&cfg.model).to_string();
    let reply = cfg.provider.complete_chat(&cfg, &model, messages)?;
    Ok(Value::String(reply))
}

/// `llm_system(prompt, system, model?) -> string`
///
/// Convenience wrapper for llm_call with a system prompt.
pub fn llm_system(
    prompt: &str,
    system: &str,
    model_override: Option<&str>,
) -> Result<Value, String> {
    llm_call_sys(prompt, model_override, Some(system))
}

/// `llm_stream_print(prompt, system?, model?) -> string`
///
/// Streams the LLM response token-by-token to stdout, then returns the full
/// accumulated text. Uses SSE streaming (stream:true). Supports both Anthropic
/// and OpenAI providers (auto-detected via LLM_PROVIDER env var).
#[cfg(feature = "native-llm")]
pub fn llm_stream_print(
    prompt: &str,
    system: Option<&str>,
    model_override: Option<&str>,
) -> Result<Value, String> {
    use std::io::{BufRead, BufReader, Write};

    let cfg = Config::from_env()?;
    let model = model_override.unwrap_or(&cfg.model).to_string();

    // Build messages list
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = system {
        if !sys.is_empty() {
            messages.push(ChatMessage { role: "system".to_string(), content: sys.to_string() });
        }
    }
    messages.push(ChatMessage { role: "user".to_string(), content: prompt.to_string() });

    match cfg.provider {
        Provider::Anthropic => {
            let mut system_parts: Vec<String> = Vec::new();
            let mut msgs_json: Vec<serde_json::Value> = Vec::new();
            for m in &messages {
                if m.role == "system" {
                    system_parts.push(m.content.clone());
                } else {
                    msgs_json.push(serde_json::json!({ "role": m.role, "content": m.content }));
                }
            }
            let mut body = serde_json::json!({
                "model": model, "max_tokens": 4096,
                "messages": msgs_json, "stream": true
            });
            if !system_parts.is_empty() {
                body["system"] = serde_json::Value::String(system_parts.join("\n\n"));
            }
            let resp = ureq::post(&cfg.base_url)
                .set("Content-Type", "application/json")
                .set("Authorization", &format!("Bearer {}", cfg.api_key))
                .set("anthropic-version", "2023-06-01")
                .set("x-api-key", &cfg.api_key)
                .send_json(body)
                .map_err(|e| format!("llm_stream HTTP error: {}", e))?;
            let reader = BufReader::new(resp.into_reader());
            let mut full_text = String::new();
            for line in reader.lines() {
                let line = line.map_err(|e| format!("llm_stream read error: {}", e))?;
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" { break; }
                    if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                        if event["type"] == "content_block_delta" {
                            if let Some(text) = event["delta"]["text"].as_str() {
                                print!("{}", text);
                                let _ = std::io::stdout().flush();
                                full_text.push_str(text);
                            }
                        }
                    }
                }
            }
            println!();
            Ok(Value::String(full_text))
        }
        Provider::OpenAI => {
            let msgs_json: Vec<serde_json::Value> = messages
                .iter()
                .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
                .collect();
            let body = serde_json::json!({
                "model": model, "messages": msgs_json, "stream": true
            });
            let resp = ureq::post(&cfg.base_url)
                .set("Content-Type", "application/json")
                .set("Authorization", &format!("Bearer {}", cfg.api_key))
                .send_json(body)
                .map_err(|e| format!("llm_stream HTTP error: {}", e))?;
            let reader = BufReader::new(resp.into_reader());
            let mut full_text = String::new();
            for line in reader.lines() {
                let line = line.map_err(|e| format!("llm_stream read error: {}", e))?;
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" { break; }
                    if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(text) = event["choices"][0]["delta"]["content"].as_str() {
                            print!("{}", text);
                            let _ = std::io::stdout().flush();
                            full_text.push_str(text);
                        }
                    }
                }
            }
            println!();
            Ok(Value::String(full_text))
        }
    }
}

#[cfg(not(feature = "native-llm"))]
pub fn llm_stream_print(
    _prompt: &str,
    _system: Option<&str>,
    _model_override: Option<&str>,
) -> Result<Value, String> {
    Err("llm_stream_print: recompile with --features native-llm".to_string())
}

/// `llm_judge(responses, criteria, model?) -> dict[]`
///
/// Scores each response (array of strings) against `criteria` and returns
/// an array of `{idx, score, reason}` dicts sorted best-first.
#[cfg(feature = "native-llm")]
pub fn llm_judge(
    responses: &Value,
    criteria: &str,
    model_override: Option<&str>,
) -> Result<Value, String> {
    let items = match responses {
        Value::Array(a) => a.items.borrow().clone(),
        _ => return Err("llm_judge: first arg must be an array of strings".to_string()),
    };

    let mut prompt = format!(
        "Score each response below (1-10) based on: {criteria}\n\
         Return ONLY JSON: [{{\"idx\":0,\"score\":8,\"reason\":\"...\"}}, ...]\n\n"
    );
    for (i, item) in items.iter().enumerate() {
        prompt.push_str(&format!("[{}]: {}\n---\n", i, item.to_display_string()));
    }

    let sys = "You are a precise evaluator. Output only valid JSON with no extra text.";
    let raw = llm_call_sys(&prompt, model_override, Some(sys))?;
    let text = match raw { Value::String(s) => s, _ => return Ok(Value::Array(HArray::from_vec(vec![]))) };

    // Find first '[' and scan for the array
    let bytes = text.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] == b'[' {
            for end in (start + 1..=bytes.len()).rev() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text[start..end]) {
                    return Ok(json_to_value(&v));
                }
            }
        }
    }
    Ok(Value::Array(HArray::from_vec(vec![])))
}

#[cfg(not(feature = "native-llm"))]
pub fn llm_judge(
    _responses: &Value,
    _criteria: &str,
    _model_override: Option<&str>,
) -> Result<Value, String> {
    Err("llm_judge: recompile with --features native-llm".to_string())
}

/// `llm_compare(a, b, criteria, model?) -> dict`
///
/// Compares two responses and returns `{winner: "A"|"B", reason: "..."}`.
#[cfg(feature = "native-llm")]
pub fn llm_compare(
    a: &str,
    b: &str,
    criteria: &str,
    model_override: Option<&str>,
) -> Result<Value, String> {
    let prompt = format!(
        "Compare these two responses based on: {criteria}\n\n\
         [A]: {a}\n\n[B]: {b}\n\n\
         Return ONLY JSON: {{\"winner\":\"A\",\"reason\":\"...\"}}"
    );
    let sys = "You are an impartial judge. Output only valid JSON.";
    let raw = llm_call_sys(&prompt, model_override, Some(sys))?;
    let text = match raw { Value::String(s) => s, _ => return Ok(Value::Null) };

    let bytes = text.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] == b'{' {
            for end in (start + 1..=bytes.len()).rev() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text[start..end]) {
                    return Ok(json_to_value(&v));
                }
            }
        }
    }
    Ok(Value::Null)
}

#[cfg(not(feature = "native-llm"))]
pub fn llm_compare(
    _a: &str,
    _b: &str,
    _criteria: &str,
    _model_override: Option<&str>,
) -> Result<Value, String> {
    Err("llm_compare: recompile with --features native-llm".to_string())
}

/// `batch_llm_call(prompts, model?, concurrency?) -> string[]`
///
/// Send multiple prompts to the LLM sequentially and return all responses in
/// the same order.  `prompts` may be either:
///   - an array of strings, or
///   - an array of dicts with keys `prompt` (required), `system` (optional),
///     and `model` (optional — overrides the function-level `model` arg).
///
/// `model` sets a default model for all calls; per-prompt dict entries take
/// precedence.  `concurrency` is accepted but currently ignored (calls are
/// sequential with a brief inter-call sleep to respect rate limits).
#[cfg(feature = "native-llm")]
pub fn batch_llm_call(
    prompts_val: &Value,
    default_model: Option<&str>,
    _concurrency: usize,
) -> Result<Value, String> {
    let items = match prompts_val {
        Value::Array(a) => a.items.borrow().clone(),
        _ => {
            return Err(
                "batch_llm_call: first argument must be an array of strings or dicts".to_string(),
            )
        }
    };

    let cfg = Config::from_env()?;

    let mut results: Vec<Value> = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let (prompt, sys_opt, model_str) = match item {
            Value::String(s) => (s.clone(), None::<String>, None::<String>),
            Value::Dict(d) => {
                let d = d.borrow();
                let prompt = d
                    .get("prompt")
                    .map(|v| v.to_display_string())
                    .ok_or_else(|| {
                        format!("batch_llm_call: prompts[{i}] dict missing 'prompt' key")
                    })?;
                let sys = d.get("system").map(|v| v.to_display_string());
                let model = d.get("model").map(|v| v.to_display_string());
                (prompt, sys, model)
            }
            _ => {
                return Err(format!(
                    "batch_llm_call: prompts[{i}] must be a string or dict"
                ))
            }
        };

        let model = model_str
            .as_deref()
            .or(default_model)
            .unwrap_or(&cfg.model);

        let mut messages: Vec<ChatMessage> = Vec::new();
        if let Some(sys) = &sys_opt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: prompt,
        });

        let reply = cfg.provider.complete_chat(&cfg, model, &messages)?;
        results.push(Value::String(reply));

        if i + 1 < items.len() {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    Ok(Value::Array(HArray::from_vec(results)))
}

/// `batch_llm_chat(messages_array, model?, concurrency?) -> string[]`
///
/// Send multiple chat conversations to the LLM sequentially.
/// `messages_array` is an array of arrays, where each inner array contains
/// the messages (dicts with `role` and `content`) for one chat call.
/// Returns an array of reply strings in the same order.
#[cfg(feature = "native-llm")]
pub fn batch_llm_chat(
    messages_array_val: &Value,
    default_model: Option<&str>,
    _concurrency: usize,
) -> Result<Value, String> {
    let outer = match messages_array_val {
        Value::Array(a) => a.items.borrow().clone(),
        _ => {
            return Err(
                "batch_llm_chat: first argument must be an array of message arrays".to_string(),
            )
        }
    };

    let cfg = Config::from_env()?;
    let model = default_model.unwrap_or(&cfg.model);

    let mut results: Vec<Value> = Vec::with_capacity(outer.len());
    for (i, inner_val) in outer.iter().enumerate() {
        let messages = parse_messages(inner_val)
            .map_err(|e| format!("batch_llm_chat: messages_array[{i}]: {e}"))?;
        let reply = cfg.provider.complete_chat(&cfg, model, &messages)?;
        results.push(Value::String(reply));

        if i + 1 < outer.len() {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    Ok(Value::Array(HArray::from_vec(results)))
}

/// `llm_embed(text: string, model?: string) -> float[]`
///
/// Embed `text` and return the embedding vector as an OMC float array.
/// Always uses the OpenAI embeddings endpoint (`LLM_EMBED_URL` /
/// `LLM_EMBED_MODEL`) regardless of `LLM_PROVIDER`.
#[cfg(feature = "native-llm")]
pub fn llm_embed(text: &str, model_override: Option<&str>) -> Result<Value, String> {
    let api_key = api_key()?;
    let embed_url = std::env::var("LLM_EMBED_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1/embeddings".to_string());
    let embed_model = model_override
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            std::env::var("LLM_EMBED_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string())
        });

    let body = serde_json::json!({
        "model": embed_model,
        "input": text,
    });

    let resp: serde_json::Value = post_json(&embed_url, &api_key, None, body)?;

    let floats = resp["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| "llm_embed: unexpected response — no embedding array".to_string())?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| "llm_embed: non-numeric value in embedding".to_string())
                .map(Value::HFloat)
        })
        .collect::<Result<Vec<Value>, String>>()?;

    Ok(Value::Array(HArray::from_vec(floats)))
}

// ---------------------------------------------------------------------------
// Helpers exposed to interpreter.rs for argument parsing
// ---------------------------------------------------------------------------

/// Parse a `Value` (must be `Array` of `Dict`s) into `Vec<ChatMessage>`.
#[cfg(feature = "native-llm")]
pub fn parse_messages(v: &Value) -> Result<Vec<ChatMessage>, String> {
    let arr = match v {
        Value::Array(a) => a.items.borrow().clone(),
        _ => return Err("llm_chat: first argument must be an array of message dicts".to_string()),
    };

    arr.iter()
        .enumerate()
        .map(|(i, item)| {
            let dict = match item {
                Value::Dict(d) => d.borrow().clone(),
                _ => {
                    return Err(format!(
                        "llm_chat: messages[{i}] must be a dict with 'role' and 'content'"
                    ))
                }
            };
            let role = dict
                .get("role")
                .map(|v| v.to_display_string())
                .unwrap_or_else(|| "user".to_string());
            let content = dict
                .get("content")
                .map(|v| v.to_display_string())
                .unwrap_or_default();
            Ok(ChatMessage { role, content })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A single chat message (role + content text).
#[cfg(feature = "native-llm")]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Copy, PartialEq)]
#[cfg(feature = "native-llm")]
enum Provider {
    OpenAI,
    Anthropic,
}

#[cfg(feature = "native-llm")]
struct Config {
    provider: Provider,
    base_url: String,
    model: String,
    api_key: String,
}

#[cfg(feature = "native-llm")]
impl Config {
    fn from_env() -> Result<Self, String> {
        // Provider: explicit LLM_PROVIDER wins; otherwise infer from key env vars.
        let explicit_provider = std::env::var("LLM_PROVIDER")
            .unwrap_or_default()
            .to_lowercase();
        let provider = match explicit_provider.as_str() {
            "anthropic" => Provider::Anthropic,
            "openai"    => Provider::OpenAI,
            _ => {
                // Auto-detect: if any Anthropic-specific var is set, use Anthropic.
                let has_anthropic_key =
                    std::env::var("ANTHROPIC_API_KEY").map(|v| !v.is_empty()).unwrap_or(false)
                    || std::env::var("ANTHROPIC_AUTH_TOKEN").map(|v| !v.is_empty()).unwrap_or(false);
                if has_anthropic_key { Provider::Anthropic } else { Provider::OpenAI }
            }
        };

        let (default_url, default_model) = match provider {
            Provider::Anthropic => (
                "https://api.anthropic.com/v1/messages".to_string(),
                "claude-haiku-4-5".to_string(),
            ),
            Provider::OpenAI => (
                "https://api.openai.com/v1/chat/completions".to_string(),
                "gpt-4o-mini".to_string(),
            ),
        };

        // Base URL: LLM_BASE_URL > ANTHROPIC_BASE_URL > default.
        let base_url = std::env::var("LLM_BASE_URL")
            .or_else(|_| std::env::var("ANTHROPIC_BASE_URL"))
            .unwrap_or(default_url);

        // Append /v1/messages for Anthropic when the base URL ends with the proxy host.
        let base_url = if provider == Provider::Anthropic
            && !base_url.contains("/messages")
            && !base_url.ends_with("/v1/chat/completions")
        {
            format!("{}/v1/messages", base_url.trim_end_matches('/'))
        } else {
            base_url
        };

        let model = std::env::var("LLM_MODEL").unwrap_or(default_model);
        let api_key = api_key()?;

        Ok(Config { provider, base_url, model, api_key })
    }
}

#[cfg(feature = "native-llm")]
impl Provider {
    fn complete_single(
        self,
        cfg: &Config,
        model: &str,
        prompt: &str,
    ) -> Result<String, String> {
        let msg = ChatMessage { role: "user".to_string(), content: prompt.to_string() };
        self.complete_chat(cfg, model, &[msg])
    }

    fn complete_chat(
        self,
        cfg: &Config,
        model: &str,
        messages: &[ChatMessage],
    ) -> Result<String, String> {
        match self {
            Provider::OpenAI => complete_openai(cfg, model, messages),
            Provider::Anthropic => complete_anthropic(cfg, model, messages),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible chat completions
// ---------------------------------------------------------------------------

#[cfg(feature = "native-llm")]
fn complete_openai(
    cfg: &Config,
    model: &str,
    messages: &[ChatMessage],
) -> Result<String, String> {
    let msgs_json: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();

    let body = serde_json::json!({
        "model": model,
        "messages": msgs_json,
    });

    let resp: serde_json::Value = post_json(&cfg.base_url, &cfg.api_key, None, body)?;

    resp["choices"][0]["message"]["content"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| {
            format!(
                "llm_call: unexpected OpenAI response — no content: {}",
                resp
            )
        })
}

// ---------------------------------------------------------------------------
// Anthropic messages API
// ---------------------------------------------------------------------------

#[cfg(feature = "native-llm")]
fn complete_anthropic(
    cfg: &Config,
    model: &str,
    messages: &[ChatMessage],
) -> Result<String, String> {
    // Anthropic requires system messages to be a top-level field, not in messages[].
    let mut system_parts: Vec<String> = Vec::new();
    let mut msgs_json: Vec<serde_json::Value> = Vec::new();

    for m in messages {
        if m.role == "system" {
            system_parts.push(m.content.clone());
        } else {
            msgs_json.push(serde_json::json!({ "role": m.role, "content": m.content }));
        }
    }

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": msgs_json,
    });

    if !system_parts.is_empty() {
        body["system"] = serde_json::Value::String(system_parts.join("\n\n"));
    }

    // Anthropic-specific headers: anthropic-version, x-api-key
    let extra = Some(vec![
        ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ("x-api-key".to_string(), cfg.api_key.clone()),
    ]);

    let resp: serde_json::Value = post_json(&cfg.base_url, &cfg.api_key, extra, body)?;

    resp["content"][0]["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| {
            format!(
                "llm_call: unexpected Anthropic response — no text: {}",
                resp
            )
        })
}

// ---------------------------------------------------------------------------
// HTTP POST helper (ureq, feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "native-llm")]
fn post_json(
    url: &str,
    bearer_token: &str,
    extra_headers: Option<Vec<(String, String)>>,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut req = ureq::post(url)
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", bearer_token));

    if let Some(headers) = extra_headers {
        for (k, v) in &headers {
            req = req.set(k, v);
        }
    }

    let resp = req
        .send_json(body)
        .map_err(|e| format!("llm HTTP error: {}", e))?;

    let status = resp.status();
    let body_str = resp
        .into_string()
        .map_err(|e| format!("llm read body error: {}", e))?;

    if status < 200 || status >= 300 {
        return Err(format!("llm API error (HTTP {}): {}", status, body_str));
    }

    serde_json::from_str(&body_str)
        .map_err(|e| format!("llm JSON parse error: {} — body: {}", e, body_str))
}

// ---------------------------------------------------------------------------
// API key resolution
// ---------------------------------------------------------------------------

#[cfg(feature = "native-llm")]
fn api_key() -> Result<String, String> {
    for var in &[
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",  // Claude Code / apiproxy token
    ] {
        if let Ok(k) = std::env::var(var) {
            if !k.is_empty() {
                return Ok(k);
            }
        }
    }
    Err(
        "llm_call/llm_chat/llm_embed: no API key found. \
         Set LLM_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN)."
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Build a Value::Dict for an llm_models() response entry
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn model_entry(id: &str, provider: &str) -> Value {
    let mut m = BTreeMap::new();
    m.insert("id".to_string(), Value::String(id.to_string()));
    m.insert("provider".to_string(), Value::String(provider.to_string()));
    Value::dict_from(m)
}

/// `llm_models() -> dict[]` — return a static list of well-known model ids.
pub fn llm_models() -> Value {
    let entries: Vec<Value> = [
        // OpenAI
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o1-mini", "openai"),
        ("o1-preview", "openai"),
        // Anthropic
        ("claude-opus-4-5", "anthropic"),
        ("claude-sonnet-4-5", "anthropic"),
        ("claude-3-5-haiku-20241022", "anthropic"),
    ]
    .iter()
    .map(|(id, prov)| model_entry(id, prov))
    .collect();

    Value::Array(HArray::from_vec(entries))
}

// ---------------------------------------------------------------------------
// Return-value helpers: build HInt for success/failure sentinel
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn ok_flag() -> Value {
    Value::HInt(HInt::new(1))
}

// ---------------------------------------------------------------------------
// llm_tools — structured tool-calling (function calling)
// ---------------------------------------------------------------------------

/// Convert an OMC `Value` to a `serde_json::Value` for embedding in API requests.
#[cfg(feature = "native-llm")]
fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::HInt(n) => serde_json::json!(n.value),
        Value::HFloat(f) => serde_json::json!(f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(a) => {
            serde_json::Value::Array(a.items.borrow().iter().map(value_to_json).collect())
        }
        Value::Dict(d) => {
            let map: serde_json::Map<String, serde_json::Value> =
                d.borrow().iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect();
            serde_json::Value::Object(map)
        }
        _ => serde_json::Value::Null,
    }
}

/// Convert a `serde_json::Value` back to an OMC `Value`.
#[cfg(feature = "native-llm")]
fn json_to_value(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::HInt(HInt::new(i))
            } else {
                Value::HFloat(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Value::Array(HArray::from_vec(arr.iter().map(json_to_value).collect()))
        }
        serde_json::Value::Object(map) => {
            let mut btree = BTreeMap::new();
            for (k, v) in map {
                btree.insert(k.clone(), json_to_value(v));
            }
            Value::dict_from(btree)
        }
    }
}

/// `llm_tools(messages, tools, model?) -> dict`
///
/// Send messages with tool definitions. Returns a dict:
/// - `{type: "text", content: "...", stop_reason: "end_turn"}` for text responses
/// - `{type: "tool_use", id: "...", name: "...", input: {...}, stop_reason: "tool_use"}` for tool calls
///
/// Tool definitions are dicts with: `{name, description, parameters}` where
/// `parameters` is a JSON Schema object.
#[cfg(feature = "native-llm")]
pub fn llm_tools(
    messages_val: &Value,
    tools_val: &Value,
    model_override: Option<&str>,
) -> Result<Value, String> {
    let cfg = Config::from_env()?;
    let model = model_override.unwrap_or(&cfg.model).to_string();
    let messages = parse_messages(messages_val)?;

    let tools_json: Vec<serde_json::Value> = match tools_val {
        Value::Array(a) => a.items.borrow().iter().map(value_to_json).collect(),
        _ => return Err("llm_tools: second arg must be array of tool dicts".to_string()),
    };

    match cfg.provider {
        Provider::Anthropic => complete_anthropic_tools(&cfg, &model, &messages, &tools_json),
        Provider::OpenAI => complete_openai_tools(&cfg, &model, &messages, &tools_json),
    }
}

#[cfg(feature = "native-llm")]
fn complete_anthropic_tools(
    cfg: &Config,
    model: &str,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> Result<Value, String> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut msgs_json: Vec<serde_json::Value> = Vec::new();
    for m in messages {
        if m.role == "system" {
            system_parts.push(m.content.clone());
        } else {
            msgs_json.push(serde_json::json!({"role": m.role, "content": m.content}));
        }
    }

    // Anthropic uses `input_schema` instead of `parameters`
    let anthropic_tools: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            let mut tool = t.clone();
            if let Some(obj) = tool.as_object_mut() {
                if let Some(params) = obj.remove("parameters") {
                    obj.insert("input_schema".to_string(), params);
                }
            }
            tool
        })
        .collect();

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": msgs_json,
        "tools": anthropic_tools,
    });
    if !system_parts.is_empty() {
        body["system"] = serde_json::Value::String(system_parts.join("\n\n"));
    }

    let extra = Some(vec![
        ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ("x-api-key".to_string(), cfg.api_key.clone()),
    ]);
    let resp: serde_json::Value = post_json(&cfg.base_url, &cfg.api_key, extra, body)?;

    let stop_reason = resp["stop_reason"].as_str().unwrap_or("end_turn").to_string();
    let content = resp["content"].as_array().ok_or_else(|| {
        format!("llm_tools: unexpected Anthropic response: {}", resp)
    })?;

    let mut result = BTreeMap::new();
    result.insert("stop_reason".to_string(), Value::String(stop_reason.clone()));

    if stop_reason == "tool_use" {
        for block in content {
            if block["type"].as_str() == Some("tool_use") {
                result.insert("type".to_string(), Value::String("tool_use".to_string()));
                result.insert(
                    "id".to_string(),
                    Value::String(block["id"].as_str().unwrap_or("").to_string()),
                );
                result.insert(
                    "name".to_string(),
                    Value::String(block["name"].as_str().unwrap_or("").to_string()),
                );
                result.insert("input".to_string(), json_to_value(&block["input"]));
                result.insert("content".to_string(), Value::Null);
                return Ok(Value::dict_from(result));
            }
        }
    }

    let text = content
        .iter()
        .find_map(|b| {
            if b["type"].as_str() == Some("text") {
                b["text"].as_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .unwrap_or_default();

    result.insert("type".to_string(), Value::String("text".to_string()));
    result.insert("content".to_string(), Value::String(text));
    result.insert("name".to_string(), Value::Null);
    result.insert("id".to_string(), Value::Null);
    result.insert("input".to_string(), Value::Null);
    Ok(Value::dict_from(result))
}

#[cfg(feature = "native-llm")]
fn complete_openai_tools(
    cfg: &Config,
    model: &str,
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> Result<Value, String> {
    let msgs_json: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();

    let openai_tools: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| serde_json::json!({"type": "function", "function": t}))
        .collect();

    let body = serde_json::json!({
        "model": model,
        "messages": msgs_json,
        "tools": openai_tools,
    });

    let resp: serde_json::Value = post_json(&cfg.base_url, &cfg.api_key, None, body)?;
    let choice = &resp["choices"][0];
    let finish_reason = choice["finish_reason"].as_str().unwrap_or("stop").to_string();
    let message = &choice["message"];

    let mut result = BTreeMap::new();
    result.insert("stop_reason".to_string(), Value::String(finish_reason.clone()));

    if finish_reason == "tool_calls" {
        if let Some(calls) = message["tool_calls"].as_array() {
            if let Some(call) = calls.first() {
                let fn_obj = &call["function"];
                let name = fn_obj["name"].as_str().unwrap_or("").to_string();
                let args_str = fn_obj["arguments"].as_str().unwrap_or("{}");
                let input: serde_json::Value =
                    serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                result.insert("type".to_string(), Value::String("tool_use".to_string()));
                result.insert(
                    "id".to_string(),
                    Value::String(call["id"].as_str().unwrap_or("").to_string()),
                );
                result.insert("name".to_string(), Value::String(name));
                result.insert("input".to_string(), json_to_value(&input));
                result.insert("content".to_string(), Value::Null);
                return Ok(Value::dict_from(result));
            }
        }
    }

    let text = message["content"].as_str().unwrap_or("").to_string();
    result.insert("type".to_string(), Value::String("text".to_string()));
    result.insert("content".to_string(), Value::String(text));
    result.insert("name".to_string(), Value::Null);
    result.insert("id".to_string(), Value::Null);
    result.insert("input".to_string(), Value::Null);
    Ok(Value::dict_from(result))
}

// ---------------------------------------------------------------------------
// substrate_embed — phi-pi-fib harmonic text embedding (no API call)
// ---------------------------------------------------------------------------

/// `substrate_embed(text, dims?) -> float[]`
///
/// Creates a harmonic embedding of `text` using Fibonacci-indexed phi/pi
/// frequencies. Returns an L2-normalised float array of length `dims`
/// (default 16). Works offline — no API key required.
#[cfg(feature = "native-llm")]
pub fn substrate_embed(text: &str, dims: usize) -> Value {
    const PHI: f64 = 1.618_033_988_749_895;
    let pi = std::f64::consts::PI;

    let mut fibs: Vec<f64> = vec![1.0, 1.0];
    while fibs.len() < dims + 2 {
        let n = fibs.len();
        fibs.push(fibs[n - 1] + fibs[n - 2]);
    }

    let mut vec = vec![0.0f64; dims];
    for ch in text.chars() {
        let c = ch as u32 as f64;
        for (j, v) in vec.iter_mut().enumerate() {
            let freq = fibs[j] * PHI / pi;
            *v += (freq * c).sin();
        }
    }

    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-9 {
        for x in &mut vec {
            *x /= norm;
        }
    }

    Value::Array(HArray::from_vec(vec.into_iter().map(Value::HFloat).collect()))
}

// ---------------------------------------------------------------------------
// Stubs when `native-llm` feature is disabled (WASM / embedded builds)
// ---------------------------------------------------------------------------

/// Stub: `llm_call` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn llm_call(_prompt: &str, _model_override: Option<&str>) -> Result<Value, String> {
    Err("llm_call: recompile with --features native-llm".to_string())
}

#[cfg(not(feature = "native-llm"))]
pub fn llm_call_sys(
    _prompt: &str,
    _model_override: Option<&str>,
    _system: Option<&str>,
) -> Result<Value, String> {
    Err("llm_call: recompile with --features native-llm".to_string())
}

/// Stub: `llm_chat` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn llm_chat(_messages: &[ChatMessage], _model_override: Option<&str>) -> Result<Value, String> {
    Err("llm_chat: recompile with --features native-llm".to_string())
}

/// Stub: `llm_embed` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn llm_embed(_text: &str, _model_override: Option<&str>) -> Result<Value, String> {
    Err("llm_embed: recompile with --features native-llm".to_string())
}

/// Stub `ChatMessage` so the interpreter dispatch can refer to it in both cfg paths.
#[cfg(not(feature = "native-llm"))]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Stub: `parse_messages` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn parse_messages(_v: &Value) -> Result<Vec<ChatMessage>, String> {
    Err("parse_messages: recompile with --features native-llm".to_string())
}

/// Stub: `batch_llm_call` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn batch_llm_call(
    _prompts_val: &Value,
    _default_model: Option<&str>,
    _concurrency: usize,
) -> Result<Value, String> {
    Err("batch_llm_call: recompile with --features native-llm".to_string())
}

/// Stub: `batch_llm_chat` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn batch_llm_chat(
    _messages_array_val: &Value,
    _default_model: Option<&str>,
    _concurrency: usize,
) -> Result<Value, String> {
    Err("batch_llm_chat: recompile with --features native-llm".to_string())
}

/// Stub: `llm_tools` requires the `native-llm` Cargo feature.
#[cfg(not(feature = "native-llm"))]
pub fn llm_tools(
    _messages_val: &Value,
    _tools_val: &Value,
    _model_override: Option<&str>,
) -> Result<Value, String> {
    Err("llm_tools: recompile with --features native-llm".to_string())
}

/// `substrate_embed` is always available (no API call, pure math).
#[cfg(not(feature = "native-llm"))]
pub fn substrate_embed(text: &str, dims: usize) -> Value {
    const PHI: f64 = 1.618_033_988_749_895;
    let pi = std::f64::consts::PI;
    let mut fibs: Vec<f64> = vec![1.0, 1.0];
    while fibs.len() < dims + 2 {
        let n = fibs.len();
        fibs.push(fibs[n - 1] + fibs[n - 2]);
    }
    let mut vec = vec![0.0f64; dims];
    for ch in text.chars() {
        let c = ch as u32 as f64;
        for (j, v) in vec.iter_mut().enumerate() {
            let freq = fibs[j] * PHI / pi;
            *v += (freq * c).sin();
        }
    }
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-9 {
        for x in &mut vec {
            *x /= norm;
        }
    }
    Value::Array(HArray::from_vec(vec.into_iter().map(Value::HFloat).collect()))
}
