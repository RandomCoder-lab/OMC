# Task 01: llm_call + llm_chat Native Builtins

## Goal
Add `llm_call`, `llm_chat`, and `llm_embed` as native builtins to the OMC interpreter.
This is the most critical feature for the recursive self-improvement loop.

## Repo
/home/thearchitect/OMC — you are in a git worktree. Commit your work, do not push.

## Context
- Core interpreter: omnimcode-core/src/interpreter.rs (~14100 lines)
- Cargo.toml already has: `ureq = { version = "2", features = ["json", "tls"], optional = true }`
  and `native-llm = ["dep:ureq"]` feature (enabled by default). See omnimcode-core/Cargo.toml.
- Pattern: grep for `json_parse` in interpreter.rs to see how builtins work.
  - Line ~2219: builtin name guard (`| "json_parse" | "json_stringify"`)
  - Line ~3482: implementation (`"json_parse" => { ... }`)
  - Line ~13717: ALL_BUILTINS list

## Step 1: Check Cargo.toml
Read omnimcode-core/Cargo.toml. Confirm ureq is there with native-llm feature.
If not, add it yourself.

## Step 2: Add feature gate import at top of interpreter.rs
Find where other `#[cfg(feature = ...)]` or `use` statements are near the top.
Add (inside the file, not at module level if there are conflicts):
```rust
#[cfg(feature = "native-llm")]
use ureq;
```
Actually, ureq doesn't need a use statement — just call ureq::post() directly.

## Step 3: Add to builtin match guard
Find the line with `| "json_parse" | "json_stringify"` and add to that same block:
```
| "llm_call" | "llm_chat" | "llm_embed"
```

## Step 4: Add implementations after json_stringify block (~line 3515)
Find where json_stringify ends and insert:

```rust
#[cfg(feature = "native-llm")]
"llm_call" => {
    // llm_call(prompt, model?, system?) -> String
    if args.is_empty() {
        return Err("llm_call requires (prompt, model?, system_prompt?)".to_string());
    }
    let prompt = match &args[0] {
        Value::String(s) => s.clone(),
        other => other.to_display_string(),
    };
    let model = args.get(1)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::var("OMC_LLM_MODEL")
            .unwrap_or_else(|_| "claude-opus-4-5".to_string()));
    let system = args.get(2)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "llm_call: ANTHROPIC_API_KEY env var not set".to_string())?;

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    });
    if let Some(sys) = system {
        body["system"] = serde_json::Value::String(sys);
    }

    let resp = ureq::post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", &api_key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(body)
        .map_err(|e| format!("llm_call: HTTP error: {}", e))?;
    let resp_json: serde_json::Value = resp.into_json()
        .map_err(|e| format!("llm_call: JSON parse error: {}", e))?;
    let text = resp_json["content"][0]["text"].as_str()
        .ok_or_else(|| format!("llm_call: unexpected response: {}", resp_json))?;
    Ok(Value::String(text.to_string()))
}
#[cfg(not(feature = "native-llm"))]
"llm_call" => Err("llm_call: build with native-llm feature (enabled by default)".to_string()),

#[cfg(feature = "native-llm")]
"llm_chat" => {
    // llm_chat(messages_array, model?, system?) -> String
    // messages_array: array of [role, content] pairs or array of dicts {role, content}
    if args.is_empty() {
        return Err("llm_chat requires (messages, model?, system?)".to_string());
    }
    let messages = match &args[0] {
        Value::Array(a) => {
            let arr = a.borrow();
            let mut msgs = Vec::new();
            for item in arr.iter() {
                match item {
                    Value::Array(pair) => {
                        let p = pair.borrow();
                        if p.len() >= 2 {
                            let role = p[0].to_display_string();
                            let content = p[1].to_display_string();
                            msgs.push(serde_json::json!({"role": role, "content": content}));
                        }
                    }
                    Value::Dict(d) => {
                        let d = d.borrow();
                        let role = d.get("role").map(|v| v.to_display_string()).unwrap_or_else(|| "user".to_string());
                        let content = d.get("content").map(|v| v.to_display_string()).unwrap_or_default();
                        msgs.push(serde_json::json!({"role": role, "content": content}));
                    }
                    _ => {}
                }
            }
            msgs
        }
        _ => return Err("llm_chat: first argument must be an array of messages".to_string()),
    };
    let model = args.get(1)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| std::env::var("OMC_LLM_MODEL")
            .unwrap_or_else(|_| "claude-opus-4-5".to_string()));
    let system = args.get(2)
        .and_then(|v| if let Value::String(s) = v { Some(s.clone()) } else { None });

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "llm_chat: ANTHROPIC_API_KEY env var not set".to_string())?;

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": messages
    });
    if let Some(sys) = system {
        body["system"] = serde_json::Value::String(sys);
    }

    let resp = ureq::post("https://api.anthropic.com/v1/messages")
        .set("x-api-key", &api_key)
        .set("anthropic-version", "2023-06-01")
        .set("content-type", "application/json")
        .send_json(body)
        .map_err(|e| format!("llm_chat: HTTP error: {}", e))?;
    let resp_json: serde_json::Value = resp.into_json()
        .map_err(|e| format!("llm_chat: JSON parse error: {}", e))?;
    let text = resp_json["content"][0]["text"].as_str()
        .ok_or_else(|| format!("llm_chat: unexpected response: {}", resp_json))?;
    Ok(Value::String(text.to_string()))
}
#[cfg(not(feature = "native-llm"))]
"llm_chat" => Err("llm_chat: build with native-llm feature".to_string()),

"llm_embed" => {
    // llm_embed(text) -> Array of floats (placeholder - returns substrate embedding)
    // Uses phi-pi-fib substrate encoding as a free local embedding
    if args.is_empty() {
        return Err("llm_embed requires (text)".to_string());
    }
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        other => other.to_display_string(),
    };
    // Use sha256 to create a deterministic 128-dim embedding
    // (real embedding requires an API call to text-embedding-3 or similar)
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    let mut embedding = Vec::with_capacity(128);
    for i in 0..128 {
        let byte_idx = i % 32;
        let bit = (hash[byte_idx] >> (i % 8)) & 1;
        // Use phi/pi/fib weighting for substrate-aware embedding
        let phi = 1.6180339887_f64;
        let val = (bit as f64 * 2.0 - 1.0) * phi.powi((i as i32 % 8) - 4);
        embedding.push(Value::Float(val));
    }
    Ok(Value::Array(Rc::new(RefCell::new(embedding))))
}
```

## Step 5: Add to ALL_BUILTINS
Find the ALL_BUILTINS array near line 13717. Add:
```
"llm_call", "llm_chat", "llm_embed",
```

## Step 6: Fix any Rc/RefCell imports if needed
The impl uses `Rc::new(RefCell::new(...))`. Check if `use std::rc::Rc; use std::cell::RefCell;`
are already imported at the top of interpreter.rs (they almost certainly are).

## Step 7: Build
```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core
```
Fix any compiler errors. Common issues:
- Missing `use sha2::Digest;` — check if sha2 is already used elsewhere
- ureq version API differences — check ureq 2.x docs if needed
- `#[cfg]` attributes on match arms may not work — wrap the match arm body in cfg instead

## Step 8: Write test file
Save as examples/test_llm_call.omc:
```
# Test llm_call builtin
# Set ANTHROPIC_API_KEY before running
h reply = llm_call("What is 2+2? Just say the number, nothing else.")
print(reply)

# Test llm_chat
h msgs = [["user", "Hello!"], ["assistant", "Hi there!"], ["user", "What is 3+3?"]]
h reply2 = llm_chat(msgs)
print(reply2)
```

## Step 9: Commit
```bash
git add -A
git commit -m "feat: llm_call + llm_chat + llm_embed native builtins via ureq"
```
