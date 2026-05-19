# Task 33: Native HTTP builtins — http_get, http_post, http_put, http_delete

## Goal
Add 4 native HTTP builtins to the OMC interpreter using the already-available `ureq` crate.
ureq is already a dependency (feature `native-llm`) and is available in the workspace.

## Working directory
/home/thearchitect/OMC

## What already exists
- `omnimcode-core/src/llm_builtins.rs` — already uses ureq for HTTP in llm_call
- ureq 2.x is in Cargo.toml under `native-llm` feature
- `omnimcode-core/src/interpreter.rs` — the main interpreter, ~14000+ lines

## Functions to add

### 1. http_get(url, headers?)
- url: string
- headers: optional dict {header_name: header_value}
- returns: dict {status: int, body: string, ok: bool}

### 2. http_post(url, body, headers?)
- url: string  
- body: string (raw body)
- headers: optional dict
- returns: dict {status: int, body: string, ok: bool}

### 3. http_post_json(url, data, headers?)
- url: string
- data: dict or array (will be json_stringify'd)
- headers: optional dict  
- returns: dict {status: int, body: string, json: parsed_json_or_null, ok: bool}

### 4. http_put(url, body, headers?)
- Same signature as http_post but uses PUT

### 5. http_delete(url, headers?)
- url: string
- headers: optional dict
- returns: dict {status: int, body: string, ok: bool}

## Implementation steps

### Step 1: Add to llm_builtins.rs (or create http_builtins.rs)
Add a new file `omnimcode-core/src/http_builtins.rs` with these functions:

```rust
use crate::interpreter::Value;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

pub fn http_get(args: &[Value]) -> Result<Value, String> {
    // args[0] = url string
    // args[1] = optional headers dict
    // Use ureq::get(url).call()
    // Return Value::Dict with status, body, ok
}

pub fn http_post(args: &[Value]) -> Result<Value, String> { ... }
pub fn http_post_json(args: &[Value]) -> Result<Value, String> { ... }
pub fn http_put(args: &[Value]) -> Result<Value, String> { ... }
pub fn http_delete(args: &[Value]) -> Result<Value, String> { ... }
```

### Step 2: Wire into interpreter.rs

In the builtin match guard (around line 2219 where "llm_call" | "llm_chat" are listed), add:
```
| "http_get" | "http_post" | "http_post_json" | "http_put" | "http_delete"
```

Add dispatch arms that call `crate::http_builtins::http_get(args)` etc.

### Step 3: Add to ALL_BUILTINS and HEAL_BUILTIN_NAMES

Search for where `"llm_call"` is listed in the ALL_BUILTINS array or similar, add the new names there too.

### Step 4: Add to docs.rs

Add a category "http" with doc entries for each function.

### Step 5: Write a test

Create `examples/test_http.omc`:
```omc
h result = http_get("https://httpbin.org/get", null)
print(result["status"])
print(result["ok"])

h post_result = http_post_json("https://httpbin.org/post", {key: "value"}, null)
print(post_result["status"])
```

## Important ureq patterns

```rust
// GET
let resp = ureq::get(&url)
    .call()
    .map_err(|e| format!("http_get failed: {e}"))?;
let status = resp.status();
let body = resp.into_string().map_err(|e| format!("read body: {e}"))?;

// POST with JSON body
let resp = ureq::post(&url)
    .set("Content-Type", "application/json")
    .send_string(&json_body)
    .map_err(|e| format!("http_post failed: {e}"))?;

// Add custom headers
let mut req = ureq::get(&url);
for (k, v) in &headers {
    req = req.set(k, v);
}
```

## Value dict construction pattern (from existing llm_builtins.rs)

```rust
fn make_response_dict(status: u16, body: String) -> Value {
    let mut map = HashMap::new();
    map.insert("status".to_string(), Value::Int(status as i64));
    map.insert("body".to_string(), Value::Str(body));
    map.insert("ok".to_string(), Value::Bool(status >= 200 && status < 300));
    Value::Dict(Rc::new(RefCell::new(map)))
}
```

## After implementing

Run:
```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core 2>&1 | tail -20
```

Fix any compile errors. When it builds clean, commit:
```bash
git add omnimcode-core/src/http_builtins.rs omnimcode-core/src/interpreter.rs omnimcode-core/src/docs.rs omnimcode-core/src/lib.rs examples/test_http.omc
git commit -m "feat: http_get + http_post + http_post_json + http_put + http_delete native HTTP builtins"
```

## DO NOT
- Do not use async/tokio — ureq is synchronous
- Do not change Cargo.toml (ureq is already there)
- Do not launch sub-agents
- Do not modify any other files beyond what's listed above
