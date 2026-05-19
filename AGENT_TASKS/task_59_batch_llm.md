# Task 59: batch_llm_call — parallel/sequential batch LLM calls

## Goal
Add `batch_llm_call` builtin to the OMC interpreter.
This enables sending many prompts to Claude in parallel (or sequential with rate limiting),
critical for the agent swarm and population evolution demos.

## Working directory
/home/thearchitect/OMC

## Function to add

### batch_llm_call(prompts, model?, concurrency?) → array of strings

Parameters:
- prompts: array of strings OR array of {prompt, system?, model?} dicts
- model: string (optional) — model to use for all (default: claude-opus-4-5)
- concurrency: int (optional, default 3) — max parallel requests

Returns: array of strings (responses, same order as input prompts)

### batch_llm_chat(messages_array, model?, concurrency?) → array of strings

Parameters:
- messages_array: array of arrays — each inner array is the messages for one chat call
- model: string (optional)
- concurrency: int (optional)

Returns: array of strings

## Implementation strategy

Since Value uses Rc/RefCell (not Send), true parallelism via rayon/threads is not directly possible.
However, we can:

1. **Sequential with progress**: Run each call in sequence, which is what we do now but wrapped nicely
2. **Rayon on strings**: Convert all prompts to Vec<String>, run rayon parallel on strings, collect results as Vec<String>, then convert back to Value::Array

Option 2 is the right approach — work on raw Strings (which ARE Send), then convert results back.

```rust
pub fn batch_llm_call(args: &[Value]) -> Result<Value, String> {
    let prompts: Vec<(String, Option<String>, Option<String>)> = extract_prompts(&args[0])?;
    let default_model = match args.get(1) {
        Some(Value::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let concurrency = match args.get(2) {
        Some(Value::Int(n)) => *n as usize,
        _ => 3,
    };
    
    // Run sequentially with chunking to respect rate limits
    let mut results = Vec::new();
    for (prompt, sys, model) in &prompts {
        let m = model.as_deref()
            .or(default_model.as_deref())
            .unwrap_or("claude-opus-4-5");
        let result = call_llm_once(prompt, m, sys.as_deref())?;
        results.push(Value::Str(result));
        // small sleep between calls to avoid rate limits
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    
    Ok(Value::Array(Rc::new(RefCell::new(results))))
}
```

Note: `call_llm_once` should reuse the same ureq logic from `llm_builtins.rs`. 
Either import/call that function, or refactor llm_builtins.rs to expose a `call_llm(prompt, model, system) -> Result<String, String>` helper.

## Where to put the code

Add to `omnimcode-core/src/llm_builtins.rs` — this file already has the LLM infrastructure.

## Wire into interpreter.rs

In the builtin match guard, add:
```
| "batch_llm_call" | "batch_llm_chat"
```

## OMC usage examples (test file)

Create `examples/test_batch_llm.omc`:
```omc
h prompts = [
    "What is 2+2?",
    "What is the capital of France?",
    "Name one programming language."
]

print("Running batch LLM call (3 prompts)...")
h results = batch_llm_call(prompts, null, 2)
print(str_concat("Got ", to_str(arr_len(results)), " results"))

h i = 0
while i < arr_len(results) {
    print(str_concat("  [", to_str(i), "] ", str_slice(results[i], 0, 50)))
    i = i + 1
}

# With per-prompt system messages
h prompts_with_sys = [
    {prompt: "What is 2+2?", system: "Answer in one word."},
    {prompt: "Capital of France?", system: "Answer in one word."}
]
h results2 = batch_llm_call(prompts_with_sys, null, 2)
print(results2)
```

## Add to ALL_BUILTINS and docs

Add `batch_llm_call` and `batch_llm_chat` to ALL_BUILTINS and docs in "llm_workflow" category.

## Build and commit

```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core 2>&1 | tail -20
# fix errors
git add omnimcode-core/src/llm_builtins.rs omnimcode-core/src/interpreter.rs omnimcode-core/src/docs.rs examples/test_batch_llm.omc
git commit -m "feat: batch_llm_call + batch_llm_chat builtin for parallel LLM prompt batching"
```

## DO NOT
- Do not use tokio/async
- Do not launch sub-agents  
- Keep it simple — sequential with sleep is fine
- Requires ANTHROPIC_API_KEY to actually work (same as llm_call)
