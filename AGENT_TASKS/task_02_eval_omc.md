# Task 02: eval_omc Runtime Self-Modification Builtin

## Goal
Add `eval_omc(code_str)` and `eval_omc_ctx(code_str)` as native builtins.
This enables OMC programs to parse and run OMC code at runtime — the key to self-modification.

## Repo: /home/thearchitect/OMC

## Files to Modify
- `omnimcode-core/src/interpreter.rs`

## Context
- The builtin match guard is near line 2219 (search for `| "json_parse" | "json_stringify"`)
- Builtin implementations are in the match body (search for `"json_parse" => {`)
- ALL_BUILTINS list is near line 13717
- Lexer: `omnimcode-core/src/lexer.rs` — find the tokenize function
- Parser: `omnimcode-core/src/parser.rs` — find the parse function
- Run `grep -n "pub fn " omnimcode-core/src/interpreter.rs | head -30` to see available methods

## What to Implement

### `eval_omc(code_str, scope_dict?)` -> Value
Parse and run OMC code in a fresh interpreter. Returns last evaluated value.
Optional second arg: dict of variable bindings to pre-populate.

### `eval_omc_ctx(code_str)` -> Value  
Run OMC code sharing the current interpreter's function definitions.
(Inherits all currently defined functions from the parent scope.)

### `omc_parse(code_str)` -> Dict
Parse OMC code and return minimal parse info (for metaprogramming).

## Implementation

First, understand how the interpreter runs code by reading the top of interpreter.rs and looking for how `run_program` or `interpret` works.

Look at how `Interpreter::new()` is created. You need to create a fresh interpreter for eval_omc.

```rust
"eval_omc" => {
    if args.is_empty() {
        return Err("eval_omc requires (code_str)".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("eval_omc: argument must be a string".to_string()),
    };
    
    // Tokenize + parse
    let tokens = crate::lexer::tokenize(&code)
        .map_err(|e| format!("eval_omc: lex error: {}", e))?;
    let stmts = crate::parser::parse(tokens)
        .map_err(|e| format!("eval_omc: parse error: {}", e))?;
    
    // Fresh interpreter
    let mut fresh = Interpreter::new();
    
    // If second arg is dict, populate scope
    if args.len() > 1 {
        if let Value::Dict(map) = &args[1] {
            let borrowed = map.borrow();
            for (k, v) in borrowed.iter() {
                fresh.env.borrow_mut().set(k.clone(), v.clone());
            }
        }
    }
    
    // Execute
    let mut last = Value::Null;
    for stmt in &stmts {
        last = fresh.exec_stmt(stmt).map_err(|e| format!("eval_omc: {}", e))?;
    }
    Ok(last)
}
```

For `eval_omc_ctx`, iterate over `self.env` and copy all `Value::Fn` entries into the fresh interpreter.

**IMPORTANT**: The actual method/field names may differ. Read the interpreter source carefully and adapt. The key is:
1. Get code as string from args
2. Tokenize (find the tokenize function — it's likely `crate::lexer::tokenize` or similar)
3. Parse (find the parse function)
4. Create fresh Interpreter
5. Execute statements
6. Return last value

## Step to handle unknown API
Run these to understand the structure:
```bash
grep -n "pub fn " omnimcode-core/src/interpreter.rs | head -40
grep -n "pub fn " omnimcode-core/src/lexer.rs | head -20
grep -n "pub fn " omnimcode-core/src/parser.rs | head -20
grep -n "exec_stmt\|run_stmt\|eval_stmt" omnimcode-core/src/interpreter.rs | head -10
grep -n "fn new()" omnimcode-core/src/interpreter.rs | head -5
```

## Test
```omc
h code = "h x = 10; h y = 20; x + y"
h result = eval_omc(code)
print(result)
# Expected: 30

h gen_fn = "fn double(x) x * 2"
eval_omc(gen_fn)
h r = eval_omc("double(21)")
print(r)
# Expected: 42
```

Save test as `examples/test_eval_omc.omc`.

## Build & Test
```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core 2>&1 | tail -20
cargo test -p omnimcode-core 2>&1 | tail -20
```

## Commit
```
git add -A
git commit -m "feat: eval_omc + eval_omc_ctx runtime self-evaluation builtins"
```
