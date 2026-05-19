# Task 34: Introspection builtins — get_scope_vars, list_defined_fns, fn_arity, fn_source

## Goal
Add 4 runtime introspection builtins to the OMC interpreter.
These let OMC programs inspect themselves at runtime — critical for self-improvement and meta-programming.

## Working directory
/home/thearchitect/OMC

## Functions to add

### 1. get_scope_vars() → dict
Returns all currently-defined variables in scope as a dict {name: value}.
Useful for debugging and inspection.

### 2. list_defined_fns() → array of strings
Returns all currently-defined function names as an array.
Example: ["fib", "is_palindrome", "my_fn"]

### 3. fn_arity(fn_name) → int or null
Given a function name string, returns the number of parameters.
Returns null if the function doesn't exist.
Example: fn_arity("fib") → 1 (because fn fib(n))

### 4. fn_source(fn_name) → string or null
Given a function name string, returns the source-reconstructed OMC code for that function.
Reconstructs from the AST — "fn name(params) { ... }"
Returns null if the function doesn't exist.

## Where to make changes

### interpreter.rs
The interpreter has an `env` / environment/scope structure. You need to:

1. Find where variables are stored (likely a HashMap<String, Value> or similar)
2. Find where functions are stored (likely a separate map of function definitions)

Look for patterns like:
- `self.env` or `env` HashMap
- `self.functions` or `module.functions` 
- `FunctionDef`, `CompiledFunction`, or `UserFunction` variants

The interpreter is large (~14000 lines). Search for:
- `"fn_arity"` (may not exist yet)
- `BuiltinFunction` or `Builtin` match arms
- Where `"llm_call"` is dispatched (near line 2219)

### Implementation approach

For `get_scope_vars()`:
- The interpreter likely has access to the current environment frame
- Return a dict copy of all bindings visible in the current scope
- Filter out function objects (only return data values)

For `list_defined_fns()`:
- Look at where user-defined functions are stored
- Return their names as an array of strings

For `fn_arity(name)`:
- Find the function definition by name
- Return the parameter count

For `fn_source(name)`:
- Find the function definition
- Reconstruct "fn name(p1, p2) { <body as OMC string> }"
- You can do a simple AST-to-string reconstruction for the signature, 
  and use "{ <native body> }" for the body if full reconstruction is too complex
- At minimum: return "fn name(p1, p2) { ... }"

## Wire into interpreter.rs

In the builtin match guard, add:
```
| "get_scope_vars" | "list_defined_fns" | "fn_arity" | "fn_source"
```

Add dispatch arms.

## Add to ALL_BUILTINS and docs

Add these 4 names to the ALL_BUILTINS list and add doc entries in the "meta" or "introspection" category.

## Test file

Create `examples/test_introspection.omc`:
```omc
fn my_func(a, b, c) { return a + b + c }
fn other(x) { return x * 2 }

h fns = list_defined_fns()
print(fns)

h arity = fn_arity("my_func")
print(arity)

h src = fn_source("my_func")
print(src)

h vars = get_scope_vars()
print(dict_keys(vars))
```

Expected output:
- fns should contain "my_func" and "other"
- arity should be 3
- src should show the function signature
- vars should show defined variables

## Build and commit

```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core 2>&1 | tail -20
# fix errors
git add omnimcode-core/src/interpreter.rs omnimcode-core/src/docs.rs examples/test_introspection.omc
git commit -m "feat: get_scope_vars + list_defined_fns + fn_arity + fn_source introspection builtins"
```

## DO NOT
- Do not launch sub-agents
- Do not break existing tests
- Keep changes focused to interpreter.rs and docs.rs only
