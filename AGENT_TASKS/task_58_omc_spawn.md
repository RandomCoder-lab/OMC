# Task 58: omc_spawn + omc_pipe process execution builtins

## Goal
Add 2 process-execution builtins to the OMC interpreter.
These let OMC programs spawn subprocesses, capture output, and pipe between commands.
CRITICAL for the recursive self-improvement loop: can run `omc` itself as a subprocess.

## Working directory
/home/thearchitect/OMC

## Functions to add

### 1. omc_spawn(cmd, args?, env_vars?, timeout_ms?) → dict
Spawns a subprocess and waits for it to complete.

Parameters:
- cmd: string — the command to run (e.g., "omc", "python3", "cargo")
- args: array of strings (optional) — command line arguments
- env_vars: dict (optional) — extra environment variables
- timeout_ms: int (optional, default 30000) — timeout in milliseconds

Returns: dict {
  stdout: string,    # captured stdout
  stderr: string,    # captured stderr  
  exit_code: int,    # 0 = success
  ok: bool           # exit_code == 0
}

Example in OMC:
```omc
h result = omc_spawn("omc", ["examples/test_llm_call.omc"], null, 30000)
print(result["stdout"])
print(result["exit_code"])

# Self-hosting: run OMC code via OMC
h code_result = omc_spawn("omc", ["--eval", "print(42 + 1)"], null, 5000)
```

### 2. omc_pipe(commands) → dict
Pipes multiple commands together (like shell pipe: cmd1 | cmd2 | cmd3).

Parameters:
- commands: array of arrays — each inner array is [cmd, arg1, arg2, ...]

Returns: dict {
  stdout: string,   # output of final command
  stderr: string,   # stderr from all commands
  exit_code: int,
  ok: bool
}

Example in OMC:
```omc
h result = omc_pipe([
    ["echo", "hello world"],
    ["tr", "a-z", "A-Z"]
])
print(result["stdout"])  # "HELLO WORLD"
```

## Rust implementation

Use `std::process::{Command, Stdio}`.

```rust
use std::process::{Command, Stdio};
use std::io::Write;
use std::time::Duration;

pub fn omc_spawn(args: &[Value]) -> Result<Value, String> {
    let cmd = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return Err("omc_spawn: first arg must be string".to_string()),
    };
    
    let cmd_args: Vec<String> = if args.len() > 1 {
        match &args[1] {
            Value::Array(arr) => {
                arr.borrow().iter().map(|v| match v {
                    Value::Str(s) => s.clone(),
                    other => other.to_string(),
                }).collect()
            }
            Value::Null => vec![],
            _ => return Err("omc_spawn: args must be array".to_string()),
        }
    } else {
        vec![]
    };
    
    // timeout
    let timeout_ms = if args.len() > 3 {
        match &args[3] {
            Value::Int(n) => *n as u64,
            _ => 30000,
        }
    } else {
        30000
    };
    
    let mut child = Command::new(&cmd)
        .args(&cmd_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("omc_spawn: failed to start {cmd}: {e}"))?;
    
    // collect output with timeout using wait_with_output
    let output = child.wait_with_output()
        .map_err(|e| format!("omc_spawn: wait failed: {e}"))?;
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);
    
    let mut map = HashMap::new();
    map.insert("stdout".to_string(), Value::Str(stdout));
    map.insert("stderr".to_string(), Value::Str(stderr));
    map.insert("exit_code".to_string(), Value::Int(exit_code as i64));
    map.insert("ok".to_string(), Value::Bool(exit_code == 0));
    Ok(Value::Dict(Rc::new(RefCell::new(map))))
}
```

## File to create

Create `omnimcode-core/src/process_builtins.rs` with `omc_spawn` and `omc_pipe` functions.

## Wire into interpreter.rs

In the builtin match guard, add:
```
| "omc_spawn" | "omc_pipe"
```

Add dispatch arms.

## Add to ALL_BUILTINS and docs

Add these 2 names to ALL_BUILTINS list and add doc entries in a "process" category.

## Test file

Create `examples/test_omc_spawn.omc`:
```omc
# Test basic spawn
h result = omc_spawn("echo", ["Hello from subprocess!"], null, 5000)
print(result["stdout"])
print(result["ok"])
print(result["exit_code"])

# Test pipe
h pipe_result = omc_pipe([["echo", "hello"], ["tr", "a-z", "A-Z"]])
print(pipe_result["stdout"])

# Self-improvement: OMC spawns OMC
h code = "print(\"OMC runs OMC!\")\nprint(1 + 1)"
file_write("/tmp/test_spawn.omc", code)
h omc_result = omc_spawn("omc", ["/tmp/test_spawn.omc"], null, 10000)
print(omc_result["stdout"])
```

## Build and commit

```bash
cd /home/thearchitect/OMC
cargo build -p omnimcode-core 2>&1 | tail -20
# fix errors
git add omnimcode-core/src/process_builtins.rs omnimcode-core/src/interpreter.rs omnimcode-core/src/docs.rs omnimcode-core/src/lib.rs examples/test_omc_spawn.omc
git commit -m "feat: omc_spawn + omc_pipe process execution builtins"
```

## DO NOT
- Do not use async/tokio
- Do not launch sub-agents
- Do not break existing code
