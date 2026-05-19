//! Process execution builtins: `omc_spawn` and `omc_pipe`.
//!
//! These let OMC programs spawn subprocesses, capture output, and pipe between
//! commands — critical for the recursive self-improvement loop (OMC running OMC).
//!
//! ## omc_spawn(cmd, args?, env_vars?, timeout_ms?) -> dict
//!
//! Spawns a subprocess and waits for it to complete.
//!
//! Parameters:
//!   - cmd: string — the command to run (e.g., "omc", "python3", "cargo")
//!   - args: array of strings (optional) — command line arguments
//!   - env_vars: dict (optional) — extra environment variables to inject
//!   - timeout_ms: int (optional, default 30000) — timeout in milliseconds
//!
//! Returns: dict { stdout: string, stderr: string, exit_code: int, ok: bool }
//!
//! ## omc_pipe(commands) -> dict
//!
//! Pipes multiple commands together (like shell: cmd1 | cmd2 | cmd3).
//!
//! Parameters:
//!   - commands: array of arrays — each inner array is [cmd, arg1, arg2, ...]
//!
//! Returns: dict { stdout: string, stderr: string, exit_code: int, ok: bool }

use crate::value::{HArray, HInt, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_result(stdout: String, stderr: String, exit_code: i32) -> Value {
    let mut map = BTreeMap::new();
    map.insert("stdout".to_string(), Value::String(stdout));
    map.insert("stderr".to_string(), Value::String(stderr));
    map.insert("exit_code".to_string(), Value::HInt(HInt::new(exit_code as i64)));
    map.insert("ok".to_string(), Value::Bool(exit_code == 0));
    Value::dict_from(map)
}

fn value_to_string_vec(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Array(arr) => {
            let items = arr.items.borrow();
            items.iter().map(|item| match item {
                Value::String(s) => Ok(s.clone()),
                other => Ok(other.to_display_string()),
            }).collect()
        }
        Value::Null => Ok(vec![]),
        _ => Err("expected array of strings".to_string()),
    }
}

// ---------------------------------------------------------------------------
// omc_spawn
// ---------------------------------------------------------------------------

/// `omc_spawn(cmd, args?, env_vars?, timeout_ms?) -> dict`
///
/// Spawns a subprocess synchronously and captures its output.
/// Note: timeout_ms is accepted but currently not enforced (no tokio/threads).
/// The process runs to completion. Pass a reasonable timeout so callers can
/// document intent even if enforcement is added later.
pub fn omc_spawn(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("omc_spawn: first argument (cmd) is required".to_string());
    }

    // arg 0: cmd (string)
    let cmd = match &args[0] {
        Value::String(s) => s.clone(),
        other => return Err(format!(
            "omc_spawn: first arg must be string (command), got {}",
            other.type_name()
        )),
    };

    // arg 1: args array (optional)
    let cmd_args: Vec<String> = if args.len() > 1 {
        value_to_string_vec(&args[1])
            .map_err(|e| format!("omc_spawn: args (2nd param): {}", e))?
    } else {
        vec![]
    };

    // arg 2: env_vars dict (optional)
    let env_vars: Vec<(String, String)> = if args.len() > 2 {
        match &args[2] {
            Value::Dict(d) => {
                d.borrow().iter().map(|(k, v)| {
                    (k.clone(), v.to_display_string())
                }).collect()
            }
            Value::Null => vec![],
            _ => return Err("omc_spawn: env_vars (3rd param) must be a dict or null".to_string()),
        }
    } else {
        vec![]
    };

    // arg 3: timeout_ms (optional, default 30000). Currently just validated.
    // Actual timeout enforcement would need threads or tokio; for now we
    // accept the parameter so the interface is stable.
    let _timeout_ms: u64 = if args.len() > 3 {
        match &args[3] {
            Value::HInt(n) => n.value.max(0) as u64,
            Value::Null => 30000,
            _ => 30000,
        }
    } else {
        30000
    };

    let mut child = Command::new(&cmd);
    child.args(&cmd_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    for (k, v) in &env_vars {
        child.env(k, v);
    }

    let output = child.spawn()
        .map_err(|e| format!("omc_spawn: failed to start {:?}: {}", cmd, e))?
        .wait_with_output()
        .map_err(|e| format!("omc_spawn: wait failed: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let exit_code = output.status.code().unwrap_or(-1);

    Ok(make_result(stdout, stderr, exit_code))
}

// ---------------------------------------------------------------------------
// omc_pipe
// ---------------------------------------------------------------------------

/// `omc_pipe(commands) -> dict`
///
/// Pipes commands together like a shell pipe: cmd1 | cmd2 | cmd3.
/// Each element of `commands` is an array whose first element is the program
/// name and remaining elements are arguments.
///
/// stderr from each stage is collected and concatenated in the result.
pub fn omc_pipe(args: &[Value]) -> Result<Value, String> {
    if args.is_empty() {
        return Err("omc_pipe: commands array is required".to_string());
    }

    let commands: Vec<(String, Vec<String>)> = match &args[0] {
        Value::Array(outer) => {
            let items = outer.items.borrow();
            items.iter().enumerate().map(|(i, elem)| {
                match elem {
                    Value::Array(inner) => {
                        let inner_items = inner.items.borrow();
                        if inner_items.is_empty() {
                            return Err(format!("omc_pipe: command at index {} is an empty array", i));
                        }
                        let cmd = match &inner_items[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(format!(
                                "omc_pipe: command name at index {} must be string, got {}",
                                i, other.type_name()
                            )),
                        };
                        let cmd_args: Vec<String> = inner_items[1..].iter()
                            .map(|v| match v {
                                Value::String(s) => s.clone(),
                                other => other.to_display_string(),
                            })
                            .collect();
                        Ok((cmd, cmd_args))
                    }
                    other => Err(format!(
                        "omc_pipe: each command must be an array, got {} at index {}",
                        other.type_name(), i
                    )),
                }
            }).collect::<Result<Vec<_>, String>>()?
        }
        _ => return Err("omc_pipe: argument must be an array of command arrays".to_string()),
    };

    if commands.is_empty() {
        return Ok(make_result(String::new(), String::new(), 0));
    }

    let mut combined_stderr = String::new();

    // Build the pipeline: first command reads from nothing, each subsequent
    // command reads from the previous command's stdout.
    if commands.len() == 1 {
        let (cmd, cmd_args) = &commands[0];
        let output = Command::new(cmd)
            .args(cmd_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("omc_pipe: failed to start {:?}: {}", cmd, e))?
            .wait_with_output()
            .map_err(|e| format!("omc_pipe: wait failed: {}", e))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        return Ok(make_result(stdout, stderr, exit_code));
    }

    // Multi-stage pipeline: run each stage, feeding stdout of each into the
    // stdin of the next. Collect all stderr. Return stdout of the last stage.
    let mut current_stdin: Vec<u8> = Vec::new();
    let mut last_exit_code = 0i32;

    for (idx, (cmd, cmd_args)) in commands.iter().enumerate() {
        let is_last = idx == commands.len() - 1;

        let mut child_cmd = Command::new(cmd);
        child_cmd.args(cmd_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if idx > 0 {
            child_cmd.stdin(Stdio::piped());
        }

        let mut child = child_cmd.spawn()
            .map_err(|e| format!("omc_pipe: failed to start {:?} (stage {}): {}", cmd, idx, e))?;

        // Write previous stdout to this stage's stdin
        if idx > 0 {
            if let Some(mut stdin_pipe) = child.stdin.take() {
                stdin_pipe.write_all(&current_stdin)
                    .map_err(|e| format!("omc_pipe: failed to write stdin for stage {}: {}", idx, e))?;
                // Drop the pipe so the child sees EOF
                drop(stdin_pipe);
            }
        }

        let output = child.wait_with_output()
            .map_err(|e| format!("omc_pipe: wait failed for stage {}: {}", idx, e))?;

        // Collect stderr from every stage
        let stage_stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if !stage_stderr.is_empty() {
            if !combined_stderr.is_empty() {
                combined_stderr.push('\n');
            }
            combined_stderr.push_str(&stage_stderr);
        }

        last_exit_code = output.status.code().unwrap_or(-1);

        if is_last {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            return Ok(make_result(stdout, combined_stderr, last_exit_code));
        }

        // Pass this stage's stdout as next stage's stdin
        current_stdin = output.stdout;
    }

    // Unreachable when commands is non-empty, but satisfy the compiler.
    Ok(make_result(String::new(), combined_stderr, last_exit_code))
}

// ---------------------------------------------------------------------------
// Value type_name helper (mirror the interpreter's type_of logic)
// ---------------------------------------------------------------------------

trait TypeName {
    fn type_name(&self) -> &'static str;
}

impl TypeName for Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::HInt(_) => "int",
            Value::HFloat(_) => "float",
            Value::String(_) => "string",
            Value::Bool(_) => "bool",
            Value::Array(_) => "array",
            Value::Dict(_) => "dict",
            Value::Function { .. } => "function",
            Value::Null => "null",
            Value::Circuit(_) => "circuit",
            Value::Singularity { .. } => "singularity",
        }
    }
}
