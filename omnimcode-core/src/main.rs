// omnimcode-core/src/main.rs - OMNIcode Standalone Executable Entry Point

use omnimcode_core::parser::Parser;
use omnimcode_core::interpreter::Interpreter;

use std::env;
use std::fs;
use std::io::{self, Write};


fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse simple flag-style args: --check, --fmt, --help / -h
    // Anything else after the flag is the input file.
    let mut mode = "run";
    let mut file_arg: Option<&str> = None;
    for a in args.iter().skip(1) {
        match a.as_str() {
            "--check" | "-c" => mode = "check",
            "--fmt" | "--format" | "-f" => mode = "fmt",
            "--help" | "-h" => mode = "help",
            other if !other.starts_with('-') => file_arg = Some(other),
            other => {
                eprintln!("Unknown flag: {}", other);
                eprintln!("Try --help for usage.");
                std::process::exit(2);
            }
        }
    }

    if mode == "help" {
        print_help();
        return;
    }

    let exit_code: i32 = match (mode, file_arg) {
        ("run", None) => { repl(); 0 }
        ("run", Some(path)) => match read_and_run(path) {
            Ok(()) => 0,
            Err(e) => { eprintln!("Error: {}", e); 1 }
        },
        ("check", Some(path)) => check_program(path),
        ("check", None) => {
            eprintln!("--check requires a file argument.");
            2
        }
        ("fmt", Some(path)) => format_program_to_stdout(path),
        ("fmt", None) => {
            eprintln!("--fmt requires a file argument.");
            2
        }
        _ => unreachable!(),
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn print_help() {
    let prog = env::args().next().unwrap_or_else(|| "omnimcode-standalone".to_string());
    println!("Usage:");
    println!("  {} [FILE]              run a program (or start REPL if no file)", prog);
    println!("  {} --check FILE        run heal pass, print diagnostics, exit", prog);
    println!("  {} --fmt FILE          pretty-print AST as canonical OMC source", prog);
    println!("  {} --help              this message", prog);
    println!();
    println!("Environment variables:");
    println!("  OMC_VM=1               execute through the Rust bytecode VM");
    println!("  OMC_HEAL=1             auto-heal AST before execution (iterative)");
    println!("  OMC_HEAL_RETRY=1       catch runtime errors, heal, retry once");
    println!("  OMC_HEAL_QUIET=1       suppress heal-pass diagnostic output");
    println!("  OMC_DISASM=1           dump bytecode disassembly before VM run");
    println!("  OMC_OPT=0              disable optimizer (on by default)");
    println!("  OMC_OPT_STATS=1        print optimizer pass statistics");
    println!("  OMC_STDLIB_PATH=...    colon-separated module search path");
}

fn read_and_run(path: &str) -> Result<(), String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("reading {}: {}", path, e))?;
    execute_program(&content)
}

/// `--check`: parse, run heal_ast_until_fixpoint, print diagnostics to
/// stdout, never execute. Exit code is the number of diagnostics
/// (clamped to 1). Useful for CI / lint workflows.
fn check_program(path: &str) -> i32 {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("reading {}: {}", path, e); return 1; }
    };
    let mut parser = Parser::new(&content);
    let statements = match parser.parse() {
        Ok(s) => s,
        Err(e) => { eprintln!("parse error: {}", e); return 1; }
    };
    let interpreter = Interpreter::new();
    let (_healed, diagnostics, iters, outcome) =
        interpreter.heal_ast_until_fixpoint(statements, 5);
    if diagnostics.is_empty() {
        println!("{}: clean ({} iteration{})", path, iters,
                 if iters == 1 { "" } else { "s" });
        return 0;
    }
    println!("{}: {} diagnostic(s) over {} iteration(s) ({})",
             path, diagnostics.len(), iters, outcome);
    for d in &diagnostics {
        println!("  {}", d);
    }
    1
}

/// `--fmt`: parse, pretty-print the AST back to canonical OMC source,
/// write to stdout. Lossy on whitespace/comments — produces canonical
/// indentation, parentheses around BIN ops (avoids precedence ambiguity),
/// and consistent statement spacing.
fn format_program_to_stdout(path: &str) -> i32 {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("reading {}: {}", path, e); return 1; }
    };
    let mut parser = Parser::new(&content);
    let statements = match parser.parse() {
        Ok(s) => s,
        Err(e) => { eprintln!("parse error: {}", e); return 1; }
    };
    print!("{}", omnimcode_core::formatter::format_program(&statements));
    0
}

fn execute_program(source: &str) -> Result<(), String> {
    let mut parser = Parser::new(source);
    let mut statements = parser.parse()?;

    // OMC_HEAL=1 — run the host-side self-healing pass (iteratively
    // until fixpoint, max 5 passes). Catches harmonic violations,
    // identifier typos, literal divide-by-zero, and arity mismatches
    // at call sites. Diagnostics print to stderr; healed AST
    // executes normally.
    //
    // OMC_HEAL_QUIET=1 suppresses the diagnostic output (still heals).
    // OMC_HEAL_RETRY=1 (handled later) catches a runtime error after
    //                  execution, runs heal_ast, and retries once.
    let heal_quiet = std::env::var("OMC_HEAL_QUIET").as_deref() == Ok("1");
    if std::env::var("OMC_HEAL").as_deref() == Ok("1") {
        let interpreter = Interpreter::new();
        let (healed, diagnostics, iters, outcome) =
            interpreter.heal_ast_until_fixpoint(statements, 5);
        if !diagnostics.is_empty() && !heal_quiet {
            eprintln!(
                "--- OMC_HEAL: {} diagnostic(s) across {} iteration(s) ({}) ---",
                diagnostics.len(), iters, outcome
            );
            for d in &diagnostics {
                eprintln!("  {}", d);
            }
            eprintln!("--- end OMC_HEAL ---");
        }
        statements = healed;
    }

    // Opt-in bytecode VM path. The tree-walk interpreter remains the default
    // (full language coverage); the VM is a faster dispatch for the subset of
    // programs whose ASTs the compiler currently supports.
    if std::env::var("OMC_VM").as_deref() == Ok("1") {
        let mut module = omnimcode_core::compiler::compile_program(&statements)?;
        // OMC_OPT=0 disables the optimizer (handy for debugging). On by default.
        if std::env::var("OMC_OPT").as_deref() != Ok("0") {
            let stats = omnimcode_core::bytecode_opt::optimize_module(&mut module);
            if std::env::var("OMC_OPT_STATS").as_deref() == Ok("1") {
                eprintln!(
                    "[opt] folded={} cached={} dead_loads={} not={} neg={} (total {})",
                    stats.constants_folded,
                    stats.unary_calls_cached,
                    stats.dead_loads_removed,
                    stats.double_nots_collapsed,
                    stats.double_negs_collapsed,
                    stats.total()
                );
            }
        }
        // OMC_DISASM=1 prints the compiled bytecode (post-optimization) to
        // stderr before execution. Useful for debugging and verifying that
        // optimizer/inliner did what was expected.
        if std::env::var("OMC_DISASM").as_deref() == Ok("1") {
            eprint!("{}", omnimcode_core::disasm::disassemble_module(&module));
        }
        let mut vm = omnimcode_core::vm::Vm::new();
        // Pre-register user function definitions into the VM's internal
        // interpreter so the `call(fn, args)` builtin and other dynamic
        // dispatch paths can resolve them. The VM still uses its own
        // compiled function table for direct Op::Call dispatch; this
        // duplication only kicks in for first-class function reflection.
        // Imports are no-ops in the bytecode compiler, so the VM
        // never wires up `math.fib_up_to`-style aliased calls on its
        // own. Run a pre-pass that walks top-level Statement::Import
        // and merges each module's functions into interp.functions.
        // Dot-namespaced calls then route through call_module_function
        // and resolve normally.
        vm.interp_mut().process_imports(&statements)?;
        vm.interp_mut().register_user_functions(&statements);
        // Also register every lambda body the compiler collected. Lambda
        // invocation routes through call_first_class_function → the
        // interpreter's tree-walk path; that path looks up by name in
        // `self.interp.functions`, so the lambda body AST needs to live
        // there too. Fast bytecode-VM body execution is future work.
        for (lname, lparams, lbody) in &module.lambda_asts {
            vm.interp_mut().register_lambda(lname, lparams.clone(), lbody.clone());
        }
        vm.run_module(&module)?;
        return Ok(());
    }

    let mut interpreter = Interpreter::new();
    // OMC_HEAL_RETRY=1 — catch runtime errors after execution starts,
    // run the heal pass on a fresh copy of the AST, and retry. Captures
    // bugs that the static heal pass missed (e.g. dynamic /0, missing
    // names that only surface at call time). Single retry: if the
    // healed AST also errors, that error propagates unmodified.
    //
    // The retry runs even WITHOUT OMC_HEAL=1 — it's a separate opt-in
    // that catches errors after the fact rather than preventing them.
    if std::env::var("OMC_HEAL_RETRY").as_deref() == Ok("1") {
        let retry_source = statements.clone();
        match interpreter.execute(statements) {
            Ok(()) => return Ok(()),
            Err(first_err) => {
                if !heal_quiet {
                    eprintln!("--- OMC_HEAL_RETRY: caught error, attempting heal+retry ---");
                    eprintln!("  first error: {}", first_err);
                }
                // Fresh interpreter so any partial side-effects from
                // the first run don't leak into the retry.
                let mut retry_interp = Interpreter::new();
                let (healed, diags, _, _) =
                    retry_interp.heal_ast_until_fixpoint(retry_source, 5);
                if !diags.is_empty() && !heal_quiet {
                    eprintln!("  healing pass found {} diagnostic(s):", diags.len());
                    for d in &diags {
                        eprintln!("    {}", d);
                    }
                }
                if !heal_quiet {
                    eprintln!("--- retrying with healed AST ---");
                }
                return retry_interp.execute(healed);
            }
        }
    }
    interpreter.execute(statements)?;

    Ok(())
}

fn repl() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("         OMNIcode Interactive Shell (v1.0.0-standalone)         ");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Type OMNIcode statements. Press Ctrl+C to exit.");
    println!();

    let stdin = io::stdin();
    let mut interpreter = Interpreter::new();
    let mut input_buffer = String::new();

    loop {
        print!("omc> ");
        io::stdout().flush().unwrap();

        input_buffer.clear();
        match stdin.read_line(&mut input_buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {
                if input_buffer.trim().is_empty() {
                    continue;
                }

                // Try to parse and execute
                let trimmed = input_buffer.trim();
                let mut parser = Parser::new(trimmed);
                
                match parser.parse() {
                    Ok(statements) => {
                        match interpreter.execute(statements) {
                            Ok(()) => {},
                            Err(e) => eprintln!("Error: {}", e),
                        }
                    }
                    Err(e) => {
                        eprintln!("Parse error: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Thank you for using OMNIcode!");
    println!("═══════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_hello_world() {
        let result = execute_program("print(\"Hello\");");
        assert!(result.is_ok());
    }
}
