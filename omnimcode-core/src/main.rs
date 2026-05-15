// omnimcode-core/src/main.rs - OMNIcode Standalone Executable Entry Point

use omnimcode_core::parser::Parser;
use omnimcode_core::interpreter::Interpreter;

use std::env;
use std::fs;
use std::io::{self, Write};


fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse simple flag-style args. Anything else is the input file
    // (or the install spec when --install is set).
    let mut mode = "run";
    let mut file_arg: Option<&str> = None;
    for a in args.iter().skip(1) {
        match a.as_str() {
            "--check" | "-c" => mode = "check",
            "--fmt" | "--format" | "-f" => mode = "fmt",
            "--install" | "-i" => mode = "install",
            "--list" | "-l" => mode = "list",
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
        ("install", spec) => install_command(spec),
        ("list", _) => list_command(),
        _ => unreachable!(),
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

/// Register the `py_*` builtin family on `interp`. Embedded Python
/// is now always-on (used to be feature-gated + OMC_PYTHON=1) — the
/// standalone binary ships with numpy/pandas/sklearn reachable from
/// any OMC program out of the box.
///
/// Set OMC_NO_PYTHON=1 in the environment to skip registration if
/// you genuinely don't want CPython initialised in your process
/// (saves ~5 MB resident from the embedded interpreter).
fn maybe_register_python(interp: &mut Interpreter) {
    if std::env::var("OMC_NO_PYTHON").as_deref() == Ok("1") {
        return;
    }
    omnimcode_core::python_embed::register_python_builtins(interp);
}

/// `--install [URL_OR_NAME]`. With no argument: read `omc.toml` in
/// the current directory and install every entry in [dependencies].
/// With a URL: fetch it and store the file under `omc_modules/`,
/// using the basename (sans .omc) as the module name. With a name
/// that doesn't look like a URL: error (no central registry yet —
/// users provide explicit URLs in omc.toml).
///
/// Eats our own dogfood: uses the embedded Python `requests` for
/// the HTTP fetch and `tomllib` for the manifest parse. Zero new
/// Rust dependencies.
fn install_command(spec: Option<&str>) -> i32 {
    use omnimcode_core::python_embed::{install_url_via_python, parse_omc_toml_via_python};

    if std::env::var("OMC_NO_PYTHON").as_deref() == Ok("1") {
        eprintln!("--install requires Python (used for HTTP fetch + TOML parse).");
        eprintln!("Unset OMC_NO_PYTHON or run with Python embedding enabled.");
        return 2;
    }

    // Ensure omc_modules/ exists.
    if let Err(e) = std::fs::create_dir_all("omc_modules") {
        eprintln!("install: cannot create omc_modules/: {}", e);
        return 1;
    }

    match spec {
        Some(spec) => {
            let url = if spec.starts_with("http://") || spec.starts_with("https://") {
                spec.to_string()
            } else {
                eprintln!("install: argument must be a URL (no central registry yet).");
                eprintln!("        For a manifest install, run `omc --install` with no arg");
                eprintln!("        and create an omc.toml in this directory.");
                return 2;
            };
            // Derive name from URL basename.
            let name = url
                .rsplit('/')
                .next()
                .unwrap_or("module")
                .trim_end_matches(".omc");
            match install_url_via_python(name, &url) {
                Ok(path) => {
                    println!("installed: {} -> {}", name, path);
                    0
                }
                Err(e) => {
                    eprintln!("install({}): {}", name, e);
                    1
                }
            }
        }
        None => {
            let manifest_path = "omc.toml";
            let manifest_text = match std::fs::read_to_string(manifest_path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("install: cannot read {}: {}", manifest_path, e);
                    eprintln!("        Create one with [dependencies] entries:");
                    eprintln!("");
                    eprintln!("            [dependencies]");
                    eprintln!("            np = \"https://raw.githubusercontent.com/.../np.omc\"");
                    return 1;
                }
            };
            let deps = match parse_omc_toml_via_python(&manifest_text) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("install: omc.toml parse: {}", e);
                    return 1;
                }
            };
            if deps.is_empty() {
                println!("install: no [dependencies] in omc.toml — nothing to do.");
                return 0;
            }
            let mut failures = 0;
            for (name, url) in &deps {
                match install_url_via_python(name, url) {
                    Ok(path) => println!("installed: {} -> {}", name, path),
                    Err(e) => {
                        eprintln!("install({}): {}", name, e);
                        failures += 1;
                    }
                }
            }
            if failures > 0 { 1 } else { 0 }
        }
    }
}

/// `--list`: enumerate everything in omc_modules/.
fn list_command() -> i32 {
    let dir = std::path::Path::new("omc_modules");
    if !dir.exists() {
        println!("(no omc_modules/ in this directory)");
        return 0;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("list: cannot read omc_modules/: {}", e);
            return 1;
        }
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("omc") {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect();
    names.sort();
    if names.is_empty() {
        println!("(no installed modules — use --install to add some)");
    } else {
        for n in names {
            println!("  {}", n);
        }
    }
    0
}

fn print_help() {
    let prog = env::args().next().unwrap_or_else(|| "omnimcode-standalone".to_string());
    println!("Usage:");
    println!("  {} [FILE]              run a program (or start REPL if no file)", prog);
    println!("  {} --check FILE        run heal pass, print diagnostics, exit", prog);
    println!("  {} --fmt FILE          pretty-print AST as canonical OMC source", prog);
    println!("  {} --install [URL]     install package from URL into omc_modules/", prog);
    println!("                         (no URL = read omc.toml [dependencies])", );
    println!("  {} --list              list packages installed under omc_modules/", prog);
    println!("  {} --help              this message", prog);
    println!();
    println!("omc.toml format (for --install with no arg):");
    println!("  [dependencies]");
    println!("  np = \"https://example.com/raw/np.omc\"");
    println!("  pd = \"https://example.com/raw/pd.omc\"");
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
    println!("  OMC_NO_PYTHON=1        skip embedded CPython initialisation");
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
        // OMC_PYTHON=1 — register py_import / py_call / py_eval / etc.
        // so OMC code can drive numpy, pandas, requests, any pip lib.
        // Off by default; the standalone binary still builds without
        // libpython if `python-embed` feature isn't on at build time.
        maybe_register_python(vm.interp_mut());
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
    maybe_register_python(&mut interpreter);
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
    println!("OMNIcode interactive shell");
    println!("Type :help for commands, :quit to exit. Statements end with ;");
    println!();

    let stdin = io::stdin();
    let mut interpreter = Interpreter::new();
    maybe_register_python(&mut interpreter);
    let mut buffer = String::new();
    let mut continuing = false;

    loop {
        print!("{}", if continuing { "...> " } else { "omc> " });
        io::stdout().flush().unwrap();

        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => { println!(); break; }
            Err(e) => { eprintln!("Error reading input: {}", e); break; }
            Ok(_) => {}
        }

        let trimmed = line.trim();

        // REPL meta-commands (only at the start of a fresh input).
        if !continuing {
            match trimmed {
                "" => continue,
                ":quit" | ":q" | ":exit" => break,
                ":help" | ":h" | ":?" => {
                    repl_print_help();
                    continue;
                }
                ":reset" => {
                    interpreter = Interpreter::new();
                    println!("interpreter state reset");
                    continue;
                }
                _ => {}
            }
        }

        buffer.push_str(&line);

        // Heuristic for multi-line input: count unmatched braces/parens/brackets.
        // If they're unbalanced (more openers than closers), keep reading.
        // Skips characters inside string literals so `"{"` doesn't confuse us.
        if !is_balanced(&buffer) {
            continuing = true;
            continue;
        }

        // First attempt: parse as-typed.
        let trimmed_buffer = buffer.trim().to_string();
        let mut parser = Parser::new(&trimmed_buffer);
        match parser.parse() {
            Ok(statements) => {
                continuing = false;
                let to_run = buffer.clone();
                buffer.clear();
                repl_execute(&mut interpreter, &to_run, statements);
            }
            Err(msg) if msg.contains("Semicolon") && !trimmed_buffer.ends_with(';') => {
                // Bare-expression mode: parser wanted a `;` but the
                // user hit enter without one. Try parsing with `;`
                // appended; if that yields a single Expression
                // statement, evaluate it and print the result. This
                // is what makes `1 + 2` (no semicolon) print 3.
                let with_semi = format!("{};", trimmed_buffer);
                let mut p2 = Parser::new(&with_semi);
                match p2.parse() {
                    Ok(statements) => {
                        continuing = false;
                        let to_run = buffer.clone();
                        buffer.clear();
                        repl_execute(&mut interpreter, &to_run, statements);
                    }
                    Err(msg2) => {
                        eprintln!("Parse error: {}", msg2);
                        continuing = false;
                        buffer.clear();
                    }
                }
            }
            Err(msg) => {
                // Other parse errors that look like "needs more input"
                // (unterminated string, missing closing brace not caught
                // by is_balanced) → ask for another line. Otherwise
                // show the error and reset.
                if msg.contains("Eof") || msg.contains("end of") {
                    continuing = true;
                } else {
                    eprintln!("Parse error: {}", msg);
                    continuing = false;
                    buffer.clear();
                }
            }
        }
    }

    println!("bye");
}

fn repl_print_help() {
    println!("REPL commands:");
    println!("  :help, :h, :?   show this message");
    println!("  :quit, :q       exit the REPL");
    println!("  :reset          discard all defined variables and functions");
    println!();
    println!("Tips:");
    println!("  Statements need a trailing `;`. Multi-line input continues");
    println!("  while braces/parens are unbalanced (use a closing `}}` or");
    println!("  `)` to finish). Type a bare expression with `;` to see its");
    println!("  result via println().");
}

/// Run a parsed REPL input. If the input is a single expression
/// statement (with no trailing semicolon in the source), evaluate it
/// and print the result — Python REPL style. Otherwise execute as
/// normal statements.
fn repl_execute(
    interp: &mut Interpreter,
    raw_source: &str,
    statements: Vec<omnimcode_core::ast::Statement>,
) {
    use omnimcode_core::ast::Statement;
    // Detect implicit-print case: exactly one Expression statement
    // and the source has no trailing `;`. This makes `1 + 2` (no
    // semicolon) print `3`, while `1 + 2;` runs silently.
    let trimmed = raw_source.trim();
    let is_bare_expr = !trimmed.ends_with(';')
        && statements.len() == 1
        && matches!(&statements[0], Statement::Expression(_));

    if is_bare_expr {
        if let Statement::Expression(e) = &statements[0] {
            match interp.eval_for_repl(e) {
                Ok(v) => println!("{}", v.to_display_string()),
                Err(msg) => eprintln!("Error: {}", msg),
            }
            return;
        }
    }

    if let Err(e) = interp.execute(statements) {
        eprintln!("Error: {}", e);
    }
}

/// Counts unmatched openers in `s`, ignoring contents of string
/// literals. Returns true when all brackets/parens/braces are balanced
/// — i.e. when the REPL can attempt to parse the input.
fn is_balanced(s: &str) -> bool {
    let mut depth: i32 = 0;
    let mut in_str = false;
    let mut prev = '\0';
    for c in s.chars() {
        if in_str {
            // Honor backslash escapes so `"\""` doesn't end the string early.
            if c == '"' && prev != '\\' { in_str = false; }
            prev = c;
            continue;
        }
        match c {
            '"' => in_str = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        prev = c;
    }
    depth <= 0 && !in_str
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
