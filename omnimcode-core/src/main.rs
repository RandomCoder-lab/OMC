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
            "--init" => mode = "init",
            "--test" | "-t" => mode = "test",
            "--bench" | "-b" => mode = "bench",
            "--audit" | "-a" => mode = "audit",
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
        ("init", _) => init_command(),
        ("test", Some(path)) => test_command(path),
        ("test", None) => { eprintln!("--test requires a file argument."); 2 }
        ("bench", Some(path)) => bench_command(path),
        ("bench", None) => { eprintln!("--bench requires a file argument."); 2 }
        ("audit", Some(path)) => audit_command(path),
        ("audit", None) => { eprintln!("--audit requires a file argument."); 2 }
        _ => unreachable!(),
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

/// Register the `py_*` builtin family on `interp`. Embedded Python
/// is on by default (python-embed feature, in default features) — the
/// standalone binary ships with numpy/pandas/sklearn reachable from
/// any OMC program out of the box.
///
/// Set OMC_NO_PYTHON=1 in the environment to skip registration if
/// you genuinely don't want CPython initialised in your process.
/// Disable the `python-embed` Cargo feature at build time for WASM /
/// no_std targets where libpython can't link.
#[cfg(feature = "python-embed")]
fn maybe_register_python(interp: &mut Interpreter) {
    if std::env::var("OMC_NO_PYTHON").as_deref() == Ok("1") {
        return;
    }
    omnimcode_core::python_embed::register_python_builtins(interp);
}

/// Stub when `python-embed` is OFF (e.g. WASM target). Lets the rest
/// of main.rs call this unconditionally; OMC programs that use
/// `py_*` builtins will get "Undefined function" errors at runtime
/// which is the desired behavior — fail loudly, don't pretend Python
/// is there when it isn't.
#[cfg(not(feature = "python-embed"))]
fn maybe_register_python(_interp: &mut Interpreter) {}

/// `--install [SPEC]`. SPEC can be:
///
///   * a URL                 → fetch and store under that basename
///   * a registry short name → look up in the central registry,
///                             fetch, verify sha256, store
///   * absent                → read `omc.toml` and install every
///                             entry in [dependencies]
///
/// omc.toml [dependencies] entries can be:
///
///   * `name = "<URL>"`                   → fetch URL directly
///   * `name = "<short-name>"`            → registry lookup
///   * `name = { url = "...", sha256 = "..." }` → URL + verification
///
/// Eats our own dogfood: HTTP fetch via embedded Python `requests`,
/// TOML parse via `tomllib`, sha256 via `hashlib`. Zero new Rust
/// dependencies.
#[cfg(feature = "python-embed")]
fn install_command(spec: Option<&str>) -> i32 {
    use omnimcode_core::python_embed::{
        install_url_via_python, parse_omc_toml_via_python, registry_lookup,
    };

    if std::env::var("OMC_NO_PYTHON").as_deref() == Ok("1") {
        eprintln!("--install requires Python (used for HTTP fetch + TOML parse).");
        eprintln!("Unset OMC_NO_PYTHON or run with Python embedding enabled.");
        return 2;
    }

    if let Err(e) = std::fs::create_dir_all("omc_modules") {
        eprintln!("install: cannot create omc_modules/: {}", e);
        return 1;
    }

    match spec {
        Some(spec) => {
            let (name, url, sha) = if spec.starts_with("http://") || spec.starts_with("https://")
            {
                let name = spec
                    .rsplit('/')
                    .next()
                    .unwrap_or("module")
                    .trim_end_matches(".omc")
                    .to_string();
                (name, spec.to_string(), None)
            } else {
                // Treat as a registry short name.
                match registry_lookup(spec) {
                    Ok((url, sha)) => (spec.to_string(), url, Some(sha)),
                    Err(e) => {
                        eprintln!("install: {}", e);
                        eprintln!("        Use a full URL or check the registry.");
                        return 1;
                    }
                }
            };
            match install_url_via_python(&name, &url, sha.as_deref()) {
                Ok(path) => {
                    let v = if sha.is_some() { " (sha256 ok)" } else { "" };
                    println!("installed: {} -> {}{}", name, path, v);
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
                    eprintln!("        Run `omnimcode-standalone --init` to create one.");
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
            for (name, value) in &deps {
                // value is a URL OR a registry short name.
                let (url, sha) = if value.starts_with("http://") || value.starts_with("https://") {
                    (value.clone(), None)
                } else {
                    match registry_lookup(value) {
                        Ok((u, s)) => (u, Some(s)),
                        Err(e) => {
                            eprintln!("install({}): registry lookup failed: {}", name, e);
                            failures += 1;
                            continue;
                        }
                    }
                };
                match install_url_via_python(name, &url, sha.as_deref()) {
                    Ok(path) => {
                        let v = if sha.is_some() { " (sha256 ok)" } else { "" };
                        println!("installed: {} -> {}{}", name, path, v);
                    }
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

#[cfg(not(feature = "python-embed"))]
fn install_command(_spec: Option<&str>) -> i32 {
    eprintln!("--install requires the `python-embed` feature (HTTP / TOML / sha256).");
    eprintln!("Rebuild with `cargo build --release` (default features include python-embed).");
    2
}

/// `--test FILE`: load FILE, find every top-level `fn test_*()`,
/// run each in a fresh interpreter scope, report pass/fail per test
/// and a final summary. A test PASSES if it returns without raising;
/// FAILS if it errors. Exit code = number of failures (clamped to 1).
///
/// Convention: test fns take no args, return anything (return value
/// ignored). Use OMC's `error("msg")` to assert failure.
fn test_command(path: &str) -> i32 {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("test: read {}: {}", path, e); return 2; }
    };
    let test_names = match scan_fn_prefix(&source, "test_") {
        Ok(n) => n,
        Err(e) => { eprintln!("test: parse {}: {}", path, e); return 2; }
    };
    if test_names.is_empty() {
        println!("test: no `fn test_*()` functions in {}", path);
        return 0;
    }
    println!("running {} test(s) from {}", test_names.len(), path);
    let mut passed = 0;
    let mut failed: Vec<(String, String)> = Vec::new();
    for name in &test_names {
        // Re-parse + re-execute per-test so each gets a fresh state.
        // Slower than reusing the interpreter, but means one test's
        // mutations can't leak into the next.
        match run_named_fn(&source, name) {
            Ok(()) => { passed += 1; println!("  ok    {}", name); }
            Err(e) => {
                failed.push((name.clone(), e.clone()));
                println!("  FAIL  {}", name);
            }
        }
    }
    println!("");
    println!("result: {} passed, {} failed", passed, failed.len());
    for (name, err) in &failed {
        println!("  {}: {}", name, err.lines().next().unwrap_or(err));
    }
    if failed.is_empty() { 0 } else { 1 }
}

/// `--bench FILE`: load FILE, find every top-level `fn bench_*()`,
/// run each, time it, report ms total. Convention: bench fns take
/// no args, return anything. Use `now_ms()` inside if you want
/// per-iteration breakdowns.
fn bench_command(path: &str) -> i32 {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => { eprintln!("bench: read {}: {}", path, e); return 2; }
    };
    let bench_names = match scan_fn_prefix(&source, "bench_") {
        Ok(n) => n,
        Err(e) => { eprintln!("bench: parse {}: {}", path, e); return 2; }
    };
    if bench_names.is_empty() {
        println!("bench: no `fn bench_*()` functions in {}", path);
        return 0;
    }
    println!("running {} bench(es) from {}", bench_names.len(), path);
    use std::time::Instant;
    for name in &bench_names {
        let start = Instant::now();
        let res = run_named_fn(&source, name);
        let elapsed = start.elapsed();
        match res {
            Ok(()) => println!("  {:30}  {:>8.2} ms", name, elapsed.as_secs_f64() * 1000.0),
            Err(e) => println!("  {:30}  ERROR: {}", name, e.lines().next().unwrap_or(&e)),
        }
    }
    0
}

/// Find every top-level `fn NAME(...)` whose name starts with `prefix`.
/// Used by --test and --bench to discover their respective fn families.
fn scan_fn_prefix(source: &str, prefix: &str) -> Result<Vec<String>, String> {
    use omnimcode_core::ast::Statement;
    let mut parser = Parser::new(source);
    let stmts = parser.parse()?;
    let mut out = Vec::new();
    for s in &stmts {
        if let Statement::FunctionDef { name, .. } = s {
            if name.starts_with(prefix) {
                out.push(name.clone());
            }
        }
    }
    Ok(out)
}

/// Run a single named fn from `source` with no args, in a fresh
/// interpreter. Returns Ok(()) if the fn returns without raising,
/// Err(msg) if any statement in the body errored.
fn run_named_fn(source: &str, name: &str) -> Result<(), String> {
    // Append a top-level call to the named fn so the regular
    // execute_program path runs it after the rest of the file
    // (including all other fn defs the test might depend on).
    let augmented = format!("{}\n{}();\n", source, name);
    execute_program(&augmented)
}

/// `--audit FILE`: run FILE under both engines (tree-walk + VM)
/// and exit with code 1 if their stdout differs. Catches the class
/// of bug that originally surfaced via the float-truncation issue:
/// both engines silently produced different wrong answers, with no
/// signal that anything was broken.
///
/// Used in CI / pre-merge to guarantee parity. Captures stdout via
/// std::process::Command rather than re-implementing the run path,
/// so it works on any program (and uses the SAME binary the user
/// would run normally).
fn audit_command(path: &str) -> i32 {
    use std::process::Command;
    let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("omnimcode-standalone"));
    let tw_out = match Command::new(&exe).arg(path).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => { eprintln!("audit: tree-walk run failed: {}", e); return 2; }
    };
    let vm_out = match Command::new(&exe).env("OMC_VM", "1").arg(path).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => { eprintln!("audit: VM run failed: {}", e); return 2; }
    };
    if tw_out == vm_out {
        println!("audit: tree-walk and VM produce identical output ({} bytes)", tw_out.len());
        0
    } else {
        eprintln!("audit: ENGINE DIVERGENCE on {}", path);
        // Show first ~10 lines of diff so the user can see where.
        let tw_lines: Vec<&str> = tw_out.lines().collect();
        let vm_lines: Vec<&str> = vm_out.lines().collect();
        let max = tw_lines.len().max(vm_lines.len());
        let mut shown = 0;
        for i in 0..max {
            let tw_l = tw_lines.get(i).copied().unwrap_or("<eof>");
            let vm_l = vm_lines.get(i).copied().unwrap_or("<eof>");
            if tw_l != vm_l {
                eprintln!("  line {}:", i + 1);
                eprintln!("    tree-walk: {}", tw_l);
                eprintln!("    VM:        {}", vm_l);
                shown += 1;
                if shown >= 10 { eprintln!("  (truncated)"); break; }
            }
        }
        1
    }
}

/// `--init`: scaffold a new OMC project in the current directory.
/// Creates `omc.toml` (with no dependencies yet) and `main.omc`
/// (a hello-world). Refuses to overwrite existing files.
fn init_command() -> i32 {
    let toml_path = "omc.toml";
    let main_path = "main.omc";
    if std::path::Path::new(toml_path).exists() {
        eprintln!("init: {} already exists — refusing to overwrite", toml_path);
        return 1;
    }
    if std::path::Path::new(main_path).exists() {
        eprintln!("init: {} already exists — refusing to overwrite", main_path);
        return 1;
    }
    let toml_content = r#"# OMNIcode project manifest. Run `omnimcode-standalone --install`
# to fetch + cache every dependency listed below into omc_modules/.

[package]
name = "my-omc-project"
version = "0.1.0"
description = "an omnicode project"

[dependencies]
# Short names resolve through the central registry (sha256-verified).
# Uncomment as needed:
#
# np      = "np"          # numpy bridge
# pd      = "pd"          # pandas bridge
# sk      = "sklearn"     # scikit-learn bridge
# requests = "requests"   # HTTP client
# sqlite   = "sqlite"     # embedded SQL
#
# You can also pin to a specific URL:
# my_lib   = "https://example.com/raw/my_lib.omc"
"#;
    let main_content = r#"# Welcome to OMNIcode. Edit this file, run with `omnimcode-standalone main.omc`.
println("Hello, harmonic world.");

# Try the embedded Python (always-on):
# h np = py_import("numpy");
# h xs = py_call(np, "arange", [10]);
# println(concat_many("first 10 ints from numpy: ", xs));

# Or import a registry package after `--install`:
# import "np" as np;
# println(concat_many("np.pi = ", np.pi()));
"#;
    if let Err(e) = std::fs::write(toml_path, toml_content) {
        eprintln!("init: write {}: {}", toml_path, e);
        return 1;
    }
    if let Err(e) = std::fs::write(main_path, main_content) {
        eprintln!("init: write {}: {}", main_path, e);
        return 1;
    }
    println!("created {} and {}", toml_path, main_path);
    println!("");
    println!("Next steps:");
    println!("  edit omc.toml — uncomment deps you want");
    println!("  omnimcode-standalone --install");
    println!("  omnimcode-standalone main.omc");
    0
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
    println!("  {} --init              scaffold a new project (omc.toml + main.omc)", prog);
    println!("  {} --install [SPEC]    install package into omc_modules/", prog);
    println!("                         SPEC = URL, registry name, or absent (reads omc.toml)");
    println!("  {} --list              list packages installed under omc_modules/", prog);
    println!("  {} --test FILE         run every fn test_*() in FILE, report pass/fail", prog);
    println!("  {} --bench FILE        run every fn bench_*() in FILE, report ms each", prog);
    println!("  {} --audit FILE        run FILE under both engines, exit 1 on output divergence", prog);
    println!("  {} --help              this message", prog);
    println!();
    println!("omc.toml format (for --install with no arg):");
    println!("  [dependencies]");
    println!("  np = \"np\"                                # registry name");
    println!("  custom = \"https://example.com/raw/x.omc\" # explicit URL");
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
