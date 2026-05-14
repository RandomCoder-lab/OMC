// omnimcode-core/src/main.rs - OMNIcode Standalone Executable Entry Point

use omnimcode_core::parser::Parser;
use omnimcode_core::interpreter::Interpreter;

use std::env;
use std::fs;
use std::io::{self, Write};


fn main() {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            // REPL mode
            repl();
        }
        2 => {
            // File mode
            let filename = &args[1];
            match fs::read_to_string(filename) {
                Ok(content) => {
                    if let Err(e) = execute_program(&content) {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error reading file '{}': {}", filename, e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Usage: {} [program.omc]", args[0]);
            eprintln!("  If no file specified, starts REPL mode");
            std::process::exit(1);
        }
    }
}

fn execute_program(source: &str) -> Result<(), String> {
    let mut parser = Parser::new(source);
    let mut statements = parser.parse()?;

    // OMC_HEAL=1 — run the host-side self-healing pass over the AST
    // before interpretation. Catches harmonic violations, identifier
    // typos, literal divide-by-zero, and arity mismatches at call
    // sites. Diagnostics print to stderr; healed AST executes
    // normally. Same heal classes as the OMC-written self-healing
    // demo in examples/self_healing_h5.omc, but applied to ANY OMC
    // program through the standard toolchain.
    //
    // OMC_HEAL_QUIET=1 suppresses the diagnostic output (heal still
    // happens; just runs silently).
    if std::env::var("OMC_HEAL").as_deref() == Ok("1") {
        let interpreter = Interpreter::new();
        let (healed, diagnostics) = interpreter.heal_ast(statements);
        if !diagnostics.is_empty()
            && std::env::var("OMC_HEAL_QUIET").as_deref() != Ok("1")
        {
            eprintln!("--- OMC_HEAL: {} diagnostic(s) ---", diagnostics.len());
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
