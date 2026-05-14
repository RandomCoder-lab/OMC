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
    let statements = parser.parse()?;

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
                    "[opt] folded={} dead_loads={} not={} neg={} (total {})",
                    stats.constants_folded,
                    stats.dead_loads_removed,
                    stats.double_nots_collapsed,
                    stats.double_negs_collapsed,
                    stats.total()
                );
            }
        }
        let mut vm = omnimcode_core::vm::Vm::new();
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
