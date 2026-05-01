// src/main.rs - OMNIcode Standalone Executable Entry Point

mod ast;
mod value;
mod parser;
mod interpreter;
mod runtime;
mod circuits;      // Genetic logic circuits
mod evolution;     // Genetic operators
mod circuit_dsl;   // Circuit DSL and transpiler [Tier 2]
mod optimizer;     // Circuit optimization engine [Tier 3]
mod hbit;          // HBit dual-band processing [Tier 2+]
mod phi_pi_fib;    // O(log_phi_pi_fibonacci n) search algorithm [Tier 4]
mod phi_disk;      // Phi Disk cache system [Tier 4]

use std::env;
use std::fs;
use std::io::{self, Write};
use parser::Parser;
use interpreter::Interpreter;

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
