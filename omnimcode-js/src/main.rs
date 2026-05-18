//! omcj — OMC-to-JavaScript transpiler binary.
//!
//! Usage:
//!   omcj input.omc                     # writes input.js
//!   omcj input.omc -o output.js
//!   omcj input.omc --runtime https://cdn.example.com/omc-runtime.js
//!   omcj input.omc --stdout            # write to stdout

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "omcj",
    about = "OMCJ — OMC to JavaScript transpiler",
    long_about = "Transpiles OMC source files to readable ES2020 modules.\n\
                  The emitted code imports built-ins from `omc-runtime.js` \
                  (copy it next to the output or adjust --runtime).",
    version
)]
struct Args {
    /// Input .omc source file.
    input: PathBuf,

    /// Output .js file.  Defaults to <input>.js next to the input file.
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Import path for the OMC runtime module (default: ./omc-runtime.js).
    #[arg(long, default_value = "./omc-runtime.js", value_name = "PATH")]
    runtime: String,

    /// Write transpiled JS to stdout instead of a file.
    #[arg(long)]
    stdout: bool,

    /// Silence informational messages.
    #[arg(short, long)]
    quiet: bool,
}

fn main() {
    let args = Args::parse();

    // Read source.
    let src = match std::fs::read_to_string(&args.input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("omcj: cannot read {:?}: {e}", args.input);
            std::process::exit(1);
        }
    };

    // Transpile.
    let js = match omnimcode_js::transpile(&src, &args.runtime) {
        Ok(js) => js,
        Err(e) => {
            eprintln!("omcj: transpile error: {e}");
            std::process::exit(1);
        }
    };

    // Output destination.
    if args.stdout {
        print!("{}", js);
        return;
    }

    let out_path = args.output.unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default();
        let dir = args.input.parent().unwrap_or_else(|| std::path::Path::new("."));
        dir.join(stem).with_extension("js")
    });

    match std::fs::write(&out_path, &js) {
        Ok(()) => {
            if !args.quiet {
                eprintln!(
                    "omcj: wrote {} bytes → {:?}",
                    js.len(),
                    out_path
                );
            }
        }
        Err(e) => {
            eprintln!("omcj: cannot write {:?}: {e}", out_path);
            std::process::exit(1);
        }
    }
}
