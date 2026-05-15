// omnimcode-wasm/src/lib.rs
//
// WebAssembly interface for OMNIcode. Exposes a small JS-facing API
// for running OMC programs in browsers, Node, or any wasm-bindgen
// host. Pyo3 is excluded (libpython doesn't link in wasm32), so
// `py_*` builtins fail at runtime — that's the intended behavior:
// fail loudly rather than pretend Python is there.
//
// Usage from JS:
//
//     import init, { OmcRuntime } from './pkg/omnimcode_wasm.js';
//     await init();
//     const omc = new OmcRuntime();
//     omc.run("println(fold(7));");              // prints to console
//     const v = omc.eval("3 + 4 * 2");           // returns "11"
//     const r = omc.get_var("x");                 // after `h x = 89;`
//
// The crate ships as a single .wasm + JS glue file via wasm-pack;
// publish to npm with `wasm-pack publish`.

use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use wasm_bindgen::prelude::*;

/// Persistent interpreter instance. JS code creates one per session;
/// state (variables, defined fns, imported modules) survives across
/// `run` / `eval` / `get_var` calls.
#[wasm_bindgen]
pub struct OmcRuntime {
    interp: Interpreter,
}

#[wasm_bindgen]
impl OmcRuntime {
    /// Construct a fresh runtime.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Friendly panic messages in the browser console — helps the
        // first-run debugging experience.
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        OmcRuntime { interp: Interpreter::new() }
    }

    /// Run a complete OMC program. Returns "" on success, an error
    /// message on failure. `println` / `print` output goes to the
    /// JS console (via the default stdout redirection wasm provides).
    pub fn run(&mut self, source: &str) -> Result<(), JsError> {
        let mut parser = Parser::new(source);
        let stmts = parser
            .parse()
            .map_err(|e| JsError::new(&format!("parse: {}", e)))?;
        self.interp
            .execute(stmts)
            .map_err(|e| JsError::new(&e))?;
        Ok(())
    }

    /// Evaluate a single expression and return its result as a string.
    /// Wraps the expression in a `__wasm_result =` binding and pulls
    /// the variable out afterwards — keeps the public API simple
    /// without exposing the Value enum to JS.
    pub fn eval(&mut self, expr: &str) -> Result<String, JsError> {
        let augmented = format!("h __wasm_result = ({});", expr);
        let mut parser = Parser::new(&augmented);
        let stmts = parser
            .parse()
            .map_err(|e| JsError::new(&format!("parse: {}", e)))?;
        self.interp
            .execute(stmts)
            .map_err(|e| JsError::new(&e))?;
        let v = self
            .interp
            .get_var_for_testing("__wasm_result")
            .ok_or_else(|| JsError::new("eval: result not captured"))?;
        Ok(v.to_display_string())
    }

    /// Fetch a top-level variable by name. Returns the value's display
    /// representation (matches what `println` would produce). Returns
    /// `null` if the variable isn't defined.
    pub fn get_var(&self, name: &str) -> Option<String> {
        self.interp
            .get_var_for_testing(name)
            .map(|v| v.to_display_string())
    }

    /// True if a top-level variable is defined.
    pub fn has_var(&self, name: &str) -> bool {
        self.interp.get_var_for_testing(name).is_some()
    }

    /// Reset the runtime to a fresh state — clears variables,
    /// user-defined functions, and imported modules. Useful for
    /// REPL "clear context" patterns.
    pub fn reset(&mut self) {
        self.interp = Interpreter::new();
    }
}

impl Default for OmcRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot stateless eval: parses + runs `source`, returns "" on
/// success. Doesn't preserve any state. Lower-overhead entry point
/// for `omc.execute(...)`-style snippet runners.
#[wasm_bindgen]
pub fn run_once(source: &str) -> Result<(), JsError> {
    let mut parser = Parser::new(source);
    let stmts = parser
        .parse()
        .map_err(|e| JsError::new(&format!("parse: {}", e)))?;
    let mut interp = Interpreter::new();
    interp.execute(stmts).map_err(|e| JsError::new(&e))?;
    Ok(())
}

/// Returns the OMC version string. Useful for "what build am I
/// running?" probes from JS.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
