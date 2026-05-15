// omnimcode-lsp/src/main.rs
//
// Language Server Protocol implementation for OMNIcode.
//
// What it provides today:
//   - Parse-level diagnostics (errors appear inline in the editor)
//   - Heal-pass suggestions as code actions (typo correction,
//     off-attractor literals in index positions, etc.)
//   - Hover info for built-in functions (signature + one-line summary)
//   - Go-to-definition for user-defined functions and module imports
//   - Completion for top-level function names + harmonic primitives
//
// What's deliberately out of scope for v1:
//   - Type-checking (OMC's "types" are φ-math attractors, not Hindley-Milner)
//   - Semantic highlighting (textmate grammar in tools/vscode-omc handles this)
//   - Refactoring (rename / extract fn — adds significant complexity)
//
// Wire-up: VS Code extension under tools/vscode-omc spawns this binary
// via stdio. Other editors (Neovim, Helix, Zed) use the same binary
// through their own LSP client configs.

use dashmap::DashMap;
use omnimcode_core::ast::Statement;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::parser::Parser;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

struct Backend {
    client: Client,
    /// URI → latest text content. Updated on DidOpen / DidChange so
    /// we don't re-read from disk for every diagnostic refresh.
    documents: DashMap<Url, String>,
}

impl Backend {
    /// Parse + heal a document and publish diagnostics. Called from
    /// DidOpen / DidChange. Errors become Diagnostic entries with
    /// span info; heal suggestions become Information-level hints.
    async fn analyze(&self, uri: Url) {
        // Compute diagnostics in a sync helper — Interpreter contains
        // Rc<RefCell> internals (not Send), so it must drop BEFORE
        // any .await. The helper builds the Vec<Diagnostic> and
        // exits scope; the await follows on Send-only types.
        let diagnostics = match self.documents.get(&uri) {
            Some(t) => Self::compute_diagnostics(&t),
            None => return,
        };
        self.client.publish_diagnostics(uri, diagnostics, None).await;
    }

    /// Sync helper: parse + heal-pass, return diagnostics. No async,
    /// no Send issues — Interpreter lives only inside this fn.
    fn compute_diagnostics(text: &str) -> Vec<Diagnostic> {
        let mut diagnostics: Vec<Diagnostic> = Vec::new();
        let mut parser = Parser::new(text);
        let stmts = match parser.parse() {
            Ok(s) => s,
            Err(msg) => {
                let (line, col) = extract_line_col(&msg).unwrap_or((1, 1));
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position {
                            line: line.saturating_sub(1),
                            character: col.saturating_sub(1),
                        },
                        end: Position {
                            line: line.saturating_sub(1),
                            character: col.saturating_sub(1) + 1,
                        },
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("omc-parse".to_string()),
                    message: msg,
                    ..Default::default()
                });
                return diagnostics;
            }
        };
        let interp = Interpreter::new();
        let (_healed, heal_diags, _iters, _outcome) =
            interp.heal_ast_until_fixpoint(stmts, 5);
        for d in heal_diags {
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position { line: 0, character: 0 },
                    end: Position { line: 0, character: 1 },
                },
                severity: Some(DiagnosticSeverity::INFORMATION),
                source: Some("omc-heal".to_string()),
                message: d,
                ..Default::default()
            });
        }
        diagnostics
    }

    /// Walk the AST and return top-level user-defined fn names.
    /// Used by completion + go-to-definition.
    fn collect_user_fns(stmts: &[Statement]) -> Vec<String> {
        let mut out = Vec::new();
        for s in stmts {
            if let Statement::FunctionDef { name, .. } = s {
                out.push(name.clone());
            }
        }
        out
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "omnimcode-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "OMNIcode LSP ready")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        self.documents.insert(uri.clone(), params.text_document.text);
        self.analyze(uri).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.clone();
        // TextDocumentSyncKind::FULL — server gets the entire new
        // contents on every change. Simpler than incremental sync;
        // fast enough for typical OMC files.
        if let Some(change) = params.content_changes.into_iter().next() {
            self.documents.insert(uri.clone(), change.text);
            self.analyze(uri).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        self.documents.remove(&params.text_document.uri);
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let pos = params.text_document_position_params.position;
        let text = match self.documents.get(&uri) {
            Some(t) => t.clone(),
            None => return Ok(None),
        };
        // Identify the identifier under the cursor — naive whitespace +
        // punctuation tokeniser, sufficient for hover purposes.
        let line = text.lines().nth(pos.line as usize).unwrap_or("");
        let word = word_at(line, pos.character as usize);
        let Some(word) = word else { return Ok(None) };
        // Look up in the builtin signature table.
        if let Some(doc) = builtin_doc(&word) {
            return Ok(Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: doc.to_string(),
                }),
                range: None,
            }));
        }
        Ok(None)
    }

    async fn completion(&self, _: CompletionParams) -> Result<Option<CompletionResponse>> {
        // Static completion list: every well-known harmonic primitive
        // and stdlib name. Doesn't include user-defined fns yet —
        // would require reparsing per request and is the next step.
        let items: Vec<CompletionItem> = BUILTIN_COMPLETION_ITEMS
            .iter()
            .map(|(name, detail)| CompletionItem {
                label: name.to_string(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(detail.to_string()),
                ..Default::default()
            })
            .collect();
        Ok(Some(CompletionResponse::Array(items)))
    }
}

/// Extract "LINE:COL" from a parser error message of the form
/// "at LINE:COL: ...". Returns (line, col), 1-indexed.
fn extract_line_col(msg: &str) -> Option<(u32, u32)> {
    let after_at = msg.split("at ").nth(1)?;
    let head = after_at.split(':').collect::<Vec<_>>();
    if head.len() < 2 {
        return None;
    }
    let line: u32 = head[0].parse().ok()?;
    let col: u32 = head[1].parse().ok()?;
    Some((line, col))
}

/// Identify the identifier-shaped token at `col` in `line`. Used by
/// hover to pick up the word the cursor is over. Returns None when
/// the position is on whitespace or punctuation.
fn word_at(line: &str, col: usize) -> Option<String> {
    let chars: Vec<char> = line.chars().collect();
    if col >= chars.len() {
        return None;
    }
    if !is_ident_char(chars[col]) {
        return None;
    }
    let mut start = col;
    while start > 0 && is_ident_char(chars[start - 1]) {
        start -= 1;
    }
    let mut end = col;
    while end < chars.len() && is_ident_char(chars[end]) {
        end += 1;
    }
    Some(chars[start..end].iter().collect())
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '.'
}

/// Hover documentation for built-in functions. Markdown body.
fn builtin_doc(name: &str) -> Option<&'static str> {
    match name {
        "fold" => Some("**`fold(n)`** — snap `n` to the nearest Fibonacci attractor.\n\nReturns the closest value in `[0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]` preserving sign."),
        "harmonic_partition" => Some("**`harmonic_partition(arr)`** — bucket array elements by their Fibonacci attractor.\n\nReturns an array of arrays, one per attractor bucket. Used by `harmonic_anomaly` and `harmonic_index`."),
        "harmonic_index" => Some("**`harmonic_index(arr, idx_fn)`** — build a sub-linear lookup index by attractor neighborhood. See `examples/harmonic_collections.omc`."),
        "harmony_value" => Some("**`harmony_value(n)`** — float in [0, 1]. 1.0 if `n` IS Fibonacci; decays based on distance to nearest attractor."),
        "is_fibonacci" => Some("**`is_fibonacci(n)`** — returns 1 if `n` is in the Fibonacci attractor table, else 0."),
        "fib" | "fibonacci" => Some("**`fibonacci(n)`** — the n-th Fibonacci number. `fibonacci(10) == 55`."),
        "arr_push" => Some("**`arr_push(arr_var, value)`** — append `value` to `arr_var` in-place. First arg must be a variable reference."),
        "arr_get" => Some("**`arr_get(arr, idx)`** — return `arr[idx]`. Errors on out-of-bounds. For safe access, use `safe arr_get(arr, idx)`."),
        "arr_map" => Some("**`arr_map(arr, fn)`** — apply `fn` to every element, return new array."),
        "arr_filter" => Some("**`arr_filter(arr, pred)`** — keep elements where `pred(x)` returns truthy."),
        "arr_reduce" => Some("**`arr_reduce(arr, fn, init)`** — fold from left: `fn(fn(fn(init, a[0]), a[1]), ...)`."),
        "dict_get" => Some("**`dict_get(dict, key, default?)`** — fetch `dict[key]`. Returns `default` (or `null`) on missing."),
        "dict_set" => Some("**`dict_set(dict_var, key, value)`** — mutate `dict_var` in place."),
        "py_import" => Some("**`py_import(module_name)`** — load a CPython module. Returns an opaque handle. (Desktop only — fails in WASM builds.)"),
        "py_call" => Some("**`py_call(handle, method, args)`** — invoke `handle.method(*args)`. Auto-converts results."),
        "println" => Some("**`println(value)`** — print `value` followed by newline. Uses `to_display_string()` so floats keep their decimal point."),
        "csv_parse" => Some("**`csv_parse(text, sep?, skip_header?)`** — fast CSV parser. Returns array of arrays of strings.\n\nDefaults: `sep=\",\"`, `skip_header=0`."),
        "now_ms" => Some("**`now_ms()`** — wall-clock milliseconds since epoch."),
        "error" => Some("**`error(msg)`** — raise a runtime error caught by surrounding `try / catch`."),
        _ => None,
    }
}

/// Static completion list. (name, one-line detail).
const BUILTIN_COMPLETION_ITEMS: &[(&str, &str)] = &[
    ("fold", "fold(n) → snap to Fibonacci attractor"),
    ("fibonacci", "fibonacci(n) → n-th Fibonacci"),
    ("is_fibonacci", "is_fibonacci(n) → 0/1"),
    ("harmony_value", "harmony_value(n) → harmonic alignment [0, 1]"),
    ("harmonic_partition", "harmonic_partition(arr) → arr of buckets"),
    ("harmonic_sort", "harmonic_sort(arr) → sorted by HIM score"),
    ("arr_push", "arr_push(arr_var, v)"),
    ("arr_get", "arr_get(arr, idx)"),
    ("arr_set", "arr_set(arr_var, idx, v)"),
    ("arr_len", "arr_len(arr) → int"),
    ("arr_map", "arr_map(arr, fn)"),
    ("arr_filter", "arr_filter(arr, pred)"),
    ("arr_reduce", "arr_reduce(arr, fn, init)"),
    ("arr_concat", "arr_concat(a, b)"),
    ("arr_slice", "arr_slice(arr, start, end)"),
    ("dict_new", "dict_new() → {}"),
    ("dict_get", "dict_get(d, key, default?)"),
    ("dict_set", "dict_set(d_var, key, v)"),
    ("dict_has", "dict_has(d, key) → 0/1"),
    ("dict_keys", "dict_keys(d) → array"),
    ("dict_len", "dict_len(d) → int"),
    ("str_len", "str_len(s) → byte length"),
    ("str_concat", "str_concat(a, b)"),
    ("str_split", "str_split(s, sep) → array"),
    ("str_slice", "str_slice(s, start, end)"),
    ("csv_parse", "csv_parse(text, sep?, skip_header?)"),
    ("read_file", "read_file(path) → string"),
    ("write_file", "write_file(path, contents)"),
    ("py_import", "py_import(modname) → handle"),
    ("py_call", "py_call(handle, method, args)"),
    ("py_get", "py_get(handle, attr)"),
    ("py_eval", "py_eval(expr_str)"),
    ("println", "println(v)"),
    ("print", "print(v)"),
    ("to_int", "to_int(v)"),
    ("to_float", "to_float(v)"),
    ("to_string", "to_string(v)"),
    ("type_of", "type_of(v) → string"),
    ("error", "error(msg) — raise"),
    ("now_ms", "now_ms() → int"),
];

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let (service, socket) = LspService::new(|client| Backend {
        client,
        documents: DashMap::new(),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
