//! MCP server for OMC.
//!
//! Implements just enough of the Model Context Protocol over stdio
//! JSON-RPC for an LLM client (Claude Desktop, Cursor, etc.) to:
//!   - eval OMC code
//!   - introspect the builtin surface (help / list / categories)
//!   - explain runtime errors against the curated catalog
//!   - enumerate OMC-unique primitives so the LLM knows what's
//!     worth reaching for OMC instead of NumPy
//!
//! Protocol: line-delimited JSON-RPC 2.0 over stdin/stdout. The
//! handshake (initialize → initialized notification → tools/list →
//! tools/call) follows MCP. We keep the surface minimal — no
//! resources, no prompts, no sampling, just tools.
//!
//! Configure in Claude Desktop:
//!   {
//!     "mcpServers": {
//!       "omc": { "command": "/path/to/omnimcode-mcp" }
//!     }
//!   }

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use std::io::{self, BufRead, Write};

use omnimcode_core::canonical;
use omnimcode_core::docs;
use omnimcode_core::errors;
use omnimcode_core::interpreter::Interpreter;
use omnimcode_core::memory::MemoryStore;
use omnimcode_core::parser::Parser;
use omnimcode_core::predict::{CodeCorpus, predict_continuations};
use omnimcode_core::tokenizer;
use omnimcode_core::value::Value;

#[derive(Debug, Deserialize)]
struct RpcRequest {
    jsonrpc: String,
    id: Option<Json>,
    method: String,
    #[serde(default)]
    params: Json,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Json,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut interp = Interpreter::new();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req): Result<RpcRequest, _> = serde_json::from_str(&line) else {
            // Garbage on the line — skip it. MCP clients sometimes
            // send junk during startup.
            continue;
        };
        if req.jsonrpc != "2.0" {
            continue;
        }
        // Notifications (no id field) don't get a response.
        let Some(id) = req.id.clone() else {
            // initialized, etc. — acknowledge implicitly.
            continue;
        };

        let response = handle(&mut interp, &req.method, &req.params, id);
        let s = serde_json::to_string(&response).unwrap();
        let _ = writeln!(out, "{}", s);
        let _ = out.flush();
    }
}

fn handle(interp: &mut Interpreter, method: &str, params: &Json, id: Json) -> RpcResponse {
    match method {
        "initialize" => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "omnimcode-mcp",
                    "version": "1.0.0"
                }
            })),
            error: None,
        },
        "tools/list" => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "tools": list_tools() })),
            error: None,
        },
        "tools/call" => {
            let name = params.get("name").and_then(Json::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match dispatch_tool(interp, name, &args) {
                Ok(text) => {
                    let final_text = maybe_auto_summarize(text);
                    RpcResponse {
                        jsonrpc: "2.0",
                        id,
                        result: Some(json!({
                            "content": [{ "type": "text", "text": final_text }],
                            "isError": false
                        })),
                        error: None,
                    }
                },
                Err(msg) => RpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({
                        "content": [{ "type": "text", "text": msg }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        }
        _ => RpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code: -32601,
                message: format!("Method not found: {}", method),
            }),
        },
    }
}

/// Tool catalog exposed to MCP clients. Keep descriptions punchy —
/// the LLM uses them to decide which tool to call.
fn list_tools() -> Vec<Json> {
    vec![
        json!({
            "name": "omc_eval",
            "description": "Evaluate OMC source code and return stdout. Use this to run OMC programs, test snippets, or compute results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "OMC source code to evaluate." }
                },
                "required": ["code"]
            }
        }),
        json!({
            "name": "omc_help",
            "description": "Look up signature + description + example for an OMC builtin. Returns 'did you mean' suggestions on miss.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Builtin name, e.g. arr_softmax" }
                },
                "required": ["name"]
            }
        }),
        json!({
            "name": "omc_list_builtins",
            "description": "List all documented OMC builtins, optionally filtered by category (substrate, ml_kernels, autograd, generators, ...).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": { "type": "string", "description": "Optional category filter." }
                }
            }
        }),
        json!({
            "name": "omc_categories",
            "description": "List all builtin categories. Use this before omc_list_builtins to see what's available.",
            "inputSchema": { "type": "object" }
        }),
        json!({
            "name": "omc_unique_builtins",
            "description": "List OMC-unique builtins with NO Python/NumPy equivalent. These are the reason to reach for OMC over numpy: substrate-aware primitives, harmonic ops, native lazy generators.",
            "inputSchema": { "type": "object" }
        }),
        json!({
            "name": "omc_explain_error",
            "description": "Given an OMC error message, return a structured explanation: what it means, typical cause, one-line fix.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The OMC error message." }
                },
                "required": ["message"]
            }
        }),
        json!({
            "name": "omc_did_you_mean",
            "description": "Closest known builtin names for a typo. Useful when you've guessed a name that doesn't exist.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The (probably wrong) name." }
                },
                "required": ["name"]
            }
        }),
        json!({
            "name": "omc_predict",
            "description": "Substrate-indexed code completion. Given a partial OMC code prefix \
                            (e.g. `fn prom_linear_`), return the top-k ranked continuations from \
                            a content-addressed corpus of OMC files. Each result is a viable \
                            branch.\n\
                            \n\
                            The `format` arg controls how much context each suggestion costs:\n\
                            - `hash` (default, ~50 bytes/suggestion): fn_name + file + \
                              canonical_hash + substrate_distance. Use this for browsing — \
                              cheap context. Fetch the body on demand with omc_fetch_by_hash.\n\
                            - `signature` (~100 bytes/suggestion): adds the fn signature line. \
                              Enough for an LLM to know the call shape.\n\
                            - `full`: includes the complete source. Use only when you'll \
                              actually edit/adapt the body.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths OR directories to ingest. Directories are walked recursively for .omc files — pass `examples/lib` to query against the entire lib tree."
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Partial OMC source (e.g. `fn prom_linear_`). May be incomplete."
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 5,
                        "description": "Number of ranked continuations to return."
                    },
                    "format": {
                        "type": "string",
                        "enum": ["hash", "signature", "codec", "full"],
                        "default": "hash",
                        "description": "Response detail level. See tool description."
                    }
                },
                "required": ["paths", "prefix"]
            }
        }),
        json!({
            "name": "omc_corpus_size",
            "description": "Diagnostic: report how many top-level fns are ingested across a list \
                            of OMC source paths. Useful for verifying paths resolve before \
                            building a larger predict query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths to ingest."
                    }
                },
                "required": ["paths"]
            }
        }),
        json!({
            "name": "omc_compress_context",
            "description": "Compress an arbitrary OMC source string into a substrate-keyed \
                            codec payload. Returns a dict with a canonical_hash (alpha-rename \
                            invariant identity) plus sampled_tokens (structural thumbnail). \
                            The LLM can hold the compressed payload in context as a cheap \
                            reference, then recover the original source via omc_decompress \
                            against a corpus that contains the same canonical form.\n\
                            \n\
                            Symmetric to omc_fetch_by_hash but for arbitrary text instead \
                            of pre-indexed corpus entries. Use when the LLM wants to remember \
                            a chunk of code it's just seen without paying its full byte cost.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "OMC source string to compress."
                    },
                    "every_n": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 3,
                        "description": "Token sampling stride. 1 = keep all tokens (no compression, useful for lossless transport). 3 (default) gives ~3x token-count reduction."
                    }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "omc_decompress",
            "description": "Recover the original OMC source from a substrate-keyed codec \
                            payload (or just a canonical_hash) by library lookup against a \
                            corpus. Returns {found, source, fn_name, file} on hit or \
                            {found: false} on miss.\n\
                            \n\
                            Generalizes omc_fetch_by_hash: accepts either a full codec \
                            payload (dict with content_hash) or a bare canonical_hash int. \
                            Lookup is alpha-rename invariant — works even if the fn was \
                            renamed in source after compression.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths to search for a matching canonical form."
                    },
                    "codec": {
                        "type": "object",
                        "description": "Codec payload from omc_compress_context. Either this or canonical_hash is required."
                    },
                    "canonical_hash": {
                        "type": "integer",
                        "description": "Bare canonical hash. Either this or codec is required."
                    }
                },
                "required": ["paths"]
            }
        }),
        json!({
            "name": "omc_fetch_by_hash",
            "description": "Recover a function body by its canonical hash. The companion to \
                            omc_predict with format=hash: the LLM browses cheaply via hash \
                            digests, then fetches the actual source only when ready to use \
                            it. Walks the same paths corpus as omc_predict; returns the full \
                            source of the matching fn, or notFound:true if no fn in the \
                            corpus has that hash.\n\
                            \n\
                            The canonical_hash is alpha-rename invariant — a fn that's been \
                            renamed still recovers from the same hash.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Source file paths to search."
                    },
                    "canonical_hash": {
                        "type": "integer",
                        "description": "The canonical_hash returned by a previous omc_predict call."
                    }
                },
                "required": ["paths", "canonical_hash"]
            }
        }),
        json!({
            "name": "omc_memory_store",
            "description": "Substrate-keyed conversation memory: persist a chunk of text \
                            (an agent turn, a reasoning trace, a piece of context the LLM \
                            wants to remember later) content-addressed by canonical hash. \
                            Returns {content_hash, namespace, bytes}. The hash is the same \
                            primitive as omc_compress_context's content_hash — they're \
                            interchangeable.\n\
                            \n\
                            Survives MCP process restart (filesystem-backed at \
                            ~/.omc/memory/<namespace>/). Use a per-conversation namespace \
                            (e.g. \"agent_<session_id>\") to keep threads separate.\n\
                            \n\
                            Together with omc_memory_recall, lets an LLM agent's prior turns \
                            stay in cheap reference form (a hash) in the current context, \
                            recovering full content only when reasoning needs it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Content to store. Can be OMC source, prose, JSON, or any UTF-8 text."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "Logical partition. Sanitized to ASCII alphanumeric + _-."
                    }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "omc_memory_recall",
            "description": "Recover stored text by canonical hash. Returns {found, text, ...} \
                            or {found: false} if no namespace contains an entry with that \
                            hash. If namespace is given, only that namespace is searched; \
                            otherwise, every namespace under the memory root is walked.\n\
                            \n\
                            Companion to omc_memory_store. Together they let prior agent \
                            turns stay in hash form in the current context, recovered on \
                            demand only when reasoning needs them.\n\
                            \n\
                            **v0.12.1: prefer `content_hash_str` (decimal string) over \
                            `content_hash` (integer) for any hash > 2^53 ≈ 9e15.** JSON's \
                            number type is f64 and silently rounds large ints. The store \
                            response always includes both forms; pass back the string form \
                            to be safe.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "integer",
                        "description": "Hash returned by a prior omc_memory_store. Lossy above 2^53."
                    },
                    "content_hash_str": {
                        "type": "string",
                        "description": "Decimal-string form. Lossless. Preferred for hashes > 2^53."
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional. If omitted, searches all namespaces."
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_recall_summary",
            "description": "v0.12.0 Axis 7 — high-leverage summary recall. Returns ~100-300 \
                            bytes of `what is this content` metadata (content_hash, byte_count, \
                            first_line, preview, attractor) instead of the full body. \
                            **Lossless** — the verbatim is always still recoverable via \
                            omc_memory_recall.\n\
                            \n\
                            Real measured savings on 100KB body: ~400× context-token reduction. \
                            Designed for the **list-then-recall** workflow: get cheap previews \
                            of many candidate hashes, pick the relevant one, issue a single \
                            full recall.\n\
                            \n\
                            Best paired with omc_memory_list which gives you the hashes; then \
                            walk them through recall_summary; then recall the one(s) that matter.\n\
                            \n\
                            **v0.12.1: prefer `content_hash_str` (decimal string) for hashes > 2^53.**",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {"type": "integer", "description": "Lossy above 2^53."},
                    "content_hash_str": {"type": "string", "description": "Decimal-string form. Preferred for large hashes."},
                    "namespace": {"type": "string"}
                }
            }
        }),
        json!({
            "name": "omc_memory_recall_codec",
            "description": "v0.12.0 Axis 7 — codec-form recall for context-cost reduction. \
                            Returns a substrate-codec payload (content_hash + every-N sampled \
                            tokens + phi_pi_fib attractor + sizing metadata) instead of the \
                            full text. **Lossless** — the verbatim body remains recoverable \
                            via omc_memory_recall with the same content_hash.\n\
                            \n\
                            Honest savings on 100KB content (measured): every_n=5 → 1.5× \
                            context savings, every_n=13 → 3.8×, every_n=21 → 6.2×. JSON \
                            tokens cost ~10 bytes each, so savings only kick in past stride \
                            5. Don't expect 50-500×; expect 2-6× at reasonable strides.\n\
                            \n\
                            Use this when the LLM has a structural fingerprint use case (e.g., \
                            verifying that two entries describe the same content via attractor \
                            equality, or remembering 'I've seen this hash before' without \
                            re-reading the body) — not as a general full-text replacement.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "integer",
                        "description": "Hash returned by a prior omc_memory_store. Lossy above 2^53."
                    },
                    "content_hash_str": {
                        "type": "string",
                        "description": "Decimal-string form. Preferred for hashes > 2^53."
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional. If omitted, searches all namespaces."
                    },
                    "every_n": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "description": "Sampling stride; higher = smaller + lossier."
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_list",
            "description": "Browse a namespace's stored entries, most recent first. Each \
                            entry has {content_hash, bytes, stored_at_unix, preview}. The \
                            preview is the first ~80 chars of the text, stripped of \
                            newlines — enough to disambiguate when picking which entry to \
                            recall.\n\
                            \n\
                            Use to see what an agent has stored without paying the byte \
                            cost of recalling every entry. Limit defaults to 20.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "Namespace to browse."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 20,
                        "description": "Maximum entries to return."
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_stats",
            "description": "Diagnostic: total entries and stored bytes for a namespace, plus \
                            the configured fibtier cap. Useful for an agent to know how \
                            much of its memory budget is in use.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default"
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_evict",
            "description": "Manually prune a namespace's index down to the most recent \
                            `keep` entries. Body files on disk are NOT removed — an LLM \
                            with the hash can still recall. Use to force-bound memory \
                            growth, or to compact a long-running agent's state at a \
                            session boundary.\n\
                            \n\
                            Returns {dropped, kept}. The default fibtier behavior runs \
                            this automatically after each store using OMC_MEMORY_MAX_ENTRIES \
                            (default 232 = sum of first 10 Fibonacci tier sizes); this \
                            tool exposes manual control.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default"
                    },
                    "keep": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of most-recent entries to retain. 0 clears the index entirely."
                    }
                },
                "required": ["keep"]
            }
        }),
        json!({
            "name": "omc_memory_store_delta",
            "description": "v0.10.1 Axis 5 — store text as a delta against an explicit base \
                            entry. Useful for iterative drafts: store v1 normally, then v2/v3 \
                            as deltas off v1. Each delta is roughly constant size if the \
                            edits are localized. Falls back to a regular store if the prefix \
                            shared with base is <64 bytes or the delta wouldn't actually save \
                            space.\n\
                            \n\
                            Bodies are tagged with `OMCD` magic and rebuilt on recall by \
                            fetching the base. Returns the same hash you'd get from a regular \
                            store (hash of the FULL text), so other tools work unchanged.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "default": "default"},
                    "text": {"type": "string", "description": "The new content (full text, not a diff)."},
                    "base_hash": {"type": "integer", "description": "Base hash. Lossy above 2^53; prefer base_hash_str."},
                    "base_hash_str": {"type": "string", "description": "Decimal-string form of base hash. Lossless."}
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "omc_memory_compact_substrate",
            "description": "v0.10.0 Axis 4 — substrate-tokenizer compaction. Re-encodes \
                            aged pool bodies through the OMC substrate tokenizer (encode + \
                            varint pack + deflate). Wins on OMC-flavored content because the \
                            substrate dictionary already exploits OMC syntax patterns; falls \
                            back gracefully on prose (the rewrite is skipped when it doesn't \
                            save ≥16 bytes).\n\
                            \n\
                            Bodies are tagged with the 4-byte `OMCT` magic and inflated \
                            transparently on recall.\n\
                            \n\
                            Returns the same shape as omc_memory_compact. Schedule both: \
                            run omc_memory_compact_substrate first (best for OMC content), \
                            then omc_memory_compact (fallback for everything else).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default"
                    },
                    "age_threshold_secs": {
                        "type": "integer",
                        "default": 86400,
                        "minimum": 0
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_compact_bpe",
            "description": "v0.11.2 SBPE — self-training BPE codec. First axis to actually \
                            beat plain zlib on real content. Trains a per-body byte-pair \
                            encoding (512 greedy frequency merges by default), then ships \
                            the merge table + token stream as two zlib-deflated blobs. \
                            The data trains its own vocabulary at compression time and the \
                            merge table travels inline.\n\
                            \n\
                            Measured 5.21× on 100KB native .omc vs 4.70× for plain zlib \
                            (Axis 3 / OMCZ). Header amortizes for bodies ≥16KB; smaller \
                            bodies fall back to no-op (the safety check skips when SBPE \
                            doesn't save ≥16 bytes vs raw).\n\
                            \n\
                            Bodies tagged with `OMCB` magic, transparently decompressed on \
                            recall. Use as a replacement for omc_memory_compact when content \
                            is large enough to amortize the inline merge table — for cold \
                            archival of substantial bodies, this is now the best axis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "default": "default"},
                    "age_threshold_secs": {"type": "integer", "default": 86400, "minimum": 0}
                }
            }
        }),
        json!({
            "name": "omc_memory_compact_hbit",
            "description": "v0.11.0 Axis 6 — HBit dual-band codec. Substrate-tokenize each \
                            aged body, then split each i64 token id into a high-32-bit band \
                            and a low-32-bit band. Each band is zigzag-delta-varint-packed \
                            and deflated separately. Wins when the two bands have different \
                            entropy distributions, which is typical for substrate-tokenized \
                            natural language (the hi band changes more slowly than the lo \
                            band as tokens cluster within substrate attractor neighborhoods).\n\
                            \n\
                            Bodies tagged with `OMCH` magic, transparently rebuilt on recall. \
                            Skips entries already in any compressed form. Falls back when the \
                            two-band layout doesn't save ≥16 bytes vs the raw body.\n\
                            \n\
                            Schedule: try omc_memory_compact_hbit first on substrate-friendly \
                            content; fall back to omc_memory_compact_substrate, then \
                            omc_memory_compact. Returns {compacted, bytes_before, bytes_after}.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "default": "default"},
                    "age_threshold_secs": {"type": "integer", "default": 86400, "minimum": 0}
                }
            }
        }),
        json!({
            "name": "omc_memory_compact",
            "description": "v0.9.3 Axis 3 — fibtier-aware progressive compression. \
                            Walk a namespace's index and rewrite pool bodies older than \
                            `age_threshold_secs` as zlib-deflated blobs (3-10× smaller on \
                            disk). Recall path transparently inflates them; content is \
                            unchanged from the LLM's perspective. Aged-content compression \
                            stacks on top of Axis 2 dedup.\n\
                            \n\
                            Returns {compacted, bytes_before, bytes_after}. Skips entries \
                            already in OMCZ form. Skips entries where deflate doesn't save \
                            at least 16 bytes (small high-entropy text can EXPAND under \
                            deflate).\n\
                            \n\
                            Typical use: schedule a daily compact for namespaces older \
                            than 86400 (1 day). Or fold into a session-boundary hook.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default"
                    },
                    "age_threshold_secs": {
                        "type": "integer",
                        "default": 86400,
                        "minimum": 0,
                        "description": "Only entries older than this (in seconds since stored_at) are compacted. 0 = compact everything."
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_create_manifest",
            "description": "v0.9.1 Axis 1 — Merkle manifest hashes. Bundle N leaf \
                            content_hashes into ONE manifest hash. The LLM holds the manifest \
                            hash in context (~5 tokens) and expands on demand via \
                            omc_memory_recall_manifest, which returns the leaf list. Leaves are \
                            then recalled individually only when needed. Compression on the \
                            'reference cost in context' axis grows linearly with N: 100 entries \
                            = 1 manifest hash in context instead of 100 hashes.\n\
                            \n\
                            The manifest is itself a regular memory entry (stored with body \
                            `{\"manifest\":1,\"entries\":[..]}`) so it persists across MCP restart \
                            and can be evicted/listed like any other entry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "default",
                        "description": "Namespace the manifest lives in. Leaf hashes can come from any namespace; the manifest just references them."
                    },
                    "entries": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Leaf content_hashes from prior omc_memory_store. Lossy above 2^53; prefer entries_str."
                    },
                    "entries_str": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Decimal-string forms of leaf hashes. Lossless. Preferred."
                    }
                }
            }
        }),
        json!({
            "name": "omc_memory_recall_manifest",
            "description": "Recall a manifest hash and return the leaf list. If `expand` is true, \
                            also fetches each leaf's full text in one call (use when you know \
                            you'll need all leaves; cheaper than N round-trips).\n\
                            \n\
                            Returns {entries: [leaf_hashes]} OR {entries: [leaf_hashes], \
                            expanded: [{hash, text}, ...]}. If the hash points at a regular \
                            (non-manifest) entry, returns {is_manifest: false, text: <body>}.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Optional. If omitted, searches all namespaces."
                    },
                    "content_hash": {
                        "type": "integer",
                        "description": "Manifest hash. Lossy above 2^53; prefer content_hash_str."
                    },
                    "content_hash_str": {
                        "type": "string",
                        "description": "Decimal-string form of the manifest hash. Lossless."
                    },
                    "expand": {
                        "type": "boolean",
                        "default": false,
                        "description": "If true, recall every leaf in one call."
                    }
                }
            }
        }),
    ]
}

/// v0.12.1 — robust hash argument reader. JSON's number type can only
/// faithfully represent integers up to 2^53; any FNV1a 64-bit hash above
/// that gets silently rounded to the nearest f64 by the LLM/MCP client
/// layer, making the entry unrecoverable. Accept the hash as either an
/// `integer` (legacy, lossy above 2^53) or a `string` (decimal, lossless).
/// Prefer the string form when both are present.
fn read_hash_arg(args: &Json, tool: &str) -> Result<i64, String> {
    if let Some(s) = args.get("content_hash_str").and_then(Json::as_str) {
        return s.parse::<i64>().map_err(|e|
            format!("{}: 'content_hash_str' is not a valid i64: {}", tool, e));
    }
    args.get("content_hash").and_then(Json::as_i64)
        .ok_or_else(|| format!(
            "{}: missing 'content_hash' (integer) or 'content_hash_str' (decimal string). \
             Prefer 'content_hash_str' for hashes > 2^53 to avoid JSON-float precision loss.",
            tool))
}

/// v0.12.1 — emit a hash in both forms so the caller can pass back the
/// lossless string version. Inserts both `content_hash` and `content_hash_str`.
fn hash_fields(h: i64) -> serde_json::Map<String, Json> {
    let mut m = serde_json::Map::new();
    m.insert("content_hash".to_string(), json!(h));
    m.insert("content_hash_str".to_string(), json!(h.to_string()));
    m
}

/// v0.13.0 Option-A — smart-response MCP.
///
/// Wraps a dispatched tool result. If `OMC_MCP_AUTO_SUMMARY=1` and the
/// response carries a `text` field bigger than the threshold (default 1024
/// bytes, override via `OMC_MCP_AUTO_SUMMARY_THRESHOLD`), the full text is
/// cached in the MemoryStore (`_auto_summary_cache` namespace) and the
/// LLM-facing response is rewritten to a tiny envelope with the
/// `expand_with` instructions.
///
/// The LLM then decides: use the preview, or call
/// `omc_memory_recall(content_hash_str=..., namespace=_auto_summary_cache)`
/// to fetch the full body. For sessions where the LLM only needs the
/// preview ~60-80% of the time, this is a real 2-5× LLM token saving on
/// recall-heavy workflows. Lossless — the full body is always recoverable.
fn maybe_auto_summarize(raw_response: String) -> String {
    if std::env::var("OMC_MCP_AUTO_SUMMARY").ok().as_deref() != Some("1") {
        return raw_response;
    }
    let threshold: usize = std::env::var("OMC_MCP_AUTO_SUMMARY_THRESHOLD")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    if raw_response.len() < threshold * 2 {
        return raw_response;  // not worth the rewrite framing
    }
    let mut v: Json = match serde_json::from_str(&raw_response) {
        Ok(v) => v,
        Err(_) => return raw_response,
    };
    // Only trigger on responses carrying a long `text` field.
    let text_len = v.get("text").and_then(Json::as_str)
        .map(|s| s.len()).unwrap_or(0);
    if text_len < threshold { return raw_response; }
    let text = v.get("text").and_then(Json::as_str).unwrap().to_string();
    let store = MemoryStore::from_env();
    let hash = match store.store("_auto_summary_cache", &text) {
        Ok(h) => h,
        Err(_) => return raw_response,
    };
    let preview: String = text.chars()
        .filter(|c| !c.is_control())
        .take(200).collect();
    if let Json::Object(ref mut map) = v {
        map.remove("text");
        map.insert("_auto_summarized".to_string(), json!(true));
        map.insert("preview".to_string(), json!(preview));
        map.insert("original_byte_count".to_string(), json!(text.len()));
        map.insert("expand_with".to_string(), json!({
            "tool": "omc_memory_recall",
            "content_hash_str": hash.to_string(),
            "namespace": "_auto_summary_cache",
            "note": "Call this tool to retrieve the full body if the preview \
                     isn't enough. The body is cached losslessly under this hash."
        }));
    }
    serde_json::to_string_pretty(&v).unwrap_or(raw_response)
}

fn dispatch_tool(interp: &mut Interpreter, name: &str, args: &Json) -> Result<String, String> {
    match name {
        "omc_eval" => {
            let code = args.get("code").and_then(Json::as_str)
                .ok_or_else(|| "omc_eval: missing 'code' arg".to_string())?;
            eval_program(interp, code)
        }
        "omc_help" => {
            let name = args.get("name").and_then(Json::as_str)
                .ok_or_else(|| "omc_help: missing 'name' arg".to_string())?;
            match docs::lookup(name) {
                Some(d) => Ok(serde_json::to_string_pretty(&json!({
                    "name": d.name,
                    "category": d.category,
                    "signature": d.signature,
                    "description": d.description,
                    "example": d.example,
                    "unique_to_omc": d.unique_to_omc
                })).unwrap()),
                None => {
                    let suggestions = docs::did_you_mean(name, 5);
                    Ok(serde_json::to_string_pretty(&json!({
                        "found": false,
                        "name": name,
                        "did_you_mean": suggestions
                    })).unwrap())
                }
            }
        }
        "omc_list_builtins" => {
            let cat = args.get("category").and_then(Json::as_str);
            let names = docs::names_in(cat);
            Ok(serde_json::to_string_pretty(&json!(names)).unwrap())
        }
        "omc_categories" => {
            let cats = docs::categories();
            Ok(serde_json::to_string_pretty(&json!(cats)).unwrap())
        }
        "omc_unique_builtins" => {
            let names: Vec<&str> = docs::BUILTINS.iter()
                .filter(|b| b.unique_to_omc)
                .map(|b| b.name)
                .collect();
            Ok(serde_json::to_string_pretty(&json!(names)).unwrap())
        }
        "omc_explain_error" => {
            let msg = args.get("message").and_then(Json::as_str)
                .ok_or_else(|| "omc_explain_error: missing 'message' arg".to_string())?;
            match errors::match_error(msg) {
                Some(p) => Ok(serde_json::to_string_pretty(&json!({
                    "matched": true,
                    "pattern": p.pattern,
                    "category": p.category,
                    "explanation": p.explanation,
                    "typical_cause": p.typical_cause,
                    "fix": p.fix
                })).unwrap()),
                None => Ok(serde_json::to_string_pretty(&json!({
                    "matched": false,
                    "explanation": "No catalog pattern matched. Try `omc_did_you_mean` if it looks like a typo."
                })).unwrap()),
            }
        }
        "omc_did_you_mean" => {
            let name = args.get("name").and_then(Json::as_str)
                .ok_or_else(|| "omc_did_you_mean: missing 'name' arg".to_string())?;
            let suggestions = docs::did_you_mean(name, 5);
            Ok(serde_json::to_string_pretty(&json!(suggestions)).unwrap())
        }
        "omc_predict" => {
            let paths = parse_paths_arg(args, "omc_predict")?;
            let prefix = args.get("prefix").and_then(Json::as_str)
                .ok_or_else(|| "omc_predict: missing 'prefix' arg".to_string())?;
            // top_k optional, defaults to 5. Clamp to [1, 50] so a
            // misconfigured client can't ask for the entire corpus.
            let top_k = args.get("top_k").and_then(Json::as_i64)
                .unwrap_or(5)
                .clamp(1, 50) as usize;
            let format = args.get("format")
                .and_then(Json::as_str)
                .unwrap_or("hash");
            let corpus = build_corpus(&paths)?;
            let suggestions = predict_continuations(&corpus, prefix, top_k);
            let suggestion_jsons: Vec<Json> = suggestions.iter()
                .map(|s| project_suggestion(s, format))
                .collect();
            let payload = json!({
                "prefix": prefix,
                "corpus_size": corpus.len(),
                "top_k": top_k,
                "format": format,
                "suggestions": suggestion_jsons,
            });
            Ok(serde_json::to_string_pretty(&payload).unwrap())
        }
        "omc_corpus_size" => {
            let paths = parse_paths_arg(args, "omc_corpus_size")?;
            let corpus = build_corpus(&paths)?;
            let payload = json!({
                "paths": paths,
                "fn_count": corpus.len(),
            });
            Ok(serde_json::to_string_pretty(&payload).unwrap())
        }
        "omc_fetch_by_hash" => {
            let paths = parse_paths_arg(args, "omc_fetch_by_hash")?;
            let target = args.get("canonical_hash").and_then(Json::as_i64)
                .ok_or_else(|| "omc_fetch_by_hash: missing 'canonical_hash' (i64) arg".to_string())?;
            let corpus = build_corpus(&paths)?;
            match corpus.entries.iter().find(|e| e.canonical_hash == target) {
                Some(entry) => {
                    let payload = json!({
                        "found": true,
                        "canonical_hash": entry.canonical_hash,
                        "fn_name": entry.fn_name,
                        "file": entry.file,
                        "source": entry.source,
                    });
                    Ok(serde_json::to_string_pretty(&payload).unwrap())
                }
                None => {
                    let payload = json!({
                        "found": false,
                        "canonical_hash": target,
                        "searched_paths": paths,
                        "corpus_size": corpus.len(),
                    });
                    Ok(serde_json::to_string_pretty(&payload).unwrap())
                }
            }
        }
        "omc_compress_context" => {
            let text = args.get("text").and_then(Json::as_str)
                .ok_or_else(|| "omc_compress_context: missing 'text' arg".to_string())?;
            let every_n = args.get("every_n").and_then(Json::as_i64)
                .unwrap_or(3)
                .max(1) as usize;
            let codec = encode_codec_payload(text, every_n);
            // Caller-facing payload: codec dict + the text length so the
            // LLM can compute its own compression ratio against the JSON
            // it receives (vs the raw input it had).
            let payload = json!({
                "original_bytes": text.len(),
                "codec": codec,
            });
            Ok(serde_json::to_string_pretty(&payload).unwrap())
        }
        "omc_memory_store" => {
            let text = args.get("text").and_then(Json::as_str)
                .ok_or_else(|| "omc_memory_store: missing 'text' arg".to_string())?;
            let namespace = args.get("namespace").and_then(Json::as_str)
                .unwrap_or("default");
            let store = MemoryStore::from_env();
            let hash = store.store(namespace, text)?;
            let mut resp = hash_fields(hash);
            resp.insert("namespace".to_string(), json!(namespace));
            resp.insert("bytes".to_string(), json!(text.len()));
            Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
        }
        "omc_memory_recall_summary" => {
            let target = read_hash_arg(args, "omc_memory_recall_summary")?;
            let namespace = args.get("namespace").and_then(Json::as_str);
            let store = MemoryStore::from_env();
            match store.recall_summary(namespace, target)? {
                Some(p) => {
                    let mut resp = hash_fields(p.content_hash);
                    resp.insert("found".to_string(), json!(true));
                    resp.insert("byte_count".to_string(), json!(p.byte_count));
                    resp.insert("first_line".to_string(), json!(p.first_line));
                    resp.insert("preview".to_string(), json!(p.preview));
                    resp.insert("attractor".to_string(), json!(p.attractor));
                    resp.insert("attractor_str".to_string(), json!(p.attractor.to_string()));
                    Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
                }
                None => {
                    let mut resp = hash_fields(target);
                    resp.insert("found".to_string(), json!(false));
                    resp.insert("namespace".to_string(), json!(namespace));
                    Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
                }
            }
        }
        "omc_memory_recall_codec" => {
            let target = read_hash_arg(args, "omc_memory_recall_codec")?;
            let namespace = args.get("namespace").and_then(Json::as_str);
            let every_n = args.get("every_n").and_then(Json::as_u64).unwrap_or(3) as usize;
            let want_array = args.get("include_tokens_array").and_then(Json::as_bool).unwrap_or(false);
            let store = MemoryStore::from_env();
            match store.recall_codec(namespace, target, every_n)? {
                Some(payload) => {
                    let mut resp = hash_fields(payload.content_hash);
                    resp.insert("found".to_string(), json!(true));
                    resp.insert("sampled_tokens_packed".to_string(), json!(payload.sampled_tokens_packed));
                    resp.insert("sampled_tokens".to_string(),
                        if want_array { json!(payload.sampled_tokens) } else { json!(null) });
                    resp.insert("sampled_token_count".to_string(), json!(payload.sampled_tokens.len()));
                    resp.insert("attractor".to_string(), json!(payload.attractor));
                    resp.insert("attractor_str".to_string(), json!(payload.attractor.to_string()));
                    resp.insert("every_n".to_string(), json!(payload.every_n));
                    resp.insert("original_byte_count".to_string(), json!(payload.original_byte_count));
                    resp.insert("original_token_count".to_string(), json!(payload.original_token_count));
                    resp.insert("compression_ratio".to_string(), json!(payload.compression_ratio));
                    Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
                }
                None => {
                    let mut resp = hash_fields(target);
                    resp.insert("found".to_string(), json!(false));
                    resp.insert("namespace".to_string(), json!(namespace));
                    Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
                }
            }
        }
        "omc_memory_recall" => {
            let target = read_hash_arg(args, "omc_memory_recall")?;
            let namespace = args.get("namespace").and_then(Json::as_str);
            let store = MemoryStore::from_env();
            match store.recall(namespace, target)? {
                Some(text) => {
                    let mut resp = hash_fields(target);
                    resp.insert("found".to_string(), json!(true));
                    resp.insert("bytes".to_string(), json!(text.len()));
                    resp.insert("text".to_string(), json!(text));
                    Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
                }
                None => {
                    let mut resp = hash_fields(target);
                    resp.insert("found".to_string(), json!(false));
                    resp.insert("namespace".to_string(), json!(namespace));
                    Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
                }
            }
        }
        "omc_memory_list" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let limit = args.get("limit").and_then(Json::as_i64).unwrap_or(20)
                .clamp(1, 1000) as usize;
            let store = MemoryStore::from_env();
            let entries = store.list(namespace, limit)?;
            let entry_jsons: Vec<Json> = entries.iter().map(|e| json!({
                "content_hash": e.content_hash,
                "content_hash_str": e.content_hash.to_string(),
                "bytes": e.bytes,
                "stored_at_unix": e.stored_at_unix,
                "preview": e.preview,
            })).collect();
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "count": entries.len(),
                "entries": entry_jsons,
            })).unwrap())
        }
        "omc_memory_stats" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let store = MemoryStore::from_env();
            let (count, bytes) = store.stats(namespace)?;
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "total_entries": count,
                "total_bytes": bytes,
                "fibtier_cap": store.max_entries_per_namespace,
            })).unwrap())
        }
        "omc_memory_evict" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let keep = args.get("keep").and_then(Json::as_i64)
                .ok_or_else(|| "omc_memory_evict: missing 'keep' (i64) arg".to_string())?
                .max(0) as usize;
            let store = MemoryStore::from_env();
            let dropped = store.evict_to_cap(namespace, keep)?;
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "dropped": dropped,
                "kept": keep,
            })).unwrap())
        }
        "omc_memory_store_delta" => {
            let text = args.get("text").and_then(Json::as_str)
                .ok_or_else(|| "omc_memory_store_delta: missing 'text'".to_string())?;
            // v0.12.1: accept base_hash as int OR base_hash_str as decimal string
            let base = if let Some(s) = args.get("base_hash_str").and_then(Json::as_str) {
                s.parse::<i64>().map_err(|e|
                    format!("omc_memory_store_delta: 'base_hash_str' not a valid i64: {}", e))?
            } else {
                args.get("base_hash").and_then(Json::as_i64)
                    .ok_or_else(|| "omc_memory_store_delta: missing 'base_hash' (integer) or \
                                    'base_hash_str' (decimal string)".to_string())?
            };
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let store = MemoryStore::from_env();
            let hash = store.store_as_delta(namespace, text, base)?;
            let pool_p = format!("{}", store.root.display());
            let mut resp = hash_fields(hash);
            resp.insert("namespace".to_string(), json!(namespace));
            resp.insert("base_hash".to_string(), json!(base));
            resp.insert("base_hash_str".to_string(), json!(base.to_string()));
            resp.insert("text_bytes".to_string(), json!(text.len()));
            resp.insert("pool_root".to_string(), json!(pool_p));
            Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
        }
        "omc_memory_compact_bpe" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let age = args.get("age_threshold_secs").and_then(Json::as_i64).unwrap_or(86400);
            let store = MemoryStore::from_env();
            let (n, before, after) = store.compact_namespace_bpe(namespace, age)?;
            let ratio = if after > 0 { before as f64 / after as f64 } else { 0.0 };
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "compacted": n,
                "bytes_before": before,
                "bytes_after": after,
                "compression_ratio": ratio,
                "age_threshold_secs": age,
                "format": "OMCB",
            })).unwrap())
        }
        "omc_memory_compact_hbit" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let age = args.get("age_threshold_secs").and_then(Json::as_i64).unwrap_or(86400);
            let store = MemoryStore::from_env();
            let (n, before, after) = store.compact_namespace_hbit(namespace, age)?;
            let ratio = if after > 0 { before as f64 / after as f64 } else { 0.0 };
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "compacted": n,
                "bytes_before": before,
                "bytes_after": after,
                "compression_ratio": ratio,
                "age_threshold_secs": age,
                "format": "OMCH",
            })).unwrap())
        }
        "omc_memory_compact_substrate" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let age = args.get("age_threshold_secs").and_then(Json::as_i64).unwrap_or(86400);
            let store = MemoryStore::from_env();
            let (n, before, after) = store.compact_namespace_substrate(namespace, age)?;
            let ratio = if after > 0 { before as f64 / after as f64 } else { 0.0 };
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "compacted": n,
                "bytes_before": before,
                "bytes_after": after,
                "compression_ratio": ratio,
                "age_threshold_secs": age,
                "format": "OMCT",
            })).unwrap())
        }
        "omc_memory_compact" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            let age = args.get("age_threshold_secs").and_then(Json::as_i64).unwrap_or(86400);
            let store = MemoryStore::from_env();
            let (n, before, after) = store.compact_namespace(namespace, age)?;
            let ratio = if after > 0 { before as f64 / after as f64 } else { 0.0 };
            Ok(serde_json::to_string_pretty(&json!({
                "namespace": namespace,
                "compacted": n,
                "bytes_before": before,
                "bytes_after": after,
                "compression_ratio": ratio,
                "age_threshold_secs": age,
            })).unwrap())
        }
        "omc_memory_create_manifest" => {
            let namespace = args.get("namespace").and_then(Json::as_str).unwrap_or("default");
            // v0.12.1: accept entries as ints OR entries_str as decimal strings
            let mut leaves: Vec<i64> = Vec::new();
            if let Some(strs) = args.get("entries_str").and_then(Json::as_array) {
                for v in strs.iter() {
                    let s = v.as_str().ok_or_else(||
                        "omc_memory_create_manifest: 'entries_str' must be array of decimal strings".to_string())?;
                    leaves.push(s.parse::<i64>().map_err(|e|
                        format!("omc_memory_create_manifest: bad entry_str '{}': {}", s, e))?);
                }
            } else {
                let entries_v = args.get("entries").and_then(Json::as_array)
                    .ok_or_else(|| "omc_memory_create_manifest: missing 'entries' (i64 array) or 'entries_str' (decimal-string array)".to_string())?;
                for v in entries_v.iter() {
                    let h = v.as_i64()
                        .ok_or_else(|| "omc_memory_create_manifest: 'entries' must be i64 hashes (use 'entries_str' for hashes > 2^53)".to_string())?;
                    leaves.push(h);
                }
            }
            let store = MemoryStore::from_env();
            let manifest_hash = store.create_manifest(namespace, &leaves)?;
            let mut resp = serde_json::Map::new();
            resp.insert("manifest_hash".to_string(), json!(manifest_hash));
            resp.insert("manifest_hash_str".to_string(), json!(manifest_hash.to_string()));
            resp.insert("namespace".to_string(), json!(namespace));
            resp.insert("leaf_count".to_string(), json!(leaves.len()));
            Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap())
        }
        "omc_memory_recall_manifest" => {
            let target = read_hash_arg(args, "omc_memory_recall_manifest")?;
            let namespace = args.get("namespace").and_then(Json::as_str);
            let expand = args.get("expand").and_then(Json::as_bool).unwrap_or(false);
            let store = MemoryStore::from_env();
            match store.recall_manifest(namespace, target)? {
                None => {
                    let text = store.recall(namespace, target)?.unwrap_or_default();
                    let mut resp = hash_fields(target);
                    resp.insert("is_manifest".to_string(), json!(false));
                    resp.insert("text".to_string(), json!(text));
                    resp.insert("bytes".to_string(), json!(text.len()));
                    return Ok(serde_json::to_string_pretty(&Json::Object(resp)).unwrap());
                }
                Some(leaves) => {
                    let leaves_str: Vec<String> = leaves.iter().map(|h| h.to_string()).collect();
                    let mut out = json!({
                        "is_manifest": true,
                        "manifest_hash": target,
                        "manifest_hash_str": target.to_string(),
                        "entries": leaves.clone(),
                        "entries_str": leaves_str,
                        "leaf_count": leaves.len(),
                    });
                    if expand {
                        let mut expanded: Vec<Json> = Vec::with_capacity(leaves.len());
                        for h in &leaves {
                            let body = store.recall(None, *h)?;
                            expanded.push(json!({
                                "hash": h,
                                "found": body.is_some(),
                                "text": body.unwrap_or_default(),
                            }));
                        }
                        out["expanded"] = json!(expanded);
                    }
                    Ok(serde_json::to_string_pretty(&out).unwrap())
                }
            }
        }
        "omc_decompress" => {
            let paths = parse_paths_arg(args, "omc_decompress")?;
            // Accept either a bare canonical_hash or a codec dict that
            // contains content_hash. This is the generalization of
            // omc_fetch_by_hash that the LLM can use whether it kept
            // the full codec payload or distilled to just the hash.
            let target = if let Some(h) = args.get("canonical_hash").and_then(Json::as_i64) {
                h
            } else if let Some(codec) = args.get("codec") {
                codec.get("content_hash").and_then(Json::as_i64)
                    .ok_or_else(|| "omc_decompress: codec dict missing 'content_hash'".to_string())?
            } else {
                return Err("omc_decompress: requires either 'canonical_hash' or 'codec'".to_string());
            };
            let corpus = build_corpus(&paths)?;
            match corpus.entries.iter().find(|e| e.canonical_hash == target) {
                Some(entry) => Ok(serde_json::to_string_pretty(&json!({
                    "found": true,
                    "canonical_hash": entry.canonical_hash,
                    "fn_name": entry.fn_name,
                    "file": entry.file,
                    "source": entry.source,
                })).unwrap()),
                None => Ok(serde_json::to_string_pretty(&json!({
                    "found": false,
                    "canonical_hash": target,
                    "searched_paths": paths,
                    "corpus_size": corpus.len(),
                })).unwrap()),
            }
        }
        _ => Err(format!("Unknown tool: {}", name)),
    }
}

/// Compact one Suggestion into the requested response format.
///
/// - `hash` (~50 bytes): identity only. The LLM uses it to remember a
///   match it might fetch later via omc_fetch_by_hash.
/// - `signature` (~100 bytes): adds the fn signature line so the LLM
///   knows the call shape without paying for the body.
/// - `codec` (~150-300 bytes): hash + sampled-token thumbnail. Carries
///   structural information about the fn (matmul-heavy vs dict-traversal
///   etc.) without paying for the body. Use when the LLM wants to
///   distinguish between similarly-named candidates by shape.
/// - `full`: everything including the body. Use when the LLM intends
///   to read or adapt the implementation.
///
/// `prefix_match_len` and `substrate_distance` are included at every
/// level — they're the ranking explanation and cost essentially nothing.
fn project_suggestion(s: &omnimcode_core::predict::Suggestion, format: &str) -> Json {
    match format {
        "full" => json!({
            "fn_name": s.fn_name,
            "source": s.source,
            "file": s.file,
            "canonical_hash": s.canonical_hash,
            "attractor": s.attractor,
            "prefix_match_len": s.prefix_match_len,
            "substrate_distance": s.substrate_distance,
            "query_attractor": s.query_attractor,
        }),
        "signature" => json!({
            "fn_name": s.fn_name,
            "signature": extract_signature(&s.source),
            "file": s.file,
            "canonical_hash": s.canonical_hash,
            "prefix_match_len": s.prefix_match_len,
            "substrate_distance": s.substrate_distance,
        }),
        "codec" => {
            let codec = encode_codec_payload(&s.source, 3);
            json!({
                "fn_name": s.fn_name,
                "file": s.file,
                "canonical_hash": s.canonical_hash,
                "prefix_match_len": s.prefix_match_len,
                "substrate_distance": s.substrate_distance,
                "codec": codec,
            })
        }
        // "hash" is the default and the most compressed form.
        _ => json!({
            "fn_name": s.fn_name,
            "file": s.file,
            "canonical_hash": s.canonical_hash,
            "prefix_match_len": s.prefix_match_len,
            "substrate_distance": s.substrate_distance,
        }),
    }
}

/// Canonicalize → tokenize → sample-every-Nth → produce the codec
/// payload dict the v0.0.5 substrate-codec spec defines. Mirrors the
/// omc_codec_encode builtin but builds a JSON value directly (no
/// Value/Interpreter round-trip). every_n=1 means "keep all tokens"
/// (no compression, useful for lossless transport); the practical
/// default is 3 (matches the builtin's default), giving ~3× token-
/// count reduction.
///
/// The content_hash is alpha-rename invariant — the LLM can recover
/// the original source via omc_fetch_by_hash or omc_decompress
/// against any corpus that contains a fn with the same canonical form.
fn encode_codec_payload(source: &str, every_n: usize) -> Json {
    let every_n = every_n.max(1);
    let canon = canonical::canonicalize(source).unwrap_or_else(|_| source.to_string());
    let tokens = tokenizer::encode(&canon);
    // Cap the sampled-token thumbnail to MAX_THUMBNAIL_TOKENS so codec
    // format stays bounded regardless of fn size. The hash is the
    // identity (alpha-rename invariant, full lossless recovery via
    // omc_decompress); the thumbnail is just enough structural signal
    // to disambiguate candidates without paying for full source.
    const MAX_THUMBNAIL_TOKENS: usize = 16;
    // Effective stride: at least every_n, scaled up if needed to keep
    // the sample below the cap. Preserves the every_n contract for
    // small fns; uniformly subsamples for large ones.
    let effective_n = (tokens.len() / MAX_THUMBNAIL_TOKENS.max(1)).max(every_n);
    let sampled: Vec<i64> = tokens.iter().enumerate()
        .filter(|(i, _)| i % effective_n == 0)
        .take(MAX_THUMBNAIL_TOKENS)
        .map(|(_, t)| *t)
        .collect();
    // Use tokenizer::code_hash so content_hash matches predict's
    // canonical_hash. Both hash the TOKEN-PACKED bytes (not the raw
    // canonical-source bytes) — without this alignment, a suggestion's
    // canonical_hash wouldn't equal the codec's content_hash, and the
    // LLM couldn't use them interchangeably with omc_fetch_by_hash /
    // omc_decompress.
    let (attractor, hash, dist) = tokenizer::code_hash(&canon);
    let ratio = if !sampled.is_empty() {
        source.len() as f64 / sampled.len() as f64
    } else { 0.0 };
    json!({
        "sampled_tokens": sampled,
        "content_hash": hash,
        "attractor": attractor,
        "dist": dist,
        "original_tok_count": tokens.len(),
        "source_bytes": source.len(),
        "every_n": every_n,
        "compression_ratio": ratio,
    })
}


/// Extract the function signature line from a fn body's source. The
/// signature is everything from `fn` through the closing paren of the
/// argument list, plus any `-> ReturnType` annotation. Stops at the
/// opening `{` of the body.
///
/// Robust to multi-line signatures (joins lines, collapses whitespace).
fn extract_signature(source: &str) -> String {
    // Join everything before the first `{` then collapse whitespace.
    let head = source.split_once('{').map(|(h, _)| h).unwrap_or(source);
    let cleaned: String = head.split_whitespace().collect::<Vec<_>>().join(" ");
    cleaned.trim().to_string()
}

/// Extract a `paths` array argument from a tool's JSON args. Used by
/// both omc_predict and omc_corpus_size — same shape, same validation.
fn parse_paths_arg(args: &Json, tool: &str) -> Result<Vec<String>, String> {
    let paths_val = args.get("paths")
        .ok_or_else(|| format!("{}: missing 'paths' arg", tool))?;
    let arr = paths_val.as_array()
        .ok_or_else(|| format!("{}: 'paths' must be an array of strings", tool))?;
    arr.iter()
        .map(|v| v.as_str()
            .ok_or_else(|| format!("{}: every 'paths' entry must be a string", tool))
            .map(|s| s.to_string()))
        .collect()
}

/// Build a CodeCorpus by reading + ingesting every file in `paths`.
///
/// Each entry can be a file OR a directory. Directories are walked
/// recursively for `*.omc` files. This is what makes cross-corpus
/// blending cheap — an LLM can pass `["examples/lib"]` and ingest
/// the entire lib tree without enumerating files itself.
///
/// I/O errors surface as MCP-style strings so the client sees a
/// clean `isError: true` text instead of a panic.
fn build_corpus(paths: &[String]) -> Result<CodeCorpus, String> {
    let mut corpus = CodeCorpus::new();
    for path in paths {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            // Walk the directory recursively for .omc files.
            walk_omc_files(p, &mut corpus)?;
        } else {
            let src = std::fs::read_to_string(path)
                .map_err(|e| format!("omc_predict: read {}: {}", path, e))?;
            corpus.ingest_file(path, &src);
        }
    }
    Ok(corpus)
}

/// Recursively ingest every `*.omc` file under `dir` into `corpus`.
/// Stable iteration order (sorted by filename) so the same paths
/// argument produces the same corpus across runs — predictability is
/// part of the substrate contract.
fn walk_omc_files(dir: &std::path::Path, corpus: &mut CodeCorpus) -> Result<(), String> {
    let read_dir = std::fs::read_dir(dir)
        .map_err(|e| format!("read_dir {}: {}", dir.display(), e))?;
    let mut entries: Vec<std::path::PathBuf> = read_dir
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            walk_omc_files(&entry, corpus)?;
        } else if entry.extension().and_then(|s| s.to_str()) == Some("omc") {
            let path_str = entry.to_string_lossy().to_string();
            if let Ok(src) = std::fs::read_to_string(&entry) {
                corpus.ingest_file(&path_str, &src);
            }
            // Per-file read errors are silently skipped — a single
            // unreadable file shouldn't break a directory ingest.
        }
    }
    Ok(())
}

/// Evaluate an OMC program. Errors come back as structured strings
/// (the MCP client sees isError=true alongside the text). Each
/// tools/call uses a fresh interpreter to avoid state bleed.
///
/// Returns the display string of the final statement's value, or
/// "null" if the program ends on a non-expression. This matches the
/// REPL convention LLMs expect when iterating quickly.
fn eval_program(_interp: &mut Interpreter, code: &str) -> Result<String, String> {
    let mut parser = Parser::new(code);
    let stmts = parser.parse()
        .map_err(|e| format!("parse error: {}", e))?;
    // Fresh Interpreter per call: keeps the MCP server stateless,
    // which is what most LLM clients expect. Tooling can layer
    // session state on top if needed.
    let mut fresh = Interpreter::new();
    fresh.execute(stmts).map_err(|e| format!("runtime error: {}", e))?;

    // Collect stdout captured by print/println.
    let printed = fresh.take_output_lines();

    // Prefer the last top-level expression value, then fall back to
    // any function-level return value (e.g. `return 42;` at top level).
    let v = fresh.take_last_expression_value()
        .or_else(|| fresh.take_return_value());
    let expr_result = match v {
        Some(v) => display_value(&v),
        None    => "null".to_string(),
    };

    // If any print() calls were made, return them first, then the
    // expression result (unless it's "null", which is uninformative).
    if printed.is_empty() {
        Ok(expr_result)
    } else {
        let mut out = printed.join("\n");
        if expr_result != "null" {
            out.push('\n');
            out.push_str(&expr_result);
        }
        Ok(out)
    }
}

fn display_value(v: &Value) -> String {
    // Compact, LLM-friendly rendering. HInt shows value + substrate
    // metadata so the LLM sees the resonance/HIM that distinguishes
    // OMC from numpy. Arrays unwrap their RefCell wrapper visually
    // — the inner Debug format leaks Rust internals that aren't useful.
    match v {
        Value::HInt(h) => format!(
            "HInt {{ value: {}, resonance: {:.3}, him: {:.3} }}",
            h.value, h.resonance, h.him_score
        ),
        Value::HFloat(f) => format!("{}", f),
        Value::String(s) => format!("\"{}\"", s),
        Value::Bool(b) => format!("{}", b),
        Value::Null => "null".to_string(),
        Value::Array(arr) => {
            let items = arr.items.borrow();
            let parts: Vec<String> = items.iter().map(display_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Dict(d) => {
            let d = d.borrow();
            let parts: Vec<String> = d.iter()
                .map(|(k, v)| format!("\"{}\": {}", k, display_value(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Function { name, .. } => format!("<fn {}>", name),
        _ => format!("{:?}", v),
    }
}
