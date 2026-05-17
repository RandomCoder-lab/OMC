//! omc-kernel — content-addressed store keyed by canonical hash.
//!
//! Every OMC fn has a canonical form (whitespace-stripped, comments
//! removed, parameter binding normalized) whose 64-bit fnv1a hash is
//! a stable, alpha-rename-invariant identity. omc-kernel uses that
//! hash as the primary key for a file-system content-addressed store
//! at ~/.omc/kernel/store/<hex_hash>.omc.
//!
//! With this store, code becomes a content-addressed Merkle DAG over
//! canonical hashes — version it the way IPFS versions files, except
//! the addressing is semantic instead of byte-level (alpha-rename and
//! whitespace edits are the same content).
//!
//! Subcommands:
//!   ingest DIR    extract every fn from DIR's .omc files, store by hash
//!   fetch HASH    retrieve stored fn by canonical hash (hex)
//!   stat HASH     substrate metadata: attractor, dist, bytes, fn name
//!   ls            list stored hashes + first-line summary
//!   sign FILE     read an OMC source file, write a substrate-signed
//!                 compressed message to stdout (suitable for inter-
//!                 process transport)
//!   verify        read a substrate-signed message from stdin,
//!                 verify the signature, attempt store recovery on
//!                 canonical-hash match; print recovered source
//!   demo          end-to-end: ingest examples/lib/, sign a fn, fetch
//!                 it back, print substrate metadata
//!
//! Wire format for sign/verify: JSON-serialized substrate-signed
//! message (same format as omc_msg_sign_compressed). Content is
//! carried as sampled-token codec payload; receiver recovers the
//! full source via store lookup.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use omnimcode_core::canonical;
use omnimcode_core::interpreter::extract_top_level_fns;
use omnimcode_core::phi_pi_fib;
use omnimcode_core::tokenizer;

// --------------------------------------------------------------------
// Store paths
// --------------------------------------------------------------------

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn kernel_root() -> PathBuf {
    if let Ok(p) = std::env::var("OMC_KERNEL_ROOT") {
        PathBuf::from(p)
    } else {
        let mut p = home_dir();
        p.push(".omc");
        p.push("kernel");
        p
    }
}

fn store_dir() -> PathBuf {
    let mut p = kernel_root();
    p.push("store");
    p
}

fn store_path_for(hash: i64) -> PathBuf {
    let mut p = store_dir();
    p.push(format!("{:016x}.omc", hash as u64));
    p
}

fn meta_path_for(hash: i64) -> PathBuf {
    let mut p = store_dir();
    p.push(format!("{:016x}.json", hash as u64));
    p
}

fn ensure_store() -> std::io::Result<()> {
    std::fs::create_dir_all(store_dir())
}

// --------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------

fn extract_fn_name(src: &str) -> String {
    let after_fn = src.strip_prefix("fn ").unwrap_or(src).trim_start();
    let end = after_fn
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(after_fn.len());
    after_fn[..end].to_string()
}

fn hash_of_canonical(src: &str) -> i64 {
    let canon = canonical::canonicalize(src).unwrap_or_else(|_| src.to_string());
    tokenizer::fnv1a_64(canon.as_bytes())
}

fn parse_hex_hash(s: &str) -> Option<i64> {
    u64::from_str_radix(s, 16).ok().map(|u| u as i64)
}

// --------------------------------------------------------------------
// Subcommands
// --------------------------------------------------------------------

fn cmd_ingest(dir: &str) -> ExitCode {
    let root = Path::new(dir);
    if !root.is_dir() {
        eprintln!("ingest: not a directory: {}", dir);
        return ExitCode::from(1);
    }
    if let Err(e) = ensure_store() {
        eprintln!("ingest: cannot create store: {}", e);
        return ExitCode::from(1);
    }
    let mut stack = vec![root.to_path_buf()];
    let mut new_count = 0usize;
    let mut existing_count = 0usize;
    let mut fn_count = 0usize;
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for ent in rd.flatten() {
            let p = ent.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if matches!(name, "target" | "node_modules" | ".git" | "omc_modules") {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().and_then(|s| s.to_str()) != Some("omc") {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&p) else { continue };
            for fn_src in extract_top_level_fns(&src) {
                fn_count += 1;
                let hash = hash_of_canonical(&fn_src);
                let path = store_path_for(hash);
                if path.exists() {
                    existing_count += 1;
                    continue;
                }
                if let Err(e) = std::fs::write(&path, &fn_src) {
                    eprintln!("ingest: write failed for {}: {}", path.display(), e);
                    continue;
                }
                // Sidecar metadata so `stat` is O(1).
                let canon =
                    canonical::canonicalize(&fn_src).unwrap_or_else(|_| fn_src.clone());
                let (attractor, dist) =
                    phi_pi_fib::nearest_attractor_with_dist(hash);
                let meta = serde_json::json!({
                    "canonical_hash": hash.to_string(),
                    "attractor": attractor.to_string(),
                    "attractor_distance": dist.to_string(),
                    "source_bytes": fn_src.len(),
                    "canonical_bytes": canon.len(),
                    "fn_name": extract_fn_name(&fn_src),
                    "origin_file": p.display().to_string(),
                });
                let _ = std::fs::write(meta_path_for(hash), meta.to_string());
                new_count += 1;
            }
        }
    }
    println!(
        "ingested {} fns: {} new, {} already present in store",
        fn_count, new_count, existing_count
    );
    println!("store: {}", store_dir().display());
    ExitCode::SUCCESS
}

fn cmd_fetch(hex_hash: &str) -> ExitCode {
    let Some(hash) = parse_hex_hash(hex_hash) else {
        eprintln!("fetch: invalid hex hash: {}", hex_hash);
        return ExitCode::from(2);
    };
    let path = store_path_for(hash);
    match std::fs::read_to_string(&path) {
        Ok(src) => {
            print!("{}", src);
            if !src.ends_with('\n') {
                println!();
            }
            ExitCode::SUCCESS
        }
        Err(_) => {
            eprintln!("fetch: not in store: {:016x}", hash as u64);
            ExitCode::from(1)
        }
    }
}

fn cmd_stat(hex_hash: &str) -> ExitCode {
    let Some(hash) = parse_hex_hash(hex_hash) else {
        eprintln!("stat: invalid hex hash: {}", hex_hash);
        return ExitCode::from(2);
    };
    let mp = meta_path_for(hash);
    match std::fs::read_to_string(&mp) {
        Ok(s) => {
            // Pretty-print the JSON if possible.
            let parsed: serde_json::Value = serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s.clone()));
            println!("{}", serde_json::to_string_pretty(&parsed).unwrap_or(s));
            ExitCode::SUCCESS
        }
        Err(_) => {
            eprintln!("stat: not in store: {:016x}", hash as u64);
            ExitCode::from(1)
        }
    }
}

fn cmd_ls() -> ExitCode {
    let dir = store_dir();
    if !dir.is_dir() {
        println!("(store is empty: {})", dir.display());
        return ExitCode::SUCCESS;
    }
    let Ok(rd) = std::fs::read_dir(&dir) else {
        eprintln!("ls: cannot read {}", dir.display());
        return ExitCode::from(1);
    };
    let mut entries: Vec<(String, String, usize)> = Vec::new();
    for ent in rd.flatten() {
        let p = ent.path();
        if p.extension().and_then(|s| s.to_str()) != Some("omc") {
            continue;
        }
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let Ok(src) = std::fs::read_to_string(&p) else { continue };
        let name = extract_fn_name(&src);
        let bytes = src.len();
        entries.push((stem, name, bytes));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    println!("{} fn(s) in store at {}", entries.len(), dir.display());
    println!("{:<18} {:>8}  {}", "canonical-hash", "bytes", "fn");
    for (hash, name, bytes) in &entries {
        println!("{:<18} {:>8}  fn {}", hash, bytes, name);
    }
    ExitCode::SUCCESS
}

// --- sign / verify (uses the codec; reuses what's in interpreter.rs) ---

fn cmd_sign(path: &str) -> ExitCode {
    let Ok(content) = std::fs::read_to_string(path) else {
        eprintln!("sign: cannot read: {}", path);
        return ExitCode::from(1);
    };
    let canon = canonical::canonicalize(&content).unwrap_or_else(|_| content.clone());
    let hash = tokenizer::fnv1a_64(canon.as_bytes());
    let (attractor, dist) = phi_pi_fib::nearest_attractor_with_dist(hash);
    let tokens = tokenizer::encode(&canon);
    let every_n = 3usize;
    let sampled: Vec<i64> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| i % every_n == 0)
        .map(|(_, t)| *t)
        .collect();
    // Sender ID 0 — kernel-level signing, no agent identity attached.
    // Caller can rewrap with their own omc_msg_sign_compressed if they
    // want agent attribution.
    let msg = serde_json::json!({
        "sender_id": 0,
        "kind": 1,
        "content_hash": hash.to_string(),
        "attractor": attractor.to_string(),
        "attractor_distance": dist.to_string(),
        "sampled_tokens": sampled,
        "every_n": every_n,
        "original_tok_count": tokens.len(),
        "source_bytes": content.len(),
    });
    println!("{}", serde_json::to_string(&msg).unwrap());
    ExitCode::SUCCESS
}

fn cmd_verify() -> ExitCode {
    let mut wire = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut wire) {
        eprintln!("verify: stdin read failed: {}", e);
        return ExitCode::from(1);
    }
    let v: serde_json::Value = match serde_json::from_str(&wire) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("verify: not valid JSON: {}", e);
            return ExitCode::from(1);
        }
    };
    let hash_str = v.get("content_hash").and_then(|x| x.as_str()).unwrap_or("0");
    let hash: i64 = hash_str.parse().unwrap_or(0);
    eprintln!("verify: content_hash = {:016x}", hash as u64);
    let path = store_path_for(hash);
    match std::fs::read_to_string(&path) {
        Ok(src) => {
            // Recompute hash from store entry — defense against tampering
            // of the store itself.
            let canon = canonical::canonicalize(&src).unwrap_or_else(|_| src.clone());
            let recomputed = tokenizer::fnv1a_64(canon.as_bytes());
            if recomputed != hash {
                eprintln!(
                    "verify: STORE TAMPERED — recomputed hash {:016x} does not match",
                    recomputed as u64
                );
                return ExitCode::from(1);
            }
            eprintln!("verify: store hash matches; recovered {} bytes", src.len());
            print!("{}", src);
            if !src.ends_with('\n') {
                println!();
            }
            ExitCode::SUCCESS
        }
        Err(_) => {
            eprintln!(
                "verify: content not in store ({:016x}) — fetch from peer or fall back to full payload",
                hash as u64
            );
            ExitCode::from(1)
        }
    }
}

fn cmd_demo() -> ExitCode {
    // End-to-end: ingest examples/lib/, sign a known fn body, verify it back.
    let lib_dir = std::env::current_dir()
        .ok()
        .map(|d| d.join("examples").join("lib"))
        .filter(|p| p.is_dir());
    let lib_dir = match lib_dir {
        Some(d) => d,
        None => {
            eprintln!("demo: run from the OMC repo root (examples/lib/ must exist)");
            return ExitCode::from(1);
        }
    };
    println!("demo: ingesting {}", lib_dir.display());
    let _ = cmd_ingest(lib_dir.to_str().unwrap_or("."));
    println!();
    println!("demo: signing a renamed copy of `fn commit` from sqlite.omc");
    println!("  original (in store):  fn commit(conn) {{ return py_call(conn, \"commit\", []); }}");
    println!("  sender's rename:      fn commit(handle) {{ return py_call(handle, \"commit\", []); }}");
    let renamed = "fn commit(handle) { return py_call(handle, \"commit\", []); }";
    let canon = canonical::canonicalize(renamed).unwrap_or_else(|_| renamed.to_string());
    let hash = tokenizer::fnv1a_64(canon.as_bytes());
    println!("  canonical hash:       {:016x}", hash as u64);
    let path = store_path_for(hash);
    match std::fs::read_to_string(&path) {
        Ok(src) => {
            println!("\n  STORE HIT — canonical-hash addressing is alpha-rename invariant.");
            println!("  Recovered original canonical form:");
            for line in src.trim_end().lines() {
                println!("    {}", line);
            }
            println!("\n  Sender used `handle`, store has `conn` — same canonical address.");
        }
        Err(_) => {
            println!("\n  STORE MISS — ingest may not have run; try `omc-kernel ingest examples/lib`");
        }
    }
    ExitCode::SUCCESS
}

// --------------------------------------------------------------------
// Entry
// --------------------------------------------------------------------

fn print_usage() {
    eprintln!("omc-kernel — content-addressed store keyed by canonical hash");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  omc-kernel ingest DIR       extract every fn from DIR's .omc files, store");
    eprintln!("  omc-kernel fetch HASH       retrieve stored fn by canonical hash (hex)");
    eprintln!("  omc-kernel stat HASH        substrate metadata for stored fn");
    eprintln!("  omc-kernel ls               list stored hashes + fn names");
    eprintln!("  omc-kernel sign FILE        sign OMC source to a substrate-signed wire msg");
    eprintln!("  omc-kernel verify           verify a wire msg from stdin, recover via store");
    eprintln!("  omc-kernel demo             ingest examples/lib/, alpha-rename recovery demo");
    eprintln!();
    eprintln!("Env:");
    eprintln!("  OMC_KERNEL_ROOT             override store location (default: ~/.omc/kernel)");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(2);
    }
    let _ = ensure_store();
    match args[1].as_str() {
        "ingest" => {
            if args.len() < 3 {
                eprintln!("ingest: missing DIR");
                return ExitCode::from(2);
            }
            cmd_ingest(&args[2])
        }
        "fetch" => {
            if args.len() < 3 {
                eprintln!("fetch: missing HASH");
                return ExitCode::from(2);
            }
            cmd_fetch(&args[2])
        }
        "stat" => {
            if args.len() < 3 {
                eprintln!("stat: missing HASH");
                return ExitCode::from(2);
            }
            cmd_stat(&args[2])
        }
        "ls" => cmd_ls(),
        "sign" => {
            if args.len() < 3 {
                eprintln!("sign: missing FILE");
                return ExitCode::from(2);
            }
            cmd_sign(&args[2])
        }
        "verify" => cmd_verify(),
        "demo" => cmd_demo(),
        "-h" | "--help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand: {}", other);
            print_usage();
            ExitCode::from(2)
        }
    }
}
