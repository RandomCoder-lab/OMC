//! omc-grep — code archaeology by canonical hash + substrate distance.
//!
//! Walks a directory of OMC files, extracts every top-level fn,
//! canonicalizes each one (whitespace-stripped, comments removed,
//! parameters alpha-renamed to a canonical order), computes the
//! canonical hash and the nearest substrate (Fibonacci) attractor.
//!
//! Reports:
//!   * EXACT clusters: groups of 2+ fns with identical canonical hash
//!     — these are true duplicates regardless of rename/whitespace
//!   * NEAR clusters: fns within `--near` substrate-distance of each
//!     other but not exact matches
//!
//! Usage:
//!   omc-grep [--near N] [--min-cluster K] [--show-all] DIR
//!
//! The alpha-rename invariance is what nothing else does — text grep,
//! ast-grep, tree-sitter queries all miss `fn foo(x)` ≡ `fn foo(y)`.
//! OMC's canonical form normalizes the parameter binding, so they
//! become the same hash.
//!
//! Phase 1 (this file): OMC files only. Phase 2 will add Python via
//! the stdlib `ast` module.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use omnimcode_core::canonical;
use omnimcode_core::interpreter::extract_top_level_fns;
use omnimcode_core::phi_pi_fib;
use omnimcode_core::tokenizer;

/// A single fn occurrence: where it was found + its canonical form.
#[derive(Clone)]
struct FnEntry {
    file: PathBuf,
    line: u32,
    name: String,
    source: String,
    canonical: String,
    canon_hash: i64,
    attractor: i64,
    attr_dist: i64,
}

fn extract_fn_name(src: &str) -> String {
    // src starts with "fn NAME(...) { ... }". Pull NAME.
    let after_fn = src.strip_prefix("fn ").unwrap_or(src).trim_start();
    let end = after_fn
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(after_fn.len());
    after_fn[..end].to_string()
}

fn find_line_of(haystack: &str, needle: &str) -> u32 {
    if let Some(idx) = haystack.find(needle) {
        haystack[..idx].chars().filter(|&c| c == '\n').count() as u32 + 1
    } else {
        0
    }
}

fn walk_omc_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for ent in rd.flatten() {
            let p = ent.path();
            // Skip common build/dep directories.
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if matches!(name, "target" | "node_modules" | ".git" | "__pycache__" | "omc_modules") {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("omc") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Strip "fn NAME(...)" so the hash reflects only the body. Lets us
/// find alpha-equivalent fns under DIFFERENT NAMES (e.g. dispatch
/// helpers that got copied and renamed but never reworked).
fn body_only(canonical: &str) -> String {
    if let Some(open) = canonical.find('{') {
        // Find matching close brace, return everything between (and the braces).
        let bytes = canonical.as_bytes();
        let mut depth = 0i32;
        let mut k = open;
        while k < bytes.len() {
            match bytes[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return canonical[open..=k].to_string();
                    }
                }
                _ => {}
            }
            k += 1;
        }
    }
    canonical.to_string()
}

fn ingest_file(path: &Path, body_only_mode: bool) -> Vec<FnEntry> {
    let Ok(src) = std::fs::read_to_string(path) else { return Vec::new() };
    let mut out = Vec::new();
    for fn_src in extract_top_level_fns(&src) {
        let canonical = canonical::canonicalize(&fn_src)
            .unwrap_or_else(|_| fn_src.clone());
        let hash_input = if body_only_mode {
            body_only(&canonical)
        } else {
            canonical.clone()
        };
        let canon_hash = tokenizer::fnv1a_64(hash_input.as_bytes());
        let (attractor, attr_dist) =
            phi_pi_fib::nearest_attractor_with_dist(canon_hash);
        let line = find_line_of(&src, &fn_src);
        let name = extract_fn_name(&fn_src);
        out.push(FnEntry {
            file: path.to_path_buf(),
            line,
            name,
            source: fn_src,
            canonical,
            canon_hash,
            attractor,
            attr_dist,
        });
    }
    out
}

fn print_exact_clusters(entries: &[FnEntry], min_cluster: usize) -> usize {
    let mut by_hash: BTreeMap<i64, Vec<&FnEntry>> = BTreeMap::new();
    for e in entries {
        by_hash.entry(e.canon_hash).or_default().push(e);
    }
    let mut clusters: Vec<_> = by_hash
        .into_iter()
        .filter(|(_, v)| v.len() >= min_cluster)
        .collect();
    // Sort by cluster size descending, then by hash for stability.
    clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));

    println!(
        "\n=== EXACT clusters ({} cluster{}, threshold ≥{}) ===",
        clusters.len(),
        if clusters.len() == 1 { "" } else { "s" },
        min_cluster
    );
    if clusters.is_empty() {
        println!("  (none — every fn in this corpus has a unique canonical hash)");
    }
    let mut total_dupes = 0;
    for (hash, members) in &clusters {
        total_dupes += members.len() - 1; // first occurrence is the "original"
        // Group by distinct name to see if it's rename-duplication or pure copies.
        let mut names: Vec<&str> = members.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        names.dedup();
        let kind = if names.len() == 1 {
            format!("copies of `{}`", names[0])
        } else {
            format!("alpha-equivalent across {} names: {}", names.len(), names.join(", "))
        };
        println!(
            "\n  hash={:016x}  attr={}  dist={}  members={}  ({})",
            *hash as u64,
            members[0].attractor,
            members[0].attr_dist,
            members.len(),
            kind
        );
        for m in members {
            println!("    {}:{}  fn {}", m.file.display(), m.line, m.name);
        }
    }
    total_dupes
}

fn print_near_clusters(entries: &[FnEntry], near_dist: i64, min_cluster: usize) {
    if near_dist <= 0 {
        return;
    }
    // Bucket by attractor — fns sharing the nearest Fibonacci land in
    // the same bucket. Inside a bucket, any pair within `near_dist` of
    // each other (by raw hash) is a near-cluster.
    let mut by_attr: BTreeMap<i64, Vec<&FnEntry>> = BTreeMap::new();
    let mut by_hash: BTreeMap<i64, Vec<&FnEntry>> = BTreeMap::new();
    for e in entries {
        by_attr.entry(e.attractor).or_default().push(e);
        by_hash.entry(e.canon_hash).or_default().push(e);
    }
    let mut printed = 0usize;
    println!(
        "\n=== NEAR clusters (substrate distance ≤ {}, excluding exact dupes) ===",
        near_dist
    );
    let mut shown_pairs: std::collections::BTreeSet<(i64, i64)> = std::collections::BTreeSet::new();
    for (_attr, bucket) in &by_attr {
        // For each pair in the bucket, if hashes differ and |h1-h2| <= near_dist, print.
        for i in 0..bucket.len() {
            for j in (i + 1)..bucket.len() {
                let a = bucket[i];
                let b = bucket[j];
                if a.canon_hash == b.canon_hash {
                    continue; // exact dupe — already in EXACT section
                }
                let d = (a.canon_hash - b.canon_hash).abs();
                if d > near_dist {
                    continue;
                }
                let key = if a.canon_hash < b.canon_hash {
                    (a.canon_hash, b.canon_hash)
                } else {
                    (b.canon_hash, a.canon_hash)
                };
                if !shown_pairs.insert(key) {
                    continue;
                }
                printed += 1;
                println!(
                    "\n  pair-distance={}  attr={}  ",
                    d, a.attractor
                );
                println!(
                    "    {}:{}  fn {}   [hash={:016x}]",
                    a.file.display(),
                    a.line,
                    a.name,
                    a.canon_hash as u64
                );
                println!(
                    "    {}:{}  fn {}   [hash={:016x}]",
                    b.file.display(),
                    b.line,
                    b.name,
                    b.canon_hash as u64
                );
            }
        }
    }
    if printed == 0 {
        println!("  (none within distance {})", near_dist);
    }
    let _ = (by_hash, min_cluster); // reserved for future "near + multi-member" reporting
}

fn print_summary(entries: &[FnEntry], files: &[PathBuf], total_dupes: usize) {
    let total_fns = entries.len();
    let unique_hashes: std::collections::BTreeSet<i64> =
        entries.iter().map(|e| e.canon_hash).collect();
    let dup_pct = if total_fns > 0 {
        100.0 * total_dupes as f64 / total_fns as f64
    } else {
        0.0
    };
    println!("\n=== Summary ===");
    println!("  files scanned     : {}", files.len());
    println!("  fns extracted     : {}", total_fns);
    println!("  unique canonical  : {}", unique_hashes.len());
    println!(
        "  duplicate fns     : {} ({:.1}% redundant)",
        total_dupes, dup_pct
    );
}

fn print_usage() {
    eprintln!("omc-grep — canonical-hash code archaeology");
    eprintln!();
    eprintln!("Usage: omc-grep [OPTIONS] DIR");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --body-only      hash the fn body only (drop name + signature);");
    eprintln!("                   finds alpha-equivalent fns under DIFFERENT NAMES");
    eprintln!("  --near N         also report fn pairs within substrate distance N");
    eprintln!("                   (sharing same Fibonacci attractor) [default: 0 = off]");
    eprintln!("  --min-cluster K  only report exact clusters with K+ members [default: 2]");
    eprintln!("  --show-all       include single-fn entries in the output");
    eprintln!("  -h, --help       this help");
    eprintln!();
    eprintln!("Currently handles: .omc files. Walks DIR recursively.");
    eprintln!("Skips: target/, node_modules/, .git/, __pycache__/, omc_modules/");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut near_dist: i64 = 0;
    let mut min_cluster: usize = 2;
    let mut show_all = false;
    let mut body_only_mode = false;
    let mut dir: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--body-only" => body_only_mode = true,
            "--near" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--near requires a value");
                    return ExitCode::from(2);
                }
                near_dist = args[i].parse().unwrap_or(0);
            }
            "--min-cluster" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--min-cluster requires a value");
                    return ExitCode::from(2);
                }
                min_cluster = args[i].parse().unwrap_or(2);
            }
            "--show-all" => show_all = true,
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            s if s.starts_with("--") => {
                eprintln!("unknown flag: {}", s);
                print_usage();
                return ExitCode::from(2);
            }
            s => {
                if dir.is_some() {
                    eprintln!("multiple directories not supported (yet)");
                    return ExitCode::from(2);
                }
                dir = Some(s.to_string());
            }
        }
        i += 1;
    }
    let dir = match dir {
        Some(d) => d,
        None => {
            print_usage();
            return ExitCode::from(2);
        }
    };
    let root = Path::new(&dir);
    if !root.is_dir() {
        eprintln!("not a directory: {}", dir);
        return ExitCode::from(1);
    }
    let files = walk_omc_files(root);
    let mode = if body_only_mode {
        "body-only (alpha-equivalent across DIFFERENT NAMES)"
    } else {
        "full-canonical (same name + same body)"
    };
    println!(
        "omc-grep: scanning {} (.omc files: {})  mode: {}",
        dir,
        files.len(),
        mode
    );
    let mut entries = Vec::new();
    for f in &files {
        entries.extend(ingest_file(f, body_only_mode));
    }
    if entries.is_empty() {
        println!("\n  no top-level fns found");
        return ExitCode::SUCCESS;
    }
    let total_dupes = print_exact_clusters(&entries, min_cluster);
    if near_dist > 0 {
        print_near_clusters(&entries, near_dist, min_cluster);
    }
    let _ = show_all; // reserved
    print_summary(&entries, &files, total_dupes);
    ExitCode::SUCCESS
}
