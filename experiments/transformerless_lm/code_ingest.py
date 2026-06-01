#!/usr/bin/env python3
"""code_ingest.py — computing & code into the knowledge web: source code across languages + hardware,
plus computing standards (RFCs). Complements arXiv cs.* (research/PhD-level) and Wikipedia (CS concepts).

  --repos : sample real source files from major open-source repos across languages + systems + hardware
            (CPython, Rust, Go, Linux, LLVM, Postgres, PyTorch, RISC-V…). GitHub trees API (1-2 calls/repo,
            within the 60/hr unauth budget) + raw.githubusercontent (CDN, not API-rate-limited).
  --rfc N : fetch N foundational RFCs (plaintext) — networking/protocols/computing standards, basics->deep.
Field label = language (python/rust/c/…) or "computing" (RFCs). Files -> library/, folded by the disk grower.
Agnostic: real public code/standards as data. Polite + resumable.
"""
import sys, re, time, json, hashlib, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
UA = "OMC-knowledge-web/1.0 (research; local)"
LANG = {"py": "python", "rs": "rust", "go": "go", "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp",
        "js": "javascript", "ts": "typescript", "java": "java", "rb": "ruby", "v": "verilog", "scala": "scala",
        "swift": "swift", "kt": "kotlin", "kts": "kotlin", "hs": "haskell", "jl": "julia", "r": "r",
        "php": "php", "cs": "csharp", "lua": "lua", "ex": "elixir", "exs": "elixir", "clj": "clojure",
        "zig": "zig", "ml": "ocaml", "pl": "perl", "pm": "perl", "f90": "fortran", "f": "fortran",
        "erl": "erlang", "dart": "dart", "nim": "nim", "d": "d", "sh": "shell", "sv": "systemverilog",
        "vhd": "vhdl", "asm": "assembly", "s": "assembly", "m": "objc", "groovy": "groovy", "elm": "elm"}
# flagship repo per language: basics→systems→PhD-grade, languages + hardware (HDL) + OS/compilers
REPOS = [("python/cpython", "py"), ("rust-lang/rust", "rs"), ("golang/go", "go"),
         ("torvalds/linux", "c"), ("llvm/llvm-project", "cpp"), ("postgres/postgres", "c"),
         ("redis/redis", "c"), ("git/git", "c"), ("nodejs/node", "js"), ("facebook/react", "js"),
         ("numpy/numpy", "py"), ("pytorch/pytorch", "py"), ("microsoft/TypeScript", "ts"),
         ("openjdk/jdk", "java"), ("bitcoin/bitcoin", "cpp"), ("riscv/riscv-isa-manual", "c"),
         ("ruby/ruby", "rb"), ("sqlite/sqlite", "c"), ("ggerganov/llama.cpp", "cpp"),
         ("apple/swift", "swift"), ("JetBrains/kotlin", "kt"), ("JuliaLang/julia", "jl"),
         ("php/php-src", "c"), ("dotnet/runtime", "cs"), ("lua/lua", "c"), ("elixir-lang/elixir", "ex"),
         ("clojure/clojure", "java"), ("ziglang/zig", "zig"), ("ocaml/ocaml", "ml"), ("Perl/perl5", "pl"),
         ("erlang/otp", "erl"), ("dart-lang/sdk", "dart"), ("nim-lang/Nim", "nim"), ("ghc/ghc", "hs"),
         ("scala/scala3", "scala"), ("crystal-lang/crystal", "rb"), ("vlang/v", "v"),
         ("JuliaLang/julia", "jl"), ("haskell/cabal", "hs"), ("kubernetes/kubernetes", "go"),
         ("tensorflow/tensorflow", "cpp"), ("django/django", "py"), ("rails/rails", "rb"),
         ("vuejs/core", "ts"), ("denoland/deno", "rs"), ("chipsalliance/rocket-chip", "scala"),
         ("verilator/verilator", "cpp"), ("SymbiFlow/prjxray", "py")]


def get(url, as_json=False):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = r.read().decode("utf-8", errors="replace")
            return json.loads(data) if as_json else data
        except Exception as e:
            if "rate limit" in str(e).lower() or "403" in str(e):
                print(f"[code] GitHub rate-limited — stopping repos pass", flush=True); return None
            time.sleep(4 * (attempt + 1))
    return None


def fetch_repo(repo, ext, n_files, existing):
    info = get(f"https://api.github.com/repos/{repo}", as_json=True)
    if not info:
        return 0
    branch = info.get("default_branch", "main")
    tree = get(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1", as_json=True)
    if not tree:
        return 0
    src = [t["path"] for t in tree.get("tree", [])
           if t.get("type") == "blob" and t["path"].rsplit(".", 1)[-1].lower() in LANG]
    if not src:
        return 0
    step = max(1, len(src) // n_files)
    sample = src[::step][:n_files]
    field = LANG.get(ext, "code")
    rid = repo.replace("/", "-")
    got = 0
    for path in sample:
        raw = get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}")
        if raw and 200 < len(raw) < 400000:
            lg = LANG.get(path.rsplit(".", 1)[-1].lower(), field)
            h = hashlib.md5(path.encode()).hexdigest()[:8]
            name = f"{lg}__{rid}_{h}.txt"
            if name not in existing:
                (LIB / name).write_text(raw); existing.add(name); got += 1
        time.sleep(1.2)
    return got


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", action="store_true")
    ap.add_argument("--files-per-repo", type=int, default=12, dest="fpr")
    ap.add_argument("--rfc", type=int, default=0)
    args = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    existing = {f.name for f in LIB.glob("*.txt")}
    total = 0
    if args.repos:
        for repo, ext in REPOS:
            g = fetch_repo(repo, ext, args.fpr, existing)
            total += g
            print(f"[code] {repo:28} +{g} source files", flush=True)
            if g == 0 and total == 0:
                pass
            time.sleep(1)
    if args.rfc:
        man = HERE / "library" / "_rfc_cursor"
        cur = int(man.read_text()) if man.exists() else 0
        got = 0
        while got < args.rfc and cur < 9600:
            cur += 1
            name = f"computing__rfc{cur}.txt"
            if name not in existing:
                txt = get(f"https://www.rfc-editor.org/rfc/rfc{cur}.txt")
                if txt and len(txt) > 2000:
                    (LIB / name).write_text(txt); existing.add(name); got += 1; total += 1
            man.write_text(str(cur)); time.sleep(1.2)
        print(f"[code] +{got} RFCs (cursor at {cur})", flush=True)
    print(f"\n[code] saved {total} code/computing files into library/", flush=True)


if __name__ == "__main__":
    main()
