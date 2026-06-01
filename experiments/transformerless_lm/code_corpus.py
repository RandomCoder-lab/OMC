#!/usr/bin/env python3
"""code_corpus.py — download CODE as a corpus (no API rate limit). Instead of GitHub's 60/hr file API,
stream each repo's whole-repo tarball from codeload.github.com (one request per repo, unauthenticated, no
limit), extract a SAMPLE of source files on the fly, and EARLY-STOP once enough are collected — so the
download is bounded even for huge repos (we never finish reading a 250MB tarball; we stop after N files).
Source files land in library/<language>__<repo>_<hash>.txt for the disk grower to fold. Agnostic: real
public source as data; the web learns each language's keywords/identifiers/idioms by co-occurrence.

  python code_corpus.py                 # all repos, ~per-repo sample
  python code_corpus.py --per-repo 25
"""
import sys, io, time, tarfile, hashlib, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "library"
UA = {"User-Agent": "OMC-knowledge-web/1.0 (research; local)"}

LANG = {"py": "python", "rs": "rust", "go": "go", "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "hpp": "cpp",
        "js": "javascript", "ts": "typescript", "java": "java", "rb": "ruby", "v": "verilog", "scala": "scala",
        "swift": "swift", "kt": "kotlin", "hs": "haskell", "jl": "julia", "r": "r", "php": "php", "cs": "csharp",
        "lua": "lua", "ex": "elixir", "exs": "elixir", "clj": "clojure", "zig": "zig", "ml": "ocaml",
        "pl": "perl", "pm": "perl", "f90": "fortran", "erl": "erlang", "dart": "dart", "nim": "nim", "d": "d",
        "sh": "shell", "sv": "systemverilog", "vhd": "vhdl", "groovy": "groovy", "elm": "elm", "jl2": "julia"}

REPOS = ["python/cpython", "rust-lang/rust", "golang/go", "torvalds/linux", "llvm/llvm-project",
         "postgres/postgres", "redis/redis", "git/git", "nodejs/node", "facebook/react", "numpy/numpy",
         "microsoft/TypeScript", "openjdk/jdk", "bitcoin/bitcoin", "ruby/ruby", "sqlite/sqlite",
         "ggerganov/llama.cpp", "apple/swift", "JetBrains/kotlin", "JuliaLang/julia", "php/php-src",
         "dotnet/runtime", "lua/lua", "elixir-lang/elixir", "clojure/clojure", "ziglang/zig", "ocaml/ocaml",
         "Perl/perl5", "erlang/otp", "dart-lang/sdk", "nim-lang/Nim", "ghc/ghc", "scala/scala3",
         "crystal-lang/crystal", "vlang/v", "kubernetes/kubernetes", "django/django", "rails/rails",
         "denoland/deno", "chipsalliance/rocket-chip", "verilator/verilator", "haskell/cabal"]

DONE = LIB / "_code_corpus_done.txt"


def stream_repo(repo, per_repo, existing, byte_cap=80_000_000):
    rid = repo.replace("/", "-")
    for branch in ("main", "master", "trunk"):
        url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)
        except Exception:
            continue
        got, read = 0, 0
        try:
            tar = tarfile.open(fileobj=resp, mode="r|gz")
            for m in tar:
                read += max(m.size, 0)
                if not m.isfile():
                    continue
                ext = m.name.rsplit(".", 1)[-1].lower()
                lang = LANG.get(ext)
                if not lang or not (200 < m.size < 200_000):
                    continue
                try:
                    data = tar.extractfile(m).read().decode("utf-8", "replace")
                except Exception:
                    continue
                h = hashlib.md5(m.name.encode()).hexdigest()[:8]
                name = f"{lang}__{rid}_{h}.txt"
                if name not in existing:
                    (LIB / name).write_text(data); existing.add(name); got += 1
                if got >= per_repo or read > byte_cap:
                    break
            resp.close()
            return got
        except Exception as e:
            print(f"  [{repo}] stream error after {got}: {e}", flush=True)
            return got
    return -1   # no branch worked


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo", type=int, default=20)
    args = ap.parse_args()
    LIB.mkdir(exist_ok=True)
    existing = {f.name for f in LIB.glob("*.txt")}
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    total = 0
    for repo in REPOS:
        if repo in done:
            continue
        g = stream_repo(repo, args.per_repo, existing)
        if g >= 0:
            done.add(repo); DONE.write_text("\n".join(sorted(done)))
            print(f"[code] {repo:32} +{g} files", flush=True); total += g
        else:
            print(f"[code] {repo:32} (no branch / unreachable)", flush=True)
        time.sleep(1)
    print(f"\n[code] +{total} source files into library/ (codeload, no API limit)", flush=True)


if __name__ == "__main__":
    main()
