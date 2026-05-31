#!/usr/bin/env python3
"""codemem — write-don't-train addressed memory over any codebase.

Point it at a folder; it trains a tiny char-LM, WRITES a (hidden->next-token) datastore over your code
with no backprop, and IVF-indexes it. Then query it three ways:

  codemem build  <path>  [-o STORE] [--ext .py,.rs,.md] [--max-mb 8] [--steps 1500] [--d 96] ...
  codemem info   <STORE>
  codemem eval   <STORE> [--holdout-frac 0.1 | --holdout FILE]   # LM-only vs LM+memory perplexity
  codemem search <STORE> "how does the IVF index work" [-k 5]    # find related code (provenance)
  codemem complete <STORE> "def build(" [--n 200] [--lam 0.5] [--temp 0.8] [--no-memory]

The proven recipe (GENERATOR_PLAN.md): writing knowledge into an addressed datastore injects it without
training — strongest on CODE (+48% next-char on a held-out, disjoint slice). IVF retrieval keeps ~99-100%
of brute-force quality at 400x+ fewer comparisons (the "Index Done Right" frontier study, 159 runs).
"""
import sys, argparse, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from written_memory import WrittenMemory, MemConfig

DEFAULT_EXT = ".py,.rs,.md,.toml,.txt,.js,.ts,.c,.h,.cpp,.go,.java,.sh,.omc"
SKIP_DIRS = {".git", "node_modules", "target", "__pycache__", ".venv", "venv", "dist", "build",
             ".auto-claude", ".mypy_cache", ".pytest_cache"}


def gather(path: Path, exts: set[str], max_bytes: int, log=print) -> tuple[str, int]:
    """Concatenate files under `path` (filtered by extension), newest-first by size cap. Each file is
    prefixed with a `# ==== relpath ====` banner so provenance snippets show their origin."""
    if path.is_file():
        return path.read_text(errors="replace"), 1
    parts, total, nfiles = [], 0, 0
    files = []
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        if any(d in f.parts for d in SKIP_DIRS):
            continue
        files.append(f)
    for f in files:
        try:
            body = f.read_text(errors="replace")
        except Exception:
            continue
        rel = f.relative_to(path)
        banner = f"\n# ==== {rel} ====\n"
        chunk = banner + body
        if total + len(chunk) > max_bytes:
            remaining = max_bytes - total
            if remaining > len(banner) + 200:
                parts.append(chunk[:remaining]); total += remaining; nfiles += 1
            break
        parts.append(chunk); total += len(chunk); nfiles += 1
    log(f"[codemem] gathered {nfiles} files, {total:,} chars from {path}")
    return "".join(parts), nfiles


def cmd_build(args):
    path = Path(args.path).resolve()
    exts = set(e if e.startswith(".") else "." + e for e in args.ext.split(","))
    text, nfiles = gather(path, exts, int(args.max_mb * 1024 * 1024))
    if len(text) < args.seq * 4:
        sys.exit(f"[codemem] too little text ({len(text)} chars) — check --ext / path")
    store = Path(args.out) if args.out else (path / ".codemem" if path.is_dir() else path.parent / f"{path.stem}.codemem")
    cfg = MemConfig(d=args.d, blocks=args.blocks, seq=args.seq, ds_keys=args.ds_keys,
                    nlist=args.nlist, nprobe=args.nprobe, topk=args.topk,
                    tokenizer=args.tokenizer, bpe_vocab=args.bpe_vocab)
    t0 = time.time()
    mem = WrittenMemory.build(text, cfg, train_steps=args.steps, seed=args.seed)
    # auto-tune λ on the last slice of the corpus (held-out-ish), then persist
    lam = mem.tune_lambda(text[-min(len(text)//10, 50000):])
    mem.save(store)
    print(f"\n[codemem] BUILT store at {store}")
    print(f"[codemem]   {mem.K.shape[0]:,} entries · {mem.index.C.shape[0]} cells · vocab {mem.vocab} "
          f"· λ*={lam:.2f} · {nfiles} files · {time.time()-t0:.0f}s")
    print(f"[codemem] try:  codemem eval {store}   |   codemem search {store} \"...\"   |   "
          f"codemem complete {store} \"def \"")


def _load(args):
    store = Path(args.store)
    if not (store / "memory.pt").exists():
        sys.exit(f"[codemem] no store at {store} (run `codemem build` first)")
    return WrittenMemory.load(store)


def cmd_info(args):
    store = Path(args.store)
    meta = json.loads((store / "meta.json").read_text())
    print(json.dumps(meta, indent=2))


def cmd_eval(args):
    mem = _load(args)
    if args.holdout:
        test = Path(args.holdout).read_text(errors="replace")
    else:
        n = len(mem.source); test = mem.source[int(n * (1 - args.holdout_frac)):]
    lam = args.lam if args.lam is not None else mem.cfg.lam
    sc = mem.score(test, lam=lam, max_windows=args.windows)
    print(f"[codemem] eval on {len(test):,} held-out chars  (λ={sc['lam']:.2f}, "
          f"{sc['chars_per_tok']:.2f} chars/tok)")
    print(f"[codemem]   LM-only     bits/char = {sc['lm_bpc']:.4f}")
    print(f"[codemem]   LM + memory bits/char = {sc['mem_bpc']:.4f}   ({sc['improve_pct']:+.2f}% token-CE)")
    print(f"[codemem]   -> written memory {'injects knowledge' if sc['improve_pct']>0 else 'does not help here'}")


def cmd_search(args):
    mem = _load(args)
    hits = mem.nearest_contexts(args.query, k=args.k)
    print(f"[codemem] nearest written contexts for: {args.query!r}\n")
    for i, r in enumerate(hits, 1):
        snip = r["snippet"].strip().replace("\n", "\n      ")
        print(f"  [{i}] dist={r['distance']:.3f}  pos={r['position']}  next={r['predicted']!r}")
        print(f"      {snip}\n")


def cmd_complete(args):
    mem = _load(args)
    lam = args.lam if args.lam is not None else mem.cfg.lam
    if args.chunked:
        if mem.Ptok is None:
            sys.exit("[codemem] this store predates span-copy; rebuild it to use --chunked")
        mem.calibrate_copy(mem.source[-8000:], target_rate=args.copy_rate)
        out, st = mem.complete_chunked(args.prefix, max_tokens=args.n, span=args.span, lam=lam,
                                       temp=args.temp, max_run=args.max_run, seam=args.seam,
                                       rep_penalty=args.rep_penalty)
        print(f"[codemem] span-copy [copy_rate={st['copy_rate']} ({st['copied']}cp/{st['generated']}gen), "
              f"longest_verbatim_run={st['longest_verbatim_run']} tok, "
              f"distinct_regions={st['distinct_regions']}, span={args.span}, max_run={args.max_run}]:\n")
        print(out)
    else:
        out = mem.complete(args.prefix, n_tokens=args.n, lam=lam, temp=args.temp,
                           use_memory=not args.no_memory, greedy=args.greedy)
        tag = "LM-only" if args.no_memory else f"LM+memory(λ={lam:.2f})"
        print(f"[codemem] completion [{tag}, temp={args.temp}]:\n")
        print(out)


def main():
    ap = argparse.ArgumentParser(prog="codemem", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a memory store from a folder/file")
    b.add_argument("path")
    b.add_argument("-o", "--out", default=None, help="store dir (default <path>/.codemem)")
    b.add_argument("--ext", default=DEFAULT_EXT)
    b.add_argument("--max-mb", type=float, default=8.0)
    b.add_argument("--d", type=int, default=96)
    b.add_argument("--blocks", type=int, default=3)
    b.add_argument("--seq", type=int, default=128)
    b.add_argument("--steps", type=int, default=1500)
    b.add_argument("--ds-keys", type=int, default=400_000, dest="ds_keys")
    b.add_argument("--nlist", type=int, default=1024)
    b.add_argument("--nprobe", type=int, default=4)
    b.add_argument("--topk", type=int, default=256)
    b.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe",
                   help="subword bpe (default, better generation) or char baseline")
    b.add_argument("--bpe-vocab", type=int, default=2048, dest="bpe_vocab")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_build)

    i = sub.add_parser("info", help="show store metadata")
    i.add_argument("store"); i.set_defaults(fn=cmd_info)

    e = sub.add_parser("eval", help="LM-only vs LM+memory perplexity on held-out text")
    e.add_argument("store")
    e.add_argument("--holdout", default=None, help="file to score (default: tail of the corpus)")
    e.add_argument("--holdout-frac", type=float, default=0.1, dest="holdout_frac")
    e.add_argument("--lam", type=float, default=None)
    e.add_argument("--windows", type=int, default=64)
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("search", help="find related code (provenance via nearest written contexts)")
    s.add_argument("store"); s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(fn=cmd_search)

    c = sub.add_parser("complete", help="generate with memory injection")
    c.add_argument("store"); c.add_argument("prefix")
    c.add_argument("--n", type=int, default=200)
    c.add_argument("--lam", type=float, default=None)
    c.add_argument("--temp", type=float, default=0.8)
    c.add_argument("--no-memory", action="store_true")
    c.add_argument("--greedy", action="store_true")
    c.add_argument("--chunked", action="store_true", help="span-copy: paste verbatim corpus chunks when memory matches")
    c.add_argument("--span", type=int, default=8, help="tokens copied per confident match (--chunked)")
    c.add_argument("--copy-rate", type=float, default=0.6, dest="copy_rate",
                   help="target fraction of positions to copy (--chunked); lower = more LM, more novel")
    c.add_argument("--max-run", type=int, default=24, dest="max_run",
                   help="novelty pressure: cap contiguous tokens copied from one region before a seam")
    c.add_argument("--seam", type=int, default=2, help="LM tokens generated at a forced seam (--chunked)")
    c.add_argument("--rep-penalty", type=float, default=1.3, dest="rep_penalty",
                   help="repetition penalty for LM tokens (anti-loop at seams)")
    c.set_defaults(fn=cmd_complete)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
