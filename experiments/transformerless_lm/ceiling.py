"""ceiling.py — the generator's TRUE ceiling at its ACTUAL job, with the substrate pieces stacked.

Reframe (user, 2026-05-30): in the full system the generator doesn't know+reason+speak — the SUBSTRATE
carries knowledge (write-don't-train memory), addressing (IVF) finds it, token-chunking (BPE) gives real
units. So measure bits-per-char on a HELD-OUT, UNSEEN domain as a function of GENERATOR SIZE, bare vs
+substrate. Three questions that define the ceiling:
  (1) where does the BARE generator plateau as d grows? (the size wall)
  (2) does the substrate gap PERSIST as size grows, or does a bigger generator make addressing redundant?
  (3) does SMALL+substrate BEAT a much-bigger BARE model on the unseen domain? (above-weight, quantified)

Honest: bits-per-char is a PREDICTION proxy (the rigorous foundation), not a direct reasoning score.
The meaningful headline is the unseen-domain comparison: the substrate supplies knowledge the bare model
— at any size trained only on A — simply does not have.
"""
import sys, time, json, traceback
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from am2_write_memory import train_base
from written_memory import WrittenMemory, MemConfig, IVFIndex
from bpe_tokenizer import make_tokenizer

HERE = Path(__file__).parent


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="96,192,288")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--mb", type=float, default=5.0)
    ap.add_argument("--bpe_vocab", type=int, default=2048)
    ap.add_argument("--ds", type=int, default=250000)
    args = ap.parse_args()

    raw = (HERE / "omc_corpus.txt").read_text(errors="replace")[:int(args.mb * 1024 * 1024)]
    n = len(raw)
    A = raw[:n // 2]                                   # train domain
    Brest = raw[n // 2:]                               # UNSEEN during training
    bn = len(Brest)
    B_store = Brest[:int(bn * 0.8)]                    # written to memory
    B_val = Brest[int(bn * 0.8):int(bn * 0.9)]        # tune lambda
    B_test = Brest[int(bn * 0.9):]                     # eval (disjoint from store)
    tok = make_tokenizer("bpe", A + B_store, vocab_size=args.bpe_vocab)   # corpus-derived (law-compliant)
    vocab = tok.vocab_size
    A_enc = torch.tensor(tok.encode(A), dtype=torch.long)
    print(f"[ceiling] vocab={vocab}  A(train)={len(A):,}  B_store={len(B_store):,}  "
          f"B_test={len(B_test):,}  sizes={args.sizes}", flush=True)

    rows = []
    for d in [int(s) for s in args.sizes.split(",")]:
        t0 = time.time()
        try:
            model, _ = train_base(vocab, d, args.blocks, args.seq, A_enc, A_enc, args.steps, 3e-4, 32, 0)
            cfg = MemConfig(d=d, blocks=args.blocks, seq=args.seq, ds_keys=args.ds, nlist=512,
                            nprobe=4, topk=256, tokenizer="bpe", bpe_vocab=args.bpe_vocab)
            wm = WrittenMemory(cfg, tok, model)
            wm.source = B_store
            enc, offs = tok.encode_with_offsets(B_store)
            wm.K, wm.V, wm.P, wm.Ptok = wm._write_pass(torch.tensor(enc, dtype=torch.long),
                                                        torch.tensor(offs, dtype=torch.long))
            wm.index = IVFIndex.build(wm.K, cfg.nlist, cfg.kmeans_iters, 0)
            lam = wm.tune_lambda(B_val)
            bare = wm.score(B_test, lam=0.0)["lm_bpc"]
            plus = wm.score(B_test, lam=lam)["mem_bpc"]
            params = sum(p.numel() for p in model.parameters())
            row = dict(d=d, params=params, bare_bpc=round(bare, 4), sub_bpc=round(plus, 4),
                       lam=round(lam, 2), gain=round(100 * (bare - plus) / bare, 2),
                       secs=round(time.time() - t0))
            rows.append(row)
            (HERE / "results_ceiling.json").write_text(json.dumps(rows, indent=2))
            print(f"[ceiling] d={d} ({params/1e6:.2f}M)  bare={row['bare_bpc']}  "
                  f"+substrate={row['sub_bpc']} (λ{lam:.2f}, {row['gain']:+.1f}%)  [{row['secs']}s]", flush=True)
            del model, wm
            import gc; gc.collect()
        except Exception:
            print(f"[ceiling] d={d} FAILED:\n{traceback.format_exc()}", flush=True)

    if not rows:
        print("[ceiling] no rows", flush=True); return
    print("\n[ceiling] === TRUE-CEILING SCALING CURVE (bits/char on UNSEEN domain) ===", flush=True)
    print("| d | params | bare | +substrate | gain |", flush=True)
    for r in rows:
        print(f"| {r['d']} | {r['params']/1e6:.2f}M | {r['bare_bpc']} | {r['sub_bpc']} | {r['gain']:+.1f}% |", flush=True)
    sm, big = rows[0], rows[-1]
    print(f"\n[ceiling] Q3 punch-above-weight:", flush=True)
    print(f"[ceiling]   smallest+substrate (d={sm['d']}, {sm['params']/1e6:.2f}M) = {sm['sub_bpc']} bpc", flush=True)
    print(f"[ceiling]   largest BARE      (d={big['d']}, {big['params']/1e6:.2f}M) = {big['bare_bpc']} bpc", flush=True)
    beats = sm["sub_bpc"] < big["bare_bpc"]
    print(f"[ceiling]   -> small+substrate {'BEATS' if beats else 'does NOT beat'} the "
          f"{big['params']/sm['params']:.0f}x-bigger bare model on unseen domain", flush=True)
    # Q2: does the substrate gain shrink or hold as size grows?
    print(f"[ceiling] Q2 substrate gain by size: " +
          "  ".join(f"d{r['d']}={r['gain']:+.1f}%" for r in rows), flush=True)
    print("[ceiling] wrote results_ceiling.json", flush=True)


if __name__ == "__main__":
    main()
