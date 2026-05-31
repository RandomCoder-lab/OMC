"""AM-7 — THE 35B MICROCOSM: can a SMALL LM + written memory of everything rival a BIG LM trained on
everything? (the user's headline thesis, in miniature, on a code corpus.)

Framing: a big model "trained on 35B tokens" has all that knowledge in its WEIGHTS. The thesis: train a
SMALL model on a TINY slice, let it READ everything via an addressed memory (IVF, the AM-6 winner), and
see if small+memory approaches big-trained-on-all. If yes, memory-scale substitutes for training-scale —
you don't train on 35B, you MEMORIZE it (O(1) writes vs gradient descent).

Conditions, all eval'd on a held-out TEST slice (disjoint from the memory store; λ tuned on a separate
VAL slice, NOT on test):
  1. SMALL-floor      : d_small LM trained on C_small only, no memory      (the floor)
  2. SMALL+MEM        : SMALL LM + IVF memory written from C_full           <- THE THESIS
  3. BIG              : d_big LM trained on C_full (everything), no memory  (the target to rival)
  4. SMALL-alldata    : d_small LM trained on C_full, no memory            (isolates params vs data)
  5. BIG+MEM          : d_big LM trained on C_small + IVF memory(C_full)   (does memory help big too)

Headline: gap-closed % = (SMALL+MEM - SMALL_floor) / (BIG - SMALL_floor) in loss. 100% => memory fully
replaced training on the extra data at small-model size. Honest: small-scale char-LM, perplexity proxy.
Everything no_grad; IVF retrieval vectorized per-cell; multi-seed.
"""
import sys, os, time, json, gc, argparse
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import get_batch
from am2_write_memory import train_base, hidden_and_logits, build_datastore
from idx_night import kmeans, cdist_argmin

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "12")))
HERE = Path(__file__).parent
LED = HERE / "MICROCOSM_RESULTS.md"


def log(s):
    with open(LED, "a") as f:
        f.write(s + "\n")
    print(s, flush=True)


@torch.no_grad()
def lm_probs(model, enc, seq, vocab, pos, n_batches=4, bs=32, seed=999):
    """Stack a few batches of (p_lm, hidden, target) from enc starting near pos."""
    model.eval()
    ps, hs, ts = [], [], []
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        x, y = get_batch(enc, bs, seq, g)
        logits, h = hidden_and_logits(model, x)
        ps.append(F.softmax(logits.reshape(-1, vocab), dim=-1))
        hs.append(h.reshape(-1, h.shape[-1]))
        ts.append(y.reshape(-1))
    return torch.cat(ps), torch.cat(hs), torch.cat(ts)


@torch.no_grad()
def ivf_pknn(hq, K, V, cell_of_key, C, nprobe, k, vocab, qch=128):
    """Vectorized IVF retrieval -> p_knn [Nq, vocab]. (Same engine proven in idx_night, 159 runs.)"""
    Nq = hq.shape[0]
    buckets = {}
    order = cell_of_key.argsort()
    sc = cell_of_key[order]
    uniq, cnts = torch.unique_consecutive(sc, return_counts=True)
    off = 0
    for c, n in zip(uniq.tolist(), cnts.tolist()):
        buckets[c] = order[off:off + n]; off += n
    qcells = torch.cdist(hq, C).topk(nprobe, dim=1, largest=False).indices   # [Nq, nprobe]
    run_d = torch.full((Nq, k), 1e30)
    run_t = torch.zeros((Nq, k), dtype=torch.long)
    for col in range(nprobe):
        qc = qcells[:, col]
        for cell in torch.unique(qc).tolist():
            keys_c = buckets.get(cell)
            if keys_c is None:
                continue
            qids = (qc == cell).nonzero().flatten()
            if qids.numel() == 0:
                continue
            Kc, Vc = K[keys_c], V[keys_c]
            kk = min(k, keys_c.numel())
            for s in range(0, qids.numel(), qch):
                qi = qids[s:s + qch]
                d = torch.cdist(hq[qi], Kc)
                cd, ci = d.topk(kk, dim=1, largest=False)
                allk = torch.cat([run_d[qi], cd], 1)
                allt = torch.cat([run_t[qi], Vc[ci]], 1)
                md, mi = allk.topk(k, dim=1, largest=False)
                run_d[qi] = md
                run_t[qi] = torch.gather(allt, 1, mi)
                del d, cd, ci, allk, allt
    matched = run_d[:, 0] < 1e29
    w = F.softmax(-run_d, dim=1)
    p_knn = torch.zeros(Nq, vocab)
    p_knn.scatter_add_(1, run_t, w)
    return p_knn, matched


def ce(p, tgt):
    return F.nll_loss(torch.log(p + 1e-9), tgt).item()


def mem_loss(model, K, V, C, cok, vocab, seq, val_enc, test_enc, nprobe, k):
    """Pick λ on VAL, report interpolated CE on TEST (λ NOT tuned on test)."""
    pv, hv, tv = lm_probs(model, val_enc, seq, vocab, 0, seed=111)
    pt, ht, tt = lm_probs(model, test_enc, seq, vocab, 0, seed=999)
    pkv, _ = ivf_pknn(hv, K, V, cok, C, nprobe, k, vocab)
    pkt, _ = ivf_pknn(ht, K, V, cok, C, nprobe, k, vocab)
    best = (1e9, 0.0)
    for i in range(19):
        lam = i / 20
        L = ce((1 - lam) * pv + lam * pkv, tv)
        if L < best[0]: best = (L, lam)
    lam = best[1]
    return ce((1 - lam) * pt + lam * pkt, tt), lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="omc_corpus.txt")
    ap.add_argument("--full_mb", type=float, default=8.0)     # "everything" = first N MB
    ap.add_argument("--small_kb", type=int, default=200)      # tiny train slice (KB)
    ap.add_argument("--d_small", type=int, default=64)
    ap.add_argument("--d_big", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--ds", type=int, default=400000)         # memory keys from C_full
    ap.add_argument("--nprobe", type=int, default=4)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    raw = (HERE / args.corpus).read_text(errors="replace")[:int(args.full_mb * 1024 * 1024)]
    chars = sorted(set(raw)); stoi = {c: i for i, c in enumerate(chars)}; vocab = len(chars)
    enc = torch.tensor([stoi[c] for c in raw], dtype=torch.long)
    n = enc.numel()
    # held-out tail: val + test (disjoint from everything that gets trained/memorized)
    test_n = 80000; val_n = 80000
    C_test = enc[n - test_n:]
    C_val = enc[n - test_n - val_n:n - test_n]
    C_full = enc[:n - test_n - val_n]                          # "everything" available
    C_small = C_full[:args.small_kb * 1024]                    # the tiny slice
    log(f"\n# AM-7 35B MICROCOSM — {time.strftime('%Y-%m-%d %H:%M')}  corpus={args.corpus} ({args.full_mb}MB)")
    log(f"vocab={vocab}  C_full={C_full.numel():,}  C_small={C_small.numel():,}  "
        f"val={C_val.numel():,}  test={C_test.numel():,}  d_small={args.d_small} d_big={args.d_big}")

    def n_params(d):
        # rough: embed + blocks(attn ~4d^2 + ffn 8d^2) + head
        return vocab * d + args.blocks * (12 * d * d) + d * vocab

    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        log(f"\n## seed {seed}")
        # train the models
        m_small, _ = train_base(vocab, args.d_small, args.blocks, args.seq, C_small, C_small, args.steps, 3e-4, 32, seed)
        m_small_full, _ = train_base(vocab, args.d_small, args.blocks, args.seq, C_full, C_full, args.steps, 3e-4, 32, seed)
        m_big, gbig = train_base(vocab, args.d_big, args.blocks, args.seq, C_full, C_full, args.steps, 3e-4, 32, seed)

        # floors / ceilings (no memory) on TEST
        def lm_only(model):
            p, h, t = lm_probs(model, C_test, args.seq, vocab, 0, seed=999)
            return ce(p, t)
        L_small = lm_only(m_small)
        L_small_full = lm_only(m_small_full)
        L_big = lm_only(m_big)

        # memory of EVERYTHING, indexed by IVF (built from each model's OWN hidden space)
        def with_mem(model):
            K, V = build_datastore(model, C_full, args.seq, min(args.ds, C_full.numel() - args.seq - 1), vocab, gbig)
            C = kmeans(K, args.nlist, iters=8, seed=seed)
            cok = cdist_argmin(K, C)
            L, lam = mem_loss(model, K, V, C, cok, vocab, args.seq, C_val, C_test, args.nprobe, args.k)
            del K, V, C, cok; gc.collect()
            return L, lam
        L_small_mem, lam_s = with_mem(m_small)
        L_big_mem, lam_b = with_mem(m_big)

        gap = L_small - L_big
        closed = 100.0 * (L_small - L_small_mem) / gap if abs(gap) > 1e-6 else 0.0
        log(f"| condition | test CE | params |")
        log(f"|---|---|---|")
        log(f"| SMALL-floor (d{args.d_small}, train small) | {L_small:.4f} | {n_params(args.d_small)/1e6:.2f}M |")
        log(f"| SMALL+MEM (d{args.d_small}+mem-of-all, λ={lam_s:.2f}) | {L_small_mem:.4f} | {n_params(args.d_small)/1e6:.2f}M+mem |")
        log(f"| SMALL-alldata (d{args.d_small}, train ALL) | {L_small_full:.4f} | {n_params(args.d_small)/1e6:.2f}M |")
        log(f"| BIG (d{args.d_big}, train ALL) | {L_big:.4f} | {n_params(args.d_big)/1e6:.2f}M |")
        log(f"| BIG+MEM (d{args.d_big}+mem, λ={lam_b:.2f}) | {L_big_mem:.4f} | {n_params(args.d_big)/1e6:.2f}M+mem |")
        log(f"**seed {seed}: gap SMALL→BIG closed by memory = {closed:.0f}%** "
            f"(small {L_small:.3f} → small+mem {L_small_mem:.3f} → big {L_big:.3f})  [{time.time()-t0:.0f}s]")
        rows.append(dict(seed=seed, L_small=L_small, L_small_mem=L_small_mem, L_small_full=L_small_full,
                         L_big=L_big, L_big_mem=L_big_mem, closed=closed, lam_s=lam_s, lam_b=lam_b,
                         p_small=n_params(args.d_small), p_big=n_params(args.d_big)))
        (HERE / "results_am7.json").write_text(json.dumps(rows, indent=2))
        del m_small, m_small_full, m_big; gc.collect()

    def mean(key):
        return sum(r[key] for r in rows) / len(rows)
    log(f"\n## === MICROCOSM VERDICT ({len(rows)} seeds) ===")
    log(f"SMALL-floor   test CE = {mean('L_small'):.4f}")
    log(f"SMALL+MEM     test CE = {mean('L_small_mem'):.4f}   <- small model READING everything")
    log(f"SMALL-alldata test CE = {mean('L_small_full'):.4f}")
    log(f"BIG           test CE = {mean('L_big'):.4f}   <- big model TRAINED on everything")
    log(f"BIG+MEM       test CE = {mean('L_big_mem'):.4f}")
    pr = mean('p_big') / mean('p_small')
    log(f"\nGap SMALL→BIG closed by memory: **{mean('closed'):.0f}%**  "
        f"(at {pr:.1f}x FEWER params than BIG, memory on CPU)")
    if mean('closed') >= 70:
        log("-> THESIS SUPPORTED at this scale: a small model READING everything via addressed memory "
            "approaches a big model TRAINED on everything. Memory-scale substitutes for training-scale.")
    elif mean('closed') >= 30:
        log("-> PARTIAL: memory closes a meaningful fraction of the gap but doesn't fully replace training.")
    else:
        log("-> NOT at this scale: memory did not substitute for training the big model. Honest negative.")
    log(f"\n# AM-7 DONE {time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
