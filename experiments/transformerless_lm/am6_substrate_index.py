"""AM-6 — the SUBSTRATE INDEX: can dodecahedral addressing make big-k retrieval cheap?

Brute-force kNN (AM-5) hit +11.35% scanning all 600K keys — the cost bottleneck. Here we BUCKET the
datastore by dodecahedral face-codes (P random 3-projections -> 12 faces each -> 12^P buckets, the
proven uniform addressing), retrieve only candidates in the query's bucket(s), exact-kNN within them.
Measure the TRADEOFF: gain-retained vs candidates-scanned (speedup) vs brute force. Honest expectation:
a tradeoff exists (crude ANN loses recall); the question is whether our addressing keeps most of the
gain at large speedup. addressing = WHERE to look (index, proven-safe), exact-kNN = the actual scoring.
"""
import sys, argparse, time, math, itertools
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import get_batch
from am2_write_memory import train_base, hidden_and_logits, build_datastore

HERE = Path(__file__).parent
PHI = (1 + 5 ** 0.5) / 2


def normals():
    raw = [[0, 1, PHI], [0, 1, -PHI], [0, -1, PHI], [0, -1, -PHI],
           [1, PHI, 0], [1, -PHI, 0], [-1, PHI, 0], [-1, -PHI, 0],
           [PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, 1], [-PHI, 0, -1]]
    N = torch.tensor(raw, dtype=torch.float32)
    return N / N.norm(dim=1, keepdim=True)               # [12,3]


def faces(vecs, R, NRM, topw=1):
    """vecs [N,d] -> for each of P projections, the top-`topw` dodecahedral faces. Returns [P,N,topw]."""
    out = []
    for p in range(R.shape[0]):
        proj = vecs @ R[p]                                # [N,3]
        sims = proj @ NRM.t()                             # [N,12]
        out.append(sims.topk(topw, dim=-1).indices)       # [N,topw]
    return torch.stack(out)                               # [P,N,topw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--qpos", type=int, default=16)
    ap.add_argument("--ds", type=int, default=250000)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--ch", type=int, default=48)        # cdist chunk — small to cap peak RAM
    args = ap.parse_args()

    a = (HERE / "tinyshakespeare.txt").read_text(errors="replace")
    b = (HERE / "pride_prejudice.txt").read_text(errors="replace")
    chars = sorted(set(a + b)); stoi = {c: i for i, c in enumerate(chars)}; vocab = len(chars)
    enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    A, B = enc(a), enc(b)
    A_train = A[:int(A.numel() * 0.9)]
    bcut = int(B.numel() * 0.85); B_store, B_test = B[:bcut], B[bcut:]

    t0 = time.time()
    model, g = train_base(vocab, args.d, args.blocks, args.seq, A_train, A_train, args.steps, 3e-4, 32, 0)
    K, V = build_datastore(model, B_store, args.seq, min(args.ds, B_store.numel() - args.seq - 1), vocab, g)
    Ndb = K.shape[0]
    print(f"[am6] trained + datastore {Ndb:,} keys ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    torch.set_grad_enabled(False)        # CRITICAL: the per-query eval loop must not accumulate graph
    gg = torch.Generator().manual_seed(999)
    x, y = get_batch(B_test, args.qpos, args.seq, gg)
    logits, h = hidden_and_logits(model, x)
    hq = h.reshape(-1, h.shape[-1]); tgt = y.reshape(-1)
    p_lm = F.softmax(logits.reshape(-1, vocab), dim=-1)
    Nq = hq.shape[0]
    lm_loss = F.nll_loss(torch.log(p_lm + 1e-9), tgt).item()

    def sweep_lambda(p_knn):
        best = (lm_loss, 0.0)
        for i in range(19):
            lam = i / 20
            L = F.nll_loss(torch.log((1 - lam) * p_lm + lam * p_knn + 1e-9), tgt).item()
            if L < best[0]: best = (L, lam)
        return best

    # brute-force reference (scan all keys) — small chunks to cap peak RAM
    p_knn_bf = torch.zeros(Nq, vocab)
    CH = args.ch
    for s in range(0, Nq, CH):
        dist = torch.cdist(hq[s:s + CH], K)
        nd, ni = dist.topk(args.k, dim=-1, largest=False)
        p_knn_bf[s:s + CH].scatter_add_(1, V[ni], F.softmax(-nd, dim=-1))
        del dist, nd, ni
    bf = sweep_lambda(p_knn_bf)
    bf_gain = 100 * (lm_loss - bf[0]) / lm_loss
    print(f"[am6] LM-only={lm_loss:.4f}  brute-force(scan {Ndb:,}) -> {bf[0]:.4f} ({bf_gain:+.2f}%, λ={bf[1]:.2f})\n", flush=True)

    NRM = normals()
    print(f"[am6] {'config':22s} {'avg_cands':>10s} {'speedup':>8s} {'gain':>8s} {'retained':>9s}", flush=True)
    results = {}
    torch.manual_seed(0)
    for P in [1, 2, 3]:
        R = torch.randn(P, args.d, 3)
        kf = faces(K, R, NRM, topw=1)[:, :, 0]            # [P,Ndb] key bucket per projection
        key_code = sum(kf[p].long() * (12 ** p) for p in range(P))   # [Ndb]
        # bucket -> key indices
        order = key_code.argsort()
        sorted_codes = key_code[order]
        buckets = {}
        uniq, counts = torch.unique_consecutive(sorted_codes, return_counts=True)
        off = 0
        for c, cnt in zip(uniq.tolist(), counts.tolist()):
            buckets[c] = order[off:off + cnt]; off += cnt
        for topw in [1, 2]:                                # single-bucket vs multi-probe
            qf = faces(hq, R, NRM, topw=topw)              # [P,Nq,topw]
            p_knn = torch.zeros(Nq, vocab); cand_count = 0
            for qi in range(Nq):
                # candidate bucket codes = cartesian product of the topw faces per projection
                per_p = [qf[p, qi].tolist() for p in range(P)]
                codes = set()
                for combo in itertools.product(*per_p):
                    codes.add(sum(int(combo[p]) * (12 ** p) for p in range(P)))
                idxs = [buckets[c] for c in codes if c in buckets]
                if not idxs:
                    continue
                cand = torch.cat(idxs)
                cand_count += cand.numel()
                kk = min(args.k, cand.numel())
                dist = torch.cdist(hq[qi:qi + 1], K[cand])      # [1, |cand|]
                nd, ni = dist.topk(kk, dim=-1, largest=False)
                slots = cand[ni[0]]
                w = F.softmax(-nd[0], dim=-1)
                p_knn[qi].scatter_add_(0, V[slots], w)
            avg_cands = cand_count / Nq
            loss, lam = sweep_lambda(p_knn)
            gain = 100 * (lm_loss - loss) / lm_loss
            retained = 100 * gain / bf_gain if bf_gain > 0 else 0
            tag = f"P={P} probe={topw}"
            print(f"[am6] {tag:22s} {avg_cands:10.0f} {Ndb/max(avg_cands,1):7.0f}x {gain:+7.2f}% {retained:8.0f}%", flush=True)
            results[tag] = dict(avg_cands=round(avg_cands), speedup=round(Ndb/max(avg_cands,1)),
                                gain=round(gain, 2), retained=round(retained))
    import json
    (HERE / "results_am6.json").write_text(json.dumps(
        dict(lm_only=round(lm_loss, 4), brute_gain=round(bf_gain, 2), configs=results), indent=2))
    print(f"\n[am6] wrote results_am6.json  (brute baseline {bf_gain:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()
