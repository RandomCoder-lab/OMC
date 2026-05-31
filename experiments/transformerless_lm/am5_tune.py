"""AM-5 — establish the HONEST baseline for write-don't-train memory before fancy levers.

AM-3/4 lowballed the basic kNN knobs (k=32, temp=1.0, 200K of 619K keys). Train the base LM ONCE on A
(Shakespeare), build the FULL B_store datastore (unseen P&P), then sweep k / temperature / λ on held-out
B_test. Compute neighbors once at max-k and slice — so the whole grid is one kNN pass. Reports the true
ceiling of the simple approach so later substrate levers are measured against an honest baseline.
"""
import sys, argparse, time
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import get_batch
from am2_write_memory import train_base, hidden_and_logits, build_datastore

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--qpos", type=int, default=64)
    ap.add_argument("--ds", type=int, default=600000)     # use (almost) all of B_store
    ap.add_argument("--maxk", type=int, default=512)
    args = ap.parse_args()

    a = (HERE / "tinyshakespeare.txt").read_text(errors="replace")
    b = (HERE / "pride_prejudice.txt").read_text(errors="replace")
    chars = sorted(set(a + b)); stoi = {c: i for i, c in enumerate(chars)}; vocab = len(chars)
    enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    A, B = enc(a), enc(b)
    A_train = A[:int(A.numel() * 0.9)]
    bcut = int(B.numel() * 0.85); B_store, B_test = B[:bcut], B[bcut:]
    print(f"[am5] vocab={vocab} A_train={A_train.numel()} B_store={B_store.numel()} B_test={B_test.numel()}", flush=True)

    t0 = time.time()
    model, g = train_base(vocab, args.d, args.blocks, args.seq, A_train, A_train, args.steps, 3e-4, 32, 0)
    print(f"[am5] base LM trained ({time.time()-t0:.0f}s)", flush=True)
    ds = min(args.ds, B_store.numel() - args.seq - 1)
    K, V = build_datastore(model, B_store, args.seq, ds, vocab, g)
    print(f"[am5] datastore written: {K.shape[0]:,} keys ({time.time()-t0:.0f}s)", flush=True)

    # one query batch; compute neighbors ONCE at maxk, then slice k / sweep temp / sweep λ
    model.eval()
    gg = torch.Generator().manual_seed(999)
    x, y = get_batch(B_test, args.qpos, args.seq, gg)
    logits, h = hidden_and_logits(model, x)
    hq = h.reshape(-1, h.shape[-1]); tgt = y.reshape(-1)
    p_lm = F.softmax(logits.reshape(-1, vocab), dim=-1)
    N = hq.shape[0]
    nd = torch.zeros(N, args.maxk); ni = torch.zeros(N, args.maxk, dtype=torch.long)
    CH = 256
    for s in range(0, N, CH):
        dist = torch.cdist(hq[s:s + CH], K)
        d, i = dist.topk(args.maxk, dim=-1, largest=False)
        nd[s:s + CH] = d; ni[s:s + CH] = i
    lm_loss = F.nll_loss(torch.log(p_lm + 1e-9), tgt).item()
    print(f"[am5] LM-only on B_test = {lm_loss:.4f}  (neighbors ready, {time.time()-t0:.0f}s)\n", flush=True)

    lambdas = [i / 20 for i in range(0, 19)]
    best = (lm_loss, None)
    for k in [16, 64, 256, args.maxk]:
        ndk, nik = nd[:, :k], ni[:, :k]
        for temp in [1.0, 5.0, 20.0, 100.0]:
            w = F.softmax(-ndk / temp, dim=-1)
            p_knn = torch.zeros(N, vocab); p_knn.scatter_add_(1, V[nik], w)
            bl = (lm_loss, 0.0)
            for lam in lambdas:
                L = F.nll_loss(torch.log((1 - lam) * p_lm + lam * p_knn + 1e-9), tgt).item()
                if L < bl[0]: bl = (L, lam)
            gain = 100.0 * (lm_loss - bl[0]) / lm_loss
            print(f"[am5]   k={k:>4} temp={temp:>5.0f}  best λ={bl[1]:.2f}  loss={bl[0]:.4f}  ({gain:+.2f}%)", flush=True)
            if bl[0] < best[0]: best = (bl[0], dict(k=k, temp=temp, lam=bl[1]))
    g_best = 100.0 * (lm_loss - best[0]) / lm_loss
    print(f"\n[am5] === HONEST BASELINE ===", flush=True)
    print(f"[am5] LM-only {lm_loss:.4f} -> best write-memory {best[0]:.4f}  ({g_best:+.2f}%)  cfg={best[1]}", flush=True)
    import json
    (HERE / "results_am5.json").write_text(json.dumps(dict(lm_only=round(lm_loss, 4),
        best_loss=round(best[0], 4), gain=round(g_best, 2), cfg=best[1], ds=int(K.shape[0])), indent=2))
    print("[am5] wrote results_am5.json", flush=True)


if __name__ == "__main__":
    main()
