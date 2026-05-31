"""AM-4 — the dual-band gate as a CONFIDENCE signal on the written memory (compounding on AM-3).

AM-3 used a FIXED interpolation weight λ (trust the memory equally everywhere). But we built the HBit
dual-band gate (value_divergence / hbit_divergence) precisely to say "in-tune vs off-manifold." Applied
to retrieval: when the nearest written key is CLOSE (the memory has a good match for this context →
in-tune), trust it MORE; when it's far (a gap → off-manifold), fall back to the LM. So λ becomes
PER-TOKEN, gated by retrieval confidence = the dual-band α/β gate, on memory.

Test: same Shakespeare->P&P knowledge-injection setup as AM-3. fixed-λ (AM-3) vs gated-λ. Does the gate
compound the gain? Pre-registered: gated ≥ fixed.
"""
import sys, argparse, time
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import get_batch
from am2_write_memory import train_base, hidden_and_logits, build_datastore

HERE = Path(__file__).parent


@torch.no_grad()
def eval_gated(model, eval_enc, seq, vocab, K, V, qpos, k, lambdas):
    model.eval()
    g = torch.Generator().manual_seed(999)
    x, y = get_batch(eval_enc, qpos, seq, g)
    logits, h = hidden_and_logits(model, x)
    hq = h.reshape(-1, h.shape[-1]); tgt = y.reshape(-1)
    p_lm = F.softmax(logits.reshape(-1, vocab), dim=-1)
    N = hq.shape[0]
    p_knn = torch.zeros(N, vocab); d1 = torch.zeros(N)
    CH = 512
    for s in range(0, N, CH):
        dist = torch.cdist(hq[s:s + CH], K)
        nd, ni = dist.topk(k, dim=-1, largest=False)
        w = F.softmax(-nd, dim=-1)
        p_knn[s:s + CH].scatter_add_(1, V[ni], w)
        d1[s:s + CH] = nd[:, 0]                         # nearest-neighbor distance = the gate signal
    lm_loss = F.nll_loss(torch.log(p_lm + 1e-9), tgt).item()

    def loss_of(p):
        return F.nll_loss(torch.log(p + 1e-9), tgt).item()

    # (1) FIXED λ — the AM-3 baseline
    best_fixed = (0.0, lm_loss)
    for lam in lambdas:
        L = loss_of((1 - lam) * p_lm + lam * p_knn)
        if L < best_fixed[1]: best_fixed = (lam, L)

    # (2) GATED λ — per-token confidence from nearest distance (the dual-band gate):
    #     close (in-tune) -> conf~1 -> trust memory; far (gap) -> conf->0 -> fall back to LM.
    med = d1.median()
    conf = torch.exp(-d1 / (med + 1e-9)).clamp(0, 1).unsqueeze(1)
    best_gated = (0.0, lm_loss)
    for lam in lambdas:
        li = lam * conf
        L = loss_of((1 - li) * p_lm + li * p_knn)
        if L < best_gated[1]: best_gated = (lam, L)
    return lm_loss, best_fixed, best_gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="tinyshakespeare.txt")
    ap.add_argument("--b", default="pride_prejudice.txt")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--qpos", type=int, default=48)
    ap.add_argument("--ds", type=int, default=200000)
    args = ap.parse_args()

    a_text = (HERE / args.a).read_text(errors="replace")
    b_text = (HERE / args.b).read_text(errors="replace")
    chars = sorted(set(a_text + b_text)); stoi = {c: i for i, c in enumerate(chars)}; vocab = len(chars)
    enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    A, B = enc(a_text), enc(b_text)
    A_train = A[:int(A.numel() * 0.9)]
    bcut = int(B.numel() * 0.85); B_store, B_test = B[:bcut], B[bcut:]
    print(f"[am4] vocab={vocab} A_train={A_train.numel()} B_store={B_store.numel()} B_test={B_test.numel()}", flush=True)

    t0 = time.time()
    model, g = train_base(vocab, args.d, args.blocks, args.seq, A_train, A_train,
                          args.steps, args.lr, args.bs, args.seed)
    print(f"[am4] base LM trained on A only ({time.time()-t0:.0f}s)", flush=True)
    K, V = build_datastore(model, B_store, args.seq, args.ds, vocab, g)
    lambdas = [i / 20 for i in range(0, 19)]
    lm, bf, bg = eval_gated(model, B_test, args.seq, vocab, K, V, args.qpos, args.k, lambdas)

    gf = 100.0 * (lm - bf[1]) / lm
    gg = 100.0 * (lm - bg[1]) / lm
    print(f"\n[am4] === dual-band gate on written memory (Shakespeare->P&P, held-out) ===", flush=True)
    print(f"[am4] LM-only                       = {lm:.4f}", flush=True)
    print(f"[am4] + memory, FIXED λ={bf[0]:.2f}          = {bf[1]:.4f}   ({gf:+.2f}%)   [AM-3 baseline]", flush=True)
    print(f"[am4] + memory, GATED λmax={bg[0]:.2f}        = {bg[1]:.4f}   ({gg:+.2f}%)   [dual-band confidence]", flush=True)
    print(f"\n[am4] === VERDICT ===", flush=True)
    print(f"[am4] gate compounds the gain: {'YES' if bg[1] < bf[1] - 1e-4 else 'NO'}  "
          f"(fixed {gf:+.2f}% -> gated {gg:+.2f}%, +{gg-gf:.2f} pts)", flush=True)
    import json
    (HERE / "results_am4.json").write_text(json.dumps(
        dict(lm_only=round(lm, 4), fixed=round(bf[1], 4), fixed_lambda=bf[0], gain_fixed=round(gf, 2),
             gated=round(bg[1], 4), gated_lambda=bg[0], gain_gated=round(gg, 2)), indent=2))
    print("[am4] wrote results_am4.json", flush=True)


if __name__ == "__main__":
    main()
