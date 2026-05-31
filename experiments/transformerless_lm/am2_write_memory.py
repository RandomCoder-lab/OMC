"""AM-2 — write-don't-train addressed memory (kNN-LM style), substrate as the index.

The user's insight (2026-05-30), after PKM failed: don't TRAIN the memory (gradient slots are sparse,
slow — PKM lost). WRITE to it by address. A base LM trains ONCE; then a single forward pass over the
corpus WRITES (hidden_state -> next_token) to a datastore with NO backprop. At inference, address the
current hidden state, read the nearest written entries -> a kNN next-token distribution, interpolate
with the LM. "Remembering" (addressed writes, O(1)/item) vs "learning" (gradient weights, slow).

AM-2a (this file): does written memory help, and does writing MORE help monotonically? (exact kNN read,
to establish the mechanism honestly). AM-2b (next, if positive): replace exact kNN with the SUBSTRATE
index (haddr face buckets / locality) -> sublinear read, our contribution over kNN-LM's brute FAISS.

Precedent: kNN-LM (Khandelwal 2020) — writing a datastore cuts perplexity with zero memory training,
and a small LM + datastore beats a bigger one. We test that mechanism on our stack.
"""
import sys, time, statistics, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset, get_batch
from am1_addressed_memory import LM, DenseFFN


def train_base(vocab, d, n_blocks, seq, tr, va, steps, lr, bs, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = LM(vocab, d, n_blocks, seq, lambda: DenseFFN(d, 4 * d))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x, y = get_batch(tr, bs, seq, g)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model, g


def hidden_and_logits(model, x):
    """Return (logits, h) where h is the pre-head hidden state (ln_f output) — the kNN-LM key."""
    B, T = x.shape
    h = model.embed(x) + model.pe[:T]
    m = model.mask[:T, :T]
    for b in model.blocks:
        h = b(h, m)
    h = model.ln_f(h)
    return model.head(h), h


@torch.no_grad()
def build_datastore(model, enc, seq, n_keys, vocab, gen):
    """WRITE pass — no backprop. Collect (hidden_state -> next_token) over the corpus."""
    model.eval()
    keys, vals = [], []
    bs = 64
    while sum(k.shape[0] for k in keys) < n_keys:
        x, y = get_batch(enc, bs, seq, gen)
        _, h = hidden_and_logits(model, x)         # [B,T,d]
        keys.append(h.reshape(-1, h.shape[-1]))    # [B*T, d]
        vals.append(y.reshape(-1))                 # [B*T]
    K = torch.cat(keys)[:n_keys]
    V = torch.cat(vals)[:n_keys]
    return K, V


@torch.no_grad()
def eval_knn(model, va, seq, vocab, K, V, q_positions, k, temp, lambdas):
    """Eval LM-only vs LM+written-memory across interpolation weights λ. Lower loss = better."""
    model.eval()
    g = torch.Generator().manual_seed(999)
    x, y = get_batch(va, q_positions, seq, g)       # [Q,T]
    logits, h = hidden_and_logits(model, x)
    hq = h.reshape(-1, h.shape[-1])                 # [Q*T, d]
    tgt = y.reshape(-1)                             # [Q*T]
    p_lm = F.softmax(logits.reshape(-1, vocab), dim=-1)
    # exact kNN read: nearest written keys -> their next-tokens -> distribution.
    # Chunk queries so the [chunk, Nds] distance matrix never blows memory at large datastores.
    p_knn = torch.zeros(hq.shape[0], vocab)
    CH = 512
    for s in range(0, hq.shape[0], CH):
        q = hq[s:s + CH]
        d2 = torch.cdist(q, K)                       # [chunk, Nds]
        nd, ni = d2.topk(k, dim=-1, largest=False)   # [chunk,k]
        w = F.softmax(-nd / temp, dim=-1)            # [chunk,k]
        p_knn[s:s + CH].scatter_add_(1, V[ni], w)    # aggregate weights by retrieved token
    lm_loss = F.nll_loss(torch.log(p_lm + 1e-9), tgt).item()
    best = (0.0, lm_loss)
    curve = []
    for lam in lambdas:
        p = (1 - lam) * p_lm + lam * p_knn
        loss = F.nll_loss(torch.log(p + 1e-9), tgt).item()
        curve.append((round(lam, 2), round(loss, 4)))
        if loss < best[1]:
            best = (lam, loss)
    return lm_loss, best, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tinyshakespeare")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--qpos", type=int, default=64)            # query windows (×seq tokens scored)
    ap.add_argument("--ds-sizes", default="10000,50000,200000")
    args = ap.parse_args()

    chars, stoi, itos, enc = make_dataset(seq_len=args.seq, source=args.source)
    vocab = len(chars)
    n = enc.numel(); cut = int(n * 0.9)
    tr, va = enc[:cut], enc[cut:]
    print(f"[am2] source={args.source} vocab={vocab} train={tr.numel()} val={va.numel()} "
          f"d={args.d} blocks={args.blocks} seq={args.seq} steps={args.steps} k={args.k}", flush=True)

    t0 = time.time()
    model, g = train_base(vocab, args.d, args.blocks, args.seq, tr, va, args.steps, args.lr, args.bs, args.seed)
    print(f"[am2] base LM trained ({time.time()-t0:.0f}s)", flush=True)

    lambdas = [i / 20 for i in range(0, 19)]   # 0.00 .. 0.90
    sizes = [int(s) for s in args.ds_sizes.split(",")]
    print(f"\n[am2] === write-don't-train: LM-only vs LM + written memory ===", flush=True)
    results = {}
    lm_only = None
    for ds in sizes:
        K, V = build_datastore(model, tr, args.seq, ds, vocab, g)
        lm_loss, (best_lam, best_loss), curve = eval_knn(
            model, va, args.seq, vocab, K, V, args.qpos, args.k, args.temp, lambdas)
        lm_only = lm_loss
        impr = 100.0 * (lm_loss - best_loss) / lm_loss
        results[ds] = dict(lm_loss=round(lm_loss, 4), best_lambda=best_lam,
                           best_loss=round(best_loss, 4), improve_pct=round(impr, 2))
        print(f"[am2] datastore={ds:>7,} keys: LM-only={lm_loss:.4f}  "
              f"LM+memory(λ={best_lam:.2f})={best_loss:.4f}  improvement={impr:+.2f}%", flush=True)

    # monotonic-with-more-writing check
    losses_by_size = [results[s]["best_loss"] for s in sizes]
    mono = all(losses_by_size[i] >= losses_by_size[i + 1] - 1e-4 for i in range(len(losses_by_size) - 1))
    print(f"\n[am2] === VERDICT ===", flush=True)
    print(f"[am2] writing memory helps: {'YES' if results[sizes[-1]]['improve_pct'] > 0 else 'NO'} "
          f"(best improvement {max(r['improve_pct'] for r in results.values()):+.2f}% at no extra training)", flush=True)
    print(f"[am2] more writing -> better (monotonic): {'YES' if mono else 'NO'}  "
          f"losses {losses_by_size}", flush=True)
    import json
    Path(__file__).parent.joinpath("results_am2.json").write_text(json.dumps(results, indent=2))
    print("[am2] wrote results_am2.json", flush=True)


if __name__ == "__main__":
    main()
