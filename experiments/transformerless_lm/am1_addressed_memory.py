"""AM-1a — addressed memory (Product-Key Memory) vs dense FFN. Pre-registered (GENERATOR_PLAN.md).

The FFN is a key-value memory scanned densely. PKM addresses it: O(m) access for M=m^2 slots.
Keys are LEARNED, dot-product scored; the top-k SELECTION is the addressing (index) — NOT a Fibonacci
score (that scorer was falsified). Attention + PE are identical across conditions (reused from models.py),
so the only variable is dense-FFN vs addressed-memory.

Conditions: dense (H=4d) | dense_big (H=M, same capacity as PKM, DENSE access) | pkm (M slots, addressed).
Reports val cross-entropy + params + estimated FFN active-FLOPs/token, 3 seeds.
"""
import sys, time, math, statistics, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from corpus import make_dataset, get_batch
from models import Attention, crt_pe   # identical attention + positional encoding across conditions


class DenseFFN(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, H), nn.GELU(), nn.Linear(H, d))
        self.ffn_flops = 2 * d * H            # two matmuls / token
    def forward(self, x): return self.net(x)


class PKM(nn.Module):
    """Product-Key Memory: addressed key-value memory. M = m^2 slots, ~O(m + k*d) access per token."""
    def __init__(self, d, m, topk=8):
        super().__init__()
        self.m, self.M, self.topk = m, m * m, topk
        self.dq = d if d % 2 == 0 else d + 1
        self.half = self.dq // 2
        self.query = nn.Linear(d, self.dq)
        self.k1 = nn.Parameter(torch.randn(m, self.half) * (self.half ** -0.5))
        self.k2 = nn.Parameter(torch.randn(m, self.half) * (self.half ** -0.5))
        self.values = nn.EmbeddingBag(self.M, d, mode="sum")   # M value vectors, gathered+weighted
        nn.init.normal_(self.values.weight, std=d ** -0.5)
        self.bn = nn.BatchNorm1d(self.dq)                       # query BN (PKM paper: spreads key usage)
        self.ffn_flops = d * self.dq + m * self.dq + topk * d   # query + sub-key scoring + value gather
    def forward(self, x):
        B, T, d = x.shape
        q = self.bn(self.query(x).reshape(B * T, self.dq))
        q1, q2 = q[:, :self.half], q[:, self.half:]
        s1, s2 = q1 @ self.k1.t(), q2 @ self.k2.t()             # (N, m) each
        k = self.topk
        v1, i1 = s1.topk(k, dim=-1)
        v2, i2 = s2.topk(k, dim=-1)
        cand_scores = (v1.unsqueeze(2) + v2.unsqueeze(1)).reshape(B * T, k * k)
        cand_idx = (i1.unsqueeze(2) * self.m + i2.unsqueeze(1)).reshape(B * T, k * k)
        sc, ci = cand_scores.topk(k, dim=-1)
        slot = torch.gather(cand_idx, 1, ci)                    # (N, k) addresses into M
        w = F.softmax(sc, dim=-1)
        out = self.values(slot, per_sample_weights=w)          # (N, d) weighted sum of addressed slots
        return out.reshape(B, T, d)


class Block(nn.Module):
    def __init__(self, d, seq_len, ff):
        super().__init__()
        self.attn = Attention(d, gate_mode="none", seq_len=seq_len)
        self.ff = ff
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class LM(nn.Module):
    def __init__(self, vocab, d, n_blocks, seq_len, ff_factory):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.register_buffer("pe", crt_pe(seq_len, d))
        self.blocks = nn.ModuleList([Block(d, seq_len, ff_factory()) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False); self.head.weight = self.embed.weight
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))
    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pe[:T]
        m = self.mask[:T, :T]
        for b in self.blocks: h = b(h, m)
        return self.head(self.ln_f(h))


def train_one(arch, d, m, topk, n_blocks, seq_len, vocab, tr, va, steps, lr, bs, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    H = 4 * d
    if arch == "dense":      ff = lambda: DenseFFN(d, H)
    elif arch == "dense_big":ff = lambda: DenseFFN(d, m * m)
    elif arch == "pkm":      ff = lambda: PKM(d, m, topk)
    else: raise ValueError(arch)
    model = LM(vocab, d, n_blocks, seq_len, ff)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    params = sum(p.numel() for p in model.parameters())
    ffn_flops = model.blocks[0].ff.ffn_flops
    @torch.no_grad()
    def val_loss():
        model.eval(); tot = 0.0
        for _ in range(20):
            x, y = get_batch(va, bs, seq_len, g)
            tot += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
        model.train(); return tot / 20
    model.train()
    for step in range(steps):
        x, y = get_batch(tr, bs, seq_len, g)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return val_loss(), params, ffn_flops


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="dense,dense_big,pkm")  # run subset; merges into results_am1.json
    ap.add_argument("--source", default="tinyshakespeare")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--m", type=int, default=64)          # M = 4096 slots
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    try:
        chars, stoi, itos, enc = make_dataset(seq_len=args.seq, source=args.source)
    except Exception as e:
        print(f"[am1] source {args.source!r} unavailable ({e}); falling back to omc", flush=True)
        chars, stoi, itos, enc = make_dataset(seq_len=args.seq, source="omc")
    vocab = len(chars)
    n = enc.numel(); cut = int(n * 0.9)
    tr, va = enc[:cut], enc[cut:]
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[am1] source={args.source} vocab={vocab} train={tr.numel()} val={va.numel()} "
          f"d={args.d} m={args.m}(M={args.m*args.m}) topk={args.topk} blocks={args.blocks} "
          f"seq={args.seq} steps={args.steps} seeds={seeds}", flush=True)

    out_path = Path(__file__).parent / "results_am1.json"
    results = {}
    if out_path.exists():
        try: results = json.loads(out_path.read_text())
        except Exception: results = {}
    for arch in args.archs.split(","):
        losses = []; params = ffn = 0
        t0 = time.time()
        for sd in seeds:
            vl, params, ffn = train_one(arch, args.d, args.m, args.topk, args.blocks, args.seq,
                                        vocab, tr, va, args.steps, args.lr, args.bs, sd)
            losses.append(vl)
            print(f"[am1]   {arch:9s} seed={sd} val={vl:.4f}", flush=True)
        mean = statistics.mean(losses)
        std = statistics.pstdev(losses)
        results[arch] = dict(mean=mean, std=std, params=params, ffn_flops=ffn,
                             losses=[round(x, 4) for x in losses])
        print(f"[am1] {arch:9s} val={mean:.4f}±{std:.4f}  params={params:,}  "
              f"ffn_flops/tok={ffn:,}  ({time.time()-t0:.0f}s)", flush=True)

    out_path.write_text(json.dumps(results, indent=2))
    print("[am1] wrote/merged results_am1.json", flush=True)
    if all(k in results for k in ("dense", "dense_big", "pkm")):
        d_, db, pk = results["dense"], results["dense_big"], results["pkm"]
        print("\n[am1] === VERDICT (pre-registered, GENERATOR_PLAN.md) ===", flush=True)
        print(f"[am1] dense     {d_['mean']:.4f}  (FFN-FLOPs {d_['ffn_flops']:,}, params {d_['params']:,})", flush=True)
        print(f"[am1] dense_big {db['mean']:.4f}  (FFN-FLOPs {db['ffn_flops']:,}, params {db['params']:,})", flush=True)
        print(f"[am1] pkm       {pk['mean']:.4f}  (FFN-FLOPs {pk['ffn_flops']:,}, params {pk['params']:,})", flush=True)
        print(f"[am1] P1 capacity-per-FLOP: pkm {pk['mean']:.4f} vs dense {d_['mean']:.4f}  "
              f"({d_['ffn_flops']/max(pk['ffn_flops'],1):.1f}x cheaper FFN) -> "
              f"{'HOLDS' if pk['mean'] <= d_['mean'] + 0.02 else 'FAILS'}", flush=True)
        print(f"[am1] P2 addressing≈capacity, cheaper: pkm {pk['mean']:.4f} vs dense_big {db['mean']:.4f}  "
              f"({db['ffn_flops']/max(pk['ffn_flops'],1):.1f}x cheaper FFN than dense_big) -> "
              f"{'HOLDS' if pk['mean'] <= db['mean'] + 0.03 else 'FAILS'}", flush=True)
    else:
        print(f"[am1] have {list(results)} — run remaining archs to complete the verdict.", flush=True)


if __name__ == "__main__":
    main()
