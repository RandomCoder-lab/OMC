"""Conventional baseline — the missing control.

A vanilla char-level Transformer matched to FibRec's size (~333K params), trained
on the same corpus / seq_len / step budget, measured with the same corpus 4-gram
validity. Answers: is the substrate LM competitive as a GENERATOR at equal
params+budget, or is the substrate's value purely in addressing/retrieval?

Until now every comparison was substrate-vs-substrate, so 0.547 validity had no
absolute reference. This provides it.
"""
import math, sys, time, statistics
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).parent
SEQ_LEN, BATCH, LR = 256, 24, 3e-4
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # configurable for crossover test
D, L, H = 64, 6, 4          # ~326K params ≈ FibRec v1's 332,880
K = 4

corpus = (HERE / 'omc_corpus.txt').read_text(errors='replace')
chars = sorted(set(corpus)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
ids = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)
n = len(ids); split = int(n * 0.9); train_ids, val_ids = ids[:split], ids[split:]
grams = set(corpus[i:i+K] for i in range(0, min(len(corpus), 20_000_000) - K))
def validity(s): return 0.0 if len(s) < K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, mask):
        y = self.ln1(x)
        a, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x

class Tx(nn.Module):
    def __init__(self, V, d, L, h, T):
        super().__init__()
        self.tok = nn.Embedding(V, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
        for b in self.blocks: h = b(h, mask)
        return self.head(self.lnf(h))

def get_batch(src, g):
    p = torch.randint(0, len(src) - SEQ_LEN - 1, (BATCH,), generator=g)
    x = torch.stack([src[i:i+SEQ_LEN] for i in p])
    y = torch.stack([src[i+1:i+SEQ_LEN+1] for i in p])
    return x, y

@torch.no_grad()
def generate(model, prompt, n_new, seed):
    g = torch.Generator().manual_seed(seed); model.eval()
    ctx = [stoi.get(c, 0) for c in prompt][-SEQ_LEN:]
    out = []
    for _ in range(n_new):
        x = torch.tensor(ctx[-SEQ_LEN:], dtype=torch.long).unsqueeze(0)
        logits = model(x)[0, -1]
        p = F.softmax(logits, dim=-1)
        nx = int(torch.multinomial(p, 1, generator=g).item())
        out.append(nx); ctx.append(nx)
    return ''.join(chars[i] for i in out)

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Tx(V, D, L, H, SEQ_LEN)
    params = sum(p.numel() for p in model.parameters())
    print(f"vanilla transformer: params={params:,} (FibRec v1=332,880)  V={V}  d={D} L={L} h={H}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1); eg = torch.Generator().manual_seed(999)
    t0 = time.time()
    for step in range(STEPS + 1):
        x, y = get_batch(train_ids, g)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                vs = []
                for _ in range(4):
                    xv, yv = get_batch(val_ids, eg)
                    vs.append(F.cross_entropy(model(xv).reshape(-1, V), yv.reshape(-1)).item())
            model.train()
            print(f"  step {step:4d}  val={statistics.mean(vs):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    # generation validity, same prompts as measure_validity.py
    prompts = ["fn ", "h x = ", "fn add(a, b) {", "return "]
    gv = [validity(generate(model, pr, 140, s)) for s in range(3) for pr in prompts]
    print(f"BASELINE transformer gen_validity={statistics.mean(gv):.3f}  (n={len(gv)})", flush=True)
