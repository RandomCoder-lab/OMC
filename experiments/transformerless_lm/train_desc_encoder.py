"""NEXT-1 model — two-tower contrastive encoder: description ↔ code shared space.

Goal: retrieve CODE from a DESCRIPTION query (cross-space). Raw char-histogram scores only
2/10 here because descriptions (English) and code (symbols) have different surface forms.
A learned two-tower encoder maps both into ONE space so emb(desc_i)·emb(code_i) is high.
Char-level hashed-bigram input → universal (no tokenizer/word-list). Tiny, CPU-trainable.

Honest expectation: lifts cases with learnable desc↔code structure (descriptive names/comments);
terse-name+no-synonym cases (gcd↔"greatest common divisor") stay hard without that phrasing in
training — a real limit, not hidden.
"""
import json, math, random, statistics
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).parent
DIM, EMB, EPOCHS, BATCH, LR, TEMP = 2048, 128, 120, 128, 2e-3, 0.07

rows = [json.loads(l) for l in (HERE/'desc_code_dataset.jsonl').read_text().splitlines()]
random.seed(0); random.shuffle(rows)
nval = max(40, len(rows)//10); val, train = rows[:nval], rows[nval:]
print(f"[enc] {len(rows)} pairs → train {len(train)} / val {len(val)}", flush=True)

def feat(s):
    """hashed char-bigram histogram, DIM-dim, normalized (universal char-level input)."""
    h = torch.zeros(DIM); b = s.encode('utf-8','ignore')
    if len(b) > 1:
        a = torch.tensor(list(b[:-1])); c = torch.tensor(list(b[1:]))
        h.scatter_add_(0, (a*257+c) % DIM, torch.ones(len(a)))
    n = h.norm(); return h/n if n > 0 else h

# precompute features
def featmat(rs, key): return torch.stack([feat(r[key]) for r in rs])
Dtr, Ctr = featmat(train,'description'), featmat(train,'code')
Dva, Cva = featmat(val,'description'),  featmat(val,'code')

class Tower(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(
        nn.Linear(DIM,512), nn.GELU(), nn.Linear(512,EMB))
    def forward(s,x): return F.normalize(s.f(x), dim=-1)

desc_t, code_t = Tower(), Tower()
opt = torch.optim.AdamW(list(desc_t.parameters())+list(code_t.parameters()), lr=LR, weight_decay=1e-4)

def infonce(de, ce):
    logits = de @ ce.T / TEMP
    labels = torch.arange(len(de))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def recall_at(de, ce, k=5):
    sims = de @ ce.T
    ranks = sims.argsort(dim=1, descending=True)
    hits = sum(1 for i in range(len(de)) if i in ranks[i,:k].tolist())
    return hits/len(de)

g = torch.Generator().manual_seed(0)
for ep in range(EPOCHS):
    desc_t.train(); code_t.train()
    perm = torch.randperm(len(train), generator=g)
    for i in range(0, len(train), BATCH):
        idx = perm[i:i+BATCH]
        loss = infonce(desc_t(Dtr[idx]), code_t(Ctr[idx]))
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 30 == 0 or ep == EPOCHS-1:
        with torch.no_grad():
            desc_t.eval(); code_t.eval()
            r = recall_at(desc_t(Dva), code_t(Cva), 5)
        print(f"[enc] epoch {ep:3d} val desc→code recall@5 = {r:.2f}", flush=True)

# ── final eval: learned vs char-histogram baseline (cross-space desc→code) ──
with torch.no_grad():
    desc_t.eval(); code_t.eval()
    learned = recall_at(desc_t(Dva), code_t(Cva), 5)
    # baseline: raw histogram, desc query vs code (no learning)
    base = recall_at(F.normalize(Dva,dim=1), F.normalize(Cva,dim=1), 5)
print(f"\n[enc] === cross-space desc→code recall@5 (val, n={len(val)}) ===", flush=True)
print(f"[enc]   char-histogram (no learning) = {base:.2f}", flush=True)
print(f"[enc]   LEARNED two-tower encoder    = {learned:.2f}", flush=True)

# named NL probes (intuition) — retrieve over ALL code embeddings
allrows = train + val
Call = code_t(featmat(allrows,'code')) if True else None
with torch.no_grad():
    Call = code_t(featmat(allrows,'code'))
    name_of = [r['name'] for r in allrows]
    def rank(q, tgt):
        qe = desc_t(feat(q).unsqueeze(0))
        order = (qe @ Call.T).squeeze(0).argsort(descending=True).tolist()
        for rk,ix in enumerate(order):
            if name_of[ix]==tgt: return rk
        return 9999
    print("[enc] NL→code ranks (learned encoder):", flush=True)
    for q,t in [("greatest common divisor","gcd"),("merge sort","merge_sort"),
                ("check if a number is prime","is_prime"),("fibonacci","fibonacci"),
                ("binary search","binary_search"),("dot product","dot")]:
        if t in name_of: print(f"[enc]   {q:28s}→{t:12s} rank={rank(q,t)}", flush=True)

torch.save({'desc_state':desc_t.state_dict(),'code_state':code_t.state_dict(),
            'DIM':DIM,'EMB':EMB}, HERE/'desc_encoder.pt')
print(f"[enc] saved desc_encoder.pt", flush=True)
print("[enc] ENC DONE", flush=True)
