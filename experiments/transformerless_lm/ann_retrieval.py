"""NEXT-4 — substrate-native ANN: sub-linear retrieval over locality fps.

The efficiency angle. Brute-force locality retrieval is O(N) per query. A random-hyperplane
LSH bucketing (universal, no training) over the locality fps gives sub-linear search:
hash query → search only its bucket(s). Measure speedup vs recall@5 retention. This is where
substrate addressing (bucketed geometry) could match dense-retrieval efficiency at scale.
"""
import torch, torch.nn.functional as F, random, statistics, time
from pathlib import Path
from locality_fp import build_vocab, hist_fp

HERE = Path(__file__).parent
W, N, NQ, K, BITS = 89, 4000, 200, 5, 10
text = (HERE/'omc_corpus.txt').read_text(errors='replace')[:3_000_000]
stoi, V = build_vocab(text)
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)

random.seed(0)
pos = random.sample(range(len(text)-W), N+NQ); ip, qp = pos[:N], pos[NQ-NQ+NQ:] if False else pos[N:]
idx = F.normalize(torch.stack([hist_fp(ids,p,W,V,bigram=True) for p in ip]).float(), dim=1)
qs  = F.normalize(torch.stack([hist_fp(ids,p,W,V,bigram=True) for p in qp]).float(), dim=1)
D = idx.shape[1]

# random-hyperplane LSH: BITS planes → BITS-bit bucket code
torch.manual_seed(0)
planes = F.normalize(torch.randn(BITS, D), dim=1)
def codes(M): return (M @ planes.T > 0).int()      # [n, BITS]
def code_key(c): return tuple(c.tolist())
idx_codes = codes(idx)
buckets = {}
for i in range(N): buckets.setdefault(code_key(idx_codes[i]), []).append(i)

def brute_top(q): return set((idx @ q).topk(K).indices.tolist())
def ann_top(q, qc, probe=2):
    # search query bucket + nearest `probe` buckets by Hamming distance
    keys = list(buckets)
    kt = torch.tensor(keys)
    ham = (kt != qc).sum(1)
    near = [keys[i] for i in ham.argsort()[:probe].tolist()]
    cand = [i for k in near for i in buckets[k]]
    if not cand: return set()
    sub = idx[cand]; top = (sub @ q).topk(min(K,len(cand))).indices.tolist()
    return set(cand[t] for t in top)

if __name__ == '__main__':
    qc = codes(qs)
    # recall of ANN vs brute-force, and avg candidates examined
    rec=[]; examined=[]
    for j in range(NQ):
        g = brute_top(qs[j])
        a = ann_top(qs[j], qc[j], probe=3)
        rec.append(len(g & a)/K)
        keys=list(buckets); kt=torch.tensor(keys); ham=(kt!=qc[j]).sum(1)
        near=[keys[i] for i in ham.argsort()[:3].tolist()]
        examined.append(sum(len(buckets[k]) for k in near))
    print(f"[ann] LSH ANN over locality-fps (N={N}, {BITS}-bit, probe=3):", flush=True)
    print(f"[ann]   recall@{K} vs brute-force = {statistics.mean(rec):.2f}", flush=True)
    print(f"[ann]   examined ~{statistics.mean(examined):.0f}/{N} candidates → {N/max(statistics.mean(examined),1):.1f}x fewer", flush=True)
    print(f"[ann]   (substrate addressing as a hash bucket → sub-linear retrieval; recall/speedup tunable via probe)", flush=True)
    print("[ann] ANN DONE", flush=True)
