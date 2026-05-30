"""ADDR-3 — Retrieval-relevance benchmark: does φ-retrieval return RELEVANT windows?

We proved φ FACE uniformity (χ²17) but never measured retrieval RELEVANCE. Ground-truth
content similarity = char-trigram Jaccard between windows (a real, label-free content metric).
For each query, gold = top-k index windows by Jaccard. Measure recall@k of:
  - φ-fp cosine retrieval   (the current index mechanism)
  - locality-fp cosine retrieval (ADDR-1)
vs the gold Jaccard ranking. Hypothesis: φ ≈ random (no locality); locality recovers gold.
"""
import torch, random, statistics
from pathlib import Path
from corpus_address_index import substrate_fingerprint
from locality_fp import build_vocab, hist_fp

HERE = Path(__file__).parent
W = 89          # window length
N_IDX = 600     # index windows
N_Q = 100       # queries
K = 5

text = (HERE/'omc_corpus.txt').read_text(errors='replace')[:2_000_000]
stoi, V = build_vocab(text)
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def trigrams(s): return set(s[i:i+3] for i in range(len(s)-2))
def jaccard(a, b):
    A, B = trigrams(a), trigrams(b)
    return len(A & B) / max(len(A | B), 1)

random.seed(0)
positions = random.sample(range(len(text)-W), N_IDX + N_Q)
idx_pos, q_pos = positions[:N_IDX], positions[N_IDX:]
idx_txt = [text[p:p+W] for p in idx_pos]
q_txt   = [text[p:p+W] for p in q_pos]

# fingerprints for the index
phi_idx = torch.stack([substrate_fingerprint(t, 64) for t in idx_txt])
loc_idx = torch.stack([hist_fp(ids, p, W, V) for p in idx_pos])
import torch.nn.functional as F
phi_idx_n = F.normalize(phi_idx.float(), dim=1)
loc_idx_n = F.normalize(loc_idx.float(), dim=1)

def topk_cos(qfp, idx_n, k):
    sims = (idx_n @ F.normalize(qfp.float().unsqueeze(0), dim=1).T).squeeze(1)
    return set(sims.argsort(descending=True)[:k].tolist())

if __name__ == '__main__':
    rec_phi, rec_loc, rec_rand, jac_phi, jac_loc = [], [], [], [], []
    for qi, (qp, qt) in enumerate(zip(q_pos, q_txt)):
        gold = set(sorted(range(N_IDX), key=lambda j: jaccard(qt, idx_txt[j]), reverse=True)[:K])
        phi_hits = topk_cos(substrate_fingerprint(qt, 64), phi_idx_n, K)
        loc_hits = topk_cos(hist_fp(ids, qp, W, V), loc_idx_n, K)
        rec_phi.append(len(phi_hits & gold) / K)
        rec_loc.append(len(loc_hits & gold) / K)
        rec_rand.append(K / N_IDX)  # expected random recall
        # also: mean Jaccard of what each retriever returned (absolute relevance)
        jac_phi.append(statistics.mean(jaccard(qt, idx_txt[j]) for j in phi_hits))
        jac_loc.append(statistics.mean(jaccard(qt, idx_txt[j]) for j in loc_hits))
    print(f"[rel] retrieval-relevance benchmark (W={W}, index={N_IDX}, queries={N_Q}, k={K})", flush=True)
    print(f"[rel] gold = top-{K} by char-trigram Jaccard (true content similarity)", flush=True)
    print(f"[rel] recall@{K} vs gold:  φ-fp={statistics.mean(rec_phi):.3f}  locality-fp={statistics.mean(rec_loc):.3f}  random={statistics.mean(rec_rand):.3f}", flush=True)
    print(f"[rel] mean Jaccard of retrieved:  φ-fp={statistics.mean(jac_phi):.3f}  locality-fp={statistics.mean(jac_loc):.3f}", flush=True)
    verdict = "φ-retrieval ≈ RANDOM (no content relevance)" if statistics.mean(rec_phi) < 3*statistics.mean(rec_rand) else "φ-retrieval has some relevance"
    print(f"[rel] VERDICT: {verdict}; locality-fp {'RECOVERS relevance' if statistics.mean(rec_loc)>0.3 else 'modest'}", flush=True)
    print("[rel] REL-BENCH DONE", flush=True)
