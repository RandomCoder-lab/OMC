"""Push past the 29% coarse-to-fine ceiling with a LOCALITY-bearing fingerprint.

DIAGNOSIS (established): φ-hash fingerprints have NO content-locality — overlapping
windows are near-orthogonal (great for uniform bucketing χ²=17, useless for similarity
navigation; containing-region median rank ~80 → beam can't help).
FIX: a locality fingerprint (char-count histogram) where overlapping windows share
features → containing region median rank ~7 → beam navigation becomes viable.
Keep φ for uniform addressing/keys; use locality-fp for the navigation/similarity layer.
"""
import torch, torch.nn.functional as F, random, statistics
from pathlib import Path
HERE = Path(__file__).parent
PARA, SENT, WORD = 233, 89, 34

text = (HERE/'omc_corpus.txt').read_text(errors='replace')[:1_500_000]
chars = sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}; V=len(chars)
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)

def hist(start, L):
    h = torch.zeros(V); seg = ids[start:start+L]
    h.scatter_add_(0, seg, torch.ones(len(seg))); return h

# build tree with histogram fps at every node
tree=[]
for p in range(0, min(len(ids)-PARA, 400*PARA), PARA):
    node={'pos':p,'h':hist(p,PARA),'sents':[]}
    for s in range(p, p+PARA-SENT, SENT):
        node['sents'].append({'pos':s,'h':hist(s,SENT),
            'words':[{'pos':w,'h':hist(w,WORD)} for w in range(s,s+SENT-WORD,WORD)]})
    tree.append(node)

def nrm(t): return F.normalize(t.float().unsqueeze(0),dim=1)
def topb(qh, nodes, b):
    if not nodes: return []
    M=F.normalize(torch.stack([n['h'] for n in nodes]).float(),dim=1)
    sims=(M@nrm(qh).T).squeeze(1)
    return [nodes[i] for i in sims.argsort(descending=True)[:b].tolist()]

def navigate_hist(qpos, b):
    examined=0
    paras=topb(hist(qpos,PARA), tree, b); examined+=len(tree)
    spool=[s for p in paras for s in p['sents']]
    sents=topb(hist(qpos,SENT), spool, b); examined+=len(spool)
    wpool=[w for s in sents for w in s['words']]
    words=topb(hist(qpos,WORD), wpool, 1); examined+=len(wpool)
    if not words: return -1, examined, None
    # containing paragraph of the chosen leaf
    par0 = max(p['pos'] for p in paras if p['pos']<=words[0]['pos']) if paras else -1
    return words[0]['pos'], examined, paras

if __name__=='__main__':
    random.seed(0)
    qs=[]
    for _ in range(80):
        nd=random.choice(tree); s=random.choice(nd['sents']); wd=random.choice(s['words'])
        qs.append((wd['pos'], nd['pos']))
    print(f"[loc] locality (histogram) fp navigation vs φ-hash 29% ceiling:", flush=True)
    for b in (1,3,8,16):
        para_hit=0; exact=0; exam=[]
        for qpos,truepar in qs:
            fpos,ex,paras=navigate_hist(qpos,b); exam.append(ex)
            if paras and any(p['pos']==truepar for p in paras): para_hit+=1   # right paragraph in beam
            if fpos==qpos: exact+=1
        print(f"[loc] beam B={b:<2} para-in-beam={para_hit/80:.2f}  exact-leaf={exact/80:.2f}  examined~{statistics.mean(exam):.0f}", flush=True)
    print("[loc] LOC-NAV DONE", flush=True)
