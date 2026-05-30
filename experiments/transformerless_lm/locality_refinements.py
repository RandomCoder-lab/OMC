"""ADDR-5 (overlapping nodes) + ADDR-7 (richer locality fp) — refinements.

ADDR-7: does a char-BIGRAM histogram (more discriminative) beat char-UNIGRAM on
        retrieval relevance (recall@5 vs Jaccard gold)?
ADDR-5: do OVERLAPPING/dense nodes (stride < window) lift navigation recovery past
        the 0.88 from non-overlapping locality-fp nodes?
"""
import torch, torch.nn.functional as F, random, statistics
from pathlib import Path
from locality_fp import build_vocab, hist_fp, _topb

HERE = Path(__file__).parent
text = (HERE/'omc_corpus.txt').read_text(errors='replace')[:2_000_000]
stoi, V = build_vocab(text)
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# ── ADDR-7: bigram vs unigram on relevance ──
def trigrams(s): return set(s[i:i+3] for i in range(len(s)-2))
def jaccard(a,b):
    A,B=trigrams(a),trigrams(b); return len(A&B)/max(len(A|B),1)
W, N_IDX, N_Q, K = 89, 600, 100, 5
random.seed(0)
pos = random.sample(range(len(text)-W), N_IDX+N_Q); ipos,qpos = pos[:N_IDX],pos[N_IDX:]
itxt=[text[p:p+W] for p in ipos]; qtxt=[text[p:p+W] for p in qpos]
def recall(bigram):
    idxfp = F.normalize(torch.stack([hist_fp(ids,p,W,V,bigram) for p in ipos]).float(),dim=1)
    recs=[]
    for qp,qt in zip(qpos,qtxt):
        gold=set(sorted(range(N_IDX),key=lambda j:jaccard(qt,itxt[j]),reverse=True)[:K])
        q=F.normalize(hist_fp(ids,qp,W,V,bigram).float().unsqueeze(0),dim=1)
        hits=set((idxfp@q.T).squeeze(1).argsort(descending=True)[:K].tolist())
        recs.append(len(hits&gold)/K)
    return statistics.mean(recs)

# ── ADDR-5: overlapping nodes for navigation ──
def build_tree_stride(para_st, sent_st, word_st, PARA=233, SENT=89, WORD=34, max_paras=200, bigram=False):
    tree=[]
    for p in range(0, min(len(ids)-PARA, max_paras*PARA), para_st):
        node={'pos':p,'fp':hist_fp(ids,p,PARA,V,bigram),'sents':[]}
        for s in range(p, p+PARA-SENT+1, sent_st):
            node['sents'].append({'pos':s,'fp':hist_fp(ids,s,SENT,V,bigram),
                'words':[{'pos':w,'fp':hist_fp(ids,w,WORD,V,bigram)} for w in range(s,s+SENT-WORD+1,word_st)]})
        tree.append(node)
    return tree
def nav_recovery(tree, b=8, bigram=False):
    random.seed(1); hit=0; trials=60
    leaves=[(nd,s,w) for nd in tree for s in nd['sents'] for w in s['words']]
    for _ in range(trials):
        nd,s,wd=random.choice(leaves); q=wd['pos']
        paras=_topb(hist_fp(ids,q,233,V,bigram),tree,b)
        sp=[ss for p in paras for ss in p['sents']]; sents=_topb(hist_fp(ids,q,89,V,bigram),sp,b)
        wp=[ww for ss in sents for ww in ss['words']]; words=_topb(hist_fp(ids,q,34,V,bigram),wp,1)
        if words and words[0]['pos']==q: hit+=1
    return hit/trials

if __name__=='__main__':
    print("[ref] ADDR-7 retrieval relevance recall@5 (vs Jaccard gold):", flush=True)
    print(f"[ref]   char-unigram histogram = {recall(False):.3f}", flush=True)
    print(f"[ref]   char-bigram  histogram = {recall(True):.3f}", flush=True)
    print("[ref] ADDR-5 navigation exact-recovery (beam=8):", flush=True)
    t_base = build_tree_stride(233, 89, 34)   # non-overlapping (window=stride)
    print(f"[ref]   non-overlapping nodes  = {nav_recovery(t_base):.3f}", flush=True)
    t_ovl = build_tree_stride(89, 34, 34)     # overlapping paragraphs(stride89<233) & sentences(stride34<89)
    print(f"[ref]   overlapping nodes      = {nav_recovery(t_ovl):.3f}", flush=True)
    print("[ref] REFINE DONE", flush=True)
