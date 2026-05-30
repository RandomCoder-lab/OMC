"""Phase 3 — PARAMETERS-AS-ADDRESSES in a REAL attention LM (not the n-gram toy).

The Phase-2c win (4× param reduction at no cost) was on an attention-free n-gram LM.
This scales it to a genuine char Transformer: the FFN linears (the bulk of transformer
params) become AddressedLinear with address-sharing. Tests:
  (1) does the param-reduction win HOLD on a real generative LM with attention?
  (2) does φ-structured bucketing (substrate-shaped sharing) beat naive modulo?
Metric: degeneracy-GUARDED corpus 4-gram validity. ~matched arch, from scratch.
"""
import sys, math, time, statistics
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

from addressed_param_store import AddressedParamStore
from addressed_linear import AddressedLinear

HERE = Path(__file__).parent
SEQ_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 128
STEPS   = int(sys.argv[2]) if len(sys.argv) > 2 else 800
D, L, H, FF = 64, 4, 4, 256          # FF hidden = 4*D
BATCH, LR, K = 24, 3e-3, 4
SEEDS = [0, 1]

corpus = (HERE/'omc_corpus.txt').read_text(errors='replace')
chars = sorted(set(corpus)); stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}; V=len(chars)
ids = torch.tensor([stoi[c] for c in corpus], dtype=torch.long); n=len(ids); sp=int(n*0.9); tr,va=ids[:sp],ids[sp:]
grams=set(corpus[i:i+K] for i in range(0,min(len(corpus),20_000_000)-K))
def distinct(s): return len(set(s))/max(len(s),1)
def maxrun(s):
    mr=cur=1
    for i in range(1,len(s)):
        cur=cur+1 if s[i]==s[i-1] else 1; mr=max(mr,cur)
    return mr
def rawval(s): return 0.0 if len(s)<K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)
def gval(s): return 0.0 if (distinct(s)<0.15 or maxrun(s)>40) else rawval(s)

def batch(src,g):
    p=torch.randint(0,len(src)-SEQ_LEN-1,(BATCH,),generator=g)
    return torch.stack([src[i:i+SEQ_LEN] for i in p]), torch.stack([src[i+1:i+SEQ_LEN+1] for i in p])

class FFN(nn.Module):
    def __init__(self, mode, nb, bm, tag, store):
        super().__init__()
        if mode == "normal":
            self.w1=nn.Linear(D,FF); self.w2=nn.Linear(FF,D); self.addr=False
        else:
            self.w1=AddressedLinear(D,FF,name=f"{tag}.w1",store=store,bias=True,n_buckets=nb,bucket_mode=bm)
            self.w2=AddressedLinear(FF,D,name=f"{tag}.w2",store=store,bias=True,n_buckets=max(1,nb//4),bucket_mode=bm)
            self.addr=True
    def forward(self,x): return self.w2(F.gelu(self.w1(x)))

class Block(nn.Module):
    def __init__(self,i,mode,nb,bm,store):
        super().__init__()
        self.ln1=nn.LayerNorm(D); self.attn=nn.MultiheadAttention(D,H,batch_first=True)
        self.ln2=nn.LayerNorm(D); self.ff=FFN(mode,nb,bm,f"blk{i}",store)
    def forward(self,x,m):
        y=self.ln1(x); a,_=self.attn(y,y,y,attn_mask=m,need_weights=False); x=x+a
        return x+self.ff(self.ln2(x))

class AddrTx(nn.Module):
    def __init__(self,mode="normal",nb=0,bm="mod"):
        super().__init__()
        self.store=AddressedParamStore()
        self.tok=nn.Embedding(V,D); self.pos=nn.Embedding(SEQ_LEN,D)
        self.blocks=nn.ModuleList([Block(i,mode,nb,bm,self.store) for i in range(L)])
        self.lnf=nn.LayerNorm(D); self.head=nn.Linear(D,V); self.head.weight=self.tok.weight
    def forward(self,x):
        T=x.shape[1]; h=self.tok(x)+self.pos(torch.arange(T))
        m=torch.triu(torch.full((T,T),float('-inf')),1)
        for b in self.blocks: h=b(h,m)
        return self.head(self.lnf(h))

@torch.no_grad()
def gen_validity(m,seed):
    g=torch.Generator().manual_seed(seed); m.eval(); vs=[]
    for pr in ["fn ","h x = ","fn add(a, b) {","return "]:
        ctx=[stoi.get(c,0) for c in pr][-SEQ_LEN:]
        s=""
        for _ in range(140):
            x=torch.tensor(ctx[-SEQ_LEN:],dtype=torch.long).unsqueeze(0)
            p=F.softmax(m(x)[0,-1],dim=-1); nx=int(torch.multinomial(p,1,generator=g)); s+=itos.get(nx,'?'); ctx.append(nx)
        vs.append(gval(s))
    m.train(); return statistics.mean(vs)

def run(mode,nb,bm,seed):
    torch.manual_seed(seed); g=torch.Generator(); g.manual_seed(seed+3)
    m=AddrTx(mode,nb,bm)
    ffp=sum(p.numel() for blk in m.blocks for p in blk.ff.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=LR); m.train()
    for _ in range(STEPS):
        x,y=batch(tr,g); F.cross_entropy(m(x).reshape(-1,V),y.reshape(-1)).backward(); opt.step(); opt.zero_grad()
    return ffp, gen_validity(m,seed)

if __name__=='__main__':
    t0=time.time()
    print(f"[Phase3] addressed Transformer: seq_len={SEQ_LEN} D={D} L={L} FF={FF} steps={STEPS} V={V}", flush=True)
    configs=[("normal-FFN","normal",0,"mod"),
             ("addressed-mod-4x","addr",FF//4,"mod"),
             ("addressed-phi-4x","addr",FF//4,"phi"),
             ("addressed-mod-8x","addr",FF//8,"mod"),
             ("addressed-phi-8x","addr",FF//8,"phi")]
    rows=[]
    for name,mode,nb,bm in configs:
        ps=[]; vs=[]
        for seed in SEEDS:
            fp,gv=run(mode,nb,bm,seed); ps.append(fp); vs.append(gv)
        rows.append((name,ps[0],statistics.mean(vs)))
        print(f"[Phase3] {name:18s} ffn_params={ps[0]:<7} guarded_validity={statistics.mean(vs):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    base=rows[0][1]
    print("\n[Phase3] === PARAMS-AS-ADDRESSES on a real Transformer ===", flush=True)
    for name,fp,gv in rows:
        print(f"[Phase3] {name:18s} ffn_params={fp:<7} ({base/fp:.1f}x)  validity={gv:.3f}", flush=True)
