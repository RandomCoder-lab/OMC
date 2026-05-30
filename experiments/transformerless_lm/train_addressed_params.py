"""Capability test: PARAMETERS AS ADDRESSES (user's sifted-out idea).

Claim: weights fetched/shared by address can REDUCE parameter count (many rows
collide onto one address → one shared prototype) while keeping quality — and let
you swap params by address. Tested on a clean neural n-gram char-LM whose hidden
layer is an AddressedLinear; sweep n_buckets (unique addresses) and measure the
params-vs-validity tradeoff, against a normal-nn.Linear baseline of identical arch.

Metric: corpus 4-gram gen-validity (same as the rest of the program).
"""
import statistics, math, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

from addressed_param_store import AddressedParamStore
from addressed_linear import AddressedLinear

HERE = Path(__file__).parent
CTX, D, HID = 8, 32, 256           # 8-char context, 32-dim embed, 256 hidden
STEPS, BATCH, LR, SEEDS = 1200, 64, 3e-3, [0, 1]
K = 4

corpus = (HERE/'omc_corpus.txt').read_text(errors='replace')
chars = sorted(set(corpus)); stoi={c:i for i,c in enumerate(chars)}; V=len(chars)
itos={i:c for c,i in stoi.items()}
ids = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)
n=len(ids); split=int(n*0.9); tr,va=ids[:split],ids[split:]
grams=set(corpus[i:i+K] for i in range(0,min(len(corpus),20_000_000)-K))
def validity(s): return 0.0 if len(s)<K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)

def batch(src,g):
    p=torch.randint(0,len(src)-CTX-1,(BATCH,),generator=g)
    xs=torch.stack([src[i:i+CTX] for i in p]); ys=src[p+CTX]
    return xs,ys

class NGramLM(nn.Module):
    """embed last CTX chars → concat → hidden → head. The hidden layer is either
    a normal Linear or an AddressedLinear (param-shared by address)."""
    def __init__(self, addressed=False, n_buckets=0):
        super().__init__()
        self.embed=nn.Embedding(V,D)
        self.addressed=addressed
        if addressed:
            self.store=AddressedParamStore()
            self.h=AddressedLinear(CTX*D, HID, name="ngram.hidden", store=self.store, n_buckets=n_buckets)
        else:
            self.h=nn.Linear(CTX*D, HID, bias=False)
        self.act=nn.GELU()
        self.head=nn.Linear(HID,V)
    def forward(self,x):                    # x:[B,CTX]
        e=self.embed(x).reshape(x.shape[0],-1)   # [B,CTX*D]
        return self.head(self.act(self.h(e)))    # [B,V]

@torch.no_grad()
def gen_validity(m,seed):
    g=torch.Generator().manual_seed(seed); m.eval(); outs=[]
    for pr in ["fn ","h x = ","fn add(a, b) {","return "]:
        ctx=[stoi.get(c,0) for c in pr][-CTX:]
        if len(ctx)<CTX: ctx=[0]*(CTX-len(ctx))+ctx
        s=""
        for _ in range(120):
            x=torch.tensor(ctx[-CTX:],dtype=torch.long).unsqueeze(0)
            p=F.softmax(m(x)[0],dim=-1); nx=int(torch.multinomial(p,1,generator=g))
            s+=itos.get(nx,'?'); ctx.append(nx)
        outs.append(validity(pr+s))
    m.train(); return statistics.mean(outs)

def run(addressed,n_buckets,seed):
    torch.manual_seed(seed); g=torch.Generator(); g.manual_seed(seed+3)
    m=NGramLM(addressed=addressed,n_buckets=n_buckets)
    hid_params=sum(p.numel() for p in m.h.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=LR); m.train()
    for _ in range(STEPS):
        x,y=batch(tr,g); loss=F.cross_entropy(m(x),y)
        opt.zero_grad(); loss.backward(); opt.step()
    return hid_params, gen_validity(m,seed)

if __name__=='__main__':
    t0=time.time()
    print(f"[addr] neural n-gram LM: CTX={CTX} D={D} HID={HID} V={V}  normal hidden params={CTX*D*HID}", flush=True)
    configs=[("normal-Linear",False,0),
             ("addressed nb=full",True,0),
             ("addressed nb=64",True,64),
             ("addressed nb=16",True,16),
             ("addressed nb=4",True,4)]
    rows=[]
    for name,addr,nb in configs:
        ps=[]; vs=[]
        for seed in SEEDS:
            hp,gv=run(addr,nb,seed); ps.append(hp); vs.append(gv)
        hp=ps[0]; gv=statistics.mean(vs)
        rows.append((name,hp,gv))
        print(f"[addr] {name:20s} hidden_params={hp:<7} gen_validity={gv:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    base=rows[0][1]
    print("\n[addr] === PARAMETERS-AS-ADDRESSES capability ===")
    for name,hp,gv in rows:
        print(f"[addr] {name:20s} params={hp:<7} ({base/hp:.1f}x reduction)  validity={gv:.3f}")
    print("[addr] VERDICT: addressing reduces params; the question is validity retention vs reduction.")
