"""R1 — Self-Witness REVISED: real quality-selected bloom centroid (not the
ungated EMA proxy that 'failed' as B.1).

Recipe:
  1. Load a trained model; generate samples; keep the highest 4-gram-validity ones.
  2. Bloom centroid = Fibonacci-position-weighted mean of their hidden states
     (clause-initial positions weighted more — compute_bloom_embedding's recipe).
  3. A/B fresh training: control (no Self-Witness) vs treatment (Self-Witness with
     the FROZEN real centroid, sw_fixed=True). Same seed/budget; only SW differs.

Tests the user's hypothesis: B.1 wasn't a failure of the idea, just of the proxy.
"""
import statistics, math
from pathlib import Path
import torch, torch.nn.functional as F

from train_substrate_attention import FibRecLMSubsim
from train_address_navigator import FibRecLMNavigator
from omc_assistant import _load_lm
from grimoire_spells import navigated_generate

HERE = Path(__file__).parent
PHI = (1+5**0.5)/2
K = 4
STEPS, BATCH, LR, SEEDS = 500, 16, 3e-4, [0, 1]

# corpus + validity
corpus = (HERE/'omc_corpus.txt').read_text(errors='replace')
grams = set(corpus[i:i+K] for i in range(0, min(len(corpus),20_000_000)-K))
def validity(s): return 0.0 if len(s)<K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)

# ── 1. compute the REAL bloom centroid from a trained model ──
print("[R1] computing real bloom centroid from bloom_256_curriculum_model.pt ...", flush=True)
tm, tstoi, titos, tsl = _load_lm(HERE/'bloom_256_curriculum_model.pt')
fnpos = [i for i in range(len(corpus)-1) if corpus.startswith('fn ', i)][:40]
samples = []
for j,p in enumerate(fnpos):
    pr = corpus[p:p+40]
    txt = navigated_generate(tm, tstoi, titos, tsl, pr, 100, nav=False, seed=2000+j)
    samples.append((validity(pr+txt), pr+txt))
samples.sort(reverse=True)
keep = samples[:max(1,len(samples)//3)]
print(f"[R1] kept {len(keep)}/{len(samples)} samples, mean validity {statistics.mean(v for v,_ in keep):.3f}", flush=True)
# Fibonacci-position-weighted mean hidden over kept samples
FIB=[1,2,3,5,8,13,21,34,55,89,144,233]
def fibbin(p):
    for i,f in enumerate(FIB):
        if p< f: return i
    return len(FIB)
d = tm.base.embed.embedding_dim
acc = torch.zeros(d); wsum=0.0
tm.eval()
with torch.no_grad():
    for _,txt in keep:
        ids = torch.tensor([tstoi.get(c,0) for c in txt[:tsl]], dtype=torch.long).unsqueeze(0)
        emb = tm.base.embed(ids)[0]      # [L,d]  embedding-space (derivation's b space)
        for pos in range(emb.shape[0]):
            w = PHI**(-fibbin(pos))
            acc += w*emb[pos]; wsum += w
centroid = (acc/wsum).detach()
torch.save(centroid, HERE/'bloom_centroid_real.pt')
print(f"[R1] real bloom centroid saved, norm={centroid.norm():.3f}", flush=True)

# ── 2. A/B fresh training ──
ck = torch.load(HERE/'bloom_256_curriculum_model.pt', map_location='cpu', weights_only=False)
vocab=ck['vocab']; V=len(vocab); stoi={c:i for i,c in enumerate(vocab)}; itos={i:c for i,c in enumerate(vocab)}
d_model=ck.get('d_model',64); n_blocks=ck.get('n_blocks',4); seq_len=ck.get('seq_len',256); K_init=ck.get('K_init',89)
ids=torch.tensor([stoi.get(c,0) for c in corpus],dtype=torch.long); split=int(len(ids)*0.9)
tr,va=ids[:split],ids[split:]
def batch(src,g):
    p=torch.randint(0,len(src)-seq_len-1,(BATCH,),generator=g)
    return torch.stack([src[i:i+seq_len] for i in p]), torch.stack([src[i+1:i+seq_len+1] for i in p])
@torch.no_grad()
def vloss(m,g):
    m.eval(); t=0.0
    for _ in range(6):
        x,y=batch(va,g); t+=F.cross_entropy(m(x)[0].reshape(-1,V),y.reshape(-1)).item()
    m.train(); return t/6
@torch.no_grad()
def gval(m,seed):
    return statistics.mean(validity(navigated_generate(m,stoi,itos,seq_len,pr,100,nav=False,seed=seed*9+i))
                           for i,pr in enumerate(["fn ","h x = ","fn add(a, b) {","return "]))
def run(sw,seed):
    torch.manual_seed(seed); g=torch.Generator(); g.manual_seed(seed+5)
    base=FibRecLMSubsim(vocab_size=V,d_model=d_model,n_blocks=n_blocks,seq_len=seq_len,K=K_init,mode="cross",K_sig=32)
    if sw:
        base.self_witness=True; base.sw_fixed=True; base._sw_init=True
        base.bloom_centroid.copy_(centroid)
    m=FibRecLMNavigator(base,d_model)
    opt=torch.optim.AdamW(m.parameters(),lr=LR); m.train()
    for _ in range(STEPS):
        x,y=batch(tr,g); loss=F.cross_entropy(m(x)[0].reshape(-1,V),y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    eg=torch.Generator(); eg.manual_seed(999)
    return vloss(m,eg), gval(m,seed)

if __name__=='__main__':
    res={"control":{"v":[],"g":[]},"witness":{"v":[],"g":[]}}
    for seed in SEEDS:
        for cond,sw in [("control",False),("witness",True)]:
            v,gv=run(sw,seed); res[cond]["v"].append(v); res[cond]["g"].append(gv)
            print(f"[R1] seed={seed} {cond:8s} val={v:.4f} gen_validity={gv:.3f}", flush=True)
    cv,wv=statistics.mean(res["control"]["v"]),statistics.mean(res["witness"]["v"])
    cg,wg=statistics.mean(res["control"]["g"]),statistics.mean(res["witness"]["g"])
    print(f"\n[R1] === REVISED Self-Witness (real bloom centroid) ===")
    print(f"[R1] control  val={cv:.4f} gen_validity={cg:.3f}")
    print(f"[R1] witness  val={wv:.4f} gen_validity={wg:.3f}")
    print(f"[R1] Δval={wv-cv:+.4f}(lower=better)  Δgen_validity={wg-cg:+.4f}(higher=better)")
    print("[R1] VERDICT: "+("REAL CENTROID HELPS — B.1 was a proxy artifact" if (wg>cg and wv<=cv+0.01) else "still no help even with real centroid"))
