"""D.1 A/B: Bloom self-distillation — does feeding accepted high-quality
self-generations back into training improve the model?

Cycle (per seed):
  1. Generate samples from M0; score by corpus 4-gram validity; KEEP the top
     fraction → the "bloom" (the accept gate, à la compute_bloom_embedding's
     quality weighting, here using the C.3 validity metric).
  2. CONTROL  : fine-tune a fresh copy of M0 on corpus windows only.
     TREATMENT : fine-tune a fresh copy on corpus windows + bloom samples.
     (same seed, steps, optimizer — only the data differs.)
  3. Measure val char-loss AND generation validity for each.

Prediction: if the bloom carries real signal, TREATMENT ≥ CONTROL on generation
validity. If self-distillation on a 0.55-validity generator just amplifies its own
errors (the Φ_R / function-word-loop failure mode), TREATMENT ≤ CONTROL — also a
valid, reportable result.
"""
import statistics, random, copy
from pathlib import Path
import torch, torch.nn.functional as F

from train_substrate_attention import FibRecLMSubsim
from train_address_navigator import FibRecLMNavigator
from grimoire_spells import navigated_generate

HERE = Path(__file__).parent
CKPT = HERE / 'bloom_256_curriculum_model.pt'
K = 4
STEPS, BATCH, LR, SEEDS = 250, 16, 3e-5, [0, 1]
N_BLOOM_GEN, BLOOM_KEEP = 60, 0.4         # generate 60, keep top 40% by validity
GEN_LEN = 100

# ── corpus + validity reference ───────────────────────────────────────────────
fndefs = (HERE / 'omc_fndefs.txt').read_text(errors='replace')
corpus_full = (HERE / 'omc_corpus.txt').read_text(errors='replace')
grams = set(corpus_full[i:i+K] for i in range(0, min(len(corpus_full), 20_000_000) - K))
def validity(s):
    return 0.0 if len(s) < K else sum(1 for i in range(len(s)-K) if s[i:i+K] in grams)/(len(s)-K)

# ── load checkpoint metadata + a fresh-model factory ──────────────────────────
ckpt = torch.load(str(CKPT), map_location='cpu', weights_only=False)
vocab = ckpt['vocab']; V = len(vocab)
stoi = {c: i for i, c in enumerate(vocab)}; itos = {i: c for i, c in enumerate(vocab)}
d_model = ckpt.get('d_model', 64); n_blocks = ckpt.get('n_blocks', 4)
seq_len = ckpt.get('seq_len', 256); K_init = ckpt.get('K_init', 89)
base_state = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('intent_head.')}

def fresh_model():
    base = FibRecLMSubsim(vocab_size=V, d_model=d_model, n_blocks=n_blocks,
                          seq_len=seq_len, K=K_init, mode="cross", K_sig=32)
    m = FibRecLMNavigator(base, d_model)
    m.load_state_dict(base_state, strict=False)
    return m

ids = torch.tensor([stoi.get(c, 0) for c in fndefs], dtype=torch.long)
split = int(len(ids) * 0.9); train_ids, val_ids = ids[:split], ids[split:]

def win_batch(src, gen, extra=None, p_extra=0.0):
    xs, ys = [], []
    for _ in range(BATCH):
        if extra is not None and random.random() < p_extra:
            e = extra[random.randrange(len(extra))]
            if len(e) >= seq_len + 1:
                p = random.randrange(len(e) - seq_len); seq = e[p:p+seq_len+1]
            else:
                seq = torch.cat([e, torch.zeros(seq_len+1-len(e), dtype=torch.long)])
        else:
            p = torch.randint(0, len(src)-seq_len-1, (1,), generator=gen).item()
            seq = src[p:p+seq_len+1]
        xs.append(seq[:seq_len]); ys.append(seq[1:seq_len+1])
    return torch.stack(xs), torch.stack(ys)

@torch.no_grad()
def val_loss(model, gen):
    model.eval(); t = 0.0
    for _ in range(8):
        x, y = win_batch(val_ids, gen)
        t += F.cross_entropy(model(x)[0].reshape(-1, V), y.reshape(-1)).item()
    model.train(); return t / 8

@torch.no_grad()
def gen_validity(model, seed):
    vs = []
    for i, pr in enumerate(["fn ", "h x = ", "fn add(a, b) {", "return "]):
        t = navigated_generate(model, stoi, itos, seq_len, pr, GEN_LEN, nav=False, seed=seed*10+i)
        vs.append(validity(t))
    return statistics.mean(vs)

def finetune(model, seed, bloom=None):
    g = torch.Generator(); g.manual_seed(seed+7)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    p_extra = 0.0 if bloom is None else 0.25
    extra = None if bloom is None else [torch.tensor([stoi.get(c,0) for c in b], dtype=torch.long) for b in bloom]
    model.train()
    for _ in range(STEPS):
        x, y = win_batch(train_ids, g, extra=extra, p_extra=p_extra)
        loss = F.cross_entropy(model(x)[0].reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

if __name__ == '__main__':
    import time; t0 = time.time()
    print(f"loaded {CKPT.name}: V={V} seq_len={seq_len}  distinct {K}-grams={len(grams):,}", flush=True)
    # ── build the bloom from M0's accepted generations ──
    m0 = fresh_model()
    samples = []
    fn_pos = [i for i in range(len(fndefs)-1) if fndefs.startswith('fn ', i)]
    for j in range(N_BLOOM_GEN):
        pr = fndefs[fn_pos[j % len(fn_pos)]:][:40]
        txt = navigated_generate(m0, stoi, itos, seq_len, pr, GEN_LEN, nav=False, seed=1000+j)
        samples.append((validity(pr+txt), pr+txt))
    samples.sort(reverse=True)
    keep = max(1, int(len(samples)*BLOOM_KEEP))
    bloom = [s for _, s in samples[:keep]]
    acc_v = statistics.mean(v for v, _ in samples[:keep])
    rej_v = statistics.mean(v for v, _ in samples[keep:]) if keep < len(samples) else 0
    print(f"bloom: kept {keep}/{len(samples)}  accepted_validity={acc_v:.3f}  rejected={rej_v:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    res = {"control": {"val": [], "gv": []}, "treatment": {"val": [], "gv": []}}
    eg = torch.Generator(); eg.manual_seed(999)
    for seed in SEEDS:
        for cond, bl in [("control", None), ("treatment", bloom)]:
            torch.manual_seed(seed)
            m = fresh_model()
            m = finetune(m, seed, bloom=bl)
            v = val_loss(m, eg); gv = gen_validity(m, seed)
            res[cond]["val"].append(v); res[cond]["gv"].append(gv)
            print(f"seed={seed} {cond:9s} val={v:.4f} gen_validity={gv:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== D.1 Bloom self-distillation A/B ===")
    for cond in ("control", "treatment"):
        print(f"{cond:9s} val={statistics.mean(res[cond]['val']):.4f}  gen_validity={statistics.mean(res[cond]['gv']):.3f}")
    dv = statistics.mean(res['treatment']['val']) - statistics.mean(res['control']['val'])
    dg = statistics.mean(res['treatment']['gv']) - statistics.mean(res['control']['gv'])
    print(f"Δ val (T−C) = {dv:+.4f} (lower=better)   Δ gen_validity (T−C) = {dg:+.4f} (higher=better)")
    print("BLOOM HELPS" if (dg > 0 and dv <= 0.01) else "BLOOM NEUTRAL/HURTS — self-distillation on weak generator")
