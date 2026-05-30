"""B.1 A/B: Self-Witness bloom-heartbeat vs baseline, small-scale, multi-seed.

Isolates the architectural effect: same model/optimizer/data/seed, only the
self_witness flag differs. Reports best val char-loss per (seed, condition) and
the mean Δ. Pre-registered prediction: witnessed ≤ baseline (amplifying on-manifold
positions should help) — but it must be reported even if it loses (project ethic).
"""
import sys, time, math, statistics
from pathlib import Path
import torch, torch.nn.functional as F

from train_substrate_attention import FibRecLMSubsim
from train_address_navigator import FibRecLMNavigator

HERE = Path(__file__).parent
D_MODEL, N_BLOCKS, SEQ_LEN, K = 64, 2, 89, 89
STEPS, BATCH, LR, EVAL_EVERY = 500, 16, 5e-4, 100
SEEDS = [0, 1, 2]

text = (HERE / 'omc_corpus.txt').read_text(errors='replace')[:6_000_000]
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
V = len(chars)
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = len(ids); split = int(n * 0.9)
train_ids, val_ids = ids[:split], ids[split:]
print(f"corpus={n:,} vocab={V} train={len(train_ids):,} val={len(val_ids):,}", flush=True)

def batch(src, gen):
    pos = torch.randint(0, len(src) - SEQ_LEN - 1, (BATCH,), generator=gen)
    x = torch.stack([src[p:p+SEQ_LEN] for p in pos])
    y = torch.stack([src[p+1:p+SEQ_LEN+1] for p in pos])
    return x, y

@torch.no_grad()
def evaluate(model, gen):
    model.eval(); tot = 0.0
    for _ in range(10):
        x, y = batch(val_ids, gen)
        logits, _ = model(x)
        tot += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item()
    model.train(); return tot / 10

def run(self_witness: bool, seed: int) -> float:
    torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed + 100)
    eg = torch.Generator(); eg.manual_seed(999)
    base = FibRecLMSubsim(vocab_size=V, d_model=D_MODEL, n_blocks=N_BLOCKS,
                          seq_len=SEQ_LEN, K=K, mode="cross", K_sig=32)
    base.self_witness = self_witness
    model = FibRecLMNavigator(base, D_MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best = float('inf')
    for step in range(STEPS + 1):
        x, y = batch(train_ids, g)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            v = evaluate(model, eg); best = min(best, v)
    return best

if __name__ == '__main__':
    t0 = time.time()
    results = {"baseline": {}, "witnessed": {}}
    for seed in SEEDS:
        for cond, sw in [("baseline", False), ("witnessed", True)]:
            v = run(sw, seed)
            results[cond][seed] = v
            print(f"seed={seed} {cond:9s} best_val={v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== Self-Witness A/B ===")
    b = [results["baseline"][s] for s in SEEDS]
    w = [results["witnessed"][s] for s in SEEDS]
    print(f"baseline  mean={statistics.mean(b):.4f}  {[f'{x:.4f}' for x in b]}")
    print(f"witnessed mean={statistics.mean(w):.4f}  {[f'{x:.4f}' for x in w]}")
    d = statistics.mean(w) - statistics.mean(b)
    wins = sum(1 for s in SEEDS if results["witnessed"][s] < results["baseline"][s])
    print(f"Δ(witnessed−baseline) = {d:+.4f}  ({100*d/statistics.mean(b):+.2f}%)  wins={wins}/{len(SEEDS)}")
    print("PROVEN" if d < 0 and wins >= 2 else "FALSIFIED / inconclusive")
