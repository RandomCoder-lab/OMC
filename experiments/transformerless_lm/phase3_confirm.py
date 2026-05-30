"""Phase 3 confirm — settle the noisy n=2 frontier with n=5 seeds at ONE matched
reduction ratio. Reports mean±std so noise is visible. Answers cleanly:
  (a) does FFN address-sharing cost validity vs full FFN (and how much)?
  (b) is φ-bucketing different from modulo at matched param budget?
"""
import statistics, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import addressed_transformer as AT   # reuse the proven model + data + guarded metric

SEEDS = [0, 1, 2, 3, 4]
AT.STEPS = 800

def run_cfg(mode, nb, bm):
    ps, vs = [], []
    for s in SEEDS:
        fp, gv = AT.run(mode, nb, bm, s)
        ps.append(fp); vs.append(gv)
    return ps[0], vs

if __name__ == '__main__':
    t0 = time.time()
    print(f"[confirm] n={len(SEEDS)} seeds, seq_len={AT.SEQ_LEN}, steps={AT.STEPS}", flush=True)
    # matched-ish param budget: mod nb=16 (16 buckets) vs phi (≈13 tiers) vs normal
    cfgs = [("normal-FFN","normal",0,"mod"),
            ("mod-16buckets","addr",16,"mod"),
            ("phi-tiers","addr",16,"phi")]
    out = []
    for name,mode,nb,bm in cfgs:
        fp, vs = run_cfg(mode,nb,bm)
        m = statistics.mean(vs); sd = statistics.pstdev(vs)
        out.append((name,fp,m,sd,vs))
        print(f"[confirm] {name:14s} params={fp:<7} validity={m:.3f}±{sd:.3f}  vals={[round(v,2) for v in vs]}  ({time.time()-t0:.0f}s)", flush=True)
    print("\n[confirm] === VERDICT (n=5, mean±std) ===", flush=True)
    base = out[0][2]
    for name,fp,m,sd,_ in out:
        print(f"[confirm] {name:14s} params={fp:<7} validity={m:.3f}±{sd:.3f}  Δvs_full={m-base:+.3f}", flush=True)
    mod = next(o for o in out if o[0]=="mod-16buckets")
    phi = next(o for o in out if o[0]=="phi-tiers")
    diff = phi[2]-mod[2]; noise = (mod[3]+phi[3])
    print(f"[confirm] φ−mod = {diff:+.3f} (combined std {noise:.3f}) → "
          + ("φ DIFFERS from mod" if abs(diff)>noise else "φ ≈ mod (within noise)"), flush=True)
    print("[confirm] PHASE3-CONFIRM DONE", flush=True)
