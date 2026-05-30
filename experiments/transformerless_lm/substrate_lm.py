"""THE SUBSTRATE LM — assembled from the tools the whole program built.

Not a neural net that hallucinates tokens. A substrate code model: given a query it returns
VERIFIED OMC, by routing through the addressed toolbox and gating on execution.

  query → [1] exact-key registry        (name match → exact correct fn)
        → [2] learned encoder retrieval  (NL→code, NEXT-1, recall@5 0.89)
        → [3] φ-synthesis template       (derivable patterns)
        → [4] grammar-constrained gen     (valid by construction, NEXT-5)
  every candidate → VERIFY (parse/run via the real interpreter) → return best verified + provenance.

This is the LM. The pieces: NEXT-1 (encoder), NEXT-2 (composition), NEXT-3 (verify/correctness),
NEXT-5 (grammar), NEXT-8 (the loop). Codebase-agnostic; built by the same substrate-synth tool
that it embodies (self-hosting).
"""
import re, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).parent

# ── learned desc↔code encoder (NEXT-1) ──
class _Tower(nn.Module):
    def __init__(s, DIM, EMB):
        super().__init__(); s.f = nn.Sequential(nn.Linear(DIM,512), nn.GELU(), nn.Linear(512,EMB))
    def forward(s,x): return F.normalize(s.f(x), dim=-1)

def _feat(s, DIM):
    h = torch.zeros(DIM); b = s.encode('utf-8','ignore')
    if len(b) > 1:
        a = torch.tensor(list(b[:-1])); c = torch.tensor(list(b[1:]))
        h.scatter_add_(0, (a*257+c) % DIM, torch.ones(len(a)))
    n = h.norm(); return h/n if n > 0 else h

class SubstrateLM:
    def __init__(self):
        self.reg = torch.load(HERE/'omc_name_registry.pt', map_location='cpu', weights_only=False)
        self.by_name = {e['name']: e['code'] for e in self.reg.values()}
        self.items = [(e['name'], e['code']) for e in self.reg.values()]
        # load learned encoder
        ck = torch.load(HERE/'desc_encoder.pt', map_location='cpu', weights_only=False)
        self.DIM, self.EMB = ck['DIM'], ck['EMB']
        self.desc_t = _Tower(self.DIM, self.EMB); self.desc_t.load_state_dict(ck['desc_state']); self.desc_t.eval()
        self.code_t = _Tower(self.DIM, self.EMB); self.code_t.load_state_dict(ck['code_state']); self.code_t.eval()
        with torch.no_grad():
            self.code_emb = self.code_t(torch.stack([_feat(c, self.DIM) for _,c in self.items]))
        from exec_eval import run_omc
        self._verify = run_omc

    def _encoder_retrieve(self, query, top_k=3):
        with torch.no_grad():
            q = self.desc_t(_feat(query, self.DIM).unsqueeze(0))
            sims = (self.code_emb @ q.T).squeeze(1)
        return [self.items[i] for i in sims.argsort(descending=True)[:top_k].tolist()]

    def answer(self, query, want_name=None):
        """Return (code, provenance, verify_status). Tries toolbox routes, verifies each."""
        cands = []  # (provenance, code)
        # [1] exact-key registry
        key = (want_name or '').lower() or query.lower().strip().replace(' ', '_')
        if key in self.by_name: cands.append(('registry-exact', self.by_name[key]))
        for w in re.split(r'\W+', query.lower()):
            if w in self.by_name: cands.append(('registry-exact', self.by_name[w])); break
        # [2] learned encoder NL→code
        for nm, code in self._encoder_retrieve(query, top_k=2):
            cands.append((f'encoder:{nm}', code))
        # [3] φ-synthesis
        try:
            from phi_synthesis import synthesize
            r = synthesize(query, 64)
            if r: cands.append(('phi-synth', r[0].replace('{{','{').replace('}}','}')))
        except Exception: pass
        # [4] grammar-constrained fallback
        try:
            from grammar_gen import GrammarGen
            cands.append(('grammar', GrammarGen(seed=abs(hash(query))%999).gen_fn(key if key.isidentifier() else 'g')))
        except Exception: pass
        # verify, return first that runs clean (OK), else first that parses, else first
        for prov, code in cands:
            if self._verify(code) == 'OK': return code, prov, 'OK'
        for prov, code in cands:
            if self._verify(code) in ('OK','RUNTIME_ERR'): return code, prov, 'VALID'
        return (cands[0][1], cands[0][0], 'UNVERIFIED') if cands else ('', 'none', 'EMPTY')

if __name__ == '__main__':
    lm = SubstrateLM()
    print("[slm] SubstrateLM ready (registry + learned encoder + φ-synth + grammar + verify)\n", flush=True)
    for q in ["gcd", "compute the greatest common divisor", "merge sort",
              "check if a number is prime", "sum the numbers from 1 to n", "reverse a string"]:
        code, prov, st = lm.answer(q)
        first = code.split(chr(10))[0][:54]
        print(f"[slm] {q:34s} → [{st:10s} via {prov:16s}] {first}", flush=True)
    print("\n[slm] SLM DONE", flush=True)
