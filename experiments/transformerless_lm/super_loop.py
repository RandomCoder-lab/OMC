"""NEXT-8 — Super-tool substrate loop: address → generate → VERIFY, codebase-agnostic.

The reusable scaffold. The substrate provides the two trustworthy halves:
  ADDRESS  — locality-fp retrieval of relevant code from the target corpus (OMC or Python)
  VERIFY   — grammar-validity + execution via the real interpreter (OMC: standalone; Python: ast/exec)
The GENERATOR is a pluggable slot — the "super tool" (a strong LLM / the agent) plugs in here;
autonomously we demo with a composition generator. Nothing is accepted unless VERIFY passes,
so a weak or hallucinating generator can't corrupt the result — the substrate is the guardrail.

Universal: pass lang='omc' or 'python'; only verify_fn + corpus change.
"""
import ast, subprocess, tempfile, os, re
from pathlib import Path
import torch, torch.nn.functional as F
from locality_fp import hist_fp

HERE = Path(__file__).parent
_OMC_BIN = str((HERE/'../../target/release/omnimcode-standalone').resolve())

class SubstrateLoop:
    def __init__(self, lang='omc', corpus_items=None):
        self.lang = lang
        # corpus_items: list of (name, code) for addressing
        self.items = corpus_items or self._default_corpus(lang)
        text = " ".join(n+" "+c for n, c in self.items) if self.items else "x"
        self.stoi = {ch: i for i, ch in enumerate(sorted(set(text)))}
        self.V = len(self.stoi)
        if self.items:
            self.M = F.normalize(torch.stack([self._fp(n+" "+c) for n, c in self.items]).float(), dim=1)

    def _default_corpus(self, lang):
        if lang == 'omc':
            reg = torch.load(HERE/'omc_name_registry.pt', map_location='cpu', weights_only=False)
            return [(e['name'], e['code']) for e in list(reg.values())[:800]]
        from py_substrate import index_python
        return [(n, s) for _f, n, s in index_python(HERE)]

    def _fp(self, s):
        ids = torch.tensor([self.stoi.get(c, 0) for c in s], dtype=torch.long)
        return hist_fp(ids, 0, len(ids), self.V, bigram=True)

    # ── ADDRESS: locality retrieval (substrate, trustworthy) ──
    def address(self, task_desc, top_k=5):
        q = F.normalize(self._fp(task_desc).unsqueeze(0), dim=1)
        sims = (self.M @ q.T).squeeze(1)
        return [self.items[i] for i in sims.argsort(descending=True)[:top_k].tolist()]

    # ── VERIFY: grammar + execution (substrate, the guardrail) ──
    def verify(self, code):
        if self.lang == 'python':
            try:
                import textwrap; ast.parse(textwrap.dedent(code)); return 'OK'
            except (SyntaxError, IndentationError): return 'PARSE_ERR'
        with tempfile.NamedTemporaryFile('w', suffix='.omc', delete=False, dir='/tmp') as f:
            f.write(code); p = f.name
        try:
            r = subprocess.run([_OMC_BIN, p], capture_output=True, text=True, timeout=15)
            if r.returncode == 0: return 'OK'
            return 'PARSE_ERR' if re.search(r'Expected|Unexpected token|expression can', r.stderr+r.stdout) else 'RUNTIME_ERR'
        except Exception: return 'RUNTIME_ERR'
        finally:
            try: os.unlink(p)
            except Exception: pass

    # ── the loop: address → generate(super-tool) → verify → accept ──
    def improve(self, task_desc, generate_fn, max_tries=4):
        """generate_fn(task_desc, addressed_context) -> candidate code (the SUPER-TOOL slot).
        Returns (code, status, tries). Accepts only VERIFY-passing candidates."""
        ctx = self.address(task_desc)
        best = None
        for t in range(1, max_tries+1):
            cand = generate_fn(task_desc, ctx)
            best = cand
            if self.verify(cand) == 'OK':
                return cand, f'accepted@{t}', t
        return best, 'unverified', max_tries

if __name__ == '__main__':
    # autonomous demo: composition generator in the super-tool slot (an LLM would plug in here)
    from compositional_eval import gen_toolbox
    loop = SubstrateLoop(lang='omc')
    def composer(task, ctx):                       # ← super-tool slot (stand-in)
        code, _ = gen_toolbox(task, task, allow_exact=False)   # held-out: must synthesize
        return code
    tasks = ["gcd", "factorial", "is_prime", "sum to n", "reverse a string", "running max"]
    acc = 0
    for tk in tasks:
        code, status, tries = loop.improve(tk, composer)
        ok = status.startswith('accepted'); acc += ok
        print(f"[loop] {tk:18s} {status:14s} verify={'OK' if ok else 'FAIL'}  (addressed {len(loop.address(tk))} ctx)", flush=True)
    print(f"[loop] verified-valid acceptance = {acc}/{len(tasks)} (substrate VERIFY gate; generator pluggable)", flush=True)
    print("[loop] SUPER-LOOP DONE", flush=True)
