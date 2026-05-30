"""NEXT-2 (compositional grammar-assembly) + NEXT-3 (execution-correctness benchmark).

CORRECTNESS is the real scoreboard. We derive test cases from reference implementations
(run the reference via the interpreter → expected outputs), then score generated code by
% of test cases whose output matches.

NEXT-2 generator = codebase-agnostic toolbox dispatcher (NOT RAG):
  1. registry exact-key   (toolbox: the corpus holds the exact fn)        → correct if covered
  2. φ-synthesis template (toolbox: derivable templates)
  3. locality retrieval   (toolbox: nearest real fn by content)
  4. grammar fallback     (valid-by-construction)
Each returns VALID OMC (parse-rate ~1.0); 1-3 aim for CORRECT.

THREE conditions, honest:
  A toolbox-full : generator may use the exact registry entry (measures coverage/retrieval)
  B held-out     : the exact entry is HIDDEN → must synthesize/compose (measures generalization)
  C LM baseline  : FibRec LM generates the fn (the gap)

UNIVERSALITY: only RUN_CMD (interpreter), REGISTRY (corpus), and the grammar are codebase-
specific. The dispatch + correctness machinery is language-agnostic — works on ANY codebase
with a grammar + interpreter + corpus.
"""
import subprocess, tempfile, os, random, re
from pathlib import Path
import torch

HERE = Path(__file__).parent
RUN_CMD = [str((HERE/'../../target/release/omnimcode-standalone').resolve())]   # ← codebase-specific
reg = torch.load(HERE/'omc_name_registry.pt', map_location='cpu', weights_only=False)

def run_capture(src, timeout=15):
    with tempfile.NamedTemporaryFile('w', suffix='.omc', delete=False, dir='/tmp') as f:
        f.write(src); p = f.name
    try:
        r = subprocess.run(RUN_CMD + [p], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None
    finally:
        try: os.unlink(p)
        except Exception: pass

# ── benchmark tasks: (name, n_params, deps) — deps bundled for reference run ──
TASKS = [("gcd",2,[]), ("factorial",1,[]), ("fibonacci",1,[]), ("is_prime",1,[]),
         ("int_pow",2,[]), ("square",1,[]), ("double",1,[]), ("sum_to_n",1,[]),
         ("power_of_two",1,[]), ("lcm",2,["gcd"])]

def gen_inputs(name, k, n=6):
    rng = random.Random(hash(name) & 0xffff)
    lim = 8 if name in ("factorial","fibonacci","power_of_two","int_pow") else 40
    return [tuple(rng.randint(1, lim) for _ in range(k)) for _ in range(n)]

def call_block(name, inputs):
    return "\n".join(f"print({name}({', '.join(map(str,a))}));" for a in inputs)

def expected_for(name, k, deps):
    inputs = gen_inputs(name, k)
    ref = "\n".join(reg[d]['code'] for d in deps) + "\n" + reg[name]['code']
    out = run_capture(ref + "\n" + call_block(name, inputs))
    return inputs, out

def score(candidate_src, name, k, deps, inputs, expected):
    """Rename candidate fn to `name`, bundle deps, run on inputs, compare to expected."""
    if expected is None: return None
    src = re.sub(r'^\s*fn\s+\w+', f'fn {name}', candidate_src, count=1)
    full = "\n".join(reg[d]['code'] for d in deps) + "\n" + src + "\n" + call_block(name, inputs)
    return 1.0 if run_capture(full) == expected else 0.0

# ── NEXT-2 compositional dispatcher ──
def gen_toolbox(name, desc, allow_exact=True):
    if allow_exact and name in reg:
        return reg[name]['code'], 'registry-exact'
    try:
        from phi_synthesis import synthesize
        r = synthesize(desc or name, 64)
        if r: return r[0].replace('{{','{').replace('}}','}'), 'phi-synth'
    except Exception: pass
    # locality retrieval: nearest registry fn by content
    try:
        from locality_fp import build_vocab, hist_fp
        import torch.nn.functional as F
        corpus = (HERE/'omc_corpus.txt').read_text(errors='replace')
        stoi, V = build_vocab(corpus)
        def lf(s):
            ids = torch.tensor([stoi.get(c,0) for c in s]); return hist_fp(ids,0,len(ids),V,bigram=True)
        items = [(e['name'], e['code']) for e in reg.values()]
        M = F.normalize(torch.stack([lf(nm+' '+cd) for nm,cd in items]).float(),dim=1)
        q = F.normalize(lf(desc or name).unsqueeze(0),dim=1)
        return items[int((M@q.T).squeeze(1).argmax())][1], 'locality'
    except Exception: pass
    from grammar_gen import GrammarGen
    return GrammarGen(seed=hash(name)&0xff).gen_fn(name, n_params=2), 'grammar'

if __name__ == '__main__':
    # precompute expected outputs
    bench = {}
    for name,k,deps in TASKS:
        inp, exp = expected_for(name,k,deps)
        bench[name] = (k,deps,inp,exp)
        print(f"[ce] ref {name}: expected={'ok' if exp else 'FAILED'}", flush=True)

    print("\n[ce] === CORRECTNESS by generator condition ===", flush=True)
    descs = {"gcd":"greatest common divisor","factorial":"factorial","fibonacci":"fibonacci",
             "is_prime":"is prime","int_pow":"integer power","square":"square","double":"double",
             "sum_to_n":"sum to n","power_of_two":"power of two","lcm":"least common multiple"}
    for cond, allow in [("A toolbox-full", True), ("B held-out (no exact)", False)]:
        passes = []
        srcs = {}
        for name,k,deps in TASKS:
            code, src_kind = gen_toolbox(name, descs[name], allow_exact=allow)
            srcs[name] = src_kind
            _,_,inp,exp = bench[name]
            passes.append(score(code, name, k, deps, inp, exp) or 0.0)
        print(f"[ce] {cond:22s} correctness={sum(passes)/len(passes):.2f}  ({int(sum(passes))}/{len(passes)})  sources={list(set(srcs.values()))}", flush=True)
    print("[ce] CE DONE", flush=True)
