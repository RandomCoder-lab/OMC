"""DEVELOP EVAL #3 — execution-grounded generation quality (vs noisy 4-gram proxy).

The standalone OMC interpreter accepts valid syntax (exit 0) and rejects parse
errors (exit 1, 'Error: at L:C'). So PARSE-RATE = fraction of generated text that
is syntactically-valid OMC — a far truer quality signal than corpus 4-gram overlap.

Measures parse-rate across three sources:
  (1) registry functions  — real corpus fns (ceiling, ~100%)
  (2) φ-synthesis templates — derivable, should parse
  (3) bloom_best LM generations — the honest test of the substrate generator
"""
import subprocess, tempfile, statistics, re, os
from pathlib import Path

HERE = Path(__file__).parent
BIN = str((HERE/'../../target/release/omnimcode-standalone').resolve())

def run_omc(snippet, timeout=15):
    """Return 'OK' | 'PARSE_ERR' | 'RUNTIME_ERR' | 'TIMEOUT'."""
    with tempfile.NamedTemporaryFile('w', suffix='.omc', delete=False, dir='/tmp') as f:
        f.write(snippet); path = f.name
    try:
        r = subprocess.run([BIN, path], capture_output=True, text=True, timeout=timeout)
        os.unlink(path)
        if r.returncode == 0:
            return 'OK'
        err = (r.stderr + r.stdout)
        # parse errors say "Unexpected token"/"Expected"; runtime says "panicked"/"runtime"
        if re.search(r'Expected|Unexpected token|expression can', err):
            return 'PARSE_ERR'
        return 'RUNTIME_ERR'
    except subprocess.TimeoutExpired:
        os.unlink(path); return 'TIMEOUT'
    except Exception:
        try: os.unlink(path)
        except Exception: pass
        return 'RUNTIME_ERR'

def brace_trim(text):
    """Extract first brace-balanced fn body if present, else return text."""
    i = text.find('fn ')
    if i < 0: return text
    depth = 0; started = False
    for j in range(i, len(text)):
        if text[j] == '{': depth += 1; started = True
        elif text[j] == '}':
            depth -= 1
            if started and depth == 0: return text[i:j+1]
    return text[i:]

def rate(snippets):
    cats = {}
    for s in snippets:
        c = run_omc(s); cats[c] = cats.get(c, 0) + 1
    n = max(len(snippets), 1)
    parse_ok = (cats.get('OK', 0) + cats.get('RUNTIME_ERR', 0)) / n  # parsed = OK or ran-then-errored
    return parse_ok, cats

if __name__ == '__main__':
    import torch
    print(f"[exec] interpreter: {BIN}", flush=True)

    # (1) registry functions — ceiling
    reg = torch.load(HERE/'omc_name_registry.pt', map_location='cpu', weights_only=False)
    reg_fns = [e['code'] for e in list(reg.values())[:40]]
    p_reg, c_reg = rate(reg_fns)
    print(f"[exec] (1) registry fns      parse-rate={p_reg:.2f}  {c_reg}", flush=True)

    # (2) φ-synthesis templates
    from phi_synthesis import synthesize, TEMPLATES
    syn = []
    for q in ["fibonacci","merge sort","binary search","gcd","prime sieve","sigmoid",
              "running average","dot product","factorial","insertion sort"]:
        r = synthesize(q, 64)
        if r: syn.append(r[0].replace('{{','{').replace('}}','}'))
    p_syn, c_syn = rate(syn)
    print(f"[exec] (2) φ-synthesis ({len(syn)})   parse-rate={p_syn:.2f}  {c_syn}", flush=True)

    # (3) bloom_best LM generations
    try:
        from omc_assistant import _load_lm
        from grimoire_spells import navigated_generate
        m, stoi, itos, sl = _load_lm(HERE/'bloom_best.pt')
        gens = []
        for i, pr in enumerate(["fn ","fn add(a, b) {","fn sum(arr) {","fn max(a, b) {"]*5):
            g = navigated_generate(m, stoi, itos, sl, pr, 120, nav=False, seed=i)
            gens.append(brace_trim(pr + g))
        p_gen, c_gen = rate(gens)
        print(f"[exec] (3) bloom_best LM ({len(gens)})  parse-rate={p_gen:.2f}  {c_gen}", flush=True)
    except Exception as e:
        print(f"[exec] (3) bloom_best LM — error: {e}", flush=True)

    print("[exec] EXEC-EVAL DONE", flush=True)
