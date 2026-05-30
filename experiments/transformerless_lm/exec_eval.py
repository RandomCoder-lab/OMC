"""ADDR-2 — Execution-grounded eval (the standard generation metric).

Replaces corpus 4-gram validity (which rated bloom_best 0.52 while only 10% of its output
was valid OMC). parse_rate() = fraction of snippets the real OMC interpreter accepts.
Importable: `from exec_eval import parse_rate, run_omc`.
"""
import subprocess, tempfile, os, re
from pathlib import Path

BIN = str((Path(__file__).parent/'../../target/release/omnimcode-standalone').resolve())

def run_omc(snippet: str, timeout: int = 15) -> str:
    """'OK' (valid+ran) | 'RUNTIME_ERR' (parsed, errored) | 'PARSE_ERR' | 'TIMEOUT'."""
    with tempfile.NamedTemporaryFile('w', suffix='.omc', delete=False, dir='/tmp') as f:
        f.write(snippet); path = f.name
    try:
        r = subprocess.run([BIN, path], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0: return 'OK'
        err = r.stderr + r.stdout
        return 'PARSE_ERR' if re.search(r'Expected|Unexpected token|expression can', err) else 'RUNTIME_ERR'
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
    except Exception:
        return 'RUNTIME_ERR'
    finally:
        try: os.unlink(path)
        except Exception: pass

def parse_rate(snippets):
    """Fraction that PARSE (OK or RUNTIME_ERR = syntactically valid). Returns (rate, breakdown)."""
    cats = {}
    for s in snippets:
        c = run_omc(s); cats[c] = cats.get(c, 0) + 1
    n = max(len(snippets), 1)
    return (cats.get('OK', 0) + cats.get('RUNTIME_ERR', 0)) / n, cats

def run_rate(snippets):
    """Fraction that RUN clean (OK only)."""
    cats = {}
    for s in snippets:
        c = run_omc(s); cats[c] = cats.get(c, 0) + 1
    return cats.get('OK', 0) / max(len(snippets), 1), cats

if __name__ == '__main__':
    ok = run_omc("fn f(n){ return n+1; } print(f(2));")
    bad = run_omc("fn f(n { ret =")
    print(f"[exec_eval] sanity: valid→{ok}  garbage→{bad}  (want OK / PARSE_ERR)")
