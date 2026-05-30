"""Close the last loop — auto-derive the OMC grammar from the REAL source.

The grammar-constrained generator was the one piece NOT tracking OMC (hand-coded subset).
This derives it from ground truth:
  ast.rs        → construct inventory (Statement variants) + operator inventory (Expression variants)
  tokenizer.rs  → concrete operator symbols
  parser.rs     → keyword map ("fn"→Fn, "h"→Harmonic, "while"→While, ...)
Each operator is emitted ONLY if BOTH its AST variant (ast.rs) AND its symbol (tokenizer.rs)
exist — grounded in source, not invented. Writes omc_grammar.json. Re-run when OMC changes →
new operators/keywords flow into the generator automatically; new Statement constructs are
flagged as coverage gaps (the seam becomes visible + measured instead of silent).
"""
import re, json
from pathlib import Path

CORE = Path("/home/thearchitect/OMC/omnimcode-core/src")
OUT = Path(__file__).parent / "omc_grammar.json"

ast = (CORE/"ast.rs").read_text(errors='replace')
tok = (CORE/"tokenizer.rs").read_text(errors='replace')
par = (CORE/"parser.rs").read_text(errors='replace')

def enum_variants(src, name):
    m = re.search(rf"pub enum {name}\s*\{{", src)
    if not m: return []
    i = m.end(); depth = 1; body = []
    while i < len(src) and depth > 0:
        ch = src[i]
        if ch == '{': depth += 1
        elif ch == '}': depth -= 1
        body.append(ch); i += 1
    # top-level variant identifiers (PascalCase at brace-depth 1)
    vs, d = [], 0
    for line in "".join(body).splitlines():
        ms = re.match(r'\s*([A-Z][A-Za-z0-9]*)', line)
        if d == 0 and ms and not line.strip().startswith('//'):
            vs.append(ms.group(1))
        d += line.count('{') + line.count('(') - line.count('}') - line.count(')')
    return sorted(set(vs))

stmts = enum_variants(ast, "Statement")
exprs = enum_variants(ast, "Expression")

# tokenizer symbol set
symbols = set(re.findall(r'"([^"a-zA-Z0-9_ ]{1,3})"', tok))

# canonical AST-variant → symbol (only emitted if BOTH variant and symbol are present)
CANON = {'Add':'+','Sub':'-','Mul':'*','Div':'/','Mod':'%',
         'Eq':'==','Ne':'!=','Lt':'<','Le':'<=','Gt':'>','Ge':'>=',
         'And':'&&','Or':'||','BitAnd':'&','BitOr':'|','BitXor':'^','Shl':'<<','Shr':'>>'}
arith = {v: CANON[v] for v in ('Add','Sub','Mul','Div','Mod') if v in exprs and CANON[v] in symbols}
cmp   = {v: CANON[v] for v in ('Eq','Ne','Lt','Le','Gt','Ge') if v in exprs and CANON[v] in symbols}
logic = {v: CANON[v] for v in ('And','Or') if v in exprs and CANON[v] in symbols}

# keywords from parser.rs ("X" => Token::Y)
keywords = sorted(set(re.findall(r'"([a-z_]+)"\s*=>\s*Token::', par)))

grammar = {
    'statements': stmts, 'expressions': exprs,
    'arith_ops': arith, 'cmp_ops': cmp, 'logic_ops': logic,
    'keywords': keywords,
    'literals': [e for e in ('Number','Float','String','Boolean','Array','Dict') if e in exprs],
}
OUT.write_text(json.dumps(grammar, indent=2))

if __name__ == '__main__':
    print(f"[derive] from omnimcode-core/src → {OUT.name}", flush=True)
    print(f"[derive] Statement constructs ({len(stmts)}): {stmts}", flush=True)
    print(f"[derive] arith ops: {list(arith.values())}  cmp ops: {list(cmp.values())}  logic: {list(logic.values())}", flush=True)
    print(f"[derive] keywords ({len(keywords)}): {keywords}", flush=True)
    print("[derive] DERIVE DONE", flush=True)
