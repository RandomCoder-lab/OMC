"""NEXT-1 data — build a (description, code) dataset from the raw corpus.

Descriptions come from signal that actually exists in raw code:
  1. LEADING COMMENTS before a fn (natural-language — the rich signal; 19.7k in corpus)
  2. snake_case NAME words ("merge_sort" → "merge sort"; universal weak signal, 92% multi-word)
  3. structural tags (recursive / loop / array / returns) — augmentation from the code itself
Output: desc_code_dataset.jsonl  [{name, description, code, has_comment}]. Pairs for the
contrastive locality encoder (NEXT-1). Then validate: do description→code retrievals bridge
NL queries better than raw char-histogram over code?
"""
import re, json, statistics
from pathlib import Path

HERE = Path(__file__).parent
text = (HERE/'omc_corpus.txt').read_text(errors='replace')
lines = text.splitlines()

def split_name(name):
    s = re.sub(r'[_]+', ' ', name)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.lower().strip()

def clean_comment(cs):
    out = []
    for c in cs:
        c = c.lstrip().lstrip('#').strip()
        c = re.sub(r'^[-=*\s]+$', '', c)          # drop separator bars
        c = re.sub(r'^[-=]{2,}', '', c).strip()
        if len(c) >= 4 and not c.startswith('---'):
            out.append(c)
    return " ".join(out)

def struct_tags(code):
    t = []
    if re.search(r'\b\w+\([^)]*\)\s*\{[^{}]*\b\w+\(', code) and code.count(re.match(r'fn (\w+)', code).group(1)) > 1 if re.match(r'fn (\w+)', code) else False:
        t.append('recursive')
    if 'while ' in code: t.append('loop')
    if 'arr_' in code: t.append('array')
    if 'dict_' in code: t.append('dict')
    if 'str_' in code: t.append('string')
    if re.search(r'return\s+(0|1|n\s*[<>=])', code): t.append('predicate')
    return " ".join(t)

records = []
i = 0
while i < len(lines):
    m = re.match(r'^(\s*)fn\s+(\w+)\s*\(', lines[i])
    if not m:
        i += 1; continue
    indent, name = m.group(1), m.group(2)
    # collect contiguous preceding comment lines
    j = i - 1; cs = []
    while j >= 0 and (lines[j].lstrip().startswith('#') or lines[j].strip() == ''):
        if lines[j].lstrip().startswith('#'): cs.insert(0, lines[j])
        elif cs: break          # blank line after we already saw comments → stop
        j -= 1
    # grab body by brace balance
    depth = 0; body = []; k = i; started = False
    while k < len(lines):
        body.append(lines[k]); depth += lines[k].count('{') - lines[k].count('}')
        if '{' in lines[k]: started = True
        if started and depth <= 0: break
        k += 1
    code = "\n".join(body)
    comment = clean_comment(cs)
    desc = " ".join(filter(None, [comment, split_name(name), struct_tags(code)]))
    records.append({'name': name, 'description': desc, 'code': code, 'has_comment': bool(comment)})
    i = k + 1

# dedupe by name (keep first)
seen = set(); uniq = []
for r in records:
    if r['name'] in seen: continue
    seen.add(r['name']); uniq.append(r)

# merge registry fns (name+code) for full coverage — comment-less but name/struct described
import torch as _t
_reg = _t.load(HERE/'omc_name_registry.pt', map_location='cpu', weights_only=False)
for e in _reg.values():
    if e['name'] in seen: continue
    code = e['code']
    desc = " ".join(filter(None, [split_name(e['name']), struct_tags(code)]))
    seen.add(e['name']); uniq.append({'name': e['name'], 'description': desc, 'code': code, 'has_comment': False})

out = HERE/'desc_code_dataset.jsonl'
with open(out, 'w') as f:
    for r in uniq: f.write(json.dumps(r) + "\n")

n = len(uniq); nc = sum(r['has_comment'] for r in uniq)
print(f"[ds] built {n} (description, code) pairs → {out.name}", flush=True)
print(f"[ds] with real NL comment: {nc} ({100*nc//max(n,1)}%); name+struct only: {n-nc}", flush=True)
print(f"[ds] mean desc len: {statistics.mean(len(r['description']) for r in uniq):.0f} chars", flush=True)
for r in uniq[:3]:
    print(f"[ds] e.g. {r['name']!r}: desc={r['description'][:90]!r}", flush=True)

# ── validation: does description-indexing bridge NL→code better than code-indexing? ──
import torch, torch.nn.functional as F
from locality_fp import build_vocab, hist_fp
allt = " ".join(r['description']+" "+r['code'] for r in uniq)
stoi, V = build_vocab(allt)
def lf(s):
    ids = torch.tensor([stoi.get(c,0) for c in s], dtype=torch.long); return hist_fp(ids,0,len(ids),V,bigram=True)
desc_idx = F.normalize(torch.stack([lf(r['description']) for r in uniq]).float(), dim=1)
code_idx = F.normalize(torch.stack([lf(r['code']) for r in uniq]).float(), dim=1)
# NL queries → does the right-named fn rank top-5?
probes = [("greatest common divisor","gcd"),("merge sort algorithm","merge_sort"),
          ("check if prime","is_prime"),("quick sort","quick_sort"),
          ("factorial","factorial"),("fibonacci sequence","fibonacci"),
          ("binary search","binary_search"),("insertion sort","insertion_sort"),
          ("dot product","dot"),("sum to n","sum_to_n")]
def rank_of(qfp, idx, target):
    sims=(idx@F.normalize(qfp.unsqueeze(0),dim=1).T).squeeze(1)
    order=sims.argsort(descending=True).tolist()
    for rank,ix in enumerate(order):
        if uniq[ix]['name']==target: return rank
    return 9999
print("[ds] NL→code bridge (rank of correct fn; lower=better):", flush=True)
dh=ch=0
for q,tgt in probes:
    if tgt not in seen: continue
    rd=rank_of(lf(q),desc_idx,tgt); rc=rank_of(lf(q),code_idx,tgt)
    dh+=(rd<5); ch+=(rc<5)
    print(f"[ds]   {q:26s}→{tgt:12s}  desc-index rank={rd:<5} code-index rank={rc}", flush=True)
print(f"[ds] top-5 hits: description-index {dh}/{len(probes)} vs code-index {ch}/{len(probes)}", flush=True)
print("[ds] DS DONE", flush=True)
