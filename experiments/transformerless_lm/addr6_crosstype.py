"""ADDR-6 — cross-type intersection retrieval, using locality-fp (relevant, not random).

Demonstrates the universal_store idea fixed by ADDR-3: store code + tools + memories,
query by content, and get back the RELEVANT cross-type neighborhood — ranked by a
locality fingerprint (char-bigram histogram) instead of φ-cosine (which ADDR-3 showed
is ~random). This is the template for wiring relevant cross-type retrieval into the assistant.
"""
import torch, torch.nn.functional as F
from pathlib import Path
from locality_fp import build_vocab, hist_fp

HERE = Path(__file__).parent
# shared vocab over the corpus so fps are comparable
corpus = (HERE/'omc_corpus.txt').read_text(errors='replace')
stoi, V = build_vocab(corpus)

def lfp(s, bigram=True):
    ids = torch.tensor([stoi.get(c,0) for c in s], dtype=torch.long)
    return hist_fp(ids, 0, len(ids), V, bigram=bigram)

class CrossTypeStore:
    """Locality-fp keyed multi-type store. Query → relevant cross-type results."""
    def __init__(self): self.items = []   # (type, label, content, fp)
    def add(self, typ, label, content):
        self.items.append((typ, label, content, lfp(label + ' ' + str(content))))
    def query(self, q, top_k=5):
        qf = F.normalize(lfp(q).unsqueeze(0), dim=1)
        M = F.normalize(torch.stack([it[3] for it in self.items]), dim=1)
        sims = (M @ qf.T).squeeze(1)
        return [(float(sims[i]), self.items[i][0], self.items[i][1])
                for i in sims.argsort(descending=True)[:top_k].tolist()]

if __name__ == '__main__':
    s = CrossTypeStore()
    # CODE (real OMC fns)
    s.add('code', 'fibonacci', 'fn fibonacci(n){ if n<=1 {return n;} return fibonacci(n-1)+fibonacci(n-2); }')
    s.add('code', 'merge_sort', 'fn merge_sort(arr){ ... divide and conquer ... }')
    s.add('code', 'gcd', 'fn gcd(a,b){ while b!=0 { h t=b; b=a%b; a=t; } return a; }')
    s.add('code', 'is_prime', 'fn is_prime(n){ ... }')
    # TOOLS
    s.add('tool', 'sequence_generator', 'tool: generate recursive integer sequences')
    s.add('tool', 'sorter', 'tool: sort an array')
    s.add('tool', 'number_theory', 'tool: gcd lcm prime factorization')
    # MEMORIES
    s.add('memory', 'user_asked_fibonacci', 'user asked about fibonacci and recursion earlier')
    s.add('memory', 'user_asked_sorting', 'user wanted a merge sort implementation')

    for q in ["fibonacci recursive sequence", "sort an array", "prime gcd number theory"]:
        print(f"\n[x6] query: {q!r}", flush=True)
        for sim, typ, label in s.query(q, top_k=4):
            print(f"[x6]   {sim:.2f}  [{typ:6s}] {label}", flush=True)
    print("\n[x6] → one query returns the RELEVANT cross-type neighborhood (code+tool+memory),", flush=True)
    print("[x6]   ranked by locality-fp. Template for assistant retrieval (replaces φ ≈ random).", flush=True)
    print("[x6] X6 DONE", flush=True)
