"""DEVELOP ADDRESSING #2 — coarse-to-fine navigation + cross-type intersections.

(A) COARSE-TO-FINE: build a nested paragraph→sentence→word tree over the corpus.
    A query descends scale-by-scale (paragraph 233 → sentence 89 → word 34),
    only searching the CHILDREN of the chosen node at each level. Payoff: correct-
    region retrieval at a fraction of the flat search cost. We measure localization
    accuracy (did it descend to the right region?) and the search-cost reduction vs
    flat all-windows search.

(B) CROSS-TYPE INTERSECTION: one φ-address holds a SUPERPOSITION across types
    (code / tool / memory). Query the address → get all co-located types back.

Pure φ math (canonical normals, post-fix); universal, no word lists.
"""
import statistics, time, random
from pathlib import Path
import torch, torch.nn.functional as F
from corpus_address_index import substrate_fingerprint, batch_substrate_fingerprint
from hierarchical_address import assign_address

HERE = Path(__file__).parent
PARA, SENT, WORD = 233, 89, 34          # Fibonacci scales: paragraph / sentence / word

def fp_window(ids, tok_ord, start, L, d=64):
    w = ids[start:start+L].unsqueeze(0)
    return batch_substrate_fingerprint(w, tok_ord, d)[0]

def build_tree(ids, tok_ord, max_paras=400):
    """Nested: paragraph nodes → child sentence offsets → child word offsets."""
    tree = []
    for p in range(0, min(len(ids)-PARA, max_paras*PARA), PARA):
        node = {'pos': p, 'fp': fp_window(ids, tok_ord, p, PARA)}
        # children sentences within [p, p+PARA)
        sents = []
        for s in range(p, p+PARA-SENT, SENT):
            sents.append({'pos': s, 'fp': fp_window(ids, tok_ord, s, SENT),
                          'words': [{'pos': w, 'fp': fp_window(ids, tok_ord, w, WORD)}
                                    for w in range(s, s+SENT-WORD, WORD)]})
        node['sents'] = sents
        tree.append(node)
    return tree

def nearest(query_fp, nodes):
    """Nearest node by cosine; returns (node, n_examined)."""
    if not nodes: return None, 0
    M = F.normalize(torch.stack([n['fp'] for n in nodes]).float(), dim=1)
    q = F.normalize(query_fp.float().unsqueeze(0), dim=1)
    sims = (M @ q.T).squeeze(1)
    return nodes[int(sims.argmax())], len(nodes)

def topb(query_fp, nodes, b):
    """Top-b nodes by cosine; returns (list[node], n_examined)."""
    if not nodes: return [], 0
    M = F.normalize(torch.stack([n['fp'] for n in nodes]).float(), dim=1)
    q = F.normalize(query_fp.float().unsqueeze(0), dim=1)
    sims = (M @ q.T).squeeze(1)
    idx = sims.argsort(descending=True)[:b].tolist()
    return [nodes[i] for i in idx], len(nodes)

def navigate(query_fp, tree):
    """[deprecated] single-fp descent — fails: φ-fp is not scale-invariant."""
    examined = 0
    para, n = nearest(query_fp, tree); examined += n
    sent, n = nearest(query_fp, para['sents']); examined += n
    word, n = nearest(query_fp, sent['words']); examined += n
    return word['pos'], examined, (para['pos'], sent['pos'], word['pos'])

def navigate_scalematched(qpos, ids, tok_ord, tree):
    """CORRECT coarse→fine: re-fingerprint the query's context AT EACH SCALE."""
    examined = 0
    qp_para = fp_window(ids, tok_ord, qpos, PARA)
    para, n = nearest(qp_para, tree); examined += n
    qp_sent = fp_window(ids, tok_ord, qpos, SENT)
    sent, n = nearest(qp_sent, para['sents']); examined += n
    qp_word = fp_window(ids, tok_ord, qpos, WORD)
    word, n = nearest(qp_word, sent['words']); examined += n
    return word['pos'], examined, (para['pos'], sent['pos'], word['pos'])

def navigate_beam(qpos, ids, tok_ord, tree, b=3):
    """Scale-matched BEAM descent: keep top-b at each level so a misrouted argmax
    doesn't drop the right branch. Returns (final_word_pos, examined)."""
    examined = 0
    qp = fp_window(ids, tok_ord, qpos, PARA)
    paras, n = topb(qp, tree, b); examined += n
    qs = fp_window(ids, tok_ord, qpos, SENT)
    sent_pool = [s for p in paras for s in p['sents']]
    sents, n = topb(qs, sent_pool, b); examined += n
    qw = fp_window(ids, tok_ord, qpos, WORD)
    word_pool = [w for s in sents for w in s['words']]
    word, n = nearest(qw, word_pool); examined += n
    return (word['pos'] if word else -1), examined

if __name__ == '__main__':
    t0 = time.time()
    text = (HERE/'omc_corpus.txt').read_text(errors='replace')[:1_500_000]
    chars = sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    tok_ord = torch.tensor([ord(c) for c in chars], dtype=torch.int64)
    print(f"[nav] building nested para→sent→word tree on {len(text):,} chars ...", flush=True)
    tree = build_tree(ids, tok_ord)
    n_words_total = sum(len(s['words']) for nd in tree for s in nd['sents'])
    print(f"[nav] tree: {len(tree)} paragraphs, {n_words_total} word-leaves ({time.time()-t0:.0f}s)", flush=True)

    # ── (A) coarse-to-fine localization + cost ──
    random.seed(0)
    para_spans = [(nd['pos'], nd['pos']+PARA) for nd in tree]
    trials = 80
    # CLEAN METRIC: exact-leaf recovery — query is a real word leaf; does navigation
    # return that exact leaf position? Flat search recovers it at full cost (ceiling);
    # navigation trades recall for cost. Sweep beam width to push recall up.
    qpositions = []
    for _ in range(trials):
        nd = random.choice(tree); s = random.choice(nd['sents']); wd = random.choice(s['words'])
        qpositions.append(wd['pos'])
    flat_cost = n_words_total

    # baselines
    rec_naive = sum(1 for q in qpositions if navigate(fp_window(ids,tok_ord,q,WORD),tree)[2][2]==q)/trials
    sm = [(navigate_scalematched(q,ids,tok_ord,tree)) for q in qpositions]
    rec_sm = sum(1 for q,r in zip(qpositions,sm) if r[2][2]==q)/trials
    print(f"[nav] (A) EXACT word-leaf recovery (query=real leaf; flat ceiling=1.00 @ {flat_cost} examined):", flush=True)
    print(f"[nav]     naive single-fp argmax   recovery={rec_naive:.2f}", flush=True)
    print(f"[nav]     scale-matched argmax     recovery={rec_sm:.2f}  examined~{statistics.mean(e for _,e,_ in sm):.0f}", flush=True)
    for b in (1, 3, 5, 8):
        res = [navigate_beam(q, ids, tok_ord, tree, b=b) for q in qpositions]
        rec = sum(1 for q,(fp,_) in zip(qpositions,res) if fp==q)/trials
        exmean = statistics.mean(e for _,e in res)
        print(f"[nav]     beam B={b:<2} scale-matched   recovery={rec:.2f}  examined~{exmean:.0f}  ({flat_cost/exmean:.1f}x fewer than flat)", flush=True)

    # ── (B) cross-type intersection ──
    from universal_store import UniversalAddressStore
    store = UniversalAddressStore(d_model=64)
    def tool_sort(arr): return sorted(arr)
    # Force three different TYPES to a shared address by giving them the same fp.
    shared_fp = substrate_fingerprint("fibonacci", 64)
    store.store("fn fibonacci(n){...}", type='text', fp=shared_fp)
    store.store(tool_sort, type='tool', fp=shared_fp, name='sort')
    store.store("user asked about fibonacci earlier", type='memory', fp=shared_fp, turn=3)
    hitset = store.retrieve(shared_fp, top_k=8)
    types_at_addr = sorted(set(e.type for _, e in hitset))
    print(f"[nav] (B) cross-type intersection at φ(fibonacci): one address returns types {types_at_addr} "
          f"({len(hitset)} co-located entries) — code+tool+memory as a superposition.", flush=True)
    print("[nav] HIER-NAV DONE", flush=True)
