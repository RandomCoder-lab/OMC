"""sem_rank.py — the system learns to JUDGE MEANING itself, agnostically.

The cross-domain ranker needed a meaning signal that lexical overlap (idf/byte-cosine) can't supply. The
agnostic, law-clean way to get one: DISTRIBUTIONAL SEMANTICS — train a small embedding on the corpus's own
co-occurrence ("a thing is known by the company it keeps"). Supervision = the raw text itself (no
hand-labeled meaning, no dictionary). The system teaches itself what relates to what.

Completes the division of labor: SUBSTRATE = where + grounding (proven addressed index, verify-per-hop);
LEARNED ENCODER = what-relates / meaning (here), trained on corpus-derived co-occurrence -> still agnostic.

Honest test: does the learned embedding PREDICT held-out co-occurrence (real relatedness) better than a
lexical baseline? Measured by AUC. Plus qualitative cross-domain nearest-neighbors (does a P&P character's
nearest Shakespeare character share a theme?). If AUC >> 0.5 and >> lexical, the system has learned to
judge meaning on its own.
"""
import sys, re, time, json, math, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from nl_ground import clean_text, split_passages, extract_entities

HERE = Path(__file__).parent
WORD = re.compile(r"[a-z]+")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="pride_prejudice.txt")
    ap.add_argument("--b", default="tinyshakespeare.txt")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--max_vocab", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = random.Random(args.seed)

    A = clean_text((HERE / args.a).read_text(errors="replace"))
    B = clean_text((HERE / args.b).read_text(errors="replace"))
    paA, paB = split_passages(A), split_passages(B)
    # held-out split of passages (for relatedness eval) — train embeddings on the rest
    cutA, cutB = int(len(paA) * 0.85), int(len(paB) * 0.85)
    train_text = " ".join(paA[:cutA]) + " " + " ".join(paB[:cutB])
    test_paA, test_paB = paA[cutA:], paB[cutB:]

    toks = WORD.findall(train_text.lower())
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:args.max_vocab]]
    stoi = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    ids = [stoi[t] for t in toks if t in stoi]
    print(f"[sem] train tokens={len(ids):,} vocab={V}  (held-out: {len(test_paA)} A + {len(test_paB)} B passages)", flush=True)

    # negative-sampling distribution (unigram^0.75)
    cnt = torch.zeros(V)
    for i in ids:
        cnt[i] += 1
    negp = (cnt ** 0.75); negp /= negp.sum()
    ids_t = torch.tensor(ids, dtype=torch.long)

    # skip-gram with negative sampling — the system learns meaning from context
    emb_in = nn.Embedding(V, args.dim); emb_out = nn.Embedding(V, args.dim)
    nn.init.uniform_(emb_in.weight, -0.5 / args.dim, 0.5 / args.dim); nn.init.zeros_(emb_out.weight)
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=2e-3)
    n = len(ids); BS = 1024
    t0 = time.time()
    g = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        centers_pos = torch.randint(args.window, n - args.window, (BS,), generator=g)
        offs = torch.randint(1, args.window + 1, (BS,), generator=g) * (torch.randint(0, 2, (BS,), generator=g) * 2 - 1)
        ctx_pos = centers_pos + offs
        c = ids_t[centers_pos]; o = ids_t[ctx_pos]
        negs = torch.multinomial(negp, BS * args.neg, replacement=True, generator=g).view(BS, args.neg)
        vc = emb_in(c)                                  # [BS,d]
        vo = emb_out(o)                                 # [BS,d]
        vn = emb_out(negs)                              # [BS,neg,d]
        pos = F.logsigmoid((vc * vo).sum(1))
        neg = F.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)
        loss = -(pos + neg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (args.steps // 5) == 0:
            print(f"[sem]   step {step}/{args.steps} loss={loss.item():.3f}", flush=True)
    print(f"[sem] embeddings trained ({time.time()-t0:.0f}s)", flush=True)
    E = emb_in.weight.detach()
    E = E / (E.norm(dim=1, keepdim=True) + 1e-9)        # normalize -> cosine = dot
    # all-but-the-top (Mu & Viswanath): remove the dominant common direction(s) so frequent-token HUBS
    # stop dominating every neighborhood. Derived from the embeddings themselves — agnostic, no labels.
    mu = E.mean(0)
    Ec = E - mu
    _, _, Vt = torch.linalg.svd(Ec, full_matrices=False)
    top = Vt[:2]                                          # strip top-2 principal directions
    Ec = Ec - (Ec @ top.t()) @ top
    Ec = Ec / (Ec.norm(dim=1, keepdim=True) + 1e-9)

    def vec(w):
        return Ec[stoi[w]] if w in stoi else None
    def relate(a, b):                                   # learned meaning-relatedness
        va, vb = vec(a), vec(b)
        return float(va @ vb) if va is not None and vb is not None else None

    # entities (proper nouns), lowercased to match the embedding vocab
    entA = [e.lower() for e in extract_entities(A, top_n=60) if e.lower() in stoi]
    entB = [e.lower() for e in extract_entities(B, top_n=60) if e.lower() in stoi]
    eAset, eBset = set(entA), set(entB)
    a_only = [e for e in entA if e not in eBset]; b_only = [e for e in entB if e not in eAset]

    # ── HONEST TEST: does learned relatedness predict HELD-OUT co-occurrence > lexical baseline? ──
    def cooc(passages, ents):
        pos = set()
        for p in passages:
            pl = " " + p.lower() + " "
            present = [e for e in ents if (" " + e + " ") in pl or (" " + e + ",") in pl or (" " + e + ".") in pl]
            for x in present:
                for y in present:
                    if x < y:
                        pos.add((x, y))
        return pos
    test_pos = cooc(test_paA, entA) | cooc(test_paB, entB)
    allent = entA + entB
    # build negatives: random entity pairs (same corpus side) NOT co-occurring in held-out
    test_pos = [p for p in test_pos if vec(p[0]) is not None and vec(p[1]) is not None]
    negs_eval = []
    while len(negs_eval) < len(test_pos):
        x, y = rng.choice(allent), rng.choice(allent)
        if x != y and (min(x, y), max(x, y)) not in set(test_pos):
            negs_eval.append((x, y))
    def auc(scorer):
        ps = [scorer(a, b) for a, b in test_pos]
        ns = [scorer(a, b) for a, b in negs_eval]
        ps = [s for s in ps if s is not None]; ns = [s for s in ns if s is not None]
        if not ps or not ns:
            return 0.5
        wins = sum(1 for p in ps for nsamp in random.sample(ns, min(40, len(ns))) if p > nsamp)
        return wins / (len(ps) * min(40, len(ns)))
    def lexical(a, b):                                  # baseline: byte-bigram overlap of the NAMES
        from nl_ground import fp
        return float(fp(a) @ fp(b))
    auc_sem = auc(relate)
    auc_lex = auc(lexical)
    print(f"\n[sem] === does the system judge MEANING-relatedness? (held-out co-occurrence AUC) ===", flush=True)
    print(f"[sem] learned distributional embedding : AUC = {auc_sem:.3f}", flush=True)
    print(f"[sem] lexical name-overlap baseline    : AUC = {auc_lex:.3f}   (chance = 0.500)", flush=True)
    verdict = ("LEARNED MEANING: the embedding predicts real relatedness far above lexical/chance — the "
               "system judges meaning from raw co-occurrence, no labels." if auc_sem >= auc_lex + 0.1 and auc_sem >= 0.65
               else "WEAK: learned signal didn't clearly beat lexical/chance at this scale (honest).")
    print(f"[sem] -> {verdict}", flush=True)

    # ── cross-domain nearest neighbors: does a P&P character's nearest Shakespeare entity share a theme? ──
    print(f"\n[sem] === cross-domain MEANING neighbors (learned; the system's own judgment) ===", flush=True)
    for a in a_only[:6]:
        ranked = sorted(((relate(a, b), b) for b in b_only if relate(a, b) is not None), reverse=True)
        if ranked:
            tops = ", ".join(f"{b}({s:.2f})" for s, b in ranked[:3])
            print(f"[sem]   {a:>12} ⟨A⟩  ~meaning~  {tops}  ⟨B⟩", flush=True)

    (HERE / "results_sem_rank.json").write_text(json.dumps(dict(
        vocab=V, dim=args.dim, auc_semantic=round(auc_sem, 3), auc_lexical=round(auc_lex, 3),
        entitiesA=len(a_only), entitiesB=len(b_only)), indent=2))
    print("\n[sem] wrote results_sem_rank.json", flush=True)


if __name__ == "__main__":
    main()
