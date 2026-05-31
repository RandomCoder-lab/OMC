"""reasoning_nav.py — TEST: is reasoning grounded multi-hop NAVIGATION (scales with the index),
rather than next-token prediction (scales with parameters)?

Controlled transitive-reachability benchmark with KNOWN ground truth. A synthetic functional graph
(each entity has exactly one true successor → unique chains) is written as a shuffled fact corpus
("ENT_a leads_to ENT_b ."), plus distractor facts about unrelated entities (retrieval noise). Task:
given a start entity, is the entity k hops down its chain reachable — and recover the path.

Three reasoners compared at each hop-depth k:
  1. LM-NAV     — a small word-LM's memorized 1-hop transition, chained externally (reasoning in WEIGHTS).
  2. SINGLE-HOP — one addressed retrieval (can answer depth-1 only; the no-chaining control).
  3. GROUNDED-NAV — multi-hop walk through the addressed space; each hop is followed ONLY if a retrieved
     fact verifiably asserts cur->next (grounded; cannot hallucinate). Reasoning in the INDEX.

Pre-registered prediction: as the corpus grows past the small LM's capacity, LM-NAV decays with depth
(1-hop errors compound), while GROUNDED-NAV holds (retrieval scales with the index, not params). If so,
reasoning-as-navigation beats reasoning-in-weights AT SCALE — the punch-above-weight for REASONING,
mirroring the knowledge result. Honest failure modes measured: where grounded-nav breaks (retrieval can't
disambiguate the right fact under distractor load) is the real new ceiling.
"""
import sys, time, json, random
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from am1_addressed_memory import LM, DenseFFN
from corpus import get_batch

HERE = Path(__file__).parent
DIM = 1024


# ── addressed fingerprint (char-bigram histogram — the locality primitive) ──────
def fp(s: str) -> torch.Tensor:
    v = torch.zeros(DIM)
    s = s.lower()
    for i in range(len(s) - 1):
        v[(ord(s[i]) * 131 + ord(s[i + 1])) % DIM] += 1.0
    nrm = v.norm()
    return v / nrm if nrm > 0 else v


def pseudoword(rng):
    C, V = "bcdfghjklmnprstvwz", "aeiou"
    return "".join(rng.choice(C) + rng.choice(V) for _ in range(rng.randint(2, 3))) + str(rng.randint(10, 99))


def gen_graph(n_train_chains, n_new_chains, chain_len, n_distract, seed):
    """TRAIN chains: their facts go to BOTH the LM training set and the index. NEW chains: facts go to the
    index ONLY (added with zero LM retraining) — the reasoning analog of unseen-domain knowledge. All
    entities are distinct across chains (functional graph: unique successor)."""
    rng = random.Random(seed)
    ents = set()
    def newent():
        while True:
            e = pseudoword(rng)
            if e not in ents:
                ents.add(e); return e
    succ = {}
    def mk_chains(n):
        out = []
        for _ in range(n):
            chain = [newent() for _ in range(chain_len + 1)]
            for a, b in zip(chain, chain[1:]):
                succ[a] = b
            out.append(chain)
        return out
    train_chains = mk_chains(n_train_chains)
    new_chains = mk_chains(n_new_chains)
    def facts_of(chains):
        return [f"{a} leads_to {b} ." for c in chains for a, b in zip(c, c[1:])]
    distract = [f"{newent()} leads_to {newent()} ." for _ in range(n_distract)]
    train_facts = facts_of(train_chains)                     # LM trains CLEAN on chain facts (fair/strong)
    all_facts = train_facts + facts_of(new_chains) + distract  # index = train + NEW + distractor noise
    rng.shuffle(train_facts); rng.shuffle(all_facts)
    return train_facts, all_facts, succ, train_chains, new_chains, sorted(ents)


# ── addressed retrieval index (vectorized) ──────────────────────────────────────
class Index:
    def __init__(self, facts):
        self.facts = facts
        self.subj = [f.split(" leads_to ")[0] for f in facts]
        self.obj = [f.split(" leads_to ")[1].split(" .")[0].strip() for f in facts]
        self.M = torch.stack([fp(f) for f in facts])         # [N, DIM]

    def retrieve(self, qstr, topk=8):
        sims = self.M @ fp(qstr)
        idx = sims.topk(min(topk, len(self.facts)), largest=True).indices.tolist()
        return idx


def grounded_nav(start, max_hops, index):
    """Hop through the addressed space; follow a retrieved fact ONLY if it verifiably asserts cur->next."""
    cur, path = start, [start]
    for _ in range(max_hops):
        nxt = None
        for i in index.retrieve(cur, topk=8):
            if index.subj[i] == cur:                 # GROUNDING: the retrieved fact really is about cur
                nxt = index.obj[i]; break
        if nxt is None or nxt in path:
            break
        path.append(nxt); cur = nxt
    return path


def single_hop(start, index):
    for i in index.retrieve(start, topk=8):
        if index.subj[i] == start:
            return index.obj[i]
    return None


# ── LM-in-weights reasoner (word-level) ─────────────────────────────────────────
def train_word_lm(facts, ents, d=128, blocks=2, seq=16, steps=600, seed=0):
    vocab_toks = ["<pad>", "leads_to", "."] + ents
    stoi = {t: i for i, t in enumerate(vocab_toks)}
    itos = {i: t for t, i in stoi.items()}
    stream = []
    for f in facts:
        a, rest = f.split(" leads_to ")
        b = rest.split(" .")[0].strip()
        stream += [stoi[a], stoi["leads_to"], stoi[b], stoi["."]]
    enc = torch.tensor(stream, dtype=torch.long)
    vocab = len(vocab_toks)
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = LM(vocab, d, blocks, seq, lambda: DenseFFN(d, 4 * d))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        x, y = get_batch(enc, 64, seq, g)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, stoi, itos


@torch.no_grad()
def lm_one_hop(model, cur, stoi, itos, ent_set):
    if cur not in stoi:
        return None
    ctx = torch.tensor([[stoi[cur], stoi["leads_to"]]], dtype=torch.long)
    pred = int(model(ctx)[0, -1].argmax())
    tok = itos.get(pred)
    return tok if tok in ent_set else None


def lm_nav(start, max_hops, model, stoi, itos, ent_set):
    cur, path = start, [start]
    for _ in range(max_hops):
        nxt = lm_one_hop(model, cur, stoi, itos, ent_set)
        if nxt is None or nxt in path:
            break
        path.append(nxt); cur = nxt
    return path


def true_chain(start, succ, depth):
    cur, out = start, []
    for _ in range(depth):
        if cur not in succ:
            break
        cur = succ[cur]; out.append(cur)
    return out


def cohort_acc(starts, succ, L, index, model, stoi, itos, ent_set):
    depths = list(range(1, L + 1))
    acc = {m: {k: [0, 0] for k in depths} for m in ("grounded", "lm", "single")}
    for s in starts:
        tc = true_chain(s, succ, L)
        gp = grounded_nav(s, L, index)
        lp = lm_nav(s, L, model, stoi, itos, ent_set)
        sh = single_hop(s, index)
        for k in depths:
            if k > len(tc):
                continue
            target = tc[k - 1]
            acc["grounded"][k][1] += 1; acc["grounded"][k][0] += int(len(gp) > k and gp[k] == target)
            acc["lm"][k][1] += 1;       acc["lm"][k][0] += int(len(lp) > k and lp[k] == target)
            acc["single"][k][1] += 1;   acc["single"][k][0] += int(k == 1 and sh == target)
    pct = lambda m, k: round(100.0 * acc[m][k][0] / acc[m][k][1]) if acc[m][k][1] else 0
    return {m: {k: pct(m, k) for k in depths} for m in acc}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_len", type=int, default=6)
    ap.add_argument("--train_chains", type=int, default=200)
    ap.add_argument("--new_chains", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=120)
    ap.add_argument("--lm_d", type=int, default=128)
    ap.add_argument("--lm_steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    L = args.chain_len
    depths = list(range(1, L + 1))

    n_distract = (args.train_chains + args.new_chains) * L
    train_facts, all_facts, succ, train_chains, new_chains, ents = gen_graph(
        args.train_chains, args.new_chains, L, n_distract, args.seed)
    ent_set = set(ents)
    index = Index(all_facts)                                 # index has TRAIN + NEW (new = zero retraining)
    print(f"[nav] reasoning-as-navigation: SEEN vs UNSEEN chains  chain_len={L}", flush=True)
    print(f"[nav] LM trained on {len(train_facts):,} facts; index holds {len(all_facts):,} "
          f"(incl. {len(new_chains)} NEW chains added with ZERO retraining)", flush=True)
    t0 = time.time()
    model, stoi, itos = train_word_lm(train_facts, ents, d=args.lm_d, steps=args.lm_steps, seed=args.seed)
    lm_params = sum(p.numel() for p in model.parameters())
    print(f"[nav] LM {lm_params/1e6:.2f}M trained ({round(time.time()-t0)}s)\n", flush=True)

    seen = cohort_acc([c[0] for c in train_chains][:args.n_eval], succ, L, index, model, stoi, itos, ent_set)
    unseen = cohort_acc([c[0] for c in new_chains][:args.n_eval], succ, L, index, model, stoi, itos, ent_set)

    def show(title, res):
        print(f"[nav] --- {title} ---", flush=True)
        print("[nav] depth:        " + "  ".join(f"k={k}" for k in depths), flush=True)
        print("[nav] grounded-nav  " + "  ".join(f"{res['grounded'][k]:3d}%" for k in depths), flush=True)
        print("[nav] lm-in-weights " + "  ".join(f"{res['lm'][k]:3d}%" for k in depths), flush=True)
        print("[nav] single-hop    " + "  ".join(f"{res['single'][k]:3d}%" for k in depths), flush=True)
    show("SEEN chains (LM trained on these)", seen)
    print("[nav]", flush=True)
    show("UNSEEN chains (in index only — added with ZERO LM retraining)", unseen)

    print(f"\n[nav] === VERDICT (deep reachability, k={L}) ===", flush=True)
    print(f"[nav] SEEN:   grounded={seen['grounded'][L]}%  lm-in-weights={seen['lm'][L]}%", flush=True)
    print(f"[nav] UNSEEN: grounded={unseen['grounded'][L]}%  lm-in-weights={unseen['lm'][L]}%", flush=True)
    gu, lu = unseen['grounded'][L], unseen['lm'][L]
    if gu >= lu + 30 and gu >= 50:
        print(f"[nav] -> HYPOTHESIS SUPPORTED: on knowledge added AFTER training, grounded navigation reasons "
              f"({gu}%) where the fixed LM cannot ({lu}%). Reasoning over the LIVE INDEX, retrain-free — "
              f"the reasoning analog of write-don't-train knowledge injection.", flush=True)
    elif gu >= lu + 10:
        print(f"[nav] -> PARTIAL: grounded-nav beats fixed-LM on unseen chains, but modestly.", flush=True)
    else:
        print(f"[nav] -> NOT supported (grounded-nav did not clearly beat the LM on unseen chains). Honest negative.", flush=True)
    (HERE / "results_reasoning_nav.json").write_text(json.dumps(
        dict(seen=seen, unseen=unseen, lm_params=lm_params,
             train_facts=len(train_facts), index_facts=len(all_facts)), indent=2))
    print("[nav] wrote results_reasoning_nav.json", flush=True)


if __name__ == "__main__":
    main()
