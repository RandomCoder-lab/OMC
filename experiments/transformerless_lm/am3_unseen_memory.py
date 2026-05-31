"""AM-3 — knowledge via addressed memory, NOT training (the thesis-decider).

Train the LM on domain A (Shakespeare). WRITE memory from an UNSEEN domain B (Pride & Prejudice).
Test on a held-out slice of B that is DISJOINT from the memory (so it cannot just look up the exact
next char — it must generalize from nearby B-context). Compare on B_test:
  - LM-only            : the A-trained model on B (out of domain).
  - LM + memory(B)     : same model, addressed memory written from B_store.   <- knowledge injection
  - LM + memory(A)     : control — memory written from A (Shakespeare).        <- "any memory" baseline
If LM+memory(B) >> LM-only AND > LM+memory(A), the model gained B-knowledge by WRITING, not training.
Disjoint store/test = honest (no exact-answer lookup). Shared vocab across A and B.
"""
import sys, argparse, time
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from am1_addressed_memory import LM, DenseFFN
from am2_write_memory import train_base, hidden_and_logits, build_datastore, eval_knn

HERE = Path(__file__).parent


def load_text(name):
    return (HERE / name).read_text(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="tinyshakespeare.txt")   # train domain
    ap.add_argument("--b", default="pride_prejudice.txt")   # unseen memory domain
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--qpos", type=int, default=48)
    ap.add_argument("--ds", type=int, default=200000)       # datastore size (B_store and A control)
    args = ap.parse_args()

    a_text, b_text = load_text(args.a), load_text(args.b)
    chars = sorted(set(a_text + b_text))                    # SHARED vocab across both domains
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    A = enc(a_text)
    B = enc(b_text)
    # B split: store (write memory) vs test (eval) — DISJOINT, no exact-answer leakage
    bcut = int(B.numel() * 0.85)
    B_store, B_test = B[:bcut], B[bcut:]
    # A split: train vs held-out (for a sanity reference)
    acut = int(A.numel() * 0.9)
    A_train, A_test = A[:acut], A[acut:]
    print(f"[am3] vocab(shared)={vocab}  A(train {args.a})={A_train.numel()}  "
          f"B_store({args.b})={B_store.numel()}  B_test={B_test.numel()}", flush=True)

    t0 = time.time()
    model, g = train_base(vocab, args.d, args.blocks, args.seq, A_train, A_test,
                          args.steps, args.lr, args.bs, args.seed)
    print(f"[am3] base LM trained on A only ({time.time()-t0:.0f}s)", flush=True)

    # datastores: memory of UNSEEN B, and control memory of SEEN A (same size)
    DSB_K, DSB_V = build_datastore(model, B_store, args.seq, args.ds, vocab, g)
    DSA_K, DSA_V = build_datastore(model, A_train, args.seq, args.ds, vocab, g)
    lambdas = [i / 20 for i in range(0, 19)]

    # all evaluated on the SAME held-out B_test
    lmB, bestB, _ = eval_knn(model, B_test, args.seq, vocab, DSB_K, DSB_V, args.qpos, args.k, 1.0, lambdas)
    _,   bestA, _ = eval_knn(model, B_test, args.seq, vocab, DSA_K, DSA_V, args.qpos, args.k, 1.0, lambdas)

    gainB = 100.0 * (lmB - bestB[1]) / lmB
    gainA = 100.0 * (lmB - bestA[1]) / lmB
    print(f"\n[am3] === knowledge-via-memory on UNSEEN domain B (held-out, disjoint from store) ===", flush=True)
    print(f"[am3] LM-only on B_test                      = {lmB:.4f}", flush=True)
    print(f"[am3] LM + memory(B)  λ={bestB[0]:.2f}              = {bestB[1]:.4f}   ({gainB:+.2f}%)  <- injected B-knowledge", flush=True)
    print(f"[am3] LM + memory(A)  λ={bestA[0]:.2f} [control]    = {bestA[1]:.4f}   ({gainA:+.2f}%)  <- 'any memory'", flush=True)
    net = gainB - gainA
    print(f"\n[am3] === VERDICT ===", flush=True)
    print(f"[am3] B-memory gain over LM-only : {gainB:+.2f}%", flush=True)
    print(f"[am3] net knowledge injection (B-memory minus A-control): {net:+.2f}%", flush=True)
    if gainB > 3.0 and net > 1.0:
        print(f"[am3] -> THESIS SUPPORTED: writing UNSEEN-domain memory injected real knowledge "
              f"the LM never trained on, beyond 'any memory'.", flush=True)
    elif gainB > 0 and net > 0:
        print(f"[am3] -> PARTIAL: B-memory helps and beats the control, but modestly at this scale.", flush=True)
    else:
        print(f"[am3] -> NOT at this scale: addressed memory of unseen data did not clearly inject knowledge.", flush=True)
    import json
    (HERE / "results_am3.json").write_text(json.dumps(
        dict(lm_only=round(lmB, 4), mem_B=round(bestB[1], 4), lam_B=bestB[0],
             mem_A=round(bestA[1], 4), lam_A=bestA[0], gain_B=round(gainB, 2),
             gain_A=round(gainA, 2), net=round(net, 2), vocab=vocab, ds=args.ds), indent=2))
    print("[am3] wrote results_am3.json", flush=True)


if __name__ == "__main__":
    main()
