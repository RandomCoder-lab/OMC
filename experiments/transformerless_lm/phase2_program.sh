#!/bin/bash
# ============================================================================
# PHASE 2 — 8-hour autonomous program (revisions + improvements + capability)
# Builds on Phase-1 findings. KEY IMPROVEMENT this phase: K-shrink was a NO-OP
# (EXP-2); it is now FIXED (stateless_fibgen_forward honors K_active, and
# set_K_active_recursive reaches FibRecLMSubsim). Phase 2 tests whether the
# now-functional K-shrink helps, and runs the best-config long ladder.
# Metric = corpus 4-gram gen-validity, ~330K params, from scratch.
# Budget ~5.7h (margin under 8h). Robust: failures don't abort the rest.
# ============================================================================
cd "$(dirname "$0")"
LOG=AUTONOMOUS_LOG.md
exec >> /tmp/phase2_master.log 2>&1

note()   { { echo ""; echo "### $1"; echo "_$(date '+%m-%d %H:%M')_"; } >> $LOG; }
result() { echo "- $1" >> $LOG; }
val()    { python3 -u measure_validity.py "$1" 2>/dev/null | grep -oE '0\.[0-9]+' | head -1; }
cbest()  { grep BEST "/tmp/auto_${1}.log" 2>/dev/null | tail -1 | grep -oE 'char=[0-9.]+'; }
# train <tag> <steps> <d> <nblk> <lr> <extra-args...>
train()  {
  local tag=$1 steps=$2 d=$3 nb=$4 lr=$5; shift 5
  python3 -u train_address_navigator.py --steps "$steps" --seq-len 256 \
    --K-init 89 --K-min 13 --d-model "$d" --n-blocks "$nb" \
    --batch-size 24 --lr "$lr" --corpus omc_corpus.txt --out "${tag}.pt" --device cpu "$@" \
    > "/tmp/auto_${tag}.log" 2>&1 || echo "  [train $tag FAILED]"
}

echo "==== PHASE 2 START $(date) ===="
# Wait for Phase-1 + K_init follow-up to fully finish.
echo "[p2] waiting for K_init=144 result ..."
until grep -q "EXP-Kinit" $LOG 2>/dev/null && grep -q "K_init=144" $LOG 2>/dev/null; do sleep 60; done
until ! pgrep -f "train_address_navigator.py --steps 500.*K-init 144" >/dev/null 2>&1; do sleep 60; done
sleep 10
note "PHASE 2 — K-shrink fix + best-config ladder"
result "Phase-1 recap: best FibRec ~0.58 (1800 steps), comparable to transformer; K-shrink was a no-op, NOW FIXED."

# EXP-6  K-shrink A/B (now that it's functional): static-K vs functional-K-shrink.
# Same config/seed/budget; only the K schedule differs. Does shrinking K during
# training (the substrate "abstraction ladder") help validity?
note "EXP-6 K-shrink A/B (600 steps, d64/n4/lr3e-4) — tests the FIX"
train "p2_static"  600 64 4 3e-4 --no-k-shrink
result "static K=89 (control)      $(cbest p2_static)  validity=$(val p2_static.pt)"
train "p2_kshrink" 600 64 4 3e-4
result "functional K-shrink 89->13 $(cbest p2_kshrink)  validity=$(val p2_kshrink.pt)"

# EXP-9  TOKEN-CHUNK SCALE sweep (user's idea — group addresses into larger forms,
# char→word→sentence→paragraph, as in the nested dode geometry). Generation-side
# probe: train at seq_len = 34 / 89 / 233 (≈ word / sentence / paragraph context
# chunk) and see whether LARGER-FORM context raises validity. Fibonacci scales.
note "EXP-9 Token-chunk scale sweep (seq_len 34/89/233 ≈ word/sentence/paragraph, 400 steps, d64/n4/lr3e-4, static-K)"
for sl in 34 89 233; do
  python3 -u train_address_navigator.py --steps 400 --seq-len $sl \
    --K-init 89 --K-min 13 --d-model 64 --n-blocks 4 --batch-size 24 --lr 3e-4 \
    --no-k-shrink --corpus omc_corpus.txt --out "p2_chunk_${sl}.pt" --device cpu \
    > "/tmp/auto_p2_chunk_${sl}.log" 2>&1 || echo "  [chunk $sl FAILED]"
  result "seq_len=$sl (chunk) $(cbest p2_chunk_${sl})  validity=$(val p2_chunk_${sl}.pt)"
done

# EXP-7  Best-config long ladder (Phase-1 recommendation): n6 + lr1e-4, long,
# static-K (comparable to Phase-1 EXP-4=0.583). Strongest FibRec config —
# does it beat 0.583 / the transformer?
note "EXP-7 Best-config long ladder (n6, lr1e-4, 1000 steps, static-K)"
train "p2_best" 1000 64 6 1e-4 --no-k-shrink
result "n6/lr1e-4/1000/staticK     $(cbest p2_best)  validity=$(val p2_best.pt)"

echo "==== PHASE 2 COMPLETE $(date) ===="
note "PHASE 2 COMPLETE — see results above"
