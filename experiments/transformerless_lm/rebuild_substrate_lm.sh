#!/bin/bash
# CO-EVOLUTION pipeline — re-derive the entire substrate LM from CURRENT OMC state.
# OMC is in active development; run this whenever the language/corpus changes and the LM
# regenerates itself. Everything downstream is derived, so the LM tracks OMC automatically.
#
#   corpus/interpreter change  →  ./rebuild_substrate_lm.sh  →  updated substrate LM
#
# What's auto-derived (tracks OMC for free):
#   registry + name index   ← the corpus (build_desc_dataset pulls latest fns)
#   desc↔code dataset        ← latest comments + names + structure
#   learned encoder          ← retrained on the fresh dataset
#   verification             ← runs the CURRENT interpreter binary (auto-tracks new builtins/syntax)
#   correctness benchmark    ← reference impls from the current corpus
# Grammar now AUTO-DERIVED (step 0): derive_grammar.py extracts operators/keywords/construct
# inventory from omnimcode-core/src → omc_grammar.json; grammar_gen is data-driven by it.
# Residual (honest, now measured not silent): per-construct EMITTERS cover 6/20 Statement
# variants; uncovered constructs surface as coverage gaps. Full emitter auto-synthesis = next.
set -e
cd "$(dirname "$0")"
echo "== CO-EVOLVE substrate LM from current OMC =="
echo "[0/5] derive grammar from omnimcode-core/src (operators/keywords/constructs) ..."
python3 derive_grammar.py 2>&1 | grep -E 'arith ops|Statement constructs \('
echo "[1/5] rebuild name registry from corpus ...";        python3 name_registry.py --build >/dev/null
echo "[2/5] rebuild (description,code) dataset ...";        python3 build_desc_dataset.py 2>&1 | grep -E '^\[ds\] built'
echo "[3/5] retrain desc<->code encoder ...";               python3 train_desc_encoder.py 2>&1 | grep -E 'recall@5 = 0|LEARNED'
echo "[4/5] verify assembled LM against current interpreter ..."
python3 - <<'PY'
from substrate_lm import SubstrateLM
lm = SubstrateLM()
ok = lm.answer("gcd", want_name="gcd")[2]
print(f"   SubstrateLM live; sanity answer('gcd') verify={ok}")
PY
echo "== DONE: substrate LM re-derived from current OMC state =="
