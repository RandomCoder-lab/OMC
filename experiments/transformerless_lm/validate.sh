#!/bin/bash
# validate.sh — wait for wave-2 (through node_expand) to fully finish + the ground to be rebuilt, then run
# the WHOLE suite on solid ground: see-web, agnostic fact-check, runner (both gears), wavefront
# (pool/bridge/infer), and cross-script translation. Read-only; fires once wave-2 frees the DB.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
echo "[validate] waiting for wave-2 / node_expand to finish ..."
while pgrep -f "run_wave2|run_more_langs|node_expand.py|dbstack.py|align.py|bible_corpus.py|se_dump.py|arxiv_ingest.py|wiki_ingest.py" >/dev/null 2>&1; do sleep 30; done
# confirm ground rebuilt
python3 - <<'PY'
import sqlite3,sys
d=sqlite3.connect('knowledge.db')
try:
  nd=d.execute('SELECT COUNT(*) FROM node_deg').fetchone()[0]
  st=d.execute('SELECT COUNT(*) FROM stop').fetchone()[0]
  print(f"[validate] ground: node_deg={nd:,}  stopwords={st:,}")
  if nd==0: print("[validate] WARNING: node_deg empty — running degraded");
except Exception as e:
  print("[validate] node_deg/stop check:",e)
PY
echo ""; echo "############ FULL SUITE — on rebuilt ground $(date) ############"
echo "===== web size ====="; python3 kask.py stat
echo; echo "===== AGNOSTIC fact-check: 'gravity bends light' (should show refraction dominant, gravity weak/specialized) ====="
python3 factcheck.py --agnostic "gravity bends light"
echo; echo "===== fact-check: 'the heart pumps blood' ====="
python3 factcheck.py "the heart pumps blood"
echo; echo "===== RUNNER coherence from gravity (should climb physics, no boilerplate) ====="
python3 runner.py gravity --steps 6 --mode coherence
echo; echo "===== RUNNER explore from gravity (distant-but-grounded leaps) ====="
python3 runner.py gravity --steps 6 --mode explore
echo; echo "===== WAVEFRONT bridge gravity time (expect relativity) ====="
python3 wavefront.py bridge gravity time
echo; echo "===== WAVEFRONT pool algorithm ====="
python3 wavefront.py pool algorithm
echo; echo "===== WAVEFRONT infer light wave particle (expect photon/quantum, NOT function words) ====="
python3 wavefront.py infer light wave particle
echo; echo "===== TRANSLATION across scripts ====="
for w in Бог 神 konig deus rei; do python3 kask.py translate "$w"; done
echo; echo "===== OMC SELF-KNOWLEDGE: does the web know its own design? ====="
echo "-- concepts containing 'substrate' --"; python3 kask.py search substrate
echo "-- stitch: about substrate --"; python3 stitch.py about substrate 2>/dev/null | head -10
echo "-- bridge: substrate ⟷ addressing --"; python3 wavefront.py bridge substrate addressing 2>/dev/null | head -8
echo; echo "############ SUITE DONE $(date) ############"
