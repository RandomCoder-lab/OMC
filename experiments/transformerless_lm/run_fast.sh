#!/bin/bash
# run_fast.sh — wait for languages+fetchers to finish, then FAST-fold the ~27k new files (fastfold.py:
# RAM-deduped edges + node expansion, preserves the DB's align/parallel/code edges), then run the WHOLE
# suite on solid ground. Replaces the slow dbstack path.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
echo "[fast] waiting for languages + fetchers to finish ..."
while pgrep -f "run_more_langs|bible_corpus.py|align.py|se_dump.py|arxiv_ingest.py|wiki_ingest.py" >/dev/null 2>&1; do sleep 30; done
echo "[fast] fetchers done $(date) — FAST FOLD (fastfold.py) ..."
OMP_NUM_THREADS=12 python3 -u fastfold.py
echo ""; echo "############ FULL SUITE — solid ground $(date) ############"
echo "===== web size ====="; python3 kask.py stat
echo; echo "===== AGNOSTIC: 'gravity bends light' ====="; python3 factcheck.py --agnostic "gravity bends light"
echo; echo "===== fact-check: 'the heart pumps blood' ====="; python3 factcheck.py "the heart pumps blood"
echo; echo "===== RUNNER coherence from gravity ====="; python3 runner.py gravity --steps 6 --mode coherence
echo; echo "===== RUNNER explore from gravity ====="; python3 runner.py gravity --steps 6 --mode explore
echo; echo "===== WAVEFRONT bridge gravity time ====="; python3 wavefront.py bridge gravity time
echo; echo "===== WAVEFRONT pool algorithm ====="; python3 wavefront.py pool algorithm
echo; echo "===== WAVEFRONT infer light wave particle ====="; python3 wavefront.py infer light wave particle
echo; echo "===== TRANSLATION across scripts ====="; for w in Бог 神 konig deus rei; do python3 kask.py translate "$w"; done
echo; echo "===== OMC SELF-KNOWLEDGE ====="
echo "-- search substrate --"; python3 kask.py search substrate
echo "-- stitch about substrate --"; python3 stitch.py about substrate 2>/dev/null | head -10
echo "-- bridge substrate addressing --"; python3 wavefront.py bridge substrate addressing 2>/dev/null | head -8
echo; echo "############ SUITE DONE $(date) ############"
