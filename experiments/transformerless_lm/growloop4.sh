#!/bin/bash
# growloop4.sh — MODERN-PRIORITY worker (v5). The corpus was ~99% classical Gutenberg, so modern concepts
# (qubit/gene/algorithm) never reached node-frequency. This flips the mix: the modern feed runs EVERY cycle
# (Wikipedia --random is the fast broad-coverage lever; arXiv/PubMed/code add depth), and Gutenberg is
# throttled to a small pull every 5th cycle so the classical base keeps its long tail without drowning the
# modern share. Same RAM-flat dbstack fold. Stop: touch /tmp/STOP_GROW.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
LOG=/tmp/growloop.log; HOURS=${1:-12}; END=$(($(date +%s) + HOURS*3600)); C=0
echo "[growloop v5 MODERN-PRIORITY] START $(date) +${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
  [ -f /tmp/STOP_GROW ] && { echo "[growloop] STOP $(date)" >> "$LOG"; break; }
  AVD=$(df --output=avail /home 2>/dev/null|tail -1); [ "${AVD:-0}" -lt 2000000 ] && { echo "[growloop] disk<2G pause" >>"$LOG"; sleep 300; continue; }
  C=$((C+1))
  # MODERN every cycle: Wikipedia (broad/fast) + arXiv (science) + PubMed (biomed); code+RFC inside feed.sh
  OMP_NUM_THREADS=2 python3 wiki_ingest.py  --random 120 --batch 10 >> /tmp/wiki.log   2>&1
  OMP_NUM_THREADS=2 python3 arxiv_ingest.py --per-cat 16            >> /tmp/arxiv.log  2>&1
  OMP_NUM_THREADS=2 python3 pubmed_ingest.py --per-term 40          >> /tmp/pubmed.log 2>&1
  # code+RFC every 4th cycle (GitHub 60/hr budget); RFCs cheap so every other
  if [ $((C % 4)) -eq 0 ]; then OMP_NUM_THREADS=2 python3 code_ingest.py --repos --files-per-repo 10 --rfc 40 >> /tmp/code.log 2>&1
  else                          OMP_NUM_THREADS=2 python3 code_ingest.py --rfc 40 >> /tmp/code.log 2>&1; fi
  # Gutenberg throttled: small pull every 5th cycle (keep the classical long tail, don't drown modern)
  [ $((C % 5)) -eq 0 ] && OMP_NUM_THREADS=2 python3 ingest.py --seq 30 >> /tmp/ingest.log 2>&1
  OMP_NUM_THREADS=4 python3 dbstack.py >> /tmp/dbstack.log 2>&1
  N=$(ls library/*.txt 2>/dev/null|wc -l)
  echo "[growloop v5] cycle $C $(date): library $N, disk $(df -h /home 2>/dev/null|awk 'NR==2{print $4}'), RAM $(free -h|awk '/Mem:/{print $7}')" >> "$LOG"
  sleep 20
done
echo "[growloop v5] ENDED $(date)" >> "$LOG"
