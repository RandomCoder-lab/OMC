#!/bin/bash
# growloop3.sh — ALL-SOURCE worker (v4): Gutenberg every cycle + modern feed (all connectors) every 3rd
# cycle + RAM-flat dbstack into knowledge.db. One worker owns all six streams. Stop: touch /tmp/STOP_GROW.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
LOG=/tmp/growloop.log; HOURS=${1:-12}; END=$(($(date +%s) + HOURS*3600)); C=0
echo "[growloop v4 ALL-SOURCE] START $(date) +${HOURS}h" >> "$LOG"
while [ "$(date +%s)" -lt "$END" ]; do
  [ -f /tmp/STOP_GROW ] && { echo "[growloop] STOP $(date)" >> "$LOG"; break; }
  AVD=$(df --output=avail /home 2>/dev/null|tail -1); [ "${AVD:-0}" -lt 2000000 ] && { echo "[growloop] disk<2G pause" >>"$LOG"; sleep 300; continue; }
  OMP_NUM_THREADS=4 python3 ingest.py --seq 80 >> /tmp/ingest.log 2>&1
  C=$((C+1)); [ $((C % 3)) -eq 0 ] && bash feed.sh >> /tmp/feed.log 2>&1
  OMP_NUM_THREADS=4 python3 dbstack.py >> /tmp/dbstack.log 2>&1
  N=$(ls library/*.txt 2>/dev/null|wc -l)
  echo "[growloop] cycle $(date): library $N, disk $(df -h /home 2>/dev/null|awk 'NR==2{print $4}'), RAM $(free -h|awk '/Mem:/{print $7}')" >> "$LOG"
  sleep 20
done
echo "[growloop] ENDED $(date)" >> "$LOG"
