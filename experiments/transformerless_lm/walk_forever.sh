#!/bin/bash
# walk_forever.sh — the navigator keeps walking: batches of wandering discovery over the whole web,
# accumulating compass-gated memories (dedup→reinforce) until told to stop. Gentle pacing between batches.
# Stop: touch /tmp/STOP_WALK.
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
LOG=/tmp/walk.log; B=0
echo "[walk] START $(date)" >> "$LOG"
while [ ! -f /tmp/STOP_WALK ]; do
  B=$((B+1))
  OMP_NUM_THREADS=2 python3 -u navigator.py --rounds 25 --wander --quiet >> "$LOG" 2>&1
  M=$(python3 -c "import sqlite3;print(sqlite3.connect('knowledge.db',timeout=30).execute(\"SELECT COUNT(*) FROM memory WHERE kind='derived'\").fetchone()[0])" 2>/dev/null)
  echo "[walk] batch $B done $(date) — derived memories: $M" >> "$LOG"
  sleep 20
done
echo "[walk] STOPPED $(date)" >> "$LOG"
