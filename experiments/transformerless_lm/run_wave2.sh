#!/bin/bash
# wave2: engineering subfields + music + art. Fetches write library/ (safe alongside the language
# expansion). New SE sites run AFTER the current se_dump (shared Posts.xml). Then wait for everything →
# fold → node_expand (serialized DB writers, no collision).
cd /home/thearchitect/OMC/experiments/transformerless_lm || exit 1
ART="Painting,Sculpture,Renaissance art,Impressionism,Cubism,Baroque,Art history,Color theory,Modern art,Abstract art,Printmaking,Drawing,Perspective (graphical),Art movement,Surrealism,Romanticism,Portrait,Landscape painting,Fresco,Watercolor painting"
MUS="Music theory,Harmony,Counterpoint,Melody,Rhythm,Musical notation,Chord (music),Scale (music),Classical music,Jazz,Sonata,Symphony,Orchestra,Tempo,Timbre,Polyphony,Key (music),Interval (music),Octave,Musical composition"

OMP_NUM_THREADS=2 nohup python3 -u arxiv_ingest.py --per-cat 35 >> /tmp/arxiv2.log 2>&1 &
OMP_NUM_THREADS=2 nohup python3 -u wiki_ingest.py --random 500 >> /tmp/wiki2.log 2>&1 &
OMP_NUM_THREADS=2 nohup python3 -u wiki_ingest.py --topics "$ART" >> /tmp/wiki2.log 2>&1 &
OMP_NUM_THREADS=2 nohup python3 -u wiki_ingest.py --topics "$MUS" >> /tmp/wiki2.log 2>&1 &

echo "[wave2] waiting for current se_dump to finish before new SE sites ..."
while pgrep -f "se_dump.py" >/dev/null 2>&1; do sleep 20; done
echo "[wave2] SE: engineering/robotics/music/design/physics/scicomp ..."
OMP_NUM_THREADS=2 python3 -u se_dump.py >> /tmp/se.log 2>&1

echo "[wave2] waiting for all fetchers + language expansion ..."
while pgrep -f "run_more_langs|bible_corpus.py|align.py|arxiv_ingest.py|wiki_ingest.py|dbstack.py" >/dev/null 2>&1; do sleep 30; done
echo "[wave2] all clear $(date) — folding everything into DB ..."
OMP_NUM_THREADS=4 python3 -u dbstack.py
echo "[wave2] node_expand (stopwords + new-term addressability) ..."
OMP_NUM_THREADS=4 python3 -u node_expand.py
echo "[wave2] DONE $(date)"
python3 -c "import sqlite3,json;d=sqlite3.connect('knowledge.db');print('passages',d.execute('SELECT COUNT(*) FROM passages').fetchone()[0],'edges',d.execute('SELECT COUNT(*) FROM edges').fetchone()[0],'fields',d.execute('SELECT COUNT(DISTINCT dom) FROM passages').fetchone()[0],'nodes',len(json.load(open('.kwebcache/web.json'))['nodes']))"
