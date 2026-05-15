# Movie Recommendation Engine — OMC Stress Test

A real recommendation engine over MovieLens latest-small. Built to
stress-test the language at scale and surface pain points.

## What's in here

* `recommend.omc` — engine source. Loads CSV → aggregates per-movie
  → builds `harmonic_index` → compares harmonic vs linear lookup at
  100 / 1k / 10k / 100k record scales.
* `PAIN_POINTS.md` — comprehensive, prioritized list of every issue
  found while writing the engine. Read this for the takeaway.
* `sample_100.csv`, `sample_1k.csv` — small samples (committed).
* `sample_10k.csv`, `sample_100k.csv` — gitignored. Re-download
  with the command below.

## Re-downloading the data

```bash
cd /tmp
curl -sL -o ml.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip -p ml.zip ml-latest-small/ratings.csv > /home/thearchitect/OMC/examples/recommend/sample_100k.csv
head -10001 /home/thearchitect/OMC/examples/recommend/sample_100k.csv > /home/thearchitect/OMC/examples/recommend/sample_10k.csv
rm ml.zip
```

CSV schema: `userId,movieId,rating,timestamp`. ~100k ratings from
~600 users on ~9700 movies.

## Running

```bash
./target/release/omnimcode-standalone examples/recommend/recommend.omc
OMC_VM=1 ./target/release/omnimcode-standalone examples/recommend/recommend.omc
```

Both engines should produce identical hit counts (post-CRIT-1 fix).
The 100k stage will hang on a vanilla build — see HIGH-1 in
PAIN_POINTS.md.
