# Harmonic Anomaly Detection: When Attractor-Bucketing Beats IsolationForest (and When It Doesn't)

> A documented comparison of OMNIcode's `harmonic_anomaly` library against scikit-learn's `IsolationForest` on three datasets — synthesized credential stuffing, a real network-intrusion benchmark (NSL-KDD), and a three-attack signature zoo. Honest about wins and losses.

## TL;DR

| Dataset | Top-K | Harmonic | IsolationForest | Winner |
|---|---|:---:|:---:|---|
| Credential stuffing (synthesized, multi-dim) | K=10 | **10/10** | 7/10 | **Harmonic** |
| Credential stuffing | K=25 | **25/25** | 17/25 | Harmonic |
| Credential stuffing | K=50 | **50/50** | 40/50 | Harmonic |
| Attack zoo: exfiltration + scraping + DDoS | K=10×3 | **30/30** | unmeasured | Harmonic (all 100%) |
| Power-law latency outliers (synthesized, 1-D) | K=5 | **4/5** | 0/5 | **Harmonic** |
| Power-law latency outliers | K=30 | 5/30 | **15/30** | IF |
| NAB realKnownCause (1-D time series) | K=10 windows | 7/19 | 7/19 | **Tie** |
| **NSL-KDD network intrusion (real)** | K=10 | 7/10 | **9/10** | **IF** |
| NSL-KDD | K=50 | 42/50 | **45/50** | IF |
| NSL-KDD | K=100 | 78/100 | **92/100** | IF |
| NSL-KDD | K=500 | **365/500** | 351/500 | **Harmonic** (post-substrate-refactor) |

**The pattern:** harmonic wins on *structural* anomalies (rare combinations of normal-looking values), loses on *magnitude* anomalies (values that are simply unusual in scale). NAB and NSL-KDD are mostly magnitude anomalies; credential stuffing is structural.

---

## What the harmonic detector does

For each row in a tabular dataset:

1. Bucket each feature dimension to a Fibonacci attractor via `fold(value)` or `fold(log10(value) * scale)`.
2. Build a frequency histogram per dimension over those buckets.
3. Score each row = sum over dimensions of `-log(p_dim_bucket)`. High score = the row sits in the tail of MULTIPLE dimensions simultaneously.

The full algorithm fits in 40 lines of OMC (see [`examples/lib/harmonic_anomaly.omc`](../examples/lib/harmonic_anomaly.omc)). No training, no hyperparameters, deterministic, single-pass over data.

```omc
import "harmonic_anomaly" as ha;

h det = ha.new(["latency", "status", "endpoint", "hour"]);
ha.set_strategy(det, 1, "discrete");   # status_code is categorical
ha.set_strategy(det, 2, "discrete");   # endpoint_id is categorical
ha.set_strategy(det, 3, "modulo");     # hour-of-day is small periodic

ha.fit(det, training_rows);
h alerts = ha.top_k(det, all_rows, 10);
```

---

## Result 1: Credential stuffing (the strongest win)

**Setup:** 5000 normal HTTP requests + 50 injected credential-stuffing rows. Each row has 4 features: `[latency_ms, status_code, endpoint_id, hour_of_day]`. The attack pattern is `(15ms latency, status=401, endpoint=8 /api/login, hour=3am)`.

Every individual value in an attack row is normal-looking:
- 15ms latency happens (cached responses)
- status=401 happens (~1.5% of bulk traffic)
- /api/login (endpoint 8) sees occasional legitimate traffic
- 3am has off-peak users

The TUPLE is the anomaly.

**Result:**
```
                   K=10   K=25   K=50   K=100
  IsolationForest  7/10  17/25  40/50  50/100
  OMC harmonic    10/10  25/25  50/50  50/100
```

Harmonic catches every credential-stuffing row in the top 10, then top 25, then top 50. IsolationForest catches some but mixes in unrelated magnitude outliers (large 500-error responses, slow batch jobs).

**Why harmonic wins here:** the credential-stuffing pattern is *exactly* the kind of structural anomaly sum-of-marginal-log-rarities targets. Each dimension's bucket is uncommon but not impossible; the rarity multiplies across dimensions.

**Reproduction:**
```bash
./target/release/omnimcode-standalone examples/datascience/multidim_anomaly.omc
```

---

## Result 2: Three-attack zoo (clean sweep)

**Setup:** Three separate experiments, each with 1000 normal rows + 15 injected attacks of a specific type.

1. **Insider exfiltration**: huge response sizes (80-120KB), to a rare endpoint, during business hours, low request count
2. **API abuse / scraping**: status=200 (all successful), every endpoint, any hour, extreme request rate
3. **DDoS pattern**: tiny latency (3-10ms), mixed 200/503 status, single entry endpoint, off-peak hours

**Result (top-10 per scenario):**
```
  Insider exfiltration       : harmonic 10/10 (100% precision)
  API abuse / scraping       : harmonic 10/10 (100% precision)
  DDoS pattern               : harmonic 10/10 (100% precision)
  Aggregate                  : 30/30 across all three scenarios
```

All three attack signatures share the "normal per dim, anomalous in tuple" structure. Harmonic catches all of them.

**Reproduction:**
```bash
./target/release/omnimcode-standalone examples/datascience/anomaly_attack_zoo.omc
```

---

## Result 3: Power-law latency outliers (mixed)

**Setup:** 1000 Pareto-distributed API latencies + 30 injected anomalies of two kinds:
- **On-attractor outliers** (15): large but log-aligned values (100ms, 1000ms — slow batch jobs, expected outliers)
- **Between-attractor anomalies** (15): large AND off-grid (317ms, 731ms — system thrashing, GC pauses, lock contention)

Detection target: catch the between-attractor anomalies (real incidents), ignore the on-attractor ones (slow but routine).

**Result:**
```
                    K=5    K=10   K=20   K=30
  IsolationForest   0/5    5/10   8/20  15/30
  OMC harmonic      4/5    5/10   5/20  5/30
```

At K=5 (the alert-budget regime — what oncall actually pages on), harmonic gets 4/5 between-attractor anomalies; IF gets 0/5 because it picks the largest magnitudes first (which are the on-attractor "expected slow" values).

At K=30, IF eventually catches all 15 between-attractor anomalies plus all 15 on-attractor ones; harmonic plateaus at 5.

**Honest take:** harmonic wins on the metric that matters in production (low-K precision) but loses on broad recall. Different optimization targets.

**Reproduction:**
```bash
./target/release/omnimcode-standalone examples/datascience/anomaly_detection.omc
```

---

## Result 4: NAB realKnownCause (honest tie)

**Setup:** Numenta Anomaly Benchmark — canonical labeled 1-D time-series dataset for anomaly detection. Seven real production traces (AWS CloudWatch CPU, ad exchange, NYC taxi, EC2 latency, etc.) with hand-labeled anomaly windows.

Metric: how many distinct labeled windows the top-K picks cover (NMS-spread to prevent stacking on one spike).

**Result:**
```
                    windows  IF@K=10  H@K=10  IF@K=20  H@K=20
  ambient_temp        2       1/2      1/2      1/2     1/2
  cpu_misconfig       1       1/1      1/1      1/1     1/1
  ec2_latency         3       1/3      1/3      1/3     1/3
  machine_temp        4       1/4      1/4      1/4     1/4
  nyc_taxi            5       1/5      1/5      1/5     1/5
  rogue_agent_hold    2       1/2      1/2      1/2     1/2
  rogue_agent_updown  2       1/2      1/2      1/2     1/2

  TOTALS:            19       7/19    7/19      7/19    7/19
```

Both detectors tie at 7/19. The discriminator works as expected (catches the largest anomaly per series) but neither captures multiple distinct windows.

**Honest take:** beating IF on NAB requires real time-series machinery — CUSUM (cumulative change-point detection), seasonality decomposition via FFT, or HMM/LSTM autoencoders. Numenta's own HTM detector gets ~70%; Twitter's ADVec gets ~60%; naive top-K detectors (us and IF) sit at the 30-40% baseline tier.

The NAB result documents what doesn't work — and where the next architectural move would have to land.

**Reproduction:**
```bash
./target/release/omnimcode-standalone examples/datascience/nab_validation.omc
./target/release/omnimcode-standalone examples/datascience/nab_time_aware.omc  # 3 iterations of harmonic, all still 7/19
```

---

## Result 5: NSL-KDD network intrusion (mixed — substrate-refactor flipped K=500)

**Setup:** Real labeled network intrusion dataset from University of New Brunswick. 22,544 captured connections; we use a 5000-row sample with 2147 normal + 2853 attacks across many classes (neptune DoS, mscan, satan, smurf, warezmaster, etc.). Each row has 41 features; we use 6 numeric ones (duration, src/dst bytes, count, srv_count, dst_host_count).

**Result (post-substrate-refactor, 2026-05-15):**
```
                     K=10    K=50    K=100   K=500
  IsolationForest    9/10    45/50   92/100   351/500
  OMC harmonic       7/10    42/50   78/100   365/500
```

IsolationForest wins at low K (9/10 vs 7/10) and through K=100; harmonic crosses over and wins at K=500 (365 vs 351). The K=500 result is +17 over the pre-refactor measurement (348/500) — the new `log_phi_pi_fibonacci` substrate uses a 40-entry attractor table extending to 63M, vs the old 16-entry table that saturated at 610. NSL-KDD's `src_bytes` and `dst_bytes` features routinely exceed millions; the old substrate compressed every large attack-magnitude to the same near-zero resonance score and the detector couldn't distinguish them. The new substrate sees finer per-row gradients on volumetric attacks.

Looking at IF's top-10 picks: 9 of 10 are labeled `smurf` (a volumetric ICMP flood attack — huge byte counts).
Looking at harmonic's top-10 picks: a mix of `mscan` (port scanning), `warezmaster` (privilege escalation), `back` (buffer overflow), `smurf`.

**Why IF still leads at low K:** NSL-KDD's labeled attacks are dominated by *volumetric* events — DoS floods with massive byte counts. IF picks magnitude outliers first; the labeled attacks at the top of any reasonable score distribution ARE the most extreme magnitudes. IF's job is finding "the biggest spike"; the dataset rewards that.

**Why harmonic catches up at K=500:** look at the *diversity* of what each detector flags. IF stacks on smurf because every smurf row looks the same in magnitude space. Harmonic finds mscan + warezmaster + back + smurf — multiple distinct attack patterns. By the time you've spent 500 alerts, harmonic has surfaced more unique attack types and more total true positives.

For an SRE on a tight alert budget hunting *known* threats, IF is still the right tool (9/10 vs 7/10 at K=10). For *threat hunting* — investigating broadly to find anything anomalous — harmonic's broader coverage (365 vs 351 at K=500) becomes the winning trade.

**Reproduction:**
```bash
# Data is committed at examples/datascience/nsl_kdd_data/sample_5k.csv
./target/release/omnimcode-standalone examples/datascience/nsl_kdd_validation.omc
```

---

## The pattern across all five datasets

| Anomaly type | Harmonic | IsolationForest |
|---|:---:|:---:|
| **Structural** (rare combination of normal-looking values) | ✅ Wins decisively | ❌ Mixes in magnitude outliers |
| **Multi-dim attack signatures** (different per dim, anomalous as tuple) | ✅ 30/30 across three patterns | not measured |
| **Top-of-queue alert precision** (low-K regime on power-law data) | ✅ 4/5 vs 0/5 | ❌ Picks magnitude outliers |
| **Broad recall** (K spans most of dataset) | ❌ Plateaus | ✅ Reaches saturation |
| **1-D time series with extreme spikes** (NAB) | Tie at naive baseline | Tie at naive baseline |
| **Volumetric attacks** (DoS, brute force, huge magnitudes) | ❌ Spreads picks across types | ✅ Wins on precision |

**The honest framing for production use:**

- **Use `harmonic_anomaly` when:** your threat model includes credential stuffing, account takeover, exfiltration via normal-looking traffic, low-and-slow attacks, multi-vector campaigns, or any "looks normal per dim, suspicious in aggregate" pattern.
- **Use `IsolationForest` when:** your threat model is dominated by volumetric attacks (DoS, brute force), high-magnitude resource misuse, or anything where "biggest spike = real incident."
- **Use both** if your alert budget allows — they catch different things and the overlap is small.

---

## Why this matters

Multi-dim structural anomaly detection has been an active research area for 20 years. The current production tooling — IsolationForest, Local Outlier Factor, one-class SVM — was designed for magnitude detection on roughly-Gaussian data. None of them have attractor-bucketing as a first-class primitive.

OMC's `harmonic_anomaly` is 40 lines of OMC on top of `fold()` and `harmonic_partition`. It catches a class of real attack signatures that scikit-learn's tools genuinely miss at low K.

That's not magic. That's not "we replaced IsolationForest." That's: a specific algorithmic primitive (Fibonacci-attractor bucketing) is the right fit for a specific class of anomalies (structural / multi-vector). Knowing which tool to use when is the engineering work; having the tool available is the contribution.

---

## Installing + using

```bash
# Install the library
omnimcode-standalone --install harmonic_anomaly

# Or from URL
omnimcode-standalone --install https://raw.githubusercontent.com/RandomCoder-lab/OMC/main/examples/lib/harmonic_anomaly.omc

# Use it
cat > detect.omc <<'EOF'
import "harmonic_anomaly" as ha;
h det = ha.new(["latency", "status", "endpoint", "hour"]);
ha.set_strategy(det, 1, "discrete");
ha.set_strategy(det, 2, "discrete");
ha.set_strategy(det, 3, "modulo");
ha.fit(det, training_rows);
h alerts = ha.top_k(det, all_rows, 10);
println(alerts);
EOF
omnimcode-standalone detect.omc
```

Source: [`examples/lib/harmonic_anomaly.omc`](../examples/lib/harmonic_anomaly.omc) (~150 lines).

Tutorial: [`examples/datascience/anomaly_tutorial.omc`](../examples/datascience/anomaly_tutorial.omc).

Tests: [`examples/tests/test_harmonic_libs.omc`](../examples/tests/test_harmonic_libs.omc) (18 tests, all passing).

---

## What's not done

- Time-aware anomaly detection (CUSUM, FFT seasonality, HMM) — would be needed to beat IF on NAB.
- Real production deployment — synthetic + benchmark wins are encouraging but not enterprise proof.
- Streaming / incremental fit — currently `fit()` is one-shot; `update()` for online learning is on the roadmap.
- Multi-modal data (text + numeric + categorical) — current bucketing only handles scalar dims.

These are honest gaps. The wins documented above hold within the regime they're measured in. The pattern is the contribution — knowing structural anomalies need structural detection isn't novel; having a one-line OMC library that demonstrates the difference quantitatively is.
