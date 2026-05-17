# Geodesic attention A/B — first Prometheus replication attempt

## Result (3 seeds × 250 steps, 8-token windows)

| seed | vanilla loss | geodesic loss | delta | outcome |
|---|--:|--:|--:|---|
| 42 | 2.464 | 2.713 | +10.1% | geodesic worse |
| 7  | 2.507 | 2.479 | −1.1% | geodesic better |
| 123 | 2.272 | 2.620 | +15.3% | geodesic worse |
| **mean** | **2.414** | **2.604** | **+7.9%** | **1/3 wins** |

**Verdict: inconclusive, leaning negative.** The PyTorch result that
won 3/3 seeds at -0.4% did NOT replicate in this Prometheus run.

## Honest caveat — the K-frozen attention bug

While the A/B was training, a code review surfaced a real bug in
`prom_attention_forward`:

```omc
h k = tape_matmul(x_id, K_w);    # K_w is a trainable param
h k_val = tape_value(k);          # rip out the value
h kt_val = arr_transpose(k_val);  # transpose in OMC space
h kt = tape_const(kt_val);        # re-inject as a CONSTANT
h scores = tape_matmul(q, kt);    # gradient flows ONLY to q
```

The `tape_value → arr_transpose → tape_const` sequence severs
gradient flow through K. `K_w` gets zero gradient from the attention
score path. K is effectively frozen at its random init throughout
training.

**This means both arms (A and B) ran broken attention.** The
geodesic bias was being added to scores `q · K_random^T`, not
scores from a learned K. We're testing whether the geodesic bias
helps when keys are random — an entirely different question from
the PyTorch experiment where K was trainable.

## Why the result is unsurprising given the bug

In the PyTorch experiment, K was trained alongside Q and V. The
geodesic bias added a positional inductive prior on top of
*learned* attention. The model could discover patterns like "attend
to nearby positions for short-range dependencies" and the bias
nudged it toward Fibonacci-coprime distance metrics specifically.

In our Prometheus run, K is fixed at random. The attention scores
have no learned structure. Adding a positional bias to random
scores either:
- adds random noise (no benefit) — most likely
- accidentally provides the ONLY structure → tiny effect either direction

The result is consistent with "broken attention plus a bias either
hurts (overrides random noise that happened to work) or doesn't help
much (random noise was already meaningless)."

## What's needed for a meaningful replication

1. **Add `tape_transpose` Rust builtin.** Differentiable transpose
   so K trains through the score path. ~30 lines forward + backward.
2. **Verify K's gradient is non-zero** after one training step.
3. **Re-run the A/B with both arms having trainable K.**
4. If geodesic still loses 0/3 at this scale, then we have a real
   negative — substrate bias doesn't help when corpus is small +
   model is small + training is short. That's a legit honest finding.
5. If geodesic wins or ties, the PyTorch result replicates.

## Lesson

We shipped a layer without testing its end-to-end gradient flow.
`test_prometheus.omc` has 10 tests covering every other layer and
zero touching attention. That's the regression-prevention gap to
close before any further A/B testing.

The fail-forward path:
1. Fix K (add tape_transpose)
2. Add `test_attention_backward_flows_to_QKV` to lock it
3. Re-run this A/B
4. Report whichever result lands (real win OR real null)
