# DRIP Active Prefetch: Design and Causal Backtest (2026-07-13)

## 1. Status and conclusion

This is an **experimental ablation**, not the default DRIP policy.

- `DRIP-PrefetchTransition`: predicts the next evidence working set from
  historically observed adjacent-window transitions.
- `DRIP-Prefetch`: combines the transition predictor with a query-centroid
  semantic trend predictor.
- `DRIP` and `DRIPNOdetector` keep their original default behavior. The two
  prefetch variants are available only through `--drip-ablation`.

The final causal backtest does **not** support promoting the hybrid predictor to
the formal method. Transition-only gives a very small gain on SQuAD, is neutral
on HotpotQA and MIND, and the hybrid predictor hurts MIND answerability. The
main bottleneck is not admission control: it is that the current streams contain
too little learnable next-window evidence signal.

## 2. Why this is active prefetch

The reactive DRIP path observes a hot-cache support failure, probes the cold
corpus, and assigns Measured Evidence Flow (MEF) to candidates for the current
window. That is admission after a miss, not prefetching.

The new path predicts evidence for window `t+1` at the end of window `t`, before
any query in `t+1` is visible. It never reads gold support, future queries,
dataset names, or future timestamps. A predicted document receives temporary
Forecasted Evidence Utility (FEU), and still has to pass the existing
primal-dual replacement rule.

## 3. Transition predictor

Let `x_t(d)` be the empirical frequency of observable support proxies in window
`t`, and let `n_t(d)` be the corresponding count. The sparse transition memory is

```text
N_t(i,j) = beta_D N_{t-1}(i,j) + n_{t-1}(i) x_t(j).
```

The Markov forecast and finite-sample confidence are

```text
p_M,t+1(j)
  = sum_i x_t(i) N_t(i,j) / N_t(i,*),

kappa_M,t
  = sum_i x_t(i) N_t(i,*) / (N_t(i,*) + 1).
```

`N/(N+1)` prevents a single observed transition from receiving full confidence.
The model is exponentially forgotten with the existing demand decay `beta_D`,
so an old regime does not remain dominant after drift.

## 4. Semantic trend predictor

The hybrid variant also computes an online smoothed query centroid:

```text
mu_t = normalize(beta_D mu_{t-1} + (1-beta_D) mean(q in Q_t) emb(q)),
v_t  = mu_t - mu_{t-1}.
```

A discounted scalar least-squares momentum coefficient is estimated from only
past centroid velocities:

```text
a_t = clip(
  sum_h beta_D^h <v_{t-h}, v_{t-h-1}>
  / sum_h beta_D^h ||v_{t-h-1}||^2,
  0, 1),

mu_hat_t+1 = normalize(mean(Q_t) + a_t v_t).
```

The cold index is probed around `mu_hat_t+1`. A document receives trend mass only
when it has positive forecast advantage:

```text
g_t(d) = max(0, sim(mu_hat_t+1,d) - sim(mean(Q_t),d)) / sqrt(rank(d)).
```

Trend confidence combines velocity alignment, the learned momentum, and sample
reliability. It is additionally calibrated online with a Beta-Bernoulli posterior
mean:

```text
pi_t = (trend_hits + 1) / (trend_trials + 2).
```

Thus a trend channel that repeatedly misses the next window rapidly loses
weight. This calibration was necessary: the uncalibrated trend version caused
many writes and almost no next-window hits.

## 5. FEU and replacement rule

The transition and trend distributions are confidence-weighted and normalized
into `p_hat_t+1(d)`. Their temporary prefetch credit is

```text
F_t(d) = |Q_t| kappa_t p_hat_t+1(d).
```

`F_t(d)` lasts for one admission decision only. It is removed at the beginning
of the next window without removing genuine MEF accumulated by reactive misses.
Admission remains cost-aware:

```text
Delta_t(c,v)
  = D_t(c) + F_t(c) - m P_t(v) - lambda_t,

admit c replacing v iff Delta_t(c,v) > 0.
```

The prefetcher may add execution slots, but it cannot expand the denominator of
the long-run replacement constraint. The dual update uses the original reactive
opportunity `B_reactive,t`:

```text
lambda_t+1
  = [lambda_t + eta_t (R_t / max(1,B_reactive,t) - b_rep)]_+.
```

This separation fixes an important experimental confound: an earlier prototype
made prefetch writes appear cheaper merely by increasing its own budget
denominator.

## 6. Final backtest protocol

- Seed: 42.
- Hot cache: approximately 10% of the document pool.
- Initialization: causal prefix with three non-evaluation warm-up windows.
- No exact query overlap between warm-up and evaluation.
- Natural temporal workloads preserve chronological order and use span sampling.
- Metrics are full-stream averages.

## 7. Final results

`Pred` is the number of predicted candidate occurrences, `Hit` is the number
that appears in the next observable support state, and `PWrite` is the number of
admissions assisted by FEU.

| Dataset | Policy | HasAns | AMAT | Replacements | Pred | Hit | PWrite |
|---|---|---:|---:|---:|---:|---:|---:|
| SQuAD | LRU | 11.3 | 9.871 | 12,136 | 0 | 0 | 0 |
| SQuAD | DRIPNOdetector | 11.1 | 9.887 | 3,246 | 0 | 0 | 0 |
| SQuAD | DRIP-Prefetch | 11.2 | 9.881 | 3,204 | 380 | 7 | 20 |
| HotpotQA comparison | LRU | 5.2 | 10.484 | 1,637 | 0 | 0 | 0 |
| HotpotQA comparison | DRIPNOdetector | 5.5 | 10.452 | 2,339 | 0 | 0 | 0 |
| HotpotQA comparison | DRIP-Prefetch | 5.5 | 10.452 | 2,339 | 255 | 1 | 11 |
| StreamingQA | LRU | 8.1 | 10.192 | 17,882 | 0 | 0 | 0 |
| StreamingQA | DRIPNOdetector | 6.1 | 10.388 | 5,508 | 0 | 0 | 0 |
| StreamingQA | DRIP-Prefetch | 6.1 | 10.388 | 5,486 | 355 | 0 | 8 |
| MIND news | LRU | 80.1 | 2.990 | 2,719 | 0 | 0 | 0 |
| MIND news | DRIPNOdetector | 79.9 | 3.007 | 2,675 | 0 | 0 | 0 |
| MIND news | DRIP-Prefetch | 79.4 | 3.063 | 2,640 | 203 | 1 | 90 |

### Transition-only control

| Dataset | Core HasAns / Repl. | Transition HasAns / Repl. | Pred / Hit / PWrite |
|---|---:|---:|---:|
| SQuAD | 11.1 / 3,246 | 11.3 / 3,229 | 366 / 7 / 14 |
| HotpotQA comparison | 5.5 / 2,339 | 5.5 / 2,339 | 45 / 0 / 0 |
| MIND news | 79.9 / 2,675 | 79.9 / 2,675 | 9 / 0 / 0 |

## 8. Interpretation

The result is governed by two independent workload properties:

1. **Evidence reuse/cacheability**: will an admitted document be needed again?
2. **Transition predictability**: can its next use be inferred before the query
   arrives?

HotpotQA comparison and StreamingQA have very low adjacent-window evidence
overlap (0.46% and 0.29%, respectively), so a causal predictor has little signal.
MIND has high reuse and 26.96% adjacent overlap, but recency already captures it;
the semantic trend channel adds pollution rather than information. SQuAD has
enough repeated motifs for transition-only to produce a small gain, but its
7/366 next-window precision is still too low for a strong result.

This also explains why a neural network is not automatically the answer. With
only 50 evaluation windows and no independent training traces, a neural model
would either overfit or leak test-stream future information. A learned model is
appropriate only after obtaining many independent historical user/session traces
and enforcing a chronological train/validation/test split.

## 9. Recommended next version

Do not tune the current query-centroid trend predictor into the main paper. The
next meaningful predictor should use causal context that contains information
LRU does not have:

- variable-order PPM/Markov transitions conditioned on user, site, task, action,
  or session state;
- recurring workflow motifs rather than global document-to-document transitions;
- a confidence gate trained/evaluated on disjoint historical traces;
- explicit prefetch precision, coverage, pollution, cold-probe traffic, and net
  AMAT in addition to Has-Answer and Replacements.

Dataset construction should independently control and report (a) cross-phase
evidence overlap, (b) within-phase evidence reuse, and (c) transition
predictability. Drift severity alone does not imply that active prefetch is
possible.

## 10. Reproduction commands

Run from `/home/jyliu/RAG-project`. GPU IDs can be changed.

### SQuAD controlled topic drift

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
experiments/direct/run.py \
  --datasets squad --n-source 4000 --n-stream-queries 15000 \
  --n-windows 50 --window-size 300 --drift full_gradual \
  --init-mode causal-prefix --warmup-windows 3 --kb-pool-ratio 0.1 \
  --drip-ablation \
  --strategies LRU DRIPNOdetector DRIP-Prefetch \
  --output prefetch_v5_final_seed42_squad_r1000.json
```

### HotpotQA comparison

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
experiments/direct/run.py \
  --datasets hotpotqa_comparison --n-source 4000 --n-stream-queries 2500 \
  --n-windows 50 --window-size 50 --drift full_gradual \
  --init-mode causal-prefix --warmup-windows 3 --kb-pool-ratio 0.1 \
  --drip-ablation \
  --strategies LRU DRIPNOdetector DRIP-Prefetch \
  --output prefetch_v5_final_seed42_hotpotqa_comparison_r1000.json
```

### StreamingQA natural temporal stream

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
experiments/direct/run.py \
  --datasets streamingqa_official --n-windows 50 --window-size 500 \
  --drift temporal --temporal-sampling span \
  --init-mode causal-prefix --warmup-windows 3 --kb-pool-ratio 0.1 \
  --drip-ablation \
  --strategies LRU DRIPNOdetector DRIP-Prefetch \
  --output prefetch_v5_final_span_seed42_streamingqa_official_r1000.json
```

### MIND natural temporal access trace

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
experiments/hidden/run.py \
  --datasets mind_news_access --workload natural_temporal \
  --n-stream-queries 25000 --n-windows 50 --window-size 500 \
  --temporal-sampling span --init-mode causal-prefix --warmup-windows 3 \
  --kb-pool-ratio 0.1 --drip-ablation \
  --strategies LRU DRIPNOdetector DRIP-Prefetch \
  --output prefetch_v5_final_span_seed42_mind_news_r1000.json
```

For transition-only, replace `DRIP-Prefetch` with
`DRIP-PrefetchTransition`.

## 11. Result files

- `experiments/direct/data/prefetch_v5_final_seed42_squad_r1000.json`
- `experiments/direct/data/prefetch_v5_final_seed42_hotpotqa_comparison_r1000.json`
- `experiments/direct/data/prefetch_v5_final_span_seed42_streamingqa_official_r1000.json`
- `experiments/agent/data/prefetch_v5_final_span_seed42_mind_news_r1000.json`
- `experiments/direct/data/prefetch_transition_final_seed42_squad_r1000.json`
- `experiments/direct/data/prefetch_transition_final_seed42_hotpotqa_comparison_r1000.json`
- `experiments/agent/data/prefetch_transition_final_span_seed42_mind_news_r1000.json`

## 12. Implementation map

- `algorithms/drip/cache_manager/predictive_prefetch.py`: transition/trend
  predictor, FEU lifecycle, calibration, and diagnostics.
- `algorithms/drip/cache_manager/core.py`: window hooks, prefetch execution
  slots, probe accounting, and logging.
- `algorithms/drip/cache_manager/primal_dual.py`: separates execution budget
  from the replacement-constraint denominator.
- `algorithms/drip/ablation.py`: experimental factories only.
- `algorithms/drip/tests/test_predictive_prefetch.py`: causality and FEU tests.

## 13. Research basis

- USENIX PPM predictive cache: <https://www.usenix.org/conference/usenix-1996-annual-technical-conference/predicting-future-file-system-actions-prior>
- Perceptron data-cache prefetching: <https://arxiv.org/abs/1712.00905>
- Neural memory-access prediction: <https://arxiv.org/abs/1803.02329>

