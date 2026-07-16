# DRIP V3: causal protocol, constrained updater, and verified results

This document is the authoritative record for the current direct-evidence
implementation.  It supersedes result claims based on future-aware
initialization, overlapping warm-up queries, prefix-only natural traces, or the
old fixed churn penalty.

## 1. Current algorithm

### Miss-Conditioned Evidence Frequency (MEF)

Only an under-covered query contributes candidate evidence:

```text
E_t(q,d)
= I[d in TopK(q)] * max(0, sim(q,d))
  / (rank_q(d) * (epsilon + 1 - sim(q,d))^alpha)
  + b_1 * I[rank_q(d) = 1]

D_t(d) = beta_D,t(d) D_{t-1}(d) + gamma * sum_{q in U_t} E_t(q,d)
```

`D_t` is a decayed, miss-conditioned weighted frequency.  It is not a
calibrated prediction and should not be described as hubness.

### Hit-Conditioned Service Utility (HSU)

```text
A_t(d) = sum_{q in Q_t} I[d in H_t(q)] / max(1, |H_t(q)|)
S_t(d) = beta_S,t(d) S_{t-1}(d) + A_t(d)
P_t(d) = D_t(d) + S_t(d)
```

`P_t` is the Composite Residency Utility (CRU).  A candidate is valued by MEF;
a resident is protected by both MEF and observed service.

### Minimax semantic-recency retention

Let `p_sem(v)` be the percentile rank of CRU among residents and `p_rec(v)` the
percentile rank of last-access recency.  The current writer uses

```text
Omega_t(v) = max(p_sem(v), p_rec(v))
v_t        = argmin_{v in K_t} Omega_t(v)
```

This protects a resident when either semantic utility or temporal locality is
strong.  It avoids a dataset-specific convex mixing weight between the two
signals.

### Primal-dual replacement control

The long-run controller targets a fraction `b_rep` of the executable per-window
write cap:

```text
lambda_{t+1}
= [lambda_t + eta_t * (R_t / max(1, B_write,t) - b_rep)]_+

eta_t = 1 / sqrt(t + 1)

Delta_t(c,v) = D_t(c) - m P_t(v) - lambda_t
admit c replacing v iff Delta_t(c,v) > 0
```

The official default is `b_rep=0.50`.  It was selected once on seed 42 and is
not tuned separately for each dataset.

Code mapping:

```text
algorithms/drip/cache_manager/evidence_core.py  MEF, HSU, CRU
algorithms/drip/cache_manager/robust_expert.py  minimax victim ordering
algorithms/drip/cache_manager/primal_dual.py    dual price and admission
algorithms/drip/cache_manager/policies.py       DRIP / DRIPNOdetector
```

## 2. Protocol corrections

All current runs use:

- `init_mode=causal-prefix`;
- a warm-up history disjoint from evaluated queries;
- `KB/pool = 10%` unless explicitly stated;
- no duplicate evaluation queries;
- full-stream averages, not an unexplained H1/H2 slice;
- `temporal_sampling=span` for natural traces.

Protocol diagnostics for the seed-42 validation streams:

| Workload | Pool / KB | Eval queries | Warm-up overlap | Support reuse | Drift audit |
|---|---:|---:|---:|---:|---:|
| SQuAD controlled | 4,000 / 400 | 15,000 | 0 | 73.51% | centroid 0.149, JS 0.481 |
| HotpotQA comparison | 30,454 / 3,045 | 2,500 | 0 | 17.24% | centroid 0.258, JS 0.472 |
| MIND natural | 51,282 / 5,128 | 25,000 | 0 | 86.81% | 97.23% time-span coverage |
| StreamingQA official | 29,819 / 2,982 | 25,000 | 0 | 14.90% | 95.62% time-span coverage |

HotpotQA and StreamingQA have little reusable support.  Their low absolute
Has-Answer rates are therefore workload cacheability limits, not evidence that a
larger cache manager can manufacture reuse that is absent from the stream.

## 3. Global target selection

Seed 42 is the validation stream.  A candidate target is feasible only if its
replacement count is no more than `1.5x LRU` on every validation workload.

| Workload | LRU H/R | ARC H/R | DRIP-25 H/R | DRIP-50 H/R | DRIP-75 H/R |
|---|---:|---:|---:|---:|---:|
| SQuAD | 11.3 / 12,136 | 10.6 / 661 | 11.2 / 2,527 | **11.5 / 3,583** | 11.6 / 3,734 |
| HotpotQA | 5.2 / 1,637 | 6.6 / 5,334 | 5.3 / 1,755 | **5.5 / 2,338** | 5.8 / 3,216 |
| MIND | 80.1 / 2,719 | 80.2 / 2,698 | 75.3 / 1,723 | **79.9 / 2,677** | 80.0 / 2,702 |
| StreamingQA | 8.1 / 17,882 | 8.2 / 3,994 | 6.3 / 3,729 | **6.1 / 5,688** | 6.5 / 7,981 |

`DRIP-75` violates the constraint on HotpotQA (`1.96x LRU`).  Both 0.25 and
0.50 are feasible; 0.50 has the better average Has-Answer rate, so it is frozen
for held-out runs.

Raw files:

```text
experiments/direct/data/target_sweep_disjoint_seed42_squad_direct_r1000.json
experiments/direct/data/target_sweep_disjoint_seed42_hotpotqa_comparison_r1000.json
experiments/agent/data/target_sweep_span_seed42_mind_news_r1000.json
experiments/direct/data/target_sweep_span_seed42_streamingqa_official_r1000.json
```

## 4. Held-out controlled workloads

The following values are mean +/- population standard deviation over held-out
experiment seeds 43 and 44.  Data construction remains fixed by `DATA_SEED=42`;
the held-out seed changes stream ordering and initialization randomness.
`DRIP-Core` is the validated no-detector writer (`DRIPNOdetector` in the
registry); `DRIP+Det` is the same writer with working-set detection (`DRIP`).

| Workload | Method | HasAns | AMAT | Recall@5 | Replacements |
|---|---|---:|---:|---:|---:|
| SQuAD | LRU | 11.30 +/- 0.00 | 9.870 | 10.64 | 12,125 |
| | FIFO | 11.25 +/- 0.05 | 9.873 | 10.58 | 12,153 |
| | TinyLFU | 9.95 +/- 0.05 | 10.004 | 9.33 | 12,491 |
| | GPTCache | 11.35 +/- 0.05 | 9.862 | 10.43 | 2,853 |
| | Proximity | 10.30 +/- 0.20 | 9.968 | 9.52 | 13,008 |
| | ARC | 10.90 +/- 0.00 | 9.913 | 9.76 | **613** |
| | DRIP-Core | **11.60 +/- 0.00** | **9.843** | 10.81 | 3,163 |
| | DRIP+Det | **11.60 +/- 0.20** | 9.844 | **10.83** | 3,729 |
| HotpotQA | LRU | 4.90 +/- 0.50 | 10.514 | 17.37 | **1,614** |
| | FIFO | 4.70 +/- 0.70 | 10.530 | 17.56 | 1,607 |
| | TinyLFU | 4.90 +/- 0.50 | 10.514 | 17.50 | 1,605 |
| | GPTCache | 4.90 +/- 0.50 | 10.512 | 16.15 | 3,377 |
| | Proximity | 4.75 +/- 0.55 | 10.522 | 14.36 | 2,042 |
| | ARC | **7.15 +/- 0.75** | **10.282** | **22.64** | 5,422 |
| | DRIP-Core | 5.45 +/- 0.45 | 10.456 | 19.41 | 2,309 |
| | DRIP+Det | 5.40 +/- 0.40 | 10.458 | 19.38 | 2,305 |

Interpretation:

- On SQuAD, DRIP-Core improves Has-Answer by 0.30 points over LRU while using
  73.9% fewer replacements.  It gains 0.25 points over GPTCache for 10.9% more
  writes.
- On HotpotQA, DRIP-Core improves Has-Answer by 0.55 points over LRU but writes
  43.1% more.  ARC has the highest quality, while DRIP-Core uses 57.4% fewer
  replacements; this is a cost-quality Pareto point, not SOTA dominance.

Raw files use the prefixes:

```text
experiments/direct/data/main_baselines_heldout_seed{43,44}_squad_r1000.json
experiments/direct/data/main_baselines_heldout_seed{43,44}_hotpotqa_comparison_r1000.json
```

## 5. Natural temporal traces

The natural runs use seed 42 because this seed was also the global target
selection stream; they are protocol/boundary evidence rather than held-out
significance claims.

| Workload | Method | HasAns | AMAT | Recall@5 | Replacements |
|---|---|---:|---:|---:|---:|
| MIND | LRU | 80.1 | 2.990 | 80.10 | 2,719 |
| | FIFO | 79.7 | 3.025 | 79.75 | 2,771 |
| | TinyLFU | 80.0 | 3.003 | 79.97 | 2,720 |
| | GPTCache | 25.6 | 8.444 | 25.56 | 3,524 |
| | Proximity | 25.0 | 8.499 | 25.01 | 2,745 |
| | ARC | **80.2** | **2.979** | **80.21** | 2,698 |
| | DRIP-Core | 79.9 | 3.007 | 79.93 | **2,675** |
| | DRIP+Det | 79.9 | 3.008 | 79.92 | 2,677 |
| StreamingQA | LRU | 8.1 | 10.192 | 5.67 | 17,882 |
| | FIFO | 8.0 | 10.197 | 5.72 | 18,168 |
| | TinyLFU | 7.5 | 10.248 | 5.29 | 17,619 |
| | GPTCache | **9.2** | **10.084** | **5.90** | **3,531** |
| | Proximity | 8.2 | 10.184 | 5.31 | 21,074 |
| | ARC | 8.2 | 10.178 | 5.52 | 3,994 |
| | DRIP-Core | 6.1 | 10.388 | 4.26 | 5,508 |
| | DRIP+Det | 6.1 | 10.391 | 4.27 | 5,688 |

MIND has high support reuse and all access-key-aware policies converge near
80%.  StreamingQA has only 14.9% repeated support; here GPTCache-style semantic
admission is best and DRIP is not competitive.  This is a measured boundary of
the current evidence ledger, not a result to hide behind a post-drift slice.

```text
experiments/agent/data/main_baselines_span_seed42_mind_news_r1000.json
experiments/direct/data/main_baselines_span_seed42_streamingqa_official_r1000.json
```

## 6. Detector ablation

`DRIPNOdetector` differs only by disabling working-set KCUSUM and selective
forgetting.  On held-out seeds 43/44:

| Workload | DRIP HasAns / R | NoDet HasAns / R | Result |
|---|---:|---:|---|
| SQuAD | 11.60 / 3,729 | 11.60 / 3,163 | No quality gain; detector adds 566 writes |
| HotpotQA | 5.40 / 2,305 | 5.45 / 2,309 | Statistically indistinguishable |

On seed-42 natural traces, MIND is 79.9/2,677 versus 79.9/2,675, and
StreamingQA is 6.1/5,688 versus 6.1/5,508.  A stationary SQuAD control raises no
`drifted=True` event and produces identical 9.4/3,250 outcomes.  The detector is
therefore statistically structured but has no consistent cache-management
benefit in the current experiments.  It must remain an experimental extension,
not a claimed established contribution.

Raw files:

```text
experiments/direct/data/detector_ablation_selected50_seed42_*.json
experiments/direct/data/heldout_nodet_selected50_seed{43,44}_*.json
experiments/direct/data/stationary_control_selected50_seed42_squad_r1000.json
experiments/agent/data/detector_ablation_selected50_span_seed42_mind_news_r1000.json
```

## 7. Fairness checks

### Candidate breadth

ARC probes 50 candidates while the DRIP default probes 8.  A one-variable DRIP
TopK=50 rerun changes SQuAD from 11.5/3,583 to 11.8/3,660 and HotpotQA from
5.5/2,338 to 5.5/2,341.  The 6.25x candidate-scoring expansion gives no robust
gain, so TopK=8 remains the efficiency default.

```text
experiments/direct/data/topk50_validation_selected50_seed42_*.json
```

### ARC beta

ARC was checked at beta in `{0.15, 0.20, 0.30, 0.70}`.  On HotpotQA, beta=0.70
raises Has-Answer from 6.6 to 7.1 but increases replacements from 5,334 to 9,839.
On SQuAD all settings remain within 0.1 points.  ARC should therefore be reported
as a Pareto baseline; selecting beta=0.70 only for HotpotQA would be post-hoc
dataset-specific tuning.

## 8. Reproduction commands

Run from the project root:

```bash
cd /home/jyliu/RAG-project
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DATA_SEED=42 EXPERIMENT_SEED=42
```

Validation target sweep:

```bash
$PY experiments/direct/run.py \
  --datasets squad --n-windows 50 --window-size 300 \
  --drift full_gradual --init-mode causal-prefix --warmup-windows 3 \
  --kb-pool-ratio 0.1 --drip-ablation \
  --strategies LRU AgentRAGCache \
    DRIP-PDConsensus25 DRIP-PDConsensus50 DRIP-PDConsensus75 \
  --output target_sweep_disjoint_seed42_squad_direct_r1000.json

$PY experiments/direct/run.py \
  --datasets hotpotqa_comparison --n-windows 50 --window-size 50 \
  --drift full_gradual --init-mode causal-prefix --warmup-windows 3 \
  --kb-pool-ratio 0.1 --drip-ablation \
  --strategies LRU AgentRAGCache \
    DRIP-PDConsensus25 DRIP-PDConsensus50 DRIP-PDConsensus75 \
  --output target_sweep_disjoint_seed42_hotpotqa_comparison_r1000.json
```

Natural traces:

```bash
$PY experiments/direct/run.py \
  --datasets streamingqa_official --n-windows 50 --window-size 500 \
  --drift temporal --temporal-sampling span \
  --init-mode causal-prefix --warmup-windows 3 --kb-pool-ratio 0.1 \
  --strategies LRU AgentRAGCache DRIP-PDConsensus50 \
  --drip-ablation \
  --output target_sweep_span_seed42_streamingqa_official_r1000.json

$PY experiments/hidden/run.py \
  --datasets mind_news_access --n-windows 50 --window-size 500 \
  --drift sudden --workload natural_temporal --temporal-sampling span \
  --init-mode causal-prefix --warmup-windows 3 --kb-pool-ratio 0.1 \
  --strategies LRU AgentRAGCache DRIP-PDConsensus50 \
  --drip-ablation \
  --output target_sweep_span_seed42_mind_news_r1000.json
```

Strict detector ablation replaces the strategy list with:

```text
--strategies DRIP DRIPNOdetector
```

Result JSON is always written under the corresponding runner's `data/`
directory, and figures are written under its `figures/` directory.

The complete eight-strategy command used for the broad tables is:

```text
--strategies LRU FIFO TinyLFU GPTCacheStyle Proximity \
  AgentRAGCache DRIPNOdetector DRIP
```

For held-out controlled runs, set `EXPERIMENT_SEED=43` and then `44`; the output
filenames are listed in Section 4.

## 9. Defensible current claim

The validated contribution is evidence-conditioned MEF/HSU accounting,
minimax semantic-recency retention, and primal-dual replacement control.  It
finds useful cost-quality operating points across controlled and natural
workloads, but does not universally dominate LRU or ARC.  Query-hidden evidence
completion remains part of the architecture and research direction, while its
dataset construction and effectiveness are not yet strong enough for a main
result.  The detector is likewise retained in code, but current evidence does
not support presenting it as the source of performance gains.
