# Cost-Aware Direct-Evidence Drift Plan

This note is a reviewable plan for the narrower storyline:

```text
Cost-aware cache adaptation under query-visible evidence drift.
```

The paper does not need to solve general hidden-evidence discovery in the main
claim. Hidden evidence can stay as a diagnostic or future extension. The main
system question is whether a hot RAG cache can adapt to evidence drift without
thrashing the L1 cache with excessive writes and evictions.

## Scope

### Main scope

1. Real temporal visible drift.
2. Controlled direct-evidence topic drift.
3. Cost/churn stress under the same visible-evidence setting.

### Out of main scope

General query-invisible evidence discovery:

```text
q -> A1 -> A2 -> ... -> Am
```

That is a valid extension, but it mixes cache management with multi-hop
retrieval/reasoning. Main results should not depend on it.

## Algorithm

The runnable prototype is:

```text
algorithms/drip/cache_manager/cost_aware.py
```

Registry keys:

```text
CostAwareDRIP
CostAwareDRIP-NoDrift
CostAwareDRIP-NoChurn
```

The policy uses only query-visible direct candidates. For each window, it:

1. observes cache hits/misses;
2. retrieves top-k pool candidates only for undercovered queries;
3. estimates drift severity from observable query/cache signals;
4. decays utility faster and opens the write budget under drift;
5. admits candidates only when replacement net gain is positive.

### Drift controller

```text
rho_t =
  w_q   * QueryShift_t
+ w_cov * CoverageDrop_t
+ w_m   * MissRate_t
+ w_n   * CandidateNovelty_t
```

Where:

```text
QueryShift_t       = 1 - cos(mean(q_t), EMA_mean(q))
CoverageDrop_t     = max(0, EMA_coverage - coverage_t)
MissRate_t         = fraction of queries not served by current KB
CandidateNovelty_t = average 1 - max_sim(candidate, current_KB)
```

`rho_t` is clipped to `[0, 1]`.

### Utility update

For a query-visible candidate `d` retrieved from an undercovered query:

```text
U_t(d) =
  decay(rho_t) * U_{t-1}(d)
+ s(q_t, d)
+ rho_t * n_t(d)
```

The direct evidence term is:

```text
s(q, d) =
  direct_weight * sim(q, d) / log2(rank + 1)
+ top1_bonus * I[rank = 1]
```

The novelty term is:

```text
n_t(d) = novelty_weight * (1 - max_sim(d, current_KB))
```

This keeps the method in the direct-evidence regime. It does not use gold
support labels or hidden route hints.

### Cost-aware admission

For candidate `c` and weakest resident victim `v`:

```text
Delta_t(c, v) =
  U_t(c)
- L_t(v)
- C_t(c, v)
```

where:

```text
L_t(v)      = eviction_loss_weight * max(0, Priority(v))
C_t(c, v)  = write_cost + churn_cost + admission_margin_t
```

Admit only if:

```text
Delta_t(c, v) > 0
```

`admission_margin_t` is a small hysteresis cost, not a separate theory-side
threshold:

```text
admission_margin_t = margin_max - rho_t * (margin_max - margin_min)
```

Write budget:

```text
base_cap = min(WRITE_CAP, ceil(KB_size * base_write_fraction))
cap_t    = base_cap * (1 + drift_write_boost * rho_t) / cooldown
```

Anti-churn controls:

```text
min_residency_windows
duplicate_threshold
write_ema cooldown
recent-reentry penalty
```

## Experiment Matrix

### E1. Real temporal visible drift

Purpose: show the boundary condition where recency is naturally strong, and
evaluate quality per write.

```bash
cd /data/jyliu/RAG-project
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/direct/run.py \
  --datasets streamingqa_temporal \
  --drift temporal \
  --n-windows 50 \
  --window-size 100 \
  --strategies ARC LRU FIFO DRIP-QueryVisible CostAwareDRIP CostAwareDRIP-NoDrift OnDemandFetch Oracle \
  --output costaware_streamingqa_temporal_50w100.json
```

Expected interpretation:

```text
LRU/FIFO may remain very strong because StreamingQA is visible and temporal.
CostAwareDRIP should be judged by quality-cost tradeoff, not only raw recall.
```

### E2. Controlled direct-evidence topic drift

Purpose: isolate direct topic/evidence distribution shift without hidden support
completion.

```bash
cd /data/jyliu/RAG-project
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded \
  --q-type comparison \
  --n-source 2000 \
  --n-stream-queries 500 \
  --n-windows 20 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 6250 \
  --strategies ARC LRU FIFO DRIP-QueryVisible CostAwareDRIP CostAwareDRIP-NoDrift OnDemandFetch Oracle \
  --output costaware_2wiki_direct_topic_20w25_kb6250_graphret.json
```

Expected interpretation:

```text
This should be the main visible-topic drift figure.
CostAwareDRIP should compete on H2 has-answer / support coverage per write.
```

### E3. Cost/churn stress

Purpose: test whether the policy avoids thrashing when capacity is tight.

Run the same E2 command with:

```text
--kb-budget 1250
--kb-budget 2500
--kb-budget 6250
```

Plot:

```text
x-axis: writes or churn_rate_mean
y-axis: has_answer_h2 / support_coverage_h2 / recall@5_h2
```

Expected interpretation:

```text
The useful result is a Pareto frontier:
same quality with fewer writes, or higher quality under the same write budget.
```

### Optional E4. Hidden-evidence diagnostic

Purpose: show the boundary of direct-evidence caching, not make it the main
claim.

Use existing bridge workloads only as diagnostics:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded \
  --q-type bridge_comparison \
  --n-source 2000 \
  --n-stream-queries 500 \
  --n-windows 20 \
  --window-size 25 \
  --drift full_gradual \
  --workload cluster_shift \
  --retrieval graph \
  --kb-budget 6250 \
  --strategies ARC LRU FIFO DRIP-QueryVisible DRIP-QueryHidden CostAwareDRIP Oracle \
  --output costaware_hidden_diagnostic_20w25_kb6250_graphret.json
```

Interpretation:

```text
If CostAwareDRIP fails here, that is acceptable.
It is a direct-evidence cache policy, not a latent multi-hop retriever.
```

## Required Metrics

Quality:

```text
recall@5_h2
kb_coverage_h2
has_answer_h2
support_coverage_h2
cold_fetches_h2 / cold_fetches_per_query
```

Cost:

```text
update_cost
evictions
churn_rate_mean
write_budget_mean
maint_retrieval_cost
serve_retrieval_cost
cost
```

Controller diagnostics:

```text
drift_log[].severity
drift_log[].query_shift
drift_log[].coverage_drop
cost_log[].writes
cost_log[].write_budget
cost_log[].admission_margin
cost_log[].avg_net_gain
```

Derived metrics for tables:

```text
writes_per_has_answer_h2 = update_cost / max(1, has_answer_h2)
quality_per_write        = recall@5_h2 / max(1, update_cost)
```

For the paper, prefer a Pareto plot over a single scoreboard.

## Ablations

Use three variants:

| Strategy | Question |
|---|---|
| `CostAwareDRIP` | Full cost-aware drift-controlled policy. |
| `CostAwareDRIP-NoDrift` | Is drift control actually useful beyond net-gain admission? |
| `CostAwareDRIP-NoChurn` | Are min-residency and cooldown preventing cache thrash? |

Keep `qtype` fixed for controlled studies. The point is to avoid mixing
evidence-regime routing with drift/cost control. Later agent mixed workloads
can remove qtype after the cache objective is stable.

## Claim Boundary

Strong claim:

```text
Under query-visible evidence drift, cost-aware admission improves the
quality-cost tradeoff of a hot RAG cache.
```

Do not claim:

```text
The method solves general hidden or latent multi-hop evidence discovery.
```

Hidden evidence can be framed as future work or as a separate extension that
requires path-level retrieval.
