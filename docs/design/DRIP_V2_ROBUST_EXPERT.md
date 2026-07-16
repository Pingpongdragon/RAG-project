# Robust-DRIP V2: Learning When to Trust Semantic Evidence

## Motivation

Current experiments expose two regimes rather than one universally best eviction rule:

- StreamingQA has strong temporal locality, where LRU/TinyLFU outperform fixed semantic
  priority;
- controlled topic drift benefits from DRIP's rank-distance demand, where DRIP outperforms
  recency and ARC-style caching.

A fixed mixture cannot be optimal in both regimes. Robust-DRIP therefore treats semantic
demand as learned advice and recency as a classical fallback, then learns their trust online.
This follows the robust learning-augmented caching principle: exploit useful predictions
without losing a non-predictive policy when the predictor is wrong.

## Experts

For each resident document `d`, the semantic expert uses the existing DRIP priority:

```text
P_sem,t(d) = S_t(d) + D_t(d)
```

The recency expert uses the last window in which the resident served a hot-cache query:

```text
P_rec,t(d) = last_seen_t(d)
```

Each priority is converted to a within-cache percentile in `[0,1]`. This avoids combining
raw values with unrelated units.

## Observable Expert Loss

Let `A_t(d)` be the number of current-window hot-cache top-1 hits served by resident `d`.
Before adding current-window credit, expert `e` receives loss

```text
ell_e,t = sum_d A_t(d) * (1 - rank_e,t(d)) / sum_d A_t(d).
```

This is causal and annotation-free: it uses only access traces available at serving time.
An expert is penalized when it had placed a now-requested resident near the eviction end.

## Parameter-Free Exponential Weights

Within one detected regime, cumulative expert loss is

```text
L_e,t = sum_{s <= t} ell_e,s.
```

The learning rate follows the two-expert Hedge schedule

```text
eta_t = min(1, sqrt(2 log 2 / t)),
w_e,t = exp(-eta_t L_e,t) / sum_j exp(-eta_t L_j,t).
```

No dataset-specific expert weight or learning rate is tuned. When query MMD rejects the
current reference distribution, cumulative losses are reset to zero so stale-regime regret
does not lock the controller into the old expert.

## Eviction and Admission

The combined resident protection score is

```text
P_mix,t(d) = w_sem,t * rank_sem,t(d) + w_rec,t * rank_rec,t(d).
```

Victims are considered in ascending `P_mix`. Candidate generation and the admission gate
remain exactly V1:

```text
Delta_t(c,v) = D_t(c) - m P_sem,t(v) - C_t.
```

This isolation is intentional: a gain cannot be attributed to more candidates, a larger write
budget, oracle support labels, or a relaxed replacement threshold. V2 changes only which
resident the writer is willing to sacrifice.

### Parameter-free consensus alternative

The adaptive-weight experiment is retained as a diagnostic. A stricter robust alternative
does not learn a mixture weight and instead uses minimax protection:

```text
P_safe,t(d) = max(rank_sem,t(d), rank_rec,t(d)),
v_t = argmin_d P_safe,t(d).
```

This rule evicts a resident early only when both semantic and recency experts regard it as
weak. It is a parameter-free consensus policy and must be compared against both static
experts; it should not be described as online expert learning.

### Ghost-regret feedback

The hit-rank loss above only evaluates documents that remain resident, so it cannot directly
attribute a future miss to a past eviction. The stronger online variant keeps a ghost record
for each evicted document. If an under-covered query later rediscovers that document in its
cold-corpus top-k, the eviction produces an observable regret event.

At eviction time, expert responsibility is the complement of its protection percentile:

```text
b_e,t(v) = 1 - rank_e,t(v).
```

At the first future cold-probe rediscovery of `v`, the event loss is `b_e,t(v)`, the ghost is
consumed, and the same parameter-free Hedge update is applied. This signal is causal, uses no
gold support or future oracle, and evaluates the actual consequence of eviction rather than
only the ordering of cache hits.

## Required Evaluation

1. `Robust-DRIP` versus V1, LRU, TinyLFU, ARC, and LeCaR-style expert caching.
2. Static semantic and static recency expert ablations.
3. Expert-weight trajectories around known and detected change points.
4. Cache ratios from 0.1% to 10%, not only the current 10% setting.
5. At least five stream seeds, with parameters selected on a validation stream and frozen.
6. FAISS hot/cold indexes with p50/p95 latency, index-write time, and bytes written.

The simulator remains a policy-screening tool. End-to-end system claims require real index
measurements; precomputed NumPy embedding search is not sufficient evidence.
