# DRIP Current Storyline

> Current version: 2026-06-21. This note is the paper-facing story after the
> PPR/DRF/hubness/role-rerank prototypes were retired.

## 0. Talk Summary

The clean story is:

```text
Many agents share one bounded RAG hot cache.
Agent demand drifts, so the system needs a detector to decide when the cache
should move and how aggressively.
But detection alone only says "the hot cache is misaligned"; it does not say
which documents should be written.
DRIP fills that gap with evidence-visibility-aware admission:
  query-visible evidence: q -> d
  query-hidden evidence:  q -> A -> B
Pair Lease then keeps the completed support pair A+B resident together.
```

So the division of labor is:

```text
detector: when/how much to update
router/evidence: what documents to update
pair lease: which support documents must stay together
```

The current strongest result is the controlled multi-agent multi-hop reuse
setting, where no separate topic shift is added. In that setting, DRIP improves
the query-hidden cache-management ablation:

```text
DRIP-QueryVisible -> DRIP
R@5 H2:       4.1 -> 5.9
KB Cov H2:    6.9 -> 9.6
Reuse:        7.7 -> 10.4
Writes:      9542 -> 4174
```

This is the main paper claim. The combined topic-shift + multi-hop experiment
is currently better used as a stress test and future-work motivation, because
direct tail evidence and hidden bridge evidence still compete for slots.

## 1. Core Claim

DRIP is a drift-aware multi-agent hot-cache manager for Agent RAG.

The key separation is:

```text
detector: when and how aggressively should the cache move?
DRIP admission: which documents should be written or retained?
```

This avoids making the detector the whole contribution. The detector only says
"the shared hot cache is no longer aligned with current agent demand." It does
not know which documents should be admitted. The main algorithmic contribution
is evidence-visibility-aware admission:

```text
query-visible evidence: q -> d
query-hidden evidence:  q -> A -> B
```

In hidden multi-hop cases, the cache problem is not just drift detection. The
cache must complete missing support documents that the query itself does not
lexically or semantically expose.

One-sentence version:

```text
DRIP uses drift-aware update control and evidence-visibility-aware admission
to maintain a shared RAG hot cache under multi-agent demand changes.
```

## 2. System Decomposition

### 2.1 Multi-agent drift controller

Code:

```text
algorithms/drip/detection/multi_agent_drift.py
algorithms/drip/cache_manager/__init__.py
```

Input:

```text
current query embeddings
current query-to-hot-cache similarities
agent_id / user_id / client_id when available
```

No support labels, answers, qtype, route_hint, or gold documents are used by the
detector.

Signals:

```text
unsupported_rate_t = mean[ max_{d in C_t} sim(q,d) < tau ]
top1_misalignment_t = 1 - mean_q max_{d in C_t} sim(q,d)
topk_misalignment_t = 1 - mean_q mean_topk_{d in C_t} sim(q,d)
centroid_shift_t = 1 - cos(mu_t, baseline_mu)
```

The detector outputs:

```text
rho_t in [0,1]
```

and DRIP uses it as a controller:

```text
write_cap_t = write_cap_0 * (1 + alpha * rho_t)
decay_t     = decay_0 * (1 - beta  * rho_t)
margin_t    = margin_0 * (1 - gamma * rho_t)
```

Interpretation:

```text
rho_t high  -> write more, forget old demand faster, lower admission barrier
rho_t low   -> stay conservative
```

### 2.2 Evidence visibility router

Code:

```text
algorithms/drip/cache_manager/query_router.py
```

The router decides whether an under-covered query should use direct evidence or
hidden-support completion:

```text
r(q) in {QUERY_VISIBLE, QUERY_HIDDEN}
```

Current full multihop experiments use the workload/planner route hint to isolate
cache-management quality. This should be described as a controlled routing
condition, not as an unsupervised hidden-query classifier result.

### 2.3 Query-visible admission

For query-visible evidence:

```text
E_vis(q,d) = sim(q,d)
D_dir,t(d) <- lambda D_dir,t-1(d) + E_vis(q_t,d)
```

This is the direct dense path. The ablation name is:

```text
DRIP-QueryVisible
```

### 2.4 Query-hidden support completion

For hidden multi-hop evidence, the query exposes or retrieves an anchor support
`A`, but the reusable support `B` is hidden:

```text
q -> A -> B
```

DRIP scores hidden support candidates with:

```text
E_ESC(q, B | A) = sim(phi(q,A), B) * link(A,B) * cue(q,B)
```

where:

```text
sim(phi(q,A), B): semantic compatibility between the query-anchor target and B
link(A,B):        entity/support link strength between A and B
cue(q,B):         generic query/B compatibility cue
```

Demand and pair lease:

```text
D_brg,t(B) <- lambda D_brg,t-1(B) + E_ESC(q_t,B|A)
L_t(A), L_t(B) <- rho L_{t-1} + E_ESC(q_t,B|A)
```

Resident priority:

```text
P_t(d) = S_t(d) + D_dir,t(d) + D_brg,t(d) + lambda_pair L_t(d)
```

Admission:

```text
admit c replacing v iff U_t(c) > margin_t * P_t(v)
```

with a direct-first writer:

```text
direct candidates use budget first
hidden bridge candidates use remaining / reserved budget
bridge eviction avoids direct-protected and recently served residents
```

This is the final DRIP path:

```text
DRIP = DRIP-QueryHidden = QueryVisible + ESC + Pair Lease
```

## 3. Workload Construction

### 3.1 No-shift multihop reuse

Workload:

```text
multi_agent_bridge_reuse
```

Construction:

```text
1. group queries by support title
2. choose a support title B absent from question text as hidden support
3. H1: exposure queries introduce the support need
4. H2: reuse queries from different synthetic agents share the same hidden B
5. initial KB keeps other supports as resident anchors A
6. hidden B is held out of the initial KB
```

This tests whether the cache can write and retain hidden multi-hop support
documents without topic drift.

### 3.2 Topic-shift + multihop reuse

Workload:

```text
topic_shift_bridge_reuse
```

Construction:

```text
1. cluster queries into head/tail topics by query embedding
2. select hidden support B groups that have both head-topic and tail-topic queries
3. H1: head-topic exposure queries
4. H2: tail-topic reuse queries
5. background queries drift gradually from head to tail
6. initial KB keeps anchor A and holds out hidden B
```

This combines two difficulties:

```text
query distribution drift
hidden multi-hop support completion
```

Current results show this combined benchmark is harder and introduces slot
competition between tail direct evidence and hidden bridge evidence.

## 4. Current Results

### 4.1 Single-hop temporal query shift

File:

```text
motivation/motivation_1/data/results_streamingqa_temporal_final_clean.json
```

Setting:

```text
StreamingQA temporal
pool=29,819, KB=400
100 windows x 50 queries
```

| Strategy | R@5 H1 | R@5 H2 | Writes | MaintR | Cost |
|---|---:|---:|---:|---:|---:|
| ARC | 20.5 | 4.4 | 1083 | 4145 | 5228 |
| FIFO | 50.2 | 32.8 | 2109 | 2907 | 5016 |
| LRU | **52.8** | **33.1** | 2033 | 2843 | 4876 |
| SemFlow | 48.9 | 30.3 | 2261 | 14800 | 17061 |
| DRIP | 48.5 | 29.3 | 2174 | 15010 | 17184 |
| Oracle | 84.4 | 79.4 | 22140 | 0 | 22140 |

Reading:

```text
Single-hop temporal drift is not the main DRIP win.
When evidence is query-visible, recency/access-history baselines are strong.
```

### 4.2 No-shift multi-agent multihop reuse

File:

```text
motivation/motivation_2/data/full100_2wiki_no_shift_multiagent_kb750_graphret_current.json
```

Setting:

```text
2Wiki bridge_comparison
n_source=5000, pool=22,984
KB=750
100 windows x 50 queries
workload=multi_agent_bridge_reuse
retrieval=graph
```

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 3.4 | **10.1** | **9.6** | 0.0 | **15.5** | **17.0** | 3.617 | 3265 |
| DRIP-QueryVisible | 4.1 | 6.9 | 7.2 | 0.0 | 6.8 | 7.7 | 3.712 | 9542 |
| DRIP | **5.9** | 9.6 | 8.8 | **0.3** | 8.1 | 10.4 | **3.648** | 4174 |

Reading:

```text
DRIP improves query-hidden multihop cache management over QueryVisible:
  R@5 H2:    4.1 -> 5.9
  KB Cov H2: 6.9 -> 9.6
  Reuse:     7.7 -> 10.4
  Writes:    9542 -> 4174
```

ARC still has strong hidden-B/reuse residency because it is conservative, but
DRIP has better retrieval-facing H2 quality with fewer writes than the
query-visible ablation.

### 4.3 Topic-shift + multihop reuse

File:

```text
motivation/motivation_2/data/full100_2wiki_topic_shift_bridge_kb750_graphret_routed.json
```

Setting:

```text
2Wiki bridge_comparison
n_source=5000, pool=22,984
KB=750
100 windows x 50 queries
drift=full_gradual
workload=topic_shift_bridge_reuse
retrieval=graph
```

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 2.4 | **7.7** | **9.0** | 0.0 | **13.6** | **15.7** | **3.641** | 2785 |
| DRIP-QueryVisible | **4.8** | 7.2 | 7.6 | 0.1 | 6.8 | 8.2 | 3.694 | 10009 |
| DRIP | 4.1 | 7.0 | 8.3 | **0.4** | 6.5 | 8.3 | 3.667 | 4238 |

Reading:

```text
This combined benchmark is not yet DRIP's main win.
DRIP improves H1 and writes less, but H2 R@5 is below QueryVisible.
The likely bottleneck is budget competition:
  tail direct evidence wants slots
  hidden bridge evidence also wants slots
```

This motivates a future drift-aware direct/hidden budget scheduler, but should
not replace the no-shift multihop table as the main evidence.

## 5. Paper-Facing Story

Recommended narrative:

```text
1. Multiple agents share a bounded RAG hot cache.
2. Agent demand drifts, so a detector is needed to decide when the cache should
   move and how aggressively.
3. But drift detection alone is insufficient: it only says "the hot cache is
   misaligned", not which documents should be written.
4. DRIP separates evidence visibility:
   query-visible evidence is written through direct semantic demand;
   query-hidden evidence is completed through q -> A -> B support completion.
5. Pair Lease keeps A+B resident as a support unit, improving multi-hop reuse.
```

Short version:

```text
Detector controls when to move.
Evidence visibility controls what to move.
Pair Lease controls what must stay together.
```

## 6. Claims To Avoid

Do not claim:

```text
DRIP solves all query drift.
DRIP beats every baseline on every metric.
The detector discovers hidden multi-hop supports.
The current topic-shift + multihop benchmark is solved.
```

Safer claim:

```text
DRIP's validated contribution is evidence-visibility-aware hot-cache admission.
It improves controlled no-shift multi-agent multihop reuse, while the combined
topic-shift + multihop setting reveals the next bottleneck: direct/hidden budget
competition under drift.
```
