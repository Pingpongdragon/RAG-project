# Query Shift Storyline and Benchmark Plan

Date: 2026-06-15
Branch: arc-baseline-and-detectors
Status: CEO review execution doc

## 0. Decision

Do not tune GraphIndex further until the story and benchmark are fixed.

The current risk is not that the graph formula lacks one more boost term. The
risk is that reviewers read the method as another query-time retriever, while
the paper claims shared RAG hot-tier cache management.

Order of work:

```text
1. Lock the paper story.
2. Build the query-shift benchmark matrix.
3. Re-run baselines, especially ARC-style cache.
4. Change DRIP/GraphIndex only against benchmark failures.
```

This is the whole game. If the benchmark does not require future reuse, prefetch
is just extra retrieval work with a nicer name.

## 1. One-Line Claim

DRIP manages a shared RAG hot tier under query demand shift.

The corpus may be static, but the incoming query distribution changes over time,
so the optimal resident evidence working set also changes.

```tex
q_t \sim P_t(q), \qquad P_t(q) \neq P_{t+\Delta}(q)
```

The ideal cache at time `t` is:

```tex
C_t^*
=
\arg\max_{C: |C| \le B}
\mathbb{E}_{q \sim P_t(q)}[\mathrm{Hit}(q, C)]
```

DRIP tries to track `C_t -> C_t^*` with route-aware admission.

## 2. What ARC Covers, And What It Does Not

ARC is the nearest-neighbor threat. It is not a strawman.

ARC studies compact passage cache for agent RAG. It scores passages with:

```tex
\mathrm{Priority}(p)
=
\frac{
  \beta \log(h_k(p)+1) + (1-\beta)\mathrm{DRF}(p)
}{
  \log(w(p)+1)
}
```

where:

```tex
\mathrm{DRF}(p)
=
\sum_{q: p \in Ret(q)}
\frac{1}{rank(q,p) \cdot dist(q,p)^\alpha}
```

ARC is strong for stable, query-visible passage caching:

```text
history says this passage is useful
+ embedding geometry says this passage is central
-> keep it in cache
```

But ARC does not explicitly cover the two dimensions DRIP needs:

| Dimension | ARC | DRIP |
|---|---|---|
| Demand dynamics | Mostly fixed `P(q|Theta)`, streaming warm-up | Explicit query demand shift over time |
| Evidence visibility | Query-visible retrieval | Single visible, direct multi-hop visible, bridge-hidden |
| Hidden bridge | No `q -> A -> entity/relation -> B` path | GraphIndex / entity-chained prefetch |
| Cache claim | Compact per-agent corpus | Shared hot-tier working set under drift |

So the paper should not say "we invented query-distribution cache for agent
RAG." ARC already does that.

The paper should say:

```text
ARC handles stable query-visible cache value.
DRIP handles shifting demand and route-specific evidence visibility,
including bridge evidence not reachable from the query embedding alone.
```

## 3. The Two-Axis Benchmark

Cache failure has two causes that should be measured separately.

```text
Axis 1: Demand Dynamics
  static
  random stream
  temporal drift
  burst drift

Axis 2: Evidence Visibility
  single-hop visible
  direct multi-hop visible
  bridge-hidden
```

Why this matters:

```text
If demand is stable:
  ARC/LFU/hubness should look strong.

If demand shifts:
  old working sets become stale, so decay/detection/update budget matters.

If evidence is query-visible:
  DRIP-Dense can work.

If evidence is bridge-hidden:
  query embedding cannot see B, so GraphIndex must use traces from A.
```

The target benchmark matrix:

| Workload | Demand dynamics | Evidence visibility | Main question |
|---|---|---|---|
| StreamingQA-Single | temporal drift | single-hop visible | Does query-side admission avoid breaking simple QA? |
| HotpotQA-DirectStream | static/random or drifted | direct multi-hop visible | Does semantic propagation co-reside visible evidence? |
| 2Wiki-Bridge-Diagnosis | static/random | bridge-hidden | Does query-only cache miss hidden B? |
| TemporalBridgeReuseStream | temporal/burst drift | bridge-hidden | Does prefetch reduce future cold-tier access? |

The last row is the missing proof.

## 4. Query Shift Types

Do not treat "shift" as one vague word. Define it.

| Shift type | Meaning | Example |
|---|---|---|
| Topic shift | Domain/topic mixture changes | medical questions -> legal questions |
| Entity shift | Same task type, new entity cluster | NBA players -> football players |
| Relation shift | Predicate being asked changes | born-in -> spouse-of -> directed-by |
| Task shift | Question structure changes | single-hop lookup -> comparison -> bridge |
| Granularity shift | Required chunk level changes | overview doc -> exact clause/chunk |
| Burst shift | Short-lived concentrated demand | API outage, breaking news, product bug |
| Cyclic shift | Old demand returns | monthly report, semester topic, recurring workflow |

For this paper, the strongest initial scope is:

```text
topic/entity/relation drift + burst reuse
```

Task shift is useful for analysis, but if it drives the whole benchmark, it can
confuse the story because router quality becomes a moving target.

## 5. Algorithm Roles

The method should be described as a route-aware cache manager, not one scoring
formula.

Naming:

```text
DRIP-Dense = Semantic Flow.
```

It is the query-visible module inside DRIP. When a query is under-covered by
the hot tier, DRIP-Dense sends demand from the query to dense top documents and
their semantic neighbors. It handles single-hop and direct multi-hop evidence,
where the needed chunk is visible from the query embedding.

DRIP is the full route-aware system:

```text
DRIP = DRIP-Dense + GraphIndex bridge evidence + budgeted writer/router.
```

```text
QueryRouter
  -> Single / MultiDirect:
       DRIP-Dense direct-demand admission

  -> Bridge:
       GraphIndex trace-conditioned bridge admission

BudgetedWriter
  -> decay old demand
  -> admit high-value candidates
  -> evict low-value or stale residents
```

Full cache utility:

```tex
V_t(d)
=
D_t(d)
+ G_t(d)
+ \beta_h H(d)
- \beta_r R(d, C_t)
```

Direct demand:

```tex
D_t(d)
=
\sum_{\tau < t}
\lambda^{t-\tau}
\frac{[s(q_\tau,d)]_+}{rank(q_\tau,d)^\eta}
```

Bridge demand:

```tex
G_t(B)
=
\sum_{\tau < t}
\lambda^{t-\tau}
\sum_{A,e}
s(q_\tau,A)
\cdot rel(q_\tau,A,e,B)
\cdot \frac{IDF(e)}{g(e)^\rho}
```

Important: `lambda` is not cosmetic. It encodes query shift. Old demand decays
so the cache can move.

## 6. Why Prefetch Is Not Query Rewrite

Reader objection:

```text
Shouldn't an agent just rewrite/decompose the query and retrieve B?
```

Answer:

```text
Yes, for the current query. That is reactive serving.
DRIP uses that trace as a cache admission signal for future demand.
```

Pipeline:

```text
current query misses hot tier
  -> agent/retriever exposes A
  -> A reveals bridge entity/relation B
  -> DRIP decides whether B is worth keeping warm
  -> later related queries hit hot tier instead of cold tier
```

If B is never reused, DRIP should not write it. This is why the benchmark must
measure reuse and cold-tier savings, not just current-query recall.

## 7. Required Metrics

Current metrics are necessary but not enough.

Keep:

```text
Recall@5 H1/H2
KB coverage H1/H2
Update cost
Maintenance retrieval cost
Serve retrieval cost
```

Add:

| Metric | Meaning |
|---|---|
| `cold_fetches_per_query` | How much work still falls through to cold tier |
| `reuse_hit_rate` | Fraction of later related queries served by prefetched/resident evidence |
| `first_exposure_cost` | Cold-tier work paid when A/B is first discovered |
| `amortized_cold_cost` | Cold cost spread over later reuse |
| `bridge_prefetch_precision` | Of prefetched B candidates, how many are later useful |
| `bridge_resident_survival` | Whether useful B remains resident until reuse |
| `wasted_prefetch_rate` | Prefetched candidates never used before eviction |
| `shift_recovery_lag` | Queries/windows needed to adapt after drift |
| `stale_resident_rate` | Cache budget occupied by old-window evidence |

This is where DRIP should beat ARC:

```text
not just higher recall on one query,
but lower repeated cold-tier pressure after demand shifts.
```

## 8. Dataset Plan

Use existing datasets, but transform them into cache workloads.

| Source | Use |
|---|---|
| WildChat / Google Trends | Motivation only: real query demand changes |
| StreamingQA | Natural temporal single-hop calibration |
| HotpotQA | Direct multi-hop visibility workload |
| 2WikiMultihopQA | Bridge-hidden workload |
| MuSiQue | Harder bridge/compositional appendix |
| LOCOMO / LongMemEval | Related-work contrast: memory benchmark, not primary cache benchmark |

New constructed benchmark:

```text
Query-Shifted RAG Cache Benchmark
```

Workload constructors:

```text
StaticStream:
  shuffled queries, no explicit phase structure

TemporalDriftStream:
  windows grouped by topic/entity/relation

BurstDriftStream:
  short high-density bursts with support reuse

BridgeReuseStream:
  q1 exposes A -> B
  later q2/q3/q4 need same B or neighboring bridge evidence
```

Construction rule for bridge reuse:

```text
group questions by:
  support title B
  bridge entity
  relation cue
  topic/entity cluster

then build windows where:
  early query reveals B through A
  later queries require B without necessarily sharing the same A
```

This creates the hot-tier claim. Without this, 2Wiki only diagnoses blindness.

## 9. Baselines

The comparison set should make the paper hard to dismiss.

| Baseline | Why |
|---|---|
| Static | No adaptation floor |
| FIFO / LRU | Classic cache policies |
| TinyLFU | Frequency-aware cache |
| GPTCacheStyle / Proximity | Semantic response/query reuse style |
| AgentRAGCache | ARC-style nearest-neighbor baseline |
| DRIP-Dense | Our query-visible semantic demand admission method |
| DRIP without GraphIndex | Shows direct channel only |
| DRIP without decay/detector | Shows drift handling matters |
| DRIP full | Route-aware + bridge-aware + shift-aware |
| Oracle | Headroom |

ARC must be a main baseline, not an appendix. It is the reviewer threat.

## 10. Implementation Sequence

Recommended order:

```text
Phase 1: Story and protocol
  - update motivation text to "shared hot-tier under query demand shift"
  - add benchmark matrix to experiment protocol
  - define shift types and metrics

Phase 2: Workload constructor
  - implement grouping by support title/entity/relation
  - generate StaticStream, TemporalDriftStream, BurstDriftStream, BridgeReuseStream
  - write dataset manifest for reproducibility

Phase 3: Evaluation plumbing
  - add cold_fetches_per_query
  - add reuse_hit_rate
  - add wasted_prefetch_rate
  - add shift_recovery_lag
  - log first exposure vs later reuse

Phase 4: Algorithm changes
  - add rank-aware direct demand using ARC-style DRF signal
  - make decay explicit in direct and graph evidence
  - make GraphIndex relation/path score answer benchmark failures
  - add writer gates for wasted prefetch control

Phase 5: Paper update
  - rewrite intro and method around two axes
  - position ARC as stable query-visible cache baseline
  - show DRIP wins on drift + bridge reuse, not just one-shot QA
```

## 11. What Is Not In Scope Yet

| Item | Reason |
|---|---|
| Full agent planner / query rewrite | Baseline or trace source, not the cache method |
| Synthesized memory objects like Mem0 | Different cache object, not original evidence chunk |
| Open-ended conversational memory benchmark as main eval | Useful related work, but not shared document hot-tier |
| Full knowledge graph construction | Too broad; use relation-ish local cues first |
| Claiming prefetch helps without reuse metrics | This is the failure mode to avoid |

## 12. CEO Review Verdict

Use **SELECTIVE EXPANSION** posture.

Hold the method scope: DRIP is still a bounded RAG hot-tier manager.

Expand the benchmark scope: add the two-axis benchmark and reuse/cold-cost metrics.

Do not expand into full memory management or agent planning. That path is bigger, but
it muddies the contribution and walks into Mem0/LOCOMO territory.

Recommended next action:

```text
Write the benchmark constructor before changing GraphIndex again.
```

If the constructor shows no support reuse, the method should be simplified.
If it shows reuse, then GraphIndex has a real job.
