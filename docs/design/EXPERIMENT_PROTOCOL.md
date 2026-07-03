# EXPERIMENT_PROTOCOL

This document records the current experiment policy after implementation
unification.

## Core Implementation Rule

DRIP has one current runnable implementation:

```text
algorithms/drip/cache_manager/
```

New runs, paper text, figures, and tables should use the two paper-facing
visibility variants below: `DRIP-QueryVisible` and `DRIP-QueryHidden`.

| Layer | Current location | Notes |
|---|---|---|
| DRIPCore manager | `algorithms/drip/cache_manager/__init__.py` | Drift control, route, evidence ledger, admission. |
| Drift detector | `algorithms/drip/detection/multi_agent_drift.py` | Multi-agent query-cache alignment detector; controls update aggressiveness only. |
| Paper variants | `algorithms/drip/cache_manager/drip.py` | `DRIP-QueryVisible`, `DRIP-QueryHidden`, and `DRIP`. |
| Query routing | `algorithms/drip/cache_manager/query_router.py` | `QUERY_VISIBLE` / `QUERY_HIDDEN`; qtype is diagnostic by default. |
| Dense/direct evidence | `algorithms/drip/cache_manager/embedding_index.py` | Direct candidates for query-visible evidence. |
| Hidden support | `algorithms/drip/cache_manager/support_completion.py` | Evidence-conditioned hidden-support completion and pair lease. |
| Graph metadata | `algorithms/drip/cache_manager/graph_index.py` | Entity metadata and postings for hidden-support completion. |
| Detector baselines | `algorithms/drip/detection/baseline_detectors.py` | Optional detector ablations. |

Earlier prototype implementations have been removed from the runnable cache
registry; `algorithms/drip/cache_manager/` is the only current DRIP code path.

## Strategy Registry

All runnable cache policies are registered in:

```text
algorithms/cache/registry.py
```

Current DRIP-related entries:

| Strategy | Meaning |
|---|---|
| `DRIP-QueryVisible` | Embedding/direct evidence channel only. |
| `DRIP-QueryHidden` | Query-visible A plus evidence-conditioned hidden-support completion B and pair lease. |
| `DRIP` | Main method alias for `DRIP-QueryHidden`. |

Old graph-walk/entity-rerank/decomposition variants are no longer registry entries.

Detector ablations should not change the query-visible/query-hidden router.
They should only change the controller signal `rho_t` and therefore the update
policy:

```text
NoDetector / fixed rho=0
MultiAgentDriftDetector
DriftLensDetector or ADWIN/MMD ablations
```

## Main Comparison Set

Use a compact cache-replacement comparison set:

```text
Static
LRU
TinyLFU
GPTCacheStyle
AgentRAGCache
DRIP-QueryVisible
DRIP-QueryHidden
Oracle
```

Additional paradigm references such as DocArrival, KnowledgeEdit,
OnDemandFetch, LogDrivenArrival, and MemGPTStyle can remain in appendices or
diagnostic runs.

For paper tables, label `AgentRAGCache` as `ARC`. Keep the runnable strategy
name as `AgentRAGCache` because that is the registry key. The DRIP family
should appear only as `DRIP-QueryVisible`, `DRIP-QueryHidden`, or the main
alias `DRIP`.

## Benchmark Design

The benchmark is two-dimensional. A method only answers the paper question if
it is evaluated across both axes:

| Axis | Levels | Why it matters |
|---|---|---|
| Demand dynamics | `static`, `random`, `temporal_drift`, `burst_drift` | Cache management is about a moving hot working set, not one independent QA batch. |
| Evidence visibility | `single_hop_visible`, `direct_multi_hop_visible`, `bridge_hidden` | Query-visible methods can solve the first two; bridge-hidden evidence needs a content/link signal after observing the first hop. |

Suggested dataset mapping:

| Visibility level | Dataset / filter | Primary purpose |
|---|---|---|
| `single_hop_visible` | StreamingQA or single-hop QA stream | Calibration: access history and `DRIP-QueryVisible` should both be competitive. |
| `direct_multi_hop_visible` | HotpotQA `comparison` | Query embedding can see both target entities; `DRIP-QueryVisible` should help through neighborhood demand. |
| `bridge_hidden` | 2WikiMultihopQA `bridge_comparison`, `compositional`, `inference` | Query exposes A, but reusable B must be inferred through content/entity links. |

Workload constructors:

| Workload | Meaning | Current runner |
|---|---|---|
| `cluster_shift` | Existing KMeans head/tail split with sudden/gradual/full-gradual drift. | `--workload cluster_shift --drift sudden` |
| `random_static` | Same query pool sampled uniformly with replacement/reshuffle; no intended drift. | `--workload random_static` |
| `temporal_bridge_reuse` | Bridge groups appear in phases: exposure queries reveal a shared support title, later reuse queries test whether the hot tier kept it warm. | `--workload temporal_bridge_reuse` |
| `burst_bridge_reuse` | Same bridge groups but compressed into bursts to test fast recovery and churn. | `--workload burst_bridge_reuse` |

`temporal_bridge_reuse` is the key prefetch benchmark. One-shot 2Wiki bridge
diagnoses query-only blindness, but it does not by itself justify prefetch.
Prefetch only becomes a cache-management win when a hidden evidence document is
reused by later queries, reducing future cold-tier pressure.

## Core Metrics

Report at minimum:

- `Recall@5 H1/H2`
- `KB coverage H1/H2`
- `Update cost`
- `Maintenance retrieval cost`
- `Serve retrieval cost` when the policy fetches from the cold pool online

Route-aware / reuse-aware metrics:

- `cold_fetches_per_query`: average number of gold support documents missing
  from the effective hot tier before serve-time retrieval.
- `reuse_hit_rate`: among reuse-labeled queries, fraction whose reuse support
  document is resident before the query is evaluated.
- `reuse_queries`: number of reuse-labeled queries used for the rate.
- `first_exposure_cost`: average missing gold documents on exposure queries.
- `amortized_cold_cost`: average missing gold documents after the first
  exposure within each reuse group.
- `bridge_prefetch_precision`: among bridge-channel candidates/writes, fraction
  that are gold or future-reuse support. Use DRIP `bridge_log` for diagnostics.
- `bridge_resident_survival`: whether the reuse support remains resident from
  exposure to later reuse.
- `wasted_prefetch_rate`: written bridge candidates that are never reused later
  in the constructed stream.
- `shift_recovery_lag`: windows after a demand shift until recall or reuse hit
  rate reaches a configured fraction of its final plateau.
- `stale_resident_rate`: resident documents tied only to old demand after the
  shift.

The first implemented pass records `cold_fetches_per_query`,
`reuse_hit_rate`, `reuse_queries`, `first_exposure_cost`, and
`amortized_cold_cost` directly in `motivation_2/run.py`. The remaining metrics
are diagnostic targets for the next GraphIndex iteration.

## Ablation Interpretation

| Comparison | Meaning |
|---|---|
| `DRIP-QueryHidden` vs `DRIP-QueryVisible` | Gain from evidence-conditioned hidden-support completion plus pair retention. |
| `DRIP-QueryHidden` vs `AgentRAGCache` | Whether hidden-support evidence beats query-distribution geometry on hidden-support tasks. |
| `DRIP-QueryHidden` vs `Oracle` | Remaining headroom from imperfect candidate scoring and admission. |

GraphIndex should be changed only after these metrics expose a concrete
failure mode. For example:

| Failure observed | Component to inspect |
|---|---|
| High bridge candidate volume but low future reuse hit | GraphIndex relation/entity gates and candidate precision. |
| Good bridge candidates but low writes | Budgeted writer / priority threshold. |
| Good writes but poor survival until reuse | Eviction priority / serve decay. |
| Strong recall but high cold cost | Serve-time fallback is masking bad hot-tier residency. |
