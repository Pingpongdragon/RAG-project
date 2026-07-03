# Code Explained

This document points to the current DRIP code path. Older pipeline and
detector-wrapped prototypes are retired; the active cache-policy path is the
`algorithms/drip/cache_manager/` package.

## Active Code Path

| File | Role |
|---|---|
| `algorithms/drip/cache_manager/__init__.py` | DRIPCore cache manager: drift control, route, evidence ledger, support-priority admission. |
| `algorithms/drip/detection/multi_agent_drift.py` | Multi-agent drift detector/controller for update aggressiveness. |
| `algorithms/drip/cache_manager/query_router.py` | Deterministic QUERY_VISIBLE / QUERY_HIDDEN route selection. |
| `algorithms/drip/cache_manager/embedding_index.py` | Dense direct candidates. |
| `algorithms/drip/cache_manager/graph_index.py` | Entity metadata and postings used by bridge completion. |
| `algorithms/drip/cache_manager/support_completion.py` | Hidden-support completion and pair lease retention. |
| `algorithms/drip/cache_manager/drip.py` | Paper-facing policy classes: `DRIP-QueryVisible`, `DRIP-QueryHidden`, and `DRIP`. |
| `algorithms/drip/cache_manager/config.py` | DRIPCore hyperparameters. |
| `algorithms/drip/detection/baseline_detectors.py` | Detector ablation baselines. |

## Current Algorithm

The paper-facing variants separate query-visible and query-hidden evidence:

| Strategy | Enabled channels |
|---|---|
| `DRIP-QueryVisible` | Dense/direct evidence only. |
| `DRIP-QueryHidden` | Query-visible A plus evidence-conditioned hidden-support completion B and pair lease. |
| `DRIP` | Alias for the current `DRIP-QueryHidden` policy. |

For each query window:

1. Score current queries against the resident cache.
2. Detect multi-agent drift severity from query-cache alignment.
3. Use severity to adjust demand decay, write cap, and admission margin.
4. Route each under-covered query as QUERY_VISIBLE or QUERY_HIDDEN.
5. Credit dense/direct evidence for QUERY_VISIBLE.
6. Credit hidden-support evidence for QUERY_HIDDEN: retrieve missing B conditioned on resident/easy A.
7. Maintain serve evidence for resident documents that covered queries.
8. Protect completed A+B support pairs with pair lease.
9. Replace resident documents only when candidate demand beats the weakest resident support priority.

The strategy registry entry is `STRATEGY_FACTORIES["DRIP"]` in
`algorithms/cache/registry.py`.
