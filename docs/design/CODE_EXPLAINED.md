# Code Explained

This document points to the current DRIP code path. Older notes about
`DRIPPipeline`, `KBUpdateAgent`, `DRIPKBCurator`, `RoutedCache`, and
`algorithms/cache/ours/drip.py` refer to retired implementations.

## Active Code Path

| File | Role |
|---|---|
| `algorithms/drip/support_flow/__init__.py` | DRIPCore cache manager: route, evidence ledger, support-priority admission. |
| `algorithms/drip/support_flow/query_router.py` | Deterministic SINGLE / MULTI_DIRECT / BRIDGE route selection. |
| `algorithms/drip/support_flow/embedding_index.py` | Dense direct candidates. |
| `algorithms/drip/support_flow/graph_index.py` | Graph bridge evidence with IDF, entity-degree penalty, novelty, and complementarity. |
| `algorithms/drip/support_flow/config.py` | DRIPCore hyperparameters. |
| `algorithms/cache/ours/query_driven.py` | Retained minimal SemFlow / QueryDriven direct-demand baseline. |
| `algorithms/cache/ours/config.py` | QueryDriven baseline hyperparameters. |
| `algorithms/drip/detection/` | Optional detector utilities, not the active cache-policy entry point. |

## Final Algorithm

For each query window:

1. Score current queries against the resident cache.
2. Route each under-covered query as SINGLE, MULTI_DIRECT, or BRIDGE.
3. Credit dense/direct evidence for SINGLE and MULTI_DIRECT.
4. Credit GraphIndex bridge evidence for BRIDGE.
5. Maintain serve evidence for resident documents that covered queries.
6. Replace resident documents only when candidate demand beats the weakest resident support priority.

The strategy registry entry is `STRATEGY_FACTORIES["DRIP"]` in
`algorithms/cache/registry.py`.
