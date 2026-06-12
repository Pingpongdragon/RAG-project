# EXPERIMENT_PROTOCOL

This document records the current experiment policy after implementation
unification.

## Core Implementation Rule

DRIP has one current runnable implementation:

```text
algorithms/drip/support_flow/
```

`algorithms/cache/ours/query_driven.py` is retained only as the minimal
direct-demand QueryDriven baseline used by motivation experiments.

| Layer | Current location | Notes |
|---|---|---|
| DRIPCore manager | `algorithms/drip/support_flow/__init__.py` | Route, evidence ledger, admission. |
| Query routing | `algorithms/drip/support_flow/query_router.py` | SINGLE / MULTI_DIRECT / BRIDGE. |
| Dense evidence | `algorithms/drip/support_flow/embedding_index.py` | Direct candidates. |
| Graph bridge evidence | `algorithms/drip/support_flow/graph_index.py` | Entity bridge scoring. |
| QueryDriven baseline | `algorithms/cache/ours/query_driven.py` | Direct demand only. |
| Detector utilities | `algorithms/drip/detection/` | Optional detector library. |

Deleted implementations:

```text
algorithms/cache/ours/routed_cache.py
algorithms/cache/ours/drip.py
```

## Strategy Registry

All runnable cache policies are registered in:

```text
algorithms/cache/registry.py
```

Current DRIP-related entries:

| Strategy | Meaning |
|---|---|
| `QueryDriven` | Minimal direct-demand baseline. |
| `QueryDrivenLoose` | Sensitivity variant. |
| `DRIP` | Current DRIPCore method. |
| `DRIPCore` / `SupportFlow` | Backward-compatible aliases for the current core. |

`RoutedCache` and `DRIPDetector` are no longer registry entries.

## Main Comparison Set

Use a compact cache-replacement comparison set:

```text
Static
LRU
TinyLFU
GPTCacheStyle
AgentRAGCache
QueryDriven
DRIP
Oracle
```

Additional paradigm references such as DocArrival, KnowledgeEdit,
OnDemandFetch, LogDrivenArrival, and MemGPTStyle can remain in appendices or
diagnostic runs.

## Core Metrics

Report at minimum:

- `Recall@5 H1/H2`
- `KB coverage H1/H2`
- `Update cost`
- `Maintenance retrieval cost`
- `Serve retrieval cost` when the policy fetches from the cold pool online

## Ablation Interpretation

| Comparison | Meaning |
|---|---|
| `DRIP` vs `QueryDriven` | Gain from routing plus graph bridge evidence. |
| `DRIP` vs `AgentRAGCache` | Whether entity bridge evidence beats query-distribution geometry on hidden bridge tasks. |
| `DRIP` vs `Oracle` | Remaining headroom from imperfect candidate scoring and admission. |
