# Unification Note

The implementation has been unified around one DRIP algorithm:

```text
algorithms/drip/cache_manager/
```

There is only one current DRIP cache-policy path. The paper-facing variants
separate query-visible and query-hidden evidence over the same manager:

```text
DRIP-QueryVisible = dense/direct channel only
DRIP-QueryHidden  = query-visible A + hidden-support completion B
DRIP              = current main alias for DRIP-QueryHidden
```

Use these component names in the paper and experiments:

| Component | Current implementation |
|---|---|
| Route-aware cache manager | `algorithms/drip/cache_manager/__init__.py` |
| Multi-agent drift controller | `algorithms/drip/detection/multi_agent_drift.py` |
| Paper variants | `algorithms/drip/cache_manager/drip.py` |
| Query routing | `algorithms/drip/cache_manager/query_router.py` |
| Dense/direct evidence | `algorithms/drip/cache_manager/embedding_index.py` |
| Hidden-support completion | `algorithms/drip/cache_manager/support_completion.py` |
| Graph metadata | `algorithms/drip/cache_manager/graph_index.py` |
| Detector baselines | `algorithms/drip/detection/drift_detector.py`, `algorithms/drip/detection/baseline_detectors.py` |

Retired names such as `KBUpdateAgent`, `DRIPPipeline`, and `DRIPKBCurator` are
legacy implementation details and should not appear as current method modules.
Detector-wrapped cache variants should not be used as current experiment
strategies.
