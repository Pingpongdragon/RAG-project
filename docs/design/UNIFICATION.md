# Unification Note

The implementation has been unified around one DRIP algorithm:

```text
algorithms/drip/support_flow/
```

The old split between `algorithms/drip` as a full pipeline and
`algorithms/cache/ours` as the experiment backend is gone. `QueryDriven`
remains under `algorithms/cache/ours` as the minimal motivation baseline; the
old `RoutedCache` and detector-wrapped DRIP variants have been deleted so there
is only one current DRIP path.

Use these component names in the paper and experiments:

| Component | Current implementation |
|---|---|
| Route-aware cache manager | `algorithms/drip/support_flow/__init__.py` |
| Query routing | `algorithms/drip/support_flow/query_router.py` |
| Dense/direct evidence | `algorithms/drip/support_flow/embedding_index.py` |
| Graph bridge evidence | `algorithms/drip/support_flow/graph_index.py` |
| Minimal SemFlow baseline | `algorithms/cache/ours/query_driven.py` |
| Drift detector | `algorithms/drip/detection/drift_detector.py` |

Retired names such as `KBUpdateAgent`, `DRIPPipeline`, and `DRIPKBCurator` are
legacy implementation details and should not appear as current method modules.
Retired `RoutedCache` and `DRIPDetector` cache variants should not be used as
current experiment strategies.
