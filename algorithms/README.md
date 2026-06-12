# algorithms/ - cache policies and detector utilities

This directory now has one canonical DRIP implementation.

## Canonical Method

The paper method is the DRIP-Core support-cache policy:

- `algorithms/drip/support_flow/__init__.py` - DRIPCore cache manager.
- `algorithms/drip/support_flow/query_router.py` - route decisions.
- `algorithms/drip/support_flow/embedding_index.py` - dense/direct evidence.
- `algorithms/drip/support_flow/graph_index.py` - graph bridge evidence.
- `algorithms/drip/support_flow/config.py` - support-flow hyperparameters.

Experiments should instantiate DRIP through:

```python
from algorithms.cache.registry import STRATEGY_FACTORIES
strategy = STRATEGY_FACTORIES["DRIP"](doc_pool, doc_embs, title_to_idx)
```

## Retired Cache/Ours Implementations

`algorithms/cache/ours/query_driven.py` remains as the minimal SemFlow /
QueryDriven motivation baseline. The old bridge and detector-wrapped variants
were removed to avoid confusing them with the current GraphIndex scoring path.
Do not import `RoutedCache` or `DRIPDetector` as current method modules.

## Detector Library

`algorithms/drip/` also keeps detector interfaces and detector baselines:

- `algorithms/drip/interfaces.py`
- `algorithms/drip/detection/drift_detector.py`
- `algorithms/drip/detection/baseline_detectors.py`

The old `DRIPPipeline`, `KBUpdateAgent`, `DRIPKBCurator`, interest-model
pipeline, and cache/ours variants were removed because they represented retired
or simplified designs.

## Baselines

Cache baselines live under `algorithms/cache/`:

- `recency/` — LRU, FIFO, temporal variants.
- `frequency/` — TinyLFU.
- `semantic/` — GPTCache-style semantic cache.
- `paradigm_ref/` — Static, DocArrival, KnowledgeEdit, AgentRAGCache, etc.
- `oracle/` — Belady-style upper bound.

## Current DRIP Algorithm

The current algorithm is:

```text
Route each query as single, multi-direct, or bridge
  -> accumulate dense or graph evidence into demand
  -> maintain serve evidence for resident documents
  -> admit candidate c over resident e iff demand gain beats support priority
```
