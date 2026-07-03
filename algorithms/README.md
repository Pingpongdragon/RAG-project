# algorithms/ - cache policies and detector utilities

This directory now has one canonical DRIP implementation.

## Canonical Method

The paper method is the DRIP cache manager support-cache policy:

- `algorithms/drip/cache_manager/__init__.py` - DRIPCore cache manager.
- `algorithms/drip/detection/multi_agent_drift.py` - multi-agent drift detector/controller.
- `algorithms/drip/cache_manager/query_router.py` - query-visible/query-hidden route decisions.
- `algorithms/drip/cache_manager/embedding_index.py` - dense/direct evidence.
- `algorithms/drip/cache_manager/graph_index.py` - entity metadata/index used by hidden-support completion.
- `algorithms/drip/cache_manager/support_completion.py` - hidden-support completion and pair lease.
- `algorithms/drip/cache_manager/config.py` - DRIP cache-manager hyperparameters.

Experiments should instantiate DRIP through:

```python
from algorithms.cache.registry import STRATEGY_FACTORIES
strategy = STRATEGY_FACTORIES["DRIP"](doc_pool, doc_embs, title_to_idx)
```

## Retired Cache/Ours Implementations

The standalone direct-demand baseline has been folded into DRIP's direct path.
Old bridge, direct-only, and detector-wrapped variants were removed to avoid
confusing them with the current DRIP path. Use `STRATEGY_FACTORIES["DRIP"]` for
the paper method.

## Detector Library

`algorithms/drip/` also keeps detector interfaces and detector baselines:

- `algorithms/drip/interfaces.py`
- `algorithms/drip/detection/multi_agent_drift.py`
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
Detect multi-agent query-cache drift severity
  -> adjust demand decay, write cap, and admission margin
  -> route each query as QUERY_VISIBLE or QUERY_HIDDEN
  -> accumulate dense or hidden-support evidence into demand
  -> maintain serve evidence for resident documents
  -> protect completed A+B support pairs with pair lease
  -> admit candidate c over resident e iff demand gain beats support priority
```
