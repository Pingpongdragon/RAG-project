# Algorithm Design

The old `DRIPPipeline -> KBUpdateAgent -> DRIPKBCurator` design has been
retired. It used an engineering-style action ladder (`NoOp`, `Mild`,
`Aggressive`) and fixed cache-size rewrite ratios, which is no longer the paper
algorithm.

Use these files as the current source of truth:

- [FINAL_METHOD.md](FINAL_METHOD.md) — paper-level method definition.
- [DRIP_ALGORITHM_EXPLAINED.md](DRIP_ALGORITHM_EXPLAINED.md) — implementation-facing explanation.
- [algorithms/drip/support_flow/__init__.py](../../algorithms/drip/support_flow/__init__.py) — canonical DRIPCore implementation.
- [algorithms/drip/support_flow/query_router.py](../../algorithms/drip/support_flow/query_router.py) — route selection.
- [algorithms/drip/support_flow/graph_index.py](../../algorithms/drip/support_flow/graph_index.py) — bridge evidence scoring.

Current method summary:

```text
QueryRouter
  -> dense/direct evidence or GraphIndex bridge evidence
  -> demand + serve evidence ledger
  -> support-priority admission
```

The retired `algorithms/cache/ours` simplified cache implementation has been
deleted to keep the codebase aligned with the latest DRIPCore path.
