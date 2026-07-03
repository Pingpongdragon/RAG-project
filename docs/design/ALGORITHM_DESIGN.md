# Algorithm Design

The old `DRIPPipeline -> KBUpdateAgent -> DRIPKBCurator` design has been
retired. It used an engineering-style action ladder (`NoOp`, `Mild`,
`Aggressive`) and fixed cache-size rewrite ratios, which is no longer the paper
algorithm.

Use these files as the current source of truth:

- [FINAL_METHOD.md](FINAL_METHOD.md) — paper-level method definition.
- [DRIP_ALGORITHM_EXPLAINED.md](DRIP_ALGORITHM_EXPLAINED.md) — implementation-facing explanation.
- [algorithms/drip/cache_manager/__init__.py](../../algorithms/drip/cache_manager/__init__.py) — canonical DRIPCore implementation.
- [algorithms/drip/detection/multi_agent_drift.py](../../algorithms/drip/detection/multi_agent_drift.py) — multi-agent drift controller.
- [algorithms/drip/cache_manager/drip.py](../../algorithms/drip/cache_manager/drip.py) — paper-facing `DRIP-QueryVisible`, `DRIP-QueryHidden`, and `DRIP` variants.
- [algorithms/drip/cache_manager/query_router.py](../../algorithms/drip/cache_manager/query_router.py) — query visibility selection.
- [algorithms/drip/cache_manager/support_completion.py](../../algorithms/drip/cache_manager/support_completion.py) — evidence-conditioned hidden-support completion and pair lease.
- [algorithms/drip/cache_manager/graph_index.py](../../algorithms/drip/cache_manager/graph_index.py) — graph metadata for bridge completion.

Current method summary:

```text
MultiAgentDriftDetector -> update aggressiveness rho_t
QueryRouter
  -> query-visible dense/direct evidence or query-hidden support evidence
  -> demand + serve + pair-lease evidence
  -> support-priority admission
```

Current ablations:

| Strategy | Enabled channels |
|---|---|
| `DRIP-QueryVisible` | Dense/direct channel only. |
| `DRIP-QueryHidden` | Query-visible A plus evidence-conditioned hidden support B and pair lease. |
| `DRIP` | Current main alias for `DRIP-QueryHidden`. |
