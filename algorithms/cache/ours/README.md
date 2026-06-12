# ours/ - minimal motivation baselines

The current DRIP implementation lives in `algorithms/drip/support_flow/`.
This directory only keeps the direct-demand QueryDriven baseline used by older
motivation experiments.

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `query_driven.py` | `QueryDriven` | Direct demand-ledger admission without graph bridge expansion | baseline |
| `query_driven.py` | `QueryDrivenLoose` | Wider, looser sensitivity variant | sensitivity only |
| `config.py` | `QueryDrivenConfig` | QueryDriven hyperparameters | baseline config |

Retired `routed_cache.py` and `drip.py` were deleted. Bridge evidence should be
implemented and studied through `algorithms/drip/support_flow/graph_index.py`.
