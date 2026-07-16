# DRIP 正式实验

本目录保存论文最终实验，不再使用 `motivation_1/2` 这种阶段性名称。

| 目录 | 场景 | 主入口 |
|---|---|---|
| `direct/` | 自然时间漂移与 query-visible direct evidence 漂移 | `python experiments/direct/run.py ...` |
| `hidden/` | 多跳与 query-hidden evidence diagnostic | `python experiments/hidden/run.py ...` |
| `agent/` | 真实 agent/session/access trace | `python experiments/agent/run_access_trace.py ...` |

三个分支共享：

```text
algorithms/cache/registry.py       策略注册表
algorithms/drip/                   唯一 DRIP 实现
experiments/common/stream_protocol.py      因果采样与协议审计
experiments/common/factorized_workload.py  受控 evidence-regime 流
experiments/common/session_workload.py     会话流协议
```

三个目录不强行合成一个巨型 runner：`direct/` 主要是 dense/direct cache
evaluation，`hidden/` 还包含 query decomposition 与 graph retrieval，`agent/` 则
执行严格因果的时间或会话 replay。它们共享策略接口，但保留不同的数据协议。

`motivation/` 现在只保存问题必要性、引言图、历史分析和绘图材料，不再承担正式实验入口。
