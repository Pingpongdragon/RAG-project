# Shared Experiment Protocols

本目录只保存 `direct/`、`hidden/` 和 `agent/` 共同使用的实验协议，不实现 cache
策略，也不属于线上 RAG 主流程。

| 文件 | 职责 |
|---|---|
| `factorized_workload.py` | 在 sparse gold-evidence 空间构造 one-shot、gradual、recurring、shuffled 与 stationary 受控流 |
| `stream_protocol.py` | 自然时间采样、causal warm-up、初始 cache 和泄漏/reuse/drift 审计 |
| `session_workload.py` | 在保持会话内 turn order 的前提下构造 MT-RAG session replay |

边界规则：

1. 原始数据始终保存在 `datasets/`。
2. 各分支 loader 只负责格式适配和官方顺序，不制造有利于某个策略的流。
3. 受控 drift 只能在本目录构造，latent regime/gold support 不得传给在线策略。
4. `algorithms/` 不能 import 本目录；只能由实验 runner 调用算法。

