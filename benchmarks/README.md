# Benchmarks and Audits

本目录回答局部机制问题，不承担论文正式 direct/hidden/agent 主实验：

| 文件 | 回答的问题 |
|---|---|
| `audit_semantic_pages.py` | semantic page 是否比 document LRU 提供更多 reuse？ |
| `audit_region_shadow_prefetch.py` | region forecast 与 shadow prefetch 是否具有因果收益？ |
| `faiss_hot_tier.py` | FAISS hot/cold search 与 replacement 的真实延迟是多少？ |
| `run_controlled_topic_trace.py` | TopicDynamics 在 oracle evidence-access trace 上是否可行？ |
| `tune_controlled_topic.py` | 仅用 calibration prefix 调整受控 trace 参数 |

`archive_legacy/` 是旧的 WoW/Hotpot 合成漂移框架与历史结果，仅供追溯。它不属于
当前论文主链，不应被新的实验代码 import。

