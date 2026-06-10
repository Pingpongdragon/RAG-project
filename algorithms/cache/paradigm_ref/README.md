# paradigm_ref/ — 范式参照 + ARC baseline

混合目录：大部分是 motivation 阶段的"范式参照"（展示不同更新范式长什么样，**正文不必全留**），
外加一个真正的主 baseline `AgentRAGCache`。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `agent_rag_cache.py` | `AgentRAGCache` | **Agent-RAG ARC (2511.02919)**：DRF(秩-距离-频率) + hubness 几何打分，无漂移检测 | ✅ **主 baseline**（最近邻威胁） |
| `agent_rag_cache.py` | `AgentRAGCache(use_hubness=False)` | ARC(w/o hubness) 消融，注册名 `AgentRAGCache_NoHub` | ✅ **消融**（论文 Table 自带这行） |
| `supply_side.py` | `Static` | 不更新（下界） | ✅ **下界**，必留 |
| `supply_side.py` | `DocArrival` | 文档到达驱动（LightRAG/HippoRAG 风格） | ⚪ 范式参照，可选 |
| `supply_side.py` | `KnowledgeEdit` | 知识编辑驱动（RECIPE 风格） | ⚪ 范式参照，可选 |
| `supply_side.py` | `RandomFIFO` | 随机新文档 FIFO 替换 | ⚪ 朴素对照，可选 |
| `reactive.py` | `OnDemandFetch` | 被动逐条 fetch（CRAG 风格） | ⚪ 可选（对照"主动预取 vs 被动"） |
| `reactive.py` | `LogDrivenArrival` | 日志驱动周期性到达 | ⚪ 可选 |
| `reactive.py` | `MemGPTStyle` | MemGPT 分页风格 | ⚪ 可选 |

## AgentRAGCache 参数（全部来自 ARC 论文）
α=0.4, β∈{0.7,0.15,0.2}按数据集, τ=0.2, K=50；HUB_K=10 是论文未给、我们补的默认。
详见 `docs/design/CODE_EXPLAINED.md §8`。

> ⚠️ 当前实跑 escalation 触发过频 → cache churn → 数字偏低。跑正式实验前需调 τ / escalation 粒度，
> 否则会被读成"故意调废 baseline"。
