# Motivation 材料

当前论文首先定义两个研究场景前提：

1. **活跃请求的 domain mixture 会随时间变化**；
2. **每个活跃 domain 内存在大量可复用的相同或相关证据请求**。

第一个前提导致历史 resident set 在换域后变旧，第二个前提保证
cache 本身有价值。二者交集定义为 `cacheable domain-mixture drift`。
此后才使用完整的方法必要性证据链：

1. RAGCache/AgentRAGCache 证明固定领域局部性下 cache 有价值；
2. 同一 SQuAD 源池在 domain schedule 改变后出现 residency mismatch；
3. DRIP 在同容量下联合报告 hit、writes 和 cold reads，验证是否恢复。

正式实验代码已经迁移到：

| 场景 | 目录 |
|---|---|
| Direct evidence、受控 evidence-domain shift | [`experiments/direct/`](../experiments/direct/) |
| Hidden evidence、多跳边界 | [`experiments/hidden/`](../experiments/hidden/) |
| 真实 agent/session/access trace | [`experiments/agent/`](../experiments/agent/) |

本目录保留的内容：

| 目录 | 内容 |
|---|---|
| `motivation_0/` | WildChat、Google Trends 等宏观需求漂移证据 |
| `plotting/` | 引言图和历史结果绘图脚本 |
| `paper_figs/` | 论文引用的 motivation 图片 |
| `docs/` | 历史分析、叙事和验证记录 |
| `archive/` | 不再属于当前主线的旧快照 |

因此，`motivation/` 不再包含 cache policy、正式 runner 或主实验结果入口。

## 论文中的两张 motivation 图

1. `fig1_domain_mixture_drift` 保留原 WildChat + Google Trends 图，说明真实聚合
   请求中 domain mixture 会变化。它支持“这是现实运行条件”，不支持
   “所有部署都必然漂移”。
2. `fig2_cacheable_domain_drift` 联合展示域内证据复用和 shift-induced
   residency mismatch。MIND 使用更严格的跨用户同新闻复用率，SQuAD/WoW
   使用受控域内 evidence reuse。MT-RAG 仅作低复用负对照，不进入动机主图。

论证不要求自然时间戳：单 agent 任务切换和多 agent 活跃比例变化都可
形成有因果顺序的 domain mixture。但 SQuAD/WoW/FEVER 始终标为 controlled，
只有 MIND 标为 natural chronological trace。

正式输出为：

```text
motivation_0/figures/user_query_topic_drift.{pdf,png}
paper_figs/intro/fig2_cacheable_domain_drift.{pdf,png}
```

图二配套数值审计为：

```text
docs/verification/FIG2_CACHEABLE_DOMAIN_DRIFT.md
```

重画并导出两张图：

```bash
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
  motivation/motivation_0/plot_mo0_drift.py
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python \
  motivation/plotting/plot_intro_domain_adapt.py
```
