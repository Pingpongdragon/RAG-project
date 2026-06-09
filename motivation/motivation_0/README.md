# Motivation 0 — Query Demand Drift (Fig. 0)

> 对应 `motivation.tex` 的 **Fig.~\ref{fig:motivation0}** —
> *"Query demand drift persists independent of corpus updates."*
> 论证 Intro §*The Prevalence of Query Distribution Drift*：真实多用户负载的需求分布随时间漂移，
> 即使底层 corpus 完全固定，hot-tier 的 working set 也必须随需求变化。

## 这一层在叙事中的角色

这是三级 system audit 的**入口证据（macroscopic / L1）**：用真实世界数据说明
"文档不变、需求在变"是一个 first-class 的普遍现象，从而确立 shared hot-tier
必须 demand-aware 的动机。它不涉及任何 cache 策略对比，只做现象刻画。

## 图结构

| 子图 | 数据 | 论证 |
|---|---|---|
| (a) | WildChat-1M 月级 topic mixture（32.5K ChatGPT 对话） | 集体话题在月尺度上显著重排 |
| (b) | Google Trends 多年主流 LLM 应用域权重 | 用户注意力在多年尺度上持续被重新加权 |

> 规划中的 (c) 子图（Agent 链式查询占比，弥合宏观↔微观断层）见
> [../docs/narrative/NEXT_STEPS_AUDIT_TODO.md](../docs/narrative/NEXT_STEPS_AUDIT_TODO.md) §2，本轮未实现。

## 文件

| 文件 | 角色 |
|---|---|
| `plot_mo0_drift.py` | 主绘图脚本：读 `data/` → 输出 `figures/` 与 `paper_figs/intro/fig0_*.pdf` |
| `plot_config.py` | 配色 / 字号等绘图常量 |
| `data/wildchat_sampled.json` | WildChat 采样对话（topic 分类输入） |
| `data/google_trends_cache.json` | Google Trends 抓取缓存（避免重复请求 pytrends） |
| `figures/` | 本目录内部预览图 |

> 论文最终引用的 PDF 在 `../paper_figs/intro/fig0_intro_user_query_topic_drift.pdf`。

## 复现

```bash
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
cd /home/jyliu/RAG-project/motivation/motivation_0
$PY plot_mo0_drift.py     # 重新分类 WildChat + 读 Trends 缓存 → 重画 Fig 0
```
