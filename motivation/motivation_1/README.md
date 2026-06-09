# Motivation 1 — Single-Hop Calibration (Fig. 1)

> 对应 `motivation.tex` 的 **Fig.~\ref{fig:streamingqa}** —
> *"Single-hop calibration under natural demand drift."*
> 论证 Intro §*A signal-source audit of existing policies*：在单跳 StreamingQA 上做
> **signal-source 审计**，说明各类 cache 家族（document-centric / semantic-response /
> recency-frequency / agent-reactive）在哪种 signal 下够用、哪种不够用。

## 这一层在叙事中的角色

这是三级 system audit 的**校准层（L1，单跳）**。核心 claim 是诚实的划界，而非 showcase：

- 当每条 query 通常直接指向所需证据时，**access history 已经是强信号**。
- StreamingQA（14 年新闻 QA，5 个历史 era R1–R5，天然 demand drift）上 **LRU = 28.8% Recall@5**，
  SemFlow = 27.1% 略低但可接受 → 说明 semantic admission 在单跳上**不破坏** workload，但也不是增益来源。
- 把 `Recency-TTL`（oracle 文档年份）作为 document timestamp 信号的天花板：仍然崩到接近 0 →
  证明 timestamp 缺乏 **intra-era resolution**（同一 era 数千篇同日候选无法区分）。
- `OnDemandFetch` 恢复质量，但代价是大量 logical cold-tier fetch（≈29 次/query）→ 是 fallback 不是常驻策略。

> 真正的增益场景在下一层（Fig 2 / `motivation_2`）的静态多跳里才出现。

## 图与数据对齐

| Fig 1 子图 | 内容 | 数据文件 |
|---|---|---|
| (a) | 各策略 Recall@5（含 LRU / SemFlow / Recency-TTL oracle） | `data/results_streamingqa_temporal.json` |
| (b) | OnDemandFetch 的 logical cold-tier fetch 计数（cold-tier 压力代理） | 同上 |
| (c) | 各更新策略每窗口维护开销（4–10ms CPU 模拟） | 同上 |

> 绘图脚本不在本目录，而在 [../plotting/plot_motivation_v2.py](../plotting/plot_motivation_v2.py) 的 `fig1()`，
> 它读取 `motivation_1/data/results_streamingqa_temporal.json` → 输出 `../paper_figs/intro/fig1_*.pdf`。

## 文件

| 文件 | 角色 |
|---|---|
| `run.py` | 实验入口；`--drift temporal --datasets streamingqa` 跑 StreamingQA era 流 |
| `strategies.py` | 全部策略实现（Static / LRU / TinyLFU / MissLRU / SemFlow(=QueryDriven) / OnDemandFetch / Recency-TTL / Oracle …） |
| `loaders.py` | 通用数据集 loader（HotpotQA / FEVER / SQuAD …，供历史实验用） |
| `loaders_temporal.py` | StreamingQA 的时间分段 loader（R1–R5 era 流） |
| `config.py` | 数据集配置（n_source / kb 预算公式等） |
| `utils.py` | embedding 缓存、检索 helper、drift 流构造 |
| `NOTES.md` | QDC 生效的三个充要条件（Cond-A/B/C），是 `docs/experiments/DATASET_ANALYSIS.md` 的现场笔记 |

## 复现 Fig 1 数据

```bash
PY=/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python
cd /home/jyliu/RAG-project/motivation/motivation_1
$PY run.py --drift temporal --datasets streamingqa \
   --output data/results_streamingqa_temporal.json
# 然后回到 ../plotting 跑 plot_motivation_v2.py 的 fig1()
```

## 历史产物（非当前 tex 主线）

`figures/` 与 `data/` 里还保留了早期单跳实验（HotpotQA-comp / FEVER / 2wiki，sudden/gradual，
50w/80w/100w 多个规模）的结果与图，以及旧的 `mo1_{sudden,gradual}*.pdf` 2×2 图。
这些是 StreamingQA 校准定型前的探索证据，**当前 Fig 1 不再使用**，但实验数据按要求保留未删。
`data/*.json.bak.*` 为历史备份。
