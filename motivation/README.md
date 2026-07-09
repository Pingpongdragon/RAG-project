# Motivation — Cost-Aware RAG Hot-Tier Adaptation under Evidence Drift

> 当前 CostAwareDRIP 主线实验、推荐命令、结果读取方式统一看
> [COSTAWARE_RUNBOOK.md](COSTAWARE_RUNBOOK.md)。历史 `motivation_1/2`
> 目录名保留，但论文主线按 E1/E2/E3 组织。

本目录是论文 Introduction / Motivation 部分的全部材料：LaTeX 源文、三级 system audit 的
实验代码与数据、绘图脚本、叙事文档，以及最终算法设计。

## 一句话主线

> 在多用户/agent 共享 RAG 部署里，L2 全量语料可以近似 append-only，但 L1 hot-tier
> 缓存受容量、写入、换出和延迟成本约束。本文用分层实验说明：真实 temporal drift
> 下 recency 是强边界，controlled direct-evidence topic drift 下需要主动写入新证据，
> 而真正的主问题是 **cost-aware admission**：什么时候值得把 L2 检索到的新 evidence
> 写进 L1，并避免 cache churn。
> 当前可审阅方案见 [docs/design/COST_AWARE_DRIP_EXPERIMENT_PLAN.md](../docs/design/COST_AWARE_DRIP_EXPERIMENT_PLAN.md)。

## 分层实验逻辑

| 层级 | 论文图 | 现象 | 目录 | 结论 |
|---|---|---|---|---|
| **L1** 宏观需求漂移 | Fig 0 | WildChat / Google Trends topic 重排 | [motivation_0/](motivation_0/) | 漂移是 first-class 普遍现象 |
| **S1** 真实 temporal visible drift | Fig 1 | StreamingQA 14 年 era 流 | [motivation_1/](motivation_1/) | LRU/FIFO 很强；这是 visible temporal 的边界，不是 hidden evidence 问题 |
| **S2** controlled direct-evidence topic drift | Fig 2 | 2Wiki/Hotpot/MuSiQue topic shift | [motivation_2/](motivation_2/) | 新 topic evidence 需要主动 admission；比较质量-成本 tradeoff |
| **S3** cost/churn stress | Fig 3 / appendix | KB budget sweep + write budget/churn | [motivation_1/](motivation_1/), [motivation_2/](motivation_2/) | 主张不是“写得更多”，而是同等质量下写得更少/更稳 |
| **Diagnostic** hidden evidence | appendix | 2Wiki `Q→A→B` / bridge workloads | [motivation_2/](motivation_2/) | 只作为 direct-evidence policy 的边界；不作为主贡献 |

## 顶层文件 / 目录

| 项 | 角色 |
|---|---|
| `motivation.tex` | 论文 Intro/Motivation 源文（图叙事注释见文件头部） |
| `references.bib` | 参考文献（缺失 cite key 见 [docs/narrative/NEXT_STEPS_AUDIT_TODO.md](docs/narrative/NEXT_STEPS_AUDIT_TODO.md) §3） |
| `motivation_0/` | **Fig 0** — query demand drift（WildChat / Trends） |
| `motivation_1/` | **Fig 1** — StreamingQA real temporal visible drift + recency boundary |
| `motivation_2/` | **Fig 2/3** — controlled direct-evidence topic drift + cost/churn stress |
| `plotting/` | 论文图绘制：`plot_motivation_v2.py`（fig1/fig2）、benchmark 脚本 |
| `paper_figs/intro/` | 论文最终引用的 `fig0/1/2_*.pdf`（tex `\includegraphics` 指向这里） |
| `baselines_src/` | 第三方 baseline 源码参考（GPTCache / MemGPT），只读 |
| `docs/` | 全部文档（叙事 / 实验 / 设计 / 文献 / 复现）→ 见 [docs/README.md](docs/README.md) |
| `archive/` | 历史备份、日志、过时文档、旧快照 → 见 [archive/README.md](archive/README.md) |

## 文档导航（docs/）

- **叙事 & 评审** [docs/narrative/](docs/narrative/) — 故事线、评审意见、会议记录、待办
- **实验总结** [docs/experiments/](docs/experiments/) — 各图结果、数据集诊断、评审回应
- **算法设计** [docs/design/](docs/design/) — **[COST_AWARE_DRIP_EXPERIMENT_PLAN.md](../docs/design/COST_AWARE_DRIP_EXPERIMENT_PLAN.md)**（当前主线）+ 历史 DRIP 设计文档
- **文献背景** [docs/literature/](docs/literature/) — query shift 文献、agent memory 笔记
- **复现验证** [docs/verification/](docs/verification/) — StreamingQA 复现指南

## 约定

- 当前主实验算法名为 `CostAwareDRIP`，消融为 `CostAwareDRIP-NoDrift` / `CostAwareDRIP-NoChurn`。
- `DRIP-QueryHidden` / ESC / pair lease 保留为 hidden-evidence diagnostic，不再承担主论文贡献。
- 实验数据（`motivation_*/data/*.json`）与 embedding 缓存（`motivation_*/cache/`）一律保留，不删。
- 细粒度机制设计只写在 `docs/design/`，不进 README / tex 主线。
