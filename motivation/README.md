# Motivation — Shared RAG Hot-Tier Admission under Query Drift

本目录是论文 Introduction / Motivation 部分的全部材料：LaTeX 源文、三级 system audit 的
实验代码与数据、绘图脚本、叙事文档，以及最终算法设计。

## 一句话主线

> 在多用户并发 agent 共享 RAG 的部署里，hot-tier 缓存面临 **query demand drift**：
> 文档不变、需求在变。本文用一个三级 system audit 刻画"哪种 admission signal 在哪种
> 访问模式下够用"，并据此提出 **DRIP** —— 一个从 `DRIP-Dense` 到
> `DRIP-ESC` 再到 `DRIP-ESC-Lease` 的 route-aware cache manager。
> 完整设计见 [docs/design/ALGORITHM_DESIGN.md](docs/design/ALGORITHM_DESIGN.md)。

## 三级 audit ↔ 图 ↔ 目录映射

| 层级 | 论文图 | 现象 | 目录 | 结论 |
|---|---|---|---|---|
| **L1** 宏观需求漂移 | Fig 0 | WildChat / Google Trends topic 重排 | [motivation_0/](motivation_0/) | 漂移是 first-class 普遍现象 |
| **L1** 单跳校准 | Fig 1 | StreamingQA 14 年 era 流 | [motivation_1/](motivation_1/) | access history（LRU）已够用 |
| **L2** direct evidence | Fig 2a | HotpotQA 直接证据多跳 | [motivation_2/](motivation_2/) | DRIP-Dense +9.7pp，语义扩散有效 |
| **L3** bridge evidence | Fig 2b | 2Wiki `Q→A→B` 桥接多跳 | [motivation_2/](motivation_2/) | query 信号触达不到 B，21pp Oracle gap → 引出 DRIP-ESC / DRIP-ESC-Lease |

## 顶层文件 / 目录

| 项 | 角色 |
|---|---|
| `motivation.tex` | 论文 Intro/Motivation 源文（图叙事注释见文件头部） |
| `references.bib` | 参考文献（缺失 cite key 见 [docs/narrative/NEXT_STEPS_AUDIT_TODO.md](docs/narrative/NEXT_STEPS_AUDIT_TODO.md) §3） |
| `motivation_0/` | **Fig 0** — query demand drift（WildChat / Trends） |
| `motivation_1/` | **Fig 1** — StreamingQA 单跳 signal audit |
| `motivation_2/` | **Fig 2** — 静态多跳 direct vs bridge 诊断 |
| `plotting/` | 论文图绘制：`plot_motivation_v2.py`（fig1/fig2）、benchmark 脚本 |
| `paper_figs/intro/` | 论文最终引用的 `fig0/1/2_*.pdf`（tex `\includegraphics` 指向这里） |
| `baselines_src/` | 第三方 baseline 源码参考（GPTCache / MemGPT），只读 |
| `docs/` | 全部文档（叙事 / 实验 / 设计 / 文献 / 复现）→ 见 [docs/README.md](docs/README.md) |
| `archive/` | 历史备份、日志、过时文档、旧快照 → 见 [archive/README.md](archive/README.md) |

## 文档导航（docs/）

- **叙事 & 评审** [docs/narrative/](docs/narrative/) — 故事线、评审意见、会议记录、待办
- **实验总结** [docs/experiments/](docs/experiments/) — 各图结果、数据集诊断、评审回应
- **算法设计** [docs/design/](docs/design/) — **[ALGORITHM_DESIGN.md](docs/design/ALGORITHM_DESIGN.md)**（最终框架）+ DESIGN_DIRECTIONS（gap 分类）
- **文献背景** [docs/literature/](docs/literature/) — query shift 文献、agent memory 笔记
- **复现验证** [docs/verification/](docs/verification/) — StreamingQA 复现指南

## 约定

- 当前算法名统一为 `DRIP`，消融命名统一为 `DRIP-Dense` / `DRIP-ESC` / `DRIP-ESC-Lease`。
- 实验数据（`motivation_*/data/*.json`）与 embedding 缓存（`motivation_*/cache/`）一律保留，不删。
- 细粒度机制设计只写在 `docs/design/`，不进 README / tex 主线。
