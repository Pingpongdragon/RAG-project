# ARC 对照分析（最近邻威胁与差异化）

> **ARC** = *Cache Mechanism for Agent RAG Systems*, arXiv:2511.02919 (2025.11)。
> 这是 DRYAD **最相似的已发表工作（最近邻威胁）**，必须在 related work 显式引用并区分。
> 本文记录两者的重叠、差异、以及 DRYAD 仍可投的定位收窄。

---

## 1. ARC 是什么（从摘要与公开 review 确认）

- **目标**：为每个 LLM agent 动态维护"小而高价值的语料"（compact high-relevance corpus），annotation-free。
- **机制**：综合**历史 query 分布模式** + **缓存项在 embedding 空间的几何**，自动维持高相关缓存。
- **结果**：存储压到原语料的 **0.015%**、has-answer 提升至 **~79.8%**、延迟降 **~80%**；三个检索数据集。
- **定位**：单 agent 的紧凑 RAG 文档缓存；用 query 频率/分布 + 嵌入几何打分做准入。

## 2. 重叠（必须承认）

DRYAD 与 ARC 的**大框架相同**：*用 query 分布信号为 agent 动态管理一个紧凑的 RAG 文档缓存*。
→ 因此 DRYAD **不能再把"用 query 分布管理 agent 文档缓存"当作核心新意**——这一格被 ARC 占了。

## 3. 差异（DRYAD 护城河，按强度排序）

| 维度 | ARC | DRYAD | 强度 |
|---|---|---|---|
| **漂移处理** | 历史 query 分布几何，**累积/静态统计**，无显式漂移检测 | **显式 alignment-feature FID 漂移检测 + 决策预算 λ·B**：对 query–KB 对齐分布偏移做统计检验，决定"何时、以多大力度"换 | 🟢 强 |
| **多跳 / bridge** | 纯 embedding 几何（query↔缓存项相似度），**够不到 second-hop** | **entity-chained prefetch**：读 A 抽实体找 B，触达 query 邻域外的桥接文档 | 🟢 强 |
| **drift 设定** | 未强调随时间的 query shift | 核心即 **query distribution shift over time**（sudden/gradual/cyclic）+ adaptation-speed 指标 | 🟡 中 |
| **双层** | 单层 doc 缓存 | doc cache + agent memory 两层共享一个漂移信号 | 🟡 中（memory 层未实现，暂不作主卖点） |

## 4. 一句话区分（写进 related work）

> *ARC maintains a compact per-agent corpus by scoring documents with cumulative query-frequency and embedding-space geometry. DRYAD differs on two axes ARC does not address: (i) it **detects** query–KB alignment drift with an alignment-feature FID test and sizes each update by the detected drift, rather than relying on cumulative geometry; and (ii) it admits **bridge documents unreachable from the query embedding** via entity-chained prefetch, whereas ARC's geometric scoring can only cache items similar to past queries.*

## 5. 仍可投吗 —— 能，但定位必须收窄

- **核心新意上移到 ARC 没做的两点**：①漂移**检测**驱动的更新（FID + 决策预算）；②**entity-chained bridge** 多跳准入。
- 对照口径：**ARC 解决"哪些高频文档值得缓存"；DRYAD 解决"分布漂移时如何检测并换入 query 看不见的桥接文档"**。前者是 frequency/geometry，后者是 drift-detection + content-side signal。
- **真正风险点（非 ARC 本身）**：当前 2Wiki 上 DRYAD 比 SemFlow 仅 +0.8~2.7pp，bridge 增益偏小。
  若最终 FID 检测消融与 bridge 增益都拉不开差距，会被指"ARC 变体"。
  → 必做：(a) FID 检测 vs 无检测 vs gap 的消融拉出差距；(b) bridge 增益在更纯/更大 pool 上放大；(c) 把 ARC 作为主 baseline 之一直接同台比较。

## 6. 数据集参考（可借鉴 ARC 的三检索数据集设定）

ARC 用"三个检索数据集"评 storage% / has-answer / latency。DRYAD 已有 StreamingQA/HotpotQA/2Wiki/MuSiQue，
可**对齐 ARC 的指标口径**（storage 占比、has-answer/recall、latency/cold-fetch）以便直接对比。
具体 ARC 用哪三个集待查全文确认（arxiv 当前被网络策略挡，需在可访问环境核对 2511.02919 全文的 §experiments）。

## 7. 待核实（arxiv 访问受限，二手摘要）

- ARC 的确切数据集名、baseline 列表、是否真有任何 drift/temporal 设定 → 需读全文 §4 确认。
- ARC 是否完全单 agent（DRYAD 强调多用户共享）→ 全文确认后可作为额外差异点。

> 来源：arXiv:2511.02919 摘要 + themoonlight/huggingface review 摘要（WebSearch 片段，2026-06）。全文待核。
