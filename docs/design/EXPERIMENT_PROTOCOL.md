# EXPERIMENT_PROTOCOL — 数据集、Baseline、核心算法统一原则

> 本文确立 DRYAD 论文的实验协议：用哪些**数据集**、保留哪几个**baseline**、以及"核心算法只有一份"的**统一原则**。
> 依据两轮文献调研（见 [REFERENCES.md](REFERENCES.md)）。目的：把散落的 18 个策略收敛成干净、有说服力、可复现的实验矩阵。

---

## 1. 核心算法统一原则（一份实现，多处调参）

> 原则：**DRYAD 的核心逻辑只有一份权威实现**；不同实验（Fig1 单跳 / Fig2 多跳 / benchmark 对比）只允许调**超参**，不允许各自分叉出不同的核心逻辑。

| 层 | 权威实现位置 | 说明 |
|---|---|---|
| 检测①+决策②（生产骨架） | `algorithms/qarc/`（DriftLens + KBUpdateAgent + curator） | DRYAD 的 detect/decide |
| 准入③ doc 层（demand 账本 + 实体桥接） | `motivation/motivation_2/strategies.py::DRYAD`（继承 RoutedCache←QueryDriven） | DRYAD 的 admit/evict |

**收敛动作**：
- `motivation_1/strategies.py` 与 `motivation_2/strategies.py` 的 `QueryDriven` 参数分叉（DEMAND_DECAY 0.85 vs 0.92 等）**只视为同一算法的不同超参配置**，不是两个算法。
  - 处理：保留两套文件作 Fig1/Fig2 的定型实验快照（实验数据不动），但在两处 `QueryDriven` 顶部加注释互指，声明"同一核心、参数随实验调整"。
  - 长期（可选）：抽出公共 `QueryDriven` 到一个共享模块，两个 run.py 用不同 config 传参——但**仅在确认不改变已定型实验结果时**才做。
- DRYAD 最终算法（含三模块）以 `algorithms/qarc` + `motivation_2::DRYAD` 为唯一事实来源，其余策略类都是 baseline 或消融。

---

## 2. 数据集（4 主 + 1 可选，正交映射 claim）

| 数据集 | 对应 claim / 论文层级 | 触发 DRYAD 的哪部分 | 出处 |
|---|---|---|---|
| **StreamingQA** | L1 单跳时间漂移（漂移检测器真正被触发） | 模块① 检测 + R1 | arXiv:2205.11388 |
| **HotpotQA** | L2 direct 多跳（邻域 demand 有效） | R1 邻域扩散 + co-admit | Yang 2018 |
| **2WikiMultihopQA** | L3 bridge 多跳（核心战场） | R3 实体链 prefetch | Ho 2020 |
| **MuSiQue** | 更难组合多跳（R3 泛化/难度上限） | R3 在深 bridge 上 | Trivedi 2022 |
| Bamboogle（可选，125 题） | zero-shot bridge 泛化小测 | R3 | Press 2022 |

**为何这个组合足够**：
- 正是近 2 年多跳 RAG 论文标准四件套（HotpotQA+2Wiki+MuSiQue+Bamboogle），审稿人不会质疑标准性。
- 四集正交映射四个 claim（时间漂移/direct/bridge/难bridge），无冗余。
- **memory 层不引入对话集**：DRYAD 的 memory 是检索 episodic（实体链经验），不是对话记忆。
  用 2Wiki/MuSiQue 的 gold 实体链派生合成 episodic 是自洽的（诚实标注）。
- **明确不加**：LoCoMo / LongMemEval（对话记忆，task mismatch）、TempLAMA / TimeQA（probing/推理，非流式 cache）。

---

## 3. Baseline 精选（从 13+2 → 8 个主表）

> 原则：每个 baseline **类别**只留 1 个代表 + 上界 Oracle + 下界 Static。Miss* 变体、MemGPT、RECIPE 进消融或 related work。

| 主表 baseline | 类别 | 代表论文 | 对应 DRYAD 的对立面 |
|---|---|---|---|
| **Static** | 下界 | — | 不更新 cache 的地板 |
| **LRU** | 经典 eviction（access-history） | — | bridge 够不到的反例（audit 主角） |
| **TinyLFU** | 频率 eviction | Einziger 2017 | 频率族代表 |
| **GPTCacheStyle** | 语义缓存 | GPTCache 2023 | response 级相似度命中 |
| **DocArrival** | 文档到达驱动 | LightRAG 2410.05779 / HippoRAG 2405.14831 | corpus-side 到达（吞 LogDrivenArrival） |
| **ERASE** | 知识编辑/KB 更新 | ERASE 2406.11830 | corpus-side freshness 编辑（doc 层对照） |
| **ComRAG** | agent memory 管理 | ComRAG 2506.21098 | memory 层 admission/eviction（替代 MemGPTStyle） |
| **OnDemandFetch** | 被动检索 | CRAG 2401.15884 思路 | miss 就取的高成本上界式对照 |
| **Oracle** | 上界 | — | 非 causal 天花板 |

**砍掉（理由）**：
- `RandomFIFO` — 被 Static/LRU 覆盖，无信息量。
- `MissLRU / MissTinyLFU` — Miss-only 更新是实现细节，进消融提一句。
- `LogDrivenArrival` — DocArrival 变体，合并。
- `MemGPTStyle` — 被 ComRAG（更新更可比）替代；MemGPT 作 framing 引用。
- `KnowledgeEdit(RECIPE)` — RECIPE 改参数不改 external KB，与 cache 设定不对口；ERASE 更对口；RECIPE 进 related work。
- `QueryDrivenLoose` — QueryDriven 的调参变体，进消融。

**ComRAG / ERASE 为何保留**：二者都是"在预算/阈值下决定 external KB 或 memory 存什么删什么"，与 DRYAD 同类可比——ERASE 对 doc 层、ComRAG 对 memory 层，正好覆盖 DRYAD 双层各自的最强对照。

---

## 4. 实验矩阵（主结果表）

```
                StreamingQA   HotpotQA   2Wiki(bridge)   MuSiQue
                  (L1)         (L2)         (L3)          (难L3)
Static            ...          ...          ...           ...
LRU               ...          ...          ...           ...
TinyLFU           ...          ...          ...           ...
GPTCacheStyle     ...          ...          ...           ...
DocArrival        ...          ...          ...           ...
ERASE             ...          ...          ...           ...
ComRAG            ...          ...          ...           ...
OnDemandFetch     ...          ...          ...           ...
DRYAD (ours)      不输         +增益         收窄gap        收窄gap
Oracle (上界)     ...          ...          ...           ...
```

报告 Recall@5 H2 + cold-fetch 成本 + 每窗维护开销。

## 5. 消融（对应 DRYAD 三模块）

| 消融 | 隔离哪部分 | 现有类 |
|---|---|---|
| DRYAD − R3 | 实体桥接的 bridge 增益 | `RoutedCache` 去 R3 → `QueryDriven` |
| DRYAD − 模块①② | 显式漂移检测/决策（退回"写入帽=失败数"） | `RoutedCache` |
| 检测器：FID vs gap vs ADWIN/MMD | 对齐特征 FID 的贡献 | （需实现变体） |
| Miss-only 更新 | MissLRU/MissTinyLFU | 已有 |
| memory 层 开/关 | agentic 增益（semantic-memory hit-rate） | （需实现 S4） |

---

## 6. 落地待办

- [x] 确立数据集 + baseline 精选（本文）
- [ ] benchmark/ 与 motivation_2 的 STRATEGY_FACTORIES 收敛到精选 8 个（删冗余注册，保留类定义备消融）
- [ ] FINAL_METHOD 补 G1–G5 缺口（见该文档）
- [ ] memory 层最小实现 + 独立指标（S4）
