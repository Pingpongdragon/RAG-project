# EXPERIMENT_PROTOCOL — 数据集、Baseline、核心算法统一原则

> 本文确立 DRYAD 论文的实验协议：用哪些**数据集**、保留哪几个**baseline**、以及"核心算法只有一份"的**统一原则**。
> 依据两轮文献调研（见 [REFERENCES.md](REFERENCES.md)）。目的：把散落的 18 个策略收敛成干净、有说服力、可复现的实验矩阵。

---

## 1. 核心算法统一原则（一份实现，多处调参）

> 原则：**DRYAD 的核心逻辑只有一份权威实现**；不同实验（Fig1 单跳 / Fig2 多跳 / benchmark 对比）只允许调**超参**，不允许各自分叉出不同的核心逻辑。

| 层 | 权威实现位置（目标） | 说明 |
|---|---|---|
| 检测①+决策②（生产骨架） | `algorithms/qarc/`（DriftLens + KBUpdateAgent） | DRYAD 的 detect/decide |
| 准入③ doc 层（demand 账本 + 实体桥接） | `algorithms/cache/ours/dryad.py`（继承 RoutedCache←QueryDriven） | DRYAD 的 admit/evict；**目标**位置（迁移后），现暂在 motivation_2/strategies.py |

> 迁移目标：所有 cache 策略集中到 `algorithms/cache/`（见 §4.2），motivation 仅 import 不定义。

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

## 3. Baseline 精选（重选：以 cache replacement 族为主）

> 原则（**修正后**）：本文定位是「**自适应缓存系统在 query shift 下换入/换出文档**」。因此主 baseline 必须是
> **cache replacement policy**（能映射到统一 `on_access/admit/evict` 接口、对同一 workload 做换入换出）。
> 对话记忆/知识编辑/文档到达这类**不是 replacement policy**的方法，降级为「范式对照」，不进主 cache 表。

### 主表 baseline（cache replacement 族 + 语义/RAG 缓存族）

| 主表 baseline | 类别 | 出处 | 角色 |
|---|---|---|---|
| **Random / FIFO** | 下界 | 经典 | 无信号地板 |
| **LRU** | recency | 教科书 | 最基本对手；StreamingQA(L1) 上 access-history 成立的点 |
| **LFU / TinyLFU** | frequency | Einziger 2017 (1512.00727) | 频率族代表（admission policy） |
| **ARC** ⭐ | 经典自适应 | Megiddo&Modha FAST'03 | **最大缺口，必补**：随 workload 自调 recency/freq，与"适应 shift"正面对位 |
| **LeCaR / CACHEUS** ⭐ | learned 自适应 | HotStorage'18 / FAST'21 | 学习型自适应缓存，working-set≫cache 场景的 SOTA 对手 |
| **MissLRU / MissTinyLFU** | miss 驱动 + 语义信号 | — | 与 SemFlow/DRYAD 同源（miss 时用语义信号），多跳主对照 |
| **GPTCacheStyle** | 语义缓存 | GPTCache | embedding 相似度命中下界 |
| **RAGCache** ⭐ | RAG 知识缓存 | Jin 2024 (2404.12457) | RAG 缓存方向必比近邻 |
| **Online semantic eviction** | 语义淘汰（带 mismatch cost） | 2508.07675 | 与本命题正面重叠，重点对比/引用 |
| **ARC** ⭐ | per-agent RAG cache（最近邻威胁） | 2511.02919 | **必比+必区分**：query 分布几何缓存；见 [ARC_COMPARISON.md](ARC_COMPARISON.md) |
| **Belady / OPT** | 上界 | Belady 1966 | 离线最优（=现有 Oracle，**建议改名 Belady/OPT** 让审稿人秒懂） |
| **SemFlow (QueryDriven)** | ours | 本文 | demand 账本（L1/L2） |
| **RoutedCache** | ours | 本文 | +实体桥接（L3 消融） |
| **DRYAD** | ours | 本文 | 完整：FID 检测 + 双层准入 |

⭐ = 当前代码**尚缺、需新增**的关键对手（ARC / LeCaR 或 CACHEUS / RAGCache）。

### 降级为「范式对照」（不进主 cache 表，放 related work 或单独 paradigm 小节）

`ComRAG`(对话QA记忆)、`ERASE`(知识编辑)、`RECIPE`、`DocArrival`(=LightRAG/HippoRAG 文档到达)、
`MemGPTStyle`、`OnDemandFetch`(=CRAG 按需取)。
**理由**：它们对象/决策动作/目标函数与 cache replacement 不同，不是同类可比；作主 baseline 会被质疑稻草人。
保留它们用于说明"DRYAD 不是在做对话记忆/知识编辑/文档摄入"，即划清范式边界。

> **取舍修正记录**：
> - 之前误把 ComRAG/ERASE 当主 baseline → 修正为「范式对照」。
> - 主对照改为 cache replacement 族；**优先补 ARC**（自适应缓存经典对手，审稿人最期待，比 ComRAG/ERASE 重要得多）。
> - Oracle → 建议正名 **Belady/OPT**。
> - agent 数据集仍待补（支撑 memory 层 claim）。

---

## 4. 代码架构：统一接口 + 按族分文件 + 单一 harness

> 目标（用户决策）：**所有策略（baseline + ours）都在 `algorithms/` 里、每个策略独立、实现同一抽象接口**；
> motivation 与 main 实验**共用一套 harness**，motivation 只是 main 的 policy 子集 + 切片视图。
> 这与自适应缓存领域惯例一致（LeCaR/CACHEUS 开源实现都是统一 `Cache` 接口 + 单一 trace driver）。

### 4.1 统一抽象接口（窗口级 cache policy）

所有策略实现同一接口（现有 `BaseStrategy` 即此模式的 window-batch 变体，语义对齐经典 `on_access/admit/evict`）：
```
class CachePolicy:
    def set_kb(ids)                                  # 初始化常驻集
    def step(window_queries, window_q_embs, w)       # 观察一窗 → 换入/换出（admit/evict）
    # 统计：update_cost / maint_retrieval_cost / serve_retrieval_cost
```

### 4.2 按族分目录（建议结构）

```
algorithms/
├── cache/                      # ← 窗口级 cache policy 家族（motivation + main 共用）
│   ├── base.py                 #   CachePolicy 抽象 + 成本统计
│   ├── registry.py             #   STRATEGY_FACTORIES（唯一注册处）
│   ├── recency/                #   LRU, RecencyTTL, TemporalAware
│   ├── frequency/              #   LFU, TinyLFU, MissLRU, MissTinyLFU
│   ├── adaptive/               #   ARC, LeCaR/CACHEUS  ← 需新增
│   ├── semantic/               #   GPTCacheStyle, RAGCache  ← RAGCache 需新增
│   ├── oracle/                 #   Belady/OPT (=现 Oracle)
│   ├── ours/                   #   SemFlow(QueryDriven), RoutedCache, DRYAD
│   └── paradigm_ref/           #   范式对照: DocArrival, KnowledgeEdit, OnDemandFetch, MemGPTStyle
├── qarc/  comrag/  erase/      # ← 既有 per-query KBUpdateStrategy 家族（benchmark 用，暂留）
└── base.py                     # 既有 KBUpdateStrategy 接口
```

> DRYAD 的检测器（`qarc/detection/drift_detector.py`）从 `cache/ours/dryad.py` 内部 import，
> 不再靠 sys.path hack——因为两者都在 `algorithms` 包内，是正常包内引用。

### 4.3 motivation 与 main 统一 harness

- **同一 driver、同一 trace/window 构造、同一组指标**。motivation 三级 audit = main harness 的三个 dataset 配置：
  - StreamingQA(L1 单跳) / HotpotQA(L2 direct) / 2Wiki(L3 bridge) / MuSiQue(难 bridge)
- motivation 视图 = 同一份 per-window recall 日志里**只渲染 {LRU, MissLRU, SemFlow, Belady} 四条线**的子集。
- Belady/OPT 既是 motivation 上界也是 main 上界，天然统一，避免两套口径被质疑 cherry-pick。
- `motivation_1/2` 的 run.py 改为从 `algorithms.cache.registry` 导入策略，自身不再定义策略类。
  参数差异（DEMAND_DECAY 0.85 vs 0.92）通过 config 注入，不是代码分叉。

### 4.4 新增的"适应 shift"指标（撑核心卖点）

除 Recall@k，补两个：
- **adaptation speed / recovery time**：shift 后多少窗恢复 hit/recall（画 change-point 附近时间序列曲线）。
- **cold-fetch cost**：cold-tier 取回次数（vs OnDemand 的高成本）。

### 4.5 迁移风险与原则

- **实验数据/图不动**：`motivation_*/data/*.json`、`figures/`、`cache/` 全部保留。
- 迁移仅动**策略代码归属**，不改策略**逻辑/超参**——保证现有 Fig1/Fig2 复现一致。
- 分阶段：先建 `algorithms/cache/` 并迁移 + 让 run.py 导入跑通验证数字一致，再补 ARC/RAGCache 等新对手。

---

## 5. 实验矩阵（主结果表）

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

## 6. 消融（对应 DRYAD 三模块）

| 消融 | 隔离哪部分 | 现有类 |
|---|---|---|
| DRYAD − R3 | 实体桥接的 bridge 增益 | `RoutedCache` 去 R3 → `QueryDriven` |
| DRYAD − 模块①② | 显式漂移检测/决策（退回"写入帽=失败数"） | `RoutedCache` |
| 检测器：FID vs gap vs ADWIN/MMD | 对齐特征 FID 的贡献 | （需实现变体） |
| Miss-only 更新 | MissLRU/MissTinyLFU | 已有 |
| memory 层 开/关 | agentic 增益（semantic-memory hit-rate） | （需实现 S4） |

---

## 7. 落地待办

- [x] 确立数据集 + baseline 精选（本文）
- [ ] benchmark/ 与 motivation_2 的 STRATEGY_FACTORIES 收敛到精选 8 个（删冗余注册，保留类定义备消融）
- [ ] FINAL_METHOD 补 G1–G5 缺口（见该文档）
- [ ] memory 层最小实现 + 独立指标（S4）
