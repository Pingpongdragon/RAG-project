# UNIFICATION — `updator/` 与 `motivation/` 算法统一到 DRYAD

> 目的：项目里有两套独立实现的 KB 更新算法，本文记录它们的关系、不一致点，以及如何统一成同一个最终 method **DRYAD**。
>
> - `motivation/`：论文的**初步故事线 / 三级 audit 验证平台**。有数据、embedding 缓存、Fig 0/1/2 评测口径。
> - `updator/`：最终算法与 **baseline 对比**的平台（QARC/ComRAG/ERASE/Static/Random 统一接口）。
>
> 结论：**两边用同一个最终 method DRYAD**。DRYAD = `updator/qarc` 的「检测+决策」 + `motivation` 的「准入/准出+实体桥接」。

---

## 1. 两处现状对照

| 维度 | motivation (`motivation_2/strategies.py`) | updator (`updator/qarc/`) |
|---|---|---|
| 需求建模 | **per-doc** `demand[pi]` / `serve[pi]` 指数衰减 | **per-cluster** AutoKMeans 兴趣中心 + 权重 |
| 漂移检测 | **隐式**：失败 query 数 = 漂移强度（写入帽=失败数） | **显式**：DriftLens FID(对齐特征) + AlignmentGap |
| 决策 | 无独立决策层（gate 自限） | `KBUpdateAgent`：NoOp/Mild/Aggressive/Recalibrate + λ |
| 准入 | 逐条 admission gate `demand[c] > vscore(e)` | 子模贪心 `greedy_submodular_select`（(1-1/e) 近似） |
| 淘汰 | `serve+demand − λ_red·redundancy` 升序 | 兴趣相关度升序 |
| 冗余/多样性 | `LAMBDA_RED/RED_THRESH/TAU_ADMIT` 去重 | Facility-Location 多样性项 |
| **实体/bridge** | **有**：`RoutedCache` R3 entity-chained | **无** |
| 实验入口 | `motivation_2/run.py`（HotpotQA/2Wiki/MuSiQue + drift） | `updator/base.py` `KBUpdateStrategy` 接口（无统一 run） |

**核心判断**：两套是**同一思想（需求驱动的 KB 策展）的两个实现**，各有对方缺的关键件——
updator 有显式检测/决策但无 bridge；motivation 有 bridge 但检测是隐式的。DRYAD 取两者并集。

---

## 2. DRYAD 的统一接缝（关键设计）

把两半接成一条流水线，接缝是 **预算变量 λ·B**：

```
updator/qarc:  DriftLens → AlignmentGap → KBUpdateAgent.decide → 输出 (action, λ)
                                                                      │
                                                          budget = λ·B │  ← 接缝
                                                                      ▼
motivation:    SemFlow demand/serve + RoutedCache R3 → admit_and_evict(budget=λ·B)
```

- **取代点**：SemFlow 原本"写入帽 = 本窗失败数"（隐式预算）→ 改为"写入帽 = 模块②输出的 `λ·B`"（显式预算）。
- **保留点**：SemFlow 的 demand/serve 账本、统一 gate、冗余惩罚、R3 实体桥接全部保留，是模块③的实现。
- **新增点**：模块①②（DriftLens + Agent）从 updator/qarc 移植成 motivation 可调用的轻量组件。

---

## 3. 术语统一表（两处代码改名/对齐到 DRYAD 词汇）

| DRYAD 统一词 | motivation 旧词 | updator 旧词 | 处理 |
|---|---|---|---|
| DRYAD（method 名） | SemFlow / RoutedCache / `[METHOD]` | QARC | 论文统一叫 DRYAD；QARC 作为 updator 内部 pipeline 名可保留为"DRYAD 的检测+决策实现" |
| 模块① DETECT | （隐式失败数） | DriftLens / drift_detector | 统一称 "drift detection (DriftLens-FID)" |
| 模块② DECIDE | （隐式写入帽） | KBUpdateAgent | 统一称 "update agent / policy" |
| 模块③ ADMIT | demand/serve gate | 子模 curator | 统一称 "demand-driven admission" |
| R3 bridge | RoutedCache entity chain | （无） | 统一称 "entity-chained prefetch" |

> 注意：**不强行把 per-doc 和 per-cluster 合并成一种数据结构**。它们是同一目标函数的两种实现粒度
> （per-doc demand ≈ per-cluster 兴趣覆盖的细粒度版）。论文正文用统一抽象 `f(S)=interest_coverage+diversity` 描述，
> 实现上 motivation 侧用 per-doc（细、利于 bridge 单文档定位），updator 侧用 per-cluster（粗、利于子模理论保证）。
> 二者在论文里作为**同一 method 的两种 admission 后端**呈现，消融里可互比。

---

## 4. 落地步骤（已做 / 待做）

- [x] 写清 DRYAD 三模块框架 → [ALGORITHM_DESIGN.md](ALGORITHM_DESIGN.md)
- [x] 写清两处统一关系 → 本文件
- [ ] **代码统一（motivation 侧，作为论文验证）**：
  - 在 `motivation_2/strategies.py` 把 `RoutedCache` 定型为 `DRYAD`：保留 R1+R3，接入轻量 drift 检测/决策（先可用简化版：用失败率+EMA 近似 DriftLens，确认有效后再移植完整 FID）。
  - 跑 2Wiki bridge gradual，验证收窄 21pp gap。
- [ ] **代码统一（updator 侧，作为 baseline 对比）**：
  - 把 R3 entity-chained prefetch 作为一个 admission 信号加进 `updator/qarc/curation/kb_curator.py` 的子模候选，使 updator 的 DRYAD 与 motivation 的 DRYAD 是同一算法。
  - 用 `updator/base.py` 接口让 DRYAD 与 ComRAG/ERASE/Static/Random 同台比较。
- [ ] **统一命名**：定稿后把 tex / 代码 / 文档里的 `[METHOD]` 与 QARC 统一替换为 DRYAD（保留 QARC 作内部 pipeline 别名注释）。

---

## 5. 给审稿人的一致性陈述（备用话术）

> *DRYAD is a single framework with three modules — drift detection, an update policy, and demand-driven admission with entity-chained prefetch. The detection/policy modules follow our QARC pipeline (alignment-feature FID + a rule agent); the admission module is the SemFlow demand ledger extended with an entity-chained bridge signal. We instantiate the admission backend at two granularities (per-document on the motivation testbed, per-cluster submodular in the baseline comparison) and show they are the same policy under a shared coverage-plus-diversity objective.*
