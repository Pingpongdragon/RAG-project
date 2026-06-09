# DRYAD — 最终算法设计：Drift-aware Demand-driven Admission

> ⚠️ **已被取代（superseded）。** 本文是 DRYAD 的**早期单层 doc-cache 版**设计。
> 最新的双层（agent memory + doc cache）权威设计见 → **[FINAL_METHOD.md](FINAL_METHOD.md)**。
> 本文保留作为单层框架的历史记录与三模块（检测/决策/准入）的细节参考。


> **DRYAD** = **Dr**ift-aware demand-driven admission **with entit**y-ch**a**ine**d** prefetch。
> 共享 RAG hot-tier 的统一 KB 更新框架：**检测（何时更新）→ 决策（更新多激进）→ 准入/准出（换入哪些、换出哪些，含 bridge 信号）**。
>
> 这是论文最终 method（tex 里 `[METHOD]` 的统一替换名）。它把两处既有实现收敛成一套：
> - `updator/qarc/`：完整的**漂移检测（DriftLens FID）+ Agent 决策 + 子模策展**流水线 → 提供 DRYAD 的「检测 + 决策」骨架。
> - `motivation/motivation_2/strategies.py` 的 `QueryDriven`(SemFlow) + `RoutedCache`(R3 实体桥接) → 提供 DRYAD 的「准入/准出 + bridge 修复」引擎，且已在三级 audit 上验证。
>
> 上游素材：[STORYLINE_v1.md](../narrative/STORYLINE_v1.md)、[DESIGN_DIRECTIONS.md](DESIGN_DIRECTIONS.md)（gap G1–G5）、
> [updator/README.md](../../../updator/README.md)、两处代码对照见 [UNIFICATION.md](UNIFICATION.md)。

---

## 0. 问题陈述（锁定边界）

多用户并发 agent 共享的 two-tier RAG：cold corpus 完整且**静态**（无可靠 doc timestamp），
hot tier 是内存内小型 ANN，容量 `B` ≪ corpus，需在 SLO 内服务 query 流。query 流随时间 **demand drift**。
问题：**只看 query-side 信号（不看 doc 创建时间）的前提下，每个窗口该把哪些 cold doc 换入/换出 hot tier，
使未来 query 的 Recall@K 最大、cold-tier fetch 最少。**

排除两类作弊：① 给 LRU/TTL 真实 doc 时间戳（泄露 temporal oracle，退化成 document-stream cache）；
② Oracle 用未来 gold SF（非 causal，仅上界）。

---

## 1. 审稿人质疑与反驳（method 形态的出发点）

> **审稿人**：多跳失败是"算法换入换出哪些文档（admission）"的问题，不是 LRU 这个淘汰算法本身的问题。

**我们完全同意——而且这正是本文的论点，不是反对。** 回应顺着往下推：

1. 我们 audit 的对象从来不是"谁的淘汰算法好"，而是 **admission 决策所依赖的信号源**。
   LRU 只是 *access-history* 这一信号家族的代表（TinyLFU / MissLRU 同族）。
2. 三级 audit 证明同一件事在不同信号下换入换出的能力：
   - **L1 单跳**：access-history 信号足以决定换入换出，LRU 不输（StreamingQA 28.8% ≈ SemFlow 27.1%）。
   - **L2 direct 多跳**：query 语义邻域信号能改进换入换出，SemFlow **+9.7pp**（HotpotQA 44.6→54.3）。
   - **L3 bridge 多跳**：**任何只看 query-side 信号的 admission policy（LRU / TinyLFU / SemFlow 全部）都无法把隐藏的 second-hop 文档 B 换进来**——B 在 query embedding 空间里对 query 不可见，21pp Oracle gap 与具体淘汰算法无关。
3. 因此根因是 **admission 可用的信号空间不足以识别该换入哪个 bridge 文档**，不是 eviction 算法的锅。
4. DRYAD 不是又一个淘汰算法，而是一套**完整 admission 框架**：它给 admission 决策补上 query-side 给不出的新信号——
   **从已入库文档 A 的内容沿实体链找到 B（entity-chained prefetch）**，并配上 drift 检测决定何时、以多大力度执行换入换出。

> 一句话写进 rebuttal：*"We agree bridge failure is an admission problem, not an LRU-eviction problem. That is precisely our claim: across L1–L3 the eviction rule is held fixed while the admission **signal** varies, and the 21pp gap persists for every query-side signal because no query-side signal can even see the second-hop document. DRYAD closes this by adding a content-side entity-chained admission signal, gated by drift detection."*

---

## 2. DRYAD 总体架构（三模块）

```
 query window
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 模块 ①  DETECT — 何时该更新？                                          │
│   DriftLens FID(对齐特征) + AlignmentGap   ← 来自 updator/qarc         │
│   输出 drift_signal {is_drifted, fid, gap}                            │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 模块 ②  DECIDE — 更新多激进？                                          │
│   Agent 规则: NoOp / Mild(λ=0.2) / Aggressive(λ=0.5) / Recalibrate    │
│   warmup + cooldown + 连续漂移重校准      ← 来自 updator/qarc          │
│   输出 update_budget λ·B（本窗允许换入换出的条数上限）                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 模块 ③  ADMIT/EVICT — 换入哪些、换出哪些？（在 λ·B 预算内）             │
│   候选信号三路汇入同一 demand/serve 账本：                              │
│     R1 query 语义邻域 demand   (SemFlow, 解 L1/L2)                     │
│     R3 entity-chained bridge   (RoutedCache, 解 L3 ← 新信号)           │
│     R2 cold-fetch 高置信回写    (兜底 COLD)                            │
│   统一准入门 + 冗余惩罚淘汰     ← 来自 motivation SemFlow              │
└─────────────────────────────────────────────────────────────────────┘
```

**设计主张**：检测/决策（模块①②）回答"何时、多大力度更新"，是 `updator/qarc` 已实现的；
准入/准出（模块③）回答"换入换出哪些"，是 `motivation` 验证过的 SemFlow+R3。
DRYAD = 把这两半接成一条流水线，**模块②输出的 `λ·B` 直接作为模块③的写入预算上限**，
取代 SemFlow 原来"写入帽=失败数"的隐式预算——这是两处统一的接缝。

<!-- SECTION-3-PLACEHOLDER -->

---

## 3. 模块 ① DETECT — 漂移检测（来自 updator/qarc/detection）

**不用 raw query embedding，用 query 与 KB 的「对齐特征」**，因为 query 变 ≠ 要更新（新 query 可能仍被 KB 覆盖），
只有 *query–KB 对齐模式* 偏离历史正常水平才该更新。

```
alignment_features(q) = [ sim(q, c_1), …, sim(q, c_K),     # 与 K 个 KB 主题中心
                          top1_sim, …, topN_sim ]           # 与最相似 N 篇 KB 文档
   c_k = KMeans(当前 KB 文档) 的中心
```

- **Offline**：基线 μ₀, Σ₀（正则化协方差 Σ+εI）；随机采样历史 query 窗口算 FID 分布，取 P95 作 threshold。
- **Online**：`FID(当前窗口, 基线) = ‖μ₁−μ₂‖² + Tr(Σ₁+Σ₂−2√(Σ₁Σ₂))`；`is_drifted = FID > threshold`。
- **AlignmentGap**：`G(t) = 1 − avg_q max_{d∈KB} sim(q,d)`，KB 覆盖度的直接度量。
- **KB 更新后必须重建基线**（KB 换文档→中心变→对齐特征定义变）。

> 与 SemFlow 隐式漂移信号的关系：SemFlow 用"失败 query 数"作隐式漂移强度（写入帽=失败数）。
> DRYAD 把它升级为**显式**双信号 `{FID, Gap}`——FID 抓"对齐分布偏移"，Gap 抓"覆盖度下降"，
> 比单纯失败计数更能区分"query 漂了但 KB 仍覆盖"（不该更新）与"真不对齐了"（该更新）。

代码：[`updator/qarc/detection/drift_detector.py`](../../../updator/qarc/detection/drift_detector.py) `DriftLensDetector`。

---

## 4. 模块 ② DECIDE — Agent 决策（来自 updator/qarc/decision）

把 DriftLens 论文里的 Human-in-the-Loop 换成规则 Agent。观察 `{is_drifted, gap, 历史趋势}`，输出 action + 替换比例 λ：

| 规则 | 条件 | action | λ（占 B 比例） |
|---|---|---|---|
| Warmup | 前 N 窗 | Aggressive | 0.5 |
| R1 正常 | 未漂移 & Gap 正常 | **NoOp** | 0 |
| R2 Gap 偏高 | 未漂移 & Gap > EMA+k·MAD | Mild | 0.2 |
| R3 漂移 | is_drifted | Aggressive | 0.5 |
| R4 连续漂移 | 连续 N 次 | Recalibrate + Aggressive | 0.5 + 重建基线 |

- Gap 自适应阈值 `EMA(Gap)+k·MAD(Gap)`，不用固定阈值。
- 更新后 cooldown 窗内不再触发。
- **输出 `λ·B` 就是模块③这一窗的换入换出预算上限。**

代码：[`updator/qarc/decision/kb_agent.py`](../../../updator/qarc/decision/kb_agent.py) `KBUpdateAgent.decide`。

---

## 5. 模块 ③ ADMIT/EVICT — 准入/准出（来自 motivation SemFlow + RoutedCache）

三路候选信号汇入**同一个 per-doc demand/serve 账本**（指数衰减 0.92），在 `λ·B` 预算内竞争：

### R1 — query 语义邻域 demand（SemFlow，解 L1/L2）
每个 miss query 取 cold pool top-K 邻居，按归一化相似度 `×NEIGH_GAMMA(0.4)` 摊 demand，top-1 额外 +1。
co-admit 语义共现文档 → 支撑 comparison 多跳（两实体同驻）。已验证 HotpotQA +9.7pp。

### R3 — entity-chained bridge prefetch（RoutedCache，解 L3 ← DRYAD 的关键新信号）
```
对每个 BRIDGE miss：
  A = top-step1_k(query, pool)          # first hop，query 邻域可达
  E = extract_entities(A 的内容)         # 读 A 的实体（spaCy NER / pool_ents）
  for B in pool if B 与 E 共享实体:      # 沿实体边找 second hop
      demand[B] += entity_overlap(B,E) × ALPHA_BRIDGE
```
信号空间与 R1 正交：R1 用 `sim(query, doc)`，R3 用 `sim(A内容, B) via 共享实体`，**只有 R3 触达 `Q→A→B` 的 B**。
entity→doc 倒排表增量维护（O(|diff|)）。

### R2 — cold-fetch 高置信回写（兜底 COLD）
hot tier 整体无证据时临时取 cold top-K（当窗可见，恢复 recall），验证有用的 doc 以 `ALPHA_FETCH(>1)` 高置信写回账本，
把"反应式检索"转成"反应式检索 + 选择性固化"，避免对重复 demand 反复付 cold 成本。

### 统一准入门 + 冗余惩罚淘汰
```
候选(非KB)按 demand 降序；KB doc 按 vscore 升序：
   vscore(d) = serve[d] + demand[d] − LAMBDA_RED·max(0, redundancy(d) − RED_THRESH)
准入：demand[c] > vscore(e) 才用 c 换 e；去重：sim(c, KB) > TAU_ADMIT(0.95) 跳过
预算：本窗换入换出条数 ≤ λ·B（来自模块②）  ← 取代 SemFlow 原"写入帽=失败数"
```

代码：[`motivation_2/strategies.py`](../../motivation_2/strategies.py) `QueryDriven`(R1+gate) + `RoutedCache`(R3)。

---

## 6. 端到端伪代码（单窗口）

```python
def dryad_step(window_queries, q_embs, w):
    # ① DETECT
    feats = alignment_features(q_embs, kb_centroids, kb_emb)
    drift = drift_lens.detect(feats)              # {is_drifted, fid}
    gap   = alignment_gap(q_embs, kb_emb)
    # ② DECIDE
    action, lam = agent.decide(drift, gap, w)     # NoOp/Mild/Aggressive/Recalibrate
    if action == NoOp:
        update_serve_only(window_queries); return
    budget = int(lam * B)
    # ③ ADMIT/EVICT  (在 budget 内)
    decay(demand, serve)
    for q in window_queries:
        if hit(q): serve[best_kb(q)] += 1; continue
        regime = route(q)                          # DIRECT / BRIDGE / COLD
        if   regime == DIRECT: r1_neighborhood_demand(q)      # SemFlow
        elif regime == BRIDGE: r3_entity_chain(q)             # RoutedCache
        else:                  r2_cold_fetch_writeback(q)     # 兜底
    admit_and_evict(demand, serve, budget)         # 统一 gate + 冗余惩罚
    if action == Recalibrate: drift_lens.set_baseline(kb_emb, recent_q)
    entity_index.sync(kb_diff)
```

---

## 7. 与 gap 分类的映射

| Gap (DESIGN_DIRECTIONS) | DRYAD 部件 |
|---|---|
| G-1 Bridge identification | 模块③ R3 entity-chained prefetch |
| G-2 Same-window causality | 模块③ R2 当窗临时可见 |
| G-3 Fetch-to-persist handoff | 模块③ R2 高置信回写 |
| G-4 Regime-routing | 模块③ route() + 模块② Agent |
| G-5 Capacity/eviction | 模块② λ·B 预算 + 模块③ 统一 gate |
| （新增）何时更新 | 模块① DriftLens 检测 |

---

## 8. 验证计划（落到 motivation_2 基础设施）

主战场 2Wiki bridge gradual（Fig 2b），同口径对比 DRYAD vs SemFlow(QueryDriven) / MissLRU / OnDemandFetch / Oracle。
当前基线（已存于 `data/results_2wiki_bc_entity_expand_gradual_q2k.json`）：
**QueryDriven 36.0 / MissLRU 29.0 / OnDemand 48.8 / Oracle 57.0**。

- **核心预期**：DRYAD 把 SemFlow→Oracle 的 **21pp gap 显著收窄**（R3 把 bridge doc B 换进 KB）。
- **消融**：
  1. 去 R3（只 R1+R2）→ 应退回 SemFlow 水平 ⇒ 证明 R3 是 bridge 增益来源。
  2. 去模块①②（写入帽=失败数）→ 隔离"显式漂移检测"的贡献。
  3. 去 R2 回写 → 隔离 fetch-to-persist 收益。
- **成本轴**：cold-tier fetch 次数（vs OnDemand ~29/query）、每窗维护开销（应仍 4–10ms 量级）。
- **与 updator baseline 比较**：DRYAD vs ComRAG / ERASE / Static / Random（updator/base.py 统一接口），证明完整框架优于既有 KB 更新范式。

---

## 9. 风险与诚实边界

- **路由误判**：规则 route() 边界会错标；缓解：错判只改候选来源，统一 gate 仍是最终裁判。
- **实体抽取质量**：R3 依赖 NER；长尾/cold-start 实体召回有限 → 列为 limitation。
- **predicate 漂移**：`A→B` 实体边可能随时间语义改变（隐式/时序桥接）→ 本设计只处理一跳 `A→B`，更长链 `A→B→C` 留 future work。
- **检测开销**：DriftLens 每窗算 FID + KMeans(KB)；需确认仍在维护预算内（实验报告）。
- **不过度承诺**：DRYAD 按 regime 对症下药——L1 不输 access-history，L2 靠 R1，L3 靠 R3 收窄 Oracle gap；不声称任何场景最优。

---

## 附：术语对照（统一到 DRYAD）

| DRYAD 术语 | motivation 代码 | updator/qarc 代码 |
|---|---|---|
| 模块① 漂移检测 | （原隐式：失败数） | `DriftLensDetector` (FID + AlignmentGap) |
| 模块② 决策 | （原隐式：写入帽=失败数） | `KBUpdateAgent.decide` (λ) |
| 模块③ R1 demand 准入 | `QueryDriven` demand/serve + gate | 子模 `QARCKBCurator`（per-cluster 对应物） |
| 模块③ R3 bridge | `RoutedCache` entity chain | （无，DRYAD 新增到 updator 一侧） |
| 模块③ 冗余淘汰 | `LAMBDA_RED/RED_THRESH` | Facility-Location 多样性项 |
| `[METHOD]` 占位 | → `DRYAD` | → `DRYAD` |

> 两处实现差异（per-doc vs per-cluster、显式 vs 隐式检测）与统一接缝详见 [UNIFICATION.md](UNIFICATION.md)。
