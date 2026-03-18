# Updator 模块文档 — KB 动态更新策略

> 本文档总结 `updator/` 目录下所有代码的具体逻辑，涵盖 3 种 KB 更新策略 + 2 种 Baseline + 统一接口层。

---

## 1. 整体架构

```
updator/
├── base.py                  # 统一抽象基类 + 2 个 Baseline
├── __init__.py              # 包级导出
├── comrag/                  # ComRAG (ACL 2025) — 查询驱动的 QA 记忆路由
│   ├── memory.py            #   双向量库 (V_high / V_low) + 质心聚类
│   ├── pipeline.py          #   三层路由 + 自适应温度 + 端到端管线
│   └── updater.py           #   评分 → 路由 → 聚类放置
├── erase/                   # ERASE (Li et al. 2024) — 文档驱动的事实编辑
│   ├── knowledge_base.py    #   可编辑事实库: 增删改查 + 历史追踪
│   └── updater.py           #   三步流水线: Retrieve → Update → Add
└── qarc/                    # QARC (ours) — 兴趣驱动的对齐漂移检测 + 子模KB策展
    ├── drift_detector.py    #   Part 1: Query-KB 对齐特征的正则化 FID 漂移检测
    ├── kb_agent.py          #   Part 2: Agent-in-the-Loop KB 更新决策
    ├── interest_model.py    #   AutoKMeans 兴趣聚类 + AlignmentGap
    ├── kb_curator.py        #   子模函数 KB 策展 (bootstrap + re-curation)
    └── pipeline.py          #   Bootstrap → Online 主流水线
```

---

## 2. 统一接口层 (`base.py`)

### 2.1 抽象基类 `KBUpdateStrategy`

所有方法必须实现的统一接口，确保 QARC / ComRAG / ERASE / Static / Random 可以公平对比：

| 方法 | 作用 |
|------|------|
| `initialize(doc_pool, doc_embeddings, kb_budget)` | 传入文档池 + embedding，构建初始 KB |
| `process_query(query_text, query_embedding, step, gold_doc_ids)` | 逐条处理查询：检索 → 判断是否更新 → 执行更新 → 返回结果 |
| `get_kb_doc_ids()` → `Set[str]` | 当前 KB 文档 ID 集合 |
| `get_kb_size()` → `int` | 当前 KB 文档数 |
| `get_metrics()` → `MethodMetrics` | 累积统计（查询数、更新数、KB 大小历史等） |

### 2.2 数据结构

- **`ProcessResult`**: 单次查询的返回值 — `retrieved_doc_ids`, `update_performed`, `kb_size`, `extra_metrics`
- **`MethodMetrics`**: 累积指标 — `total_queries`, `total_updates`, `total_docs_added/removed`, `kb_size_history`, `extra_series`

### 2.3 初始 KB 选择: `select_diverse_initial_kb()`

贪心 MaxMin Diversity: 逐步选择与已选集合**余弦距离最远**的文档。比 `pool[:budget]` 更公平，不受排列顺序影响。

### 2.4 Baseline

| 策略 | 逻辑 |
|------|------|
| **`StaticKBStrategy`** | 初始化后**永不更新** KB。用于展示不更新时性能随 drift 衰退 |
| **`RandomKBStrategy`** | 每隔 `update_interval` 步，从文档池随机采样 `kb_budget` 条替换整个 KB |

---

## 3. ComRAG — 查询驱动的 QA 记忆路由

**论文**: ComRAG — Conversational RAG with Dynamic Memory (ACL 2025 Industry Track)

### 3.1 核心思想

ComRAG **不修改 KB**，而是维护一个不断增长的 **QA 记忆库**：
- 高质量历史回答可以被直接复用或作为正面参考
- 低质量历史回答作为"反面教材"避免 LLM 重蹈覆辙

### 3.2 数据结构 (`memory.py`)

**双向量库**:
- `V_high` (高质量): 存储评分 `s ≥ γ` 的 QA 对 → 直接复用 / 正面参考
- `V_low` (低质量): 存储评分 `s < γ` 的 QA 对 → 反面教材

**每个向量库 (`CentroidClusterStore`) 的质心聚类管理**:

```
新 QA 到来:
  1. 与已有记录 sim ≥ δ (近重复)?
     → 保留评分更高者 (近重复替换)
  2. 与某质心 sim ≥ τ?
     → 加入该簇, c_k = 均值(簇内所有 embedding)
  3. 都不满足?
     → 创建新簇 (发现新话题)
```

超参数:
- τ (tau) = 0.75: 聚类相似度阈值
- δ (delta) = 0.9: 直接复用/替换阈值
- γ (gamma) = 0.6: 质量分界线

### 3.3 三层查询路由 (`pipeline.py`)

给定新查询 q，计算与 V_high 中最相似记录的 sim：

| 条件 | 策略 | 动作 |
|------|------|------|
| `sim ≥ δ` | **直接复用** | 返回历史回答，不调 LLM (省 token) |
| `τ ≤ sim < δ` | **参考生成** | 用 V_high 中相关 QA 作为 ICL 示例辅助 LLM |
| `sim < τ` | **KB回退+避免** | KB 检索文档 + V_low 中低质量 QA 作为反面教材 |

### 3.4 自适应温度 (`compute_adaptive_temperature`)

```
T(Δ) = exp(-k × min_gap)  截断到 [T_min, T_max]
```
- 历史分数差异小 → 高温 → 鼓励探索
- 历史分数差异大 → 低温 → 保持稳定

### 3.5 更新阶段 (`updater.py`)

每次 LLM 回答后:
1. **评分**: `s = Scorer(q, â)` (默认 BERT-Score F1)
2. **路由**: `s ≥ γ` → V_high, `s < γ` → V_low
3. **放置**: `CentroidClusterStore.add()` 处理近重复替换/聚类分配

---

## 4. ERASE — 文档驱动的事实编辑

**论文**: Language Modeling with Editable External Knowledge (Li et al., 2024)

### 4.1 核心思想

ERASE 让 KB 中的每条**原子事实**都变成"可编辑的"：
- 每条事实 f_j 附带历史记录 `H_j = [(timestamp, True/False), ...]`
- 新文档到来时，LLM 判断已有事实的真假是否变化
- 假事实可尝试改写为新的真事实

### 4.2 KB 数据模型 (`knowledge_base.py`)

- **`FactEntry`**: 原子事实 + embedding + 真假历史 `H_j`
  - `is_currently_true`: 取历史中最新状态
  - `reinforce(ts)` / `make_false(ts)` / `rewrite(new_fact, new_emb, ts)`
- **`ERASEKnowledgeBase`**: 稠密向量检索 + 事实级 CRUD
  - `retrieve()`: 推理用 (阈值 0.7, 默认只取 true)
  - `retrieve_for_update()`: 更新用 (阈值 0.3, 包含 false)

### 4.3 三步更新流水线 (`updater.py`)

当一篇新文档 d 到达时:

```
Step 1 — Retrieve(K, d):
  用文档 embedding 检索 top-k 候选事实 (低阈值 0.3)

Step 2 — Update(f_j, H_j, d):  ← 核心步骤, 两轮 LLM
  [第一轮 - 分类] 对每个候选事实:
    LLM 判断 → "Reinforce" / "Make False" / "No Change"
    → 相应更新历史 H_j

  [第二轮 - 改写] 对被标记为 False 的事实:
    LLM 尝试改写为新的真事实
    → 例: "Elizabeth II is Queen" → "Charles III is King"

Step 3 — Add(d):
  LLM 从文档中提取新的原子事实 → embed → 加入 KB
```

### 4.4 推理 (Appendix A.3)

检索时将事实 + 历史一起给 LLM:
```
"Elizabeth II is the Queen of England (true at 2020, false at 2022-09)"
```
让 LLM 理解知识的时间演变。

---

## 5. QARC — 兴趣驱动的对齐漂移检测 + 子模 KB 策展 (ours)

### 5.1 核心思想

QARC 不是检测 "query 偏移" 或 "docs 偏移"，而是检测 **"query 与 KB 的对齐模式是否偏离了历史正常水平"**：
- query 变了 ≠ 要更新 KB (新 query 可能仍然被 KB 覆盖)
- query-KB 不对齐了 → 才需要更新 KB

### 5.2 Part 1: 对齐漂移检测 (`drift_detector.py`)

**对齐特征 (Alignment Features)**: 对每个 query，不使用 raw embedding，而是计算与 KB 的对齐特征:

```
alignment_features(q) = [
    sim(q, c₁), ..., sim(q, cₖ),     # 与 K 个 KB 主题中心的匹配度
    top1_sim, ..., topN_sim           # 与最相似 N 篇 KB 文档的匹配度
]
```

其中 `cₖ` = KMeans(KB 文档) 的聚类中心。

**正则化 FID 检测**:
- **Offline**: 存储基线对齐特征的 μ₀ 和 Σ₀ (正则化协方差 Σ + εI)
- **Threshold**: 随机采样历史 query 窗口 → 计算 FID → 取 P95 百分位
- **Online**: FID(当前窗口, 基线) > threshold → 对齐偏移 → KB 需更新

```
FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·√(Σ₁·Σ₂))
```

**KB 更新后**: KB 换了文档 → 聚类中心变了 → 对齐特征定义变了 → 必须用近期 query 重建基线。

### 5.3 Part 2: Agent-in-the-Loop 决策 (`kb_agent.py`)

替代 DriftLens 论文中的 Human-in-the-Loop。三个观察信号:
1. `drift_result.is_drifted` — 对齐特征是否漂移
2. `gap_result.gap` — AlignmentGap 对齐度差距
3. 历史趋势 — 连续漂移次数, Gap 的 EMA/MAD 走势

**决策规则**:

| 阶段/规则 | 条件 | 动作 | 替换比例 |
|-----------|------|------|----------|
| **Warmup** | 前 N 个窗口 | AggressiveUpdate | λ=0.5 |
| Rule 1 (正常) | 未漂移 & Gap 正常 | NoOp | — |
| Rule 2 (Gap偏高) | 未漂移 & Gap > EMA+k·MAD | MildUpdate | λ=0.2 |
| Rule 3 (漂移) | 已漂移 | AggressiveUpdate | λ=0.5 |
| Rule 4 (连续漂移) | 连续 N 次漂移 | Recalibrate + 更新 | λ=0.5 + 重建基线 |

**冷却机制**: 更新后 cooldown_windows 个窗口内不触发新更新。

**Gap 自适应阈值**:
```
gap_threshold = EMA(Gap) + k × MAD(Gap)
```
不使用固定阈值，而是跟踪 Gap 的指数移动平均和偏差。

### 5.4 兴趣模型 (`interest_model.py`)

| 组件 | 作用 |
|------|------|
| `QueryWindowBuffer` | 滑动窗口缓冲区，积累 W 个查询的 embedding |
| `auto_kmeans(X)` | 对窗口内 query 做 Cosine K-Means，自动选最佳 K (轮廓系数) |
| `compute_alignment_gap(Q, KB)` | G(t) = 1 - avg max_sim(q, KB)，衡量 KB 对齐度 |

### 5.5 子模函数 KB 策展 (`kb_curator.py`, 903 行)

**核心**: 将 KB 更新建模为子模函数最大化问题：

```
max_{S:|S|≤B} f(S) = Σᵢ wᵢ · max_{d∈S} CosSim(cᵢ, d) + η · Diversity(S)
```

- `wᵢ`: 兴趣权重 (auto_kmeans 输出的各簇比例)
- `cᵢ`: 兴趣中心 (各簇 centroid)
- `η`: 多样性正则系数

**更新流程 (`recurate`)**:
1. 计算当前 KB 中每篇文档的边际贡献
2. 移除贡献最低的 `⌊λ_max × B⌋` 篇文档
3. 用兴趣中心检索候选文档 (top-k per centroid)
4. 贪心子模最大化选择新文档填充空位
5. 保证 KB 大小始终 ≤ budget

**Bootstrap**: 初始 KB 建立支持两种模式:
- 冷启动: `bootstrap_diversity()` — MaxMin 多样性选择
- 热启动: `bootstrap_from_queries()` — 用历史 query 的兴趣中心加权选择

### 5.6 主流水线 (`pipeline.py`)

```
Bootstrap:
  文档池 → 初始 KB (多样性或兴趣加权)
  DriftLens 延迟初始化 (warmup 后才有足够 query 历史)

Online Loop (每个查询窗口):
  1. process_query() → 缓冲 embedding + KB 检索
  2. 窗口满 → _process_window():
     a. AutoKMeans 聚类 → 兴趣中心 + 权重
     b. DriftLens 对齐漂移检测 (FID)
     c. compute_alignment_gap → 对齐度差距
     d. Agent.decide(drift, gap) → 更新决策
     e. 执行更新 (如有) → recurate + 重建 DriftLens 基线
```

**Query 历史管理**: Pipeline 维护一个 ring buffer (`deque`)，记录每个 query embedding：
- `set_baseline(kb_embs, query_embs)` 需要两个参数
- KB 更新后用全部历史 query 重建基线（因为对齐特征定义依赖当前 KB）

---

## 6. 三种方法的核心对比

| 维度 | ComRAG | ERASE | QARC |
|------|--------|-------|------|
| **更新触发** | 每次查询后 (隐式) | 新文档到达时 | 检测到对齐漂移时 |
| **检测机制** | 相似度阈值路由 (τ/δ) | 无显式检测 | 正则化 FID + AlignmentGap |
| **更新对象** | QA 记忆库 (V_high/V_low) | 原子事实的真假 | KB 文档集合组成 |
| **更新方式** | 聚类放置 + 近重复替换 | LLM 分类/改写事实 | 子模函数贪心增删 |
| **KB 是否改变** | ❌ 不修改 KB | ✅ 编辑事实内容 | ✅ 替换 KB 中的文档 |
| **LLM 依赖** | 生成回答时 | 分类 + 改写 + 提取 | 可选 (仅生成回答时) |
| **感知用户兴趣** | ❌ (只感知查询历史) | ❌ (文档驱动) | ✅ (显式兴趣建模) |
| **漂移检测** | 无 | 无 | ✅ (对齐特征 FID) |

---

## 7. 实验适配层 (`experiments/adapters.py`)

将三种策略 + 两种 Baseline 适配到 `KBUpdateStrategy` 统一接口：

| 适配器 | 内部策略 |
|--------|----------|
| `QARCStrategyAdapter` | 封装 `QARCPipeline` |
| `ComRAGStrategyAdapter` | 封装 `ComRAGPipeline` + `DynamicMemory` |
| `ERASEStrategyAdapter` | 封装 `ERASEUpdater` + `ERASEKnowledgeBase` |
| `StaticKBStrategy` | 直接继承自 `base.py` |
| `RandomKBStrategy` | 直接继承自 `base.py` |

所有适配器实现相同的 `initialize()` / `process_query()` / `get_kb_doc_ids()` 接口，使实验框架可以无差别地运行任意策略组合。

---

## 8. 文件行数统计

| 文件 | 行数 | 核心职责 |
|------|------|----------|
| `base.py` | 351 | 抽象基类 + Static/Random Baseline |
| `comrag/memory.py` | 476 | 双向量库 + 质心聚类 |
| `comrag/pipeline.py` | 395 | 三层路由 + 自适应温度 |
| `comrag/updater.py` | 175 | 评分 → 路由 → 放置 |
| `erase/knowledge_base.py` | 348 | 可编辑事实库 |
| `erase/updater.py` | 540 | 三步流水线 + Prompt 模板 |
| `qarc/drift_detector.py` | 366 | 对齐特征 FID 漂移检测 |
| `qarc/kb_agent.py` | 360 | Agent 决策 (4 规则 + warmup + 冷却) |
| `qarc/interest_model.py` | 363 | AutoKMeans + AlignmentGap |
| `qarc/kb_curator.py` | 903 | 子模函数 KB 策展 |
| `qarc/pipeline.py` | 393 | Bootstrap → Online 主流水线 |
| **合计** | **4,744** | |
