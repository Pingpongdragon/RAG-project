# 实验评估指标说明

> 适用范围：`benchmarks/experiment_framework.py` 中所有方法（QARC / ComRAG / ERASE / StaticKB / RandomKB）

---

## 一、评估粒度

| 粒度 | 说明 |
|:---|:---|
| **单点 (per-query)** | 每条 query 处理完毕后立即采样，反映某一时刻的瞬时状态 |
| **时序 (per-window)** | 窗口内所有 query 的单点值做平均，产生一个时间序列 |
| **全局 (aggregate)** | 所有窗口的均值，用于横向方法对比 |

每个窗口包含 `window_size=20` 条 query。1000 条 query → 50 个窗口，形成长度为 50 的时间序列。

---

## 二、各指标详解

### 1. Recall@k（检索召回率）

**类型：** 单点 → 时序（窗口内均值）→ 全局均值

**含义：**  
每条 query 都有一个或多个 gold doc（标准答案文档）。
检索系统从 KB 里返回 top-k 个文档（k=10），若 gold doc 出现在其中，则该 query 命中（hit=1），否则 hit=0。

$$\text{Recall@k}(W) = \frac{1}{|W|} \sum_{q \in W} \mathbf{1}[\text{gold}_q \cap \text{top-k retrieved}_q \neq \emptyset]$$

**关键时序细节：**  
- 检索（`retrieve()`）发生在 `process_query()` 开头，此时 KB 是**当前时刻**的状态
- QARC 的一个窗口内，KB 只在第 20 条 query 处理完后才可能更新，因此窗口内前 19 条 query 用的 KB 与第 20 条**一致**（都是上一次更新后的状态）
- 窗口均值跨越了 KB 更新事件的前后，**反映的是整窗口内平均的检索能力**

**如何解读：**  
越高越好。在话题漂移时会下降，能快速恢复说明方法适应性强。

---

### 2. Gold-in-KB Rate（金标文档覆盖率）

**类型：** 单点 → 时序（窗口内均值）→ 全局均值

**含义：**  
每条 query 处理完后，检查 gold doc 是否**在当前 KB 中存在**（不要求被检索到，只要在 KB 里即可）。

$$\text{Gold-in-KB}(W) = \frac{1}{|W|} \sum_{q \in W} \mathbf{1}[\text{gold}_q \cap KB_{\text{current}} \neq \emptyset]$$

**关键时序细节（和 Recall@k 的不同之处）：**

```
每条 query 的执行顺序：
  ① retrieve(q) → ret_ids          ← 用的是 KB_old (更新前)
  ② q 入 buffer
  ③ 若 buffer 满 → KB 可能更新为 KB_new
  ④ kb_ids = get_kb_doc_ids()      ← 用的是 KB_new (更新后，若发生更新)

→ Recall@k 用①的结果（旧 KB）
→ Gold-in-KB 用④的 KB 状态（可能是新 KB）
```

因此在窗口末尾（第 20 条 query）：Recall 反映旧 KB 的检索能力，Gold-in-KB 已经反映新 KB 的内容。

**特殊情况：**
- **QARC**：Gold-in-KB 高，因为 submodular 更新会主动把与用户兴趣相关的文档（包括潜在 gold doc）挑入 KB
- **ComRAG**：Gold-in-KB 低但 Recall 在 Cyclic Return 场景高——原因是 ComRAG 的 recall 来自 QA 记忆路由（绕过 KB 直接从历史记忆回答），不代表 gold doc 在 KB 里
- **StaticKB**：Gold-in-KB 等于 ComRAG（两者 KB 相同，ComRAG 从不更新 KB）

**如何解读：**  
高 Gold-in-KB + 高 Recall：KB 内容好且检索有效  
高 Gold-in-KB + 低 Recall：KB 有 gold doc 但检索排名靠后（embedding 相似度不够）  
低 Gold-in-KB + 高 Recall：依赖 KB 外部机制（如 ComRAG 的记忆路由）

---

### 3. Topic Alignment（话题对齐度）

**类型：** 时序（每个窗口末尾的快照）→ 全局均值

**含义：**  
衡量 KB 的话题分布与当前用户兴趣的匹配程度。

$$\text{TopicAlign}(W) = \cos\bigl(P_{\text{window}}(\text{topic}),\ P_{\text{KB}}(\text{topic})\bigr)$$

其中：
- $P_{\text{window}}(\text{topic})$：当前窗口内的查询话题分布（归一化频率直方图）
- $P_{\text{KB}}(\text{topic})$：当前时刻 KB 中文档的话题分布（归一化频率直方图）

两个分布拉成向量，取余弦相似度，范围 $[0, 1]$。

**时序细节：**  
- 窗口话题分布：基于该窗口内的 query 标签（每窗口 20 条，固定）
- KB 话题分布：窗口结束时的快照（KB 在最后一条 query 后已更新）
- 是**窗口级**快照，不是 query 级单点

**如何解读：**  
接近 1 → KB 覆盖的话题与当前 query 话题高度吻合  
接近 0 → KB 内容与用户当前兴趣偏差很大（话题错位）  
在 Sudden Shift 场景下，所有方法的该指标都会在话题切换时骤降

**注意：** 该指标依赖 pool 文档的话题标签（预设的，非真实话题），反映的是"结构对齐"，不完全等同于语义覆盖。

---

### 4. Avg Retrieval Sim（平均检索相似度）

**类型：** 单点 → 时序（窗口内均值）→ 全局均值

**含义：**  
每条 query 检索返回 top-k 文档，取**第一位文档**与 query 的余弦相似度。

$$\text{AvgSim}(W) = \frac{1}{|W|} \sum_{q \in W} \cos(q,\ \text{top-1 retrieved}_q)$$

**时序细节：** 与 Recall@k 完全同步——都用 `retrieve()` 返回的同一批结果，都是检索那一刻的 KB 状态。

**如何解读：**  
反映 KB 中最相似文档与 query 的语义贴近程度。该指标在各方法间差距较小（0.67–0.73），因为检索用的是同一套 embedding，差距主要来自 KB 内容质量。

---

### 5. Adaptation Speed（适应速度）

**类型：** 全局标量（基于时序计算），单位：窗口数

**含义：**  
话题分布发生显著变化（$|\Delta P_{\text{topic}}| > 0.3$）后，Topic Alignment 需要多少个窗口才能恢复到变化前水平的 80%。

**计算逻辑：**
```python
for 每个窗口 i:
    if |topic_dist[i] - topic_dist[i-1]| > 0.3:   # 显著话题切换
        target = topic_alignment[i-1] × 0.8        # 恢复目标
        recovery_windows = 第一个 alignment ≥ target 的窗口距离
avg(recovery_windows) → adaptation_speed
```

**如何解读：**  
越小越好（0 = 立即恢复，无需等待）  
若整个实验无显著话题切换（如 Cyclic Return 的 QARC=0.0），说明 alignment 始终维持，无需恢复

**局限性：**  
- 依赖 Topic Alignment 作为代理指标（间接）
- 只统计"显著漂移"事件，Gradual Drift 下可能检测不到足够的切换点
- 不区分"从未下降"和"从未恢复"（都可能报 0）

---

## 三、指标时序示意

```
时间轴 →  Q1  Q2  ...  Q20 | Q21 Q22  ...  Q40 | ...
              (Window 0)        (Window 1)      
              
Recall@k:  [per-query采样，窗口内平均]  [per-query采样]
Gold-in-KB:[per-query采样，KB可能在Q20后刚更新]
TopicAlign:                             ↑窗口末快照       ↑窗口末快照
KB update:                     ↑ (若触发)               ↑ (若触发)
```

---

## 四、指标在各场景的预期行为

| 场景 | Recall@k | Gold-in-KB | TopicAlign | AvgSim | AdaptSpd |
|:---|:---|:---|:---|:---|:---|
| **Gradual Drift** | 随话题缓慢漂移而缓慢下降，好方法能跟上 | 跟随 KB 更新情况 | 缓慢变化 | 相对稳定 | 无明显区分 |
| **Sudden Shift** | 话题突变时骤降，恢复速度是关键 | 突变后骤降，主动更新方法更快回升 | 骤降→恢复 | 小幅波动 | **主要区分场景** |
| **Cyclic Return** | 话题轮回时，有历史 memory 的方法占优 | 反映 KB 是否保留了回归话题的文档 | 周期性波动 | 稳定 | 弱区分 |

---

## 五、Gold-in-KB 的测量偏差说明

由于 Gold-in-KB 测量的是 `process_query` 执行**后**的 KB 状态，而 Recall@k 测量的是**前**的检索结果，两者在窗口边界处存在半个窗口的错位。

具体来说：
- 窗口内前 19 条 query：Gold-in-KB 和 Recall@k 都基于**同一旧 KB**（对齐）
- 窗口第 20 条 query：Recall@k 基于旧 KB，Gold-in-KB 基于新 KB（**错位**）

这种错位的影响在 `window_size=20` 时约占 1/20 = 5%，对窗口均值影响较小，但需要注意 Gold-in-KB 整体上对动态更新方法会略微**高估**（因为能看到本窗口末的更新结果）。

---

# 对比方法基本思想

## ComRAG（ACL 2025）

**核心思路：KB 不动，靠 QA 记忆路由补偿话题漂移。**

ComRAG 将知识存储分为两层：
- **静态 KB**（Document Store）：随机初始化后**从不更新**，与 StaticKB 完全相同
- **动态 QA 记忆**（Memory Store）：每条 query 处理后，将该 query 的 embedding + 对应 gold doc 存入记忆，不断累积

**检索时的三级路由策略**（按 query 与记忆层中已有条目的相似度分档）：

| 相似度 | 策略 | 做法 |
|:---|:---|:---|
| ≥ δ（=0.9） | Direct Reuse | 直接返回记忆中该历史 query 对应的 gold doc |
| τ ≤ sim < δ（=0.75–0.9） | Reference Generation | 从相似历史条目的 gold doc 集合混合检索 |
| < τ | KB Retrieval | 回退到静态 KB 的标准 embedding 检索 |

**优势**：Cyclic Return 场景下话题轮回时，历史记忆中已有完全匹配的 gold doc，直接 reuse → Recall 极高  
**劣势**：KB 随话题漂移不更新，Gold-in-KB 始终低；记忆无限增长（无遗忘机制）；Recall 高依赖的是"记住了答案"而非"KB 覆盖了知识"

**代码位置**：[benchmarks/experiment_framework.py](experiment_framework.py#L379)，`ComRAGAdapter`

---

## ERASE（Li et al. 2024）

**核心思路：以 gold doc 为驱动，逐条 query-level 增量编辑 KB。**

ERASE 不做批量更新，而是**每条 query 处理完后**，立即对 gold doc 执行一次 KB 编辑操作：

**编辑规则**（针对每个 gold doc）：
1. 从 KB 中检索与该 gold doc 最相似的已有条目
2. 若相似度 ≥ `update_threshold`（=0.7）→ **REWRITE**：用新 gold doc 覆盖写该条目（保留位置，更新内容）
3. 若相似度 < 阈值 → **ADD**：将 gold doc 作为新条目加入；若 KB 满则按时间戳末尾淘汰最老的条目

**ERASE 依赖外部 gold doc 作为更新信号**：它不做任何漂移检测，不关心 query 分布，只要 query 有 gold doc 就更新 KB。在实验框架中，gold doc 通过 `feed_gold_docs()` 在 Recall@k 计算后传入，**模拟实际系统中的人工标注或弱监督信号**。

**优势**：每条 query 都即时响应，KB 与 gold doc 保持同步，Gold-in-KB 率高  
**劣势**：
- **依赖 gold doc 标注**（实际场景中难以实时获取）
- 无全局视角：逐条编辑可能造成 KB 内容碎片化
- 时间成本高（~40s），因为每条 query 都要做一次相似度计算 + 写操作

**代码位置**：[benchmarks/experiment_framework.py](experiment_framework.py#L530)，`ERASEAdapter`

---

# QARC 算法三阶段详解

QARC 将 KB 自适应更新问题分为三个串联阶段，每个窗口（20条 query）结束时依次执行：

```
查询流 → [窗口缓冲 20 条] → Phase 1: 检测 → Phase 2: 决策 → Phase 3: 更新
```

## Phase 1：检测阶段

**1.1 窗口 query 聚类（AutoKMeans）**

窗口内 20 条 query 的 embedding 先做 L2 归一化，然后用 **Cosine K-Means** 聚类：

- 自动选择最优 K（遍历 k ∈ [2, 10]，选**轮廓系数 Silhouette Score 最高**的 k）
- 每个簇产生一个**兴趣中心** centroid_i 和**兴趣权重** weight_i（该簇 query 占窗口总数的比例）
- 代码：[updator/qarc/curation/interest_model.py](../updator/qarc/curation/interest_model.py)，`auto_kmeans()`

聚类结果有两个用途：
- 用于 **Phase 2 检测**：计算对齐特征
- 用于 **Phase 3 更新**：submodular 选择时按兴趣加权

**1.2 DriftLens——对齐漂移检测**

借鉴 DriftLens（Greco et al.）的核心框架，但检测对象从"query 分布"改为"**query-KB 对齐模式**"：

> DriftLens 原文：将 query embedding 送入神经网络降维，per-window 建立高斯分布（均值+协方差），离线校准阈值，在线比对 FID 是否超阈值。同时提供 per-label 子分布检测（支持分类器辅助告警）。

**我们的改动：不检测 query 分布本身，而是检测 query 与 KB 的对齐特征**。原因是：
> "query 分布变了 ≠ 需要更新 KB（新 query 可能仍被 KB 覆盖）；但 query-KB 对齐下降了 → KB 确实需要更新"

**对齐特征的构造：**

对每条 query q，计算：

$$\text{feat}(q) = [\underbrace{\text{sim}(q,c_1),\ldots,\text{sim}(q,c_K)}_{\text{与 K 个 KB 主题质心的相似度}},\ \underbrace{\text{top1\_sim},\ldots,\text{topN\_sim}}_{\text{RAG 过程中与 KB 最相似 N 篇文档的相似度}}]$$

- 前 K 个（K=5）：query 与当前 KB 文档做 KMeans 聚类后 K 个质心的余弦相似度 → 捕捉**话题级对齐**
- 后 N 个（N=10）：RAG 实际检索时 top-N 的余弦相似度 → 捕捉**细粒度文档级对齐**

**Offline 阶段：**
1. 对 KB 文档做 KMeans → K 个主题质心
2. 历史 query 计算对齐特征 → 建立基线分布（均值 μ₀、正则化协方差 Σ₀ = Cov + εI）
3. 随机采样 500 个窗口 → 计算各自 FID → 取 P95 作为阈值

**Online 阶段：**

$$\text{FID}(\text{window}, \text{baseline}) = \|\mu_w - \mu_0\|^2 + \text{Tr}(\Sigma_w + \Sigma_0 - 2\sqrt{\Sigma_w \Sigma_0})$$

FID > threshold → **对齐漂移**，KB 需要更新。

**1.3 AlignmentGap——对齐度绝对值**

DriftLens 检测的是相对变化（与历史比），Gap 补充绝对状态：

$$G(t) = 1 - \frac{1}{|W|} \sum_{q \in W} \max_{d \in KB} \text{CosSim}(q, d)$$

本质：窗口里每条 query 能在 KB 里找到多相似的文档，取平均，再用 1 减掉。
- G ≈ 0 → KB 完美覆盖
- G ≈ 1 → KB 与当前 query 完全失配

两个信号互补：
- DriftLens → "对齐模式变了吗？"（相对，需要历史基线）
- Gap → "现在对齐得好不好？"（绝对，可实时计算）

---

## Phase 2：决策阶段（Agent-in-the-Loop）

`KBUpdateAgent` 替代 DriftLens 原文的 Human-in-the-Loop，自动将检测信号转化为更新策略：

| 优先级 | 条件 | 动作 | λ_max |
|:---:|:---|:---|:---:|
| 0 (最高) | Warmup 前 N 窗口 | AGGRESSIVE（热启动） | 0.5 |
| - | 冷却期中 | NO_OP | 0 |
| 4 | 连续漂移 ≥ 3 次 | RECALIBRATE + AGGRESSIVE | 0.5 |
| 3 | FID > threshold | AGGRESSIVE | 0.5 |
| 2 | Gap > EMA + k×MAD | MILD | 0.2 |
| 1 | 正常 | NO_OP | 0 |

Gap 阈值自适应：`threshold = EMA(G) + k × MAD(G)`，不用固定值。每次更新后进入 2 窗口冷却期。

---

## Phase 3：子模更新阶段（ReCurate）

确认需要更新后，`QARCKBCurator.recurate()` 分三步执行：

**A. 候选检索**：用每个兴趣中心 $c_i$ 在文档池中 FAISS 检索 top-k 相似文档，去重后得候选集

**B. 贪心子模选择**：最大化兴趣覆盖目标函数

$$f_\text{interest}(S) = \sum_{i=1}^{m} \alpha_i \cdot \max_{d \in S} \text{CosSim}(c_i, d)$$

单调子模函数，贪心算法保证 ≥ (1−1/e) ≈ 63.2% 最优近似。
采用 Lazy Greedy + 最大堆优化（维护每个簇的当前最大覆盖，避免重复计算边际增益）。

**C. 增量限速替换**：

$$\text{to\_add} = K_\text{ideal} \setminus K_\text{old}, \quad \text{to\_remove} = K_\text{old} \setminus K_\text{ideal}$$

实际替换量受 $\lambda_\text{max}$ 限制：最多替换 $\lfloor \lambda_\text{max} \times |KB| \rfloor$ 篇，防止单次更新过于剧烈。

KB 变化后：
- 使 FAISS KB 检索索引缓存失效（下次查询时重建）
- 重新调用 DriftLens `set_baseline()` + `calibrate_threshold()`（KB 变了，对齐特征定义随之变化，基线需重建）
