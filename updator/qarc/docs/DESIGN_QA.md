# QARC 设计 Q&A — 为什么这么做、每一步在干什么

---

## 一句话总结

QARC 流水线做的事情：
> **攒一窗口 query → 聚类发现"用户当前关心什么" → 检测"用户关心的东西跟 KB 还对不对齐" → 决定要不要换 KB 里的文档 → 换了之后重新校准检测器**

---

## Q1: 为什么 DriftLens 检测之后还要做 query 聚类 (auto_kmeans)？它们各自在干什么？

**它们职责完全不同，不是重叠的**：

```
auto_kmeans: "用户当前关心什么？"（兴趣建模）→ 输出给 KB 更新用
DriftLens:   "用户关心的东西跟 KB 还对齐吗？"（变化检测）→ 输出给 Agent 决策用
```

### auto_kmeans 的作用：发现兴趣中心，告诉 KB 该选什么文档

```python
# pipeline.py _process_window()
centroids, labels, weights = auto_kmeans(X)
# centroids = 用户当前关心的几个主题中心（向量）
# weights = 每个主题的重要程度（比如 60% 关心体育、40% 关心科学）
```

这些 centroids + weights 会传给 `curator.recurate(centroids, weights, lambda_max, eta)`

curator 拿到后："哦用户60%关心体育，我去文档池里找体育相关的文档放进KB"

**如果没有 auto_kmeans → curator 不知道该选什么文档，没有优化方向。**

### DriftLens 的作用：判断"要不要触发 KB 更新"

DriftLens 不管具体兴趣是什么，它只回答一个 boolean 问题：
> "这批 query 跟 KB 的匹配模式，跟历史上正常水平相比，变了没有？"

```python
# drift_detector.py
对齐特征 = [sim(q, centroid_1), ..., sim(q, centroid_K), top1_sim, ..., topN_sim]
FID(当前窗口对齐特征, 历史基线对齐特征) > threshold → 漂移了
```

**如果没有 DriftLens → 每个窗口都更新 KB，浪费开销；或者永远不更新，KB 过时。**

### 类比

| 角色 | 对应模块 | 问什么问题 |
|------|---------|-----------|
| 体温计 | DriftLens | "你发烧了吗？"（二值判断） |
| 医生 | Agent | "发烧了→吃药；没发烧→观察"（决策） |
| 药方 | auto_kmeans + curator | "吃什么药、剂量多少"（具体更新内容） |

---

## Q2: DriftLens 里的 "topic分类器" 是怎么训练和更新的？

**DriftLens 本身没有分类器。** 这里的 "分类" 来自 KMeans 聚类：

```python
# drift_detector.py set_baseline()

# Step 1: 对 KB 的 50 篇文档做 KMeans → 3~5 个簇
kmeans = KMeans(n_clusters=k)
kmeans.fit(kb_embeddings)  # 当前 KB 里的文档
self._kb_centroids = kmeans.cluster_centers_  # 这就是"主题中心"
```

这些主题中心用来定义对齐特征："query 跟每个主题中心的相似度是多少"。

### 何时重新训练？

**每次 KB 更新后都会重新训练**——因为 KB 换了文档，主题中心就变了：

```
pipeline.py _process_window():
    if KB更新了:
        _recalibrate_detector()
            → detector.set_baseline(新KB_embs, 近期query历史)
                → KMeans(新KB) → 新的主题中心
                → 用新主题中心重算所有历史query的对齐特征
                → 建立新的"正常对齐模式"基线
            → detector.calibrate_threshold(query历史, window_size)
                → 随机采样窗口 → FID分布 → 取P95作为新阈值
```

### 为什么要重新训练？

```
KB有体育+科学文档 → 主题中心=[体育向量, 科学向量]
→ 对齐特征=[跟体育的sim, 跟科学的sim, top1_sim, ...]

KB更新后变成体育+历史文档 → 主题中心=[体育向量, 历史向量]
→ 对齐特征定义变了! 旧基线已经无意义了
→ 必须用新的主题中心重建基线
```

---

## Q3: DriftLens 检测到漂移后，具体怎么决定更新 KB？

**Agent (kb_agent.py) 拿到两个信号做决策**：

```
信号1: DriftResult.is_drifted  → bool (对齐模式偏移了?)
信号2: AlignmentGap.gap        → float (KB 跟 query 的匹配差距有多大?)
```

### 决策规则（按优先级从高到低）：

```
┌ Warmup 期（前3个窗口）→ 总是 AGGRESSIVE_UPDATE (50%替换)
│   因为冷启动KB不好，先激进调整
│
├ Rule 4: 连续3次漂移 → RECALIBRATE (50%替换 + 重建基线)
│   "旧的 DriftLens 基线已经过时了"
│
├ Rule 3: 漂移 → AGGRESSIVE_UPDATE (50%替换)
│   "用户兴趣变了，大换血"
│
├ Rule 2: 没漂移但 Gap 异常偏高 → MILD_UPDATE (20%替换)
│   "DriftLens说没变，但缝隙在变大，轻度调整"
│
└ Rule 1: 没漂移 + Gap 正常 → NO_OP
    "一切正常，不动"
```

### Gap 阈值怎么定？

不是固定值，而是自适应的 **EMA + k·MAD**：

```python
gap_threshold = 指数移动平均(历史Gap) + 1.5 × 平均绝对偏差(历史Gap)
```

比如历史 Gap 都在 0.15 左右，那阈值大概是 0.15 + 1.5×0.02 = 0.18
一旦 Gap 超过 0.18，即使 DriftLens 没报漂移，也会触发轻度更新。

---

## Q4: curator.recurate() 具体怎么选择和淘汰文档？

### 四步流程：

```
A. 候选检索: 每个兴趣中心 → 去文档池检索 top-100 相似文档
B. 子模选择: 贪心最大化 f(S) = f_interest + η·f_diversity → "理想KB"
C. 增量替换: 理想KB vs 当前KB 的差集，受 λ_max 限制
D. 一致性检查: (可选) ERASE 风格的事实核查
```

### B 的子模目标函数：

```
f(S) = f_interest(S) + η · f_diversity(S)

f_interest(S) = Σ αᵢ · max_{d∈S} CosSim(cᵢ, d)
              = 对每个兴趣主题，KB中最相关文档的匹配度，按兴趣权重加权

f_diversity(S) = (1/|Pool|) · Σ_{d∈Pool} max_{d'∈S} CosSim(d, d')
              = KB能覆盖文档池多大比例的多样性
```

**为什么是子模函数？** 因为 max 操作保证了边际递减：第一篇体育文档收益大，第二篇就没那么大了。贪心算法对子模函数有 (1-1/e) ≈ 63% 的近似保证。

### C 的增量替换：

```
理想KB = {d1, d2, d3, d7, d8}    (子模选择的结果)
当前KB = {d1, d2, d5, d6, d9}    (现在有的)

需要添加 = {d3, d7, d8}
需要移除 = {d5, d6, d9}

如果 λ_max=0.2 且 KB 有 50 篇 → 最多换 10 篇
需要添加 3 + 需要移除 3 = 6 篇变动 < 10 → 全部执行

如果超过限额:
  添加: 按边际增益排序，取最有价值的 top-10
  移除: 按与当前兴趣的相关性排序，移除最不相关的 top-10
```

---

## Q5: 整个流水线的数据流向图

```
用户发送 query
    │
    ▼
process_query(text, embedding)
    │
    ├── 1. 归一化 + 记入 query_history (deque, 最多500条)
    │
    ├── 2. 从 KB 检索 top-k 文档 → 返回给用户
    │
    ├── 3. 送入 QueryWindowBuffer
    │       └── 窗口满了 (8条)?
    │             │ 是
    │             ▼
    │       _process_window()
    │         │
    │         ├── auto_kmeans(8条query) → centroids + weights
    │         │                            "用户关心什么"
    │         │
    │         ├── detector.detect(8条query)
    │         │     └── 计算对齐特征 → FID(窗口 vs 基线) → DriftResult
    │         │                         "跟KB还对齐吗"
    │         │
    │         ├── compute_alignment_gap(queries, KB) → AlignmentGapResult
    │         │                                        "具体差多少"
    │         │
    │         ├── agent.decide(drift_result, gap_result) → AgentDecision
    │         │     └── 4条规则 → action + lambda_max + eta
    │         │                   "要不要更新、换多少"
    │         │
    │         └── if action != NO_OP:
    │               curator.recurate(centroids, weights, lambda_max, eta)
    │               │  └── 子模优化 → 增量替换KB文档
    │               │
    │               _recalibrate_detector()
    │                  └── 重建对齐基线 (KB变了→特征定义变了)
    │
    └── 返回 {documents, answer, window_event}
```

---

## Q6: DriftLens 的对齐特征具体是什么？

对每条 query q，计算一个向量：

```
alignment_features(q) = [
    CosSim(q, KB主题中心_1),     ← "跟主题1多匹配"
    CosSim(q, KB主题中心_2),     ← "跟主题2多匹配"
    CosSim(q, KB主题中心_3),     ← "跟主题3多匹配"
    top_1_KB文档相似度,          ← "最相似KB文档有多像"
    top_2_KB文档相似度,          ← "第二相似有多像"
    ...,
    top_N_KB文档相似度,          ← "第N相似有多像"
]
```

维度 = K(主题数) + N(top-N) ≈ 3+5 = 8维

### 为什么这样设计？

```
直接对比query embedding:    query变了 → 但KB可能仍然覆盖 → 不应更新
直接对比document embedding: KB没变 → 永远不漂移 → 没用

对比"query跟KB的匹配度":    query对KB的匹配方式变了 → 该更新了 ✓
```

### 检测方法：

```
基线 = 历史query对齐特征的 (均值μ₀, 正则化协方差Σ₀)
窗口 = 当前窗口query对齐特征的 (均值μ_w, 正则化协方差Σ_w)

FID = ||μ₀ - μ_w||² + Tr(Σ₀ + Σ_w - 2·√(Σ₀·Σ_w))

FID > 阈值 → 对齐模式偏移 → KB需要更新
```

使用正则化 Σ + εI 确保协方差可逆（小窗口时原始DriftLens会崩溃）。

---

## Q7: auto_kmeans 和 DriftLens 里的 KMeans 有什么区别？

| 属性 | auto_kmeans (interest_model) | KMeans (drift_detector) |
|------|-----|-----|
| **聚什么** | 当前窗口的 query | 当前 KB 的 document |
| **目的** | 发现用户兴趣主题 | 定义对齐特征的参考坐标系 |
| **输出给谁** | curator (选文档用) | alignment_features (检测用) |
| **何时跑** | 每个窗口都跑 | 每次KB更新后重新跑 |
| **K 的选法** | 自动搜索 (轮廓系数) | min(n_clusters, n_kb/3) |

**它们聚类的对象完全不同**：一个聚 query，一个聚 document。

---

## Q8: Warmup 期（前3个窗口）为什么总是激进更新？

```
窗口1: KB 是冷启动的 (多样性最大化选的，不针对任何用户)
       → DriftLens 还没初始化 (需要积累够历史才行)
       → 反正 KB 也不好，先激进换

窗口2: KB 稍微好一点了，但还不稳定
       → 继续激进换

窗口3: KB 经过2次激进更新，已经比较对齐
       → 最后一次激进换 + 用积累的 24 条 query 初始化 DriftLens
       → 从此以后进入"正常检测模式"
```

相当于 DriftLens 论文里的 Offline Training Phase：
先积攒数据 → 训练检测器 → 然后才能做 Online Detection。

---

## Q9: 冷启动时 DriftLens 为什么不能立刻初始化？

DriftLens 的阈值校准需要：
1. 从历史 query 中随机采样窗口
2. 每个采样窗口计算 FID
3. 取 FID 分布的 P95 作为阈值

如果只有 8 条 query（1个窗口），采样来采样去都差不多，阈值没意义。

**需要至少 max(3×window_size, 20) 条 query**（约3个窗口）才能有效校准。

所以流水线设计为：
```
warmup_windows=3 期间: Agent 强制激进更新 (不依赖检测)
warmup 结束后:        用积累的 query 历史初始化 DriftLens
之后的窗口:           正常 detect → decide → execute 循环
```

---

## 模块职责一览

```
interest_model.py    "用户关心什么"    QueryWindowBuffer + auto_kmeans + AlignmentGap
drift_detector.py    "跟KB对齐吗"     对齐特征 + 正则化FID + 阈值校准
kb_agent.py          "要不要换"        4条规则 + 自适应Gap阈值 + Warmup + 冷却
kb_curator.py        "换什么文档"      子模优化 + 贪心选择 + 增量替换
pipeline.py          "把以上串起来"    Bootstrap → 窗口循环 → query历史管理
```
