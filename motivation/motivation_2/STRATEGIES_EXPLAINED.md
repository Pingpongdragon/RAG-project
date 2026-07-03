# Motivation 2 — KB Update Strategies 代码详解

> 文件：`strategies.py` · 配置：`config.py` · 流构造：`utils.py`

---

## 0. 公共基类 BaseStrategy

所有策略共享同一接口，每个 window 流程：
1. `prepare_window(...)` — 仅 OnDemandFetch 用，预取"虚拟 KB"
2. **外部评测**（run.py 在 prepare→评测→step 之间算 KB Coverage / Recall@K）
3. `step(window_queries, window_query_embs, window_idx)` — 更新持久化 KB

关键字段：
- `self.kb: set[doc_id]`
- `update_cost`：累计 KB 写入次数
- `maint_retrieval_cost`：离线维护时池查询次数
- `serve_retrieval_cost`：在线 per-query 池查询次数（仅 OnDemandFetch）

关键超参（`config.py`）：

| 常量 | 值 | 用途 |
|------|----|------|
| `SF_HIT_THRESH` | 0.55 | query 与 KB doc 余弦 ≥ 0.55 算 hit |
| `WRITE_CAP` | — | 通用写入帽（DocArrival/KnowledgeEdit/QDC/LogDriven 共用） |
| `PROBE_TOPK` | 8 | 失败 query 在池里取的候选宽度 |
| `DOC_ARRIVE` | 80 | DocArrival 每 window 随机采样"到货"数 |
| `FIFO_BATCH` | 40 | RandomFIFO 每 window 替换数 |
| `LOG_LAG_WINDOWS` | 5 | LogDrivenArrival 的审查周期 |
| `n_windows` | 100 | 总 window 数 |
| `window_size` | 50 | 每 window query 数 |

---

## 1. Static — 基线（冻结）

```python
class Static(BaseStrategy):
    def step(self, ...): pass
```

KB 在初始化后**永不更新**。用于显示 drift 后曲线"自由下落"的下限。成本恒为 0。

---

## 2. RandomFIFO — 盲目 supply-side

```python
arrivals = rng.choice(all_ids, FIFO_BATCH=40, replace=False)
for did in arrivals:
    old = insert_order.pop(0)   # 最旧的先走（FIFO）
    kb.discard(old); kb.add(did)
```

每 window 随机抽 40 个池文档，按 FIFO 替换 KB 最旧条目。与 query 完全解耦，论证"没有 relevance 信号的 ingest 注入噪音的速度比注入有用知识更快"。

---

## 3. DocArrival — HippoRAG2 / LightRAG 风格

```python
arrivals = rng.choice(all_ids, DOC_ARRIVE=80)
for did in arrivals:
    sims = kb_emb @ new_emb
    if sims.max() > 0.7:   # 与 KB 高度相似 → 替换最相似 KB doc（"更新版本"）
        replace_argmax(kb, did)
    elif sims.max() < 0.3: # 与 KB 完全无关 → 淘汰最 stale KB doc（新主题插入）
        replace(stale, did)
    # 灰色地带 (0.3–0.7) 跳过
```

只看**文档侧**相似度，不看 query。双阈值模拟 HippoRAG2 的"高度相似就替换、完全新就插入、灰色地带丢弃"。漂移后到货流仍按池分布走，KB 更新方向偏旧分布。

---

## 4. KnowledgeEdit — RECIPE 风格

```python
targets = rng.choice(kb_list, EDIT_BATCH)
for tid in targets:
    sims = doc_embs @ doc_embs[tid]    # 找 tid 在整个池里的近邻
    sims[kb_idx] = -1                  # 排除已在 KB 内的
    cands = where((sims > 0.4) & (sims < 0.8))
    replace(tid, argmax(sims[cands]))  # 用最相似的外部 doc 替换
```

比 DocArrival 主动：主动选 KB 内文档，找池里"语义相邻"版本换掉。但仍无 query 信号——drift 后这些 edit 还在**旧 KB 局部邻域**打转，换进来的仍是旧分布近邻。

---

## 5. OnDemandFetch — CRAG / Agent-RAG 风格

```python
def prepare_window(...):
    for q in window_queries:
        if max(q · kb) >= SF_HIT_THRESH: continue
        top = argpartition(q · pool, -FETCH_TOP_K)
        fetched_this_window |= set(top)   # 临时 augment，本 window 可见
        serve_retrieval_cost += FETCH_TOP_K

def get_effective_kb(...): return kb | fetched_this_window
def step(...): pass   # KB 永不持久化更新
```

KB 完全不更新，但每 window 为失败 query 临时拉池 top-K。评测用 `get_effective_kb()`，所以你看到 drift 后它的 KB Coverage 比 Static 高——**那是临时的，不算持久化**。用来论证："就算外部检索完美，也无法替代 KB 巩固"（每 query 付 latency、重复拉、跨 window 不学习）。

---

## 6. LogDrivenArrival — 滞后日志分析

```python
def step(...):
    # A) 周期边界 (window_idx % 5 == 0)：执行上轮积累的 fix
    if pending_adds and window_idx % LOG_LAG_WINDOWS == 0:
        usefulness = mean(window_q · kb_emb)
        evict_order = argsort(usefulness)           # 最不被本 window 用的先走
        for did in pending_adds: replace(least_useful, did)

    # B) 每 window 都把失败 query 累进 buffer
    fail = (max(q · kb) < SF_HIT_THRESH)
    fail_buffer.append(nqe[fail])

    # C) 周期末 ((window+1) % 5 == 0)：分析 buffer，决定下一轮要加什么
    if (window+1) % LOG_LAG_WINDOWS == 0:
        for q in fail_buffer: cand_set |= top-K(q · pool)
        pending_adds = top-LOG_FIX_CAP by mean(fail_buffer · doc_emb)
```

有 query 信号，但存在 **5-window 滞后**：下一轮修复用的是上一轮失败分布，drift 已往前走。用来反衬 QDC 的"实时累积"价值。

---

## 7. DRIP-Dense (QDC) — 本文方法

> 核心：**两个衰减计数 (`demand`, `serve`) + 一个 admission gate；写入量自动正比于漂移强度。**

### 数据结构

```python
demand[pool_idx]: float  # 累计"被失败 query 看上"的归一化相似度，衰减率 0.85
serve[pool_idx]:  float  # 累计"在 KB 里被成功 query 用过"的次数，衰减率 0.92
SERVE_PRIOR = 1.0        # 初始 KB 每个 doc 拿 1 单位 serve（贝叶斯先验）
```

### 单 window 流程

```python
def step(...):
    max_s = (nqe @ kb_emb.T).max(axis=1)
    self._decay()   # demand *= 0.85; serve *= 0.92; 低于 0.01 删除

    # (1) 成功 query → 给最佳 KB doc +1.0 serve
    for q in succ: serve[argmax(q·kb)] += 1.0

    # (2) 失败 query → 取池 top-8，归一化后累积到 demand（∑weights=1.0，与 serve 单位对齐）
    for q in fail:
        top8 = top-8(q · pool)
        weights = sims[top8] / sims[top8].sum()
        for w, pi in zip(weights, top8): demand[pi] += w
        maint_retrieval_cost += 8

    # (3) Candidate (非 KB)：按 demand desc
    cands = sorted(non-KB, key=demand desc)

    # (4) Eviction (KB)：按 serve+demand asc
    evict_val[e] = serve[e] + demand[e]
    evictable = sorted(KB, key=evict_val asc)

    # (5) 统一准入门：demand[c] > evict_val[e] 才替换
    for (cval, c), e in zip(cands, evictable):
        if cval <= evict_val[e]: break  # 降序 vs 升序 ⇒ 后面更不满足，直接停
        replace(e, c); serve.pop(e)
```

### 设计要点

| 要点 | 实现 | 原因 |
|------|------|------|
| 写入帽 = 失败数 | 无显式 cap，gate 自限 | 漂移强→失败多→写多；漂移弱→几乎不写 |
| demand/serve 单位对齐 | 成功: serve+=1; 失败: ∑demand+=1 | 门槛是同尺度比较，无需额外超参缩放 |
| demand 衰减更快 (0.85 < 0.92) | 旧失败信号比旧服务信号更早过期 | 与 drift 节奏匹配 |
| SERVE_PRIOR=1.0 | 初始 KB doc 各发 1 单位 serve | 防 cold-start 首 window 被撬；几 window 无用后自然衰减为 0 |
| serve.pop(ep) 在替换时 | 新装入的 doc 无历史 serve | 避免旧 serve 分数跟着 pool_idx 被继承 |

### 为何击败各 baseline

- 击败 **Static**：会变。
- 击败 **RandomFIFO / DocArrival**：用 query 失败信号，写的是 workload **真需要**的 doc。
- 击败 **KnowledgeEdit**：用 query-side 信号而非 KB-internal 相似度（后者在漂移下是错信号）。
- 击败 **LogDrivenArrival**：实时累积，无 5-window 滞后；反复失败的 doc demand 持续积累。
- 击败 **OnDemandFetch**：持久化 → serve 时 0 额外 latency；admission gate 让它"懂得不写"。

---

## 8. Oracle — 上界（非 causal）

```python
def step(...):
    sf_pool = {title_to_idx[t] for q in window for t in q.sf_titles}  # 用真值 SF 标题
    doc_scores = mean(norm_qe @ doc_embs.T, axis=0)
    new_kb = top-budget(sf_pool by doc_scores)
    new_kb |= top-(budget-|sf_pool|)(pool\sf_pool by doc_scores)   # 补足 budget
    kb = new_kb   # 每 window 全重建
```

使用未来真值（SF 标题），是固定 KB 容量下 Recall@K 的**严格上界**。漂移前后均为平线，证明"如果你知道查询的支持文档，完全可以一直在天花板"——所有 gap 都是"未知 query 分布"造成的。

---

## 9. Query Stream 构造与 Drift 模式

> 代码：`utils.py` — `cluster_and_build_stream`

```python
KMeans(n_clusters=8) on all query embeddings (all-MiniLM-L6-v2)
head = top-3 最大 clusters; tail = 其余 5 个
H1 (前 50 windows, 2500 queries): 97% head + 3% tail
H2 (后 50 windows, 2500 queries): 取决于 drift_mode
```

参考方法论：Lupart et al., *MS-Shift*, ECIR 2023（query 语义偏移建模）。

### sudden drift

```python
H2: 3% head + 97% tail (一刀切)
```

**实际效果（数据验证）**：Static KB Coverage 在 w50 从 ~100% 直接跌到 ~20-25%，单 window 断崖。

### gradual drift

```python
H2 内部每个 window 单独设 head 比例：
  head_pct(wi) = 0.97 - wi * (0.94 / 49)   # wi=0..49, 从0.97线性降到0.03
```

**实际效果（数据验证）**：
```
static sudden  w44-65: [93, 100, 97, 92, 100, 100,  25, 20, 20, 13, 11, 20, ...]  # w50 断崖
static gradual w44-65: [94, 100, 97, 95, 100, 100,  96, 97, 96, 93, 94, 87, 83, 83, 79, 80, 26, 27, ...]  # w50 后延迟~12个window才下落
```

Gradual 比 Sudden **晚约 12 个 window** 才跌落，而不是理想的整个 H2 斜坡。

**原因**：H1 已抽走 ~97% 的 head 池，H2 开头每 window 的 `pick(heads, n_h_w)` 在约 10-12 个 window 后就抽空 head 池，之后变成纯 tail → 形成"延迟断崖"而非平滑斜坡。

**是否符合 Lupart et al.？** 意图一致（线性 ramp），但实现存在 pool exhaustion 问题。真正平滑的 ramp 需要把 H2 的 head 采样改为**有放回**或在 H2 开始前重置/扩充 head 池。

> ⚠️ 当前 gradual 图视觉上确实与 sudden 不同（延迟断崖 vs. 即时断崖），足以展示"渐变 vs. 突变"的对比，但如果 reviewer 细究 drift 构造，需要解释上述池耗尽机制。

---

## 10. 评测口径（run.py）

每 window 对每个策略：
1. `prepare_window`（OnDemandFetch 预取临时 fetch）
2. 用 **effective KB** 计算两个指标：
   - `kb_coverage_per_window`：本 window 所有 SF docs 中有多少在 effective KB 里（比例）
   - `recall@5_per_window`：top-5 检索对 SF docs 的召回率
3. `step`：策略更新持久化 KB

> OnDemandFetch 的 effective KB = `kb ∪ fetched_this_window`，其他策略 = `kb`。

---

## 11. 一图看懂"信号 vs. 成本"

| 策略 | query 信号 | 持久化更新 | 写入触发 | 主要成本 |
|------|-----------|-----------|---------|---------|
| Static | ✗ | — | 永不 | 0 |
| RandomFIFO | ✗ | ✓ | 周期 40/w | maint scan |
| DocArrival | ✗ | ✓ | 到货双阈值 | maint scan |
| KnowledgeEdit | ✗ | ✓ | 编辑预算 | 全池 NN |
| OnDemandFetch | ✓ per-query | ✗（仅临时） | 每次失败 | **serve 检索（在线latency）** |
| LogDrivenArrival | ✓ buffered | ✓ | 5-window 周期 | maint scan + 滞后 |
| **DRIP-Dense** | ✓ **累积衰减** | ✓ | 失败×gate（自适应） | maint scan |
| Oracle | 真值 SF | ✓ | 每 window 全建 | N/A，非 causal |
