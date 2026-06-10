# 算法代码解释（给作者读懂自己的代码）

> 目的：把你方法（当前代码包名 `algorithms/qarc/`，论文名 DRYAD）的每个模块讲清楚——
> 每个文件干什么、核心公式、参数从哪来、哪些是论文给的哪些是我们补的默认值。
> 同时澄清命名混乱、解释新加的两个 baseline 为何会差。
>
> ⚠️ 命名待统一：见本文 §0。代码里 `qarc` = 历史代号，论文叫 DRYAD，新框架（跨层预取）还没定名。

---

## §0 命名总表（先看这个，否则后面会乱）

| 名字 | 出现位置 | 实际含义 | 建议 |
|---|---|---|---|
| **QARC** | `algorithms/qarc/` 包名、所有 import | 历史代号 = 漂移检测+决策+策展骨架 | 🔴 论文不用，待改成统一名 |
| **DRYAD** | docs、论文草稿 | 旧框架算法名（热库子模策展） | 🟡 旧名 |
| **DRYAD / RoutedCache** | `cache/ours/dryad.py`、`routed_cache.py` | motivation 实验台后端（demand-ledger + 实体桥接） | 🟡 实验台实现 |
| **SemFlow / QueryDriven** | `cache/ours/query_driven.py` | demand-ledger 准入（无 regime 分支） | 🟢 组件名，可留 |
| **(待定)** | `docs/design/REFRAME_PREFETCH.md` | 新框架 = drift-aware 跨层预取 | 🔵 还没名字，待你拍板 |

**核心混乱**：包名(QARC) ≠ 论文名(DRYAD) ≠ 新方向(预取)。建议定一个论文名，把 `algorithms/qarc/`
连同所有 import 全局替换掉，SemFlow/RoutedCache 作为模块名保留。

---

## §1 你的方法 = 三个模块串成一条流水线

```
                  algorithms/qarc/pipeline.py  (主流水线, 381 行)
                            │
   ┌────────────────────────┼────────────────────────┐
   ▼                        ▼                        ▼
 ① detection/          ② decision/             ③ curation/
 drift_detector.py     kb_agent.py             interest_model.py
 (检测对齐漂移)         (决定换不换/换多少)      + kb_curator.py
                                                (选哪些文档进 KB)
```

一句话：**①检测 query 与 KB 的对齐是否漂移 → ②Agent 据此决定更新力度 λ·B → ③子模函数选文档换入换出。**

---

## §2 模块① 漂移检测 `detection/drift_detector.py`（356 行）

**类**：`DriftLensDetector`

**它解决什么**：不检测"query 变没变"，而是检测"**query 与 KB 的对齐模式**是否偏离历史正常水平"。
因为 query 变了不一定要更新 KB（新 query 可能仍被 KB 覆盖），只有 query-KB 不对齐了才要更新。

**是不是自己实现的 / 能不能引用源码**：
- **理论来自 DriftLens 论文**（Greco et al. ICDMW 2021 原版 / arXiv:2406.17813 2024 完整版），方法是
  在 embedding 上用 per-label Fréchet 距离做漂移检测。
- **我们做了两处改写**（所以是 adapt 不是照搬）：
  1. **特征空间换了**（核心创新）：原版在 raw embedding 上算 FID；我们在 **query-KB 对齐特征**上算。
  2. **正则化修复**：原版小窗口下协方差奇异会崩，我们加 `Σ + ε·I`。
- **论文写法**：cite DriftLens 原文，措辞 "we adapt DriftLens to operate on query-KB alignment
  features with a regularized covariance"。代码顶部 docstring 已标出处。

**对齐特征**（`_compute_alignment_features`，第 102 行）：对每个 query 算
```
alignment_features(q) = [ sim(q,c₁),...,sim(q,c_K),  top1_sim,...,topN_sim ]
```
- `c_k` = KB 文档 KMeans 的 K 个聚类中心（主题中心）。
- `topN_sim` = q 与最相似 N 篇 KB 文档的相似度。

**正则化 FID**（`_regularized_fid`，第 137 行）：
```
FID = ‖μ₁−μ₂‖² + Tr(Σ₁ + Σ₂ − 2·√(Σ₁Σ₂))
```

**离线基线数据从哪来**（关键问题）：**不是外部数据集，是系统 warmup 阶段自己攒的 query 历史。**
- `set_baseline(kb_embs, query_embs)`（第 175 行）：用「warmup 攒的历史 query + 当前 KB」算对齐特征，
  存基线 μ₀/Σ₀。
- `calibrate_threshold`（第 232 行）：随机采样历史 query 窗口算 FID 分布，取 **P95** 当阈值（5% 假阳率）。
- `detect`（第 291 行）：当前窗口 FID > 阈值 → 判定漂移。
- 这就是论文 "annotation-free" 站得住的原因：基线是正常对齐的自采样，无需标注、无需外部 drift-free 数据。

**参数**：`n_clusters=5`（主题数）、`top_n_sims=10`、`threshold_percentile=95`、`cov_reg=1e-5`（ε）。
这些是 DriftLens-style 检测的标准设置，非论文硬性规定，可调。

**检测器 baseline（消融用，`detection/baseline_detectors.py`）**：同接口可插拔替换 DriftLensDetector：
- `NoDetector`：永不报漂移 → 消融底（"不检测会怎样"）。
- `ADWINDetector`：标量 change-point 族代表（Bifet & Gavalda）。在「query 到 KB 的 max 相似度」标量流上
  用 ADWIN2 Hoeffding-Bernstein 界检测。比 FID 慢几窗才触发——体现标量法不如对齐特征灵敏。
- `MMDDetector`：分布距离族。在 **raw query embedding**（非对齐特征）上算 RBF-MMD → 证明"对齐特征优于 raw"。

---

## §3 模块② 决策 Agent `decision/kb_agent.py`（360 行）

**类**：`KBUpdateAgent`（`UpdateAction` 枚举 + `AgentDecision`）

**它解决什么**：替代 DriftLens 论文里的 Human-in-the-Loop，自动决定"**换不换、换多大力度**"。

**三个观察信号**：① drift 是否触发 ② AlignmentGap 大小 ③ 历史趋势（连续漂移次数、Gap 的 EMA/MAD）。

**决策规则**（`decide`，第 169 行）：

| 规则 | 条件 | 动作 | 替换比例 λ |
|---|---|---|---|
| Warmup | 前 N 窗 | 激进更新 | 0.5 |
| Rule 1 | 未漂移 & Gap 正常 | NoOp | — |
| Rule 2 | 未漂移 & Gap > EMA+k·MAD | 轻度更新 | 0.2 |
| Rule 3 | 已漂移 | 激进更新 | 0.5 |
| Rule 4 | 连续 N 次漂移 | 重校准 + 更新 | 0.5 + 重建基线 |

**Gap 自适应阈值**（`gap_threshold`，第 161 行）：`EMA(Gap) + k·MAD(Gap)`，不用固定阈值。

**冷却机制**：更新后若干窗口内不再触发，防抖。

**这是 ARC 没有的**：ARC 每 query 逐项逐出、无 λ·B 预算概念；你这里是"按检测到的漂移强度主动定档替换比例"。

<!-- PART2 -->

---

## §4 模块③a 兴趣模型 `curation/interest_model.py`（374 行）

| 组件 | 作用 |
|---|---|
| `QueryWindowBuffer` | 滑动窗口，积累 W 条 query embedding |
| `auto_kmeans(X)` | 对窗口内 query 做 Cosine K-Means，用轮廓系数自动选最佳 K |
| `compute_alignment_gap(Q, KB)` | `G(t) = 1 − avg max_sim(q, KB)`，衡量 KB 对齐度（越大越不对齐） |

AlignmentGap 是模块②决策的输入信号之一；auto_kmeans 的兴趣中心 + 权重喂给模块③做子模选择。

---

## §5 模块③b 子模 KB 策展 `curation/kb_curator.py`（985 行）

**类**：`QARCKBCurator`（含 `DocumentPool` 文档池 + `Document` + 贪心子模函数）

**核心**：把 KB 更新建模为**子模函数最大化**：
```
max_{S:|S|≤B}  f(S) = Σᵢ wᵢ · max_{d∈S} CosSim(cᵢ, d)  +  η · Diversity(S)
```
- `wᵢ` = 兴趣权重（auto_kmeans 各簇比例），`cᵢ` = 兴趣中心，`η` = 多样性正则系数。
- **理论保证**：单调子模 + 贪心 → ≥ (1−1/e) ≈ 0.632 最优近似。

**更新流程 `recurate`**（第 600 行）：① 算 KB 中每篇文档边际贡献 → ② 移除贡献最低的 ⌊λ·B⌋ 篇
（λ 来自模块②决策）→ ③ 用兴趣中心检索候选 → ④ 贪心子模填空位 → ⑤ 保证 KB ≤ budget。

**Bootstrap**：冷启动 `bootstrap_diversity()`（MaxMin 多样性）；热启动 `bootstrap_from_queries()`（历史 query 兴趣加权）。

**性能**：`_compute_gain`（第 401 行）用增量边际增益，比朴素子模快 100-200 倍。

---

## §6 主流水线 `pipeline.py`（381 行）

**类**：`QARCPipeline`（`QARCPhase` 枚举：BOOTSTRAP / WARMUP / ONLINE）

```
bootstrap(): 文档池 → 初始 KB；DriftLens 延迟初始化（warmup 后才有足够 query 历史）
process_query(): 缓冲 query embedding + KB 检索
_process_window()（窗口满时）:
  a. auto_kmeans → 兴趣中心+权重
  b. DriftLensDetector.detect → 漂移?
  c. compute_alignment_gap → Gap
  d. KBUpdateAgent.decide(drift, gap) → 决策(λ)
  e. 执行 recurate(λ) + 重建 DriftLens 基线
```
`_init_detector`（第 160 行）：攒够 `max(3×window_size, 20)` 条 query 才初始化检测器。
`_recalibrate_detector`（第 182 行）：KB 更新后用近期 query 重建基线（因对齐特征定义依赖当前 KB）。

---

## §7 新加的两个 baseline + 为什么它们会差（重点回答）

> 你问："Proximity 和 ARC(w/o hubness) 由于没用到多跳和用户漂移信息，这样会差吗？解释一下。"
> 答：**会差，而且这正是要展示的——但必须确保是"方法本身够不到"而非"我们故意调废"，否则审稿人判稻草人。**

### 7.1 ARC(w/o hubness) — `AgentRAGCache(use_hubness=False)`，注册名 `AgentRAGCache_NoHub`
- 这是**论文自己的消融**（ARC §5 报了 "ARC w/o hubness" 一行），不是我们发明的弱化版。
- 去掉 hubness 后 `Priority(p) = DRF(p)`（纯距离-秩-频率），少了 embedding 空间中心性这一项。
- **会差多少**：论文 Table 1 显示带 hubness 在 SQuAD 上 +3pp 左右。所以它**不会差很多**——它仍是个强 baseline，
  只是少一个组件。用途：证明 hubness 这一项的边际价值。

### 7.2 Proximity (Bergman 2025) — `algorithms/cache/semantic/proximity.py`
机制：维护 (过去 query → 它取到的文档) 的 FIFO 账本；新 query 若与某过去 query 相似度 ≥ τ 就复用其文档，
否则 miss 取 top-1 入 KB；超预算按 query-doc 对 FIFO 逐出。

**为什么在「多跳 + 漂移」下会差（两个 gap，都是方法固有，非人为设障）**：

1. **没有多跳/bridge 概念**。Proximity 复用的是"相似过去 query 取到的文档"，而那次取还是 plain
   `sim(q, doc)`。bridge 文档（第二跳需要、但与 query 不相似）**任何过去 query 都不会取到它**，
   所以永远进不了账本、永远无法复用。→ 这是机制决定的，不是我们限制的。

2. **没有漂移处理**。账本只按"年龄"FIFO 逐出 + 相似度门控。query 分布一漂移：
   (a) 旧 (query,doc) 对变得不相关，却只因"老"才被逐出，不是因"失去相关性"；
   (b) 漂移后的新 query 找不到相似的过去 query（cos < τ），于是**整个漂移过渡期一直 miss**，
   直到账本重新填满——**没有任何机制去"检测"漂移并提前适应**。它付出完整的 miss 代价。

→ 这两个 gap **正是你方法主攻的两条轴**（漂移检测 + 实体桥接）。GPTCache 同样有 gap 1 和 gap 2。
所以它们在多跳+漂移设定下落后，是"gap 真实存在且有后果"的证据，不是赢了个被打残的对手。

### 7.3 防稻草人的硬要求（写进实验时务必遵守）
- 每个 baseline 都用**论文忠实机制** + **同样的 budget / cost 记账**（代码已保证：都走 `BaseStrategy`，
  `maint_retrieval_cost` / `update_cost` 统一）。
- 不要给 baseline 设比自己方法更小的 budget、更少的检索预算。
- 报告时按 ARC 论文口径：**best 加粗、second-best 下划线**，给 has-answer + AMAT 两表。
- **关键**：要单独给一组"无漂移/单跳"的对照——如果在那里 Proximity/GPTCache 不差，
  就证明它们的差**专门来自**漂移+多跳，而非实现劣质。这是反稻草人的最强证据。

---

## §8 参数澄清（你问"参数是不是太多"）

`AgentRAGCache` 的 5 个参数**全部来自 ARC 论文**，没有一个是我们加的：

| 代码参数 | 论文符号 | 论文取值 | 备注 |
|---|---|---|---|
| `ALPHA` | α (DRF 距离敏感度) | 0.4 | 论文给定 |
| `BETA` | β (hubness vs DRF) | {0.7, 0.15, 0.2} 按数据集 | 论文按集变；代码默认 0.3，跑实验按表设 |
| `TAU` | τ (escalation 阈值) | 0.2 | 论文给定 |
| `TOP_K` | K (检索宽度) | 50 | 论文给定 |
| `HUB_K` | hubness 的 k | 论文未给具体值 | 我们补默认 10 |

→ 只有 `HUB_K=10` 和 `BETA` 默认值是我们补的，其余照搬论文。不是参数膨胀。


