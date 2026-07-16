# M-RAG 与 DRIP 分区路由审计（2026-07-16）

## 结论先行

M-RAG 可以支持“先分区、再检索”的架构动机，但不能直接作为 DRIP 的缓存
算法。M-RAG 的动作是选择数据库分区和改写 demonstration memory，奖励是当前
生成结果的 ROUGE/BLEU 改善；DRIP 的动作是把具体 evidence 写入共享 hot tier，
奖励主要来自未来请求是否复用该 evidence，并且必须支付 cold-read 与物理写入
成本。二者的 action、reward horizon 和系统约束均不同。

本次审计还发现并修复了 controlled runner 的因果时序错误：旧 runner 在评分后
把 gold evidence embedding 交给 DomainAdapt，因而旧结果只能解释为
post-service evidence-neighborhood placement，不能证明 current-query routing。
causal-v2 runner 现在在评分前使用原始 query embedding 生成路由，并把 gold
evidence ID 严格保留到评分之后。

## 1. 当前 DRIP 如何分区

设归一化文档向量为

\[
x_d=\frac{e(d)}{\lVert e(d)\rVert_2}.
\]

`SemanticTopicPartition` 先在归一化向量上执行 MiniBatchKMeans：

\[
\min_{\{c_z\},a_d}\sum_d \lVert x_d-c_{a_d}\rVert_2^2.
\]

拟合后把中心再次归一化。主分区使用 cosine affinity 决定：

\[
z^*(d)=\arg\max_z x_d^\top c_z.
\]

每篇文档还保留 Top-m 软归属，供诊断和 post-service evidence histogram 使用：

\[
p(z\mid d)=
\frac{\exp(x_d^\top c_z/\tau)}
{\sum_{j\in\operatorname{TopM}(d)}\exp(x_d^\top c_j/\tau)}.
\]

受控主实验使用更严格的 balanced semantic pages。先做 spherical-like KMeans，
再把超出 page capacity 的文档按次优 centroid affinity 重分配，满足

\[
|P_z|\le S,\qquad C=\lceil N/S\rceil.
\]

这样 route width 的 cold traffic 上界可审计；普通 KMeans 的极大 bucket 不会被
伪装成一次廉价 partition probe。

在线路由保持两级分辨率：

\[
r_t(z)=e(q_t)^\top c_z,
\qquad
\mathcal Z_t=\operatorname{TopL}_z r_t(z),
\]

\[
C_t(q)=\operatorname{TopK}_{d\in\cup_{z\in\mathcal Z_t}P_z}
e(q_t)^\top e(d).
\]

分区只缩小搜索空间，hot tier 的 cache object 始终是具体文档。主配置中的历史
topic prior 权重为 0，因为它在先前 held-out audit 中降低结果。

## 2. M-RAG 实际做了什么

M-RAG（ACL 2024）尝试 Randomization/LSH、K-means Clustering、图上的 spectral
partitioning，以及 Category partitioning。分区策略和数量由开发集按生成任务选择；
论文最终分别采用 summarization 的 Indexing-4、translation 的 Randomization-3，
以及 dialogue 的 Category-10。

Agent-S 构造一个 M 维状态。每一维是在相应分区内执行 Top-1 retrieval 得到的
最大相似度：

\[
s_m^{(S)}=
\max_{(\tilde x,\tilde y)\in D_m}
\operatorname{sim}(\sigma(\tilde x\oplus\tilde y),
\sigma(x\oplus y)).
\]

动作是选择一个分区，\(a^{(S)}=m\)。Agent-R 在被选分区中用冻结 LLM 生成 K
个候选 response memory，再选择一个候选。它的即时奖励是当前生成指标的改善：

\[
r_t^{(R)}=\Delta(h',y)-\Delta(h,y),
\]

该累计奖励再共享给 Agent-S。两个策略均由 DQN 和 replay memory 训练。

与 DRIP 冲突的地方：

1. M-RAG 没有 cache capacity、replacement、write cost 或 churn 目标；
2. Agent-R 改写 response/demonstration，不是选择持久 evidence residency；
3. reward 使用当前样本 reference 与 LLM generation，DRIP 的主要收益在未来窗口；
4. Agent-S 为构造状态需要先探测所有 M 个分区，论文给出的检索复杂度是
   \(O(M\log N)\)，与 DRIP 限制 cold reads 的目标冲突；
5. 实验是静态训练/验证/测试语言生成任务，没有非平稳 query stream 或共享缓存；
6. DQN replay 在非平稳 reward 下可能保留过时经验，M-RAG 没有 dynamic-regret
   或 switching-cost 保证。

因此，M-RAG 是 partitioned retrieval baseline/related work，而不是可直接复用的
cache baseline。

## 3. causal-v2 协议修复

旧流程：

1. 用 persistent cache 评分；
2. 展平 gold evidence；
3. 把 `doc_embs[gold_positions]` 交给 `step`；
4. DomainAdapter 在 step 内首次路由。

修复后：

1. 从当前窗口的 `qidx` 读取原始 query embeddings；
2. 对多 support query 只为事件对齐重复同一 query embedding；
3. 在评分前调用 `prepare_window`；
4. persistent-cache hit 只由评分前 hot tier 计算；
5. 单独报告 routed region recall、routed candidate recall 与 candidate reads；
6. 评分后才向 `step` 暴露 evidence IDs。

实现位置：

- `benchmarks/run_controlled_topic_trace.py`
- `algorithms/drip/tests/test_domain_adaptation.py`

## 4. 修复后的结果

### SQuAD recurring，五个 held-out construction seeds

配置：1,200 source docs，20x25 queries，cache 24，write cap 5，balanced page 16，
route width 2，window candidate budget 24。

| Policy | Evidence hit (%) | Std (pp) | Mean writes |
|---|---:|---:|---:|
| LRU | **22.44** | 0.29 | 98.4 |
| Classical ARC | 19.16 | 0.70 | 98.0 |
| AgentRAGCache | 18.08 | 0.72 | 75.2 |
| DRIP-Reactive | 20.72 | 0.24 | 30.0 |
| DRIP-DomainAdapt | 21.84 | 1.40 | **27.4** |

DomainAdapt 相对 Reactive 平均增加 1.12 hit points，并少 2.6 次写入，但仍低于
LRU 0.60 points；平均 routed candidate recall 为 53.08%，每个 seed 产生 480 次
candidate reads。这个点支持“较少写入下的 routing gain”，不支持全成本 SOTA。

在 candidate budget 4 下，DomainAdapt 是 20.96%，Reactive 是 20.72%，route
recall 仅 15.20%，candidate reads 降为 80。收益只有 0.24 points，说明存在明显的
hit--read trade-off。

### SQuAD 边界场景（seed 42）

| Stream | LRU hit | Reactive hit | DomainAdapt hit | Domain writes | Route reads |
|---|---:|---:|---:|---:|---:|
| Recurring | 22.0 | 20.6 | 23.4 | 32 | 480 |
| Shuffled | 15.8 | 19.8 | 19.8 | 28 | 480 |
| Stationary | **76.2** | 69.4 | 64.0 | 27 | 480 |

DomainAdapt 在 shuffled 不增加 hit，在 stationary 比 Reactive 低 5.4 points。
分区路由不应在 stationary/high-locality stream 上无条件开启。

### FEVER causal-v2（seed 42）

| Policy | Evidence hit (%) | Writes | Routed reads |
|---|---:|---:|---:|
| LRU | **17.93** | 100 | 0 |
| DRIP-Reactive | 15.59 | 29 | 0 |
| DRIP-DomainAdapt | 15.98 | **27** | 480 |

DomainAdapt 仅比 Reactive 高 0.39 points；candidate recall 为 39.18%，region recall
为 48.93%。旧的 16.96% 结果来自错误的 gold-evidence embedding 路由，不能继续
引用。

## 5. 两个学习式变体的 falsification

### Contextual history residual

实验保存相似历史 query 到事后 evidence region 的映射，用 kNN residual 修正
centroid score。SQuAD seed 42 上，它把 region recall 从 64.2% 提到 77.6%，candidate
recall 从 54.0% 提到 61.4%，但 cache hit 均为 23.4%。在五个 calibration seeds、
candidate budget 4 下，静态 DomainAdapt 平均 21.16%，contextual variant 只有
20.56%。结论：更准确识别 region 没有自动转化为更好的 residency。

### Downstream-validated candidate credit

该变体模仿 M-RAG 的 downstream reward：只有 selected region 覆盖事后 evidence
时才奖励 routed neighbors，并排除 gold 本身的重复 credit。五个 calibration seeds
平均只有 19.80%，低于静态 DomainAdapt 的 21.16%，写入还从 30.8 增至 32.4。

两个失败分支均已从核心代码删除，只保留 JSON 结果作为 negative evidence。

### Delayed cost-aware route gate

该变体在 `{0,4,12,24}` 个 routed candidates 之间使用 discounted UCB，reward
是“上一窗口候选在下一窗口 evidence 中的 recall”减去读取成本。五个 calibration
seeds 上，它把 routed reads 从固定路由的 480 降到平均 180.8，但 hit 从 20.80%
降至 20.36%，低于 Reactive 的 20.56%，写入从 27.6 增至 31.4。日志中除一个窗口
外，跨窗口 candidate overlap 均为 0；该反馈不足以学习有效 gate。实验模块已删除。

### Region-conditioned placement

将当前 evidence-region distribution 直接加入 document utility，同样失败。seed-11
calibration 上，placement weight 0.25/0.5/1/2 的 hit 分别为
19.4%/19.2%/19.4%/19.6%，均低于 Reactive 20.6%，且写入更多。该分支也已从核心
代码删除。

## 6. 推荐的算法方向

### A. 当前应采用：严格拆开 current retrieval 与 future residency

causal-v2 证据表明 routed candidates 对下一窗口的直接重叠近乎为零。区域路由
应该只回答“当前请求去哪里检索”，而 persistent cache admission 只接受当前请求
真实取回、并由后续访问证明具有复用价值的具体 evidence。不要再把未验证的语义
邻居作为未来 cache demand。换言之，主系统应报告两条相互独立的曲线：

- current retrieval：support recall 对 cold reads；
- persistent cache：future exact-evidence hit 对 writes/churn。

如果论文继续以 cache 为核心，DomainAdapt 应降级为可插拔 cold-index 层，核心算法
回到 document-level cost-aware replacement。

### B. 值得继续的学习问题：multi-view current-query routing

M-RAG 最有价值的观察是 partition strategy 依任务变化。可把 current-query arm 定义为
`semantic`、`entity/metadata`、`union` 和 `no-route`；single-hop/entity query 与
multi-hop/concept query 可以选择不同 view。每个 view 内仍按 centroid/ANN 检索，
hot-tier placement 不接收 arm/topic bonus，继续由同一 switching-price controller
管理。必须比较：

- semantic only；
- metadata/entity only；
- fixed union；
- learned view gate；
- full-pool dense retrieval upper bound。

### C. 若坚持优化 cache：转向 set-level online placement

独立 document score 无法表达容量内文档之间的互补与重复。下一种真正不同的 cache
算法应直接优化 resident set，而不是再加 topic score：

\[
X_{t+1}=\arg\max_{|X|\le B}
\widehat F_t(X)-\lambda_t|X\triangle X_t|,
\]

其中 \(\widehat F_t(X)\) 只能由已完成请求的 exact-evidence reuse 构造，并通过
facility-location/coverage 或 submodular marginal gain 抑制重复文档。它需要与
document-wise utility、TinyLFU 和 oracle Belady 上界比较。若这种 set-level objective
仍不能超过 Reactive/LRU，则当前数据流没有足够可预测结构，应该如实收缩论文贡献。

## 来源

- [M-RAG, ACL 2024](https://aclanthology.org/2024.acl-long.108/)
- [MBA-RAG, COLING 2025](https://aclanthology.org/2025.coling-main.218/)
- [AURORA continual indexing, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.495/)
- [SOLAR semantic replacement preprint](https://arxiv.org/abs/2607.00394)
