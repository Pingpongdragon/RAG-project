# DRIP 数据协议与算法审计（2026-07-11）

## 1. 先给结论

旧的 cache-ratio 表不能直接进入论文。它只有 1%、5%、10% 三个点，而且使用了：

1. 基于整个未来 stream 的 `ctx_titles` 构造初始 KB；
2. 部分 drift mode 对 exact query 有放回采样；
3. router 默认读取 `qtype/route_hint`；
4. StreamingQA 由问题文本中的年份分箱，并非官方 timestamp protocol。

这些问题不一定只偏向 DRIP，也可能强化 LRU/TinyLFU，但会使比较缺少可解释性。
因此旧图统一标为 **legacy diagnostic**，新 ratio curve 必须在 causal protocol 上重跑。

## 2. 数据集合理性

| Workload | 当前判断 | 论文中的正确定位 |
|---|---|---|
| HotpotQA comparison | 合理，但漂移是人为控制的 | Controlled direct-evidence topic shift |
| 2Wiki comparison | 合理，但漂移是人为控制的 | Controlled direct-evidence topic shift |
| 2Wiki simple/mixed | 很重要的反例 | Direct router 的 hidden/mixed boundary test |
| 当前 StreamingQA loader | 不足以支持真实时间漂移 | Year-topic proxy；不能称 official temporal |
| Mind2Web cluster shift | 没有原生时间戳 | Agent/domain-shift diagnostic，不是 temporal drift |

Hotpot/2Wiki 的 KMeans + mixture shift 本身并非“不合理”。它属于标准的 controlled
covariate-shift stress test，优势是 change point 和强度可控；问题在于不能把它包装成
自然生产流。正式论文应把 controlled 与 natural workload 分开报告。

### 推荐的公开 natural workloads

1. **StreamingQA official metadata**：官方 JSONL 包含 `question_ts`、
   `evidence_ts` 和 `evidence_id`。按原生时间戳排序，而不是从问题文本抽年份。
   数据说明见 [DeepMind StreamingQA repository](https://github.com/google-deepmind/streamingqa)。
2. **MIND**：匿名用户 impression log 自带 user、timestamp、click history 和曝光文章，
   适合把 user 当作 client/agent、clicked article 当作访问目标，构造真实共享 hot tier。
   它包含约 100 万用户和 16 万篇新闻，见
   [Microsoft MIND publication](https://www.microsoft.com/en-us/research/publication/mind-a-large-scale-dataset-for-news-recommendation/)。
3. **TripClick**：2013--2020 年约 520 万次医疗搜索交互，适合 chronological
   retrieval-cache workload，见 [TripClick paper](https://arxiv.org/abs/2103.07901)。
4. **WebLINX**：100K interaction、2300 个多轮网页导航 demonstration，适合 agent
   trace 和网站 OOD；它不是自然 temporal stream，但比自定义 Mind2Web cluster
   更接近真实 agent interaction，见 [WebLINX benchmark](https://mcgill-nlp.github.io/weblinx/)。
5. **Multi-User MultiWOZ**：两位用户和一个 agent，可用于 router/多用户上下文测试，
   但没有全局真实时间，因此不应承担 temporal 主结论，见
   [EMNLP 2023 paper](https://aclanthology.org/2023.findings-emnlp.213/)。

建议主实验分成两层：MIND/official StreamingQA/TripClick 提供 natural stream；
Hotpot/2Wiki 提供 direct/hidden 机制可控的解释性测试。这样不会被批评为只在自定义
场景中制造漂移。

## 3. 新实验协议

### 3.1 Causal initialization

默认 `--init-mode causal-prefix`：只使用评估开始前的 query embedding，从 cold
corpus 检索候选形成共同初始 KB；不读取 `sf_titles`、`ctx_titles` 或未来窗口。
旧初始化改名为 `--init-mode legacy-head`，仅用于复现。

### 3.2 Query stream sampling

每个 topic/round 使用 shuffled cycle：一个分层 pool 中的 query 全部使用一次后才
进入下一轮。若目标流长于 pool，重复仍可能发生，但 JSON 会记录 exact-query
duplicate rate，避免隐藏对 LRU/frequency cache 有利的重复。

每个 runner 还输出后验 `support_reuse` 审计，包括 repeated-support rate、完整
support 已在历史出现的 query 比例、相邻窗口 support Jaccard 和最大 support 频次。
这些 gold 统计只用于判断 workload 是否真的可缓存，不参与初始化或在线策略。

Embedding cache key 绑定实际 document title/text 和 query text 的内容指纹。这样不同
data seed 即使 pool/query 数量相同，也不会误读另一 seed 的 embedding。

### 3.3 Ratio curve

新图使用 hot/pool = `{1%, 2%, 5%, 10%, 20%}`，至少 3 个 seed。每个点报告：

- Has-Answer 与 AMAT；
- Replacements / 1K queries；
- 95% bootstrap confidence interval；
- stream duplicate rate 和 init mode。

旧曲线的结果只能用于提出假设：StreamingQA 极小缓存更适合 recency，Hotpot direct
随容量增加更适合 semantic evidence。是否成立必须由新协议确认。

## 4. 三组件学术化方向

### 4.1 Detector：从 repeated MMD test 改为 sequential kernel CUSUM

当前每个窗口做一次 `p <= 0.05` 的置换检验，长时间连续检测会累积 false alarm。
更合理的是 sequential statistic：

```text
z_t = standardized_MMD(P_ref, P_t)
G_t = max(0, G_{t-1} + z_t - kappa)
alarm iff G_t > h(ARL_0)
```

阈值由目标 Average Run Length `ARL_0` 校准，评价指标是 false-alarm ARL 和 expected
detection delay，而不是单窗口分类准确率。Online Kernel CUSUM 的理论目标正是 ARL、
detection delay 和在线常数复杂度，见
[Wei and Xie, Online Kernel CUSUM](https://arxiv.org/abs/2211.15070)。

Detector 只回答 **when to forget stale state**；告警只改变 ledger retention，不同时
放大 write budget 和降低 admission margin，保持因果路径可消融。

### 4.2 Router：从 lexical heuristic 改为 selective evidence observability

当前大写实体数、cue word 和 title mention 无法区分 2Wiki direct 与 mixed hidden。
新的 router 应输出 `{direct, hidden, abstain}`，而不是强制二分类：

```text
V(q) = observability features(
  dense top-k concentration,
  independent query-span/candidate agreement,
  entity diversity,
  graph-path availability)
```

用训练 split 的 direct/hidden 标签做 conformal calibration，测试时不读取 `qtype`。
若预测集合只有 `{direct}` 或 `{hidden}`，走对应 generator；若集合包含两类，则并行
产生候选，由同一个 updater 竞争。论文报告 coverage、selective accuracy、abstention
rate，以及端到端 Has-Answer，而不是只报告 router accuracy。

当前代码已先完成底线修复：`use_oracle_route_hint=False`。现有 heuristic 只保留为
fallback/ablation，不能作为核心学术贡献。

### 4.3 Updater：从手工 churn penalty 改为 primal-dual replacement price

把“最大化 answerability，同时 replacement rate 不超过系统预算”写成约束问题：

```text
max_pi  sum_t U_t(K_t)
s.t.    E[R_t] <= B_rep
```

在线拉格朗日价格：

```text
lambda_{t+1} = [lambda_t + eta_t (R_t / B_write,t - b_rep)]_+
Delta_t(c,v) = U_t(c) - U_t(v) - lambda_t
admit iff Delta_t(c,v) > 0
```

这会把 `replacement_cost + pressure_mu + EMA_decay` 三个经验参数替换为一个系统可解释
的 replacement budget `b_rep`；`eta_t` 可用 `1/sqrt(t)`。Updater 回答 **how much
adaptation is affordable**，与 detector 和 router 正交。

## 5. Evidence normalization 的结论

已备份并测试的 L1 normalization 没有跨 workload 提升。旧 fixed-cost updater 下三组
实验都退化；当前 primal-dual updater 下，HotpotQA 可用 0.1 Has-Answer 损失换取
25.9% 更少 replacements，但 StreamingQA proxy 和 Mind2Web 分别损失 3.7 和 3.0 个
Has-Answer 点。它只缩小 candidate demand，没有同步归一化 resident utility；在线
dual price 能适应一部分尺度变化，但不能恢复被抹平的 query confidence。该版本保留
在 ablation，不应进入主算法。

## 6. 实施顺序

1. 先用 causal protocol 重跑 ratio curve；
2. 实现并单测 kernel CUSUM，先验证 ARL/delay，再接 cache；
3. 用 2Wiki mixed 建 selective router，必须允许 abstain；
4. 实现 primal-dual updater，在固定 Has-Answer 下比较 replacements；
5. 最后只做单组件消融，不同时改 detector/router/updater。
