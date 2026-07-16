# DRIP TopicDynamics：公式与代码映射

正文不需要再把每个计数器写成独立贡献。active topic layer 只有两步。

## 1. 显式 evidence-topic 漂移

已完成窗口的 evidence `Y_t` 通过冷库 partition memberships `m(d)` 编码为：

```text
h_t = normalize(sum_{d in Y_t} m(d)).
```

检测器使用更新前的 EWMA reference `mu_(t-1)`：

```text
delta_t = JS(h_t || mu_(t-1))
G_t = max(0, G_(t-1) + delta_t - kappa)
alarm_t = I[G_t >= tau]
```

代码：`topic_partition.py::topic_histogram` 和
`topic_dynamics.py::ExplicitTopicDriftDetector`。

## 2. Soft topic transition

不再创建 hard prototype state，也不使用 expert mixture：

```text
T_t = beta_T T_(t-1) + h_(t-1) h_t^T
h_hat_(t+1) = normalize(h_t^T row_normalize(T_t)).
```

具体文档使用 mixture-conditioned episodic lookup。设历史窗口 `i` 的 signature 与
真实 evidence 分别为 `h_i,Y_i`，候选分数为：

```text
p_hat_(t+1)(d)
  = sum_{i<t} decay_i exp(-JS(h_hat_(t+1),h_i)/temperature)
      count_{Y_i}(d) / |Y_i|.
```

因此候选只来自因果历史。代码：`topic_dynamics.py::SoftTopicDynamics`。

## 3. 当前临时 placement adapter

LRU 不进入 DRIP。当前 resident/candidate utility 只是：

```text
u_t(d) = normalize(S_t(d) + D_t(d) + D_topic,t(d)),
```

其中 `S_t` 是 hit-conditioned service credit，`D_t` 是 miss evidence，
`D_topic,t` 只在显式 alarm 且预测置信度达到门槛后获得质量。Reactive swap 暂时使用：

```text
replace v with c iff u_t(c) - u_t(v) - lambda_t > 0.
```

代码：`policy.py::_node_utility`、`_topic_dynamics_prefetch`、`_write`。这部分是
待继续设计的 placement 接口，不是 TopicDynamics 的理论核心。

## 4. 参数表

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `topic_metadata_field` | empty | 使用一个稳定 metadata 字段分区 |
| `topic_semantic_topics` | 0 | semantic topic 数；与 metadata 二选一 |
| `topic_partition_metric` | cosine | semantic assignment metric |
| `topic_soft_memberships` | 2 | 每篇文档保留的 soft topics |
| `topic_partition_temperature` | 0.10 | membership softmax temperature |
| `topic_drift_reference_rate` | 0.05 | detector reference EWMA rate |
| `topic_drift_slack` | 0.01 | CUSUM 正常波动扣除量 |
| `topic_drift_threshold` | 0.10 | drift alarm threshold |
| `topic_transition_decay` | 0.95 | soft transition 历史保留率 |
| `topic_document_decay` | 0.50 | historical topic-episode 保留率 |
| `topic_min_transition_support` | 1.0 | 发布预测所需的有效转移质量 |
| `topic_min_forecast_confidence` | 0.50 | 发布预测的最低置信度 |
| `topic_candidate_budget` | 0 | 候选上限；0 关闭 TopicDynamics |
| `topic_apply_forecast_to_cache` | false | 是否启用尚未定稿的 provisional placement |

正式实验必须记录 partition fingerprint、detector 参数和 causal feedback protocol。
