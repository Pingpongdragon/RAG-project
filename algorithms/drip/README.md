# DRIP active implementation

当前 active path 只有一条研究主线：

```text
completed evidence
  -> metadata / semantic topic partition
  -> explicit topic-drift signal
  -> soft next-topic forecast
  -> causal document candidates
  -> replaceable placement adapter
```

LRU 是 `algorithms/cache/recency/lru.py` 中的独立 baseline，不再作为 DRIP
expert、特征或 fallback。旧 hard prototype states、global/state Hedge 和
LRU/Reactive Fixed-Share 已从 active code 删除。

## 文件职责

| 文件 | 职责 |
|---|---|
| `topic_partition.py` | 建立稳定、带 fingerprint 的 metadata/semantic 冷库 topic 目录，并把已完成 evidence 编码成 soft histogram |
| `topic_dynamics.py` | 用 JS--CUSUM 显式检测分布变化；学习 soft topic transition，并用历史 mixture episodes 选候选 |
| `policy.py` | 收集 HSU/MEF feedback，调用 TopicDynamics，并把候选交给当前 placement adapter |
| `controller.py` | 仅保留 replacement shadow-price controller；不再含 LRU expert |
| `config.py` | 唯一参数表和 metadata/semantic factory |
| `index.py` | miss 后的 bounded cold-corpus Top-K probe |

## 分区

Metadata 分区复用冷库已有字段，Topic ID 使用 type-sensitive canonical order：

```python
config = DRIPConfig.topic_dynamics(
    candidate_budget=32,
    metadata_field="subcategory",
)
```

Semantic 分区先对文档 embedding 做一次离线 KMeans，再以 cosine affinity
产生 hard topic 和 sparse soft memberships：

```python
config = DRIPConfig.topic_dynamics(
    candidate_budget=32,
    semantic_topics=32,
    topic_partition_metric="cosine",
    topic_soft_memberships=2,
)
```

两种实现共享 `TopicPartition` API：`memberships()`、`hard_bucket()`、
`soft_bucket()`、`topic_histogram()`、`summary()`。正式结果应记录
`corpus_fingerprint` 和 `partition_fingerprint`，避免冷库重排后复用错误状态。

## 当前如何检测与预测

窗口 `t` 服务完成后，实际 evidence access 形成 `h_t`。检测器先用更新前的
EWMA reference 计算 Jensen--Shannon divergence，再累积 one-sided CUSUM；超过
阈值才输出 `drift_alarm=True`。

预测器不再创建 hard latent state。它直接累计 soft transition
`h_(t-1) outer h_t`，由当前 distribution 预测下一窗口 distribution。具体文档
通过预测 mixture 检索最相近的历史窗口，再汇总这些窗口真实使用过的 evidence；
因此不会生成未来或从未出现过的文档 ID，也不会把各 topic 独立频率错误相加。

每个 `topic_log` 项现在直接记录：

- `topic_drift_score` / `topic_drift_cusum` / `topic_drift_alarm`
- `topic_current` / `topic_predicted`
- `topic_predicted_distribution`
- `topic_transition_support` / `topic_forecast_confidence`
- 实际 materialized proposal 和 promotion 数

## Placement 边界

`policy.py::_write()` 目前只是可运行的临时 adapter：resident utility 为
`HSU + MEF + gated topic demand`，写入受共同容量、每窗口写上限、speculative
fraction 和 shadow price 约束。它不是已经定稿的论文 placement，也不应被描述成
全局最优 switching-cost solver。

`topic_apply_forecast_to_cache=False` 是默认值：系统仍检测、预测并记录候选，但不
自动写入。只有专门研究旧 adapter 时才显式设为 `True`。

因此旧 TopicState 实验数字不能沿用到这个版本。下一轮实验应分别验证：

1. metadata 与 semantic partition 是否产生稳定、可预测的 `h_t`；
2. detector 的 delay / false alarm；
3. next-topic accuracy 与 calibration；
4. 再比较候选如何进入 hot tier。

## 因果顺序

评估仍遵守：先用冻结的 `K_t` 对整个窗口计分，服务完成后才把 evidence feedback
交给 `step()`，更新只能生成 `K_(t+1)`。算法不读取未来 query、stream constructor
的 latent regime label，或当前请求计分前的 gold support。
