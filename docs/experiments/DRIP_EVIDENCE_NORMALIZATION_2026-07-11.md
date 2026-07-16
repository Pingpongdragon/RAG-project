# DRIP Direct Evidence Normalization Experiment

日期：2026-07-11

## 备份与隔离

本实验不覆盖当前 DRIP V1：

- 正式 `DRIP` 和 `DRIP-DirectFull` 继续使用
  `algorithms/drip/cache_manager/evidence_core.py::_credit_dense`；
- 新公式只存在于消融策略 `DRIP-DirectNormalized`；
- V1 的 detector、serve ledger、decay、replacement controller 和全部参数保持不变；
- 已有权威 V1 结果文件不修改，新结果使用独立文件名。

备份复现检查：三份新配对实验中的 `DRIP-DirectFull` 与对应旧权威结果在
Has-Answer、AMAT、Replacements、逐窗口 Recall@5、逐窗口 Has-Answer 和完整
replacement cost log 上均逐项相等。`DRIP-DirectNormalized` 也有回归测试保证不会
进入正式 `STRATEGY_FACTORIES`。

V1 权威结果：

- `experiments/direct/data/full_system_authoritative_streamingqa_50w100_kbp10.json`
- `experiments/direct/data/full_system_authoritative_hotpot_gradual_50w50_kbp10.json`
- `experiments/agent/data/full_system_authoritative_mind2web_20w25_kbp10.json`

## 严格单变量公式

V1 原始 evidence：

```text
e_raw(q,d)
= gamma * sim(q,d) / [rank_q(d) * (epsilon + 1 - sim(q,d))^alpha]
  + b_1 * I[rank_q(d)=1]
```

Normalized variant 只增加 query 内归一化：

```text
E_norm(q,d) = e_raw(q,d) / sum_{d' in nonresident TopK(q)} e_raw(q,d')
```

因此每个 under-covered query 写入 demand ledger 的总 evidence mass 固定为 1，候选
之间的相对次序和 V1 完全相同。这个实验暂不删除 `gamma`、`top1_bonus` 或其他
参数，避免一次改变多个因素。

## 固定协议

- KB/pool = 0.1；
- 对照策略：`DRIP-DirectFull` 与 `DRIP-DirectNormalized`；
- StreamingQA temporal：50 windows x 100 queries；
- HotpotQA comparison full-gradual：50 windows x 50 queries；
- Mind2Web cluster-shift：20 windows x 25 queries；
- 比较完整流的 Has-Answer、AMAT、Recall@5 与 Replacements。

## 结果

| Dataset | Variant | Has-Answer ↑ | AMAT ↓ | Recall@5 ↑ | Repl. ↓ |
|---|---|---:|---:|---:|---:|
| StreamingQA | V1 Full | **50.6** | **5.942** | **40.48** | 2401 |
|  | Normalized | 49.7 | 6.032 | 39.92 | **2349** |
| HotpotQA | V1 Full | **30.1** | **7.992** | **42.96** | 2137 |
|  | Normalized | 25.3 | 8.468 | 38.76 | **1988** |
| Mind2Web | V1 Full | **50.2** | **5.980** | **45.30** | 363 |
|  | Normalized | 45.4 | 6.460 | 44.40 | **317** |

Normalized 相对 V1 的变化：

| Dataset | Delta Has-Answer | Delta AMAT | Delta Recall@5 | Delta Repl. |
|---|---:|---:|---:|---:|
| StreamingQA | -0.9 | +0.090 | -0.56 | -52 |
| HotpotQA | -4.8 | +0.476 | -4.20 | -149 |
| Mind2Web | -4.8 | +0.480 | -0.90 | -46 |

结果文件：

- `experiments/direct/data/normalized_evidence_streamingqa_50w100_kbp10.json`
- `experiments/direct/data/normalized_evidence_hotpot_gradual_50w50_kbp10.json`
- `experiments/agent/data/normalized_evidence_mind2web_20w25_kbp10.json`

## 诊断

归一化没有改变一个 query 内的候选排序，但显著改变了两个尺度：

1. 所有 under-covered query 都只贡献总质量 1，高置信、尖锐的 query 不再比模糊
   query 贡献更多 evidence；
2. demand 被压低后，serve priority 与固定 replacement penalty 没有同步归一化。

这一点可以从成功 admission 的平均 net gain 看出：

| Dataset | V1 avg. net gain | Normalized avg. net gain |
|---|---:|---:|
| StreamingQA | 1.065 | 0.311 |
| HotpotQA | 0.831 | 0.239 |
| Mind2Web | 0.720 | 0.316 |

与此同时，平均 replacement penalty 仍处于约 0.29--0.36。Normalized 因此更少
写入，但减少的是有用 admission，而不是在相同 answerability 下消除无效 churn。
HotpotQA 的质量损失最大，说明 direct topic-drift workload 尤其依赖累计 evidence
强度区分高价值候选。

## 决策

**在 fixed-cost V1 中不采用简单 query-level L1 normalization。** 三个 workload
没有一个获得质量提升，replacement 降低也没有达到“质量相近时少写”的目标。

若后续继续研究归一化，必须把 demand、serve 和 replacement price 一起放到统一
尺度上，例如使用完整的 normalized utility + primal-dual replacement controller；
不能只归一化公式左侧的 candidate evidence 后沿用 V1 阈值。该方案应作为独立 V2，
不能通过调低 V1 replacement cost 来把本次负结果包装成正结果。

## 当前 primal-dual updater 下的复核

正式 DRIP 后续改为 causal initialization 和 primal-dual replacement price。为避免
把旧 fixed-cost 结论错误外推到新 updater，又做了严格配对复核：两者都关闭 detector，
唯一差别仍是每个 query 的 evidence 是否 L1 归一化。

| Workload | Variant | Has-Answer ↑ | AMAT ↓ | Recall@5 ↑ | Repl. ↓ |
|---|---|---:|---:|---:|---:|
| HotpotQA causal | PrimalDual | **6.9** | **10.312** | **19.98** | 1722 |
| | + Normalized | 6.8 | 10.324 | 18.60 | **1276** |
| StreamingQA year proxy | PrimalDual | **46.3** | **6.366** | **37.50** | 2634 |
| | + Normalized | 42.6 | 6.742 | 34.64 | **2273** |
| Mind2Web diagnostic | PrimalDual | **46.4** | **6.360** | **42.80** | **258** |
| | + Normalized | 43.4 | 6.660 | 39.00 | 271 |

HotpotQA 上 normalization 得到一个有吸引力的成本点：Has-Answer 仅下降 0.1，
replacements 下降 25.9%。但该现象没有跨 workload 泛化；StreamingQA proxy 和
Mind2Web 都出现 3 个点以上的 Has-Answer 损失，Mind2Web 甚至写得更多。因此正式
方法仍不启用 normalization，`DRIP-PrimalDualNormalized` 只保留为消融。

对应结果：

- `experiments/direct/data/normalized_primaldual_hotpot_causal_50w50_kbp10.json`
- `experiments/direct/data/normalized_primaldual_streamingqa_proxy_causal_50w100_kbp10.json`
- `experiments/agent/data/normalized_primaldual_mind2web_causal_20w25_kbp10.json`

注意：StreamingQA 这一行是缺少官方 timestamp 的 year proxy，只能作为归一化行为
回归，不能作为论文中的 natural temporal 结果。
