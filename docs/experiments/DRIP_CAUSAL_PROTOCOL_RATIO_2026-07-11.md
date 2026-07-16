# DRIP causal protocol 与 cache-ratio 审计（2026-07-11）

## 1. 实验口径

- Dataset: HotpotQA comparison, 4,000 source queries, pool 30,454 documents;
- Stream: 50 windows x 50 queries, full gradual topic shift;
- Initialization: `causal-prefix`, 3 个评估前历史窗口，只用 query/corpus embedding;
- Sampling: shuffled cycle，2,500 个 stream query 全部唯一，duplicate rate = 0;
- Ratios: 1%, 2%, 5%, 10%, 20%;
- 当前是 seed 42 的 protocol audit，不是最终多 seed 主表。

旧 ratio 结果使用 future-context head initialization 和部分有放回采样，已降级为
legacy diagnostic。新结果不能与旧数字直接横向比较。

## 2. Causal ratio 结果

| Ratio | Method | HasAns | AMAT | Recall@5 | Replacements |
|---:|---|---:|---:|---:|---:|
| 1% | LRU | 0.2 | 10.98 | 3.1 | 2257 |
| | DRIP Direct | 0.2 | 10.98 | 3.2 | 1957 |
| | DRIP MentionRouted | 0.2 | 10.98 | 3.2 | 2021 |
| | DRIP PrimalDual (target=0.5) | 0.2 | 10.98 | 3.2 | 2251 |
| 2% | LRU | **0.9** | **10.91** | 6.2 | 2087 |
| | DRIP Direct | 0.8 | 10.92 | 5.9 | 2339 |
| | DRIP MentionRouted | 0.7 | 10.93 | **6.2** | 2586 |
| | DRIP PrimalDual (target=0.5) | 0.8 | 10.92 | 5.9 | 2358 |
| 5% | LRU | 3.3 | 10.67 | 12.8 | **1801** |
| | DRIP Direct | **3.4** | **10.66** | 13.5 | 2854 |
| | DRIP MentionRouted | **3.4** | **10.66** | **13.8** | 3653 |
| | DRIP PrimalDual (target=0.5) | 3.2 | 10.68 | 13.2 | 2373 |
| 10% | LRU | 6.8 | 10.32 | 19.6 | **1577** |
| | DRIP Direct | 7.5 | 10.25 | 22.0 | 2948 |
| | DRIP MentionRouted | **7.8** | **10.22** | **22.2** | 3803 |
| | DRIP PrimalDual (target=0.5) | 7.4 | 10.26 | 21.1 | 2284 |
| 20% | LRU | 13.7 | 9.63 | 28.9 | **1200** |
| | DRIP Direct | 14.9 | 9.51 | 31.5 | 2520 |
| | DRIP MentionRouted | **15.3** | **9.47** | **31.9** | 3243 |
| | DRIP PrimalDual (target=0.5) | 14.8 | 9.52 | 31.1 | 2125 |

图：`docs/experiments/figures/hotpot_causal_cache_ratio.{pdf,png}`。

## 3. Primal-dual Pareto operating point

`replacement_target=0.25` 不是按数据集调的 admission threshold，而是系统允许使用的
write-budget 比例：

| Ratio | HasAns | AMAT | Replacements | vs. Direct replacements |
|---:|---:|---:|---:|---:|
| 1% | 0.2 | 10.98 | 1326 | -32.2% |
| 10% | 6.9 | 10.31 | 1721 | -41.6% |
| 20% | 14.2 | 9.58 | 1466 | -41.8% |

在 10% ratio，target=0.5 是质量优先点：HasAns 7.4 vs. Direct 7.5，写入少
22.5%。target=0.25 是成本优先点：HasAns 6.9，写入少 41.6%，且仍略高于 LRU
的 6.8。论文应报告 Pareto frontier，而不是隐藏 operating point。

## 4. 最关键的数据审计

2,500 个 query 包含 5,000 次 support occurrence，却有 4,138 个 unique support：

- mean occurrences per support = 1.208;
- 只有 17.96% 的 support 文档重复出现；
- 当前 query 的完整 support 在过去全部出现过的比例仅 4.44%；
- adjacent-window support overlap 仅 0.90%。

这说明 Hotpot comparison 在去掉 future-context initialization 和 exact-query repetition
后，并不是强 cache-reuse workload。它仍可作为“semantic prediction 能否在 controlled
topic shift 下稍微优于 recency”的机制测试，但不适合作为系统主结果。低 Has-Answer
不是 detector/updater 单点失败，而是数据本身几乎每次要求新文档。

主系统 workload 应换成有真实重复访问和时间戳的 MIND/TripClick；Hotpot/2Wiki 只保留
为 direct/hidden evidence controllability test。

## 5. Detector 诊断

10% gradual run 中，三个 DRIP variant 都只在 window 25 告警一次；目标 ARL=100，
bootstrap calibrated ARL=100.445。说明 detector 输出不依赖 updater variant，也没有
旧 repeated-MMD test 的逐窗多重检验问题。

补充真实流检查：

- stationary 50-window stream：0 次告警，calibrated ARL=100.914；
- sudden drift：在真实 change point window 25 告警，delay=0 window，且只告警一次；
- full gradual drift：在 midpoint window 25 告警，且只告警一次。

这只是单 seed 的行为回归。论文还需在 synthetic controlled shift 上画 ARL/delay vs.
shift magnitude，而不能仅用三条 workload 断言统计最优性。

## 6. 正式 DRIP 入口验证

正式 `DRIP` / `DRIPNOdetector` 已切换到 `replacement_budget_rate=0.25` 的
primal-dual updater：

| Drift | Method | HasAns | AMAT | Replacements |
|---|---|---:|---:|---:|
| full gradual | DRIP w/o detector | 6.9 | 10.31 | 1722 |
| | DRIP | 6.9 | 10.31 | 1721 |
| sudden | DRIP w/o detector | 7.6 | 10.24 | 1722 |
| | DRIP | 7.6 | 10.24 | 1722 |

因此当前证据支持“detector 的 sequential calibration 合理”，但不支持“detector 已经
提升 cache utility”。在自然 temporal/multi-user workload 出结果前，detector 贡献应
写成 statistically calibrated component，而不是性能主胜点。
