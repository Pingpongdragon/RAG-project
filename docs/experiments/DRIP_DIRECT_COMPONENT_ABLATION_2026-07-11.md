# DRIP Direct + Detector 完整系统与组件消融

日期：2026-07-11

## 1. 结论先行

本轮只验证 direct evidence + detector/controller 主流程，hidden evidence 不纳入
结论。当前实现不是“所有数据集都击败 LRU”的方法，而是在 query-visible topic
drift 上提高 hot-tier answerability，并用 replacement controller 控制这种语义适配
带来的额外写入。

- **Direct evidence 必须保留。** 去掉后，三个 workload 的 Has-Answer 分别下降
  23.3、7.1 和 10.6 个百分点，是最主要的质量来源。
- **Serve ledger 必须保留。** 它在三个 workload 上同时提高 Has-Answer 并减少
  replacements，说明它确实保护了仍在服务查询的 resident，而不是多余账本。
- **Replacement penalty、dynamic pressure 和 demand decay 都有用。** 它们主要把
  Hotpot 上无约束 writer 的 3342 次替换降到 2137 次；代价是约 0.3 个
  Has-Answer 点。这个组件的贡献应表述为 Pareto trade-off，而不是纯质量增益。
- **Detector 目前是 change-point churn stabilizer，不是质量提升器。** 在三个漂移
  workload 上，开启 detector 后 Has-Answer 基本不变，分别少 11、31、1 次替换；
  stationary 对照中 0 次告警且与 no-detector 结果完全相同。
- **完整系统的主正例是 HotpotQA controlled topic drift。** DRIP 的 Has-Answer 为
  30.1%，高于最佳非 DRIP baseline 的 23.0%；相对 ARC 少 54.1% replacements。
- **StreamingQA 是明确边界条件。** TinyLFU/LRU/FIFO 依靠强 temporal locality
  仍然更好；这不是数据构造错误的充分证据，而是 recency 能直接观测该 workload
  的需求变化。

## 2. 实验协议

- hot tier 容量固定为 document pool 的约 10%。
- 所有方法共享同一 embedding 和 serving retriever，只改变 cache maintenance。
- 指标均取完整 query stream，不使用 H1/H2 作为主结果。
- Has-Answer 与 AMAT 衡量 cache residency；Recall@5 衡量固定 retriever 在 resident
  文档上的下游排序；Replacements 是一进一出的 cache replacement 次数。
- 主实验使用固定随机种子。LRU/FIFO/TinyLFU 的初始同分 resident 已改为固定种子
  随机打破平局，避免 Python string hash 改变结果。
- Workload：StreamingQA temporal、HotpotQA comparison full-gradual、Mind2Web agent
  trace；另用 HotpotQA stationary 检查 detector 假阳性。

## 3. 完整系统结果

| Dataset | Method | Has-Answer ↑ | AMAT ↓ | Recall@5 ↑ | Repl. ↓ |
|---|---|---:|---:|---:|---:|
| StreamingQA | LRU | 52.4 | 5.762 | **42.26** | 1583 |
|  | FIFO | 52.4 | 5.764 | 42.00 | 1576 |
|  | TinyLFU | **52.6** | **5.738** | 42.20 | **1575** |
|  | GPTCache | 24.1 | 8.586 | 17.84 | 3581 |
|  | Proximity | 27.3 | 8.266 | 20.56 | 2148 |
|  | ARC | 42.5 | 6.746 | 33.60 | 4240 |
|  | **DRIP** | 50.6 | 5.942 | 40.48 | 2401 |
| HotpotQA | LRU | 22.3 | 8.772 | 37.56 | 935 |
|  | FIFO | 21.4 | 8.856 | 36.66 | 965 |
|  | TinyLFU | 22.2 | 8.784 | 37.40 | **930** |
|  | GPTCache | 20.0 | 8.996 | 26.20 | 3383 |
|  | Proximity | 23.0 | 8.700 | 26.54 | 1374 |
|  | ARC | 19.3 | 9.068 | 38.28 | 4658 |
|  | **DRIP** | **30.1** | **7.992** | **42.96** | 2137 |
| Mind2Web | LRU | 39.0 | 7.100 | 42.40 | 97 |
|  | FIFO | 40.6 | 6.940 | 42.30 | 105 |
|  | TinyLFU | 40.6 | 6.940 | 42.60 | 98 |
|  | GPTCache | 34.0 | 7.600 | 34.30 | 420 |
|  | Proximity | 39.6 | 7.040 | 32.50 | **88** |
|  | ARC | **55.6** | **5.440** | 45.00 | 953 |
|  | **DRIP** | 50.2 | 5.980 | **45.30** | 363 |

### 结果解释

1. **HotpotQA：质量主正例。** DRIP 相对 LRU 提高 7.8 个 Has-Answer 点、降低
   0.780 AMAT，并提高 5.40 个 Recall@5 点。与 ARC 相比，DRIP 同时提高质量，
   replacements 从 4658 降到 2137。与 recency baseline 相比，DRIP 仍多写约
   2.3 倍，因此“低 churn”只能相对 ARC 宣称，不能相对所有 cache 宣称。
2. **StreamingQA：recency boundary。** DRIP 距最佳 Has-Answer 低 2.0 点，而且多
   826 次 replacements。这里应保留 LRU/FIFO/TinyLFU 的强结果，并将其作为方法
   适用边界，而不是继续为每个 temporal dataset 单独调参。
3. **Mind2Web：成本/质量折中。** ARC 的 Has-Answer 高 5.4 点，但 DRIP 少 590 次
   replacements（下降 61.9%）。DRIP 的 Recall@5 略高。该数据初始 hot tier 已有
   很高 support coverage，因此目前更适合作为 preservation/churn diagnostic，
   不能单独证明 drift adaptation。

## 4. Direct 组件消融

| Dataset | Variant | Has-Answer ↑ | AMAT ↓ | Recall@5 ↑ | Repl. ↓ |
|---|---|---:|---:|---:|---:|
| StreamingQA | Full | 50.6 | 5.942 | 40.48 | 2401 |
|  | NoDetector | 50.7 | 5.934 | 40.70 | 2412 |
|  | NoServe | 48.3 | 6.166 | 39.10 | 2532 |
|  | NoEvidence | 27.3 | 8.266 | 20.56 | 0 |
|  | NoReplacementPenalty | 50.3 | 5.970 | 40.38 | 2520 |
|  | NoDynamicPressure | 50.4 | 5.956 | 40.40 | 2475 |
|  | NoDemandDecay | 50.2 | 5.980 | 40.40 | 2474 |
| HotpotQA | Full | 30.1 | 7.992 | 42.96 | 2137 |
|  | NoDetector | 30.1 | 7.988 | 43.06 | 2168 |
|  | NoServe | 28.4 | 8.160 | 42.28 | 2320 |
|  | NoEvidence | 23.0 | 8.700 | 26.54 | 0 |
|  | NoReplacementPenalty | 30.4 | 7.960 | 43.86 | 3342 |
|  | NoDynamicPressure | 31.2 | 7.880 | 44.12 | 2725 |
|  | NoDemandDecay | 30.5 | 7.952 | 43.44 | 2505 |
| Mind2Web | Full | 50.2 | 5.980 | 45.30 | 363 |
|  | NoDetector | 50.2 | 5.980 | 45.30 | 364 |
|  | NoServe | 49.0 | 6.100 | 45.10 | 426 |
|  | NoEvidence | 39.6 | 7.040 | 32.50 | 0 |
|  | NoReplacementPenalty | 50.0 | 6.000 | 45.40 | 521 |
|  | NoDynamicPressure | 49.8 | 6.020 | 45.50 | 390 |
|  | NoDemandDecay | 49.6 | 6.040 | 45.20 | 402 |

### 逐组件判断与处理

#### Direct evidence：有效，保留

它决定新 demand 从哪里来。`NoEvidence` 的 0 replacements 不是效率优势，而是
策略完全失去 admission 信号、退化为近似静态 cache。当前 rank + distance credit
在三个 workload 上都有显著质量增益，无需更换。

#### Serve ledger：有效，保留

去掉 serve 后，质量下降且 replacements 上升。说明 `S_t(d)` 正确表达了 resident
的机会成本。默认 `serve_topk=3` 应保留；它不是只在一个数据集上换取分数的参数。

#### Fixed replacement penalty：有效，保留

Hotpot 上完全去掉 penalty 只增加 0.3 个 Has-Answer 点，却额外产生 1205 次
replacements；StreamingQA 和 Mind2Web 上 Full 还同时获得更高质量和更少写入。
因此固定写成本 `lambda_rep` 是必要项。

#### Dynamic replacement pressure：有效，保留但作为成本旋钮

去掉 pressure 后 Hotpot 增加 1.1 个 Has-Answer 点，但多 588 次 replacements；另外
两个 workload 上 Full 同时少写并保持或提高 Has-Answer。参数 sweep 显示
`mu=0.5` 是 Hotpot 的 Pareto knee：`mu=0.25/0.5/0.75/1.0` 对应
`(30.6,2352)/(30.1,2137)/(29.4,1985)/(28.6,1866)`。默认采用 0.5，论文中应报告
Pareto 曲线，不声称它无条件提高质量。

#### Demand decay：有效，保留

不衰减会让旧 regime demand 长期占据 resident priority。Full 在 StreamingQA 与
Mind2Web 上同时提高质量并减少写入；Hotpot 以 0.4 个 Has-Answer 点换取 368 次
更少 replacements。这仍是符合目标函数的稳定性组件。

#### Query MMD detector/controller：统计检测有效，控制收益较小

最初的 support-weighted MMD 把 cache support gap 乘进 kernel，会在 stationary
query distribution 下重复告警。原因是 cache 本身由历史查询更新，使 reference
query 与 current query 的 support gap 不再满足置换检验的 exchangeability。该版
已废弃。

当前 detector 只在 query embedding 上执行 RBF-MMD permutation test：

```text
H0: P_t(q) = P_ref(q)
drift iff permutation_p_value <= 0.05
```

support gap 只记录为 diagnostic，不进入 p-value。告警后 controller 仅把旧
demand/serve ledger 额外乘 `1 - eta`，默认 `eta=0.25`，同时重建新 regime
reference；它不再放宽 write cap、gain margin 或 replacement penalty。

| Workload | Alarm windows | Full vs. NoDetector |
|---|---|---|
| StreamingQA temporal | 5, 9, 20, 26, 30, 40, 49 | -0.1 HA，-11 repl. |
| Hotpot gradual | 8, 26, 38 | 0.0 HA，-31 repl. |
| Mind2Web | 11, 19 | 0.0 HA，-1 repl. |
| Hotpot stationary | none | 指标与逐窗口结果完全一致 |

因此 detector 可以保留为完整流程的一部分，但当前论文表述只能是“经过校准的
change-point detector，使 controller 在不损害质量时略微减少旧状态写入”，不能
把它写成主要性能来源。下一步若希望 detector 成为强贡献，应优化 change-point
后的 controller action，而不是重新混合更多启发式 drift signals。

为了区分 detector 与 controller 的问题，另做了 50 个随机种子的 synthetic
null/power audit（每次 3 个 warmup 窗口，199 次 permutation）：同分布 null 中
3/50 告警，经验 type-I error 为 6%；强 topic shift 中 50/50 检出，power 为
100%。null 与 shift 的中位 p-value 分别为 0.475 和 0.005。该结果支持
query-only MMD 的基本校准与强漂移检出能力，但只覆盖这一 synthetic 设置，不能
替代正式实验中的多强度 power curve。

## 5. 当前主公式

Direct evidence：

```text
E_t(q,d) = gamma * I[d in TopK(q)] * max(0,sim(q,d))
           / (rank_q(d) * (epsilon + 1 - sim(q,d))^alpha)
           + b_1 * I[rank_q(d)=1]
```

Demand 与 serve ledger：

```text
D_t(d) = beta_D,t * D_{t-1}(d) + sum_{q in U_t} E_t(q,d)
S_t(d) = beta_S,t * S_{t-1}(d) + A_t(d)
P_t(v) = D_t(v) + S_t(v)
```

Detector 告警时的旧状态折扣：

```text
rho_t = I[p_t <= alpha_mmd]
beta_D,t = beta_D * (1 - eta * rho_t)
beta_S,t = beta_S * (1 - eta * rho_t)
```

Replacement-aware admission：

```text
phi_t = min(4, EMA(R_{t-1}) / max(1, B_write))
C_t   = lambda_rep * (1 + mu * phi_t)
Delta_t(c,v) = D_t(c) - m * P_t(v) - C_t
admit c replacing v iff Delta_t(c,v) > 0
```

## 6. 权威结果文件

完整系统（最终 deterministic baseline 版本）：

- `experiments/direct/data/full_system_authoritative_streamingqa_50w100_kbp10.json`
- `experiments/direct/data/full_system_authoritative_hotpot_gradual_50w50_kbp10.json`
- `experiments/agent/data/full_system_authoritative_mind2web_20w25_kbp10.json`

组件消融：

- `experiments/direct/data/component_ablation_authoritative_streamingqa_50w100_kbp10.json`
- `experiments/direct/data/component_ablation_authoritative_hotpot_gradual_50w50_kbp10.json`
- `experiments/agent/data/component_ablation_authoritative_mind2web_20w25_kbp10.json`
- `experiments/direct/data/component_ablation_authoritative_hotpot_stationary_50w50_kbp10.json`

组件文件中的 DRIP 行是权威结果；其 baseline 行早于最终 deterministic tie-break
修复，因此 baseline 对比只使用上面的完整系统文件。

## 7. 尚未证明的部分

- 当前是单个固定 stream seed，不足以给出均值、方差和显著性区间。
- detector 的真实 stationary 检查仍只有一个 stream seed；50-seed synthetic audit
  也只覆盖一种噪声和一种强漂移。正式论文还应报告多漂移强度、不同窗口规模及
  连续监测下的 type-I error / power curve。
- Mind2Web 的初始化协议使初始 cache coverage 偏高，需增加更严格 cold-start 或
  head-to-tail split 后再作为主要 adaptation 证据。
- 本报告不评价 hidden evidence、GraphIndex 或 pair lease；它们既未被证明无效，
  也不能使用本轮 direct 结果作为有效证据。
