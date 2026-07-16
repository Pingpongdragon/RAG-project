# DRIP multi-dataset cache-ratio and CUSUM audit (2026-07-12)

## 1. 结论先行

本轮在统一因果协议下比较了 LRU、FIFO、TinyLFU、GPTCache-style、Proximity、
AgentRAGCache (ARC)、DRIPNOdetector 和 DRIP，并扫描 `KB/pool` 为
`1%/2%/5%/10%`。当前证据支持以下结论：

1. **LRU 强不是 runner 错误。** 在 MIND 自然新闻访问流和 StreamingQA year-proxy
   中，近期访问就是未来需求的强信号，LRU/FIFO 本来就是非常强的边界基线。
2. **当前 DRIP 的稳定优势是 write-quality trade-off，而不是全面超过 LRU。**
   MIND 1% 上，DRIP 的 Has-Answer 为 `90.3%`，LRU 为 `91.4%`，但 replacements
   从 `7,720` 降到 `2,550`，减少 `67.0%`。
3. **CUSUM 能检测 working-set change，但不是主要性能来源。** 相对完全相同的
   no-detector 版本，它在 MIND 1% 上仅 `+0.1` Has-Answer point、少 43 次替换；
   2% 和 5% 上反而分别下降 `0.5` 和 `0.3` point。其收益不具备跨容量一致性。
4. **HotpotQA/2Wiki 的因果流不适合当 cache 主结果。** 去掉 future-aware 初始化后，
   当前请求能够仅靠历史 support 回答的比例分别只有 `4.48%` 和 `0.24%`。这两个
   数据集只能验证 direct-evidence candidate generation，不能证明 cache policy 的
   长期复用收益。
5. **当前主表并未真正使用 hidden query router。** Router 代码存在，但 direct
   主策略 `EmbeddingOnlyDRIPCore` 会把 hidden/BOTH 决策强制改为 visible。Mind2Web
   上 ARC 明显优于 DRIP，也反映了 direct-only 方法与 hidden/structured reuse 的失配。

## 2. 统一协议

| Workload | Stream | Corpus | 构造和用途 | 重要限制 |
|---|---:|---:|---|---|
| MIND news access | 100,000 events | 51,282 | 原始时间顺序的自然 item-access trace | 单 support，不验证 hidden reasoning |
| StreamingQA proxy | 5,000 queries | 29,819 | 按可用 year proxy 排序 | mirror 缺官方 question/evidence timestamp |
| HotpotQA comparison | 2,500 queries | 30,454 | full-gradual direct topic shift | 几乎没有 causal support reuse |
| 2Wiki comparison | 2,500 queries | 37,214 | full-gradual direct topic shift | 几乎没有 causal support reuse |
| Mind2Web agent | 2,500 actions | 4,752 | 100 个本地 task 的 controlled reuse | 75.6% exact-query repetition，不是自然 trace |

所有结果都遵循：

- 当前窗口先用更新前的 cache 计分，再执行维护；
- 初始化只使用评估流之前的 causal prefix；
- 不使用未来 gold support 初始化 cache；
- 同一 workload 的所有方法共享 stream、embedding、cache budget 和 retriever；
- MIND 的 observed access 只给真实访问 key 记一次 demand，不扩成语义 Top-K；
- Oracle 不进入主比较。

## 3. 数据是否真的可缓存

| Workload | Repeated support | Past-answerable query | Adjacent-window support Jaccard |
|---|---:|---:|---:|
| MIND news access | 95.971% | 95.971% | 31.447% |
| Mind2Web controlled | 86.749% | 82.880% | 8.331% |
| StreamingQA proxy | 63.800% | 63.800% | 11.803% |
| HotpotQA comparison | 17.240% | 4.480% | 0.456% |
| 2Wiki comparison | 0.520% | 0.240% | 0.041% |

这个表比单看 drift 曲线更关键。若未来请求几乎从不复用过去 support，任何在线 cache
都无法把当前 miss 转化为未来 hit；此时低 Has-Answer 是 workload 的不可缓存性，不能
归因于 detector 或 eviction policy。

## 4. MIND 自然访问流

### 4.1 1% cache ratio

| Method | Has-Answer | AMAT | Recall@5 | Replacements |
|---|---:|---:|---:|---:|
| LRU | **91.4** | **1.860** | **91.397** | 7,720 |
| FIFO | 88.4 | 2.162 | 88.379 | 10,811 |
| TinyLFU | 57.6 | 5.237 | 57.626 | 41,562 |
| ARC | 79.2 | 3.085 | 79.154 | **1,244** |
| DRIPNOdetector | 90.2 | 1.981 | 90.186 | 2,593 |
| DRIP | 90.3 | 1.967 | 90.333 | 2,550 |

DRIP 相对 LRU 少 `5,170` 次 replacement (`-67.0%`)，代价是 `-1.1` Has-Answer
points。这是清晰的成本优先 Pareto operating point，而不是 quality winner。

### 4.2 Capacity sensitivity

| Ratio | LRU HasAns / Repl. | ARC HasAns / Repl. | No-det HasAns / Repl. | DRIP HasAns / Repl. |
|---:|---:|---:|---:|---:|
| 1% | 91.4 / 7,720 | 79.2 / 1,244 | 90.2 / 2,593 | 90.3 / 2,550 |
| 2% | 94.7 / 4,541 | 89.5 / 2,116 | 92.0 / 2,170 | 91.5 / 2,254 |
| 5% | 95.7 / 3,661 | 95.0 / 3,339 | 92.6 / 2,002 | 92.3 / 2,021 |
| 10% | 96.0 / 3,362 | 96.1 / 3,337 | 93.5 / 1,771 | 93.3 / 1,771 |

随着 cache 变大，LRU/ARC 已经覆盖绝大多数热点，DRIP 的保守 admission 会保留较多
过时 resident，quality gap 反而扩大。这个趋势说明当前 fixed `b_rep=0.25` 不是所有
cache ratio 下都处于同一个 Pareto 位置；论文不能只报 1%。

## 5. 其他 workload 的主要发现

### StreamingQA year-proxy

- 1%：LRU `31.5/3,165`，DRIP `29.4/2,045`，no-detector `28.2/1,773`；
- 2%：LRU `41.5/2,398`，DRIP `41.4/2,529`；
- 5%：DRIP `44.1/2,742`，LRU `43.8/2,155`；
- 10%：LRU `48.2/1,872`，DRIP `46.3/2,626`。

每个二元组为 `Has-Answer / Replacements`。Detector 在紧缓存下提高 quality，但同时
增加 writes；在 5%/10% 下几乎没有作用。该 workload 只能作为 temporal proxy，不能
写成官方 StreamingQA timestamp evaluation。

### HotpotQA and 2Wiki direct comparison

- HotpotQA 10%：LRU `6.8/1,577`，ARC `7.2/5,303`，DRIP `6.9/1,721`；
- 2Wiki 10%：LRU `1.8/2,004`，ARC `0.9/7,654`，DRIP `1.7/1,580`；
- DRIP 与 no-detector 的 Has-Answer 在所有 ratio 上相同；
- ARC 在两者上 replacements 很高，尤其 2Wiki 10% 为 7,654。

这些结果能说明 primal-dual writer 抑制 ARC-style over-writing，却不能证明长期 cache
utility，因为 causal support reuse 太低。

### Mind2Web controlled diagnostic

10% 时：ARC `58.4/623`，DRIP `52.1/1,065`，LRU `37.8/92`，Proximity
`45.0/49`。ARC 在 quality 和 writes 上都优于 DRIP。该结果不能被隐藏：当前 direct
主方法没有启用 hidden support completion，而且 stream 是有限 task 的 controlled
reuse。它适合暴露方法边界，不适合声称真实 agent-trace SOTA。

完整表格和曲线位于：

```text
docs/experiments/tables/cache_ratio_*.md
docs/experiments/figures/cache_ratio_*.{png,pdf}
```

## 6. 为什么选 CUSUM

当前 detector 不是对每个窗口独立做一次阈值判断，而是顺序累计小偏移：

```text
z_t = (MMD^2(P_ref^x, P_t^x) - mu_0) / sigma_0
G_t = max(0, G_{t-1} + z_t - kappa)
alarm iff G_t >= h(ARL_0)
```

其中 `P_t^x` 是 observable support proxy working set。选择 kernel CUSUM 有三个合理
原因：

1. CUSUM 累计持续的小偏移，比单窗口 MMD threshold 更适合 gradual drift；
2. RBF-kernel MMD 可比较高维、多模态 embedding distribution，不只检测均值变化；
3. bootstrap 用 `target_arl=100` 校准 `h`，false alarm 约束具有明确统计含义。

统计 change 与 cache actionability 又被分开：

```text
chi_t = min(1, G_t / h)
V_hat_t = sum_i [p_hat_t(c_i) - p_hat_t(v_i)]_+
a_t = V_hat_t / (V_hat_t + V_0 + epsilon)
rho_t = chi_t * a_t
```

`V_hat_t` 是在当前窗口局部平稳假设下，最优 candidate-victim swaps 预计减少的 miss
mass。只有“分布确实变了”且“存在可行动替换”时，`rho_t` 才大。

## 7. 为什么 CUSUM 对 cache 管理作用很小

### 7.1 Detector 没有直接控制 admission

当前 `rho_t` 只进入 selective ledger forgetting：

```text
beta_X,t(d) = beta_X [1 - rho_t (1 - r_t(d))],  X in {D,S}
```

`r_t(d)` 是 MMD witness 给出的 current-regime affinity。Detector **不会**改变：

- cold candidate generator；
- write cap；
- gain margin；
- primal-dual replacement price；
- near-duplicate rule。

所以它只能让旧 resident 较快失去保护，不能保证正确 candidate 出现或被选中。

### 7.2 主 updater 已经响应 miss

无 detector 的 DRIP 每个窗口已经根据 miss 更新 demand，并通过 primal-dual price 控制
写入。对于 direct abrupt shift，miss 本身就是强信号，CUSUM 再判断一次 topic shift
常常只是重复信息。

### 7.3 Signal 很弱且短暂

- HotpotQA 50 个窗口只有 6 个窗口 `rho_t>0`，平均 `rho_t=0.031`；
- 2Wiki 只有一次 hard alarm，出现在 window 40；
- StreamingQA hard alarm 出现在 window 41；
- hard alarm 后 reference 立即 rebase，并重新 warm up 3 个窗口。

例如 HotpotQA 中完全属于旧 regime 的文档，其 demand retention 只从 `0.92` 暂时降到
约 `0.891`。这很难改变 victim 排序。`ARL_0=100` 对 50-window 实验也相当保守。

### 7.4 数据决定了可获得的收益上限

HotpotQA/2Wiki 几乎没有 future support reuse，检测再准也没有未来 hit 可以回收；MIND
具有极强 temporal locality，LRU 已接近最优。Detector 只可能在两者之间一个较窄的
区域产生边际收益。

### 7.5 当前 actionability estimator 过于短视

MIND 的相同 100k trace 上，detector 相对 no-detector 的精确差值为：

| Ratio | Hit delta | Has-Answer delta | Replacement delta |
|---:|---:|---:|---:|
| 1% | +147 | +0.147 pp | -43 |
| 2% | -480 | -0.480 pp | +84 |
| 5% | -274 | -0.274 pp | +19 |
| 10% | -192 | -0.192 pp | 0 |

三个 ratio 都有 32 次 hard alarm、72 个非零 controller window，平均 `rho_t` 约
`0.126`。这说明变化主要来自 controller 与 cache state 的交互，而不是 detector
是否看到不同的外生流。当前

```text
V_hat_t = sum_i [p_hat_t(c_i) - p_hat_t(v_i)]_+
```

只用当前窗口频率并假设下一窗口局部平稳。1% cache 中，释放短期失效 resident 往往
有用；cache 变大后，暂时不活跃的 resident 可能是会周期性回归的 long-tail item，
witness forgetting 反而破坏长期复用。因而当前 `a_t` 不是 capacity- and
reuse-horizon-aware 的可靠 replacement value。

### 7.6 QA support proxy 仍有偏差

MIND exact access 使用真实在线 access key；普通 QA 在 miss 时使用 cold-probe top-1，
hit 时使用当前 resident top-1。这个 proxy 是 retriever- and policy-conditioned，可能
测到“当前策略相信的 support”而不是真正 hidden support。Router/candidate generation
错误不能由 CUSUM 修复。

## 8. Query router 的实际状态

`evidence_router.py` 实现了三路决策：

```text
QUERY_VISIBLE / QUERY_HIDDEN / QUERY_BOTH (abstain)
```

它使用 query cue、cache similarity、first-hop strength 和 entity diversity；默认不读
`qtype/route_hint`。但当前主方法继承 `EmbeddingOnlyDRIPCore`，设置：

```text
force_query_visible = True
```

因此 hidden/BOTH 会被覆盖成 visible。换言之，**router 存在，但当前主曲线没有获得
hidden routing 或 graph completion 的收益**。Hidden 分支应继续作为独立方法方向，
但在完成 causal hidden dataset 和端到端结果前，不应混入当前 direct 主表的贡献声明。

## 9. 方法决策

1. 把 **primal-dual replacement constraint** 保留为当前主贡献；
2. 把 **support KCUSUM** 写成 optional controller，并报告严格 no-detector 消融；
3. 不声称 detector 普遍提高 cache utility，目前跨容量结果不支持；
4. MIND 自然流作为 cache-system 主结果，HotpotQA/2Wiki 作为 evidence diagnostic；
5. 下一版应把 detector 直接接到预测需求，而不是只接 ledger decay：

```text
p_hat_t(d) <- adaptive working-set probability
Delta_t(c,v) = p_hat_t(c) - p_hat_t(v) - lambda_t
```

CUSUM/run-length posterior 可控制 `p_hat_t` 的学习率或状态重置，MMD witness 则直接校准
candidate/victim value。这样 detector、prediction 和 admission 才优化同一个 future
miss objective，而不是三个相邻但松耦合的模块。

若仍希望保留显式 sequential detector，更直接的研究版本应累计“可避免的 cache
损失”，而不是 generic embedding discrepancy：

```text
g_t = [V_hat_t - lambda_t R_hat_t]_+
G_t^cache = max(0, G_{t-1}^cache + g_t - kappa)
reconfigure iff G_t^cache > C_rebuild
```

这里阈值 `C_rebuild` 具有真实系统成本含义：只有持续积累的预计 miss reduction 已经
足以支付重配置成本时才触发。MMD/CUSUM 仍可用于定位 change-point 和计算 witness，
但不再单独决定 cache action。该式是后续设计建议，尚未在当前结果中实现。
