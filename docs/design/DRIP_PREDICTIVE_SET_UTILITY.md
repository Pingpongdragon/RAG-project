# DRIP: Predictive Evidence-Set Utility

本文档记录当前 `DRIP-WorkingSet` 的统一论文口径。方法不再被描述为
recency fallback、miss admission、prefetch writer 和 hidden writer 四条规则。
它们只是证据状态的不同观测或候选生成器，所有写入共用一个受约束的
predictive set-exchange objective。

## 1. 可观测证据后验

对 hot tier 无法支持的 query `q`，冷库 Top-K probe 先产生未归一化证据量：

```text
e_bar_t(q,d)
  = I[d in C_k(q)]
    [sim(q,d)]_+
    / {rank_q(d) [epsilon + 1 - sim(q,d)]^alpha}
    + b_1 I[rank_q(d)=1].
```

`DRIP-WorkingSet` 将它归一化为 miss-conditioned posterior：

```text
pi_t(d | q, miss)
  = e_bar_t(q,d) / sum_{d' in C_k(q)} e_bar_t(q,d').
```

这一步使每个 query 只分配一个单位的证据质量，避免 Top-K 大小或 embedding
相似度尺度直接改变写入强度。在线 sufficient statistics 为

```text
D_t(d) = beta_D D_{t-1}(d)
         + sum_{q in U_t} pi_t(d | q, miss),

S_t(d) = beta_S S_{t-1}(d) + A_t(d),
```

其中 `D_t` 是 **Miss-Conditioned Evidence Frequency (MEF)**，`S_t` 是
**Hit-Conditioned Service Utility (HSU)**。`A_t(d)` 只来自当前窗口实际服务过 query
的 resident。

## 2. Transition-Conditioned Evidence Forecasting

学术表述不是“提前塞文档”，而是 **transition-conditioned evidence
forecasting**：使用截至窗口 `t` 的历史，估计下一窗口的证据分布。
令 `x_t(d)` 为当前窗口从 hot hit、允许的 cold probe 或 observed item access
得到的 support-proxy 经验分布。

第一个估计器是稀疏 working-set Markov 估计：

```text
P_hat_t(j | i) = N_t(i,j) / sum_v N_t(i,v),
x_hat^M_{t+1}(j) = sum_i x_t(i) P_hat_t(j | i).
```

第二个估计器是非参数条件均值。令 `z_t` 为当前窗口 query context
的归一化表示，历史中只在 `x_{i+1}` 真正到达后才保存 `(z_i,x_{i+1})`：

```text
x_hat^K_{t+1}(d | z_t)
  = sum_{i<t} w_i(z_t) x_{i+1}(d),

w_i(z_t)
  = kappa_h(z_t,z_i) / sum_{j<t} kappa_h(z_t,z_j).
```

邻居数取 `ceil(sqrt(n))`，核带宽取因果邻居距离的中位数，不按数据集调阈值。
两个估计器依据有效样本量、语境熟悉度、target-distribution consistency
和在线命中后验得到置信度 `c^M_t,c^K_t`，然后只产生一个预测分布：

```text
x_hat_{t+1}
  = (c^M_t x_hat^M_{t+1} + c^K_t x_hat^K_{t+1})
    / (c^M_t + c^K_t).

c_t = 1 - (1 - c^M_t)(1 - c^K_t).
```

稀疏 Top-K 支撑集得到一次性 **Forecasted Evidence Utility (FEU)**：

```text
F_t(d) = |Q_t| c_t x_hat_{t+1}(d).
```

`F_t` 只影响 `K_{t+1}`，下一窗口开始时清除。因此错误预测不会像长期
demand 一样无限滞留。第一次 `A -> B` 只能学习转移；再次观察到 `A`
时才能为 `B` 产生因果预测。

## 3. 统一的证据集合效用

令 `g subseteq D` 表示一个完整 evidence unit。direct 和 forecast 文档是单元组
`g={d}`；hidden reasoning chain 是多元 hyperedge `g={a,b,...}`。单文档状态

```text
X_t(d) = S_t(d) + D_t(d) + F_t(d)
```

与 recency 先各自经验 CDF 校准，再用 Hedge 在线聚合：

```text
u_hat_t(d)
  = w_sem,t CDF_t(X_t(d))
    + w_rec,t CDF_t(last_seen_t(d)),

w_e,t proportional to exp(-eta_t L_e,t).
```

这就是“在 recency 充分时保守退化到接近 LRU”的数学含义：提高的是
recency expert 权重，而不是切换到另一套 writer。

对任意容量可行的 prospective cache `K`，预测证据集合效用为

```text
U_hat_t(K)
  = sum_{d in K} u_hat_t(d)
    + sum_{g in E_t} m_hat_t(g) I[g subseteq K].
```

第二项是 hidden evidence complementarity：只有完整 hyperedge 驻留才获得关系效用，
仅保留半条证据链不得分。代码将上式实现为 singleton node utility 与
relation hyperedge utility 的等价分解。

## 4. 唯一的缓存动作公式

令动作 `a=(G,V)`：`G` 是 direct、forecast 或 hidden generator 提出的完整
evidence group，`C=G\K_t` 是实际缺失的待写入文档，`V subseteq K_t`
且 `|V|=|C|`。所有来源只使用一个净边际收益：

```text
Delta_t(a)
  = U_hat_t((K_t \ V) union C)
    - U_hat_t(K_t)
    - lambda_t |C|.

execute a iff Delta_t(a) > 0.
```

victim `V` 由移除后的真实集合边际损失最小化得到。所以准入和准出不是两个
独立阈值；hidden group 也不需要另一个 bundle writer。near-duplicate 检查只是可行性
约束，不改变目标函数。

## 5. Replacement 长期约束

将写入当成受约束资源，而不是多个手工 penalty：

```text
lambda_{t+1}
  = [lambda_t
     + eta_t (R_t / max(1,B_write,t) - b_rep)]_+,

eta_t = 1 / sqrt(t+1).
```

`lambda_t` 是 replacement constraint 的 shadow price，也是动作公式中唯一的写成本。
`b_rep` 是部署级 replacement budget，应当以 Pareto sweep 报告，不应为每个数据集
暗中调整。

## 6. 四因素 workload protocol

受控数据不再只改 topic mixture，而是显式报告：

1. `drift_magnitude`: 跨 regime evidence-set Jaccard distance；
2. `within_regime_repeated_support_rate`: 证据写入后的未来复用潜力；
3. `causal_transition_accuracy`: 只用过去 regime transitions 预测下一状态的准确率；
4. `labeled_visibility_rate`: direct/hidden 标签的离线审计。

SQuAD/HotpotQA 用 gold evidence 文本的 sparse TF-IDF 做离线构造，detector 仍只看
在线 dense support proxy，避免同 embedding 构造并验证自己。HotpotQA 可显式声明
`min_support_frequency` 构造 reusable-evidence core；query 文本仍只使用一次。

`factorized_recurring` 与 `factorized_shuffled` 保持 regime 边际频率和 evidence reuse
一致，只改变 transition predictability。StreamingQA 保持自然 chronology，作为低复用/低可预测
边界。MIND 分为含目标标题的 Access workload 和仅使用点击前 history/context
的 Context workload，后者才是因果 next-access forecasting 设置。

## 7. 代码位置

```text
algorithms/drip/cache_manager/predictive_prefetch.py
  transition-conditioned evidence forecasting

algorithms/drip/cache_manager/predictive_utility.py
  U_hat_t(K), Delta_t(a), unified victim selection

algorithms/drip/cache_manager/working_set.py
  DRIP-WorkingSet 组合入口与 NoForecast 严格消融

experiments/common/factorized_workload.py
  representation-independent controlled workloads

experiments/common/stream_protocol.py
  drift/reuse/predictability/visibility 后验审计
```

## 8. 可复现命令

SQuAD recurring 与 matched low-predictability control：

```bash
cd /data/jyliu/RAG-project
export DRIP_REPLACEMENT_TARGET=0.25

python experiments/direct/run.py \
  --datasets squad --n-source 1000 --n-windows 20 --window-size 25 \
  --workload factorized_recurring --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.05 --warmup-windows 1 \
  --output factorized_squad_recurring.json

python experiments/direct/run.py \
  --datasets squad --n-source 1000 --n-windows 20 --window-size 25 \
  --workload factorized_shuffled --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.05 --warmup-windows 1 \
  --output factorized_squad_shuffled.json
```

HotpotQA reusable direct-evidence core：

```bash
python experiments/direct/run.py \
  --datasets hotpotqa_comparison --n-source 4000 \
  --n-windows 8 --window-size 25 --workload factorized_recurring \
  --factorized-min-support-frequency 2 --factorized-family-mode anchor \
  --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.1 --warmup-windows 1 \
  --output factorized_hotpot_reusable.json
```

MIND causal context view：

```bash
python experiments/hidden/run.py \
  --datasets mind_news_context --workload natural_temporal \
  --n-windows 50 --window-size 500 --temporal-sampling prefix \
  --drip-ablation \
  --strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet \
  --kb-pool-ratio 0.01 --warmup-windows 1 \
  --output mind_context_predictive.json
```
