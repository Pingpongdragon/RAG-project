# DRIPNOdetector Objective and Formula Notes

本文件先固定论文口径，再改代码。当前主实验先看
`DRIPNOdetector`：不依赖 drift detector，只使用 query-visible/direct
evidence、serve 账本和 replacement 控制。

## 1. Max 目标

热缓存为 `K_t`，容量预算为 `B`，当前窗口查询为 `Q_t`。主目标不是只最大化
recall，而是在相近回答能力下减少替换：

```text
max_{K_t: |K_t| <= B} J_t(K_t)

J_t(K_t)
= Answerability_t(K_t)
  - lambda_amat * AMAT_t(K_t)
  - lambda_rep  * ReplacementCost_t(K_t, K_{t-1})
```

其中：

```text
Answerability_t(K_t)
= (1 / |Q_t|) * sum_{q in Q_t} I[required_support(q) subseteq K_t]

AMAT_t(K_t)
= hit_cost + miss_rate_t(K_t) * miss_penalty

ReplacementCost_t(K_t, K_{t-1})
= |K_t \ K_{t-1}| / max(1, |K_t|)
```

实验里对应四个核心指标：

```text
Recall@5          检索质量
Has-Answer Rate   required support 是否已经在 hot cache 中
AMAT              命中/未命中的平均访问代价
Replacements      cache replacement 次数，越少越好
```

## 2. 当前代码实际公式

当前 `DRIPNOdetector` 继承 `EmbeddingOnlyDRIPCore`，实际走的是
`evidence_core.py` 里的 direct 分支：

```text
force_query_visible = True
use_bridge          = False
use_pair_lease      = False
use_drift_detector  = False
```

### 2.1 Serve 账本

如果 query 被当前 KB 命中，就给 top resident 记服务价值：

```text
S_t(d)
= serve_decay * S_{t-1}(d) + recent_hit_credit_t(d)
```

代码位置：

```text
DRIPCore._credit_serve(...)
```

### 2.2 Demand 账本

对 under-covered query，从全 pool 取 dense top-k candidate。当前 direct
evidence 很简单：

```text
E_current(q, d)
= direct_gamma * max(0, sim(q, d))
   + direct_top1_bonus * I[rank(d) = 1]
```

需求账本为：

```text
D_t(d)
= demand_decay * D_{t-1}(d) + sum_{q in under-covered Q_t} E_current(q, d)
```

代码位置：

```text
EvidenceConditionedDRIPCore._credit_dense(...)
```

### 2.3 Resident priority

当前 writer 保护 resident 的公式：

```text
P_t(v)
= S_t(v) + D_direct,t(v)
```

hidden 版本还会加 `D_bridge` 和 `pair_lease`；但 `DRIPNOdetector` 中它们为 0：

```text
P_t(v)
= S_t(v)
 + D_direct,t(v)
 + D_bridge,t(v)
 + pair_lease_weight * PairLease_t(v)
```

代码位置：

```text
EvidenceConditionedDRIPCore._priority_for_route(...)
```

### 2.4 当前 admission

当前 admission 没有显式 replacement cost，只靠 margin 抑制抖动：

```text
admit c replacing v iff
D_t(c) > gain_margin * P_t(v)
```

也就是：

```text
Delta_current(c, v)
= D_t(c) - gain_margin * P_t(v)

admit iff Delta_current(c, v) > 0
```

代码位置：

```text
EvidenceConditionedDRIPCore._write(...)
```

## 3. 下一步要实现的公式

我们保留当前有效的 demand/serve 主线，但把 direct evidence 和 replacement
控制写得更像 cache 论文里的效用最大化。

### 3.1 ARC-style direct evidence

把 direct evidence 从单纯 similarity 改成 rank + distance 加权：

```text
E_t(q, d)
= sigma_t(q)
  * I[d in TopK(q)] * max(0, sim(q, d))
  / (rank_q(d) * (epsilon + 1 - sim(q, d))^alpha)
```

其中：

```text
sigma_t(q) = uncovered_support_slots(q)
```

当前实现里 `sigma_t(q)` 先取 1，因为 `DRIPCore.step` 已经只对
under-covered query 调用 `_credit_dense`。后续如果要更严格，可以把缺失
support slot 数传进来。

### 3.2 Demand / serve update

```text
D_t(d) = beta_d * D_{t-1}(d) + sum_q E_t(q, d)
S_t(d) = beta_s * S_{t-1}(d) + A_t(d)
```

`A_t(d)` 是 resident 在当前窗口服务 query 的 credit。

### 3.3 Replacement-aware priority

```text
P_t(v) = lambda_s * S_t(v) + lambda_d * D_t(v)
```

目前先保持 `lambda_s = lambda_d = 1`，避免引入过多参数。

### 3.4 合并后的替换惩罚项

用户提出把 fixed replacement cost 和 churn pressure 合成一项。采用：

```text
C_t(c, v) = lambda_replace * (1 + mu * phi_t)
```

其中：

```text
phi_t = EMA(replacements_per_window) / replacement_budget
```

`phi_t` 越大，说明最近换入换出压力越高，下一轮 admission 会更保守。

### 3.5 新 admission

```text
Delta_t(c, v)
= D_t(c)
  - gain_margin * P_t(v)
  - C_t(c, v)

admit c replacing v iff Delta_t(c, v) > 0
```

这里 `C_t` 是唯一显式替换惩罚项；不再维护单独的
`lambda_replace + lambda_churn * ChurnPressure` 两项，论文和代码口径更干净。

## 4. 代码映射

```text
algorithms/drip/cache_manager/policies.py
  DRIPNOdetector: 当前主实验方法；关闭 drift detector。

algorithms/drip/cache_manager/evidence_core.py
  _credit_dense: direct evidence E_t(q,d)
  _priority_for_route: resident priority P_t(v)
  _write: admission Delta_t(c,v)

motivation/motivation_1/run.py
motivation/motivation_2/run.py
  输出 Recall@5 / Has-Answer / AMAT / Replacements。
```

这份文档记录的是“改代码前”的状态和“本轮要落地”的公式。后续如果 detector
调好，可以把 `rho_t` 重新作为 `sigma_t(q)`、write cap 或 decay 的外部控制信号，
但不影响当前 no-detector 主实验口径。
