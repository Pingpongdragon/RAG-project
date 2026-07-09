# DRIP 参数与论文公式对照

当前 DRIP 只维护两个实验入口：

```text
DRIPNOdetector  当前主实验：不依赖 detector，只用 direct evidence + serve/demand + replacement control。
DRIP            后续 detector 调好后使用的版本，目前不作为主结论。
```

## 1. 当前文件入口

```text
algorithms/drip/cache_manager/core.py
  主窗口循环：观察 cache 覆盖、路由 evidence、更新 serve/demand、调用 writer。

algorithms/drip/cache_manager/policies.py
  策略入口：DRIPNOdetector / DRIP。

algorithms/drip/cache_manager/evidence_core.py
  direct evidence、hidden diagnostic、replacement-aware admission。

algorithms/drip/cache_manager/drip_config.py
  参数表：
    DRIPCoreConfig = 当前主实验最小参数；
    DRIPHiddenDiagnosticConfig = hidden / detector 旧分支参数。
```

旧 `CostAwareDRIP` 文件已经从 active 目录移除；现在不再有两套 config。
日常调主实验只看 `DRIPCoreConfig`，可以先忽略 `DRIPHiddenDiagnosticConfig`。

## 2. Max 目标

```text
max_{K_t: |K_t| <= B} J_t(K_t)

J_t(K_t)
= Answerability_t(K_t)
  - lambda_amat * AMAT_t(K_t)
  - lambda_rep  * ReplacementCost_t(K_t, K_{t-1})
```

对应实验指标：

```text
Recall@5
Has-Answer Rate
AMAT
Replacements
```

## 3. Direct evidence

代码位置：

```text
algorithms/drip/cache_manager/evidence_core.py::_credit_dense
```

公式：

```text
E_t(q, d)
= sigma_t(q)
  * I[d in TopK(q)] * max(0, sim(q, d))
  / (rank_q(d) * (epsilon + 1 - sim(q, d))^alpha)
```

当前实现里 `sigma_t(q)=1`，因为只有 under-covered query 才会进入
`_credit_dense`。后续可以把缺失 support slot 数传进去。

相关参数在 `DRIPCoreConfig`：

```text
direct_topk
direct_gamma
direct_top1_bonus
direct_evidence_alpha
direct_evidence_epsilon
```

## 4. Serve / demand update

代码位置：

```text
core.py::_credit_serve
evidence_core.py::_credit_dense
evidence_core.py::_decay
```

公式：

```text
D_t(d) = beta_d * D_{t-1}(d) + sum_q E_t(q, d)
S_t(d) = beta_s * S_{t-1}(d) + A_t(d)
```

相关参数：

```text
demand_decay
serve_decay
serve_topk
serve_prior
min_stat
demand_ledger_cap
```

## 5. Resident priority

代码位置：

```text
evidence_core.py::_priority_for_route
```

公式：

```text
P_t(v)
= S_t(v)
 + D_direct,t(v)
 + D_bridge,t(v)
 + pair_lease_weight * PairLease_t(v)
```

对于当前主实验 `DRIPNOdetector`：

```text
use_bridge     = False
use_pair_lease = False

P_t(v) = S_t(v) + D_direct,t(v)
```

## 6. Replacement-aware admission

代码位置：

```text
evidence_core.py::_replacement_penalty
evidence_core.py::_write
```

合并后的替换惩罚：

```text
C_t(c, v) = lambda_replace * (1 + mu * phi_t)
phi_t     = EMA(replacements_per_window) / replacement_budget
```

admission：

```text
Delta_t(c, v)
= D_t(c)
  - gain_margin * P_t(v)
  - C_t(c, v)

admit c replacing v iff Delta_t(c, v) > 0
```

相关参数：

```text
gain_margin
replacement_cost
replacement_pressure_mu
replacement_ema_decay
tau_duplicate
```

## 7. Hidden / detector 旧参数

当前主实验 `DRIPNOdetector` 明确关闭 detector：

```text
use_drift_detector=False
```

bridge / GraphIndex / detector 参数都集中在：

```text
DRIPHiddenDiagnosticConfig
```

`detection/multi_agent_drift.py` 仍保留，之后调好 detector 后可作为 `DRIP`
版本的 `rho_t` 信号：

```text
rho_t = MultiAgentDriftDetector(...)
```

相关参数：

```text
use_drift_detector
drift_warmup_windows
drift_min_agent_queries
drift_z_threshold
drift_centroid_threshold
drift_write_boost
drift_decay_boost
drift_margin_discount
```
