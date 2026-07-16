# DRIP: Actionable Support Working-Set Adaptation

本文记录 2026-07-12 之后的 detector/controller 主设计。核心变化不是换一个
query-drift threshold，而是重新定义 detector 的任务：**只检测会改变 hot-cache
最优内容的 support working-set shift**。

## 1. 系统定位

DRIP 不再把“query topic 发生变化”直接等同于“cache 应该重写”。完整链路分成五层：

```text
online request
  -> observable support proxy
  -> support working-set KCUSUM
  -> replacement-value / resident-obsolescence estimator
  -> support-selective forgetting
  -> primal-dual admission under the long-run write constraint
```

各层回答不同问题：

1. `support observer`：当前真正需要哪些文档；
2. `KCUSUM`：support 分布是否持续偏离旧 regime；
3. `value estimator`：旧 resident 是否失效，替换后预计能否减少未来 miss；
4. `selective forgetting`：只忘掉与新 regime 不一致的历史状态；
5. `primal-dual writer`：即使发生漂移，也必须支付长期 replacement shadow price。

因此 detector 是 cache state estimator，不是独立的写入策略。统计 change point 和
cache actionability 必须分别报告。

## 2. 无标签泄漏的 support proxy

对每个当前请求 `q`，构造在线可观察的 support proxy `x(q)`：

```text
x(q) = observed access key                         item-access trace
x(q) = served resident                             supported QA request
x(q) = Top-1_D(q)                                  under-covered QA request
```

item trace 的 key 在请求发生后可观察；普通 QA 的 hit proxy 来自本次 hot-tier
serving result，miss proxy 来自当前允许的 cold probe。两者都不读取 gold support
或未来请求，也不会为 detector 给已命中的请求增加冷库访问。

窗口 support working set 为：

```text
P_t^x = (1 / |Q_t|) sum_q delta_{phi(x(q))}
```

其中 `phi` 是固定文档 encoder。

## 3. Support working-set KCUSUM

使用 RBF MMD 比较冻结 reference 与当前 support distribution：

```text
M_t = MMD_RBF^2(P_ref^x, P_t^x)
z_t = (M_t - mu_0) / sigma_0
G_t = max(0, G_{t-1} + z_t - kappa)
alarm_t = I[G_t >= h(ARL_0)]
```

`mu_0`、`sigma_0` 和控制限 `h` 全部由 warm-up bootstrap 校准。`ARL_0` 是稳定
working set 下的目标误报间隔。为了让 gradual shift 平滑进入 controller，定义：

```text
chi_t = min(1, G_t / h)
```

`chi_t` 只是控制限消耗比例，不是 posterior probability；正式 change point 仍只由
`alarm_t` 定义。确认 change point 后 rebase reference。

## 4. 旧 resident 失效与预计 miss reduction

令 `n_t(d)` 是当前窗口 support proxy `d` 的出现次数：

```text
p_hat_t(d) = n_t(d) / sum_j n_t(j)
```

当前 resident obsolescence 为：

```text
O_t = 1 - sum_{v in K_t} p_hat_t(v)
```

它表示当前 support mass 中 hot tier 无法覆盖的部分。为了判断写入是否有未来价值，
在短期局部平稳假设下，把概率最高的 non-resident `c_i` 与概率最低的 resident
`v_i` 成对，估计一个 write batch 的 counterfactual value：

```text
V_hat_t = sum_{i=1}^{L_t} [p_hat_t(c_i) - p_hat_t(v_i)]_+
L_t = min(B_write,t, |C_t|, |K_t|)
```

`V_hat_t` 同时要求三件事成立：cold candidate 有需求、victim 已失效、替换预计降低
下一窗口 miss。它不是未来 oracle，而是局部平稳 plug-in estimate。

用 warm-up 的中位数 `V_0` 自归一化：

```text
a_t = V_hat_t / (V_hat_t + V_0 + epsilon)
rho_t = chi_t * a_t
```

`rho_t` 是 controller intensity。topic 变了但 `V_hat_t=0` 时，统计日志可以报警，
cache 状态不会因此被重置。

## 5. MMD witness 与 support-selective forgetting

全局乘一个 `beta` 会同时削弱 candidate demand 和 victim priority，admission gain
容易相互抵消。DRIP 改用 MMD witness 区分新旧 regime 文档：

```text
w_t(d)
= E_{x~P_t^x} k(phi(d), x)
  - E_{x~P_ref^x} k(phi(d), x)
```

对当前 tracked documents 的 `w_t(d)` 做 median/MAD 标准化，再经 sigmoid 得到：

```text
r_t(d) in [0,1]
```

`r_t(d)` 越大，文档越属于当前 support regime。对 demand 和 serve 使用统一的
效用存活公式：

```text
beta_D,t(d) = beta_D [1 - rho_t (1 - r_t(d))]
beta_S,t(d) = beta_S [1 - rho_t (1 - r_t(d))]

D_t(d) = beta_D,t(d) D_{t-1}(d) + E_t(d)
S_t(d) = beta_S,t(d) S_{t-1}(d) + A_t(d)
```

这里 `beta_D` / `beta_S` 是跨一个窗口的时间存活率，方括号是旧 regime utility
在当前 working set 下仍然相关的存活率。两个失效风险相乘，因此不再需要
`drift_state_discount`、`fast_beta` 或 detector-specific gain margin。

稳定期 `rho_t=0` 时严格退化为原 DRIP；当前 regime 文档 `r_t(d)=1` 时也保留原
half-life；只有同时满足“检测证据强”和“文档属于旧 regime”才加速遗忘。

## 6. Writer 保持独立的长期成本约束

Resident priority 与 admission 不变：

```text
P_t(v) = S_t(v) + D_t(v)
Delta_t(c,v) = D_t(c) - m P_t(v) - lambda_t
admit iff Delta_t(c,v) > 0
```

dual price 继续执行：

```text
lambda_{t+1}
= [lambda_t + eta_t (R_t / B_write,t - b_rep)]_+
```

Detector 不直接扩大 write cap，也不绕过 `lambda_t`。这保证 detector 负责状态估计，
primal-dual controller 负责长期成本，两者的消融可以独立解释。

## 7. 参数与复杂度

Active detector/controller 参数只剩：

```text
drift_warmup_windows
drift_target_arl
```

`beta_D`、`beta_S` 是原账本参数，建议在论文中同时报告对应 half-life：

```text
H(beta) = log(0.5) / log(beta)
```

默认 `beta_D=0.92` 对应约 8.31 windows，`beta_S=0.75` 对应约 2.41 windows。
KCUSUM 使用已有 reference cap；witness 按 batch 计算，不构造全部 tracked-document
与 reference 的永久矩阵。

## 8. 实验必须回答的问题

主消融至少包含：

```text
DRIPNOdetector                   evidence + primal-dual
DRIP                             + support KCUSUM + actionable forgetting
historical query-CUSUM           detector-target ablation
global ledger reset              controller ablation
```

Detector 不能只报告 alarm/JSD。必须报告：

1. post-alarm future miss reduction；
2. stale-resident eviction precision；
3. Has-Answer / AMAT 改变量；
4. 为这些收益支付的额外 Replacements；
5. `DRIP` 相对 `DRIPNOdetector` 的 paired-window confidence interval。

如果自然 workload 上 detector 仍没有可见净收益，应保留为 optional component，不能
因为公式更完整就把它宣称为主性能来源。

## 9. 理论来源

- MMD two-sample testing: Gretton et al., *A Kernel Two-Sample Test*, JMLR 2012.
- Sequential kernel change detection: Flynn and Yoo, *Change Detection with the
  Kernel Cumulative Sum Algorithm*, 2019.
- Exponential forgetting under concept drift provides the general state-decay
  interpretation；本设计进一步把统一遗忘改为 support-conditioned survival。
