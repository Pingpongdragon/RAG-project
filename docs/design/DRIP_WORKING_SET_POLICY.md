# DRIP Working-Set Policy

本文件保留 `DRIP-WorkingSet` 的入口说明。完整公式、协议和复现命令统一维护在
[`DRIP_PREDICTIVE_SET_UTILITY.md`](DRIP_PREDICTIVE_SET_UTILITY.md)，不再为 direct、
prefetch 和 hidden 分支分别维护一套准入规则。

## 统一方法

窗口 `t` 结束后，策略根据因果历史估计下一窗口证据状态：

```text
observed state:    MEF D_t(d), HSU S_t(d)
forecast state:    FEU F_t(d) = |Q_t| c_t x_hat_{t+1}(d)
document utility:  u_hat_t(d) = Hedge(semantic, recency)
set utility:       U_hat_t(K)
```

`x_hat_{t+1}` 由离散 evidence-transition estimator 与 Nadaraya--Watson
conditional-mean estimator 融合得到。训练样本 `(z_i, x_{i+1})` 只在 `i+1` 窗口
到达后写入历史，因此没有未来查询或 gold-support 泄漏。

Direct、forecast 与 hidden relation completion 只负责提出 evidence group `G`。
令 `C=G\K_t` 为缺失成员，`V` 为同样大小的 resident victim 集，唯一动作规则为：

```text
Delta_t(G,V)
  = U_hat_t((K_t \ V) union C)
    - U_hat_t(K_t)
    - lambda_t |C|.

execute iff Delta_t(G,V) > 0.
```

因此：

1. recency 足够时，Hedge 提高 recency expert 权重，仍使用同一个 optimizer；
2. 可预测转移出现时，FEU 提前提高未来 evidence group 的边际效用；
3. hidden chain 作为 hyperedge 仅在完整驻留时获得互补效用；
4. `lambda_t` 是长期 replacement constraint 的统一影子价格。

## 代码入口

```text
cache_manager/predictive_prefetch.py  条件证据分布预测
cache_manager/predictive_utility.py   集合效用与统一交换动作
cache_manager/primal_dual.py          replacement shadow price
cache_manager/working_set.py          完整策略与 NoForecast 严格消融
```

实验中通过以下入口启用：

```bash
--drip-ablation \
--strategies LRU DRIP-WorkingSet-NoForecast DRIP-WorkingSet
```

`DRIP-WorkingSet-NoForecast` 只移除 `F_t`，保留相同的集合效用、Hedge、hidden
hyperedge 和 primal--dual controller，因此是预测组件的严格消融。
