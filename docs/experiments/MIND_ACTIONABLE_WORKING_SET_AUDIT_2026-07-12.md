# Actionable Support Working-Set Audit (2026-07-12)

## 1. 结论

本轮把旧 query-distribution CUSUM 改成 support working-set KCUSUM，并把固定
`drift_state_discount` 改成 MMD-witness selective forgetting。完整 200k MIND
自然访问流上得到严格 Pareto 改善：

```text
DRIP - DRIPNOdetector
  Has-Answer       +342 hits / 200,000 = +0.171 percentage points
  AMAT             1.908 -> 1.891
  Replacements     4,658 -> 4,581 (-77, -1.65%)
```

paired-window bootstrap 95% CI 为 `[+0.120,+0.222]` percentage points。Detector
收益仍然不大，不能取代 primal-dual updater 作为主性能来源；但它现在检测的是与
cache action 对齐的变量，并在自然流上同时改善质量和写入成本。

## 2. 最终系统结构

```text
request
  -> observable support proxy
  -> support working-set KCUSUM
  -> expected miss-reduction estimator
  -> MMD-witness selective demand/serve forgetting
  -> primal-dual admission
```

完整公式见：

```text
docs/design/DRIP_ACTIONABLE_WORKING_SET_DESIGN.md
```

实现位置：

```text
algorithms/drip/detection/working_set_cusum.py
algorithms/drip/cache_manager/core.py
```

## 3. 协议

- dataset：MIND-small natural positive-click stream；
- causal warm-up：1,500 events；
- evaluation：200,000 events，400 windows x 500；
- corpus：51,282 news items；
- hot budget：513，即 pool 的 1%；
- item access 在本次请求结束后暴露 exact key；
- 当前窗口先按更新前 cache 计分，再允许策略更新；
- detector/controller 不读取 `sf_titles`、future request 或 timestamp oracle；
- seed：42。

## 4. 四项主指标

| Method | Has-Answer | AMAT | Recall@5 | Replacements |
|---|---:|---:|---:|---:|
| LRU | **91.9305** | **1.807** | **91.9305** | 14,519 |
| DRIPNOdetector | 90.9180 | 1.908 | 90.9180 | 4,658 |
| DRIP + actionable support KCUSUM | 91.0890 | 1.891 | 91.0890 | **4,581** |

在该 exact item-access workload 上，Recall@5 与 Has-Answer 数值一致。相对 LRU，
完整 DRIP 损失 `0.8415` Has-Answer points，但 replacements 减少 9,938 次，即
`68.45%`。这个结果仍然是“接近 LRU 质量、显著降低写入”的成本优先 operating
point，而不是在自然 temporal locality 上全面击败 LRU。

## 5. 与旧 query-CUSUM 的严格比较

| Detector | Hits | Has-Answer | AMAT | Replacements | Delta vs no-detector |
|---|---:|---:|---:|---:|---:|
| none | 181,836 | 90.9180 | 1.908 | 4,658 | baseline |
| old query-CUSUM + global reset | 182,134 | 91.0670 | 1.893 | 4,601 | +298 hits, -57 repl. |
| new support KCUSUM + selective reset | **182,178** | **91.0890** | **1.891** | **4,581** | **+342 hits, -77 repl.** |

`DRIPNOdetector` 在新旧结果文件中完全相同：181,836 hits、4,658 replacements。
因此 detector-target 重构没有改变实验流、初始化、evidence 或 primal-dual baseline。
新版本相对旧 detector 额外增加 44 hits，并再减少 20 replacements。

## 6. Detector 是否真的检测 working-set change

最终 detector 在 400 windows 中 hard alarm 63 次；140 个窗口的 CUSUM statistic
大于零并产生非零 controller intensity。以下是独立于 embedding detector 的
post-hoc audit：

| Diagnostic | Alarm windows | Non-alarm windows | Ratio | Top-20 shift recall |
|---|---:|---:|---:|---:|
| Item-ID distribution JSD | 0.19642 | 0.18034 | 1.089x | 40% |
| Category distribution JSD | 0.01328 | 0.00919 | 1.445x | 45% |
| New-item rate | 4.083% | 3.822% | 1.068x | 20% |

Alarm 窗口不仅 category 变化更大，exact support-item distribution 的变化也更大。
但新-item rate 只小幅上升，再次说明 MIND 是高复用、平滑变化 workload，detector
的边际收益天然有限。

## 7. 写入后是否减少未来 miss

当前窗口的 plug-in replacement value 为：

```text
V_hat_t = sum_i [p_hat_t(c_i) - p_hat_t(v_i)]_+
```

Controller action 在窗口更新之后生效，因此检查 hard alarm 后的未来窗口。DRIP 相对
no-detector 的平均 Has-Answer 增量为：

| Future horizon | Mean delta | Positive alarm episodes |
|---|---:|---:|
| next 1 window | +0.206 pp | 35 / 62 |
| next 3 windows | +0.161 pp | 45 / 62 |
| next 5 windows | +0.152 pp | 47 / 62 |

这比只验证“alarm 与 category JSD 相关”更接近 cache controller 的真实目标：告警后
更新确实在后续请求中减少 miss。并非每次 alarm 都有收益，因此后续仍应研究更好的
value calibration，而不能把所有统计 change point 都解释成必写事件。

## 8. Controlled behavior regression

在一个 10-item causal trace 中，前 4 windows 只访问 `{0,...,4}`，之后 abrupt
切换到 `{5,...,9}`，cache budget 为 5：

| Method | Post-shift hits | Replacements |
|---|---:|---:|
| DRIPNOdetector | 360 | 5 |
| DRIP + support KCUSUM | **440** | 5 |

两者最终写入成本相同，detector 通过更快释放旧 resident state 减少了适应延迟。该
行为已固化为 `test_actionable_working_set_shift_reduces_future_misses_at_same_write_cost`。

## 9. 参数结论

删除：

```text
drift_state_discount
```

保留：

```text
drift_warmup_windows = 3
drift_target_arl     = 100
```

动态 retention 不新增 fast-beta：

```text
beta_X,t(d) = beta_X [1 - rho_t (1 - r_t(d))], X in {D,S}
```

基础 beta 可以用 half-life 解释：`beta_D=0.92` 对应 8.31 windows，
`beta_S=0.75` 对应 2.41 windows。

## 10. Paper decision

1. primal-dual replacement constraint 仍是最强、最稳定的主贡献；
2. detector 应命名为 actionable support working-set detector，而不是 generic query
   drift detector；
3. 自然流结果现在支持把 detector 写入完整系统，但必须同时给 no-detector 消融并
   诚实说明增益为 0.171 points；
4. LRU 仍是 temporal locality 的强边界；
5. hidden evidence 是另一条证据可见性问题，不由 MIND 或本 detector 审计替代。

## 11. Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 \
EMBED_MODEL=all-MiniLM-L6-v2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/hidden/run.py \
  --datasets mind_news_access \
  --workload natural_temporal \
  --n-windows 400 --window-size 500 \
  --warmup-windows 3 --kb-pool-ratio 0.01 \
  --strategies LRU DRIPNOdetector DRIP \
  --output mind_access_support_witness_full_400w500.json
```

Authoritative result:

```text
experiments/agent/data/mind_access_support_witness_full_400w500.json
```

