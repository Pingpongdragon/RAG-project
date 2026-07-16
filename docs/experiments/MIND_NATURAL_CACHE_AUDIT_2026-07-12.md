# Natural Cacheability and Drift Audit (2026-07-12)

> **Scope: one natural workload only.** 跨 5 个 workload 的完整结果、复现命令和
> JSON/表格/曲线位置请从 `docs/experiments/README.md` 进入；本文件只深入审计
> MIND natural access，不代表完整实验清单。

> **Historical detector baseline.** 本文件记录旧 query-CUSUM + global ledger
> discount 结果。新的 support working-set detector、selective retention 和完整
> 200k 对照见 `MIND_ACTIONABLE_WORKING_SET_AUDIT_2026-07-12.md`。旧文件保留，
> 用于严格比较 detector target，而不是覆盖历史数字。

## 1. 结论先行

这轮实验把两个问题分开了：

1. **cache 不是伪需求。** 在 MIND-small 的自然点击流中，200,000 条评估请求
   只有 6,761 个唯一 support，96.62% 的 support occurrence 在过去出现过；LRU
   的 Has-Answer 达到 91.93%。这与原 Hotpot causal stream 的低复用完全不同。
2. **显式 drift detector 暂时不是核心增益。** Kernel CUSUM 确实更常在类别分布
   变化较大的窗口报警，但相对同一 DRIP 的 no-detector 版本只提升 0.149 个百分点
   （298/200,000 个 hit），少 57 次 replacement。检测信号是真实的，controller
   的实用收益却很小。
3. **primal-dual replacement control 是当前最稳的结果。** DRIP 相对 LRU 只损失
   0.864 个 Has-Answer 百分点，却把 replacements 从 14,519 降到 4,601，减少
   68.31%。实际 replacement load 为 `4601 / 17866 = 25.75%`，接近预设长期约束
   `b_rep=25%`。
4. **新的 contrastive evidence 公式没有支配旧公式。** 它更学术、参数更少，也能
   降低写入，但 StreamingQA proxy 上的 Has-Answer 明显下降。因此只保留为
   ablation，不替换当前 rank-distance evidence。

## 2. 数据集和协议

### 2.1 为什么改用 MIND

MIND 是 Microsoft 发布的真实多用户新闻推荐日志，原始 MIND-small training archive
包含新闻语料、用户历史和带正负标签的 impression。数据主页：
<https://msnews.github.io/>。

本实验使用原始 `behaviors.tsv` 和 `news.tsv`：

- cold corpus：51,282 篇新闻；
- users：50,000；
- positive click events：236,344；
- 评估流严格按原 timestamp、behavior row、click order 排序；
- 每个 positive click 是一个单 support item-access request；
- clicked news ID 是请求发生后可观察的 access key，不是未来 oracle label。

本地 archive MD5 为 `bd6ae77fa15949653f39829e946d327c`，与 MIND 官方发布值
一致。原始 ZIP 不提交到 Git，使用仍受官方数据条款约束。

### 2.2 Causal protocol

- 前 1,500 条事件只用于 causal initialization；
- 后 200,000 条事件用于评估，二者时间上严格不重叠；
- hot cache budget `B=513`，即 `KB/pool=1.00%`；
- 当前窗口先按更新前的 cache 计算 Has-Answer，再允许策略更新；
- item trace 的 miss 只给实际访问项记一次 demand，不能把一个 click 扩成
  Top-K 个语义“伪访问”；
- 普通 QA workload 不含 `access_title`，仍使用 cold-corpus Top-K evidence。

## 3. Cacheability audit

| Statistic | 200k natural trace |
|---|---:|
| Evaluated events | 200,000 |
| Unique support items | 6,761 |
| Repeated support occurrences | 193,239 |
| Repeated support rate | 96.6195% |
| Queries answerable from past support | 96.6195% |
| Mean adjacent-window support Jaccard | 32.4627% |
| Maximum support frequency | 4,316 |

这些数字说明 MIND 有明确、自然形成的 working-set reuse。它适合回答“cache 是否有
意义”和“写入控制是否有效”，但它是 item-access workload，不应被包装成 hidden
multi-hop evidence benchmark。

## 4. 当前核心公式

### 4.1 Query-visible evidence

普通 QA 的当前正式 evidence 为：

```text
E_t(q,d)
= I[d in TopK(q)] * gamma * max(0,s(q,d))
  / (r_q(d) * (epsilon + 1 - s(q,d))^alpha)
  + b_1 I[r_q(d)=1].
```

自然 item trace 在请求结束后已经观察到 key `x_t`，因此采用标准 access evidence：

```text
E_t(x_t,d) = I[d = x_t].
```

这两个分支不能混淆：前者是在不知道 support ID 时从语义检索产生候选，后者是
标准 cache trace 中已经发生的对象访问。

### 4.2 Demand, serve, and resident priority

```text
D_t(d) = beta_D D_{t-1}(d) + sum_{q in U_t} E_t(q,d)
S_t(d) = beta_S S_{t-1}(d) + A_t(d)
P_t(v) = S_t(v) + D_t(v)
```

`U_t` 是当前 cache 无法支持的请求集合，`A_t(d)` 是 resident 的 serve credit。

### 4.3 Primal-dual replacement control

把 replacement load 写成长期约束：

```text
E[R_t / B_write,t] <= b_rep
```

在线 dual price 为：

```text
lambda_{t+1}
= [lambda_t + eta_t (R_t / B_write,t - b_rep)]_+
eta_t = 1 / sqrt(t + 1).
```

Admission rule：

```text
Delta_t(c,v) = D_t(c) - m P_t(v) - lambda_t
admit c replacing v iff Delta_t(c,v) > 0.
```

因此 `lambda_t` 不是人工拍出的固定 threshold，而是违反写预算约束时自动上涨的
shadow price。

### 4.4 Detector and controller

Detector 使用 bootstrap-calibrated kernel CUSUM：

```text
z_t = (MMD^2(P_ref,P_t) - mu_0) / sigma_0
G_t = max(0, G_{t-1} + z_t - kappa)
alarm iff G_t >= h(ARL_0).
```

告警时只折扣旧 regime 的账本：

```text
beta_D,t = beta_D (1 - eta_drift)
beta_S,t = beta_S (1 - eta_drift).
```

它不同时扩大 write cap 或修改 admission margin，避免一个 alarm 控制多个旋钮。

## 5. Main result: 200k events

| Method | Has-Answer | AMAT | Recall@5 | Replacements |
|---|---:|---:|---:|---:|
| LRU | **91.9305** | **1.807** | **91.9305** | 14,519 |
| DRIP w/o detector | 90.9180 | 1.908 | 90.9180 | 4,658 |
| DRIP + CUSUM | 91.0670 | 1.893 | 91.0670 | **4,601** |

在该 item-access workload 上，query 文本就是被访问新闻的内容，resident support 的
Recall@5 与 Has-Answer 数值一致；主 cache 结论仍以 Has-Answer、AMAT 和
Replacements 为准。

DRIP 相对 LRU：

- Has-Answer：`-0.8635` percentage points；
- AMAT：`+0.086`；
- Replacements：`-9,918`，即 `-68.31%`。

Detector 相对 no-detector：

- `+298` hits / 200,000 queries，即 `+0.149` percentage points；
- paired window bootstrap 95% CI `[+0.1125, +0.1875]` points；
- replacements `4658 -> 4601`；
- 400 个窗口报警 64 次，平均每 6.25 个窗口一次。

所以 detector 的增益在统计上稳定，但系统量级很小，不足以把它描述成当前方法的
主要性能来源。

## 6. Are the alarms meaningful?

以相邻窗口 category distribution 的 Jensen-Shannon divergence 做独立 post-hoc
audit：

| Diagnostic | Alarm windows | Non-alarm windows |
|---|---:|---:|
| Mean category JSD | 0.013172 | 0.009194 |
| Mean new-item rate | 4.197% | 3.923% |

告警窗口的 category JSD 平均高约 43%，且 20 个最大 category shift 中有 40% 被
告警命中（随机按 64/400 alarm rate 约为 16%）。所以 detector 能感知真实 query
distribution change；问题在于这些变化对 cache replacement 的边际价值不大，或
单次 25% ledger discount 不是足够有效的 action。

## 7. Contrastive evidence ablation

尝试的参数更少的公式为：

```text
b_q = s_(K+1)
tau_q = max(0.02, 1.4826 * MAD(s_(1:K+1)))
a_q(d) = (s(q,d) - b_q) / tau_q
p_q(d) = exp(a_q(d)) / sum_j exp(a_q(j))
M_q = gap_t(q) * [log(1 + sum_j exp(a_q(j))) - log(1 + K)]_+
E_t(q,d) = M_q p_q(d).
```

优点是 query-specific calibration、flat retrieval neighborhood 得到零 evidence、
没有 rank/epsilon/alpha/top-1 bonus 四组独立旋钮。可是 StreamingQA year-proxy
诊断结果不占优：

| Evidence | Has-Answer | AMAT | Replacements |
|---|---:|---:|---:|
| Current rank-distance | 46.3 | 6.366 | 2,634 |
| Contrastive log-evidence | 43.0 | 6.700 | 1,933 |
| Bounded contrastive | 40.7 | 6.934 | 1,845 |

该 StreamingQA mirror 缺失官方 `question_ts/evidence_ts`，这里只能作为公式
diagnostic，不能作为正式 temporal result。即便如此，contrastive 版本也只形成
“更少写入、明显更低 answerability”的点，不应替换当前公式。

## 8. Paper decision

- 保留 **primal-dual replacement control** 作为主方法贡献；它在自然 200k trace
  上准确实现成本约束并给出清晰 Pareto trade-off。
- 把 **kernel CUSUM** 降为 optional controller / ablation。除非后续找到 detector
  带来至少可见系统收益的自然 change-point workload，否则不要把它写成主性能来源。
- 保留 **contrastive evidence** 为负结果/ablation，不合入正式 evidence 公式。
- 继续把 hidden evidence 作为独立研究问题；MIND 不验证 hidden multi-hop。

## 9. Reproduction

```bash
EMBED_MODEL=all-MiniLM-L6-v2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiments/hidden/run.py \
  --datasets mind_news_access \
  --workload natural_temporal \
  --n-windows 400 --window-size 500 \
  --warmup-windows 3 --kb-pool-ratio 0.01 \
  --strategies LRU DRIPNOdetector DRIP \
  --output mind_access_full_400w500_kbp1.json
```

Authoritative result:
`experiments/agent/data/mind_access_full_400w500_kbp1.json`.
