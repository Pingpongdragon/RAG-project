# DRIP V2: Evidence-Breadth Routing

日期：2026-07-11

## 1. 设计目标

V1 使用 dense top-k evidence、serve/demand ledger 和 replacement gate。它在
direct topic drift 上优于 recency baseline，但在强时间局部性数据上可能不如 LRU：
同一个 semantic eviction rule 不适合所有 query windows。

V2 不再增加一个需要人工阈值标注的 drift detector。它回答一个更直接、可在线观测
的问题：当前 support failure 是否暴露了一个可辨识的 evidence set？

- 若 query 明确指向多个互补候选，保护 semantic evidence bundle；
- 若 cold probe 只有一个明确候选，使用 recency eviction；
- candidate admission 和 replacement penalty 继续使用 V1，避免一次改动多个组件。

本文暂称该方法为 **Evidence-Breadth Routed DRIP (DRIP-EBR)**。代码中的实验名为
`DRIP-MentionRouted`，尚未加入正式 baseline registry。

## 2. Candidate-Local IDF Mention

对 under-covered query `q` 的 cold top-k 候选集 `C_q`，只在候选标题内部计算
document frequency：

```text
idf_C(w) = log((|C_q| + 1) / (df_C(w) + 1))
```

候选 `d` 的 query-visible mention 分数为：

```text
M(q,d)
= sum_{w in tok(q) intersect tok(title(d))} idf_C(w)
  / sum_{w in tok(title(d))} idf_C(w)
```

候选集共有的站点名、主题词或模板词满足 `df_C(w)=|C_q|`，其 IDF 为 0，不会被
误当成区分性 evidence。该计算只读取在线 query、候选标题和 dense rank，不读取
`sf_titles`、`qtype`、数据集名称或未来 query。

V2 direct credit 为：

```text
E(q,d_r)
= gamma * max(0, sim(q,d_r))
  / [r * (epsilon + 1 - sim(q,d_r))^alpha]
  + b_1 * I[r=1]
  + (b_1/r) * M(q,d_r) * I[r>1]
```

因此 top-1 保持 V1 credit；只有 rank>1 且被 query 明确提及的候选获得互补 evidence
credit。它不是简单地扩大 top-k 写入。

## 3. Window Evidence Breadth

对窗口中每个触发 cold probe 的 query，定义：

```text
z(q) = I[exists d_r in C_q, r>1 and M(q,d_r)>0]
rho_t = (1 / |U_t|) * sum_{q in U_t} z(q)
```

`rho_t` 表示当前失败流中有多少 query 暴露了互补 evidence。路由规则为：

```text
if rho_t >= 1/2:
    victim order = ascending [S_t(v) + D_t(v)]
else:
    victim order = ascending last-access time
```

`1/2` 是窗口多数决，不是按数据集调出的阈值。路由只改变 victim ordering；候选
生成、write budget 和 admission gate 保持一致：

```text
Delta_t(c,v) = D_t(c) - m P_t(v) - C_t
C_t = lambda_rep * (1 + mu * phi_t)
admit iff Delta_t(c,v) > 0
```

## 4. 学术叙事

核心命题不是“semantic cache 总比 LRU 强”，而是：

> eviction policy 应由 failure 中可观测的 evidence breadth 决定。直接 evidence
> 暴露时，document-level recency 会拆散互补 evidence；evidence 稀疏且时间局部性
> 强时，semantic ledger 又会过度保留旧主题。DRIP-EBR 在两种 regime 间做无监督
> 的在线选择，并用统一 replacement price 控制写放大。

这比用一个 detector 同时调 decay、margin、budget 和 penalty 更容易分析和消融：
detector 只决定使用哪一种 eviction inductive bias，writer 仍是同一条净收益规则。

## 5. 实现边界

- 当前 router 处理 query-visible/direct evidence；hidden evidence 仍由 bridge 模块
  单独研究，不应把未完成 hidden 结果包装进主表。14 万文档 2Wiki-simple stress
  test 已验证：inference/compositional query 也可能具有高 title-mention breadth，
  但 direct writer 无法找到 hidden support，因此 breadth 不能替代 bridge completion。
- 当前结果是固定 embedding 上的 hot-tier simulator；FAISS microbenchmark 只补充
  索引搜索与 replacement 成本，不包含网络、磁盘、LLM 生成和分布式同步。
- 1% StreamingQA 是明确失败边界，不能宣称全工作负载 SOTA。
