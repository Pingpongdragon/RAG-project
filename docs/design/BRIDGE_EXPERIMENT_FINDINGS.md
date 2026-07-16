# Bridge 缓存小规模验证 — 阶段性发现

> **当前可讲版本 (2026-06-21):** 这份文档前半部分保留了 PPR / DRF /
> hubness / role rerank 等历史试错记录, 但当前代码和汇报主线已经收束为
> `DRIP = ESC hidden-support completion + Pair Lease retention`。论文口径应把
> **单跳 query shift** 和 **无主题漂移 multi-hop support completion** 分开讲。

## 当前结论速览

### Branch A: single-hop query shift 是校准分支

完整实验:

```text
StreamingQA temporal
pool=29,819, KB=400
100 windows x 50 queries
结果文件: experiments/direct/data/results_streamingqa_temporal_final_clean.json
```

| Strategy | R@5 H1 | R@5 H2 | Writes | MaintR | Cost |
|---|---:|---:|---:|---:|---:|
| ARC | 20.5 | 4.4 | 1083 | 4145 | 5228 |
| FIFO | 50.2 | 32.8 | 2109 | 2907 | 5016 |
| LRU | **52.8** | **33.1** | 2033 | 2843 | 4876 |
| DRIP-Dense | 48.9 | 30.3 | 2261 | 14800 | 17061 |
| DRIP | 48.5 | 29.3 | 2174 | 15010 | 17184 |
| Oracle | 84.4 | 79.4 | 22140 | 0 | 22140 |

读法: 当证据直接可见时, recency / access-history 已经很强。这个分支用于校准
边界, 不作为 DRIP 的主胜场。

### Branch B: no-topic-shift multihop 是当前主胜场

完整实验:

```text
static corpus, detector-free
workload=multi_agent_bridge_reuse
KB=750, n_source=3000
20 windows x 25 queries
retrieval=graph
结果文件:
  experiments/hidden/data/hidden_2wiki_musique_simplified_sd_20w25_kb750_graphret.json
```

2Wiki bridge-comparison:

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 15.5 | 24.9 | 27.9 | 0.2 | **17.9** | 20.8 | 2.886 | 1452 |
| DRIP-Dense | 11.3 | 16.5 | 36.5 | 2.2 | 9.4 | 14.4 | 2.540 | 1056 |
| DRIP-ESC | 17.3 | 26.9 | 45.1 | **3.8** | 16.6 | **25.2** | 2.194 | 824 |
| DRIP | **21.1** | **30.4** | **45.8** | 3.6 | 16.6 | **25.2** | **2.170** | **823** |

MuSiQue:

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 20.3 | 24.5 | 22.1 | 2.4 | 14.6 | 18.8 | 2.080 | 1796 |
| DRIP-Dense | 24.6 | 30.8 | 37.2 | 5.0 | **16.8** | **25.2** | 1.688 | **789** |
| DRIP-ESC | 24.1 | 29.0 | 36.0 | 5.6 | 14.4 | 22.4 | 1.728 | 793 |
| DRIP | **26.5** | **32.6** | **37.3** | **7.2** | 14.4 | 22.0 | **1.686** | 800 |

Hotpot 当前构造几乎没有 shared hidden-B group (`reuse=10`), 更像 direct-visible
sanity check, 不作为 hidden bridge 主证据。

100x50 完整受控多跳结果已经复跑。当前更适合作为主表的是:

```text
2Wiki bridge_comparison
n_source=5000, pool=22,984, KB=750
100 windows x 50 queries
workload=multi_agent_bridge_reuse
retrieval=graph
结果文件:
experiments/hidden/data/full100_2wiki_no_shift_multiagent_kb750_graphret_current.json
```

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 3.4 | **10.1** | **9.6** | 0.0 | **15.5** | **17.0** | 3.617 | 3265 |
| DRIP-QueryVisible | 4.1 | 6.9 | 7.2 | 0.0 | 6.8 | 7.7 | 3.712 | 9542 |
| DRIP | **5.9** | 9.6 | 8.8 | **0.3** | 8.1 | 10.4 | **3.648** | 4174 |

读法:

```text
DRIP 在无主题漂移的受控多跳复用里有效:
  R@5 H2: 4.1 -> 5.9 vs QueryVisible
  KB Cov H2: 6.9 -> 9.6
  Reuse: 7.7 -> 10.4
  Writes: 9542 -> 4174

但 ARC 的 Hidden-B/Reuse 仍高, 因为它更保守、写得更少;
DRIP 的优势是 retrieval-facing R@5 + 较高 coverage per write, 不是全面碾压 ARC。
```

同时新增的组合 benchmark:

```text
workload=topic_shift_bridge_reuse
drift=full_gradual
结果文件:
experiments/hidden/data/full100_2wiki_topic_shift_bridge_kb750_graphret_routed.json
```

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 2.4 | **7.7** | **9.0** | 0.0 | **13.6** | **15.7** | **3.641** | 2785 |
| DRIP-QueryVisible | **4.8** | 7.2 | 7.6 | 0.1 | 6.8 | 8.2 | 3.694 | 10009 |
| DRIP | 4.1 | 7.0 | 8.3 | **0.4** | 6.5 | 8.3 | 3.667 | 4238 |

结论:

```text
topic shift + multihop 目前不是主胜场。
DRIP 在 H1 提升, 但 H2 低于 QueryVisible, 说明漂移后的 tail direct evidence
与 hidden bridge evidence 存在 slot 竞争。

因此论文主 claim 应先放在 no-topic-shift multihop cache residency。
topic shift + multihop 是下一步组合 benchmark, 需要额外的漂移感知 direct/hidden
预算调度, 不能直接拿当前 DRIP 当最终结果。
```

### 当前方法公式

```text
E_ESC(q, B | A) = sim(phi(q,A), B) * link(A,B) * cue(q,B)

L_t(A), L_t(B) <- rho L_{t-1} + E_ESC(q,B|A)

D_t(d) = D_vis,t(d) + D_hid,t(d) + lambda_pair L_t(d)

P_t(d) = S_t(d) + D_t(d)

direct-first admission, bridge-leftover admission
```

### 当前 reuse / drift+multihop 构造

`multi_agent_bridge_reuse` 是无主题漂移的多跳驻留测试:

```text
1. 按 support title 分组。
2. 只保留 title B 不出现在 question text 里的组, B 作为 hidden support。
3. 每组前半段放 1 个 exposure query。
4. 后半段放多个 reuse query, 来自不同 synthetic agents, 共享同一个 B。
5. 初始化 KB 时保留其他 support docs 作为 anchor A, 显式 hold out hidden B。
6. 指标:
   hidden_B_hit = B 是否在 KB;
   reuse_hit = reuse query 到来时 B 是否仍在 KB;
   has_answer = query 的全部 support docs 是否都在 KB。
```

`topic_shift_bridge_reuse` 是新加的组合 benchmark:

```text
1. 先用 query embedding 聚类出 head/tail topics。
2. 只选同一个 hidden B 同时拥有 head-topic query 和 tail-topic query 的 group。
3. H1 放 head-topic exposure query。
4. H2 / gradual 后段放 tail-topic reuse query。
5. 背景 query 按 full_gradual 从 head 逐步切到 tail。
6. 同样 anchor A resident, hidden B held out。
```

这个组合 benchmark 同时测试两个难点:

```text
query distribution shift
hidden multi-hop support completion
```

但当前结果显示二者叠加后会出现 direct evidence 与 bridge evidence 的 slot 竞争,
所以它应该作为下一阶段, 不应替代无漂移多跳主表。

一句话:

```text
DRIP is evidence-conditioned support completion for multi-hop cache residency.
```

当前不要再把主方法说成 PPR、ARC-style DRF/hubness、role rerank, 也不要说
DRIP 已经端到端解决 topic shift + multihop。更稳的主张是: **先在无主题漂移下证明
hidden multi-hop support 能被写入并驻留; topic shift + multihop 是下一步组合 benchmark。**

---

测试脚本: `algorithms/drip/tests/test_bridge_bucketed.py`
合成场景: 12 个 (A,B) bridge 三元组 + 120 噪声, KB预算24, 72查询, 窗口6。
B 是 query 够不着的第二跳, 与 A 共享稀有实体。

## 关键发现 (按因果链)

1. **Benchmark CLI 从不触发 bridge 分支。**
   `DRIPStrategyAdapter.initialize` (benchmarks/archive_legacy/adapters.py:89) 从不调用
   `set_pool_entities`, 且喂给 step() 的 query 只有 {"question": text},
   丢掉了 sf_titles/qtype。`step()` 中 `if not has_metadata()` (line 87) 把
   route 强制降级为 SINGLE。=> 现有所有 benchmark 跑法里 bridge 根本没跑过。
   这本身就是 "H2 不动" 的一部分原因 (机制从未被执行)。

2. **bridge 候选生成几乎全军覆没在 relation 门。**
   合成探针 (graph_evidence 单 query):
   raw_paths=8 -> after_degree_gate=2 -> after_relation_gate=0。
   `_relation_score` (graph_index.py:218) 要求 query 词与 B 的 title/context
   词重叠。第二跳 B 的标题通常与 query 无词面重叠 (这正是"第二跳"的定义),
   所以 relation 门把真正的 bridge B 杀光。这与 SUPPORT_FLOW_METHOD.md:163
   "shared entity + IDF 不够判别" 的已知缺陷吻合。

3. **即便 B 拿到 demand, 也排在直接候选后面。**
   stock 诊断: 12 个 gold-B 中仅 3 个拿到 demand (且来自 first-hop dense
   credit, 不是实体桥), demand 0.05~0.20, 而非 KB 候选里 "other" 的 demand
   更高 (0.14+)。小预算被直接候选吃光。

## 结论 (对设计的影响)

- "分桶预算 + S_brg" (协议第1步) **无法单独验证**, 因为在 relation 门之后
  bridge 桶是空的 —— 候选根本进不来。
- => 病的主次顺序被这次实验**反转**了: 不是 "先预算、后匹配", 而是
  **匹配(候选生成)是第一瓶颈**。relation 门 + 弱实体链先要解决,
  否则分桶/serve腿无米下锅。
- 这反而**强化了上 PPR 的理由**: PPR 用图传播质量替代脆弱的 relation 词重叠
  门, 让第二跳 B 不靠"与 query 词面重叠"也能拿到证据质量。

## 下一步
- 把合成数据改成 relation-realistic (B 标题/上下文含 query 词), 隔离验证分桶；
  或直接跳到 PPR 原型, 因为 PPR 同时解 1+2。

---

# 第二阶段: 局部 PPR 真实数据验证 (结论: 当前实现无效)

测试: `test_bridge_ppr.py`(合成) + `test_bridge_ppr_real.py`(真实 hotpotqa_entity_walk)
环境: conda env `ljy_rag_ft` (base 无 sklearn/torch)。
嵌入: TF-IDF+SVD-128 (验证机制用; 真实实验应换 bge/sentence-transformer)。

## 结果

| 场景 | stock | local-PPR | 增量 |
|------|-------|-----------|------|
| 合成 (A-B共享稀有实体, 噪声正交) | 1/12 | 12/12 | +11 |
| 真实 (264 bridge q, 3474 docs) | 31/382 (8.1%) | 7/382 (1.8%) | **-24 (-6.3pp)** |

合成上的完美表现**完全没有迁移到真实数据**。

## 根因诊断 (100~120 bridge query 抽样)

1. PPR **能找到** gold-B: 94/100 出现在候选里 (远好于 stock 单边门)。
   => "召回"不是问题, PPR 解决了候选生成。
2. 但**子图爆炸**: d_cap=80/R=2 时候选集中位 2651 (≈全语料)。
   局部子图根本不局部 —— 真实 Wikipedia 实体高频且稠密。
3. **排名无法分离**: 收紧 d_cap 到 10/20/40, top-5 命中卡在 ~32/100 不动
   (降 d_cap 是拿召回换排名, 无净收益)。
4. 混合 dense 相似度 (PPR质量 × sim(B,A)^w, 模拟 HippoRAG): top5 仅 39->42/120。
   纯图传播的稳态质量分不开 gold-B 与噪声。

## 结论

- 当前局部 PPR 实现 (组件0-3) 在真实数据上**不可用**。核心病: 稀疏稳态质量
  无法把真 second-hop 从稠密邻域噪声里排出来; query 相关性信号在纯传播里丢失。
- 合成实验是"PPR 必胜"的构造(稀有桥实体+正交噪声), 不能代表真实分布。
  这是一次方法论教训: 合成 sanity check 通过 != 真实有效。
- HippoRAG 真实做法是 PPR + dense + LLM 重排的混合, 且离线全图; 我们的
  流式局部近似丢了太多信息。简单的 dense 重加权补不回来。

## 下一步候选 (未决, 需与人讨论)
- A. 换真实嵌入(bge-small) 重测 —— 当前 TF-IDF 嵌入质量低, 可能拖累 first-hop
     种子, 进而毒化整个 PPR。这是最大未控变量, 应优先排除。
- B. 缩短到 L=1 + 强 beam (退化为"带重排的单跳实体扩展"), 放弃多跳普适性。
- C. 承认纯图传播不够, 回到"实体链 + 学习式重排"路线。

---

# 第三阶段: 真实嵌入 (bge-large) + 真实 2wiki (结论: 方向成立, 增益尚小)

脚本: `experiments/hidden/run_ppr_2wiki.py`
环境: ljy_rag_ft。数据: 2wiki_expanded, pool=8465, queries=1500。
嵌入: BAAI/bge-large-en-v1.5 (1024d, 缓存)。实体: spaCy NER (缓存)。
KB预算=192, 窗口=50, d_cap=30, c=0.5, L=3, R=2, K0=5。

## 结果

| | stock(单边门) | local-PPR | 增量 |
|--|--|--|--|
| bridge gold in KB | 120/3427 (3.5%) | 160/3427 (4.7%) | +40 (+1.2pp) |
| writes (改写成本) | 2796 | 1122 | **-60%** |

## 解读

- **方向翻正**: 换真实 bge 嵌入后 PPR 从 -6.3pp(TF-IDF) 变为 +1.2pp。
  证实第二阶段的失败主因是 TF-IDF 嵌入毒化 first-hop 种子。
- **效率是真实亮点**: PPR 用 40% 的写入拿到更高命中, update cost 维度明显占优。
- **但绝对增益小**: 3.5%->4.7% 均在低基数 (KB 192 vs gold 3427, 远未饱和)。
  +1.2pp 与 ARC_COMPARISON.md 自承的 "DRIP vs DRIP-Dense +0.8~2.7pp" 同量级,
  尚不足以拉开 "ARC 变体" 的指控。

## 仍未控的变量 / 下一步
- 只跑了 1 组参数 (d_cap=30)。未扫 c/L/R/d_cap/K0, 不知 PPR 上限。
- KB 预算过小导致 floor effect; 应加大预算或换 recall@k 指标看分离度。
- 组件 4-6 (β/γ 双权重 + serve 双通道 S_brg) 仍未实现; 当前只换了 E_graph 来源。
  writes 大降提示: 配合分桶预算可能进一步放大效率优势。
- 应对比 DRIP-Dense / DRIP-ESC (registry 里的实体感知 baseline), 而非只对 stock。

---

# 第四阶段: 多跳代码调参路线 (先修正确性, 再扫参数)

当前不要继续盲调 `graph_index.py` 里的单个权重。多跳链路已经显示出三个不同瓶颈:

1. **执行正确性**: bridge 分支是否真的跑到、query embedding 是否对齐。
2. **候选生成/分离**: `E_graph` 是否能把 hidden B 从实体邻域噪声里分出来。
3. **缓存准入**: bridge 候选是否能拿到预算, 且以较少 writes 进入/留在 hot tier。

调参必须按这个顺序走。否则会把代码 bug、retriever 信号和 writer 策略混在一起。

## 0. 先修一个 PPR 专用脚本的 embedding 对齐 bug

`experiments/hidden/run.py` 的标准写法是:

```python
wqe = np.array([query_embs[q["qidx"]] for q in wq])
```

但 `experiments/hidden/run_ppr_2wiki.py` 目前在 `run()` 里用了:

```python
qe = query_embs[w0:w0 + WINDOW]
```

这和前面注释的 "`qidx` 映射回嵌入" 不一致。`build_query_stream()` 会重排/复制 query,
所以顺序切片可能把窗口 query 喂错 embedding, 毒化 dense first-hop seed, 进而污染 PPR。

应改成:

```python
qe = np.array([query_embs[q["_qidx"]] for q in win], dtype="f")
```

第三阶段的 `+1.2pp / writes -60%` 只能先当方向性信号; 修完这个 bug 后必须复跑。

## 1. 固定非 Graph 组件, 只调 E_graph

为了隔离变量, 第一轮只允许改变 PPR 参数:

```text
c      restart probability
L      PPR 截断步数
R      BFS 半径
K0     dense first-hop seed 数
d_cap  实体度数硬上限
```

其他保持固定:

```text
router / detector / DRIP-Dense / writer / gain_margin / KB init
```

第一轮目标不是 H2 最终最高, 而是回答:

```text
PPR 的候选质量上限是多少?
用多少候选规模能达到这个上限?
```

必须记录这些诊断:

```text
seed_recall_A          gold A 是否进 TopK0
candidate_recall_B     gold B 是否出现在 PPR 子图/候选中
topk_recall_B          gold B 在 PPR 排名前 k 的比例
subgraph_size_p50/p90  |V_q| 分布
selected_docs/query    每个 query 输出多少 bridge candidate
```

如果没有这些诊断, 单看 `bridge gold in KB` 无法判断是 graph 找不到、rank 排不出,
还是 writer 没写进去。

## 2. 推荐 PPR 扫参顺序

先扫控制子图大小的参数, 再扫传播参数。

### 2.1 先扫 `d_cap`

固定:

```text
c=0.5, L=3, R=2, K0=5
```

扫:

```text
d_cap in {10, 20, 30, 50, 80}
```

看:

```text
candidate_recall_B
top5/top20_recall_B
subgraph_size_p50/p90
runtime/query
```

判断:

- 若 `candidate_recall_B` 低: `d_cap` 太小, 真桥实体被截掉。
- 若 `candidate_recall_B` 高但 `topk_recall_B` 低: 子图太稠密, 需要更强 rerank/更小 R,
  不是继续放大图。
- 若 `subgraph_size_p90` 接近全语料: 当前设置不可用, 先收 `R` 或 `d_cap`。

### 2.2 再扫 `R` 和 `L`

建议先限制:

```text
R in {1, 2}
L in {2, 3}
```

不要一上来 `R=3/L=4`。第二阶段已经证明真实 Wikipedia 实体图很容易爆炸。

判断:

- `R=1` 有收益: 多跳其实可以退化为 "A 的实体邻居 + rerank", 低成本且更稳。
- `R=2` 明显更好: PPR 多跳传播确实提供额外 bridge signal。
- `R=2` 只增加候选不增加 topk: 保留 `R=1`, 把精力放 rerank/writer。

### 2.3 最后扫 `c` 和 `K0`

```text
c in {0.3, 0.5, 0.7}
K0 in {3, 5, 8}
```

解释:

- `c` 越大越贴近 dense seed, 更稳但桥接弱。
- `c` 越小传播越远, 召回可能升, 噪声也会升。
- `K0` 太小会漏 first-hop A; 太大会把错误 seed 带入 PPR。

若真实 bge first-hop 已经很强, 优先 `K0=3/5`。若 `seed_recall_A` 低, 再扩大 K0。

## 3. 第二轮才调 writer: 分桶预算 + bridge serve

等 `E_graph` 的 `topk_recall_B` 稳定后, 再调 writer。这里的目标是把 PPR 的候选质量变成
hot-tier 里的常驻收益, 同时保持 writes 低。

应实现/验证 `LOCAL_PPR_SUPPORT_FLOW.md` 组件 4-7:

```text
D_dir, D_brg       demand 双通道
S_dir, S_brg       serve 双通道
beta               direct vs bridge 偏好
gamma              served utility vs expected demand
budget_brg         route-aware bridge 写预算
```

推荐从最小改动开始:

```text
beta = f_bridge_window      # 窗口内 bridge query 占比, 或固定 0.5
gamma = 0.3                 # 早期 serve 少, demand 权重大一点
bridge_reserve in {0.25, 0.5}
gain_margin 固定不动
```

writer 层必须同时报告:

```text
bridge_candidates
bridge_gold_candidates
bridge_writes
bridge_gold_writes
write_gold_rate
total_writes
cold_fetches_per_query
reuse_hit_rate
```

当前第三阶段最有价值的不是 `+1.2pp`, 而是:

```text
writes 2796 -> 1122 (-60%)
```

这说明 PPR 可能不是显著提高绝对 H2 命中, 而是用更低改写成本达到相近/略高命中。
因此 DRIP 的卖点应优先表述为:

```text
route-aware bridge cache improves evidence residency per write
```

而不是只说:

```text
absolute recall 大幅提升
```

## 4. 评价指标排序

多跳缓存调参不要只看 `Recall@5`。推荐排序:

1. `candidate_recall_B`: graph 是否找到 gold B。
2. `topk_recall_B`: graph 是否能把 gold B 排到可写候选前面。
3. `bridge gold in KB`: writer 是否把 B 写进去并留住。
4. `writes`: 为此付出多少换入换出。
5. `reuse_hit_rate` / `cold_fetches_per_query`: 预取是否真的减少未来冷取。
6. `Recall@5`: 最终 retrieval 是否能从 hot tier 中把证据拿出来。

其中 1-2 是 Graph/PPR 质量, 3-5 是 cache management 质量, 6 是端到端检索表现。

## 5. 建议的近期执行命令

先修 `run_ppr_2wiki.py` embedding 对齐后, 跑一个最小矩阵:

```bash
for d in 10 20 30 50; do
  /home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run_ppr_2wiki.py \
    --dataset 2wikimultihopqa \
    --n-source 1500 \
    --d-cap $d \
    --R 1 \
    --L 2 \
    --K0 5 \
    --c 0.5
done
```

若 `R=1` 有稳定正收益, 再对最好的 `d_cap` 跑:

```bash
R in {1,2}, L in {2,3}, c in {0.3,0.5,0.7}
```

不要同时扫所有参数。每轮只回答一个问题:

```text
d_cap: 过滤 hub 是否过强/过弱?
R/L: 多跳传播是否真的有用?
c: dense seed 与 graph propagation 的平衡点在哪?
K0: first-hop seed 是否漏召?
```

## 6. 当前代码应如何落地

短期不要把 `PPRDRIPCore` 留在 `algorithms/drip/tests/` 里作为正式方法。应拆成:

```text
algorithms/drip/cache_manager/local_ppr.py       LocalPPRBridgeEvidence
algorithms/drip/cache_manager/graph_index.py     保留 stock GraphIndex, 或作为 fallback
algorithms/drip/cache_manager/__init__.py         通过 config 选择 graph_backend={"stock","ppr"}
```

这样实验可以同时跑:

```text
DRIP-stock
DRIP-PPR
DRIP-PPR+bucketed-writer
```

否则现在 monkey-patch `gi.graph_evidence = ppr_graph_evidence` 的方式很适合快速验证,
但不适合继续做系统性调参和论文复现。

---

# 第五阶段: Mo2 全量复跑 + baseline 对比 (结论: PPR 有效, 但还不够)

脚本: `experiments/hidden/run.py`
策略: `LRU / ARC / DRIP-Dense / DRIP / DRIP_PPR / Oracle`
数据: `2wikimultihopqa --expanded --q-type bridge_comparison --n-source 1500`
流: `temporal_bridge_reuse`, 50 windows x 50 queries = 2500 stream queries。
KB=300, pool=8404。输出:
`experiments/hidden/data/mo2_full_2wiki_bridge_ppr_baselines.json`。

本轮同时做了两个代码修复/接入:

1. `run_ppr_2wiki.py` 修正 query embedding 对齐:

```python
qe = np.array([query_embs[q["_qidx"]] for q in win], dtype="f")
```

2. 把 PPR 从测试 monkey-patch 拆成正式策略:

```text
algorithms/drip/cache_manager/local_ppr.py
registry key: DRIP_PPR
```

## 全量结果

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes | MaintR |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 5.4 | 5.4 | 3.818 | 3.3 | 2042 | 2113 |
| ARC | 4.2 | **9.4** | **3.648** | **17.1** | **842** | 2253 |
| DRIP-Dense | **5.9** | 6.0 | 3.802 | 4.5 | 2025 | 10430 |
| DRIP-stock | 4.5 | 6.0 | 3.793 | 6.5 | 4990 | 19960 |
| DRIP-PPR | 5.5 | 6.9 | 3.778 | 7.8 | 4275 | 19728 |
| Oracle | 52.6 | 100.0 | 0.0 | 100.0 | 10165 | 0 |

## 和 stock DRIP 的差异

PPR 相对 stock DRIP:

```text
R@5 H2:       4.5 -> 5.5   (+1.0pp)
KB Cov H2:    6.0 -> 6.9   (+0.9pp)
Reuse:        6.5 -> 7.8   (+1.3pp)
Writes:      4990 -> 4275  (-14.3%)
ColdQ:       3.793 -> 3.778 (小幅改善)
```

更关键的是写入质量:

```text
DRIP-stock:
  write_gold = 459 / 4990
  avg write_gold_rate = 9.2%

DRIP-PPR:
  write_gold = 2071 / 4275
  avg write_gold_rate = 46.0%
```

这说明 PPR 的主要收益不是简单提高候选数量, 而是显著提高 bridge 写入精度:

```text
PPR 让 writer 更常写到真正的 gold bridge docs,
但这些 gold docs 还没有稳定转化为 KB residency / reuse / R@5 的大幅提升。
```

## 和 baseline 的关系

结论必须谨慎:

- DRIP-PPR **优于 DRIP-stock**: graph evidence 来源是有效的。
- DRIP-PPR **R@5 H2 低于 DRIP-Dense**: 5.5 vs 5.9, 还不能说端到端 retrieval 赢。
- DRIP-PPR **reuse / cold-cost 明显低于 ARC**: ARC 的 reuse=17.1, writes=842, 说明 ARC 的
  conservative priority 对 cache stability 仍然更强。
- DRIP-PPR 的亮点是 **write precision**: 46.0% gold writes, 远高于 stock DRIP。

所以当前故事不能写成:

```text
PPR solves bridge retrieval.
```

更准确的是:

```text
Local PPR fixes the bridge candidate/admission signal: it writes many more true
bridge documents with fewer total writes. The remaining bottleneck is retention
and reuse conversion, not candidate discovery.
```

## 当前推荐公式 (候选, 不是最终版)

PPR 版 bridge evidence:

```text
seed(q) = TopK0 dense first-hop documents

V_q = BFS(seed(q), radius R, entity degree <= d_cap)

w(A,B) = sum_{e in Ent(A) cap Ent(B)}
         normalized_IDF(e) / deg(e)^rho

W = row_normalize(w)

pi^(0) = s(q)
pi^(l) = c * s(q) + (1-c) * W^T pi^(l-1),  l=1..L

E_graph(q,B) = pi^(L)(B)
```

Then DRIP still uses the same demand/admission shell:

```text
D_t(B) <- lambda * D_{t-1}(B) + eta_brg * E_graph(q,B)

admit c replacing v iff D_t(c) > gain_margin * P_t(v)
```

但全量结果说明 admission shell 还不够。下一步不能继续只调 PPR 参数,
而应该实现 `LOCAL_PPR_SUPPORT_FLOW.md` 的 writer 组件:

```text
D_dir / D_brg
S_dir / S_brg
route-aware bridge budget
retention-aware priority
```

也就是说:

```text
Graph/PPR 已经把 gold B 写进来了;
现在要让 B 留得住, 并在 reuse query 到来时仍在 hot tier。
```

---

# 第六阶段: 简洁 writer 修正 (结论: 有效, 可作为下一版主线)

实现:

```text
algorithms/drip/cache_manager/local_ppr.py
registry:
  DRIP_PPR_WRITER
  DRIP_PPR_WRITER_AGGR
  DRIP_PPR_WRITER_STICKY
```

目标不是再改 PPR, 而是验证第五阶段的判断:

```text
PPR 已能找出更高质量 bridge candidates;
剩余问题是 writer 是否给 bridge 候选预算并保护刚写入的 bridge docs。
```

## 简洁公式

保留 PPR evidence:

```text
E_G(q,B) = pi_q^(L)(B)
```

需求更新不变:

```text
D_t(d) = lambda D_{t-1}(d) + E_r(q_t,d)
```

只给 writer 加两条简单规则:

```text
budget_brg = ceil(f_brg * budget)
budget_dir = budget - budget_brg
```

和 bridge residency credit:

```text
P_t(d) = S_t(d) + D_t(d) + kappa B_t(d) - Red(d)
```

其中 `B_t(d)` 只给由 bridge bucket 写入的 resident 文档, 并随时间衰减。Admission 仍然是:

```text
admit c replacing v iff D_t(c) > m_r * P_t(v)
```

bridge bucket 使用较小 margin `m_brg`, direct bucket 使用原 margin。

这保持了公式的主干:

```text
PPR finds bridge evidence.
Route-aware writer decides whether the evidence is worth a cache replacement.
```

## 结果

同一 Mo2 设置:

```text
2wikimultihopqa expanded, bridge_comparison, n_source=1500
temporal_bridge_reuse, 50x50, KB=300
```

输出:

```text
experiments/hidden/data/mo2_full_2wiki_ppr_writer_baselines.json
experiments/hidden/data/mo2_full_2wiki_ppr_writer_variants.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 4.2 | **9.4** | **3.648** | **17.1** | **842** |
| DRIP-Dense | 5.9 | 6.0 | 3.802 | 4.5 | 2025 |
| DRIP-PPR | 5.5 | 6.9 | 3.778 | 7.8 | 4275 |
| DRIP-PPR+Writer | **6.2** | 7.7 | 3.751 | 7.2 | 1185 |
| DRIP-PPR+Writer aggressive | **6.2** | 7.9 | 3.756 | 7.4 | 1329 |
| DRIP-PPR+Writer sticky | 5.8 | 7.1 | 3.757 | 6.5 | 931 |

## 解读

- `DRIP-PPR+Writer` 是目前最好的 DRIP 变体: R@5 H2 从 PPR 的 5.5 提到 6.2,
  超过 DRIP-Dense 的 5.9, 同时 writes 从 4275 降到 1185。
- aggressive 版只把 coverage/reuse 略微提高, R@5 不变, writes 增加。说明继续降
  bridge admission margin 有边际收益, 但不是主要瓶颈。
- sticky 版 writes 最低, 但 R@5 下降。说明过度保护 bridge docs 会牺牲当前检索可见性,
  不宜把 `kappa` 设太大。
- ARC 仍然在 reuse/cold-cost 上更强。我们的优势现在是:

```text
DRIP-Dense-like retrieval quality + much lower writes + route-aware bridge precision.
```

而不是全面击败 ARC。

## 当前建议

保留主公式:

```text
E_r(q,d) = E_D(q,d) or E_G(q,d)
D_t(d) = lambda D_{t-1}(d) + E_r(q_t,d)
P_t(d) = S_t(d) + D_t(d) + kappa B_t(d) - Red(d)
admit by route-aware budgets
```

论文/汇报里不要展示 aggressive/sticky 变体, 它们只作为消融。主方法用
`DRIP-PPR+Writer` 的默认设置:

```text
f_brg = 0.5
m_brg = 0.75
kappa = 1.0
```

下一步如果继续调, 只扫:

```text
f_brg in {0.4, 0.5, 0.6}
m_brg in {0.65, 0.75, 0.85}
kappa fixed at 1.0
```

不要再同时改 PPR 的 `c/L/R/K0/d_cap`, 否则解释会重新变浑。

---

# 第七阶段: ARC-style DRF 保留修正 (结论: 当前最强主线)

动机来自前一阶段的反例:

```text
PPR 已能把 gold bridge docs 写进来,
但 resident bridge doc 在后续 reuse query 里拿不到 graph evidence 续命。
```

也就是说, 之前的 writer 只解决了 admission, 没解决 retention。ARC 虽然没有多跳图,
但它的 DRF + hubness 是很强的 cache stability prior。新的修正不是再加复杂分支,
而是把 ARC 最有用的稳定项接到 PPR bridge evidence 后面。

## 参考方法

- ARC / Agent RAG Cache: DRF + hubness 解释了为什么 ARC 在 reuse/cold-cost 上强。
- HippoRAG / HippoRAG2: PPR over knowledge graph 解释了为什么 bridge evidence 应来自图传播。
- LightRAG / GraphRAG: entity-relation graph 说明离线实体/关系抽取可作为下一步增强。
- classic ARC / TinyLFU: 频率保留比纯 recency 更适合重复访问。

## 关键代码修正

实现:

```text
algorithms/drip/cache_manager/local_ppr.py
  PPRBridgeDRFDRIPCore

algorithms/cache/registry.py
  DRIP_PPR_DRF
  DRIP_PPR_DRF_AGGR
```

最关键的修正:

```text
PPR evidence now credits resident docs too.
```

之前 PPR 调用会排除 `kb_pos`, 所以已经写进 KB 的 hidden B 在后续 bridge query
里不会再收到图证据。新变体只在 DRF 版本中允许 PPR 返回 resident docs:

```text
candidates, stats = ppr.evidence(first_hops, kb_pos=None)
```

这让 resident B 可以通过图证据续命, 而不仅靠 dense query hit。

## 简洁公式

主公式保持三行:

```text
E_G(q,d) = PPR_q(d)

F_t(d) = rho F_{t-1}(d) + E_G(q_t,d)

P_t(d) = D_t(d) + mu log(1 + F_t(d)) + beta log(1 + H_t(d)) - Red(d)
```

admission 仍然是:

```text
admit c replacing v iff D_t(c) > m_r P_t(v)
```

其中:

```text
F_t(d)  = DRF-like frequency / bridge evidence ledger
H_t(d)  = local hubness in current KB
Red(d)  = redundancy penalty
```

这比第六阶段更适合作为论文公式, 因为它和 ARC 的结构接近:

```text
ARC:  priority = DRF + hubness
ours: priority = PPR-bridge demand + DRF + hubness
```

差异只在 evidence 来源:

```text
ARC evidence comes from dense retrieval history.
DRIP-PPR-DRF evidence also comes from hidden bridge graph paths.
```

## 完整 Mo2 结果

设置不变:

```text
2wikimultihopqa expanded, bridge_comparison, n_source=1500
temporal_bridge_reuse, 50x50, KB=300
```

输出:

```text
experiments/hidden/data/mo2_full_2wiki_ppr_drf_final_baselines.json
experiments/hidden/data/mo2_full_2wiki_ppr_drf_variants_v2.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| LRU | 5.3 | 5.4 | 3.818 | 3.1 | 2045 |
| ARC | 4.2 | **9.4** | **3.648** | **17.1** | 842 |
| DRIP-Dense | 5.9 | 6.0 | 3.802 | 4.5 | 2025 |
| DRIP-PPR+Writer | 6.2 | 7.7 | 3.751 | 7.2 | 1185 |
| DRIP-PPR-DRF | 6.9 | 8.5 | 3.736 | 8.9 | 708 |
| DRIP-PPR-DRF aggressive | **7.3** | 9.0 | 3.722 | 9.6 | **686** |

对比 ARC:

```text
R@5 H2:    4.2 -> 7.3  (+3.1pp, +74% relative)
Writes:    842 -> 686  (-18.5%)
KB Cov H2: 9.4 -> 9.0  (-0.4pp, nearly tied)
Reuse:    17.1 -> 9.6  (still weaker)
ColdQ:   3.648 -> 3.722 (still weaker)
```

对比 DRIP-Dense:

```text
R@5 H2:    5.9 -> 7.3  (+1.4pp)
KB Cov H2: 6.0 -> 9.0  (+3.0pp)
Reuse:     4.5 -> 9.6  (+5.1pp)
Writes:   2025 -> 686  (-66%)
```

## 失败轮次

第一版 DRF 失败:

```text
DRIP_PPR_DRF: R@5 H2=1.3, writes=50
```

根因:

```text
DRF serve gate 太宽, 低相似 resident 也被保护, cache 被冻住。
```

修正:

```text
DRF serve credit only for true cache hits:
max_sim(q, KB) >= SF_HIT_THRESH
```

同时 resident bridge docs 通过 `E_G` 而不是低阈值 dense similarity 续命。

过度 aggressive 也失败:

```text
reserve=0.75, margin=0.45:
R@5 H2=6.9, Cov H2=8.5, writes=1128
```

说明继续多写会把可检索 direct docs 挤掉, 不是主路。

## 当前主张

现在可以更强地说:

```text
On the bridge-reuse workload, DRIP-PPR-DRF substantially improves retrieval
quality over ARC and DRIP-Dense while using fewer writes.
```

但还不能说:

```text
DRIP-PPR-DRF dominates ARC on every cache metric.
```

ARC 仍然赢 reuse/cold-cost, 因为它的 full-pool miss-driven DRF maintenance cost
很低, 且非常保守。我们的优势是:

```text
specialized bridge evidence -> much higher R@5 at lower write cost
```

不是:

```text
lower maintenance retrieval cost
```

## LLM entity extraction 是否现在上

暂时不要放进这轮主结果。原因:

```text
当前最大增益来自 retention 修正, 不是 entity recall。
```

LLM entity extraction 应作为下一组实验:

```text
spaCy entities vs LLM canonical entities
same PPR-DRF writer
same 50x50 full Mo2 run
```

如果 LLM 版要进论文, 必须离线缓存实体结果, 并单独报告抽取成本。不要把它和
DRF writer 同时作为一个未拆开的新方法, 否则审稿人会问增益到底来自图质量还是缓存策略。

## 当前推荐参数

主方法:

```text
DRIP_PPR_DRF_AGGR

PPR: c=0.5, L=3, R=2, K0=5, d_cap=30
writer: bridge_reserve=0.6, bridge_margin=0.65
retention: drf_weight=0.25, hub_weight=0.05, drf_decay=0.98
```

保守消融:

```text
DRIP_PPR_DRF

bridge_reserve=0.5, bridge_margin=0.75
drf_weight=0.15, hub_weight=0.05, drf_decay=0.98
```

下一步只做两个方向:

1. 用 LLM/typed entity extraction 替换 spaCy, 只验证 graph evidence 质量。
2. 降低 DRIP 的 maintenance retrieval accounting, 否则 ARC 会一直在 MaintR 上天然占优。

---

# 第八阶段: mask / 大 KB / coverage 口径澄清

问题: `temporal_bridge_reuse` 之前无条件把 stream 里的 support docs 从初始 KB 里删掉。
这适合 causal prefetch benchmark:

```text
不给未来答案泄漏, 看算法能否从 exposure query 预取 reuse support doc。
```

但它不适合 capacity sanity check:

```text
如果 KB 容量已经大于全部 stream support docs, coverage 是否能接近上界?
```

因此 `experiments/hidden/run.py` 新增两个开关:

```text
--no-mask-stream-gold   不从初始 KB 移除 stream support docs
--init-stream-gold      oracle-init: 先把 stream support docs 放入 KB, 再补满 slack
```

## 8.1 KB=4500 + no-mask, 但仍用 head-biased init

命令:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa --expanded --q-type bridge_comparison \
  --n-source 1500 --kb-budget 4500 \
  --strategies LRU ARC DRIP-Dense DRIP_PPR_DRF_AGGR Oracle \
  --workload temporal_bridge_reuse \
  --no-mask-stream-gold \
  --output mo2_full_2wiki_ppr_drf_kb4500_nomask.json
```

诊断:

```text
pool=8404
stream queries=2500
unique stream gold docs=4008
KB=4500
head-biased init resident gold=2726/4008
```

所以即使 KB 容量大于 unique gold, 如果初始化仍按 centroid 选文档,
它也不会自动包含所有 gold docs。

结果:

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| LRU | 31.6 | 72.0 | 1.108 | 83.5 | 493 |
| ARC | 38.3 | 75.7 | 1.028 | 83.3 | 1798 |
| DRIP-Dense | 32.5 | 72.7 | 1.092 | 84.3 | 794 |
| DRIP-PPR-DRF aggressive | **43.9** | **79.4** | **0.946** | 83.6 | 2310 |
| Oracle | 51.0 | 100.0 | 0.0 | 100.0 | 15790 |

结论:

```text
低 coverage 的主因不是 PPR 找不到, 而是 benchmark 之前强制 mask,
以及 head-biased init 并不等价于 gold-init。
```

在 no-mask 大 KB 设定下, DRIP-PPR-DRF 在 R@5 / coverage / cold cost 上超过
ARC 和 DRIP-Dense, 但 writes 更高。

## 8.2 KB=4500 + stream-gold oracle init

命令:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa --expanded --q-type bridge_comparison \
  --n-source 1500 --kb-budget 4500 \
  --strategies LRU ARC DRIP-Dense DRIP_PPR_DRF_AGGR Oracle \
  --workload temporal_bridge_reuse \
  --init-stream-gold \
  --output mo2_full_2wiki_ppr_drf_kb4500_goldinit.json
```

初始化:

```text
Stream-gold init: gold=4008 resident_gold=4008 KB=4500/4500
```

结果:

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| LRU | **51.4** | **99.9** | **0.00** | **100.0** | 9 |
| ARC | 43.5 | 84.5 | 0.50 | 90.9 | 1345 |
| DRIP-Dense | 50.8 | 98.3 | 0.04 | 97.8 | 132 |
| DRIP-PPR-DRF aggressive | 45.3 | 81.9 | 0.51 | 85.2 | 1789 |
| Oracle | 51.0 | 100.0 | 0.0 | 100.0 | 15790 |

结论:

```text
如果初始 KB 已经包含全部 stream gold, coverage 确实接近 100%。
```

这验证了你的直觉。但这个设定不是正式方法对比, 因为它把答案直接放进初始 KB。
它的用途是 sanity check:

```text
容量足够时, evaluator 和 retrieval 上界正常;
coverage 低不是指标 bug, 而是 init / mask / eviction 定义导致。
```

同时它暴露一个新事实:

```text
当 KB 已接近 oracle 时, aggressive writer 会破坏已有 gold residency。
```

所以正式 benchmark 应分三类报告:

1. `masked causal`: 看预取能力, 不能要求初始 coverage 高。
2. `no-mask head-biased`: 看真实大 KB 初始化下的自适应能力。
3. `stream-gold init`: 只作容量/evaluator 上界 sanity check, 不作为算法胜负。

---

# 第九阶段: head-biased init + PPR retrieval 正式复跑

判断修正:

```text
正式对比不应使用 --init-stream-gold。
```

原因:

```text
初始 KB 应模拟 drift 前的 head-biased state,
而不是由 evaluation stream 的未来 support docs 直接构造。
```

`--init-stream-gold` 只保留为 capacity/evaluator sanity check。

本轮还修了一个 runner bug:

```text
--retrieval graph 不应自动触发 LLM decomposition。
```

之前 `retrieval == graph` 时即使没有 `--llm-expand`, 也会先调用
`batch_decompose()`, 导致网络/代理失败。现在只有显式传 `--llm-expand`
才做 LLM sub-question decomposition。

## 设置

```text
2wikimultihopqa expanded
q_type=bridge_comparison
n_source=1500
workload=temporal_bridge_reuse
KB=4500
init=head-biased, no stream-gold oracle init
retrieval=graph/PPR
strategies: LRU / ARC / DRIP-Dense / DRIP_PPR_DRF_AGGR / Oracle
```

命令:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa --expanded --q-type bridge_comparison \
  --n-source 1500 --kb-budget 4500 \
  --strategies LRU ARC DRIP-Dense DRIP_PPR_DRF_AGGR Oracle \
  --workload temporal_bridge_reuse \
  --no-mask-stream-gold \
  --retrieval graph \
  --output mo2_full_2wiki_ppr_drf_kb4500_nomask_graphret.json
```

输出:

```text
experiments/hidden/data/mo2_full_2wiki_ppr_drf_kb4500_nomask_graphret.json
```

## 结果

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes | MaintR |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 32.6 | 72.4 | 1.09 | **86.1** | 497 | 769 |
| ARC | 39.2 | 75.7 | 1.03 | 83.3 | 1798 | 789 |
| DRIP-Dense | 33.3 | 72.7 | 1.09 | 84.3 | 794 | 3865 |
| DRIP-PPR-DRF aggressive | **44.3** | **79.4** | **0.95** | 83.6 | 2310 | 15632 |

对比 ARC:

```text
R@5 H2:    39.2 -> 44.3 (+5.1pp)
KB Cov H2: 75.7 -> 79.4 (+3.7pp)
ColdQ:     1.03 -> 0.95
```

对比 DRIP-Dense:

```text
R@5 H2:    33.3 -> 44.3 (+11.0pp)
KB Cov H2: 72.7 -> 79.4 (+6.7pp)
ColdQ:     1.09 -> 0.95
```

和 dense retrieval 的 no-mask 大 KB 结果相比:

```text
DRIP R@5 H2: 43.9(dense) -> 44.3(graph/PPR)
ARC  R@5 H2: 38.3(dense) -> 39.2(graph/PPR)
```

因此 PPR retrieval 本身只小幅抬高最终 R@5。主要收益仍来自:

```text
PPR evidence + DRF/hubness writer 让 KB 里 resident gold 更多。
```

不是:

```text
换成 graph retriever 后所有方法都大幅变强。
```

## 当前简洁公式

### 1. Head-biased 初始化

不用 evaluation stream 的未来 gold docs:

```text
a_head(d) = max_{c in C_head} cos(e_d, c)
K_0 = TopB_d a_head(d)
```

如果有明确 pre-drift head context, 可先放入 head docs, 再按 `a_head` 补满。

### 2. Bridge evidence: local PPR

```text
seed(q) = TopK dense(q,d)
V_q = BFS(seed(q), R, deg(entity) <= d_cap)

w(i,j) = sum_{e in Ent(i) cap Ent(j)}
         IDF(e) / deg(e)^rho

pi_q = c s_q + (1-c) W^T pi_q
E_G(q,d) = pi_q(d)
```

### 3. Demand + bridge-aware retention

```text
D_t(d) = lambda D_{t-1}(d) + E_r(q_t,d)

P_t(d) = D_t(d) + mu R_t(d)
```

其中:

```text
R_t(d): decayed reuse credit, counted from true cache hits and bridge-PPR evidence
```

`H_t(d)` 和 `Red(d)` 不进入主公式:

```text
H_t(d): optional system stabilizer, appendix only
Red(d): implementation hygiene / duplicate suppression, appendix only
```

主贡献只表述为:

```text
multi-hop bridge demand E_G + route-aware admission
```

### 4. Route-aware admission

```text
B_brg = ceil(f_brg B)
B_dir = B - B_brg

admit c replacing v iff D_t(c) > m_route P_t(v)
```

当前主线参数已经被第十一阶段的 `DRIP_BRIDGE_ECHO` 替代。这里保留的是
第九/十阶段的 R/H 消融上下文:

```text
PPR: c=0.5, L=3, R=2, K0=5, d_cap=30
writer: f_brg=0.6, m_brg=0.65
retention: mu=0.25, decay=0.98
```

一句话版本:

```text
Use local PPR to estimate hidden bridge demand, then use route-aware admission
and lightweight reuse credit to keep bridge evidence resident.
```

## 仍需注意

当前 `temporal_bridge_reuse` workload 把所有 evaluation queries 标为 tail:

```text
head=0, tail=2500
```

所以本轮的 head-biased init 主要来自 head cluster centroid, 而不是 stream 内 H1
head context。若要更严格模拟 “drift 前已经服务过 head queries”, 应新增一个 workload:

```text
H1: real head queries build initial/resident KB
H2: bridge reuse tail queries evaluate adaptation
```

这会比 `stream-gold init` 更合理, 也比全 mask 更接近真实系统。

---

# 第十阶段: 去掉 R/H 的消融

问题:

```text
R_t(d) reuse credit 和 H_t(d) hubness 是否只是借 ARC 的壳?
不用它们会怎样?
```

本轮在同一正式设置下跑完整 50x50:

```text
2wikimultihopqa expanded
q_type=bridge_comparison
n_source=1500
workload=temporal_bridge_reuse
KB=4500
init=head-biased, no stream-gold oracle init
retrieval=graph/PPR
```

输出:

```text
experiments/hidden/data/mo2_full_2wiki_ppr_drf_ablate_rh_graphret.json
```

## 变体定义

```text
DRIP_PPR_DRF_AGGR   R on,  H on
DRIP_PPR_DRF_NOH    R on,  H off
DRIP_PPR_DRF_NOR    R off, H on
DRIP_PPR_DRF_NORH   R off, H off
DRIP_PPR_ROUTE_ONLY R off, H off, and no resident bridge stickiness
```

其中:

```text
R: decayed reuse / bridge-PPR retention credit
H: local embedding hubness
```

## 结果

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 39.2 | 75.7 | 1.028 | 83.3 | 1798 |
| DRIP-Dense | 33.3 | 72.7 | 1.092 | 84.3 | 794 |
| DRIP-PPR full (R+H) | 44.3 | **79.4** | **0.946** | 83.6 | 2310 |
| DRIP-PPR no-H | **44.8** | 72.4 | 1.138 | 66.1 | 3342 |
| DRIP-PPR no-R | 40.8 | 78.5 | 0.963 | **86.0** | 3104 |
| DRIP-PPR no-R/H | 42.3 | 72.5 | 1.132 | 68.9 | 3592 |
| DRIP-PPR route-only | 41.1 | 68.0 | 1.239 | 61.2 | 3686 |

## 解读

1. **不用 R/H 仍然有效。**

`DRIP-PPR no-R/H` 和 `route-only` 在 R@5 H2 上仍超过 ARC/DRIP-Dense:

```text
no-R/H:     42.3 > ARC 39.2 > DRIP-Dense 33.3
route-only: 41.1 > ARC 39.2 > DRIP-Dense 33.3
```

这说明核心收益不是 ARC-like DRF/hubness, 而是:

```text
local PPR bridge evidence + route-aware bridge admission
```

2. **H 不应作为主公式。**

去掉 H 后 R@5 反而最高:

```text
full 44.3 -> no-H 44.8
```

但 no-H 的 coverage / cold / reuse 明显变差:

```text
KB Cov: 79.4 -> 72.4
ColdQ:  0.946 -> 1.138
Reuse:  83.6 -> 66.1
Writes: 2310 -> 3342
```

所以 H 更像系统 regularizer: 帮助稳定 residency 和降低冷取, 但不是 retrieval
quality 的主要来源。论文主公式里最好不写 H, 或只放 appendix/消融。

3. **R 有实际作用, 但也不是核心发现。**

去掉 R 后 R@5 明显下降:

```text
full 44.3 -> no-R 40.8
```

但 coverage / cold 仍接近 full:

```text
KB Cov: 79.4 -> 78.5
ColdQ:  0.946 -> 0.963
```

因此 R 是有用的 retention term, 尤其帮助 PPR 证据跨窗口留存;
但它应该被写成 "lightweight reuse credit", 不要写成 ARC-style DRF 主贡献。

## 推荐讲法

主方法公式保留:

```text
E_G(q,d) = local-PPR bridge evidence
D_t(d) = lambda D_{t-1}(d) + E_r(q_t,d)
B_brg = ceil(f_brg B)
admit c replacing v iff D_t(c) > m_route P_t(v)
```

主公式中的 resident priority 最简化:

```text
P_t(d) = D_t(d) + mu R_t(d)
```

并明确:

```text
H_t(d) is not part of the core method.
Red(d) is not part of the core method.
```

第十一阶段后不再建议把 `DRIP-PPR no-H` 当主方法, 因为名字和实现仍然带
DRF/R/H 消融痕迹。主线改成 `DRIP_BRIDGE_ECHO`: 不用 hubness, 不用 DRF,
不把 Red(d) 写入 priority。

---

# 第十一阶段: 清理主线为 BridgeEcho (不用 DRF / hubness / Red)

动机:

```text
hubness 和 DRF 都太像 ARC / AgentRAGCache;
Red(d) 消融显示几乎无收益;
所以主方法必须只保留 multi-hop 特有的 bridge signal。
```

代码:

```text
algorithms/drip/cache_manager/local_ppr.py
  PPRBridgeEchoDRIPCore

algorithms/cache/registry.py
  DRIP_BRIDGE_ECHO
```

临时参数矩阵跑完后已删除临时 registry keys, 只保留主线 `DRIP_BRIDGE_ECHO`。

## 简洁公式

### 1. Bridge evidence

仍然使用 local PPR:

```text
seed(q) = TopK0 dense(q,d)
V_q = BFS(seed(q), R, deg(entity) <= d_cap)

w(i,j) = sum_{e in Ent(i) cap Ent(j)}
         IDF(e) / deg(e)^rho

pi_q = c s_q + (1-c) W^T pi_q
E_G(q,d) = pi_q(d)
```

### 2. Demand

```text
D_t(d) = lambda D_{t-1}(d) + E_r(q_t,d)
```

其中 bridge route 用:

```text
E_r(q,d) = E_G(q,d)
```

direct route 用 dense evidence。

### 3. BridgeEcho residency

新增自有项叫 `A_t(d)` (bridge echo), 不再叫 DRF/R/H:

```text
A_t(d) = xi A_{t-1}(d) + Echo_t(d)
```

代码里的 `Echo_t(d)` 只来自两个来源:

```text
1. d 出现在 local-PPR bridge candidates:
   Echo_t(d) += E_G(q,d) / rank_q(d)

2. d 已经有 bridge echo 或由 bridge bucket 写入, 且被真实 query 命中:
   Echo_t(d) += cos(q,d) / rank_q(d)
```

明确不做:

```text
no dense-candidate frequency credit
no ARC-style miss-driven DRF
no local hubness H_t(d)
no redundancy penalty Red(d) in resident priority
```

resident priority:

```text
P_t(d) = S_t(d) + D_t(d) + mu log(1 + A_t(d))
```

其中 `S_t(d)` 是 resident doc 的真实服务命中 credit。它不是 ARC 的 DRF:

```text
S_t(d) 只对已经在 KB 里的文档、且实际命中当前 query 的文档加分;
不扫描 full pool, 不对 miss candidates 做全局频率统计。
```

### 4. Route-aware admission

```text
B_brg = ceil(f_brg B)
B_dir = B - B_brg

B_eff = ceil(tau_w B)

admit c replacing v iff D_t(c) > m_route P_t(v)
```

最终主线参数:

```text
PPR:    c=0.5, L=3, R=2, K0=5, d_cap=30
writer: f_brg=0.5, m_brg=1.0, m_dir=1.5
echo:   mu=0.75, xi=0.99
write cap: tau_w=0.45
```

一句话版本:

```text
Local PPR discovers hidden bridge evidence; BridgeEcho keeps documents that
repeatedly reappear as bridge evidence, without importing ARC's hubness/DRF prior.
```

## 完整复跑

设置:

```text
2wikimultihopqa expanded
q_type=bridge_comparison
n_source=1500
workload=temporal_bridge_reuse
stream=50x50
KB=4500
init=head-biased, no stream-gold mask
retrieval=graph/PPR
```

输出:

```text
experiments/hidden/data/mo2_full_2wiki_bridge_echo_budget045_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 39.2 | 75.7 | **1.028** | **83.3** | 1798 |
| DRIP-Dense | 33.3 | 72.7 | 1.092 | 84.3 | 794 |
| DRIP-BridgeEcho | **39.5** | **75.8** | 1.040 | 80.9 | **1651** |
| Oracle | 51.7 | 100.0 | 0.000 | 100.0 | 15790 |

BridgeEcho 写入质量:

```text
write_gold = 814 / 1651 = 49.3%
```

## 结论

BridgeEcho 证明 “自己的、非 ARC-like” 主线在 cache-management 指标上也成立:

```text
R@5 H2:     39.5 > ARC 39.2 > DRIP-Dense 33.3
KB Cov H2:  75.8 > ARC 75.7 > DRIP-Dense 72.7
Writes:     1651 < ARC 1798
```

差距很小, 不能夸大成大幅胜出; 但方向终于对了。此前 coverage 低的根因不是
PPR 找不到 support docs, 而是 writer 过度换入导致 resident gold 被冲掉:

```text
no write cap:
  R@5=42.1, Cov=72.1, Writes=3989

serve fix only:
  R@5=42.2, Cov=72.5, Writes=3494

write cap tau_w=0.5:
  R@5=40.0, Cov=75.5, Writes=1803

write cap tau_w=0.45:
  R@5=39.5, Cov=75.8, Writes=1651

write cap tau_w=0.4:
  R@5=38.9, Cov=76.1, Writes=1479
```

解释:

```text
PPR + BridgeEcho 能写到真正 bridge docs;
但 cache management 需要限制每窗换入量, 否则正确写入也会制造 churn。
```

这不是回到 DRF/hubness 的理由。下一步若继续提升, 应把 `tau_w` 从手调常数改成
multi-hop 自有的低 churn gate, 例如:

```text
BridgeConsensus(d) = same window 内多个 query / 多条 PPR path 同时支持 d
只 admit consensus bridge docs, 而不是所有 positive PPR bridge docs。
```

也就是说, 下一步改 candidate admission gate, 不要再把 ARC 的 hubness/DRF 放回主公式。

---

# ARC 对标容量校准 (结论: 不能用大 KB 当主对比)

用户提醒是对的: 如果把 `Cache Mechanism for Agent RAG Systems` 里的 ARC 作为 baseline,
cache 容量不能设得太大。该论文默认 cache capacity 是 3.0MB, 并做 1MB 到 5MB
容量消融。以当前 bge-large 1024d float32 embedding 粗略换算:

```text
KB=300   ≈ 1.2MB
KB=750   ≈ 3.0MB   # ARC 默认容量级别
KB=1250  ≈ 5.0MB
KB=4500  ≈ 18MB    # 只能当大容量 sanity
KB=6250  ≈ 25MB    # 不能作为 ARC 主对标
```

因此后续主表应优先报告 `KB=300/750/1250`, 而不是 `KB=4500/6250`。

## ARC-aligned 小容量复跑

设置:

```text
2wikimultihopqa expanded
q_type=bridge_comparison
n_source=2000
workload=cluster_shift
stream=20x25
retrieval=graph/PPR
strategies=ARC, DRIP-Dense, DRIP_OVERFLOW
```

输出:

```text
experiments/hidden/data/cap_2wiki_kb300_20w25_graphret.json
experiments/hidden/data/cap_2wiki_kb750_20w25_graphret.json
experiments/hidden/data/small_2wiki_overflow_v2_20w25_graphret.json
```

| KB | Approx MB | Strategy | R@5 H2 | KB Cov H2 | ColdQ | Writes |
|---:|---:|---|---:|---:|---:|---:|
| 300 | 1.2 | ARC | 7.8 | 10.2 | 3.49 | 696 |
| 300 | 1.2 | DRIP-Dense | 14.1 | 15.0 | 3.16 | 614 |
| 300 | 1.2 | DRIP_OVERFLOW | **14.3** | 14.9 | **3.15** | **610** |
| 750 | 3.0 | ARC | 17.3 | 23.6 | 2.77 | 1337 |
| 750 | 3.0 | DRIP-Dense | **26.2** | **32.0** | **2.12** | **663** |
| 750 | 3.0 | DRIP_OVERFLOW | **26.2** | **32.0** | **2.12** | **663** |
| 1250 | 5.0 | ARC | 23.3 | 35.5 | 2.22 | 1790 |
| 1250 | 5.0 | DRIP-Dense | **29.3** | 41.1 | 1.67 | **502** |
| 1250 | 5.0 | DRIP_OVERFLOW | 28.8 | **41.8** | **1.65** | 511 |

## 当前解释

在 ARC 论文对应的小容量区间, DRIP 的 cache management 是有效的:

```text
KB=750:
  R@5 H2:    17.3 -> 26.2  (+8.9pp vs ARC)
  KB Cov H2: 23.6 -> 32.0  (+8.4pp vs ARC)
  Writes:    1337 -> 663   (-50.4% vs ARC)
```

但这轮也暴露了一个更尖锐的问题:

```text
DRIP_OVERFLOW ≈ DRIP-Dense
```

也就是说, 小容量下真正赢 ARC 的主要来源仍是 direct semantic support-debt
写入准则, 不是 PPR/multihop overflow。当前 bridge overflow 的正确作用是
"不破坏 direct floor", 但还没有稳定提供额外收益。

下一步如果要证明多跳模块本身, 不能继续只靠加大 KB。更合理的方向是:

```text
small cache:     direct evidence dominates, bridge must stay conservative
medium cache:    only admit bridge docs with repeated support / consensus
large cache:     bridge can use overflow slots, but large-cap result不能拿去压 ARC
```

因此当前方法故事线应改成:

```text
1. ARC-sized cache 下, DRIP 的语义 support-debt admission 明显强于 ARC。
2. 多跳 bridge 不能抢 direct slot; 它只能使用 overflow / consensus slot。
3. 未来主创新应落在 bridge consensus admission, 而不是继续扩大 cache 容量。
```

---

# 第七阶段: full temporal 复核 + 数据构造诊断 (结论: 多跳模块还不能作为主胜点)

本轮目标是回答一个尖锐问题:

```text
如果不是继续堆 PPR/role/hubness, 是否能设计一个自己的多跳写入准则,
并在 ARC-sized cache 上稳定超过 ARC / embedding-only?
```

我试了两条路线:

1. `Route-Sized Lease`: 写入后给 support docs 短期驻留 lease, 防止高精度写入被下一窗 churn 掉。
2. `DRIP_DECOMP`: 把问题分解成 support-seeking sub-questions, 用子问题 embedding 给 hidden support 建 demand。

`Route-Sized Lease` 小规模无收益, 已从正式 registry 清掉, 只保留结果文件作诊断:

```text
small_2wiki_rsl_kb750_temporal_20w25_graphret.json
small_2wiki_rsl_v2_kb750_temporal_20w25_graphret.json
```

## 7.1 full temporal 结果

设置:

```text
2wikimultihopqa expanded
q_type=bridge_comparison
n_source=1500
workload=temporal_bridge_reuse
stream=50x50
retrieval=graph
KB=750   # 约等于 ARC 3MB 容量
```

输出:

```text
experiments/hidden/data/mo2_full_2wiki_decomp_heur_kb750_temporal_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| LRU | **13.0** | 13.3 | 3.604 | 6.0 | 1512 |
| ARC | 10.4 | **19.4** | **3.306** | **28.8** | 1857 |
| DRIP-Dense | 12.8 | 14.3 | 3.539 | 11.4 | 3647 |
| DRIP_DECOMP heuristic | 11.1 | 14.9 | 3.512 | 16.9 | 3958 |

解读:

- heuristic decomposition 能把 `Cov H2` 从 14.3 提到 14.9, `Reuse` 从 11.4 提到 16.9,
  说明 multi-hop sub-question 信号不是完全没用。
- 但它牺牲 R@5, 且 coverage/reuse 仍明显低于 ARC。
- 结论: `DRIP_DECOMP` 目前只能作为探索性消融, 不能作为最终主算法。

## 7.2 数据构造诊断

同一 full temporal stream 的结构:

```text
pool=8404, stream=2500
每个 query support docs = 4
support title 出现在 query 文本中: 5000 / 10000 = 50.0%
unique gold supports:
  all = 4008
  H1  = 3132
  H2  = 2827
KB = 750
每窗 unique gold p50/p90/max = 196 / 199 / 200
reuse groups = 632, avg size = 2.61
reuse_support_title 出现在 question 中: 224 / 1652 = 13.6%
```

这说明当前 `temporal_bridge_reuse` 不是一个干净的 "A resident, hidden B missing" benchmark:

- 每个 query 有 4 个 support docs, 不是简单二跳 A-B。
- H2 unique gold supports 有 2827 个, KB 只有 750, 所以 coverage 不可能接近 50%
  除非 workload 有更强重复或 oracle 式选择。
- 50% support titles 直接出现在 question, 因此 embedding-only 已经能解决很多 support。
- reuse group 平均只有 2.61 次, 多跳预取的 amortization 空间不大。
- ARC 在 reuse/cold 上强, 很大程度来自保守稳定性, 不是因为它理解多跳。

`anchored_bridge_reuse` 更像 "A 已在 KB, B 被 hold out", 但也有问题:

```text
small_2wiki_decomp_heur_kb750_anchored_20w25_graphret.json

ARC:            R@5 H2=22.9, Cov H2=31.3, Reuse=18.8
DRIP-Dense:  R@5 H2=30.5, Cov H2=35.5, Reuse=9.4
DRIP_DECOMP:    R@5 H2=28.4, Cov H2=32.7, Reuse=10.7
DRIP_CHAIN:     R@5 H2=29.3, Cov H2=33.9, Reuse=10.7
```

这里首窗 R@5 很高, 说明 anchor init 过强; 但 chain/decomp 仍不如 embedding-only,
说明当前 graph/decomp 候选噪声仍大。

## 7.3 当前最终算法建议

不要把 PPR / role rerank / hubness / DRF 写成最终主公式。当前能站住的主方法是:

```text
Route-Sized Support Debt Cache
```

核心公式保持短:

```text
r(q) -> k_q                     # query route decides required support slots
c_t(q) = max(0, k_q - hit_KB(q)) # support deficit

E(q,d) = dense(q,d)             # current robust path
          + alpha * decomp(q,d) # optional sub-question evidence, ablation only

D_t(d) = lambda D_{t-1}(d) + sum_q 1[c_t(q)>0] E(q,d)
P_t(d) = S_t(d) + D_t(d)

admit d over victim v iff D_t(d) > margin * P_t(v)
```

方法叙事:

```text
DRIP is not "PPR beats ARC".
DRIP is route-sized support admission:
the cache writes enough evidence slots for the query route, instead of treating
all misses as single-document misses.
```

目前可报告的较好结果仍是 ARC-aligned 小容量 `cluster_shift`, 不是 full temporal:

```text
KB=750, 20x25 cluster_shift:
ARC:           R@5 H2=17.3, Cov H2=23.6, Writes=1337
DRIP-Dense: R@5 H2=26.2, Cov H2=32.0, Writes=663
```

这说明 cache-management 主体有效; 但 multi-hop bridge module 还没稳定胜过
embedding-only。

## 7.4 下一步必须换的不是参数, 是 benchmark / signal

如果论文要主打 "special multi-hop + query drift", 需要构造更干净的 benchmark:

```text
1. 初始 KB 包含 A, 明确 hold out B。
2. B 不出现在 query text, 但可从 A 的 relation/entity evidence 推出。
3. 每个 B 至少有 5+ 次未来 reuse, 否则预取没有 amortization。
4. KB 容量保持 ARC-aligned: 300/750/1250。
5. 同时报告:
   candidate_recall_B
   topk_recall_B
   B write precision
   B residency half-life
   reuse hit rate
```

在这个 benchmark 没修好前, 继续调 PPR/lease/decomp 很可能只是把 retrieval 噪声写进 cache。

---

# 第八阶段: Router 修正假设复测 (结论: naive hybrid router 会过写)

用户提出的怀疑是对的: router 可能有问题。诊断发现:

```text
bridge_comparison 在 full temporal 中:
  SINGLE=0
  MULTI_DIRECT=0
  BRIDGE=2500/2500
```

也就是说, 现有 router 把所有 `bridge_comparison` 都当成纯 `BRIDGE`。由于每个
2Wiki bridge-comparison query 有 4 个 support docs, 这确实太粗:

```text
visible leg:  film/work support A, 可由 query embedding 直接找到
hidden leg:   person/attribute support B, 需要 decomposed / graph / bridge signal
```

因此我试了两种 router-credit 修法。

## 8.1 Aggressive hybrid: direct leg 用正常 dense top-k

改法:

```text
bridge_comparison:
  graph/decomp/PPR leg 仍然跑
  direct leg 改用 dense top-8 + direct_gamma + top1_bonus
```

结果:

```text
experiments/hidden/data/small_2wiki_routerhybrid_kb750_temporal_20w25_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | Reuse | Writes |
|---|---:|---:|---:|---:|
| ARC | 24.8 | 32.5 | 28.5 | 1188 |
| DRIP-Dense | 23.3 | 27.1 | 20.0 | 1598 |
| DRIP | 21.8 | 27.4 | 18.2 | 1679 |
| DRIP_CHAIN | 22.8 | 26.6 | 18.8 | 1689 |
| DRIP_DECOMP | 22.4 | 26.7 | 21.2 | 1621 |

失败原因很明确:

```text
write_gold_rate 从原来的约 52.5% 降到 38.0%;
writes 从 804 增到 1598。
```

正常 dense top-k 对 bridge-comparison 太激进, 写了大量非 support 相似文档, churn 加重。

## 8.2 Conservative hybrid-v2: direct leg 看更多候选, 但仍 query-local normalized

改法:

```text
bridge_comparison:
  direct leg 从 first_hops(top3) 扩到 dense top8 输入
  但仍用 bridge_direct_gamma + no top1 bonus
  在 PPRBridgeDebt 中实际按 direct_debt_topk 做 query-local normalized credit
```

结果:

```text
experiments/hidden/data/small_2wiki_routerhybrid_v2_kb750_temporal_20w25_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | Reuse | Writes |
|---|---:|---:|---:|---:|
| ARC | 24.8 | 32.5 | 28.5 | 1188 |
| DRIP-Dense | **31.9** | 34.1 | 15.2 | 804 |
| DRIP | 26.3 | 29.7 | 17.0 | 1395 |
| DRIP_CHAIN | 28.5 | 30.8 | 20.0 | 1230 |
| DRIP_DECOMP | 31.7 | **34.3** | 17.0 | 1033 |
| DRIP_RSD | 26.5 | 29.0 | 17.0 | 782 |

解读:

- `DRIP-Dense` 和 `DRIP_DECOMP` 基本回到原结果。
- `DRIP_DECOMP` 仍只有很小 coverage/reuse 收益: 34.1 -> 34.3, 15.2 -> 17.0。
- PPR/stock DRIP 和 chain 仍低于 embedding-only, 说明 graph candidates 仍在引入噪声。

## 8.3 当前结论

router 确实需要更细:

```text
bridge_comparison != pure BRIDGE
bridge_comparison = visible direct leg + hidden bridge leg
```

但 naive 修法会过写。正确方向不是把 direct leg 放大, 而是:

```text
1. direct leg 保持保守 normalized support-debt;
2. hidden bridge leg 只能用更强 evidence gate;
3. PPR/chain/decomp 必须先证明 hidden-B precision, 再给写入预算。
```

本轮没有把 router-hybrid 改动留在正式代码里。正式代码保持当前保守 bridge-direct,
因为它在现有 benchmark 上更稳。

目前可用判断:

```text
有效:
  Route-sized support slots (bridge_comparison_slots=4)
  conservative embedding-only support-debt writer

弱有效:
  heuristic query decomposition (coverage/reuse 小幅提升, 但 writes 增加)

无稳定收益:
  stock PPR
  pair/chain PPR
  aggressive hybrid router
  lease-only retention
```

---

# 第九阶段: Resident-Anchor Bridge Cache (RABC)

目标从 "PPR 能不能传播到 B" 改成更明确的 cache-management 问题:

```text
如果 cache 里已经有可见 first-hop anchor A,
系统是否应该主动维护与 A 组成 support chain 的 hidden bridge B?
```

这比 PPR 更适合论文表述:

```text
PPR 是图检索/传播机制;
RABC 是 cache admission 机制: resident evidence 触发 hidden support residency.
```

## 9.1 方法公式 (简洁版)

对 query `q`, 先取 cache 中已命中的 resident anchors:

```text
A_t(q) = { a in KB_t : sim(q,a) >= tau_A }
```

只从这些 resident anchors 生成 bridge candidate:

```text
E_AB(q,b) =
  sum_{a in A_t(q)}
  sim(q,a) *
  sum_{e in Ent(a) cap Ent(b)}
    idf(e) / deg(e)^rho *
    title_bonus(e,b)
```

其中:

```text
title_bonus(e,b) > 1  iff normalized(e) matches title(b)
```

高频实体不再完全展开; 若 `deg(e) > d_cap`, 只允许连到 canonical title page:

```text
deg(e) > d_cap:
  expand only b where e matches title(b)
```

写入准则保持 direct-safe:

```text
direct writes first
bridge writes only use min(leftover_budget, f_brg * budget)
```

当前默认:

```text
tau_A=0.62
f_brg=0.10
m_brg=1.20
d_cap=20
resident_anchor_topk=5
dense_anchor_topk=0
```

代码:

```text
algorithms/drip/cache_manager/local_ppr.py
  AnchorBridgeEvidence
  AnchorBridgeDRIPCore

algorithms/drip/cache_manager/drip.py
  DRIPAnchor

registry key:
  DRIP_ANCHOR
```

## 9.2 Benchmark 修正

原 `temporal_bridge_reuse` 太散, 不是干净的 "已有 A, 缺 B" 场景。
新增:

```text
workload = resident_anchor_bridge_reuse
```

构造:

```text
1. 选择 question 中不出现 title 的 hidden support B;
2. 选择高复用 hidden groups;
3. 每组 1 个 exposure + 多个 reuse;
4. 初始 KB 放入其他 support docs 作为 resident anchors A;
5. 初始 KB 明确 hold out hidden B.
```

当前默认:

```text
top 80 hidden groups
8 reuse / group
```

代码:

```text
experiments/hidden/utils.py
  _build_resident_anchor_bridge_stream

experiments/hidden/run.py
  _init_anchored_bridge_kb
```

## 9.3 小规模机制验证结果

设置:

```text
2Wiki bridge_comparison
n_source=3000, n_stream_queries=1200
20 windows x 25 queries
KB=750, graph retrieval
resident_anchor_bridge_reuse
```

输出:

```text
experiments/hidden/data/small_2wiki_resident_anchor_kb750_20w25_graphret_v4.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 21.0 | 32.4 | 2.36 | **10.3** | 1109 |
| DRIP-Dense | **36.1** | **54.3** | **1.59** | 3.4 | 450 |
| DRIP_ANCHOR | 33.2 | 49.1 | 1.69 | 7.8 | 589 |

解释:

```text
RABC 把 hidden reuse 从 3.4% 提到 7.8%,
证明 resident-anchor bridge signal 有作用;
但它牺牲了部分 direct support residency, 所以 R@5/Cov 低于 embedding-only.
```

## 9.4 完整 50x50 stress test 结果

设置:

```text
2Wiki bridge_comparison
n_source=3000
50 windows x 50 queries
KB=750, graph retrieval
resident_anchor_bridge_reuse (top80 hidden groups, 8 reuse/group)
```

输出:

```text
experiments/hidden/data/full_2wiki_resident_anchor_top80_titlecap_kb750_50w50_graphret.json
experiments/hidden/data/full_2wiki_resident_anchor_top80_decomp_kb750_50w50_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 5.3 | **12.2** | **3.47** | **17.7** | 2228 |
| DRIP-Dense | **7.9** | 8.9 | 3.56 | 1.6 | 3621 |
| DRIP_ANCHOR | **7.9** | 9.0 | 3.57 | 1.6 | 3963 |
| DRIP_DECOMP(heuristic) | 6.8 | 8.8 | 3.56 | 2.8 | 4046 |

解读:

```text
1. DRIP/embedding path 的 retrieval R@5 仍高于 ARC。
2. 但 hidden reuse / KB coverage 仍是 ARC 更强。
3. RABC 在 full 50x50 上没有稳定转化为 hidden-B residency。
4. heuristic decomposition 也没有解决 hidden-B precision, 反而略增 writes。
```

## 9.5 当前科学结论

RABC 是一个更干净、可写成论文方法的方向:

```text
resident anchor -> hidden bridge candidate -> direct-safe bridge admission
```

但当前 spaCy/entity-overlap 实现还不足以支撑 "大幅超过 ARC/DRIP-Dense" 的强主张。
主要失败点不是 router, 也不是 writer reserve, 而是:

```text
candidate B 没有稳定命中 reuse_support_title;
写到的 gold support 很多, 但不是未来复用的 hidden B。
```

下一步如果要把这条线做成强论文结果, 应该换 bridge evidence, 而不是继续调 writer:

```text
1. 用 LLM/entity-linker 从 resident A 抽 relation-specific target entity;
2. 或用 anchor-conditioned subquery:
     q_AB = "the <relation> of <anchor title/context>"
   再 dense retrieve B;
3. 保留 RABC 的 direct-safe admission shell.
```

可汇报的保守故事:

```text
Bridge cache 是真实 failure mode;
RABC 给出了 cache-management 形式化;
当前实验显示小规模机制有效, 但 full stress test 中 bottleneck 转移到
anchor-to-hidden entity linking, 需要更强 bridge evidence.
```

---

# 第十阶段: Evidence-Conditioned Support Completion (ESC) + multi-agent bridge reuse

本轮目标是回应第九阶段失败点:

```text
RABC 写到很多 gold support, 但不是未来复用的 hidden B。
```

因此不再继续调 PPR 或实体共现权重, 而是把 bridge evidence 改成
anchor-conditioned next-hop retrieval。灵感来自 IRCoT / Self-Ask 的
"previous evidence -> next retrieval query" 思路, 但这里用于 cache admission,
不是用于端到端 QA 推理。

## 10.1 新方法: ACTR / ESC

当前代码名:

```text
DRIP-ESC
algorithms/drip/cache_manager/local_ppr.py
algorithms/drip/cache_manager/drip.py
```

简洁公式:

```text
A_t(q) = {a in KB_t : sim(q,a) >= tau_A}

h(q) = relation cues from question text
       e.g. director / producer / country / born / author ...

z(q,a) = BGE("Target relation: h(q,a);
              Original question: q;
              Known evidence: title(a), text(a);
              Retrieve the missing supporting evidence.")

C(a) = entity-neighbor candidates of a

E_brg(q,b) =
  sum_{a in A_t(q)}
    sim(q,a) / sqrt(rank(a))
    * [(1-alpha) edge(a,b) + alpha sim(z(q,a), b)^p]
    * [1 + eta edge(a,b)]
```

Admission 保持 direct-safe:

```text
direct candidates use the direct floor
bridge candidates use a small overflow reserve
```

当前默认:

```text
tau_A = 0.58
bridge_max_reserve = 0.25
bridge_margin = 0.95
alpha = 0.60
p = 1.0
```

注意: relation cue 是从 question text 抽取, 不是 gold support / qtype oracle。
这比之前的 role rerank 更适合作为论文方法: 它是一般的下一跳查询生成,
不是 2Wiki film/person 特判。

## 10.2 新 workload: multi_agent_bridge_reuse

旧 `resident_anchor_bridge_reuse` 的问题:

```text
小规模 hidden groups 太少;
full 50x50 时虽然 hidden groups 多, 但 reuse target 与写入信号不稳定对齐。
```

新 workload:

```text
experiments/hidden/utils.py
  _build_multi_agent_bridge_stream
```

构造:

```text
1. 选择 hidden support B:
   - B 不出现在 question text 中;
   - B 被 >=3 个不同 query 共享。

2. 初始 KB:
   - 保留其他 support docs 作为 resident anchors A;
   - hold out hidden B。

3. 流:
   - exposure queries 出现在前半段;
   - reuse queries 由多个 synthetic agents 在后半段重复请求同一 B;
   - agent_id 只用于 workload 标记, 策略不可见。
```

这更贴近论文问题:

```text
multi-agent shared cache under distribution shift should amortize hidden bridge
evidence across future agents.
```

## 10.3 小规模有效结果

设置:

```text
2Wiki bridge_comparison
n_source=3000, n_stream_queries=1200
20 windows x 25 queries
KB=750, graph retrieval
multi_agent_bridge_reuse
```

输出:

```text
experiments/hidden/data/small_2wiki_multiagent_esc_text_rel_kb750_20w25_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 15.5 | 24.9 | 2.89 | 20.8 | 1452 |
| DRIP-Dense | **18.7** | 24.1 | 2.26 | 5.6 | **684** |
| DRIP-ESC | 15.9 | **29.3** | **2.16** | **34.8** | 942 |

解读:

```text
ESC_TEXT 明显提高 hidden bridge residency:
  Reuse: 20.8 -> 34.8 vs ARC (+14.0pp)
  Reuse: 5.6  -> 34.8 vs embedding-only (+29.2pp)
  KB Cov H2: 24.9 -> 29.3 vs ARC (+4.4pp)

代价:
  R@5 H2 低于 embedding-only (15.9 vs 18.7),
  但略高于 ARC (15.9 vs 15.5).
```

这说明:

```text
多跳 cache management 的收益不是自动体现在 R@5;
它首先体现在 hidden support residency / future-agent reuse。
```

## 10.4 完整 50x50 仍未成立

设置:

```text
2Wiki bridge_comparison
n_source=3000, n_stream_queries=3000
50 windows x 50 queries
KB=750, graph retrieval
multi_agent_bridge_reuse
```

输出:

```text
experiments/hidden/data/full_2wiki_multiagent_esc_text_rel_kb750_50w50_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 6.1 | **13.9** | **3.44** | **19.6** | **2157** |
| DRIP-Dense | **7.1** | 8.3 | 3.56 | 2.6 | 3641 |
| DRIP-ESC | 6.9 | 8.2 | 3.57 | 2.5 | 3925 |

结论必须诚实:

```text
ESC_TEXT 在 small/local multi-agent reuse 上有效;
但在 50x50 long-horizon stress test 上没有击败 ARC。
```

日志诊断:

```text
DRIP-ESC:
  bridge_gold_updates = 3844
  write_gold = 2852
  gold_candidates = 7808

但 reuse_hit_rate 只有 2.5%。
```

这说明它仍然写了很多 gold support, 但不是长期复用的 hidden B,
或者写入后在长流里被 churn 掉。ARC 虽然没有多跳设计, 但 conservative
admission / low write churn 在 long-horizon residency 上更强。

## 10.5 当前论文判断

不能写:

```text
Our method achieves SOTA over ARC on the full long-horizon benchmark.
```

可以写:

```text
We identify bridge-support residency as a cache-management failure mode.
We propose evidence-conditioned support completion (ESC), which converts
resident first-hop evidence into relation-conditioned bridge admission.
On controlled multi-agent bridge reuse, ESC improves hidden-support reuse and
KB coverage over ARC and embedding-only admission. On longer 50x50 streams,
the remaining bottleneck is long-horizon retention and exact hidden-target
selection.
```

下一步若要冲强结果:

```text
1. 加 hidden-target diagnostics:
   bridge_hidden_candidates / bridge_hidden_writes / hidden_evictions.

2. 换更可分解的数据集:
   HotpotQA bridge 或 MuSiQue compositional, 但需要先预编码缓存;
   当前 Hotpot 28k docs 重新编码过慢, 本轮已停止。

3. 用 LLM/entity linker 只做 relation extraction / entity linking,
   不用 gold labels:
     h(q,a) = LLM relation target for anchor a
   然后仍走 ESC admission。

4. 重新设计 retention:
   不能简单 hard shelf; small 结果显示 shelf 保护 noisy bridge docs,
   反而降低 reuse (34.8 -> 28.8)。
```

---

# 第十一阶段: hidden-target diagnostics + generic relation linking + confirmed retention

本轮新增:

```text
1. hidden-target diagnostics
   bridge_hidden_updates / top1 / top5 / top10 / MRR
   write_hidden_candidates / write_hidden / hidden_evictions
   direct_writes / bridge_writes / route budgets

2. generic relation/entity linking
   不再使用 director/producer/writer 这类数据集角色模板。
   从 question 中抽 relation cues 和 capitalized/entity spans, 构造:

     z_text(q,A) = "relation cues + question entities + known evidence A"

   候选 B 的得分增加通用 cue-overlap bonus, 而不是 2Wiki-specific rule。

3. confirmed bridge retention
   对反复被 bridge completion 指向的候选积累确认信用:

     Z_t(d) = decayed bridge support debt
     C_t(d) = decayed confirmed bridge mass, activated after repeated hits
     U_brg(d) = Z_t(d) + mu C_t(d)

   resident priority 也加 C_t(d), 但不做 hard shelf。
```

## 11.1 小规模 20x25 结果

设置:

```text
2Wiki bridge_comparison, n_source=3000
multi_agent_bridge_reuse, 20 windows x 25 queries
KB=750, graph retrieval
```

默认稳健版输出:

```text
experiments/hidden/data/small_2wiki_multiagent_esc_text_diag_confirm_v3_kb750_20w25_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 15.5 | 24.8 | 2.89 | 20.4 | 1442 |
| DRIP-Dense | **18.6** | 24.0 | 2.27 | 5.6 | **686** |
| DRIP-ESC | 14.7 | **31.2** | **2.12** | **44.8** | 986 |

hidden diagnostics:

```text
bridge_hidden_updates = 345
bridge_hidden_top5    = 301
write_hidden_candidates = 221
write_hidden = 41
hidden_evictions = 0
direct_writes = 722
bridge_writes = 264
```

解读:

```text
relation/entity linking 已经基本找到 hidden B:
  301 / 345 hidden hits are ranked in top-5.

但只有 41 hidden writes:
  主要瓶颈不是找不到 B, 而是 bridge admission/slot competition。
```

## 11.2 aggressive bridge lane 消融

为验证 admission 瓶颈, 额外开启:

```text
bridge_rank_admit = True
soft bridge victim fallback = True
```

输出:

```text
experiments/hidden/data/small_2wiki_multiagent_esc_text_rankadmit_softvictim_kb750_20w25_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 15.5 | 24.8 | 2.89 | 20.4 | 1442 |
| DRIP-Dense | **18.6** | 24.0 | 2.27 | 5.6 | **686** |
| DRIP-ESC aggressive | 14.8 | **31.8** | **2.19** | **46.4** | 1440 |

hidden diagnostics:

```text
write_hidden: 41 -> 118
bridge_writes: 264 -> 693
write_gold: 541 -> 735
```

结论:

```text
aggressive bridge lane 证明了 admission/slot victim 是真实瓶颈:
放开 bridge victim 后 hidden writes 大幅增加。

但它损害 H1/R@5, 因为过早替换 anchor/direct residents。
所以 aggressive lane 只作为消融, 不作为默认主方法。
```

## 11.3 完整 50x50 结果

设置:

```text
2Wiki bridge_comparison, n_source=3000
multi_agent_bridge_reuse, 50 windows x 50 queries
KB=750, graph retrieval
```

输出:

```text
experiments/hidden/data/full_2wiki_multiagent_esc_text_diag_confirm_v2_kb750_50w50_graphret.json
```

| Strategy | R@5 H2 | KB Cov H2 | ColdQ | Reuse | Writes |
|---|---:|---:|---:|---:|---:|
| ARC | 6.1 | **13.9** | **3.44** | **19.6** | **2151** |
| DRIP-Dense | 7.1 | 8.3 | 3.56 | 2.6 | 3641 |
| DRIP-ESC | **7.2** | 9.8 | 3.52 | 9.3 | 3776 |

hidden diagnostics:

```text
bridge_hidden_updates = 931
bridge_hidden_top5    = 813
write_hidden_candidates = 870
write_hidden = 14
hidden_evictions = 1
```

长流结论:

```text
ESC 在 full long-horizon 上有小幅 R@5 H2 提升:
  ARC 6.1 -> ESC 7.2
  EMB_ONLY 7.1 -> ESC 7.2

但 KB coverage / reuse 仍不如 ARC:
  ARC Cov H2 13.9, Reuse 19.6
  ESC Cov H2 9.8, Reuse 9.3

核心原因不是 PPR/ESC 找不到 hidden B:
  hidden B top5 = 813 次。

核心原因是 admission/retention 不够:
  hidden candidates = 870, hidden writes = 14。
```

## 11.4 当前方法判断

现在不能再说:

```text
PPR 无效, 因为 graph 找不到 B。
```

更准确是:

```text
纯 PPR 无效, 因为它把 query-conditioned relation signal 稀释到稠密实体邻域里,
排名和 admission 都会被 hub/noise 文档干扰。

ESC 有效地修了候选定位:
用 resident anchor A + relation cues 让 hidden B 经常排到 top5。

剩余问题是 cache admission:
真正未来要复用的 hidden B 进入候选, 但没有稳定赢得 slot。
```

下一步应从 "找 B" 转向 "给 B 留 slot":

```text
1. 不再继续调 PPR 的 R/L/c/d_cap。
2. 设计 bridge lane 的 victim policy:
   允许少量替换低价值 anchor, 但不能伤 H1 direct floor。
3. serve/retrieval 层要让 resident hidden B 在 multi-hop query 里进入 top-5;
   否则 KB coverage 提升不会自动变成 R@5。
```

---

# 第十二阶段: 第一优先级改为 no-shift multihop cache

现在的实验顺序应固定为:

```text
1. Multihop no topic shift:
   先证明多跳 support doc 能被 cache 管住。

2. ARC-paper style:
   固定小 cache capacity, 大 corpus + QA stream,
   主指标用 has-answer / support residency, 不是 R@5。

3. Topic shift + multihop:
   只有在 1 成立后, 才加入 head->tail topic shift。
```

原因:

```text
multi-hop cache management 与 topic shift 是两个正交难点。
如果直接叠加, 结果差时无法判断是:
  A. hidden B 找不到/写不进/留不住,
  B. topic shift 使旧 topic support 被驱逐,
  C. retrieval 层没把 resident support 取出来。
```

## 12.1 数据构造

当前 no-shift multihop 使用:

```text
workload = multi_agent_bridge_reuse
initial KB = anchor support A kept, hidden support B held out
front half = exposure queries
back half  = reuse queries from other synthetic agents
```

它不是普通 topic shift。它刻意测试:

```text
resident A + query q 能否引导 cache 写入 hidden B,
并让后续 reuse query 直接从 shared cache 命中 B。
```

## 12.2 主指标

现在多跳 cache 的指标优先级是:

```text
hidden_B_hit_rate       hidden bridge support B 是否在 KB
has_answer_rate         query 的所有 support docs 是否都在 KB
support_coverage_rate   support docs 的平均驻留比例
reuse_hit_rate          reuse query 到来时 hidden B 是否仍在 KB
writes                  达到上述效果的换入成本
```

`R@5` 暂时降级为 retrieval 层指标。cache manager 的主责任是让 gold support
resident; resident 以后是否能被 top-5 retriever 取出, 是下一层问题。

## 12.3 2Wiki 20x25 has-answer 结果

设置:

```text
2Wiki bridge_comparison, n_source=3000
multi_agent_bridge_reuse, 20 windows x 25 queries
KB=750, graph retrieval
output = experiments/hidden/data/small_2wiki_multiagent_hasanswer_20w25_graphret.json
```

| Strategy | Has-answer | Support Cov | Hidden-B Hit | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|
| ARC | 0.2 | 27.9 | 17.9 | 20.8 | 2.886 | 1452 |
| DRIP-Dense | 2.8 | 43.4 | 4.1 | 5.6 | 2.264 | 684 |
| DRIP-ESC | **4.8** | **47.1** | **29.3** | **45.2** | **2.116** | 987 |
| DRIP-ESC_CONFIRM | 4.0 | 45.7 | 19.4 | 29.6 | 2.172 | 931 |
| Oracle | 100.0 | 100.0 | 100.0 | 100.0 | 0.000 | 4938 |

相对 ARC:

```text
has-answer:   0.2 -> 4.8  (+4.6pp)
support cov: 27.9 -> 47.1 (+19.2pp)
hidden-B:    17.9 -> 29.3 (+11.4pp)
reuse:       20.8 -> 45.2 (+24.4pp)
writes:      1452 -> 987  (-32.0%)
```

相对 pure embedding cache:

```text
has-answer:   2.8 -> 4.8  (+2.0pp)
support cov: 43.4 -> 47.1 (+3.7pp)
hidden-B:     4.1 -> 29.3 (+25.2pp)
reuse:        5.6 -> 45.2 (+39.6pp)
```

## 12.4 当前判断

这轮说明 `DRIP-ESC` 在 no-shift multihop cache 口径下是有效的:

```text
它不是靠单跳 embedding-only 取胜。
主要增益来自 hidden B residency 和 reuse conversion。
```

但它还不是最终论文级结果:

```text
absolute has-answer 仍只有 4.8%, 说明多 support 全部驻留仍很难。
DRIP-ESC_CONFIRM 更稳但太保守, 不是当前主线。
当前主线应暂定 DRIP-ESC, 再做 2Wiki/Hotpot/MuSiQue no-shift 扩展验证。
```

下一步:

```text
1. 等当前 2Wiki 50x50 full run 完成, 但注意它可能是 has-answer patch 前启动。
2. 用同一 has-answer 指标复跑 2Wiki full。
3. 再跑 Hotpot / MuSiQue no-shift multihop。
4. 第一阶段稳定后, 再切到 ARC-style 小 cache + 大 corpus。
5. 最后叠加 topic shift + multihop。
```

## 12.5 Pair lease: 让 A+B 同时留在 cache

问题:

```text
ESC_TEXT 能把 hidden B 找出来并写入, 但 has-answer 仍低。
原因是 B 驻留不等于完整答案驻留:
  B 在 KB, 但 anchor A 或其他 support 被换走 => has-answer 仍为 0。
```

新增机制:

```text
DRIP-ESC-Lease
```

核心思想:

```text
ESC 生成 B 时记录 provenance: 哪个 anchor A 贡献了这个 B。
如果 B 被 bridge route 写入, 则给 (A,B) 一个短期 decayed lease,
让 A 和 B 在 resident priority 中一起被保护。
```

简洁公式:

```text
E_ESC(q,B|A) = target_sim(q,A,B) * link(A,B) * cue(q,B)

L_t(A), L_t(B) <- rho L_{t-1} + E_ESC(q,B|A)

D_t(d) = D_vis,t(d) + D_hid,t(d) + lambda_pair L_t(d)

P_t(d) = S_t(d) + D_t(d)
```

Admission 仍不变:

```text
admit c replacing v iff U(c) > margin * P_t(v)
```

区别:

```text
原 ESC_TEXT:
  保护单个 B。

ESC_TEXT_PAIR:
  保护产生 B 的 support pair (A,B)。
```

这不是 ARC 的 DRF/hubness:

```text
L_t 只来自 ESC 的 anchor-conditioned bridge evidence,
不是 miss frequency, 不是 query-distance frequency, 也不计算 hubness。
```

代码位置:

```text
algorithms/drip/cache_manager/support_completion.py
  EvidenceConditionedBridgeEvidence.last_pair_scores
  EvidenceConditionedDRIPCore.pair_lease

algorithms/drip/cache_manager/drip.py
  DRIPESCLease

registry key:
  DRIP-ESC-Lease
```

### 12.5.1 2Wiki 20x25 pair-lease 结果

设置同 12.3, 输出:

```text
experiments/hidden/data/small_2wiki_multiagent_pairlease_20w25_graphret.json
```

| Strategy | Has-answer | Support Cov | Hidden-B | Reuse | R@5 H2 | KB Cov H2 | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARC | 0.2 | 27.9 | 17.9 | 20.8 | 15.5 | 24.9 | 1452 |
| DRIP-Dense | 2.8 | 43.4 | 4.1 | 5.6 | 18.7 | 24.1 | 684 |
| DRIP-ESC | 4.8 | **47.1** | **29.3** | **45.2** | 14.7 | 31.5 | 987 |
| DRIP-ESC-Lease | **5.2** | 45.8 | **29.3** | **45.2** | **19.3** | **35.7** | **797** |

Pair diagnostics:

```text
pair_activations = 298
pair_lease_docs_last = 612
pair_lease_mass_last = 439.21
```

相对原 ESC_TEXT:

```text
has-answer: 4.8 -> 5.2 (+0.4pp)
R@5 H2:     14.7 -> 19.3 (+4.6pp)
KB Cov H2:  31.5 -> 35.7 (+4.2pp)
writes:     987 -> 797 (-19.3%)
```

解读:

```text
pair lease 确实修了一部分 A+B 不同时驻留的问题,
尤其体现在 H2 coverage 和 H2 R@5 上。

但全局 support_coverage_rate 从 47.1 降到 45.8,
说明它保护 bridge pair 时牺牲了一部分 H1/direct/background support。
```

下一步不能只看全局 support coverage, 应增加 target-only 指标:

```text
target_has_answer_rate     只在 exposure/reuse bridge query 上算
target_support_coverage    只在 exposure/reuse bridge query 上算
background_support_coverage 单独报告被牺牲多少
```

若 target-only 明显升, 这就是论文故事:

```text
support-pair lease improves multihop answer residency under fixed cache budget.
```

---

# 第十三阶段: 二跳 pair 是否足够? 跨数据集验证

问题:

```text
当前 ESC_TEXT_PAIR 是二跳 support pair:
  resident/easy A -> hidden B

它能否代表 multi-hop cache?
```

判断:

```text
二跳 pair 是当前论文主线的合适最小单元, 但不能直接声称已经解决任意 k-hop。

HotpotQA / 2Wiki 多数问题可被建模为两个 support docs 的 bridge completion。
MuSiQue 更接近 3/4-hop support chain, 如果 pair lease 不激活, 需要扩展为
support-set lease, 而不是继续调 PPR。
```

## 13.1 跨数据集 20x25 结果

设置:

```text
KB=750, n_source=3000, 20 windows x 25 queries
workload=multi_agent_bridge_reuse
retrieval=graph
strategies=ARC / DRIP-Dense / DRIP-ESC / DRIP-ESC-Lease / Oracle
```

输出:

```text
2Wiki:
experiments/hidden/data/small_2wiki_multiagent_pairlease_20w25_graphret.json

Hotpot:
experiments/hidden/data/multids_hotpot_musique_pairlease_20w25_graphret.json

MuSiQue corrected:
experiments/hidden/data/musique_pairlease_diagfix_20w25_graphret.json
```

| Dataset | Strategy | Has-answer | Support Cov | Hidden-B | Reuse | R@5 H2 | KB Cov H2 | Writes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | ARC | 0.2 | 27.9 | 17.9 | 20.8 | 15.5 | 24.9 | 1452 |
| 2Wiki | EMB_ONLY | 2.8 | 43.4 | 4.1 | 5.6 | 18.7 | 24.1 | 684 |
| 2Wiki | ESC_TEXT | 4.8 | **47.1** | **29.3** | **45.2** | 14.7 | 31.5 | 987 |
| 2Wiki | ESC_TEXT_PAIR | **5.2** | 45.8 | **29.3** | **45.2** | **19.3** | **35.7** | **797** |
| Hotpot | ARC | 14.6 | 39.6 | 16.0 | 32.0 | 51.6 | 52.6 | 1675 |
| Hotpot | EMB_ONLY | 52.2 | 73.0 | 24.0 | 48.0 | 64.4 | 66.3 | **451** |
| Hotpot | ESC_TEXT | 58.4 | 76.6 | 22.0 | 44.0 | 66.6 | 68.5 | 455 |
| Hotpot | ESC_TEXT_PAIR | **62.4** | **78.2** | 22.0 | 44.0 | **71.6** | **72.9** | 492 |
| MuSiQue | ARC | 2.4 | 22.1 | 14.6 | 18.8 | 20.3 | 24.5 | 1796 |
| MuSiQue | EMB_ONLY | 5.8 | 36.0 | 14.1 | 22.4 | 23.5 | 27.8 | 792 |
| MuSiQue | ESC_TEXT | 8.4 | **41.9** | **16.3** | **26.0** | 28.0 | 35.2 | **701** |
| MuSiQue | ESC_TEXT_PAIR | **8.6** | 41.2 | 15.6 | 24.8 | **29.5** | **36.4** | 705 |

Pair lease diagnostics:

| Dataset | pair_activations | pair_lease_docs_last | pair_lease_mass_last |
|---|---:|---:|---:|
| 2Wiki | 298 | 612 | 439.21 |
| Hotpot | 190 | 506 | 290.92 |
| MuSiQue | 250 | 477 | 336.81 |

## 13.2 解读

2Wiki:

```text
pair lease 有效。
它把 H2 R@5 从 14.7 提到 19.3, KB Cov H2 从 31.5 提到 35.7,
同时 writes 从 987 降到 797。
```

Hotpot:

```text
pair lease 也有效:
has-answer 58.4 -> 62.4
R@5 H2 66.6 -> 71.6
KB Cov H2 68.5 -> 72.9

但本轮 Hotpot 没有找到足够 multi-agent shared hidden-B groups,
runner fallback 到 hidden anchored reuse, 所以它证明二跳 support completion,
不能作为强 multi-agent reuse 证据。
```

MuSiQue corrected:

```text
旧结论 "pair_activations=0" 作废。
原因不是方法完全无效, 而是 hidden-support workload 没有给 MuSiQue query 标 route_hint,
导致诊断显示 route_labeled=0, 并且之前 mixed result 把 route 问题和算法问题混在一起。

修复后:
  route_labeled=25/25 per window
  route_match=25/25 per window
  pair_activations=250

ESC_TEXT 相对 EMB_ONLY:
  has-answer 5.8 -> 8.4
  target has-answer 3.7 -> 4.9
  R@5 H2 23.5 -> 28.0
  KB Cov H2 27.8 -> 35.2
  writes 792 -> 701

ESC_TEXT_PAIR 相对 ESC_TEXT:
  has-answer 8.4 -> 8.6
  target has-answer 4.9 -> 5.4
  R@5 H2 28.0 -> 29.5
  KB Cov H2 35.2 -> 36.4

结论:
  ESC evidence 在 MuSiQue 也有效; pair lease 有正收益但较小。
  MuSiQue 仍比 2Wiki/Hotpot 难, 因为很多 query 需要 3/4 个 supports 同时驻留,
  二跳 pair 只能修 A+B, 不能保证完整 support set。
```

## 13.3 当前方法边界

可以声称:

```text
ESC + Pair Lease solves two-hop bridge support completion:
find missing B conditioned on resident A, then protect A+B together.
```

不能声称:

```text
已经解决任意 k-hop support-chain caching。
```

如果论文时间紧:

```text
主线聚焦 bridge support completion:
  resident/easy A -> missing/hidden B

2Wiki/Hotpot 是主要有效性证据。
MuSiQue 是 harder multi-support setting: ESC 仍有效, pair lease 增益较小,
可作为 "needs support-set lease" 的边界分析, 不再当作方法失败。
```

如果继续推进 MuSiQue, 下一步不是调 pair 参数, 而是改机制:

```text
Support-set lease:
  对每个 under-covered query, 估计一个 support set G_q = top anchors + bridge candidates.
  当其中任一 candidate 被写入, 对 G_q 中已驻留/新写入的多个 support docs 给 group lease.

公式:
  L_t(d) <- rho L_{t-1}(d) + sum_{q: d in G_q} E(q, G_q) / |G_q|
  D_t(d) = D_vis,t(d) + D_hid,t(d) + lambda_set L_t(d)
  P_t(d) = S_t(d) + D_t(d)

这样 pair 是 |G_q|=2 的特例, MuSiQue 可以覆盖 3/4-hop chain。
```

## 13.4 本轮问题汇总与代码修复

问题 1: MuSiQue bridge route 没有可诊断标签。

```text
现象:
  route_labeled=0, 容易误读成 bridge 分支没跑或 pair lease 没触发。

原因:
  MuSiQue 原始 qtype 不稳定/缺失; hidden-support workload 只写了 reuse_support_title,
  没写 route_hint=bridge。

修复:
  experiments/hidden/utils.py
    _query_copy(... support_title=...) -> out.setdefault("route_hint", "bridge")
```

问题 2: route 诊断和 router 不一致。

```text
现象:
  QueryRouter.route() 已经读 route_hint,
  但 DRIPCore._expected_route() 只读 qtype/type, 所以 route_match 统计失真。

修复:
  algorithms/drip/cache_manager/__init__.py
    qtype = route_hint or qtype or type

验证:
  musique_pairlease_diagfix_20w25_graphret.json
  route_labeled=25/25, route_match=25/25, routes.BRIDGE=25/25
```

问题 3: 全局 support coverage 混合了 target/background。

```text
现象:
  support_coverage_rate 低时, 无法判断是 target bridge query 没保住,
  还是 background direct query 被牺牲。

修复:
  experiments/hidden/run.py 新增:
    target_has_answer_rate
    target_support_coverage_rate
    target_cold_fetches_per_query
    background_has_answer_rate
    background_support_coverage_rate
    background_cold_fetches_per_query
```

当前主方法公式保持简洁:

```text
E_ESC(q, B | A) = sim(phi(q, A), B) * link(A, B) * cue(q, B)

L_t(A), L_t(B) <- rho L_{t-1} + E_ESC(q_t, B | A)

D_t(d) = D_vis,t(d) + D_hid,t(d) + lambda_pair L_t(d)

P_t(d) = S_t(d) + D_t(d)

admit c replacing v iff U(c) > margin * P_t(v)
```

这里 `E_ESC` 解决 hidden support discovery, `L_t` 解决 A+B 同时驻留。
没有使用 ARC 的 DRF/hubness, 也没有把 PPR 作为当前主线公式。
当前 registry 只保留 `DRIP`, `DRIP-ESC`, `DRIP-ESC-Lease`,
`DRIP-Dense`; PPR/entity-rerank/decomp 原型已从主入口下线。

---

# 第十四阶段: 主入口瘦身 + direct-first writer 验证

用户反馈是正确的: 如果最终方法还暴露 PPR 的 `c/L/R/K0/d_cap`,
再叠加 role rerank / hubness / DRF / Red, 公式会失去论文主线。
本轮把当前代码入口收敛为一个简洁 cache-management 方法:

```text
DRIP = ESC hidden-support completion + Pair Lease retention
```

## 14.1 删除/下线内容

当前可运行 registry 只保留:

```text
DRIP
DRIP-Dense
DRIP-ESC
DRIP-ESC-Lease
```

下线/删除:

```text
algorithms/drip/cache_manager/local_ppr.py
algorithms/drip/tests/test_bridge_ppr.py
algorithms/drip/tests/test_bridge_ppr_real.py
experiments/hidden/run_ppr_2wiki.py
```

同时清理 `experiments/hidden/config.py` 和 `run.py` 里的旧展示名:

```text
DRIP_PPR_*
DRIP_ENTITY_RERANK*
DRIP_DECOMP
DRIP_OVERFLOW / PAIR / CHAIN / RSD
```

保留的 `experiments/hidden/graph_retrieval.py` 是 retrieval backend,
不属于 DRIP cache policy 的主公式。

## 14.2 Writer 修正: direct-first, bridge-leftover

之前 bridge bucket 会先拿固定预算, 在某些窗口挤掉 query-visible direct support。
本轮改成:

```text
1. direct candidates 先用本窗口全部 write budget 尝试补可见 support;
2. bridge candidates 只使用 direct 写完后的剩余 budget;
3. bridge eviction 避免替换 direct-protected / recently-served / direct-demand resident docs。
```

公式上不增加新项, 只是 admission order 更合理:

```text
D_t(d) = D_dir,t(d) + D_brg,t(d) + lambda_pair L_t(d)
P_t(d) = S_t(d) + D_t(d)

direct admit first:
  admit c iff D_dir(c) > m_dir P_t(v)

bridge admit from leftover:
  admit b iff D_brg(b) > m_brg P_t(v),
  where v is not direct-protected when possible.
```

这避免了“为了补 hidden B, 把 A 或当前直接证据冲掉”的失败模式。

## 14.3 完整验证: 2Wiki multi-agent bridge reuse

命令:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/hidden/run.py \
  --datasets 2wikimultihopqa \
  --expanded \
  --q-type bridge_comparison \
  --n-source 3000 \
  --n-stream-queries 1200 \
  --n-windows 20 \
  --window-size 25 \
  --drift full_gradual \
  --workload multi_agent_bridge_reuse \
  --retrieval graph \
  --kb-budget 750 \
  --strategies ARC DRIP-Dense DRIP-ESC DRIP \
  --output cleanup_2wiki_esc_pair_directfloor_20w25_graphret.json
```

结果:

| Strategy | R@5 H2 | KB Cov H2 | hidden-B hit | has-answer | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARC | 15.4 | 24.9 | 17.9 | 0.2 | 20.8 | 2.884 | 1443 |
| DRIP-Dense | 19.4 | 25.1 | 4.1 | 3.6 | 5.6 | 2.202 | 649 |
| DRIP-ESC | 17.2 | 27.0 | 16.8 | 3.6 | 25.6 | 2.192 | 820 |
| DRIP | **19.9** | **31.9** | **23.0** | **4.6** | **35.2** | **2.132** | 785 |

和 direct-first 前的 DRIP 对比:

```text
R@5 H2:     19.2 -> 19.9
KB Cov H2:  31.5 -> 31.9
has-answer: 4.2 -> 4.6
ColdQ:      2.136 -> 2.132
Reuse:      36.4 -> 35.2
Writes:     776 -> 785
```

结论:

```text
ESC + Pair Lease 不是纯 embedding-only:
  hidden-B hit: 4.1 -> 23.0
  KB Cov H2:   25.1 -> 31.9
  Reuse:        5.6 -> 35.2

direct-first writer 让最终 R@5 H2 也略高于 embedding-only:
  19.4 -> 19.9
```

当前最稳的论文表述:

```text
DRIP does not make PPR the method. DRIP is a cache manager that repairs
under-covered multi-hop supports: ESC discovers the missing support B conditioned
on resident A, and Pair Lease keeps A+B resident long enough for reuse.
```

---

# 第十五阶段: 两条分支的当前收束

现在要把两个问题分开讲, 否则论文故事会混:

```text
Branch A: single-hop query shift
Branch B: no-topic-shift multi-hop support completion
```

它们不是同一个 benchmark。Branch A 主要证明边界: 当证据直接可见、时间局部性强时,
LRU/FIFO 已经很强, DRIP 不应该硬说大胜。Branch B 才是当前方法主贡献:
给 multi-hop hidden support 做 cache management。

## 15.1 Branch A: single-hop query shift

设置:

```text
StreamingQA temporal
pool=29,819, KB=400
100 windows x 50 queries
结果文件:
experiments/direct/data/results_streamingqa_temporal_final_clean.json
```

结果:

| Strategy | R@5 H1 | R@5 H2 | Writes | MaintR | Cost |
|---|---:|---:|---:|---:|---:|
| AgentRAGCache / ARC | 20.5 | 4.4 | 1083 | 4145 | 5228 |
| FIFO | 50.2 | 32.8 | 2109 | 2907 | 5016 |
| LRU | **52.8** | **33.1** | 2033 | 2843 | 4876 |
| DRIP-Dense | 48.9 | 30.3 | 2261 | 14800 | 17061 |
| DRIP | 48.5 | 29.3 | 2174 | 15010 | 17184 |
| Oracle | 84.4 | 79.4 | 22140 | 0 | 22140 |

结论:

```text
single-hop temporal query shift is a calibration branch, not DRIP's main win.
```

可以说:

```text
1. ARC 在 temporal single-hop 下很弱, 因为历史 DRF/hubness 不等于时间局部性。
2. LRU/FIFO 很强, 说明单跳时间漂移里 recency/access history 是强 baseline。
3. DRIP-Dense/DRIP 没有在这个分支上大幅超过 LRU; 因此论文不应把 single-hop temporal
   写成主贡献。
4. 这个分支的作用是校准: DRIP 的 multi-hop 组件不应该用来解释单跳胜负。
```

不能说:

```text
DRIP already solves query shift end to end.
```

更稳的讲法:

```text
On directly visible single-hop drift, recency baselines are already strong.
Our contribution targets a harder orthogonal failure mode: hidden support
completion in multi-hop cache management.
```

## 15.2 Branch B: no-topic-shift multi-hop support completion

设置:

```text
static corpus, no detector
workload=multi_agent_bridge_reuse
KB=750, n_source=3000
20 windows x 25 queries
retrieval=graph for serving metric
strategies=ARC / DRIP-Dense / DRIP-ESC / DRIP
```

注意: 命令里仍传 `--drift full_gradual`, 这是 run.py 的 stream-order 参数;
本节不把它解释成 topic shift 贡献。这里评估的是:

```text
在固定语料和固定小 cache 下, 能不能把 hidden multi-hop support 写进并留在 hot tier。
```

结果文件:

```text
2Wiki:
experiments/hidden/data/cleanup_2wiki_esc_pair_directfloor_20w25_graphret.json

Hotpot + MuSiQue:
experiments/hidden/data/cleanup_hotpot_musique_esc_pair_directfloor_20w25_graphret.json
```

### 2Wiki bridge-comparison

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 15.4 | 24.9 | 27.9 | 0.2 | 17.9 | 20.8 | 2.884 | 1443 |
| DRIP-Dense | 19.4 | 25.1 | 45.0 | 3.6 | 4.1 | 5.6 | 2.202 | 649 |
| DRIP-ESC | 17.2 | 27.0 | 45.2 | 3.6 | 16.8 | 25.6 | 2.192 | 820 |
| DRIP | **19.9** | **31.9** | **46.7** | **4.6** | **23.0** | **35.2** | **2.132** | 785 |

2Wiki 结论:

```text
ESC + Pair Lease 相对 embedding-only 的核心增益:
  hidden-B: 4.1 -> 23.0
  KB Cov H2: 25.1 -> 31.9
  Reuse: 5.6 -> 35.2
  R@5 H2: 19.4 -> 19.9
```

这说明它不是纯单跳 embedding cache。它确实把 hidden support B 写进并保住,
但最终 R@5 只小幅提高, 因为 retrieval 排序仍然会影响 top-5。

### HotpotQA

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 57.8 | 59.4 | 46.0 | 22.0 | 10.0 | 20.0 | 1.080 | 1528 |
| DRIP-Dense | **67.8** | **68.7** | **77.5** | **59.6** | 35.0 | **70.0** | **0.450** | **523** |
| DRIP-ESC | 67.4 | 68.1 | 76.8 | 58.4 | 35.0 | **70.0** | 0.464 | 534 |
| DRIP | 66.0 | 66.9 | 75.9 | 57.8 | 35.0 | **70.0** | 0.482 | 556 |

Hotpot 解读:

```text
当前 Hotpot multi_agent_bridge_reuse 构造没有找到足够 shared hidden-B groups:
日志显示 fallback, reuse=10。
```

因此这轮 Hotpot 不适合作为 hidden bridge 主证据。它说明:

```text
when evidence is directly visible, embedding-only/direct cache is already enough.
Adding ESC is unnecessary and can mildly hurt.
```

这不是方法失败, 而是 workload 不匹配。Hotpot 可以放 direct-visible sanity / appendix,
不要作为 ESC 主胜场。

### MuSiQue

| Strategy | R@5 H2 | KB Cov H2 | Support Cov | Has-answer | Hidden-B | Reuse | ColdQ | Writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARC | 20.3 | 24.5 | 22.1 | 2.4 | 14.6 | 18.8 | 2.080 | 1796 |
| DRIP-Dense | 23.5 | 26.9 | 35.3 | 5.4 | 13.9 | 22.0 | 1.752 | 819 |
| DRIP-ESC | 23.1 | 28.0 | 35.5 | 5.0 | 14.6 | 22.8 | 1.738 | 829 |
| DRIP | **24.7** | **30.9** | **36.4** | **6.4** | **15.1** | **23.2** | **1.708** | 823 |

MuSiQue 结论:

```text
DRIP 相对 embedding-only:
  R@5 H2: 23.5 -> 24.7
  KB Cov H2: 26.9 -> 30.9
  Has-answer: 5.4 -> 6.4
  ColdQ: 1.752 -> 1.708
```

增益较小但方向一致。原因是 MuSiQue 常需要 3/4 个 supports 同时驻留,
二跳 Pair Lease 只能保 A+B, 不能保证完整 support set。

## 15.3 当前最终故事线

论文主故事建议这样讲:

```text
1. Single-hop query shift is not enough to motivate DRIP:
   recency/access-history baselines are strong when evidence is directly visible.

2. The real gap is hidden multi-hop support residency:
   query-visible embedding admission writes A, but misses reusable B.

3. DRIP fixes this with two cache-management mechanisms:
   ESC discovers B conditioned on resident/easy A;
   Pair Lease keeps A+B resident as a support unit.

4. This is a no-topic-shift multi-hop cache result first.
   Topic shift + multi-hop should be a later combined benchmark, not the current
   main claim.
```

当前公式保持:

```text
E_ESC(q, B | A) = sim(phi(q,A), B) * link(A,B) * cue(q,B)

L_t(A), L_t(B) <- rho L_{t-1} + E_ESC(q,B|A)

D_t(d) = D_vis,t(d) + D_hid,t(d) + lambda_pair L_t(d)

P_t(d) = S_t(d) + D_t(d)

direct-first admission, bridge-leftover admission
```

一句话:

```text
DRIP is not a PPR method and not an ARC-style DRF/hubness method.
DRIP is evidence-conditioned support completion for multi-hop cache residency.
```

---

# 16. ARC-style 指标对齐 + Query-visible 完整复跑

参照 `Cache Mechanism for Agent RAG Systems.pdf`, cache-management 主指标不应只看
retrieval `R@5`, 而应优先看 cache 本身是否能回答问题。注意: ARC 论文的公式是
平均 item miss rate 的反面; 在 SQuAD/MMLU 这类单支持文档设置下, 它等价于
has-answer, 但在 multi-hop 多支持文档设置下需要拆成两个指标:

```text
miss(q_t) = # gold support docs not resident in cache

SupportCov = mean_q (# resident gold supports / # gold supports)  # ARC formula analog
HasAnswer = 1[miss(q_t) = 0]                                      # strict multihop answerability
ColdQ = mean_q miss(q_t)
```

对应关系:

```text
ARC paper: Has-answer Rate + AMAT
Our run:   support_coverage_rate + strict has_answer_rate
           + cold_fetches_per_query / latency / writes
```

`R@5` 仍保留, 但作为 retriever 层指标, 不再作为 cache management 的唯一主结果。

## 16.1 代码接入

`experiments/direct/run.py` 已加入:

```text
has_answer_rate
has_answer_h1 / has_answer_h2 / has_answer_per_window
support_coverage_rate
support_coverage_h1 / support_coverage_h2 / support_coverage_per_window
cold_fetches_per_query
cold_fetches_h1 / cold_fetches_h2 / cold_fetches_per_window
```

新增 direct branch 图表脚本:

```text
motivation/plotting/plot_direct_cache_metrics.py
```

输出:

```text
motivation/paper_figs/intro/fig_direct_query_visible_cache_metrics.pdf
motivation/paper_figs/intro/fig_direct_query_visible_cache_metrics.png
motivation/paper_figs/intro/direct_query_visible_cache_metrics.md
```

## 16.2 StreamingQA temporal, 100x50 完整 direct 复跑

命令:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/home/jyliu/miniconda3/envs/ljy_rag_ft/bin/python experiments/direct/run.py \
  --datasets streamingqa_temporal \
  --drift temporal \
  --n-windows 100 \
  --window-size 50 \
  --strategies LRU FIFO TinyLFU Proximity GPTCacheStyle DocArrival \
    AgentRAGCache_NoHub ARC DRIP-QueryVisible DRIP OnDemandFetch Oracle \
  --output qdirect_streamingqa_temporal_fig1_metrics_100w50_current.json
```

输出:

```text
experiments/direct/data/qdirect_streamingqa_temporal_fig1_metrics_100w50_current.json
```

| Strategy | Has-answer | SupportCov | ColdQ | R@5 H2 | KB Cov H2 | Writes |
|---|---:|---:|---:|---:|---:|---:|
| ARC | 13.7 | 13.7 | 0.86 | 4.4 | 4.2 | 1083 |
| DRIP-QueryVisible | 42.9 | 42.9 | 0.57 | 31.6 | 34.1 | 2936 |
| LRU | **47.6** | **47.6** | **0.52** | **33.0** | **35.1** | 2045 |
| FIFO | 46.1 | 46.1 | 0.54 | 32.8 | 35.0 | 2118 |
| OnDemandFetch | 83.4 | 83.4 | 0.17 | 60.0 | 76.3 | 0 |
| Oracle | 100.0 | 100.0 | 0.00 | 79.4 | 100.0 | 22140 |

解读:

```text
DRIP-QueryVisible 大幅超过 ARC:
  Has-answer: 13.7 -> 42.9
  R@5 H2:      4.4 -> 31.6
  KB Cov H2:   4.2 -> 34.1

但 LRU/FIFO 在 StreamingQA temporal 上也很强。
这说明 direct/query-visible shift 并不是 DRIP 独有胜场:
当 gold support 会被近期 query 直接访问时, recency/access-history 已经是强 baseline。
```

因此 direct 分支论文措辞应是:

```text
DRIP is drift-adaptive and strongly outperforms ARC under query-visible temporal shift,
but simple recency remains competitive when supports are directly observed.
```

不要写成:

```text
DRIP universally dominates all cache baselines on single-hop shift.
```

## 16.3 2Wiki comparison direct, 100x50, KB=6250

已有完整结果:

```text
experiments/hidden/data/qdirect_2wiki_comparison_100w50_kb6250_dense_current.json
```

| Strategy | Has-answer | SupportCov | ColdQ | R@5 H2 | KB Cov H2 | Writes |
|---|---:|---:|---:|---:|---:|---:|
| ARC | 19.5 | 38.8 | 1.22 | 34.5 | 34.4 | 9248 |
| DRIP-QueryVisible | **32.0** | **46.7** | **1.07** | **40.4** | **40.3** | **7657** |
| Oracle | 100.0 | 100.0 | 0.00 | 99.8 | 100.0 | 129350 |

这里 DRIP-QueryVisible 同时提升 answerability、support residency 和 R@5,
并且 writes 少于 ARC。

## 16.4 当前判断

ARC 论文主指标帮助我们把故事拆清楚:

```text
Query-visible shift:
  主要看 HasAnswer / ColdQ。
  DRIP-QueryVisible 明显强于 ARC, 但要承认 LRU/FIFO 也强。

Query-hidden multihop:
  不能只看 R@5。
  应重点看 hidden_B_hit, all_support_has_answer, SupportCov, ColdQ, writes。
  DRIP 的主创新应该放在这里: evidence-conditioned support completion.
```

最终讲法:

```text
ARC optimizes generic embedding-cache priority (DRF + hubness).
DRIP optimizes evidence visibility: when direct evidence is visible, use direct admission;
when support is hidden, complete the missing support and retain support pairs/groups.
```

---

# 17. ARC-style 精简 baseline + cache-size ablation

第 16 节的 StreamingQA 表里, `OnDemandFetch` 的 `Has-answer` 曾用
`effective_kb` 统计, 把 serve-time 临时 fetch 也算进了 cache。这个口径不适合
对标 ARC 论文, 因为 ARC 的 has-answer 应表示:

```text
answer can be served from the persistent hot cache without full-index access
```

已修正:

```text
experiments/direct/run.py
```

现在 residency metrics 使用 persistent `s.kb`, retrieval `R@K` 仍可使用
`effective_kb`。因此 OnDemand 的正确解读是:

```text
R@5 high, hot-cache HasAnswer low, ServeR huge.
```

也就是说它是一个 strong retrieval-time baseline, 但不是一个好的 cache-management
baseline。

## 17.1 本轮 baseline 集合

按 `Cache Mechanism for Agent RAG Systems.pdf` 主线收窄为:

```text
LRU
FIFO
TinyLFU        # LFU proxy
Proximity
GPTCacheStyle
AgentRAGCache_NoHub
ARC
DRIP
OnDemandFetch # extra retrieval-time baseline
```

主图不放 Oracle, 因为 Oracle 是 upper bound, 会压缩主方法和 baseline 的视觉差异。

## 17.2 输出

```text
motivation/paper_figs/intro/fig_direct_arcstyle_cache_ablation.pdf
motivation/paper_figs/intro/fig_direct_arcstyle_cache_ablation.png
motivation/paper_figs/intro/direct_arcstyle_cache_ablation.md
```

实验文件:

```text
StreamingQA:
experiments/direct/data/qdirect_streamingqa_arcstyle_100w50_kb200_current.json
experiments/direct/data/qdirect_streamingqa_arcstyle_100w50_kb400_current.json
experiments/direct/data/qdirect_streamingqa_arcstyle_100w50_kb800_current.json

2Wiki:
experiments/hidden/data/qdirect_2wiki_comparison_arcstyle_100w50_kb1250_dense_current.json
experiments/hidden/data/qdirect_2wiki_comparison_arcstyle_100w50_kb2500_dense_current.json
experiments/hidden/data/qdirect_2wiki_comparison_allbaselines_100w50_kb6250_dense_current.json
```

## 17.3 主容量结果

### StreamingQA temporal, KB=400

| Strategy | Hot Has-answer | R@5 H2 | KB Cov H2 | ColdQ | Writes | ServeR |
|---|---:|---:|---:|---:|---:|---:|
| LRU | **47.2** | **33.0** | **35.2** | **0.53** | 2037 | 0 |
| FIFO | 45.8 | 32.8 | 35.0 | 0.54 | 2107 | 0 |
| ARC | 13.7 | 4.4 | 4.2 | 0.86 | 1083 | 0 |
| DRIP | 42.9 | 31.6 | 34.1 | 0.57 | 2936 | 0 |
| OnDemandFetch | 21.6 | 60.0 | 76.3 | 0.78 | 0 | 216050 |

结论:

```text
DRIP >> ARC, but LRU/FIFO remain strongest or tied under this query-visible
temporal stream.
```

这说明 query-visible shift 不是最能凸显 DRIP 独特性的场景; 它证明 DRIP
能适应 shift, 但 recency baseline 在直接可见 support 上也很强。

### 2Wiki comparison direct, KB=6250

| Strategy | Hot Has-answer | SupportCov | R@5 H2 | KB Cov H2 | ColdQ | Writes | ServeR |
|---|---:|---:|---:|---:|---:|---:|---:|
| LRU | 25.9 | 45.7 | 34.8 | 34.9 | 1.09 | 1835 | 0 |
| FIFO | 25.9 | 45.5 | 34.5 | 34.5 | 1.09 | 1855 | 0 |
| ARC | 19.5 | 38.8 | 34.5 | 34.4 | 1.22 | 9248 | 0 |
| DRIP | **32.0** | **46.7** | **40.4** | **40.3** | **1.07** | 7657 | 0 |
| OnDemandFetch | 28.7 | 37.6 | 95.8 | 96.5 | 1.25 | 0 | 150400 |

结论:

```text
2Wiki direct is the better query-visible figure:
DRIP beats ARC and recency on hot-cache answerability/support residency,
while OnDemand only wins R@5 by paying full-index serve cost.
```

## 17.4 Cache-size ablation

### StreamingQA

| KB | LRU HasAns | FIFO HasAns | ARC HasAns | DRIP HasAns | OnDemand HasAns |
|---:|---:|---:|---:|---:|---:|
| 200 | 32.6 | 30.6 | 9.5 | 30.8 | 18.4 |
| 400 | 47.2 | 45.8 | 13.7 | 42.9 | 21.6 |
| 800 | 49.8 | 49.1 | 21.0 | 48.9 | 22.2 |

### 2Wiki

| KB | LRU HasAns | FIFO HasAns | ARC HasAns | DRIP HasAns | OnDemand HasAns |
|---:|---:|---:|---:|---:|---:|
| 1250 | 0.5 | 0.5 | 1.7 | 2.9 | 1.4 |
| 2500 | 3.1 | 3.4 | 4.8 | 8.3 | 5.8 |
| 6250 | 25.9 | 25.9 | 19.5 | 32.0 | 28.7 |

结论:

```text
Small cache: all methods struggle; recency often strong because query-visible
supports are recently accessed.

Moderate/large cache: DRIP's evidence-driven admission becomes useful,
especially on 2Wiki where each query needs multiple supports.

OnDemandFetch: high R@5 comes from full-index serve retrieval, but hot-cache
HasAnswer remains low and ServeR is 150k-216k, so it is not a replacement
policy.
```

论文图建议:

```text
Main direct figure: use 2Wiki KB=6250, because DRIP clearly beats ARC/recency
on hot-cache answerability and support coverage.

Appendix/direct sanity: include StreamingQA temporal, but phrase carefully:
DRIP is much better than ARC, yet LRU/FIFO are competitive when supports are
directly visible and temporally local.
```
