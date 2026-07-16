# DRIP V2 Evidence-Breadth Routing Experiment

日期：2026-07-11

## 1. 备份与实验隔离

- 正式 `DRIP` / `DRIPNOdetector` 和已有 V1 JSON 均未被覆盖；
- V2 通过 `--drip-ablation` 临时注册，实验名 `DRIP-MentionRouted`；
- V1 的 evidence、serve/demand ledger、write budget 和 replacement gate 保持不变；
- V2 只增加 candidate-local IDF mention credit，并按 window evidence breadth 在
  semantic 与 recency victim order 之间路由；
- `DATA_SEED=42` 固定数据样本，`EXPERIMENT_SEED` 只改变 stream/初始次序，避免把
  “换了一批数据”误报成随机种子稳定性。

简单 query-level L1 normalization 的负结果独立记录在
`DRIP_EVIDENCE_NORMALIZATION_2026-07-11.md`，没有并入 V2。

## 2. 数据规模

| Workload | Cold pool | Hot tier (10%) | Stream | Drift |
|---|---:|---:|---:|---|
| StreamingQA | 29,819 | 2,982 | 50 x 100 | real temporal order |
| HotpotQA comparison | 30,454 | 3,045 | 50 x 50 | controlled full-gradual |
| 2Wiki comparison | 37,214 | 3,721 | 50 x 50 | controlled full-gradual |
| Mind2Web agent diagnostic | 4,752 | 475 | 20 x 25 | cluster shift |
| 2Wiki-simple stress diagnostic | 140,947 | 14,095 | 100 x 50 | gradual mixed-type |

主结果是 3--4 万文档级算法验证，并补充一个 14 万文档 stress diagnostic；它们仍
不是百万文档在线服务实验。首次 2Wiki 运行约
231 秒，其中约 150 秒用于生成 37,214 个 embedding；缓存 embedding 后，同一组策略
约 68--69 秒。StreamingQA/HotpotQA 完整 baseline 运行约 62/53 秒。运行快主要因为
所有 embedding 已预计算，策略在内存 NumPy 数组上执行，并未调用 LLM、网络或磁盘。

## 3. 五随机种子主结果

下表为 seed 42--46 的 `mean +/- sample std`。Recall@5 为全窗口平均。

| Dataset | Method | HasAns up | AMAT down | Recall@5 up | Repl. down |
|---|---|---:|---:|---:|---:|
| StreamingQA | LRU | 51.56 +/- 0.91 | 5.845 +/- 0.091 | 42.16 +/- 0.43 | 1592 +/- 18 |
|  | TinyLFU | 51.82 +/- 0.89 | 5.820 +/- 0.089 | 42.26 +/- 0.64 | **1584 +/- 12** |
|  | ARC | 42.54 +/- 1.05 | 6.746 +/- 0.108 | 33.96 +/- 0.97 | 4210 +/- 66 |
|  | DRIP V1 | 49.52 +/- 1.03 | 6.048 +/- 0.101 | 40.64 +/- 0.59 | 2437 +/- 26 |
|  | **DRIP-EBR** | **52.72 +/- 1.19** | **5.728 +/- 0.121** | **42.92 +/- 0.73** | 2358 +/- 20 |
| HotpotQA | LRU | 23.84 +/- 2.84 | 8.616 +/- 0.284 | 39.22 +/- 1.72 | 891 +/- 36 |
|  | TinyLFU | 23.90 +/- 3.45 | 8.611 +/- 0.345 | 39.33 +/- 2.09 | **887 +/- 37** |
|  | ARC | 20.38 +/- 1.20 | 8.963 +/- 0.118 | 38.69 +/- 0.65 | 4803 +/- 134 |
|  | DRIP V1 | 31.22 +/- 2.58 | 7.878 +/- 0.257 | 44.75 +/- 1.52 | 2061 +/- 59 |
|  | **DRIP-EBR** | **35.56 +/- 2.05** | **7.445 +/- 0.205** | **47.46 +/- 1.14** | 2654 +/- 81 |
| 2Wiki | LRU | 30.76 +/- 4.37 | 7.923 +/- 0.437 | 43.29 +/- 2.21 | 889 +/- 29 |
|  | TinyLFU | 31.12 +/- 4.00 | 7.888 +/- 0.402 | 43.57 +/- 1.83 | **885 +/- 23** |
|  | ARC | 15.20 +/- 1.22 | 9.478 +/- 0.121 | 32.06 +/- 1.16 | 6290 +/- 142 |
|  | DRIP V1 | 34.02 +/- 3.27 | 7.601 +/- 0.327 | 45.34 +/- 1.61 | 2105 +/- 43 |
|  | **DRIP-EBR** | **38.00 +/- 2.18** | **7.200 +/- 0.219** | **47.55 +/- 1.40** | 2776 +/- 66 |

配对 Has-Answer 检验（同一 seed、双侧 paired t-test）：

| Dataset | vs. TinyLFU delta | p-value | vs. DRIP V1 delta | p-value |
|---|---:|---:|---:|---:|
| StreamingQA | +0.90 pt | 0.01225 | +3.20 pt | 0.00019 |
| HotpotQA | +11.66 pt | 0.00007 | +4.34 pt | 0.00007 |
| 2Wiki | +6.88 pt | 0.00316 | +3.98 pt | 0.00367 |

五个种子仍然较少，p-value 只能作为稳定性证据，不能代替更多数据集和真实系统验证。

## 4. Cache-ratio 边界

| Dataset | Hot ratio | LRU HasAns | TinyLFU HasAns | DRIP V1 HasAns | DRIP-EBR HasAns |
|---|---:|---:|---:|---:|---:|
| StreamingQA | 1% | **41.8** | 17.2 | 40.0 | 35.9 |
|  | 5% | 49.8 | 49.9 | 49.0 | **50.9** |
|  | 10% | 52.4 | 52.6 | 50.6 | **53.1** |
| HotpotQA | 1% | 0.4 | 0.4 | 1.2 | **1.9** |
|  | 5% | 9.6 | 9.8 | 16.0 | **20.3** |
|  | 10% | 22.3 | 22.2 | 30.1 | **34.4** |

StreamingQA 1% 是清楚的反例：容量极小时，维护互补 semantic evidence 的机会成本
过高，LRU 的纯 recency 更适合。5%--10% 时，V2 才有空间同时保留近期文档和 evidence
bundle。HotpotQA 的优势随容量增加而扩大，符合 direct evidence breadth 的设计预期。

## 5. Agent workload 诊断

Mind2Web 三种子结果中，ARC 的 Has-Answer 为 `50.73 +/- 4.90`，V2 为
`45.87 +/- 3.76`；V2 replacement 为 `427 +/- 31`，明显低于 ARC 的
`960 +/- 43`，但 quality 尚未超过 ARC。candidate-local IDF 已消除了共享网站名导致的
虚假 mention，V2 基本回到 V1，而不是在 agent workload 上形成质量 SOTA。因此该
数据集应作为 Pareto/迁移诊断，不应写成主胜例。

## 6. 14 万文档 Mixed-Type Stress Test

`2Wiki-simple` 混合 comparison、inference 和 compositional query。它使用已有的
140,947-document / 40,000-query embedding cache，测试 5,000 个流事件，KB/pool=10%。

| Method | HasAns up | AMAT down | Repl. down |
|---|---:|---:|---:|
| LRU | 56.1 | 5.39 | 1,609 |
| TinyLFU | **57.1** | **5.29** | **1,594** |
| DRIP V1 | 51.8 | 5.82 | 4,124 |
| DRIP-EBR | 50.4 | 5.96 | 5,115 |

该结果否定了“只要标题 mention 多就使用 semantic eviction”的过宽版本。V2 在
100 个窗口全部选择 semantic route，尽管 mixed workload 的 inference/compositional
support 并不完全 query-visible。进一步诊断显示：至少两个候选标题被提及的 query
比例在 Hotpot direct、2Wiki direct、2Wiki-simple 中分别约为 96.3%、92.7%、88.3%，
无法可靠区分 direct 与 hidden evidence；同窗和相邻窗口 candidate-reuse 也不能解释
差异。因此不继续添加 reuse threshold。该 stress test 应作为 hidden-evidence 边界：
后续需要 bridge/ESC completion，而不是用 direct router 强行覆盖。

四策略完整运行耗时约 204 秒；单独 V2 约 92 秒。它说明当前实现可以运行到 14 万
文档，但性能和算法质量都还不足以支持“通用 mixed-workload SOTA”的表述。

## 7. FAISS 单线程 CPU 微基准

使用 384 维 `IndexFlatIP`，1,000 个单 query、top-5 search，200 次
`IndexIDMap2 remove+add`。不含网络、磁盘、LLM 和分布式同步。

| Dataset | Cold search mean / p99 | Hot search mean / p99 | Replacement mean / p99 |
|---|---:|---:|---:|
| StreamingQA | 2.823 / 2.976 ms | 0.175 / 0.184 ms | 0.230 / 0.245 ms |
| HotpotQA | 3.177 / 5.085 ms | 0.179 / 0.189 ms | 0.260 / 0.278 ms |
| 2Wiki | 3.950 / 5.912 ms | 0.220 / 0.233 ms | 0.327 / 0.340 ms |
| 2Wiki-simple 140k | 13.788 / 35.688 ms | 1.306 / 1.478 ms | 2.591 / 2.888 ms |

在 3 万文档规模，10% hot tier 的精确搜索比 full cold index 快约 14--18 倍；在
14 万规模约快 10.6 倍。每次写入
至少移动一个 384 维 float32 向量（1,536 bytes），且真实 ANN/分布式索引通常还有
图更新、持久化和副本同步开销。因此 replacements 不是装饰性指标，但本微基准也
不能替代完整 end-to-end serving experiment。

## 8. 结论与投稿口径

当前最可信的论文结论是：

1. direct evidence breadth 能在线判断 semantic bundle protection 是否值得启用；
2. 同一 admission gate 下，自适应 victim bias 在三套主 workload 的五个 seed 上
   稳定提高 answerability；
3. V2 的代价是比 LRU/TinyLFU 更多 replacements，因此贡献应表述为更好的
   quality--write Pareto，而不是最低写入；
4. 极小 temporal cache、Mind2Web 和 mixed hidden evidence 是现有边界；百万级 ANN
   与真实服务仍需补实验。

结果文件位于 `experiments/direct/data/v2*.json`，FAISS 原始结果位于
`docs/experiments/faiss_*.json`。
