# AAAI DRIP References

> 当前服务器没有挂载 Windows `D:` 盘。本目录已移动到项目内：
> `/home/jyliu/RAG-project/AAAI_DRIP_References`（实际对应 `/data/jyliu/RAG-project/AAAI_DRIP_References`）。
> 需要在 Windows 的 `D:\AAAI_DRIP_References` 使用时，请通过 IDE/SFTP 同步整个目录。

## 建议阅读顺序

### 第一组：最直接的 DRIP 写作参考

| 优先级 | 论文 | DRIP 最值得借鉴的部分 |
|---|---|---|
| 1 | [Mnemosyne (AAAI-26)](01_RAG_Cache/2026_Mnemosyne_Cache_Hit_Order_Fitting.pdf) | 与 cache-aided RAG 最接近。重点看 Introduction 如何从有限缓存、请求顺序和 cache miss 推出研究问题，以及效率/效果联合实验。注意它调整 multi-hop 查询执行顺序，并不是跨窗口 evidence replacement。 |
| 2 | [LogicRAG (AAAI-26)](01_RAG_Cache/2026_LogicRAG_Adaptive_Reasoning_Structures.pdf) | 参考 Problem Statement、三阶段方法叙述、框架图、主公式与 ablation 的组织方式。可借鉴它对 direct semantic retrieval 局限的论证，但不能把其逻辑图方法等同于 DRIP。 |
| 3 | [Non-Stationary KG-RAG (AAAI-25)](02_Drift_Nonstationary/2025_MAB_RAG_Nonstationary_KG.pdf) | 直接支撑“RAG 部署环境非平稳、策略需要在线适应”的叙事。重点看多目标 bandit 如何把质量与响应成本写成决策问题。 |
| 4 | [SegMem-RAG (AAAI-26)](01_RAG_Cache/2026_SegMem-RAG_Adaptive_Memory.pdf) | 支撑 experience-driven adaptive memory 和跨请求持续更新。适合 Related Work 的 adaptive memory 小节，以及与静态 corpus/retrieval policy 的区别。 |

### 第二组：Detector 与未来预取

| 优先级 | 论文 | DRIP 最值得借鉴的部分 |
|---|---|---|
| 5 | [DDG-DA (AAAI-22)](02_Drift_Nonstationary/2022_DDG-DA_Predictable_Concept_Drift.pdf) | 最重要的预取理论参考：不只检测已经发生的 drift，而是预测未来分布。可用于论证 evidence-transition predictor 的必要性。领域不是 RAG，引用时应表述为可迁移思想。 |
| 6 | [Autonomous Drift Threshold (AAAI-26)](02_Drift_Nonstationary/2026_Autonomous_Drift_Threshold.pdf) | 对当前 CUSUM 固定阈值问题最直接。参考其“检测阈值应服务下游性能，而不是独立调统计灵敏度”的论证和实验结构。 |
| 7 | [OBAL (AAAI-24)](02_Drift_Nonstationary/2024_OBAL_Multistream_Concept_Drift.pdf) | 多用户/多 agent 流中的异步漂移和跨流迁移参考。适合讨论共享 cache 中不同用户需求不能简单合并的问题。 |

### 第三组：Agent Memory 与系统边界

| 优先级 | 论文 | DRIP 最值得借鉴的部分 |
|---|---|---|
| 8 | [MemGuide (AAAI-26)](03_Agent_Memory/2026_MemGuide_Intent_Driven_Memory.pdf) | 说明 semantic similarity 不足以决定记忆价值，任务 intent 和 missing information 同样重要。可为 hidden evidence 和 agent trace 扩展提供动机。 |
| 9 | [Short-Term, Episodic, and Semantic Memory (AAAI-23)](03_Agent_Memory/2023_Short_Term_Episodic_Semantic_Memory.pdf) | 参考有限容量下 encode/store/forget 的问题定义，以及短期和长期记忆层次。它是 agent memory，不是 RAG hot-tier cache。 |
| 10 | [SubGCache (AAAI-26)](01_RAG_Cache/2026_SubGCache_Subgraph_KV_Cache.pdf) | 参考系统论文如何报告 latency、cache reuse 和跨数据集实验。它缓存的是 subgraph prompt 的 KV states，不是 DRIP 的 evidence documents，只能作为相邻系统工作。 |

## 对当前论文最有用的对应关系

| DRIP 章节 | 首选参考 |
|---|---|
| Introduction / Motivation | Mnemosyne, Non-Stationary KG-RAG, SegMem-RAG |
| Problem Formulation | LogicRAG, Non-Stationary KG-RAG |
| Drift Detector | Autonomous Drift Threshold, OBAL |
| Predictive Prefetch | DDG-DA；算法细节还应补充非 AAAI 的 MITHRIL/Palpatine |
| Cache Manager | Mnemosyne；注意其 cache object 和 DRIP 不同 |
| Agent / Hidden Evidence | MemGuide, Short-Term/Episodic/Semantic Memory |
| Experiments | Mnemosyne, LogicRAG, SubGCache |

## 写作时需要避免的混淆

1. `KV cache`、`answer cache`、`query-result cache` 和 `evidence hot tier` 是不同缓存对象。
2. Concept drift 论文通常研究预测标签关系变化；DRIP 当前主要研究 query-demand / evidence-working-set shift。
3. Mnemosyne 在一个 multi-hop query 内重排子查询，DRIP 则跨 query windows 管理常驻 evidence。
4. DDG-DA 能支持“预测未来分布”的思想，但不能直接证明 Markov evidence prefetcher 有效。
5. AAAI-26 论文适合作为最新写作和实验参照；正式 Related Work 仍需要补充系统缓存、预取和数据流领域的经典论文。

## 官方页面

- LogicRAG: https://ojs.aaai.org/index.php/AAAI/article/view/40278
- Mnemosyne: https://ojs.aaai.org/index.php/AAAI/article/view/40310
- SegMem-RAG: https://ojs.aaai.org/index.php/AAAI/article/view/40320
- SubGCache: https://ojs.aaai.org/index.php/AAAI/article/view/40827
- Non-Stationary KG-RAG: https://ojs.aaai.org/index.php/AAAI/article/view/33380
- DDG-DA: https://ojs.aaai.org/index.php/AAAI/article/view/20327
- Autonomous Drift Threshold: https://ojs.aaai.org/index.php/AAAI/article/view/39586
- OBAL: https://ojs.aaai.org/index.php/AAAI/article/view/29590
- MemGuide: https://ojs.aaai.org/index.php/AAAI/article/view/40313
- Short-Term/Episodic/Semantic Memory: https://ojs.aaai.org/index.php/AAAI/article/view/25075
