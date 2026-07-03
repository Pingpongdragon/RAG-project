# DOC 3: Query Distribution Shift 文献调研

> 生成时间: 2026-04-19
> 目的: 调研 query distribution shift 的构造方法及相关论文，评估是否可为我们的 motivation 实验提供引用支撑

---

## 一、核心结论

**我们的 query distribution shift 构造方法有先例可引，但没有完全相同场景的工作。**

我们的实验通过 **语义聚类 → head/tail 分组 → 97/3 到 3/97 比例切换** 来构造 query topic 漂移，用于评估 KB 维护策略。这种构造思路最直接的先例是 **MS-Shift**（ECIR 2023），它也通过聚类在 MS MARCO 内构造了 query-based distribution shift，只不过它用来评估 neural retriever 的泛化能力，而非 KB 管理。

### 可引用性总结

| 论文 | 可引？ | 引用角色 | 契合度 |
|---|---|---|---|
| **MS-Shift** (ECIR 2023) | ✅ 强烈推荐 | 构造方法先例 | ⭐⭐⭐⭐⭐ |
| **TCR** (arXiv 2024) | ✅ 推荐 | query shift 形式化定义 | ⭐⭐⭐⭐ |
| **StreamingQA** (arXiv 2022) | ✅ 推荐 | 时间维度知识漂移对比 | ⭐⭐⭐ |
| **TemporalWiki** (EMNLP 2022) | ✅ 可引 | 知识漂移基准对比 | ⭐⭐⭐ |
| **BEIR** (NeurIPS 2021) | ✅ 可引 | 跨域分布偏移对比 | ⭐⭐ |
| **Gama et al.** (已引) | ✅ 已引 | concept drift 分类学 | ⭐⭐⭐ |

---

## 二、详细论文分析

---

### 2.1 MS-Shift（最直接先例）⭐⭐⭐⭐⭐

**论文**: *MS-Shift: An Analysis of MS MARCO Distribution Shifts on Neural Retrieval*
**作者**: Simon Lupart, Thibault Formal, Stéphane Clinchant
**发表**: ECIR 2023 (Accepted)
**arXiv**: 2205.02870
**引用数**: ~18

#### 核心内容

MS-Shift 在 MS MARCO 数据集**内部**构造了三种基于 query 的 distribution shift：

1. **Query-Semantic Shift**: 用 sentence embedding 对 query 进行聚类，然后选择语义上不同的 cluster 作为 train/test split。这与我们用 K-Means 聚类然后分 head/tail 的做法**几乎相同**。
2. **Query-Intent Shift**: 按 query 的搜索意图（navigational / informational / transactional）分组，构造 intent 维度的漂移。
3. **Query-Length Shift**: 按 query 长度分组，构造长度维度的漂移。

#### 关键引用点

> "we build three **query-based distribution shifts** within MS MARCO (query-semantic, query-intent, query-length), which are used to evaluate the three main families of neural retrievers"

> "Overall, our study demonstrates that it is possible to **design more controllable distribution shifts** as a tool to better understand generalization of IR models."

#### 与我们的异同

| 维度 | MS-Shift | 我们的实验 |
|---|---|---|
| **构造方法** | 聚类 + train/test split | 聚类 + head/tail + 窗口流 |
| **漂移类型** | 静态 split（一次性） | 动态流（渐进/突变） |
| **评估对象** | Neural retriever 泛化 | KB 维护策略 |
| **漂移维度** | 语义/意图/长度 | 语义（topic） |
| **数据集** | MS MARCO | HotpotQA / 2Wiki / MuSiQue |
| **下游任务** | Passage Retrieval | Multi-hop QA |

#### 能否直接引用？

**✅ 可以且强烈推荐引用。** MS-Shift 是我们构造方法最直接的先例——用语义聚类在单一数据集内部制造可控的 query distribution shift。我们只是将其从"一次性 train/test split"扩展到了"动态窗口流"，并将评估对象从 retriever model 换成了 KB management strategy。

#### 建议引用写法

> Following the query-semantic shift construction of Lupart et al. (2023), we cluster queries by embedding similarity and construct a controlled distribution drift between head and tail topic groups, simulating scenarios where user interests shift over time.

---

### 2.2 TCR — Test-time Adaptation for Cross-modal Retrieval with Query Shift ⭐⭐⭐⭐

**论文**: *Test-time Adaptation for Cross-modal Retrieval with Query Shift*
**作者**: Haobin Li, Peng Hu, Qianjun Zhang, Xi Peng, Xiting Liu, Mouxing Yang
**单位**: 四川大学
**arXiv**: 2410.15624 (2024, 22 pages)
**引用数**: ~12

#### 核心内容

TCR 首次形式化定义了 **query shift** 问题：

> "query shift refers to the **online query stream originating from the domain that follows a different distribution with the source one.**"

该论文研究的场景是**跨模态检索**（image-text retrieval）中的 query 分布偏移。当 query（图片或文本）来自与训练数据不同的分布时，预训练的检索模型性能下降。TCR 提出了一种 test-time adaptation 方法来在线适应 query shift。

#### 方法概述

TCR 发现 query shift 会导致两个问题：
1. **模态内均匀性下降**（Intra-modality Uniformity ↓）：query embedding 变得更加集中，难以区分
2. **模态间间距增大**（Inter-modality Gap ↑）：query 与 gallery 的 embedding 偏离

基于此，TCR 提出：
- **Query Prediction Refinement**: 为每个 query 选最近邻 candidate，过滤不相关样本
- **Intra-modality Uniformity Learning (L_MU)**: 增强 query embedding 的散布程度
- **Inter-modality Gap Learning (L_MG)**: 将 target domain 的模态间距纠正到 source domain 水平
- **Noise-robust Adaptation (L_NA)**: 自适应阈值过滤噪声

#### 实验设置

- **Query Shift (QS)**: 仅 query 来自不同分布（通过 16 种图像/15 种文本 corruption 构造）
- **Query-Gallery Shift (QGS)**: query 和 gallery 都来自不同域（跨数据集评估）
- 基础模型: CLIP / BLIP
- 数据集: COCO-C, Flickr-C, Fashion-Gen, CUHK-PEDES, ICFG-PEDES, Nocaps

#### 与我们的异同

| 维度 | TCR | 我们的实验 |
|---|---|---|
| **问题定义** | query shift = 在线 query 流分布不同于 source | query shift = 用户关注 topic 从 head 漂移到 tail |
| **领域** | 跨模态检索（image-text） | 文本检索 / multi-hop QA |
| **漂移构造** | 对 query 加 corruption / 跨域 | 语义聚类 + 比例切换 |
| **适应方式** | 调整 encoder 参数（TTA） | 调整 KB 内容（文档替换） |
| **目标** | 让 retrieval model 鲁棒 | 让 KB 跟上 query 需求 |

#### 能否直接引用？

**✅ 推荐引用，但需说明场景差异。** TCR 的 "query shift" 定义与我们的 motivation 高度吻合——都是在线 query 流的分布发生偏移。区别在于 TCR 通过修改 model 参数来适应，而我们通过修改 KB 内容来适应。两者是**同一问题的不同层面的解法**。

#### 建议引用写法

> Li et al. (2024) formally define query shift as "the online query stream originating from a domain that follows a different distribution from the source," and propose test-time model adaptation. Our work addresses a complementary aspect: rather than adapting the retrieval model, we study how to adapt the knowledge base itself to evolving query demands.

---

### 2.3 StreamingQA — 时间维度的知识漂移基准 ⭐⭐⭐

**论文**: *StreamingQA: A Benchmark for Adaptation to New Knowledge over Time in Question Answering Models*
**作者**: Adam Liška, Tomáš Kočiský, Elena Gribovskaya, et al. (DeepMind)
**arXiv**: 2205.11388 (2022)

#### 核心内容

StreamingQA 构造了一个**基于时间戳的 QA 基准**：
- 14 年的新闻文章，按时间排列
- 每个季度生成 question（人工 + 自动）
- 按季度评估：模型读到新文章后，能否回答关于新知识的问题
- 核心发现：半参数模型（retrieval-augmented）可以通过添加新文档快速适应；但底层 LM 过时则性能不佳

#### 与我们的关系

StreamingQA 关注的是**知识侧的时效性**——随着时间推移，新事件发生，旧的答案过期。这是**supply-side drift**（文档变了）。

我们关注的是**demand-side drift**——知识库是固定的（或可更新的），但**用户关注的 topic 变了**。

#### 引用角色

引用为"现有时序 QA benchmark 关注知识过期问题（supply-side），而忽略了 query 需求的漂移（demand-side），这正是我们要填补的空白"。

---

### 2.4 TemporalWiki — 知识持续更新基准 ⭐⭐⭐

**论文**: *TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models*
**作者**: Joel Jang, Seonghyeon Ye, et al.
**发表**: EMNLP 2022
**arXiv**: 2204.14211

#### 核心内容

TemporalWiki 用 Wikipedia 连续快照的 diff（差异）来构造持续学习基准：
- 训练数据：Wikipedia 快照之间的差异
- 评估：模型能否保留旧知识 + 学习新/更新知识
- 发现：在 diff 数据上做 continual learning 比在完整快照上训练更高效（12 倍计算节省）

#### 与我们的关系

同 StreamingQA，TemporalWiki 关注的是 **知识本身的演变**（Wikipedia 条目被编辑），而非 query 分布的变化。

#### 引用角色

与 StreamingQA 一起引用，强调现有 benchmark 只覆盖了 supply-side 的知识漂移。

---

### 2.5 BEIR — 跨域检索泛化基准 ⭐⭐

**论文**: *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*
**作者**: Nandan Thakur, Nils Reimers, Andreas Rücklé, et al.
**发表**: NeurIPS 2021 (Datasets and Benchmarks)
**arXiv**: 2104.08663

#### 核心内容

BEIR 将 18 个不同领域的 IR 数据集整合为统一的 zero-shot evaluation benchmark，评估 retrieval model 的 OOD（out-of-distribution）泛化能力。不同数据集之间天然存在 domain shift。

#### 与我们的关系

BEIR 的 distribution shift 是**跨数据集/跨领域**的，是一种粗粒度的 shift。我们的 shift 是**单一数据集内部、同一领域内的 topic shift**，更加细粒度和可控。

#### 引用角色

引用为"BEIR 衡量的是跨域泛化，而我们研究的是同域内的 query topic 漂移"，突出我们问题的独特性。

---

## 三、现有工作的空白与我们的定位

### 3.1 文献全景

```
                        Distribution Shift in Retrieval/QA
                                    |
                 ┌──────────────────┼──────────────────┐
                 |                  |                  |
           知识侧漂移          跨域漂移           Query 侧漂移
         (Supply-side)      (Domain shift)      (Demand-side)
                 |                  |                  |
          StreamingQA            BEIR            MS-Shift (ECIR'23)
         TemporalWiki                            TCR (2024)
         (时间驱动)                                    |
                                              ┌───────┼───────┐
                                              |               |
                                       Retriever 适应     KB 内容适应
                                        (TCR 做了)      (我们的工作)
                                                        ← 空白！
```

### 3.2 我们填补的空白

**没有任何现有工作研究：在 query topic 分布发生漂移时，如何维护/更新 RAG 系统的知识库。**

- MS-Shift 构造了 query shift，但只用来评估 retriever model 的鲁棒性
- TCR 研究了 query shift 下的在线适应，但通过修改 model 参数而非 KB 内容
- StreamingQA / TemporalWiki 只关注知识本身的时效性
- BEIR 是跨域泛化，不是 topic drift

我们的实验是第一个：
1. **在 multi-hop QA 数据集上构造 query topic drift**
2. **比较不同 KB 维护范式**（Static / DocArrival / KnowledgeEdit / DRIP-Dense / Oracle）在 query drift 下的表现
3. **证明 demand-side 策略优于 supply-side 策略**

---

## 四、建议的论文写法

### 4.1 在 Related Work 或 Experiment 中引用

建议在 motivation 实验的 dataset construction 部分这样写：

> **Query Distribution Shift Construction.**
> To simulate scenarios where user information needs evolve over time,
> we construct a controlled query distribution drift following the
> query-semantic shift methodology of Lupart et al. (2023).
> Specifically, we cluster all queries into $K$ groups using K-Means
> on sentence embeddings, designate the largest clusters as "head"
> topics and the rest as "tail" topics, then build a query stream
> where the head-to-tail ratio shifts from 97:3 to 3:97 at the
> midpoint. This simulates a sudden topic drift—analogous to the
> concept drift taxonomy of Gama et al. (2014)—where the dominant
> query theme changes abruptly.
>
> While prior work on query distribution shift focuses on evaluating
> retriever generalization (Lupart et al., 2023; Thakur et al., 2021)
> or adapting retrieval models online (Li et al., 2024), our goal is
> different: we study how different **knowledge base maintenance
> strategies** respond to shifting query demands—a question that has not
> been addressed in the literature. Temporal QA benchmarks such as
> StreamingQA (Liška et al., 2022) and TemporalWiki (Jang et al., 2022)
> address knowledge staleness (supply-side drift) but do not consider
> query-side topic shift.

### 4.2 BibTeX 条目

```bibtex
@inproceedings{lupart2023msshift,
  title={MS-Shift: An Analysis of MS MARCO Distribution Shifts on Neural Retrieval},
  author={Lupart, Simon and Formal, Thibault and Clinchant, St{\'e}phane},
  booktitle={Proceedings of the 45th European Conference on Information Retrieval (ECIR)},
  year={2023},
  note={arXiv:2205.02870}
}

@article{li2024tcr,
  title={Test-time Adaptation for Cross-modal Retrieval with Query Shift},
  author={Li, Haobin and Hu, Peng and Zhang, Qianjun and Peng, Xi and Liu, Xiting and Yang, Mouxing},
  journal={arXiv preprint arXiv:2410.15624},
  year={2024}
}

@article{liska2022streamingqa,
  title={StreamingQA: A Benchmark for Adaptation to New Knowledge over Time in Question Answering Models},
  author={Li{\v{s}}ka, Adam and Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}} and Gribovskaya, Elena and Terzi, Tayfun and Sezener, Eren and Agrawal, Devang and de Masson d'Autume, Cyprien and Scholtes, Tim and Zaheer, Manzil and Young, Susannah and others},
  journal={arXiv preprint arXiv:2205.11388},
  year={2022}
}

@inproceedings{jang2022temporalwiki,
  title={TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models},
  author={Jang, Joel and Ye, Seonghyeon and Lee, Changho and Yang, Sohee and Shin, Joongbo and Han, Janghoon and Kim, Gyeonghun and Seo, Minjoon},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022},
  note={arXiv:2204.14211}
}

@inproceedings{thakur2021beir,
  title={BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author={Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna},
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
  year={2021},
  note={arXiv:2104.08663}
}
```

---

## 五、对 MS-Shift 和 TCR 的深入可用性分析

### 5.1 MS-Shift 能否直接用？

**能用，而且应该用。** 理由：

1. **方法论先例**: MS-Shift 的 query-semantic shift 构造与我们几乎一致——都是用 embedding 聚类将 query 分组，然后制造 train/test（或 H1/H2）之间的分布差异。这为我们的方法提供了学术合法性。

2. **可控性论证**: MS-Shift 的核心贡献之一就是论证了"在单一数据集内构造可控分布偏移是可行且有价值的"（"it is possible to design more controllable distribution shifts as a tool"）。我们可以直接引用这个结论来支持我们的实验设计。

3. **不需要使用他们的数据/代码**: 我们只需引用其构造思路。我们的数据集（HotpotQA / 2Wiki / MuSiQue）和下游任务（multi-hop QA + KB 管理评估）完全不同。

**局限**: MS-Shift 是静态的 train/test split，没有"窗口流"和"逐步漂移"的概念。我们的扩展（时间窗口 + 渐进/突变模式）是新的贡献。

### 5.2 TCR 能否直接用？

**能用，引用其定义和问题定位。** 理由：

1. **形式化定义**: TCR 给出了 query shift 的第一个正式定义（"online query stream from a different distribution"），这正是我们研究的问题。

2. **互补关系**: TCR 通过修改 model 参数来适应 query shift（TTA），我们通过修改 KB 内容来适应。两者解决的是同一问题的不同层面。可以这样引用：
   - TCR: "模型侧适应"
   - 我们: "数据侧适应"（更轻量、无需训练）

3. **场景差异需要说明**: TCR 是 image-text 跨模态检索，我们是 text-only multi-hop QA。TCR 用 corruption（加噪声）构造 shift，我们用 topic clustering 构造。

**局限**: TCR 的 query shift 是通过对 query 加 corruption（Gaussian noise, blur, OCR error 等）构造的，更像是"噪声"而非"topic 变化"。我们的 topic-level shift 更接近真实场景中的用户兴趣迁移。

---

## 六、最终推荐引用方案

### 必引（2篇）
- **MS-Shift** → 构造方法先例 + 可控性论证
- **Gama et al. 2014** → concept drift 分类学（已引）

### 强烈推荐引（1篇）
- **TCR** → query shift 形式化定义 + 互补关系

### 建议引（2篇）
- **StreamingQA** → 知识时效性基准（supply-side 对比）
- **TemporalWiki** → 同上

### 可选引（1篇）
- **BEIR** → 跨域泛化基准（粗粒度 shift 对比）

---

*文档结束*
