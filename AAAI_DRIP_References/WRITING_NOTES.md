# DRIP AAAI 写作约束

本文档记录对核心参考论文的写法抽取，供后续实验结果回填时保持全文一致。

## 参考论文的可复用结构

- **Mnemosyne (AAAI-26)**：先用一张图给出两个可观察 cache miss 原因，再提出
  一个具体研究问题，随后用两个模块回答。避免在引言中提前展开所有实现细节。
- **LogicRAG (AAAI-26)**：Problem Statement 只定义输入、输出和主要缺口；方法开头
  先给完整三阶段流程，再逐阶段给公式。每个公式服务一个决策。
- **Non-Stationary KG-RAG (AAAI-25)**：把 non-stationarity 和 multi-objective cost
  分成两个挑战，并逐一对应方法部件。下游反馈只在请求完成后更新策略。
- **DDG-DA (AAAI-22)**：明确限定 predictable drift，不把部分可预测的实验场景
  外推为所有漂移。DRIP 同样应区分现实 domain-mixture shift、受控 recurring shift
  和不可预测边界。

## DRIP 当前写作规则

1. 引言只走一条证据链：domain mixture changes，within-domain evidence is reusable，
   historical residency mismatches after a shift，DRIP repairs placement under cost。
2. 两个场景各用一个段首粗体句和一张单栏图；不在后文重复图注中的数据集审计。
3. 方法按四阶段写：offline directory，current-query retrieval，current-request
   downstream feedback，switching-cost-aware placement。
4. `CurrentRequestUtilityEstimator` 不读取先前请求。历史只以衰减后的 document-level
   sufficient statistics 留在 cache manager 中，不能写成整个系统完全无状态。
5. 当前 feedback 实验使用 official response embedding 时，必须称为
   `reference-response attribution proxy`；只有真实生成响应才能称为 generator feedback。
6. 新实验完成前不把 feedback-aware DRIP 的收益写入 Abstract、Conclusion 或主结果。
   回填时同时报告 persistent hit、service recall、writes、routed reads 和 attribution
   accuracy，不能只报告 retrieval recall。
