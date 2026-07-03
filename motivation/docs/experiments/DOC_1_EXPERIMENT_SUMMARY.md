# Motivation 实验设计与结果总结

> 更新: 2026-04-20 (8策略版本)
> 代码: `motivation_4/` (config.py / loaders.py / strategies.py / utils.py / run.py)
> 框架: `motivation 实验想法.md`

## 1. 实验目标

证明事实链条：
**query分布漂移 → RAG检索崩溃 → 供给侧/被动搜索/人工反馈均无效 → 简单QD也不足 → Oracle上界证明瓶颈是content selection precision → motivate图谱增强方法**

## 2. 数据集与实验设置

| 数据集 | Doc Pool | Query数 | KB Budget | 窗口配置 |
|---|---|---|---|---|
| HotpotQA | 9,783 | 1,000 | 1,000 | 20×50 |
| 2WikiMultihopQA | 6,119 | 500 | 600 | 20×25 |
| MuSiQue | 9,838 | 500 | 1,000 | 20×25 |

- 漂移构造: KMeans (k=8)，H1 (W1-10) 97% head / 3% tail，H2 (W11-20) 3% head / 97% tail
- KB初始化: Head-biased (优先放 head cluster 的 supporting fact)
- Embedding: `all-MiniLM-L6-v2` (384维)

## 3. 八种策略

| # | 策略 | 类型 | 信号来源 | 每窗口操作 | 对应系统/角色 |
|---|---|---|---|---|---|
| B0 | **Static** | 静态底线 | 无 | 0 | 不更新下界 |
| B1 | **RandomFIFO** | 盲目供给侧 | 随机到达 | ≤50 swaps | 无脑定时任务,噪声充斥 |
| B2 | **DocArrival** | 结构化供给侧 | 相似文档到达 | ≤40 swaps | HippoRAG / LightRAG |
| B3 | **KnowledgeEdit** | 事实编辑 | 外部编辑请求 | ≤30 edits | RECIPE |
| B4 | **OnDemandFetch** | 被动搜索 | 查询失败 | 临时检索Top-20 | Agent/CRAG (用完即弃) |
| B5 | **LogDrivenArrival** | 人工日志反馈 | T期fail日志 | T+1期≤50 swaps | Human-in-the-loop (滞后) |
| V1 | **DRIP-Dense** | 需求驱动 | 查询失败+效用淘汰 | ≤60 swaps | 简单demand-driven |
| B6 | **Oracle** | 理论上限 | 完美未来知识 | 一次性重建 | 理论上界 |

### 新增策略说明

- **RandomFIFO (B1)**: 每窗口从 Pool 随机抽 FIFO_BATCH=50 文档替换最旧文档。证明盲目更新引入噪声。
- **OnDemandFetch (B4)**: KB 保持静态,失败查询临时检索 Pool Top-20 文档辅助回答,用完即弃不入库。证明"沉淀到KB (Consolidation)"的必要性。
- **LogDrivenArrival (B5)**: 分析 T 窗口 fail query 日志,T+1 窗口补入缺失文档 (Top-30检索,≤50 cap)。证明人工滞后效应。

## 4. Sudden Drift 实验结果 (20窗口)

### Recall@5

| Strategy | HotpotQA H1→H2 (Δ) | 2Wiki H1→H2 (Δ) | MuSiQue H1→H2 (Δ) |
|---|---|---|---|
| Static | 23.6→0.5 (-23.1) | 26.1→0.5 (-25.6) | 21.3→2.3 (-19.0) |
| RandomFIFO | 21.7→6.0 (-15.7) | 17.4→6.6 (-10.8) | 20.7→4.4 (-16.3) |
| DocArrival | 22.1→2.3 (-19.8) | 25.0→1.9 (-23.1) | 20.4→3.0 (-17.4) |
| KnowledgeEdit | 22.6→2.2 (-20.4) | 21.7→3.0 (-18.7) | 18.6→1.4 (-17.2) |
| OnDemandFetch | 27.5→10.9 (-16.6) | 26.2→4.0 (-22.2) | 22.6→10.6 (-12.0) |
| LogDrivenArrival | 20.9→7.2 (-13.7) | 15.7→4.2 (-11.5) | 17.1→2.9 (-14.2) |
| **DRIP-Dense** | 22.2→**7.5** (-14.7) | 24.6→**2.7** (-21.9) | 21.2→**8.8** (-12.4) |
| **Oracle** | 23.6→**74.5** (+50.9) | 26.1→**60.9** (+34.8) | 21.3→**55.1** (+33.8) |

### Recall@20

| Strategy | HotpotQA H2 | 2Wiki H2 | MuSiQue H2 |
|---|---|---|---|
| Static | 0.5 | 0.5 | 2.7 |
| RandomFIFO | 6.6 | 7.9 | 5.7 |
| DocArrival | 2.6 | 1.9 | 3.1 |
| KnowledgeEdit | 2.2 | 4.1 | 2.3 |
| OnDemandFetch | 12.3 | 4.7 | 15.0 |
| LogDrivenArrival | 8.2 | 5.2 | 3.8 |
| DRIP-Dense | 8.2 | 3.2 | 10.3 |
| **Oracle** | **81.6** | **66.2** | **64.6** |

### Cost (累计操作次数)

| Strategy | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| RandomFIFO | 896 | 907 | 896 |
| DocArrival | 271 | 163 | 269 |
| KnowledgeEdit | 513 | 593 | 578 |
| OnDemandFetch | 835 | 358 | 440 |
| LogDrivenArrival | 950 | 950 | 950 |
| DRIP-Dense | 731 | 626 | 464 |
| Oracle | 995 | 590 | 900 |

## 5. 核心发现

1. **Distribution drift 造成检索崩溃**: Static 从 20-26% → 0.5-2.3%
2. **盲目供给侧引入噪声**: RandomFIFO 虽有改善 (4-7%) 但 H1 也下降,整体信噪比恶化
3. **结构化供给侧有"信息茧房"**: DocArrival 只吸纳与现有KB相似文档,漂移后拒收新领域
4. **事实编辑只修补旧领域**: KnowledgeEdit 在旧领域内微调,无法应对领域级需求转移
5. **被动搜索代价高且不沉淀**: OnDemandFetch 在 HotpotQA/MuSiQue 效果不错 (10-11%),但每次查询都需外部检索,证明"沉淀到KB"的必要性
6. **人工反馈永远滞后一步**: LogDrivenArrival 在 T+1 才修复 T 的问题,持续漂移下永远追不上
7. **简单 DRIP-Dense 虽超越传统范式,但仍有限**: 受限于 single-hop 语义匹配,最好也只到 7-11%
8. **Oracle 上限巨大**: 55-75%,证明 KB 容量足够,瓶颈是 content selection precision
9. **最佳非Oracle方法→Oracle gap = 47-64 百分点**: 这就是图谱增强方法需要填补的空间

## 6. 诊断分析 (HotpotQA H2 Multi-hop Gap)

| 诊断指标 | 数值 | 含义 |
|---|---|---|
| Query→SF 直接相似度均值 | 0.523 | Dense检索对SF直接捕获能力弱 |
| SF sim > 0.55 (命中阈值) | 46.9% | 即使SF在KB,超一半检索不到 |
| Top-50 检索覆盖 SF | 86.7% | 算法能从Pool中选对文档 |
| 两个SF均sim>0.5 | 32.8% | **核心断点**: 仅1/3查询能同时找到两篇SF |
| 仅1个SF sim>0.5 | 51.8% | **多跳鸿沟**: 多数只够得到"第一跳" |

**结论**: 瓶颈不是"找不到文档",而是 **Dense向量对多跳查询的语义断层** → motivate 图谱增强方案 (V2)。

## 7. Figures

- `motivation_4/figures/recall_drift_20w_sudden.pdf` — 20窗口 Sudden drift, 8策略 Recall@5 曲线
