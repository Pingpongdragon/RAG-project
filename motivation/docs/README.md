# docs/ — Motivation 文档索引

按用途分类的全部文档。代码与数据在上一级的 `motivation_0/1/2`，本目录只放叙事性 / 分析性材料。

## narrative/ — 叙事与评审

| 文件 | 内容 |
|---|---|
| [STORYLINE_v1.md](narrative/STORYLINE_v1.md) | **当前主叙事（v4）**：L1/L2/L3 三级访问模式、五幕结构、图-文映射 |
| [NEXT_STEPS_AUDIT_TODO.md](narrative/NEXT_STEPS_AUDIT_TODO.md) | 评审落地未尽事项（并发动机实验、Fig 0c、缺失 bib） |
| [评审意见.md](narrative/评审意见.md) | reviewer 指出的"宏观↔微观断层"及重构建议 |
| [会议记录_05.txt](narrative/会议记录_05.txt) | 组会记录：标题、结构、图表、bridge 量化要求 |

## experiments/ — 实验总结

| 文件 | 内容 |
|---|---|
| [DOC_1_EXPERIMENT_SUMMARY.md](experiments/DOC_1_EXPERIMENT_SUMMARY.md) | 8 策略总框架与多数据集 Recall@5 结果 |
| [DOC_5_STREAMINGQA_ANALYSIS.md](experiments/DOC_5_STREAMINGQA_ANALYSIS.md) | **Fig 1** StreamingQA era 分析（R1–R5 覆盖率/失效原因） |
| [DATASET_ANALYSIS.md](experiments/DATASET_ANALYSIS.md) | QDC 生效的三充要条件 + 8 数据集 breakdown |
| [REVIEWER_RESPONSE_SUMMARY.md](experiments/REVIEWER_RESPONSE_SUMMARY.md) | v3 revision 实验清单（cache baseline / 噪声敏感性 / 真实漂移） |

## design/ 与 literature/ — 已上移到项目顶层 docs/

算法设计（DRIP）与文献综述属于全项目级，已移到 [../../docs/](../../docs/)：
- 算法设计 → [../../docs/design/](../../docs/design/)（FINAL_METHOD / REFERENCES / UNIFICATION / ALGORITHM_DESIGN / DESIGN_DIRECTIONS）
- 文献背景 → [../../docs/literature/](../../docs/literature/)（query shift / agent memory notes）

本目录（motivation/docs）此后只保留 motivation 实验台专属的 narrative / experiments / verification。

## verification/ — 复现验证

| 文件 | 内容 |
|---|---|
| [VERIFY_STREAMINGQA.md](verification/VERIFY_STREAMINGQA.md) | Fig 1 数据分割 / baseline / FAQ / 代码路径映射 |
