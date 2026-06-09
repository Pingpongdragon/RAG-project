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

## design/ — 算法设计

| 文件 | 内容 |
|---|---|
| [FINAL_METHOD.md](design/FINAL_METHOD.md) | 🟢 **DRYAD 权威设计**：漂移检测驱动的**双层缓存**（agent memory + doc hot-tier），三模块 DETECT/DECIDE/ADMIT + 实体桥接 + 审稿人反驳 |
| [REFERENCES.md](design/REFERENCES.md) | 文献调研：drift detection / RAG cache / agentic memory，related work + baseline + 白点 |
| [UNIFICATION.md](design/UNIFICATION.md) | `updator/` 与 `motivation/` 两套代码统一到 DRYAD 的关系与接缝 |
| [ALGORITHM_DESIGN.md](design/ALGORITHM_DESIGN.md) | ⚠️ 早期单层 doc-cache 版，已被 FINAL_METHOD 取代，留作历史 |
| [DESIGN_DIRECTIONS.md](design/DESIGN_DIRECTIONS.md) | gap 分类（G1–G5）与机制方向（D1–D6），设计素材来源 |

## literature/ — 文献背景

| 文件 | 内容 |
|---|---|
| [DOC_3_QUERY_SHIFT_LITERATURE.md](literature/DOC_3_QUERY_SHIFT_LITERATURE.md) | query distribution shift 相关工作调研（MS-Shift / TCR / StreamingQA） |
| [agent_memory_notes.md](literature/agent_memory_notes.md) | LLM agent memory 分层综述笔记（shared KB 定位） |

## verification/ — 复现验证

| 文件 | 内容 |
|---|---|
| [VERIFY_STREAMINGQA.md](verification/VERIFY_STREAMINGQA.md) | Fig 1 数据分割 / baseline / FAQ / 代码路径映射 |
