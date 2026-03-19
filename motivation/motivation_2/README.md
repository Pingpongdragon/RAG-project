# Motivation 2: Query-Document Alignment

## 核心叙事

当用户查询话题发生漂移 (Query Shift) 时，知识库文档与新查询不再对齐，检索性能骤降。
通过添加与新查询对齐的文档，性能可以恢复 —— 这正是 QARC 动态知识库管理要解决的问题。

## 目录结构

```
motivation_2/
├── gen_misalignment.py          ← 聚类版 Misalignment 实验
├── query_doc_alignment.py       ← 绘图脚本 (2 面板)
├── README.md
├── data/                        ← 实验数据 & 日志
│   ├── results_scaling.json         4 种 Retriever 的 Scaling Recovery 数据
│   ├── results_misalignment.json    5 级 JSD Misalignment 数据
│   ├── cluster_info.json            聚类元数据 (K=7, 语义标签)
│   └── full_run.log                 完整实验日志
├── figures/                     ← 图表输出
│   ├── query_doc_alignment.png
│   └── query_doc_alignment.pdf
├── gen_scaling_recovery.py      ← 生成 results_scaling.json
└── common.py                    ← 共享工具 (compute_jsd)
```

## 图表说明 (2 面板)

- **(a) Query Shift → Performance Drop → Recovery**: 4 种检索器均下降 ~40%，添加对齐文档恢复 +29~34%，证明 query shift 是模型无关的通用问题。
- **(b) Higher Misalignment → Lower Performance**: JSD 0→0.38，Hit@10 56.4%→37.9%，量化了 KB-Query 分布偏移与性能的单调递减关系。

## 如何复现

```bash
python motivation/motivation_2/gen_misalignment.py --quick   # ~5 min
python motivation/motivation_2/gen_misalignment.py --full    # ~60 min
python motivation/motivation_2/query_doc_alignment.py        # 生成图表
```
