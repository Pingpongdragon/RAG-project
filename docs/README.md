# RAG-project 文档中心

全项目文档导航。算法设计与文献综述在本目录；motivation 实验台专属的叙事/实验/验证文档仍在 [motivation/docs/](../motivation/docs/)。

## 目录

### design/ — DRYAD 算法设计
| 文件 | 内容 |
|---|---|
| [FINAL_METHOD.md](design/FINAL_METHOD.md) | 🟢 **权威设计**：漂移检测驱动的双层缓存（agent memory + doc hot-tier），三模块 DETECT/DECIDE/ADMIT + 实体桥接 |
| [REFERENCES.md](design/REFERENCES.md) | 文献调研：drift detection / RAG cache / agentic memory + baseline 选择 + 白点 |
| [UNIFICATION.md](design/UNIFICATION.md) | `algorithms/`（生产）与 `motivation/`（实验台）两套代码如何统一到 DRYAD |
| [ALGORITHM_DESIGN.md](design/ALGORITHM_DESIGN.md) | ⚠️ 早期单层版，已被 FINAL_METHOD 取代，留作历史 |
| [DESIGN_DIRECTIONS.md](design/DESIGN_DIRECTIONS.md) | gap 分类 G1–G5，设计素材来源 |

### literature/ — 文献背景
| 文件 | 内容 |
|---|---|
| [DOC_3_QUERY_SHIFT_LITERATURE.md](literature/DOC_3_QUERY_SHIFT_LITERATURE.md) | query distribution shift 相关工作调研 |
| [agent_memory_notes.md](literature/agent_memory_notes.md) | LLM agent memory 分层综述笔记 |

### 顶层
| 文件 | 内容 |
|---|---|
| [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | 项目交接说明 |

## 代码与实验的文档（在各自目录）

- **算法实现** → [algorithms/README.md](../algorithms/README.md)：DRYAD 核心（qarc）+ baseline（comrag/erase/base）
- **实验驱动** → [benchmark/](../benchmark/)：DRYAD vs baseline 对比框架
- **motivation 实验台** → [motivation/docs/README.md](../motivation/docs/README.md)：三级 audit 的叙事/实验/验证

## 全项目结构速览

```
RAG-project/
├── docs/             # 本目录：算法设计 + 文献（全项目级文档）
├── algorithms/       # 核心算法 DRYAD(qarc) + baseline(comrag/erase/base)
├── benchmark/        # 唯一实验驱动（DRYAD vs baseline），legacy_benchmarks/ 存旧数据
├── motivation/       # 论文 motivation 实验台（三级 audit, Fig 0/1/2）
├── core/             # 底层公共库（retriever/generator/evaluator…）
├── config/ models/   # 配置与模型缓存
├── datasets/         # 数据集
├── third_party/      # HippoRAG/LightRAG/RECIPE（参考实现，不 import）
└── main.py           # RAG 管道入口
```
