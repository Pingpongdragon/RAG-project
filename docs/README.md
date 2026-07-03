# RAG-project 文档中心

全项目文档导航。算法设计与文献综述在本目录；motivation 实验台专属的叙事/实验/验证文档仍在 [motivation/docs/](../motivation/docs/)。

## 目录

### design/ — DRIP 算法设计
| 文件 | 内容 |
|---|---|
| [FINAL_METHOD.md](design/FINAL_METHOD.md) | 🟢 **权威设计**：漂移检测驱动的 L1 文档热缓存，三模块 DETECT/DECIDE/ADMIT + 实体桥接 |
| [DRIP_ALGORITHM_EXPLAINED.md](design/DRIP_ALGORITHM_EXPLAINED.md) | 面向论文与实验执行的 DRIP 说明书：模块解释、伪代码、消融映射、最终实验矩阵 |
| [REFERENCES.md](design/REFERENCES.md) | 文献调研：drift detection / RAG cache / adaptive cache + baseline 选择 |
| [UNIFICATION.md](design/UNIFICATION.md) | `algorithms/`（生产）与 `motivation/`（实验台）两套代码如何统一到 DRIP |
| [ARC_COMPARISON.md](design/ARC_COMPARISON.md) | ⚠️ 最近邻威胁 ARC(2511.02919) 的对照与差异化定位 |
| [ALGORITHM_DESIGN.md](design/ALGORITHM_DESIGN.md) | ⚠️ 早期单层版，已被 FINAL_METHOD 取代，留作历史 |
| [DESIGN_DIRECTIONS.md](design/DESIGN_DIRECTIONS.md) | gap 分类 G1–G5，设计素材来源 |

### literature/ — 文献背景
| 文件 | 内容 |
|---|---|
| [DOC_3_QUERY_SHIFT_LITERATURE.md](literature/DOC_3_QUERY_SHIFT_LITERATURE.md) | query distribution shift 相关工作调研 |

### 顶层
| 文件 | 内容 |
|---|---|
| [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | 项目交接说明 |

## 代码与实验的文档（在各自目录）

- **算法实现** → [algorithms/README.md](../algorithms/README.md)：DRIP 核心 + baseline（comrag/erase/base）
- **实验驱动** → [benchmark/](../benchmark/)：DRIP vs baseline 对比框架
- **motivation 实验台** → [motivation/docs/README.md](../motivation/docs/README.md)：三级 audit 的叙事/实验/验证

## 全项目结构速览

```
RAG-project/
├── docs/             # 本目录：算法设计 + 文献（全项目级文档）
├── algorithms/       # 核心算法 DRIP + baseline(comrag/erase/base)
├── benchmark/        # 唯一实验驱动（DRIP vs baseline），legacy_benchmarks/ 存旧数据
├── motivation/       # 论文 motivation 实验台（三级 audit, Fig 0/1/2）
├── core/             # 底层公共库（retriever/generator/evaluator…）
├── config/ models/   # 配置与模型缓存
├── datasets/         # 数据集
├── third_party/      # HippoRAG/LightRAG/RECIPE（参考实现，不 import）
└── main.py           # RAG 管道入口
```
