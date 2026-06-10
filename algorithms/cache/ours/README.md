# ours/ — 你的方法及其实验台后端

> ⚠️ **命名待统一**：这里有三个名字（DRYAD / RoutedCache / QueryDriven=SemFlow），
> 加上 `algorithms/qarc/` 包名，共四套指向重叠的东西。见 `docs/design/CODE_EXPLAINED.md §0`。
> 正式论文应只保留**一个**最终方法，其余作组件或消融。

| 文件 | 类 | 是什么 | 论文用？ |
|---|---|---|---|
| `query_driven.py` | `QueryDriven` (SemFlow) | demand-ledger 准入：失败 query 的语义需求累积驱动换入换出，无 regime 分支 | 🟢 你方法的核心机制之一 |
| `query_driven.py` | `QueryDrivenLoose` | 上者的宽松版（探针更宽、门控更松），敏感性分析用 | ⚪ 消融/敏感性，可选 |
| `routed_cache.py` | `RoutedCache` | demand-ledger + **实体桥接 (entity-chain bridge)** 后端 | 🟢 多跳 bridge 能力所在 |
| `dryad.py` | `DRYAD` | 漂移检测 + 决策 + 上述后端的整合版（论文主方法骨架） | 🟢 **最终方法候选** |

## 与 baseline 的本质区别（写进论文 related work）
- vs ARC：你有**显式漂移检测**(FID) + **λ·B 决策预算**；ARC 是累积几何打分、逐项逐出、无预算。
- vs Proximity/GPTCache：你有**实体桥接**触达 bridge 文档；它们纯 query-doc 相似度够不到。

## 跑实验建议
**正文主方法只报 1 个**（DRYAD 或它的最终命名版），不要 QueryDriven/RoutedCache/DRYAD 三个都上主表——
否则读者分不清哪个是"你的方法"。其余进附录做消融（如 w/o bridge = QueryDriven vs w/ bridge = RoutedCache）。
