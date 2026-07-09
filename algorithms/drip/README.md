# DRIP 目录说明

当前 DRIP 目录只保留两条实验入口：

```text
DRIPNOdetector  当前主实验方法：不依赖 detector。
DRIP            后续接 drift detector 的版本，目前不是主结论。
```

`drip.py` 已经改名为 `policies.py`，因为它不是 main 文件，只是策略入口。
真正的主窗口循环在 `cache_manager/core.py`。

## 文件地图

| 文件 | 作用 | 当前是否主路径 |
|---|---|---|
| `__init__.py` | 顶层导出，只暴露 `DRIP` / `DRIPNOdetector` | 是 |
| `README.md` | 本目录说明 | 是 |
| `PARAMETER_MAPPING.md` | 参数和论文公式对照 | 是 |
| `cache_manager/__init__.py` | cache manager 公开导出层，不放算法逻辑 | 是 |
| `cache_manager/core.py` | DRIPCore 主循环：观察窗口、路由 evidence、更新账本、调用 writer | 是 |
| `cache_manager/policies.py` | 策略入口：`DRIPNOdetector` / `DRIP` | 是 |
| `cache_manager/drip_config.py` | 参数表；主实验只看 `DRIPCoreConfig` | 是 |
| `cache_manager/evidence_core.py` | direct evidence、replacement-aware writer、hidden diagnostic | 是 |
| `cache_manager/dense_index.py` | query-visible dense candidate 检索 | 是 |
| `cache_manager/evidence_router.py` | 判断 evidence 是 visible 还是 hidden | 是 |
| `cache_manager/entity_graph_index.py` | hidden diagnostic 使用的实体/关系索引 | appendix / diagnostic |
| `detection/multi_agent_drift.py` | drift detector/controller，后续给 `DRIP` 用 | 暂不用于主实验 |
| `tests/` | GraphIndex / bridge diagnostic 单测 | diagnostic |

## 为什么现在只有一个 config？

之前有：

```text
config.py
cost_aware_config.py
```

这是因为旧 CostAware prototype 单独维护了一套参数。现在 CostAware prototype
已经移出 active 路径，当前只保留：

```text
cache_manager/drip_config.py
```

里面分成两个类：

```text
DRIPCoreConfig              当前主实验最小参数
DRIPHiddenDiagnosticConfig  hidden / detector 旧分支参数
```

日常调主实验先只看 `DRIPCoreConfig`。

## 当前主公式对应文件

```text
E_t(q,d)
  -> cache_manager/evidence_core.py::_credit_dense

S_t(d), D_t(d)
  -> cache_manager/core.py::_credit_serve
  -> cache_manager/evidence_core.py::_decay / _credit_dense

P_t(v)
  -> cache_manager/evidence_core.py::_priority_for_route

C_t(c,v), Delta_t(c,v)
  -> cache_manager/evidence_core.py::_replacement_penalty
  -> cache_manager/evidence_core.py::_write
```

## 中文文件头

当前 active Python 文件都保留了中文 module docstring。之后如果新增文件，也按这个
规则写：文件第一段先说明“这个文件负责什么、不负责什么”。
