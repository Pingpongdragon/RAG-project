"""DRIP cache manager 的公开导出层。

这里不放算法逻辑，只把常用类从职责清楚的文件里重新导出：

  - ``core.py``: 主窗口循环 ``DRIPCore``；
  - ``policies.py``: 实验策略入口 ``DRIP`` / ``DRIPNOdetector``；
  - ``drip_config.py``: 参数表 ``DRIPCoreConfig``；
  - ``dense_index.py`` / ``entity_graph_index.py`` / ``evidence_router.py``:
    检索、实体索引、evidence 路由组件。
"""

from .core import DRIPCore
from .dense_index import EmbeddingIndex
from .drip_config import DRIPCoreConfig
from .entity_graph_index import GraphIndex
from .evidence_router import QUERY_HIDDEN, QUERY_VISIBLE, QueryRouter, RouteDecision
from .policies import (
    DRIP,
    DRIPDense,
    DRIPESC,
    DRIPESCLease,
    DRIPNOdetector,
    DRIPQueryHidden,
    DRIPQueryVisible,
)

__all__ = [
    "DRIP",
    "DRIPNOdetector",
    "DRIPCore",
    "DRIPCoreConfig",
    "QueryRouter",
    "EmbeddingIndex",
    "GraphIndex",
    "RouteDecision",
    "QUERY_VISIBLE",
    "QUERY_HIDDEN",
    "DRIPQueryVisible",
    "DRIPQueryHidden",
    "DRIPDense",
    "DRIPESC",
    "DRIPESCLease",
]
