"""
benchmark.data — 实验数据构造模块

提供四种漂移场景的数据集构建函数:
  - build_gradual_drift:       高斯渐变 topic 漂移 (WoW)
  - build_sudden_shift:        Sigmoid 阶跃 topic 切换 (WoW)
  - build_cyclic_return:       周期性 topic 回归 (WoW)
  - build_hotpotqa_entity_walk: HotpotQA 实体图随机游走
"""

from benchmarks.archive_legacy.data.experiment_builders import (
    build_gradual_drift,
    build_sudden_shift,
    build_cyclic_return,
    build_hotpotqa_entity_walk,
)

from benchmarks.archive_legacy.data.structures import (
    QueryItem,
    PoolDocument,
    ExperimentDataset,
)

__all__ = [
    "build_gradual_drift",
    "build_sudden_shift",
    "build_cyclic_return",
    "build_hotpotqa_entity_walk",
    "QueryItem",
    "PoolDocument",
    "ExperimentDataset",
]
