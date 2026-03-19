"""
benchmarks — 实验数据集构造模块

模块结构:
    benchmarks/
    ├── config.py                  # 所有可调参数集中管理
    ├── data_structures.py         # QueryItem, PoolDocument, ExperimentDataset
    ├── experiment_framework.py    # 评估框架 (Adapters, Metrics, run_comparison)
    ├── run_experiments.py         # CLI 实验入口
    ├── datasets/                  # 数据加载器 (WoW, HotpotQA)
    ├── schedules/                 # Topic 概率调度器
    ├── builders/                  # 实验构建入口
    ├── data/                      # 预生成的 JSON 数据集
    └── README.md

用法:
    from benchmarks import build_gradual_drift, QueryItem
    from benchmarks.config import BenchmarkConfig, GradualDriftConfig
"""

from benchmarks.data_structures import QueryItem, PoolDocument, ExperimentDataset
from benchmarks.config import (
    BenchmarkConfig,
    GradualDriftConfig,
    SuddenShiftConfig,
    CyclicReturnConfig,
    HotpotQAConfig,
    PoolConfig,
    WoWConfig,
    BIG_TOPICS,
    DIVERSE_TOPICS,
    CYCLE_TOPICS,
    FOOD_CHAIN,
    HEALTH_CHAIN,
)
from benchmarks.schedules import (
    TopicSchedule,
    GaussianDriftSchedule,
    SigmoidShiftSchedule,
    CyclicSchedule,
)
from benchmarks.builders import (
    build_gradual_drift,
    build_sudden_shift,
    build_cyclic_return,
    build_hotpotqa_entity_walk,
    build_all_datasets,
)

__all__ = [
    # Data structures
    "QueryItem", "PoolDocument", "ExperimentDataset",
    # Config
    "BenchmarkConfig", "GradualDriftConfig", "SuddenShiftConfig",
    "CyclicReturnConfig", "HotpotQAConfig", "PoolConfig", "WoWConfig",
    "BIG_TOPICS", "DIVERSE_TOPICS", "CYCLE_TOPICS", "FOOD_CHAIN", "HEALTH_CHAIN",
    # Schedules
    "TopicSchedule", "GaussianDriftSchedule", "SigmoidShiftSchedule", "CyclicSchedule",
    # Builders
    "build_gradual_drift", "build_sudden_shift", "build_cyclic_return",
    "build_hotpotqa_entity_walk", "build_all_datasets",
]
