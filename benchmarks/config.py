"""
benchmarks 配置参数 — 所有可调参数集中管理。

修改此文件即可调整实验规模、调度器参数、数据源等，
无需修改任何 loader / builder 代码。
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
#  Topic 分组常量（WoW 数据集中的真实 topic 名称）
# ============================================================

FOOD_CHAIN = ["Pasta", "Pizza", "Baking", "Wine tasting"]
HEALTH_CHAIN = ["Obesity", "Physical fitness", "Chronic fatigue syndrome"]
DIVERSE_TOPICS = ["Red", "Manta ray", "Superman", "Niagara Falls", "Ferrari"]
CYCLE_TOPICS = ["Pasta", "Ferrari", "Hair coloring"]
BIG_TOPICS = ["Pasta", "Brown hair", "Jazz", "Ferrari", "Obesity"]


# ============================================================
#  数据集构建参数
# ============================================================

@dataclass
class WoWConfig:
    """WoW 数据源配置"""
    split: str = "validation"
    min_conversations: int = 15       # select_topics 时要求的最少对话数


@dataclass
class PoolConfig:
    """文档池配置"""
    max_total: int = 5000             # 文档池上限
    keep_all_gold: bool = True        # 始终保留所有 gold 文档


@dataclass
class GradualDriftConfig:
    """Exp 1: Gradual Drift 参数"""
    total_queries: int = 300
    sigma: float = 0.18               # 高斯标准差，越大重叠越多
    n_topics: int = 10                # 选取的 topic 数量
    preferred_topics: List[str] = field(default_factory=lambda: list(BIG_TOPICS))
    seed: int = 42
    pool: PoolConfig = field(default_factory=PoolConfig)
    wow: WoWConfig = field(default_factory=WoWConfig)


@dataclass
class SuddenShiftConfig:
    """Exp 2: Sudden Shift 参数"""
    total_queries: int = 300
    steepness: float = 30.0           # sigmoid 陡峭程度，越大切换越突然
    n_topics: int = 10
    preferred_topics: List[str] = field(default_factory=lambda: list(DIVERSE_TOPICS))
    seed: int = 42
    pool: PoolConfig = field(default_factory=PoolConfig)
    wow: WoWConfig = field(default_factory=WoWConfig)


@dataclass
class CyclicReturnConfig:
    """Exp 3: Cyclic Return 参数"""
    total_queries: int = 300
    n_cycles: int = 2                 # 周期数
    sigma: float = 0.08               # 每个高斯峰宽度
    n_topics: int = 5
    preferred_topics: List[str] = field(default_factory=lambda: list(CYCLE_TOPICS))
    seed: int = 42
    pool: PoolConfig = field(default_factory=PoolConfig)
    wow: WoWConfig = field(default_factory=WoWConfig)


@dataclass
class HotpotQAConfig:
    """Exp 4: HotpotQA Entity Walk 参数"""
    total_queries: int = 400
    split: str = "validation_distractor"
    seed: int = 42
    pool: PoolConfig = field(default_factory=lambda: PoolConfig(max_total=50000))


@dataclass
class BenchmarkConfig:
    """全局配置 — 聚合所有实验的参数"""
    gradual_drift: GradualDriftConfig = field(default_factory=GradualDriftConfig)
    sudden_shift: SuddenShiftConfig = field(default_factory=SuddenShiftConfig)
    cyclic_return: CyclicReturnConfig = field(default_factory=CyclicReturnConfig)
    hotpotqa: HotpotQAConfig = field(default_factory=HotpotQAConfig)

    @classmethod
    def default(cls) -> "BenchmarkConfig":
        return cls()

    @classmethod
    def quick_test(cls) -> "BenchmarkConfig":
        """缩小规模的快速测试配置"""
        small_pool = PoolConfig(max_total=500)
        return cls(
            gradual_drift=GradualDriftConfig(total_queries=30, n_topics=3, pool=small_pool),
            sudden_shift=SuddenShiftConfig(total_queries=30, n_topics=3, pool=small_pool),
            cyclic_return=CyclicReturnConfig(total_queries=30, n_topics=3, pool=small_pool),
            hotpotqa=HotpotQAConfig(total_queries=40, pool=PoolConfig(max_total=1000)),
        )
