"""DRIP 算法包入口。

现在只对外暴露两个策略：
  - ``DRIP``：保留给依赖 drift detector 的版本；
  - ``DRIPNOdetector``：当前主实验先跑的不依赖 detector 版本。
"""

from .cache_manager import (
    DRIP,
    DRIPNOdetector,
)

__all__ = [
    "DRIP",
    "DRIPNOdetector",
]
