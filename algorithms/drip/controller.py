"""DRIP 的在线写入预算控制器。

``PrimalDualController`` 把长期 replacement budget 转换成每次物理
写入的影子价格。它只使用截至当前窗口的实际写入反馈。
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DualFeedback:
    price_before: float
    price_after: float
    load: float
    target: float
    step_size: float


class PrimalDualController:
    """用一个影子价格约束长期 replacement 负载。"""

    def __init__(self, target_rate, initial_price):
        self.target_rate = float(target_rate)
        self.price = max(0.0, float(initial_price))
        self.age = 0

    def update(self, writes, write_budget):
        write_budget = int(write_budget)
        price_before = float(self.price)
        load = float(writes) / max(1.0, float(write_budget))
        if write_budget > 0:
            self.age += 1
            step_size = 1.0 / math.sqrt(float(self.age))
            self.price = max(
                0.0,
                price_before + step_size * (load - self.target_rate),
            )
        else:
            step_size = 0.0
        return DualFeedback(
            price_before=price_before,
            price_after=float(self.price),
            load=load,
            target=float(self.target_rate),
            step_size=float(step_size),
        )
