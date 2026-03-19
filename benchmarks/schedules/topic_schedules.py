"""
Topic 概率调度器 - 控制查询流中 topic 随时间的分布变化。

每个调度器实现 P(topic | t)，其中 t in [0, 1] 是归一化的流位置。
不同调度器模拟不同类型的兴趣漂移模式:

  GaussianDriftSchedule  - 各 topic 高斯激活，平滑过渡（渐进漂移）
  SigmoidShiftSchedule   - 陡峭 sigmoid 阶跃切换（突变转移）
  CyclicSchedule         - 周期性高斯峰，A->B->C->A->B->C（周期回归）
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class TopicSchedule(ABC):
    """P(topic | t) 的抽象基类，t in [0, 1]。"""

    @abstractmethod
    def get_probs(self, t: float) -> Dict[str, float]:
        """返回时刻 t 的归一化 topic 概率分布。"""
        ...

    def get_probs_array(self, t: float, topics: List[str]) -> np.ndarray:
        """返回与 topics 列表对齐的概率数组。"""
        p = self.get_probs(t)
        a = np.array([p.get(tp, 0.0) for tp in topics])
        s = a.sum()
        return a / s if s > 0 else np.ones(len(topics)) / len(topics)


class GaussianDriftSchedule(TopicSchedule):
    """
    各 topic 拥有一个高斯激活曲线，中心等间距分布在 [0, 1] 上。

    w_i(t) = exp(-0.5 * ((t - c_i) / sigma)^2)

    sigma 控制重叠程度 - 越大，相邻 topic 共存越多。

    Args:
        topics: topic 名称列表
        sigma:  高斯标准差（默认 0.18）
    """

    def __init__(self, topics: List[str], sigma: float = 0.18):
        self.topics = topics
        n = len(topics)
        self.centres = [i / max(n - 1, 1) for i in range(n)]
        self.sigma = sigma

    def get_probs(self, t: float) -> Dict[str, float]:
        raw = {
            tp: math.exp(-0.5 * ((t - c) / self.sigma) ** 2)
            for tp, c in zip(self.topics, self.centres)
        }
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


class SigmoidShiftSchedule(TopicSchedule):
    """
    Topics 通过陡峭 sigmoid 曲线切换 - 模拟突然的兴趣转移。

    每对相邻 topic 之间有一个 sigmoid 过渡点，steepness 控制切换的陡峭程度。

    Args:
        topics:    topic 名称列表
        steepness: sigmoid 陡峭程度（默认 30.0，越大越突兀）
    """

    def __init__(self, topics: List[str], steepness: float = 30.0):
        self.topics = topics
        n = len(topics)
        self.transitions = [(i + 0.5) / n for i in range(n - 1)]
        self.steepness = steepness

    def get_probs(self, t: float) -> Dict[str, float]:
        n = len(self.topics)
        w = np.zeros(n)
        for i in range(n):
            left = 1.0 if i == 0 else 1.0 / (1.0 + math.exp(
                -self.steepness * (t - self.transitions[i - 1])))
            right = 1.0 if i == n - 1 else 1.0 / (1.0 + math.exp(
                self.steepness * (t - self.transitions[i])))
            w[i] = left * right
        total = w.sum() or 1.0
        return {self.topics[i]: float(w[i] / total) for i in range(n)}


class CyclicSchedule(TopicSchedule):
    """
    Topics 周期性激活: A -> B -> C -> A -> B -> C -> ...

    每个 topic 拥有等间隔高斯峰，形成平滑的周期循环。

    Args:
        topics:   topic 名称列表
        n_cycles: 总周期数（默认 2）
        sigma:    每个高斯峰的宽度（默认 0.08）
    """

    def __init__(self, topics: List[str], n_cycles: int = 2, sigma: float = 0.08):
        self.topics = topics
        self.n_cycles = n_cycles
        self.sigma = sigma
        n = len(topics)
        cycle_len = 1.0 / n_cycles
        self.peaks: Dict[str, List[float]] = {}
        for i, tp in enumerate(topics):
            offset = (i / n) * cycle_len
            self.peaks[tp] = [c * cycle_len + offset for c in range(n_cycles)]

    def get_probs(self, t: float) -> Dict[str, float]:
        raw = {}
        for tp, peaks in self.peaks.items():
            raw[tp] = sum(
                math.exp(-0.5 * ((t - p) / self.sigma) ** 2) for p in peaks
            )
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}
