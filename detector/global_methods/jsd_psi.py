"""
B1: JSD + PSI Baseline

工业标准方法: 将窗口内概率向量取平均，与 KB 分布用
JSD (Jensen-Shannon Divergence) 和 PSI (Population Stability Index) 比较

优点: 简单快速，易于解释
缺点: 平均操作丢失样本级分布形状信息，无统计检验 (无 p-value)
"""

import numpy as np
from typing import Dict, Tuple
from scipy.spatial.distance import jensenshannon
import logging

from ..base import DOMAINS

logger = logging.getLogger(__name__)


class JSDPSIDetector:
    def __init__(
        self,
        kb_dist_vec: np.ndarray,
        jsd_threshold: float = 0.1,
        psi_threshold: float = 0.2,
    ):
        """
        Args:
            kb_dist_vec: shape (C,) — KB 聚合分布 (各类别比例)
        """
        self.kb_dist_vec = kb_dist_vec.copy()
        self.jsd_threshold = jsd_threshold
        self.psi_threshold = psi_threshold

    def detect(self, query_probs: np.ndarray) -> Tuple[float, float, bool]:
        """
        Args:
            query_probs: shape (W, C) — 窗口内所有查询的概率向量

        Returns:
            (jsd, psi, is_shift)
        """
        avg = np.mean(query_probs, axis=0)

        jsd = float(jensenshannon(self.kb_dist_vec, avg))

        eps = 1e-6
        e = np.maximum(self.kb_dist_vec, eps)
        a = np.maximum(avg, eps)
        psi = float(np.abs(np.sum((a - e) * np.log(a / e))))

        is_shift = (jsd > self.jsd_threshold) or (psi > self.psi_threshold)
        return jsd, psi, is_shift