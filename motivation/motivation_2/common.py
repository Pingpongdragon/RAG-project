"""
Motivation 2: 共享工具模块

提供 gen_scaling_recovery.py 使用的分布计算与 JSD 工具。
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.distance import jensenshannon
import os

OUT_DIR = Path(__file__).resolve().parent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个分布之间的 JS 散度"""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))
