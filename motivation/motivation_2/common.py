"""
Motivation 2: 共享工具模块
- 蒸馏分类器封装
- 分布计算 & JSD
- 共享常量
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.distance import jensenshannon

# 导入蒸馏分类器
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from detector.distill_trainer import OnlineDetector, LABEL_MAP, ID2LABEL
import os

# ==========================================
# 共享常量
# ==========================================
DETECTOR_PATH = "/home/jyliu/RAG_project/detector/mini_router_best"
DOMAIN_NAMES = list(LABEL_MAP.keys())  # ['entertainment', 'stem', 'humanities', 'lifestyle']
NUM_DOMAINS = len(DOMAIN_NAMES)

OUT_DIR = Path(__file__).resolve().parent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# 分类器工具
# ==========================================
def get_detector(path: str = DETECTOR_PATH) -> OnlineDetector:
    """加载蒸馏分类器 (单例模式)"""
    return OnlineDetector(path)


def classify_queries(queries: List[str], detector: OnlineDetector) -> List[str]:
    """用蒸馏分类器批量分类 query 的领域"""
    results = detector.predict_batch(queries, batch_size=64)
    return [r["top_label"] for r in results]


# ==========================================
# 分布计算
# ==========================================
def get_domain_distribution(labels: List[str]) -> np.ndarray:
    """计算领域分布向量 [entertainment, stem, humanities, lifestyle]"""
    counts = np.zeros(NUM_DOMAINS, dtype=float)
    for lbl in labels:
        if lbl in LABEL_MAP:
            counts[LABEL_MAP[lbl]] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(NUM_DOMAINS) / NUM_DOMAINS
    return counts


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个分布之间的 JS 散度"""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))