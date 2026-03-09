"""
Motivation 2: 共享工具模块

⚠️ 注意：此文件是历史实验代码，依赖已删除的硬分类蒸馏模块。
⚠️ 当前项目已迁移到 ComRAG 动态聚类方案，不再使用固定的4 domain分类。
⚠️ 此文件仅供历史参考，相关函数已注释。

原功能：
- 蒸馏分类器封装
- 分布计算 & JSD
- 共享常量
"""
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.distance import jensenshannon
import os

# ==========================================
# 历史常量 (已废弃 - 硬编码4 domain)
# ==========================================
# 这些是历史实验用的固定domain定义
LEGACY_LABEL_MAP = {
    "entertainment": 0,
    "stem": 1,
    "humanities": 2,
    "lifestyle": 3
}
LEGACY_ID2LABEL = {v: k for k, v in LEGACY_LABEL_MAP.items()}
DOMAIN_NAMES = list(LEGACY_LABEL_MAP.keys())  # ['entertainment', 'stem', 'humanities', 'lifestyle']
NUM_DOMAINS = len(DOMAIN_NAMES)

OUT_DIR = Path(__file__).resolve().parent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# 已废弃：蒸馏分类器相关函数
# ==========================================
# 以下函数依赖已删除的 detector.distill_trainer 模块
# 保留代码仅供参考

# def get_detector(path: str = DETECTOR_PATH) -> OnlineDetector:
#     """加载蒸馏分类器 (单例模式) - 已废弃"""
#     from detector.distill_trainer import OnlineDetector
#     return OnlineDetector(path)

# def classify_queries(queries: List[str], detector) -> List[str]:
#     """用蒸馏分类器批量分类 query 的领域 - 已废弃"""
#     results = detector.predict_batch(queries, batch_size=64)
#     return [r["top_label"] for r in results]


# ==========================================
# 分布计算工具（仍可用）
# ==========================================
def get_domain_distribution(labels: List[str]) -> np.ndarray:
    """
    计算领域分布向量 [entertainment, stem, humanities, lifestyle]
    
    ⚠️ 注意：此函数假设使用历史的4 domain标签
    Args:
        labels: domain标签列表
    Returns:
        4维分布向量
    """
    counts = np.zeros(NUM_DOMAINS, dtype=float)
    for lbl in labels:
        if lbl in LEGACY_LABEL_MAP:
            counts[LEGACY_LABEL_MAP[lbl]] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(NUM_DOMAINS) / NUM_DOMAINS
    return counts


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个分布之间的 JS 散度（通用工具）"""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))
