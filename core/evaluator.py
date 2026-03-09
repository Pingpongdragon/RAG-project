"""
core/evaluator.py — 通用检索与生成评估模块

提供与数据源无关的评估函数:
  - Recall@K: 检索召回率
  - Exact Match (EM): 精确匹配
  - Token F1: 词级 F1
  - Answer Relevancy: LLM 评分 (可选)

不再依赖硬编码的 domain shift 数据文件,
所有数据由实验脚本传入
"""

import logging
import re
from typing import List, Dict, Set, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 文本规范化
# ============================================================

def normalize_answer(s: str) -> str:
    """规范化答案文本 (小写 + 去标点 + 去冠词 + 压缩空白)"""
    s = s.lower().strip()
    # 去标点
    s = re.sub(r"[^\w\s]", " ", s)
    # 去冠词
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# 检索指标
# ============================================================

def recall_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> float:
    """
    Recall@K = |retrieved ∩ gold| / |gold|

    Args:
        retrieved_ids: 检索返回的文档 ID 列表
        gold_ids:      真实相关文档 ID 列表
    """
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_ids)) / len(gold_ids)


def precision_at_k(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> float:
    """Precision@K"""
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & set(gold_ids)) / len(retrieved_ids)


def gold_in_kb_rate(
    kb_doc_ids: Set[str],
    gold_ids: List[str],
) -> float:
    """KB 中包含多少比例的 gold 文档"""
    if not gold_ids:
        return 0.0
    return len(set(gold_ids) & kb_doc_ids) / len(gold_ids)


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    gold_ids: List[str],
) -> float:
    """MRR — 第一个命中的倒数排名"""
    gold_set = set(gold_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# 生成指标
# ============================================================

def exact_match(prediction: str, gold: str) -> float:
    """Exact Match (规范化后)"""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1"""
    pred_tokens = set(normalize_answer(prediction).split())
    gold_tokens = set(normalize_answer(gold).split())
    if not gold_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ============================================================
# 批量评估
# ============================================================

def evaluate_retrieval_batch(
    all_retrieved: List[List[str]],
    all_gold: List[List[str]],
) -> Dict[str, float]:
    """
    批量评估检索质量

    Args:
        all_retrieved: 每条查询的检索结果 ID 列表
        all_gold:      每条查询的 gold 文档 ID 列表

    Returns:
        {"recall@k": float, "precision@k": float, "mrr": float}
    """
    recalls, precisions, mrrs = [], [], []
    for retrieved, gold in zip(all_retrieved, all_gold):
        recalls.append(recall_at_k(retrieved, gold))
        precisions.append(precision_at_k(retrieved, gold))
        mrrs.append(mean_reciprocal_rank(retrieved, gold))

    return {
        "recall@k": float(np.mean(recalls)) if recalls else 0.0,
        "precision@k": float(np.mean(precisions)) if precisions else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def evaluate_generation_batch(
    predictions: List[str],
    golds: List[str],
) -> Dict[str, float]:
    """
    批量评估生成质量

    Returns:
        {"em": float, "f1": float}
    """
    ems, f1s = [], []
    for pred, gold in zip(predictions, golds):
        ems.append(exact_match(pred, gold))
        f1s.append(token_f1(pred, gold))

    return {
        "em": float(np.mean(ems)) if ems else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }


# ============================================================
# 时间序列分析 (用于实验结果后处理)
# ============================================================

def compute_adaptation_speed(
    metric_series: List[float],
    window_size: int = 20,
    threshold_ratio: float = 0.9,
) -> float:
    """
    计算适应速度: 从 metric 下降到恢复至阈值 (peak * ratio) 所需的窗口数

    Args:
        metric_series: 时间序列 (如 recall 曲线)
        window_size:   滑动窗口大小
        threshold_ratio: 恢复阈值比例

    Returns:
        恢复所需的窗口数 (越小越好), 未恢复返回 len(series)/window_size
    """
    if len(metric_series) < window_size * 2:
        return float(len(metric_series)) / window_size

    # 滑动窗口平均
    n_windows = len(metric_series) // window_size
    window_avgs = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_avgs.append(np.mean(metric_series[start:end]))

    if not window_avgs:
        return 0.0

    peak = max(window_avgs)
    threshold = peak * threshold_ratio

    # 找最低点后首次恢复到阈值的窗口
    trough_idx = int(np.argmin(window_avgs))
    for i in range(trough_idx, len(window_avgs)):
        if window_avgs[i] >= threshold:
            return float(i - trough_idx)

    return float(len(window_avgs) - trough_idx)
