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



def kb_turnover_rate(
    kb_history: List[Set[str]],
) -> float:
    """
    KB 周转率 — 衡量 KB 文档集合在流式处理过程中的变更程度

    per-step turnover = |KB_t Δ KB_{t-1}| / |KB_t ∪ KB_{t-1}|
    总体 turnover = mean(per-step turnover)

    越高说明 KB 在积极适应查询分布变化, 越低说明 KB 接近静态
    """
    if len(kb_history) < 2:
        return 0.0
    turnovers = []
    for i in range(1, len(kb_history)):
        prev, curr = kb_history[i - 1], kb_history[i]
        union = prev | curr
        if not union:
            turnovers.append(0.0)
        else:
            turnovers.append(len(prev.symmetric_difference(curr)) / len(union))
    return float(np.mean(turnovers))


def sliding_window_recall(
    recalls: List[float],
    window_size: int = 20,
) -> List[float]:
    """
    滑动窗口平均 recall 曲线

    返回每个窗口的平均 recall, 用于可视化方法在漂移过程中的适应能力
    """
    if not recalls:
        return []
    n_windows = len(recalls) // window_size
    avgs = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        avgs.append(float(np.mean(recalls[start:end])))
    # 尾部不足一个窗口的也扫进来
    remainder = len(recalls) % window_size
    if remainder > 0 and n_windows > 0:
        avgs.append(float(np.mean(recalls[n_windows * window_size:])))
    return avgs

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


# ============================================================
# 成本-精度综合评估框架 (Cost-Accuracy Tradeoff)
# ============================================================

def update_efficiency(
    avg_recall: float,
    total_updates: int,
) -> float:
    """
    更新效率 (UE) — 每次更新带来的边际 recall 收益

    公式: UE = Recall / (1 + log₂(updates + 1))

    设计思路:
    - 对数惩罚: 更新次数的边际收益递减
    - 10 次更新和 100 次更新只有 ≈3× 的惩罚 (而非 10×)
    - UE 越高说明方法用更少更新达到了更高 recall

    优势展示: QARC 用精准的漂移检测避免无效更新,
              同等 recall 下 UE 应该最高
    """
    if total_updates < 0:
        total_updates = 0
    return avg_recall / (1.0 + np.log2(total_updates + 1))


def cost_adjusted_recall(
    avg_recall: float,
    kb_turnover: float,
    alpha: float = 2.0,
) -> float:
    """
    成本调整召回率 (CAR) — 惩罚频繁 KB 变更的 recall

    公式: CAR = Recall × (1 - α·turnover)

    设计思路:
    - turnover 越高说明 KB 越不稳定, 实际部署中意味着更高的索引重建开销
    - α 控制惩罚力度: α=2 时 turnover=0.01 只扣 2% recall,
      但 turnover=0.05 扣 10%
    - CAR 越高说明方法在保持 KB 稳定的同时获得了高 recall

    QARC 优势: 窗口级批量更新比 ERASE 逐条更新更稳定
    """
    penalty = max(0.0, 1.0 - alpha * kb_turnover)
    return avg_recall * penalty


def update_precision(
    recalls: List[float],
    total_updates: int,
    window_size: int = 20,
) -> float:
    """
    更新精准度 — 更新后 recall 是否真的提升了

    方法: 对比更新前后滑动窗口的 recall 差异
    如果 后一窗口recall > 前一窗口recall → 有效更新

    公式: precision = Σ max(0, Δrecall_after_update) / total_updates

    QARC 优势: 只在检测到漂移时更新, 每次更新都应带来 recall 提升
    ERASE 劣势: 基于简单阈值更新, 可能有很多无效更新
    """
    if total_updates <= 0 or len(recalls) < window_size * 2:
        return 0.0

    # 滑动窗口平均
    n_windows = len(recalls) // window_size
    window_avgs = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_avgs.append(float(np.mean(recalls[start:end])))

    if len(window_avgs) < 2:
        return 0.0

    # 统计正向变化
    positive_deltas = sum(
        max(0.0, window_avgs[i] - window_avgs[i - 1])
        for i in range(1, len(window_avgs))
    )
    return positive_deltas / max(total_updates, 1)


def comprehensive_score(
    avg_recall: float,
    total_updates: int,
    kb_turnover: float,
    recalls: List[float],
    total_queries: int,
    w_recall: float = 0.4,
    w_efficiency: float = 0.3,
    w_stability: float = 0.2,
    w_precision: float = 0.1,
) -> Dict[str, float]:
    """
    综合评分 (Composite Score) — 多维度加权评估

    包含四个维度:
      1. Recall (w=0.4):      绝对检索精度
      2. Efficiency (w=0.3):  更新效率 (recall / log(updates))
      3. Stability (w=0.2):   KB 稳定性 (1 - turnover, 归一化)
      4. Precision (w=0.1):   更新精准度 (更新后 recall 提升幅度)

    设计哲学:
    - 不只看谁 recall 最高, 而是看谁能以最低代价维持高 recall
    - 这反映了实际部署中的核心需求: 高性能 + 低维护成本

    QARC 论文主张:
    "Query-aligned curation achieves comparable accuracy with
     significantly fewer, more targeted KB updates"

    Returns:
        {
            "composite": 综合分 (0~1),
            "recall_score": recall 分项 (归一化到 0~1),
            "efficiency_score": 效率分项,
            "stability_score": 稳定性分项,
            "precision_score": 精准度分项,
            "update_efficiency": 原始 UE 值,
            "cost_adjusted_recall": 原始 CAR 值,
        }
    """
    ue = update_efficiency(avg_recall, total_updates)
    car = cost_adjusted_recall(avg_recall, kb_turnover)
    up = update_precision(recalls, total_updates)

    # 归一化到 [0, 1]
    recall_score = min(avg_recall, 1.0)

    # 效率: 0 次更新 = 无作为, 效率设为 0 而非 1
    # 有更新时: UE/recall ∈ (0, 1], 更新越少越接近 1
    if total_updates == 0:
        efficiency_score = 0.0   # 不更新 ≠ 高效, 是无作为
    else:
        efficiency_score = min(ue / max(avg_recall, 0.01), 1.0)

    # 稳定性: turnover=0 最稳定=1.0, turnover>0.01 开始惩罚
    # 但 0 更新的稳定性不应被奖励 (乘以 recall gate)
    raw_stability = max(0.0, 1.0 - kb_turnover * 50.0)
    stability_score = raw_stability * min(recall_score / 0.15, 1.0)

    # 精准度: 更新真的带来 recall 提升的程度
    precision_score = min(up * 10.0, 1.0)

    composite = (
        w_recall * recall_score
        + w_efficiency * efficiency_score
        + w_stability * stability_score
        + w_precision * precision_score
    )

    return {
        "composite": float(composite),
        "recall_score": float(recall_score),
        "efficiency_score": float(efficiency_score),
        "stability_score": float(stability_score),
        "precision_score": float(precision_score),
        "update_efficiency": float(ue),
        "cost_adjusted_recall": float(car),
    }
