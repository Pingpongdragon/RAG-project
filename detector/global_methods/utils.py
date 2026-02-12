"""
共享工具函数

Fréchet Distance (FID):
    原始用于 GAN 图像质量评估:
        Heusel et al., "GANs Trained by a Two Time-Scale Update Rule
        Converge to a Local Nash Equilibrium", NeurIPS 2017

    DriftLens 将其用于 drift detection:
        Greco et al., "Unsupervised Concept Drift Detection from Deep Learning
        Representations in Real-time", IEEE TKDE 2025
        https://ieeexplore.ieee.org/document/11103500

    公式: FID(P,Q) = ||μ_P - μ_Q||² + Tr(Σ_P + Σ_Q - 2(Σ_P Σ_Q)^{1/2})
    当维度很低(如 4 类) 时简化为标量方差情况: 退化但仍有意义
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def estimate_gaussian(
    probs: np.ndarray,
    weights: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计高斯参数 (μ, Σ)

    Args:
        probs: shape (N, C) — 概率向量
        weights: shape (N,) — 每条样本的权重 (用于 soft 分配)
                 None → 均匀权重 (等价于硬分配)

    Returns:
        mean: shape (C,)
        cov: shape (C, C)
    """
    if probs.shape[0] == 0:
        C = probs.shape[1] if probs.ndim == 2 else 1
        return np.zeros(C), np.eye(C) * 1e-6

    if weights is None:
        mean = np.mean(probs, axis=0)
        cov = np.cov(probs.T, ddof=0)
    else:
        # 加权均值和协方差
        w = weights / weights.sum()
        mean = np.average(probs, axis=0, weights=w)
        diff = probs - mean
        cov = (diff * w[:, None]).T @ diff

    # 正则化，防止奇异
    if cov.ndim < 2:
        cov = np.array([[max(cov, 1e-6)]])
    else:
        cov += np.eye(cov.shape[0]) * 1e-6

    return mean, cov


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """
    Fréchet Distance (FID)

    Heusel et al., NeurIPS 2017 (原始 FID)
    Greco et al., IEEE TKDE 2025 (用于 drift detection)

    FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2·(Σ1·Σ2)^{1/2})

    小维度 (C=4) 下数值稳定版本
    """
    diff = mu1 - mu2
    diff_sq = np.dot(diff, diff)

    product = sigma1 @ sigma2
    sqrt_product = sqrtm(product)

    # sqrtm 可能返回复数 (数值误差)
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    trace_term = np.trace(sigma1 + sigma2 - 2 * sqrt_product)
    fid = float(diff_sq + trace_term)
    return max(fid, 0.0)  # 数值误差可能导致微小负值


def bootstrap_threshold(
    reference_probs: np.ndarray,
    n_bootstrap: int = 1000,
    window_size: int = 50,
    percentile: float = 95.0,
    compute_fn=None
) -> float:
    """
    Bootstrap 阈值估计

    从参考集中重复采样窗口，计算统计量的分布，
    取 percentile 分位数作为阈值

    这是 DriftLens (Greco et al., TKDE 2025) 的阈值估计策略:
    "Estimates threshold distance values from the threshold dataset
     to discriminate between drift and no-drift conditions"

    Args:
        reference_probs: shape (N, C)
        n_bootstrap: 重采样次数
        window_size: 每次采样的窗口大小
        percentile: 阈值对应的分位数
        compute_fn: 输入 (window_probs,) → 返回 float 统计量

    Returns:
        threshold: float
    """
    if compute_fn is None:
        raise ValueError("必须提供 compute_fn")

    N = reference_probs.shape[0]
    stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = np.random.choice(N, size=min(window_size, N), replace=True)
        window = reference_probs[idx]
        stats[i] = compute_fn(window)

    # 去极端值后取分位数 (DriftLens 的做法)
    q_low, q_high = np.percentile(stats, [1, 99])
    trimmed = stats[(stats >= q_low) & (stats <= q_high)]
    threshold = float(np.percentile(trimmed, percentile))
    return threshold