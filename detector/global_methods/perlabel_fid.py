"""
B2: Per-Label FID (DriftLens 风格)

论文:
    Greco et al., "Unsupervised Concept Drift Detection from Deep Learning
    Representations in Real-time"
    IEEE Transactions on Knowledge and Data Engineering (TKDE), 2025
    https://ieeexplore.ieee.org/document/11103500

    初步版本:
    Greco et al., "Drift Lens: Real-time Unsupervised Concept Drift Detection
    by Evaluating Per-Label Embedding Distributions"
    ICDMW 2021

核心思路:
    1. Offline: 用 KB 参考集建立 per-label 和 per-batch 高斯基线
       - per-batch: 对所有样本的概率向量拟合 N(μ_batch, Σ_batch)
       - per-label: 按硬预测标签分组，每组拟合 N(μ_k, Σ_k)
    2. Offline: Bootstrap 采样估计阈值
    3. Online: 每个窗口计算与基线的 FID，超过阈值判定漂移

DriftLens 原版用 embedding 向量，我们适配为 softmax 概率向量
DriftLens 原版按硬预测标签分组 (argmax) → 我们这里也保持一致作为 baseline
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from ..base import DOMAINS, NUM_CLASSES
from .utils import estimate_gaussian, frechet_distance, bootstrap_threshold

logger = logging.getLogger(__name__)


class PerLabelFIDDetector:
    """
    DriftLens 风格的 Per-Label FID 漂移检测
    """

    def __init__(
        self,
        kb_reference_probs: np.ndarray,
        n_bootstrap: int = 1000,
        threshold_percentile: float = 95.0,
        window_size: int = 50,
    ):
        """
        Args:
            kb_reference_probs: shape (N, C) — KB 所有文档过 router 的概率向量
            n_bootstrap: Bootstrap 阈值估计的采样次数
            threshold_percentile: 阈值分位数
            window_size: 检测窗口大小 (用于阈值估计)
        """
        self.ref = np.array(kb_reference_probs, dtype=np.float64)
        self.window_size = window_size

        # ===== 1. 估计基线高斯 =====
        # Per-batch: 所有样本
        self.batch_mu, self.batch_sigma = estimate_gaussian(self.ref)

        # Per-label: 按 argmax 硬分配
        self.label_mu = {}
        self.label_sigma = {}
        hard_labels = np.argmax(self.ref, axis=1)

        for k in range(NUM_CLASSES):
            mask = hard_labels == k
            subset = self.ref[mask]
            if len(subset) < 2:
                # 样本不足，用全局估计
                self.label_mu[k], self.label_sigma[k] = self.batch_mu.copy(), self.batch_sigma.copy()
                logger.warning(f"⚠️ Label {DOMAINS[k]}: 仅 {len(subset)} 条样本, 用全局替代")
            else:
                self.label_mu[k], self.label_sigma[k] = estimate_gaussian(subset)

        # ===== 2. Bootstrap 阈值 (DriftLens 策略) =====
        self.batch_threshold = bootstrap_threshold(
            self.ref, n_bootstrap, window_size, threshold_percentile,
            compute_fn=self._compute_batch_fid_from_window
        )
        logger.info(f"📏 Per-batch FID threshold = {self.batch_threshold:.4f}")

        # Per-label 阈值
        self.label_thresholds = {}
        for k in range(NUM_CLASSES):
            subset = self.ref[hard_labels == k]
            if len(subset) >= 10:
                th = bootstrap_threshold(
                    subset, n_bootstrap, min(window_size, len(subset)),
                    threshold_percentile,
                    compute_fn=lambda w, _k=k: self._compute_label_fid(w, _k)
                )
            else:
                th = self.batch_threshold * 2  # 保守阈值
            self.label_thresholds[k] = th
            logger.info(f"   {DOMAINS[k]:15s} FID threshold = {th:.4f}")

    def _compute_batch_fid_from_window(self, window: np.ndarray) -> float:
        mu, sigma = estimate_gaussian(window)
        return frechet_distance(self.batch_mu, self.batch_sigma, mu, sigma)

    def _compute_label_fid(self, window: np.ndarray, label_idx: int) -> float:
        mu, sigma = estimate_gaussian(window)
        return frechet_distance(self.label_mu[label_idx], self.label_sigma[label_idx], mu, sigma)

    def detect(self, query_probs: np.ndarray) -> Tuple[float, Dict[str, float], bool]:
        """
        检测

        Args:
            query_probs: shape (W, C)

        Returns:
            (batch_fid, per_label_fid_dict, is_shift)
        """
        # Per-batch FID
        q_mu, q_sigma = estimate_gaussian(query_probs)
        batch_fid = frechet_distance(self.batch_mu, self.batch_sigma, q_mu, q_sigma)

        # Per-label FID (硬分配)
        hard_labels = np.argmax(query_probs, axis=1)
        per_label_fid = {}
        any_label_shift = False

        for k in range(NUM_CLASSES):
            mask = hard_labels == k
            subset = query_probs[mask]
            if len(subset) < 3:
                per_label_fid[DOMAINS[k]] = 0.0
                continue

            label_mu, label_sigma = estimate_gaussian(subset)
            fid_k = frechet_distance(
                self.label_mu[k], self.label_sigma[k],
                label_mu, label_sigma
            )
            per_label_fid[DOMAINS[k]] = fid_k

            if fid_k > self.label_thresholds[k]:
                any_label_shift = True

        is_shift = (batch_fid > self.batch_threshold) or any_label_shift
        return batch_fid, per_label_fid, is_shift