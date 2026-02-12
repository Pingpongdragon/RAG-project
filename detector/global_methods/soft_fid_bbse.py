"""
Ours: Soft-Weighted FID + BBSE Ensemble

==========================
核心创新: Soft-Weighted FID
==========================

问题:
    DriftLens (Greco et al., TKDE 2025) 按 argmax 硬分配样本到 per-label 组。
    当模型不确定时 (如 P = [0.35, 0.30, 0.20, 0.15])，硬分配会:
      1. 将边界样本分到错误的组 → 污染高斯估计
      2. 丢失"模型不确定"这个重要信号

解决:
    每条样本对 **所有** label 的高斯估计都有贡献，
    权重 = P(label_k | x) (即 softmax 概率)

    加权高斯估计:
      μ_k = Σ_i w_{i,k} · x_i / Σ_i w_{i,k}
      Σ_k = Σ_i w_{i,k} · (x_i - μ_k)(x_i - μ_k)^T / Σ_i w_{i,k}

    其中 w_{i,k} = P(class=k | query_i)

    这保证:
      - 高置信样本主要贡献给对应 class 的高斯
      - 低置信样本均匀分散到所有 class → 不会污染任何单一 class
      - 模型越校准，效果越好 (所以先做 Temperature Scaling)

========================
双视角 Ensemble
========================

视角 1 — Soft FID: 检测 **条件分布** P(prob_vec | class=k) 的变化
    → 捕捉: "同一类别内的概率向量分布形状变了"
    → 例如: STEM 的问题从 "物理题" 变成 "CS 题"，虽然都预测为 STEM，
            但概率分布的置信度/形状不同

视角 2 — BBSE: 检测 **标签先验** P(class=k) 的变化
    → 捕捉: "用户感兴趣的领域比例变了"
    → 例如: 以前 40% STEM，现在 70% STEM

    BBSE 论文:
        Lipton et al., "Detecting and Correcting for Label Shift
        with Black Box Predictors", ICML 2018
        https://arxiv.org/abs/1802.03916

    校准改进:
        Alexandari et al., "Maximum Likelihood with Bias-Corrected
        Calibration is Hard-To-Beat at Label Shift Adaptation",
        ICML 2020

判定: Soft-FID AND BBSE 都报警才确认 shift → 降低假阳性
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

from ..base import DOMAINS, NUM_CLASSES
from .utils import estimate_gaussian, frechet_distance, bootstrap_threshold

logger = logging.getLogger(__name__)


class SoftFIDBBSEDetector:
    """
    Soft-Weighted FID + BBSE 双视角漂移检测
    """

    def __init__(
        self,
        kb_reference_probs: np.ndarray,
        confusion_matrix: Optional[np.ndarray] = None,
        n_bootstrap: int = 1000,
        threshold_percentile: float = 95.0,
        window_size: int = 50,
        bbse_l1_threshold: float = 0.3,
    ):
        """
        Args:
            kb_reference_probs: shape (N, C) — KB 概率向量
            confusion_matrix: shape (C, C) — C[i,j] = 真实 i 被预测为 j 的次数
            n_bootstrap: 阈值估计采样次数
            threshold_percentile: 阈值分位数
            window_size: 检测窗口大小
            bbse_l1_threshold: BBSE L1 距离阈值
        """
        self.ref = np.array(kb_reference_probs, dtype=np.float64)
        self.window_size = window_size
        self.bbse_l1_threshold = bbse_l1_threshold

        # ===== 1. Soft-Weighted Per-Label 基线高斯 =====
        # 每条 KB 文档的概率向量本身就是软权重
        self.soft_label_mu = {}
        self.soft_label_sigma = {}

        for k in range(NUM_CLASSES):
            weights = self.ref[:, k]  # P(class=k | doc_i)
            self.soft_label_mu[k], self.soft_label_sigma[k] = estimate_gaussian(
                self.ref, weights=weights
            )

        # Per-batch 基线
        self.batch_mu, self.batch_sigma = estimate_gaussian(self.ref)

        # ===== 2. Bootstrap 阈值 =====
        # Soft-FID global
        self.batch_threshold = bootstrap_threshold(
            self.ref, n_bootstrap, window_size, threshold_percentile,
            compute_fn=self._compute_soft_global_fid
        )
        logger.info(f"📏 [Ours] Soft batch FID threshold = {self.batch_threshold:.4f}")

        # Soft-FID per-label
        self.label_thresholds = {}
        for k in range(NUM_CLASSES):
            th = bootstrap_threshold(
                self.ref, n_bootstrap, window_size, threshold_percentile,
                compute_fn=lambda w, _k=k: self._compute_soft_label_fid(w, _k)
            )
            self.label_thresholds[k] = th
            logger.info(f"   {DOMAINS[k]:15s} soft FID threshold = {th:.4f}")

        # ===== 3. BBSE (Lipton et al., ICML 2018) =====
        self.kb_dist_vec = np.mean(self.ref, axis=0)
        self.C_norm = None
        if confusion_matrix is not None:
            cm = np.array(confusion_matrix, dtype=np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            self.C_norm = cm / np.maximum(row_sums, 1e-10)
            cond = np.linalg.cond(self.C_norm.T)
            logger.info(f"📊 [Ours] BBSE 混淆矩阵条件数 = {cond:.1f}")
            if cond > 100:
                logger.warning(f"⚠️ BBSE 条件数 > 100, 估计不稳定")

    def _compute_soft_global_fid(self, window: np.ndarray) -> float:
        mu, sigma = estimate_gaussian(window)
        return frechet_distance(self.batch_mu, self.batch_sigma, mu, sigma)

    def _compute_soft_label_fid(self, window: np.ndarray, label_idx: int) -> float:
        weights = window[:, label_idx]
        mu, sigma = estimate_gaussian(window, weights=weights)
        return frechet_distance(
            self.soft_label_mu[label_idx], self.soft_label_sigma[label_idx],
            mu, sigma
        )

    def _bbse_estimate(self, query_probs: np.ndarray) -> Tuple[np.ndarray, float]:
        """BBSE: C^T · w = μ̂ → w"""
        if self.C_norm is None:
            return self.kb_dist_vec.copy(), 0.0

        mu_hat = np.mean(query_probs, axis=0)
        try:
            w = np.linalg.solve(self.C_norm.T, mu_hat)
            w = np.maximum(w, 0)
            s = w.sum()
            w = w / s if s > 0 else np.ones(NUM_CLASSES) / NUM_CLASSES
        except np.linalg.LinAlgError:
            w = mu_hat

        l1 = float(np.sum(np.abs(w - self.kb_dist_vec)))
        return w, l1

    def detect(
        self, query_probs: np.ndarray
    ) -> Tuple[float, Dict[str, float], float, np.ndarray, bool]:
        """
        双视角检测

        Returns:
            (soft_fid_global, soft_fid_per_label, bbse_l1,
             estimated_dist, is_shift)
        """
        # ===== 视角 1: Soft-Weighted FID =====
        q_mu, q_sigma = estimate_gaussian(query_probs)
        soft_global = frechet_distance(self.batch_mu, self.batch_sigma, q_mu, q_sigma)

        soft_per_label = {}
        any_label_shift = False
        for k in range(NUM_CLASSES):
            weights = query_probs[:, k]
            q_label_mu, q_label_sigma = estimate_gaussian(query_probs, weights=weights)
            fid_k = frechet_distance(
                self.soft_label_mu[k], self.soft_label_sigma[k],
                q_label_mu, q_label_sigma
            )
            soft_per_label[DOMAINS[k]] = fid_k
            if fid_k > self.label_thresholds[k]:
                any_label_shift = True
                logger.info(
                    f"   [Soft-FID] {DOMAINS[k]}: {fid_k:.4f} > {self.label_thresholds[k]:.4f}"
                )

        fid_shift = (soft_global > self.batch_threshold) or any_label_shift

        # ===== 视角 2: BBSE =====
        est_dist, bbse_l1 = self._bbse_estimate(query_probs)
        bbse_shift = bbse_l1 > self.bbse_l1_threshold

        # ===== AND 投票 =====
        is_shift = fid_shift and bbse_shift

        if is_shift:
            logger.warning(
                f"🚨 [Ours] FID={soft_global:.4f} (th={self.batch_threshold:.4f}) | "
                f"BBSE L1={bbse_l1:.4f} (th={self.bbse_l1_threshold})"
            )

        return soft_global, soft_per_label, bbse_l1, est_dist, is_shift