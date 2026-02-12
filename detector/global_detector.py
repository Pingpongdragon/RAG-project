"""
Global Shift Detector — 调度层

根据 detection_method 分发到具体方法:
    "jsd_psi"       — B1: Baseline
    "perlabel_fid"  — B2: DriftLens 风格 (Greco et al., TKDE 2025)
    "ours"          — Soft-Weighted FID + BBSE Ensemble
"""

import numpy as np
from typing import Dict, Optional
import logging

from .base import DOMAINS, NUM_CLASSES, DetectionResult
from .global_shift import JSDPSIDetector, PerLabelFIDDetector, SoftFIDBBSEDetector

logger = logging.getLogger(__name__)


class GlobalShiftDetector:
    def __init__(
        self,
        kb_reference_probs: np.ndarray,
        confusion_matrix: Optional[np.ndarray] = None,
        detection_method: str = "ours",
        jsd_threshold: float = 0.1,
        psi_threshold: float = 0.2,
        bbse_l1_threshold: float = 0.3,
        n_bootstrap: int = 1000,
        threshold_percentile: float = 95.0,
        window_size: int = 50,
    ):
        self.method = detection_method
        self.kb_ref = np.array(kb_reference_probs, dtype=np.float64)
        self.kb_dist_vec = np.mean(self.kb_ref, axis=0)

        # 保存配置 (用于 update_kb)
        self._config = dict(
            confusion_matrix=confusion_matrix,
            detection_method=detection_method,
            jsd_threshold=jsd_threshold,
            psi_threshold=psi_threshold,
            bbse_l1_threshold=bbse_l1_threshold,
            n_bootstrap=n_bootstrap,
            threshold_percentile=threshold_percentile,
            window_size=window_size,
        )

        # B1
        self.jsd_psi = JSDPSIDetector(self.kb_dist_vec, jsd_threshold, psi_threshold)

        # B2
        self.perlabel_fid = PerLabelFIDDetector(
            self.kb_ref, n_bootstrap, threshold_percentile, window_size
        )

        # Ours
        self.soft_fid_bbse = SoftFIDBBSEDetector(
            self.kb_ref, confusion_matrix,
            n_bootstrap, threshold_percentile, window_size,
            bbse_l1_threshold
        )

    def detect(
        self, result: DetectionResult, query_probs: np.ndarray
    ) -> DetectionResult:
        """
        执行全局漂移检测

        Args:
            result: 当前检测结果
            query_probs: shape (W, C)

        Returns:
            更新后的 result
        """
        avg = np.mean(query_probs, axis=0)
        result.query_distribution = {
            d: float(avg[i]) for i, d in enumerate(DOMAINS)
        }

        # ===== B1: JSD + PSI =====
        if self.method == "jsd_psi":
            jsd, psi, shift = self.jsd_psi.detect(query_probs)
            result.jsd_score = jsd
            result.psi_score = psi
            result.is_global_shift = shift

        # ===== B2: Per-Label FID (DriftLens) =====
        elif self.method == "perlabel_fid":
            fid_g, fid_pl, shift = self.perlabel_fid.detect(query_probs)
            result.fid_global = fid_g
            result.fid_per_label = fid_pl
            result.is_global_shift = shift

        # ===== Ours: Soft FID + BBSE =====
        elif self.method == "ours":
            sfid_g, sfid_pl, bbse_l1, est, shift = self.soft_fid_bbse.detect(query_probs)
            result.soft_fid_global = sfid_g
            result.soft_fid_per_label = sfid_pl
            result.bbse_l1 = bbse_l1
            result.estimated_true_distribution = {
                d: float(est[i]) for i, d in enumerate(DOMAINS)
            }
            result.is_global_shift = shift

        if result.is_global_shift:
            logger.warning(
                f"🔴 GLOBAL SHIFT @ step {result.step} (method={self.method})"
            )

        return result