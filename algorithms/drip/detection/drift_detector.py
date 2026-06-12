
"""
DRIP Part 1: 对齐漂移检测 — 基于 Query-KB 对齐特征的 FID

=== 核心思想 ===
检测的不是 "query 分布是否偏移", 也不是 "document 分布是否偏移",
而是: "query 与 KB 的对齐模式是否偏离了历史正常水平"

query 变了 ≠ 要更新 KB (新 query 可能仍然被 KB 覆盖)
query-KB 不对齐了 → 才需要更新 KB

=== 对齐特征 (Alignment Features) ===
对每个 query, 不使用 raw embedding, 而是计算与 KB 的对齐特征:

    alignment_features = [
        sim(q, centroid_1), ..., sim(q, centroid_K),   # 与 K 个 KB 主题的匹配度
        top1_sim, top2_sim, ..., topN_sim,             # 与最相似 N 篇 KB 文档的匹配度
    ]

=== 检测方法: 正则化 FID ===
DriftLens 原始方法在小窗口上因协方差矩阵奇异而崩溃
(window_size ≈ feat_dim → np.cov 不满秩 → sqrtm 失败)

本实现使用正则化 FID (Fréchet Inception Distance) 的变体:
  1. Offline: 存储基线对齐特征的均值 μ₀ 和正则化协方差 Σ₀
  2. Threshold: 随机采样窗口 → 计算 FID → 取百分位值
  3. Online: FID(window, baseline) > threshold → 漂移

正则化: Σ = np.cov(X) + ε·I, 确保可逆

=== 触发含义 ===
FID > threshold → 新 query 与 KB 的匹配模式偏离了历史正常水平 → KB 需更新
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from scipy import linalg
from algorithms.drip.interfaces import BaseDriftDetector

logger = logging.getLogger(__name__)


from algorithms.drip.interfaces import DriftResult  # 从接口层导入


class DriftLensDetector(BaseDriftDetector):
    """基于 Query-KB 对齐特征的漂移检测器。

    使用正则化 FID 在对齐特征空间中检测分布偏移。

    流程:
        detector = DriftLensDetector()
        detector.set_baseline(kb_embs, history_query_embs)
        detector.calibrate_threshold(history_query_embs, window_size)
        result = detector.detect(window_query_embs)
    """

    def __init__(
        self,
        n_clusters: int = 5,
        top_n_sims: int = 10,
        threshold_percentile: float = 95.0,
        threshold_n_samples: int = 500,
        cov_reg: float = 1e-5,
        random_state: int = 42,
        # 以下参数保留接口兼容性但不再使用
        batch_n_pc: int = 10,
        per_label_n_pc: int = 5,
    ):
        """
        Args:
            n_clusters:           KB 文档聚类数 (主题数)
            top_n_sims:           对齐特征中 top-N 相似度个数
            threshold_percentile: 阈值百分位 (P95 = 5% 假阳率)
            threshold_n_samples:  阈值校准采样次数
            cov_reg:              协方差正则化系数 ε (Σ + ε·I)
            random_state:         随机种子
        """
        self.n_clusters = n_clusters
        self.top_n_sims = top_n_sims
        self.threshold_percentile = threshold_percentile
        self.threshold_n_samples = threshold_n_samples
        self.cov_reg = cov_reg
        self.random_state = random_state

        # 内部状态
        self._threshold: Optional[float] = None
        self._is_ready = False
        self._score_history: List[float] = []

        # KB 结构
        self._kb_centroids: Optional[np.ndarray] = None   # (K, d)
        self._kb_embeddings: Optional[np.ndarray] = None  # (n_kb, d)

        # 基线统计 (对齐特征空间)
        self._baseline_mean: Optional[np.ndarray] = None  # (feat_dim,)
        self._baseline_cov: Optional[np.ndarray] = None   # (feat_dim, feat_dim)

    # ─── 对齐特征计算 ───

    def _compute_alignment_features(
        self, query_embs: np.ndarray
    ) -> np.ndarray:
        """计算 query-KB 对齐特征。

        对每个 query:
          [sim(q,c1), ..., sim(q,cK), top1_sim, ..., topN_sim]

        Returns:
            alignment_features: (n, K + top_n) float64
        """
        K = self._kb_centroids.shape[0]
        top_n = min(self.top_n_sims, self._kb_embeddings.shape[0])

        # 与 KB 主题中心的相似度 (n, K)
        centroid_sims = query_embs @ self._kb_centroids.T

        # 与所有 KB 文档的相似度 → 取 top-N (n, top_n)
        all_doc_sims = query_embs @ self._kb_embeddings.T  # (n, n_kb)

        if all_doc_sims.shape[1] <= top_n:
            topn_sims = np.sort(all_doc_sims, axis=1)[:, ::-1]
        else:
            idx = np.argpartition(all_doc_sims, -top_n, axis=1)[:, -top_n:]
            topn_sims = np.take_along_axis(all_doc_sims, idx, axis=1)
            topn_sims = np.sort(topn_sims, axis=1)[:, ::-1]

        alignment_features = np.concatenate(
            [centroid_sims, topn_sims], axis=1
        ).astype(np.float64)

        return alignment_features

    # ─── 正则化 FID ───

    def _regularized_fid(
        self,
        mu1: np.ndarray, sigma1: np.ndarray,
        mu2: np.ndarray, sigma2: np.ndarray,
    ) -> float:
        """计算正则化 Fréchet Inception Distance。

        FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·√(Σ₁·Σ₂))

        协方差矩阵已经过 ε·I 正则化, 确保可逆。
        """
        diff = mu1 - mu2
        diff_sq = np.sum(diff ** 2)

        product = sigma1 @ sigma2
        covmean = linalg.sqrtm(product)

        # sqrtm 可能返回复数 (由于数值误差)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)

        fid = diff_sq + tr
        return max(float(fid), 0.0)

    def _compute_stats(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算均值和正则化协方差。"""
        mu = features.mean(axis=0)
        if features.shape[0] < 2:
            cov = np.eye(features.shape[1]) * self.cov_reg
        else:
            cov = np.cov(features, rowvar=False)
            cov += np.eye(cov.shape[0]) * self.cov_reg
        return mu, cov

    # ─── Offline Phase ───

    def set_baseline(
        self,
        kb_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> bool:
        """Offline: 建立"正常 query-KB 对齐"的基线分布。

        1. KMeans(KB 文档) → K 个主题中心
        2. 历史 query → 对齐特征 (与 KB 的匹配度)
        3. 计算对齐特征的均值和正则化协方差 → 基线

        Args:
            kb_embeddings:    当前 KB embedding (n_kb, d)
            query_embeddings: 历史 query embedding (n_q, d)

        Returns:
            是否成功
        """
        from sklearn.cluster import KMeans

        n_kb = kb_embeddings.shape[0]
        n_q = query_embeddings.shape[0]

        if n_kb < 5:
            logger.warning(f"DriftLens: KB 不足 ({n_kb})")
            return False
        if n_q < 5:
            logger.warning(f"DriftLens: query 不足 ({n_q})")
            return False

        # Step 1: KMeans on KB → 主题结构
        k = min(self.n_clusters, n_kb // 3, n_q // 3)
        k = max(k, 2)
        kmeans = KMeans(
            n_clusters=k, random_state=self.random_state, n_init=3
        )
        kmeans.fit(kb_embeddings)
        self._kb_centroids = kmeans.cluster_centers_
        self._kb_embeddings = kb_embeddings.copy()

        # Step 2: 计算历史 query 的对齐特征
        alignment_feats = self._compute_alignment_features(query_embeddings)

        # Step 3: 基线统计
        self._baseline_mean, self._baseline_cov = self._compute_stats(
            alignment_feats
        )

        self._is_ready = True

        feat_dim = alignment_feats.shape[1]
        logger.info(
            f"DriftLens Offline: 对齐基线 — "
            f"KB={n_kb}, queries={n_q}, K={k}, feat_dim={feat_dim}"
        )
        return True

    def calibrate_threshold(
        self,
        query_embeddings: np.ndarray,
        window_size: int,
    ) -> Optional[float]:
        """Offline: 用历史 query 的对齐特征校准阈值。

        从历史 query 随机采样窗口 → 计算 FID → 取百分位值。

        Args:
            query_embeddings: 历史 query embedding (n_q, d)
            window_size:      窗口大小

        Returns:
            校准的阈值, 或 None
        """
        if not self._is_ready:
            logger.warning("DriftLens: 未设基线, 无法校准")
            return None

        n_q = query_embeddings.shape[0]
        if n_q < 2 * window_size:
            logger.warning(
                f"DriftLens: query 不足 ({n_q} < {2 * window_size})"
            )
            return None

        alignment_feats = self._compute_alignment_features(query_embeddings)

        rng = np.random.RandomState(self.random_state)
        fid_scores = []
        for _ in range(self.threshold_n_samples):
            idx = rng.choice(len(alignment_feats), size=window_size, replace=True)
            win_feats = alignment_feats[idx]
            mu_w, cov_w = self._compute_stats(win_feats)
            fid = self._regularized_fid(
                self._baseline_mean, self._baseline_cov, mu_w, cov_w
            )
            if np.isfinite(fid):
                fid_scores.append(fid)

        if not fid_scores:
            logger.warning("DriftLens: 校准无有效 FID 值")
            return None

        fid_scores.sort()
        idx = int(len(fid_scores) * self.threshold_percentile / 100.0)
        idx = min(idx, len(fid_scores) - 1)
        self._threshold = fid_scores[idx]

        logger.info(
            f"DriftLens Offline: 阈值校准 — "
            f"n_samples={len(fid_scores)}, "
            f"P{self.threshold_percentile:.0f}={self._threshold:.4f}"
        )
        return self._threshold

    # ─── Online Phase ───

    def detect(self, window_query_embs: np.ndarray) -> DriftResult:
        """Online: 检测当前窗口 query-KB 对齐模式是否偏离基线。

        1. 计算窗口 query 的对齐特征
        2. 正则化 FID(窗口 vs 基线)
        3. FID > threshold → KB 需更新

        Args:
            window_query_embs: 当前窗口 query embedding (w, d)

        Returns:
            DriftResult
        """
        default = DriftResult(
            fid_score=0.0, threshold=self.threshold, is_drifted=False,
        )

        if not self._is_ready or window_query_embs.shape[0] < 2:
            return default

        alignment_feats = self._compute_alignment_features(window_query_embs)
        mu_w, cov_w = self._compute_stats(alignment_feats)
        fid = self._regularized_fid(
            self._baseline_mean, self._baseline_cov, mu_w, cov_w
        )

        if not np.isfinite(fid):
            logger.warning(f"DriftLens: FID 非有限值 ({fid})")
            return default

        is_drifted = fid > self.threshold
        self._score_history.append(fid)

        logger.info(
            f"DriftLens: FID={fid:.4f}, "
            f"threshold={self.threshold:.4f}, "
            f"drifted={is_drifted}"
        )
        return DriftResult(
            fid_score=fid, threshold=self.threshold, is_drifted=is_drifted,
        )

    # ─── Properties ───

    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else float('inf')

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def score_history(self) -> List[float]:
        return list(self._score_history)

    def get_state(self) -> Dict[str, Any]:
        return {
            "is_ready": self._is_ready,
            "threshold": self.threshold,
            "n_clusters": (
                self._kb_centroids.shape[0] if self._kb_centroids is not None else 0
            ),
            "n_scores": len(self._score_history),
            "detection_method": "regularized_FID(alignment_features)",
        }
