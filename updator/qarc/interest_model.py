"""
QARC 兴趣建模模块 — 查询窗口聚类 + 对齐度差距计算 + GMM漂移检测

=== 本文件在 QARC 框架中的角色 ===
QARC = Query-Aligned Retrieval-augmented Knowledge Curation
本模块负责"感知用户兴趣"这一核心任务，具体包含四个组件：

┌──────────────────────────────────────────────────────────┐
│ 组件1: QueryWindowBuffer — 滑动窗口缓冲区                │
│   收集连续 W 个查询的 embedding，窗口满后触发分析         │
│                                                          │
│ 组件2: AutoKMeans — 自动聚类                              │
│   对窗口内的查询 embedding 做 K-Means 聚类               │
│   自动选择最佳 K（轮廓系数），输出兴趣重心 + 权重         │
│                                                          │
│ 组件3: AlignmentGap — 对齐度差距                          │
│   G(t) = 1 - avg max_sim(query, KB)                      │
│   衡量当前 KB 与用户兴趣的匹配程度                       │
│   G≈0 表示 KB 完美覆盖，G≈1 表示完全失配                 │
│                                                          │
│ 组件4: AdaptiveThreshold — 自适应阈值 (EMA + k·MAD)      │
│   用于 Phase 2（Exploit）判断是否需要触发 KB 重新策展     │
│                                                          │
│ 组件5: GMMDriftDetector — 混合高斯漂移检测                │
│   受 DriftLens 论文启发，用 GMM 建模嵌入分布             │
│   通过对比 reference/window 两个 GMM 的 KL 散度检测漂移   │
└──────────────────────────────────────────────────────────┘

=== 与 ComRAG 聚类的区别 ===
- ComRAG: 聚类的是 QA 历史记录（累积式），用于检索路由
- QARC:   聚类的是当前窗口查询 embedding（滑动窗口），用于驱动 KB 重新策展
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class InterestCluster:
    """一个发现的兴趣主题簇。

    Attributes:
        centroid: 簇中心向量（embedding 空间）
        weight:   该簇占窗口查询的比例 α_i（用于子模函数加权）
        query_count: 分配到该簇的查询数量
    """
    centroid: np.ndarray
    weight: float
    query_count: int
    cluster_id: int = 0
    representative_queries: List[str] = field(default_factory=list)

    def __repr__(self):
        return (
            f"InterestCluster[{self.cluster_id}] "
            f"weight={self.weight:.3f}, n_queries={self.query_count}"
        )


@dataclass
class AlignmentGapResult:
    """对齐度差距计算结果。

    Attributes:
        gap: G(t) = 1 - avg_max_sim，越大表示 KB 越不匹配用户兴趣
        avg_max_sim: 每个查询与 KB 最相似文档的平均相似度
        per_query_sims: 每个查询的最大相似度列表
    """
    gap: float
    avg_max_sim: float
    per_query_sims: List[float]
    window_size: int


# ─────────────────────────────────────────────────────────
# 组件1: 查询窗口缓冲区
# ─────────────────────────────────────────────────────────

class QueryWindowBuffer:
    """滑动窗口缓冲区，积累查询直到窗口满。

    工作流程:
        1. 每来一个查询，调用 add() 存入
        2. 检查 is_full，如果满了就 flush() 取出所有数据
        3. 外部拿到数据后进行 AutoKMeans 聚类 + AlignmentGap 计算
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._embeddings: List[np.ndarray] = []
        self._texts: List[str] = []
        self._max_sims: List[float] = []

    def add(self, embedding: np.ndarray, text: str = "",
            max_sim_to_kb: float = 0.0):
        """添加一个查询到缓冲区。"""
        self._embeddings.append(embedding)
        self._texts.append(text)
        self._max_sims.append(max_sim_to_kb)

    @property
    def is_full(self) -> bool:
        return len(self._embeddings) >= self.window_size

    @property
    def size(self) -> int:
        return len(self._embeddings)

    def flush(self) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """取出并清空缓冲区内容，返回 (embeddings, texts, max_sims)。"""
        embeddings = self._embeddings.copy()
        texts = self._texts.copy()
        sims = self._max_sims.copy()
        self._embeddings.clear()
        self._texts.clear()
        self._max_sims.clear()
        return embeddings, texts, sims

    def peek_embeddings(self) -> np.ndarray:
        """查看当前缓冲区内的 embedding 矩阵（不清空）。"""
        if not self._embeddings:
            return np.empty((0, 0))
        return np.vstack(self._embeddings)


# ─────────────────────────────────────────────────────────
# 组件2: AutoKMeans 自动聚类
# ─────────────────────────────────────────────────────────
#
# 算法流程:
#   1. 对 k ∈ [k_min, k_max] 分别跑 Cosine K-Means
#   2. 对每个 k 计算轮廓系数 (silhouette score)
#   3. 选择轮廓系数最高的 k 作为最终聚类数
#   4. 输出: (centroids, labels, weights)
#
# 使用 cosine similarity 而非欧氏距离，因为 embedding 都是 L2 归一化的

def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """计算 L2 归一化向量的余弦相似度矩阵。"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normed = X / np.clip(norms, 1e-10, None)
    return X_normed @ X_normed.T


def _kmeans_cosine(
    X: np.ndarray, k: int,
    max_iter: int = 50, n_init: int = 3, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """基于余弦相似度的 K-Means。

    Args:
        X: (n, d) L2 归一化的 embedding 矩阵
        k: 聚类数

    Returns:
        (centroids, labels, inertia)
        - centroids: (k, d) 各簇中心
        - labels: (n,) 每个点的簇标签
        - inertia: 平均最大余弦相似度（越大越好）
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    best_centroids = None
    best_labels = None
    best_inertia = -np.inf

    for init_round in range(n_init):
        # KMeans++ 初始化: 选择彼此最远的初始中心
        idx = [rng.randint(n)]
        for _ in range(1, k):
            sims = X @ X[idx].T
            max_sim = sims.max(axis=1)
            probs = 1.0 - max_sim
            probs = np.clip(probs, 0.0, None)
            probs_sum = probs.sum()
            if probs_sum < 1e-12:
                idx.append(rng.randint(n))
            else:
                probs /= probs_sum
                idx.append(rng.choice(n, p=probs))

        centroids = X[idx].copy()

        for _ in range(max_iter):
            # E-step: 每个点分配到最近的中心
            sims = X @ centroids.T
            labels = sims.argmax(axis=1)

            # M-step: 更新中心为簇内均值
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    c = X[mask].mean(axis=0)
                    norm = np.linalg.norm(c)
                    new_centroids[j] = c / max(norm, 1e-10)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        sims = X @ centroids.T
        inertia = sims.max(axis=1).mean()

        if inertia > best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_inertia


def _silhouette_cosine(X: np.ndarray, labels: np.ndarray) -> float:
    """基于余弦距离的轮廓系数（Silhouette Score）。

    轮廓系数衡量聚类质量:
    - 接近 +1: 聚类紧凑且分离良好
    - 接近  0: 簇之间有重叠
    - 接近 -1: 聚类质量差
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2 or k >= n:
        return -1.0

    # 大数据集随机采样加速
    max_sample = min(n, 500)
    if n > max_sample:
        idx = np.random.choice(n, max_sample, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    # 余弦距离 = 1 - 余弦相似度
    sim_matrix = X_sample @ X_sample.T
    dist_matrix = 1.0 - sim_matrix

    m = X_sample.shape[0]
    sil_scores = np.zeros(m)

    for i in range(m):
        own_label = labels_sample[i]
        own_mask = labels_sample == own_label
        own_count = own_mask.sum()

        # a(i): 同簇平均距离
        a_i = dist_matrix[i, own_mask].sum() / (own_count - 1) if own_count > 1 else 0.0

        # b(i): 最近异簇的平均距离
        b_i = np.inf
        for label in unique_labels:
            if label == own_label:
                continue
            other_mask = labels_sample == label
            if other_mask.sum() > 0:
                b_i = min(b_i, dist_matrix[i, other_mask].mean())

        if b_i == np.inf:
            sil_scores[i] = 0.0
        else:
            sil_scores[i] = (b_i - a_i) / max(max(a_i, b_i), 1e-10)

    return float(sil_scores.mean())


def auto_kmeans(
    X: np.ndarray, k_min: int = 2, k_max: int = 10, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """自动选择 K 的 K-Means 聚类。

    算法:
        1. 对 k ∈ [k_min, min(k_max, n-1)] 跑 cosine K-Means
        2. 计算轮廓系数，选最高分的 k
        3. 返回最佳聚类的 (centroids, labels, weights)

    Args:
        X: (n, d) L2 归一化的 embedding 矩阵

    Returns:
        centroids: (k, d) 兴趣簇中心
        labels: (n,) 聚类标签
        weights: (k,) 各簇权重（查询占比 α_i）
    """
    n = X.shape[0]

    # 边界情况处理
    if n <= 1:
        return X.copy(), np.array([0]), np.array([1.0])
    if n <= k_min:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        centroids = X / np.clip(norms, 1e-10, None)
        return centroids, np.arange(n), np.ones(n) / n

    k_max_actual = min(k_max, n - 1)
    k_min_actual = min(k_min, k_max_actual)

    best_k = k_min_actual
    best_score = -2.0
    best_centroids = None
    best_labels = None

    for k in range(k_min_actual, k_max_actual + 1):
        centroids, labels, inertia = _kmeans_cosine(X, k, seed=seed)
        score = _silhouette_cosine(X, labels)

        logger.debug(f"  AutoKMeans k={k}: silhouette={score:.4f}, inertia={inertia:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_centroids = centroids
            best_labels = labels

    # 计算各簇权重
    weights = np.zeros(best_k)
    for j in range(best_k):
        weights[j] = (best_labels == j).sum() / n

    logger.info(
        f"AutoKMeans: selected k={best_k} (silhouette={best_score:.4f}), "
        f"weights={weights.tolist()}"
    )

    return best_centroids, best_labels, weights


# ─────────────────────────────────────────────────────────
# 组件3: 对齐度差距 Alignment Gap
# ─────────────────────────────────────────────────────────
#
# 公式: G(t) = 1 - (1/|W|) × Σ_{q∈W} max_{d∈K} CosSim(Emb(q), Emb(d))
#
# 直觉:
#   - 对窗口内每个查询，找 KB 中最相似的文档
#   - 取所有查询的最大相似度的平均值
#   - 用 1 减去它，得到"差距"
#   - G ≈ 0 → KB 完美覆盖用户兴趣
#   - G ≈ 1 → KB 与用户兴趣完全不匹配

def compute_alignment_gap(
    query_embeddings: np.ndarray,
    kb_embeddings: np.ndarray,
) -> AlignmentGapResult:
    """计算 Interest-KB 对齐度差距 G(t)。

    Args:
        query_embeddings: (n_queries, d) 窗口内查询的 embedding
        kb_embeddings:    (n_docs, d) 当前 KB 文档的 embedding

    Returns:
        AlignmentGapResult，包含 gap 值和详细信息
    """
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.reshape(1, -1)

    if kb_embeddings.size == 0:
        n = query_embeddings.shape[0]
        return AlignmentGapResult(gap=1.0, avg_max_sim=0.0,
                                  per_query_sims=[0.0] * n, window_size=n)

    # L2 归一化
    q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_normed = query_embeddings / np.clip(q_norms, 1e-10, None)
    kb_norms = np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
    kb_normed = kb_embeddings / np.clip(kb_norms, 1e-10, None)

    # 相似度矩阵: (n_queries, n_docs)
    sim_matrix = query_normed @ kb_normed.T
    per_query_max_sim = sim_matrix.max(axis=1)

    avg_max_sim = float(per_query_max_sim.mean())
    gap = 1.0 - avg_max_sim

    return AlignmentGapResult(
        gap=gap, avg_max_sim=avg_max_sim,
        per_query_sims=per_query_max_sim.tolist(),
        window_size=query_embeddings.shape[0],
    )


# ─────────────────────────────────────────────────────────
# 组件4: 自适应阈值 (EMA + k·MAD)
# ─────────────────────────────────────────────────────────
#
# 用于 Phase 2 (Exploit) 阶段判断是否触发 KB 重新策展
#
# 公式:
#   G_ema(t)  = β × G_ema(t-1) + (1-β) × G(t)    # 指数移动平均
#   G_mad(t)  = β × G_mad(t-1) + (1-β) × |G(t) - G_ema(t)|  # 平均绝对偏差
#   threshold = G_ema + k × G_mad
#   当 G(t) > threshold 时触发重新策展
#
# 直觉: 跟踪 Gap 的"正常范围"，超出范围说明兴趣发生了显著偏移

class AdaptiveThreshold:
    """EMA + k·MAD 自适应阈值。

    Phase 1 (Explore): 只积累历史，不做触发判断
    Phase 2 (Exploit): 用 EMA + k·MAD 计算阈值，超过则触发重新策展
    """

    def __init__(self, beta: float = 0.9, k: float = 2.0):
        """
        Args:
            beta: EMA 平滑因子（越大越平滑，对突变越迟钝）
            k:    MAD 乘数（越大阈值越宽松，越难触发）
        """
        self.beta = beta
        self.k = k
        self.g_ema: Optional[float] = None
        self.g_mad: Optional[float] = None
        self.history: List[float] = []

    def initialize_from_history(self, g_values: List[float]):
        """从 Phase 1 积累的 Gap 历史初始化（解决冷启动问题）。"""
        if not g_values:
            self.g_ema = 0.5
            self.g_mad = 0.1
            return
        self.g_ema = float(np.mean(g_values))
        self.g_mad = float(np.median(np.abs(np.array(g_values) - self.g_ema)))
        self.history = list(g_values)
        logger.info(
            f"AdaptiveThreshold initialized: EMA={self.g_ema:.4f}, "
            f"MAD={self.g_mad:.4f}, threshold={self.threshold:.4f}"
        )

    def update(self, g: float) -> bool:
        """更新阈值并判断是否超过。返回 True 表示应触发重新策展。"""
        self.history.append(g)
        if self.g_ema is None:
            self.g_ema = g
            self.g_mad = 0.0
            return False
        self.g_ema = self.beta * self.g_ema + (1 - self.beta) * g
        self.g_mad = self.beta * self.g_mad + (1 - self.beta) * abs(g - self.g_ema)
        return g > self.threshold

    @property
    def threshold(self) -> float:
        if self.g_ema is None:
            return float('inf')
        return self.g_ema + self.k * max(self.g_mad, 1e-4)

    def get_state(self) -> Dict[str, Any]:
        return {
            "g_ema": self.g_ema, "g_mad": self.g_mad,
            "threshold": self.threshold, "history_len": len(self.history),
        }


# ─────────────────────────────────────────────────────────
# 组件5: GMM 漂移检测器 (受 DriftLens 论文启发)
# ─────────────────────────────────────────────────────────
#
# === 论文背景 ===
# DriftLens (Greco et al., IEEE TKDE 2025) 用单高斯分布建模
# 每个标签的嵌入距离分布，通过分布对比检测概念漂移。
#
# === 我们的增强: 混合高斯 (GMM) ===
# 单高斯假设分布是单峰的，但用户兴趣往往是多峰的
# （比如同时关注"体育"和"科技"两个主题）。
# 使用 GMM 可以自然捕获这种多模态分布。
#
# === 算法流程 ===
#
# 1. 特征提取:
#    对每个 embedding x，计算它到各兴趣中心的相似度向量:
#    φ(x) = [CosSim(x, c₁), CosSim(x, c₂), ..., CosSim(x, cₖ)]
#    这将高维 embedding 映射到低维"兴趣距离空间"
#
# 2. 基线建模 (set_reference):
#    用当前 KB 文档的 φ(x) 拟合一个 GMM（通过 BIC 自动选分量数）
#    → 得到 reference GMM P_ref
#
# 3. 窗口检测 (compute_drift_score):
#    对新到的查询窗口计算 φ(x)，拟合 window GMM P_win
#    计算对称 KL 散度: D = (KL(P_ref||P_win) + KL(P_win||P_ref)) / 2
#    D 越大表示分布偏移越严重
#
# 4. 阈值判断:
#    用类似 AdaptiveThreshold 的 EMA + k·MAD 跟踪 D 的正常范围
#    D > threshold 时触发漂移警报

class GMMDriftDetector:
    """混合高斯模型漂移检测器。

    核心思想: 将 embedding 投影到"兴趣距离空间"，
    用 GMM 建模该空间的分布，通过 KL 散度检测分布变化。
    """

    def __init__(
        self,
        n_components_range: tuple = (1, 5),
        covariance_type: str = "diag",
        beta: float = 0.85,
        k_drift: float = 2.5,
        min_samples: int = 15,
        random_state: int = 42,
    ):
        """
        Args:
            n_components_range: GMM 分量数搜索范围 (min, max)，通过 BIC 自动选择
            covariance_type: "diag"=对角协方差（快速）, "full"=完整协方差
            beta: 漂移分数 EMA 平滑因子
            k_drift: MAD 乘数，控制漂移触发灵敏度
            min_samples: 拟合 GMM 的最小样本数
        """
        self.n_components_range = n_components_range
        self.covariance_type = covariance_type
        self.beta = beta
        self.k_drift = k_drift
        self.min_samples = min_samples
        self.random_state = random_state

        # 基线 GMM 状态
        self._ref_gmm = None
        self._ref_centroids = None
        self._ref_means = None
        self._ref_covs = None
        self._ref_weights = None

        # 漂移分数跟踪
        self._drift_ema: Optional[float] = None
        self._drift_mad: Optional[float] = None
        self._drift_history: List[float] = []
        self._is_initialized = False

    # ── 特征提取: embedding → 兴趣距离向量 ──

    @staticmethod
    def _compute_distance_features(
        embeddings: np.ndarray,
        reference_points: np.ndarray,
    ) -> np.ndarray:
        """计算距离特征: φ(x) = [CosSim(x, ref₁), ..., CosSim(x, refₘ)]。

        将高维 embedding 映射到低维兴趣距离空间。
        """
        e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(e_norms, 1e-10, None)
        r_norms = np.linalg.norm(reference_points, axis=1, keepdims=True)
        reference_points = reference_points / np.clip(r_norms, 1e-10, None)
        return embeddings @ reference_points.T  # (n, m)

    # ── GMM 拟合（BIC 模型选择）──

    def _fit_gmm(self, X: np.ndarray):
        """用 BIC 准则选择最优分量数并拟合 GMM。"""
        from sklearn.mixture import GaussianMixture

        n_min, n_max = self.n_components_range
        n_max = min(n_max, max(1, X.shape[0] // 3))
        n_min = min(n_min, n_max)

        best_gmm = None
        best_bic = np.inf

        for n_comp in range(n_min, n_max + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=self.covariance_type,
                    max_iter=100, n_init=2,
                    random_state=self.random_state,
                    reg_covar=1e-5,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except Exception:
                continue

        if best_gmm is None:
            gmm = GaussianMixture(
                n_components=1, covariance_type=self.covariance_type,
                random_state=self.random_state, reg_covar=1e-5,
            )
            gmm.fit(X)
            best_gmm = gmm

        return best_gmm

    # ── KL 散度计算 ──

    @staticmethod
    def _gmm_kl_divergence_mc(gmm_p, gmm_q, n_samples: int = 2000, seed: int = 42) -> float:
        """蒙特卡洛估计 KL(P || Q)。

        从 P 采样，计算 log p(x) - log q(x) 的均值。
        """
        rng = np.random.RandomState(seed)
        samples, _ = gmm_p.sample(n_samples)
        log_p = gmm_p.score_samples(samples)
        log_q = gmm_q.score_samples(samples)
        return max(float(np.mean(log_p - log_q)), 0.0)

    @staticmethod
    def _symmetric_kl(gmm_p, gmm_q, n_samples: int = 2000, seed: int = 42) -> float:
        """对称 KL 散度: (KL(P||Q) + KL(Q||P)) / 2。"""
        kl_pq = GMMDriftDetector._gmm_kl_divergence_mc(gmm_p, gmm_q, n_samples, seed)
        kl_qp = GMMDriftDetector._gmm_kl_divergence_mc(gmm_q, gmm_p, n_samples, seed)
        return (kl_pq + kl_qp) / 2.0

    # ── 公开接口 ──

    def set_reference(self, kb_embeddings: np.ndarray, interest_centroids: np.ndarray):
        """设置基线 GMM（在 bootstrap 后或每次重新策展后调用）。

        Args:
            kb_embeddings: 当前 KB 文档 embedding
            interest_centroids: AutoKMeans 输出的兴趣中心
        """
        if kb_embeddings.shape[0] < self.min_samples:
            logger.warning(f"GMMDrift: KB 文档数不足 ({kb_embeddings.shape[0]}), 跳过")
            return
        self._ref_centroids = interest_centroids.copy()
        features = self._compute_distance_features(kb_embeddings, self._ref_centroids)
        self._ref_gmm = self._fit_gmm(features)
        self._ref_means = self._ref_gmm.means_.copy()
        self._ref_covs = self._ref_gmm.covariances_.copy()
        self._ref_weights = self._ref_gmm.weights_.copy()
        self._is_initialized = True
        logger.info(
            f"GMMDrift: 基线设置完成, {self._ref_gmm.n_components} 个分量, "
            f"BIC={self._ref_gmm.bic(features):.1f}"
        )

    def compute_drift_score(
        self, window_embeddings: np.ndarray, interest_centroids: np.ndarray,
    ) -> Dict[str, Any]:
        """计算当前窗口的漂移分数。

        流程:
            1. 用 reference centroids 计算距离特征（保持维度一致）
            2. 拟合 window GMM
            3. 计算 symmetric KL 散度
            4. 更新 EMA 并判断是否触发

        Returns:
            包含 drift_score, triggered, threshold 等信息的字典
        """
        result = {
            "drift_score": 0.0, "triggered": False,
            "threshold": float('inf'), "window_components": 0,
            "ref_components": 0, "is_initialized": self._is_initialized,
        }

        if not self._is_initialized or window_embeddings.shape[0] < self.min_samples:
            return result

        features = self._compute_distance_features(window_embeddings, self._ref_centroids)
        window_gmm = self._fit_gmm(features)
        drift_score = self._symmetric_kl(self._ref_gmm, window_gmm)
        triggered = self._update_drift_ema(drift_score)

        result.update({
            "drift_score": drift_score, "triggered": triggered,
            "threshold": self.drift_threshold,
            "window_components": window_gmm.n_components,
            "ref_components": self._ref_gmm.n_components,
            "drift_ema": self._drift_ema, "drift_mad": self._drift_mad,
        })
        logger.info(
            f"GMMDrift: score={drift_score:.4f}, threshold={self.drift_threshold:.4f}, "
            f"triggered={triggered}"
        )
        return result

    def _update_drift_ema(self, score: float) -> bool:
        """更新漂移分数的 EMA/MAD 并检查是否超过阈值。"""
        self._drift_history.append(score)
        if self._drift_ema is None:
            self._drift_ema = score
            self._drift_mad = 0.0
            return False
        self._drift_ema = self.beta * self._drift_ema + (1 - self.beta) * score
        self._drift_mad = self.beta * self._drift_mad + (1 - self.beta) * abs(score - self._drift_ema)
        return score > self.drift_threshold

    @property
    def drift_threshold(self) -> float:
        if self._drift_ema is None:
            return float('inf')
        return self._drift_ema + self.k_drift * max(self._drift_mad, 1e-4)

    def initialize_from_explore_history(self):
        """从 Explore 阶段积累的漂移分数历史初始化 EMA/MAD。"""
        if len(self._drift_history) >= 2:
            self._drift_ema = float(np.mean(self._drift_history))
            self._drift_mad = float(np.median(np.abs(
                np.array(self._drift_history) - self._drift_ema
            )))
            logger.info(
                f"GMMDrift: 从历史初始化 — EMA={self._drift_ema:.4f}, "
                f"MAD={self._drift_mad:.4f}, threshold={self.drift_threshold:.4f}"
            )

    def get_state(self) -> Dict[str, Any]:
        return {
            "is_initialized": self._is_initialized,
            "drift_ema": self._drift_ema, "drift_mad": self._drift_mad,
            "drift_threshold": self.drift_threshold,
            "n_history": len(self._drift_history),
            "ref_components": self._ref_gmm.n_components if self._ref_gmm else 0,
        }
