"""
QARC 兴趣建模工具 — 查询窗口缓冲 + 自动聚类 + 对齐度差距

本模块提供 QARC 流水线所需的基础工具:

┌──────────────────────────────────────────────────────────┐
│ QueryWindowBuffer — 滑动窗口缓冲区                       │
│   收集连续 W 个查询的 embedding，窗口满后触发分析         │
│                                                          │
│ auto_kmeans — 自动聚类                                    │
│   对窗口内查询 embedding 做 Cosine K-Means               │
│   自动选择最佳 K（轮廓系数），输出兴趣中心 + 权重         │
│                                                          │
│ compute_alignment_gap — 对齐度差距 G(t)                   │
│   G(t) = 1 - avg max_sim(query, KB)                      │
│   衡量当前 KB 与用户兴趣的匹配程度                       │
│   G≈0 表示 KB 完美覆盖，G≈1 表示完全失配                 │
└──────────────────────────────────────────────────────────┘

漂移检测逻辑已移至 drift_detector.py (Part 1: DriftLens)
更新决策逻辑已移至 kb_agent.py (Part 2: Agent-in-the-Loop)
"""

import numpy as np
import logging
from typing import List, Optional, Tuple
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
        weight:   该簇占窗口查询的比例（用于 submodular 加权）
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
# 查询窗口缓冲区
# ─────────────────────────────────────────────────────────

class QueryWindowBuffer:
    """滑动窗口缓冲区，积累查询直到窗口满。

    工作流程:
        1. 每来一个查询，调用 add() 存入
        2. 检查 is_full，如果满了就 flush() 取出所有数据
        3. 外部拿到数据后进行 auto_kmeans 聚类 + alignment_gap 计算
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
# Cosine K-Means 自动聚类
# ─────────────────────────────────────────────────────────
#
# 使用余弦相似度 (而非欧氏距离), 因为 embedding 都是 L2 归一化的
# 对 k ∈ [k_min, k_max] 分别跑, 用轮廓系数选最优 k

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
        # KMeans++ 初始化
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
            sims = X @ centroids.T
            labels = sims.argmax(axis=1)

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
    """基于余弦距离的轮廓系数 (Silhouette Score)。

    接近 +1: 聚类紧凑且分离良好
    接近  0: 簇之间有重叠
    接近 -1: 聚类质量差
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2 or k >= n:
        return -1.0

    max_sample = min(n, 500)
    if n > max_sample:
        idx = np.random.choice(n, max_sample, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    sim_matrix = X_sample @ X_sample.T
    dist_matrix = 1.0 - sim_matrix

    m = X_sample.shape[0]
    sil_scores = np.zeros(m)

    for i in range(m):
        own_label = labels_sample[i]
        own_mask = labels_sample == own_label
        own_count = own_mask.sum()

        a_i = dist_matrix[i, own_mask].sum() / (own_count - 1) if own_count > 1 else 0.0

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
    """自动选择 K 的 Cosine K-Means 聚类。

    遍历 k ∈ [k_min, k_max], 选轮廓系数最高的 k。

    Args:
        X: (n, d) L2 归一化的 embedding 矩阵

    Returns:
        centroids: (k, d) 兴趣簇中心
        labels:    (n,) 聚类标签
        weights:   (k,) 各簇权重 (查询占比)
    """
    n = X.shape[0]

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

        logger.debug(
            f"  AutoKMeans k={k}: silhouette={score:.4f}, inertia={inertia:.4f}"
        )

        if score > best_score:
            best_score = score
            best_k = k
            best_centroids = centroids
            best_labels = labels

    weights = np.zeros(best_k)
    for j in range(best_k):
        weights[j] = (best_labels == j).sum() / n

    logger.info(
        f"AutoKMeans: k={best_k} (silhouette={best_score:.4f}), "
        f"weights={weights.tolist()}"
    )

    return best_centroids, best_labels, weights


# ─────────────────────────────────────────────────────────
# 对齐度差距 Alignment Gap
# ─────────────────────────────────────────────────────────
#
# G(t) = 1 - (1/|W|) * sum_{q in W} max_{d in K} CosSim(q, d)
#
# G ≈ 0 → KB 完美覆盖用户兴趣
# G ≈ 1 → KB 与用户兴趣完全不匹配

def compute_alignment_gap(
    query_embeddings: np.ndarray,
    kb_embeddings: np.ndarray,
    precomputed_max_sims: Optional[List[float]] = None,
) -> AlignmentGapResult:
    """计算 Interest-KB 对齐度差距 G(t)。

    优先使用 precomputed_max_sims（来自 RAG 检索已计算的 max sim），
    避免对 KB embedding 做冗余全量相似度计算。

    注意: QARC 的 KB 是经 budget 约束的小型文档子集（默认 50 篇），
    不是完整的文档池。因此即使 fallback 到暴力计算也是 O(W×B) 量级。

    Args:
        query_embeddings:     (n_queries, d) 窗口查询 embedding
        kb_embeddings:        (n_docs, d) KB 文档 embedding
        precomputed_max_sims: 每条 query 在 RAG 检索时已得到的
                              max CosSim(q, KB)，如提供则直接复用。

    Returns:
        AlignmentGapResult
    """
    n = query_embeddings.shape[0] if query_embeddings.ndim > 1 else 1

    # 优先复用 RAG 检索时已计算的 max_sim，避免冗余计算
    if precomputed_max_sims is not None and len(precomputed_max_sims) == n:
        per_query_max_sim = np.array(precomputed_max_sims)
    else:
        # Fallback: 重新计算（bootstrap 或无预计算值时）
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if kb_embeddings.size == 0:
            return AlignmentGapResult(gap=1.0, avg_max_sim=0.0,
                                      per_query_sims=[0.0] * n, window_size=n)

        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_normed = query_embeddings / np.clip(q_norms, 1e-10, None)
        kb_norms = np.linalg.norm(kb_embeddings, axis=1, keepdims=True)
        kb_normed = kb_embeddings / np.clip(kb_norms, 1e-10, None)

        sim_matrix = query_normed @ kb_normed.T
        per_query_max_sim = sim_matrix.max(axis=1)

    avg_max_sim = float(per_query_max_sim.mean())
    gap = 1.0 - avg_max_sim

    return AlignmentGapResult(
        gap=gap, avg_max_sim=avg_max_sim,
        per_query_sims=per_query_max_sim.tolist(),
        window_size=n,
    )

