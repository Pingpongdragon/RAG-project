"""
ComRAG Updater: 基于聚类的增量式 KB 更新

配合 detector/clustering_detector.py 的 ClusteringDetector 使用:
1. 接收 DetectionResult 中的簇分布
2. 按比例分配 KB 容量给各个簇
3. 低质量簇获得额外容量 boost
4. 用 centroid 向量从 DocPool 检索相关文档 (inject)
5. 过剩的簇做 FIFO 淘汰 (evict)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 接口协议
# ============================================================

class KBProtocol(Protocol):
    """知识库需要实现的接口"""
    def add_document(self, doc: Any, cluster_id: int) -> None: ...
    def remove_document(self, doc: Any, cluster_id: int) -> None: ...
    def get_documents_by_cluster(self, cluster_id: int) -> List[Any]: ...
    def count(self) -> int: ...


class DocPoolProtocol(Protocol):
    """全局文档池需要实现的接口"""
    def search_by_vector(self, vector: np.ndarray, top_k: int = 10) -> List[Any]: ...


# ============================================================
# 内存 Mock 实现 (测试用)
# ============================================================

class SimpleKB:
    """内存知识库 (测试用)"""
    def __init__(self):
        self._docs: Dict[int, List[Any]] = {}

    def add_document(self, doc: Any, cluster_id: int = -1) -> None:
        self._docs.setdefault(cluster_id, []).append(doc)

    def remove_document(self, doc: Any, cluster_id: int = -1) -> None:
        if cluster_id in self._docs:
            try:
                self._docs[cluster_id].remove(doc)
            except ValueError:
                pass

    def get_documents_by_cluster(self, cluster_id: int) -> List[Any]:
        return self._docs.get(cluster_id, [])

    def count(self) -> int:
        return sum(len(docs) for docs in self._docs.values())


class SimpleDocPool:
    """Mock 全局文档池"""
    def search_by_vector(self, vector: np.ndarray, top_k: int = 10) -> List[Any]:
        import random
        return [
            {"id": f"doc_{random.randint(1000, 9999)}", "embedding": vector}
            for _ in range(top_k)
        ]


# ============================================================
# ComRAG Updater
# ============================================================

class ComRAGUpdater:
    """
    ComRAG 风格的增量式 KB 更新器。

    核心策略:
    - KB 总容量按簇的 query 分布比例分配
    - 低质量簇获得 boost (优先补充知识)
    - inject: 用 centroid 向 DocPool 检索
    - evict: FIFO 淘汰最旧文档
    """

    def __init__(
        self,
        kb: Any = None,
        doc_pool: Any = None,
        max_kb_size: int = 1000,
        low_quality_boost: float = 1.5,
    ):
        self.kb = kb or SimpleKB()
        self.doc_pool = doc_pool or SimpleDocPool()
        self.max_kb_size = max_kb_size
        self.low_quality_boost = low_quality_boost

        self._cluster_docs: Dict[int, List[Any]] = {}
        self.update_count = 0
        self.total_cost = 0

    def should_update(self, detection_result) -> bool:
        """判断是否需要更新 (新簇出现 or 分布漂移)"""
        return detection_result.is_new_cluster or detection_result.is_distribution_shift

    def update(self, detection_result, detector=None) -> Dict:
        """
        根据 DetectionResult 增量调整 KB。

        Args:
            detection_result: ClusteringDetector.detect() 的返回值
            detector: ClusteringDetector 实例 (用于获取 centroid 和质量信息)

        Returns:
            操作摘要 dict
        """
        if not self.should_update(detection_result):
            return {"action": "no_update"}

        dist = detection_result.cluster_distribution
        if not dist:
            return {"action": "no_update"}

        # 确定低质量簇
        low_q_ids = set()
        if detector:
            for c in detector.get_low_quality_clusters():
                low_q_ids.add(c.cluster_id)

        # 1. 计算目标容量
        target_counts = self._compute_targets(dist, low_q_ids)

        # 2. 增量调整
        added_total = 0
        removed_total = 0

        for cid, target in target_counts.items():
            current_docs = self._cluster_docs.get(cid, [])
            delta = target - len(current_docs)

            if delta > 0 and detector:
                added = self._inject(cid, delta, detector)
                added_total += added
            elif delta < 0:
                removed = self._evict(cid, abs(delta))
                removed_total += removed

        self.update_count += 1
        cost = added_total + removed_total
        self.total_cost += cost

        return {
            "action": "incremental_align",
            "added": added_total,
            "removed": removed_total,
            "cost": cost,
            "total_clusters": len(dist),
        }

    def _compute_targets(self, dist: Dict[int, float], low_q_ids: set) -> Dict[int, int]:
        """按分布比例 + 低质量 boost 计算各簇目标容量"""
        raw = {}
        for cid, ratio in dist.items():
            w = ratio
            if cid in low_q_ids:
                w *= self.low_quality_boost
            raw[cid] = w

        total_w = sum(raw.values()) or 1.0
        return {
            cid: max(1, int(self.max_kb_size * w / total_w))
            for cid, w in raw.items()
        }

    def _inject(self, cluster_id: int, count: int, detector) -> int:
        """从 DocPool 检索文档注入 KB"""
        if cluster_id not in detector.clusters:
            return 0

        centroid = detector.clusters[cluster_id].centroid
        candidates = self.doc_pool.search_by_vector(centroid, top_k=count * 2)

        added = 0
        for doc in candidates:
            if added >= count:
                break
            self.kb.add_document(doc, cluster_id=cluster_id)
            self._cluster_docs.setdefault(cluster_id, []).append(doc)
            added += 1

        if added:
            logger.info(f"Injected {added} docs into cluster {cluster_id}")
        return added

    def _evict(self, cluster_id: int, count: int) -> int:
        """FIFO 淘汰"""
        docs = self._cluster_docs.get(cluster_id, [])
        to_remove = docs[:count]
        self._cluster_docs[cluster_id] = docs[count:]

        for doc in to_remove:
            self.kb.remove_document(doc, cluster_id=cluster_id)

        if to_remove:
            logger.info(f"Evicted {len(to_remove)} docs from cluster {cluster_id}")
        return len(to_remove)

    def inject_for_cluster(self, cluster_id: int, centroid: np.ndarray, count: int = 10):
        """手动为指定簇补充文档 (low-quality 簇主动补充)"""
        candidates = self.doc_pool.search_by_vector(centroid, top_k=count)
        for doc in candidates:
            self.kb.add_document(doc, cluster_id=cluster_id)
            self._cluster_docs.setdefault(cluster_id, []).append(doc)
        logger.info(f"Manual inject: {len(candidates)} docs for cluster {cluster_id}")

    def get_statistics(self) -> Dict:
        return {
            "update_count": self.update_count,
            "total_cost": self.total_cost,
            "kb_size": self.kb.count() if hasattr(self.kb, 'count') else "N/A",
            "clusters_tracked": len(self._cluster_docs),
            "strategy": "ComRAG (Dynamic Incremental)",
        }
