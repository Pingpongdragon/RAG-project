"""
core/kb_base.py — 统一知识库基础设施 (topic-based, 无硬编码领域)

设计思想:
  - 基于 topic 聚类, 不预设固定领域
  - 文档按 embedding 相似度自动归入 TopicCluster
  - 热度衰减 + 容量淘汰 保证 KB 大小可控
  - 对外提供 add / search / evict / statistics 四类操作

                ┌─────────────────────────────────┐
                │     TopicKnowledgeBase           │
                │  capacity: int                   │
                │  clusters: List[TopicCluster]    │
                │  ──────────────────────          │
                │  add_document(doc, step)         │
                │  search(query_vec, step, top_k)  │
                │  evict(n)                        │
                │  get_all_doc_ids() -> Set[str]   │
                └─────────────────────────────────┘
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class KBDocument:
    """知识库文档 — 最小存储单元"""
    doc_id: str
    content: str
    embedding: np.ndarray
    topic: str = ""            # 文档所属 topic (可选, 由外部标注)
    title: str = ""
    access_count: int = 0
    last_access_step: int = -1
    added_step: int = 0


class TopicCluster:
    """
    话题簇 — 语义相近的文档自动聚合

    每个簇维护:
      - centroid: 所有文档 embedding 的归一化均值
      - heat: 访问热度 (指数衰减)
      - docs: 簇内文档列表
    """

    def __init__(self, cluster_id: str, centroid: np.ndarray):
        self.cluster_id = cluster_id
        self.centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        self.docs: List[KBDocument] = []
        self.heat: float = 1.0
        self.last_access_step: int = -1
        self.creation_step: int = -1
        self.size: int = 0

    def add_doc(self, doc: KBDocument):
        """添加文档并增量更新质心"""
        self.docs.append(doc)
        self.size += 1
        # 增量更新质心: new_centroid = mean(所有 doc embeddings)
        embeddings = np.array([d.embedding for d in self.docs])
        self.centroid = np.mean(embeddings, axis=0)
        self.centroid = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)

    def update_heat(self, step: int, decay: float = 0.95):
        """访问时更新热度 (指数衰减 + 1)"""
        if self.last_access_step >= 0:
            elapsed = step - self.last_access_step
            self.heat = self.heat * (decay ** elapsed) + 1.0
        else:
            self.heat += 1.0
        self.last_access_step = step

    def get_virtual_heat(self, step: int, decay: float = 0.95) -> float:
        """不修改状态, 只读取当前虚拟热度"""
        if self.last_access_step >= 0:
            elapsed = step - self.last_access_step
            return self.heat * (decay ** elapsed)
        return self.heat

    def similarity(self, query_vec: np.ndarray) -> float:
        """与查询向量的余弦相似度"""
        return float(np.dot(self.centroid, query_vec) / (
            np.linalg.norm(self.centroid) * np.linalg.norm(query_vec) + 1e-8
        ))

    def remove_coldest(self, n: int) -> int:
        """移除 n 个最冷的文档 (按 last_access_step 升序)"""
        if n <= 0 or not self.docs:
            return 0
        n = min(n, len(self.docs))
        self.docs.sort(key=lambda d: (d.last_access_step, d.access_count))
        removed = self.docs[:n]
        self.docs = self.docs[n:]
        self.size = len(self.docs)
        # 重新计算质心
        if self.docs:
            embeddings = np.array([d.embedding for d in self.docs])
            self.centroid = np.mean(embeddings, axis=0)
            self.centroid = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)
        return len(removed)


# ============================================================
# 主类: TopicKnowledgeBase
# ============================================================

class TopicKnowledgeBase:
    """
    基于 topic 聚类的知识库 — 无硬编码领域

    文档通过 embedding 相似度自动归入最近的 TopicCluster,
    若不够相似则创建新簇. 超出容量时淘汰最冷的簇/文档.

    参数:
        capacity:             KB 最大文档数
        similarity_threshold: 归入已有簇的最低余弦相似度
        max_clusters:         最大簇数量
        heat_decay:           热度衰减系数
    """

    def __init__(
        self,
        capacity: int = 5000,
        similarity_threshold: float = 0.40,
        max_clusters: int = 50,
        heat_decay: float = 0.95,
    ):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.max_clusters = max_clusters
        self.heat_decay = heat_decay

        self.clusters: List[TopicCluster] = []
        self._doc_count: int = 0
        self._cluster_counter: int = 0
        self._doc_id_set: Set[str] = set()  # 快速去重

    # ---- 写操作 ----

    def add_document(self, doc: KBDocument, step: int = 0) -> bool:
        """
        添加文档到 KB

        1. 找最近簇 (cosine sim > threshold) → 归入
        2. 否则创建新簇 (若未达上限)
        3. 超容量则淘汰
        """
        if doc.doc_id in self._doc_id_set:
            return False  # 去重

        doc.added_step = step

        # 找最近簇
        best_cluster = None
        best_sim = -1.0
        for cluster in self.clusters:
            sim = cluster.similarity(doc.embedding)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster and best_sim >= self.similarity_threshold:
            best_cluster.add_doc(doc)
            best_cluster.update_heat(step)
        elif len(self.clusters) < self.max_clusters:
            # 创建新簇
            new_cluster = TopicCluster(
                cluster_id=f"topic_{self._cluster_counter}",
                centroid=doc.embedding.copy(),
            )
            self._cluster_counter += 1
            new_cluster.add_doc(doc)
            new_cluster.update_heat(step)
            new_cluster.creation_step = step
            self.clusters.append(new_cluster)
        elif best_cluster:
            # 已达上限, 强制归入最近簇
            best_cluster.add_doc(doc)
            best_cluster.update_heat(step)
        else:
            return False

        self._doc_count += 1
        self._doc_id_set.add(doc.doc_id)

        # 容量检查
        while self._doc_count > self.capacity and self.clusters:
            self._evict(step)

        return True

    def add_documents_batch(
        self, docs: List[KBDocument], step: int = 0
    ) -> int:
        """批量添加, 返回成功添加的数量"""
        added = 0
        for doc in docs:
            if self.add_document(doc, step):
                added += 1
        return added

    def remove_document(self, doc_id: str) -> bool:
        """按 doc_id 删除文档"""
        for cluster in self.clusters:
            for i, doc in enumerate(cluster.docs):
                if doc.doc_id == doc_id:
                    cluster.docs.pop(i)
                    cluster.size -= 1
                    self._doc_count -= 1
                    self._doc_id_set.discard(doc_id)
                    # 空簇清理
                    if cluster.size == 0:
                        self.clusters.remove(cluster)
                    return True
        return False

    def clear(self):
        """清空所有文档"""
        self.clusters.clear()
        self._doc_count = 0
        self._doc_id_set.clear()

    # ---- 读操作 ----

    def search(
        self,
        query_vec: np.ndarray,
        step: int = 0,
        top_k: int = 10,
        n_probe_clusters: int = 3,
    ) -> List[Tuple[KBDocument, float]]:
        """
        检索: 先定位 top-n_probe 簇, 再在簇内精排

        Args:
            query_vec: 查询向量 (D,)
            step: 当前步 (用于更新热度)
            top_k: 返回文档数
            n_probe_clusters: 探测簇数

        Returns:
            List[(KBDocument, score)] 按相似度降序
        """
        if not self.clusters:
            return []

        # 1. 簇级粗排
        cluster_sims = [(c, c.similarity(query_vec)) for c in self.clusters]
        cluster_sims.sort(key=lambda x: x[1], reverse=True)

        # 2. 在 top 簇内精排
        candidates = []
        for cluster, _ in cluster_sims[:n_probe_clusters]:
            cluster.update_heat(step)
            for doc in cluster.docs:
                sim = float(np.dot(query_vec, doc.embedding) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc.embedding) + 1e-8
                ))
                candidates.append((doc, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. 更新访问统计
        results = []
        for doc, score in candidates[:top_k]:
            doc.access_count += 1
            doc.last_access_step = step
            results.append((doc, score))

        return results

    def get_all_doc_ids(self) -> Set[str]:
        """返回 KB 中所有文档 ID"""
        return set(self._doc_id_set)

    def get_all_docs(self) -> List[KBDocument]:
        """返回所有文档"""
        docs = []
        for cluster in self.clusters:
            docs.extend(cluster.docs)
        return docs

    @property
    def doc_count(self) -> int:
        return self._doc_count

    # ---- 淘汰 ----

    def _evict(self, step: int, n: int = None):
        """淘汰最冷的簇或其中的文档"""
        if not self.clusters:
            return

        # 找虚拟热度最低的簇
        target = min(
            self.clusters,
            key=lambda c: c.get_virtual_heat(step, self.heat_decay),
        )

        if target.size <= 30:
            # 小簇: 整体删除
            for doc in target.docs:
                self._doc_id_set.discard(doc.doc_id)
            self._doc_count -= target.size
            self.clusters.remove(target)
        else:
            # 大簇: 移除 30% 最冷文档
            to_remove = max(1, int(target.size * 0.3))
            for doc in target.docs[:to_remove]:
                self._doc_id_set.discard(doc.doc_id)
            removed = target.remove_coldest(to_remove)
            self._doc_count -= removed

    # ---- 统计 ----

    def get_statistics(self) -> Dict:
        """获取 KB 统计 (topic 分布等)"""
        topic_dist = {}
        for cluster in self.clusters:
            topic_dist[cluster.cluster_id] = {
                "size": cluster.size,
                "heat": cluster.heat,
                "last_access": cluster.last_access_step,
            }

        return {
            "total_docs": self._doc_count,
            "capacity": self.capacity,
            "num_clusters": len(self.clusters),
            "utilization": self._doc_count / self.capacity if self.capacity > 0 else 0,
            "topic_clusters": topic_dist,
        }
