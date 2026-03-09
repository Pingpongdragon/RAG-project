"""
ComRAG 动态记忆模块: 双向量库 + 基于质心的记忆管理机制

论文: ComRAG — Conversational Retrieval-Augmented Generation (ACL 2025 Industry Track)
链接: https://arxiv.org/abs/2506.21098

=== 论文核心思想 (Section 4.2: Dynamic Memory Mechanism) ===

ComRAG 的核心观察是：在对话式问答 (CQA) 中，用户的问题会重复出现相似模式。
与其每次都走 KB 检索 → LLM 生成 这条昂贵路径，不如维护一个"历史 QA 记忆库"，
让高质量的历史回答可以被直接复用或作为参考。

记忆库分两个向量库:
  - V_high (高质量库): 存储评分 s >= γ 的 QA 对
    → 这些回答质量好，可以被直接复用(策略1)或作为正面参考(策略2)
  - V_low  (低质量库): 存储评分 s <  γ 的 QA 对
    → 这些回答质量差，用作"反面教材"告诉 LLM 避免犯同样的错(策略3)

记忆管理采用基于质心的聚类:
  - 每个簇有一个质心 c = 均值(簇内所有 embedding)
  - 新 QA 到来时:
    1) 如果与某已有记录相似度 >= δ (近重复): 保留评分较高的那个 (论文 Algorithm 2)
    2) 否则如果与某质心相似度 >= τ: 加入该簇, 更新质心
    3) 否则: 创建新簇
  - 这确保了记忆库不会无限膨胀，同时保持答案质量持续提升

超参数 (论文 Section 5.4):
  - τ (tau):   聚类相似度阈值 (默认 0.75) — 决定什么程度的相似算"同一话题"
  - δ (delta): 直接复用/替换阈值 (默认 0.9) — 决定什么程度的相似算"同一问题"
  - γ (gamma): 质量分界线 (默认 0.6) — 区分高质量和低质量回答

=== 与 QARC/ERASE 的关键区别 ===
- ComRAG 的检测是隐式的: 通过 τ/δ 相似度阈值进行路由，不显式检测分布漂移
- ComRAG 不修改 KB: 只维护一个不断增长的 QA 记忆库，KB 内容始终不变
- QARC 会显式检测兴趣漂移 (GMM + AlignmentGap) 并主动重新选择 KB 文档
- ERASE 是文档驱动的: 新信息到来时编辑已有事实，而非路由查询
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class QARecord:
    """一条 QA 记录，对应论文中的 (q, Emb(q), â, s)。

    在论文 Section 4.2 中，每条记录包含:
    - question:  原始问题 q
    - answer:    生成的回答 â = LLM(q, context)
    - embedding: 问题的稠密向量 Emb(q)，经 L2 归一化
    - score:     评分 s = Scorer(q, â)，论文用 BERT-Score F1
    """
    question: str
    answer: str
    embedding: np.ndarray       # Emb(q), L2 归一化
    score: float                # Scorer(q, â) ∈ [0, 1]
    record_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"qa_{int(self.timestamp * 1000)}"


@dataclass
class SearchResult:
    """向量库检索的单条结果。"""
    record: QARecord
    similarity: float
    cluster_id: int


# ============================================================
# 单向量库: 基于质心的聚类记忆 (论文 Section 4.2)
# ============================================================

class CentroidClusterStore:
    """
    单个向量库（V_high 或 V_low）的基于质心的记忆管理。

    === 论文公式对应 ===
    质心计算:     c_k = (1/|C_k|) × Σ_{q_i ∈ C_k} Emb(q_i)
    簇分配:       分配到 argmax_k CosSim(Emb(q), c_k)，前提是 sim >= τ
    新建簇:       当所有质心的 sim < τ 时，以 c_new = Emb(q) 创建新簇
    近重复替换:   当与某已有记录 sim >= δ 时，仅保留评分更高者

    这个机制保证:
    1) 记忆在语义上是有组织的(不是散乱存放)
    2) 同一主题下只保留最优回答(近重复替换)
    3) 记忆量可控(不会为每个问题都存一条)
    """

    def __init__(
        self,
        store_name: str = "high",
        tau: float = 0.75,
        delta: float = 0.9,
    ):
        self.store_name = store_name
        self.tau = tau                          # 聚类阈值
        self.delta = delta                      # 替换阈值

        self.clusters: Dict[int, List[QARecord]] = defaultdict(list)   # 簇ID → 记录列表
        self.centroids: Dict[int, np.ndarray] = {}                     # 簇ID → 质心向量
        self._next_cluster_id = 0

    # ---- 核心接口 ----

    def add(self, record: QARecord) -> Dict[str, Any]:
        """
        添加一条 QA 记录 — 对应论文 Algorithm 2 的核心逻辑。

        决策树:
        1. 在所有已有记录中找最相似的
           → 如果 sim >= δ (近重复):
             - 新记录评分更高 → 替换旧记录 (提升记忆质量)
             - 旧记录评分更高 → 跳过 (已有记录已经够好了)
        2. 否则，在质心中找最相似的
           → 如果 sim >= τ → 加入该簇，更新质心
           → 如果 sim <  τ → 创建新簇 (发现了新话题)
        """
        emb = record.embedding

        # 步骤1: 查找最相似的已有记录 (用于替换检查)
        nearest_record, nearest_sim, nearest_cid = self._find_nearest_record(emb)

        # 步骤2: 近重复替换机制 (sim >= δ)
        if nearest_record is not None and nearest_sim >= self.delta:
            if record.score > nearest_record.score:
                # 论文 Algorithm 2: "replace if new score > old score"
                self._remove_record(nearest_cid, nearest_record)
                self._add_to_cluster(nearest_cid, record)
                self._update_centroid(nearest_cid)
                logger.info(
                    f"[{self.store_name}] 替换 cluster {nearest_cid}: "
                    f"评分 {nearest_record.score:.3f} → {record.score:.3f}"
                )
                return {
                    "action": "replaced",
                    "cluster_id": nearest_cid,
                    "replaced_record": nearest_record,
                }
            else:
                logger.debug(
                    f"[{self.store_name}] 跳过 (已有评分 "
                    f"{nearest_record.score:.3f} >= 新评分 {record.score:.3f})"
                )
                return {"action": "skipped", "cluster_id": nearest_cid}

        # 步骤3: 质心匹配
        best_cid, best_sim = self._find_nearest_centroid(emb)

        if best_cid >= 0 and best_sim >= self.tau:
            # 属于已有话题簇
            self._add_to_cluster(best_cid, record)
            self._update_centroid(best_cid)
            return {"action": "added_to_cluster", "cluster_id": best_cid}
        else:
            # 发现新话题，创建新簇
            new_cid = self._create_cluster(record)
            logger.info(
                f"[{self.store_name}] 新簇 {new_cid} "
                f"(最近质心 sim={best_sim:.3f} < τ={self.tau})"
            )
            return {"action": "new_cluster", "cluster_id": new_cid}

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """暴力搜索 top-k 最相似的 QA 记录（适用于记忆库较小的场景）。"""
        query_emb = self._normalize(query_embedding)
        all_results = []

        for cid, records in self.clusters.items():
            for rec in records:
                sim = self._cosine_sim(query_emb, rec.embedding)
                all_results.append(SearchResult(
                    record=rec, similarity=sim, cluster_id=cid,
                ))

        all_results.sort(key=lambda x: x.similarity, reverse=True)
        return all_results[:top_k]

    def search_centroid_first(
        self, query_embedding: np.ndarray, top_k: int = 5, n_probe_clusters: int = 3
    ) -> List[SearchResult]:
        """
        两阶段检索 — 对应论文 Algorithm 1 的加速检索策略:
        第一阶段: 用质心做粗筛，找到最相关的 n_probe_clusters 个簇
        第二阶段: 在候选簇内做记录级精细搜索

        当记忆库很大时，这比暴力搜索快得多 (避免遍历所有记录)。
        """
        query_emb = self._normalize(query_embedding)

        # 第一阶段: 质心级检索
        centroid_sims = []
        for cid, centroid in self.centroids.items():
            sim = self._cosine_sim(query_emb, centroid)
            centroid_sims.append((cid, sim))
        centroid_sims.sort(key=lambda x: x[1], reverse=True)

        candidate_cids = [cid for cid, _ in centroid_sims[:max(n_probe_clusters, top_k)]]

        # 第二阶段: 候选簇内检索
        all_results = []
        for cid in candidate_cids:
            for rec in self.clusters.get(cid, []):
                sim = self._cosine_sim(query_emb, rec.embedding)
                all_results.append(SearchResult(
                    record=rec, similarity=sim, cluster_id=cid,
                ))

        all_results.sort(key=lambda x: x.similarity, reverse=True)
        return all_results[:top_k]

    def get_max_similarity(self, query_embedding: np.ndarray) -> Tuple[Optional[SearchResult], float]:
        """获取与 query 最相似的单条记录及其相似度 — 用于路由决策。"""
        results = self.search(query_embedding, top_k=1)
        if results:
            return results[0], results[0].similarity
        return None, 0.0

    # ---- 统计信息 ----

    @property
    def total_records(self) -> int:
        return sum(len(recs) for recs in self.clusters.values())

    @property
    def num_clusters(self) -> int:
        return len(self.centroids)

    def get_statistics(self) -> Dict:
        cluster_sizes = {cid: len(recs) for cid, recs in self.clusters.items()}
        avg_scores = {}
        for cid, recs in self.clusters.items():
            if recs:
                avg_scores[cid] = float(np.mean([r.score for r in recs]))
        return {
            "store_name": self.store_name,
            "total_records": self.total_records,
            "num_clusters": self.num_clusters,
            "cluster_sizes": cluster_sizes,
            "cluster_avg_scores": avg_scores,
        }

    # ---- 内部方法 ----

    def _find_nearest_record(self, emb: np.ndarray) -> Tuple[Optional[QARecord], float, int]:
        """在所有已有记录中找最相似的（用于替换检查）。"""
        best_record = None
        best_sim = -1.0
        best_cid = -1
        emb = self._normalize(emb)
        for cid, records in self.clusters.items():
            for rec in records:
                sim = self._cosine_sim(emb, rec.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_record = rec
                    best_cid = cid
        return best_record, best_sim, best_cid

    def _find_nearest_centroid(self, emb: np.ndarray) -> Tuple[int, float]:
        """找最近的簇质心。"""
        best_cid = -1
        best_sim = -1.0
        emb = self._normalize(emb)
        for cid, centroid in self.centroids.items():
            sim = self._cosine_sim(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        return best_cid, best_sim

    def _create_cluster(self, record: QARecord) -> int:
        """创建新簇: c_new = Emb(q)"""
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self.centroids[cid] = self._normalize(record.embedding.copy())
        self.clusters[cid] = [record]
        return cid

    def _add_to_cluster(self, cluster_id: int, record: QARecord):
        self.clusters[cluster_id].append(record)

    def _remove_record(self, cluster_id: int, record: QARecord):
        recs = self.clusters[cluster_id]
        self.clusters[cluster_id] = [r for r in recs if r.record_id != record.record_id]
        if not self.clusters[cluster_id]:
            del self.clusters[cluster_id]
            del self.centroids[cluster_id]

    def _update_centroid(self, cluster_id: int):
        """重新计算质心: c = (1/|C|) × Σ Emb(q_i)"""
        recs = self.clusters.get(cluster_id, [])
        if not recs:
            return
        embeddings = np.array([r.embedding for r in recs])
        centroid = embeddings.mean(axis=0)
        self.centroids[cluster_id] = self._normalize(centroid)

    @staticmethod
    def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        """余弦相似度（输入已 L2 归一化，所以直接点积即可）。"""
        return float(np.dot(v1, v2))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v


# ============================================================
# 动态记忆: 双向量库 (论文 Section 4.2)
# ============================================================

class DynamicMemory:
    """
    ComRAG 双向量库管理器 — 论文的核心数据结构。

    === 论文架构对应 ===
    - high_store (V_high): 高质量 QA 对 (score >= γ)
      用途: 策略1 直接复用 + 策略2 正面参考
    - low_store  (V_low):  低质量 QA 对 (score < γ)
      用途: 策略3 反面教材 (告诉 LLM 这些回答是错的，不要学)

    === 三层路由策略 (论文 Algorithm 1, Section 4.3) ===
    给定新查询 q，计算 Emb(q) 与 V_high 中最相似记录的相似度 sim:
    1) sim >= δ (近重复)       → 直接复用: 返回历史回答，不调 LLM (省 token)
    2) τ <= sim < δ (相关)     → 参考生成: 用高质量 QA 作为 ICL 示例辅助 LLM
    3) sim < τ (不相关)        → KB回退 + 避免低质量: 用 KB 文档 + V_low 中的反面教材

    === 为什么分高低两个库？ ===
    论文的核心 insight: 低质量回答不是垃圾，而是有价值的负面信号。
    通过告诉 LLM "这些回答是错的，请避免类似错误"，可以显著提升回答质量。
    这就是策略3中 "avoidance" 的含义。
    """

    def __init__(self, tau: float = 0.75, delta: float = 0.9, gamma: float = 0.6):
        self.tau = tau
        self.delta = delta
        self.gamma = gamma

        self.high_store = CentroidClusterStore(store_name="V_high", tau=tau, delta=delta)
        self.low_store = CentroidClusterStore(store_name="V_low", tau=tau, delta=delta)

        self._total_added = 0
        self._total_replaced = 0
        self._total_new_clusters = 0

    def add(
        self,
        question: str,
        answer: str,
        embedding: np.ndarray,
        score: float,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        添加一条新 QA 记录 — 对应论文 Algorithm 2 (Update Phase)。

        流程:
        1. 根据评分 s 与阈值 γ 的比较，决定存入 V_high 还是 V_low
        2. 在目标库内，由 CentroidClusterStore.add() 执行:
           - 近重复检测与替换 (sim >= δ)
           - 聚类分配 (sim >= τ) 或新建簇 (sim < τ)
        """
        record = QARecord(
            question=question,
            answer=answer,
            embedding=embedding,
            score=score,
            metadata=metadata or {},
        )

        # Algorithm 2, Line 3: 根据评分选择目标库
        if score >= self.gamma:
            target_store = self.high_store
            store_label = "V_high"
        else:
            target_store = self.low_store
            store_label = "V_low"

        result = target_store.add(record)
        result["target_store"] = store_label
        result["score"] = score

        self._total_added += 1
        if result["action"] == "replaced":
            self._total_replaced += 1
        elif result["action"] == "new_cluster":
            self._total_new_clusters += 1

        return result

    def route_query(self, query_embedding: np.ndarray) -> Dict[str, Any]:
        """
        三层路由决策 — 对应论文 Algorithm 1, Section 4.3。

        输入: 查询 embedding  Emb(q)
        输出: 路由策略 + 相关上下文

        决策逻辑:
        1) 查 V_high 中与 q 最相似的记录，得到 sim
        2) sim >= δ  → direct_reuse        (直接复用高质量回答)
        3) τ <= sim  → reference_generation (用 V_high 中的 QA 作为正面参考)
        4) sim < τ   → kb_avoidance         (用 KB + V_low 中的 QA 作为反面教材)
        """
        query_emb = CentroidClusterStore._normalize(query_embedding)
        best_high, max_sim_high = self.high_store.get_max_similarity(query_emb)

        # 策略1: 直接复用 — 高相似度意味着问过几乎一样的问题，直接返回历史回答
        if best_high is not None and max_sim_high >= self.delta:
            logger.info(
                f"[路由] (1) 直接复用 (sim={max_sim_high:.3f} >= δ={self.delta})"
            )
            return {
                "strategy": "direct_reuse",
                "max_similarity": max_sim_high,
                "best_match": best_high,
                "high_q_references": [],
                "low_q_negatives": [],
            }

        # 策略2: 参考生成 — 有相关但不完全一样的高质量 QA，作为 ICL 示例
        if best_high is not None and max_sim_high >= self.tau:
            high_refs = self.high_store.search(query_emb, top_k=5)
            logger.info(
                f"[路由] (2) 参考生成 "
                f"(τ={self.tau} <= sim={max_sim_high:.3f} < δ={self.delta})"
            )
            return {
                "strategy": "reference_generation",
                "max_similarity": max_sim_high,
                "best_match": best_high,
                "high_q_references": high_refs,
                "low_q_negatives": [],
            }

        # 策略3: KB回退 + 避免低质量 — 没有好的历史参考，回退到 KB 检索
        low_negatives = self.low_store.search(query_emb, top_k=5)
        logger.info(
            f"[路由] (3) KB回退+避免 "
            f"(sim={max_sim_high:.3f} < τ={self.tau}, "
            f"反面教材数={len(low_negatives)})"
        )
        return {
            "strategy": "kb_avoidance",
            "max_similarity": max_sim_high,
            "best_match": None,
            "high_q_references": [],
            "low_q_negatives": low_negatives,
        }

    def get_statistics(self) -> Dict:
        return {
            "total_added": self._total_added,
            "total_replaced": self._total_replaced,
            "total_new_clusters": self._total_new_clusters,
            "high_store": self.high_store.get_statistics(),
            "low_store": self.low_store.get_statistics(),
            "thresholds": {"tau": self.tau, "delta": self.delta, "gamma": self.gamma},
        }
