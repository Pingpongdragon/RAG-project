"""
ComRAG Dynamic Memory: Dual Vector Store + Centroid-Based Memory Mechanism

Paper: ComRAG (ACL 2025 Industry Track)
https://arxiv.org/abs/2506.21098

Core Design:
- V_high: High-quality CQA vector store (score >= gamma)
- V_low:  Low-quality CQA vector store  (score <  gamma)
- Centroid-based clustering for memory management within each store
- Near-duplicate replacement (sim >= delta) keeps only best answer (Algorithm 2)

Hyperparameters (Paper Section 5.4):
- tau:   Cluster similarity threshold (default=0.75)
- delta: Direct reuse / replacement threshold (default=0.9)
- gamma: Quality score boundary (default=0.6)
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class QARecord:
    """One QA record: corresponds to (q, Emb(q), a_hat, s) in V_high / V_low."""
    question: str
    answer: str
    embedding: np.ndarray       # Emb(q), L2-normalized
    score: float                # Scorer(q, a_hat)
    record_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"qa_{int(self.timestamp * 1000)}"


@dataclass
class SearchResult:
    """Single search result from the vector store."""
    record: QARecord
    similarity: float
    cluster_id: int


# ============================================================
# Centroid-Based Cluster Memory (Paper Section 4.2)
# ============================================================

class CentroidClusterStore:
    """
    Centroid-based memory mechanism for a single vector store.

    Paper formulas:
    - centroid: c = (1/|C|) * sum Emb(q_i)
    - Assignment: C = argmax CosSim(Emb(q), c) if sim >= tau
    - New cluster: c = Emb(q) if sim < tau for all centroids
    - Replacement: if sim >= delta and new score > old score, replace
    """

    def __init__(
        self,
        store_name: str = "high",
        tau: float = 0.75,
        delta: float = 0.9,
    ):
        self.store_name = store_name
        self.tau = tau
        self.delta = delta

        self.clusters: Dict[int, List[QARecord]] = defaultdict(list)
        self.centroids: Dict[int, np.ndarray] = {}
        self._next_cluster_id = 0

    # ---- Core Interface ----

    def add(self, record: QARecord) -> Dict[str, Any]:
        """
        Add a QA record (Algorithm 2 core logic).

        Returns:
            Summary dict with action, cluster_id, and optional replaced_record.
        """
        emb = record.embedding

        # Step 1: Find most similar existing record (for replacement check)
        nearest_record, nearest_sim, nearest_cid = self._find_nearest_record(emb)

        # Step 2: Replacement mechanism (sim >= delta)
        if nearest_record is not None and nearest_sim >= self.delta:
            if record.score > nearest_record.score:
                self._remove_record(nearest_cid, nearest_record)
                self._add_to_cluster(nearest_cid, record)
                self._update_centroid(nearest_cid)
                logger.info(
                    f"[{self.store_name}] Replaced in cluster {nearest_cid}: "
                    f"score {nearest_record.score:.3f} -> {record.score:.3f}"
                )
                return {
                    "action": "replaced",
                    "cluster_id": nearest_cid,
                    "replaced_record": nearest_record,
                }
            else:
                logger.debug(
                    f"[{self.store_name}] Skipped (existing score "
                    f"{nearest_record.score:.3f} >= new {record.score:.3f})"
                )
                return {"action": "skipped", "cluster_id": nearest_cid}

        # Step 3: Find nearest centroid
        best_cid, best_sim = self._find_nearest_centroid(emb)

        if best_cid >= 0 and best_sim >= self.tau:
            self._add_to_cluster(best_cid, record)
            self._update_centroid(best_cid)
            return {"action": "added_to_cluster", "cluster_id": best_cid}
        else:
            new_cid = self._create_cluster(record)
            logger.info(
                f"[{self.store_name}] New cluster {new_cid} "
                f"(nearest_sim={best_sim:.3f} < tau={self.tau})"
            )
            return {"action": "new_cluster", "cluster_id": new_cid}

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search for top-k most similar QA records."""
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
        Two-phase search (Paper Algorithm 1):
        1. Top-k centroid retrieval
        2. Record-level search within candidate clusters
        """
        query_emb = self._normalize(query_embedding)

        centroid_sims = []
        for cid, centroid in self.centroids.items():
            sim = self._cosine_sim(query_emb, centroid)
            centroid_sims.append((cid, sim))
        centroid_sims.sort(key=lambda x: x[1], reverse=True)

        candidate_cids = [cid for cid, _ in centroid_sims[:max(n_probe_clusters, top_k)]]

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
        """Get the single most similar record and its similarity score."""
        results = self.search(query_embedding, top_k=1)
        if results:
            return results[0], results[0].similarity
        return None, 0.0

    # ---- Statistics ----

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

    # ---- Internal Methods ----

    def _find_nearest_record(self, emb: np.ndarray) -> Tuple[Optional[QARecord], float, int]:
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
        """Recompute centroid: c = (1/|C|) * sum Emb(q_i)"""
        recs = self.clusters.get(cluster_id, [])
        if not recs:
            return
        embeddings = np.array([r.embedding for r in recs])
        centroid = embeddings.mean(axis=0)
        self.centroids[cluster_id] = self._normalize(centroid)

    @staticmethod
    def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v


# ============================================================
# Dynamic Memory: Dual Vector Store (Paper Section 4.2)
# ============================================================

class DynamicMemory:
    """
    ComRAG dual vector store manager.

    - high_store (V_high): QA pairs with score >= gamma
    - low_store  (V_low):  QA pairs with score <  gamma

    Usage:
        memory = DynamicMemory(tau=0.75, delta=0.9, gamma=0.6)
        memory.add(question, answer, embedding, score)
        route = memory.route_query(query_embedding)
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
        Add a new QA record (Algorithm 2).
        Automatically routes to high_store or low_store based on score vs gamma.
        Internally handles centroid-based replacement logic.
        """
        record = QARecord(
            question=question,
            answer=answer,
            embedding=embedding,
            score=score,
            metadata=metadata or {},
        )

        # Algorithm 2, Line 3: choose target store
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
        Three-tier routing (Algorithm 1, Section 4.3).

        1) sim >= delta       -> direct_reuse        (reuse high-quality answer directly)
        2) tau <= sim < delta -> reference_generation (use high-quality QA as reference)
        3) sim < tau          -> kb_avoidance         (use KB + avoid low-quality answers)

        Returns dict with strategy, similarity, matches, references, and negatives.
        """
        query_emb = CentroidClusterStore._normalize(query_embedding)
        best_high, max_sim_high = self.high_store.get_max_similarity(query_emb)

        # Strategy 1: Direct Reuse
        if best_high is not None and max_sim_high >= self.delta:
            logger.info(
                f"[Route] (1) Direct Reuse (sim={max_sim_high:.3f} >= delta={self.delta})"
            )
            return {
                "strategy": "direct_reuse",
                "max_similarity": max_sim_high,
                "best_match": best_high,
                "high_q_references": [],
                "low_q_negatives": [],
            }

        # Strategy 2: Reference Generation
        if best_high is not None and max_sim_high >= self.tau:
            high_refs = self.high_store.search(query_emb, top_k=5)
            logger.info(
                f"[Route] (2) Reference Generation "
                f"(tau={self.tau} <= sim={max_sim_high:.3f} < delta={self.delta})"
            )
            return {
                "strategy": "reference_generation",
                "max_similarity": max_sim_high,
                "best_match": best_high,
                "high_q_references": high_refs,
                "low_q_negatives": [],
            }

        # Strategy 3: KB + Low-Quality Avoidance
        low_negatives = self.low_store.search(query_emb, top_k=5)
        logger.info(
            f"[Route] (3) KB + Avoidance "
            f"(sim={max_sim_high:.3f} < tau={self.tau}, "
            f"low_q_negatives={len(low_negatives)})"
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
