
"""
Unified Experiment Evaluation Framework (v2)

Key changes from v1:
  - "domain" -> "topic" (matches WoW's natural topic field)
  - Hard "domain coverage" replaced with Topic Alignment (cosine of distributions)
  - Added Gold-in-KB rate metric (KB curation quality upper bound)
  - No hard domain boundaries in evaluation

Provides:
  1. EmbeddingHelper           — wraps SentenceTransformer / random fallback
  2. Per-window metrics        — Recall@k, Topic Alignment, Gold-in-KB, Avg Sim
  3. Method adapters           — DRIP / StaticKB / RandomKB common interface
  4. run_experiment()          — stream queries, collect per-window metrics
  5. run_comparison()          — run all methods on one dataset, produce tables
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.data_structures import ExperimentDataset, PoolDocument, QueryItem

logger = logging.getLogger(__name__)


# ============================================================
# Embedding Helper
# ============================================================

class EmbeddingHelper:
    """
    Wraps SentenceTransformer for batch embedding.
    Falls back to random vectors when ``use_random=True`` (for quick testing).
    """

    def __init__(self, use_random: bool = False, dim: int = 768, device: str | None = None):
        self.use_random = use_random
        self.dim = dim
        self._model = None
        self._device = device

        if not use_random:
            try:
                from sentence_transformers import SentenceTransformer
                from config.settings import EMBEDDING_MODEL, CACHE_FOLDER
                self._model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    cache_folder=CACHE_FOLDER,
                    trust_remote_code=True,
                    device=device or 'cpu',
                )
                self.dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model on {device or 'cpu'}, dim={self.dim}")
            except Exception as e:
                logger.warning(f"Embedding model unavailable: {e}. Falling back to random.")
                self.use_random = True

    def embed(self, text: str) -> np.ndarray:
        if self.use_random:
            v = np.random.randn(self.dim).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-10)
        return self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.use_random:
            v = np.random.randn(len(texts), self.dim).astype(np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            return v / np.clip(norms, 1e-10, None)
        return self._model.encode(
            texts, convert_to_numpy=True,
            normalize_embeddings=True, batch_size=batch_size,
        )


# ============================================================
# Metrics
# ============================================================

@dataclass
class WindowMetrics:
    """Per-window evaluation metrics."""
    window_idx: int
    query_range: Tuple[int, int]          # (start, end) in stream
    n_queries: int
    # --- Topic distribution in this window (soft, not a single label) ---
    window_topic_dist: Dict[str, float]   # empirical P(topic) in window
    # --- Core metrics ---
    retrieval_recall_at_k: float          # frac queries with gold doc in top-k
    gold_in_kb_rate: float                # frac queries with gold doc in KB
    topic_alignment: float                # cos(KB topic dist, window topic dist)
    avg_retrieval_sim: float              # average top-1 cosine similarity
    kb_size: int
    latency_ms: float
    # --- KB topic distribution snapshot ---
    kb_topic_dist: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Full result for one method on one experiment."""
    method_name: str
    experiment_name: str
    window_metrics: List[WindowMetrics]
    total_queries: int
    total_time_s: float
    query_time_s: float = 0.0    # sum of process_query() call durations
    update_time_s: float = 0.0   # sum of feed_gold_docs() call durations (ERASE)

    # Aggregated
    avg_recall: float = 0.0
    avg_gold_in_kb: float = 0.0
    avg_topic_alignment: float = 0.0
    avg_sim: float = 0.0
    adaptation_speed: float = 0.0

    def compute_aggregates(self):
        if not self.window_metrics:
            return
        self.avg_recall = float(np.mean(
            [w.retrieval_recall_at_k for w in self.window_metrics]))
        self.avg_gold_in_kb = float(np.mean(
            [w.gold_in_kb_rate for w in self.window_metrics]))
        self.avg_topic_alignment = float(np.mean(
            [w.topic_alignment for w in self.window_metrics]))
        self.avg_sim = float(np.mean(
            [w.avg_retrieval_sim for w in self.window_metrics]))
        self.adaptation_speed = self._compute_adaptation_speed()

    def _compute_adaptation_speed(self) -> float:
        """
        After a significant topic mix change (|delta topic_dist| > 0.3),
        how many windows until topic_alignment recovers to 0.8 × pre-shift level?
        """
        if len(self.window_metrics) < 3:
            return 0.0

        alignments = [w.topic_alignment for w in self.window_metrics]
        dists = [w.window_topic_dist for w in self.window_metrics]

        recovery_windows = []
        for i in range(1, len(dists)):
            # Check whether the topic mix changed significantly
            all_topics = set(dists[i].keys()) | set(dists[i - 1].keys())
            delta = sum(abs(dists[i].get(t, 0) - dists[i - 1].get(t, 0))
                        for t in all_topics) / 2.0
            if delta > 0.3:
                pre = alignments[i - 1] if alignments[i - 1] > 0.05 else 0.5
                target = pre * 0.8
                for j in range(i, len(alignments)):
                    if alignments[j] >= target:
                        recovery_windows.append(j - i)
                        break
                else:
                    recovery_windows.append(len(alignments) - i)

        return float(np.mean(recovery_windows)) if recovery_windows else 0.0

    def to_dict(self) -> Dict:
        self.compute_aggregates()
        return {
            "method": self.method_name,
            "experiment": self.experiment_name,
            "total_queries": self.total_queries,
            "total_time_s": round(self.total_time_s, 2),
            "avg_recall@k": round(self.avg_recall, 4),
            "avg_gold_in_kb": round(self.avg_gold_in_kb, 4),
            "avg_topic_alignment": round(self.avg_topic_alignment, 4),
            "avg_retrieval_sim": round(self.avg_sim, 4),
            "adaptation_speed": round(self.adaptation_speed, 2),
            "per_window": [
                {
                    "window": w.window_idx,
                    "topic_dist": {k: round(v, 3) for k, v in w.window_topic_dist.items()},
                    "recall@k": round(w.retrieval_recall_at_k, 4),
                    "gold_in_kb": round(w.gold_in_kb_rate, 4),
                    "topic_align": round(w.topic_alignment, 4),
                    "avg_sim": round(w.avg_retrieval_sim, 4),
                    "kb_size": w.kb_size,
                    "kb_topics": {k: round(v, 3) for k, v in w.kb_topic_dist.items()},
                    "latency_ms": round(w.latency_ms, 1),
                }
                for w in self.window_metrics
            ],
        }


# ============================================================
# Metric helpers
# ============================================================

def _topic_distribution(items_with_topic, topic_key: str = "topic") -> Dict[str, float]:
    """Compute normalised topic distribution from a collection."""
    counts: Counter = Counter()
    for item in items_with_topic:
        if isinstance(item, dict):
            counts[item.get(topic_key, "?")] += 1
        elif hasattr(item, topic_key):
            counts[getattr(item, topic_key)] += 1
        else:
            counts["?"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def topic_alignment_score(
    dist_a: Dict[str, float],
    dist_b: Dict[str, float],
) -> float:
    """
    Cosine similarity between two topic distributions.
    Both treated as sparse vectors over the topic vocabulary.
    Returns value in [0, 1]; higher ⇒ better alignment.
    """
    all_topics = set(dist_a) | set(dist_b)
    if not all_topics:
        return 0.0
    va = np.array([dist_a.get(t, 0.0) for t in all_topics])
    vb = np.array([dist_b.get(t, 0.0) for t in all_topics])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ============================================================
# Method Adapter Protocol
# ============================================================

class MethodAdapter(ABC):
    """所有 KB 更新方法的实验适配器接口。

    每个方法 (DRIP/ComRAG/ERASE/Static/Random) 实现此接口,
    由 ExperimentRunner 统一调度。"""

    name: str = "base"

    @abstractmethod
    def initialize(
        self,
        pool_docs: List[PoolDocument],
        embedder: EmbeddingHelper,
        doc_embeddings: Dict[str, np.ndarray],
        shared_bootstrap: Optional[list] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def process_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
    ) -> Tuple[List[str], Optional[str]]:
        """Return (retrieved doc IDs, answer or None)."""
        raise NotImplementedError

    @abstractmethod
    def get_kb_doc_ids(self) -> set:
        raise NotImplementedError

    @abstractmethod
    def get_kb_topic_distribution(
        self, pool_docs_map: Dict[str, PoolDocument],
    ) -> Dict[str, float]:
        """Topic distribution of current KB."""
        ...

    def on_window_end(self):
        """Hook called at end of each evaluation window."""
        pass


# ============================================================
# DRIP Adapter
# ============================================================

class DRIPAdapter(MethodAdapter):
    name = "DRIP"

    def __init__(self, kb_budget: int = 50, window_size: int = 20, **kw):
        self.kb_budget = kb_budget
        self.window_size = window_size
        self.extra_kw = kw
        self.pipeline = None
        self._doc_embeddings: Dict[str, np.ndarray] = {}

    def initialize(self, pool_docs, embedder, doc_embeddings, shared_bootstrap=None):
        from algorithms.drip.curation.kb_curator import Document, DocumentPool, DRIPKBCurator
        from algorithms.drip.pipeline import DRIPPipeline, DRIPPhase

        self._doc_embeddings = doc_embeddings
        pool = DocumentPool()
        for pd in pool_docs:
            pool.add_document(Document(
                doc_id=pd.doc_id, text=pd.text,
                embedding=doc_embeddings[pd.doc_id],
                metadata={"topic": pd.topic, "title": pd.title},
            ))

        curator = DRIPKBCurator(
            document_pool=pool, kb_budget=self.kb_budget,
            lambda_max=self.extra_kw.get("exploit_lambda_max", 0.2),
            candidate_top_k=self.extra_kw.get("candidate_top_k", 100),
        )
        # Random bootstrap: just load doc_ids into KB
        if shared_bootstrap:
            curator.kb_docs.clear()
            for did in shared_bootstrap:
                if did in curator.pool.documents:
                    curator.kb_docs[did] = curator.pool.documents[did]
            logger.info(f"DRIP: random bootstrap ({len(curator.kb_docs)} docs)")

        from algorithms.drip.config import DRIPConfig
        cfg_overrides = {k: v for k, v in self.extra_kw.items()
                         if k not in ("candidate_top_k", "exploit_lambda_max")}
        cfg_overrides["window_size"] = self.window_size
        cfg = DRIPConfig.from_dict(cfg_overrides)
        self.pipeline = DRIPPipeline(curator=curator, cfg=cfg)
        self.pipeline.phase = DRIPPhase.ONLINE
        logger.info("DRIP: enter ONLINE phase (drift detection active)")

    def process_query(self, query_text, query_embedding):
        r = self.pipeline.process_query(query_text, query_embedding=query_embedding)
        return [d.doc_id for d in r.get("documents", [])], r.get("answer")

    def get_kb_doc_ids(self):
        return {d.doc_id for d in self.pipeline.get_current_kb_docs()} if self.pipeline else set()

    def feed_gold_docs(self, gold_doc_ids: List[str]):
        """Micro-update: inject gold docs into KB, evicting least relevant."""
        if self.pipeline is None:
            return
        curator = self.pipeline.curator
        for did in gold_doc_ids:
            if did in curator.kb_docs:
                continue  # Already in KB
            if did not in curator.pool.documents:
                continue  # Not in pool
            if curator.kb_size >= self.kb_budget:
                # Evict least relevant doc to recent query interest
                kb_embs = curator.get_kb_embeddings()
                kb_ids = list(curator.kb_docs.keys())
                recent = list(self.pipeline._query_history)[-50:]
                if recent:
                    interest = np.mean(np.vstack(recent), axis=0)
                    interest = interest / max(np.linalg.norm(interest), 1e-10)
                    sims = kb_embs @ interest
                    worst_idx = int(np.argmin(sims))
                    del curator.kb_docs[kb_ids[worst_idx]]
                else:
                    del curator.kb_docs[kb_ids[0]]
            curator.kb_docs[did] = curator.pool.documents[did]
        # KB changed: invalidate FAISS index cache
        curator._invalidate_kb_index()

    def get_kb_topic_distribution(self, pool_docs_map):
        counts: Counter = Counter()
        for d in self.pipeline.get_current_kb_docs():
            counts[d.metadata.get("topic", "?")] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}



# ============================================================
# ComRAG Adapter  (QA history accumulation + three-tier routing)
# ============================================================

class ComRAGAdapter(MethodAdapter):
    """
    Adapter for ComRAG: static KB + accumulating QA dual stores + memory-augmented retrieval.

    Adaptation mechanism:
      - KB is FIXED after bootstrap (same as StaticKB)
      - QA history grows as queries arrive, tracking gold doc associations
      - Three-tier routing based on similarity to history:
        1) Direct reuse    (sim >= delta) — return gold docs from matched history
        2) Reference gen   (tau <= sim < delta) — blend history gold docs + KB
        3) KB + avoidance  (sim < tau) — standard KB retrieval

    Gold docs from previous queries are stored and reused when similar queries arrive.
    """

    name = "ComRAG"

    def __init__(self, kb_budget: int = 50, top_k: int = 5,
                 tau: float = 0.75, delta: float = 0.9, gamma: float = 0.6):
        self.kb_budget = kb_budget
        self.top_k = top_k
        self.tau = tau
        self.delta = delta
        self.gamma = gamma
        self._pool_map: Dict[str, PoolDocument] = {}
        # Static KB (same init as StaticKB)
        self.kb_ids: List[str] = []
        self.kb_embs = None
        # ComRAG dual stores
        self.memory = None  # DynamicMemory
        # Gold doc tracking for memory-augmented retrieval
        self._gold_doc_history: List = []   # list of (emb_normed, gold_ids)
        self._last_query_embedding = None

    def initialize(self, pool_docs, embedder, doc_embeddings, shared_bootstrap=None):
        from algorithms.comrag.memory import DynamicMemory

        for pd in pool_docs:
            self._pool_map[pd.doc_id] = pd

        # Use shared random bootstrap (list of doc_ids)
        if shared_bootstrap:
            self.kb_ids = list(shared_bootstrap)
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pool_docs), min(self.kb_budget, len(pool_docs)), replace=False)
            self.kb_ids = [pool_docs[i].doc_id for i in idx]
        self.kb_embs = np.vstack([doc_embeddings[did] for did in self.kb_ids])
        norms = np.linalg.norm(self.kb_embs, axis=1, keepdims=True)
        self.kb_embs = self.kb_embs / np.clip(norms, 1e-10, None)

        # Init ComRAG dual memory (starts empty, grows with queries)
        self.memory = DynamicMemory(
            tau=self.tau, delta=self.delta, gamma=self.gamma)

    def process_query(self, query_text, query_embedding):
        q = query_embedding.reshape(1, -1)
        qnorm = np.linalg.norm(q)
        q_normed = q / qnorm if qnorm > 1e-10 else q

        # 1. Route through ComRAG memory
        route = self.memory.route_query(query_embedding)
        strategy = route["strategy"]

        # 2. Strategy-aware retrieval: use gold docs from matched history
        ret_ids = []

        if strategy == "direct_reuse" and route["best_match"] is not None:
            gold = self._find_nearest_gold_docs(route["best_match"].record.embedding)
            ret_ids.extend(gold)

        elif strategy == "reference_generation" and route.get("high_q_references"):
            seen = set()
            for ref in route["high_q_references"]:
                gold = self._find_nearest_gold_docs(ref.record.embedding)
                for did in gold:
                    if did not in seen:
                        ret_ids.append(did)
                        seen.add(did)

        # 3. Fill remaining slots from KB
        if len(ret_ids) < self.top_k:
            sims = (self.kb_embs @ q_normed.T).flatten()
            k = min(len(sims), self.top_k * 2)
            top_idx = np.argpartition(sims, -k)[-k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            seen = set(ret_ids)
            for i in top_idx:
                did = self.kb_ids[i]
                if did not in seen:
                    ret_ids.append(did)
                    seen.add(did)
                if len(ret_ids) >= self.top_k:
                    break

        ret_ids = ret_ids[:self.top_k]

        # 4. Score & add to QA memory
        sims_all = (self.kb_embs @ q_normed.T).flatten()
        score = float(np.max(sims_all)) if len(sims_all) > 0 else 0.5
        self.memory.add(
            question=query_text,
            answer=query_text,
            embedding=query_embedding,
            score=score,
        )

        self._last_query_embedding = query_embedding.copy()
        return ret_ids, None

    def feed_gold_docs(self, gold_doc_ids: List[str]):
        """Store gold doc associations for memory-augmented retrieval."""
        if self._last_query_embedding is not None:
            emb = self._last_query_embedding.copy()
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            self._gold_doc_history.append((emb, list(gold_doc_ids)))

    def _find_nearest_gold_docs(self, embedding):
        """Find gold doc IDs from the most similar historical query."""
        if not self._gold_doc_history:
            return []
        emb = embedding.reshape(1, -1)
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        stored_embs = np.vstack([e for e, _ in self._gold_doc_history])
        sims = (stored_embs @ emb.T).flatten()
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= 0.7:
            return list(self._gold_doc_history[best_idx][1])
        return []

    def get_kb_doc_ids(self):
        return set(self.kb_ids)

    def get_kb_topic_distribution(self, pool_docs_map):
        counts: Counter = Counter()
        for did in self.kb_ids:
            pd = pool_docs_map.get(did)
            if pd:
                counts[pd.topic] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}


# ============================================================
# ERASE Adapter  (similarity-based update/add KB editing)
# ============================================================

class ERASEAdapter(MethodAdapter):
    """
    Adapter for ERASE: retrieve-update-add KB editing.

    After each query, take the gold document (if available) and:
      - sim > update_threshold → REWRITE the most similar KB entry
      - else → ADD as new entry (evict oldest if over budget)

    No LLM calls — uses similarity heuristic for speed.
    """

    name = "ERASE"

    def __init__(self, kb_budget: int = 50, top_k: int = 5,
                 update_threshold: float = 0.7):
        self.kb_budget = kb_budget
        self.top_k = top_k
        self.update_threshold = update_threshold
        self._pool_map: Dict[str, PoolDocument] = {}
        self.kb = None  # ERASEKnowledgeBase
        self._doc_embeddings: Dict[str, np.ndarray] = {}
        self._query_count = 0
        self._pending_gold_ids: List[str] = []

    def initialize(self, pool_docs, embedder, doc_embeddings, shared_bootstrap=None):
        from algorithms.erase.knowledge_base import ERASEKnowledgeBase

        self._doc_embeddings = doc_embeddings
        self._pool_map = {pd.doc_id: pd for pd in pool_docs}

        # Use shared random bootstrap (list of doc_ids)
        if shared_bootstrap:
            selected_ids = list(shared_bootstrap)
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pool_docs), min(self.kb_budget, len(pool_docs)), replace=False)
            selected_ids = [pool_docs[i].doc_id for i in idx]

        # Initialize ERASE KB with bootstrap docs as initial facts
        self.kb = ERASEKnowledgeBase(similarity_threshold=0.3)
        for did in selected_ids:
            pd = self._pool_map.get(did)
            if pd is None:
                continue
            emb = doc_embeddings[did].copy()
            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            self.kb.add_fact(
                fact=pd.text,
                embedding=emb,
                timestamp="0",
                source=did,
                metadata={"doc_id": did, "topic": pd.topic},
            )

    def process_query(self, query_text, query_embedding):
        self._query_count += 1
        q = query_embedding.copy()
        qnorm = np.linalg.norm(q)
        if qnorm > 1e-10:
            q = q / qnorm

        # Retrieve from ERASE KB
        from algorithms.erase.knowledge_base import RetrievalResult
        results = self.kb.retrieve(q, top_k=self.top_k, threshold=0.0, only_true=True)
        ret_ids = []
        for r in results:
            did = r.entry.metadata.get("doc_id", r.entry.fact_id)
            ret_ids.append(did)

        # Process pending gold docs from previous queries
        # (We process them here so they're available for next query)
        for gold_id in self._pending_gold_ids:
            self._apply_erase_update(gold_id)
        self._pending_gold_ids = []

        return ret_ids, None

    def feed_gold_docs(self, gold_doc_ids: List[str]):
        """Queue gold docs for ERASE processing after the query."""
        self._pending_gold_ids.extend(gold_doc_ids)

    def _apply_erase_update(self, doc_id: str):
        """Apply ERASE-style update/add for a single document."""
        pd = self._pool_map.get(doc_id)
        if pd is None:
            return
        emb = self._doc_embeddings.get(doc_id)
        if emb is None:
            return
        emb_normed = emb / max(np.linalg.norm(emb), 1e-10)

        # Step 1: Retrieve most similar existing fact
        results = self.kb.retrieve(emb_normed, top_k=1, threshold=0.0, only_true=True)

        ts = str(self._query_count)

        if results and results[0].similarity >= self.update_threshold:
            # REWRITE: update existing fact with new doc content
            old_entry = results[0].entry
            self.kb.rewrite_fact(
                fact_id=old_entry.fact_id,
                new_fact=pd.text,
                new_embedding=emb_normed,
                timestamp=ts,
            )
            # Update metadata
            self.kb.entries[old_entry.fact_id].metadata["doc_id"] = doc_id
            self.kb.entries[old_entry.fact_id].metadata["topic"] = pd.topic
        else:
            # ADD: new fact
            # If over budget, evict oldest
            true_facts = self.kb.get_true_facts()
            if len(true_facts) >= self.kb_budget:
                # Evict fact with oldest timestamp
                oldest = min(true_facts,
                             key=lambda f: f.history[-1].timestamp if f.history else "0")
                self.kb.remove_fact(oldest.fact_id)

            self.kb.add_fact(
                fact=pd.text,
                embedding=emb_normed,
                timestamp=ts,
                source=doc_id,
                metadata={"doc_id": doc_id, "topic": pd.topic},
            )

    def get_kb_doc_ids(self):
        ids = set()
        for entry in self.kb.get_true_facts():
            ids.add(entry.metadata.get("doc_id", entry.fact_id))
        return ids

    def get_kb_topic_distribution(self, pool_docs_map):
        counts: Counter = Counter()
        for entry in self.kb.get_true_facts():
            topic = entry.metadata.get("topic", "?")
            counts[topic] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}


# ============================================================
# Static KB Baseline  (Vanilla RAG — no adaptation)
# ============================================================

class StaticKBAdapter(MethodAdapter):
    name = "StaticKB"

    def __init__(self, kb_budget: int = 50, top_k: int = 5):
        self.kb_budget = kb_budget
        self.top_k = top_k
        self.kb_embs = None
        self.kb_ids: List[str] = []
        self._pool_map: Dict[str, PoolDocument] = {}

    def initialize(self, pool_docs, embedder, doc_embeddings, shared_bootstrap=None):
        for pd in pool_docs:
            self._pool_map[pd.doc_id] = pd

        if shared_bootstrap:
            self.kb_ids = list(shared_bootstrap)
        else:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(pool_docs), min(self.kb_budget, len(pool_docs)), replace=False)
            self.kb_ids = [pool_docs[i].doc_id for i in idx]
        self.kb_embs = np.vstack([doc_embeddings[did] for did in self.kb_ids])
        norms = np.linalg.norm(self.kb_embs, axis=1, keepdims=True)
        self.kb_embs = self.kb_embs / np.clip(norms, 1e-10, None)

    def process_query(self, query_text, query_embedding):
        q = query_embedding.reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-10)
        sims = (self.kb_embs @ q.T).flatten()
        k = min(self.top_k, len(sims))
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [self.kb_ids[i] for i in idx], None

    def get_kb_doc_ids(self):
        return set(self.kb_ids)

    def get_kb_topic_distribution(self, pool_docs_map):
        counts: Counter = Counter()
        for did in self.kb_ids:
            pd = pool_docs_map.get(did)
            if pd:
                counts[pd.topic] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}


# ============================================================
# Random KB Baseline  (random refresh every window)
# ============================================================

class RandomKBAdapter(MethodAdapter):
    name = "RandomKB"

    def __init__(self, kb_budget: int = 50, top_k: int = 5, seed: int = 42):
        self.kb_budget = kb_budget
        self.top_k = top_k
        self._rng = np.random.RandomState(seed)
        self.all_ids: List[str] = []
        self.all_embs = None
        self.kb_idx: np.ndarray = np.array([], dtype=int)
        self._pool_map: Dict[str, PoolDocument] = {}
        self._qcount = 0
        self._window_size = 20

    def initialize(self, pool_docs, embedder, doc_embeddings, shared_bootstrap=None):
        self.all_ids = [p.doc_id for p in pool_docs]
        self.all_embs = np.vstack([doc_embeddings[d] for d in self.all_ids])
        norms = np.linalg.norm(self.all_embs, axis=1, keepdims=True)
        self.all_embs = self.all_embs / np.clip(norms, 1e-10, None)
        self._pool_map = {p.doc_id: p for p in pool_docs}
        self.kb_idx = self._rng.choice(
            len(self.all_ids), min(self.kb_budget, len(self.all_ids)), replace=False)

    def process_query(self, query_text, query_embedding):
        self._qcount += 1
        if self._qcount % self._window_size == 0:
            self.kb_idx = self._rng.choice(
                len(self.all_ids), min(self.kb_budget, len(self.all_ids)), replace=False)
        kb_embs = self.all_embs[self.kb_idx]
        q = query_embedding.reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-10)
        sims = (kb_embs @ q.T).flatten()
        k = min(self.top_k, len(sims))
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [self.all_ids[self.kb_idx[i]] for i in idx], None

    def get_kb_doc_ids(self):
        return {self.all_ids[i] for i in self.kb_idx}

    def get_kb_topic_distribution(self, pool_docs_map):
        counts: Counter = Counter()
        for i in self.kb_idx:
            pd = pool_docs_map.get(self.all_ids[i])
            if pd:
                counts[pd.topic] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(
    dataset: ExperimentDataset,
    method: MethodAdapter,
    embedder: EmbeddingHelper,
    eval_window_size: int = 20,
    retrieve_top_k: int = 10,
    precomputed_doc_embs: Optional[Dict[str, np.ndarray]] = None,
    precomputed_query_embs: Optional[np.ndarray] = None,
    shared_bootstrap: Optional[list] = None,
) -> ExperimentResult:
    """
    Stream queries through a method and collect per-window metrics.

    Metrics collected per window:
      - Retrieval Recall@k:  gold doc in top-k retrieved
      - Gold-in-KB rate:     gold doc currently in KB
      - Topic Alignment:     cos(KB topic dist, window topic dist)
      - Avg Retrieval Sim:   mean cosine of top-1 retrieved doc
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running: {method.name} on {dataset.name}")
    logger.info(f"{'=' * 60}")

    # 1. Embeddings (reuse precomputed if available)
    if precomputed_doc_embs is not None:
        doc_embeddings = precomputed_doc_embs
        logger.info(f"Using precomputed doc embeddings ({len(doc_embeddings)} docs)")
    else:
        logger.info("Pre-embedding pool documents...")
        doc_texts = [d.text for d in dataset.document_pool]
        doc_ids = [d.doc_id for d in dataset.document_pool]
        doc_embs_mat = embedder.embed_batch(doc_texts)
        doc_embeddings = {did: doc_embs_mat[i] for i, did in enumerate(doc_ids)}
    pool_map = {d.doc_id: d for d in dataset.document_pool}

    if precomputed_query_embs is not None:
        q_embs = precomputed_query_embs
        logger.info(f"Using precomputed query embeddings ({len(q_embs)} queries)")
    else:
        logger.info("Pre-embedding queries...")
        q_texts = [q.question for q in dataset.query_stream]
        q_embs = embedder.embed_batch(q_texts)

    # 3. Init method
    logger.info(f"Initializing {method.name}...")
    method.initialize(dataset.document_pool, embedder, doc_embeddings, shared_bootstrap=shared_bootstrap)

    # 4. Stream
    windows: List[WindowMetrics] = []
    t0_total = time.time()
    n = len(dataset.query_stream)
    _query_time_s = 0.0
    _update_time_s = 0.0

    for ws in range(0, n, eval_window_size):
        we = min(ws + eval_window_size, n)
        win_queries = dataset.query_stream[ws:we]

        # Topic distribution of this window (soft, not single label)
        win_topic_dist = _topic_distribution(win_queries)

        recalls, gold_hits, top1_sims, latencies = [], [], [], []

        for qi, qitem in enumerate(win_queries):
            qe = q_embs[ws + qi]

            t0 = time.time()
            ret_ids, _ = method.process_query(qitem.question, qe)
            _dt = time.time() - t0
            latencies.append(_dt * 1000)
            _query_time_s += _dt

            # Recall@k
            gold = set(qitem.gold_doc_ids)
            if gold:
                hit = 1.0 if gold & set(ret_ids[:retrieve_top_k]) else 0.0
                recalls.append(hit)

                # Gold-in-KB
                kb_ids = method.get_kb_doc_ids()
                gold_hits.append(1.0 if gold & kb_ids else 0.0)

                # Feed gold docs to ERASE for update/add
                if hasattr(method, 'feed_gold_docs'):
                    _t_upd = time.time()
                    method.feed_gold_docs(list(gold))
                    _update_time_s += time.time() - _t_upd

            # Top-1 sim
            if ret_ids:
                re = doc_embeddings.get(ret_ids[0])
                if re is not None:
                    sim = float(np.dot(
                        qe / max(np.linalg.norm(qe), 1e-10),
                        re / max(np.linalg.norm(re), 1e-10),
                    ))
                    top1_sims.append(sim)

        # KB topic distribution
        kb_topic_dist = method.get_kb_topic_distribution(pool_map)
        ta = topic_alignment_score(win_topic_dist, kb_topic_dist)

        wm = WindowMetrics(
            window_idx=len(windows),
            query_range=(ws, we),
            n_queries=we - ws,
            window_topic_dist=win_topic_dist,
            retrieval_recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
            gold_in_kb_rate=float(np.mean(gold_hits)) if gold_hits else 0.0,
            topic_alignment=ta,
            avg_retrieval_sim=float(np.mean(top1_sims)) if top1_sims else 0.0,
            kb_size=len(method.get_kb_doc_ids()),
            latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            kb_topic_dist=kb_topic_dist,
        )
        windows.append(wm)

        # Pretty log
        top_topic = max(win_topic_dist, key=win_topic_dist.get) if win_topic_dist else "?"
        logger.info(
            f"  Win {wm.window_idx:2d}: "
            f"top_topic={top_topic[:18]:18s} "
            f"recall={wm.retrieval_recall_at_k:.3f}  "
            f"gold_kb={wm.gold_in_kb_rate:.3f}  "
            f"align={wm.topic_alignment:.3f}  "
            f"sim={wm.avg_retrieval_sim:.3f}"
        )
        method.on_window_end()

    result = ExperimentResult(
        method_name=method.name,
        experiment_name=dataset.name,
        window_metrics=windows,
        total_queries=n,
        total_time_s=time.time() - t0_total,
        query_time_s=_query_time_s,
        update_time_s=_update_time_s,
    )
    result.compute_aggregates()
    logger.info(
        f"\n{method.name} Summary: "
        f"recall={result.avg_recall:.4f}  "
        f"gold_kb={result.avg_gold_in_kb:.4f}  "
        f"align={result.avg_topic_alignment:.4f}  "
        f"adapt={result.adaptation_speed:.1f}w"
    )
    return result


def run_comparison(
    dataset: ExperimentDataset,
    methods: List[MethodAdapter],
    embedder: EmbeddingHelper,
    eval_window_size: int = 20,
    output_path: Optional[str] = None,
) -> Dict[str, ExperimentResult]:
    """Run all methods on one dataset and produce comparison table."""
    # Pre-embed once for all methods
    logger.info("Pre-embedding pool documents (shared across methods)...")
    doc_texts = [d.text for d in dataset.document_pool]
    doc_ids = [d.doc_id for d in dataset.document_pool]
    doc_embs_mat = embedder.embed_batch(doc_texts)
    doc_embeddings = {did: doc_embs_mat[i] for i, did in enumerate(doc_ids)}

    logger.info("Pre-embedding queries (shared across methods)...")
    q_texts = [q.question for q in dataset.query_stream]
    q_embs = embedder.embed_batch(q_texts)

    # Shared random bootstrap — same starting KB for all methods (fast)
    _kb_budget = methods[0].kb_budget if hasattr(methods[0], 'kb_budget') else 50
    _rng = np.random.RandomState(42)
    _n = min(_kb_budget, len(dataset.document_pool))
    _indices = _rng.choice(len(dataset.document_pool), _n, replace=False)
    shared_bootstrap_ids = [dataset.document_pool[i].doc_id for i in _indices]
    logger.info(f"Shared random bootstrap: {len(shared_bootstrap_ids)} docs selected")

    results = {}
    for m in methods:
        results[m.name] = run_experiment(
            dataset, m, embedder, eval_window_size,
            precomputed_doc_embs=doc_embeddings,
            precomputed_query_embs=q_embs,
            shared_bootstrap=shared_bootstrap_ids,
        )

    # Table
    print(f"\n{'=' * 85}")
    print(f"  Comparison: {dataset.name}  ({dataset.description})")
    print(f"{'=' * 85}")
    hdr = f"{'Method':<12} {'Recall@k':>10} {'Gold-KB':>10} {'TopicAlign':>12} {'AvgSim':>10} {'AdaptSpd':>10} {'QryTime(s)':>12} {'UpdTime(s)':>12} {'Total(s)':>10}"
    print(hdr)
    print("-" * 109)
    for name, r in results.items():
        print(f"{name:<12} {r.avg_recall:>10.4f} {r.avg_gold_in_kb:>10.4f} "
              f"{r.avg_topic_alignment:>12.4f} {r.avg_sim:>10.4f} "
              f"{r.adaptation_speed:>10.1f} "
              f"{r.query_time_s:>12.2f} {r.update_time_s:>12.2f} "
              f"{r.total_time_s:>10.2f}")
    print()

    if output_path:
        out = {n: r.to_dict() for n, r in results.items()}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    return results
