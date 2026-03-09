"""
QARC Pipeline: Three-Phase Lifecycle — Bootstrap → Explore → Exploit

Part of the QARC (Query-Aligned Retrieval-augmented Knowledge Curation) framework.

Lifecycle:
  Phase 0 (Bootstrap): Before any queries — initialize KB via diversity-max from D_pool
  Phase 1 (Explore):   First N_warmup windows — aggressive re-curation every window,
                        accumulate Gap statistics for Phase 2 threshold initialization
  Phase 2 (Exploit):   Steady-state — adaptive threshold triggers re-curation only
                        when alignment gap exceeds EMA + k·MAD

Phase transitions:
  Phase 0 → 1:  Automatically after bootstrap
  Phase 1 → 2:  After N_warmup windows AND Gap variance converges (σ_recent/σ_all < ε_σ)
  Phase 2 → 1:  Re-explore when ≥ re_explore_trigger consecutive re-curations
                 (indicates drastic interest shift that Phase 2 can't handle)

Integration:
  - Uses AutoKMeans (qarc_interest_model.py) for interest profiling
  - Uses QARCKBCurator (qarc_kb_curator.py) for document selection
  - Optionally calls ERASEUpdater for consistency checks on new documents
  - Works alongside ComRAG's DynamicMemory for QA history routing
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

from updator.qarc.interest_model import (
    QueryWindowBuffer,
    InterestCluster,
    AlignmentGapResult,
    AdaptiveThreshold,
    GMMDriftDetector,
    auto_kmeans,
    compute_alignment_gap,
)
from updator.qarc.kb_curator import (
    QARCKBCurator,
    DocumentPool,
    Document,
    CurationResult,
)

logger = logging.getLogger(__name__)


# ============================================================
# Phase Enum
# ============================================================

class QARCPhase(Enum):
    BOOTSTRAP = "bootstrap"
    EXPLORE = "explore"
    EXPLOIT = "exploit"


# ============================================================
# QARC Pipeline
# ============================================================

class QARCPipeline:
    """
    Main QARC pipeline orchestrator.

    Manages the three-phase lifecycle and coordinates:
    - Query window buffering
    - Interest profiling (AutoKMeans)
    - Alignment gap computation
    - Phase transition logic
    - KB re-curation triggering

    Usage:
        pool = DocumentPool()
        pool.add_documents([...])
        curator = QARCKBCurator(pool, kb_budget=50)

        pipeline = QARCPipeline(curator, embed_fn=my_embed_fn)
        pipeline.bootstrap()  # Phase 0

        for query in query_stream:
            docs, answer = pipeline.process_query(query)
            # Use docs + answer...
    """

    def __init__(
        self,
        curator: QARCKBCurator,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        llm_fn: Optional[Callable[[str, List[str]], str]] = None,
        # Window parameters
        window_size: int = 50,
        # Phase 1 (Explore) parameters
        n_warmup_min: int = 5,
        epsilon_sigma: float = 0.3,
        explore_lambda_max: float = 0.5,
        explore_eta: float = 0.0,
        # Phase 2 (Exploit) parameters
        exploit_lambda_max: float = 0.2,
        exploit_eta: float = 0.1,
        cooldown_windows: int = 3,
        # Adaptive threshold parameters
        threshold_beta: float = 0.9,
        threshold_k: float = 2.0,
        # Re-explore trigger
        re_explore_trigger: int = 3,
        # Retrieval parameters
        retrieve_top_k: int = 5,
        # GMM drift detection (DriftLens-inspired)
        use_gmm_drift: bool = True,
        gmm_n_components_range: tuple = (1, 5),
        gmm_covariance_type: str = "diag",
        gmm_beta: float = 0.85,
        gmm_k_drift: float = 2.5,
    ):
        """
        Args:
            curator:             QARCKBCurator instance
            embed_fn:            Function to embed a text string -> np.ndarray
            llm_fn:              Function(question, context_docs) -> answer string
            window_size:         Queries per window (W_size)
            n_warmup_min:        Minimum windows in Phase 1
            epsilon_sigma:       Convergence threshold for Phase 1 → 2 transition
            explore_lambda_max:  Replacement ratio in Phase 1 (aggressive)
            explore_eta:         Diversity term in Phase 1 (0 = pure interest)
            exploit_lambda_max:  Replacement ratio in Phase 2 (conservative)
            exploit_eta:         Diversity term in Phase 2 (keeps exploration)
            cooldown_windows:    Cooldown after re-curation in Phase 2
            threshold_beta:      EMA smoothing factor
            threshold_k:         MAD multiplier for threshold
            re_explore_trigger:  Consecutive Phase 2 re-curations before re-explore
            retrieve_top_k:      Number of docs to retrieve per query
        """
        self.curator = curator
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn

        # Window
        self.window_size = window_size
        self.buffer = QueryWindowBuffer(window_size=window_size)

        # Phase 1 params
        self.n_warmup_min = n_warmup_min
        self.epsilon_sigma = epsilon_sigma
        self.explore_lambda_max = explore_lambda_max
        self.explore_eta = explore_eta

        # Phase 2 params
        self.exploit_lambda_max = exploit_lambda_max
        self.exploit_eta = exploit_eta
        self.cooldown_windows = cooldown_windows
        self.re_explore_trigger = re_explore_trigger

        # RAG params
        self.retrieve_top_k = retrieve_top_k

        # GMM drift detector
        self.use_gmm_drift = use_gmm_drift
        self.gmm_drift = GMMDriftDetector(
            n_components_range=gmm_n_components_range,
            covariance_type=gmm_covariance_type,
            beta=gmm_beta,
            k_drift=gmm_k_drift,
        ) if use_gmm_drift else None

        # State
        self.phase = QARCPhase.BOOTSTRAP
        self.adaptive_threshold = AdaptiveThreshold(
            beta=threshold_beta, k=threshold_k
        )
        self.gap_history: List[float] = []
        self.window_count = 0
        self.cooldown_remaining = 0
        self.consecutive_triggers = 0  # For re-explore detection

        # Statistics
        self.total_queries = 0
        self.total_recurations = 0
        self.phase_history: List[Tuple[int, str]] = []  # (window_idx, phase_name)

    # -------------------------------------------------------
    # Phase 0: Bootstrap
    # -------------------------------------------------------

    def bootstrap(
        self,
        historical_queries: Optional[List[np.ndarray]] = None,
    ):
        """
        Phase 0: Initialize KB.

        If historical_queries are provided, uses warm bootstrap (interest-weighted).
        Otherwise, uses diversity-max bootstrap.

        Args:
            historical_queries: Optional list of historical query embeddings
        """
        logger.info("=" * 60)
        logger.info("QARC Phase 0: Bootstrap")
        logger.info("=" * 60)

        if historical_queries and len(historical_queries) >= 10:
            # Warm start: cluster historical queries for interest model
            X = np.vstack(historical_queries)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(norms, 1e-10, None)

            centroids, labels, weights = auto_kmeans(X)
            self.curator.bootstrap_from_queries(
                query_embeddings=X,
                centroids=centroids,
                weights=weights,
                eta=0.05,
            )
            logger.info(f"Warm bootstrap with {len(historical_queries)} historical queries")
        else:
            # Cold start: diversity-max
            self.curator.bootstrap_diversity()
            logger.info("Cold bootstrap via diversity-max")

        self.phase = QARCPhase.EXPLORE
        self.phase_history.append((0, "explore"))
        logger.info("Transitioning to Phase 1 (Explore)")

        # GMM reference will be initialized after the first window
        # when we have real interest centroids from AutoKMeans

    # -------------------------------------------------------
    # Main query processing loop
    # -------------------------------------------------------

    def process_query(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process a single query through the QARC pipeline.

        Steps:
        1. Embed query (if not provided)
        2. Retrieve from current KB
        3. Generate answer via LLM (if llm_fn provided)
        4. Add to window buffer
        5. If window full → trigger interest analysis + possible re-curation

        Args:
            query_text:      The query string
            query_embedding: Pre-computed embedding (optional)

        Returns:
            Dict with keys: "documents", "answer", "max_sim", "phase",
                           "window_event" (if window was processed)
        """
        self.total_queries += 1

        # 1. Embed
        if query_embedding is None:
            if self.embed_fn is None:
                raise ValueError("No query_embedding and no embed_fn configured")
            query_embedding = self.embed_fn(query_text)

        # Normalize
        qnorm = np.linalg.norm(query_embedding)
        if qnorm > 1e-10:
            query_embedding = query_embedding / qnorm

        # 2. Retrieve from KB
        retrieved = self.curator.retrieve(query_embedding, top_k=self.retrieve_top_k)

        documents = [doc for doc, sim in retrieved]
        max_sim = retrieved[0][1] if retrieved else 0.0

        # 3. Generate answer
        answer = None
        if self.llm_fn is not None and documents:
            context_texts = [doc.text for doc in documents]
            try:
                answer = self.llm_fn(query_text, context_texts)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                answer = None

        # 4. Add to buffer
        self.buffer.add(
            embedding=query_embedding,
            text=query_text,
            max_sim_to_kb=max_sim,
        )

        result = {
            "documents": documents,
            "answer": answer,
            "max_sim": max_sim,
            "phase": self.phase.value,
            "query_index": self.total_queries,
            "window_event": None,
        }

        # 5. Check if window is full
        if self.buffer.is_full:
            window_result = self._process_window()
            result["window_event"] = window_result

        return result

    # -------------------------------------------------------
    # Window processing — interest analysis + phase logic
    # -------------------------------------------------------

    def _process_window(self) -> Dict[str, Any]:
        """
        Process a full window of queries.

        1. Flush buffer
        2. Cluster queries (AutoKMeans) → interest centroids + weights
        3. Compute Alignment Gap G(t)
        4. Phase-specific logic (Explore or Exploit)

        Returns:
            Dict with window processing details
        """
        self.window_count += 1
        embeddings, texts, sims = self.buffer.flush()

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Window #{self.window_count} | Phase: {self.phase.value} | "
            f"Queries: {len(embeddings)}"
        )
        logger.info(f"{'='*60}")

        # Stack embeddings
        X = np.vstack(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-10, None)

        # 1. AutoKMeans interest profiling
        centroids, labels, weights = auto_kmeans(X)

        # Build InterestCluster objects
        clusters = []
        for i in range(len(centroids)):
            mask = labels == i
            cluster_texts = [texts[j] for j in range(len(texts)) if mask[j]]
            clusters.append(InterestCluster(
                centroid=centroids[i],
                weight=float(weights[i]),
                query_count=int(mask.sum()),
                cluster_id=i,
                representative_queries=cluster_texts[:3],
            ))

        logger.info(f"Interest clusters: {len(clusters)}")
        for c in clusters:
            logger.info(f"  {c}")

        # 2. Compute Alignment Gap
        kb_embs = self.curator.get_kb_embeddings()
        gap_result = compute_alignment_gap(X, kb_embs)
        self.gap_history.append(gap_result.gap)

        logger.info(
            f"Alignment Gap G={gap_result.gap:.4f} "
            f"(avg_max_sim={gap_result.avg_max_sim:.4f})"
        )

        # 2b. GMM drift detection (DriftLens-inspired)
        gmm_result = None
        if self.gmm_drift is not None:
            gmm_result = self.gmm_drift.compute_drift_score(X, centroids)

        # 3. Phase-specific logic
        curation_result = None
        phase_transition = None

        if self.phase == QARCPhase.EXPLORE:
            curation_result, phase_transition = self._explore_logic(
                centroids, weights, gap_result, gmm_result
            )
        elif self.phase == QARCPhase.EXPLOIT:
            curation_result, phase_transition = self._exploit_logic(
                centroids, weights, gap_result, gmm_result
            )

        return {
            "window_index": self.window_count,
            "phase": self.phase.value,
            "gap": gap_result.gap,
            "avg_max_sim": gap_result.avg_max_sim,
            "n_clusters": len(clusters),
            "clusters": [
                {"id": c.cluster_id, "weight": c.weight, "n_queries": c.query_count}
                for c in clusters
            ],
            "curation": curation_result,
            "phase_transition": phase_transition,
            "gmm_drift": gmm_result,
        }

    # -------------------------------------------------------
    # Phase 1: Explore logic
    # -------------------------------------------------------

    def _explore_logic(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        gap_result: AlignmentGapResult,
        gmm_result: Optional[Dict] = None,
    ) -> Tuple[Optional[CurationResult], Optional[str]]:
        """
        Phase 1 (Explore): Always trigger re-curation, check for convergence.

        - Every window → ReCurate with aggressive params
        - Track Gap history for Phase 2 initialization
        - Check if Gap variance has converged → transition to Phase 2
        """
        logger.info("Phase 1 (Explore): Triggering re-curation")

        # Always re-curate in explore phase
        curation = self.curator.recurate(
            centroids=centroids,
            weights=weights,
            lambda_max=self.explore_lambda_max,
            eta=self.explore_eta,
        )
        self.total_recurations += 1

        # Update GMM reference after re-curation (KB changed)
        if self.gmm_drift is not None:
            kb_embs = self.curator.get_kb_embeddings()
            self.gmm_drift.set_reference(kb_embs, centroids)

        # Check Phase 1 → Phase 2 transition
        phase_transition = None
        if self.window_count >= self.n_warmup_min and len(self.gap_history) >= self.n_warmup_min:
            # Check convergence: σ(recent) / σ(all) < ε_σ
            recent_window = min(5, len(self.gap_history))
            recent_std = float(np.std(self.gap_history[-recent_window:]))
            total_std = float(np.std(self.gap_history))

            ratio = recent_std / max(total_std, 1e-8)

            logger.info(
                f"  Convergence check: σ_recent={recent_std:.4f}, "
                f"σ_total={total_std:.4f}, ratio={ratio:.4f}, ε={self.epsilon_sigma}"
            )

            if ratio < self.epsilon_sigma:
                # Transition to Phase 2
                self.phase = QARCPhase.EXPLOIT
                self.adaptive_threshold.initialize_from_history(self.gap_history)
                self.cooldown_remaining = 0
                self.consecutive_triggers = 0
                self.phase_history.append((self.window_count, "exploit"))
                phase_transition = "explore → exploit"

                # Initialize GMM drift EMA from explore history
                if self.gmm_drift is not None:
                    self.gmm_drift.initialize_from_explore_history()

                logger.info(
                    f"*** Phase transition: Explore → Exploit ***\n"
                    f"    Threshold initialized: "
                    f"EMA={self.adaptive_threshold.g_ema:.4f}, "
                    f"MAD={self.adaptive_threshold.g_mad:.4f}"
                )

        return curation, phase_transition

    # -------------------------------------------------------
    # Phase 2: Exploit logic
    # -------------------------------------------------------

    def _exploit_logic(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        gap_result: AlignmentGapResult,
        gmm_result: Optional[Dict] = None,
    ) -> Tuple[Optional[CurationResult], Optional[str]]:
        """
        Phase 2 (Exploit): Adaptive threshold triggers re-curation.

        - Update EMA/MAD with current Gap
        - If G > threshold AND cooldown == 0 → ReCurate
        - If consecutive triggers >= re_explore_trigger → fallback to Phase 1
        """
        # Update threshold and check trigger
        gap_triggered = self.adaptive_threshold.update(gap_result.gap)
        gmm_triggered = (gmm_result or {}).get("triggered", False)

        # Combined signal: trigger if EITHER Gap OR GMM detects drift
        triggered = gap_triggered or gmm_triggered

        trigger_src = []
        if gap_triggered:
            trigger_src.append("Gap")
        if gmm_triggered:
            trigger_src.append(f"GMM(D={gmm_result['drift_score']:.4f})")

        logger.info(
            f"Phase 2 (Exploit): G={gap_result.gap:.4f}, "
            f"threshold={self.adaptive_threshold.threshold:.4f}, "
            f"triggered={triggered} [{'+'.join(trigger_src) if trigger_src else '-'}], "
            f"cooldown={self.cooldown_remaining}"
        )

        curation = None
        phase_transition = None

        if triggered and self.cooldown_remaining <= 0:
            # Trigger re-curation
            logger.info("  Adaptive threshold exceeded → Re-curating")

            curation = self.curator.recurate(
                centroids=centroids,
                weights=weights,
                lambda_max=self.exploit_lambda_max,
                eta=self.exploit_eta,
            )
            self.total_recurations += 1
            self.cooldown_remaining = self.cooldown_windows
            self.consecutive_triggers += 1

            # Update GMM reference after re-curation
            if self.gmm_drift is not None:
                kb_embs = self.curator.get_kb_embeddings()
                self.gmm_drift.set_reference(kb_embs, centroids)

            # Check for re-explore
            if self.consecutive_triggers >= self.re_explore_trigger:
                # Drastic interest shift — fallback to Phase 1
                self.phase = QARCPhase.EXPLORE
                self.consecutive_triggers = 0
                self.phase_history.append((self.window_count, "re-explore"))
                phase_transition = "exploit → re-explore"

                logger.info(
                    f"*** Phase transition: Exploit → Re-Explore ***\n"
                    f"    {self.re_explore_trigger} consecutive triggers detected, "
                    f"    reverting to aggressive re-curation"
                )
        else:
            # No trigger or in cooldown
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
            if not triggered:
                self.consecutive_triggers = 0  # Reset streak

        return curation, phase_transition

    # -------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------

    def force_recurate(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        lambda_max: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> CurationResult:
        """Force a re-curation regardless of phase/threshold (for testing)."""
        lam = lambda_max if lambda_max is not None else self.exploit_lambda_max
        e = eta if eta is not None else self.exploit_eta
        return self.curator.recurate(centroids=centroids, weights=weights,
                                     lambda_max=lam, eta=e)

    def get_current_kb_docs(self) -> List[Document]:
        """Return current KB documents."""
        return self.curator.get_kb_docs_list()

    def get_kb_size(self) -> int:
        return self.curator.kb_size

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive pipeline statistics."""
        return {
            "phase": self.phase.value,
            "total_queries": self.total_queries,
            "total_windows": self.window_count,
            "total_recurations": self.total_recurations,
            "gap_history": list(self.gap_history),
            "phase_history": self.phase_history,
            "threshold_state": self.adaptive_threshold.get_state(),
            "gmm_drift_state": self.gmm_drift.get_state() if self.gmm_drift else None,
            "cooldown_remaining": self.cooldown_remaining,
            "consecutive_triggers": self.consecutive_triggers,
            "curator": self.curator.get_statistics(),
        }

    def get_phase(self) -> str:
        """Return current phase name."""
        return self.phase.value

    def __repr__(self):
        return (
            f"QARCPipeline(phase={self.phase.value}, "
            f"queries={self.total_queries}, windows={self.window_count}, "
            f"kb_size={self.curator.kb_size})"
        )
