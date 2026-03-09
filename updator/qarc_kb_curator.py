"""
QARC KB Curator: Submodular Document Selection + Incremental KB Replacement

Part of the QARC (Query-Aligned Retrieval-augmented Knowledge Curation) framework.

Responsibilities:
1. Phase 0 Bootstrap:  Diversity-max initialization of KB from document pool
2. ReCurate():         Interest-weighted submodular selection + incremental replacement
3. ERASE consistency:  Run ERASE-style fact checking on newly added documents

Key Concepts:
- Submodular objective:  f(S) = Σ α_i · max_{d∈S} sim(c_i, d) + η · Facility Location diversity
- Greedy optimization:   O(|candidates| · budget) — guaranteed (1 - 1/e) approximation
- Incremental replacement: bounded by λ_max to prevent catastrophic KB disruption

Differences from static KB:
- ERASE: document-driven push (documents arrive → update facts)
- ComRAG: query-driven but KB-frozen (routes queries, never changes KB)
- QARC: query-driven pull (interest shifts → re-select documents from pool)
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Set, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Document:
    """A document in the pool or KB."""
    doc_id: str
    text: str
    embedding: np.ndarray        # L2-normalized dense vector
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Document[{self.doc_id}]: {self.text[:60]}..."

    def __hash__(self):
        return hash(self.doc_id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.doc_id == other.doc_id
        return False


@dataclass
class CurationResult:
    """Result of a ReCurate operation."""
    added_ids: List[str]         # Document IDs added to KB
    removed_ids: List[str]       # Document IDs removed from KB
    kb_size: int                 # KB size after curation
    objective_before: float      # Submodular objective before
    objective_after: float       # Submodular objective after
    replacement_ratio: float     # Actual |added| / |KB| ratio


# ============================================================
# Document Pool (FAISS-compatible in-memory store)
# ============================================================

class DocumentPool:
    """
    In-memory document pool with dense retrieval.

    In production, this would wrap a FAISS index.
    For clarity, we use brute-force cosine similarity here.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._id_list: List[str] = []
        self._dirty = True

    def add_document(self, doc: Document):
        """Add a document to the pool."""
        self.documents[doc.doc_id] = doc
        self._dirty = True

    def add_documents(self, docs: List[Document]):
        """Batch add documents."""
        for doc in docs:
            self.documents[doc.doc_id] = doc
        self._dirty = True

    def _rebuild_index(self):
        """Rebuild the embedding matrix for fast retrieval."""
        if not self._dirty:
            return
        self._id_list = list(self.documents.keys())
        if self._id_list:
            embeddings = [self.documents[did].embedding for did in self._id_list]
            self._embedding_matrix = np.vstack(embeddings)
            # Normalize
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            self._embedding_matrix = self._embedding_matrix / np.clip(norms, 1e-10, None)
        else:
            self._embedding_matrix = None
        self._dirty = False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents similar to query.

        Args:
            query_embedding: (d,) normalized embedding
            top_k:          Number of results
            exclude_ids:    Document IDs to exclude

        Returns:
            List of (Document, similarity) sorted descending
        """
        self._rebuild_index()

        if self._embedding_matrix is None:
            return []

        query = query_embedding.reshape(1, -1)
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query = query / qnorm

        sims = (self._embedding_matrix @ query.T).flatten()  # (n_docs,)

        # Apply exclusion mask
        if exclude_ids:
            for i, did in enumerate(self._id_list):
                if did in exclude_ids:
                    sims[i] = -2.0

        # Top-k
        if len(sims) <= top_k:
            top_idx = np.argsort(sims)[::-1]
        else:
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        results = []
        for i in top_idx:
            if sims[i] > -1.0:
                did = self._id_list[i]
                results.append((self.documents[did], float(sims[i])))

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """Return (n, d) embedding matrix for all documents."""
        self._rebuild_index()
        if self._embedding_matrix is None:
            return np.empty((0, 0))
        return self._embedding_matrix.copy()

    def get_all_ids(self) -> List[str]:
        """Return list of all doc IDs in insertion order."""
        self._rebuild_index()
        return self._id_list.copy()

    @property
    def size(self) -> int:
        return len(self.documents)


# ============================================================
# Submodular Objective Functions
# ============================================================

def _interest_coverage(
    selected_embs: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute interest coverage objective:
    f_interest(S) = Σ_{i=1}^{m} α_i · max_{d∈S} sim(c_i, d)

    Args:
        selected_embs: (|S|, d) embeddings of selected documents
        centroids:     (m, d) interest centroids
        weights:       (m,) interest weights α_i

    Returns:
        Scalar objective value
    """
    if selected_embs.shape[0] == 0 or centroids.shape[0] == 0:
        return 0.0

    # (m, |S|)
    sim_matrix = centroids @ selected_embs.T

    # max sim for each centroid
    max_sims = sim_matrix.max(axis=1)  # (m,)

    return float((weights * max_sims).sum())


def _diversity_coverage(
    selected_embs: np.ndarray,
    pool_embs: np.ndarray,
) -> float:
    """
    Facility Location diversity objective:
    f_div(S) = Σ_{d∈D_pool} max_{d'∈S} sim(d, d')

    This is a submodular function that encourages coverage of the entire pool.

    Args:
        selected_embs: (|S|, d) embeddings of selected documents
        pool_embs:     (|D_pool|, d) embeddings of all pool documents

    Returns:
        Scalar objective (normalized by |D_pool|)
    """
    if selected_embs.shape[0] == 0 or pool_embs.shape[0] == 0:
        return 0.0

    # (|D_pool|, |S|)
    sim_matrix = pool_embs @ selected_embs.T
    max_sims = sim_matrix.max(axis=1)  # (|D_pool|,)

    return float(max_sims.mean())


def submodular_objective(
    selected_embs: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
    pool_embs: np.ndarray,
    eta: float = 0.1,
) -> float:
    """
    Combined submodular objective:
    f(S) = f_interest(S) + η · f_diversity(S)

    Args:
        selected_embs: (|S|, d) selected document embeddings
        centroids:     (m, d) interest cluster centroids
        weights:       (m,) interest weights
        pool_embs:     (|D_pool|, d) all pool embeddings
        eta:           Diversity regularization coefficient

    Returns:
        Combined objective value
    """
    f_int = _interest_coverage(selected_embs, centroids, weights)
    f_div = _diversity_coverage(selected_embs, pool_embs) if eta > 0 else 0.0
    return f_int + eta * f_div


# ============================================================
# Greedy Submodular Maximization
# ============================================================

def greedy_submodular_select(
    candidate_docs: List[Document],
    centroids: np.ndarray,
    weights: np.ndarray,
    pool_embs: np.ndarray,
    budget: int,
    eta: float = 0.1,
) -> List[Document]:
    """
    Greedy submodular maximization for document selection.

    Standard greedy: at each step, add the document with highest marginal gain.
    Guarantees (1 - 1/e) approximation for monotone submodular functions.

    Complexity: O(budget × |candidates|) — each step recomputes marginal gains.

    Args:
        candidate_docs: List of candidate documents to select from
        centroids:      (m, d) interest cluster centroids
        weights:        (m,) interest weights
        pool_embs:      (|D_pool|, d) all pool embeddings (for diversity term)
        budget:         Maximum number of documents to select
        eta:            Diversity regularization coefficient

    Returns:
        Selected documents (up to budget)
    """
    if not candidate_docs:
        return []

    budget = min(budget, len(candidate_docs))

    # Build candidate embedding matrix
    cand_embs = np.vstack([d.embedding for d in candidate_docs])
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    cand_embs = cand_embs / np.clip(cand_norms, 1e-10, None)

    selected_indices: List[int] = []
    remaining = set(range(len(candidate_docs)))

    for step in range(budget):
        best_gain = -np.inf
        best_idx = -1

        # Current selected embeddings
        if selected_indices:
            sel_embs = cand_embs[selected_indices]
        else:
            sel_embs = np.empty((0, cand_embs.shape[1]))

        current_val = submodular_objective(sel_embs, centroids, weights, pool_embs, eta)

        for idx in remaining:
            # Marginal gain of adding this candidate
            new_sel_embs = np.vstack([sel_embs, cand_embs[idx:idx+1]]) if sel_embs.shape[0] > 0 else cand_embs[idx:idx+1]
            new_val = submodular_objective(new_sel_embs, centroids, weights, pool_embs, eta)
            gain = new_val - current_val

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx < 0 or best_gain <= 0:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

        if step < 3 or step == budget - 1:
            logger.debug(
                f"  Greedy step {step+1}/{budget}: "
                f"added doc={candidate_docs[best_idx].doc_id}, "
                f"marginal_gain={best_gain:.4f}"
            )

    return [candidate_docs[i] for i in selected_indices]


# ============================================================
# KB Curator
# ============================================================

class QARCKBCurator:
    """
    QARC Knowledge Base Curator.

    Manages a dynamic KB backed by a document pool, with:
    - Phase 0: Diversity-max bootstrap (before any queries)
    - ReCurate: Interest-weighted submodular selection + incremental replacement
    - ERASE consistency check (optional, via callback)
    """

    def __init__(
        self,
        document_pool: DocumentPool,
        kb_budget: int = 50,
        lambda_max: float = 0.2,
        candidate_top_k: int = 100,
        erase_check_fn: Optional[Callable] = None,
    ):
        """
        Args:
            document_pool:   The full document pool D_pool
            kb_budget:       Maximum KB size (B)
            lambda_max:      Maximum replacement ratio per re-curation (default 0.2)
            candidate_top_k: Number of candidate docs to retrieve per centroid
            erase_check_fn:  Optional callback for ERASE-style consistency check
                             Signature: fn(doc: Document, kb_docs: List[Document]) -> None
        """
        self.pool = document_pool
        self.kb_budget = kb_budget
        self.lambda_max = lambda_max
        self.candidate_top_k = candidate_top_k
        self.erase_check_fn = erase_check_fn

        # Current KB state
        self.kb_docs: Dict[str, Document] = {}  # doc_id -> Document

        # Statistics
        self.recuration_count = 0

    @property
    def kb_size(self) -> int:
        return len(self.kb_docs)

    def get_kb_embeddings(self) -> np.ndarray:
        """Return (n, d) matrix of current KB document embeddings."""
        if not self.kb_docs:
            return np.empty((0, 0))
        embs = [doc.embedding for doc in self.kb_docs.values()]
        mat = np.vstack(embs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-10, None)

    def get_kb_doc_ids(self) -> Set[str]:
        return set(self.kb_docs.keys())

    def get_kb_docs_list(self) -> List[Document]:
        return list(self.kb_docs.values())

    # -------------------------------------------------------
    # Phase 0: Bootstrap — Diversity-max initialization
    # -------------------------------------------------------

    def bootstrap_diversity(self) -> List[Document]:
        """
        Phase 0: Initialize KB with maximum diversity from pool.

        Uses Facility Location submodular function:
        f_div(S) = Σ_{d∈D_pool} max_{d'∈S} sim(d, d')

        This covers as many pool topics as possible before any user queries arrive.

        Returns:
            List of selected documents
        """
        logger.info(
            f"Phase 0 Bootstrap: selecting {self.kb_budget} docs from pool "
            f"(pool size={self.pool.size}) via diversity-max"
        )

        if self.pool.size == 0:
            logger.warning("Empty document pool — cannot bootstrap KB")
            return []

        pool_embs = self.pool.get_all_embeddings()
        pool_ids = self.pool.get_all_ids()

        # Collect all pool docs as candidates
        all_docs = [self.pool.documents[did] for did in pool_ids]

        # For pure diversity selection, we use uniform interest (dummy centroids)
        # by setting eta=1.0 and the interest term to zero
        # Equivalently, use the facility location as the ONLY objective
        budget = min(self.kb_budget, len(all_docs))

        # Greedy facility location: iteratively pick doc with max marginal coverage
        selected = self._greedy_facility_location(all_docs, pool_embs, budget)

        # Set as current KB
        self.kb_docs.clear()
        for doc in selected:
            self.kb_docs[doc.doc_id] = doc

        logger.info(f"Phase 0 Bootstrap complete: KB size={self.kb_size}")
        return selected

    def bootstrap_from_queries(
        self,
        query_embeddings: np.ndarray,
        centroids: np.ndarray,
        weights: np.ndarray,
        eta: float = 0.05,
    ) -> List[Document]:
        """
        Bootstrap KB using historical query logs (warm start variant).

        Instead of pure diversity, uses pre-computed interest model to select
        documents that already align with known query patterns.

        Args:
            query_embeddings: Historical query embeddings
            centroids:        Interest centroids from historical queries
            weights:          Interest weights
            eta:              Small diversity term

        Returns:
            Selected documents
        """
        logger.info(
            f"Warm Bootstrap: selecting {self.kb_budget} docs using "
            f"{len(centroids)} historical interest clusters"
        )

        candidates = self._gather_candidates(centroids)
        pool_embs = self.pool.get_all_embeddings()

        selected = greedy_submodular_select(
            candidate_docs=candidates,
            centroids=centroids,
            weights=weights,
            pool_embs=pool_embs,
            budget=self.kb_budget,
            eta=eta,
        )

        self.kb_docs.clear()
        for doc in selected:
            self.kb_docs[doc.doc_id] = doc

        logger.info(f"Warm Bootstrap complete: KB size={self.kb_size}")
        return selected

    # -------------------------------------------------------
    # ReCurate: Interest-weighted submodular selection + replacement
    # -------------------------------------------------------

    def recurate(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        lambda_max: Optional[float] = None,
        eta: float = 0.1,
    ) -> CurationResult:
        """
        Re-curate KB based on current interest profile.

        Steps:
        A. Candidate retrieval: each centroid probes pool for similar docs
        B. Submodular selection: greedy maximize f(S) over candidates
        C. Incremental replacement: bounded diff between K_old and K_ideal
        D. ERASE consistency check on newly added docs (if callback provided)

        Args:
            centroids: (m, d) interest cluster centroids from AutoKMeans
            weights:   (m,) interest weights
            lambda_max: Override replacement cap (default uses self.lambda_max)
            eta:       Diversity regularization coefficient

        Returns:
            CurationResult with details of changes
        """
        self.recuration_count += 1
        lam = lambda_max if lambda_max is not None else self.lambda_max

        logger.info(
            f"ReCurate #{self.recuration_count}: "
            f"{len(centroids)} interest clusters, λ_max={lam:.2f}, η={eta:.2f}"
        )

        # A. Candidate retrieval
        candidates = self._gather_candidates(centroids)
        logger.info(f"  Gathered {len(candidates)} unique candidate docs")

        if not candidates:
            return CurationResult(
                added_ids=[], removed_ids=[],
                kb_size=self.kb_size,
                objective_before=0.0, objective_after=0.0,
                replacement_ratio=0.0,
            )

        # B. Submodular selection
        pool_embs = self.pool.get_all_embeddings()
        budget = max(self.kb_budget, self.kb_size)

        k_ideal = greedy_submodular_select(
            candidate_docs=candidates,
            centroids=centroids,
            weights=weights,
            pool_embs=pool_embs,
            budget=budget,
            eta=eta,
        )

        ideal_ids = {d.doc_id for d in k_ideal}
        current_ids = self.get_kb_doc_ids()

        # C. Incremental replacement (bounded by lambda_max)
        to_add_ids = ideal_ids - current_ids
        to_remove_ids = current_ids - ideal_ids

        max_changes = max(1, int(lam * self.kb_size)) if self.kb_size > 0 else len(to_add_ids)

        # If changes exceed budget, select the most impactful
        if len(to_add_ids) > max_changes:
            # Rank additions by marginal gain
            add_candidates = [d for d in k_ideal if d.doc_id in to_add_ids]
            add_candidates = self._rank_by_marginal_gain(
                add_candidates, centroids, weights, pool_embs, eta
            )
            to_add_ids = {d.doc_id for d in add_candidates[:max_changes]}

        if len(to_remove_ids) > max_changes:
            # Remove those least aligned with current interests
            remove_candidates = [self.kb_docs[did] for did in to_remove_ids]
            remove_candidates = self._rank_by_interest_score(
                remove_candidates, centroids, weights
            )
            # Remove the LEAST relevant
            to_remove_ids = {d.doc_id for d in remove_candidates[:max_changes]}

        # Balance: don't remove more than we add (keep KB size stable)
        n_add = len(to_add_ids)
        n_remove = min(len(to_remove_ids), n_add + max(0, self.kb_size - self.kb_budget))

        # Trim removal set
        if n_remove < len(to_remove_ids):
            remove_candidates = [self.kb_docs[did] for did in to_remove_ids]
            remove_candidates = self._rank_by_interest_score(
                remove_candidates, centroids, weights
            )
            to_remove_ids = {d.doc_id for d in remove_candidates[:n_remove]}

        # Compute objective before
        obj_before = submodular_objective(
            self.get_kb_embeddings(), centroids, weights, pool_embs, eta
        ) if self.kb_size > 0 else 0.0

        # D. Apply changes
        for did in to_remove_ids:
            del self.kb_docs[did]

        for doc in k_ideal:
            if doc.doc_id in to_add_ids:
                self.kb_docs[doc.doc_id] = doc

                # ERASE consistency check on new docs
                if self.erase_check_fn is not None:
                    try:
                        self.erase_check_fn(doc, self.get_kb_docs_list())
                    except Exception as e:
                        logger.warning(f"ERASE check failed for {doc.doc_id}: {e}")

        # Compute objective after
        obj_after = submodular_objective(
            self.get_kb_embeddings(), centroids, weights, pool_embs, eta
        ) if self.kb_size > 0 else 0.0

        replacement_ratio = len(to_add_ids) / max(self.kb_size, 1)

        result = CurationResult(
            added_ids=list(to_add_ids),
            removed_ids=list(to_remove_ids),
            kb_size=self.kb_size,
            objective_before=obj_before,
            objective_after=obj_after,
            replacement_ratio=replacement_ratio,
        )

        logger.info(
            f"  ReCurate done: +{len(to_add_ids)} / -{len(to_remove_ids)} docs, "
            f"KB size={self.kb_size}, "
            f"obj {obj_before:.4f} → {obj_after:.4f}"
        )

        return result

    # -------------------------------------------------------
    # Retrieval from KB (for RAG)
    # -------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents from current KB for a query.

        Args:
            query_embedding: (d,) normalized query embedding
            top_k:          Number of docs to return

        Returns:
            List of (Document, similarity) sorted descending
        """
        if not self.kb_docs:
            return []

        kb_embs = self.get_kb_embeddings()
        kb_ids = list(self.kb_docs.keys())

        query = query_embedding.reshape(1, -1)
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query = query / qnorm

        sims = (kb_embs @ query.T).flatten()

        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        results = []
        for i in top_idx:
            did = kb_ids[i]
            results.append((self.kb_docs[did], float(sims[i])))

        return results

    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------

    def _gather_candidates(
        self,
        centroids: np.ndarray,
    ) -> List[Document]:
        """Gather candidate documents by probing pool with each interest centroid."""
        seen_ids: Set[str] = set()
        candidates: List[Document] = []

        for i, centroid in enumerate(centroids):
            results = self.pool.search(
                query_embedding=centroid,
                top_k=self.candidate_top_k,
                exclude_ids=None,  # Include everything, even currently in KB
            )
            for doc, sim in results:
                if doc.doc_id not in seen_ids:
                    seen_ids.add(doc.doc_id)
                    candidates.append(doc)

        return candidates

    def _greedy_facility_location(
        self,
        docs: List[Document],
        pool_embs: np.ndarray,
        budget: int,
    ) -> List[Document]:
        """
        Greedy facility location for diversity-max selection (Phase 0).

        f_div(S) = Σ_{d∈D_pool} max_{d'∈S} sim(d, d')
        """
        n_pool = pool_embs.shape[0]
        doc_embs = np.vstack([d.embedding for d in docs])
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs = doc_embs / np.clip(doc_norms, 1e-10, None)

        # Precompute similarities: (n_pool, n_docs)
        all_sims = pool_embs @ doc_embs.T

        # Track current max coverage per pool doc
        current_max = np.full(n_pool, -np.inf)

        selected_indices = []
        remaining = set(range(len(docs)))

        for step in range(budget):
            best_gain = -np.inf
            best_idx = -1

            for idx in remaining:
                # Marginal gain: amount by which this doc increases coverage
                # For each pool doc: max(0, sim(pool_doc, candidate) - current_max[pool_doc])
                gains = np.maximum(0, all_sims[:, idx] - current_max)
                total_gain = gains.sum()

                if total_gain > best_gain:
                    best_gain = total_gain
                    best_idx = idx

            if best_idx < 0 or best_gain <= 0:
                break

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

            # Update current max coverage
            current_max = np.maximum(current_max, all_sims[:, best_idx])

            if step < 3 or step == budget - 1:
                logger.debug(
                    f"  FacilityLoc step {step+1}/{budget}: "
                    f"doc={docs[best_idx].doc_id}, gain={best_gain:.4f}"
                )

        return [docs[i] for i in selected_indices]

    def _rank_by_marginal_gain(
        self,
        docs: List[Document],
        centroids: np.ndarray,
        weights: np.ndarray,
        pool_embs: np.ndarray,
        eta: float,
    ) -> List[Document]:
        """Rank documents by their marginal gain to the interest objective (descending)."""
        if not docs:
            return []

        current_embs = self.get_kb_embeddings()
        base_val = submodular_objective(current_embs, centroids, weights, pool_embs, eta)

        gains = []
        for doc in docs:
            doc_emb = doc.embedding.reshape(1, -1)
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm > 1e-10:
                doc_emb = doc_emb / doc_norm
            augmented = np.vstack([current_embs, doc_emb]) if current_embs.shape[0] > 0 else doc_emb
            new_val = submodular_objective(augmented, centroids, weights, pool_embs, eta)
            gains.append((doc, new_val - base_val))

        gains.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in gains]

    def _rank_by_interest_score(
        self,
        docs: List[Document],
        centroids: np.ndarray,
        weights: np.ndarray,
    ) -> List[Document]:
        """
        Rank documents by interest relevance score (ascending = least relevant first).

        score(d) = Σ α_i · sim(c_i, d)  — how well this doc serves current interests
        """
        if not docs:
            return []

        scores = []
        for doc in docs:
            emb = doc.embedding.reshape(1, -1)
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 1e-10:
                emb = emb / emb_norm
            sims = (centroids @ emb.T).flatten()  # (m,)
            score = float((weights * sims).sum())
            scores.append((doc, score))

        # Ascending: least relevant first (for removal)
        scores.sort(key=lambda x: x[1])
        return [d for d, _ in scores]

    def get_statistics(self) -> Dict[str, Any]:
        """Return curator statistics."""
        return {
            "kb_size": self.kb_size,
            "kb_budget": self.kb_budget,
            "pool_size": self.pool.size,
            "recuration_count": self.recuration_count,
            "lambda_max": self.lambda_max,
            "kb_doc_ids": list(self.kb_docs.keys()),
        }
