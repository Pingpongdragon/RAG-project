"""
ERASE Knowledge Base: Editable External Knowledge Store

Paper: "Language Modeling with Editable External Knowledge" (Li et al., 2024)
https://arxiv.org/abs/2406.11830

Each entry in the knowledge base consists of:
  - fact f_j:    a natural-language atomic fact string
  - history H_j: [(timestamp, truth_value), ...] recording when the fact was
                  known to be true or false

The KB supports:
  - Dense vector retrieval (cosine similarity)
  - Fact-level CRUD: add, reinforce, make_false, rewrite, remove
  - History tracking for temporal reasoning at inference time
"""

import numpy as np
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class FactHistory:
    """A single history entry recording truth status at a point in time."""
    timestamp: str
    truth_value: bool  # True = fact is true, False = fact is false

    def __repr__(self):
        status = "true" if self.truth_value else "false"
        return f"({status} at {self.timestamp})"


@dataclass
class FactEntry:
    """
    A single entry in the ERASE knowledge base.

    Attributes:
        fact:      The atomic fact as natural language string
        embedding: Dense vector representation of the fact
        history:   List of (timestamp, truth_value) records
        fact_id:   Unique identifier
        source:    Source document that introduced/last modified this fact
        metadata:  Additional metadata
    """
    fact: str
    embedding: np.ndarray
    history: List[FactHistory] = field(default_factory=list)
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_currently_true(self) -> bool:
        """Check if the fact is currently considered true (latest history entry)."""
        if not self.history:
            return True  # Default: assume true if no history
        return self.history[-1].truth_value

    @property
    def latest_timestamp(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[-1].timestamp

    def reinforce(self, timestamp: str):
        """Mark fact as still true at the given timestamp."""
        self.history.append(FactHistory(timestamp=timestamp, truth_value=True))

    def make_false(self, timestamp: str):
        """Mark fact as false at the given timestamp."""
        self.history.append(FactHistory(timestamp=timestamp, truth_value=False))

    def format_history_string(self) -> str:
        """Format history for LLM context (paper Appendix A.3)."""
        if not self.history:
            return "(no history)"
        parts = [str(h) for h in self.history]
        return ", ".join(parts)

    def __repr__(self):
        status = "TRUE" if self.is_currently_true else "FALSE"
        return f"FactEntry[{self.fact_id}|{status}]: {self.fact[:60]}..."


@dataclass
class RetrievalResult:
    """Result of retrieving a fact from the knowledge base."""
    entry: FactEntry
    similarity: float


# ============================================================
# ERASE Knowledge Base
# ============================================================

class ERASEKnowledgeBase:
    """
    Editable knowledge base with dense retrieval and fact history tracking.

    Core operations:
    - add_fact():     Add a new fact with initial history
    - retrieve():     Dense vector retrieval of top-k similar facts
    - reinforce():    Mark a fact as still true
    - make_false():   Mark a fact as false
    - rewrite():      Replace a fact with a new version
    - remove():       Delete a fact entirely
    - get_true_facts(): Get all currently-true facts
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: Minimum similarity for retrieval (paper uses 0.7
                                  for inference-time retrieval, Appendix A.3)
        """
        self.entries: Dict[str, FactEntry] = {}  # fact_id -> FactEntry
        self.similarity_threshold = similarity_threshold

        # Statistics
        self._total_added = 0
        self._total_reinforced = 0
        self._total_made_false = 0
        self._total_rewritten = 0
        self._total_removed = 0

    def add_fact(
        self,
        fact: str,
        embedding: np.ndarray,
        timestamp: str,
        source: str = "",
        metadata: Optional[Dict] = None,
    ) -> FactEntry:
        """
        Add a new fact to the knowledge base (Step 3 of ERASE).

        Args:
            fact:      Atomic fact string
            embedding: Dense vector
            timestamp: When this fact was introduced
            source:    Source document
            metadata:  Additional info

        Returns:
            The created FactEntry
        """
        embedding = self._normalize(embedding)
        entry = FactEntry(
            fact=fact,
            embedding=embedding,
            history=[FactHistory(timestamp=timestamp, truth_value=True)],
            source=source,
            metadata=metadata or {},
        )
        self.entries[entry.fact_id] = entry
        self._total_added += 1

        logger.debug(f"Added fact [{entry.fact_id}]: {fact[:80]}")
        return entry

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
        only_true: bool = False,
    ) -> List[RetrievalResult]:
        """
        Dense vector retrieval of facts similar to query.

        Paper Eq (2): Retrieve(K, d) = arg top-k_{(f_j, H_j) in K} E(d)^T E(f_j)

        Args:
            query_embedding: Query vector (either document or question embedding)
            top_k:          Number of results to return
            threshold:      Minimum similarity (overrides self.similarity_threshold)
            only_true:      If True, only return currently-true facts (for inference)

        Returns:
            List of RetrievalResult sorted by similarity (descending)
        """
        if not self.entries:
            return []

        query_embedding = self._normalize(query_embedding)
        thresh = threshold if threshold is not None else self.similarity_threshold

        results = []
        for entry in self.entries.values():
            if only_true and not entry.is_currently_true:
                continue

            sim = float(np.dot(query_embedding, entry.embedding))
            if sim >= thresh:
                results.append(RetrievalResult(entry=entry, similarity=sim))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def retrieve_for_update(
        self,
        document_embedding: np.ndarray,
        top_k: int = 20,
    ) -> List[RetrievalResult]:
        """
        Retrieve facts for the update step (Step 1 of ERASE).

        Uses a lower threshold than inference (we want to find ALL potentially
        affected facts, not just highly relevant ones).

        Args:
            document_embedding: Embedding of the new document
            top_k:             Number of candidates

        Returns:
            List of RetrievalResult (all facts, not just true ones)
        """
        return self.retrieve(
            query_embedding=document_embedding,
            top_k=top_k,
            threshold=0.3,  # lower threshold for update retrieval
            only_true=False,
        )

    def reinforce_fact(self, fact_id: str, timestamp: str):
        """Reinforce: mark fact as still true (Step 2 of ERASE)."""
        if fact_id in self.entries:
            self.entries[fact_id].reinforce(timestamp)
            self._total_reinforced += 1
            logger.debug(f"Reinforced fact [{fact_id}]")

    def make_fact_false(self, fact_id: str, timestamp: str):
        """Make False: mark fact as no longer true (Step 2 of ERASE)."""
        if fact_id in self.entries:
            self.entries[fact_id].make_false(timestamp)
            self._total_made_false += 1
            logger.debug(f"Made false fact [{fact_id}]")

    def rewrite_fact(
        self,
        fact_id: str,
        new_fact: str,
        new_embedding: np.ndarray,
        timestamp: str,
    ) -> Optional[FactEntry]:
        """
        Rewrite: replace old fact with new version (Step 2 of ERASE).

        The old entry is replaced with a new one that has a fresh history
        starting with (timestamp, True).

        Args:
            fact_id:       ID of the fact to rewrite
            new_fact:      The rewritten fact text
            new_embedding: Embedding of the rewritten fact
            timestamp:     Current timestamp

        Returns:
            The new FactEntry, or None if fact_id not found
        """
        if fact_id not in self.entries:
            return None

        old_entry = self.entries[fact_id]
        new_embedding = self._normalize(new_embedding)

        # Create new entry, preserving the id for traceability
        new_entry = FactEntry(
            fact=new_fact,
            embedding=new_embedding,
            history=[FactHistory(timestamp=timestamp, truth_value=True)],
            fact_id=fact_id,  # keep same ID
            source=old_entry.source,
            metadata={
                **old_entry.metadata,
                "rewritten_from": old_entry.fact,
                "rewrite_timestamp": timestamp,
            },
        )
        self.entries[fact_id] = new_entry
        self._total_rewritten += 1

        logger.debug(
            f"Rewrote fact [{fact_id}]: '{old_entry.fact[:40]}' -> '{new_fact[:40]}'"
        )
        return new_entry

    def remove_fact(self, fact_id: str) -> bool:
        """Remove a fact entirely from the knowledge base."""
        if fact_id in self.entries:
            del self.entries[fact_id]
            self._total_removed += 1
            return True
        return False

    def get_true_facts(self) -> List[FactEntry]:
        """Get all facts currently considered true."""
        return [e for e in self.entries.values() if e.is_currently_true]

    def get_false_facts(self) -> List[FactEntry]:
        """Get all facts currently considered false."""
        return [e for e in self.entries.values() if not e.is_currently_true]

    def get_all_facts(self) -> List[FactEntry]:
        """Get all facts regardless of truth status."""
        return list(self.entries.values())

    def size(self) -> int:
        """Total number of facts in the KB."""
        return len(self.entries)

    def format_facts_for_inference(
        self,
        facts: List[RetrievalResult],
    ) -> str:
        """
        Format retrieved facts with history for LLM inference context.

        Paper Appendix A.3 format:
          f_i (v_{i0} at tau_{i0}, v_{i1} at tau_{i1}, ...)
        """
        if not facts:
            return "(No relevant facts found)"

        lines = []
        for i, r in enumerate(facts, 1):
            entry = r.entry
            hist_str = entry.format_history_string()
            lines.append(f"{entry.fact} ({hist_str})")
        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Get KB statistics."""
        true_count = sum(1 for e in self.entries.values() if e.is_currently_true)
        false_count = len(self.entries) - true_count
        return {
            "total_facts": len(self.entries),
            "true_facts": true_count,
            "false_facts": false_count,
            "total_added": self._total_added,
            "total_reinforced": self._total_reinforced,
            "total_made_false": self._total_made_false,
            "total_rewritten": self._total_rewritten,
            "total_removed": self._total_removed,
        }

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
