"""Stateless downstream-utility estimation for one completed RAG request.

The estimator intentionally has no stream history or topic state.  It scores
the documents retrieved for the *current* request from two pieces of evidence
that exist by the time placement runs:

* query--document alignment, available after retrieval;
* answer--document alignment, available after generation.

An explicit citation indicator can optionally be supplied by a citation-aware
generator.  Offline experiments may use reference answers as an attribution
proxy, but must label that protocol rather than calling it generated feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero norm")
    return value / norm


@dataclass(frozen=True)
class DocumentUtility:
    """One candidate's normalized current-request downstream utility."""

    position: int
    utility: float
    query_alignment: float
    answer_alignment: float
    cited: bool


class CurrentRequestUtilityEstimator:
    """Predict document utility without using previous requests."""

    def __init__(
        self,
        *,
        query_weight: float = 0.35,
        answer_weight: float = 0.65,
        citation_bonus: float = 1.0,
        temperature: float = 0.10,
        topk: int = 4,
    ) -> None:
        self.query_weight = float(query_weight)
        self.answer_weight = float(answer_weight)
        self.citation_bonus = float(citation_bonus)
        self.temperature = float(temperature)
        self.topk = int(topk)
        if self.query_weight < 0.0 or self.answer_weight < 0.0:
            raise ValueError("alignment weights must be non-negative")
        if self.query_weight + self.answer_weight <= 0.0:
            raise ValueError("at least one alignment weight must be positive")
        if self.citation_bonus < 0.0:
            raise ValueError("citation_bonus must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.topk < 1:
            raise ValueError("topk must be positive")

    def score(
        self,
        *,
        query_embedding: np.ndarray,
        answer_embedding: np.ndarray,
        candidate_positions: Sequence[int],
        document_embeddings: np.ndarray,
        cited_positions: Iterable[int] = (),
    ) -> tuple[DocumentUtility, ...]:
        """Return a normalized utility distribution over current candidates."""

        candidates = tuple(dict.fromkeys(int(value) for value in candidate_positions))
        if not candidates:
            return ()
        documents = np.asarray(document_embeddings, dtype=np.float32)
        if documents.ndim != 2:
            raise ValueError("document_embeddings must be two-dimensional")
        if min(candidates) < 0 or max(candidates) >= len(documents):
            raise IndexError("candidate position is outside the document pool")

        query = _unit(query_embedding, name="query_embedding")
        answer = _unit(answer_embedding, name="answer_embedding")
        selected = documents[np.asarray(candidates, dtype=np.int64)]
        norms = np.linalg.norm(selected, axis=1, keepdims=True)
        selected = selected / np.clip(norms, 1e-12, None)
        query_alignment = np.clip(selected @ query, 0.0, None)
        answer_alignment = np.clip(selected @ answer, 0.0, None)
        cited = {int(value) for value in cited_positions}
        citation = np.asarray(
            [float(position in cited) for position in candidates],
            dtype=np.float32,
        )
        raw = (
            self.query_weight * query_alignment
            + self.answer_weight * answer_alignment
            + self.citation_bonus * citation
        )
        order = np.lexsort((np.asarray(candidates), -raw))[: self.topk]
        logits = raw[order] / self.temperature
        logits -= float(logits.max(initial=0.0))
        probabilities = np.exp(logits)
        probabilities /= max(float(probabilities.sum()), 1e-12)

        return tuple(
            DocumentUtility(
                position=int(candidates[int(local)]),
                utility=float(probability),
                query_alignment=float(query_alignment[int(local)]),
                answer_alignment=float(answer_alignment[int(local)]),
                cited=bool(citation[int(local)]),
            )
            for local, probability in zip(order, probabilities)
        )

    @staticmethod
    def payload(
        utilities: Sequence[DocumentUtility],
        position_to_title: Mapping[int, str],
        *,
        source: str = "current-query-answer-alignment",
    ) -> list[dict[str, object]]:
        """Convert utilities into the post-service policy event schema."""

        return [
            {
                "title": str(position_to_title[int(item.position)]),
                "utility": float(item.utility),
                "query_alignment": float(item.query_alignment),
                "answer_alignment": float(item.answer_alignment),
                "cited": bool(item.cited),
                "source": str(source),
            }
            for item in utilities
        ]
