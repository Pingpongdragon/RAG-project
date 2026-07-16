"""Causal domain adaptation for partition-routed cold retrieval.

The adapter gives a semantic or metadata partition two online roles without
turning an entire partition into a cache object:

1. the *current* query embedding routes cold retrieval to a few regions;
2. concrete documents are ranked exactly only inside those regions.

A continuously decayed history prior and current-window topic placement remain
available as audited ablations.  Both have zero influence in the main config:
the controlled experiments showed that semantic regions are useful retrieval
indices but are not reliable cache objects or future-topic predictors.

The query path never observes current-window gold evidence.  ``observe`` is
called only after the window has been served, so its prior can affect only a
future window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .topic_partition import TopicPartition


def _normalize_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-norm row")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _stable_top(values: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    limit = min(max(0, int(limit)), len(values))
    if limit == 0:
        return np.empty(0, dtype=np.int64)
    return np.lexsort((np.arange(len(values)), -values))[:limit]


@dataclass(frozen=True)
class RoutedQuery:
    """One query's region route and concrete cold-document results."""

    regions: tuple[int, ...]
    documents: tuple[int, ...]
    scores: tuple[float, ...]
    scanned_documents: int


@dataclass(frozen=True)
class RoutedWindow:
    """Immutable routing result produced before the current window is served."""

    queries: tuple[RoutedQuery, ...]
    unique_documents: tuple[int, ...]
    scanned_documents: int
    fetched_documents: int
    prior: tuple[float, ...]
    reliability: float


class DomainAdapter:
    """Continuous domain prior plus query-conditioned region routing."""

    def __init__(
        self,
        partition: TopicPartition,
        document_embeddings: np.ndarray,
        *,
        prior_rate: float = 0.25,
        reliability_rate: float = 0.25,
        prior_weight: float = 0.25,
        route_width: int = 2,
        retrieve_topk: int = 4,
        candidate_budget: int = 0,
    ) -> None:
        if not isinstance(partition, TopicPartition):
            raise TypeError("partition must implement TopicPartition")
        self.partition = partition
        self.document_embeddings = _normalize_rows(
            document_embeddings, name="document_embeddings"
        )
        if len(self.document_embeddings) != partition.n_documents:
            raise ValueError("partition and document embeddings disagree")
        self.prior_rate = float(prior_rate)
        self.reliability_rate = float(reliability_rate)
        self.prior_weight = float(prior_weight)
        self.route_width = int(route_width)
        self.retrieve_topk = int(retrieve_topk)
        self.candidate_budget = int(candidate_budget)
        if not 0.0 <= self.prior_rate <= 1.0:
            raise ValueError("prior_rate must lie in [0, 1]")
        if not 0.0 <= self.reliability_rate <= 1.0:
            raise ValueError("reliability_rate must lie in [0, 1]")
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0.0:
            raise ValueError("prior_weight must be finite and non-negative")
        if self.route_width < 1:
            raise ValueError("route_width must be positive")
        if self.retrieve_topk < 1:
            raise ValueError("retrieve_topk must be positive")
        if self.candidate_budget < 0:
            raise ValueError("candidate_budget must be non-negative")

        self.prior = np.full(
            partition.n_topics,
            1.0 / float(partition.n_topics),
            dtype=np.float64,
        )
        centroids = np.zeros(
            (partition.n_topics, self.document_embeddings.shape[1]),
            dtype=np.float32,
        )
        for topic in range(partition.n_topics):
            members = np.asarray(partition.hard_bucket(topic), dtype=np.int64)
            if len(members):
                center = self.document_embeddings[members].mean(axis=0)
                norm = float(np.linalg.norm(center))
                if norm > 1e-12:
                    centroids[topic] = center / norm
        self.centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self.observations = 0
        # Online prequential reliability: a prior earns influence only after
        # it repeatedly agrees with subsequently observed evidence mixtures.
        self.reliability = 0.0
        self.current_distribution = np.full(
            partition.n_topics,
            1.0 / float(partition.n_topics),
            dtype=np.float64,
        )
        self.adaptation_strength = 0.0

    def observe(self, document_positions: Iterable[int]) -> np.ndarray:
        """Update the prior from one completed evidence window."""

        positions = [
            self.partition.validate_document_position(position)
            for position in document_positions
        ]
        if not positions:
            return self.prior.copy()
        histogram = self.partition.topic_histogram(positions, soft=True)
        if float(histogram.sum()) <= 0.0:
            return self.prior.copy()
        agreement = float(np.sqrt(self.prior * histogram).sum())
        self.reliability = (
            (1.0 - self.reliability_rate) * self.reliability
            + self.reliability_rate * agreement
        )
        self.prior = (
            (1.0 - self.prior_rate) * self.prior
            + self.prior_rate * np.asarray(histogram, dtype=np.float64)
        )
        self.prior /= float(self.prior.sum())
        self.current_distribution = np.asarray(histogram, dtype=np.float64)
        active_support = min(
            self.partition.n_topics, len(set(int(p) for p in positions))
        )
        if active_support <= 1:
            self.adaptation_strength = 1.0
        else:
            entropy = -float(np.sum(
                self.current_distribution
                * np.log(np.clip(self.current_distribution, 1e-12, None))
            ))
            self.adaptation_strength = float(np.clip(
                1.0 - entropy / np.log(float(active_support)), 0.0, 1.0
            ))
        self.observations += 1
        return self.prior.copy()

    def route(self, query_embeddings: np.ndarray) -> RoutedWindow:
        """Route current queries and rank documents inside selected regions."""

        queries = _normalize_rows(query_embeddings, name="query_embeddings")
        log_prior = np.log(np.clip(self.prior, 1e-8, None))
        log_prior -= float(log_prior.mean())
        routed: list[RoutedQuery] = []
        unique_documents: set[int] = set()
        total_scanned = 0
        total_fetched = 0

        for query in queries:
            region_scores = np.asarray(
                query @ self.centroids.T, dtype=np.float64
            )
            region_scores += (
                self.prior_weight * self.reliability * log_prior
            )
            regions = _stable_top(region_scores, self.route_width)
            candidates = sorted({
                int(position)
                for topic in regions
                for position in self.partition.hard_bucket(int(topic))
            })
            total_scanned += len(candidates)
            if candidates:
                candidate_array = np.asarray(candidates, dtype=np.int64)
                document_scores = np.asarray(
                    query @ self.document_embeddings[candidate_array].T,
                    dtype=np.float64,
                )
                selected_local = _stable_top(
                    document_scores, self.retrieve_topk
                )
                documents = tuple(
                    int(candidate_array[index]) for index in selected_local
                )
                scores = tuple(
                    float(document_scores[index]) for index in selected_local
                )
            else:
                documents, scores = (), ()
            unique_documents.update(documents)
            total_fetched += len(documents)
            routed.append(RoutedQuery(
                regions=tuple(int(topic) for topic in regions),
                documents=documents,
                scores=scores,
                scanned_documents=int(len(candidates)),
            ))

        # ``candidate_budget`` is a window-level cold-read budget.  Keep the
        # documents with the strongest score under any current query, then
        # remove non-materialized documents from each query's result.  A zero
        # value means no additional cap beyond per-query Top-k.
        if self.candidate_budget and len(unique_documents) > self.candidate_budget:
            best_score: dict[int, float] = {}
            for item in routed:
                for document, score in zip(item.documents, item.scores):
                    best_score[int(document)] = max(
                        float(score), best_score.get(int(document), -np.inf)
                    )
            materialized = set(sorted(
                best_score,
                key=lambda document: (-best_score[document], int(document)),
            )[: self.candidate_budget])
            routed = [RoutedQuery(
                regions=item.regions,
                documents=tuple(
                    document for document in item.documents
                    if int(document) in materialized
                ),
                scores=tuple(
                    score for document, score in zip(item.documents, item.scores)
                    if int(document) in materialized
                ),
                scanned_documents=item.scanned_documents,
            ) for item in routed]
            unique_documents = materialized
            total_fetched = sum(len(item.documents) for item in routed)

        return RoutedWindow(
            queries=tuple(routed),
            unique_documents=tuple(sorted(unique_documents)),
            scanned_documents=int(total_scanned),
            fetched_documents=int(total_fetched),
            prior=tuple(float(value) for value in self.prior),
            reliability=float(self.reliability),
        )

__all__ = ["DomainAdapter", "RoutedQuery", "RoutedWindow"]
