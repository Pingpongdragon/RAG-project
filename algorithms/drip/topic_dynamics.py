"""Explicit topic drift detection and causal soft topic dynamics.

The classes in this module operate only on feedback from completed windows.
``ExplicitTopicDriftDetector`` compares the current topic histogram with an
EWMA reference using Jensen--Shannon divergence and accumulates positive
innovations with a one-sided CUSUM.  ``SoftTopicDynamics`` learns a soft
topic-to-topic transition matrix and an episodic topic-mixture directory; it can
therefore rank only documents that have already appeared in the causal prefix.

The legacy hard-state/prototype forecaster was removed from the active code so
this module is the sole topic-dynamics implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import operator
from typing import Iterable, Sequence

import numpy as np

from .topic_partition import TopicPartition


_SPARSE_FLOOR = 1e-8


def _probability_vector(
    values: Sequence[float],
    expected_size: int,
    *,
    name: str,
) -> np.ndarray:
    """Validate and normalize a finite, non-negative vector."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (int(expected_size),):
        raise ValueError(
            f"{name} must have length {int(expected_size)}, got {len(vector)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be non-negative")
    mass = float(vector.sum())
    if mass <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return vector / mass


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jensen--Shannon divergence in natural-log units."""

    mixture = 0.5 * (left + right)
    left_mask = left > 0.0
    right_mask = right > 0.0
    divergence = 0.0
    if np.any(left_mask):
        divergence += 0.5 * float(np.sum(
            left[left_mask]
            * np.log(left[left_mask] / mixture[left_mask])
        ))
    if np.any(right_mask):
        divergence += 0.5 * float(np.sum(
            right[right_mask]
            * np.log(right[right_mask] / mixture[right_mask])
        ))
    # Round-off can produce a tiny negative number around zero.
    return float(max(0.0, divergence))


@dataclass(frozen=True)
class TopicDriftDecision:
    """One causal drift observation.

    ``reference_before`` is the distribution against which ``histogram`` was
    scored.  ``reference_after`` is the EWMA state retained for the next
    completed window.  ``cusum`` reports the threshold-tested statistic even
    when the detector resets its internal statistic after an alarm.
    """

    histogram: tuple[float, ...]
    reference_before: tuple[float, ...]
    reference_after: tuple[float, ...]
    drift_score: float
    cusum: float
    alarm: bool
    observations: int


class ExplicitTopicDriftDetector:
    """Jensen--Shannon drift detector with EWMA reference and one-sided CUSUM.

    The first completed histogram initializes the reference and never raises an
    alarm.  Every later histogram is compared with the *previous* reference,
    so the current observation cannot rewrite the baseline before it is scored.
    On an alarm, the current histogram becomes the new reference and the
    internal CUSUM is reset after the reported decision.
    """

    def __init__(
        self,
        n_topics: int,
        *,
        reference_rate: float = 0.05,
        slack: float = 0.01,
        threshold: float = 0.10,
    ) -> None:
        if isinstance(n_topics, bool):
            raise TypeError("n_topics must be an integer")
        try:
            self.n_topics = operator.index(n_topics)
        except TypeError as exc:
            raise TypeError("n_topics must be an integer") from exc
        self.reference_rate = float(reference_rate)
        self.slack = float(slack)
        self.threshold = float(threshold)
        if self.n_topics < 1:
            raise ValueError("n_topics must be positive")
        if not 0.0 < self.reference_rate <= 1.0:
            raise ValueError("reference_rate must lie in (0, 1]")
        if not np.isfinite(self.slack) or self.slack < 0.0:
            raise ValueError("slack must be a finite non-negative value")
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("threshold must be a finite positive value")

        self.reference: np.ndarray | None = None
        self.cusum = 0.0
        self.observations = 0

    def reset(self) -> None:
        """Forget the reference and all accumulated detector evidence."""

        self.reference = None
        self.cusum = 0.0
        self.observations = 0

    def observe(self, histogram: Sequence[float]) -> TopicDriftDecision:
        """Score one completed-window topic histogram and update detector state."""

        current = _probability_vector(
            histogram, self.n_topics, name="topic_histogram"
        )
        self.observations += 1
        if self.reference is None:
            self.reference = current.copy()
            frozen = tuple(float(value) for value in current)
            return TopicDriftDecision(
                histogram=frozen,
                reference_before=frozen,
                reference_after=frozen,
                drift_score=0.0,
                cusum=0.0,
                alarm=False,
                observations=int(self.observations),
            )

        reference_before = self.reference.copy()
        score = _jensen_shannon(current, reference_before)
        tested_cusum = max(0.0, float(self.cusum) + score - self.slack)
        alarm = bool(tested_cusum >= self.threshold)

        if alarm:
            # Reset only after exposing the threshold-crossing statistic.
            reference_after = current.copy()
            self.cusum = 0.0
        else:
            reference_after = (
                (1.0 - self.reference_rate) * reference_before
                + self.reference_rate * current
            )
            reference_after /= float(reference_after.sum())
            self.cusum = float(tested_cusum)
        self.reference = reference_after

        return TopicDriftDecision(
            histogram=tuple(float(value) for value in current),
            reference_before=tuple(
                float(value) for value in reference_before
            ),
            reference_after=tuple(float(value) for value in reference_after),
            drift_score=float(score),
            cusum=float(tested_cusum),
            alarm=alarm,
            observations=int(self.observations),
        )


@dataclass(frozen=True)
class SoftTopicDecision:
    """Drift diagnostics and a causal next-window document forecast."""

    drift_score: float
    drift_cusum: float
    drift_alarm: bool
    predicted_distribution: tuple[float, ...]
    predicted_topic: int | None
    forecast_confidence: float
    transition_support: float
    previous_forecast_similarity: float | None
    previous_document_recall: float | None
    previous_document_precision: float | None
    documents: tuple[int, ...]
    scores: tuple[float, ...]
    candidate_budget: int
    observed_documents: int


@dataclass
class _TopicEpisode:
    """One completed-window signature and its observed evidence multiset."""

    histogram: np.ndarray
    documents: Counter
    weight: float = 1.0


class SoftTopicDynamics:
    """Causal soft transitions and mixture-conditioned episodic reuse.

    Calling :meth:`observe_and_forecast` for completed window ``t`` performs
    four ordered operations:

    1. evaluate the forecast saved after window ``t-1`` against ``h_t``;
    2. run the explicit detector against its pre-``t`` EWMA reference;
    3. add the completed soft transition ``h_{t-1} outer h_t`` and evidence
       occurrences from window ``t``;
    4. forecast ``h_{t+1}`` and rank previously observed documents.

    No cold topic bucket is scanned and a document absent from the completed
    history can never be proposed.
    """

    def __init__(
        self,
        partition: TopicPartition,
        *,
        drift_reference_rate: float = 0.05,
        drift_slack: float = 0.01,
        drift_threshold: float = 0.10,
        transition_decay: float = 0.95,
        document_decay: float = 0.50,
        min_transition_support: float = 1.0,
        min_forecast_confidence: float = 0.50,
    ) -> None:
        if not isinstance(partition, TopicPartition):
            raise TypeError("partition must implement TopicPartition")
        self.partition = partition
        self.n_topics = int(partition.n_topics)
        self.transition_decay = float(transition_decay)
        self.document_decay = float(document_decay)
        self.min_transition_support = float(min_transition_support)
        self.min_forecast_confidence = float(min_forecast_confidence)
        if not 0.0 <= self.transition_decay <= 1.0:
            raise ValueError("transition_decay must lie in [0, 1]")
        if not 0.0 <= self.document_decay <= 1.0:
            raise ValueError("document_decay must lie in [0, 1]")
        if (
            not np.isfinite(self.min_transition_support)
            or self.min_transition_support < 0.0
        ):
            raise ValueError(
                "min_transition_support must be finite and non-negative"
            )
        if not 0.0 <= self.min_forecast_confidence <= 1.0:
            raise ValueError("min_forecast_confidence must lie in [0, 1]")

        self.detector = ExplicitTopicDriftDetector(
            self.n_topics,
            reference_rate=drift_reference_rate,
            slack=drift_slack,
            threshold=drift_threshold,
        )
        self.transitions = np.zeros(
            (self.n_topics, self.n_topics), dtype=np.float64
        )
        self.episodes: list[_TopicEpisode] = []
        self.previous_histogram: np.ndarray | None = None
        self.pending_forecast: np.ndarray | None = None
        self.pending_documents: tuple[int, ...] | None = None

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0.0:
            return 0.0
        return float(np.clip(float(left @ right) / denominator, 0.0, 1.0))

    def _decay_document_memory(self) -> None:
        retained = []
        for episode in self.episodes:
            episode.weight *= self.document_decay
            if episode.weight >= _SPARSE_FLOOR:
                retained.append(episode)
        # A bounded episodic directory is enough for online topic matching and
        # prevents metadata growth on indefinitely long streams.
        self.episodes = retained[-128:]

    def _transition_forecast(
        self, current: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        row_mass = self.transitions.sum(axis=1)
        transition_support = float(current @ row_mass)
        conditional = np.divide(
            self.transitions,
            row_mass[:, None],
            out=np.zeros_like(self.transitions),
            where=row_mass[:, None] > 0.0,
        )
        raw = current @ conditional
        raw_mass = float(raw.sum())
        predicted = (
            np.zeros(self.n_topics, dtype=np.float64)
            if raw_mass <= 0.0 else raw / raw_mass
        )
        # Support is a conservative pseudo-count confidence.  A soft forecast
        # may legitimately have high entropy, so entropy itself is not treated
        # as uncertainty; delayed forecast similarity is logged separately.
        confidence = transition_support / (1.0 + transition_support)
        return predicted, float(transition_support), float(confidence)

    def _document_candidates(
        self,
        predicted: np.ndarray,
        budget: int,
        excluded: set[int],
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Retrieve evidence from historical windows nearest to the forecast.

        Topic IDs are intentionally not treated independently: the complete
        soft mixture is the context key.  This preserves correlations between
        topics that a per-topic frequency table would destroy.
        """

        combined: dict[int, float] = {}
        for episode in self.episodes:
            divergence = _jensen_shannon(predicted, episode.histogram)
            context_weight = float(episode.weight) * float(
                np.exp(-divergence / 0.10)
            )
            occurrence_mass = float(sum(episode.documents.values()))
            if context_weight <= 0.0 or occurrence_mass <= 0.0:
                continue
            for position, count in episode.documents.items():
                if position in excluded or int(count) <= 0:
                    continue
                combined[position] = (
                    combined.get(position, 0.0)
                    + context_weight * float(count) / occurrence_mass
                )

        mass = float(sum(combined.values()))
        if mass <= 0.0 or budget <= 0:
            return (), ()
        normalized = {
            position: float(value) / mass
            for position, value in combined.items()
        }
        ranked = tuple(sorted(
            normalized,
            key=lambda position: (-normalized[position], position),
        )[:budget])
        return ranked, tuple(float(normalized[position]) for position in ranked)

    def observe_and_forecast(
        self,
        histogram: Sequence[float],
        observed_document_positions: Iterable[int],
        candidate_budget: int,
        *,
        exclude: Iterable[int] = (),
    ) -> SoftTopicDecision:
        """Observe completed feedback and forecast the next window causally."""

        current = _probability_vector(
            histogram, self.n_topics, name="topic_histogram"
        )
        if isinstance(candidate_budget, bool):
            raise TypeError("candidate_budget must be an integer")
        try:
            budget = operator.index(candidate_budget)
        except TypeError as exc:
            raise TypeError("candidate_budget must be an integer") from exc
        if budget < 0:
            raise ValueError("candidate_budget must be non-negative")

        observed = [
            self.partition.validate_document_position(position)
            for position in observed_document_positions
        ]
        excluded = {
            self.partition.validate_document_position(position)
            for position in exclude
        }

        previous_similarity = (
            None if self.pending_forecast is None
            else self._cosine(self.pending_forecast, current)
        )
        if self.pending_documents is None or not observed:
            previous_document_recall = None
            previous_document_precision = None
        else:
            pending = set(self.pending_documents)
            previous_document_recall = float(
                sum(position in pending for position in observed)
                / len(observed)
            )
            previous_document_precision = float(
                len(pending & set(observed)) / max(1, len(pending))
            )
        drift = self.detector.observe(current)

        self.transitions *= self.transition_decay
        self._decay_document_memory()
        if self.previous_histogram is not None:
            self.transitions += np.outer(self.previous_histogram, current)

        occurrence_counts = Counter(observed)
        if occurrence_counts:
            self.episodes.append(_TopicEpisode(
                histogram=current.copy(),
                documents=Counter(occurrence_counts),
            ))

        predicted, support, confidence = self._transition_forecast(current)
        eligible = bool(
            float(predicted.sum()) > 0.0
            and support >= self.min_transition_support
            and confidence >= self.min_forecast_confidence
        )
        predicted_topic = (
            int(np.argmax(predicted)) if eligible else None
        )
        if eligible:
            documents, scores = self._document_candidates(
                predicted, budget, excluded
            )
            self.pending_documents = documents
        else:
            documents, scores = (), ()
            self.pending_documents = None

        self.pending_forecast = (
            predicted.copy() if float(predicted.sum()) > 0.0 else None
        )

        self.previous_histogram = current.copy()
        return SoftTopicDecision(
            drift_score=float(drift.drift_score),
            drift_cusum=float(drift.cusum),
            drift_alarm=bool(drift.alarm),
            predicted_distribution=tuple(
                float(value) for value in predicted
            ),
            predicted_topic=predicted_topic,
            forecast_confidence=float(confidence),
            transition_support=float(support),
            previous_forecast_similarity=previous_similarity,
            previous_document_recall=previous_document_recall,
            previous_document_precision=previous_document_precision,
            documents=documents,
            scores=scores,
            candidate_budget=int(budget),
            observed_documents=int(len(observed)),
        )


__all__ = [
    "ExplicitTopicDriftDetector",
    "SoftTopicDecision",
    "SoftTopicDynamics",
    "TopicDriftDecision",
]
