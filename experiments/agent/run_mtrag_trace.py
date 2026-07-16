"""Exact evidence-residency replay for the official MTRAG benchmark.

This benchmark evaluates *placement*, not dense retrieval.  For every MTRAG
request, its qrel passages are treated as post-service evidence-access
feedback: metrics inspect the cache before those passage IDs are revealed,
then the policy may use the IDs to prepare later requests.  Consequently the
runner never embeds the 366K passages or performs a retrieval-time corpus
scan.  The loader still parses the corpus once to validate every qrel ID.

MTRAG has turn order inside each conversation but no global timestamp.  The
runner therefore uses the explicit session protocols in
``experiments.common.session_workload`` and always preserves per-conversation order.
The controlled recurring-domain protocol is labelled synthetic in the output.

Capacity is selected only from the calibration prefix.  The reference size is
the requested quantile of the number of distinct documents needed by an
oracle *inside each calibration window* to cover a requested fraction of that
window's evidence occurrences.  This is a sizing diagnostic, not an oracle
policy: no holdout supports are read during calibration or placement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.session_workload import (  # noqa: E402
    build_recurring_domain_workload,
    build_session_round_robin,
)
from experiments.agent.mtrag_loader import (  # noqa: E402
    DEFAULT_ROOT,
    DOMAINS,
    QUERY_VIEWS,
    load_mtrag_human,
)


ROUND_ROBIN = "session_round_robin"
RECURRING_DOMAIN = "controlled_recurring_domain"
POLICY_NAMES = (
    "LRU", "FIFO", "TinyLFU", "DRIP-Reactive", "DRIP-DomainAdapt",
    "CausalDomainState",
)


def _supports(event: Mapping[str, object]) -> tuple[str, ...]:
    """Return deterministic unique support IDs for one request."""

    raw = event.get("sf_titles")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("every MTRAG event needs sequence-valued sf_titles")
    # TSV row order has no retrieval semantics.  Sorting prevents that file
    # order from silently deciding which support survives an admission tie.
    supports = tuple(sorted({str(item) for item in raw}))
    if not supports:
        raise ValueError("MTRAG qrel requests must have at least one support")
    return supports


def _windows(
    events: Sequence[Mapping[str, object]], window_size: int
) -> Iterable[Sequence[Mapping[str, object]]]:
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    for start in range(0, len(events), window_size):
        window = events[start : start + window_size]
        if window:
            yield window


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    quantile = float(quantile)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return int(ordered[rank - 1])


def calibrate_capacity(
    calibration: Sequence[Mapping[str, object]],
    *,
    window_size: int,
    occurrence_coverage: float = 0.90,
    window_quantile: float = 0.90,
) -> dict[str, object]:
    """Size a cache from calibration windows, never from the holdout.

    For each window, documents are sorted by *within-window* support frequency.
    ``required_documents`` is the smallest prefix covering
    ``occurrence_coverage`` of support occurrences.  The reference capacity is
    the nearest-rank ``window_quantile`` across calibration windows.
    """

    occurrence_coverage = float(occurrence_coverage)
    if not 0.0 < occurrence_coverage <= 1.0:
        raise ValueError("occurrence_coverage must lie in (0, 1]")
    window_quantile = float(window_quantile)
    if not 0.0 < window_quantile <= 1.0:
        raise ValueError("window_quantile must lie in (0, 1]")
    if not calibration:
        raise ValueError("calibration prefix must not be empty")

    requirements: list[int] = []
    unique_working_sets: list[int] = []
    occurrences_per_window: list[int] = []
    all_occurrences = 0
    all_unique: set[str] = set()
    for window in _windows(calibration, int(window_size)):
        counts = Counter(
            support for event in window for support in _supports(event)
        )
        occurrences = int(sum(counts.values()))
        target = occurrence_coverage * occurrences
        covered = 0
        required = 0
        for frequency in sorted(counts.values(), reverse=True):
            covered += int(frequency)
            required += 1
            if covered >= target:
                break
        requirements.append(max(1, required))
        unique_working_sets.append(len(counts))
        occurrences_per_window.append(occurrences)
        all_occurrences += occurrences
        all_unique.update(counts)

    reference = max(1, _nearest_rank(requirements, window_quantile))
    sweep = sorted({
        max(1, math.ceil(reference / 4)),
        max(1, math.ceil(reference / 2)),
        reference,
        2 * reference,
    })
    return {
        "method": (
            "calibration-only dynamic working-set curve; nearest-rank "
            "window quantile"
        ),
        "occurrence_coverage_target": occurrence_coverage,
        "window_quantile": window_quantile,
        "calibration_windows": len(requirements),
        "calibration_queries": len(calibration),
        "calibration_support_occurrences": all_occurrences,
        "calibration_unique_supports": len(all_unique),
        "required_documents_per_window": requirements,
        "unique_documents_per_window": unique_working_sets,
        "support_occurrences_per_window": occurrences_per_window,
        "reference_capacity": reference,
        "recommended_sweep": sweep,
    }


class ExactCache:
    """Base class for exact evidence-key caches with optional write cap."""

    def __init__(self, capacity: int, write_budget: int | None = None) -> None:
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if write_budget is not None and int(write_budget) < 0:
            raise ValueError("write_budget must be non-negative or None")
        self.write_budget = (
            None if write_budget is None else int(write_budget)
        )
        self.residents: set[str] = set()
        self.total_writes = 0
        self.total_cold_reads = 0
        self.window_writes = 0
        self.window_cold_reads = 0
        self.window_index = -1

    def begin_window(self, window_index: int) -> None:
        self.window_index = int(window_index)
        self.window_writes = 0
        self.window_cold_reads = 0

    def end_window(
        self, events: Sequence[Mapping[str, object]]
    ) -> None:
        del events

    def observe_query(
        self, supports: Sequence[str], event: Mapping[str, object]
    ) -> None:
        raise NotImplementedError

    def _can_write(self) -> bool:
        return (
            self.write_budget is None
            or self.window_writes < self.write_budget
        )

    def _record_read(self) -> None:
        self.total_cold_reads += 1
        self.window_cold_reads += 1

    def _record_write(self) -> None:
        self.total_writes += 1
        self.window_writes += 1


class LRUCache(ExactCache):
    """Classic exact-key LRU with miss admission."""

    def __init__(self, capacity: int, write_budget: int | None = None) -> None:
        super().__init__(capacity, write_budget)
        self.last_access: dict[str, int] = {}
        self._tick = 0

    def _victim(self, protected: set[str] | None = None) -> str:
        protected = protected or set()
        candidates = self.residents - protected
        if not candidates:
            candidates = set(self.residents)
        return min(
            candidates,
            key=lambda item: (self.last_access.get(item, -1), item),
        )

    def _admit(self, item: str, protected: set[str] | None = None) -> bool:
        if item in self.residents or not self._can_write():
            return False
        if len(self.residents) >= self.capacity:
            victim = self._victim(protected)
            self.residents.remove(victim)
            self.last_access.pop(victim, None)
        self.residents.add(item)
        self.last_access[item] = self._tick
        self._record_write()
        return True

    def observe_query(
        self, supports: Sequence[str], event: Mapping[str, object]
    ) -> None:
        del event
        for item in supports:
            self._tick += 1
            if item in self.residents:
                self.last_access[item] = self._tick
                continue
            self._record_read()
            self._admit(item)


class FIFOCache(ExactCache):
    """Classic exact-key FIFO with miss admission."""

    def __init__(self, capacity: int, write_budget: int | None = None) -> None:
        super().__init__(capacity, write_budget)
        self.queue: deque[str] = deque()

    def observe_query(
        self, supports: Sequence[str], event: Mapping[str, object]
    ) -> None:
        del event
        for item in supports:
            if item in self.residents:
                continue
            self._record_read()
            if not self._can_write():
                continue
            while len(self.residents) >= self.capacity and self.queue:
                victim = self.queue.popleft()
                if victim in self.residents:
                    self.residents.remove(victim)
                    break
            self.residents.add(item)
            self.queue.append(item)
            self._record_write()


class TinyLFUCache(ExactCache):
    """Deterministic TinyLFU-style frequency admission for exact keys.

    ``fetch_frequency`` is the idealised (collision-free) TinyLFU sketch.
    Resident access counts choose a victim, while a missed candidate is
    admitted only when its historical fetch count is no smaller than the
    victim's resident frequency.
    """

    def __init__(self, capacity: int, write_budget: int | None = None) -> None:
        super().__init__(capacity, write_budget)
        self.resident_frequency: dict[str, int] = {}
        self.fetch_frequency: Counter[str] = Counter()
        self.tie_rank: dict[str, int] = {}
        self._next_rank = 0

    def observe_query(
        self, supports: Sequence[str], event: Mapping[str, object]
    ) -> None:
        del event
        for item in supports:
            if item in self.residents:
                self.resident_frequency[item] = (
                    self.resident_frequency.get(item, 0) + 1
                )
                continue
            self._record_read()
            self.fetch_frequency[item] += 1
            if not self._can_write():
                continue
            if len(self.residents) >= self.capacity:
                victim = min(
                    self.residents,
                    key=lambda resident: (
                        self.resident_frequency.get(resident, 0),
                        self.tie_rank.get(resident, -1),
                        resident,
                    ),
                )
                if (
                    self.fetch_frequency[item]
                    < self.resident_frequency.get(victim, 0)
                ):
                    continue
                self.residents.remove(victim)
                self.resident_frequency.pop(victim, None)
                self.tie_rank.pop(victim, None)
            self.residents.add(item)
            self.resident_frequency[item] = 0
            self.tie_rank[item] = self._next_rank
            self._next_rank += 1
            self._record_write()


class WindowValueCache(ExactCache):
    """Embedding-free realization of DRIP's priced document placement.

    This adapter is used only for exact evidence-residency traces.  It preserves
    the main policy's causal boundary and switching-price equation, but delegates
    retrieval to the benchmark's observable candidate list.  With
    ``candidate_mass=0`` it is the document-only Reactive ablation; with positive
    mass it gives weak credit to the current query's non-gold candidates and is
    the session-trace DomainAdapt realization.
    """

    def __init__(
        self,
        capacity: int,
        write_budget: int | None = None,
        *,
        candidate_budget: int = 24,
        score_decay: float = 0.25,
        candidate_mass: float = 0.0,
        initial_price: float = 0.25,
        target_rate: float = 0.25,
        adaptive_candidate_gate: bool = False,
        candidate_reliability_rate: float = 0.25,
    ) -> None:
        super().__init__(capacity, write_budget)
        self.candidate_budget = max(1, int(candidate_budget))
        self.score_decay = float(score_decay)
        self.candidate_mass = max(0.0, float(candidate_mass))
        self.initial_price = max(0.0, float(initial_price))
        self.price = self.initial_price
        self.target_rate = float(target_rate)
        self.adaptive_candidate_gate = bool(adaptive_candidate_gate)
        self.candidate_reliability_rate = float(candidate_reliability_rate)
        if not 0.0 <= self.score_decay <= 1.0:
            raise ValueError("score_decay must lie in [0, 1]")
        if not 0.0 <= self.target_rate <= 1.0:
            raise ValueError("target_rate must lie in [0, 1]")
        if not 0.0 <= self.candidate_reliability_rate <= 1.0:
            raise ValueError("candidate_reliability_rate must lie in [0, 1]")
        self.scores: dict[str, float] = {}
        self.pending_evidence: Counter[str] = Counter()
        self.pending_candidates: Counter[str] = Counter()
        self.dual_age = 0
        self.routed_candidate_occurrences = 0
        self.routed_candidate_documents = 0
        self.price_log: list[dict[str, float | int]] = []
        self.previous_candidate_set: set[str] = set()
        self.candidate_reliability = (
            0.0 if self.adaptive_candidate_gate else 1.0
        )
        self.candidate_gate_log: list[dict[str, float | int]] = []

    def reset_evaluation_diagnostics(self) -> None:
        """Start evaluation pricing after calibration has filled the cache.

        Main DRIP receives a cache initialized from warm-up evidence.  Charging
        those compulsory fill writes to the online dual would make the exact
        adapter incomparable by starting evaluation with a saturated price.
        Scores and residents remain causal warm-up state; only counters and the
        dual horizon restart.
        """

        self.price = self.initial_price
        self.dual_age = 0
        self.price_log.clear()
        self.routed_candidate_occurrences = 0
        self.routed_candidate_documents = 0
        self.candidate_gate_log.clear()

    def begin_window(self, window_index: int) -> None:
        super().begin_window(window_index)
        self.pending_evidence.clear()
        self.pending_candidates.clear()

    def observe_query(
        self, supports: Sequence[str], event: Mapping[str, object]
    ) -> None:
        for item in supports:
            self.pending_evidence[str(item)] += 1.0
            if item not in self.residents:
                self._record_read()

        if self.candidate_mass <= 0.0:
            return
        raw = event.get("ctx_titles", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return
        support_set = {str(item) for item in supports}
        candidates = tuple(dict.fromkeys(
            str(item) for item in raw
            if item and str(item) not in support_set
        ))
        if not candidates:
            return
        credit = self.candidate_mass / float(len(candidates))
        for item in candidates:
            self.pending_candidates[item] += credit
        self.routed_candidate_occurrences += len(candidates)
        self.routed_candidate_documents += len(set(candidates))

    def _decay_scores(self) -> None:
        for item in list(self.scores):
            value = self.score_decay * float(self.scores[item])
            if value < 1e-8:
                del self.scores[item]
            else:
                self.scores[item] = value

    def end_window(
        self, events: Sequence[Mapping[str, object]]
    ) -> None:
        del events
        self._decay_scores()
        evidence_occurrences = float(sum(self.pending_evidence.values()))
        delayed_recall = 0.0
        if evidence_occurrences > 0.0 and self.previous_candidate_set:
            delayed_recall = sum(
                float(count)
                for item, count in self.pending_evidence.items()
                if item in self.previous_candidate_set
            ) / evidence_occurrences
        if self.adaptive_candidate_gate:
            rate = self.candidate_reliability_rate
            self.candidate_reliability = (
                (1.0 - rate) * self.candidate_reliability
                + rate * delayed_recall
            )
        for item, count in self.pending_evidence.items():
            self.scores[item] = self.scores.get(item, 0.0) + float(count)
        for item, count in self.pending_candidates.items():
            self.scores[item] = (
                self.scores.get(item, 0.0)
                + self.candidate_reliability * float(count)
            )

        self.previous_candidate_set = set(sorted(
            self.pending_candidates,
            key=lambda item: (-self.pending_candidates[item], item),
        )[: self.candidate_budget])
        self.candidate_gate_log.append({
            "window": int(self.window_index),
            "delayed_recall": round(float(delayed_recall), 6),
            "reliability": round(float(self.candidate_reliability), 6),
            "previous_candidates": len(self.previous_candidate_set),
        })

        candidate_pool = set(self.pending_evidence)
        if self.candidate_reliability > 0.0:
            candidate_pool.update(self.pending_candidates)
        candidate_pool -= self.residents
        ordered = sorted(
            candidate_pool,
            key=lambda item: (-self.scores.get(item, 0.0), item),
        )[: self.candidate_budget]
        universe = self.residents | set(ordered)
        scale = max(
            (self.scores.get(item, 0.0) for item in universe),
            default=1.0,
        )
        scale = max(scale, 1e-12)
        writes_before = self.window_writes

        for candidate in ordered:
            if not self._can_write():
                break
            if len(self.residents) < self.capacity:
                self.residents.add(candidate)
                self._record_write()
                continue
            victim = min(
                self.residents,
                key=lambda item: (self.scores.get(item, 0.0), item),
            )
            gain = (
                self.scores.get(candidate, 0.0)
                - self.scores.get(victim, 0.0)
            ) / scale
            if gain <= self.price:
                continue
            self.residents.remove(victim)
            self.residents.add(candidate)
            self._record_write()

        realized = self.window_writes - writes_before
        effective_budget = (
            self.capacity if self.write_budget is None
            else max(1, self.write_budget)
        )
        self.dual_age += 1
        step = 1.0 / math.sqrt(float(self.dual_age))
        before = self.price
        load = float(realized) / float(effective_budget)
        self.price = max(
            0.0, self.price + step * (load - self.target_rate)
        )
        self.price_log.append({
            "window": int(self.window_index),
            "writes": int(realized),
            "load": round(load, 6),
            "price_before": round(before, 6),
            "price_after": round(self.price, 6),
        })


@dataclass(frozen=True)
class DomainForecast:
    observed_domain: str
    predicted_domain: str | None
    confidence: float
    state_weight: float
    global_weight: float
    gate_active: bool
    candidates: tuple[str, ...]


class CausalDomainStateCache(LRUCache):
    """LRU plus a gated causal ``domain state -> document`` predictor.

    The predictor observes only completed windows.  It learns transitions
    between each window's dominant metadata domain and decayed document reuse
    conditioned on that state.  Equal-budget global and state candidate sets
    receive delayed Hedge feedback.  Prefetch activates only after two
    comparable feedback windows and only when the state expert has strictly
    higher weight, so a non-predictive stream falls back to ordinary LRU.
    """

    def __init__(
        self,
        capacity: int,
        write_budget: int | None = None,
        *,
        candidate_budget: int | None = None,
        score_decay: float = 0.95,
        transition_decay: float = 0.95,
        hedge_eta: float = 1.0,
        minimum_confidence: float = 0.50,
        minimum_forecast_reliability: float = 0.60,
        minimum_weight_margin: float = 0.01,
    ) -> None:
        super().__init__(capacity, write_budget)
        self.candidate_budget = max(
            1,
            int(candidate_budget)
            if candidate_budget is not None
            else math.ceil(0.25 * capacity),
        )
        self.score_decay = float(score_decay)
        self.transition_decay = float(transition_decay)
        self.hedge_eta = max(0.0, float(hedge_eta))
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_forecast_reliability = float(
            minimum_forecast_reliability
        )
        self.minimum_weight_margin = max(0.0, float(minimum_weight_margin))
        if not 0.0 <= self.score_decay <= 1.0:
            raise ValueError("score_decay must lie in [0, 1]")
        if not 0.0 <= self.transition_decay <= 1.0:
            raise ValueError("transition_decay must lie in [0, 1]")

        self.global_scores: dict[str, float] = {}
        self.domain_scores: dict[str, dict[str, float]] = defaultdict(dict)
        self.transitions: dict[str, dict[str, float]] = defaultdict(dict)
        self.previous_domain: str | None = None
        self.pending_prediction: str | None = None
        self.pending_candidates: tuple[str, ...] = ()
        self.pending_gate_active = False
        self.pending_experts: dict[str, tuple[str, ...]] | None = None
        self.log_weights = {"global": 0.0, "state": 0.0}
        self.comparable_updates = 0
        self.speculative: set[str] = set()
        self.active_speculative: set[str] = set()

        self.forecasts = 0
        self.forecast_top1_correct = 0
        self.forecast_observed_mass = 0.0
        self.forecast_trials_total = 0
        self.forecast_correct_total = 0
        self.gate_windows = 0
        self.prefetch_writes = 0
        self.proactive_document_fetches = 0
        self.speculative_hits = 0
        self.forecast_log: list[dict[str, object]] = []

    @property
    def expert_weights(self) -> dict[str, float]:
        maximum = max(self.log_weights.values())
        values = {
            name: math.exp(value - maximum)
            for name, value in self.log_weights.items()
        }
        total = sum(values.values())
        return {name: value / total for name, value in values.items()}

    def begin_window(self, window_index: int) -> None:
        super().begin_window(window_index)
        # Last window's unconsumed predictions are now stale.  New inserts are
        # protected only for the window they were forecast for; this realizes
        # a bounded proactive region instead of evicting a prediction before
        # later requests in the same window can touch it.
        self.active_speculative = set()
        if self.pending_gate_active:
            self.gate_windows += 1
        if not self.pending_candidates:
            self.pending_gate_active = False
            return
        protected = set(self.pending_candidates)
        for item in self.pending_candidates:
            if item in self.residents:
                continue
            before = self.total_writes
            if self._admit_prefetch(item, protected):
                self.prefetch_writes += self.total_writes - before
                self.proactive_document_fetches += 1
                self.active_speculative.add(item)
        self.pending_candidates = ()
        self.pending_gate_active = False

    def _admit_prefetch(self, item: str, protected: set[str]) -> bool:
        if item in self.residents or not self._can_write():
            return False
        if len(self.residents) >= self.capacity:
            stale_speculative = self.speculative - protected
            if stale_speculative:
                victim = min(
                    stale_speculative,
                    key=lambda resident: (
                        self.last_access.get(resident, -1), resident
                    ),
                )
            else:
                victim = self._victim(protected)
            self.residents.remove(victim)
            self.last_access.pop(victim, None)
            self.speculative.discard(victim)
            self.active_speculative.discard(victim)
        self.residents.add(item)
        # A speculative insertion is deliberately older than every observed
        # access and remains a first-class eviction victim until it is hit.
        self.last_access[item] = -1
        self.speculative.add(item)
        self._record_write()
        return True

    def _victim(self, protected: set[str] | None = None) -> str:
        protected = protected or set()
        stale_speculative = (
            self.speculative - self.active_speculative - protected
        )
        if stale_speculative:
            return min(
                stale_speculative,
                key=lambda item: (self.last_access.get(item, -1), item),
            )
        return super()._victim(protected | self.active_speculative)

    def _admit(self, item: str, protected: set[str] | None = None) -> bool:
        before = set(self.residents)
        admitted = super()._admit(item, protected)
        if admitted:
            self.speculative.intersection_update(self.residents)
            self.active_speculative.intersection_update(self.residents)
            self.speculative.discard(item)
            self.active_speculative.discard(item)
        elif item in before:
            self.speculative.discard(item)
            self.active_speculative.discard(item)
        return admitted

    def observe_query(
        self, supports: Sequence[str], event: Mapping[str, object]
    ) -> None:
        speculative_hits = sum(
            item in self.speculative and item in self.residents
            for item in supports
        )
        self.speculative_hits += speculative_hits
        self.speculative.difference_update(supports)
        self.active_speculative.difference_update(supports)
        super().observe_query(supports, event)

    @staticmethod
    def _top(scores: Mapping[str, float], budget: int) -> tuple[str, ...]:
        return tuple(sorted(
            (item for item, value in scores.items() if value > 0.0),
            key=lambda item: (-float(scores[item]), item),
        )[:budget])

    def _decay(self) -> None:
        for scores in [self.global_scores, *self.domain_scores.values()]:
            for item in list(scores):
                value = float(scores[item]) * self.score_decay
                if value < 1e-8:
                    del scores[item]
                else:
                    scores[item] = value
        for row in self.transitions.values():
            for target in list(row):
                value = float(row[target]) * self.transition_decay
                if value < 1e-8:
                    del row[target]
                else:
                    row[target] = value

    def end_window(
        self, events: Sequence[Mapping[str, object]]
    ) -> None:
        if not events:
            return
        domains = [str(event.get("domain") or "unknown") for event in events]
        domain_counts = Counter(domains)
        observed_domain = min(
            domain_counts,
            key=lambda domain: (-domain_counts[domain], domain),
        )
        observed_supports = [
            support for event in events for support in _supports(event)
        ]

        issued_prediction = self.pending_prediction
        previous_mass = None
        previous_correct = None
        if issued_prediction is not None:
            self.forecasts += 1
            self.forecast_trials_total += 1
            previous_correct = issued_prediction == observed_domain
            self.forecast_top1_correct += int(previous_correct)
            self.forecast_correct_total += int(previous_correct)
            previous_mass = (
                domain_counts[issued_prediction] / float(len(domains))
            )
            self.forecast_observed_mass += previous_mass

        previous_rates: dict[str, float] = {}
        if self.pending_experts is not None and observed_supports:
            losses: dict[str, float] = {}
            comparable = True
            for name in ("global", "state"):
                candidates = set(self.pending_experts[name])
                if not candidates:
                    comparable = False
                    continue
                hit_rate = sum(
                    support in candidates for support in observed_supports
                ) / float(len(observed_supports))
                previous_rates[name] = hit_rate
                losses[name] = 1.0 - hit_rate
            if comparable:
                self.comparable_updates += 1
                for name in ("global", "state"):
                    self.log_weights[name] -= self.hedge_eta * losses[name]
                maximum = max(self.log_weights.values())
                for name in self.log_weights:
                    self.log_weights[name] -= maximum

        self._decay()
        if self.previous_domain is not None:
            row = self.transitions[self.previous_domain]
            row[observed_domain] = row.get(observed_domain, 0.0) + 1.0
        for support in observed_supports:
            self.global_scores[support] = (
                self.global_scores.get(support, 0.0) + 1.0
            )
            scores = self.domain_scores[observed_domain]
            scores[support] = scores.get(support, 0.0) + 1.0

        predicted_domain = None
        confidence = 0.0
        row = self.transitions.get(observed_domain, {})
        if row:
            best = min(row, key=lambda domain: (-row[domain], domain))
            confidence = float(row[best]) / (float(sum(row.values())) + 1.0)
            if confidence >= self.minimum_confidence:
                predicted_domain = best

        global_candidates = self._top(
            self.global_scores, self.candidate_budget
        )
        state_candidates = (
            ()
            if predicted_domain is None
            else self._top(
                self.domain_scores[predicted_domain], self.candidate_budget
            )
        )
        self.pending_experts = {
            "global": global_candidates,
            "state": state_candidates,
        }
        weights = self.expert_weights
        # A Beta(1, 1) posterior mean prevents one lucky transition from
        # opening the proactive gate on mixed-domain round-robin streams.
        forecast_reliability = (
            self.forecast_correct_total + 1.0
        ) / (self.forecast_trials_total + 2.0)
        gate_active = (
            predicted_domain is not None
            and bool(state_candidates)
            and self.comparable_updates >= 2
            and forecast_reliability >= self.minimum_forecast_reliability
            and weights["state"]
            > weights["global"] + self.minimum_weight_margin
        )

        candidates: tuple[str, ...] = ()
        if gate_active:
            combined: dict[str, float] = defaultdict(float)
            global_total = max(sum(self.global_scores.values()), 1e-12)
            state_scores = self.domain_scores[predicted_domain]
            state_total = max(sum(state_scores.values()), 1e-12)
            for item, value in self.global_scores.items():
                combined[item] += weights["global"] * value / global_total
            for item, value in state_scores.items():
                combined[item] += weights["state"] * value / state_total
            candidates = self._top(combined, self.candidate_budget)
        self.pending_candidates = candidates
        self.pending_gate_active = gate_active
        self.pending_prediction = predicted_domain
        self.previous_domain = observed_domain
        self.forecast_log.append({
            "window_index": self.window_index,
            "observed_domain": observed_domain,
            "observed_domain_fraction": round(
                domain_counts[observed_domain] / float(len(domains)), 6
            ),
            "previous_prediction": issued_prediction,
            "previous_prediction_correct": previous_correct,
            "previous_predicted_domain_mass": (
                None if previous_mass is None else round(previous_mass, 6)
            ),
            "next_prediction": predicted_domain,
            "next_confidence": round(confidence, 6),
            "expert_weights": {
                name: round(value, 6) for name, value in weights.items()
            },
            "previous_expert_hit_rates": {
                name: round(value, 6)
                for name, value in previous_rates.items()
            },
            "comparable_expert_updates": self.comparable_updates,
            "forecast_reliability": round(forecast_reliability, 6),
            "gate_active": gate_active,
            "proposed_documents": len(candidates),
        })

    def reset_evaluation_diagnostics(self) -> None:
        """Clear warmup counters while retaining learned causal state."""

        self.forecasts = 0
        self.forecast_top1_correct = 0
        self.forecast_observed_mass = 0.0
        self.gate_windows = 0
        self.prefetch_writes = 0
        self.proactive_document_fetches = 0
        self.speculative_hits = 0
        self.forecast_log = []

    def diagnostic_summary(self) -> dict[str, object]:
        return {
            "forecasted_windows": self.forecasts,
            "forecast_top1_accuracy": round(
                self.forecast_top1_correct / max(1, self.forecasts), 6
            ),
            "forecast_observed_domain_mass": round(
                self.forecast_observed_mass / max(1, self.forecasts), 6
            ),
            "causal_forecast_reliability": round(
                (self.forecast_correct_total + 1.0)
                / (self.forecast_trials_total + 2.0),
                6,
            ),
            "comparable_expert_updates": self.comparable_updates,
            "final_expert_weights": {
                name: round(value, 6)
                for name, value in self.expert_weights.items()
            },
            "gate_windows": self.gate_windows,
            "prefetch_writes": self.prefetch_writes,
            "proactive_document_fetches": self.proactive_document_fetches,
            "speculative_hits": self.speculative_hits,
            "useful_prefetches_per_write": round(
                self.speculative_hits / max(1, self.prefetch_writes), 6
            ),
            "forecast_log": self.forecast_log,
        }


def _make_policies(
    names: Sequence[str],
    *,
    capacity: int,
    write_budget: int | None,
    candidate_budget: int,
    score_decay: float,
    transition_decay: float,
    hedge_eta: float,
    minimum_confidence: float,
    minimum_forecast_reliability: float,
) -> dict[str, ExactCache]:
    policies: dict[str, ExactCache] = {}
    for name in names:
        if name == "LRU":
            policy = LRUCache(capacity, write_budget)
        elif name == "FIFO":
            policy = FIFOCache(capacity, write_budget)
        elif name == "TinyLFU":
            policy = TinyLFUCache(capacity, write_budget)
        elif name == "DRIP-Reactive":
            policy = WindowValueCache(
                capacity,
                write_budget,
                candidate_budget=candidate_budget,
                score_decay=score_decay,
                candidate_mass=0.0,
            )
        elif name == "DRIP-DomainAdapt":
            policy = WindowValueCache(
                capacity,
                write_budget,
                candidate_budget=candidate_budget,
                score_decay=score_decay,
                candidate_mass=1.0,
                adaptive_candidate_gate=True,
            )
        elif name == "CausalDomainState":
            policy = CausalDomainStateCache(
                capacity,
                write_budget,
                candidate_budget=candidate_budget,
                score_decay=score_decay,
                transition_decay=transition_decay,
                hedge_eta=hedge_eta,
                minimum_confidence=minimum_confidence,
                minimum_forecast_reliability=minimum_forecast_reliability,
            )
        else:
            raise ValueError(f"unknown policy: {name}")
        policies[name] = policy
    return policies


def _replay_warmup(
    policies: Mapping[str, ExactCache],
    events: Sequence[Mapping[str, object]],
    *,
    window_size: int,
) -> None:
    for window_index, window in enumerate(_windows(events, window_size)):
        for policy in policies.values():
            policy.begin_window(window_index)
        for event in window:
            supports = _supports(event)
            for policy in policies.values():
                policy.observe_query(supports, event)
        for policy in policies.values():
            policy.end_window(window)


def evaluate_trace(
    calibration: Sequence[Mapping[str, object]],
    evaluation: Sequence[Mapping[str, object]],
    *,
    capacity: int,
    window_size: int,
    policy_names: Sequence[str] = POLICY_NAMES,
    write_budget: int | None = None,
    candidate_budget: int | None = None,
    score_decay: float = 0.95,
    transition_decay: float = 0.95,
    hedge_eta: float = 1.0,
    minimum_confidence: float = 0.50,
    minimum_forecast_reliability: float = 0.60,
) -> dict[str, dict[str, object]]:
    """Causally warm policies on ``calibration`` and score ``evaluation``."""

    if not evaluation:
        raise ValueError("evaluation stream must not be empty")
    candidate_budget = (
        max(1, math.ceil(0.25 * int(capacity)))
        if candidate_budget is None
        else max(1, int(candidate_budget))
    )
    policies = _make_policies(
        policy_names,
        capacity=int(capacity),
        write_budget=write_budget,
        candidate_budget=candidate_budget,
        score_decay=score_decay,
        transition_decay=transition_decay,
        hedge_eta=hedge_eta,
        minimum_confidence=minimum_confidence,
        minimum_forecast_reliability=minimum_forecast_reliability,
    )
    _replay_warmup(
        policies, calibration, window_size=int(window_size)
    )
    warmup_counters = {
        name: (policy.total_writes, policy.total_cold_reads)
        for name, policy in policies.items()
    }
    for policy in policies.values():
        if isinstance(policy, (CausalDomainStateCache, WindowValueCache)):
            policy.reset_evaluation_diagnostics()

    records = {
        name: {
            "queries": 0,
            "strict_hits": 0,
            "any_hits": 0,
            "support_hits": 0,
            "support_occurrences": 0,
            "per_window_strict_hit_rate": [],
            "per_window_any_hit_rate": [],
            "per_window_evidence_coverage": [],
            "per_window_writes": [],
        }
        for name in policies
    }
    window_offset = math.ceil(len(calibration) / int(window_size))
    for local_window, window in enumerate(_windows(evaluation, window_size)):
        window_index = window_offset + local_window
        before_writes = {
            name: policy.total_writes for name, policy in policies.items()
        }
        for policy in policies.values():
            policy.begin_window(window_index)
        window_counts = {
            name: {"q": 0, "strict": 0, "any": 0, "hit": 0, "occ": 0}
            for name in policies
        }
        for event in window:
            supports = _supports(event)
            support_set = set(supports)
            for name, policy in policies.items():
                hits = len(support_set & policy.residents)
                counts = window_counts[name]
                counts["q"] += 1
                counts["strict"] += int(hits == len(support_set))
                counts["any"] += int(hits > 0)
                counts["hit"] += hits
                counts["occ"] += len(support_set)
            # Evidence IDs become observable only after the current metrics.
            for policy in policies.values():
                policy.observe_query(supports, event)
        for name, policy in policies.items():
            policy.end_window(window)
            counts = window_counts[name]
            record = records[name]
            record["queries"] += counts["q"]
            record["strict_hits"] += counts["strict"]
            record["any_hits"] += counts["any"]
            record["support_hits"] += counts["hit"]
            record["support_occurrences"] += counts["occ"]
            record["per_window_strict_hit_rate"].append(round(
                counts["strict"] / max(1, counts["q"]), 6
            ))
            record["per_window_any_hit_rate"].append(round(
                counts["any"] / max(1, counts["q"]), 6
            ))
            record["per_window_evidence_coverage"].append(round(
                counts["hit"] / max(1, counts["occ"]), 6
            ))
            record["per_window_writes"].append(
                policy.total_writes - before_writes[name]
            )

    result: dict[str, dict[str, object]] = {}
    for name, policy in policies.items():
        record = records[name]
        warmup_writes, warmup_reads = warmup_counters[name]
        summary: dict[str, object] = {
            "strict_all_support_hit_rate": round(
                record["strict_hits"] / max(1, record["queries"]), 6
            ),
            "any_support_hit_rate": round(
                record["any_hits"] / max(1, record["queries"]), 6
            ),
            "evidence_coverage": round(
                record["support_hits"]
                / max(1, record["support_occurrences"]),
                6,
            ),
            "strict_hits": record["strict_hits"],
            "any_hits": record["any_hits"],
            "queries": record["queries"],
            "support_hits": record["support_hits"],
            "support_occurrences": record["support_occurrences"],
            "capacity": policy.capacity,
            "final_residents": len(policy.residents),
            "cache_writes": policy.total_writes - warmup_writes,
            "warmup_cache_writes": warmup_writes,
            "cold_evidence_reads": policy.total_cold_reads - warmup_reads,
            "reactive_cold_evidence_reads": (
                policy.total_cold_reads - warmup_reads
            ),
            "warmup_cold_evidence_reads": warmup_reads,
            "write_rate_per_query": round(
                (policy.total_writes - warmup_writes)
                / max(1, record["queries"]),
                6,
            ),
            "per_window_strict_hit_rate": record[
                "per_window_strict_hit_rate"
            ],
            "per_window_any_hit_rate": record[
                "per_window_any_hit_rate"
            ],
            "per_window_evidence_coverage": record[
                "per_window_evidence_coverage"
            ],
            "per_window_writes": record["per_window_writes"],
        }
        if isinstance(policy, CausalDomainStateCache):
            diagnostics = policy.diagnostic_summary()
            proactive_writes = int(diagnostics["prefetch_writes"])
            summary["reactive_cache_writes"] = (
                int(summary["cache_writes"]) - proactive_writes
            )
            summary["prefetch_cache_writes"] = proactive_writes
            summary["proactive_document_fetches"] = int(
                diagnostics["proactive_document_fetches"]
            )
            summary["total_document_fetches"] = (
                int(summary["reactive_cold_evidence_reads"])
                + int(summary["proactive_document_fetches"])
            )
            summary["topic_state_diagnostics"] = diagnostics
        elif isinstance(policy, WindowValueCache):
            summary["reactive_cache_writes"] = int(summary["cache_writes"])
            summary["prefetch_cache_writes"] = 0
            summary["proactive_document_fetches"] = 0
            summary["total_document_fetches"] = int(
                summary["reactive_cold_evidence_reads"]
            )
            summary["routed_candidate_occurrences"] = int(
                policy.routed_candidate_occurrences
            )
            summary["final_switching_price"] = round(policy.price, 6)
            summary["final_candidate_reliability"] = round(
                policy.candidate_reliability, 6
            )
            summary["candidate_gate_log"] = list(policy.candidate_gate_log)
            summary["switching_price_log"] = list(policy.price_log)
        else:
            summary["reactive_cache_writes"] = int(summary["cache_writes"])
            summary["prefetch_cache_writes"] = 0
            summary["proactive_document_fetches"] = 0
            summary["total_document_fetches"] = int(
                summary["reactive_cold_evidence_reads"]
            )
        result[name] = summary
    return result


def _automatic_calibration_size(total: int, window_size: int) -> int:
    if total < 2:
        raise ValueError("MTRAG stream needs at least two events")
    proposed = max(window_size, math.floor(0.20 * total / window_size) * window_size)
    return min(proposed, total - 1)


def _build_stream(
    queries: Sequence[Mapping[str, object]], args: argparse.Namespace
):
    calibration_size = int(args.calibration_size)
    if calibration_size == 0:
        calibration_size = _automatic_calibration_size(
            len(queries), int(args.window_size)
        )
    if not 0 < calibration_size < len(queries):
        raise ValueError(
            "calibration_size must leave at least one evaluation event"
        )
    evaluation_size = (
        len(queries) - calibration_size
        if args.evaluation_size is None
        else int(args.evaluation_size)
    )
    common = dict(
        seed=int(args.seed),
        warmup_size=calibration_size,
        evaluation_size=evaluation_size,
        window_size=int(args.window_size),
    )
    if args.protocol == ROUND_ROBIN:
        evaluation, calibration, audit = build_session_round_robin(
            queries, **common
        )
    else:
        evaluation, calibration, audit = build_recurring_domain_workload(
            queries,
            block_size=int(args.block_size),
            **common,
        )
    return calibration, evaluation, audit


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    doc_pool, queries, title_to_idx = load_mtrag_human(
        root=args.root,
        domains=args.domains,
        query_view=args.query_view,
        max_queries=args.max_queries,
    )
    passage_count = len(doc_pool)
    # The exact trace uses only stable evidence keys.  Releasing corpus text
    # here is the important memory boundary: no 366K-document embedding/index.
    del doc_pool, title_to_idx

    calibration, evaluation, audit = _build_stream(queries, args)
    capacity_audit = calibrate_capacity(
        calibration,
        window_size=int(args.window_size),
        occurrence_coverage=float(args.capacity_coverage),
        window_quantile=float(args.capacity_quantile),
    )
    reference_capacity = int(capacity_audit["reference_capacity"])
    capacity = (
        reference_capacity if int(args.cache_size) == 0
        else int(args.cache_size)
    )
    candidate_budget = (
        max(1, math.ceil(0.25 * capacity))
        if int(args.candidate_budget) == 0
        else int(args.candidate_budget)
    )
    write_budget = (
        None if int(args.write_budget) == 0 else int(args.write_budget)
    )
    summaries = evaluate_trace(
        calibration,
        evaluation,
        capacity=capacity,
        window_size=int(args.window_size),
        policy_names=args.policies,
        write_budget=write_budget,
        candidate_budget=candidate_budget,
        score_decay=float(args.score_decay),
        transition_decay=float(args.transition_decay),
        hedge_eta=float(args.hedge_eta),
        minimum_confidence=float(args.minimum_confidence),
        minimum_forecast_reliability=float(
            args.minimum_forecast_reliability
        ),
    )
    return {
        "dataset": "official_mtrag_human_retrieval",
        "evaluation_scope": (
            "exact qrel evidence residency; not dense retrieval or answer quality"
        ),
        "method_scope": (
            "CausalDomainState is an embedding-free trace-level realization "
            "of the same causal domain-state/document-selection idea; it is "
            "not a direct invocation of algorithms/drip/policy.py"
        ),
        "protocol": {
            "access_feedback": (
                "post-service official qrel passage IDs; metrics computed before feedback"
            ),
            "current_request_leakage": False,
            "global_timestamp_available": False,
            "session_order_preserved": True,
            "workload": audit.as_dict(),
            "query_view": args.query_view,
            "domains": list(args.domains),
            "passages": passage_count,
            "queries_loaded": len(queries),
            "calibration_queries": len(calibration),
            "evaluation_queries": len(evaluation),
            "window_size": int(args.window_size),
            "block_size": (
                int(args.block_size) if args.protocol == RECURRING_DOMAIN else None
            ),
            "cache_size": capacity,
            "cache_size_source": (
                "calibration_working_set"
                if int(args.cache_size) == 0 else "explicit_override"
            ),
            "write_budget_per_window": write_budget,
            "candidate_budget": candidate_budget,
            "seed": int(args.seed),
        },
        "capacity_calibration": capacity_audit,
        "summary": summaries,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS)
    )
    parser.add_argument(
        "--query-view", choices=QUERY_VIEWS, default="rewrite"
    )
    parser.add_argument(
        "--protocol",
        choices=(ROUND_ROBIN, RECURRING_DOMAIN),
        default=ROUND_ROBIN,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=0,
        help="0 selects an approximately 20%% prefix aligned to full windows",
    )
    parser.add_argument("--evaluation-size", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=0,
        help="0 uses calibration-only working-set sizing",
    )
    parser.add_argument("--capacity-coverage", type=float, default=0.90)
    parser.add_argument("--capacity-quantile", type=float, default=0.90)
    parser.add_argument(
        "--write-budget",
        type=int,
        default=0,
        help="0 means standard unbounded miss admission within each window",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=0,
        help="0 uses ceil(0.25 * cache_size)",
    )
    parser.add_argument("--score-decay", type=float, default=0.95)
    parser.add_argument("--transition-decay", type=float, default=0.95)
    parser.add_argument("--hedge-eta", type=float, default=1.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.50)
    parser.add_argument(
        "--minimum-forecast-reliability", type=float, default=0.60
    )
    parser.add_argument(
        "--policies", nargs="+", choices=POLICY_NAMES, default=list(POLICY_NAMES)
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.window_size < 1 or args.block_size < 1:
        parser.error("window_size and block_size must be positive")
    if args.cache_size < 0 or args.candidate_budget < 0 or args.write_budget < 0:
        parser.error("cache and budget values must be non-negative")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        name: {
            "strict": summary["strict_all_support_hit_rate"],
            "any": summary["any_support_hit_rate"],
            "coverage": summary["evidence_coverage"],
            "writes": summary["cache_writes"],
        }
        for name, summary in result["summary"].items()
    }
    print(json.dumps({
        "protocol": result["protocol"],
        "capacity": result["capacity_calibration"],
        "cache": compact,
    }, indent=2))


if __name__ == "__main__":
    main()
