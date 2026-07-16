#!/usr/bin/env python3
"""审计 region forecast + document selector + shadow buffer 是否真正有效。

该脚本实现 Approach B 的最小、因果、单视图版本：

1. 冷库文档离线划分为 bounded semantic regions；
2. 窗口结束后，根据已观测 support 学习 region -> next region/document；
3. 只从预测 region 中选择少量历史上实际使用过的文档；
4. 文档先进入 TTL=1 的 shadow，命中后再晋升 hot LRU。

总容量始终固定为 ``hot + shadow = C``。主结果 ``CausalShadow`` 不读取未来
query/support；``OracleDocShadow`` 只用于回答“若未来完全可知，shadow 机制本身
最多能提升多少”，不能作为可部署结果。

示例：

    python benchmarks/audit_region_shadow_prefetch.py \
      --datasets squad_direct streamingqa_official \
      --region-sizes 128 256 512 --region-widths 1 2 4 \
      --shadow-fractions 0.1 0.2
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_semantic_pages import (  # noqa: E402
    AUDIT_VERSION as PAGE_AUDIT_VERSION,
    PROTOCOLS,
    PreparedDataset,
    build_balanced_semantic_pages,
    prepare_dataset,
    support_positions,
)
from experiments.common.stream_protocol import causal_prefix_init_kb  # noqa: E402
from experiments.common.factorized_workload import (  # noqa: E402
    WORKLOAD_CHOICES,
)
from experiments.direct.config import DATA_SEED  # noqa: E402


AUDIT_VERSION = "topic-adapt-region-shadow-v2"


def _distribution(values: Counter) -> dict[int, float]:
    total = float(sum(values.values()))
    if total <= 0.0:
        return {}
    return {int(key): float(value) / total for key, value in values.items()}


def _top_keys(values: dict[int, float], width: int) -> list[int]:
    return [
        int(key)
        for key, _ in sorted(
            values.items(), key=lambda item: (-float(item[1]), int(item[0]))
        )[:max(0, int(width))]
    ]


@dataclass(frozen=True)
class Forecast:
    """在窗口 ``t`` 结束后为 ``t+1`` 生成的因果预测。"""

    region_probability: dict[int, float]
    frequency_region_probability: dict[int, float]
    document_probability: dict[int, float]
    confidence: float
    mean_unique_documents: float


class CausalRegionModel:
    """稀疏的一阶 region transition 与 region-conditioned document model。"""

    def __init__(
        self,
        labels: np.ndarray,
        document_embeddings: np.ndarray,
        region_centroids: np.ndarray,
    ):
        self.labels = np.asarray(labels, dtype=np.int32)
        embeddings = np.asarray(document_embeddings, dtype=np.float32)
        embeddings = embeddings / np.clip(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None
        )
        centroids = np.asarray(region_centroids, dtype=np.float32)
        centroids = centroids / np.clip(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None
        )
        self.region_members = {
            int(region): np.flatnonzero(self.labels == region).astype(np.int32)
            for region in range(len(centroids))
        }
        self.region_prior: dict[int, dict[int, float]] = {}
        for region, members in self.region_members.items():
            if not len(members):
                self.region_prior[region] = {}
                continue
            # P_0(d|z)：只由静态冷库得到的 region representativeness prior。
            # 一个总伪计数在有历史时会快速退让，不会压过真实 access feedback。
            similarities = embeddings[members] @ centroids[int(region)]
            weights = np.clip(1.0 + similarities, 1e-6, None)
            weights = weights / float(weights.sum())
            self.region_prior[region] = {
                int(document): float(weight)
                for document, weight in zip(members, weights)
            }
        self.region_transitions: dict[int, Counter] = defaultdict(Counter)
        self.document_transitions: dict[int, Counter] = defaultdict(Counter)
        self.global_regions: Counter = Counter()
        self.global_documents: Counter = Counter()
        self.region_documents: dict[int, Counter] = defaultdict(Counter)
        self.previous_regions: dict[int, float] | None = None
        self.windows = 0
        self.unique_documents_total = 0

    def observe(self, region_counts: Counter, document_counts: Counter) -> Forecast:
        """加入当前窗口反馈，再只用历史预测下一窗口。"""

        current_regions = _distribution(region_counts)
        current_documents = _distribution(document_counts)

        if self.previous_regions and current_regions:
            for source, source_mass in self.previous_regions.items():
                for target, target_mass in current_regions.items():
                    self.region_transitions[int(source)][int(target)] += (
                        float(source_mass) * float(target_mass)
                    )
                for document, target_mass in current_documents.items():
                    self.document_transitions[int(source)][int(document)] += (
                        float(source_mass) * float(target_mass)
                    )

        self.global_regions.update(region_counts)
        self.global_documents.update(document_counts)
        for document, count in document_counts.items():
            self.region_documents[int(self.labels[int(document)])][int(document)] += (
                float(count)
            )
        self.windows += 1
        self.unique_documents_total += len(document_counts)
        self.previous_regions = current_regions
        return self._predict(current_regions)

    def _predict(self, current_regions: dict[int, float]) -> Forecast:
        region_scores: Counter = Counter()
        document_scores: Counter = Counter()
        confidence = 0.0

        for source, source_mass in current_regions.items():
            region_row = self.region_transitions.get(int(source), Counter())
            row_total = float(sum(region_row.values()))
            if row_total <= 0.0:
                continue
            # N/(N+1) 让一次偶然转移不会立刻得到满置信度。
            confidence += float(source_mass) * row_total / (row_total + 1.0)
            for target, value in region_row.items():
                region_scores[int(target)] += (
                    float(source_mass) * float(value) / row_total
                )

            document_row = self.document_transitions.get(int(source), Counter())
            document_total = float(sum(document_row.values()))
            if document_total > 0.0:
                for document, value in document_row.items():
                    document_scores[int(document)] += (
                        float(source_mass) * float(value) / document_total
                    )

        return Forecast(
            region_probability=_distribution(region_scores),
            frequency_region_probability=_distribution(self.global_regions),
            document_probability=_distribution(document_scores),
            confidence=min(1.0, max(0.0, float(confidence))),
            mean_unique_documents=(
                float(self.unique_documents_total) / max(1, self.windows)
            ),
        )

    def rank_documents(self, forecast: Forecast, regions: list[int]) -> list[int]:
        """在预测 region 内融合转移概率与历史区内频率。"""

        region_set = {int(value) for value in regions}
        scores: dict[int, float] = {}
        for region in region_set:
            history = self.region_documents.get(region, Counter())
            history_total = float(sum(history.values()))
            region_probability = float(
                forecast.region_probability.get(region, 0.0)
            )
            for document, prior in self.region_prior.get(region, {}).items():
                transition_value = float(
                    forecast.document_probability.get(int(document), 0.0)
                )
                # Dirichlet backoff: one static-prior pseudo-observation.
                frequency_value = (
                    region_probability
                    * (float(history.get(int(document), 0.0)) + float(prior))
                    / (history_total + 1.0)
                )
                scores[int(document)] = (
                    float(forecast.confidence) * transition_value
                    + (1.0 - float(forecast.confidence)) * frequency_value
                )
        return [
            int(document)
            for document, _ in sorted(
                scores.items(), key=lambda item: (-float(item[1]), int(item[0]))
            )
            if float(scores[document]) > 0.0
        ]


class DocumentLRU:
    """按文档计容量、按窗口末尾 support trace 更新的 LRU。"""

    def __init__(self, capacity: int, initial_documents: list[int]):
        self.capacity = max(1, int(capacity))
        self.documents: OrderedDict[int, None] = OrderedDict()
        self.cold_writes = 0
        self.evictions = 0
        for document in reversed(initial_documents[:self.capacity]):
            self.documents[int(document)] = None

    @property
    def resident(self) -> set[int]:
        return set(self.documents)

    def observe(self, supports: list[set[int]]) -> None:
        for required in supports:
            for document in sorted(required):
                document = int(document)
                if document in self.documents:
                    self.documents.move_to_end(document)
                    continue
                self.cold_writes += 1
                if len(self.documents) >= self.capacity:
                    self.documents.popitem(last=False)
                    self.evictions += 1
                self.documents[document] = None


class RegionShadowCache:
    """固定 ``hot + shadow`` 总容量的文档级 shadow cache。"""

    def __init__(
        self,
        hot_capacity: int,
        shadow_capacity: int,
        initial_hot: list[int],
        initial_shadow: list[int],
    ):
        self.hot = DocumentLRU(hot_capacity, initial_hot)
        self.shadow_capacity = max(1, int(shadow_capacity))
        self.shadow: OrderedDict[int, None] = OrderedDict(
            (int(document), None)
            for document in initial_shadow[:self.shadow_capacity]
            if int(document) not in self.hot.documents
        )
        self.shadow_is_forecast = False
        self.shadow_writes = 0
        self.shadow_evictions = 0
        self.promotions = 0
        self.predicted_occurrences = 0
        self.useful_occurrences = 0
        self.shadow_query_hits = 0
        self.shadow_document_hits = 0
        self.unused_expirations = 0
        self._used_current_shadow: set[int] = set()

    @property
    def resident(self) -> set[int]:
        return self.hot.resident | set(self.shadow)

    @property
    def trust(self) -> float:
        # Beta(1,1) posterior，避免 cold start 时永久关闭预取。
        return float(self.useful_occurrences + 1) / float(
            self.predicted_occurrences + 2
        )

    def evaluate(self, supports: list[set[int]]) -> tuple[int, int]:
        resident = self.resident
        shadow_documents = set(self.shadow)
        has_answer = supported = 0
        used_shadow: set[int] = set()
        for required in supports:
            if not required:
                continue
            supported += 1
            has_answer += int(required.issubset(resident))
            hits = required & shadow_documents
            if hits:
                self.shadow_query_hits += 1
                self.shadow_document_hits += len(hits)
                used_shadow.update(int(value) for value in hits)
        self._used_current_shadow = used_shadow
        if self.shadow_is_forecast:
            self.predicted_occurrences += len(self.shadow)
            self.useful_occurrences += len(used_shadow)
        return has_answer, supported

    def observe(self, supports: list[set[int]]) -> None:
        """命中 shadow 的文档只做逻辑晋升，不重复计 cold write。"""

        for required in supports:
            for document in sorted(required):
                document = int(document)
                if document in self.hot.documents:
                    self.hot.documents.move_to_end(document)
                    continue
                if document in self.shadow:
                    self.shadow.pop(document, None)
                    self.promotions += 1
                    if len(self.hot.documents) >= self.hot.capacity:
                        self.hot.documents.popitem(last=False)
                        self.hot.evictions += 1
                    self.hot.documents[document] = None
                    continue
                self.hot.cold_writes += 1
                if len(self.hot.documents) >= self.hot.capacity:
                    self.hot.documents.popitem(last=False)
                    self.hot.evictions += 1
                self.hot.documents[document] = None

    def install_shadow(self, candidates: list[int], is_forecast: bool = True) -> None:
        selected = []
        for document in candidates:
            document = int(document)
            if document in self.hot.documents or document in selected:
                continue
            selected.append(document)
            if len(selected) >= self.shadow_capacity:
                break
        selected_set = set(selected)

        for document in list(self.shadow):
            if document in selected_set:
                continue
            self.shadow.pop(document, None)
            self.shadow_evictions += 1
            if document not in self._used_current_shadow:
                self.unused_expirations += 1

        for document in selected:
            if document in self.shadow:
                self.shadow.move_to_end(document)
                continue
            self.shadow[document] = None
            self.shadow_writes += 1

        self.shadow_is_forecast = bool(is_forecast)
        self._used_current_shadow = set()


def _window_counters(
    supports: list[set[int]], labels: np.ndarray
) -> tuple[Counter, Counter]:
    documents: Counter = Counter()
    regions: Counter = Counter()
    for required in supports:
        for document in required:
            document = int(document)
            documents[document] += 1
            regions[int(labels[document])] += 1
    return regions, documents


def _rank_initial_documents(
    initial_documents: list[int], warmup_supports: list[set[int]]
) -> list[int]:
    frequency: Counter = Counter()
    last_seen: dict[int, int] = {}
    for query_index, required in enumerate(warmup_supports):
        for document in required:
            frequency[int(document)] += 1
            last_seen[int(document)] = int(query_index)
    return sorted(
        (int(value) for value in initial_documents),
        key=lambda document: (
            -int(frequency.get(document, 0)),
            -int(last_seen.get(document, -1)),
            int(document),
        ),
    )


def _evaluate_resident(
    resident: set[int], supports: list[set[int]]
) -> tuple[int, int]:
    supported = [required for required in supports if required]
    return (
        sum(int(required.issubset(resident)) for required in supported),
        len(supported),
    )


def simulate_full_lru(
    capacity: int,
    initial_documents: list[int],
    windows: list[list[set[int]]],
) -> dict:
    cache = DocumentLRU(capacity, initial_documents)
    hits = queries = 0
    for supports in windows:
        current_hits, current_queries = _evaluate_resident(cache.resident, supports)
        hits += current_hits
        queries += current_queries
        cache.observe(supports)
    has_answer = float(hits) / max(1, queries)
    return {
        "has_answer_rate": round(has_answer, 6),
        "normalized_amat": round(1.0 + (1.0 - has_answer) * 10.0, 6),
        "cold_document_writes": int(cache.cold_writes),
        "evictions": int(cache.evictions),
    }


def _support_coverage(regions: set[int], documents: Counter, labels: np.ndarray) -> int:
    return sum(
        int(count)
        for document, count in documents.items()
        if int(labels[int(document)]) in regions
    )


def simulate_region_shadow(
    labels: np.ndarray,
    document_embeddings: np.ndarray,
    region_centroids: np.ndarray,
    total_capacity: int,
    shadow_fraction: float,
    region_width: int,
    initial_documents: list[int],
    windows: list[list[set[int]]],
) -> dict:
    shadow_capacity = max(1, int(round(total_capacity * shadow_fraction)))
    hot_capacity = max(1, int(total_capacity) - shadow_capacity)
    initial_hot = initial_documents[:hot_capacity]
    initial_shadow = initial_documents[
        hot_capacity:hot_capacity + shadow_capacity
    ]
    causal = RegionShadowCache(
        hot_capacity, shadow_capacity, initial_hot, initial_shadow
    )
    # TopicAdapt 不预测 t+1。它把窗口 t 已观测到的 region mixture 当作
    # 当前 demand state，并用它重新激活同 region 中历史访问过、但当前不在 hot
    # tier 的文档。整个更新仍发生在窗口结束后，所以只能服务 t+1。
    adaptive = RegionShadowCache(
        hot_capacity, shadow_capacity, initial_hot, initial_shadow
    )
    oracle = RegionShadowCache(
        hot_capacity, shadow_capacity, initial_hot, initial_shadow
    )
    hot_only = DocumentLRU(hot_capacity, initial_hot)
    model = CausalRegionModel(labels, document_embeddings, region_centroids)

    causal_hits = causal_queries = 0
    adaptive_hits = adaptive_queries = 0
    oracle_hits = oracle_queries = 0
    hot_hits = hot_queries = 0
    pending: dict | None = None

    transition_region_hits = transition_region_total = 0
    frequency_region_hits = frequency_region_total = 0
    previous_region_hits = previous_region_total = 0
    selected_document_hits = selected_document_total = 0
    adaptive_document_hits = adaptive_document_total = 0
    transition_top1_hits = frequency_top1_hits = top1_trials = 0
    forecast_windows = 0
    confidence_values = []
    budget_values = []
    selected_total = 0
    selected_novel = 0
    previous_actual_regions: set[int] = set()

    for window_index, supports in enumerate(windows):
        regions, documents = _window_counters(supports, labels)
        current_region_distribution = _distribution(regions)
        current_dominant = _top_keys(current_region_distribution, 1)

        if pending is not None:
            predicted_regions = set(pending["predicted_regions"])
            frequency_regions = set(pending["frequency_regions"])
            selected_documents = set(pending["selected_documents"])
            adaptive_documents = set(pending["adaptive_documents"])
            transition_region_hits += _support_coverage(
                predicted_regions, documents, labels
            )
            frequency_region_hits += _support_coverage(
                frequency_regions, documents, labels
            )
            previous_region_hits += _support_coverage(
                previous_actual_regions, documents, labels
            )
            region_occurrences = int(sum(documents.values()))
            transition_region_total += region_occurrences
            frequency_region_total += region_occurrences
            previous_region_total += region_occurrences
            selected_document_hits += sum(
                int(count)
                for document, count in documents.items()
                if int(document) in selected_documents
            )
            selected_document_total += region_occurrences
            adaptive_document_hits += sum(
                int(count)
                for document, count in documents.items()
                if int(document) in adaptive_documents
            )
            adaptive_document_total += region_occurrences
            if current_dominant and pending["predicted_regions"]:
                top1_trials += 1
                transition_top1_hits += int(
                    int(pending["predicted_regions"][0]) == current_dominant[0]
                )
                if pending["frequency_regions"]:
                    frequency_top1_hits += int(
                        int(pending["frequency_regions"][0]) == current_dominant[0]
                    )
            forecast_windows += 1

        current_hits, current_queries = causal.evaluate(supports)
        causal_hits += current_hits
        causal_queries += current_queries
        current_hits, current_queries = adaptive.evaluate(supports)
        adaptive_hits += current_hits
        adaptive_queries += current_queries
        current_hits, current_queries = oracle.evaluate(supports)
        oracle_hits += current_hits
        oracle_queries += current_queries
        current_hits, current_queries = _evaluate_resident(
            hot_only.resident, supports
        )
        hot_hits += current_hits
        hot_queries += current_queries

        causal.observe(supports)
        adaptive.observe(supports)
        oracle.observe(supports)
        hot_only.observe(supports)

        forecast = model.observe(regions, documents)
        predicted_regions = _top_keys(
            forecast.region_probability, region_width
        )
        frequency_regions = _top_keys(
            forecast.frequency_region_probability, region_width
        )
        confidence_values.append(float(forecast.confidence))

        rho = float(forecast.confidence) * float(causal.trust)
        budget = min(
            shadow_capacity,
            int(math.ceil(float(forecast.mean_unique_documents) * rho)),
        ) if predicted_regions else 0
        ranked = model.rank_documents(forecast, predicted_regions)
        selected = [
            int(document)
            for document in ranked
            if int(document) not in causal.hot.documents
        ][:budget]
        selected_total += len(selected)
        selected_novel += sum(
            int(document not in model.global_documents) for document in selected
        )
        causal.install_shadow(selected, is_forecast=bool(selected))
        budget_values.append(len(selected))

        # Persistence-style adaptation baseline.  It does not use a learned
        # transition or a drift alarm.  The current region mixture selects the
        # region directory; within-region historical frequency selects concrete
        # documents.  confidence=0 deliberately disables the transition-document
        # term in rank_documents().
        adaptive_regions = _top_keys(current_region_distribution, region_width)
        adaptive_region_mass = sum(
            float(current_region_distribution.get(region, 0.0))
            for region in adaptive_regions
        )
        adaptive_view = Forecast(
            region_probability=current_region_distribution,
            frequency_region_probability={},
            document_probability={},
            confidence=0.0,
            mean_unique_documents=forecast.mean_unique_documents,
        )
        adaptive_budget = min(
            shadow_capacity,
            int(math.ceil(
                float(forecast.mean_unique_documents)
                * float(adaptive_region_mass)
                * float(adaptive.trust)
            )),
        ) if adaptive_regions else 0
        adaptive_ranked = model.rank_documents(
            adaptive_view, adaptive_regions
        )
        adaptive_selected = [
            int(document)
            for document in adaptive_ranked
            if int(document) not in adaptive.hot.documents
        ][:adaptive_budget]
        adaptive.install_shadow(
            adaptive_selected, is_forecast=bool(adaptive_selected)
        )

        # 非因果上界：下一窗口真实 support 可见，只判断 shadow 容量是否有价值。
        if window_index + 1 < len(windows):
            _, next_documents = _window_counters(windows[window_index + 1], labels)
            oracle_candidates = [
                int(document)
                for document, _ in sorted(
                    next_documents.items(),
                    key=lambda item: (-int(item[1]), int(item[0])),
                )
                if int(document) not in oracle.hot.documents
            ][:shadow_capacity]
        else:
            oracle_candidates = []
        oracle.install_shadow(oracle_candidates, is_forecast=bool(oracle_candidates))

        pending = {
            "predicted_regions": predicted_regions,
            "frequency_regions": frequency_regions,
            "selected_documents": selected,
            "adaptive_documents": adaptive_selected,
        }
        previous_actual_regions = set(
            _top_keys(current_region_distribution, region_width)
        )

    def cache_summary(cache: RegionShadowCache, hits: int, queries: int) -> dict:
        rate = float(hits) / max(1, queries)
        writes = int(cache.hot.cold_writes + cache.shadow_writes)
        return {
            "has_answer_rate": round(rate, 6),
            "normalized_amat": round(1.0 + (1.0 - rate) * 10.0, 6),
            "cold_document_writes": writes,
            "hot_cold_writes": int(cache.hot.cold_writes),
            "shadow_writes": int(cache.shadow_writes),
            "hot_evictions": int(cache.hot.evictions),
            "shadow_evictions": int(cache.shadow_evictions),
            "promotions": int(cache.promotions),
            "prefetch_precision": round(
                float(cache.useful_occurrences)
                / max(1, cache.predicted_occurrences),
                6,
            ),
            "prefetch_trials": int(cache.predicted_occurrences),
            "prefetch_useful": int(cache.useful_occurrences),
            "shadow_query_hits": int(cache.shadow_query_hits),
            "shadow_document_hits": int(cache.shadow_document_hits),
            "unused_expirations": int(cache.unused_expirations),
            "final_trust": round(float(cache.trust), 6),
        }

    hot_rate = float(hot_hits) / max(1, hot_queries)
    return {
        "capacity": {
            "total": int(total_capacity),
            "hot": int(hot_capacity),
            "shadow": int(shadow_capacity),
            "shadow_fraction": float(shadow_fraction),
        },
        "predictability": {
            "forecast_windows": int(forecast_windows),
            "region_locality_coverage": round(
                previous_region_hits / max(1, previous_region_total), 6
            ),
            "transition_region_coverage": round(
                transition_region_hits / max(1, transition_region_total), 6
            ),
            "frequency_region_coverage": round(
                frequency_region_hits / max(1, frequency_region_total), 6
            ),
            "transition_top1_accuracy": round(
                transition_top1_hits / max(1, top1_trials), 6
            ),
            "frequency_top1_accuracy": round(
                frequency_top1_hits / max(1, top1_trials), 6
            ),
            "selected_document_coverage": round(
                selected_document_hits / max(1, selected_document_total), 6
            ),
            "adaptive_document_coverage": round(
                adaptive_document_hits / max(1, adaptive_document_total), 6
            ),
            "confidence_mean": round(
                float(np.mean(confidence_values)) if confidence_values else 0.0,
                6,
            ),
            "prefetch_budget_mean": round(
                float(np.mean(budget_values)) if budget_values else 0.0, 3
            ),
            "novel_prefetch_fraction": round(
                selected_novel / max(1, selected_total), 6
            ),
        },
        "hot_only_lru": {
            "has_answer_rate": round(hot_rate, 6),
            "normalized_amat": round(1.0 + (1.0 - hot_rate) * 10.0, 6),
            "cold_document_writes": int(hot_only.cold_writes),
            "evictions": int(hot_only.evictions),
        },
        "topic_adapt_shadow": cache_summary(
            adaptive, adaptive_hits, adaptive_queries
        ),
        "causal_shadow": cache_summary(causal, causal_hits, causal_queries),
        "oracle_doc_shadow": cache_summary(oracle, oracle_hits, oracle_queries),
    }


def audit_dataset(dataset: PreparedDataset, args: argparse.Namespace) -> dict:
    capacity = max(
        1,
        min(
            len(dataset.doc_pool),
            int(round(len(dataset.doc_pool) * float(args.cache_ratio))),
        ),
    )
    warmup_supports, warmup_missing = support_positions(
        dataset.warmup, dataset.title_to_idx
    )
    eval_supports, eval_missing = support_positions(
        dataset.stream, dataset.title_to_idx
    )
    if warmup_missing or eval_missing:
        raise ValueError(
            f"{dataset.alias}: missing support titles "
            f"(warmup={warmup_missing}, eval={eval_missing})"
        )
    windows = [
        eval_supports[start:start + dataset.protocol.window_size]
        for start in range(0, len(eval_supports), dataset.protocol.window_size)
    ]

    initial_ids = causal_prefix_init_kb(
        dataset.doc_pool,
        dataset.doc_embs,
        dataset.warmup,
        dataset.query_embs,
        capacity,
        seed=int(args.seed) + 313,
    )
    id_to_position = {
        document["doc_id"]: index
        for index, document in enumerate(dataset.doc_pool)
    }
    initial_documents = _rank_initial_documents(
        [id_to_position[doc_id] for doc_id in initial_ids],
        warmup_supports,
    )
    full_lru = simulate_full_lru(capacity, initial_documents, windows)

    result = {
        "protocol": {
            "dataset": dataset.protocol.dataset,
            "workload": dataset.protocol.workload,
            "pool_size": len(dataset.doc_pool),
            "capacity_docs": int(capacity),
            "cache_pool_ratio": round(capacity / len(dataset.doc_pool), 6),
            "warmup_queries": len(dataset.warmup),
            "evaluation_queries": len(dataset.stream),
            "n_windows": len(windows),
            "window_size": dataset.protocol.window_size,
            "constructor": dataset.construction,
            "temporal_sampling": dataset.temporal_sampling,
        },
        "full_capacity_lru": full_lru,
        "regions": {},
    }

    for region_size in args.region_sizes:
        labels, centroids, partition = build_balanced_semantic_pages(
            dataset,
            region_size,
            seed=args.seed,
            force=args.force_partitions,
        )
        region_result = {"partition": partition, "settings": {}}
        for width in args.region_widths:
            for fraction in args.shadow_fractions:
                key = f"L{int(width)}_S{float(fraction):.2f}"
                setting = simulate_region_shadow(
                    labels=labels,
                    document_embeddings=dataset.doc_embs,
                    region_centroids=centroids,
                    total_capacity=capacity,
                    shadow_fraction=float(fraction),
                    region_width=int(width),
                    initial_documents=initial_documents,
                    windows=windows,
                )
                causal = setting["causal_shadow"]
                setting["comparison"] = {
                    "has_answer_gain_vs_full_lru_pp": round(
                        100.0 * (
                            causal["has_answer_rate"]
                            - full_lru["has_answer_rate"]
                        ),
                        3,
                    ),
                    "write_ratio_vs_full_lru": round(
                        causal["cold_document_writes"]
                        / max(1, full_lru["cold_document_writes"]),
                        3,
                    ),
                    "topic_adapt_gain_vs_full_lru_pp": round(
                        100.0 * (
                            setting["topic_adapt_shadow"]["has_answer_rate"]
                            - full_lru["has_answer_rate"]
                        ),
                        3,
                    ),
                }
                region_result["settings"][key] = setting
        result["regions"][str(region_size)] = region_result
    return result


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _best_setting(dataset: dict) -> tuple[str, str, dict]:
    candidates = []
    for region_size, region in dataset["regions"].items():
        for key, setting in region["settings"].items():
            causal = setting["causal_shadow"]
            adaptive = setting["topic_adapt_shadow"]
            candidates.append((
                float(adaptive["has_answer_rate"]),
                -int(adaptive["cold_document_writes"]),
                float(causal["has_answer_rate"]),
                region_size,
                key,
                setting,
            ))
    _, _, _, region_size, key, setting = max(candidates)
    return region_size, key, setting


def markdown_report(payload: dict, command: str) -> str:
    lines = [
        "# Causal Region-Shadow Prefetch Audit",
        "",
        f"Audit version: `{payload['audit_version']}`",
        "",
        "This audit compares topic adaptation against region forecasting with "
        "the same document selector and TTL=1 shadow buffer. `TopicAdapt` uses "
        "the latest completed-window region mixture without predicting a shift. "
        "`CausalShadow` uses only support feedback observed through window t. "
        "`OracleDocShadow` sees the next window and is an unattainable mechanism "
        "upper bound, not a method result. Total cache capacity is fixed.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Best Causal Setting Per Dataset",
        "",
        "| Dataset | Region | Setting | Locality | Markov cov. | Freq. cov. | "
        "Adapt-doc cov. | Forecast-doc cov. | Full LRU HA | TopicAdapt HA | "
        "Forecast HA | Oracle HA | Adapt gain | Forecast gain |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best = {}
    for name, dataset in payload["datasets"].items():
        region_size, key, setting = _best_setting(dataset)
        best[name] = (region_size, key, setting)
        predictability = setting["predictability"]
        causal = setting["causal_shadow"]
        adaptive = setting["topic_adapt_shadow"]
        oracle = setting["oracle_doc_shadow"]
        full_lru = dataset["full_capacity_lru"]
        lines.append(
            f"| {name} | {region_size} | `{key}` | "
            f"{_percent(predictability['region_locality_coverage'])} | "
            f"{_percent(predictability['transition_region_coverage'])} | "
            f"{_percent(predictability['frequency_region_coverage'])} | "
            f"{_percent(predictability['adaptive_document_coverage'])} | "
            f"{_percent(predictability['selected_document_coverage'])} | "
            f"{_percent(full_lru['has_answer_rate'])} | "
            f"{_percent(adaptive['has_answer_rate'])} | "
            f"{_percent(causal['has_answer_rate'])} | "
            f"{_percent(oracle['has_answer_rate'])} | "
            f"{setting['comparison']['topic_adapt_gain_vs_full_lru_pp']:+.1f} pp | "
            f"{setting['comparison']['has_answer_gain_vs_full_lru_pp']:+.1f} pp |"
        )

    lines.extend(["", "## Verdict", ""])
    wins = 0
    for name, (region_size, key, setting) in best.items():
        predictability = setting["predictability"]
        causal = setting["causal_shadow"]
        adaptive = setting["topic_adapt_shadow"]
        gain = float(
            setting["comparison"]["topic_adapt_gain_vs_full_lru_pp"]
        )
        markov_advantage = (
            float(predictability["transition_region_coverage"])
            - float(predictability["frequency_region_coverage"])
        )
        if gain > 0.0:
            wins += 1
        lines.append(
            f"- **{name}:** best `{region_size}/{key}` gives {gain:+.1f} pp "
            f"TopicAdapt Has-Answer versus full-capacity LRU "
            f"({_percent(adaptive['has_answer_rate'])} vs. "
            f"{_percent(payload['datasets'][name]['full_capacity_lru']['has_answer_rate'])}). "
            f"Markov region coverage is "
            f"{100.0 * markov_advantage:+.1f} pp versus historical frequency; "
            f"adapt/forecast document coverage is "
            f"{_percent(predictability['adaptive_document_coverage'])}/"
            f"{_percent(predictability['selected_document_coverage'])}, and "
            f"prefetch precision is {_percent(causal['prefetch_precision'])}."
        )
    lines.extend([
        "",
        (
            "TopicAdapt beats full-capacity document LRU on "
            f"{wins}/{len(best)} datasets. A positive oracle gap alone does not "
            "validate forecasting: it only says that perfect future information "
            "would make reserved shadow capacity useful."
        ),
        "",
        "## Interpretation Rules",
        "",
        "- Markov coverage must exceed the frequency baseline; otherwise the model learns popularity, not a transition.",
        "- Selected-document coverage measures the full forecasting bottleneck after region routing and within-region ranking.",
        "- `Hot-only HA` exposes the capacity tax caused by reserving shadow space.",
        "- `Oracle HA` separates a bad causal predictor from a fundamentally useless shadow mechanism.",
        "- The semantic-only version is a prerequisite test. An entity view is justified only where relation feedback exists and adds coverage beyond semantic regions.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(PROTOCOLS), default=list(PROTOCOLS)
    )
    parser.add_argument("--region-sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--region-widths", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--shadow-fractions", nargs="+", type=float, default=[0.05, 0.1, 0.2]
    )
    parser.add_argument("--cache-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=int(DATA_SEED))
    parser.add_argument("--n-source", type=int, default=None)
    parser.add_argument(
        "--workload", choices=sorted(WORKLOAD_CHOICES), default=None,
        help="override the dataset workload for adaptation controls",
    )
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--warmup-windows", type=int, default=None)
    parser.add_argument("--force-partitions", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "docs" / "experiments" /
        "REGION_SHADOW_AUDIT_2026-07-15.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "docs" / "experiments" /
        "REGION_SHADOW_AUDIT_2026-07-15.md",
    )
    args = parser.parse_args()
    if not 0.0 < args.cache_ratio <= 1.0:
        parser.error("--cache-ratio must be in (0, 1]")
    if any(value <= 0 for value in args.region_sizes + args.region_widths):
        parser.error("region sizes and widths must be positive")
    if any(not 0.0 < value < 1.0 for value in args.shadow_fractions):
        parser.error("shadow fractions must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    command = " ".join([sys.executable, *sys.argv])
    payload = {
        "audit_version": AUDIT_VERSION,
        "page_audit_dependency": PAGE_AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "assumptions": {
            "causality": "K_t and P_t serve all Q_t; feedback changes only t+1",
            "capacity": "hot_capacity + shadow_capacity = baseline capacity",
            "semantic_partition": "cold documents only; no stream labels",
            "feedback": "delayed observed support trace; gold fields emulate logged support access",
            "candidate_novelty": "only previously observed documents are selectable",
            "shadow_ttl": "one-window prediction, retained only when selected again",
            "oracle_role": "mechanism upper bound only",
        },
        "datasets": {},
    }
    for alias in args.datasets:
        prepared = prepare_dataset(alias, args)
        payload["datasets"][alias] = audit_dataset(prepared, args)
        print(f"[{alias}] region-shadow audit complete", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(markdown_report(payload, command), encoding="utf-8")
    print(f"JSON: {args.output}")
    print(f"REPORT: {args.report}")


if __name__ == "__main__":
    main()
