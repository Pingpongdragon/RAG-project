"""冷 RAG 语料库的只读 topic 目录。

本模块只负责划分文档，并把已经完成的 evidence 访问编码成 topic 直方图；它不负责
选择缓存驻留文档。显式漂移检测与 soft topic 预测位于
:mod:`algorithms.drip.topic_dynamics`。明确这条边界，是为了避免把“预测到 topic”误当成
“已经知道下一次会请求哪篇未见文档”。

Version 2 additionally gives every partition a reproducible corpus/partition
fingerprint, stable typed metadata ordering, strict numeric validation, and a
metric-explicit semantic assignment. Existing position-based public methods
remain available.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, datetime, time
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Iterable, Sequence

import numpy as np


PARTITION_VERSION = "topic-partition-v2"


def _normalize_rows(
    values: np.ndarray,
    *,
    name: str = "document_embeddings",
    reject_zero: bool = True,
) -> np.ndarray:
    """Return finite, contiguous, L2-normalized float32 rows.

    Zero document embeddings have no semantic direction and are rejected by
    default rather than being assigned to a topic through an arbitrary tie.
    ``reject_zero=False`` is reserved for exposing Euclidean KMeans centers,
    where a zero center is valid even though it has no cosine direction.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if len(array) == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.all(np.isfinite(array)):
        bad_rows = np.flatnonzero(~np.all(np.isfinite(array), axis=1))
        preview = ", ".join(str(int(row)) for row in bad_rows[:5])
        raise ValueError(f"{name} contains NaN or Inf in row(s): {preview}")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] <= 1e-12)
    if reject_zero and len(zero_rows):
        preview = ", ".join(str(int(row)) for row in zero_rows[:5])
        raise ValueError(f"{name} contains zero-norm row(s): {preview}")
    normalized = array / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _readonly_array(values: np.ndarray, dtype=None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _stable_top_indices(values: np.ndarray, limit: int) -> np.ndarray:
    """按值降序返回索引；值相同时用文档/topic ID 稳定打破平局。"""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("ranking values must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("ranking values must be finite")
    limit = min(max(0, int(limit)), len(array))
    if limit == 0:
        return np.empty(0, dtype=np.int64)
    return np.lexsort((np.arange(len(array)), -array))[:limit]


def _typed_value_key(value: object) -> tuple[str, object]:
    """Create a stable, type-sensitive key for metadata labels and IDs.

    In particular, ``True`` and ``1`` no longer collapse into the same topic.
    Supported compound labels are tuples and frozensets of supported values.
    Arbitrary objects are rejected because their repr commonly contains a
    process-local address and therefore cannot support stable fingerprints.
    """

    if isinstance(value, np.generic):
        tag = f"{type(value).__module__}.{type(value).__qualname__}"
        return tag, _typed_value_key(value.item())
    if value is None:
        return "builtins.NoneType", ""
    if isinstance(value, bool):
        return "builtins.bool", "1" if value else "0"
    if isinstance(value, int):
        return "builtins.int", str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata labels and document IDs must be finite")
        return "builtins.float", value.hex()
    if isinstance(value, str):
        return "builtins.str", value
    if isinstance(value, bytes):
        return "builtins.bytes", value.hex()
    if isinstance(value, (datetime, date, time)):
        tag = f"{type(value).__module__}.{type(value).__qualname__}"
        return tag, value.isoformat()
    if isinstance(value, tuple):
        return "builtins.tuple", tuple(_typed_value_key(item) for item in value)
    if isinstance(value, frozenset):
        items = sorted(
            (_typed_value_key(item) for item in value),
            key=_canonical_json_bytes,
        )
        return "builtins.frozenset", tuple(items)
    raise TypeError(
        "metadata labels and document IDs must be stable scalar values, "
        "dates, tuples, or frozensets; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _update_digest(digest, value: object) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        header = _canonical_json_bytes({
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        })
        payload = memoryview(array).cast("B")
    elif isinstance(value, bytes):
        header = b"bytes"
        payload = value
    else:
        header = b"json"
        payload = _canonical_json_bytes(value)
    digest.update(struct.pack("<Q", len(header)))
    digest.update(header)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _fingerprint(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        _update_digest(digest, value)
    return digest.hexdigest()


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    converted = int(value)
    if converted != value or converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


class _TypedLabelMap(Mapping):
    """Immutable mapping whose lookup respects the label's concrete type."""

    def __init__(
        self,
        labels: Sequence[object],
        typed_mapping: Mapping[tuple[str, object], int],
    ) -> None:
        self._labels = tuple(labels)
        self._typed_mapping = MappingProxyType(dict(typed_mapping))

    def __getitem__(self, label: object) -> int:
        return int(self._typed_mapping[_typed_value_key(label)])

    def __iter__(self) -> Iterator[object]:
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)


class TopicPartition:
    """元数据分区和语义分区共用的只读接口。"""

    n_documents: int
    n_topics: int
    primary_topics: np.ndarray

    def primary_topic(self, document_position: int) -> int:
        position = self.validate_document_position(document_position)
        return int(self.primary_topics[position])

    def memberships(self, document_position: int) -> tuple[tuple[int, float], ...]:
        position = self.validate_document_position(document_position)
        return self._memberships[position]

    def hard_bucket(self, topic: int) -> tuple[int, ...]:
        topic = self._validate_topic(topic)
        return self._hard_buckets[topic]

    def soft_bucket(
        self, topic: int, limit: int | None = None
    ) -> tuple[tuple[int, float], ...]:
        topic = self._validate_topic(topic)
        bucket = self._soft_buckets[topic]
        return bucket if limit is None else bucket[: max(0, int(limit))]

    def document_id(self, document_position: int) -> object:
        """Return the stable external ID associated with a document position."""

        position = self.validate_document_position(document_position)
        return self.document_ids[position]

    def document_position(self, document_id: object) -> int:
        """Resolve an external document ID without conflating different types."""

        key = _typed_value_key(document_id)
        try:
            return int(self._document_id_to_position[key])
        except KeyError as exc:
            raise KeyError(f"unknown document ID: {document_id!r}") from exc

    def topic_histogram(
        self,
        document_positions: Iterable[int],
        *,
        soft: bool = True,
        weights: Iterable[float] | None = None,
        deduplicate: bool = False,
    ) -> np.ndarray:
        """把一个已完成窗口编码成归一化 topic 直方图。

        ``weights`` supplies one finite, non-negative observation weight per
        position. With ``deduplicate=True``, only the first occurrence (and its
        corresponding weight) of each document position is retained. Defaults
        preserve the original occurrence-frequency semantics.
        """

        raw_positions = list(document_positions)
        if weights is None:
            raw_weights = [1.0] * len(raw_positions)
        else:
            raw_weights = [float(weight) for weight in weights]
            if len(raw_weights) != len(raw_positions):
                raise ValueError(
                    "weights must contain exactly one value per document position"
                )

        histogram = np.zeros(self.n_topics, dtype=np.float64)
        seen: set[int] = set()
        for raw_position, observation_weight in zip(raw_positions, raw_weights):
            position = self.validate_document_position(raw_position)
            if not math.isfinite(observation_weight) or observation_weight < 0.0:
                raise ValueError("topic histogram weights must be finite and non-negative")
            if deduplicate and position in seen:
                continue
            seen.add(position)
            if observation_weight == 0.0:
                continue
            if soft:
                for topic, membership_weight in self._memberships[position]:
                    histogram[topic] += observation_weight * membership_weight
            else:
                histogram[int(self.primary_topics[position])] += observation_weight
        mass = float(histogram.sum())
        return histogram if mass <= 0.0 else histogram / mass

    def validate_document_position(self, position: int) -> int:
        position = int(position)
        if not 0 <= position < self.n_documents:
            raise IndexError(f"document position {position} is out of range")
        return position

    def _validate_topic(self, topic: int) -> int:
        topic = int(topic)
        if not 0 <= topic < self.n_topics:
            raise IndexError(f"topic {topic} is out of range")
        return topic

    def _initialize_document_ids(
        self, document_ids: Sequence[object] | None
    ) -> None:
        explicit = document_ids is not None
        ids = list(range(self.n_documents)) if document_ids is None else list(document_ids)
        if len(ids) != self.n_documents:
            raise ValueError("document_ids must contain exactly one ID per document")
        keys = tuple(_typed_value_key(document_id) for document_id in ids)
        if len(set(keys)) != len(keys):
            raise ValueError("document_ids must be unique under typed equality")
        self.document_ids = tuple(ids)
        self._document_id_keys = keys
        self._document_ids_explicit = explicit
        self._document_id_to_position = MappingProxyType({
            key: position for position, key in enumerate(keys)
        })

    def _build_buckets(self) -> None:
        hard: list[list[int]] = [[] for _ in range(self.n_topics)]
        soft: list[list[tuple[int, float]]] = [
            [] for _ in range(self.n_topics)
        ]
        for position, memberships in enumerate(self._memberships):
            hard[int(self.primary_topics[position])].append(position)
            for topic, weight in memberships:
                soft[topic].append((position, float(weight)))
        self._hard_buckets = tuple(tuple(bucket) for bucket in hard)
        self._soft_buckets = tuple(
            tuple(sorted(bucket, key=lambda item: (-item[1], item[0])))
            for bucket in soft
        )

    def _finalize_fingerprints(
        self,
        *,
        partition_kind: str,
        corpus_content: object,
        configuration: Mapping[str, object],
    ) -> None:
        self.partition_kind = str(partition_kind)
        self._summary_details = dict(configuration)
        self.corpus_fingerprint = _fingerprint(
            "topic-corpus-v2",
            self._document_id_keys,
            corpus_content,
        )

        digest = hashlib.sha256()
        for value in (
            PARTITION_VERSION,
            self.partition_kind,
            self.corpus_fingerprint,
            dict(configuration),
            self.primary_topics,
        ):
            _update_digest(digest, value)
        for memberships in self._memberships:
            digest.update(struct.pack("<I", len(memberships)))
            for topic, weight in memberships:
                digest.update(struct.pack("<Id", int(topic), float(weight)))
        self.partition_fingerprint = digest.hexdigest()

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable structural summary of this partition."""

        sizes = sorted(len(bucket) for bucket in self._hard_buckets)
        p95_index = max(0, int(math.ceil(0.95 * len(sizes))) - 1)
        summary: dict[str, object] = {
            "version": PARTITION_VERSION,
            "kind": self.partition_kind,
            "n_documents": int(self.n_documents),
            "n_topics": int(self.n_topics),
            "document_ids_explicit": bool(self._document_ids_explicit),
            "corpus_fingerprint": self.corpus_fingerprint,
            "partition_fingerprint": self.partition_fingerprint,
            "hard_bucket_min": int(sizes[0]),
            "hard_bucket_mean": float(sum(sizes) / len(sizes)),
            "hard_bucket_p95": int(sizes[p95_index]),
            "hard_bucket_max": int(sizes[-1]),
            "empty_topics": int(sum(size == 0 for size in sizes)),
        }
        summary.update(self._summary_details)
        return summary

    @property
    def partition_summary(self) -> dict[str, object]:
        """Property alias for callers that prefer data-style access."""

        return self.summary()


class MetadataTopicPartition(TopicPartition):
    """根据稳定的文档元数据字段，对冷库进行 one-hot 分区。

    Topic IDs are assigned by a stable type-sensitive ordering, not by corpus
    encounter order. Consequently, ``True`` and ``1`` are distinct labels and
    reordering the same labeled corpus does not rename topics.
    """

    def __init__(
        self,
        document_labels: Sequence[object],
        *,
        document_ids: Sequence[object] | None = None,
    ) -> None:
        labels = list(document_labels)
        if not labels:
            raise ValueError("document_labels must contain at least one label")
        self.n_documents = len(labels)
        self.labels = tuple(labels)
        self._initialize_document_ids(document_ids)

        label_keys = tuple(_typed_value_key(label) for label in labels)
        representative: dict[tuple[str, object], object] = {}
        for key, label in zip(label_keys, labels):
            representative.setdefault(key, label)
        ordered_keys = tuple(sorted(representative, key=_canonical_json_bytes))
        typed_label_to_topic = {
            key: topic for topic, key in enumerate(ordered_keys)
        }
        self.topic_labels = tuple(representative[key] for key in ordered_keys)
        self.n_topics = len(self.topic_labels)
        self.typed_label_to_topic = MappingProxyType(typed_label_to_topic)
        self.label_to_topic = _TypedLabelMap(
            self.topic_labels, self.typed_label_to_topic
        )

        primary = np.asarray(
            [typed_label_to_topic[key] for key in label_keys], dtype=np.int64
        )
        self.primary_topics = _readonly_array(primary, dtype=np.int64)
        self._memberships = tuple(
            ((int(topic), 1.0),) for topic in self.primary_topics
        )
        self._build_buckets()
        self._finalize_fingerprints(
            partition_kind="metadata",
            corpus_content=label_keys,
            configuration={"label_order": "typed-canonical-v1"},
        )

    def topic_for_label(self, label: object) -> int:
        return int(self.typed_label_to_topic[_typed_value_key(label)])


class SemanticTopicPartition(TopicPartition):
    """对归一化 embedding 执行 MiniBatchKMeans topic 分区。

    KMeans learns the coarse centroids. By default, hard and soft assignments
    then use cosine similarity to normalized centroids. Set
    ``assignment_metric="euclidean"`` to reproduce the v1 Euclidean KMeans
    assignment. Each document saves the highest-weight ``top_m`` memberships.
    """

    def __init__(
        self,
        document_embeddings: np.ndarray,
        n_topics: int,
        *,
        top_m: int = 2,
        temperature: float = 0.10,
        random_state: int = 0,
        batch_size: int = 1024,
        max_iter: int = 100,
        assignment_metric: str = "cosine",
        document_ids: Sequence[object] | None = None,
    ) -> None:
        try:
            import sklearn
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "SemanticTopicPartition requires scikit-learn"
            ) from exc

        embeddings = _normalize_rows(
            document_embeddings,
            name="document_embeddings",
            reject_zero=True,
        )
        self.document_embeddings = _readonly_array(embeddings, dtype=np.float32)
        self.n_documents = len(self.document_embeddings)
        self._initialize_document_ids(document_ids)

        requested_topics = _positive_int("n_topics", n_topics)
        requested_top_m = _positive_int("top_m", top_m)
        self.n_topics = min(requested_topics, self.n_documents)
        self.top_m = min(requested_top_m, self.n_topics)
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.random_state = int(random_state)
        self.batch_size = _positive_int("batch_size", batch_size)
        effective_max_iter = _positive_int("max_iter", max_iter)
        metric = str(assignment_metric).strip().lower().replace("-", "_")
        if metric == "euclidean_kmeans":
            metric = "euclidean"
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(
                "assignment_metric must be 'cosine' or 'euclidean'"
            )
        self.assignment_metric = metric

        self.model = MiniBatchKMeans(
            n_clusters=self.n_topics,
            random_state=self.random_state,
            batch_size=max(self.n_topics, self.batch_size),
            max_iter=effective_max_iter,
            n_init=10,
            reassignment_ratio=0.0,
        )
        self.model.fit(self.document_embeddings)
        normalized_centers = _normalize_rows(
            self.model.cluster_centers_,
            name="cluster_centers",
            reject_zero=self.assignment_metric == "cosine",
        )
        self.centroids = _readonly_array(normalized_centers, dtype=np.float32)

        primary = np.empty(self.n_documents, dtype=np.int64)
        memberships: list[tuple[tuple[int, float], ...]] = []
        for start in range(0, self.n_documents, self.batch_size):
            stop = min(start + self.batch_size, self.n_documents)
            batch = self.document_embeddings[start:stop]
            if self.assignment_metric == "cosine":
                batch_affinities = np.asarray(
                    batch @ self.centroids.T, dtype=np.float64
                )
            else:
                distances = self.model.transform(batch)
                batch_affinities = -(distances.astype(np.float64) ** 2)

            for offset, affinity in enumerate(batch_affinities):
                topics = _stable_top_indices(affinity, self.top_m)
                primary[start + offset] = int(topics[0])
                logits = affinity[topics] / self.temperature
                logits -= float(logits.max())
                membership_weights = np.exp(logits)
                membership_weights /= float(membership_weights.sum())
                memberships.append(tuple(
                    (int(topic), float(weight))
                    for topic, weight in zip(topics, membership_weights)
                ))

        self.primary_topics = _readonly_array(primary, dtype=np.int64)
        self._memberships = tuple(memberships)
        self._build_buckets()
        self._finalize_fingerprints(
            partition_kind="semantic",
            corpus_content=self.document_embeddings,
            configuration={
                "assignment_metric": self.assignment_metric,
                "requested_topics": requested_topics,
                "effective_topics": self.n_topics,
                "requested_top_m": requested_top_m,
                "effective_top_m": self.top_m,
                "temperature": self.temperature,
                "random_state": self.random_state,
                "batch_size": self.batch_size,
                "max_iter": effective_max_iter,
                "sklearn_version": str(sklearn.__version__),
            },
        )

        # Public NumPy views are read-only. The sklearn estimator remains
        # available for compatibility, but its fitted arrays are protected too.
        for name in ("cluster_centers_", "labels_"):
            fitted = getattr(self.model, name, None)
            if isinstance(fitted, np.ndarray):
                fitted.setflags(write=False)


def build_topic_partition(
    kind: str | None = None,
    *,
    document_labels: Sequence[object] | None = None,
    document_embeddings: np.ndarray | None = None,
    n_topics: int | None = None,
    document_ids: Sequence[object] | None = None,
    **kwargs,
) -> TopicPartition:
    """Build exactly one metadata or semantic topic partition.

    ``kind`` may be omitted when exactly one of ``document_labels`` and
    ``document_embeddings`` is supplied. Additional keyword arguments are sent
    to the selected concrete constructor, so invalid cross-mode options fail
    loudly instead of being ignored.
    """

    normalized_kind = None if kind is None else str(kind).strip().lower()
    if normalized_kind in {"meta", "labels"}:
        normalized_kind = "metadata"
    if normalized_kind in {"embedding", "embeddings"}:
        normalized_kind = "semantic"
    if normalized_kind is not None and normalized_kind not in {
        "metadata", "semantic"
    }:
        raise ValueError("kind must be 'metadata' or 'semantic'")

    has_labels = document_labels is not None
    has_embeddings = document_embeddings is not None
    if normalized_kind is None:
        if has_labels == has_embeddings:
            raise ValueError(
                "provide exactly one of document_labels or document_embeddings"
            )
        normalized_kind = "metadata" if has_labels else "semantic"

    if normalized_kind == "metadata":
        if not has_labels or has_embeddings:
            raise ValueError(
                "metadata partitions require document_labels and no embeddings"
            )
        if n_topics is not None:
            raise ValueError("n_topics is inferred for metadata partitions")
        return MetadataTopicPartition(
            document_labels,
            document_ids=document_ids,
            **kwargs,
        )

    if not has_embeddings or has_labels:
        raise ValueError(
            "semantic partitions require document_embeddings and no labels"
        )
    if n_topics is None:
        raise ValueError("semantic partitions require n_topics")
    return SemanticTopicPartition(
        document_embeddings,
        n_topics,
        document_ids=document_ids,
        **kwargs,
    )
