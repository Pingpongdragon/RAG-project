"""Contracts for cold-corpus partitions and explicit soft topic dynamics."""

import numpy as np
import pytest

from algorithms.drip.topic_dynamics import (
    ExplicitTopicDriftDetector,
    SoftTopicDynamics,
)
from algorithms.drip.topic_partition import (
    MetadataTopicPartition,
    SemanticTopicPartition,
    build_topic_partition,
)


def test_metadata_partition_is_stable_typed_and_weighted():
    partition = MetadataTopicPartition(
        ["a", "a", "b", "c", "b"],
        document_ids=["d0", "d1", "d2", "d3", "d4"],
    )

    assert partition.n_topics == 3
    assert partition.topic_for_label("b") == 1
    assert partition.hard_bucket(1) == (2, 4)
    assert partition.soft_bucket(1) == ((2, 1.0), (4, 1.0))
    assert partition.document_position("d3") == 3
    assert partition.document_id(3) == "d3"
    assert np.allclose(
        partition.topic_histogram([0, 2, 2], weights=[1.0, 1.0, 3.0]),
        [0.2, 0.8, 0.0],
    )
    assert np.allclose(
        partition.topic_histogram([0, 2, 2], deduplicate=True),
        [0.5, 0.5, 0.0],
    )
    assert len(partition.corpus_fingerprint) == 64
    assert len(partition.partition_fingerprint) == 64
    assert partition.summary()["kind"] == "metadata"

    typed = MetadataTopicPartition([True, 1, True, 1])
    assert typed.n_topics == 2
    assert typed.topic_for_label(True) != typed.topic_for_label(1)


def test_metadata_topic_ids_do_not_depend_on_document_order():
    first = MetadataTopicPartition(["sports", "finance", "science"])
    second = MetadataTopicPartition(["science", "sports", "finance"])

    for label in ("sports", "finance", "science"):
        assert first.topic_for_label(label) == second.topic_for_label(label)


def test_semantic_partition_is_cosine_soft_and_read_only():
    embeddings = np.asarray(
        [
            [1.0, 0.02],
            [0.9, 0.1],
            [0.02, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    first = build_topic_partition(
        "semantic",
        document_embeddings=embeddings,
        n_topics=3,
        top_m=2,
        random_state=7,
        batch_size=6,
        assignment_metric="cosine",
    )
    second = SemanticTopicPartition(
        embeddings * 4.0,
        3,
        top_m=2,
        random_state=7,
        batch_size=6,
        assignment_metric="cosine",
    )

    assert np.array_equal(first.primary_topics, second.primary_topics)
    assert all(
        first.primary_topic(position) == first.memberships(position)[0][0]
        for position in range(6)
    )
    assert all(
        np.isclose(sum(weight for _, weight in first.memberships(position)), 1.0)
        for position in range(6)
    )
    assert first.summary()["assignment_metric"] == "cosine"
    assert first.primary_topics.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        first.primary_topics[0] = 1


@pytest.mark.parametrize(
    "embeddings, message",
    [
        (np.asarray([[np.nan, 0.0], [1.0, 0.0]]), "NaN or Inf"),
        (np.asarray([[0.0, 0.0], [1.0, 0.0]]), "zero-norm"),
    ],
)
def test_semantic_partition_rejects_invalid_embeddings(embeddings, message):
    with pytest.raises(ValueError, match=message):
        SemanticTopicPartition(embeddings, 2)


def test_explicit_detector_reports_a_topic_shift():
    detector = ExplicitTopicDriftDetector(
        2, reference_rate=0.1, slack=0.01, threshold=0.1
    )

    assert detector.observe([1.0, 0.0]).alarm is False
    assert detector.observe([0.99, 0.01]).alarm is False
    shifted = detector.observe([0.0, 1.0])

    assert shifted.alarm is True
    assert shifted.drift_score > 0.1
    assert shifted.cusum >= 0.1


def test_soft_dynamics_recovers_recurrent_topic_documents_causally():
    partition = MetadataTopicPartition(["a", "a", "b", "b"])
    model = SoftTopicDynamics(
        partition,
        drift_reference_rate=0.1,
        drift_slack=0.0,
        drift_threshold=0.1,
        transition_decay=1.0,
        document_decay=1.0,
        min_transition_support=0.5,
        min_forecast_confidence=0.4,
    )

    a0 = model.observe_and_forecast([1.0, 0.0], [0, 1], 2)
    b0 = model.observe_and_forecast([0.0, 1.0], [2, 3], 2)
    assert a0.predicted_topic is None
    assert b0.predicted_topic is None

    a1 = model.observe_and_forecast([1.0, 0.0], [0, 1], 2)
    assert a1.drift_alarm is True
    assert a1.predicted_topic == partition.topic_for_label("b")
    assert set(a1.documents) == {2, 3}

    b1 = model.observe_and_forecast([0.0, 1.0], [2, 3], 2)
    assert b1.previous_forecast_similarity == pytest.approx(1.0)


def test_soft_dynamics_never_invents_unobserved_topic_documents():
    partition = MetadataTopicPartition(["a", "a", "b", "b", "b"])
    model = SoftTopicDynamics(
        partition,
        drift_slack=0.0,
        drift_threshold=0.1,
        transition_decay=1.0,
        document_decay=1.0,
        min_transition_support=0.5,
        min_forecast_confidence=0.4,
    )

    model.observe_and_forecast([1.0, 0.0], [0], 4)
    model.observe_and_forecast([0.0, 1.0], [2], 4)
    decision = model.observe_and_forecast([1.0, 0.0], [0], 4)

    assert set(decision.documents) <= {0, 2}
    assert 1 not in decision.documents
    assert 3 not in decision.documents
    assert 4 not in decision.documents
