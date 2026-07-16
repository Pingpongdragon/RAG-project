"""Causality and scoring tests for current-request downstream feedback."""

import numpy as np

from algorithms.drip import DRIP, DRIPConfig
from core.downstream_feedback import CurrentRequestUtilityEstimator


def _policy():
    documents = [
        {"doc_id": "a", "title": "A", "text": "alpha"},
        {"doc_id": "b", "title": "B", "text": "beta"},
        {"doc_id": "c", "title": "C", "text": "gamma"},
    ]
    embeddings = np.eye(3, dtype=np.float32)
    policy = DRIP(
        "DRIP",
        documents,
        embeddings,
        {document["title"]: index for index, document in enumerate(documents)},
        DRIPConfig.reactive(downstream_feedback_mass=1.0),
    )
    policy.set_kb({"a", "b"})
    return policy


def test_estimator_uses_current_answer_alignment_and_normalizes():
    estimator = CurrentRequestUtilityEstimator(
        query_weight=0.0, answer_weight=1.0, citation_bonus=0.0, topk=2
    )
    utilities = estimator.score(
        query_embedding=np.asarray([1.0, 0.0, 0.0]),
        answer_embedding=np.asarray([0.0, 1.0, 0.0]),
        candidate_positions=[0, 1],
        document_embeddings=np.eye(3, dtype=np.float32),
    )

    assert utilities[0].position == 1
    assert np.isclose(sum(item.utility for item in utilities), 1.0)


def test_feedback_is_ignored_when_not_explicitly_enabled():
    policy = _policy()
    disabled = DRIP(
        "DRIP",
        policy.doc_pool,
        policy.doc_embs,
        policy.title_to_idx,
        DRIPConfig.reactive(downstream_feedback_mass=0.0),
    )
    updates = disabled._credit_downstream_feedback(
        {"downstream_feedback": [{"title": "C", "utility": 1.0}]},
        {0, 1},
    )

    assert updates == (0, 0.0, None, False)
    assert disabled.demand == {}


def test_post_service_payload_credits_only_named_current_candidates():
    policy = _policy()
    updates, mass, top, pressure = policy._credit_downstream_feedback(
        {
            "downstream_feedback": [
                {"title": "C", "utility": 0.8, "source": "answer"},
                {"title": "unknown", "utility": 1.0},
                {"title": "A", "utility": 0.2, "source": "answer"},
            ]
        },
        {0, 1},
    )

    assert updates == 2
    assert mass == 1.0
    assert top == 2
    assert pressure is True
    assert np.isclose(policy.demand[2], 0.8)
    assert np.isclose(policy.serve[0], 0.25 + 0.2)
