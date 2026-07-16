"""唯一公开 DRIP 入口及 evidence-only 决策的最小契约测试。"""

import numpy as np

from algorithms.cache.registry import STRATEGY_FACTORIES
from algorithms.drip import DRIP, DRIPConfig
from algorithms.drip.controller import PrimalDualController


def _inputs():
    documents = [
        {"doc_id": "a", "title": "A", "text": "alpha"},
        {"doc_id": "b", "title": "B", "text": "beta"},
        {"doc_id": "c", "title": "C", "text": "gamma"},
        {"doc_id": "d", "title": "D", "text": "delta"},
    ]
    embeddings = np.eye(4, dtype=np.float32)
    title_to_idx = {document["title"]: index
                    for index, document in enumerate(documents)}
    return documents, embeddings, title_to_idx


def test_only_one_drip_policy_is_public():
    assert set(STRATEGY_FACTORIES) == {
        "LRU", "FIFO", "TinyLFU", "ClassicalARC", "GPTCacheStyle", "Proximity",
        "AgentRAGCache", "DRIP", "Oracle",
    }
    policy = STRATEGY_FACTORIES["DRIP"](*_inputs())
    assert isinstance(policy, DRIP)
    assert isinstance(policy.config, DRIPConfig)
    assert policy.config.topic_dynamics_enabled is False


def test_topic_dynamics_factory_requires_an_explicit_cold_partition():
    config = DRIPConfig.topic_dynamics(
        candidate_budget=4,
        metadata_field="topic",
    )
    assert config.topic_dynamics_enabled is True
    assert config.topic_metadata_field == "topic"

    try:
        DRIPConfig.topic_dynamics(candidate_budget=4)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:  # pragma: no cover - contract failure
        raise AssertionError("TopicDynamics accepted an unspecified partition")


def test_each_miss_contributes_one_unit_of_mef_mass():
    policy = DRIP("DRIP", *_inputs())
    updates, mass = policy._credit_miss(
        [(2, 0.90), (3, 0.75)], kb_pos={0, 1}
    )

    assert updates == 2
    assert mass == 1.0
    assert np.isclose(sum(policy.demand.values()), 1.0)
    assert policy.demand[2] > policy.demand[3]


def test_drip_utility_is_evidence_only_and_has_no_lru_expert():
    policy = DRIP("DRIP", *_inputs())
    policy.serve = {0: 0.25, 1: 0.50}
    policy.demand = {2: 1.0}

    utility = policy._node_utility({0, 1}, {2})

    assert not hasattr(policy, "experts")
    assert not hasattr(policy, "last_seen")
    assert utility[2] == 1.0
    assert utility[2] > utility[1] > utility[0]


def test_dual_price_tracks_replacement_budget_violation():
    controller = PrimalDualController(target_rate=0.25, initial_price=0.25)
    overloaded = controller.update(writes=4, write_budget=4)
    underloaded = controller.update(writes=0, write_budget=4)

    assert overloaded.price_after > overloaded.price_before
    assert underloaded.price_after < underloaded.price_before
