"""Focused tests for the classical four-list ARC baseline."""

import numpy as np

from algorithms.cache.adaptive.arc import ClassicalARC
from algorithms.cache.params import PARAMS


def _fixture(capacity=2):
    pool = [
        {"doc_id": name, "title": name}
        for name in ("a", "b", "c", "d", "e")
    ]
    embeddings = np.eye(5, dtype=np.float32)
    title_to_idx = {doc["title"]: index for index, doc in enumerate(pool)}
    policy = ClassicalARC("ClassicalARC", pool, embeddings, title_to_idx)
    policy.set_kb({doc["doc_id"] for doc in pool[:capacity]})
    return policy, pool, embeddings


def _events(names):
    return [{"access_title": name} for name in names]


def test_arc_ghost_hits_adapt_recency_target():
    policy, _, embeddings = _fixture()
    old_cap = PARAMS.WRITE_CAP
    PARAMS.WRITE_CAP = 10
    try:
        # Promote a to T2, then c replaces the remaining T1 item into B1.
        policy.step(_events(["a", "c"]), embeddings[[0, 2]], 0)
        assert set(policy.t2) == {"a"}
        assert set(policy.t1) == {"c"}
        assert set(policy.b1) == {"b"}

        # B1 hit raises p and evicts from T2 into B2.
        policy.step(_events(["b"]), embeddings[[1]], 1)
        assert policy.target_size == 1.0
        assert policy.kb == {"b", "c"}
        assert set(policy.b2) == {"a"}

        # B2 hit lowers p and evicts from T1 into B1.
        policy.step(_events(["a"]), embeddings[[0]], 2)
        assert policy.target_size == 0.0
        assert policy.kb == {"a", "b"}
        assert set(policy.b1) == {"c"}
    finally:
        PARAMS.WRITE_CAP = old_cap


def test_arc_respects_one_shared_write_per_window():
    policy, _, embeddings = _fixture()
    old_cap = PARAMS.WRITE_CAP
    PARAMS.WRITE_CAP = 1
    try:
        policy.step(_events(["c", "d", "e"]), embeddings[[2, 3, 4]], 0)
        assert policy.update_cost == 1
        assert policy.maint_retrieval_cost == 3
        assert len(policy.kb) == 2
        assert "c" in policy.kb
        assert "d" not in policy.kb and "e" not in policy.kb
    finally:
        PARAMS.WRITE_CAP = old_cap


def test_arc_resident_hits_need_no_reads_or_writes():
    policy, _, embeddings = _fixture()
    old_cap = PARAMS.WRITE_CAP
    PARAMS.WRITE_CAP = 0
    try:
        policy.step(_events(["a", "a"]), embeddings[[0, 0]], 0)
        assert policy.update_cost == 0
        assert policy.maint_retrieval_cost == 0
        assert set(policy.t2) == {"a"}
        assert policy.kb == {"a", "b"}
    finally:
        PARAMS.WRITE_CAP = old_cap


def test_arc_new_stream_never_exceeds_cache_or_ghost_bounds():
    policy, _, embeddings = _fixture()
    old_cap = PARAMS.WRITE_CAP
    PARAMS.WRITE_CAP = 20
    try:
        sequence = ["a", "c", "b", "d", "a", "e", "c", "b", "d"]
        positions = [["a", "b", "c", "d", "e"].index(name) for name in sequence]
        policy.step(_events(sequence), embeddings[positions], 0)
        assert len(policy.kb) == 2
        assert len(policy.t1) + len(policy.t2) == 2
        assert len(policy.t1) + len(policy.t2) + len(policy.b1) + len(policy.b2) <= 4
    finally:
        PARAMS.WRITE_CAP = old_cap
