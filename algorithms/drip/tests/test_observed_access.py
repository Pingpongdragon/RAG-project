"""Item-access workload 只能使用服务后可观察的 access key。"""

import numpy as np

from algorithms.cache.frequency.tinylfu import TinyLFU
from algorithms.cache.recency.fifo import FIFO
from algorithms.cache.recency.lru import LRU
from algorithms.drip import DRIP


def _inputs():
    documents = [
        {"doc_id": "a", "title": "A", "text": "alpha"},
        {"doc_id": "b", "title": "B", "text": "beta"},
        {"doc_id": "c", "title": "C", "text": "gamma"},
    ]
    embeddings = np.eye(3, dtype=np.float32)
    return documents, embeddings, {"A": 0, "B": 1, "C": 2}


def _request_c():
    return [{
        "question": "gamma",
        "access_title": "C",
        "qtype": "observed_item_access",
    }]


def test_classic_policies_admit_the_observed_miss_key():
    for policy_class in (LRU, FIFO, TinyLFU):
        policy = policy_class(policy_class.__name__, *_inputs())
        policy.set_kb({"a", "b"})

        policy.step(_request_c(), np.asarray([[0.0, 0.0, 1.0]]), 0)

        assert "c" in policy.kb
        assert policy.update_cost == 1


def test_observed_hit_does_not_write():
    policy = DRIP("DRIP", *_inputs())
    policy.set_kb({"a", "c"})

    policy.step(_request_c(), np.asarray([[0.0, 0.0, 1.0]]), 0)

    assert policy.update_cost == 0
    assert policy.kb == {"a", "c"}
    assert policy.last_admission["under_covered"] == 0


def test_drip_uses_observed_miss_without_reading_gold_support():
    policy = DRIP("DRIP", *_inputs())
    policy.set_kb({"a", "b"})

    policy.step(_request_c(), np.asarray([[0.0, 0.0, 1.0]]), 0)

    assert "c" in policy.kb
    assert policy.update_cost == 1
    assert policy.demand == {2: 1.0}
    assert policy.maint_retrieval_cost == 1
    assert policy.last_admission["evidence_mass"] == 1.0
