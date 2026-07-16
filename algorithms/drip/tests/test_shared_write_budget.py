"""所有在线 replacement baseline 必须遵守同一个窗口写预算。"""

import numpy as np

from algorithms.cache.frequency.tinylfu import TinyLFU
from algorithms.cache.params import PARAMS
from algorithms.cache.paradigm_ref.agent_rag_cache import AgentRAGCache
from algorithms.cache.recency.fifo import FIFO
from algorithms.cache.recency.lru import LRU
from algorithms.cache.semantic.gptcache import GPTCacheStyle
from algorithms.cache.semantic.proximity import Proximity
from algorithms.drip import DRIP


def _inputs():
    documents = [
        {"doc_id": str(index), "title": str(index), "text": str(index)}
        for index in range(8)
    ]
    embeddings = np.eye(8, dtype=np.float32)
    title_to_idx = {str(index): index for index in range(8)}
    return documents, embeddings, title_to_idx


def _miss_window():
    queries = [{"question": str(index)} for index in range(2, 6)]
    embeddings = np.eye(8, dtype=np.float32)[2:6]
    return queries, embeddings


def test_all_online_writers_obey_shared_window_cap(monkeypatch):
    monkeypatch.setattr(PARAMS, "WRITE_CAP", 1)
    monkeypatch.setattr(PARAMS, "DOC_ADD_CAP", 8)
    monkeypatch.setattr(PARAMS, "DOC_ARRIVE", 8)
    monkeypatch.setattr(PARAMS, "SF_HIT_THRESH", 0.99)
    monkeypatch.setattr(GPTCacheStyle, "DEDUP_LOW", -1.0)
    monkeypatch.setattr(GPTCacheStyle, "DEDUP_HIGH", 2.0)
    queries, query_embeddings = _miss_window()

    for policy_class in (
        LRU,
        FIFO,
        TinyLFU,
        Proximity,
        GPTCacheStyle,
        AgentRAGCache,
        DRIP,
    ):
        policy = policy_class(policy_class.__name__, *_inputs())
        policy.set_kb({"0", "1"})
        policy.step(queries, query_embeddings, 0)

        assert policy.update_cost <= 1, policy_class.__name__
        assert len(policy.kb) == 2, policy_class.__name__
