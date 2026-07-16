"""GPTCache-style semantic query/result cache adapted to evidence documents.

Each resident document stores the embedding of the most recent query that
produced it. A later semantically similar query reuses that result. On a miss,
the request's post-service access/citation key is preferred when available;
ordinary QA falls back to full-pool Top-1 retrieval. Replacement is LRU, the
common GPTCache eviction choice. This is a document-result adapter, not a claim
that GPTCache itself manages RAG evidence residency.
"""

import numpy as np

from algorithms.cache.base import BaseStrategy
from algorithms.cache.params import PARAMS as _P


class GPTCacheStyle(BaseStrategy):
    TAU = 0.85

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.query_keys = {}
        self.last_used = {}
        self._tick = 0

    def set_kb(self, ids):
        super().set_kb(ids)
        for doc_id in self.kb:
            self.last_used[doc_id] = -1

    def _semantic_match(self, query):
        candidates = [doc_id for doc_id in self.kb if doc_id in self.query_keys]
        if not candidates:
            return None
        scores = np.asarray([
            float(self.query_keys[doc_id] @ query) for doc_id in candidates
        ])
        best = int(np.argmax(scores))
        return candidates[best] if float(scores[best]) >= self.TAU else None

    def step(self, window_queries, window_query_embs, window_idx):
        del window_idx
        if not self.kb:
            return
        capacity = len(self.kb)
        queries = np.asarray(window_query_embs, dtype=np.float32)
        queries /= np.clip(np.linalg.norm(queries, axis=1, keepdims=True), 1e-10, None)
        accesses = self._observed_access_positions(window_queries)
        writes = 0

        for row, query in enumerate(queries):
            self._tick += 1
            if accesses is None:
                match = self._semantic_match(query)
                if match is not None:
                    self.last_used[match] = self._tick
                    continue
                scores = self.doc_embs @ query
                target = self.p2d[int(np.argmax(scores))]
                self.maint_retrieval_cost += 1
            else:
                target = self.p2d[int(accesses[row])]
                if target not in self.kb:
                    self.maint_retrieval_cost += 1

            if target in self.kb:
                self.query_keys[target] = query.copy()
                self.last_used[target] = self._tick
                continue
            if writes >= int(_P.WRITE_CAP):
                continue
            victim = min(
                self.kb,
                key=lambda doc_id: (
                    self.last_used.get(doc_id, -1), self.d2p[doc_id]
                ),
            )
            self.kb.remove(victim)
            self.query_keys.pop(victim, None)
            self.last_used.pop(victim, None)
            self.kb.add(target)
            self.query_keys[target] = query.copy()
            self.last_used[target] = self._tick
            writes += 1

        assert len(self.kb) == capacity
        self.update_cost += writes
