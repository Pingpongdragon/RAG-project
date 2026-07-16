"""Proximity query--document cache with FIFO association replacement.

The cache reuses the document associated with the nearest historical query
when cosine similarity exceeds ``TAU``. A miss records the current query and
the document actually fetched/cited after service. Ordinary QA without an
access key uses full-pool Top-1 retrieval. Associations and resident documents
are evicted in FIFO order.
"""

import numpy as np

from algorithms.cache.base import BaseStrategy
from algorithms.cache.params import PARAMS as _P


class Proximity(BaseStrategy):
    TAU = 0.80

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self._entries = []

    def _match(self, query):
        active = [entry for entry in self._entries if entry[1] in self.kb]
        if not active:
            return None
        scores = np.asarray([float(embedding @ query) for embedding, _ in active])
        best = int(np.argmax(scores))
        return active[best][1] if float(scores[best]) >= self.TAU else None

    def _evict_fifo(self, protected):
        victim = None
        for _, doc_id in self._entries:
            if doc_id in self.kb and doc_id != protected:
                victim = doc_id
                break
        if victim is None:
            victim = min(
                (doc_id for doc_id in self.kb if doc_id != protected),
                key=self.d2p.__getitem__,
            )
        self.kb.remove(victim)
        self._entries = [entry for entry in self._entries if entry[1] != victim]

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
            if accesses is None:
                match = self._match(query)
                if match is not None:
                    continue
                target = self.p2d[int(np.argmax(self.doc_embs @ query))]
                self.maint_retrieval_cost += 1
            else:
                target = self.p2d[int(accesses[row])]
                if target not in self.kb:
                    self.maint_retrieval_cost += 1

            if target not in self.kb:
                if writes >= int(_P.WRITE_CAP):
                    continue
                self._evict_fifo(target)
                self.kb.add(target)
                writes += 1
            self._entries.append((query.copy(), target))
            if len(self._entries) > 4 * capacity:
                self._entries.pop(0)

        assert len(self.kb) == capacity
        self.update_cost += writes
