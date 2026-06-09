"""Recency / access-history policies."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import _ArrivalCacheBase

class LRU(_ArrivalCacheBase):
    """Pure least-recently-used cache.

    Admission source: random doc-arrival stream (no failure probing).
    Eviction: oldest last-touch wins (lowest last_hit window).
    Passive recency: any KB doc that gets touched by a query updates last_hit.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.last_hit = {}

    def _on_init_kb(self, did, pi):
        self.last_hit[pi] = -1
    def _on_query_hit(self, kb_pi, window_idx):
        self.last_hit[kb_pi] = window_idx
    def _evict_score(self, pi, window_idx):
        return self.last_hit.get(pi, -1)


