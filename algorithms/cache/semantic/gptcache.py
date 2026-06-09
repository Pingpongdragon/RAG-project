"""Semantic response-cache style policy."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import _ArrivalCacheBase

class GPTCacheStyle(_ArrivalCacheBase):
    """Semantic cache (GPTCache-style).

    Admission source: random arrivals with semantic-dedup gate.
      - skip arrival if sim to existing KB > DEDUP_HIGH (already covered)
      - skip arrival if sim to existing KB < DEDUP_LOW  (off-topic noise)
    Eviction: lowest decayed cache score (max sim seen from queries lately).
    """
    DEDUP_HIGH = 0.85
    DEDUP_LOW  = 0.30
    DECAY      = 0.80

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.cache_score = {}

    def _on_init_kb(self, did, pi):
        self.cache_score[pi] = 1.0
    def _on_query_seen(self, kb_pi, sim, window_idx):
        # decayed running max
        prev = self.cache_score.get(kb_pi, 0.0) * self.DECAY
        self.cache_score[kb_pi] = max(prev, sim)
    def _evict_score(self, pi, window_idx):
        return self.cache_score.get(pi, 0.0)
    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        if sim_to_kb_max > self.DEDUP_HIGH:
            return None
        if sim_to_kb_max < self.DEDUP_LOW:
            return None
        # admit with score = relevance to current KB topic distribution
        return sim_to_kb_max


