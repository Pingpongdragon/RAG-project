"""Time-aware recency policies (timestamp/freshness baselines).

Self-contained: no injectable params, only the shared _ArrivalCacheBase.
"""
import numpy as np
from algorithms.cache.base import _ArrivalCacheBase


class TemporalAware(_ArrivalCacheBase):
    """Time-aware hot-tier cache (Temporal RAG / freshness-decay style).

    Models the canonical timestamp-aware baseline: assumes document
    publication year is available, and prefers documents whose year is
    closest to the current query era. Admission and eviction both score
    documents by an exponential freshness decay around the per-window
    era anchor estimated from incoming query years.
    """
    TAU         = 2.0   # years
    NEUTRAL_GAP = 4.0   # treat unknown-year docs as moderately stale

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.doc_year = {i: doc_pool[i].get('year') for i in range(len(doc_pool))}
        self.current_era = None

    def step(self, window_queries, window_query_embs, window_idx):
        ys = [q.get('year') for q in window_queries if q.get('year') is not None]
        if ys:
            self.current_era = float(np.mean(ys))
        super().step(window_queries, window_query_embs, window_idx)

    def _gap(self, pi):
        y = self.doc_year.get(pi)
        if y is None or self.current_era is None:
            return self.NEUTRAL_GAP
        return abs(float(y) - self.current_era)

    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        return float(np.exp(-self._gap(pi) / self.TAU))

    def _evict_score(self, pi, window_idx):
        return float(np.exp(-self._gap(pi) / self.TAU))




class RecencyTTL(_ArrivalCacheBase):
    """Oracle-timestamp TTL cache.

    Demonstrates that *even with* perfect document publication timestamps,
    the timestamp signal cannot disambiguate within an era: when many
    candidates share the same year (e.g., thousands of 2015 articles
    in one StreamingQA window), TTL has no resolution to pick K out
    of them. Admission: doc.year within +/-YEAR_WINDOW of current
    query era. Eviction: oldest publication year first (classic TTL).
    """
    YEAR_WINDOW = 2.0   # admit if |doc.year - era| <= window
    NEUTRAL_YEAR = -1.0 # unknown-year docs treated as oldest

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.doc_year = {i: doc_pool[i].get('year') for i in range(len(doc_pool))}
        self.current_era = None

    def step(self, window_queries, window_query_embs, window_idx):
        ys = [q.get('year') for q in window_queries if q.get('year') is not None]
        if ys:
            self.current_era = float(np.mean(ys))
        super().step(window_queries, window_query_embs, window_idx)

    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        y = self.doc_year.get(pi)
        if y is None or self.current_era is None:
            return None
        if abs(float(y) - self.current_era) > self.YEAR_WINDOW:
            return None
        # within window: prefer closest-to-era first
        return -abs(float(y) - self.current_era)

    def _evict_score(self, pi, window_idx):
        # lower = more evictable; oldest year first
        y = self.doc_year.get(pi)
        if y is None:
            return self.NEUTRAL_YEAR
        return float(y)
