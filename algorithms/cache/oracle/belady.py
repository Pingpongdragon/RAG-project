"""Oracle / Belady upper bound (non-causal)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")

class Oracle(BaseStrategy):
    """Oracle (per-window upper bound).

    At every window, the KB is reconstructed from the gold supporting-fact
    documents of THAT window's queries; remaining capacity is filled with
    documents most similar to that window's queries. This is the tightest
    achievable upper bound on Recall@K (any retriever cannot exceed it
    given the KB capacity), shown as a *constant* upper envelope before
    AND after drift.

    Note: this is strictly a non-causal reference; it consumes ground-truth
    SF labels not available at deployment.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.t2i = title_to_idx

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        budget = len(self.kb)
        # gold SF pool indices from current window
        sf_pool = set()
        for q in window_queries:
            for t in q.get('sf_titles', []):
                if t in self.t2i:
                    sf_pool.add(self.t2i[t])
        # fill remainder by similarity to this window's queries
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        norm_qe = window_query_embs / np.clip(norms, 1e-10, None)
        doc_scores = np.mean(norm_qe @ self.doc_embs.T, axis=0)
        new_kb = set()
        for pi in sorted(sf_pool, key=lambda i: -doc_scores[i])[:budget]:
            new_kb.add(self.p2d[pi])
        if len(new_kb) < budget:
            sorted_docs = np.argsort(-doc_scores)
            for pi in sorted_docs:
                if len(new_kb) >= budget:
                    break
                if pi in sf_pool:
                    continue
                new_kb.add(self.p2d[int(pi)])
        added = len(new_kb - self.kb)
        self.update_cost += added
        self.kb = new_kb



# ── Factory registry ──────────────────────────────


