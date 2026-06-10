"""Recency / access-history policies (ARC-paper admission paradigm).

Admission follows the ARC paper's baseline description: passive miss-driven
fill — when a query is not served by the cache, escalate to the external store
(L2 pool), fetch the query's top-1 doc, and admit it ("有缺必补", no predictive
admission filter). Eviction is the classic recency rule.

Counts/recency for an evicted doc are dropped: a doc kicked out of the cache is
treated as a fresh newcomer if it is fetched again (ARC paper: baselines only
track state for docs *currently in cache*).
"""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")


class LRU(BaseStrategy):
    """Least-recently-used cache, miss-driven admission (ARC-paper paradigm).

    Per window, for each query:
      - if best KB similarity >= SF_HIT_THRESH (hit): refresh last_hit on the
        best-matching KB doc.
      - else (miss): escalate to the pool, fetch the query's top-1 doc, and
        admit it, evicting the KB doc with the smallest last_hit (LRU).
    Evicted docs lose their last_hit (treated as new if re-fetched later).
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.last_hit = {}

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.last_hit[self.d2p[did]] = -1

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T
        max_s = np.max(q_kb, axis=1)

        # hits refresh recency
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self.last_hit[int(kb_idx[pos])] = window_idx

        # misses escalate to pool and admit top-1
        fail = ~succ
        if not fail.any():
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        self.maint_retrieval_cost += int(fail.sum())
        n = 0
        for qi in range(pool_sims.shape[0]):
            top1 = int(np.argmax(pool_sims[qi]))
            cand_did = self.p2d[top1]
            if cand_did in self.kb:
                self.last_hit[top1] = window_idx
                continue
            victim = min(self.kb, key=lambda d: self.last_hit.get(self.d2p[d], -1))
            vp = self.d2p[victim]
            self.kb.discard(victim)
            self.last_hit.pop(vp, None)   # evicted -> count dropped
            self.kb.add(cand_did)
            self.last_hit[top1] = window_idx
            n += 1
        self.update_cost += n
