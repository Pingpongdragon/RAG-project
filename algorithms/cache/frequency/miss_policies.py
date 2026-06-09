"""Miss-driven, semantic-signal recency/frequency policies."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")

class MissLRU(BaseStrategy):
    """Miss-driven LRU: textbook cache.

    Per window, for each query: if best KB hit < _P.SF_HIT_THRESH (miss),
    fetch the pool's top-1 doc for that query embedding and admit it,
    evicting the KB doc with smallest last_hit. Successful queries refresh
    last_hit on the best-hit KB doc. No admission threshold, no aggregation.
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
        # refresh last_hit on successful queries
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self.last_hit[int(kb_idx[pos])] = window_idx
        # miss-driven admit
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
            # evict the LRU KB doc
            victim = min(self.kb, key=lambda d: self.last_hit.get(self.d2p[d], -1))
            vp = self.d2p[victim]
            self.kb.discard(victim)
            self.last_hit.pop(vp, None)
            self.kb.add(cand_did)
            self.last_hit[top1] = window_idx
            n += 1
        self.update_cost += n



class MissTinyLFU(BaseStrategy):
    """Miss-driven TinyLFU: textbook cache + frequency-sketch admission.

    Like MissLRU, but the admission gate uses access frequency:
      - access_freq[d]: incremented when d serves a successful query (in KB)
      - fetch_freq[d]: incremented each time d is fetched as top-1 for a
        failing query (across all windows, even when not admitted)
      - admit candidate c (replacing victim e) iff fetch_freq[c] >= access_freq[e]
      - eviction order: lowest access_freq first
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.access_freq = {}
        self.fetch_freq = {}

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.access_freq[self.d2p[did]] = 0

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
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self.access_freq[int(kb_idx[pos])] = self.access_freq.get(int(kb_idx[pos]), 0) + 1
        fail = ~succ
        if not fail.any():
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        self.maint_retrieval_cost += int(fail.sum())
        n = 0
        for qi in range(pool_sims.shape[0]):
            top1 = int(np.argmax(pool_sims[qi]))
            self.fetch_freq[top1] = self.fetch_freq.get(top1, 0) + 1
            cand_did = self.p2d[top1]
            if cand_did in self.kb:
                self.access_freq[top1] = self.access_freq.get(top1, 0) + 1
                continue
            victim = min(self.kb, key=lambda d: self.access_freq.get(self.d2p[d], 0))
            vp = self.d2p[victim]
            cand_f = self.fetch_freq.get(top1, 0)
            vict_f = self.access_freq.get(vp, 0)
            if cand_f < vict_f:
                continue  # TinyLFU gate rejects
            self.kb.discard(victim)
            self.access_freq.pop(vp, None)
            self.kb.add(cand_did)
            self.access_freq[top1] = 0
            n += 1
        self.update_cost += n


