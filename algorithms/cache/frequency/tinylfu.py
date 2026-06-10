"""Frequency policy: TinyLFU (ARC-paper admission paradigm)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")


class TinyLFU(BaseStrategy):
    """TinyLFU: frequency-based eviction with miss-driven admission.

    Admission follows the ARC paper's baseline paradigm: passive miss-driven
    fill. When a query is not served by the cache, escalate to the pool, fetch
    the query's top-1 doc, and admit it through a TinyLFU frequency gate.

    Frequency (LFU) semantics, per the ARC paper ("LFU evicts the least
    frequently used items"; counts only for docs *currently in cache*):
      - freq[d]: incremented when d serves a successful query while in cache.
      - eviction: lowest freq first (least-frequently-used).
      - TinyLFU admission gate: a fetched candidate replaces the LFU victim only
        if its (recent) fetch evidence >= the victim's in-cache frequency.
      - an evicted doc loses its freq (treated as a newcomer if re-fetched).

    In the multi-hop setting, bridge documents are never directly retrieved
    (the query omits the bridge entity), so they are never fetched on a miss and
    never enter the cache — TinyLFU cannot protect them and performs no better
    than Static. (Same blind spot as LRU; frequency vs recency only changes the
    eviction order, not the admission reach.)
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.freq = {}         # pool_idx -> in-cache access count
        self.fetch_freq = {}   # pool_idx -> times fetched as top-1 on a miss

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.freq[self.d2p[did]] = 0

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

        # hits increment in-cache frequency
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.freq[pi] = self.freq.get(pi, 0) + 1

        # misses escalate to pool, admit top-1 through the LFU gate
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
                self.freq[top1] = self.freq.get(top1, 0) + 1
                continue
            victim = min(self.kb, key=lambda d: self.freq.get(self.d2p[d], 0))
            vp = self.d2p[victim]
            # TinyLFU gate: admit only if candidate evidence >= victim frequency
            if self.fetch_freq.get(top1, 0) < self.freq.get(vp, 0):
                continue
            self.kb.discard(victim)
            self.freq.pop(vp, None)        # evicted -> count dropped
            self.kb.add(cand_did)
            self.freq[top1] = 0
            n += 1
        self.update_cost += n
