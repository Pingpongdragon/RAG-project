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
        self.tie_rank = {}
        self._next_tie_rank = 0

    def set_kb(self, ids):
        super().set_kb(ids)
        stable_ids = sorted(self.kb, key=self.d2p.__getitem__)
        rng = np.random.default_rng(_P.SEED + 703)
        init_order = [stable_ids[int(i)] for i in rng.permutation(len(stable_ids))]
        for rank, did in enumerate(init_order):
            self.freq[self.d2p[did]] = 0
            self.tie_rank[self.d2p[did]] = rank
        self._next_tie_rank = len(init_order)

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        accesses = self._observed_access_positions(window_queries)
        if accesses is not None:
            return self._step_observed_accesses(accesses)
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
            if n >= int(_P.WRITE_CAP):
                continue
            victim = min(
                self.kb,
                key=lambda d: (
                    self.freq.get(self.d2p[d], 0),
                    self.tie_rank.get(self.d2p[d], -1),
                ),
            )
            vp = self.d2p[victim]
            # TinyLFU gate: admit only if candidate evidence >= victim frequency
            if self.fetch_freq.get(top1, 0) < self.freq.get(vp, 0):
                continue
            self.kb.discard(victim)
            self.freq.pop(vp, None)        # evicted -> count dropped
            self.tie_rank.pop(vp, None)
            self.kb.add(cand_did)
            self.freq[top1] = 0
            self.tie_rank[top1] = self._next_tie_rank
            self._next_tie_rank += 1
            n += 1
        self.update_cost += n

    def _step_observed_accesses(self, accesses):
        n = 0
        for pool_idx in accesses:
            pool_idx = int(pool_idx)
            candidate = self.p2d[pool_idx]
            if candidate in self.kb:
                self.freq[pool_idx] = self.freq.get(pool_idx, 0) + 1
                continue
            self.maint_retrieval_cost += 1
            self.fetch_freq[pool_idx] = self.fetch_freq.get(pool_idx, 0) + 1
            if n >= int(_P.WRITE_CAP):
                continue
            victim = min(
                self.kb,
                key=lambda doc_id: (
                    self.freq.get(self.d2p[doc_id], 0),
                    self.tie_rank.get(self.d2p[doc_id], -1),
                ),
            )
            victim_pos = self.d2p[victim]
            if self.fetch_freq[pool_idx] < self.freq.get(victim_pos, 0):
                continue
            self.kb.discard(victim)
            self.freq.pop(victim_pos, None)
            self.tie_rank.pop(victim_pos, None)
            self.kb.add(candidate)
            self.freq[pool_idx] = 0
            self.tie_rank[pool_idx] = self._next_tie_rank
            self._next_tie_rank += 1
            n += 1
        self.update_cost += n
