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
        self.tie_rank = {}
        self._next_tie_rank = 0

    def set_kb(self, ids):
        super().set_kb(ids)
        # Batch-loaded residents have no real arrival order. Stabilize the input,
        # then use a seeded permutation for equal-recency ties so loader order is
        # not an implicit prior.
        stable_ids = sorted(self.kb, key=self.d2p.__getitem__)
        rng = np.random.default_rng(_P.SEED + 701)
        init_order = [stable_ids[int(i)] for i in rng.permutation(len(stable_ids))]
        for rank, did in enumerate(init_order):
            self.last_hit[self.d2p[did]] = -1
            self.tie_rank[self.d2p[did]] = rank
        self._next_tie_rank = len(init_order)

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        accesses = self._observed_access_positions(window_queries)
        if accesses is not None:
            return self._step_observed_accesses(accesses, window_idx)
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
            if n >= int(_P.WRITE_CAP):
                continue
            victim = min(
                self.kb,
                key=lambda d: (
                    self.last_hit.get(self.d2p[d], -1),
                    self.tie_rank.get(self.d2p[d], -1),
                ),
            )
            vp = self.d2p[victim]
            self.kb.discard(victim)
            self.last_hit.pop(vp, None)   # evicted -> count dropped
            self.tie_rank.pop(vp, None)
            self.kb.add(cand_did)
            self.last_hit[top1] = window_idx
            self.tie_rank[top1] = self._next_tie_rank
            self._next_tie_rank += 1
            n += 1
        self.update_cost += n

    def _step_observed_accesses(self, accesses, window_idx):
        """经典 key-value LRU：精确 hit，miss 后写入实际访问对象。"""

        n = 0
        for offset, pool_idx in enumerate(accesses):
            tick = int(window_idx) * max(1, len(accesses)) + int(offset)
            candidate = self.p2d[int(pool_idx)]
            if candidate in self.kb:
                self.last_hit[int(pool_idx)] = tick
                continue
            self.maint_retrieval_cost += 1
            if n >= int(_P.WRITE_CAP):
                continue
            victim = min(
                self.kb,
                key=lambda doc_id: (
                    self.last_hit.get(self.d2p[doc_id], -1),
                    self.tie_rank.get(self.d2p[doc_id], -1),
                ),
            )
            victim_pos = self.d2p[victim]
            self.kb.discard(victim)
            self.last_hit.pop(victim_pos, None)
            self.tie_rank.pop(victim_pos, None)
            self.kb.add(candidate)
            self.last_hit[int(pool_idx)] = tick
            self.tie_rank[int(pool_idx)] = self._next_tie_rank
            self._next_tie_rank += 1
            n += 1
        self.update_cost += n
