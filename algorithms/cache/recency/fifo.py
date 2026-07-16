"""FIFO cache (ARC-paper admission paradigm).

Admission follows the ARC paper's baseline description: passive miss-driven
fill. When a query is not served by the cache, escalate to the external store
(L2 pool), fetch the query's top-1 doc, and admit it. Eviction is strict
first-in-first-out: when the cache is full, drop the oldest-inserted entry,
regardless of how often or how recently it was used.

ARC paper: "FIFO removes the oldest entries in the cache." The point of FIFO as
a baseline is that it is a blind porter — miss in, queue, drop the oldest when
full — ignoring the semantic locality and query bias that ARC exploits. A
heavily-used core document gets pushed out simply for being old, as a stream of
misses keeps shoving new docs onto the queue tail.

Note: this is the miss-driven FIFO matching the paper. The repo also has
`RandomFIFO` (random arrival-stream variant) kept only as a motivation-stage
paradigm reference; it is NOT this baseline.
"""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")


class FIFO(BaseStrategy):
    """First-in-first-out cache, miss-driven admission (ARC-paper paradigm).

    Per window, for each query:
      - hit (best KB sim >= SF_HIT_THRESH): no state change (FIFO ignores usage).
      - miss: escalate to the pool, fetch the query's top-1 doc, admit it, and
        if over budget evict the oldest-inserted KB entry.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.insert_order = []   # FIFO queue of doc_ids (oldest at head)

    def set_kb(self, ids):
        super().set_kb(ids)
        # The initial batch has no real arrival order; use a seeded permutation
        # instead of loader/pool order.
        stable_ids = sorted(self.kb, key=self.d2p.__getitem__)
        rng = np.random.default_rng(_P.SEED + 702)
        self.insert_order = [
            stable_ids[int(i)] for i in rng.permutation(len(stable_ids))
        ]

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

        # FIFO ignores hits entirely (no recency/frequency credit)
        fail = max_s < _P.SF_HIT_THRESH
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
                continue
            if n >= int(_P.WRITE_CAP):
                continue
            # admit, then evict oldest if over budget
            self.kb.add(cand_did)
            self.insert_order.append(cand_did)
            n += 1
            while len(self.kb) > len(kb_list) and self.insert_order:
                old = self.insert_order.pop(0)
                if old in self.kb:        # skip stale queue entries
                    self.kb.discard(old)
        self.update_cost += n

    def _step_observed_accesses(self, accesses):
        capacity = len(self.kb)
        n = 0
        for pool_idx in accesses:
            candidate = self.p2d[int(pool_idx)]
            if candidate in self.kb:
                continue
            self.maint_retrieval_cost += 1
            if n >= int(_P.WRITE_CAP):
                continue
            self.kb.add(candidate)
            self.insert_order.append(candidate)
            n += 1
            while len(self.kb) > capacity and self.insert_order:
                oldest = self.insert_order.pop(0)
                if oldest in self.kb:
                    self.kb.discard(oldest)
        self.update_cost += n
