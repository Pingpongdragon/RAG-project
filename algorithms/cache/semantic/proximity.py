"""Proximity cache baseline — Bergman et al., "Leveraging Approximate Caching
for Faster Retrieval-Augmented Generation", EuroMLSys 2025.

As described in the Agent-RAG ARC paper (Lin et al. 2511.02919, §5 Baselines):
  "Proximity maintains historical query-document pairs and returns previously
   retrieved passages from the most semantically similar past query when the
   similarity exceeds its threshold tau, evicting the oldest query-document pair."

So Proximity is an *approximate* cache keyed on past queries: a new query reuses
a past query's cached documents if cos(q, q_past) >= tau. The cached unit is the
(query -> retrieved docs) association; eviction is FIFO on query-doc pairs.

Adapted to this repo's window-level BaseStrategy / fixed doc-budget KB so it is
comparable to the other cache policies:
  - We keep a FIFO ledger of past queries and the doc(s) they pulled into the KB.
  - Per window, each query that is within tau of some past query contributes no
    new fetch (approximate hit). Queries with no close past query "miss": we
    fetch their top-1 pool doc into the KB and log the (query, doc) pair.
  - When the KB exceeds budget, evict the doc of the oldest logged pair (FIFO).

Why Proximity is expected to do poorly under the conditions we care about
(documented here so the comparison is not read as a straw man):

  1. No multi-hop / bridge awareness. Proximity reuses whatever a *similar past
     query* fetched, and that fetch is plain top-k sim(q, doc). A bridge
     document (needed for hop 2 but not similar to the query) is never fetched
     by any past query, so it never enters the ledger and can never be reused.
     This is a property of the method, not a handicap we imposed.

  2. No drift handling. The ledger is FIFO and similarity-gated only. When the
     query distribution shifts, old (query, doc) pairs become stale but are
     evicted purely by age, not by loss of relevance; and a shifted query finds
     no close past query (cos < tau), so it always misses until the ledger
     refills — there is no mechanism that *detects* the shift and proactively
     adapts. It pays the full miss cost across the whole drift transient.

These two gaps are exactly the axes our method targets, which is why Proximity
(and GPTCache, which shares gap 1 and 2) trail under multi-hop + drift. The
point of including them is to show the gaps are real and consequential, not to
beat a weakened opponent: each runs with its paper-faithful mechanism and the
same budget / cost accounting as every other policy.
"""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")


class Proximity(BaseStrategy):
    TAU = 0.8   # query-similarity reuse threshold (paper grid-searched ~0.2 on
                # *distance*; here we gate on cosine *similarity*, so high = reuse)

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        # FIFO ledger of (query_embedding, doc_pool_idx) associations
        self._past_q = []      # list of np.ndarray (unit query vectors)
        self._pairs = []       # parallel FIFO list of doc pool_idx fetched

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)

        past = np.array(self._past_q) if self._past_q else None
        n_writes = 0
        budget = len(self.kb)

        for qi in range(nqe.shape[0]):
            q = nqe[qi]
            # approximate hit: a past query within tau already cached its docs
            if past is not None and past.shape[0] > 0:
                qsim = past @ q
                if float(qsim.max()) >= self.TAU:
                    continue  # reuse past fetch, no new work

            # miss: fetch top-1 pool doc for this query into the KB
            pool_sims = self.doc_embs @ q
            self.maint_retrieval_cost += 1
            top1 = int(np.argmax(pool_sims))
            self._past_q.append(q.copy())
            past = np.array(self._past_q)
            if self.p2d[top1] not in self.kb:
                self.kb.add(self.p2d[top1])
                self._pairs.append(top1)
                n_writes += 1

            # FIFO eviction on (query, doc) pairs to respect budget
            while len(self.kb) > budget and self._pairs:
                old_doc = self._pairs.pop(0)
                if self._past_q:
                    self._past_q.pop(0)
                self.kb.discard(self.p2d[old_doc])
                past = np.array(self._past_q) if self._past_q else None

        self.update_cost += n_writes
