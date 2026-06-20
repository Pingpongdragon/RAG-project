"""Agent-RAG Cache (ARC) baseline — Lin et al., arXiv:2511.02919 (ACL 2026 Findings).

NOT the classic Adaptive Replacement Cache (Megiddo & Modha, FAST'03). This is
the *Agent RAG Cache Mechanism* that scores cached passages by a Distance-Rank
Frequency (DRF) term plus an embedding-space hubness centrality term, with a
memory-footprint penalty, and evicts the lowest-priority item.

Faithful to the paper's Algorithm 1 priority model, adapted to this repo's
window-level BaseStrategy so it is directly comparable to DRIP / LRU under the
same cost accounting:

  Per window, for each query q:
    1. Retrieve top-k from the cache (KB). If top-1 misses the experiment's
       shared hit threshold, "escalate" to the full pool (a maint_retrieval_cost)
       and retrieve top-k from there.
    2. Update DRF for every retrieved item:  DRF(p) += 1 / (rank * dist^alpha).
       Escalated non-residents become admission candidates.
    3. At the end of the window, admit up to WRITE_CAP candidates by ARC
       priority, evicting the lowest-priority resident for each admitted item.

  Priority(p) = 1/log(w(p)+1) * [ beta * log(h_k(p)+1) + (1-beta) * DRF(p) ]

  - h_k(p): hubness = how often p appears in other cache items' k-NN lists
            (computed over the cache's own candidate set, query-agnostic).
  - w(p):   memory footprint of item p (uniform here -> penalty is constant,
            so it does not distort relative ordering; kept for faithfulness).

Paper hyper-parameters: alpha=0.4, beta in {0.7, 0.15, 0.2} per dataset,
tau=0.2, top-K=50. We expose them as class attributes.

Key contrast with DRIP (ours): ARC uses *cumulative* DRF + *static* geometric
hubness with no drift detection and assumes a fixed query distribution P(q|Theta).
It has no drift-triggered write budget (evicts item-by-item to fit capacity) and no
bridge / entity-chain admission — its geometric scoring can only cache items
similar to past queries, so multi-hop bridge documents (never directly retrieved)
stay out of the cache.
"""
import os
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")


class AgentRAGCache(BaseStrategy):
    ALPHA      = float(os.environ.get('ARC_ALPHA', '0.4'))  # distance sensitivity in DRF
    BETA       = float(os.environ.get('ARC_BETA', '0.3'))   # hubness vs query-frequency balance
    TAU        = float(os.environ.get('ARC_TAU', '0.2'))    # paper escalation distance threshold
    HUB_K      = int(os.environ.get('ARC_HUB_K', '10'))     # k for hubness kNN
    TOP_K      = int(os.environ.get('ARC_TOP_K', '50'))     # retrieval width (paper uses K=50)

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, use_hubness=True):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.use_hubness = use_hubness   # False -> ARC (w/o hubness) ablation
        self.drf = {}          # pool_idx -> float (cumulative distance-rank frequency)
        self._hub_cache = {}   # pool_idx -> float (hubness, recomputed per window)

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.drf.setdefault(self.d2p[did], 0.0)

    # ---- hubness over cache (+ optional candidate) set, query-agnostic ----
    def _refresh_hubness_for(self, kb_idx, extra_idx=None):
        if not self.use_hubness:
            return
        idx = list(int(p) for p in kb_idx)
        if extra_idx:
            idx = idx + [int(p) for p in extra_idx if int(p) not in set(idx)]
        n = len(idx)
        self._hub_cache = {}
        if n <= 1:
            for p in idx:
                self._hub_cache[p] = 0.0
            return
        emb = self.doc_embs[np.array(idx)]
        sim = emb @ emb.T
        np.fill_diagonal(sim, -np.inf)
        k = min(self.HUB_K, n - 1)
        nn = np.argpartition(sim, -k, axis=1)[:, -k:]
        counts = np.zeros(n, dtype=np.float64)
        for row in nn:
            counts[row] += 1.0
        for i, p in enumerate(idx):
            self._hub_cache[p] = float(counts[i])

    def _recompute_hubness(self, kb_idx, kb_emb):
        self._refresh_hubness_for(kb_idx)

    def _priority(self, pi, w=1.0):
        drf = self.drf.get(pi, 0.0)
        if self.use_hubness:
            h = self._hub_cache.get(pi, 0.0)
            core = self.BETA * np.log(h + 1.0) + (1.0 - self.BETA) * drf
        else:
            core = drf  # ARC (w/o hubness): pure DRF
        return core / np.log(w + 1.0 + 1.0)  # +1 extra so log>0 for uniform w=1

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        budget = len(self.kb)   # fixed cache capacity (= init KB size)
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)

        admission_candidates = set()
        for qi in range(nqe.shape[0]):
            q = nqe[qi]

            # --- 1. retrieve top-k from cache; escalate only on a shared miss ---
            kb_list = sorted(self.kb)
            if not kb_list:
                break
            kb_idx = np.array([self.d2p[d] for d in kb_list])
            kb_emb = self.doc_embs[kb_idx]
            sims = kb_emb @ q
            k = min(self.TOP_K, len(kb_idx))
            top = np.argpartition(sims, -k)[-k:]
            top = top[np.argsort(sims[top])[::-1]]
            top1_sim = float(sims[top[0]]) if len(top) else -1.0

            if top1_sim >= _P.SF_HIT_THRESH:
                # cache already serves this query well: only refresh DRF of the
                # cache hits, no escalation, no churn.
                for rank, (pi, s) in enumerate(zip(kb_idx[top], sims[top]), start=1):
                    pi = int(pi)
                    dist = max(1.0 - float(s), 1e-6)
                    self.drf[pi] = self.drf.get(pi, 0.0) + 1.0 / (rank * (dist ** self.ALPHA))
                continue

            # escalate to full pool (one pool retrieval -> maintenance cost)
            pool_sims = self.doc_embs @ q
            self.maint_retrieval_cost += 1
            kp = min(self.TOP_K, pool_sims.shape[0])
            ptop = np.argpartition(pool_sims, -kp)[-kp:]
            ptop = ptop[np.argsort(pool_sims[ptop])[::-1]]

            # --- 2. update DRF for all escalated results (accumulate demand) ---
            for rank, pi in enumerate(ptop, start=1):
                pi = int(pi)
                dist = max(1.0 - float(pool_sims[pi]), 1e-6)
                self.drf[pi] = self.drf.get(pi, 0.0) + 1.0 / (rank * (dist ** self.ALPHA))

            admission_candidates.update(
                int(pi) for pi in ptop if self.p2d[int(pi)] not in self.kb)

        # --- 3. window-level priority-gated admission (bounded writes) ---
        if not admission_candidates:
            return
        import heapq
        current_idx = [self.d2p[d] for d in self.kb]
        self._refresh_hubness_for(current_idx, admission_candidates)
        ranked_candidates = sorted(
            admission_candidates,
            key=self._priority,
            reverse=True,
        )
        heap = [(self._priority(self.d2p[d]), d) for d in self.kb]
        heapq.heapify(heap)

        n_writes = 0
        for pi in ranked_candidates:
            if n_writes >= _P.WRITE_CAP:
                break
            did = self.p2d[pi]
            if did in self.kb:
                continue
            if len(self.kb) < budget:
                self.kb.add(did)
                n_writes += 1
                heapq.heappush(heap, (self._priority(pi), did))
                continue
            while heap and heap[0][1] not in self.kb:
                heapq.heappop(heap)
            if not heap:
                break
            weak_pri, victim = heap[0]
            if self._priority(pi) <= weak_pri:
                continue
            heapq.heappop(heap)
            self.kb.discard(victim)
            self.drf.pop(self.d2p[victim], None)
            self.kb.add(did)
            n_writes += 1
            heapq.heappush(heap, (self._priority(pi), did))

        self.update_cost += n_writes
