"""Agent-RAG Cache (ARC) baseline — Lin et al., arXiv:2511.02919 (ACL 2026 Findings).

NOT the classic Adaptive Replacement Cache (Megiddo & Modha, FAST'03). This is
the *Agent RAG Cache Mechanism* that scores cached passages by a Distance-Rank
Frequency (DRF) term plus an embedding-space hubness centrality term, with a
memory-footprint penalty, and evicts the lowest-priority item.

Faithful to the paper's Algorithm 1 (query-driven escalation), adapted to this
repo's window-level BaseStrategy so it is directly comparable to QueryDriven /
DRYAD / LRU under the same cost accounting:

  Per window, for each query q:
    1. Retrieve top-k from the cache (KB). If the mean query-cache distance
       exceeds tau, "escalate" to the full pool (a maint_retrieval_cost) and
       retrieve top-k from there instead.
    2. Update DRF for every retrieved item:  DRF(p) += 1 / (rank * dist^alpha).
       New items are inserted into the cache; existing items accumulate.
    3. Evict lowest Priority(p) until the cache is within budget.

  Priority(p) = 1/log(w(p)+1) * [ beta * log(h_k(p)+1) + (1-beta) * DRF(p) ]

  - h_k(p): hubness = how often p appears in other cache items' k-NN lists
            (computed over the cache's own candidate set, query-agnostic).
  - w(p):   memory footprint of item p (uniform here -> penalty is constant,
            so it does not distort relative ordering; kept for faithfulness).

Paper hyper-parameters: alpha=0.4, beta in {0.7, 0.15, 0.2} per dataset,
tau=0.2, top-K=50. We expose them as class attributes.

Key contrast with DRYAD (ours): ARC uses *cumulative* DRF + *static* geometric
hubness with no drift detection and assumes a fixed query distribution P(q|Theta).
It has no per-window write budget (evicts item-by-item to fit capacity) and no
bridge / entity-chain admission — its geometric scoring can only cache items
similar to past queries, so multi-hop bridge documents (never directly retrieved)
stay out of the cache.
"""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")


class AgentRAGCache(BaseStrategy):
    ALPHA      = 0.4    # distance sensitivity in DRF
    BETA       = 0.3    # centrality (hubness) vs query-frequency (DRF) balance
    TAU        = 0.2    # escalation threshold on mean query-cache *distance*
    HUB_K      = 10     # k for hubness kNN over the cache candidate set
    TOP_K      = 50     # retrieval width (paper uses K=50)

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, use_hubness=True):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.use_hubness = use_hubness   # False -> ARC (w/o hubness) ablation
        self.drf = {}          # pool_idx -> float (cumulative distance-rank frequency)
        self._hub_cache = {}   # pool_idx -> float (hubness, recomputed per window)

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.drf.setdefault(self.d2p[did], 0.0)

    # ---- hubness over the cache's own candidate set (query-agnostic) ----
    def _recompute_hubness(self, kb_idx, kb_emb):
        if not self.use_hubness:
            return
        n = len(kb_idx)
        self._hub_cache = {}
        if n <= 1:
            for p in kb_idx:
                self._hub_cache[int(p)] = 0.0
            return
        sim = kb_emb @ kb_emb.T
        np.fill_diagonal(sim, -np.inf)
        k = min(self.HUB_K, n - 1)
        # for each row, indices of its top-k neighbours
        nn = np.argpartition(sim, -k, axis=1)[:, -k:]
        counts = np.zeros(n, dtype=np.float64)
        for row in nn:
            counts[row] += 1.0
        for i, p in enumerate(kb_idx):
            self._hub_cache[int(p)] = float(counts[i])

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
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)

        # hubness is query-agnostic over current cache contents
        self._recompute_hubness(kb_idx, kb_emb)

        n_writes = 0
        for qi in range(nqe.shape[0]):
            q = nqe[qi]

            # --- 1. retrieve top-k from cache; escalate if mean distance > tau ---
            kb_list = sorted(self.kb)
            if not kb_list:
                break
            kb_idx = np.array([self.d2p[d] for d in kb_list])
            kb_emb = self.doc_embs[kb_idx]
            sims = kb_emb @ q
            k = min(self.TOP_K, len(kb_idx))
            top = np.argpartition(sims, -k)[-k:]
            top = top[np.argsort(sims[top])[::-1]]
            mean_dist = float(np.mean(1.0 - sims[top]))

            escalate = mean_dist > self.TAU
            if escalate:
                # escalate to full pool (one pool retrieval -> maintenance cost)
                pool_sims = self.doc_embs @ q
                self.maint_retrieval_cost += 1
                kp = min(self.TOP_K, pool_sims.shape[0])
                ptop = np.argpartition(pool_sims, -kp)[-kp:]
                ptop = ptop[np.argsort(pool_sims[ptop])[::-1]]
                ret_idx = ptop
                ret_sims = pool_sims[ptop]
            else:
                ret_idx = kb_idx[top]
                ret_sims = sims[top]

            # --- 2. update DRF for retrieved items; insert new ones ---
            for rank, (pi, s) in enumerate(zip(ret_idx, ret_sims), start=1):
                pi = int(pi)
                dist = max(1.0 - float(s), 1e-6)
                contrib = 1.0 / (rank * (dist ** self.ALPHA))
                if pi in self.drf and self.p2d[pi] in self.kb:
                    self.drf[pi] += contrib
                else:
                    self.drf[pi] = self.drf.get(pi, 0.0) + contrib
                    if self.p2d[pi] not in self.kb:
                        self.kb.add(self.p2d[pi])
                        n_writes += 1

            # --- 3. evict lowest-priority until within budget ---
            budget = len(kb_list)  # cache capacity fixed at init size
            if len(self.kb) > budget:
                # refresh hubness over the enlarged set before eviction
                ev_list = sorted(self.kb)
                ev_idx = np.array([self.d2p[d] for d in ev_list])
                self._recompute_hubness(ev_idx, self.doc_embs[ev_idx])
                while len(self.kb) > budget:
                    victim = min(self.kb, key=lambda d: self._priority(self.d2p[d]))
                    self.kb.discard(victim)
                    self.drf.pop(self.d2p[victim], None)

        self.update_cost += n_writes
