"""Paradigm references: on-demand fetch / log-driven / MemGPT."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy, _ArrivalCacheBase
import logging
log = logging.getLogger("motivation")

class OnDemandFetch(BaseStrategy):
    """Passive on-demand search: fetch from pool per-query, don't store.

    Models CRAG / Agent-style RAG: when a query's best KB hit is below
    _P.SF_HIT_THRESH, search the external pool for top-K and use those results
    directly. The KB itself is NEVER updated (static), but the query gets
    augmented results in the same window. Cost counts external search calls.

    This demonstrates that even perfect external search cannot replace
    KB consolidation: (a) every query incurs search latency, (b) repeated
    queries re-search the same docs, (c) no learning across windows.

    For evaluation, we temporarily add fetched docs to a "virtual KB" for
    that window only, then remove them before the next window.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.fetch_k = _P.FETCH_TOP_K
        self._fetched_this_window = set()

    def prepare_window(self, window_queries, window_query_embs, window_idx):
        self._fetched_this_window = set()
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]

        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)

        for qi in range(len(window_queries)):
            q_kb_sim = nqe[qi] @ kb_emb.T
            if float(np.max(q_kb_sim)) >= _P.SF_HIT_THRESH:
                continue
            pool_sim = nqe[qi] @ self.doc_embs.T
            top_idx = np.argpartition(pool_sim, -self.fetch_k)[-self.fetch_k:]
            for pi in top_idx:
                self._fetched_this_window.add(self.p2d[pi])
            self.serve_retrieval_cost += len(top_idx)

    def get_effective_kb(self, window_queries, window_query_embs):
        """Return KB + fetched docs for this window (for recall eval)."""
        return self.kb | self._fetched_this_window

    def step(self, window_queries, window_query_embs, window_idx):
        pass



class LogDrivenArrival(BaseStrategy):
    """Lagging log-driven update: analyse previous window's failures, fix next.

    Models a human-in-the-loop or scheduled-batch pipeline:
    after window T, inspect which queries failed, find their best pool
    candidates, and add those to KB for window T+1.

    Always one window behind — demonstrates the "lagging effect" under drift:
    by the time the fix arrives, the query distribution may have shifted again.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self._pending_adds = []  # docs to add at next review cycle
        self._fail_buffer_qe = []  # accumulated fail query embeddings

    def step(self, window_queries, window_query_embs, window_idx):
        # Apply pending adds at review-cycle boundaries
        if self._pending_adds and window_idx % _P.LOG_LAG_WINDOWS == 0:
            kb_list = sorted(self.kb)
            # Evict least-useful current docs to make room
            if len(kb_list) > 0:
                kb_idx = np.array([self.d2p[d] for d in kb_list])
                kb_emb = self.doc_embs[kb_idx]
                norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
                nqe = window_query_embs / np.clip(norms, 1e-10, None)
                usefulness = np.mean(nqe @ kb_emb.T, axis=0)
                evict_order = np.argsort(usefulness)
                n = 0
                ei = 0
                for did in self._pending_adds:
                    if did in self.kb:
                        continue
                    if ei >= len(evict_order):
                        break
                    old = kb_list[evict_order[ei]]
                    self.kb.discard(old)
                    self.kb.add(did)
                    n += 1
                    ei += 1
                self.update_cost += n
            self._pending_adds = []
            self._fail_buffer_qe = []

        # Accumulate fail queries every window
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T
        max_s = np.max(q_kb, axis=1)
        fail = max_s < _P.SF_HIT_THRESH
        if fail.any():
            self._fail_buffer_qe.append(nqe[fail])

        # At end of each lag cycle, analyse accumulated failures for next cycle
        if (window_idx + 1) % _P.LOG_LAG_WINDOWS == 0 and self._fail_buffer_qe:
            fqe = np.concatenate(self._fail_buffer_qe, axis=0)
            pool_sims = fqe @ self.doc_embs.T
            cand_set = set()
            tk = min(_P.LOG_FIX_TOP_K, len(self.doc_embs))
            self.maint_retrieval_cost += len(fqe) * tk
            for qi in range(len(fqe)):
                top = np.argpartition(pool_sims[qi], -tk)[-tk:]
                cand_set.update(top.tolist())
            cand_set -= {self.d2p[d] for d in self.kb}
            scored = [(float(np.mean(fqe @ self.doc_embs[pi])), self.p2d[pi])
                      for pi in sorted(cand_set)]
            scored.sort(reverse=True)
            self._pending_adds = [did for _, did in scored[:_P.LOG_FIX_CAP]]


# ═══════════════════════════════════════════════════════════════════
# NEW BASELINES — cache-style and agent-memory-style baselines.
# IMPORTANT: these baselines do NOT receive QDC's query-failure-driven
# top-K candidate retrieval. Their admission stream is the same passive
# DocArrival stream used by HippoRAG/LightRAG; only the eviction policy
# differs. This isolates the *signal source* (failure-targeted demand)
# as QDC's contribution, rather than letting baselines piggyback on it.
# ═══════════════════════════════════════════════════════════════════



class MemGPTStyle(_ArrivalCacheBase):
    """MemGPT-style importance memory.

    Admission source: random arrivals.
    Importance(d) = decayed access frequency.
    Eviction: lowest importance.
    Passive freq: any query hit on a KB doc increments freq.
    """
    DECAY = 0.88
    INIT_FREQ = 1.0

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.freq = {}
        self.last_hit = {}

    def _importance(self, pi, window_idx):
        f = self.freq.get(pi, 0.0)
        lh = self.last_hit.get(pi, -1)
        return f * (self.DECAY ** max(0, window_idx - lh))

    def _on_init_kb(self, did, pi):
        self.freq[pi] = self.INIT_FREQ
        self.last_hit[pi] = 0
    def _on_query_hit(self, kb_pi, window_idx):
        self.freq[kb_pi] = self.freq.get(kb_pi, 0.0) + 1.0
        self.last_hit[kb_pi] = window_idx
    def _evict_score(self, pi, window_idx):
        return self._importance(pi, window_idx)
    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        # new arrivals start at INIT_FREQ; return that as ranking score
        return self.INIT_FREQ * sim_to_kb_max


# ── Updated registry: add new baselines ──


