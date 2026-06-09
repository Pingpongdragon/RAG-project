"""Shared base classes for window-level cache policies (verbatim from motivation_2)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
import logging
log = logging.getLogger("motivation")

class BaseStrategy:
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        self.name = name
        self.doc_pool = doc_pool
        self.doc_embs = doc_embs
        self.title_to_idx = title_to_idx
        self.d2p = {d['doc_id']: i for i, d in enumerate(doc_pool)}
        self.p2d = {i: d['doc_id'] for i, d in enumerate(doc_pool)}
        self.kb = set()
        self.update_cost = 0           # KB writes (insertions/replacements)
        self.maint_retrieval_cost = 0  # background pool scans (offline batch)
        self.serve_retrieval_cost = 0  # per-query pool fetches (online latency)

    @property
    def retrieval_cost(self):
        """Total pool retrievals (maintenance + serve-time)."""
        return self.maint_retrieval_cost + self.serve_retrieval_cost

    @property
    def cost(self):
        """Backward-compat alias: total operations (writes + retrievals)."""
        return self.update_cost + self.retrieval_cost

    def set_kb(self, ids):
        self.kb = set(ids)

    def prepare_window(self, window_queries, window_query_embs, window_idx):
        pass

    def step(self, window_queries, window_query_embs, window_idx):
        raise NotImplementedError



class _ArrivalCacheBase(BaseStrategy):
    """Shared scaffolding: random doc arrivals + per-window query touch.

    Each window:
      1. Sample _P.DOC_ARRIVE pool docs as "arrivals" (no failure probing).
      2. Update per-doc bookkeeping based on which KB docs got hit by queries.
      3. Subclass decides which KB docs to evict and whether to admit each
         arrival, then admits up to _P.DOC_ADD_CAP entries.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(_P.SEED + 200 + hash(name) % 1000)
        self.all_ids = [d['doc_id'] for d in doc_pool]

    # subclass hooks
    def _on_init_kb(self, did, pi):
        pass
    def _on_query_hit(self, kb_pi, window_idx):
        pass
    def _on_query_seen(self, kb_pi, sim, window_idx):
        pass
    def _evict_score(self, pi, window_idx):
        # lower = more evictable
        return 0.0
    def _admit_score(self, pi, sim_to_kb_max, window_idx):
        # higher = more worth admitting; return None to skip
        return 1.0

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self._on_init_kb(did, self.d2p[did])

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return

        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T  # (n_q, |KB|)

        # per-KB-doc passive observation: max sim from any query in window
        max_sim_per_kb = np.max(q_kb, axis=0) if q_kb.size else np.zeros(len(kb_list))
        for i, pi in enumerate(kb_idx):
            self._on_query_seen(int(pi), float(max_sim_per_kb[i]), window_idx)

        # successful hits credit recency/frequency
        max_s = np.max(q_kb, axis=1) if q_kb.size else np.zeros(len(window_queries))
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                self._on_query_hit(int(kb_idx[pos]), window_idx)

        # passive arrival stream (no failure-driven probing)
        arrivals = self.rng.choice(
            self.all_ids,
            min(_P.DOC_ARRIVE, len(self.all_ids)),
            replace=False,
        )
        self.maint_retrieval_cost += len(arrivals)

        kb_pos_set = set(int(i) for i in kb_idx)
        # admission candidates with scores
        cand_scored = []
        for did in arrivals:
            ni = self.d2p[did]
            if ni in kb_pos_set:
                continue
            ne = self.doc_embs[ni]
            sims = kb_emb @ ne
            best_sim = float(np.max(sims)) if sims.size else 0.0
            score = self._admit_score(ni, best_sim, window_idx)
            if score is None:
                continue
            cand_scored.append((score, ni))

        if not cand_scored:
            return

        # most attractive arrivals first
        cand_scored.sort(reverse=True)
        # least valuable KB docs first (eviction order)
        evict_order = sorted(
            kb_pos_set,
            key=lambda p: self._evict_score(p, window_idx),
        )

        n = 0
        for score, ni in cand_scored:
            if n >= min(_P.DOC_ADD_CAP, len(evict_order)):
                break
            ep = evict_order[n]
            old = self.p2d[ep]
            new = self.p2d[ni]
            self.kb.discard(old)
            self.kb.add(new)
            self._on_query_hit(ni, window_idx)
            n += 1
        self.update_cost += n


