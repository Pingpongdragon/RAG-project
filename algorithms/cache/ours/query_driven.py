"""SemFlow (QueryDriven) demand-ledger admission (ours)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")

class QueryDriven(BaseStrategy):
    """Query-demand-driven KB writer (single mechanism, no regime branching).

    Two long-lived per-doc statistics drive every decision:
      - demand[d]: exponentially-decayed sum of sim(q, d) over windows
        where d was a top-K pool candidate for a *failing* query. Big
        demand means many recent queries lacked KB coverage and would
        have benefited from d.
      - serve[d]: exponentially-decayed count of (window, query) pairs
        where d was the best KB hit for that query above the SF
        threshold. Big serve means d is currently doing useful work.

    Per window:
      1. Score every query against KB by top-2 coverage (cover_s).
      2. Decay both stats (demand/serve) by their respective rates.
      3. For each query that *succeeds* (max_s >= _P.SF_HIT_THRESH), credit
         the best KB doc with +1 serve.
      4. For each query that *fails* (cover_s < _P.SF_HIT_THRESH), retrieve
         top-K pool docs and add their similarity to demand[d].
      5. Sort non-KB candidates by demand desc; sort KB docs by
         (serve + demand_inside_kb) asc.
      6. Replace KB doc e by candidate c iff demand[c] > serve[e] + demand[e].
         This is the only admission test: a candidate must carry more
         accumulated evidence than the weakest currently-resident doc.
      7. Write cap per window = number of failing queries. The cap is
         therefore drift-proportional with no explicit drift detection.

    Why this should beat the baselines:
      - vs Static: ever updates at all.
      - vs RandomFIFO / DocArrival: uses query-failure signal, so writes
        target documents the workload actually needs.
      - vs KnowledgeEdit: uses query-side signal instead of pure KB-internal
        similarity, which is wrong-signal under workload drift.
      - vs LogDrivenArrival: no per-cycle lag, no fixed write quota, and
        accumulates evidence across windows so a doc relevant to many
        recurring failures wins over a doc relevant to a single query.
      - vs OnDemandFetch: persistent (free at serve time), but admits
        gracefully (only when demand outweighs current serving value).
    """
    DEMAND_DECAY = 0.92
    SERVE_DECAY  = 0.92
    PROBE_TOPK   = 8
    MIN_STAT     = 0.01

    SERVE_PRIOR  = 1.0  # initial trust per resident KB doc

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.demand = {}   # pool_idx -> float
        self.serve  = {}   # pool_idx -> float
    def set_kb(self, ids):
        super().set_kb(ids)
        # Bayesian prior: trust the initial KB. Every resident doc starts
        # with one unit of serve evidence, equivalent to "served once".
        # This protects the initial state until live failures prove a doc
        # should be replaced. The prior decays with the same rate as
        # serve, so its influence vanishes over a few windows if the doc
        # is never actually used.
        for did in self.kb:
            pi = self.d2p[did]
            self.serve[pi] = self.SERVE_PRIOR

    def _decay(self):
        d = self.DEMAND_DECAY; s = self.SERVE_DECAY; m = self.MIN_STAT
        self.demand = {p: v*d for p, v in self.demand.items() if v*d >= m}
        self.serve  = {p: v*s for p, v in self.serve.items()  if v*s >= m}

    PREFETCH_TOPK = 5
    TAU_ADMIT     = 0.95
    LAMBDA_RED    = 1.5
    RED_THRESH    = 0.85
    NEIGH_GAMMA   = 0.4
    WRITE_CAP_T1  = 200

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

        self._decay()

        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.serve[pi] = self.serve.get(pi, 0.0) + 1.0

        fail = max_s < _P.SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            self.update_cost += 0
            return

        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T

        # Top-1 miss demand
        for qi in range(n_fail):
            t1 = int(np.argmax(pool_sims[qi]))
            self.demand[t1] = self.demand.get(t1, 0.0) + 1.0

        # NEIGHBORHOOD DEMAND: warm semantic neighbors of each missed query
        topk = min(self.PREFETCH_TOPK, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * topk
        for qi in range(n_fail):
            row = pool_sims[qi]
            top = np.argpartition(row, -topk)[-topk:]
            sims = np.maximum(row[top].astype(float), 0.0)
            tot = sims.sum()
            if tot <= 0:
                continue
            w = sims / tot
            for wj, pi in zip(w, top):
                pi = int(pi)
                self.demand[pi] = self.demand.get(pi, 0.0) + float(wj) * self.NEIGH_GAMMA

        # Redundancy field
        kb_self = kb_emb @ kb_emb.T
        np.fill_diagonal(kb_self, -1.0)
        red_vec = kb_self.max(axis=1)
        red_map = {int(p): float(red_vec[i]) for i, p in enumerate(kb_idx)}
        kb_pos = set(int(i) for i in kb_idx)

        def _is_dup(cp, kb_emb_now):
            return float((self.doc_embs[cp] @ kb_emb_now.T).max()) > self.TAU_ADMIT

        # Build KB embeddings for tier-2
        kb_arr = np.array(sorted(kb_pos), dtype=int)
        kb_emb_now = self.doc_embs[kb_arr]
        kb_self2 = kb_emb_now @ kb_emb_now.T
        np.fill_diagonal(kb_self2, -1.0)
        red_vec2 = kb_self2.max(axis=1)
        red_map2 = {int(p): float(red_vec2[i]) for i, p in enumerate(kb_arr)}

        # === TIER 2: demand-gated admission with redundancy-aware eviction ===
        cands = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not cands:
            return

        def vscore(p):
            base = self.serve.get(p, 0.0) + self.demand.get(p, 0.0)
            pen = self.LAMBDA_RED * max(0.0, red_map2.get(p, 0.0) - self.RED_THRESH)
            return base - pen
        evict_val = {p: vscore(p) for p in kb_pos}
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])

        n = 0
        ei = 0
        # Optional per-window write budget (DRYAD module ② sets this to λ·B;
        # None = SemFlow's original behaviour where the gate self-limits).
        wbudget = getattr(self, '_write_budget', None)
        for cval, cp in cands:
            if ei >= len(evictable):
                break
            if wbudget is not None and n >= wbudget:
                break
            ep = evictable[ei]
            if cval <= evict_val[ep]:
                break
            if _is_dup(cp, kb_emb_now):
                continue
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.serve.pop(ep, None)
            ei += 1
            n += 1
        self.update_cost += n



class QueryDrivenLoose(QueryDriven):
    """QDC with relaxed probe width and admission gate (sensitivity check).

    Differences from QueryDriven:
      - PROBE_TOPK 8 -> 50 (each failing query credits demand to many more
        pool candidates, dramatically increasing eviction pressure).
      - Admission gate: cval > 0.7 * evict_val (vs strict cval > evict_val),
        so candidates can dislodge resident docs even at parity.
    Used to test whether QDC's losses against LRU/GPTCache come from being
    too conservative on writes vs from a fundamentally weaker signal.
    """
    PROBE_TOPK = 50
    GATE_RATIO = 0.7

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
        self._decay()
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.serve[pi] = self.serve.get(pi, 0.0) + 1.0
        fail = max_s < _P.SF_HIT_THRESH
        n_fail = int(fail.sum())
        if n_fail == 0:
            return
        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T
        probe = min(self.PROBE_TOPK, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * probe
        kb_pos = set(int(i) for i in kb_idx)
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -probe)[-probe:]
            sims = pool_sims[qi, top].astype(float)
            sims = np.maximum(sims, 0.0)
            tot = sims.sum()
            if tot <= 0:
                continue
            weights = sims / tot
            for w, pi in zip(weights, top):
                pi = int(pi)
                self.demand[pi] = self.demand.get(pi, 0.0) + float(w)
        cands = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not cands:
            return
        evict_val = {int(p): self.serve.get(int(p), 0.0) + self.demand.get(int(p), 0.0)
                     for p in kb_idx}
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])
        n = 0
        ei = 0
        for cval, cp in cands:
            if ei >= len(evictable):
                break
            ep = evictable[ei]
            if cval <= self.GATE_RATIO * evict_val[ep]:
                break
            self.kb.discard(self.p2d[ep])
            self.kb.add(self.p2d[cp])
            self.serve.pop(ep, None)
            ei += 1
            n += 1
        self.update_cost += n


