"""SemFlow (QueryDriven) demand-ledger admission baseline."""
import logging

import numpy as np

from algorithms.cache.base import BaseStrategy
from algorithms.cache.ours.config import QueryDrivenConfig
from algorithms.cache.params import PARAMS as _P

log = logging.getLogger("motivation")


class QueryDriven(BaseStrategy):
    """Minimal query-demand writer kept for motivation experiments.

    QueryDriven is intentionally direct-only. It maintains a decayed demand
    ledger for documents retrieved by weak queries and a decayed serve ledger
    for resident documents that successfully answer recent queries. Bridge
    expansion now lives only in DRIPCore.
    """

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.config = config or QueryDrivenConfig()
        self.demand = {}
        self.serve = {}
        self.last_admission = {}

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            pi = self.d2p[did]
            self.serve[pi] = self.config.serve_prior

    def _decay(self):
        d = self.config.demand_decay
        s = self.config.serve_decay
        m = self.config.min_stat
        self.demand = {p: v * d for p, v in self.demand.items() if v * d >= m}
        self.serve = {p: v * s for p, v in self.serve.items() if v * s >= m}

    def _admission_price(self, window_idx):
        return float(getattr(self, "_write_price", 0.0))

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
        support_slots = max(1, int(getattr(self.config, "support_slots", 1)))
        cover_cnt = (q_kb >= _P.SF_HIT_THRESH).sum(axis=1)

        self._decay()

        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            serve_topk = max(1, int(getattr(self.config, "serve_topk", 1)))
            serve_topk = min(serve_topk, q_kb.shape[1])
            for row in q_kb[succ]:
                pos = np.argpartition(row, -serve_topk)[-serve_topk:]
                pos = [int(p) for p in pos if row[p] >= _P.SF_HIT_THRESH]
                if not pos:
                    continue
                credit = 1.0 / len(pos)
                for p in pos:
                    pi = int(kb_idx[p])
                    self.serve[pi] = self.serve.get(pi, 0.0) + credit

        fail = cover_cnt < support_slots
        n_fail = int(fail.sum())
        if n_fail == 0:
            self.last_admission = {
                "w": window_idx,
                "weak_queries": 0,
                "write_budget": getattr(self, "_write_budget", None),
                "write_price": self._admission_price(window_idx),
                "writes": 0,
            }
            return

        fqe = nqe[fail]
        pool_sims = fqe @ self.doc_embs.T

        for qi in range(n_fail):
            t1 = int(np.argmax(pool_sims[qi]))
            self.demand[t1] = self.demand.get(t1, 0.0) + 1.0

        topk = min(self.config.prefetch_topk, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * topk
        for qi in range(n_fail):
            row = pool_sims[qi]
            top = np.argpartition(row, -topk)[-topk:]
            sims = np.maximum(row[top].astype(float), 0.0)
            total = sims.sum()
            if total <= 0.0:
                continue
            weights = sims / total
            for weight, pi in zip(weights, top):
                pi = int(pi)
                self.demand[pi] = (
                    self.demand.get(pi, 0.0)
                    + float(weight) * self.config.neigh_gamma
                )

        kb_pos = set(int(i) for i in kb_idx)
        kb_arr = np.array(sorted(kb_pos), dtype=int)
        kb_emb_now = self.doc_embs[kb_arr]
        kb_self = kb_emb_now @ kb_emb_now.T
        np.fill_diagonal(kb_self, -1.0)
        red_vec = kb_self.max(axis=1)
        red_map = {int(p): float(red_vec[i]) for i, p in enumerate(kb_arr)}

        def is_dup(cp):
            return float((self.doc_embs[cp] @ kb_emb_now.T).max()) > self.config.tau_admit

        candidates = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not candidates:
            self.last_admission = {
                "w": window_idx,
                "weak_queries": n_fail,
                "write_budget": getattr(self, "_write_budget", None),
                "write_price": self._admission_price(window_idx),
                "writes": 0,
            }
            return

        def vscore(p):
            base = self.serve.get(p, 0.0) + self.demand.get(p, 0.0)
            penalty = self.config.lambda_red * max(
                0.0, red_map.get(p, 0.0) - self.config.red_thresh
            )
            return base - penalty

        evict_val = {p: vscore(p) for p in kb_pos}
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])

        writes = 0
        evict_i = 0
        write_budget = getattr(self, "_write_budget", None)
        write_price = self._admission_price(window_idx)
        for cand_value, cp in candidates:
            if evict_i >= len(evictable):
                break
            if write_budget is not None and writes >= write_budget:
                break
            victim = evictable[evict_i]
            margin = float(getattr(self.config, "admission_gain_margin", 1.0))
            if cand_value - margin * evict_val[victim] <= write_price:
                break
            if is_dup(cp):
                continue
            self.kb.discard(self.p2d[victim])
            self.kb.add(self.p2d[cp])
            self.serve.pop(victim, None)
            evict_i += 1
            writes += 1

        self.update_cost += writes
        self.last_admission = {
            "w": window_idx,
            "weak_queries": n_fail,
            "write_budget": write_budget,
            "write_price": round(write_price, 6),
            "writes": writes,
        }


class QueryDrivenLoose(QueryDriven):
    """Sensitivity variant with wider probing and a looser admission gate."""

    probe_topk = 50
    gate_ratio = 0.7

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
        probe = min(self.probe_topk, pool_sims.shape[1])
        self.maint_retrieval_cost += n_fail * probe
        kb_pos = set(int(i) for i in kb_idx)
        for qi in range(n_fail):
            top = np.argpartition(pool_sims[qi], -probe)[-probe:]
            sims = np.maximum(pool_sims[qi, top].astype(float), 0.0)
            total = sims.sum()
            if total <= 0.0:
                continue
            weights = sims / total
            for weight, pi in zip(weights, top):
                pi = int(pi)
                self.demand[pi] = self.demand.get(pi, 0.0) + float(weight)
        candidates = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not candidates:
            return
        evict_val = {
            int(p): self.serve.get(int(p), 0.0) + self.demand.get(int(p), 0.0)
            for p in kb_idx
        }
        evictable = sorted(kb_pos, key=lambda p: evict_val[p])
        writes = 0
        evict_i = 0
        for cand_value, cp in candidates:
            if evict_i >= len(evictable):
                break
            victim = evictable[evict_i]
            if cand_value <= self.gate_ratio * evict_val[victim]:
                break
            self.kb.discard(self.p2d[victim])
            self.kb.add(self.p2d[cp])
            self.serve.pop(victim, None)
            evict_i += 1
            writes += 1
        self.update_cost += writes
