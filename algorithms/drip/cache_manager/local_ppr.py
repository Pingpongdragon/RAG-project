"""Local PPR bridge evidence for DRIP.

This module is the production-shaped version of the PPR prototype from
``algorithms/drip/tests/test_bridge_ppr.py``.  It keeps the experiment logic
identical but avoids dense per-query transition matrices so it can run inside
the Mo2 stream runner.
"""
from collections import defaultdict

import numpy as np

from algorithms.cache.params import PARAMS as _P

from . import DRIPCore


class LocalPPRBridgeEvidence:
    """Query-local PPR over GraphIndex's document/entity postings."""

    def __init__(self, graph_index, doc_embs, c=0.5, L=2, R=1, K0=5, d_cap=30):
        self.gi = graph_index
        self.doc_embs = doc_embs
        self.c = float(c)
        self.L = int(L)
        self.R = int(R)
        self.K0 = int(K0)
        self.d_cap = int(d_cap)
        self._neighbor_cache = {}

    def _doc_neighbors(self, pi, idf_max, degree_power):
        pi = int(pi)
        cached = self._neighbor_cache.get(pi)
        if cached is not None:
            return cached
        out = defaultdict(float)
        for ent in self.gi.doc_entities(pi):
            linked = self.gi.ent_to_docs.get(ent, ())
            degree = len(linked)
            if degree <= 1:
                continue
            if self.d_cap > 0 and degree > self.d_cap:
                continue
            idf = self.gi.ent_idf.get(ent, 1.0)
            weight = (idf / max(1e-9, idf_max)) / (degree ** degree_power)
            for bj in linked:
                bj = int(bj)
                if bj != pi:
                    out[bj] += weight
        norm = sum(out.values())
        if norm > 0.0:
            out = {bj: val / norm for bj, val in out.items()}
        else:
            out = {}
        self._neighbor_cache[pi] = out
        return out

    def _build_subgraph(self, seeds, idf_max, degree_power):
        nodes = set(seeds)
        frontier = set(seeds)
        adj = {}
        for _ in range(max(0, self.R)):
            nxt = set()
            for node in frontier:
                adj[node] = self._doc_neighbors(node, idf_max, degree_power)
                for nb in adj[node]:
                    if nb not in nodes:
                        nxt.add(nb)
            if not nxt:
                break
            nodes.update(nxt)
            frontier = nxt
        for node in list(nodes):
            if node not in adj:
                adj[node] = self._doc_neighbors(node, idf_max, degree_power)
        return nodes, adj

    def evidence(self, first_hops, kb_pos=None):
        """Return ``[(doc_pos, ppr_mass)]`` sorted by PPR evidence."""
        self.gi.build()
        if not self.gi.ent_to_docs:
            return [], {
                "ppr_seed_count": 0,
                "ppr_subgraph_size": 0,
                "ppr_candidate_count": 0,
            }
        kb_pos = set(int(p) for p in (kb_pos or ()))
        seeds = [
            (int(pi), max(0.0, float(sim)))
            for pi, sim in first_hops[: self.K0]
            if sim > 0.0
        ]
        if not seeds:
            return [], {
                "ppr_seed_count": 0,
                "ppr_subgraph_size": 0,
                "ppr_candidate_count": 0,
            }
        seed_sum = sum(sim for _, sim in seeds) or 1.0
        seed_vec = {pi: sim / seed_sum for pi, sim in seeds}
        seed_pos = set(seed_vec)
        idf_max = max(self.gi.ent_idf.values()) if self.gi.ent_idf else 1.0
        degree_power = float(getattr(self.gi.config, "entity_degree_power", 0.5))

        nodes, adj = self._build_subgraph(seed_vec.keys(), idf_max, degree_power)
        mass = dict(seed_vec)
        for _ in range(max(0, self.L)):
            nxt = defaultdict(float)
            for pi, val in seed_vec.items():
                nxt[pi] += self.c * val
            walk_weight = 1.0 - self.c
            if walk_weight > 0.0:
                for src, src_mass in mass.items():
                    if src_mass <= 0.0:
                        continue
                    nbrs = adj.get(src, {})
                    if not nbrs:
                        continue
                    for dst, trans in nbrs.items():
                        if dst in nodes:
                            nxt[dst] += walk_weight * src_mass * trans
            mass = dict(nxt)

        candidates = [
            (int(pi), float(score))
            for pi, score in mass.items()
            if pi not in seed_pos and pi not in kb_pos and score > 0.0
        ]
        candidates.sort(key=lambda item: -item[1])
        stats = {
            "ppr_seed_count": len(seed_pos),
            "ppr_subgraph_size": len(nodes),
            "ppr_candidate_count": len(candidates),
        }
        return candidates, stats


class PPRDRIPCore(DRIPCore):
    """DRIP with local-PPR bridge evidence replacing stock GraphIndex scoring."""

    def __init__(self, *args, ppr_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ppr_kwargs = ppr_kwargs or {}
        self._ppr_engine = None
        self._ppr_installed = False

    @property
    def _pool_ents(self):
        return self.graph_index.pool_ents

    @_pool_ents.setter
    def _pool_ents(self, pool_ents):
        self.graph_index.set_pool_entities(pool_ents)
        self._ppr_engine = None
        self._ppr_installed = False

    def _ensure_ppr(self):
        self.graph_index.build()
        if self._ppr_engine is None:
            self._ppr_engine = LocalPPRBridgeEvidence(
                self.graph_index,
                self.doc_embs,
                **self._ppr_kwargs,
            )
        return self._ppr_engine

    def _install_ppr_graph_evidence(self):
        if self._ppr_installed or not self.graph_index.has_metadata():
            return
        ppr = self._ensure_ppr()
        graph_index = self.graph_index

        def ppr_graph_evidence(query, first_hops, kb_pos, kb_emb, doc_embs):
            candidates, stats = ppr.evidence(first_hops, kb_pos=kb_pos)
            graph_index.last_stats = {
                "bridge_raw_paths": int(stats["ppr_subgraph_size"]),
                "bridge_after_degree_gate": int(stats["ppr_subgraph_size"]),
                "bridge_after_relation_gate": int(stats["ppr_candidate_count"]),
                "bridge_after_novelty_gate": int(stats["ppr_candidate_count"]),
                "bridge_after_threshold": int(stats["ppr_candidate_count"]),
                "bridge_selected": int(len(candidates)),
                "bridge_mmr_stopped": 0,
                "bridge_no_path": int(stats["ppr_seed_count"] == 0),
                "bridge_top_entities": [],
                "ppr_seed_count": int(stats["ppr_seed_count"]),
                "ppr_subgraph_size": int(stats["ppr_subgraph_size"]),
                "ppr_candidate_count": int(stats["ppr_candidate_count"]),
            }
            max_docs = int(getattr(graph_index.config, "bridge_max_docs", 20))
            return candidates[:max_docs]

        graph_index.graph_evidence = ppr_graph_evidence
        self._ppr_installed = True

    def step(self, window_queries, window_query_embs, window_idx):
        self._install_ppr_graph_evidence()
        return super().step(window_queries, window_query_embs, window_idx)


class PPRBridgeWriterDRIPCore(PPRDRIPCore):
    """PPR bridge evidence plus a minimal route-aware writer.

    The goal is intentionally modest: test whether PPR's high gold-write
    precision turns into better residency when bridge candidates get their own
    write budget and newly admitted bridge docs receive short-lived protection.
    """

    def __init__(
        self,
        *args,
        bridge_reserve=0.5,
        bridge_margin=0.75,
        bridge_stickiness=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bridge_reserve = float(bridge_reserve)
        self.bridge_margin = float(bridge_margin)
        self.bridge_stickiness = float(bridge_stickiness)
        self._bridge_recent = set()
        self._bridge_residency = {}

    def _decay(self):
        super()._decay()
        decay = float(getattr(self.config, "demand_decay", 0.92))
        floor = float(getattr(self.config, "min_stat", 0.01))
        self._bridge_residency = {
            int(p): val * decay
            for p, val in self._bridge_residency.items()
            if val * decay >= floor
        }

    def _credit_graph(self, candidates, gold_pos=None):
        for pi, score in candidates:
            if float(score) > 0.0:
                self._bridge_recent.add(int(pi))
        return super()._credit_graph(candidates, gold_pos)

    def _resident_priority(self, kb_idx, kb_emb):
        priority = super()._resident_priority(kb_idx, kb_emb)
        for p in list(priority):
            priority[p] += self.bridge_stickiness * self._bridge_residency.get(p, 0.0)
        return priority

    def _write(self, kb_idx, kb_emb, budget, gold_pos=None):
        gold_pos = gold_pos or set()
        if budget <= 0:
            return {
                "writes": 0,
                "candidates": 0,
                "gold_candidates": 0,
                "gold_writes": 0,
                "gold_rate": 0.0,
            }
        kb_pos = set(int(p) for p in kb_idx)
        all_candidates = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not all_candidates:
            return {
                "writes": 0,
                "candidates": 0,
                "gold_candidates": 0,
                "gold_writes": 0,
                "gold_rate": 0.0,
            }

        bridge_candidates = [
            item for item in all_candidates if int(item[1]) in self._bridge_recent
        ]
        direct_candidates = [
            item for item in all_candidates if int(item[1]) not in self._bridge_recent
        ]
        bridge_budget = int(np.ceil(self.bridge_reserve * budget)) if bridge_candidates else 0
        direct_budget = max(0, budget - bridge_budget)

        priority = self._resident_priority(kb_idx, kb_emb)
        victims = sorted(kb_pos, key=lambda p: priority[p])
        writes = 0
        gold_writes = 0
        victim_i = 0

        def current_kb_emb():
            return self.doc_embs[np.array(sorted(kb_pos), dtype=np.int64)]

        def admit_from(candidates, cap, margin):
            nonlocal writes, gold_writes, victim_i
            local_writes = 0
            for cand_value, cp in candidates:
                cp = int(cp)
                if local_writes >= cap or writes >= budget or victim_i >= len(victims):
                    break
                victim = int(victims[victim_i])
                gain = float(cand_value) - margin * priority[victim]
                if gain <= 0.0:
                    break
                duplicate = float((self.doc_embs[cp] @ current_kb_emb().T).max())
                if duplicate > self.config.tau_duplicate:
                    continue
                self.kb.discard(self.p2d[victim])
                self.kb.add(self.p2d[cp])
                self.serve.pop(victim, None)
                self._bridge_residency.pop(victim, None)
                kb_pos.discard(victim)
                kb_pos.add(cp)
                if cp in self._bridge_recent:
                    self._bridge_residency[cp] = (
                        self._bridge_residency.get(cp, 0.0) + 1.0
                    )
                victim_i += 1
                writes += 1
                local_writes += 1
                gold_writes += int(cp in gold_pos)

        admit_from(bridge_candidates, bridge_budget, self.bridge_margin)
        admit_from(direct_candidates, direct_budget, self.config.gain_margin)

        gold_candidates = sum(1 for _, p in all_candidates if int(p) in gold_pos)
        return {
            "writes": int(writes),
            "candidates": int(len(all_candidates)),
            "gold_candidates": int(gold_candidates),
            "gold_writes": int(gold_writes),
            "gold_rate": float(gold_writes / writes) if writes else 0.0,
        }

    def step(self, window_queries, window_query_embs, window_idx):
        self._bridge_recent = set()
        return super().step(window_queries, window_query_embs, window_idx)


class PPRBridgeDRFDRIPCore(PPRBridgeWriterDRIPCore):
    """PPR bridge evidence with ARC-style DRF/hubness retention.

    This variant keeps the writer simple: PPR supplies bridge evidence, while a
    slow DRF ledger and local hubness protect resident docs that are repeatedly
    useful.  Unlike ``PPRBridgeWriterDRIPCore``, PPR evidence is allowed to
    credit resident bridge docs so hidden B documents can stay warm across reuse
    queries even when dense query similarity does not serve them directly.
    """

    def __init__(
        self,
        *args,
        drf_weight=0.5,
        hub_weight=0.2,
        drf_decay=0.98,
        drf_alpha=0.4,
        hub_k=10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drf_weight = float(drf_weight)
        self.hub_weight = float(hub_weight)
        self.drf_decay = float(drf_decay)
        self.drf_alpha = float(drf_alpha)
        self.hub_k = int(hub_k)
        self.drf = {}

    def _install_ppr_graph_evidence(self):
        if self._ppr_installed or not self.graph_index.has_metadata():
            return
        ppr = self._ensure_ppr()
        graph_index = self.graph_index

        def ppr_graph_evidence(query, first_hops, kb_pos, kb_emb, doc_embs):
            # Include resident docs so repeated bridge evidence becomes
            # retention credit, not only admission credit.
            candidates, stats = ppr.evidence(first_hops, kb_pos=None)
            graph_index.last_stats = {
                "bridge_raw_paths": int(stats["ppr_subgraph_size"]),
                "bridge_after_degree_gate": int(stats["ppr_subgraph_size"]),
                "bridge_after_relation_gate": int(stats["ppr_candidate_count"]),
                "bridge_after_novelty_gate": int(stats["ppr_candidate_count"]),
                "bridge_after_threshold": int(stats["ppr_candidate_count"]),
                "bridge_selected": int(len(candidates)),
                "bridge_mmr_stopped": 0,
                "bridge_no_path": int(stats["ppr_seed_count"] == 0),
                "bridge_top_entities": [],
                "ppr_seed_count": int(stats["ppr_seed_count"]),
                "ppr_subgraph_size": int(stats["ppr_subgraph_size"]),
                "ppr_candidate_count": int(stats["ppr_candidate_count"]),
            }
            max_docs = int(getattr(graph_index.config, "bridge_max_docs", 20))
            return candidates[:max_docs]

        graph_index.graph_evidence = ppr_graph_evidence
        self._ppr_installed = True

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.drf.setdefault(self.d2p[did], 0.0)

    def _decay(self):
        super()._decay()
        floor = float(getattr(self.config, "min_stat", 0.01))
        self.drf = {
            int(p): val * self.drf_decay
            for p, val in self.drf.items()
            if val * self.drf_decay >= floor
        }

    def _credit_serve(self, q_kb, kb_idx):
        super()._credit_serve(q_kb, kb_idx)
        if q_kb.size == 0:
            return
        top_k = min(max(1, self.config.serve_topk), q_kb.shape[1])
        hit = np.max(q_kb, axis=1) >= _P.SF_HIT_THRESH
        for row in q_kb[hit]:
            pos = np.argpartition(row, -top_k)[-top_k:]
            pos = pos[np.argsort(row[pos])[::-1]]
            for rank, p in enumerate(pos, start=1):
                sim = float(row[p])
                if sim <= 0.0:
                    continue
                dist = max(1.0 - sim, 1e-6)
                pi = int(kb_idx[int(p)])
                self.drf[pi] = self.drf.get(pi, 0.0) + 1.0 / (
                    rank * (dist ** self.drf_alpha)
                )

    def _credit_dense(self, candidates, kb_pos, gamma=None, gold_pos=None, top1_bonus=None):
        result = super()._credit_dense(candidates, kb_pos, gamma, gold_pos, top1_bonus)
        for rank, (pi, sim) in enumerate(candidates, start=1):
            pi = int(pi)
            sim = max(0.0, float(sim))
            if sim <= 0.0:
                continue
            dist = max(1.0 - sim, 1e-6)
            self.drf[pi] = self.drf.get(pi, 0.0) + sim / (
                rank * (dist ** self.drf_alpha)
            )
        return result

    def _credit_graph(self, candidates, gold_pos=None):
        result = super()._credit_graph(candidates, gold_pos)
        for rank, (pi, score) in enumerate(candidates, start=1):
            score = float(score)
            if score <= 0.0:
                continue
            pi = int(pi)
            self.drf[pi] = self.drf.get(pi, 0.0) + score / rank
        return result

    def _hubness(self, kb_idx, kb_emb):
        n = int(len(kb_idx))
        if n <= 1 or self.hub_weight <= 0.0:
            return {int(p): 0.0 for p in kb_idx}
        sim = kb_emb @ kb_emb.T
        np.fill_diagonal(sim, -np.inf)
        k = min(max(1, self.hub_k), n - 1)
        nn = np.argpartition(sim, -k, axis=1)[:, -k:]
        counts = np.zeros(n, dtype=np.float64)
        for row in nn:
            counts[row] += 1.0
        return {int(p): float(counts[i]) for i, p in enumerate(kb_idx)}

    def _resident_priority(self, kb_idx, kb_emb):
        priority = super()._resident_priority(kb_idx, kb_emb)
        hub = self._hubness(kb_idx, kb_emb)
        for p in list(priority):
            priority[p] += self.drf_weight * np.log1p(self.drf.get(p, 0.0))
            priority[p] += self.hub_weight * np.log1p(hub.get(p, 0.0))
        return priority


class PPRBridgeEchoDRIPCore(PPRBridgeWriterDRIPCore):
    """Clean mainline PPR bridge cache without DRF, hubness, or Red priority.

    This is the paper-facing variant:

        E_G(q, d) = local-PPR bridge evidence
        D_t(d) = lambda D_{t-1}(d) + E_r(q_t, d)
        A_t(d) = xi A_{t-1}(d) + bridge_echo(q_t, d)
        P_t(d) = S_t(d) + D_t(d) + mu A_t(d)

    ``A_t`` is bridge echo: a decayed trace of repeated PPR bridge evidence for
    a document already in or near the KB. It is not ARC's miss-driven DRF:
    dense candidates never credit it, no hubness is computed, and Red(d) is not
    part of resident priority. The duplicate check remains only as a hard
    admission guard.
    """

    def __init__(
        self,
        *args,
        echo_weight=0.25,
        echo_decay=0.98,
        serve_weight=1.0,
        write_budget_scale=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.echo_weight = float(echo_weight)
        self.echo_decay = float(echo_decay)
        self.serve_weight = float(serve_weight)
        self.write_budget_scale = float(write_budget_scale)
        self.echo = {}

    def _install_ppr_graph_evidence(self):
        if self._ppr_installed or not self.graph_index.has_metadata():
            return
        ppr = self._ensure_ppr()
        graph_index = self.graph_index

        def ppr_graph_evidence(query, first_hops, kb_pos, kb_emb, doc_embs):
            # Include resident docs so repeated bridge paths can refresh A_t(d).
            candidates, stats = ppr.evidence(first_hops, kb_pos=None)
            graph_index.last_stats = {
                "bridge_raw_paths": int(stats["ppr_subgraph_size"]),
                "bridge_after_degree_gate": int(stats["ppr_subgraph_size"]),
                "bridge_after_relation_gate": int(stats["ppr_candidate_count"]),
                "bridge_after_novelty_gate": int(stats["ppr_candidate_count"]),
                "bridge_after_threshold": int(stats["ppr_candidate_count"]),
                "bridge_selected": int(len(candidates)),
                "bridge_mmr_stopped": 0,
                "bridge_no_path": int(stats["ppr_seed_count"] == 0),
                "bridge_top_entities": [],
                "ppr_seed_count": int(stats["ppr_seed_count"]),
                "ppr_subgraph_size": int(stats["ppr_subgraph_size"]),
                "ppr_candidate_count": int(stats["ppr_candidate_count"]),
            }
            max_docs = int(getattr(graph_index.config, "bridge_max_docs", 20))
            return candidates[:max_docs]

        graph_index.graph_evidence = ppr_graph_evidence
        self._ppr_installed = True

    def _decay(self):
        super()._decay()
        floor = float(getattr(self.config, "min_stat", 0.01))
        self.echo = {
            int(p): val * self.echo_decay
            for p, val in self.echo.items()
            if val * self.echo_decay >= floor
        }

    def _credit_serve(self, q_kb, kb_idx):
        super()._credit_serve(q_kb, kb_idx)
        if q_kb.size == 0:
            return
        top_k = min(max(1, self.config.serve_topk), q_kb.shape[1])
        hit = np.max(q_kb, axis=1) >= _P.SF_HIT_THRESH
        for row in q_kb[hit]:
            pos = np.argpartition(row, -top_k)[-top_k:]
            pos = pos[np.argsort(row[pos])[::-1]]
            for rank, p in enumerate(pos, start=1):
                sim = float(row[p])
                if sim <= 0.0:
                    continue
                pi = int(kb_idx[int(p)])
                if pi in self.echo or pi in self._bridge_residency:
                    self.echo[pi] = self.echo.get(pi, 0.0) + sim / rank

    def _credit_graph(self, candidates, gold_pos=None):
        result = super()._credit_graph(candidates, gold_pos)
        for rank, (pi, score) in enumerate(candidates, start=1):
            score = float(score)
            if score <= 0.0:
                continue
            pi = int(pi)
            self.echo[pi] = self.echo.get(pi, 0.0) + score / rank
        return result

    def _resident_priority(self, kb_idx, kb_emb):
        return {
            int(p): self.serve_weight * self.serve.get(int(p), 0.0)
            + self.demand.get(int(p), 0.0)
            + self.echo_weight * np.log1p(self.echo.get(int(p), 0.0))
            for p in kb_idx
        }

    def _write(self, kb_idx, kb_emb, budget, gold_pos=None):
        scaled_budget = int(np.ceil(max(0.0, self.write_budget_scale) * budget))
        return super()._write(kb_idx, kb_emb, min(budget, scaled_budget), gold_pos)


# Backward-compatible alias for notebooks/scripts created during exploration.
PPRBridgeSimpleDRIPCore = PPRBridgeEchoDRIPCore


class PPRSplitAdmissionDRIPCore(PPRBridgeEchoDRIPCore):
    """PPR bridge evidence with separate direct and bridge admission ledgers.

    Main idea:

        D_dir(d) <- direct dense evidence
        D_brg(d) <- local-PPR bridge evidence

    Direct and bridge candidates no longer compete inside one demand ledger.
    Each route gets its own candidate ranking and victim priority, while
    ``self.demand`` is kept as a compatibility aggregate for diagnostics.
    """

    def __init__(
        self,
        *args,
        direct_margin=None,
        bridge_reserve=0.65,
        bridge_margin=0.85,
        **kwargs,
    ):
        super().__init__(
            *args,
            bridge_reserve=bridge_reserve,
            bridge_margin=bridge_margin,
            **kwargs,
        )
        self.direct_margin = direct_margin
        self.direct_demand = {}
        self.bridge_demand = {}

    def _sync_demand(self):
        merged = {}
        for source in (self.direct_demand, self.bridge_demand):
            for p, v in source.items():
                merged[int(p)] = merged.get(int(p), 0.0) + float(v)
        self.demand = merged

    def _prune_map(self, values, cap):
        if cap <= 0 or len(values) <= cap:
            return dict(values)
        return dict(sorted(values.items(), key=lambda item: -item[1])[:cap])

    def _prune_demand(self):
        cap = int(getattr(self.config, "demand_ledger_cap", 0))
        if cap > 0:
            self.direct_demand = self._prune_map(self.direct_demand, cap)
            self.bridge_demand = self._prune_map(self.bridge_demand, cap)
        self._sync_demand()

    def _decay(self):
        super()._decay()
        decay = float(getattr(self.config, "demand_decay", 0.92))
        floor = float(getattr(self.config, "min_stat", 0.01))
        self.direct_demand = {
            int(p): val * decay
            for p, val in self.direct_demand.items()
            if val * decay >= floor
        }
        self.bridge_demand = {
            int(p): val * decay
            for p, val in self.bridge_demand.items()
            if val * decay >= floor
        }
        self._prune_demand()

    def _credit_dense(self, candidates, kb_pos, gamma=None, gold_pos=None, top1_bonus=None):
        """Credit only the direct ledger from dense evidence."""
        updates = 0
        mass = 0.0
        gold_updates = 0
        gold_mass = 0.0
        gamma = self.config.direct_gamma if gamma is None else gamma
        gold_pos = gold_pos or set()
        top1_bonus = self.config.direct_top1_bonus if top1_bonus is None else top1_bonus
        for rank, (pi, sim) in enumerate(candidates):
            pi = int(pi)
            if pi in kb_pos:
                continue
            score = gamma * max(0.0, float(sim))
            if rank == 0:
                score += float(top1_bonus)
            if score <= 0.0:
                continue
            self.direct_demand[pi] = self.direct_demand.get(pi, 0.0) + score
            self.demand[pi] = self.demand.get(pi, 0.0) + score
            updates += 1
            mass += score
            if pi in gold_pos:
                gold_updates += 1
                gold_mass += score
        return updates, mass, gold_updates, gold_mass

    def _credit_graph(self, candidates, gold_pos=None):
        """Credit only the bridge ledger from local-PPR evidence."""
        updates = 0
        mass = 0.0
        gold_updates = 0
        gold_mass = 0.0
        gold_pos = gold_pos or set()
        gain = float(getattr(self.config, "bridge_demand_gain", 1.0))
        for rank, (pi, raw_score) in enumerate(candidates, start=1):
            pi = int(pi)
            raw_score = float(raw_score)
            score = gain * raw_score
            if score <= 0.0:
                continue
            self._bridge_recent.add(pi)
            self.bridge_demand[pi] = self.bridge_demand.get(pi, 0.0) + score
            self.demand[pi] = self.demand.get(pi, 0.0) + score
            self.echo[pi] = self.echo.get(pi, 0.0) + raw_score / rank
            updates += 1
            mass += score
            if pi in gold_pos:
                gold_updates += 1
                gold_mass += score
        return updates, mass, gold_updates, gold_mass

    def _priority_for_route(self, kb_idx, route):
        if route == "bridge":
            return {
                int(p): self.serve_weight * self.serve.get(int(p), 0.0)
                + self.bridge_demand.get(int(p), 0.0)
                + self.echo_weight * np.log1p(self.echo.get(int(p), 0.0))
                + self.bridge_stickiness * self._bridge_residency.get(int(p), 0.0)
                for p in kb_idx
            }
        return {
            int(p): self.serve.get(int(p), 0.0)
            + self.direct_demand.get(int(p), 0.0)
            for p in kb_idx
        }

    def _resident_priority(self, kb_idx, kb_emb):
        return {
            int(p): self.serve_weight * self.serve.get(int(p), 0.0)
            + self.direct_demand.get(int(p), 0.0)
            + self.bridge_demand.get(int(p), 0.0)
            + self.echo_weight * np.log1p(self.echo.get(int(p), 0.0))
            for p in kb_idx
        }

    def _write(self, kb_idx, kb_emb, budget, gold_pos=None):
        gold_pos = gold_pos or set()
        scaled_budget = int(np.ceil(max(0.0, self.write_budget_scale) * budget))
        budget = min(int(budget), scaled_budget)
        if budget <= 0:
            return {
                "writes": 0,
                "candidates": 0,
                "gold_candidates": 0,
                "gold_writes": 0,
                "gold_rate": 0.0,
            }

        kb_pos = set(int(p) for p in kb_idx)
        direct_candidates = sorted(
            ((v, p) for p, v in self.direct_demand.items() if p not in kb_pos),
            reverse=True,
        )
        bridge_candidates = sorted(
            ((v, p) for p, v in self.bridge_demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not direct_candidates and not bridge_candidates:
            return {
                "writes": 0,
                "candidates": 0,
                "gold_candidates": 0,
                "gold_writes": 0,
                "gold_rate": 0.0,
            }

        bridge_budget = int(np.ceil(self.bridge_reserve * budget)) if bridge_candidates else 0
        direct_budget = max(0, budget - bridge_budget)
        if direct_candidates and not bridge_candidates:
            direct_budget = budget

        writes = 0
        gold_writes = 0
        protected_bridge = set()

        def current_kb_emb():
            return self.doc_embs[np.array(sorted(kb_pos), dtype=np.int64)]

        def evict(victim):
            self.kb.discard(self.p2d[victim])
            self.serve.pop(victim, None)
            self._bridge_residency.pop(victim, None)
            kb_pos.discard(victim)

        def admit(cp, route):
            self.kb.add(self.p2d[cp])
            kb_pos.add(cp)
            if route == "bridge":
                self._bridge_residency[cp] = self._bridge_residency.get(cp, 0.0) + 1.0
                protected_bridge.add(cp)

        def admit_from(candidates, cap, route, margin):
            nonlocal writes, gold_writes
            local_writes = 0
            priority = self._priority_for_route(kb_pos, route)
            victim_pool = kb_pos - protected_bridge if route == "direct" else kb_pos
            victims = sorted(victim_pool, key=lambda p: priority.get(p, 0.0))
            victim_i = 0
            for cand_value, cp in candidates:
                cp = int(cp)
                if local_writes >= cap or writes >= budget or victim_i >= len(victims):
                    break
                if cp in kb_pos:
                    continue
                victim = int(victims[victim_i])
                gain = float(cand_value) - margin * priority.get(victim, 0.0)
                if gain <= 0.0:
                    break
                duplicate = float((self.doc_embs[cp] @ current_kb_emb().T).max())
                if duplicate > self.config.tau_duplicate:
                    continue
                evict(victim)
                admit(cp, route)
                victim_i += 1
                writes += 1
                local_writes += 1
                gold_writes += int(cp in gold_pos)

        direct_margin = (
            self.config.gain_margin if self.direct_margin is None else float(self.direct_margin)
        )
        admit_from(bridge_candidates, bridge_budget, "bridge", self.bridge_margin)
        admit_from(direct_candidates, direct_budget, "direct", direct_margin)

        all_candidate_ids = {
            int(p) for _, p in direct_candidates
        } | {
            int(p) for _, p in bridge_candidates
        }
        gold_candidates = sum(1 for p in all_candidate_ids if p in gold_pos)
        return {
            "writes": int(writes),
            "candidates": int(len(all_candidate_ids)),
            "gold_candidates": int(gold_candidates),
            "gold_writes": int(gold_writes),
            "gold_rate": float(gold_writes / writes) if writes else 0.0,
        }
