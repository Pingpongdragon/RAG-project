"""DRIP cache manager 的核心窗口循环。

这个文件才是 DRIP cache manager 的主执行逻辑：

  1. 观察当前 hot cache 对窗口 query 的覆盖情况；
  2. 对 under-covered query 产生 direct / hidden evidence；
  3. 衰减并更新 serve / demand 账本；
  4. 调用子类 writer 决定是否替换 resident 文档。

文件分工：
  - ``drip_config.py``: DRIPCoreConfig 参数；
  - ``dense_index.py``: query-visible dense candidate 检索；
  - ``evidence_router.py``: visible / hidden evidence 路由；
  - ``entity_graph_index.py``: hidden-support 诊断用实体索引；
  - ``evidence_core.py``: DRIPNOdetector / hidden diagnostic 的 evidence 与 writer；
  - ``policies.py``: 对外策略名 ``DRIP`` / ``DRIPNOdetector``。
"""
from collections import Counter, defaultdict

import numpy as np

from algorithms.cache.base import BaseStrategy
from algorithms.cache.params import PARAMS as _P
from algorithms.drip.detection.multi_agent_drift import MultiAgentDriftDetector

from .drip_config import DRIPCoreConfig
from .dense_index import EmbeddingIndex
from .entity_graph_index import GraphIndex
from .evidence_router import QUERY_HIDDEN, QUERY_VISIBLE, QueryRouter, RouteDecision


class DRIPCore(BaseStrategy):
    """旧 DRIPCore：同时支持 query-visible 和 query-hidden evidence channel。"""

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.config = config or DRIPCoreConfig()
        self.router = QueryRouter(self.config)
        self.embedding_index = EmbeddingIndex(self.doc_embs)
        self.graph_index = GraphIndex(self.config, self.d2p, self.doc_pool)
        self.demand = {}
        self.serve = {}
        self.last_admission = {}
        self.bridge_log = []
        self.route_log = []
        self.drift_log = []
        self.drift_detector = None
        self._drift_severity = 0.0
        self._last_drift_result = None
        if self.config.use_drift_detector:
            self.drift_detector = MultiAgentDriftDetector(
                warmup_windows=self.config.drift_warmup_windows,
                min_agent_queries=self.config.drift_min_agent_queries,
                z_threshold=self.config.drift_z_threshold,
                centroid_threshold=self.config.drift_centroid_threshold,
            )

    @property
    def _pool_ents(self):
        return self.graph_index.pool_ents

    @_pool_ents.setter
    def _pool_ents(self, pool_ents):
        self.graph_index.set_pool_entities(pool_ents)

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.serve[self.d2p[did]] = self.config.serve_prior

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return

        kb_idx, kb_emb, nqe, q_kb = self._observe(window_query_embs)
        self._update_drift_control(window_queries, nqe, q_kb, window_idx)
        self._decay()
        self._credit_serve(q_kb, kb_idx)

        kb_pos = set(int(p) for p in kb_idx)
        probe_k = max(self.config.direct_topk, self.config.bridge_step1_k)
        n_under = 0
        under_slots = []
        route_counts = {QUERY_VISIBLE: 0, QUERY_HIDDEN: 0}
        direct_updates = bridge_updates = 0
        direct_mass = bridge_mass = 0.0
        bridge_direct_updates = 0
        bridge_direct_mass = 0.0
        direct_gold_updates = bridge_gold_updates = bridge_direct_gold_updates = 0
        direct_gold_mass = bridge_gold_mass = bridge_direct_gold_mass = 0.0
        window_gold_pos = set()
        route_labeled = route_match = 0
        graph_stats = defaultdict(int)
        graph_entities = Counter()
        bridge_hidden_updates = 0
        bridge_hidden_mass = 0.0
        bridge_hidden_top1 = 0
        bridge_hidden_top5 = 0
        bridge_hidden_top10 = 0
        bridge_hidden_mrr = 0.0
        self._window_hidden_pos = set()
        for query in window_queries:
            self._window_hidden_pos.update(self._hidden_positions(query))

        for qi, query in enumerate(window_queries):
            dense = self.embedding_index.search_one(nqe[qi], probe_k)
            first_hops = dense[: self.config.bridge_step1_k]
            first_hop_entities = [
                self.graph_index.doc_entities(pi)
                for pi, sim in first_hops
                if sim > 0.0
            ]
            route = self.router.route(query, q_kb[qi], first_hop_entities, first_hops)
            if getattr(self, "force_query_visible", False) and route.route == QUERY_HIDDEN:
                route = RouteDecision(
                    QUERY_VISIBLE,
                    route.target_slots,
                    "query_visible_only",
                )
            elif route.route == QUERY_HIDDEN and not self.graph_index.has_metadata():
                route = RouteDecision(
                    QUERY_VISIBLE,
                    route.target_slots,
                    "hidden_no_graph_metadata",
                )
            route_counts[route.route] = route_counts.get(route.route, 0) + 1
            expected_route = self._expected_route(query)
            if expected_route is not None:
                route_labeled += 1
                route_match += int(route.route == expected_route)

            covered = int((q_kb[qi] >= _P.SF_HIT_THRESH).sum())
            if covered >= route.target_slots:
                continue
            n_under += 1
            under_slots.append(route.target_slots)
            gold_pos = self._gold_positions(query)
            hidden_pos = self._hidden_positions(query)
            window_gold_pos.update(gold_pos)

            if route.route == QUERY_HIDDEN and self.graph_index.has_metadata():
                scoring_query = query
                if isinstance(query, dict):
                    scoring_query = dict(query)
                    scoring_query["_drip_query_emb"] = nqe[qi]
                graph_hops = self._bridge_graph_hops(
                    scoring_query, first_hops, dense, q_kb[qi], kb_idx, kb_pos)
                candidates = self.graph_index.graph_evidence(
                    scoring_query, graph_hops, kb_pos, kb_emb, self.doc_embs)
                for key, value in self.graph_index.last_stats.items():
                    if key == "bridge_top_entities":
                        graph_entities.update(dict(value))
                    elif isinstance(value, (int, float)):
                        graph_stats[key] += value
                for rank, (pi, score) in enumerate(candidates, start=1):
                    if int(pi) in hidden_pos:
                        bridge_hidden_updates += 1
                        bridge_hidden_mass += float(score)
                        bridge_hidden_top1 += int(rank <= 1)
                        bridge_hidden_top5 += int(rank <= 5)
                        bridge_hidden_top10 += int(rank <= 10)
                        bridge_hidden_mrr += 1.0 / float(rank)
                updates, mass, gold_updates, gold_mass = self._credit_graph(
                    candidates, gold_pos)
                bridge_updates += updates
                bridge_mass += mass
                bridge_gold_updates += gold_updates
                bridge_gold_mass += gold_mass
                updates, mass, gold_updates, gold_mass = self._credit_dense(
                    first_hops,
                    kb_pos,
                    gamma=self.config.bridge_direct_gamma,
                    gold_pos=gold_pos,
                    top1_bonus=0.0,
                )
                bridge_direct_updates += updates
                bridge_direct_mass += mass
                bridge_direct_gold_updates += gold_updates
                bridge_direct_gold_mass += gold_mass
            else:
                updates, mass, gold_updates, gold_mass = self._credit_dense(
                    dense[: self.config.direct_topk],
                    kb_pos,
                    gold_pos=gold_pos,
                    top1_bonus=self.config.direct_top1_bonus,
                )
                direct_updates += updates
                direct_mass += mass
                direct_gold_updates += gold_updates
                direct_gold_mass += gold_mass

        self.maint_retrieval_cost += n_under * probe_k
        self._prune_demand()
        budget = min(self._effective_write_cap(), sum(under_slots))
        write_stats = self._write(kb_idx, kb_emb, budget, window_gold_pos)
        writes = int(write_stats["writes"])
        self.update_cost += writes

        self.bridge_log.append({
            "w": window_idx,
            "under_covered": int(n_under),
            "bridge_updates": int(bridge_updates),
            "bridge_mass": round(bridge_mass, 4),
            "bridge_gold_updates": int(bridge_gold_updates),
            "bridge_gold_mass": round(bridge_gold_mass, 4),
            "bridge_hidden_updates": int(bridge_hidden_updates),
            "bridge_hidden_mass": round(bridge_hidden_mass, 4),
            "bridge_hidden_top1": int(bridge_hidden_top1),
            "bridge_hidden_top5": int(bridge_hidden_top5),
            "bridge_hidden_top10": int(bridge_hidden_top10),
            "bridge_hidden_mrr": round(bridge_hidden_mrr, 4),
            "bridge_direct_updates": int(bridge_direct_updates),
            "bridge_direct_mass": round(bridge_direct_mass, 4),
            "bridge_direct_gold_updates": int(bridge_direct_gold_updates),
            "bridge_direct_gold_mass": round(bridge_direct_gold_mass, 4),
            "direct_updates": int(direct_updates),
            "direct_mass": round(direct_mass, 4),
            "direct_gold_updates": int(direct_gold_updates),
            "direct_gold_mass": round(direct_gold_mass, 4),
            "graph_raw_paths": int(graph_stats["bridge_raw_paths"]),
            "graph_after_degree_gate": int(graph_stats["bridge_after_degree_gate"]),
            "graph_after_relation_gate": int(graph_stats["bridge_after_relation_gate"]),
            "graph_after_novelty_gate": int(graph_stats["bridge_after_novelty_gate"]),
            "graph_after_threshold": int(graph_stats["bridge_after_threshold"]),
            "graph_selected": int(graph_stats["bridge_selected"]),
            "graph_mmr_stopped": int(graph_stats["bridge_mmr_stopped"]),
            "graph_no_path": int(graph_stats["bridge_no_path"]),
            "graph_top_entities": graph_entities.most_common(5),
            "writes": int(writes),
            "write_candidates": int(write_stats["candidates"]),
            "write_gold_candidates": int(write_stats["gold_candidates"]),
            "write_gold": int(write_stats["gold_writes"]),
            "write_gold_rate": round(float(write_stats["gold_rate"]), 4),
            "write_hidden_candidates": int(write_stats.get("hidden_candidates", 0)),
            "write_hidden": int(write_stats.get("hidden_writes", 0)),
            "write_hidden_rate": round(float(write_stats.get("hidden_rate", 0.0)), 4),
            "hidden_evictions": int(write_stats.get("hidden_evictions", 0)),
            "direct_writes": int(write_stats.get("direct_writes", 0)),
            "bridge_writes": int(write_stats.get("bridge_writes", 0)),
            "direct_budget": int(write_stats.get("direct_budget", 0)),
            "bridge_budget": int(write_stats.get("bridge_budget", 0)),
            "bridge_reserve": round(float(write_stats.get("bridge_reserve", 0.0)), 4),
            "pair_activations": int(write_stats.get("pair_activations", 0)),
            "pair_lease_docs": int(write_stats.get("pair_lease_docs", 0)),
            "pair_lease_mass": round(float(write_stats.get("pair_lease_mass", 0.0)), 4),
        })
        self.route_log.append({
            "w": window_idx,
            "routes": route_counts,
            "route_labeled": int(route_labeled),
            "route_match": int(route_match),
            "under_covered": int(n_under),
            "target_slots": int(sum(under_slots)),
        })
        self.last_admission = {
            "w": window_idx,
            "target_slots": int(sum(under_slots)),
            "under_covered": int(n_under),
            "write_budget": int(budget),
            "writes": int(writes),
            "write_gold": int(write_stats["gold_writes"]),
            "write_gold_candidates": int(write_stats["gold_candidates"]),
            "write_hidden": int(write_stats.get("hidden_writes", 0)),
            "hidden_evictions": int(write_stats.get("hidden_evictions", 0)),
            "drift_severity": round(float(self._drift_severity), 4),
        }

    def _update_drift_control(self, window_queries, nqe, q_kb, window_idx):
        if self.drift_detector is None:
            self._drift_severity = 0.0
            self._last_drift_result = None
            return
        result = self.drift_detector.detect(nqe, q_kb, window_queries)
        self._last_drift_result = result
        self._drift_severity = float(result.severity)
        self.drift_log.append(result.to_log(window_idx))

    def _effective_write_cap(self):
        boost = 1.0 + self.config.drift_write_boost * self._drift_severity
        return int(max(0, np.ceil(float(_P.WRITE_CAP) * boost)))

    def _effective_demand_decay(self):
        decay = self.config.demand_decay * (
            1.0 - self.config.drift_decay_boost * self._drift_severity
        )
        return float(np.clip(decay, 0.0, 1.0))

    def _effective_gain_margin(self, base_margin=None):
        margin = self.config.gain_margin if base_margin is None else float(base_margin)
        discount = 1.0 - self.config.drift_margin_discount * self._drift_severity
        return float(max(0.05, margin * discount))

    def _observe(self, window_query_embs):
        kb_idx = np.array([self.d2p[d] for d in sorted(self.kb)], dtype=np.int64)
        kb_emb = self.doc_embs[kb_idx]
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb = nqe @ kb_emb.T
        return kb_idx, kb_emb, nqe, q_kb

    def _expected_route(self, query):
        if not isinstance(query, dict):
            return None
        qtype = str(
            query.get("route_hint") or query.get("qtype") or query.get("type") or ""
        ).lower()
        if not qtype:
            return None
        if "bridge" in qtype or "compositional" in qtype or "inference" in qtype:
            return QUERY_HIDDEN
        if "comparison" in qtype:
            return QUERY_VISIBLE
        if "single" in qtype or "temporal" in qtype:
            return QUERY_VISIBLE
        return None

    def _bridge_graph_hops(self, query, first_hops, dense, q_kb_row, kb_idx, kb_pos):
        """提供 bridge evidence 的 first-hop seed。

        默认只从 dense first-hop 文档扩展 bridge candidate。子类可以额外加入
        resident anchor，但 direct evidence credit 仍使用原始 ``first_hops``，
        避免 graph seed 改动意外影响 single-hop admission。
        """
        return first_hops

    def _gold_positions(self, query):
        """gold support 只用于实验日志诊断，不参与决策。"""
        if not isinstance(query, dict):
            return set()
        out = set()
        for title in query.get("sf_titles", ()):
            pi = self.title_to_idx.get(title)
            if pi is not None:
                out.add(int(pi))
        return out

    def _hidden_positions(self, query):
        """hidden reuse target 只用于实验日志诊断。"""
        if not isinstance(query, dict):
            return set()
        title = query.get("reuse_support_title")
        if not title:
            return set()
        pi = self.title_to_idx.get(title)
        return {int(pi)} if pi is not None else set()

    def _decay(self):
        d = self._effective_demand_decay()
        s = self.config.serve_decay
        m = self.config.min_stat
        self.demand = {p: v * d for p, v in self.demand.items() if v * d >= m}
        self.serve = {p: v * s for p, v in self.serve.items() if v * s >= m}
        self._prune_demand()

    def _prune_demand(self):
        cap = int(getattr(self.config, "demand_ledger_cap", 0))
        if cap <= 0 or len(self.demand) <= cap:
            return
        keep = sorted(self.demand.items(), key=lambda item: -item[1])[:cap]
        self.demand = dict(keep)

    def _credit_serve(self, q_kb, kb_idx):
        hit = np.max(q_kb, axis=1) >= _P.SF_HIT_THRESH
        if not hit.any():
            return
        k = min(max(1, self.config.serve_topk), q_kb.shape[1])
        for row in q_kb[hit]:
            pos = np.argpartition(row, -k)[-k:]
            pos = [int(p) for p in pos if row[p] >= _P.SF_HIT_THRESH]
            if not pos:
                continue
            credit = 1.0 / len(pos)
            for p in pos:
                pi = int(kb_idx[p])
                self.serve[pi] = self.serve.get(pi, 0.0) + credit
                self._after_serve_credit(pi, credit)

    def _after_serve_credit(self, pi, credit):
        """子类 hook：用于保护刚服务过 query 的 resident evidence。"""
        return None

    def _credit_dense(self, candidates, kb_pos, gamma=None, gold_pos=None, top1_bonus=None):
        """旧公式 1 的 dense 分支：E_dense(q,d) = top1_bonus + max(0, sim(q,d))。"""
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
            self.demand[pi] = self.demand.get(pi, 0.0) + score
            updates += 1
            mass += score
            if pi in gold_pos:
                gold_updates += 1
                gold_mass += score
        return updates, mass, gold_updates, gold_mass

    def _credit_graph(self, candidates, gold_pos=None):
        """旧公式 1 的 graph 分支：GraphIndex 已经计算 E_graph(q,d)。"""
        updates = 0
        mass = 0.0
        gold_updates = 0
        gold_mass = 0.0
        gold_pos = gold_pos or set()
        gain = float(getattr(self.config, "bridge_demand_gain", 1.0))
        for pi, score in candidates:
            pi = int(pi)
            score = gain * float(score)
            if score <= 0.0:
                continue
            self.demand[pi] = self.demand.get(pi, 0.0) + score
            updates += 1
            mass += score
            if pi in gold_pos:
                gold_updates += 1
                gold_mass += score
        return updates, mass, gold_updates, gold_mass

    def _resident_priority(self, kb_idx, kb_emb):
        kb_self = kb_emb @ kb_emb.T
        np.fill_diagonal(kb_self, -1.0)
        red = kb_self.max(axis=1)
        red_map = {int(p): float(red[i]) for i, p in enumerate(kb_idx)}

        def priority(p):
            base = self.serve.get(p, 0.0) + self.demand.get(p, 0.0)
            penalty = self.config.redundancy_penalty * max(
                0.0,
                red_map.get(p, 0.0) - self.config.redundancy_threshold,
            )
            return base - penalty

        return {int(p): priority(int(p)) for p in kb_idx}

    def _write(self, kb_idx, kb_emb, budget, gold_pos=None):
        """旧公式 2 admission：candidate 的 D_t(c) 要超过 victim priority。"""
        gold_pos = gold_pos or set()
        hidden_pos = getattr(self, "_window_hidden_pos", set())
        if budget <= 0:
            return {
                "writes": 0,
                "candidates": 0,
                "gold_candidates": 0,
                "gold_writes": 0,
                "gold_rate": 0.0,
                "hidden_candidates": 0,
                "hidden_writes": 0,
                "hidden_rate": 0.0,
                "hidden_evictions": 0,
            }
        kb_pos = set(int(p) for p in kb_idx)
        candidates = sorted(
            ((v, p) for p, v in self.demand.items() if p not in kb_pos),
            reverse=True,
        )
        if not candidates:
            return {
                "writes": 0,
                "candidates": 0,
                "gold_candidates": 0,
                "gold_writes": 0,
                "gold_rate": 0.0,
                "hidden_candidates": 0,
                "hidden_writes": 0,
                "hidden_rate": 0.0,
                "hidden_evictions": 0,
            }

        priority = self._resident_priority(kb_idx, kb_emb)
        victims = sorted(kb_pos, key=lambda p: priority[p])
        current_kb_emb = self.doc_embs[np.array(sorted(kb_pos), dtype=np.int64)]
        writes = 0
        gold_writes = 0
        hidden_writes = 0
        hidden_evictions = 0
        victim_i = 0

        for cand_value, cp in candidates:
            if writes >= budget or victim_i >= len(victims):
                break
            victim = victims[victim_i]
            gain = cand_value - self._effective_gain_margin() * priority[victim]
            if gain <= 0.0:
                break
            duplicate = float((self.doc_embs[cp] @ current_kb_emb.T).max())
            if duplicate > self.config.tau_duplicate:
                continue
            hidden_evictions += int(int(victim) in hidden_pos)
            self.kb.discard(self.p2d[victim])
            self.kb.add(self.p2d[cp])
            self.serve.pop(victim, None)
            victim_i += 1
            writes += 1
            gold_writes += int(int(cp) in gold_pos)
            hidden_writes += int(int(cp) in hidden_pos)
        gold_candidates = sum(1 for _, p in candidates if int(p) in gold_pos)
        hidden_candidates = sum(1 for _, p in candidates if int(p) in hidden_pos)
        return {
            "writes": int(writes),
            "candidates": int(len(candidates)),
            "gold_candidates": int(gold_candidates),
            "gold_writes": int(gold_writes),
            "gold_rate": float(gold_writes / writes) if writes else 0.0,
            "hidden_candidates": int(hidden_candidates),
            "hidden_writes": int(hidden_writes),
            "hidden_rate": float(hidden_writes / writes) if writes else 0.0,
            "hidden_evictions": int(hidden_evictions),
        }
