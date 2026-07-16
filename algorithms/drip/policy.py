"""DRIP：连续 domain adaptation 与 evidence-level hot-tier placement。

这是论文主方法的唯一实现。每个窗口依次执行：

1. 当前 query 直接路由到少量冷库分区；
2. 只在所选分区内精排具体文档；
3. 已完成请求的 evidence 更新 HSU/MEF document utility；
4. candidate 与最低效用 resident 做受预算约束的集合交换。

主方法不预测下一 topic，也不让 topic label 直接决定驻留；历史 prior 与
window-topic placement 只保留为显式消融开关。

在线策略不读取未来 query 或离线 regime 标签。exact-access
实验可在请求完成后提供 access key；oracle evidence-demand trace 可在计分后提供
gold support key，二者都只能影响下一窗口。
"""

import math

import numpy as np

from algorithms.cache.base import BaseStrategy
from algorithms.cache.params import PARAMS as _P

from .config import DRIPConfig
from .controller import PrimalDualController
from .domain_adaptation import DomainAdapter
from .index import DenseIndex
from .topic_dynamics import SoftTopicDynamics
from .topic_partition import build_topic_partition


class DRIP(BaseStrategy):
    """Evidence-only DRIP 及其显式 TopicDynamics 扩展。"""

    method_version = "drip-query-adaptive-feedback-v2"

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.config = config or DRIPConfig.from_env()
        self.index = DenseIndex(self.doc_embs)
        self._unit_doc_embs = self._normalize_rows(self.doc_embs)
        self.dual = PrimalDualController(
            self.config.replacement_target_rate,
            self.config.initial_dual_price,
        )
        self.topic_partition = None
        self.topic_dynamics = None
        self.domain_partition = None
        self.domain_adapter = None
        document_ids = [document["doc_id"] for document in self.doc_pool]
        if self.config.domain_adaptation_enabled:
            metadata_field = str(self.config.domain_metadata_field or "")
            if metadata_field:
                missing = [
                    index for index, document in enumerate(self.doc_pool)
                    if document.get(metadata_field) is None
                ]
                if missing:
                    raise ValueError(
                        f"domain metadata field {metadata_field!r} is missing "
                        f"from {len(missing)} documents"
                    )
                self.domain_partition = build_topic_partition(
                    "metadata",
                    document_labels=[
                        document[metadata_field] for document in self.doc_pool
                    ],
                    document_ids=document_ids,
                )
            else:
                self.domain_partition = build_topic_partition(
                    "semantic",
                    document_embeddings=self.doc_embs,
                    n_topics=int(self.config.domain_semantic_topics),
                    top_m=int(self.config.domain_soft_memberships),
                    temperature=float(
                        self.config.domain_partition_temperature
                    ),
                    assignment_metric=str(
                        self.config.domain_partition_metric
                    ),
                    random_state=int(_P.SEED),
                    document_ids=document_ids,
                )
            self.domain_adapter = DomainAdapter(
                self.domain_partition,
                self.doc_embs,
                prior_rate=float(self.config.domain_prior_rate),
                reliability_rate=float(
                    self.config.domain_reliability_rate
                ),
                prior_weight=float(self.config.domain_prior_weight),
                route_width=int(self.config.domain_route_width),
                retrieve_topk=int(self.config.domain_retrieve_topk),
                candidate_budget=int(self.config.domain_candidate_budget),
            )
        if self.config.topic_dynamics_enabled:
            metadata_field = str(self.config.topic_metadata_field or "")
            if metadata_field:
                missing = [
                    index for index, document in enumerate(self.doc_pool)
                    if document.get(metadata_field) is None
                ]
                if missing:
                    raise ValueError(
                        f"topic metadata field {metadata_field!r} is missing "
                        f"from {len(missing)} documents"
                    )
                self.topic_partition = build_topic_partition(
                    "metadata",
                    document_labels=[
                        document[metadata_field] for document in self.doc_pool
                    ],
                    document_ids=document_ids,
                )
            elif int(self.config.topic_semantic_topics) > 0:
                self.topic_partition = build_topic_partition(
                    "semantic",
                    document_embeddings=self.doc_embs,
                    n_topics=int(self.config.topic_semantic_topics),
                    top_m=int(self.config.topic_soft_memberships),
                    temperature=float(
                        self.config.topic_partition_temperature
                    ),
                    assignment_metric=str(
                        self.config.topic_partition_metric
                    ),
                    random_state=int(_P.SEED),
                    document_ids=document_ids,
                )
            else:
                raise ValueError(
                    "TopicDynamics requires topic_metadata_field or "
                    "topic_semantic_topics"
                )
            self.topic_dynamics = SoftTopicDynamics(
                self.topic_partition,
                drift_reference_rate=float(
                    self.config.topic_drift_reference_rate
                ),
                drift_slack=float(self.config.topic_drift_slack),
                drift_threshold=float(self.config.topic_drift_threshold),
                transition_decay=float(self.config.topic_transition_decay),
                document_decay=float(self.config.topic_document_decay),
                min_transition_support=float(
                    self.config.topic_min_transition_support
                ),
                min_forecast_confidence=float(
                    self.config.topic_min_forecast_confidence
                ),
            )

        self.demand = {}
        self.topic_demand = {}
        self.serve = {}
        self.cost_log = []
        self.prefetch_log = []
        self.topic_log = []
        self.domain_log = []
        self.downstream_log = []
        self._domain_routed_window = None
        self._domain_routed_window_idx = None
        self.speculative_residents = set()
        self._topic_pending_candidates = ()
        self._topic_pending_strength = 0.0
        self._topic_promotions_current = 0
        self.last_topic_decision = None
        self.last_admission = {}
        self.total_evictions = 0
        self._active_window_idx = -1

    def set_kb(self, ids):
        super().set_kb(ids)
        self.speculative_residents.clear()
        self.serve.clear()
        for doc_id in self.kb:
            pool_idx = int(self.d2p[doc_id])
            self.serve[pool_idx] = float(self.config.serve_prior)

    def prepare_window(self, window_queries, window_query_embs, window_idx):
        """Route current queries using only the prior from completed windows."""

        if self.domain_adapter is None:
            return
        self._domain_routed_window = self.domain_adapter.route(
            window_query_embs
        )
        self._domain_routed_window_idx = int(window_idx)
        self.serve_retrieval_cost += int(
            len(self._domain_routed_window.unique_documents)
        )

    def get_effective_kb(self, window_queries, window_query_embs):
        """Return persistent hot documents plus current routed cold fetches."""

        if self._domain_routed_window is None:
            return self.kb
        return self.kb | {
            self.p2d[int(position)]
            for position in self._domain_routed_window.unique_documents
        }

    def get_effective_kb_for_query(self, query_index):
        """Return the hot tier plus candidates routed for one current request.

        Routing is computed independently from that request's embedding.  The
        window-level method above remains for backwards compatibility, but an
        end-to-end evaluator should use this method so candidates belonging to
        other requests in the same batch cannot improve the current request.
        """

        routed = self._domain_routed_window
        index = int(query_index)
        if routed is None or index < 0 or index >= len(routed.queries):
            return self.kb
        return self.kb | {
            self.p2d[int(position)]
            for position in routed.queries[index].documents
        }

    def _ensure_domain_route(self, window_query_embs, window_idx):
        if self.domain_adapter is None:
            return None
        if (
            self._domain_routed_window is None
            or self._domain_routed_window_idx != int(window_idx)
        ):
            self.prepare_window((), window_query_embs, window_idx)
        return self._domain_routed_window

    def step(self, window_queries, window_query_embs, window_idx):
        """用窗口 ``Q_t`` 更新只能服务未来窗口的 ``K_{t+1}``。"""

        if not self.kb:
            return
        self._active_window_idx = int(window_idx)
        self._decay_ledgers()
        routed_window = self._ensure_domain_route(
            window_query_embs, window_idx
        )

        kb_idx = np.asarray(
            [self.d2p[doc_id] for doc_id in sorted(self.kb)],
            dtype=np.int64,
        )
        kb_pos = set(int(value) for value in kb_idx)
        # exact item/evidence trace 只在请求服务完成后暴露 access key。更新替换状态时
        # 不必额外执行 O(|Q||K|d) 的语义缓存扫描；普通 QA 仍走原来的 dense 路径。
        observed_accesses = self._observed_access_positions(window_queries)
        if observed_accesses is None:
            kb_emb = self._unit_doc_embs[kb_idx]
            query_embs = self._normalize_rows(window_query_embs)
            query_cache_scores = query_embs @ kb_emb.T
        else:
            query_embs = None
            query_cache_scores = None

        support_trace = []
        misses = 0
        evidence_updates = 0
        evidence_mass = 0.0
        maintenance_reads = 0
        downstream_queries = 0
        downstream_updates = 0
        downstream_mass = 0.0
        downstream_write_pressure = 0

        for row_index, query in enumerate(window_queries):
            feedback = self._credit_downstream_feedback(query, kb_pos)
            if feedback[0] > 0:
                downstream_queries += 1
                downstream_updates += int(feedback[0])
                downstream_mass += float(feedback[1])
                downstream_write_pressure += int(feedback[3])
            access_pos = (
                observed_accesses[row_index]
                if observed_accesses is not None
                else self._observed_access_position(query)
            )
            if access_pos is not None:
                support_trace.append(
                    int(feedback[2]) if feedback[2] is not None
                    else int(access_pos)
                )
                if int(access_pos) in kb_pos:
                    if feedback[0] == 0:
                        self._credit_exact_hit(int(access_pos))
                else:
                    misses += 1
                    maintenance_reads += 1
                    if feedback[0] == 0:
                        self.demand[int(access_pos)] = (
                            self.demand.get(int(access_pos), 0.0) + 1.0
                        )
                        evidence_updates += 1
                        evidence_mass += 1.0
                    if feedback[0] > 0:
                        continue
                    if routed_window is not None and row_index < len(
                        routed_window.queries
                    ):
                        routed = routed_window.queries[row_index]
                        routed_candidates = list(zip(
                            routed.documents, routed.scores
                        ))
                        updates, mass = self._credit_miss(
                            routed_candidates, kb_pos
                        )
                        evidence_updates += updates
                        evidence_mass += mass
                continue

            cache_scores = query_cache_scores[row_index]
            if cache_scores.size and float(cache_scores.max()) >= _P.SF_HIT_THRESH:
                support_trace.append(
                    int(feedback[2]) if feedback[2] is not None
                    else int(kb_idx[int(np.argmax(cache_scores))])
                )
                self._credit_semantic_hit(cache_scores, kb_idx)
                continue

            misses += 1
            if routed_window is not None and row_index < len(
                routed_window.queries
            ):
                routed = routed_window.queries[row_index]
                candidates = list(zip(routed.documents, routed.scores))
            else:
                candidates = self.index.search(
                    query_embs[row_index], self.config.direct_topk
                )
                maintenance_reads += len(candidates)
            if feedback[0] > 0:
                support_trace.append(int(feedback[2]))
            else:
                if candidates:
                    support_trace.append(int(candidates[0][0]))
                updates, mass = self._credit_miss(candidates, kb_pos)
                evidence_updates += updates
                evidence_mass += mass

        self._topic_promotions_current = len({
            int(pool_idx) for pool_idx in support_trace
            if int(pool_idx) in self.speculative_residents
        })
        for pool_idx in support_trace:
            self.speculative_residents.discard(int(pool_idx))

        domain_stats = self._update_domain_adaptation(support_trace)

        topic_stats = self._topic_dynamics_prefetch(support_trace, kb_pos)
        self.prefetch_log.append({
            "w": int(window_idx),
            **topic_stats,
            **domain_stats,
        })
        # TopicDynamics 候选来自内存中的 topic->document 表，只是点查询，不计为
        # cold index 检索或冷文档读取。
        self.maint_retrieval_cost += int(maintenance_reads)
        self._prune_demand()

        candidate_pos = {
            int(pool_idx)
            for pool_idx, value in {
                **self.demand, **self.topic_demand
            }.items()
            if int(pool_idx) not in kb_pos and float(value) > 0.0
        }
        node_utility = self._node_utility(kb_pos, candidate_pos)
        write_budget = min(
            int(_P.WRITE_CAP),
            max(int(misses), int(downstream_write_pressure))
            + int(topic_stats["topic_write_allowance"]),
        )
        write_stats = self._write(
            kb_pos,
            candidate_pos,
            node_utility,
            write_budget,
            speculative_candidates=self._topic_pending_candidates,
            speculative_budget=int(topic_stats["topic_write_allowance"]),
        )
        writes = int(write_stats["writes"])
        self.update_cost += writes

        self.last_admission = {
            "w": int(window_idx),
            "under_covered": int(misses),
            "write_budget": int(write_budget),
            "writes": writes,
            "evidence_updates": int(evidence_updates),
            "evidence_mass": round(float(evidence_mass), 6),
            "maintenance_reads": int(maintenance_reads),
            **topic_stats,
            **domain_stats,
            "downstream_feedback_queries": int(downstream_queries),
            "downstream_feedback_updates": int(downstream_updates),
            "downstream_feedback_mass": round(float(downstream_mass), 6),
            "downstream_write_pressure": int(downstream_write_pressure),
            "utility": (
                "downstream_feedback"
                if downstream_queries else "evidence_only"
            ),
            "dual_price": round(float(self.dual.price), 6),
        }

    @staticmethod
    def _normalize_rows(values):
        values = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.clip(norms, 1e-10, None)

    def _decay_ledgers(self):
        floor = float(self.config.min_stat)
        self.demand = {
            int(pool_idx): float(value) * self.config.demand_decay
            for pool_idx, value in self.demand.items()
            if float(value) * self.config.demand_decay >= floor
        }
        self.serve = {
            int(pool_idx): float(value) * self.config.serve_decay
            for pool_idx, value in self.serve.items()
            if float(value) * self.config.serve_decay >= floor
        }
        self.topic_demand = {
            int(pool_idx): float(value) * self.config.topic_document_decay
            for pool_idx, value in self.topic_demand.items()
            if float(value) * self.config.topic_document_decay >= floor
        }

    def _prune_demand(self):
        cap = int(self.config.demand_ledger_cap)
        if cap > 0 and len(self.demand) > cap:
            self.demand = dict(sorted(
                self.demand.items(), key=lambda item: -item[1]
            )[:cap])
        if cap > 0 and len(self.topic_demand) > cap:
            self.topic_demand = dict(sorted(
                self.topic_demand.items(), key=lambda item: -item[1]
            )[:cap])

    def _update_domain_adaptation(self, support_trace):
        """Update the domain prior only after the current window was served."""

        disabled = {
            "domain_enabled": False,
            "domain_count": 0,
            "domain_current": None,
            "domain_prior_top": None,
            "domain_prior_shift_l1": 0.0,
            "domain_reliability": 0.0,
            "domain_adaptation_strength": 0.0,
            "domain_regions_per_query": 0,
            "domain_documents_scanned": 0,
            "domain_documents_fetched": 0,
            "domain_unique_fetches": 0,
        }
        if self.domain_adapter is None or self.domain_partition is None:
            return disabled

        routed = self._domain_routed_window
        prior_before = self.domain_adapter.prior.copy()
        if support_trace:
            histogram = self.domain_partition.topic_histogram(
                support_trace, soft=True
            )
            prior_after = self.domain_adapter.observe(support_trace)
            current = int(np.argmax(histogram))
        else:
            histogram = np.zeros(
                self.domain_partition.n_topics, dtype=np.float64
            )
            prior_after = prior_before
            current = None
        stats = {
            "domain_enabled": True,
            "domain_count": int(self.domain_partition.n_topics),
            "domain_current": current,
            "domain_prior_top": int(np.argmax(prior_after)),
            "domain_prior_shift_l1": round(
                float(np.abs(prior_after - prior_before).sum()), 6
            ),
            "domain_reliability": round(
                float(self.domain_adapter.reliability), 6
            ),
            "domain_adaptation_strength": round(
                float(self.domain_adapter.adaptation_strength), 6
            ),
            "domain_regions_per_query": int(
                self.config.domain_route_width
            ),
            "domain_documents_scanned": int(
                0 if routed is None else routed.scanned_documents
            ),
            "domain_documents_fetched": int(
                0 if routed is None else routed.fetched_documents
            ),
            "domain_unique_fetches": int(
                0 if routed is None else len(routed.unique_documents)
            ),
        }
        self.domain_log.append({
            "w": int(self._active_window_idx),
            **stats,
            "domain_observed_distribution": [
                round(float(value), 6) for value in histogram
            ],
            "domain_prior_before": [
                round(float(value), 6) for value in prior_before
            ],
            "domain_prior_after": [
                round(float(value), 6) for value in prior_after
            ],
        })
        return stats

    def _topic_dynamics_prefetch(self, support_trace, kb_pos):
        """检测 evidence-topic 漂移并预测下一窗口的 soft topic mixture。

        该函数只把预测转成一个有界的候选接口；最终 replacement 仍由 ``_write``
        完成。候选只能来自已完成窗口中真实出现过的文档，不能从 topic label 直接
        猜测从未访问过的文档。
        """

        self._topic_pending_candidates = ()
        self._topic_pending_strength = 0.0
        self.last_topic_decision = None
        disabled = {
            "topic_count": 0,
            "topic_current": None,
            "topic_predicted": None,
            "topic_drift_score": 0.0,
            "topic_drift_cusum": 0.0,
            "topic_drift_alarm": False,
            "topic_transition_support": 0.0,
            "topic_forecast_confidence": 0.0,
            "topic_previous_forecast_similarity": None,
            "topic_previous_document_recall": None,
            "topic_previous_document_precision": None,
            "topic_activation": 0.0,
            "topic_placement_enabled": bool(
                self.config.topic_apply_forecast_to_cache
            ),
            "topic_proposals_ranked": 0,
            "topic_proposals_materialized": 0,
            "topic_added_mass": 0.0,
            "topic_write_allowance": 0,
            "topic_promotions": int(self._topic_promotions_current),
        }
        if (
            self.topic_partition is None
            or self.topic_dynamics is None
            or not support_trace
        ):
            return disabled

        histogram = self.topic_partition.topic_histogram(
            support_trace, soft=True
        )
        if float(histogram.sum()) <= 0.0:
            return disabled

        selection = self.topic_dynamics.observe_and_forecast(
            histogram,
            support_trace,
            int(self.config.topic_candidate_budget),
            exclude=kb_pos,
        )
        self.last_topic_decision = selection

        # 预测只在显式 change alarm 后获得一次 placement lease。稳定窗口继续由
        # evidence-only reactive path 维护，避免仅凭 topic persistence 持续预取。
        activation = (
            min(1.0, float(selection.forecast_confidence))
            if (
                self.config.topic_apply_forecast_to_cache
                and selection.drift_alarm
                and selection.predicted_topic is not None
            )
            else 0.0
        )

        active_candidates = int(math.ceil(
            activation * len(selection.documents)
        ))
        active_documents = selection.documents[:active_candidates]
        active_scores = selection.scores[:active_candidates]
        self._topic_pending_candidates = tuple(
            int(position) for position in active_documents
        )
        self._topic_pending_strength = float(activation)
        score_mass = float(sum(active_scores))
        added_mass = 0.0
        if activation > 0.0 and score_mass > 0.0:
            added_mass = (
                float(self.config.topic_forecast_mass) * activation
            )
            for pool_idx, score in zip(
                active_documents, active_scores
            ):
                if int(pool_idx) in kb_pos or float(score) <= 0.0:
                    continue
                self.topic_demand[int(pool_idx)] = (
                    self.topic_demand.get(int(pool_idx), 0.0)
                    + added_mass * float(score) / score_mass
                )

        write_allowance = min(int(_P.WRITE_CAP), active_candidates)
        stats = {
            "topic_count": int(self.topic_partition.n_topics),
            "topic_current": int(np.argmax(histogram)),
            "topic_predicted": (
                None if selection.predicted_topic is None
                else int(selection.predicted_topic)
            ),
            "topic_drift_score": round(
                float(selection.drift_score), 6
            ),
            "topic_drift_cusum": round(
                float(selection.drift_cusum), 6
            ),
            "topic_drift_alarm": bool(selection.drift_alarm),
            "topic_transition_support": round(
                float(selection.transition_support), 6
            ),
            "topic_forecast_confidence": round(
                float(selection.forecast_confidence), 6
            ),
            "topic_previous_forecast_similarity": (
                None
                if selection.previous_forecast_similarity is None
                else round(
                    float(selection.previous_forecast_similarity), 6
                )
            ),
            "topic_previous_document_recall": (
                None
                if selection.previous_document_recall is None
                else round(float(selection.previous_document_recall), 6)
            ),
            "topic_previous_document_precision": (
                None
                if selection.previous_document_precision is None
                else round(float(selection.previous_document_precision), 6)
            ),
            "topic_activation": round(float(activation), 6),
            "topic_placement_enabled": bool(
                self.config.topic_apply_forecast_to_cache
            ),
            "topic_proposals_ranked": int(len(selection.documents)),
            "topic_proposals_materialized": int(active_candidates),
            "topic_added_mass": round(float(added_mass), 6),
            "topic_write_allowance": int(write_allowance),
            "topic_promotions": int(self._topic_promotions_current),
        }
        self.topic_log.append({
            "w": int(self._active_window_idx),
            **stats,
            "topic_histogram": [round(float(value), 6) for value in histogram],
            "topic_predicted_distribution": [
                round(float(value), 6)
                for value in selection.predicted_distribution
            ],
        })
        return stats

    def _credit_exact_hit(self, pool_idx):
        self.serve[pool_idx] = self.serve.get(pool_idx, 0.0) + 1.0

    def _credit_semantic_hit(self, cache_scores, kb_idx):
        topk = min(max(1, int(self.config.serve_topk)), len(kb_idx))
        columns = np.argpartition(cache_scores, -topk)[-topk:]
        columns = [
            int(column)
            for column in columns
            if float(cache_scores[int(column)]) >= _P.SF_HIT_THRESH
        ]
        if columns:
            credit = 1.0 / float(len(columns))
            for column in columns:
                pool_idx = int(kb_idx[column])
                self.serve[pool_idx] = (
                    self.serve.get(pool_idx, 0.0) + credit
                )

    def _credit_miss(self, candidates, kb_pos):
        raw = []
        alpha = float(self.config.direct_evidence_alpha)
        epsilon = max(1e-9, float(self.config.direct_evidence_epsilon))
        for rank, (pool_idx, similarity) in enumerate(candidates, start=1):
            pool_idx = int(pool_idx)
            similarity = max(0.0, float(similarity))
            if pool_idx in kb_pos or similarity <= 0.0:
                continue
            distance = epsilon + max(0.0, 1.0 - similarity)
            score = similarity / (float(rank) * distance ** alpha)
            if rank == 1:
                score += float(self.config.direct_top1_bonus)
            if score > 0.0:
                raw.append((pool_idx, float(score)))

        normalizer = float(sum(score for _, score in raw))
        if normalizer <= 0.0:
            return 0, 0.0
        for pool_idx, score in raw:
            posterior = float(score / normalizer)
            self.demand[pool_idx] = (
                self.demand.get(pool_idx, 0.0) + posterior
            )
        return len(raw), 1.0

    def _credit_downstream_feedback(self, query, kb_pos):
        """Credit post-service current-request utility, never stream history.

        Returns ``(updates, mass, top_position, has_nonresident)``.  The event
        schema is a list of ``{title, utility}`` mappings created only after the
        request has been served. Invalid or unknown documents are ignored.
        """

        mass = float(self.config.downstream_feedback_mass)
        raw = query.get("downstream_feedback") if isinstance(query, dict) else None
        if mass <= 0.0 or not isinstance(raw, (list, tuple)):
            return 0, 0.0, None, False
        entries = []
        threshold = float(self.config.downstream_feedback_min_utility)
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            position = self.title_to_idx.get(title)
            try:
                utility = float(item.get("utility", 0.0))
            except (TypeError, ValueError):
                continue
            if (
                position is None or not np.isfinite(utility)
                or utility <= threshold
            ):
                continue
            entries.append((int(position), utility, str(item.get("source", ""))))
        entries.sort(key=lambda value: (-value[1], value[0]))
        entries = entries[: int(self.config.downstream_feedback_topk)]
        normalizer = float(sum(value for _, value, _ in entries))
        if normalizer <= 0.0:
            return 0, 0.0, None, False

        has_nonresident = False
        for position, value, _ in entries:
            credit = mass * float(value) / normalizer
            if position in kb_pos:
                self.serve[position] = self.serve.get(position, 0.0) + credit
            else:
                self.demand[position] = self.demand.get(position, 0.0) + credit
                has_nonresident = True
        self.downstream_log.append({
            "w": int(self._active_window_idx),
            "documents": int(len(entries)),
            "mass": round(float(mass), 6),
            "top_position": int(entries[0][0]),
            "top_utility": round(float(entries[0][1]), 6),
            "has_nonresident": bool(has_nonresident),
            "sources": sorted({source for _, _, source in entries if source}),
        })
        return len(entries), mass, int(entries[0][0]), has_nonresident

    def _node_utility(self, kb_pos, candidate_pos):
        """Return a single normalized evidence utility, with no LRU expert.

        HSU and MEF both distribute one unit of evidence mass per served query;
        TopicDynamics contributes only after an explicit drift alarm.  Dividing
        by the current maximum keeps the shadow-price scale stable without
        introducing another learned mixture.
        """

        universe = set(kb_pos) | set(candidate_pos)
        evidence_raw = {
            int(pool_idx): (
                float(self.serve.get(pool_idx, 0.0))
                + float(self.demand.get(pool_idx, 0.0))
                + float(self.topic_demand.get(pool_idx, 0.0))
            )
            for pool_idx in universe
        }
        evidence_scale = max(evidence_raw.values(), default=0.0)
        evidence_utility = {
            int(pool_idx): (
                float(value) / float(evidence_scale)
                if evidence_scale > 0.0 else 0.0
            )
            for pool_idx, value in evidence_raw.items()
        }

        return evidence_utility

    def _duplicates_retained_cache(self, candidate, retained):
        retained = sorted(int(value) for value in retained)
        if not retained:
            return False
        similarity = float((
            self._unit_doc_embs[int(candidate)]
            @ self._unit_doc_embs[np.asarray(retained, dtype=np.int64)].T
        ).max())
        return similarity > float(self.config.duplicate_threshold)

    def _write(
        self,
        kb_pos,
        candidates,
        utility,
        budget,
        *,
        speculative_candidates=(),
        speculative_budget=0,
    ):
        price = float(self.dual.price)
        writes = 0
        speculative_writes = 0
        reactive_net_gains = []
        speculative_activations = []
        desired_speculative = tuple(dict.fromkeys(
            int(position) for position in speculative_candidates
        ))
        desired_set = set(desired_speculative)
        max_speculative = int(math.ceil(
            float(self.config.topic_max_cache_fraction) * len(kb_pos)
        ))
        allowed_speculative_writes = min(
            int(budget),
            max(0, int(speculative_budget)),
            max(0, max_speculative),
        )

        # 临时的有界 placement adapter。TopicDynamics 只负责候选与置信度；这里的
        # speculative victim rule 仍是可替换接口，不把它宣称为最终 placement 方法。
        # 写入由容量、占比和写入次数三项硬上限控制，
        # 不使用下方 reactive replacement 的效用阈值。预取文档若在下一窗口命中，
        # 会在下次更新开始时转为普通 resident；长期未命中的预取 resident 优先淘汰。
        # 主动预取与 reactive replacement 共用总缓存容量和窗口写入上限。
        for candidate in desired_speculative:
            if (
                speculative_writes >= allowed_speculative_writes
                or not kb_pos
            ):
                break
            if candidate in kb_pos:
                continue
            live_speculative = self.speculative_residents & kb_pos
            stale_speculative = live_speculative - desired_set
            if stale_speculative:
                victim_pool = stale_speculative
            elif len(live_speculative) < max_speculative:
                victim_pool = kb_pos - live_speculative
            else:
                victim_pool = live_speculative
            if not victim_pool:
                break
            victim = min(
                victim_pool,
                key=lambda pool_idx: (
                    float(utility.get(pool_idx, 0.0)), int(pool_idx)
                ),
            )
            self.kb.discard(self.p2d[victim])
            self.serve.pop(victim, None)
            self.speculative_residents.discard(victim)
            kb_pos.remove(victim)

            self.kb.add(self.p2d[candidate])
            self.speculative_residents.add(candidate)
            kb_pos.add(candidate)
            writes += 1
            speculative_writes += 1
            speculative_activations.append(
                float(self._topic_pending_strength)
            )

        ordered = sorted(
            candidates,
            key=lambda pool_idx: (
                -float(utility.get(pool_idx, 0.0)), int(pool_idx)
            ),
        )

        for candidate in ordered:
            if writes >= int(budget) or not kb_pos:
                break
            if candidate in kb_pos:
                continue
            victim = min(
                kb_pos,
                key=lambda pool_idx: (
                    float(utility.get(pool_idx, 0.0)), int(pool_idx)
                ),
            )
            retained = kb_pos - {victim}
            if self._duplicates_retained_cache(candidate, retained):
                continue
            net_gain = (
                float(utility.get(candidate, 0.0))
                - float(utility.get(victim, 0.0))
                - price
            )
            if net_gain <= 0.0:
                continue

            self.kb.discard(self.p2d[victim])
            self.serve.pop(victim, None)
            self.speculative_residents.discard(victim)
            kb_pos.remove(victim)

            self.kb.add(self.p2d[candidate])
            kb_pos.add(candidate)
            writes += 1
            reactive_net_gains.append(float(net_gain))

        # 对偶控制器观测总写入负载，但当前影子价格只约束上面的剩余 reactive swap。
        # 这样会为通过门控的 TopicDynamics 分支保留一小部分探索写入额度。
        dual_feedback = self.dual.update(writes, budget)
        kb_size = max(1, len(self.kb))
        reactive_avg_net_gain = (
            float(np.mean(reactive_net_gains))
            if reactive_net_gains else 0.0
        )
        mean_speculative_activation = (
            float(np.mean(speculative_activations))
            if speculative_activations else 0.0
        )
        stats = {
            "writes": int(writes),
            "candidates": int(len(candidates)),
            "evictions": int(writes),
            "churn_rate": round(float(writes / kb_size), 6),
            "replacement_penalty": round(price, 6),
            "reactive_avg_net_gain": round(reactive_avg_net_gain, 6),
            "mean_speculative_activation": round(
                mean_speculative_activation, 6
            ),
            "direct_writes": int(writes - speculative_writes),
            "speculative_writes": int(speculative_writes),
            "speculative_residents": int(len(
                self.speculative_residents & kb_pos
            )),
            "write_budget": int(budget),
            "dual_price_before": round(dual_feedback.price_before, 6),
            "dual_price_after": round(dual_feedback.price_after, 6),
            "replacement_load": round(dual_feedback.load, 6),
            "replacement_target": round(dual_feedback.target, 6),
            "dual_step_size": round(dual_feedback.step_size, 6),
            "utility": "evidence_only",
        }
        self.total_evictions += int(writes)
        self.cost_log.append({
            "w": int(self._active_window_idx),
            **stats,
        })
        return stats
