"""DRIP 的 evidence credit 与 replacement writer。

这里实现 ``DRIPNOdetector`` 和 hidden diagnostic 共用的 evidence/writer：

  1. direct evidence 维持 query-visible support；
  2. replacement-aware writer 执行 ``Delta_t(c,v) > 0``；
  3. hidden diagnostic 可选用 ESC 从 resident/easy anchor A 找 missing B；
  4. bridge evidence 出现后，用 pair lease 一起保护 A+B。

这个模块没有 random-walk graph propagation；bridge candidate 的分数来自
anchor-conditioned semantic evidence 和 soft entity consistency。
"""
from collections import defaultdict
import re

import numpy as np

from algorithms.cache.params import PARAMS as _P

from .core import DRIPCore


class EvidenceConditionedBridgeEvidence:
    """给 hidden support B 打分，条件是 resident/easy anchor A。

    诊断公式：

        E_ESC(q, B | A) = sim(Encode(q, A), B) * Link(A, B) * Cue(q, B)

    ``Encode(q, A)`` 优先用 text encoder 编码紧凑 prompt；如果 encoder 不可用，
    则退化为 query embedding 和 anchor embedding 的归一化混合。实体重合是
    soft consistency bonus，不是硬 gate。
    """

    K_ANCHORS = 5
    TARGET_TOPK = 80
    CANDIDATE_CAP = 80
    QUERY_WEIGHT = 0.72
    ANCHOR_WEIGHT = 0.28
    TARGET_MIX = 0.60
    LINK_BOOST = 0.75
    CUE_BOOST = 0.35
    TITLE_MATCH_BOOST = 3.0
    RANK_DECAY = 0.5
    DENSE_POWER = 1.0
    D_CAP = 50

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "by", "with", "from", "that", "this", "these", "those", "which",
        "what", "who", "whom", "whose", "when", "where", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "do", "does",
        "did", "have", "has", "had", "as", "at", "it", "its", "their",
        "his", "her", "they", "them", "he", "she", "we", "you", "i",
        "answer", "following", "question", "same", "both", "also", "than",
        "more", "less", "name", "called",
    }

    def __init__(
        self,
        graph_index,
        doc_pool,
        doc_embs,
        use_text_encoder=True,
        encoder_model="BAAI/bge-large-en-v1.5",
    ):
        self.gi = graph_index
        self.doc_pool = doc_pool
        self.doc_embs = doc_embs
        self.use_text_encoder = bool(use_text_encoder)
        self.encoder_model = str(encoder_model)
        self._encoder = None
        self._target_cache = {}
        self.last_pair_scores = {}

    def _tokens(self, text):
        return [
            tok
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", str(text).lower())
            if tok not in self._STOPWORDS
        ]

    def _question_spans(self, question):
        spans = []
        seen = set()
        patterns = [
            r'"([^"]{3,80})"',
            r"'([^']{3,80})'",
            r"\b([A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,5})\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, str(question)):
                span = " ".join(str(match).split())
                key = span.lower()
                if key and key not in seen and key not in self._STOPWORDS:
                    seen.add(key)
                    spans.append(span)
        return spans

    def _relation_cues(self, question):
        masked = str(question)
        for span in sorted(self._question_spans(question), key=len, reverse=True):
            masked = re.sub(re.escape(span), " ", masked, flags=re.IGNORECASE)
        out = []
        seen = set()
        for tok in self._tokens(masked):
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out[:12]

    def _candidate_cue_score(self, pi, cues):
        if not cues:
            return 0.0
        doc = self.doc_pool[int(pi)]
        text = f"{doc.get('title') or ''} {str(doc.get('text') or '')[:700]}"
        toks = set(self._tokens(text))
        if not toks:
            return 0.0
        hits = sum(1 for cue in cues if cue in toks)
        return min(1.0, float(hits) / max(1.0, float(len(cues)) ** 0.5))

    def _target_text(self, query, ai):
        question = str(query.get("question") or "") if isinstance(query, dict) else ""
        doc = self.doc_pool[int(ai)]
        title = str(doc.get("title") or "")
        text = str(doc.get("text") or "")[:220]
        spans = "; ".join(self._question_spans(question)[:6])
        cues = " ".join(self._relation_cues(question)[:8]) or "support evidence"
        return (
            f"Question: {question}\n"
            f"Question entities: {spans}\n"
            f"Known evidence: {title}: {text}\n"
            f"Find the missing supporting document about: {cues}."
        )

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = SentenceTransformer(self.encoder_model, device=device)
        return self._encoder

    def _encode_target_texts(self, keyed_texts):
        if not keyed_texts or not self.use_text_encoder:
            return {}
        out = {}
        pending = []
        pending_keys = []
        for key, text in keyed_texts:
            cached = self._target_cache.get(key)
            if cached is None:
                pending.append(text)
                pending_keys.append(key)
            else:
                out[key] = cached
        if pending:
            try:
                prefix = (
                    "Represent this sentence for searching relevant passages: "
                    if "bge" in self.encoder_model.lower()
                    else ""
                )
                embs = self._get_encoder().encode(
                    [prefix + text for text in pending],
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                for key, emb in zip(pending_keys, embs):
                    vec = np.asarray(emb, dtype=np.float32)
                    self._target_cache[key] = vec
                    out[key] = vec
            except Exception:
                # 离线/最小环境下保持实验可运行：encoder 失败就退化到 embedding mixture。
                self.use_text_encoder = False
        return out

    def evidence(self, anchors, kb_pos=None, query=None):
        self.last_pair_scores = {}
        self.gi.build()
        kb_pos = set(int(p) for p in (kb_pos or ()))
        anchors = [
            (int(pi), max(0.0, float(sim)))
            for pi, sim in anchors[: self.K_ANCHORS]
            if float(sim) > 0.0
        ]
        if not anchors:
            return [], {"seed_count": 0, "candidate_pool": 0, "selected": 0}

        q_vec = None
        if isinstance(query, dict):
            q_vec = query.get("_drip_query_emb")
        if q_vec is None:
            q_vec = np.zeros(self.doc_embs.shape[1], dtype=np.float32)
        q_vec = np.asarray(q_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm > 0.0:
            q_vec = q_vec / q_norm

        question = str(query.get("question") or "") if isinstance(query, dict) else ""
        qidx = query.get("qidx") if isinstance(query, dict) else None
        cue_terms = self._relation_cues(question)
        anchor_sum = sum(score for _, score in anchors) or 1.0
        anchor_pos = {int(pi) for pi, _ in anchors}
        idf_max = max(self.gi.ent_idf.values()) if self.gi.ent_idf else 1.0
        degree_power = float(getattr(self.gi.config, "entity_degree_power", 0.5))
        target_texts = [
            ((int(qidx) if qidx is not None else -1, int(ai)), self._target_text(query, ai))
            for ai, _ in anchors
        ]
        encoded_targets = self._encode_target_texts(target_texts)

        scores = defaultdict(float)
        pair_scores = defaultdict(lambda: defaultdict(float))
        touched = set(anchor_pos)
        for a_rank, (ai, a_score) in enumerate(anchors, start=1):
            key = (int(qidx) if qidx is not None else -1, int(ai))
            target = encoded_targets.get(key)
            if target is None:
                target = (
                    self.QUERY_WEIGHT * q_vec
                    + self.ANCHOR_WEIGHT * self.doc_embs[int(ai)]
                )
            norm = float(np.linalg.norm(target))
            if norm <= 0.0:
                continue
            target = target / norm
            sims = self.doc_embs @ target

            edge_scores = defaultdict(float)
            for ent in self.gi.doc_entities(ai):
                linked = self.gi.ent_to_docs.get(ent, ())
                degree = len(linked)
                if degree <= 1:
                    continue
                if self.D_CAP > 0 and degree > self.D_CAP:
                    linked = [
                        int(bj) for bj in linked
                        if self.gi._entity_in_title(ent, int(bj))
                    ]
                    if not linked:
                        continue
                    degree = len(linked)
                idf = self.gi.ent_idf.get(ent, 1.0)
                edge = (idf / max(1e-9, idf_max)) / (degree ** degree_power)
                for bj in linked:
                    bj = int(bj)
                    touched.add(bj)
                    if bj in anchor_pos or bj in kb_pos:
                        continue
                    title_boost = (
                        self.TITLE_MATCH_BOOST
                        if self.gi._entity_in_title(ent, bj)
                        else 1.0
                    )
                    edge_scores[bj] += edge * title_boost

            anchor_credit = (
                (float(a_score) / anchor_sum)
                / (float(a_rank) ** self.RANK_DECAY)
            )
            ranked = sorted(
                edge_scores.items(),
                key=lambda item: (-(max(0.0, float(sims[int(item[0])]))), -item[1]),
            )[: self.TARGET_TOPK]
            for c_rank, (bj, link_score) in enumerate(ranked, start=1):
                bj = int(bj)
                dense = max(0.0, float(sims[bj]))
                if dense <= 0.0 and link_score <= 0.0:
                    continue
                dense_signal = dense ** self.DENSE_POWER
                target_score = (
                    (1.0 - self.TARGET_MIX) * max(0.0, float(link_score))
                    + self.TARGET_MIX * dense_signal
                )
                link_bonus = 1.0 + self.LINK_BOOST * min(1.0, float(link_score))
                cue_bonus = 1.0 + self.CUE_BOOST * self._candidate_cue_score(
                    bj, cue_terms
                )
                rank_bonus = 1.0 / np.sqrt(float(c_rank))
                val = anchor_credit * target_score * link_bonus * cue_bonus * rank_bonus
                if val <= 0.0:
                    continue
                scores[bj] += val
                pair_scores[bj][int(ai)] += val

        candidates = [
            (int(pi), float(score))
            for pi, score in sorted(scores.items(), key=lambda item: -item[1])[
                : self.CANDIDATE_CAP
            ]
            if score > 0.0
        ]
        keep = {pi for pi, _ in candidates}
        self.last_pair_scores = {
            int(pi): {int(ai): float(v) for ai, v in anchors.items() if v > 0.0}
            for pi, anchors in pair_scores.items()
            if int(pi) in keep
        }
        return candidates, {
            "seed_count": len(anchor_pos),
            "candidate_pool": len(touched),
            "selected": len(candidates),
        }


class EvidenceConditionedDRIPCore(DRIPCore):
    """用于 bridge support completion 的旧 DRIP core。

    这里只暴露概念性常量，避免构造函数参数过长：

        D_dir(d) <- dense/direct evidence
        D_brg(d) <- E_ESC(q, d) plus support-unit lease credit
        P(d) = S(d) + D_dir(d) + D_brg(d)
    """

    BRIDGE_GAIN = 2.0
    BRIDGE_TOPK = 5
    DIRECT_BRIDGE_GAIN = 1.5
    DIRECT_TOPK_BRIDGE = 5
    BRIDGE_RESERVE = 0.35
    BRIDGE_MARGIN = 0.85
    PAIR_LEASE_WEIGHT = 1.5
    PAIR_LEASE_DECAY = 0.985
    PAIR_LEASE_TOP_ANCHORS = 2
    RESIDENT_ANCHOR_TOPK = 5
    RESIDENT_ANCHOR_THRESHOLD = 0.58
    RESIDENT_ANCHOR_BOOST = 1.25
    DENSE_ANCHOR_TOPK = 2
    MAX_ANCHORS = 5
    DIRECT_PROTECT_DECAY = 0.98
    DIRECT_PROTECT_MIN = 0.20
    SERVE_PROTECT_MIN = 0.10
    DIRECT_DEMAND_PROTECT_MIN = 0.10

    def __init__(
        self,
        *args,
        use_bridge=True,
        use_pair_lease=True,
        use_text_encoder=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_bridge = bool(use_bridge)
        self.use_pair_lease = bool(use_pair_lease)
        self.use_text_encoder = bool(use_text_encoder)
        self._esc_engine = None
        self._esc_installed = False
        self.direct_demand = {}
        self.bridge_demand = {}
        self.bridge_pair_mass = {}
        self.pair_lease = {}
        self.direct_protect = {}
        self._current_kb_pos = set()
        self._active_window_idx = -1
        self._replacement_ema = 0.0
        self.total_evictions = 0
        self.cost_log = []

    @property
    def _pool_ents(self):
        return self.graph_index.pool_ents

    @_pool_ents.setter
    def _pool_ents(self, pool_ents):
        self.graph_index.set_pool_entities(pool_ents)
        self._esc_engine = None
        self._esc_installed = False

    def _ensure_esc(self):
        self.graph_index.build()
        if self._esc_engine is None:
            self._esc_engine = EvidenceConditionedBridgeEvidence(
                self.graph_index,
                self.doc_pool,
                self.doc_embs,
                use_text_encoder=self.use_text_encoder,
            )
        return self._esc_engine

    def _install_bridge_graph_evidence(self):
        if self._esc_installed:
            return
        graph_index = self.graph_index
        esc = self._ensure_esc() if self.use_bridge else None

        def esc_graph_evidence(query, anchors, kb_pos, kb_emb, doc_embs):
            if esc is None:
                candidates, stats = [], {
                    "seed_count": 0,
                    "candidate_pool": 0,
                    "selected": 0,
                }
            else:
                candidates, stats = esc.evidence(anchors, kb_pos=kb_pos, query=query)
            graph_index.last_stats = {
                "bridge_raw_paths": int(stats["candidate_pool"]),
                "bridge_after_degree_gate": int(stats["candidate_pool"]),
                "bridge_after_relation_gate": int(stats["selected"]),
                "bridge_after_novelty_gate": int(stats["selected"]),
                "bridge_after_threshold": int(stats["selected"]),
                "bridge_selected": int(len(candidates)),
                "bridge_mmr_stopped": 0,
                "bridge_no_path": int(stats["seed_count"] == 0),
                "bridge_top_entities": [],
                "esc_seed_count": int(stats["seed_count"]),
                "esc_candidate_pool": int(stats["candidate_pool"]),
                "esc_selected": int(stats["selected"]),
            }
            max_docs = int(getattr(graph_index.config, "bridge_max_docs", 20))
            return candidates[:max_docs]

        graph_index.graph_evidence = esc_graph_evidence
        self._esc_installed = True

    def step(self, window_queries, window_query_embs, window_idx):
        self._active_window_idx = int(window_idx)
        self._install_bridge_graph_evidence()
        return super().step(window_queries, window_query_embs, window_idx)

    def _sync_demand(self):
        merged = {}
        for source in (self.direct_demand, self.bridge_demand):
            for p, v in source.items():
                merged[int(p)] = merged.get(int(p), 0.0) + float(v)
        if self.use_pair_lease:
            for p, v in self.pair_lease.items():
                merged[int(p)] = (
                    merged.get(int(p), 0.0)
                    + self.PAIR_LEASE_WEIGHT * float(v)
                )
        self.demand = merged

    def _decay_map(self, values, decay):
        floor = float(getattr(self.config, "min_stat", 0.01))
        return {
            int(p): float(v) * decay
            for p, v in values.items()
            if float(v) * decay >= floor
        }

    def _decay(self):
        super()._decay()
        d = float(self._effective_demand_decay())
        self.direct_demand = self._decay_map(self.direct_demand, d)
        self.bridge_demand = self._decay_map(self.bridge_demand, d)
        self.direct_protect = self._decay_map(
            self.direct_protect, self.DIRECT_PROTECT_DECAY)
        if self.use_pair_lease:
            self.pair_lease = self._decay_map(
                self.pair_lease, self.PAIR_LEASE_DECAY)
            self.bridge_pair_mass = {
                (int(a), int(b)): float(v) * self.PAIR_LEASE_DECAY
                for (a, b), v in self.bridge_pair_mass.items()
                if float(v) * self.PAIR_LEASE_DECAY
                >= float(getattr(self.config, "min_stat", 0.01))
            }
        self._sync_demand()

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

    def _bridge_graph_hops(self, query, first_hops, dense, q_kb_row, kb_idx, kb_pos):
        self._current_kb_pos = set(int(p) for p in (kb_pos or ()))
        anchors = []
        seen = set()

        def add(pi, score):
            pi = int(pi)
            if pi in seen or float(score) <= 0.0:
                return
            seen.add(pi)
            anchors.append((pi, float(score)))

        if q_kb_row is not None and len(q_kb_row) > 0:
            k = min(self.RESIDENT_ANCHOR_TOPK, len(q_kb_row))
            for col in np.argsort(q_kb_row)[::-1][:k]:
                score = float(q_kb_row[int(col)])
                if score >= self.RESIDENT_ANCHOR_THRESHOLD:
                    add(int(kb_idx[int(col)]), self.RESIDENT_ANCHOR_BOOST * score)

        for pi, sim in dense[: self.DENSE_ANCHOR_TOPK]:
            add(int(pi), float(sim))

        if not anchors:
            return first_hops
        anchors.sort(key=lambda item: -item[1])
        return anchors[: self.MAX_ANCHORS]

    def _credit_dense(self, candidates, kb_pos, gamma=None, gold_pos=None, top1_bonus=None):
        gamma = self.config.direct_gamma if gamma is None else float(gamma)
        top1_bonus = self.config.direct_top1_bonus if top1_bonus is None else float(top1_bonus)
        gold_pos = gold_pos or set()
        bridge_direct_gamma = float(getattr(self.config, "bridge_direct_gamma", 0.2))
        bridge_direct = top1_bonus == 0.0 and gamma <= bridge_direct_gamma + 1e-12
        updates = 0
        mass = 0.0
        gold_updates = 0
        gold_mass = 0.0

        if bridge_direct:
            top = [
                (int(pi), max(0.0, float(sim)))
                for pi, sim in candidates[: self.DIRECT_TOPK_BRIDGE]
                if int(pi) not in kb_pos and float(sim) > 0.0
            ]
            norm = sum(score for _, score in top)
            if norm <= 0.0:
                return 0, 0.0, 0, 0.0
            for rank, (pi, sim) in enumerate(top, start=1):
                score = self.DIRECT_BRIDGE_GAIN * (sim / norm) / np.sqrt(float(rank))
                self.direct_demand[pi] = self.direct_demand.get(pi, 0.0) + score
                self.demand[pi] = self.demand.get(pi, 0.0) + score
                updates += 1
                mass += score
                if pi in gold_pos:
                    gold_updates += 1
                    gold_mass += score
            return updates, mass, gold_updates, gold_mass

        # Direct evidence 采用 ARC-style rank/distance 加权：
        # E(q,d) = sim(q,d) / (rank * (eps + 1 - sim(q,d))^alpha)。
        # 保留 sim 因子是为了避免低相似度 tail candidate 只凭 rank 被过度加分。
        alpha = max(0.0, float(getattr(self.config, "direct_evidence_alpha", 0.5)))
        eps = max(1e-9, float(getattr(self.config, "direct_evidence_epsilon", 0.05)))
        for rank, (pi, sim) in enumerate(candidates, start=1):
            pi = int(pi)
            if pi in kb_pos:
                continue
            sim = max(0.0, float(sim))
            if sim <= 0.0:
                continue
            distance = eps + max(0.0, 1.0 - sim)
            evidence = sim / (float(rank) * (distance ** alpha))
            score = gamma * evidence
            if rank == 1:
                score += top1_bonus
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
        gold_pos = gold_pos or set()
        updates = 0
        mass = 0.0
        gold_updates = 0
        gold_mass = 0.0
        top = [
            (int(pi), max(0.0, float(score)))
            for pi, score in candidates[: self.BRIDGE_TOPK]
            if float(score) > 0.0
        ]
        norm = sum(score for _, score in top)
        if norm <= 0.0:
            return 0, 0.0, 0, 0.0

        esc = self._ensure_esc() if self.use_bridge else None
        pair_scores = getattr(esc, "last_pair_scores", {}) if esc is not None else {}
        kb_pos = set(getattr(self, "_current_kb_pos", set()))
        for rank, (pi, raw) in enumerate(top, start=1):
            credit = self.BRIDGE_GAIN * (raw / norm) / np.sqrt(float(rank))
            self.bridge_demand[pi] = self.bridge_demand.get(pi, 0.0) + credit
            self.demand[pi] = self.demand.get(pi, 0.0) + credit
            updates += 1
            mass += credit
            if pi in gold_pos:
                gold_updates += 1
                gold_mass += credit
            if self.use_pair_lease:
                self._credit_pair_lease(pi, credit, pair_scores.get(int(pi), {}), kb_pos)
        return updates, mass, gold_updates, gold_mass

    def _credit_pair_lease(self, bi, credit, anchors, kb_pos):
        ranked = sorted(
            ((float(v), int(ai)) for ai, v in anchors.items() if float(v) > 0.0),
            reverse=True,
        )[: self.PAIR_LEASE_TOP_ANCHORS]
        total = sum(v for v, _ in ranked)
        if total <= 0.0:
            return
        for val, ai in ranked:
            lease = float(credit) * (val / total)
            self.bridge_pair_mass[(int(ai), int(bi))] = (
                self.bridge_pair_mass.get((int(ai), int(bi)), 0.0) + lease
            )
            if int(ai) in kb_pos:
                self.pair_lease[int(ai)] = self.pair_lease.get(int(ai), 0.0) + lease
            if int(bi) in kb_pos:
                self.pair_lease[int(bi)] = self.pair_lease.get(int(bi), 0.0) + lease

    def _activate_pair_lease(self, bi, kb_pos):
        if not self.use_pair_lease:
            return 0
        links = sorted(
            (
                float(val),
                int(ai),
            )
            for (ai, bj), val in self.bridge_pair_mass.items()
            if int(bj) == int(bi) and int(ai) in kb_pos and float(val) > 0.0
        )[::-1][: self.PAIR_LEASE_TOP_ANCHORS]
        for val, ai in links:
            self.pair_lease[ai] = self.pair_lease.get(ai, 0.0) + val
            self.pair_lease[int(bi)] = self.pair_lease.get(int(bi), 0.0) + val
            self.direct_protect[ai] = self.direct_protect.get(ai, 0.0) + val
        return len(links)

    def _priority_for_route(self, kb_idx, route):
        return {
            int(p): self.serve.get(int(p), 0.0)
            + self.direct_demand.get(int(p), 0.0)
            + self.bridge_demand.get(int(p), 0.0)
            + self.PAIR_LEASE_WEIGHT * self.pair_lease.get(int(p), 0.0)
            for p in kb_idx
        }

    def _resident_priority(self, kb_idx, kb_emb):
        return self._priority_for_route(kb_idx, "direct")

    def _replacement_pressure(self, budget):
        """最近 replacement 压力 phi_t。

        用 EMA(replacements/window) 除以本窗口可用 write budget。这样不同实验的
        window size / KB size 不同，也能得到可比较的 0 附近压力值。
        """
        scale = max(1.0, float(budget or _P.WRITE_CAP or 1))
        return float(min(4.0, self._replacement_ema / scale))

    def _replacement_penalty(self, budget):
        """统一替换惩罚 C_t = lambda_replace * (1 + mu * phi_t)。"""
        base = max(0.0, float(getattr(self.config, "replacement_cost", 0.25)))
        mu = max(0.0, float(getattr(self.config, "replacement_pressure_mu", 1.0)))
        return base * (1.0 + mu * self._replacement_pressure(budget))

    def _finalize_write_stats(self, stats, budget, replacement_penalty, net_gains):
        writes = int(stats.get("writes", 0))
        decay = float(getattr(self.config, "replacement_ema_decay", 0.75))
        decay = float(np.clip(decay, 0.0, 1.0))
        pressure_before = self._replacement_pressure(budget)
        self._replacement_ema = decay * self._replacement_ema + (1.0 - decay) * writes

        kb_size = max(1, len(self.kb))
        avg_net_gain = float(np.mean(net_gains)) if net_gains else 0.0
        stats.update({
            "evictions": writes,
            "replacement_pressure": round(float(pressure_before), 6),
            "replacement_penalty": round(float(replacement_penalty), 6),
            "avg_net_gain": round(avg_net_gain, 6),
            "churn_rate": round(float(writes / kb_size), 6),
        })
        self.total_evictions += writes
        self.cost_log.append({
            "w": int(getattr(self, "_active_window_idx", -1)),
            "write_budget": int(budget),
            "writes": writes,
            "evictions": writes,
            "churn_rate": stats["churn_rate"],
            "replacement_pressure": stats["replacement_pressure"],
            "replacement_penalty": stats["replacement_penalty"],
            "avg_net_gain": stats["avg_net_gain"],
            "direct_writes": int(stats.get("direct_writes", 0)),
            "bridge_writes": int(stats.get("bridge_writes", 0)),
        })
        return stats

    def _write(self, kb_idx, kb_emb, budget, gold_pos=None):
        gold_pos = gold_pos or set()
        hidden_pos = getattr(self, "_window_hidden_pos", set())
        budget = int(budget)
        net_gains = []
        replacement_penalty = self._replacement_penalty(budget)
        if budget <= 0:
            return self._finalize_write_stats(
                self._empty_write_stats(), budget, replacement_penalty, net_gains)

        kb_pos = set(int(p) for p in kb_idx)
        direct_candidates = sorted(
            ((float(v), int(p)) for p, v in self.direct_demand.items() if int(p) not in kb_pos),
            reverse=True,
        )
        bridge_candidates = sorted(
            ((float(v), int(p)) for p, v in self.bridge_demand.items() if int(p) not in kb_pos),
            reverse=True,
        )
        if not direct_candidates and not bridge_candidates:
            return self._finalize_write_stats(
                self._empty_write_stats(), budget, replacement_penalty, net_gains)

        direct_budget = budget if direct_candidates else 0
        bridge_budget = 0
        bridge_budget_cap = (
            int(np.ceil(self.BRIDGE_RESERVE * budget)) if bridge_candidates else 0
        )

        writes = 0
        gold_writes = 0
        hidden_writes = 0
        hidden_evictions = 0
        direct_writes = 0
        bridge_writes = 0
        pair_activations = 0
        protected_direct = set()

        def current_kb_emb():
            return self.doc_embs[np.array(sorted(kb_pos), dtype=np.int64)]

        def evict(victim):
            nonlocal hidden_evictions
            hidden_evictions += int(int(victim) in hidden_pos)
            self.kb.discard(self.p2d[victim])
            self.serve.pop(victim, None)
            self.direct_protect.pop(victim, None)
            self.pair_lease.pop(victim, None)
            kb_pos.discard(victim)

        def admit(cp, route):
            nonlocal pair_activations
            self.kb.add(self.p2d[cp])
            kb_pos.add(cp)
            if route == "direct":
                self.direct_protect[cp] = self.direct_protect.get(cp, 0.0) + 1.0
                protected_direct.add(cp)
            elif route == "bridge":
                self.bridge_demand[cp] = self.bridge_demand.get(cp, 0.0) + 1.0
                self.demand[cp] = self.demand.get(cp, 0.0) + 1.0
                pair_activations += self._activate_pair_lease(cp, kb_pos)

        def bridge_victim_pool():
            pool = {
                int(p)
                for p in kb_pos
                if p not in protected_direct
                and self.direct_protect.get(int(p), 0.0) < self.DIRECT_PROTECT_MIN
                and self.serve.get(int(p), 0.0) < self.SERVE_PROTECT_MIN
                and self.direct_demand.get(int(p), 0.0) < self.DIRECT_DEMAND_PROTECT_MIN
            }
            return pool or set(kb_pos)

        def admit_from(candidates, cap, route, margin):
            nonlocal writes, gold_writes, hidden_writes, direct_writes, bridge_writes
            priority = self._priority_for_route(kb_pos, route)
            victim_pool = kb_pos if route == "direct" else bridge_victim_pool()
            victims = sorted(victim_pool, key=lambda p: priority.get(int(p), 0.0))
            victim_i = 0
            local_writes = 0
            for value, cp in candidates:
                cp = int(cp)
                if local_writes >= cap or writes >= budget or victim_i >= len(victims):
                    break
                if cp in kb_pos:
                    continue
                victim = int(victims[victim_i])
                net_gain = (
                    float(value)
                    - margin * priority.get(victim, 0.0)
                    - replacement_penalty
                )
                if net_gain <= 0.0:
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
                hidden_writes += int(cp in hidden_pos)
                if route == "bridge":
                    bridge_writes += 1
                else:
                    direct_writes += 1
                net_gains.append(float(net_gain))

        admit_from(
            direct_candidates,
            direct_budget,
            "direct",
            self._effective_gain_margin(),
        )
        bridge_budget = min(bridge_budget_cap, max(0, budget - writes))
        admit_from(
            bridge_candidates,
            bridge_budget,
            "bridge",
            self._effective_gain_margin(self.BRIDGE_MARGIN),
        )
        self._sync_demand()

        all_candidates = {p for _, p in direct_candidates} | {p for _, p in bridge_candidates}
        gold_candidates = sum(1 for p in all_candidates if p in gold_pos)
        hidden_candidates = sum(1 for p in all_candidates if p in hidden_pos)
        stats = {
            "writes": int(writes),
            "candidates": int(len(all_candidates)),
            "gold_candidates": int(gold_candidates),
            "gold_writes": int(gold_writes),
            "gold_rate": float(gold_writes / writes) if writes else 0.0,
            "hidden_candidates": int(hidden_candidates),
            "hidden_writes": int(hidden_writes),
            "hidden_rate": float(hidden_writes / writes) if writes else 0.0,
            "hidden_evictions": int(hidden_evictions),
            "direct_writes": int(direct_writes),
            "bridge_writes": int(bridge_writes),
            "direct_budget": int(direct_budget),
            "bridge_budget": int(bridge_budget),
            "bridge_reserve": float(self.BRIDGE_RESERVE),
            "pair_activations": int(pair_activations),
            "pair_lease_docs": int(len(self.pair_lease)),
            "pair_lease_mass": float(sum(self.pair_lease.values())),
        }
        return self._finalize_write_stats(
            stats, budget, replacement_penalty, net_gains)

    @staticmethod
    def _empty_write_stats():
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
            "direct_writes": 0,
            "bridge_writes": 0,
            "direct_budget": 0,
            "bridge_budget": 0,
            "bridge_reserve": 0.0,
            "pair_activations": 0,
            "pair_lease_docs": 0,
            "pair_lease_mass": 0.0,
        }


class EmbeddingOnlyDRIPCore(EvidenceConditionedDRIPCore):
    """消融：同样 writer/router，但不使用 hidden-support ESC evidence。"""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            use_bridge=False,
            use_pair_lease=False,
            use_text_encoder=False,
            **kwargs,
        )
        self.force_query_visible = True
