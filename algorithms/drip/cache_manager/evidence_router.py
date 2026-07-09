"""DRIP cache manager 的 query evidence visibility router。"""
from dataclasses import dataclass
import re

import numpy as np

from algorithms.cache.params import PARAMS as _P


QUERY_VISIBLE = "QUERY_VISIBLE"
QUERY_HIDDEN = "QUERY_HIDDEN"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    target_slots: int
    reason: str


class QueryRouter:
    """判断缺失 evidence 是 query-visible 还是 query-hidden。

    正常路径只使用 query 文本、当前 cache 相似度、first-hop entity set。
    数据集里的 ``qtype`` / ``route_hint`` 默认不参与决策，只有 oracle
    route 消融显式打开 ``use_oracle_route_hint`` 时才使用。
    """

    _COMPARISON_CUES = (
        "compare", "same", "different", "which", "between", "both",
        "older", "younger", "larger", "smaller", "more", "less",
        "before", "after",
    )
    _HIDDEN_CHAIN_CUES = (
        "whose", "the person who", "the film that", "the album that",
        "the company that", "the city where", "the country where",
        "author of", "director of", "producer of", "located in",
        "born in", "member of", "part of",
    )

    def __init__(self, config):
        self.config = config

    def route(self, query, kb_sims, first_hop_entities, first_hops=None):
        text = query.get("question", "") if isinstance(query, dict) else str(query)
        hinted = self._route_from_oracle_hint(query)
        if hinted is not None:
            return hinted

        lower = text.lower()
        kb_sims = np.asarray(kb_sims)
        cover_cnt = int((kb_sims >= _P.SF_HIT_THRESH).sum()) if kb_sims.size else 0
        cache_top1 = float(kb_sims.max()) if kb_sims.size else 0.0
        first_hops = first_hops or ()
        pool_top1 = max((float(sim) for _, sim in first_hops), default=0.0)

        has_compare = any(cue in lower for cue in self._COMPARISON_CUES)
        has_hidden_cue = any(cue in lower for cue in self._HIDDEN_CHAIN_CUES)
        entity_count = self._rough_entity_count(text)
        multi_signal = (
            has_compare
            or has_hidden_cue
            or entity_count >= self.config.router_min_entities
        )
        if not multi_signal:
            return RouteDecision(
                QUERY_VISIBLE,
                self.config.singlehop_slots,
                "visible_single_text",
            )

        target_slots = self.config.multihop_slots
        if cover_cnt >= target_slots:
            return RouteDecision(QUERY_VISIBLE, target_slots, "visible_covered")
        if self._first_hops_are_diverse(first_hop_entities):
            return RouteDecision(QUERY_VISIBLE, target_slots, "visible_dense_diverse")

        anchor_threshold = (
            self.config.router_hidden_anchor_ratio * _P.SF_HIT_THRESH
        )
        has_visible_anchor = max(cache_top1, pool_top1) >= anchor_threshold
        if has_visible_anchor and (
            has_hidden_cue or entity_count >= self.config.router_min_entities
        ):
            return RouteDecision(QUERY_HIDDEN, target_slots, "hidden_anchor_gap")
        if has_compare:
            return RouteDecision(
                QUERY_HIDDEN,
                target_slots,
                "hidden_comparison_anchor_gap",
            )
        return RouteDecision(QUERY_VISIBLE, target_slots, "visible_dense_fallback")

    def _route_from_oracle_hint(self, query):
        """只在 oracle-router 消融中使用 qtype/route_hint。"""

        if not getattr(self.config, "use_oracle_route_hint", False):
            return None
        if not isinstance(query, dict):
            return None
        qtype = str(
            query.get("route_hint") or query.get("qtype") or query.get("type") or ""
        ).lower()
        if not qtype:
            return None
        if "bridge" in qtype and "comparison" in qtype:
            return RouteDecision(
                QUERY_HIDDEN,
                self.config.hidden_comparison_slots,
                "oracle_hidden_bridge_comparison",
            )
        if "bridge" in qtype or "compositional" in qtype or "inference" in qtype:
            return RouteDecision(
                QUERY_HIDDEN,
                self.config.multihop_slots,
                "oracle_hidden",
            )
        if "comparison" in qtype:
            return RouteDecision(
                QUERY_VISIBLE,
                self.config.multihop_slots,
                "oracle_visible_comparison",
            )
        if "single" in qtype or "temporal" in qtype:
            return RouteDecision(
                QUERY_VISIBLE,
                self.config.singlehop_slots,
                "oracle_visible_single",
            )
        return None

    def _first_hops_are_diverse(self, entity_sets):
        sets = [set(es) for es in entity_sets if es]
        if len(sets) < 2:
            return False
        union = set().union(*sets)
        common = set(sets[0])
        for ents in sets[1:]:
            common &= ents
        diversity = 1.0 - len(common) / max(1, len(union))
        return diversity >= self.config.router_dense_diversity

    @staticmethod
    def _rough_entity_count(text):
        spans = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*", text)
        stop = {"what", "who", "when", "where", "which"}
        return len({span for span in spans if span.lower() not in stop})
