"""QueryRouter for DRIP-Core."""
from dataclasses import dataclass
import re

import numpy as np

from algorithms.cache.params import PARAMS as _P


SINGLE = "SINGLE"
MULTI_DIRECT = "MULTI_DIRECT"
BRIDGE = "BRIDGE"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    target_slots: int
    reason: str


class QueryRouter:
    """Deterministic router: no LLM and no gold support labels."""

    _COMPARISON_CUES = (
        "compare", "same", "different", "which", "between", "both",
        "older", "younger", "larger", "smaller", "more", "less",
        "before", "after",
    )
    _BRIDGE_CUES = (
        "whose", "who", "what", "where", "when", "that", "which",
        "the person", "the film", "the album", "the company",
    )

    def __init__(self, config):
        self.config = config

    def route(self, query, kb_sims, first_hop_entities):
        text = query.get("question", "") if isinstance(query, dict) else str(query)
        hinted = self._route_from_hint(query)
        if hinted is not None:
            return hinted

        lower = text.lower()
        kb_sims = np.asarray(kb_sims)
        cover_cnt = int((kb_sims >= _P.SF_HIT_THRESH).sum()) if kb_sims.size else 0
        top1 = float(kb_sims.max()) if kb_sims.size else 0.0

        has_compare = any(cue in lower for cue in self._COMPARISON_CUES)
        has_bridge_cue = any(cue in lower for cue in self._BRIDGE_CUES)
        multi_signal = (
            has_compare
            or has_bridge_cue
            or self._rough_entity_count(text) >= self.config.router_min_entities
        )
        if not multi_signal:
            return RouteDecision(SINGLE, self.config.singlehop_slots, "single_text")
        if cover_cnt >= self.config.multihop_slots:
            return RouteDecision(MULTI_DIRECT, self.config.multihop_slots, "covered")
        if self._first_hops_are_diverse(first_hop_entities):
            return RouteDecision(MULTI_DIRECT, self.config.multihop_slots, "dense_diverse")
        if top1 >= self.config.router_bridge_top1_ratio * _P.SF_HIT_THRESH or has_bridge_cue:
            return RouteDecision(BRIDGE, self.config.multihop_slots, "bridge_missing")
        return RouteDecision(MULTI_DIRECT, self.config.multihop_slots, "multi_direct")

    def _route_from_hint(self, query):
        """Use an agent/dataset query-type hint when available.

        In production this field can be produced by a cached LLM router. In the
        benchmark loaders it comes from the public query type, not gold support
        labels.
        """
        if not getattr(self.config, "use_query_type_hint", True):
            return None
        if not isinstance(query, dict):
            return None
        qtype = str(query.get("route_hint") or query.get("qtype") or query.get("type") or "").lower()
        if not qtype:
            return None
        if "bridge" in qtype or "compositional" in qtype or "inference" in qtype:
            return RouteDecision(BRIDGE, self.config.multihop_slots, "query_type_bridge")
        if "comparison" in qtype:
            return RouteDecision(MULTI_DIRECT, self.config.multihop_slots, "query_type_comparison")
        if "single" in qtype or "temporal" in qtype:
            return RouteDecision(SINGLE, self.config.singlehop_slots, "query_type_single")
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
