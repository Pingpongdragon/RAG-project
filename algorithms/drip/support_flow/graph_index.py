"""GraphIndex for DRIP-Core."""
from collections import defaultdict
import re

import numpy as np


class GraphIndex:
    """Lightweight evidence graph: doc -> entities, entity -> docs, IDF."""

    def __init__(self, config, d2p, doc_pool=None):
        self.config = config
        self.d2p = d2p
        self.doc_titles = {
            d.get("doc_id"): d.get("title", "")
            for d in (doc_pool or ())
        }
        self.pool_ents = None
        self.ent_to_docs = None
        self.doc_to_ents = None
        self.ent_idf = None

    def has_metadata(self):
        return bool(self.pool_ents)

    def set_pool_entities(self, pool_ents):
        self.pool_ents = pool_ents
        self.ent_to_docs = None
        self.doc_to_ents = None
        self.ent_idf = None

    def build(self):
        if self.ent_to_docs is not None or not self.pool_ents:
            return
        ent_to_docs = defaultdict(set)
        doc_to_ents = {}
        for did, ents in self.pool_ents.items():
            pi = self.d2p.get(did)
            if pi is None:
                continue
            norm = {
                self._normalize_entity(ent)
                for ent in ents
                if len(str(ent).strip()) >= self.config.min_entity_len
            }
            title_ent = self._normalize_entity(self.doc_titles.get(did, ""))
            if len(title_ent) >= self.config.min_entity_len:
                norm.add(title_ent)
            norm = sorted(ent for ent in norm if ent)
            doc_to_ents[int(pi)] = norm
            for ent in norm:
                ent_to_docs[ent].add(int(pi))
        n_docs = max(1, len(doc_to_ents))
        self.ent_to_docs = ent_to_docs
        self.doc_to_ents = doc_to_ents
        self.ent_idf = {
            ent: float(np.log((1.0 + n_docs) / (1.0 + len(docs))) + 1.0)
            for ent, docs in ent_to_docs.items()
        }

    def doc_entities(self, pi):
        self.build()
        if not self.doc_to_ents:
            return ()
        return self.doc_to_ents.get(int(pi), ())

    def graph_evidence(self, first_hops, kb_pos, kb_emb, doc_embs):
        """Return ``[(B, E_graph(q,B))]`` for one query."""
        self.build()
        if not self.ent_to_docs:
            return []
        kb_pos = set(int(p) for p in kb_pos)
        first_hops = [(int(pi), max(0.0, float(sim))) for pi, sim in first_hops]
        first_hop_pos = {pi for pi, sim in first_hops if sim > 0.0}
        scores = defaultdict(float)

        for ai, a_sim in first_hops:
            if a_sim <= 0.0:
                continue
            for ent in self.doc_entities(ai):
                linked_docs = self.ent_to_docs.get(ent, ())
                degree = len(linked_docs)
                if degree <= 1:
                    continue
                max_degree = int(getattr(self.config, "max_entity_degree", 0))
                if max_degree > 0 and degree > max_degree:
                    continue
                degree_power = float(getattr(self.config, "entity_degree_power", 0.0))
                link = self.ent_idf.get(ent, 1.0) / (degree ** degree_power)
                for bj in linked_docs:
                    if bj in first_hop_pos or bj in kb_pos:
                        continue
                    novelty = self._novelty(bj, kb_emb, doc_embs)
                    comp = self._complementarity(ai, bj, doc_embs)
                    scores[int(bj)] += a_sim * link * novelty * comp

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        ranked = ranked[: self.config.bridge_max_docs]
        norm = max(score for _, score in ranked) or 1.0
        return [
            (pi, self.config.bridge_alpha * float(score / norm))
            for pi, score in ranked
            if score > 0.0
        ]

    @staticmethod
    def _normalize_entity(ent):
        return re.sub(r"\s+", " ", str(ent).lower().strip())

    def _novelty(self, pi, kb_emb, doc_embs):
        if kb_emb.size == 0:
            return 1.0
        max_sim = float((doc_embs[int(pi)] @ kb_emb.T).max())
        return max(self.config.graph_novelty_floor, 1.0 - max(0.0, max_sim))

    @staticmethod
    def _complementarity(ai, bj, doc_embs):
        sim = float(doc_embs[int(ai)] @ doc_embs[int(bj)])
        return max(0.05, 1.0 - max(0.0, sim))
