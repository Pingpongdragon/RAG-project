"""GraphIndex for DRIP cache manager."""
from collections import Counter, defaultdict
import math
import re

import numpy as np


class GraphIndex:
    """Lightweight evidence graph with relation-aware bridge scoring.

    The graph is still intentionally small: document/entity postings plus
    query-aware local text evidence.  It is not a global KG and it does not call
    an LLM in the hot path.
    """

    _STOPWORDS = {
        "about", "after", "also", "among", "been", "being", "both", "from",
        "have", "into", "more", "most", "other", "over", "same", "than",
        "that", "their", "then", "there", "these", "they", "this", "through",
        "what", "when", "where", "which", "while", "whose", "with", "would",
        "were", "who", "whom", "why", "how", "the", "and", "for", "was",
        "are", "did", "does", "has", "had", "his", "her", "its", "not",
        "in", "on", "of", "to", "by", "as", "at", "or", "is", "a", "an",
    }
    _GENERIC_ENTITIES = {
        "american", "arabic", "argentine", "australian", "british",
        "canadian", "chinese", "dutch", "english", "european", "french",
        "german", "greek", "hindi", "indian", "irish", "italian",
        "japanese", "kannada", "korean", "latin", "malayalam", "mexican",
        "russian", "soviet", "spanish", "swedish", "tamil", "turkish",
    }

    def __init__(self, config, d2p, doc_pool=None):
        self.config = config
        self.d2p = d2p
        self.doc_titles = {
            d.get("doc_id"): d.get("title", "")
            for d in (doc_pool or ())
        }
        self.doc_texts = {
            d.get("doc_id"): d.get("text", "")
            for d in (doc_pool or ())
        }
        self.pi_to_title = {}
        self.pi_to_text = {}
        self.pool_ents = None
        self.ent_to_docs = None
        self.doc_to_ents = None
        self.ent_idf = None
        self._context_cache = {}
        self.last_stats = {}

    def has_metadata(self):
        return bool(self.pool_ents)

    def set_pool_entities(self, pool_ents):
        self.pool_ents = pool_ents
        self.ent_to_docs = None
        self.doc_to_ents = None
        self.ent_idf = None
        self.pi_to_title = {}
        self.pi_to_text = {}
        self._context_cache = {}
        self.last_stats = {}

    def build(self):
        if self.ent_to_docs is not None or not self.pool_ents:
            return
        ent_to_docs = defaultdict(set)
        doc_to_ents = {}
        for did, ents in self.pool_ents.items():
            pi = self.d2p.get(did)
            if pi is None:
                continue
            pi = int(pi)
            self.pi_to_title[pi] = self.doc_titles.get(did, "")
            self.pi_to_text[pi] = self.doc_texts.get(did, "")
            norm = {
                self._normalize_entity(ent)
                for ent in ents
                if len(str(ent).strip()) >= self.config.min_entity_len
            }
            title_ent = self._normalize_entity(self.doc_titles.get(did, ""))
            if len(title_ent) >= self.config.min_entity_len:
                norm.add(title_ent)
            norm = sorted(ent for ent in norm if ent and ent not in self._GENERIC_ENTITIES)
            doc_to_ents[pi] = norm
            for ent in norm:
                ent_to_docs[ent].add(pi)
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

    def graph_evidence(self, query, first_hops, kb_pos, kb_emb, doc_embs):
        """Return ``[(B, E_graph(q,B))]`` for one query."""
        self.build()
        self.last_stats = {
            "bridge_raw_paths": 0,
            "bridge_after_degree_gate": 0,
            "bridge_after_relation_gate": 0,
            "bridge_after_novelty_gate": 0,
            "bridge_after_threshold": 0,
            "bridge_selected": 0,
            "bridge_mmr_stopped": 0,
            "bridge_no_path": 0,
            "bridge_top_entities": [],
        }
        if not self.ent_to_docs:
            return []
        kb_pos = set(int(p) for p in kb_pos)
        first_hops = [(int(pi), max(0.0, float(sim))) for pi, sim in first_hops]
        first_hop_pos = {pi for pi, sim in first_hops if sim > 0.0}
        evidence_mass = defaultdict(float)
        entity_counts = Counter()
        query_terms = self._query_terms(query)
        idf_max = max(self.ent_idf.values()) if self.ent_idf else 1.0

        for ai, a_sim in first_hops:
            min_firsthop = float(getattr(self.config, "bridge_min_firsthop_sim", 0.0))
            if a_sim <= min_firsthop:
                continue
            seed_ents = self._seed_entities(ai)
            if not seed_ents:
                self.last_stats["bridge_no_path"] += 1
                continue
            for ent in seed_ents:
                linked_docs = self.ent_to_docs.get(ent, ())
                self.last_stats["bridge_raw_paths"] += len(linked_docs)
                degree = len(linked_docs)
                if degree <= 1:
                    continue
                max_degree = int(getattr(self.config, "max_entity_degree", 0))
                if max_degree > 0 and degree > max_degree:
                    continue
                self.last_stats["bridge_after_degree_gate"] += len(linked_docs)
                link = self._link_strength(ent, degree, idf_max)
                for bj in linked_docs:
                    if bj in first_hop_pos or bj in kb_pos:
                        continue
                    relation = self._relation_score(query_terms, ai, ent, bj)
                    if relation <= 0.0:
                        continue
                    self.last_stats["bridge_after_relation_gate"] += 1
                    novelty = self._novelty_score(bj, kb_emb, doc_embs)
                    if novelty < float(getattr(self.config, "graph_novelty_floor", 0.0)):
                        continue
                    self.last_stats["bridge_after_novelty_gate"] += 1
                    evidence_mass[int(bj)] += self._path_evidence(
                        a_sim, link, min(1.0, relation))
                    entity_counts[ent] += 1

        if not evidence_mass:
            return []
        threshold = float(getattr(self.config, "bridge_abs_threshold", 0.0))
        saturation = max(1e-6, float(getattr(self.config, "bridge_score_saturation", 1.0)))
        graph_scores = {
            pi: self.config.bridge_alpha * (1.0 - math.exp(-score / saturation))
            for pi, score in evidence_mass.items()
            if score > 0.0
        }
        eligible = {
            pi: score
            for pi, score in graph_scores.items()
            if score >= threshold
        }
        self.last_stats["bridge_after_threshold"] = len(eligible)
        self.last_stats["bridge_top_entities"] = entity_counts.most_common(5)
        ranked = self._select_mmr(eligible, kb_emb, doc_embs)
        self.last_stats["bridge_selected"] = len(ranked)
        return ranked

    @staticmethod
    def _normalize_entity(ent):
        ent = re.sub(r"\s+", " ", str(ent).lower().strip())
        if ent.startswith("the "):
            ent = ent[4:]
        return ent

    def _seed_entities(self, pi):
        ents = list(self.doc_entities(pi))
        max_seed = int(getattr(self.config, "bridge_max_seed_entities", 0))
        if max_seed <= 0 or len(ents) <= max_seed:
            return ents

        def key(ent):
            title_bonus = 1 if self._entity_in_title(ent, pi) else 0
            return (
                title_bonus,
                self.ent_idf.get(ent, 0.0),
                -len(self.ent_to_docs.get(ent, ())),
            )

        return sorted(ents, key=key, reverse=True)[:max_seed]

    def _query_terms(self, query):
        text = query.get("question", "") if isinstance(query, dict) else str(query)
        return self._content_terms(text)

    def _content_terms(self, text):
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]+", str(text).lower())
        return {
            w.strip("'")
            for w in words
            if len(w.strip("'")) >= 3 and w.strip("'") not in self._STOPWORDS
        }

    def _relation_score(self, query_terms, ai, ent, bj):
        floor = float(getattr(self.config, "bridge_relation_floor", 0.0))
        weight = float(getattr(self.config, "bridge_relation_overlap_weight", 0.0))
        context_terms = self._entity_context_terms(ai, ent)
        target_terms = self._content_terms(self.pi_to_title.get(int(bj), ""))
        overlap_terms = query_terms & (context_terms | target_terms)
        min_overlap = int(getattr(self.config, "bridge_min_relation_overlap", 0))
        if len(overlap_terms) < min_overlap:
            return 0.0
        score = floor + weight * min(4, len(overlap_terms))
        if self._entity_in_title(ent, bj):
            score += float(getattr(self.config, "bridge_title_relation_bonus", 0.0))
        return min(1.5, score)

    def _link_strength(self, ent, degree, idf_max):
        degree_power = float(getattr(self.config, "entity_degree_power", 0.0))
        idf = self.ent_idf.get(ent, 1.0)
        idf_norm = idf / max(1e-6, float(idf_max))
        return float(np.clip(idf_norm / (degree ** degree_power), 1e-6, 1.0))

    def _path_evidence(self, a_sim, link, relation):
        alpha = max(0.0, float(getattr(self.config, "bridge_evidence_alpha", 0.45)))
        beta = max(0.0, float(getattr(self.config, "bridge_evidence_beta", 0.30)))
        gamma = max(0.0, float(getattr(self.config, "bridge_evidence_gamma", 0.25)))
        total = max(1e-6, alpha + beta + gamma)
        vals = (
            (float(np.clip(a_sim, 1e-6, 1.0)), alpha / total),
            (float(np.clip(link, 1e-6, 1.0)), beta / total),
            (float(np.clip(relation, 1e-6, 1.0)), gamma / total),
        )
        return math.exp(sum(weight * math.log(value) for value, weight in vals))

    def _select_mmr(self, scores, kb_emb, doc_embs):
        if not scores:
            return []
        max_docs = int(getattr(self.config, "bridge_max_docs", 0))
        if max_docs <= 0:
            return []
        mu = max(0.0, float(getattr(self.config, "bridge_mmr_mu", 0.0)))
        remaining = dict(scores)
        selected = []
        selected_pos = []
        while remaining and len(selected) < max_docs:
            best = None
            best_score = -float("inf")
            for pi, evidence in remaining.items():
                redundancy = self._selection_redundancy(
                    pi, kb_emb, doc_embs, selected_pos)
                score = evidence - mu * redundancy
                if score > best_score:
                    best = pi
                    best_score = score
            if best is None:
                break
            if best_score <= 0.0:
                self.last_stats["bridge_mmr_stopped"] += 1
                break
            selected.append((int(best), float(remaining[best])))
            selected_pos.append(int(best))
            remaining.pop(best, None)
        return selected

    def _selection_redundancy(self, pi, kb_emb, doc_embs, selected_pos):
        sims = []
        if kb_emb.size:
            sims.append(float((doc_embs[int(pi)] @ kb_emb.T).max()))
        if selected_pos:
            selected_emb = doc_embs[np.array(selected_pos, dtype=np.int64)]
            sims.append(float((doc_embs[int(pi)] @ selected_emb.T).max()))
        if not sims:
            return 0.0
        return max(0.0, max(sims))

    def _entity_context_terms(self, pi, ent):
        key = (int(pi), ent)
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached
        text = f"{self.pi_to_title.get(int(pi), '')}. {self.pi_to_text.get(int(pi), '')}"
        lower = text.lower()
        pos = lower.find(ent)
        if pos < 0:
            terms = self._content_terms(self.pi_to_title.get(int(pi), ""))
            self._context_cache[key] = terms
            return terms
        left = max(0, pos - 160)
        right = min(len(text), pos + len(ent) + 160)
        terms = self._content_terms(text[left:right])
        self._context_cache[key] = terms
        return terms

    def _entity_in_title(self, ent, pi):
        title = self._normalize_entity(self.pi_to_title.get(int(pi), ""))
        return bool(ent and (ent == title or ent in title or title in ent))

    def _novelty_score(self, pi, kb_emb, doc_embs):
        if kb_emb.size == 0:
            return 1.0
        max_sim = float((doc_embs[int(pi)] @ kb_emb.T).max())
        return 1.0 - max(0.0, max_sim)
