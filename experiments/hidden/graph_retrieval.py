"""
Lightweight graph-based retrieval for Mo2 (no-LLM HippoRAG/LightRAG analogue).

Graph construction (per strategy, over its current KB):
  - passage <-> entity bipartite edges weighted by entity IDF, mirroring
    HippoRAG's down-weighting of generic concept nodes that would otherwise
    dominate PPR mass purely by degree.
  - entity <-> entity edges from same-passage co-occurrence (LightRAG's
    entity-relation graph without an LLM relation extractor): PPR mass can
    flow between semantically related entities in 2 hops instead of 4.
  - dense embedding prior on the passage side for entity-poor queries.

Each LightGraphRAG instance supports incremental index() / delete() over its
own KB and computes Recall@K via Personalized PageRank seeded from the
query's entities (IDF-weighted, with substring fallback for partial-name
queries) plus the dense passage prior. Pool-wide and per-query entity
extractions are cached on disk so per-strategy state mutations remain
O(|diff|) and graph rebuilds touch only the affected nodes.
"""
import json, hashlib, re, math
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix

from config import CACHE_DIR, log


_NLP = None
_BAD_LABELS = {'CARDINAL', 'ORDINAL', 'PERCENT', 'MONEY', 'QUANTITY', 'TIME', 'DATE'}


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load('en_core_web_sm',
                          disable=['lemmatizer', 'parser', 'tagger', 'attribute_ruler'])
    return _NLP


def _normalise(text):
    t = re.sub(r'\s+', ' ', text.lower().strip())
    if t.startswith('the '):
        t = t[4:]
    return t


def _extract_for_texts(texts, batch_size=512):
    nlp = _get_nlp()
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        ents = set()
        for e in doc.ents:
            if e.label_ in _BAD_LABELS:
                continue
            n = _normalise(e.text)
            if 1 < len(n) < 80:
                ents.add(n)
        out.append(sorted(ents))
    return out


def extract_pool_entities(doc_pool, tag):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"pool_ents_{len(doc_pool)}_{tag}_v1".encode()).hexdigest()[:12]
    cache = CACHE_DIR / f'pool_ents_{key}.json'
    if cache.exists():
        log.info(f"Loading cached pool entities from {cache.name}")
        return json.load(open(cache))
    log.info(f"Extracting entities for {len(doc_pool)} pool docs (spaCy NER)...")
    texts = [f"{d['title']}. {d['text'][:512]}" for d in doc_pool]
    ents = _extract_for_texts(texts)
    out = {d['doc_id']: e for d, e in zip(doc_pool, ents)}
    json.dump(out, open(cache, 'w'))
    log.info(f"Saved {cache.name}")
    return out


def extract_query_entities(queries, tag):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"query_ents_{len(queries)}_{tag}_v1".encode()).hexdigest()[:12]
    cache = CACHE_DIR / f'query_ents_{key}.json'
    if cache.exists():
        log.info(f"Loading cached query entities from {cache.name}")
        out = json.load(open(cache))
        return {int(q['qidx']) if isinstance(out, dict) else i: e
                for i, e in enumerate(out)} if isinstance(out, list) else out
    log.info(f"Extracting entities for {len(queries)} queries (spaCy NER)...")
    texts = [q['question'] for q in queries]
    ents = _extract_for_texts(texts)
    json.dump(ents, open(cache, 'w'))
    log.info(f"Saved {cache.name}")
    return ents


class LightGraphRAG:
    """Per-strategy passage-entity graph with incremental index/delete and PPR.

    Designed as a no-LLM substitute for HippoRAG/LightRAG retrieval:
      - Nodes: passages (current KB) + entity strings extracted from those
        passages via spaCy NER.
      - Edges: passage <-> entity bipartite (uniform weight).
      - Retrieval per query:
          seed = (1-w) * uniform over query entities matched in graph
                 + w   * dense passage prior (top-N by query<->passage cosine)
          x = (1-alpha) * seed + alpha * P x          (n_iter steps)
          rank passages by stationary mass.
      - Falls back to pure dense ranking when no entity is matched.
    """

    def __init__(self, pool_entities, query_entities,
                 doc_embs, query_embs, d2p, p2d,
                 alpha=0.5, n_iter=20,
                 dense_prior_k=20, dense_prior_w=0.3,
                 use_ee_edges=True, ee_edge_weight=0.5,
                 idf_smoothing=1.0, max_ents_per_passage=20):
        self.pool_entities = pool_entities
        self.query_entities = query_entities
        self.doc_embs = doc_embs
        self.query_embs = query_embs
        self.d2p = d2p
        self.p2d = p2d
        self.alpha = alpha
        self.n_iter = n_iter
        self.dense_prior_k = dense_prior_k
        self.dense_prior_w = dense_prior_w
        self.use_ee_edges = use_ee_edges
        self.ee_edge_weight = ee_edge_weight
        self.idf_smoothing = idf_smoothing
        self.max_ents_per_passage = max_ents_per_passage
        self.passages = set()
        self.passage_to_ents = {}
        self.ent_to_passages = defaultdict(set)
        # Sparse entity co-occurrence counts over the current KB only.
        # {e1: {e2: count}} with e1 < e2 to dedupe symmetric pairs.
        self.ent_cooc = defaultdict(dict)
        # Cached transition matrix. Invalidated whenever passages change.
        self._T_cache = None           # csr_matrix or None
        self._T_cache_state = None     # (frozenset(passages), sorted_ents) snapshot

    def _invalidate_T_cache(self):
        self._T_cache = None
        self._T_cache_state = None

    def _bump_cooc(self, ents, sign):
        if not self.use_ee_edges:
            return
        elist = sorted(ents)
        m = len(elist)
        for i in range(m):
            ei = elist[i]
            row = self.ent_cooc[ei]
            for j in range(i + 1, m):
                ej = elist[j]
                v = row.get(ej, 0) + sign
                if v <= 0:
                    row.pop(ej, None)
                else:
                    row[ej] = v
            if not row:
                self.ent_cooc.pop(ei, None)

    def index(self, doc_ids):
        n = 0
        cap = self.max_ents_per_passage
        for did in doc_ids:
            if did in self.passages:
                continue
            ents = list(self.pool_entities.get(did, []))
            if cap and len(ents) > cap:
                ents = ents[:cap]
            ents = set(ents)
            self.passages.add(did)
            self.passage_to_ents[did] = ents
            for e in ents:
                self.ent_to_passages[e].add(did)
            self._bump_cooc(ents, +1)
            n += 1
        if n:
            self._invalidate_T_cache()
        return n

    def delete(self, doc_ids):
        n = 0
        for did in doc_ids:
            if did not in self.passages:
                continue
            ents = self.passage_to_ents.pop(did)
            self.passages.discard(did)
            for e in ents:
                self.ent_to_passages[e].discard(did)
                if not self.ent_to_passages[e]:
                    del self.ent_to_passages[e]
            self._bump_cooc(ents, -1)
            n += 1
        if n:
            self._invalidate_T_cache()
        return n

    def sync_to(self, target_kb):
        target = set(target_kb)
        add = target - self.passages
        rm = self.passages - target
        a = self.index(add) if add else 0
        r = self.delete(rm) if rm else 0
        return a, r

    def _query_ents(self, qidx):
        if isinstance(self.query_entities, dict):
            return self.query_entities.get(qidx, []) or self.query_entities.get(str(qidx), [])
        if 0 <= qidx < len(self.query_entities):
            return self.query_entities[qidx]
        return []

    def _dense_topk(self, qidx, passages_list, k):
        if not passages_list:
            return []
        qe = self.query_embs[qidx]
        kb_idx = np.fromiter((self.d2p[d] for d in passages_list),
                             dtype=np.int64, count=len(passages_list))
        sims = self.doc_embs[kb_idx] @ qe
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            order = np.argpartition(-sims, k)[:k]
            order = order[np.argsort(-sims[order])]
        return [passages_list[i] for i in order[:k]]

    def _entity_idf(self, ents_list, n_p):
        # IDF over current KB: rare entities become more discriminative seeds
        # and hubs (HippoRAG-style down-weighting of generic concept nodes).
        s = self.idf_smoothing
        idf = np.empty(len(ents_list), dtype=np.float64)
        for i, e in enumerate(ents_list):
            df = len(self.ent_to_passages.get(e, ()))
            idf[i] = math.log((n_p + s) / (df + s)) + 1.0
        return np.maximum(idf, 1e-3)

    def _build_transition(self, passages_list, ents_list,
                          p_to_pos, e_to_pos, ent_idf):
        n_p = len(passages_list)
        n_e = len(ents_list)
        n = n_p + n_e
        rows, cols, data = [], [], []
        col_sum = np.zeros(n, dtype=np.float64)
        # Passage column -> entity rows: weight = entity IDF (column-normalised
        # below). Effectively replaces the original 1/deg(passage) weighting
        # with one that prefers rare-entity transitions.
        for did, ents in self.passage_to_ents.items():
            pi = p_to_pos[did]
            for e in ents:
                ei_pos = e_to_pos[e]
                w = ent_idf[ei_pos]
                rows.append(n_p + ei_pos); cols.append(pi); data.append(w)
                col_sum[pi] += w
        # Entity column -> passage rows: uniform 1/deg(entity).
        for e, pset in self.ent_to_passages.items():
            ei = n_p + e_to_pos[e]
            d_e = len(pset)
            if d_e == 0:
                continue
            w = 1.0 / d_e
            for did in pset:
                pi = p_to_pos[did]
                rows.append(pi); cols.append(ei); data.append(w)
                col_sum[ei] += w
        # Optional symmetric entity<->entity co-occurrence edges (LightRAG-style
        # KG without an LLM relation extractor).
        if self.use_ee_edges and self.ent_cooc:
            ew = self.ee_edge_weight
            for e1, neigh in self.ent_cooc.items():
                if e1 not in e_to_pos:
                    continue
                ci = n_p + e_to_pos[e1]
                for e2, c in neigh.items():
                    if e2 not in e_to_pos:
                        continue
                    cj = n_p + e_to_pos[e2]
                    w = ew * float(c)
                    rows.append(cj); cols.append(ci); data.append(w)
                    rows.append(ci); cols.append(cj); data.append(w)
                    col_sum[ci] += w
                    col_sum[cj] += w
        data = np.asarray(data, dtype=np.float64)
        cols_arr = np.asarray(cols, dtype=np.int64)
        cs = col_sum[cols_arr]
        cs[cs == 0] = 1.0
        data = data / cs
        return csr_matrix((data, (rows, cols_arr)), shape=(n, n))

    def _match_query_entities(self, q_ents, ent_set):
        """Exact match first; fall back to substring containment so partial
        names ('obama' vs 'barack obama') still seed the graph. Returns a
        list of (kb_entity, raw_weight) pairs (IDF applied later)."""
        out = []
        unmatched = []
        for qe in q_ents:
            if qe in ent_set:
                out.append((qe, 1.0))
            else:
                unmatched.append(qe)
        if unmatched and ent_set:
            for qe in unmatched:
                hits = [ke for ke in ent_set
                        if (qe in ke or ke in qe) and ke != qe]
                if hits:
                    w = 0.5 / len(hits)
                    for ke in hits:
                        out.append((ke, w))
        return out

    def retrieve_with_emb(self, emb, ents, k):
        """Like retrieve() but uses an explicit embedding vector and entity list
        instead of looking up by qidx. Used for sub-question multi-query fusion.
        """
        if not self.passages:
            return [], np.array([])
        passages_list = sorted(self.passages)
        p_to_pos = {p: i for i, p in enumerate(passages_list)}
        ents_list = sorted(self.ent_to_passages.keys())
        n_p = len(passages_list)
        if not ents_list:
            # pure dense
            kb_idx = np.fromiter((self.d2p[d] for d in passages_list), dtype=np.int64, count=n_p)
            sims = self.doc_embs[kb_idx] @ emb
            top_k = min(k, n_p)
            top = np.argsort(sims)[-top_k:][::-1]
            return [passages_list[i] for i in top], sims[top]
        e_to_pos = {e: i for i, e in enumerate(ents_list)}
        n_e = len(ents_list)
        n = n_p + n_e
        ent_idf = self._entity_idf(ents_list, n_p)

        ent_set = set(ents_list)
        matched = self._match_query_entities(ents, ent_set)
        seed = np.zeros(n)
        if matched:
            for ke, raw_w in matched:
                ei = e_to_pos[ke]
                seed[n_p + ei] += raw_w * ent_idf[ei]
            tot = seed.sum()
            if tot > 0:
                seed /= tot
        kb_idx = np.fromiter((self.d2p[d] for d in passages_list), dtype=np.int64, count=n_p)
        sims = self.doc_embs[kb_idx] @ emb
        kk = min(self.dense_prior_k, n_p)
        top = np.argpartition(sims, -kk)[-kk:]
        s = np.maximum(sims[top], 0)
        prior = np.zeros(n_p)
        if s.sum() > 0:
            prior[top] = s / s.sum()
        else:
            prior[top] = 1.0 / kk
        if matched:
            seed = (1 - self.dense_prior_w) * seed
            seed[:n_p] += self.dense_prior_w * prior
        else:
            seed[:n_p] = prior
        if seed.sum() <= 0:
            kb_idx2 = np.fromiter((self.d2p[d] for d in passages_list), dtype=np.int64, count=n_p)
            sims2 = self.doc_embs[kb_idx2] @ emb
            top2 = np.argsort(sims2)[-k:][::-1]
            return [passages_list[i] for i in top2], sims2[top2]
        seed /= seed.sum()
        cache_key = (len(passages_list), len(ents_list))
        if (self._T_cache is None or self._T_cache_state != cache_key):
            T = self._build_transition(passages_list, ents_list, p_to_pos, e_to_pos, ent_idf)
            self._T_cache = T
            self._T_cache_state = cache_key
        T = self._T_cache
        r = seed.copy()
        alpha = self.alpha
        for _ in range(self.n_iter):
            r = (1 - alpha) * (T @ r) + alpha * seed
        scores = r[:n_p]
        top_k = min(k, n_p)
        top = np.argsort(scores)[-top_k:][::-1]
        return [passages_list[i] for i in top], scores[top]

    def retrieve(self, qidx, k):
        if not self.passages:
            return []
        passages_list = sorted(self.passages)
        p_to_pos = {p: i for i, p in enumerate(passages_list)}
        ents_list = sorted(self.ent_to_passages.keys())
        n_p = len(passages_list)
        if not ents_list:
            return self._dense_topk(qidx, passages_list, k)
        e_to_pos = {e: i for i, e in enumerate(ents_list)}
        n_e = len(ents_list)
        n = n_p + n_e
        ent_idf = self._entity_idf(ents_list, n_p)

        q_ents = self._query_ents(qidx)
        ent_set = set(ents_list)
        matched = self._match_query_entities(q_ents, ent_set)
        seed = np.zeros(n)
        if matched:
            # IDF-weighted seed: rare matched entities steer PPR more.
            for ke, raw_w in matched:
                ei = e_to_pos[ke]
                seed[n_p + ei] += raw_w * ent_idf[ei]
            tot = seed.sum()
            if tot > 0:
                seed /= tot
        # Dense prior on passage side
        qe = self.query_embs[qidx]
        kb_idx = np.fromiter((self.d2p[d] for d in passages_list),
                             dtype=np.int64, count=n_p)
        sims = self.doc_embs[kb_idx] @ qe
        kk = min(self.dense_prior_k, n_p)
        top = np.argpartition(sims, -kk)[-kk:]
        s = np.maximum(sims[top], 0)
        prior = np.zeros(n_p)
        if s.sum() > 0:
            prior[top] = s / s.sum()
        else:
            prior[top] = 1.0 / kk
        if matched:
            seed = (1 - self.dense_prior_w) * seed
            seed[:n_p] += self.dense_prior_w * prior
        else:
            seed[:n_p] = prior
        if seed.sum() <= 0:
            return self._dense_topk(qidx, passages_list, k)
        seed /= seed.sum()

        cache_key = (len(passages_list), len(ents_list))
        if (self._T_cache is None or
                self._T_cache_state != cache_key):
            T = self._build_transition(passages_list, ents_list,
                                       p_to_pos, e_to_pos, ent_idf)
            self._T_cache = T
            self._T_cache_state = cache_key
            self._T_cache_plist = passages_list
            self._T_cache_p2pos = p_to_pos
        else:
            T = self._T_cache
            passages_list = self._T_cache_plist
            p_to_pos = self._T_cache_p2pos
        x = seed.copy()
        for _ in range(self.n_iter):
            x = (1 - self.alpha) * seed + self.alpha * (T @ x)

        p_scores = x[:n_p]
        # Tie-break with dense sim to stabilise PPR over symmetric graphs
        p_scores = p_scores + 1e-6 * sims
        if k >= n_p:
            order = np.argsort(-p_scores)
        else:
            order = np.argpartition(-p_scores, k)[:k]
            order = order[np.argsort(-p_scores[order])]
        return [passages_list[i] for i in order[:k]]


def recall_at_k_graph(graph, queries, doc_pool, k_list,
                      sub_embs=None, sub_ents=None):
    """Recall@K using LightGraphRAG retrieval; mirrors utils.recall_at_k contract.

    If sub_embs / sub_ents provided (LLM query decomposition), runs PPR for each
    sub-question independently and takes the union of retrieved docs (recall-oriented).
    """
    if graph is None or not graph.passages:
        return {k: 0.0 for k in k_list}
    d2t = {d['doc_id']: d['title'] for d in doc_pool}
    max_k = max(k_list)
    recalls = {k: [] for k in k_list}
    for qi, q in enumerate(queries):
        gold = set(q['sf_titles'])
        if not gold:
            continue
        # Base retrieval using original query
        base_ids = graph.retrieve(q['qidx'], max_k)
        all_retrieved = list(base_ids)

        # Sub-question union (LLM expansion)
        if sub_embs is not None and qi < len(sub_embs):
            for emb, ents in zip(sub_embs[qi], sub_ents[qi] if sub_ents else [[]]*len(sub_embs[qi])):
                sub_ids, _ = graph.retrieve_with_emb(emb, ents, max_k)
                # Add sub-retrieved docs that aren't already in base (union)
                seen = set(all_retrieved)
                for d in sub_ids:
                    if d not in seen:
                        all_retrieved.append(d)
                        seen.add(d)

        retrieved_titles = [d2t.get(d) for d in all_retrieved]
        for k in k_list:
            r = len(gold & set(retrieved_titles[:k])) / len(gold)
            recalls[k].append(r)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in recalls.items()}



def recall_at_k_entity_expand(pool_ents, kb_ids, queries, d2p, doc_embs,
                               title_to_idx, query_embs,
                               step1_k=3, k_list=(5,)):
    """2-step entity expansion recall@K.

    Faster and more effective than PPR for bridge queries.
    Step 1: dense top-step1_k from KB (by embedding similarity)
    Step 2: collect entities from step1 docs -> find KB neighbors sharing those entities
    Ranking: step1 docs by embedding sim; step2 docs by entity overlap count
    """
    import numpy as _np
    from collections import defaultdict
    if not kb_ids:
        return {k: 0.0 for k in k_list}

    # Build doc_id -> title from title_to_idx + d2p
    idx2did = {v: k for k, v in d2p.items()}
    d2t = {}
    for title, idx in title_to_idx.items():
        did = idx2did.get(idx)
        if did is not None:
            d2t[did] = title

    kb_list = sorted(kb_ids)
    kb_idx = _np.array([d2p[d] for d in kb_list], dtype=_np.int64)

    # Build entity -> set of KB doc_ids index
    ent_to_kb = defaultdict(set)
    for did in kb_ids:
        for ent in pool_ents.get(did, []):
            ent_to_kb[ent].add(did)

    max_k = max(k_list)
    recalls = {k: [] for k in k_list}
    for q in queries:
        gold = set(q.get('sf_titles', []))
        if not gold:
            continue
        qe = query_embs[q['qidx']]
        sims = doc_embs[kb_idx] @ qe

        # Step 1: dense top step1_k
        s1_n = min(step1_k, len(kb_list))
        s1_idx = _np.argsort(-sims)[:s1_n]
        s1_docs = [kb_list[i] for i in s1_idx]
        s1_set = set(s1_docs)

        # Step 2: entity expansion - score by overlap count
        s1_ents = set()
        for did in s1_docs:
            s1_ents.update(pool_ents.get(did, []))

        s2_overlap = defaultdict(int)
        for ent in s1_ents:
            for nb in ent_to_kb.get(ent, set()):
                if nb not in s1_set:
                    s2_overlap[nb] += 1

        # Ranked list: step1 by embedding sim, then step2 by entity overlap count
        step2_sorted = sorted(s2_overlap.items(), key=lambda x: -x[1])
        all_ranked = s1_docs + [d for d, _ in step2_sorted]

        seen, ranked_titles = set(), []
        for did in all_ranked:
            if did not in seen:
                seen.add(did)
                ranked_titles.append(d2t.get(did))
            if len(ranked_titles) >= max_k:
                break

        for k in k_list:
            r = len(gold & set(ranked_titles[:k])) / len(gold)
            recalls[k].append(r)

    return {k: float(_np.mean(v)) if v else 0.0 for k, v in recalls.items()}
