"""
Lightweight graph-based retrieval for Mo2 (no-LLM HippoRAG/LightRAG analogue).

Builds a per-strategy passage-entity bipartite graph using spaCy NER.
Each LightGraphRAG instance supports incremental index() / delete() over its
own KB and computes Recall@K via Personalized PageRank seeded from the
query's entities (with a dense embedding prior to handle entity-poor queries).

Pool-wide and per-query entity extractions are cached on disk so the
per-strategy state mutations remain O(|diff|).
"""
import json, hashlib, re
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
                 alpha=0.5, n_iter=15,
                 dense_prior_k=10, dense_prior_w=0.3):
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
        self.passages = set()
        self.passage_to_ents = {}
        self.ent_to_passages = defaultdict(set)

    def index(self, doc_ids):
        n = 0
        for did in doc_ids:
            if did in self.passages:
                continue
            ents = set(self.pool_entities.get(did, []))
            self.passages.add(did)
            self.passage_to_ents[did] = ents
            for e in ents:
                self.ent_to_passages[e].add(did)
            n += 1
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
            n += 1
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

    def _build_transition(self, passages_list, ents_list,
                          p_to_pos, e_to_pos):
        n_p = len(passages_list)
        n_e = len(ents_list)
        n = n_p + n_e
        rows, cols, data = [], [], []
        # passage column -> entity rows: weight 1/deg(passage)
        for did, ents in self.passage_to_ents.items():
            pi = p_to_pos[did]
            d_p = len(ents)
            if d_p == 0:
                continue
            inv = 1.0 / d_p
            for e in ents:
                ei = n_p + e_to_pos[e]
                rows.append(ei); cols.append(pi); data.append(inv)
        # entity column -> passage rows: weight 1/deg(entity)
        for e, pset in self.ent_to_passages.items():
            ei = n_p + e_to_pos[e]
            d_e = len(pset)
            if d_e == 0:
                continue
            inv = 1.0 / d_e
            for did in pset:
                pi = p_to_pos[did]
                rows.append(pi); cols.append(ei); data.append(inv)
        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def retrieve(self, qidx, k):
        if not self.passages:
            return []
        passages_list = sorted(self.passages)
        p_to_pos = {p: i for i, p in enumerate(passages_list)}
        ents_list = sorted(self.ent_to_passages.keys())
        if not ents_list:
            return self._dense_topk(qidx, passages_list, k)
        e_to_pos = {e: i for i, e in enumerate(ents_list)}
        n_p = len(passages_list)
        n_e = len(ents_list)
        n = n_p + n_e

        q_ents = self._query_ents(qidx)
        matched = [e for e in q_ents if e in e_to_pos]
        seed = np.zeros(n)
        if matched:
            for e in matched:
                seed[n_p + e_to_pos[e]] += 1.0
            seed /= seed.sum()
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

        T = self._build_transition(passages_list, ents_list, p_to_pos, e_to_pos)
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


def recall_at_k_graph(graph, queries, doc_pool, k_list):
    """Recall@K using LightGraphRAG retrieval; mirrors utils.recall_at_k contract."""
    if graph is None or not graph.passages:
        return {k: 0.0 for k in k_list}
    d2t = {d['doc_id']: d['title'] for d in doc_pool}
    max_k = max(k_list)
    recalls = {k: [] for k in k_list}
    for q in queries:
        gold = set(q['sf_titles'])
        if not gold:
            continue
        retrieved_ids = graph.retrieve(q['qidx'], max_k)
        retrieved_titles = [d2t.get(d) for d in retrieved_ids]
        for k in k_list:
            r = len(gold & set(retrieved_titles[:k])) / len(gold)
            recalls[k].append(r)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in recalls.items()}
