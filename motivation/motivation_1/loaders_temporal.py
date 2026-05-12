"""
Temporal query-drift loaders for the motivation experiments.

TREC-COVID: 50 biomedical queries over the CORD-19 scientific literature
(171 K papers). Relevance judgments from TREC 2020-2021 (BEIR version).
Highly-relevant documents (rel >= 2) serve as supporting facts (SF).
Drift is grounded in the semantic shift across query topics:
  early-pandemic (origin, transmission, epidemiology)
  late-pandemic (treatments, vaccines, clinical outcomes)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger('motivation_temporal')

try:
    from config import SEED, PROJECT_DIR
except ImportError:
    SEED = 42
    PROJECT_DIR = Path('/home/jyliu/RAG-project')


def load_trec_covid(
    n_distractor_ratio: float = 4.0,
    max_docs: int = 25_000,
    max_queries: int = 50,
    rel_threshold: int = 2,
):
    """Load TREC-COVID for temporal query-drift motivation experiment.

    Returns (doc_pool, queries, title_to_idx) matching the interface expected
    by cluster_and_build_stream.  SF documents are those with rel >= rel_threshold
    in the TREC-COVID qrels (default: rel >= 2, "highly relevant").

    Distractor docs are randomly sampled non-SF docs to fill the pool.
    """
    try:
        import ir_datasets
    except ImportError:
        raise ImportError(
            "ir_datasets required: pip install ir-datasets"
        )

    log.info("[trec_covid] Loading via ir_datasets…")
    dataset = ir_datasets.load('beir/trec-covid')

    # 1. Build doc lookup
    doc_lookup = {}
    for doc in dataset.docs_iter():
        doc_lookup[doc.doc_id] = {
            'title': doc.title or doc.doc_id,
            'text': (doc.title or '') + ' ' + (doc.text or ''),
        }
    log.info(f"[trec_covid] {len(doc_lookup):,} docs loaded")

    # 2. Build qrels
    sf_for_query = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= rel_threshold:
            sf_for_query[qrel.query_id].add(qrel.doc_id)

    # 3. Load queries
    raw_queries = []
    for q in dataset.queries_iter():
        if q.query_id not in sf_for_query:
            continue
        raw_queries.append({
            'query_id': q.query_id,
            'question': q.text,
            'sf_doc_ids': list(sf_for_query[q.query_id]),
        })
    raw_queries = raw_queries[:max_queries]
    log.info(f"[trec_covid] {len(raw_queries)} queries with SF docs")

    # 4. Collect unique SF doc ids
    all_sf_ids = set()
    for q in raw_queries:
        all_sf_ids.update(q['sf_doc_ids'])
    all_sf_ids = {d for d in all_sf_ids if d in doc_lookup}
    log.info(f"[trec_covid] unique SF docs (rel>={rel_threshold}): {len(all_sf_ids):,}")

    # 5. Sample distractor docs
    n_distractor = min(
        int(len(all_sf_ids) * n_distractor_ratio),
        max_docs - len(all_sf_ids),
        len(doc_lookup) - len(all_sf_ids),
    )
    rng = np.random.default_rng(SEED + 77)
    non_sf_ids = [d for d in doc_lookup if d not in all_sf_ids]
    rng.shuffle(non_sf_ids)
    distractor_ids = set(non_sf_ids[:n_distractor])
    log.info(f"[trec_covid] distractor docs: {len(distractor_ids):,}  "
             f"(ratio={len(distractor_ids)/max(1,len(all_sf_ids)):.1f}:1)")

    # 6. Build shuffled doc_pool
    pool_ids = list(all_sf_ids | distractor_ids)
    order = np.arange(len(pool_ids))
    rng.shuffle(order)
    pool_ids = [pool_ids[int(i)] for i in order]

    doc_pool = []
    for i, did in enumerate(pool_ids):
        raw = doc_lookup[did]
        doc_pool.append({
            'doc_id': f'tc{i:06d}',
            'title': did,       # original doc_id as title for SF lookup
            'text': raw['text'][:2000],
        })
    title_to_idx = {d['title']: i for i, d in enumerate(doc_pool)}
    log.info(f"[trec_covid] doc_pool: {len(doc_pool):,} docs")

    # 7. Build query list
    queries = []
    for q in raw_queries:
        sf_titles = [did for did in q['sf_doc_ids'] if did in title_to_idx]
        if not sf_titles:
            continue
        queries.append({
            'qidx':      len(queries),
            'question':  q['question'],
            'answer':    '',
            'sf_titles': sf_titles,
            'ctx_titles': sf_titles,
            'qtype':     'trec_covid',
        })
    log.info(
        f"[trec_covid] {len(queries)} queries  "
        f"avg SF/query={sum(len(q['sf_titles']) for q in queries)/max(1,len(queries)):.1f}"
    )
    return doc_pool, queries, title_to_idx


TEMPORAL_LOADERS = {
    'trec_covid': load_trec_covid,
}
