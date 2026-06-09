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


def load_trec_covid_temporal(
    n_distractor_ratio: float = 3.0,
    max_docs: int = 30_000,
    rel_threshold: int = 1,
):
    """Real temporal TREC-COVID stream using cord19 per-round (rounds 1–5).

    Round-by-round drift (Apr–Sep 2020):
      R1 (30q, Apr-2020):  origin / transmission
      R2 (35q, May-2020):  epidemiology
      R3 (40q, Jun-2020):  treatments
      R4 (45q, Jul-2020):  comorbidities
      R5 (50q, Sep-2020):  vaccines

    Each query is tagged with its FIRST-appearance round; the helper
    `is_tail` is set True for rounds >= 3 so the existing stream builder
    naturally treats R1+R2 as head and R3+R4+R5 as tail (real semantic
    drift across pandemic phases).

    Returns (doc_pool, queries, title_to_idx) — each query carries:
        'round': 1..5,  'is_tail': bool (round >= 3)
    """
    try:
        import ir_datasets
    except ImportError:
        raise ImportError("ir_datasets required: pip install ir-datasets")

    log.info("[trec_covid_temporal] Loading rounds 1–5 …")

    # Track first-appearance round for each query
    qid_first_round: dict[str, int] = {}
    qid_text:        dict[str, str] = {}
    qid_sf:          dict[str, set[str]] = defaultdict(set)
    doc_lookup:      dict[str, dict] = {}

    for rnd in (1, 2, 3, 4, 5):
        ds = ir_datasets.load(f'cord19/trec-covid/round{rnd}')
        # docs: only collect once per round (cord19 docs same set across rounds modulo timing)
        if rnd == 5:  # round 5 has the most docs available
            for doc in ds.docs_iter():
                if doc.doc_id in doc_lookup: continue
                title = (doc.title or '').strip() or doc.doc_id
                text  = ((doc.title or '') + ' ' + (doc.abstract or ''))[:2000]
                doc_lookup[doc.doc_id] = {'title': title, 'text': text}
        for q in ds.queries_iter():
            qid = q.query_id
            if qid not in qid_first_round:
                qid_first_round[qid] = rnd
                qid_text[qid] = q.title or q.description or qid
        for qrel in ds.qrels_iter():
            if qrel.relevance >= rel_threshold:
                qid_sf[qrel.query_id].add(qrel.doc_id)

    log.info(f"[trec_covid_temporal] {len(doc_lookup):,} docs, "
             f"{len(qid_first_round)} unique queries")

    # Collect SF doc IDs that exist in our doc lookup
    all_sf_ids = set()
    for qid, sfs in qid_sf.items():
        all_sf_ids.update(d for d in sfs if d in doc_lookup)
    log.info(f"[trec_covid_temporal] SF docs (rel>={rel_threshold}): {len(all_sf_ids):,}")

    # Sample distractors
    n_distractor = min(int(len(all_sf_ids) * n_distractor_ratio),
                       max_docs - len(all_sf_ids),
                       len(doc_lookup) - len(all_sf_ids))
    rng = np.random.default_rng(SEED + 88)
    non_sf = [d for d in doc_lookup if d not in all_sf_ids]
    rng.shuffle(non_sf)
    distractor_ids = set(non_sf[:n_distractor])
    log.info(f"[trec_covid_temporal] distractors: {len(distractor_ids):,}")

    pool_ids = list(all_sf_ids | distractor_ids)
    rng.shuffle(pool_ids)
    doc_pool = [{'doc_id': f'tct{i:06d}', 'title': did, 'text': doc_lookup[did]['text']}
                for i, did in enumerate(pool_ids)]
    title_to_idx = {d['title']: i for i, d in enumerate(doc_pool)}

    # Build queries (one per unique qid), tag with round + is_tail (round>=3)
    queries = []
    for qid, rnd in sorted(qid_first_round.items(), key=lambda kv: (kv[1], kv[0])):
        sf_titles = [d for d in qid_sf.get(qid, set()) if d in title_to_idx]
        if not sf_titles:
            continue
        queries.append({
            'qidx':       len(queries),
            'question':   qid_text[qid],
            'answer':     '',
            'sf_titles':  sf_titles,
            'ctx_titles': sf_titles,
            'qtype':      'trec_covid_temporal',
            'round':      rnd,
            'is_tail':    rnd >= 3,
            'cluster':    rnd,
        })
    head = sum(1 for q in queries if not q['is_tail'])
    tail = sum(1 for q in queries if q['is_tail'])
    log.info(f"[trec_covid_temporal] {len(queries)} queries  head(R1-R2)={head}  tail(R3-R5)={tail}")
    return doc_pool, queries, title_to_idx


TEMPORAL_LOADERS['trec_covid_temporal'] = load_trec_covid_temporal


# ═════════════════════════════════════════════════════════════════════════
# StreamingQA — DeepMind real-world news QA over 14 years (2007-2020).
#
# Liška et al. (2022), "StreamingQA: A Benchmark for Adaptation to New
# Knowledge over Time in Question Answering Models", ICML 2022.
# https://arxiv.org/abs/2205.11388
#
# We use the HuggingFace mirror `bg51717/streamingqa` (test split, 36K Qs).
# Each example has (question, context, answers); the context is the
# source news article. ~5K test questions explicitly mention a year in the
# question text — we extract that year as the temporal anchor and bin
# 13 years (2008-2020) into 5 rounds:
#   R1: 2008-2010  R2: 2011-2013  R3: 2014-2016  R4: 2017-2018  R5: 2019-2020
# This matches the gradual semantic drift across financial crisis →
# Arab Spring → Eurozone → Brexit/Trump → COVID era.
# ═════════════════════════════════════════════════════════════════════════
def load_streamingqa_temporal(
    n_distractors: int = 100_000,
    rel_threshold: int = 1,
):
    """StreamingQA test split, year-anchored temporal stream (5 rounds)."""
    import re
    from datasets import load_dataset

    log.info('[streamingqa_temporal] Loading bg51717/streamingqa test split…')
    ds = load_dataset('bg51717/streamingqa', split='test')
    yr_pat = re.compile(r'\b(19[7-9]\d|20[0-2]\d)\b')

    YEAR_TO_ROUND = {}
    for y, r in ((range(2008, 2011), 1), (range(2011, 2014), 2),
                 (range(2014, 2017), 3), (range(2017, 2019), 4),
                 (range(2019, 2021), 5)):
        for yy in y:
            YEAR_TO_ROUND[yy] = r

    # Pass 1: dated queries only; first context becomes the gold doc.
    ctx_to_did: dict[str, str] = {}
    doc_pool_raw: list[dict] = []
    queries: list[dict] = []
    no_ctx = no_year = 0
    for row in ds:
        m = yr_pat.search(row['question'])
        if not m:
            no_year += 1; continue
        year = int(m.group(1))
        rnd = YEAR_TO_ROUND.get(year)
        if rnd is None:
            continue
        ctx = (row['context'] or '').strip()
        if not ctx:
            no_ctx += 1; continue
        # Dedup contexts via first-200-char hash; same article → same doc.
        ctx_key = ctx[:200]
        if ctx_key in ctx_to_did:
            did = ctx_to_did[ctx_key]
        else:
            did = f'sqd{len(doc_pool_raw):07d}'
            ctx_to_did[ctx_key] = did
            doc_pool_raw.append({
                'doc_id': did,
                'title':  did,            # title = unique key (used by stream)
                'text':   ctx[:2000],
                'year':   year,           # publication-era anchor (gold)
                'round':  rnd,
            })
        queries.append({
            'qidx':       len(queries),
            'question':   row['question'],
            'answer':     row['answers'][0] if row['answers'] else '',
            'sf_titles':  [did],
            'ctx_titles': [did],
            'qtype':      'streamingqa_temporal',
            'year':       year,
            'round':      rnd,
            'is_tail':    rnd >= 3,
            'cluster':    rnd,
        })
    log.info(f'[streamingqa_temporal] dated queries: {len(queries)} '
             f'(skipped {no_year} no-year, {no_ctx} no-context)')

    # Pass 2: pad cold pool with non-dated articles as distractors.
    seen = set(ctx_to_did)
    rng = np.random.default_rng(SEED + 17)
    extra_idx = list(range(len(ds)))  # all 36K rows; dedup by ctx hash
    rng.shuffle(extra_idx)
    n_added = 0
    n_with_year = 0
    for i in extra_idx:
        if n_added >= n_distractors:
            break
        ctx = (ds[i]['context'] or '').strip()
        if not ctx:
            continue
        key = ctx[:200]
        if key in seen:
            continue
        # Best-effort year extraction; missing year -> None (handled by
        # year-aware strategies via a neutral fallback).
        d_year = None
        for m in yr_pat.finditer(ctx[:2000]):
            yy = int(m.group(1))
            if 2005 <= yy <= 2022:
                d_year = yy
                break
        d_rnd = YEAR_TO_ROUND.get(d_year) if d_year is not None else None
        if d_year is not None:
            n_with_year += 1
        did = f'sqx{n_added:07d}'
        doc_pool_raw.append({
            'doc_id': did, 'title': did, 'text': ctx[:2000],
            'year': d_year, 'round': d_rnd,
        })
        seen.add(key)
        n_added += 1
    log.info(f'[streamingqa_temporal] distractor docs added: {n_added} '
             f'(with year: {n_with_year})')

    title_to_idx = {d['title']: i for i, d in enumerate(doc_pool_raw)}
    head = sum(1 for q in queries if not q['is_tail'])
    tail = sum(1 for q in queries if q['is_tail'])
    log.info(f'[streamingqa_temporal] pool={len(doc_pool_raw):,}  '
             f'queries={len(queries)}  head(R1-R2)={head}  tail(R3-R5)={tail}')
    return doc_pool_raw, queries, title_to_idx


TEMPORAL_LOADERS['streamingqa_temporal'] = load_streamingqa_temporal
