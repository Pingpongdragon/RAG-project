"""
Single-hop loaders for Motivation 1.

Construction follows the query-semantic shift methodology of Lupart et al.
(MS-Shift, ECIR 2023): random subsample of queries, then KMeans on raw query
embeddings to define head/tail topics for the drift schedule.

Each loader returns (doc_pool, queries, title_to_idx) where each query has
{qidx, question, answer, sf_titles, ctx_titles, qtype}.
"""
import json
from pathlib import Path

import numpy as np

from config import SEED, PROJECT_DIR, log


def _random_subsample(items, n_source, seed_offset=0):
    if not n_source or n_source >= len(items):
        return list(items)
    rng = np.random.default_rng(SEED + seed_offset)
    order = np.arange(len(items))
    rng.shuffle(order)
    selected = [items[int(i)] for i in order[:n_source]]
    log.info(f'[random] selected {len(selected)} / {len(items)} items')
    return selected


# ── SQuAD v1.1 ─────────────────────────────────────────────────────────────
SQUAD_DIR   = PROJECT_DIR / 'datasets' / 'squad'
SQUAD_TRAIN = SQUAD_DIR / 'train-v1.1.json'
SQUAD_DEV   = SQUAD_DIR / 'dev-v1.1.json'


def _load_squad_split(path):
    raw = json.load(open(path))
    items = []
    for art in raw['data']:
        title = art['title']
        for p in art['paragraphs']:
            ctx = p['context']
            for qa in p['qas']:
                items.append({
                    'title': title,
                    'context': ctx,
                    'question': qa['question'],
                    'answer': (qa.get('answers') or [{'text': ''}])[0].get('text', ''),
                })
    return items


def load_squad(n_source=4000):
    """SQuAD v1.1 single-hop reading comprehension.

    Each question references exactly ONE Wikipedia paragraph as gold evidence.
    Construction: sample n_source paragraphs (==doc_pool), then take ALL
    questions of those paragraphs as the query stream. Each paragraph is
    referenced by ~4-5 questions on average -> natural support reuse so
    cluster-level prediction has signal to exploit.
    """
    raw = []
    if SQUAD_TRAIN.exists():
        raw.extend(_load_squad_split(SQUAD_TRAIN))
    if SQUAD_DEV.exists():
        raw.extend(_load_squad_split(SQUAD_DEV))
    log.info(f"[squad] loaded {len(raw)} (question, paragraph) items")

    # group by unique paragraph
    para_keys = []
    para_to_qs = {}
    for it in raw:
        key = (it['title'], it['context'])
        if key not in para_to_qs:
            para_keys.append(key)
            para_to_qs[key] = []
        para_to_qs[key].append(it)
    log.info(f"[squad] {len(para_keys)} unique paragraphs")

    # sample paragraphs
    rng = np.random.default_rng(SEED + 31)
    order = np.arange(len(para_keys))
    rng.shuffle(order)
    if n_source and n_source < len(para_keys):
        order = order[:n_source]
    selected_keys = [para_keys[int(i)] for i in order]

    doc_pool = []
    queries = []
    for key in selected_keys:
        title, ctx = key
        pid = f'sq{len(doc_pool):06d}'
        para_title = f"{title}#p{len(doc_pool):06d}"
        doc_pool.append({'doc_id': pid, 'title': para_title, 'text': ctx})
        for it in para_to_qs[key]:
            queries.append({
                'qidx': len(queries),
                'question': it['question'],
                'answer': it['answer'],
                'sf_titles': [para_title],
                'ctx_titles': [para_title],
                'qtype': 'squad',
            })
    # shuffle queries so paragraph-grouping doesn't artificially cluster them
    qrng = np.random.default_rng(SEED + 32)
    qorder = np.arange(len(queries))
    qrng.shuffle(qorder)
    queries = [queries[int(i)] for i in qorder]
    for new_qi, q in enumerate(queries):
        q['qidx'] = new_qi
    title_to_idx = {d['title']: i for i, d in enumerate(doc_pool)}
    log.info(f"[squad] pool={len(doc_pool):,} queries={len(queries):,} "
             f"(avg {len(queries)/max(1,len(doc_pool)):.2f} q/doc)")
    return doc_pool, queries, title_to_idx


LOADERS = {
    'squad': load_squad,
}


# ── HotpotQA comparison subset ─────────────────────────────────────────────
HOTPOT_DIR  = PROJECT_DIR / 'datasets' / 'hotpotqa'
HOTPOT_TRAIN = HOTPOT_DIR / 'train_distractor.json'
HOTPOT_DEV   = HOTPOT_DIR / 'validation_distractor.json'


def load_hotpotqa_comparison(n_source=4000):
    """HotpotQA comparison subset.

    Each query asks to compare two entities, both directly named in the
    question; the two SF passages are dense-similar to the query — this is
    functionally a single-hop retrieval task. Average ~1.35 queries share an
    SF title (mild but non-trivial cross-query reuse), making it a useful
    second testbed for cluster-driven persistent maintenance.
    """
    items = []
    if HOTPOT_TRAIN.exists():
        items.extend(json.load(open(HOTPOT_TRAIN)))
    if HOTPOT_DEV.exists():
        items.extend(json.load(open(HOTPOT_DEV)))
    comp = [it for it in items if it.get('type') == 'comparison']
    log.info(f"[hotpot-comp] loaded {len(comp)} comparison items")
    comp = _random_subsample(comp, n_source, seed_offset=11)

    title_to_text = {}
    queries = []
    for it in comp:
        sf_titles = sorted(set(it.get('supporting_facts', {}).get('title', [])))
        ctx = it.get('context', {})
        ctx_titles = list(ctx.get('title', []))
        for t, sents in zip(ctx_titles, ctx.get('sentences', [])):
            if t not in title_to_text:
                title_to_text[t] = ' '.join(sents)
        if len(sf_titles) < 2:
            continue
        queries.append({
            'qidx':       len(queries),
            'question':   it['question'],
            'answer':     it.get('answer', ''),
            'sf_titles':  sf_titles,
            'ctx_titles': ctx_titles,
            'qtype':      'comparison',
        })
    titles = sorted(title_to_text.keys())
    title_to_idx = {t: i for i, t in enumerate(titles)}
    doc_pool = [
        {'doc_id': f'hc{i:06d}', 'title': t, 'text': title_to_text[t]}
        for i, t in enumerate(titles)
    ]
    log.info(f"[hotpot-comp] pool={len(doc_pool):,} queries={len(queries):,}")
    return doc_pool, queries, title_to_idx


LOADERS['hotpotqa_comparison'] = load_hotpotqa_comparison


# ── 2WikiMultihopQA comparison subset ──────────────────────────────────────
WIKI_DIR = PROJECT_DIR / 'datasets' / '2wikimultihopqa' / 'data'
WIKI_TRAIN = WIKI_DIR / 'train.json'
WIKI_DEV   = WIKI_DIR / 'dev.json'


def load_2wiki_comparison(n_source=4000):
    """2WikiMultihopQA comparison subset.

    Comparison questions name both target entities directly. SF format is
    a list of [title, sent_idx] tuples (different from HotpotQA's dict).
    """
    items = []
    if WIKI_TRAIN.exists():
        items.extend(json.load(open(WIKI_TRAIN)))
    if WIKI_DEV.exists():
        items.extend(json.load(open(WIKI_DEV)))
    comp = [it for it in items if it.get('type') == 'comparison']
    log.info(f"[2wiki-comp] loaded {len(comp)} comparison items")
    comp = _random_subsample(comp, n_source, seed_offset=13)

    title_to_text = {}
    queries = []
    for it in comp:
        sf_titles = sorted({t for t, _ in it.get('supporting_facts', [])})
        ctx = it.get('context', [])
        ctx_titles = []
        for entry in ctx:
            t = entry[0]
            sents = entry[1] if len(entry) > 1 else []
            ctx_titles.append(t)
            if t not in title_to_text:
                title_to_text[t] = ' '.join(sents) if isinstance(sents, list) else str(sents)
        if len(sf_titles) < 2:
            continue
        queries.append({
            'qidx':       len(queries),
            'question':   it['question'],
            'answer':     it.get('answer', ''),
            'sf_titles':  sf_titles,
            'ctx_titles': ctx_titles,
            'qtype':      'comparison',
        })
    titles = sorted(title_to_text.keys())
    title_to_idx = {t: i for i, t in enumerate(titles)}
    doc_pool = [
        {'doc_id': f'wc{i:06d}', 'title': t, 'text': title_to_text[t]}
        for i, t in enumerate(titles)
    ]
    log.info(f"[2wiki-comp] pool={len(doc_pool):,} queries={len(queries):,}")
    return doc_pool, queries, title_to_idx


LOADERS['2wiki_comparison'] = load_2wiki_comparison
