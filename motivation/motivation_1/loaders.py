"""
Single-hop loader for Motivation 1.

We use HotpotQA "comparison" subset:
- comparison questions ask to compare two entities; both gold passages are
  TOPICALLY similar to the query (both entities appear in the question).
- This is a SINGLE-HOP retrieval task: dense retrieval can directly find the
  two SF passages from the question alone, no bridging needed.

Loader contract matches motivation_4: returns (doc_pool, queries, title_to_idx)
where each query has {qidx, question, answer, sf_titles, ctx_titles, qtype}.
"""
import json, random
from pathlib import Path

from config import SEED, PROJECT_DIR, log

random.seed(SEED)

HOTPOT_DIR = PROJECT_DIR / 'datasets' / 'hotpotqa'
TRAIN_FILE = HOTPOT_DIR / 'train_distractor.json'
DEV_FILE   = HOTPOT_DIR / 'validation_distractor.json'


def _load_split(path):
    with open(path) as f:
        return json.load(f)


def load_hotpotqa_comparison(n_source=8000):
    """Load HotpotQA comparison-only subset.

    Returns (doc_pool, queries, title_to_idx). doc_id prefix 'hc' to keep
    distinct from motivation_4 caches.
    """
    items = []
    if TRAIN_FILE.exists():
        items.extend(_load_split(TRAIN_FILE))
    if DEV_FILE.exists():
        items.extend(_load_split(DEV_FILE))
    comp = [it for it in items if it.get('type') == 'comparison']
    log.info(f"[hotpot-comp] loaded {len(comp)} comparison items "
             f"(from {len(items)} total)")
    rng = random.Random(SEED)
    rng.shuffle(comp)
    if n_source:
        comp = comp[:n_source]

    title_to_text = {}
    queries = []
    for qi, it in enumerate(comp):
        sf = it.get('supporting_facts', {})
        sf_titles = sorted(set(sf.get('title', [])))
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


LOADERS = {
    'hotpotqa_comparison': load_hotpotqa_comparison,
}


# ── 2WikiMultihopQA comparison subset ──────────────────────────────────────
WIKI2_DIR  = PROJECT_DIR / 'datasets' / '2wikimultihopqa' / 'data'
WIKI2_TRAIN = WIKI2_DIR / 'train.json'
WIKI2_DEV   = WIKI2_DIR / 'dev.json'


def load_2wiki_comparison(n_source=8000):
    """Load 2WikiMultihopQA 'comparison' subset as a single-hop testbed.

    Comparison questions in 2Wiki compare two entities explicitly mentioned
    in the question; both SFs are directly retrievable from the query text
    without any bridging hop — functionally single-hop retrieval.
    """
    # Use dev only: train.json is ~650 MB and dev alone has ~3k comparison items
    items = []
    if WIKI2_DEV.exists():
        items.extend(json.load(open(WIKI2_DEV)))
    comp = [it for it in items if it.get('type') == 'comparison']
    log.info(f"[2wiki-comp] loaded {len(comp)} comparison items "
             f"(from {len(items)} total)")
    rng = random.Random(SEED)
    rng.shuffle(comp)
    if n_source:
        comp = comp[:n_source]

    title_to_text = {}
    queries = []
    for it in comp:
        # supporting_facts: [[title, sent_id], ...]
        sf_raw = it.get('supporting_facts', [])
        sf_titles = sorted(set(t for t, _ in sf_raw))
        # context: [[title, [sent0, sent1, ...]], ...]
        ctx = it.get('context', [])
        ctx_titles = [t for t, _ in ctx]
        for t, sents in ctx:
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
        {'doc_id': f'wc{i:06d}', 'title': t, 'text': title_to_text[t]}
        for i, t in enumerate(titles)
    ]
    log.info(f"[2wiki-comp] pool={len(doc_pool):,} queries={len(queries):,}")
    return doc_pool, queries, title_to_idx


LOADERS['2wiki_comparison'] = load_2wiki_comparison
