"""
Dataset loaders for multi-hop QA benchmarks.

Each loader returns (doc_pool, queries, title_to_idx):
  - doc_pool:     list[dict] with keys {doc_id, title, text}
  - queries:      list[dict] with keys {question, answer, sf_titles, ctx_titles}
  - title_to_idx: dict mapping document title -> position in doc_pool

Three "small" loaders read from HippoRAG's reproduce/dataset/ directory
(~1000 items each, used for quick 20-window experiments).

Two "expanded" loaders read from official full releases (train+dev splits)
for large-scale 50-window experiments.

Data sources:
  - HotpotQA:          Yang et al., EMNLP 2018, validation_distractor split
  - 2WikiMultihopQA:   Ho et al., COLING 2020, train+dev from official release
  - MuSiQue-Ans:       Trivedi et al., TACL 2022, train+dev from official release

Filtering: queries with < 2 supporting-fact titles in the pool are dropped,
  since multi-hop QA requires at least 2 evidence documents.
"""
import json
import numpy as np
from config import PROJECT_DIR, DATASET_CONFIGS, SEED, log


# ═══════════════════════════════════════════════════
#  Small loaders (HippoRAG reproduce subset)
# ═══════════════════════════════════════════════════

def load_hotpotqa():
    """Load HotpotQA validation-distractor (Yang et al., EMNLP 2018).

    Randomly samples n_source items (default 2000) from 7,405 validation
    questions.  Each item contains 10 context paragraphs, 2 of which are
    supporting facts.  Corpus is built from all context paragraphs across
    sampled items (title-deduplicated).
    """
    path = PROJECT_DIR / 'datasets' / 'hotpotqa' / 'validation_distractor.json'
    log.info(f"Loading {path}")
    with open(path) as f:
        raw = json.load(f)
    cfg = DATASET_CONFIGS['hotpotqa']
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(raw), cfg['n_source'], replace=False)
    samples = [raw[i] for i in idx]
    title_to_text = {}
    for item in samples:
        for title, sents in zip(item['context']['title'],
                                item['context']['sentences']):
            if title not in title_to_text:
                title_to_text[title] = ' '.join(sents).strip()
    doc_pool, title_to_idx = [], {}
    for i, (t, txt) in enumerate(sorted(title_to_text.items())):
        doc_pool.append({'doc_id': f'h{i:05d}', 'title': t, 'text': txt})
        title_to_idx[t] = i
    queries = []
    for item in samples:
        sfs = list({t for t in item['supporting_facts']['title']
                     if t in title_to_idx})
        if len(sfs) < 2:
            continue
        ctx = [t for t in item['context']['title'] if t in title_to_idx]
        queries.append({'question': item['question'], 'answer': item['answer'],
                       'sf_titles': sfs, 'ctx_titles': ctx})
    log.info(f"[hotpotqa] pool={len(doc_pool)}, queries={len(queries)}")
    return doc_pool, queries, title_to_idx


def load_2wikimultihopqa():
    """Load 2WikiMultihopQA from HippoRAG reproduce subset (~1000 items).

    Ho et al., COLING 2020.  The HippoRAG reproduction package includes a
    pre-processed dev subset with corpus and query JSON files.
    """
    base = PROJECT_DIR / 'HippoRAG' / 'reproduce' / 'dataset'
    log.info("Loading 2wikimultihopqa")
    with open(base / '2wikimultihopqa_corpus.json') as f:
        corpus = json.load(f)
    with open(base / '2wikimultihopqa.json') as f:
        raw = json.load(f)
    title_texts = {}
    for e in corpus:
        title_texts.setdefault(e['title'], []).append(e['text'])
    doc_pool, title_to_idx = [], {}
    for i, (t, txts) in enumerate(sorted(title_texts.items())):
        doc_pool.append({'doc_id': f'w{i:05d}', 'title': t,
                        'text': ' '.join(txts)[:512]})
        title_to_idx[t] = i
    queries = []
    for item in raw:
        sfs = list({sf[0] for sf in item['supporting_facts']
                     if sf[0] in title_to_idx})
        if len(sfs) < 2:
            continue
        ctx = list({c[0] for c in item['context'] if c[0] in title_to_idx})
        queries.append({'question': item['question'], 'answer': item['answer'],
                       'sf_titles': sfs, 'ctx_titles': ctx})
    log.info(f"[2wikimultihopqa] pool={len(doc_pool)}, queries={len(queries)}")
    return doc_pool, queries, title_to_idx


def load_musique():
    """Load MuSiQue from HippoRAG reproduce subset (~1000 items).

    Trivedi et al., TACL 2022.  The HippoRAG package includes a dev subset.
    """
    base = PROJECT_DIR / 'HippoRAG' / 'reproduce' / 'dataset'
    log.info("Loading musique")
    with open(base / 'musique_corpus.json') as f:
        corpus = json.load(f)
    with open(base / 'musique.json') as f:
        raw = json.load(f)
    title_texts = {}
    for e in corpus:
        title_texts.setdefault(e['title'], []).append(e['text'])
    doc_pool, title_to_idx = [], {}
    for i, (t, txts) in enumerate(sorted(title_texts.items())):
        doc_pool.append({'doc_id': f'm{i:05d}', 'title': t,
                        'text': ' '.join(txts)[:512]})
        title_to_idx[t] = i
    queries = []
    for item in raw:
        sfs = list({p['title'] for p in item['paragraphs']
                     if p.get('is_supporting') and p['title'] in title_to_idx})
        if len(sfs) < 2:
            continue
        ctx = list({p['title'] for p in item['paragraphs']
                     if p['title'] in title_to_idx})
        queries.append({'question': item['question'],
                       'answer': item.get('answer', ''),
                       'sf_titles': sfs, 'ctx_titles': ctx})
    log.info(f"[musique] pool={len(doc_pool)}, queries={len(queries)}")
    return doc_pool, queries, title_to_idx


# ═══════════════════════════════════════════════════
#  Expanded loaders (official full releases)
# ═══════════════════════════════════════════════════

def load_hotpotqa_expanded(n_source=None):
    """Load HotpotQA full train+dev distractor splits (Yang et al., EMNLP 2018).

    90,447 train + 7,405 dev = ~97,852 items. Each item has 10 paragraphs,
    2 supporting facts. Corpus is built from all unique titles across SAMPLED items
    to keep embedding tractable.

    Args:
        n_source: sample this many ITEMS first (so pool stays bounded);
                  resulting query count after filtering will be slightly less.
    """
    base = PROJECT_DIR / 'datasets' / 'hotpotqa'
    log.info('Loading hotpotqa_expanded (train + dev distractor)')
    all_items = []
    for split in ['train_distractor', 'validation_distractor']:
        with open(base / f'{split}.json') as f:
            all_items.extend(json.load(f))
    log.info(f'  loaded {len(all_items)} items')
    if n_source and n_source < len(all_items):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(all_items), n_source, replace=False)
        all_items = [all_items[i] for i in idx]
        log.info(f'  subsampled to {len(all_items)} items')
    title_to_text = {}
    for item in all_items:
        for title, sents in zip(item['context']['title'],
                                item['context']['sentences']):
            if title not in title_to_text:
                title_to_text[title] = ' '.join(sents).strip()[:512]
    doc_pool, title_to_idx = [], {}
    for i, (t, txt) in enumerate(sorted(title_to_text.items())):
        doc_pool.append({'doc_id': f'he{i:06d}', 'title': t, 'text': txt})
        title_to_idx[t] = i
    queries = []
    for item in all_items:
        sfs = list({t for t in item['supporting_facts']['title']
                     if t in title_to_idx})
        if len(sfs) < 2:
            continue
        ctx = [t for t in item['context']['title'] if t in title_to_idx]
        queries.append({'question': item['question'],
                       'answer': item.get('answer', ''),
                       'sf_titles': sfs, 'ctx_titles': ctx})
    log.info(f'[hotpotqa_expanded] pool={len(doc_pool)}, queries={len(queries)}')
    return doc_pool, queries, title_to_idx




def load_musique_expanded(n_source=None):
    """Load MuSiQue-Ans full train+dev (Trivedi et al., TACL 2022).

    19,938 train + 2,417 dev = 22,355 answerable items.
    Each item has ~20 paragraphs with is_supporting flags.
    Corpus: all unique (title, paragraph_text) pairs -> ~84K docs.

    Args:
        n_source: randomly sample this many queries.  None = use all.
    """
    base = PROJECT_DIR / 'datasets' / 'musique' / 'data'
    log.info('Loading musique_expanded (train + dev)')
    all_items = []
    for split in ['train', 'dev']:
        path = base / f'musique_ans_v1.0_{split}.jsonl'
        with open(path) as f:
            for line in f:
                all_items.append(json.loads(line))
    log.info(f'  loaded {len(all_items)} items from train+dev')
    title_texts = {}
    for item in all_items:
        for p in item['paragraphs']:
            t = p['title']
            title_texts.setdefault(t, []).append(p['paragraph_text'])
    doc_pool, title_to_idx = [], {}
    for i, (t, txts) in enumerate(sorted(title_texts.items())):
        seen = set()
        unique = [x for x in txts if not (x in seen or seen.add(x))]
        doc_pool.append({'doc_id': f'me{i:06d}', 'title': t,
                        'text': ' '.join(unique)[:512]})
        title_to_idx[t] = i
    queries = []
    for item in all_items:
        sfs = list({p['title'] for p in item['paragraphs']
                     if p.get('is_supporting') and p['title'] in title_to_idx})
        if len(sfs) < 2:
            continue
        ctx = list({p['title'] for p in item['paragraphs']
                     if p['title'] in title_to_idx})
        queries.append({'question': item['question'],
                       'answer': item.get('answer', ''),
                       'sf_titles': sfs, 'ctx_titles': ctx})
    if n_source and n_source < len(queries):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(queries), n_source, replace=False)
        queries = [queries[i] for i in idx]
    log.info(f'[musique_expanded] pool={len(doc_pool)}, queries={len(queries)}')
    return doc_pool, queries, title_to_idx


def load_2wiki_expanded(n_source=None):
    """Load 2WikiMultihopQA full train+dev (Ho et al., COLING 2020).

    167,454 train + 12,576 dev = 180,030 items.
    Corpus: all unique (title, paragraph) pairs -> ~385K docs.

    Args:
        n_source: randomly sample this many queries.  None = use all.
    """
    base = PROJECT_DIR / 'datasets' / '2wikimultihopqa' / 'data'
    log.info('Loading 2wiki_expanded (train + dev)')
    all_items = []
    for split in ['train', 'dev']:
        path = base / f'{split}.json'
        with open(path) as f:
            raw = json.load(f)
        all_items.extend(raw)
    log.info(f'  loaded {len(all_items)} items from train+dev')
    title_texts = {}
    for item in all_items:
        for c in item.get('context', []):
            t = c[0]
            sents = c[1] if isinstance(c[1], list) else [c[1]]
            title_texts.setdefault(t, []).append(' '.join(sents))
    doc_pool, title_to_idx = [], {}
    for i, (t, txts) in enumerate(sorted(title_texts.items())):
        seen = set()
        unique = [x for x in txts if not (x in seen or seen.add(x))]
        doc_pool.append({'doc_id': f'we{i:06d}', 'title': t,
                        'text': ' '.join(unique)[:512]})
        title_to_idx[t] = i
    queries = []
    for item in all_items:
        sfs = list({sf[0] for sf in item.get('supporting_facts', [])
                     if sf[0] in title_to_idx})
        if len(sfs) < 2:
            continue
        ctx = list({c[0] for c in item.get('context', [])
                     if c[0] in title_to_idx})
        queries.append({'question': item['question'],
                       'answer': item.get('answer', ''),
                       'sf_titles': sfs, 'ctx_titles': ctx})
    if n_source and n_source < len(queries):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(queries), n_source, replace=False)
        queries = [queries[i] for i in idx]
    log.info(f'[2wiki_expanded] pool={len(doc_pool)}, queries={len(queries)}')
    return doc_pool, queries, title_to_idx


# ── Loader registry ───────────────────────────────
LOADERS = {
    'hotpotqa':          load_hotpotqa,
    '2wikimultihopqa':   load_2wikimultihopqa,
    'musique':           load_musique,
    'hotpotqa_expanded': load_hotpotqa_expanded,
    'musique_expanded':  load_musique_expanded,
    '2wiki_expanded':    load_2wiki_expanded,
}
