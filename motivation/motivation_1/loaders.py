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


# ── 2WikiMultihopQA simple-combined (comparison + inference + compositional) ──
def load_2wiki_simple(n_source=4000):
    """2wiki simple-combined: comparison + inference + compositional types.

    Aggregates three 2-doc subtypes from 2wiki to lift cross-query document
    reuse (q/SF-title rises from 1.02 in dev-comparison-only to 1.67 in
    train-3-types). NOTE: inference/compositional involve mild bridging,
    so this is only a 'near-direct-evidence' setting — a pragmatic
    second testbed rather than a pure single-hop one.
    """
    items = []
    if WIKI_TRAIN.exists():
        items.extend(json.load(open(WIKI_TRAIN)))
    if WIKI_DEV.exists():
        items.extend(json.load(open(WIKI_DEV)))
    keep_types = {'comparison', 'inference', 'compositional'}
    simple = [it for it in items if it.get('type') in keep_types]
    log.info(f"[2wiki-simple] loaded {len(simple)} items (comp+inf+comp)")
    simple = _random_subsample(simple, n_source, seed_offset=17)

    title_to_text = {}
    queries = []
    for it in simple:
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
            'qtype':      it.get('type', 'simple'),
        })
    titles = sorted(title_to_text.keys())
    title_to_idx = {t: i for i, t in enumerate(titles)}
    doc_pool = [
        {'doc_id': f'ws{i:06d}', 'title': t, 'text': title_to_text[t]}
        for i, t in enumerate(titles)
    ]
    log.info(f"[2wiki-simple] pool={len(doc_pool):,} queries={len(queries):,}")
    return doc_pool, queries, title_to_idx


LOADERS['2wiki_simple'] = load_2wiki_simple


# ── 2WikiMultihopQA bridge_comparison subset (high-reuse near-direct setting) ──
def load_2wiki_bridge_comparison(n_source=4000):
    """2wiki bridge_comparison: comparison questions over 4 SF docs.

    Question directly names two entities (e.g. two films) and asks to compare
    a derived attribute (e.g. directors' nationality). Retrieval requires the
    two named entities plus their bridge attributes; query reuse is high
    because directors/composers/etc. recur across comparisons (q/SF=2.90,
    55.6% titles reused). Stronger structural fit than 2wiki_comparison.
    """
    items = []
    if WIKI_TRAIN.exists():
        items.extend(json.load(open(WIKI_TRAIN)))
    if WIKI_DEV.exists():
        items.extend(json.load(open(WIKI_DEV)))
    bc = [it for it in items if it.get('type') == 'bridge_comparison']
    log.info(f"[2wiki-bc] loaded {len(bc)} bridge_comparison items")
    bc = _random_subsample(bc, n_source, seed_offset=19)

    title_to_text = {}
    queries = []
    for it in bc:
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
            'qtype':      'bridge_comparison',
        })
    titles = sorted(title_to_text.keys())
    title_to_idx = {t: i for i, t in enumerate(titles)}
    doc_pool = [
        {'doc_id': f'wb{i:06d}', 'title': t, 'text': title_to_text[t]}
        for i, t in enumerate(titles)
    ]
    log.info(f"[2wiki-bc] pool={len(doc_pool):,} queries={len(queries):,}")
    return doc_pool, queries, title_to_idx


LOADERS['2wiki_bridge_comparison'] = load_2wiki_bridge_comparison


# ── TriviaQA Wikipedia (rc.wikipedia) — high-reuse single-hop setting ──
def load_triviaqa_wikipedia(n_source=6000):
    """TriviaQA rc.wikipedia: single-hop questions over Wikipedia entity pages.

    Each question directly names 1-2 entities; answer is in entity's Wikipedia
    article. Popular articles (US Presidents, Shakespeare, France, ...) appear
    in hundreds of questions → natural high q/SF reuse without bridge steps.
    Satisfies both conditions for QDC: q/SF≥1.3 AND query-emb↔SF-doc-emb aligned.

    n=6000 → pool~8k, q/SF≈1.35 (same range as HotpotQA-comparison).
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise RuntimeError("pip install datasets")

    ds = hf_load('trivia_qa', 'rc.wikipedia', split='train', trust_remote_code=True)
    items = [ex for ex in ds if ex['entity_pages']['title']]
    log.info(f"[triviaqa-wiki] loaded {len(items)} items with entity_pages")
    items = _random_subsample(items, n_source, seed_offset=23)

    title_to_text = {}
    queries = []
    for it in items:
        ep = it['entity_pages']
        titles = ep['title']
        contexts = ep['wiki_context']
        sf_titles = sorted(set(titles))
        for t, ctx in zip(titles, contexts):
            if t not in title_to_text and ctx:
                title_to_text[t] = ctx[:2000]   # cap per-doc text
        if not sf_titles:
            continue
        queries.append({
            'qidx':      len(queries),
            'question':  it['question'],
            'answer':    it['answer'].get('value', ''),
            'sf_titles': sf_titles,
            'ctx_titles': sf_titles,   # for TriviaQA, entity pages = context
            'qtype':     'triviaqa_wikipedia',
        })

    titles = sorted(title_to_text.keys())
    title_to_idx = {t: i for i, t in enumerate(titles)}
    doc_pool = [
        {'doc_id': f'tq{i:06d}', 'title': t, 'text': title_to_text[t]}
        for i, t in enumerate(titles)
    ]
    log.info(f"[triviaqa-wiki] pool={len(doc_pool):,} queries={len(queries):,}")
    return doc_pool, queries, title_to_idx


LOADERS['triviaqa_wikipedia'] = load_triviaqa_wikipedia


# ── FEVER (fact verification, high entity reuse) ────────────────────────────
def load_fever(n_source=8000):
    """FEVER fact-verification claims over Wikipedia.

    Each claim directly names the subject Wikipedia entity.
    q/SF ≈ 8 (Cond-A), Cond-B ≈ 98.6%.
    Pool: SF Wikipedia pages + 4× random distractor Wikipedia pages.
    Topic drift: film/TV (H1) → sport+music (H2).
    """
    import re
    from collections import defaultdict

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise RuntimeError("pip install datasets")

    def normalize_title(t):
        t = t.replace('_', ' ')
        t = re.sub(r'-LRB-', '(', t); t = re.sub(r'-RRB-', ')', t)
        t = re.sub(r'-LSB-', '[', t); t = re.sub(r'-RSB-', ']', t)
        t = re.sub(r'-COLON-', ':', t)
        return t.lower().strip()

    log.info("[fever] loading BEIR/fever corpus...")
    beir = hf_load('BeIR/fever', 'corpus', split='corpus', trust_remote_code=True)
    norm2raw  = {}
    norm2text = {}
    for ex in beir:
        if not ex['text']:
            continue
        nk = normalize_title(ex['title'])
        if nk not in norm2text or len(ex['text']) > len(norm2text[nk]):
            norm2text[nk] = ex['text'][:2000]
            norm2raw[nk]  = ex['title']
    log.info(f"[fever] BEIR corpus: {len(norm2text):,} titles")

    log.info("[fever] loading FEVER claims...")
    fever_ds = hf_load('fever', 'v1.0', split='train', trust_remote_code=True)
    with_ev = [ex for ex in fever_ds
               if ex['label'] in ('SUPPORTS', 'REFUTES') and ex['evidence_wiki_url']]

    claim_to_sf = defaultdict(set)
    for ex in with_ev:
        nk = normalize_title(ex['evidence_wiki_url'])
        if nk in norm2raw:
            claim_to_sf[ex['claim']].add(norm2raw[nk])
    log.info(f"[fever] {len(claim_to_sf):,} claims matched")

    FILM_KW  = {'film', 'movie', 'television', 'series', 'actor', 'actress',
                'director', 'cinema', 'animated', 'sitcom', 'drama'}
    SPORT_KW = {'football', 'soccer', 'basketball', 'tennis', 'cricket',
                'baseball', 'athlete', 'olympics', 'hockey', 'rugby'}
    MUSIC_KW = {'singer', 'band', 'album', 'song', 'musician', 'rapper',
                'guitarist', 'vocalist', 'discography'}

    def claim_topic(claim, sf_set):
        text = (claim + ' ' + ' '.join(sf_set)).lower()
        words = set(re.findall(r'\w+', text))
        if words & FILM_KW:  return 'film'
        if words & SPORT_KW: return 'sport'
        if words & MUSIC_KW: return 'music'
        return 'other'

    by_topic = defaultdict(list)
    for claim, sfs in claim_to_sf.items():
        by_topic[claim_topic(claim, sfs)].append((claim, sfs))

    log.info(f"[fever] topics: " + ", ".join(f"{t}={len(v)}" for t, v in sorted(by_topic.items())))

    h1_items = by_topic['film']
    h2_items = by_topic['sport'] + by_topic['music']
    rng = np.random.default_rng(SEED + 29)
    n_half = n_source // 2
    rng.shuffle(h1_items); rng.shuffle(h2_items)
    h1_items = h1_items[:n_half];  h2_items = h2_items[:n_half]
    log.info(f"[fever] H1(film)={len(h1_items)}  H2(sport+music)={len(h2_items)}")

    all_items = h1_items + h2_items
    sf_titles_used = set()
    queries = []
    for i, (claim, sfs) in enumerate(all_items):
        sf_titles_used.update(sfs)
        queries.append({'qidx': i, 'question': claim, 'answer': '',
                        'sf_titles': sorted(sfs), 'ctx_titles': sorted(sfs),
                        'qtype': 'fever'})

    sf_docs = []
    for t in sorted(sf_titles_used):
        nk = normalize_title(t)
        if nk in norm2text:
            sf_docs.append({'title': t, 'text': norm2text[nk]})

    n_distractor = len(sf_docs) * 4
    sf_norms = {normalize_title(d['title']) for d in sf_docs}
    distractor_pool = []
    for ex in beir:
        nk = normalize_title(ex['title'])
        if nk not in sf_norms and ex['text']:
            distractor_pool.append({'title': ex['title'], 'text': ex['text'][:2000]})
            if len(distractor_pool) >= n_distractor * 3:
                break
    rng2 = np.random.default_rng(SEED + 30)
    d_order = np.arange(len(distractor_pool)); rng2.shuffle(d_order)
    distractor_docs = [distractor_pool[int(i)] for i in d_order[:n_distractor]]

    all_docs = sf_docs + distractor_docs
    rng3 = np.random.default_rng(SEED + 31)
    doc_order = np.arange(len(all_docs)); rng3.shuffle(doc_order)
    all_docs = [all_docs[int(i)] for i in doc_order]

    title_to_idx = {d['title']: i for i, d in enumerate(all_docs)}
    doc_pool = [{'doc_id': f'fv{i:06d}', 'title': d['title'], 'text': d['text']}
                for i, d in enumerate(all_docs)]

    pool_titles = set(title_to_idx.keys())
    for q in queries:
        q['sf_titles']  = [t for t in q['sf_titles']  if t in pool_titles]
        q['ctx_titles'] = [t for t in q['ctx_titles'] if t in pool_titles]
    queries = [q for q in queries if q['sf_titles']]
    for qi, q in enumerate(queries): q['qidx'] = qi

    log.info(f"[fever] pool={len(doc_pool):,} (SF={len(sf_docs)} dist={len(distractor_docs)})  "
             f"queries={len(queries):,}  q/SF={len(queries)/max(1,len(sf_docs)):.2f}  "
             f"dist:SF={len(distractor_docs)/max(1,len(sf_docs)):.1f}:1")
    return doc_pool, queries, title_to_idx


LOADERS['fever'] = load_fever
