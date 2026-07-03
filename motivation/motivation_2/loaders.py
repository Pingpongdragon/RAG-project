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
import re
import numpy as np

from config import PROJECT_DIR, DATASET_CONFIGS, SEED, log


def _random_subsample(items, sf_titles_fn, n_source, seed_offset=0):
    """Random subsample requiring >=2 distinct supporting-fact titles per item."""
    rng = np.random.default_rng(SEED + seed_offset)
    order = np.arange(len(items))
    rng.shuffle(order)
    out = []
    for idx in order:
        it = items[int(idx)]
        if len(set(sf_titles_fn(it))) < 2:
            continue
        out.append(it)
        if n_source and len(out) >= n_source:
            break
    log.info(f'[random] selected {len(out)} / {len(items)} items')
    return out


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
                       'sf_titles': sfs, 'ctx_titles': ctx,
                       'qtype': item.get('type')})
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
                       'sf_titles': sfs, 'ctx_titles': ctx,
                       'qtype': item.get('type')})
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
                       'sf_titles': sfs, 'ctx_titles': ctx,
                       'qtype': item.get('type')})
    log.info(f"[musique] pool={len(doc_pool)}, queries={len(queries)}")
    return doc_pool, queries, title_to_idx


# ═══════════════════════════════════════════════════
#  Expanded loaders (official full releases)
# ═══════════════════════════════════════════════════

def load_hotpotqa_expanded(n_source=None, q_type=None):
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
    if q_type:
        all_items = [it for it in all_items if it.get('type') == q_type]
        log.info(f'  filtered to type={q_type}: {len(all_items)} items')
    if n_source and n_source < len(all_items):
        all_items = _random_subsample(
            all_items,
            lambda item: item['supporting_facts']['title'],
            n_source,
            seed_offset=31,
        )
        log.info(f'  randomly subsampled to {len(all_items)} items')
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
                       'sf_titles': sfs, 'ctx_titles': ctx,
                       'qtype': item.get('type')})
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
    if n_source and n_source < len(all_items):
        all_items = _random_subsample(
            all_items,
            lambda item: [p['title'] for p in item['paragraphs'] if p.get('is_supporting')],
            n_source,
            seed_offset=41,
        )
        log.info(f'  randomly subsampled to {len(all_items)} items')
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
                       'sf_titles': sfs, 'ctx_titles': ctx,
                       'qtype': item.get('type')})
    log.info(f'[musique_expanded] pool={len(doc_pool)}, queries={len(queries)}')
    return doc_pool, queries, title_to_idx


def load_2wiki_expanded(n_source=None, q_type=None):
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
    if q_type:
        all_items = [it for it in all_items if it.get("type") == q_type]
        log.info(f"  filtered to type={q_type}: {len(all_items)} items")
    if n_source and n_source < len(all_items):
        all_items = _random_subsample(
            all_items,
            lambda item: [sf[0] for sf in item.get('supporting_facts', [])],
            n_source,
            seed_offset=51,
        )
        log.info(f'  randomly subsampled to {len(all_items)} items')
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
                       'sf_titles': sfs, 'ctx_titles': ctx,
                       'qtype': item.get('type')})
    log.info(f'[2wiki_expanded] pool={len(doc_pool)}, queries={len(queries)}')
    return doc_pool, queries, title_to_idx


def _mind2web_shard_paths(n_source=None):
    """Return local Mind2Web train shard paths, downloading public shards if needed."""
    max_shards = 11
    need_shards = 1 if not n_source else max(1, min(max_shards, int(np.ceil(n_source / 100))))
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "Mind2Web loading requires huggingface_hub in the active env"
        ) from exc
    paths = []
    for sid in range(need_shards):
        filename = f"data/train/train_{sid}.json"
        try:
            paths.append(hf_hub_download(
                "osunlp/Mind2Web", filename, repo_type="dataset",
                local_files_only=True))
        except Exception:
            paths.append(hf_hub_download(
                "osunlp/Mind2Web", filename, repo_type="dataset"))
    return paths


def _compact_text(text, limit=120):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def _candidate_label(candidate, action_repr=None):
    attrs = {}
    raw_attrs = candidate.get("attributes")
    if isinstance(raw_attrs, str):
        try:
            attrs = json.loads(raw_attrs)
        except Exception:
            attrs = {}
    elif isinstance(raw_attrs, dict):
        attrs = raw_attrs
    tag = str(candidate.get("tag") or attrs.get("tag") or "element").lower()
    if action_repr:
        # Mind2Web action repr is like:
        #   [combobox] Reservation type -> SELECT: Pickup
        left = str(action_repr).split("->", 1)[0]
        left = re.sub(r"^\[[^\]]+\]\s*", "", left).strip()
        label = left
    else:
        label = (
            attrs.get("aria_label")
            or attrs.get("alt")
            or attrs.get("placeholder")
            or attrs.get("name")
            or attrs.get("id")
            or attrs.get("class")
            or candidate.get("backend_node_id")
            or ""
        )
    label = _compact_text(label, 80)
    label = label or str(candidate.get("backend_node_id") or "unknown")
    return tag, label


def _mind2web_doc_title(website, tag, label):
    key = re.sub(r"[^a-z0-9]+", " ", f"{tag} {label}".lower()).strip()
    key = re.sub(r"\s+", " ", key)[:96] or "element"
    return f"{website} :: {key}"


def load_mind2web_agent(n_source=None):
    """Load Mind2Web as a real web-agent cache benchmark.

    Mind2Web contains human action trajectories over real websites.  We map it
    to the shared cache interface as follows:

      - document: a canonical website element/control
      - query: task + website + previous action history before the next action
      - support: previous action target A plus current action target B

    The current target B is stored in ``target_title`` and is treated as hidden
    by the ``agent_multistep_reuse`` workload.  Reuse comes from different
    tasks needing the same canonical website control.
    """
    log.info("Loading Mind2Web agent trajectories from osunlp/Mind2Web")
    items = []
    for path in _mind2web_shard_paths(n_source=n_source):
        with open(path) as f:
            shard = json.load(f)
        items.extend(shard)
        if n_source and len(items) >= n_source:
            break
    if n_source:
        items = items[:n_source]

    title_to_text = {}
    queries = []
    max_neg_per_action = 24
    for task_i, item in enumerate(items):
        website = str(item.get("website") or "website")
        domain = str(item.get("domain") or "")
        subdomain = str(item.get("subdomain") or "")
        task = str(item.get("confirmed_task") or "")
        action_reprs = list(item.get("action_reprs") or [])
        actions = list(item.get("actions") or [])
        prev_titles = []
        for step_i, action in enumerate(actions):
            pos = list(action.get("pos_candidates") or [])
            if not pos:
                continue
            action_repr = action_reprs[step_i] if step_i < len(action_reprs) else ""
            tag, label = _candidate_label(pos[0], action_repr=action_repr)
            target_title = _mind2web_doc_title(website, tag, label)
            op = action.get("operation") or {}
            op_text = f"{op.get('op') or op.get('original_op') or ''} {op.get('value') or ''}".strip()
            title_to_text.setdefault(
                target_title,
                _compact_text(
                    f"Website {website}. Domain {domain} {subdomain}. "
                    f"Target element [{tag}] {label}. Operation {op_text}. "
                    f"Example action: {action_repr}",
                    512,
                ),
            )

            ctx_titles = [target_title]
            for cand in list(action.get("neg_candidates") or [])[:max_neg_per_action]:
                ntag, nlabel = _candidate_label(cand)
                nt = _mind2web_doc_title(website, ntag, nlabel)
                if nt not in title_to_text:
                    title_to_text[nt] = _compact_text(
                        f"Website {website}. Domain {domain} {subdomain}. "
                        f"Candidate element [{ntag}] {nlabel}.", 512)
                ctx_titles.append(nt)

            anchor_titles = prev_titles[-1:]
            sf_titles = list(dict.fromkeys(anchor_titles + [target_title]))
            history = " ; ".join(action_reprs[max(0, step_i - 3):step_i])
            question = _compact_text(
                f"Agent task on {website} ({domain}/{subdomain}): {task}. "
                f"Previous actions: {history if history else 'none'}. "
                f"Choose the next web element to operate.",
                420,
            )
            queries.append({
                "question": question,
                "answer": action_repr,
                "sf_titles": sf_titles,
                "ctx_titles": list(dict.fromkeys(ctx_titles + anchor_titles)),
                "qtype": "agent_multistep" if step_i > 0 else "agent_start",
                "route_hint": "bridge" if step_i > 0 else "direct",
                "target_title": target_title,
                "agent_task_id": item.get("annotation_id") or f"task-{task_i}",
                "agent_id": int(task_i % 8),
                "website": website,
                "domain": domain,
                "step_idx": int(step_i),
            })
            prev_titles.append(target_title)

    doc_pool, title_to_idx = [], {}
    for i, (title, text) in enumerate(sorted(title_to_text.items())):
        doc_pool.append({"doc_id": f"mw{i:06d}", "title": title, "text": text})
        title_to_idx[title] = i

    # Keep only queries whose supports survived the canonical element pool.
    filtered = []
    for q in queries:
        q["sf_titles"] = [t for t in q["sf_titles"] if t in title_to_idx]
        q["ctx_titles"] = [t for t in q["ctx_titles"] if t in title_to_idx]
        if q.get("target_title") in title_to_idx and q["sf_titles"]:
            filtered.append(q)
    log.info(
        f"[mind2web_agent] tasks={len(items)} pool={len(doc_pool)} "
        f"queries={len(filtered)}"
    )
    return doc_pool, filtered, title_to_idx


# ── Loader registry ───────────────────────────────
LOADERS = {
    'hotpotqa':          load_hotpotqa,
    '2wikimultihopqa':   load_2wikimultihopqa,
    'musique':           load_musique,
    'hotpotqa_expanded': load_hotpotqa_expanded,
    'musique_expanded':  load_musique_expanded,
    '2wiki_expanded':    load_2wiki_expanded,
    'mind2web_agent':    load_mind2web_agent,
}
