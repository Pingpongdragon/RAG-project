"""Hidden-evidence loader retained for the multi-hop boundary experiment.

2WikiMultihopQA is the only active hidden benchmark.  Its inference,
compositional, and bridge-comparison questions require at least two supporting
documents and expose the second-hop limitation that direct query embeddings do
not capture.  HotpotQA and MuSiQue loaders were removed from the active code
because they did not add a distinct cacheability claim under the current paper
protocol; their raw datasets and historical results remain on disk.
"""

from __future__ import annotations

import json

import numpy as np

if __package__:
    from .config import PROJECT_DIR, DATASET_CONFIGS, DATA_SEED as SEED, log
else:
    from config import PROJECT_DIR, DATASET_CONFIGS, DATA_SEED as SEED, log


HIDDEN_TYPES = {"inference", "compositional", "bridge_comparison"}


def _random_subsample(items, n_source):
    rng = np.random.default_rng(SEED + 51)
    order = np.arange(len(items))
    rng.shuffle(order)
    selected = []
    for index in order:
        item = items[int(index)]
        supports = {title for title, _ in item.get("supporting_facts", [])}
        if len(supports) < 2:
            continue
        selected.append(item)
        if n_source and len(selected) >= int(n_source):
            break
    return selected


def load_2wiki_expanded(n_source=None, q_type=None):
    """Load official 2Wiki train/dev hidden multi-hop questions."""

    if q_type == "comparison":
        raise ValueError(
            "q_type=comparison is query-visible and is not part of the hidden suite"
        )
    if q_type and q_type not in HIDDEN_TYPES:
        raise ValueError(
            f"unsupported hidden 2Wiki q_type={q_type!r}; "
            f"expected one of {sorted(HIDDEN_TYPES)}"
        )

    base = PROJECT_DIR / "datasets" / "2wikimultihopqa" / "data"
    items = []
    for split in ("train", "dev"):
        path = base / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"2Wiki split is missing: {path}")
        items.extend(json.loads(path.read_text()))
    allowed = {q_type} if q_type else HIDDEN_TYPES
    items = [item for item in items if item.get("type") in allowed]
    if n_source and n_source < len(items):
        items = _random_subsample(items, n_source)

    title_texts = {}
    for item in items:
        for context in item.get("context", []):
            title = context[0]
            sentences = context[1] if isinstance(context[1], list) else [context[1]]
            title_texts.setdefault(title, []).append(" ".join(sentences))

    doc_pool = []
    title_to_idx = {}
    for title, texts in sorted(title_texts.items()):
        unique_texts = list(dict.fromkeys(texts))
        index = len(doc_pool)
        doc_pool.append({
            "doc_id": f"we{index:06d}",
            "title": title,
            "text": " ".join(unique_texts)[:512],
        })
        title_to_idx[title] = index

    queries = []
    for item in items:
        supports = sorted({
            title
            for title, _ in item.get("supporting_facts", [])
            if title in title_to_idx
        })
        if len(supports) < 2:
            continue
        context_titles = sorted({
            context[0]
            for context in item.get("context", [])
            if context[0] in title_to_idx
        })
        queries.append({
            "qidx": len(queries),
            "question": item["question"],
            "answer": item.get("answer", ""),
            "sf_titles": supports,
            "ctx_titles": context_titles,
            "qtype": item.get("type"),
            "evidence_visibility": "hidden",
        })

    log.info(
        "[2wiki_hidden] pool=%s queries=%s types=%s",
        f"{len(doc_pool):,}",
        f"{len(queries):,}",
        sorted(allowed),
    )
    return doc_pool, queries, title_to_idx


def load_2wiki_hidden(n_source=None):
    source_size = n_source or DATASET_CONFIGS["2wikimultihopqa"].get("n_source")
    return load_2wiki_expanded(n_source=source_size)


LOADERS = {
    "2wikimultihopqa": load_2wiki_hidden,
    "2wiki_expanded": load_2wiki_expanded,
}
