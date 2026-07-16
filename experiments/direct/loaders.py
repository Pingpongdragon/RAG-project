"""Raw direct-evidence loaders for the active cross-domain experiments.

Loaders return ``(doc_pool, queries, title_to_idx)`` and never arrange requests
into a drift schedule.  Controlled one-shot, gradual, recurring, shuffled, and
stationary streams are built by ``experiments.common.factorized_workload``.

Active datasets:

* SQuAD: the formal controlled evidence-domain result used in the paper.
* FEVER: a high-reuse, gold-Wikipedia-evidence extension with offline topical
  metadata; the loader does not order those topics.

Official timestamped StreamingQA is defined in ``loaders_temporal.py``.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
import re

import numpy as np

if __package__:
    from .config import DATA_SEED as SEED, PROJECT_DIR, log
else:
    from config import DATA_SEED as SEED, PROJECT_DIR, log


SQUAD_DIR = PROJECT_DIR / "datasets" / "squad"
SQUAD_TRAIN = SQUAD_DIR / "train-v1.1.json"
SQUAD_DEV = SQUAD_DIR / "dev-v1.1.json"


def _load_squad_split(path):
    raw = json.loads(path.read_text())
    items = []
    for article in raw["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                items.append({
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answer": (qa.get("answers") or [{"text": ""}])[0].get(
                        "text", ""
                    ),
                })
    return items


def load_squad(n_source=4000):
    """Load SQuAD paragraphs with all associated unique questions.

    Each question has one gold paragraph, while each paragraph supports roughly
    four to five questions.  Paragraph sampling defines the cold source pool;
    query shuffling removes the artificial grouping inherited from the JSON
    layout.  No temporal or domain order is created here.
    """

    raw = []
    if SQUAD_TRAIN.exists():
        raw.extend(_load_squad_split(SQUAD_TRAIN))
    if SQUAD_DEV.exists():
        raw.extend(_load_squad_split(SQUAD_DEV))
    if not raw:
        raise FileNotFoundError(f"SQuAD files are missing under {SQUAD_DIR}")

    paragraph_keys = []
    paragraph_queries = {}
    for item in raw:
        key = (item["title"], item["context"])
        if key not in paragraph_queries:
            paragraph_keys.append(key)
            paragraph_queries[key] = []
        paragraph_queries[key].append(item)

    rng = np.random.default_rng(SEED + 31)
    order = np.arange(len(paragraph_keys))
    rng.shuffle(order)
    if n_source and n_source < len(order):
        order = order[: int(n_source)]

    doc_pool = []
    queries = []
    for source_position in order:
        article_title, context = paragraph_keys[int(source_position)]
        paragraph_index = len(doc_pool)
        paragraph_title = f"{article_title}#p{paragraph_index:06d}"
        doc_pool.append({
            "doc_id": f"sq{paragraph_index:06d}",
            "title": paragraph_title,
            "text": context,
        })
        for item in paragraph_queries[(article_title, context)]:
            queries.append({
                "qidx": len(queries),
                "question": item["question"],
                "answer": item["answer"],
                "sf_titles": [paragraph_title],
                "ctx_titles": [paragraph_title],
                "qtype": "squad",
                # Offline construction/audit metadata; cache policies do not read it.
                "source_article": article_title,
                "source_paragraph": paragraph_index,
                "evidence_visibility": "direct",
            })

    query_order = np.arange(len(queries))
    rng = np.random.default_rng(SEED + 32)
    rng.shuffle(query_order)
    queries = [queries[int(index)] for index in query_order]
    for query_index, query in enumerate(queries):
        query["qidx"] = query_index

    title_to_idx = {doc["title"]: index for index, doc in enumerate(doc_pool)}
    log.info(
        "[squad] pool=%s queries=%s avg_q_per_doc=%.2f",
        f"{len(doc_pool):,}",
        f"{len(queries):,}",
        len(queries) / max(1, len(doc_pool)),
    )
    return doc_pool, queries, title_to_idx


def _normalize_fever_title(title):
    title = str(title).replace("_", " ")
    replacements = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-COLON-": ":",
    }
    for source, target in replacements.items():
        title = title.replace(source, target)
    return title.lower().strip()


def _fever_domain(claim, support_titles):
    words = set(re.findall(r"\w+", f"{claim} {' '.join(support_titles)}".lower()))
    keyword_domains = {
        "film": {
            "film", "movie", "television", "series", "actor", "actress",
            "director", "cinema", "animated", "sitcom", "drama",
        },
        "sport": {
            "football", "soccer", "basketball", "tennis", "cricket",
            "baseball", "athlete", "olympics", "hockey", "rugby",
        },
        "music": {
            "singer", "band", "album", "song", "musician", "rapper",
            "guitarist", "vocalist", "discography",
        },
    }
    for domain, keywords in keyword_domains.items():
        if words & keywords:
            return domain
    return "other"


def load_fever(n_source=8000):
    """Load FEVER claims with gold Wikipedia pages and topical metadata.

    The previous loader concatenated film claims before sport/music claims and
    therefore manufactured a one-shot shift inside data loading.  This version
    samples claims without ordering by topic.  ``source_domain`` is retained
    only for offline construction and audit; the factorized workload controls
    request order and does not reveal its regime labels to online policies.
    """

    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise RuntimeError("FEVER loading requires the datasets package") from exc

    beir = hf_load("BeIR/fever", "corpus", split="corpus", trust_remote_code=True)
    normalized_to_title = {}
    normalized_to_text = {}
    for example in beir:
        if not example["text"]:
            continue
        normalized = _normalize_fever_title(example["title"])
        if (
            normalized not in normalized_to_text
            or len(example["text"]) > len(normalized_to_text[normalized])
        ):
            normalized_to_title[normalized] = example["title"]
            normalized_to_text[normalized] = example["text"][:2000]

    fever = hf_load("fever", "v1.0", split="train", trust_remote_code=True)
    claim_to_support = defaultdict(set)
    for example in fever:
        if example["label"] not in {"SUPPORTS", "REFUTES"}:
            continue
        evidence_title = example.get("evidence_wiki_url")
        if not evidence_title:
            continue
        normalized = _normalize_fever_title(evidence_title)
        if normalized in normalized_to_title:
            claim_to_support[example["claim"]].add(
                normalized_to_title[normalized]
            )

    items = [
        (claim, sorted(supports), _fever_domain(claim, supports))
        for claim, supports in claim_to_support.items()
        if supports
    ]
    rng = np.random.default_rng(SEED + 29)
    order = np.arange(len(items))
    rng.shuffle(order)
    if n_source and n_source < len(order):
        order = order[: int(n_source)]
    selected = [items[int(index)] for index in order]

    support_titles = {title for _, supports, _ in selected for title in supports}
    support_docs = [
        {
            "title": title,
            "text": normalized_to_text[_normalize_fever_title(title)],
        }
        for title in sorted(support_titles)
        if _normalize_fever_title(title) in normalized_to_text
    ]

    requested_pool = int(os.environ.get("POOL_TARGET", "0"))
    distractor_count = (
        requested_pool - len(support_docs)
        if requested_pool > len(support_docs)
        else len(support_docs) * 4
    )
    support_norms = {_normalize_fever_title(doc["title"]) for doc in support_docs}
    distractor_candidates = []
    for example in beir:
        normalized = _normalize_fever_title(example["title"])
        if normalized in support_norms or not example["text"]:
            continue
        distractor_candidates.append({
            "title": example["title"],
            "text": example["text"][:2000],
        })
        if len(distractor_candidates) >= int(distractor_count * 1.05):
            break
    distractor_order = np.arange(len(distractor_candidates))
    rng = np.random.default_rng(SEED + 30)
    rng.shuffle(distractor_order)
    distractors = [
        distractor_candidates[int(index)]
        for index in distractor_order[:distractor_count]
    ]

    documents = support_docs + distractors
    document_order = np.arange(len(documents))
    rng = np.random.default_rng(SEED + 31)
    rng.shuffle(document_order)
    documents = [documents[int(index)] for index in document_order]
    doc_pool = [
        {"doc_id": f"fv{index:06d}", "title": doc["title"], "text": doc["text"]}
        for index, doc in enumerate(documents)
    ]
    title_to_idx = {doc["title"]: index for index, doc in enumerate(doc_pool)}

    queries = []
    for claim, supports, domain in selected:
        retained = [title for title in supports if title in title_to_idx]
        if not retained:
            continue
        queries.append({
            "qidx": len(queries),
            "question": claim,
            "answer": "",
            "sf_titles": retained,
            "ctx_titles": retained,
            "qtype": "fever",
            "source_domain": domain,
            "evidence_visibility": "direct",
        })

    domain_counts = defaultdict(int)
    for query in queries:
        domain_counts[query["source_domain"]] += 1
    log.info(
        "[fever] pool=%s support_docs=%s queries=%s domains=%s",
        f"{len(doc_pool):,}",
        f"{len(support_docs):,}",
        f"{len(queries):,}",
        dict(sorted(domain_counts.items())),
    )
    return doc_pool, queries, title_to_idx


LOADERS = {
    "squad": load_squad,
    "fever": load_fever,
}
