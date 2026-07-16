"""Official chronological direct-evidence loader retained for external validity.

The paper's central intervention is controlled domain-mixture shift and does
not require natural timestamps.  StreamingQA remains as the one direct RAG
chronological workload; the 50-query TREC-COVID loaders were removed because
they cannot support a stable cache-necessity result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

if __package__:
    from .config import PROJECT_DIR, log
else:
    from config import PROJECT_DIR, log


def load_streamingqa_official():
    """Join official StreamingQA timestamps with the cached evidence text.

    The loader preserves strict ``question_ts`` order.  A downstream causal
    protocol takes a non-overlapping prefix for warm-up and scores each request
    before its evidence identifier becomes update feedback.
    """

    from datasets import load_dataset

    metadata_path = (
        Path(PROJECT_DIR)
        / "datasets"
        / "streamingqa_official"
        / "streaminqa_eval.jsonl.gz"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Official StreamingQA metadata is missing: "
            f"{metadata_path}. Download streaminqa_eval.jsonl.gz from "
            "https://github.com/google-deepmind/streamingqa."
        )

    metadata = []
    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                metadata.append(json.loads(line))

    rows = load_dataset("bg51717/streamingqa", split="test")
    row_by_id = {row["id"]: row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("StreamingQA text mirror contains duplicate qa_id values")

    metadata.sort(key=lambda row: (int(row["question_ts"]), row["qa_id"]))
    evidence_to_title = {}
    doc_by_title = {}
    queries = []
    for item in metadata:
        qa_id = item["qa_id"]
        if qa_id not in row_by_id:
            raise ValueError(f"official qa_id missing from text mirror: {qa_id}")
        row = row_by_id[qa_id]
        if row["question"].strip() != item["question"].strip():
            raise ValueError(f"question mismatch while joining qa_id={qa_id}")

        evidence_id = item["evidence_id"]
        title = evidence_to_title.get(evidence_id)
        if title is None:
            digest = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:20]
            title = f"sqo_{digest}"
            if title in doc_by_title:
                raise ValueError(f"unexpected evidence-id hash collision: {title}")
            context = (row.get("context") or "").strip()
            if not context:
                raise ValueError(f"empty evidence context for qa_id={qa_id}")
            evidence_to_title[evidence_id] = title
            doc_by_title[title] = {
                "doc_id": title,
                "title": title,
                "text": context[:2000],
                "evidence_ts": int(item["evidence_ts"]),
            }

        question_ts = int(item["question_ts"])
        answers = row.get("answers") or item.get("answers") or []
        queries.append({
            "qidx": len(queries),
            "question": row["question"],
            "answer": answers[0] if answers else "",
            "sf_titles": [title],
            "ctx_titles": [title],
            "qtype": "streamingqa_official",
            "question_ts": question_ts,
            "evidence_ts": int(item["evidence_ts"]),
            "round": datetime.fromtimestamp(
                question_ts, tz=timezone.utc
            ).month,
            "preserve_order": True,
            "evidence_visibility": "direct",
        })

    doc_pool = list(doc_by_title.values())
    title_to_idx = {doc["title"]: index for index, doc in enumerate(doc_pool)}
    log.info(
        "[streamingqa_official] queries=%s unique_evidence=%s repeated=%s "
        "range=%s..%s",
        f"{len(queries):,}",
        f"{len(doc_pool):,}",
        f"{len(queries) - len(doc_pool):,}",
        queries[0]["question_ts"],
        queries[-1]["question_ts"],
    )
    return doc_pool, queries, title_to_idx


TEMPORAL_LOADERS = {
    "streamingqa_official": load_streamingqa_official,
}
