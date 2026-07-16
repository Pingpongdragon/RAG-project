"""Read-only loader for the official IBM MTRAG human retrieval benchmark.

The upstream benchmark has no global timestamp across conversations.  This
module therefore loads conversations and their per-turn evidence, but does not
invent a chronological order or set ``preserve_order``.  A workload builder
must preserve ``turn_idx`` order within each ``conversation_id`` when it later
interleaves sessions.

Upstream layout (relative to ``datasets/mt-rag-benchmark``)::

    corpora/passage_level/{domain}.jsonl.zip
    mtrag-human/retrieval_tasks/{domain}/{domain}_lastturn.jsonl
    mtrag-human/retrieval_tasks/{domain}/{domain}_questions.jsonl
    mtrag-human/retrieval_tasks/{domain}/{domain}_rewrite.jsonl
    mtrag-human/retrieval_tasks/{domain}/qrels/dev.tsv

Only the answerable and partially-answerable retrieval tasks have qrels.  The
official query id has the form ``<conversation_id><::><turn_idx>``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import zipfile


DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
QUERY_VIEWS = ("lastturn", "questions", "rewrite")
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "mt-rag-benchmark"


def evidence_key(domain: str, source_id: str) -> str:
    """Return the globally unique evidence key used by ``sf_titles``."""

    return f"{domain}::{source_id}"


def parse_query_id(query_id: str) -> tuple[str, int]:
    """Split an official MTRAG query id into conversation and turn."""

    try:
        conversation_id, raw_turn = str(query_id).rsplit("<::>", 1)
        turn_idx = int(raw_turn)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid MTRAG query id: {query_id!r}") from exc
    if not conversation_id or turn_idx < 1:
        raise ValueError(f"invalid MTRAG query id: {query_id!r}")
    return conversation_id, turn_idx


def _validated_domains(domains: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(domain).lower() for domain in domains))
    unknown = sorted(set(selected) - set(DOMAINS))
    if unknown:
        raise ValueError(f"unknown MTRAG domains: {unknown}; expected {DOMAINS}")
    if not selected:
        raise ValueError("at least one MTRAG domain is required")
    return selected


def _read_query_view(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["_id"])
            parse_query_id(query_id)
            if query_id in rows:
                raise ValueError(f"duplicate query id {query_id!r} in {path}")
            rows[query_id] = str(row["text"])
    if not rows:
        raise ValueError(f"empty MTRAG query file: {path}")
    return rows


def _read_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if not expected.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"unexpected qrels schema in {path}: {reader.fieldnames}"
            )
        for row in reader:
            if float(row["score"]) <= 0:
                continue
            query_id = str(row["query-id"])
            parse_query_id(query_id)
            corpus_id = str(row["corpus-id"])
            if corpus_id not in qrels[query_id]:
                qrels[query_id].append(corpus_id)
    if not qrels:
        raise ValueError(f"empty MTRAG qrels file: {path}")
    return dict(qrels)


def _zip_jsonl_rows(path: Path) -> Iterable[dict]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1 or not members[0].endswith(".jsonl"):
            raise ValueError(f"expected one JSONL member in {path}, found {members}")
        with archive.open(members[0]) as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    yield json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path}:{line_number}"
                    ) from exc


def _domain_paths(root: Path, domain: str) -> tuple[Path, Path]:
    corpus = root / "corpora" / "passage_level" / f"{domain}.jsonl.zip"
    retrieval = root / "mtrag-human" / "retrieval_tasks" / domain
    required = [
        corpus,
        retrieval / "qrels" / "dev.tsv",
        *(retrieval / f"{domain}_{view}.jsonl" for view in QUERY_VIEWS),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing MTRAG files: " + ", ".join(missing))
    return corpus, retrieval


def load_mtrag_human(
    root: str | Path = DEFAULT_ROOT,
    domains: Sequence[str] = DOMAINS,
    query_view: str = "rewrite",
    max_queries: int | None = None,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Load official MTRAG passage corpora and human retrieval tasks.

    Args:
        root: Root of the cloned ``IBM/mt-rag-benchmark`` repository.
        domains: Any non-empty subset of ``DOMAINS``.
        query_view: Text exposed as ``query['question']``. ``rewrite`` is the
            strongest official causal retrieval input; the other two official
            views remain available on every query for ablations.
        max_queries: Optional deterministic query cap for smoke tests.  The
            complete selected corpora are still loaded so no gold-only corpus
            filtering is introduced.

    Returns:
        ``(doc_pool, queries, title_to_idx)`` in the shared experiment format.

    Notes:
        ``queries`` is grouped by selected domain and follows upstream file
        order.  It is not a natural global event stream.  The fields
        ``conversation_id`` and ``turn_idx`` are the authoritative constraints
        for a later session-interleaving workload builder.
    """

    root = Path(root).expanduser().resolve()
    selected_domains = _validated_domains(domains)
    query_view = str(query_view).lower()
    if query_view not in QUERY_VIEWS:
        raise ValueError(
            f"unknown query_view={query_view!r}; expected {QUERY_VIEWS}"
        )
    if max_queries is not None and int(max_queries) < 1:
        raise ValueError("max_queries must be positive or None")

    doc_pool: list[dict] = []
    title_to_idx: dict[str, int] = {}
    domain_queries: dict[str, list[dict]] = {}

    for domain in selected_domains:
        corpus_path, retrieval_dir = _domain_paths(root, domain)
        source_ids: set[str] = set()
        for row in _zip_jsonl_rows(corpus_path):
            source_id = str(row.get("_id") or row.get("id") or "")
            if not source_id:
                raise ValueError(f"corpus row without _id/id in {corpus_path}")
            if source_id in source_ids:
                raise ValueError(
                    f"duplicate corpus id {source_id!r} in {corpus_path}"
                )
            source_ids.add(source_id)
            key = evidence_key(domain, source_id)
            if key in title_to_idx:
                raise ValueError(f"duplicate global evidence key: {key}")
            raw_title = str(row.get("title") or "").strip()
            raw_text = str(row.get("text") or "").strip()
            # ``title`` must be the unique evidence key because the shared
            # evaluator maps retrieved documents back through title_to_idx.
            # Keep the readable upstream title in both metadata and text.
            searchable_text = (
                f"{raw_title}\n{raw_text}" if raw_title and not raw_text.startswith(raw_title)
                else raw_text
            )
            title_to_idx[key] = len(doc_pool)
            doc_pool.append({
                "doc_id": f"mtrag-{domain}-{source_id}",
                "title": key,
                "display_title": raw_title,
                "text": searchable_text,
                "domain": domain,
                "source_doc_id": source_id,
                "url": str(row.get("url") or ""),
            })
        views = {
            view: _read_query_view(
                retrieval_dir / f"{domain}_{view}.jsonl"
            )
            for view in QUERY_VIEWS
        }
        view_id_sets = {view: set(rows) for view, rows in views.items()}
        if len({frozenset(ids) for ids in view_id_sets.values()}) != 1:
            sizes = {view: len(ids) for view, ids in view_id_sets.items()}
            raise ValueError(
                f"query views disagree for domain={domain}: {sizes}"
            )
        qrels = _read_qrels(retrieval_dir / "qrels" / "dev.tsv")
        query_ids = list(views[query_view])
        if set(query_ids) != set(qrels):
            missing_qrels = sorted(set(query_ids) - set(qrels))[:5]
            missing_queries = sorted(set(qrels) - set(query_ids))[:5]
            raise ValueError(
                f"query/qrels mismatch for {domain}: "
                f"without_qrels={missing_qrels}, without_query={missing_queries}"
            )

        rows = []
        for source_order, query_id in enumerate(query_ids):
            conversation_id, turn_idx = parse_query_id(query_id)
            missing_supports = [
                source_id for source_id in qrels[query_id]
                if source_id not in source_ids
            ]
            if missing_supports:
                raise ValueError(
                    f"qrels reference missing passage ids for {domain}/{query_id}: "
                    f"{missing_supports[:5]}"
                )
            supports = [
                evidence_key(domain, source_id)
                for source_id in qrels[query_id]
            ]
            rows.append({
                "question": views[query_view][query_id],
                "answer": "",
                "sf_titles": supports,
                "ctx_titles": [],
                "qtype": "mtrag_retrieval",
                "query_id": query_id,
                "conversation_id": conversation_id,
                "agent_id": conversation_id,
                "turn_idx": turn_idx,
                "domain": domain,
                "collection": domain,
                "source_order": source_order,
                "lastturn": views["lastturn"][query_id],
                "questions": views["questions"][query_id],
                "rewrite": views["rewrite"][query_id],
                "query_view": query_view,
                "evidence_visibility": "hidden",
            })
        domain_queries[domain] = rows

    queries = [
        query
        for domain in selected_domains
        for query in domain_queries[domain]
    ]
    if max_queries is not None:
        queries = queries[: int(max_queries)]
    for qidx, query in enumerate(queries):
        query["qidx"] = qidx

    return doc_pool, queries, title_to_idx


def schema_audit(
    root: str | Path = DEFAULT_ROOT,
    domains: Sequence[str] = DOMAINS,
) -> dict:
    """Run an exact query/qrels/corpus-id audit and return compact counts."""

    doc_pool, queries, _ = load_mtrag_human(
        root=root,
        domains=domains,
        query_view="rewrite",
    )
    conversations = {query["conversation_id"] for query in queries}
    support_occurrences = sum(len(query["sf_titles"]) for query in queries)
    unique_supports = {
        support
        for query in queries
        for support in query["sf_titles"]
    }
    seen_by_conversation: dict[str, set[str]] = defaultdict(set)
    repeated_within_conversation = 0
    queries_answerable_from_past = 0
    for query in sorted(
        queries,
        key=lambda row: (
            row["conversation_id"], row["turn_idx"], row["query_id"]
        ),
    ):
        seen = seen_by_conversation[query["conversation_id"]]
        supports = set(query["sf_titles"])
        repeated_within_conversation += len(supports & seen)
        queries_answerable_from_past += int(bool(supports) and supports <= seen)
        seen.update(supports)
    domain_counts = {
        domain: {
            "passages": sum(doc["domain"] == domain for doc in doc_pool),
            "queries": sum(query["domain"] == domain for query in queries),
            "conversations": len({
                query["conversation_id"]
                for query in queries if query["domain"] == domain
            }),
        }
        for domain in _validated_domains(domains)
    }
    return {
        "passages": len(doc_pool),
        "queries": len(queries),
        "conversations_with_retrieval_tasks": len(conversations),
        "support_occurrences": support_occurrences,
        "unique_supports": len(unique_supports),
        "repeated_support_rate": round(
            (support_occurrences - len(unique_supports))
            / max(1, support_occurrences),
            6,
        ),
        "within_conversation_repeated_support_rate": round(
            repeated_within_conversation / max(1, support_occurrences), 6
        ),
        "past_answerable_query_rate": round(
            queries_answerable_from_past / max(1, len(queries)), 6
        ),
        "domains": domain_counts,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--max-queries", type=int, default=8)
    args = parser.parse_args()
    if args.audit:
        result = schema_audit(args.root, args.domains)
    else:
        docs, queries, title_to_idx = load_mtrag_human(
            root=args.root,
            domains=args.domains,
            max_queries=args.max_queries,
        )
        assert queries
        assert all(
            support in title_to_idx
            for query in queries
            for support in query["sf_titles"]
        )
        result = {
            "passages": len(docs),
            "queries_loaded": len(queries),
            "first_query": queries[0]["query_id"],
            "first_supports": queries[0]["sf_titles"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
