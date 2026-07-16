#!/usr/bin/env python3
"""Audit MultiDoc2Dial as a non-stationary evidence-cache workload.

The audit deliberately does not invent a global timestamp.  It measures exact
document/span reuse, document switches inside each official dialogue, and
locality in the released file order.  The latter is labelled as serialization
order and must not be presented as a natural temporal trace.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


DEFAULT_ROOT = Path("datasets/multidoc2dial/multidoc2dial")


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _percent(value: float) -> float:
    return round(100.0 * float(value), 3)


def _repeat_rate(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return (total - len(counter)) / total if total else 0.0


def _top_mass(counter: Counter[str], fraction: float) -> float:
    if not counter:
        return 0.0
    width = max(1, round(len(counter) * fraction))
    return sum(value for _, value in counter.most_common(width)) / sum(
        counter.values()
    )


def _adjacent_window_jaccard(stream: list[set[str]], window_size: int) -> float:
    windows = [
        set().union(*stream[start:start + window_size])
        for start in range(0, len(stream), window_size)
        if stream[start:start + window_size]
    ]
    scores = []
    for left, right in zip(windows, windows[1:]):
        union = left | right
        scores.append(len(left & right) / len(union) if union else 1.0)
    return mean(scores) if scores else 0.0


def audit_split(path: Path, *, window_size: int) -> dict:
    raw = _load(path)["dial_data"]
    doc_occurrences: Counter[str] = Counter()
    span_occurrences: Counter[str] = Counter()
    dialogue_doc_occurrences: Counter[str] = Counter()
    per_domain_docs: dict[str, Counter[str]] = {}
    per_domain_spans: dict[str, Counter[str]] = {}
    serialized_stream: list[set[str]] = []
    dialogue_count = 0
    agent_turns = 0
    multi_document_dialogues = 0
    within_dialogue_transitions = 0
    within_dialogue_switches = 0
    prior_turn_reuse = 0

    for domain, dialogues in raw.items():
        domain_docs = per_domain_docs.setdefault(domain, Counter())
        domain_spans = per_domain_spans.setdefault(domain, Counter())
        for dialogue in dialogues:
            dialogue_count += 1
            dialogue_documents: set[str] = set()
            seen_in_dialogue: set[str] = set()
            previous_documents: set[str] | None = None
            dialogue_agent_turns = []

            for turn in dialogue["turns"]:
                if turn.get("role") != "agent":
                    continue
                documents = {
                    str(reference["doc_id"])
                    for reference in turn.get("references", [])
                }
                spans = {
                    f'{reference["doc_id"]}::{reference["id_sp"]}'
                    for reference in turn.get("references", [])
                }
                if not documents:
                    continue

                agent_turns += 1
                dialogue_agent_turns.append(documents)
                serialized_stream.append(documents)
                doc_occurrences.update(documents)
                span_occurrences.update(spans)
                domain_docs.update(documents)
                domain_spans.update(spans)
                dialogue_documents.update(documents)
                if documents & seen_in_dialogue:
                    prior_turn_reuse += 1
                seen_in_dialogue.update(documents)

                if previous_documents is not None:
                    within_dialogue_transitions += 1
                    if documents != previous_documents:
                        within_dialogue_switches += 1
                previous_documents = documents

            dialogue_doc_occurrences.update(dialogue_documents)
            if len(dialogue_documents) > 1:
                multi_document_dialogues += 1

    def domain_summary(domain: str) -> dict:
        docs = per_domain_docs[domain]
        spans = per_domain_spans[domain]
        return {
            "document_occurrences": sum(docs.values()),
            "unique_documents": len(docs),
            "document_repeat_occurrence_rate_pct": _percent(
                _repeat_rate(docs)
            ),
            "span_occurrences": sum(spans.values()),
            "unique_spans": len(spans),
            "span_repeat_occurrence_rate_pct": _percent(
                _repeat_rate(spans)
            ),
        }

    return {
        "file": str(path),
        "ordering": "released JSON serialization; not a natural timestamp",
        "dialogues": dialogue_count,
        "agent_turns_with_evidence": agent_turns,
        "document_occurrences": sum(doc_occurrences.values()),
        "unique_documents": len(doc_occurrences),
        "document_repeat_occurrence_rate_pct": _percent(
            _repeat_rate(doc_occurrences)
        ),
        "document_top_10pct_mass_pct": _percent(
            _top_mass(doc_occurrences, 0.10)
        ),
        "span_occurrences": sum(span_occurrences.values()),
        "unique_spans": len(span_occurrences),
        "span_repeat_occurrence_rate_pct": _percent(
            _repeat_rate(span_occurrences)
        ),
        "documents_reused_across_dialogues": sum(
            1 for count in dialogue_doc_occurrences.values() if count > 1
        ),
        "multi_document_dialogue_rate_pct": _percent(
            multi_document_dialogues / dialogue_count if dialogue_count else 0
        ),
        "within_dialogue_document_switch_rate_pct": _percent(
            within_dialogue_switches / within_dialogue_transitions
            if within_dialogue_transitions else 0
        ),
        "agent_turn_prior_document_reuse_rate_pct": _percent(
            prior_turn_reuse / agent_turns if agent_turns else 0
        ),
        "serialization_adjacent_window_doc_jaccard_pct": _percent(
            _adjacent_window_jaccard(serialized_stream, window_size)
        ),
        "window_size": int(window_size),
        "per_domain": {
            domain: domain_summary(domain) for domain in sorted(per_domain_docs)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.window_size <= 0:
        raise ValueError("window-size must be positive")

    splits = {}
    for split in ("train", "validation", "test"):
        path = args.root / f"multidoc2dial_dial_{split}.json"
        splits[split] = audit_split(path, window_size=args.window_size)
    result = {
        "dataset": "MultiDoc2Dial",
        "cache_objects": {
            "document": "doc_id",
            "evidence_span": "doc_id::id_sp",
        },
        "causal_boundary": (
            "agent-turn references are post-service evidence feedback; "
            "they cannot route the same request"
        ),
        "splits": splits,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
