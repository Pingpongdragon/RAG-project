#!/usr/bin/env python3
"""Query-adaptive domain retrieval and exact cache replay on MultiDoc2Dial.

The benchmark separates two questions that must not be conflated:

1. can the current query route cold retrieval to the right domain/documents?;
2. do those retrieved candidates improve persistent future residency?

Gold references are used only after candidate construction for offline metrics
and post-service placement feedback.  MultiDoc2Dial has no global timestamp,
so both supported stream protocols preserve session order and explicitly label
the controlled recurring-domain variant.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.agent.loaders import load_multidoc2dial  # noqa: E402
from experiments.agent.run_mtrag_trace import (  # noqa: E402
    POLICY_NAMES,
    calibrate_capacity,
    evaluate_trace,
)
from experiments.common.session_workload import (  # noqa: E402
    CONTROLLED_RECURRING_DOMAIN,
    SESSION_ROUND_ROBIN,
    build_recurring_domain_workload,
    build_session_round_robin,
)


BOTH = "both"


def _stable_top(scores: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    limit = min(max(0, int(limit)), len(values))
    if not limit:
        return np.empty(0, dtype=np.int64)
    return np.lexsort((np.arange(len(values)), -values))[:limit]


def _centroids(
    document_matrix: sparse.csr_matrix,
    document_domains: list[str],
) -> tuple[list[str], sparse.csr_matrix, dict[str, np.ndarray]]:
    domains = sorted(set(document_domains))
    positions = {
        domain: np.asarray([
            index for index, value in enumerate(document_domains)
            if value == domain
        ], dtype=np.int64)
        for domain in domains
    }
    rows = [
        sparse.csr_matrix(document_matrix[positions[domain]].mean(axis=0))
        for domain in domains
    ]
    return domains, normalize(sparse.vstack(rows), norm="l2"), positions


def add_query_adaptive_candidates(
    doc_pool,
    queries,
    *,
    route_width: int,
    retrieve_topk: int,
):
    """Return copied events plus query-only retrieval diagnostics."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    document_matrix = normalize(
        vectorizer.fit_transform(document["text"] for document in doc_pool),
        norm="l2",
    ).tocsr()
    query_matrix = normalize(
        vectorizer.transform(query["question"] for query in queries),
        norm="l2",
    ).tocsr()
    document_ids = [str(document["title"]) for document in doc_pool]
    document_domains = [str(document["domain"]) for document in doc_pool]
    domains, domain_centroids, positions = _centroids(
        document_matrix, document_domains
    )
    domain_scores = (query_matrix @ domain_centroids.T).toarray()
    full_scores = (query_matrix @ document_matrix.T).toarray()

    routed_events = []
    route_support_hits = 0
    route_support_occurrences = 0
    route_strict_hits = 0
    full_support_hits = 0
    full_strict_hits = 0
    routed_domain_hits = 0
    routed_scans = 0
    routes = Counter()

    for index, source in enumerate(queries):
        selected_domain_indices = _stable_top(
            domain_scores[index], route_width
        )
        selected_domains = tuple(
            domains[position] for position in selected_domain_indices
        )
        candidate_positions = np.unique(np.concatenate([
            positions[domain] for domain in selected_domains
        ]))
        routed_scans += len(candidate_positions)
        local = _stable_top(full_scores[index, candidate_positions], retrieve_topk)
        routed_documents = tuple(
            document_ids[int(candidate_positions[position])]
            for position in local
        )
        full_documents = tuple(
            document_ids[position]
            for position in _stable_top(full_scores[index], retrieve_topk)
        )

        supports = {str(value) for value in source["sf_titles"]}
        routed_set = set(routed_documents)
        full_set = set(full_documents)
        route_support_hits += len(supports & routed_set)
        full_support_hits += len(supports & full_set)
        route_support_occurrences += len(supports)
        route_strict_hits += int(supports <= routed_set)
        full_strict_hits += int(supports <= full_set)
        routed_domain_hits += int(str(source["domain"]) in selected_domains)
        routes.update(selected_domains)

        event = dict(source)
        event["ctx_titles"] = list(routed_documents)
        event["routed_domains"] = list(selected_domains)
        event["candidate_constructor"] = (
            "current-query TF-IDF domain centroid routing then in-domain rank"
        )
        routed_events.append(event)

    total = max(1, len(queries))
    occurrences = max(1, route_support_occurrences)
    diagnostics = {
        "query_information": "question and causal dialogue history only",
        "gold_used_for_routing": False,
        "domains": domains,
        "route_width": int(route_width),
        "retrieve_topk": int(retrieve_topk),
        "mean_documents_scanned_per_query": round(routed_scans / total, 3),
        "full_pool_documents_scanned_per_query": len(doc_pool),
        "routed_domain_recall": round(routed_domain_hits / total, 6),
        "routed_support_coverage": round(
            route_support_hits / occurrences, 6
        ),
        "routed_strict_query_recall": round(route_strict_hits / total, 6),
        "full_pool_support_coverage_at_same_k": round(
            full_support_hits / occurrences, 6
        ),
        "full_pool_strict_query_recall_at_same_k": round(
            full_strict_hits / total, 6
        ),
        "route_counts": dict(sorted(routes.items())),
    }
    return routed_events, diagnostics


def _protocols(args, queries):
    requested = (
        [SESSION_ROUND_ROBIN, CONTROLLED_RECURRING_DOMAIN]
        if args.protocol == BOTH else [args.protocol]
    )
    common = dict(
        seed=int(args.seed),
        warmup_size=int(args.warmup_size),
        evaluation_size=int(args.evaluation_size),
        window_size=int(args.window_size),
    )
    for protocol in requested:
        if protocol == SESSION_ROUND_ROBIN:
            yield protocol, build_session_round_robin(queries, **common)
        else:
            yield protocol, build_recurring_domain_workload(
                queries, block_size=int(args.block_size), **common
            )


def run(args):
    doc_pool, raw_queries, _ = load_multidoc2dial(split=args.split)
    queries, retrieval = add_query_adaptive_candidates(
        doc_pool,
        raw_queries,
        route_width=int(args.route_width),
        retrieve_topk=int(args.retrieve_topk),
    )
    result = {
        "dataset": "multidoc2dial",
        "split": args.split,
        "evaluation_scope": (
            "current routed retrieval plus future exact document residency"
        ),
        "global_timestamp_available": False,
        "retrieval_audit_all_source_queries": retrieval,
        "protocols": {},
    }
    for protocol, (evaluation, warmup, audit) in _protocols(args, queries):
        capacity_audit = calibrate_capacity(
            warmup,
            window_size=int(args.window_size),
            occurrence_coverage=float(args.capacity_coverage),
            window_quantile=float(args.capacity_quantile),
        )
        capacity = (
            int(capacity_audit["reference_capacity"])
            if int(args.cache_size) == 0 else int(args.cache_size)
        )
        sweeps = {}
        for write_budget in args.write_budgets:
            sweeps[str(int(write_budget))] = evaluate_trace(
                warmup,
                evaluation,
                capacity=capacity,
                window_size=int(args.window_size),
                policy_names=args.policies,
                write_budget=int(write_budget),
                candidate_budget=int(args.candidate_budget),
                score_decay=float(args.score_decay),
            )
        result["protocols"][protocol] = {
            "workload": audit.as_dict(),
            "capacity_calibration": capacity_audit,
            "cache_size": capacity,
            "write_budget_sweep": sweeps,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument(
        "--protocol",
        choices=(SESSION_ROUND_ROBIN, CONTROLLED_RECURRING_DOMAIN, BOTH),
        default=BOTH,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-size", type=int, default=500)
    parser.add_argument("--evaluation-size", type=int, default=3800)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--route-width", type=int, default=1)
    parser.add_argument("--retrieve-topk", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--capacity-coverage", type=float, default=0.90)
    parser.add_argument("--capacity-quantile", type=float, default=0.90)
    parser.add_argument("--candidate-budget", type=int, default=4)
    parser.add_argument("--write-budgets", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--score-decay", type=float, default=0.25)
    parser.add_argument(
        "--policies", nargs="+", choices=POLICY_NAMES,
        default=["LRU", "TinyLFU", "DRIP-Reactive", "DRIP-DomainAdapt"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("retrieval", result["retrieval_audit_all_source_queries"])
    for protocol, values in result["protocols"].items():
        print(protocol, "cache", values["cache_size"])
        for budget, summaries in values["write_budget_sweep"].items():
            print(" budget", budget, {
                name: (
                    summary["strict_all_support_hit_rate"],
                    summary["cache_writes"],
                )
                for name, summary in summaries.items()
            })


if __name__ == "__main__":
    main()
