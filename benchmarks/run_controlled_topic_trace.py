"""Controlled oracle evidence-demand trace for coarse-to-fine DRIP.

The workload uses unique queries but recurring evidence families from the
shared factorized protocol.  Cold documents are partitioned independently with
dense semantic pages; workload construction uses sparse gold-evidence TF-IDF, so
the online predictor is not handed the constructor's latent regime label.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.cache.adaptive.arc import ClassicalARC
from algorithms.cache.frequency.tinylfu import TinyLFU
from algorithms.cache.params import PARAMS
from algorithms.cache.paradigm_ref.agent_rag_cache import AgentRAGCache
from algorithms.cache.recency.fifo import FIFO
from algorithms.cache.recency.lru import LRU
from algorithms.drip import DRIP, DRIPConfig
from benchmarks.audit_semantic_pages import (
    PROTOCOLS,
    build_balanced_semantic_pages,
    prepare_dataset,
)
from experiments.common.factorized_workload import (
    FACTORIZED_WORKLOADS,
    SOURCE_POOL_ROLES,
    SOURCE_POOL_TEST,
)


POLICIES = {
    "ClassicalARC": ClassicalARC,
    "LRU": LRU,
    "FIFO": FIFO,
    "TinyLFU": TinyLFU,
    "AgentRAGCache": AgentRAGCache,
    "DRIP-Reactive": DRIP,
    "DRIP-QueryOnly": DRIP,
    "DRIP-DomainAdapt": DRIP,
    "DRIP-TopicDynamics": DRIP,
}


def _initial_cache(dataset, budget):
    counts = Counter(
        title for query in dataset.warmup for title in query.get("sf_titles", ())
    )
    ranked = sorted(counts, key=lambda title: (-counts[title], title))
    resident = {
        dataset.doc_pool[dataset.title_to_idx[title]]["doc_id"]
        for title in ranked[:budget]
        if title in dataset.title_to_idx
    }
    for document in dataset.doc_pool:
        if len(resident) >= budget:
            break
        resident.add(document["doc_id"])
    return resident


def _feedback_window(dataset, queries):
    feedback = []
    positions = []
    feedback_query_rows = []
    support_sets = []
    for query_row, query in enumerate(queries):
        supports = {
            int(dataset.title_to_idx[title])
            for title in query.get("sf_titles", ())
            if title in dataset.title_to_idx
        }
        support_sets.append(supports)
        for position in sorted(supports):
            # Expose only the post-service access key.  The factorized
            # constructor stores latent workload/regime labels on its query
            # records for offline audit, but no online policy should receive
            # them, even accidentally.
            event = {"access_title": dataset.doc_pool[position]["title"]}
            feedback.append(event)
            positions.append(position)
            feedback_query_rows.append(int(query_row))
    return (
        feedback,
        np.asarray(positions, dtype=np.int64),
        support_sets,
        np.asarray(feedback_query_rows, dtype=np.int64),
    )


def run(args):
    if args.dataset not in PROTOCOLS:
        raise ValueError(f"unknown controlled protocol: {args.dataset}")
    prep_args = argparse.Namespace(
        n_source=args.n_source,
        n_windows=args.windows,
        window_size=args.window_size,
        warmup_windows=args.warmup_windows,
        workload_seed=args.workload_seed,
        workload=args.workload,
        source_pool=args.source_pool,
        source_pool_seed=args.source_pool_seed,
        source_pool_calibration_fraction=(
            args.source_pool_calibration_fraction
        ),
    )
    dataset = prepare_dataset(args.dataset, prep_args)
    labels, _, partition_stats = build_balanced_semantic_pages(
        dataset,
        target_page_size=int(args.semantic_region_size),
        seed=int(args.partition_seed),
    )
    # This is cold-corpus metadata, independent of future stream order.  Copy
    # into the in-memory pool so the policy can use the generic metadata API.
    for position, document in enumerate(dataset.doc_pool):
        document["semantic_topic"] = int(labels[position])

    PARAMS.WRITE_CAP = int(args.write_budget)
    initial = _initial_cache(dataset, int(args.cache_size))
    policies = {}
    policy_configs = {}
    for name in args.policies or list(POLICIES):
        cls = POLICIES[name]
        if name == "DRIP-TopicDynamics":
            config = DRIPConfig.topic_dynamics(
                candidate_budget=int(args.candidate_budget),
                metadata_field="semantic_topic",
                replacement_target_rate=float(args.replacement_target),
                initial_dual_price=float(args.initial_dual_price),
                demand_decay=float(args.demand_decay),
                serve_decay=float(args.serve_decay),
                topic_document_decay=float(args.decay),
                topic_forecast_mass=float(args.forecast_mass),
                topic_min_forecast_confidence=float(
                    args.min_forecast_confidence
                ),
                topic_max_cache_fraction=float(args.proactive_fraction),
                topic_apply_forecast_to_cache=bool(
                    args.enable_provisional_placement
                ),
            )
            policy = cls(
                name,
                dataset.doc_pool,
                dataset.doc_embs,
                dataset.title_to_idx,
                config,
            )
            policy_configs[name] = asdict(config)
        elif name in {
            "DRIP-QueryOnly", "DRIP-DomainAdapt"
        }:
            config = DRIPConfig.domain_adapt(
                candidate_budget=int(args.candidate_budget),
                metadata_field="semantic_topic",
                domain_prior_rate=float(args.domain_prior_rate),
                domain_reliability_rate=float(
                    args.domain_reliability_rate
                ),
                # Main DomainAdapt never predicts the next topic.  Both
                # variants route from the current query; the full method adds
                # only concentration-gated current-window placement.
                domain_prior_weight=0.0,
                domain_route_width=int(args.domain_route_width),
                domain_retrieve_topk=int(args.domain_retrieve_topk),
                replacement_target_rate=float(args.replacement_target),
                initial_dual_price=float(args.initial_dual_price),
                demand_decay=float(args.demand_decay),
                serve_decay=float(args.serve_decay),
            )
            policy = cls(
                name,
                dataset.doc_pool,
                dataset.doc_embs,
                dataset.title_to_idx,
                config,
            )
            policy_configs[name] = asdict(config)
        elif name == "DRIP-Reactive":
            config = DRIPConfig.reactive(
                replacement_target_rate=float(args.replacement_target),
                initial_dual_price=float(args.initial_dual_price),
                demand_decay=float(args.demand_decay),
                serve_decay=float(args.serve_decay),
            )
            policy = cls(
                name,
                dataset.doc_pool,
                dataset.doc_embs,
                dataset.title_to_idx,
                config,
            )
            policy_configs[name] = asdict(config)
        else:
            policy = cls(
                name,
                dataset.doc_pool,
                dataset.doc_embs,
                dataset.title_to_idx,
            )
        policy.set_kb(initial)
        policies[name] = policy

    metrics = {
        name: {
            "object_hits": 0,
            "object_requests": 0,
            "strict_hits": 0,
            "queries": 0,
            "routed_support_hits": 0,
            "routed_support_requests": 0,
            "routed_strict_hits": 0,
            "region_support_hits": 0,
            "region_support_requests": 0,
            "region_strict_hits": 0,
            "per_window_object_hit_rate": [],
            "per_window_strict_hit_rate": [],
        }
        for name in policies
    }
    started = time.time()
    ws = int(dataset.protocol.window_size)
    nw = int(dataset.protocol.n_windows)
    for window_index in range(nw):
        window = dataset.stream[window_index * ws:(window_index + 1) * ws]
        if len(window) < ws:
            break
        feedback, positions, support_sets, feedback_query_rows = (
            _feedback_window(dataset, window)
        )
        query_rows = np.asarray(
            [int(query["qidx"]) for query in window], dtype=np.int64
        )
        current_query_embeddings = dataset.query_embs[query_rows]
        # ``step`` consumes one event per revealed evidence occurrence. Repeat
        # the corresponding *query* embedding for multi-support questions so
        # routed rows stay aligned with the flattened feedback events. Gold
        # evidence embeddings are never used for pre-service routing.
        routed_query_embeddings = current_query_embeddings[feedback_query_rows]
        for name, policy in policies.items():
            policy.prepare_window(
                feedback, routed_query_embeddings, window_index
            )
            resident_positions = {
                int(policy.d2p[doc_id]) for doc_id in policy.kb
            }
            object_hits = sum(
                position in resident_positions for position in positions
            )
            strict_hits = sum(
                bool(supports) and supports <= resident_positions
                for supports in support_sets
            )
            record = metrics[name]
            record["object_hits"] += int(object_hits)
            record["object_requests"] += len(positions)
            record["strict_hits"] += int(strict_hits)
            record["queries"] += len(window)

            routed_window = getattr(policy, "_domain_routed_window", None)
            domain_partition = getattr(policy, "domain_partition", None)
            if routed_window is not None and domain_partition is not None:
                routed_docs = [set() for _ in window]
                routed_regions = [set() for _ in window]
                for event_row, query_row in enumerate(feedback_query_rows):
                    if event_row >= len(routed_window.queries):
                        break
                    routed_query = routed_window.queries[event_row]
                    routed_docs[int(query_row)].update(
                        int(position) for position in routed_query.documents
                    )
                    routed_regions[int(query_row)].update(
                        int(region) for region in routed_query.regions
                    )
                routed_support_hits = sum(
                    len(supports & routed_docs[row])
                    for row, supports in enumerate(support_sets)
                )
                routed_strict_hits = sum(
                    bool(supports) and supports <= routed_docs[row]
                    for row, supports in enumerate(support_sets)
                )
                region_support_hits = sum(
                    sum(
                        domain_partition.primary_topic(position)
                        in routed_regions[row]
                        for position in supports
                    )
                    for row, supports in enumerate(support_sets)
                )
                region_strict_hits = sum(
                    bool(supports) and all(
                        domain_partition.primary_topic(position)
                        in routed_regions[row]
                        for position in supports
                    )
                    for row, supports in enumerate(support_sets)
                )
                support_requests = sum(len(supports) for supports in support_sets)
                record["routed_support_hits"] += int(routed_support_hits)
                record["routed_support_requests"] += int(support_requests)
                record["routed_strict_hits"] += int(routed_strict_hits)
                record["region_support_hits"] += int(region_support_hits)
                record["region_support_requests"] += int(support_requests)
                record["region_strict_hits"] += int(region_strict_hits)
            record["per_window_object_hit_rate"].append(
                round(object_hits / max(1, len(positions)), 6)
            )
            record["per_window_strict_hit_rate"].append(
                round(strict_hits / max(1, len(window)), 6)
            )
            policy.step(feedback, routed_query_embeddings, window_index)

    summary = {}
    for name, policy in policies.items():
        record = metrics[name]
        cold_fetches = int(record["object_requests"] - record["object_hits"])
        topic_proposals = int(sum(
            int(item.get("topic_proposals_materialized", 0))
            for item in getattr(policy, "topic_log", ())
        ))
        proactive_fetches = int(sum(
            int(item.get("speculative_writes", 0))
            for item in getattr(policy, "cost_log", ())
        ))
        routed_candidate_reads = int(
            getattr(policy, "serve_retrieval_cost", 0)
        )
        routed_support_requests = int(record["routed_support_requests"])
        region_support_requests = int(record["region_support_requests"])
        summary[name] = {
            "evidence_hit_rate": round(
                record["object_hits"] / max(1, record["object_requests"]), 6
            ),
            "strict_query_hit_rate": round(
                record["strict_hits"] / max(1, record["queries"]), 6
            ),
            "cache_writes": int(policy.update_cost),
            "routed_candidate_support_recall": (
                round(
                    record["routed_support_hits"]
                    / max(1, routed_support_requests),
                    6,
                )
                if routed_support_requests else None
            ),
            "routed_candidate_strict_query_recall": (
                round(record["routed_strict_hits"] / max(1, record["queries"]), 6)
                if routed_support_requests else None
            ),
            "routed_region_support_recall": (
                round(
                    record["region_support_hits"]
                    / max(1, region_support_requests),
                    6,
                )
                if region_support_requests else None
            ),
            "routed_region_strict_query_recall": (
                round(record["region_strict_hits"] / max(1, record["queries"]), 6)
                if region_support_requests else None
            ),
            # Fair trace-level accounting freezes K_t for the whole window.
            # Some sequential baselines mutate metadata during step(), so
            # their internal retrieval counters are not directly comparable.
            "cold_fetches": cold_fetches,
            "routed_candidate_reads": routed_candidate_reads,
            "proactive_document_fetches": proactive_fetches,
            "topic_metadata_proposals": topic_proposals,
            "topic_index_probes": 0,
            "proactive_candidate_reads": 0,
            # Conservative upper bound: a routed candidate that is also the
            # eventual miss evidence may be counted in both terms. Report the
            # components separately so end-to-end runners can deduplicate them.
            "total_cold_store_reads": (
                cold_fetches + routed_candidate_reads + proactive_fetches
            ),
            "implementation_reported_reads": int(policy.maint_retrieval_cost),
            "candidate_reads": int(policy.maint_retrieval_cost),
            "per_window_object_hit_rate": record[
                "per_window_object_hit_rate"
            ],
            "per_window_strict_hit_rate": record[
                "per_window_strict_hit_rate"
            ],
        }
        if hasattr(policy, "cost_log"):
            summary[name]["cost_log"] = list(policy.cost_log)
        if hasattr(policy, "topic_log"):
            summary[name]["topic_log"] = list(policy.topic_log)
        if hasattr(policy, "domain_log"):
            summary[name]["domain_log"] = list(policy.domain_log)

    return {
        "dataset": args.dataset,
        "protocol": {
            "constructor": dataset.construction,
            "source_split": dataset.source_split,
            "workload": str(dataset.protocol.workload),
            "online_constructor_labels_exposed": False,
            "post_service_gold_evidence_feedback": True,
            "causal_current_query_routing": True,
            "routing_input": (
                "original query embeddings before scoring; repeated only to "
                "align multi-support feedback events"
            ),
            "evaluation_scope": (
                "oracle evidence-demand residency trace; not end-to-end RAG retrieval"
            ),
            "access_feedback": "post-service gold support IDs",
            "cache_size": int(args.cache_size),
            "write_budget": int(args.write_budget),
            "candidate_budget": int(args.candidate_budget),
            "semantic_partition": partition_stats,
            "semantic_partition_seed": int(args.partition_seed),
            "proactive_fraction": float(args.proactive_fraction),
            "replacement_target": float(args.replacement_target),
            "initial_dual_price": float(args.initial_dual_price),
            "policy_configs": policy_configs,
            "policy_versions": {
                name: str(getattr(policy, "method_version", name))
                for name, policy in policies.items()
            },
            "data_seed": int(os.environ.get("DATA_SEED", "42")),
            "workload_seed": int(args.workload_seed),
        },
        "summary": summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(PROTOCOLS), default="squad_direct")
    parser.add_argument(
        "--workload", choices=sorted(FACTORIZED_WORKLOADS), default=None,
        help="optional controlled-workload override; default uses dataset protocol",
    )
    parser.add_argument("--n-source", type=int, default=None)
    parser.add_argument("--windows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--warmup-windows", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=24)
    parser.add_argument("--write-budget", type=int, default=3)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--domain-prior-rate", type=float, default=0.25)
    parser.add_argument("--domain-reliability-rate", type=float, default=0.25)
    parser.add_argument("--domain-prior-weight", type=float, default=0.25)
    parser.add_argument("--domain-route-width", type=int, default=2)
    parser.add_argument("--domain-retrieve-topk", type=int, default=4)
    parser.add_argument("--semantic-region-size", type=int, default=128)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--workload-seed", type=int, default=42)
    parser.add_argument(
        "--source-pool",
        choices=sorted(SOURCE_POOL_ROLES),
        default=SOURCE_POOL_TEST,
        help=(
            "stable source-family pool; formal reported runs use test while "
            "the tuner uses calibration"
        ),
    )
    parser.add_argument("--source-pool-seed", type=int, default=1729)
    parser.add_argument(
        "--source-pool-calibration-fraction", type=float, default=0.5
    )
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--demand-decay", type=float, default=0.92)
    parser.add_argument("--serve-decay", type=float, default=0.75)
    parser.add_argument("--forecast-mass", type=float, default=20.0)
    parser.add_argument(
        "--min-forecast-confidence", type=float, default=0.5
    )
    parser.add_argument("--proactive-fraction", type=float, default=0.25)
    parser.add_argument(
        "--enable-provisional-placement",
        action="store_true",
        help="apply forecast candidates to the cache using the unfinished adapter",
    )
    parser.add_argument("--replacement-target", type=float, default=0.25)
    parser.add_argument("--initial-dual-price", type=float, default=0.25)
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.source_pool_calibration_fraction < 1.0:
        parser.error("--source-pool-calibration-fraction must be in (0, 1)")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "hit": values["evidence_hit_rate"],
            "writes": values["cache_writes"],
            "reads": values["total_cold_store_reads"],
        }
        for name, values in result["summary"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
