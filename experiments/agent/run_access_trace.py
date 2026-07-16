"""Causal evidence-residency evaluation for exact access traces.

This runner deliberately separates two questions that the dense-retrieval
runner used to mix together:

1. was the evidence object resident when the request arrived? and
2. can a history-only query embedding retrieve that object from the cache?

MIND click logs expose the clicked news ID after service.  The ID is therefore
valid feedback for preparing the *next* window, but never an input to the cache
state used for the current request.  MT-RAG 的多文档 qrel replay 由同目录下的
``run_mtrag_trace.py`` 单独实现。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cache.adaptive.arc import ClassicalARC
from algorithms.cache.frequency.tinylfu import TinyLFU
from algorithms.cache.params import PARAMS
from algorithms.cache.paradigm_ref.agent_rag_cache import AgentRAGCache
from algorithms.cache.recency.fifo import FIFO
from algorithms.cache.recency.lru import LRU
from algorithms.drip import DRIP, DRIPConfig
from experiments.common.stream_protocol import chronological_sample
from experiments.agent.loaders import load_mind_news_context
from experiments.agent.utils import compute_document_embeddings


POLICIES = {
    "ClassicalARC": ClassicalARC,
    "AgentRAGCache": AgentRAGCache,
    "LRU": LRU,
    "FIFO": FIFO,
    "TinyLFU": TinyLFU,
    "DRIP-Reactive": DRIP,
    "DRIP-DomainAdapt": DRIP,
    "DRIP-WindowTopic": DRIP,
    "DRIP-TopicDynamics": DRIP,
}


def _window_distribution(topic_ids, n_topics):
    counts = np.bincount(
        np.asarray(topic_ids, dtype=np.int64), minlength=int(n_topics)
    ).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def _js_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mixture = 0.5 * (left + right)

    def kl(values, reference):
        mask = values > 0.0
        return float(np.sum(
            values[mask]
            * np.log2(values[mask] / np.clip(reference[mask], 1e-12, None))
        ))

    return 0.5 * (kl(left, mixture) + kl(right, mixture))


def topic_and_proposal_audit(
    warmup_positions,
    evaluation_positions,
    position_to_topic,
    window_size,
    candidate_budget,
    decay=0.92,
):
    """Audit a strictly causal persistence forecast and concrete-doc proposal.

    Topic labels are used only by this diagnostic and the cold routing layer;
    no future label or access ID is read.  At the end of window ``t`` the
    observed topic distribution becomes the forecast for ``t+1``.  Within that
    forecast, documents are ranked by decayed historical access probability
    ``P_t(d | topic)``.  The whole topic is never loaded.
    """

    position_to_topic = np.asarray(position_to_topic, dtype=np.int64)
    n_documents = int(len(position_to_topic))
    n_topics = int(position_to_topic.max(initial=-1) + 1)
    counts = np.zeros(n_documents, dtype=np.float64)
    static_counts = np.zeros(n_topics, dtype=np.float64)

    warmup_windows = [
        np.asarray(warmup_positions[start:start + window_size], dtype=np.int64)
        for start in range(0, len(warmup_positions), window_size)
        if len(warmup_positions[start:start + window_size])
    ]
    previous_distribution = np.full(n_topics, 1.0 / max(1, n_topics))
    for positions in warmup_windows:
        counts *= float(decay)
        np.add.at(counts, positions, 1.0)
        previous_distribution = _window_distribution(
            position_to_topic[positions], n_topics
        )
        static_counts += np.bincount(
            position_to_topic[positions], minlength=n_topics
        )
    static_distribution = static_counts / max(float(static_counts.sum()), 1.0)

    top1_correct = 0
    static_top1_correct = 0
    top3_mass = 0.0
    static_top3_mass = 0.0
    nll = 0.0
    static_nll = 0.0
    js = 0.0
    static_js = 0.0
    proposal_hits = 0
    global_hits = 0
    proposal_total = 0
    proposal_useful = 0
    global_useful = 0
    proposals_issued = 0
    windows = 0

    for start in range(0, len(evaluation_positions), window_size):
        positions = np.asarray(
            evaluation_positions[start:start + window_size], dtype=np.int64
        )
        if len(positions) < window_size:
            break
        actual = _window_distribution(position_to_topic[positions], n_topics)
        predicted = previous_distribution

        top1_correct += int(int(np.argmax(predicted)) == int(np.argmax(actual)))
        static_top1_correct += int(
            int(np.argmax(static_distribution)) == int(np.argmax(actual))
        )
        k = min(3, n_topics)
        top3 = np.argpartition(predicted, -k)[-k:]
        static_top3 = np.argpartition(static_distribution, -k)[-k:]
        top3_mass += float(actual[top3].sum())
        static_top3_mass += float(actual[static_top3].sum())
        nll += -float(np.sum(actual * np.log(np.clip(predicted, 1e-9, 1.0))))
        static_nll += -float(np.sum(
            actual * np.log(np.clip(static_distribution, 1e-9, 1.0))
        ))
        js += _js_divergence(actual, predicted)
        static_js += _js_divergence(actual, static_distribution)

        # Concrete evidence proposal: P(topic next) * P_t(doc | topic).
        topic_mass = np.bincount(
            position_to_topic,
            weights=counts,
            minlength=n_topics,
        )
        conditional = counts / np.clip(
            topic_mass[position_to_topic], 1e-12, None
        )
        topic_scores = predicted[position_to_topic] * conditional
        global_scores = counts / max(float(counts.sum()), 1e-12)
        budget = min(int(candidate_budget), n_documents)
        if budget > 0:
            topic_candidates = np.argpartition(topic_scores, -budget)[-budget:]
            global_candidates = np.argpartition(global_scores, -budget)[-budget:]
            topic_set = set(int(value) for value in topic_candidates)
            global_set = set(int(value) for value in global_candidates)
            proposal_hits += sum(int(value) in topic_set for value in positions)
            global_hits += sum(int(value) in global_set for value in positions)
            proposal_useful += len(topic_set & set(int(value) for value in positions))
            global_useful += len(global_set & set(int(value) for value in positions))
            proposals_issued += budget
        proposal_total += len(positions)

        counts *= float(decay)
        np.add.at(counts, positions, 1.0)
        previous_distribution = actual
        windows += 1

    denominator = max(1, windows)
    return {
        "protocol": "previous-window topic persistence; past-access doc reuse",
        "topics": n_topics,
        "windows": windows,
        "candidate_budget": int(candidate_budget),
        "forecast_top1_accuracy": round(top1_correct / denominator, 6),
        "static_top1_accuracy": round(static_top1_correct / denominator, 6),
        "forecast_top3_mass": round(top3_mass / denominator, 6),
        "static_top3_mass": round(static_top3_mass / denominator, 6),
        "forecast_nll": round(nll / denominator, 6),
        "static_nll": round(static_nll / denominator, 6),
        "forecast_js": round(js / denominator, 6),
        "static_js": round(static_js / denominator, 6),
        "topic_doc_occurrence_recall": round(
            proposal_hits / max(1, proposal_total), 6
        ),
        "global_doc_occurrence_recall": round(
            global_hits / max(1, proposal_total), 6
        ),
        "topic_prefetch_precision": round(
            proposal_useful / max(1, proposals_issued), 6
        ),
        "global_prefetch_precision": round(
            global_useful / max(1, proposals_issued), 6
        ),
    }


def _causal_initial_cache(doc_pool, title_to_idx, warmup, budget, seed):
    frequencies = Counter(
        query["access_title"]
        for query in warmup
        if query.get("access_title") in title_to_idx
    )
    ranked = sorted(frequencies, key=lambda title: (-frequencies[title], title))
    selected = [doc_pool[title_to_idx[title]]["doc_id"] for title in ranked[:budget]]
    if len(selected) < budget:
        selected_set = set(selected)
        remaining = [
            document["doc_id"]
            for document in doc_pool
            if document["doc_id"] not in selected_set
        ]
        rng = np.random.default_rng(int(seed))
        fill = rng.choice(
            remaining, size=min(budget - len(selected), len(remaining)), replace=False
        )
        selected.extend(str(value) for value in fill)
    return set(selected)


def run_mind(args):
    doc_pool, queries, title_to_idx = load_mind_news_context()
    selected, temporal = chronological_sample(
        queries,
        warmup_size=int(args.warmup_windows) * int(args.window_size),
        evaluation_size=int(args.windows) * int(args.window_size),
        mode=args.temporal_sampling,
        block_size=int(args.window_size),
    )
    # The loader returns the complete 236K-event trace.  Selected records are
    # defensive copies, so release the source list before allocating policy
    # state and the 51K x 1024 document matrix.
    del queries
    warmup_size = int(args.warmup_windows) * int(args.window_size)
    warmup = selected[:warmup_size]
    stream = selected[warmup_size:]
    doc_embs = compute_document_embeddings(doc_pool)
    positions = np.asarray(
        [title_to_idx[query["access_title"]] for query in stream],
        dtype=np.int64,
    )
    warmup_positions = np.asarray(
        [title_to_idx[query["access_title"]] for query in warmup],
        dtype=np.int64,
    )

    topic_audits = {}
    for field in ("category", "subcategory"):
        labels = sorted({str(document.get(field, "unknown")) for document in doc_pool})
        label_to_id = {label: index for index, label in enumerate(labels)}
        doc_topics = np.asarray([
            label_to_id[str(document.get(field, "unknown"))]
            for document in doc_pool
        ], dtype=np.int64)
        topic_audits[field] = topic_and_proposal_audit(
            warmup_positions,
            positions,
            doc_topics,
            window_size=int(args.window_size),
            candidate_budget=int(args.candidate_budget),
            decay=float(args.decay),
        )

    PARAMS.WRITE_CAP = int(args.write_budget)
    init_kb = _causal_initial_cache(
        doc_pool, title_to_idx, warmup, int(args.cache_size), int(args.seed)
    )
    selected_policies = args.policies or list(POLICIES)
    policies = {}
    policy_configs = {}
    for name in selected_policies:
        policy_class = POLICIES[name]
        if name == "DRIP-TopicDynamics":
            config = DRIPConfig.topic_dynamics(
                candidate_budget=int(args.candidate_budget),
                metadata_field=str(args.topic_field),
                replacement_target_rate=float(args.replacement_target),
                initial_dual_price=float(args.initial_dual_price),
                topic_document_decay=float(args.decay),
                topic_forecast_mass=float(args.topic_forecast_mass),
                topic_min_forecast_confidence=float(
                    args.min_forecast_confidence
                ),
                topic_apply_forecast_to_cache=bool(
                    args.enable_provisional_placement
                ),
            )
            policy = policy_class(
                name,
                doc_pool,
                doc_embs,
                title_to_idx,
                config,
            )
            policy_configs[name] = asdict(config)
        elif name in {"DRIP-DomainAdapt", "DRIP-WindowTopic"}:
            config = DRIPConfig.domain_adapt(
                candidate_budget=int(args.candidate_budget),
                metadata_field=str(args.topic_field),
                domain_prior_weight=0.0,
                domain_route_width=int(args.domain_route_width),
                domain_retrieve_topk=int(args.domain_retrieve_topk),
                domain_placement_weight=(
                    float(args.domain_placement_weight)
                    if name == "DRIP-WindowTopic" else 0.0
                ),
                demand_decay=float(args.demand_decay),
                serve_decay=float(args.serve_decay),
                replacement_target_rate=float(args.replacement_target),
                initial_dual_price=float(args.initial_dual_price),
            )
            policy = policy_class(
                name,
                doc_pool,
                doc_embs,
                title_to_idx,
                config,
            )
            policy_configs[name] = asdict(config)
        elif name == "DRIP-Reactive":
            config = DRIPConfig.reactive(
                demand_decay=float(args.demand_decay),
                serve_decay=float(args.serve_decay),
                replacement_target_rate=float(args.replacement_target),
                initial_dual_price=float(args.initial_dual_price),
            )
            policy = policy_class(
                name,
                doc_pool,
                doc_embs,
                title_to_idx,
                config,
            )
            policy_configs[name] = asdict(config)
        else:
            policy = policy_class(name, doc_pool, doc_embs, title_to_idx)
        policy.set_kb(init_kb)
        policies[name] = policy

    metrics = {
        name: {
            "hits": 0,
            "requests": 0,
            "unique_hits": 0,
            "unique_requests": 0,
            "per_window_hit_rate": [],
            "per_window_writes": [],
        }
        for name in policies
    }
    started = time.time()
    for window_index in range(int(args.windows)):
        start = window_index * int(args.window_size)
        stop = start + int(args.window_size)
        window = stream[start:stop]
        window_positions = positions[start:stop]
        if len(window) < int(args.window_size):
            break
        window_doc_ids = [doc_pool[int(pos)]["doc_id"] for pos in window_positions]
        unique_window_ids = set(window_doc_ids)
        # The actual evidence embedding is available after the cold fetch and is
        # used only by replacement updates for the next window.
        feedback_embeddings = doc_embs[window_positions]
        for name, policy in policies.items():
            hits = sum(doc_id in policy.kb for doc_id in window_doc_ids)
            unique_hits = len(unique_window_ids & policy.kb)
            record = metrics[name]
            record["hits"] += int(hits)
            record["requests"] += len(window_doc_ids)
            record["unique_hits"] += int(unique_hits)
            record["unique_requests"] += len(unique_window_ids)
            record["per_window_hit_rate"].append(
                round(hits / max(1, len(window_doc_ids)), 6)
            )
            writes_before = int(policy.update_cost)
            policy.step(window, feedback_embeddings, window_index)
            record["per_window_writes"].append(
                int(policy.update_cost) - writes_before
            )

    summaries = {}
    for name, policy in policies.items():
        record = metrics[name]
        cold_fetches = int(record["requests"] - record["hits"])
        topic_proposals = int(sum(
            int(item.get("topic_proposals_materialized", 0))
            for item in getattr(policy, "topic_log", ())
        ))
        proactive_fetches = int(sum(
            int(item.get("speculative_writes", 0))
            for item in getattr(policy, "cost_log", ())
        ))
        summaries[name] = {
            "evidence_hit_rate": round(
                record["hits"] / max(1, record["requests"]), 6
            ),
            "unique_evidence_coverage": round(
                record["unique_hits"] / max(1, record["unique_requests"]), 6
            ),
            "hits": int(record["hits"]),
            "requests": int(record["requests"]),
            "cache_writes": int(policy.update_cost),
            "cold_fetches": cold_fetches,
            "proactive_document_fetches": proactive_fetches,
            "topic_metadata_proposals": topic_proposals,
            "topic_index_probes": 0,
            "proactive_candidate_reads": 0,
            "total_cold_store_reads": cold_fetches + proactive_fetches,
            "implementation_reported_reads": int(policy.maint_retrieval_cost),
            "candidate_reads": int(policy.maint_retrieval_cost),
            "write_rate_per_request": round(
                policy.update_cost / max(1, record["requests"]), 6
            ),
            "per_window_hit_rate": record["per_window_hit_rate"],
            "per_window_writes": record["per_window_writes"],
        }
        if hasattr(policy, "prefetch_log"):
            summaries[name]["forecast_log"] = list(policy.prefetch_log)
        if hasattr(policy, "topic_log"):
            summaries[name]["topic_log"] = list(policy.topic_log)
        if hasattr(policy, "domain_log"):
            summaries[name]["domain_log"] = list(policy.domain_log)

    return {
        "dataset": "mind_news_context",
        "protocol": {
            "access_feedback": "post-service exact clicked-news ID",
            "current_request_leakage": False,
            "stream": temporal.as_dict(),
            "windows": int(args.windows),
            "window_size": int(args.window_size),
            "warmup_windows": int(args.warmup_windows),
            "cache_size": int(args.cache_size),
            "write_budget": int(args.write_budget),
            "candidate_budget": int(args.candidate_budget),
            "topic_field": str(args.topic_field),
            "topic_forecast_mass": float(args.topic_forecast_mass),
            "topic_min_forecast_confidence": float(
                args.min_forecast_confidence
            ),
            "provisional_placement": bool(args.enable_provisional_placement),
            "replacement_target": float(args.replacement_target),
            "initial_dual_price": float(args.initial_dual_price),
            "policy_configs": policy_configs,
            "policy_versions": {
                name: str(getattr(policy, "method_version", name))
                for name, policy in policies.items()
            },
            "seed": int(args.seed),
        },
        "topic_audit": topic_audits,
        "summary": summaries,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--warmup-windows", type=int, default=3)
    parser.add_argument("--cache-size", type=int, default=144)
    parser.add_argument("--write-budget", type=int, default=9)
    parser.add_argument("--candidate-budget", type=int, default=72)
    parser.add_argument(
        "--decay", type=float, default=0.50,
        help="TopicDynamics document-memory decay",
    )
    parser.add_argument("--demand-decay", type=float, default=0.25)
    parser.add_argument("--serve-decay", type=float, default=0.25)
    parser.add_argument("--domain-route-width", type=int, default=2)
    parser.add_argument("--domain-retrieve-topk", type=int, default=4)
    parser.add_argument("--domain-placement-weight", type=float, default=1.0)
    parser.add_argument(
        "--topic-field", choices=("category", "subcategory"),
        default="subcategory",
    )
    parser.add_argument("--topic-forecast-mass", type=float, default=8.0)
    parser.add_argument(
        "--min-forecast-confidence", type=float, default=0.50
    )
    parser.add_argument(
        "--enable-provisional-placement", action="store_true"
    )
    parser.add_argument("--replacement-target", type=float, default=0.25)
    parser.add_argument("--initial-dual-price", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temporal-sampling", choices=("prefix", "window_span"), default="prefix"
    )
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cache_size < 1 or args.write_budget < 0 or args.candidate_budget < 0:
        parser.error("cache size must be positive; budgets must be non-negative")
    result = run_mind(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        name: {
            "hit": values["evidence_hit_rate"],
            "writes": values["cache_writes"],
            "reads": values["total_cold_store_reads"],
        }
        for name, values in result["summary"].items()
    }
    print(json.dumps({"topic": result["topic_audit"], "cache": compact}, indent=2))


if __name__ == "__main__":
    main()
