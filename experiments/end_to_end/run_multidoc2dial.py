#!/usr/bin/env python3
"""Unified cache-policy benchmark on MultiDoc2Dial.

All policies share the same cold corpus, initial hot tier, query stream, cache
capacity, write cap, embeddings, and serve--score--update order.  DRIP-Feedback
adds two operations that are charged explicitly: independent current-request
domain routing and post-service answer/document attribution.  Citation mode
uses an access key only after the current response has been scored; answer
proxy mode uses the official response text and is not labelled as an LLM
generation-quality result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cache.adaptive.arc import ClassicalARC  # noqa: E402
from algorithms.cache.frequency.tinylfu import TinyLFU  # noqa: E402
from algorithms.cache.params import PARAMS  # noqa: E402
from algorithms.cache.paradigm_ref.agent_rag_cache import AgentRAGCache  # noqa: E402
from algorithms.cache.recency.fifo import FIFO  # noqa: E402
from algorithms.cache.recency.lru import LRU  # noqa: E402
from algorithms.cache.semantic.gptcache import GPTCacheStyle  # noqa: E402
from algorithms.cache.semantic.proximity import Proximity  # noqa: E402
from algorithms.drip import DRIP, DRIPConfig  # noqa: E402
from core.downstream_feedback import CurrentRequestUtilityEstimator  # noqa: E402
from experiments.agent.loaders import load_multidoc2dial  # noqa: E402
from experiments.common.session_workload import (  # noqa: E402
    CONTROLLED_RECURRING_DOMAIN,
    SESSION_ROUND_ROBIN,
    build_recurring_domain_workload,
    build_session_round_robin,
)


BOTH = "both"
POLICY_NAMES = (
    "LRU",
    "FIFO",
    "TinyLFU",
    "ClassicalARC",
    "AgentRAGCache",
    "Proximity",
    "GPTCacheStyle",
    "DRIP-Reactive",
    "DRIP-Feedback",
)
POLICY_CLASSES = {
    "LRU": LRU,
    "FIFO": FIFO,
    "TinyLFU": TinyLFU,
    "ClassicalARC": ClassicalARC,
    "AgentRAGCache": AgentRAGCache,
    "Proximity": Proximity,
    "GPTCacheStyle": GPTCacheStyle,
    "DRIP-Reactive": DRIP,
    "DRIP-Feedback": DRIP,
}


def _fingerprint(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def _embeddings(doc_pool, queries, model_name: str):
    cache = PROJECT_ROOT / "experiments" / "end_to_end" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    document_texts = [
        f"{document['source_title']}: {document['text'][:512]}"
        for document in doc_pool
    ]
    query_texts = [str(query["question"]) for query in queries]
    answer_texts = [str(query.get("answer") or query["question"]) for query in queries]
    key = _fingerprint([model_name, *document_texts, *query_texts, *answer_texts])
    paths = {
        "documents": cache / f"multidoc2dial_{key}_documents.npy",
        "queries": cache / f"multidoc2dial_{key}_queries.npy",
        "answers": cache / f"multidoc2dial_{key}_answers.npy",
    }
    if all(path.exists() for path in paths.values()):
        return tuple(
            np.load(paths[name]).astype(np.float32)
            for name in ("documents", "queries", "answers")
        )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    outputs = []
    for name, texts in (
        ("documents", document_texts),
        ("queries", query_texts),
        ("answers", answer_texts),
    ):
        values = model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(paths[name], values)
        outputs.append(values)
    return tuple(outputs)


def _initial_cache(doc_pool, doc_embs, query_embs, warmup, capacity, seed):
    positions = []
    for event in warmup:
        scores = doc_embs @ query_embs[int(event["qidx"])]
        positions.append(int(np.argmax(scores)))
    counts = {}
    for position in positions:
        counts[position] = counts.get(position, 0) + 1
    ranked = sorted(counts, key=lambda value: (-counts[value], value))
    selected = ranked[: int(capacity)]
    if len(selected) < int(capacity):
        remaining = sorted(set(range(len(doc_pool))) - set(selected))
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(remaining))
        selected.extend(
            remaining[int(index)]
            for index in order[: int(capacity) - len(selected)]
        )
    return {doc_pool[position]["doc_id"] for position in selected}


def _make_policy(name, doc_pool, doc_embs, title_to_idx, args):
    cls = POLICY_CLASSES[name]
    if name == "DRIP-Reactive":
        config = DRIPConfig.reactive(
            demand_decay=float(args.demand_decay),
            serve_decay=float(args.serve_decay),
            downstream_feedback_mass=0.0,
            replacement_target_rate=float(args.replacement_target),
            initial_dual_price=float(args.initial_dual_price),
        )
        return cls(name, doc_pool, doc_embs, title_to_idx, config), asdict(config)
    if name == "DRIP-Feedback":
        config = DRIPConfig.domain_adapt(
            candidate_budget=int(args.route_candidate_budget),
            metadata_field="domain",
            domain_prior_weight=0.0,
            domain_route_width=int(args.route_width),
            domain_retrieve_topk=int(args.retrieve_topk),
            downstream_feedback_mass=float(args.feedback_mass),
            downstream_feedback_topk=int(args.feedback_topk),
            demand_decay=float(args.demand_decay),
            serve_decay=float(args.serve_decay),
            replacement_target_rate=float(args.replacement_target),
            initial_dual_price=float(args.initial_dual_price),
        )
        return cls(name, doc_pool, doc_embs, title_to_idx, config), asdict(config)
    return cls(name, doc_pool, doc_embs, title_to_idx), None


def _retrieve(effective_kb, query_embedding, doc_pool, doc_embs, topk):
    positions = np.asarray(sorted(
        (index for index, document in enumerate(doc_pool)
         if document["doc_id"] in effective_kb),
    ), dtype=np.int64)
    if not len(positions):
        return set()
    scores = doc_embs[positions] @ query_embedding
    width = min(int(topk), len(positions))
    local = np.lexsort((positions, -scores))[:width]
    return {doc_pool[int(positions[index])]["title"] for index in local}


def _feedback_events(
    policy,
    window,
    query_rows,
    query_embs,
    answer_embs,
    doc_embs,
    doc_pool,
    estimator,
    feedback_mode,
):
    routed = policy._domain_routed_window
    position_to_title = {
        index: document["title"] for index, document in enumerate(doc_pool)
    }
    events = []
    diagnostics = {
        "queries": 0,
        "route_candidate_gold": 0,
        "attribution_top1_gold": 0,
    }
    for local, source in enumerate(window):
        event = dict(source)
        routed_candidates = tuple(
            routed.queries[local].documents
            if routed is not None and local < len(routed.queries) else ()
        )
        cited_positions = tuple(
            int(policy.title_to_idx[title])
            for title in event.get("sf_titles", ())
            if title in policy.title_to_idx
        )
        candidates = routed_candidates
        if feedback_mode == "citation":
            candidates = tuple(dict.fromkeys((*routed_candidates, *cited_positions)))
        utilities = estimator.score(
            query_embedding=query_embs[int(query_rows[local])],
            answer_embedding=answer_embs[int(query_rows[local])],
            candidate_positions=candidates,
            document_embeddings=doc_embs,
            cited_positions=(cited_positions if feedback_mode == "citation" else ()),
        )
        event["downstream_feedback"] = estimator.payload(
            utilities,
            position_to_title,
            source=(
                "post-service-citation"
                if feedback_mode == "citation"
                else "reference-response-attribution-proxy"
            ),
        )
        gold = set(str(value) for value in event.get("sf_titles", ()))
        routed_candidate_titles = {
            position_to_title[int(position)] for position in routed_candidates
        }
        diagnostics["queries"] += 1
        diagnostics["route_candidate_gold"] += int(
            bool(gold & routed_candidate_titles)
        )
        diagnostics["attribution_top1_gold"] += int(
            bool(utilities) and position_to_title[utilities[0].position] in gold
        )
        events.append(event)
    return events, diagnostics


def evaluate(
    doc_pool,
    queries,
    title_to_idx,
    doc_embs,
    query_embs,
    answer_embs,
    warmup,
    stream,
    *,
    args,
    write_budget,
):
    PARAMS.update(
        SEED=int(args.seed),
        WRITE_CAP=int(write_budget),
        DOC_ADD_CAP=int(write_budget),
        SF_HIT_THRESH=float(args.hit_threshold),
    )
    initial = _initial_cache(
        doc_pool, doc_embs, query_embs, warmup, int(args.cache_size), args.seed
    )
    policies = {}
    configs = {}
    for name in args.policies:
        policy, config = _make_policy(
            name, doc_pool, doc_embs, title_to_idx, args
        )
        policy.set_kb(initial)
        policies[name] = policy
        if config is not None:
            configs[name] = config

    metrics = {
        name: {
            "queries": 0,
            "persistent_support_hits": 0,
            "support_occurrences": 0,
            "persistent_strict_hits": 0,
            "service_support_hits": 0,
            "service_strict_hits": 0,
            "per_window_persistent_hit": [],
            "per_window_service_recall": [],
            "per_window_writes": [],
            "prepare_seconds": 0.0,
            "step_seconds": 0.0,
            "feedback_queries": 0,
            "attribution_top1_gold": 0,
            "route_candidate_gold": 0,
        }
        for name in policies
    }
    window_size = int(args.window_size)
    for window_index, start in enumerate(range(0, len(stream), window_size)):
        window = stream[start:start + window_size]
        if len(window) < window_size:
            break
        query_rows = np.asarray([int(event["qidx"]) for event in window])
        window_query_embs = query_embs[query_rows]
        for name, policy in policies.items():
            before_writes = int(policy.update_cost)
            started = time.perf_counter()
            policy.prepare_window(window, window_query_embs, window_index)
            metrics[name]["prepare_seconds"] += time.perf_counter() - started

            resident_ids = set(policy.kb)
            persistent_window = 0
            service_window = 0
            for local, event in enumerate(window):
                gold_titles = {
                    title for title in event.get("sf_titles", ())
                    if title in title_to_idx
                }
                gold_ids = {
                    doc_pool[title_to_idx[title]]["doc_id"] for title in gold_titles
                }
                persistent = len(gold_ids & resident_ids)
                effective_kb = (
                    policy.get_effective_kb_for_query(local)
                    if hasattr(policy, "get_effective_kb_for_query")
                    else policy.kb
                )
                retrieved = _retrieve(
                    effective_kb,
                    window_query_embs[local],
                    doc_pool,
                    doc_embs,
                    int(args.retrieve_topk),
                )
                service = len(gold_titles & retrieved)
                record = metrics[name]
                record["queries"] += 1
                record["persistent_support_hits"] += persistent
                record["support_occurrences"] += len(gold_titles)
                record["persistent_strict_hits"] += int(gold_ids <= resident_ids)
                record["service_support_hits"] += service
                record["service_strict_hits"] += int(gold_titles <= retrieved)
                persistent_window += persistent
                service_window += service

            # The exact document cited/used by the completed response is a
            # post-service access key. It is injected only after all current
            # cache and service metrics above have been recorded.
            step_events = [
                {
                    **event,
                    "access_title": event["sf_titles"][0],
                }
                for event in window
            ]
            if name == "DRIP-Feedback":
                estimator = CurrentRequestUtilityEstimator(
                    query_weight=float(args.feedback_query_weight),
                    answer_weight=float(args.feedback_answer_weight),
                    citation_bonus=float(args.feedback_citation_bonus),
                    temperature=float(args.feedback_temperature),
                    topk=int(args.feedback_topk),
                )
                step_events, feedback = _feedback_events(
                    policy,
                    step_events,
                    query_rows,
                    query_embs,
                    answer_embs,
                    doc_embs,
                    doc_pool,
                    estimator,
                    str(args.feedback_mode),
                )
                metrics[name]["feedback_queries"] += feedback["queries"]
                metrics[name]["attribution_top1_gold"] += feedback[
                    "attribution_top1_gold"
                ]
                metrics[name]["route_candidate_gold"] += feedback[
                    "route_candidate_gold"
                ]
            started = time.perf_counter()
            policy.step(step_events, window_query_embs, window_index)
            metrics[name]["step_seconds"] += time.perf_counter() - started
            denominator = max(1, sum(
                len(event.get("sf_titles", ())) for event in window
            ))
            metrics[name]["per_window_persistent_hit"].append(
                round(persistent_window / denominator, 6)
            )
            metrics[name]["per_window_service_recall"].append(
                round(service_window / denominator, 6)
            )
            metrics[name]["per_window_writes"].append(
                int(policy.update_cost) - before_writes
            )

    result = {}
    for name, policy in policies.items():
        record = metrics[name]
        queries_count = max(1, int(record["queries"]))
        support_count = max(1, int(record["support_occurrences"]))
        feedback_count = max(1, int(record["feedback_queries"]))
        result[name] = {
            "persistent_support_hit_rate": round(
                record["persistent_support_hits"] / support_count, 6
            ),
            "persistent_strict_query_hit_rate": round(
                record["persistent_strict_hits"] / queries_count, 6
            ),
            "service_recall_at_k": round(
                record["service_support_hits"] / support_count, 6
            ),
            "service_strict_query_recall_at_k": round(
                record["service_strict_hits"] / queries_count, 6
            ),
            "cache_writes": int(policy.update_cost),
            "maintenance_reads": int(policy.maint_retrieval_cost),
            "serve_route_reads": int(policy.serve_retrieval_cost),
            "total_implementation_reads": int(policy.retrieval_cost),
            "prepare_seconds": round(record["prepare_seconds"], 6),
            "step_seconds": round(record["step_seconds"], 6),
            "pre_service_route_candidate_recall": (
                round(record["route_candidate_gold"] / feedback_count, 6)
                if record["feedback_queries"] else None
            ),
            "post_service_attribution_top1_accuracy": (
                round(record["attribution_top1_gold"] / feedback_count, 6)
                if record["feedback_queries"] else None
            ),
            "per_window_persistent_hit": record["per_window_persistent_hit"],
            "per_window_service_recall": record["per_window_service_recall"],
            "per_window_writes": record["per_window_writes"],
            "method_version": str(getattr(policy, "method_version", name)),
        }
    return result, configs


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
    doc_pool, queries, title_to_idx = load_multidoc2dial(split=args.split)
    doc_embs, query_embs, answer_embs = _embeddings(
        doc_pool, queries, str(args.embedding_model)
    )
    result = {
        "dataset": "multidoc2dial",
        "split": args.split,
        "protocol_note": (
            "No global timestamp. Session order is preserved; controlled "
            "domain scheduling is explicitly synthetic."
        ),
        "feedback_note": (
            "All policies receive the completed request's citation/access key "
            "after scoring. DRIP additionally consumes either current answer "
            "alignment or the same post-service citation as utility feedback."
        ),
        "config": vars(args) | {"output": str(args.output)},
        "protocols": {},
    }
    for protocol, (stream, warmup, audit) in _protocols(args, queries):
        sweeps = {}
        policy_configs = {}
        for budget in args.write_budgets:
            summaries, configs = evaluate(
                doc_pool,
                queries,
                title_to_idx,
                doc_embs,
                query_embs,
                answer_embs,
                warmup,
                stream,
                args=args,
                write_budget=int(budget),
            )
            sweeps[str(int(budget))] = summaries
            policy_configs.update(configs)
        result["protocols"][protocol] = {
            "workload": audit.as_dict(),
            "write_budget_sweep": sweeps,
            "policy_configs": policy_configs,
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
    parser.add_argument("--cache-size", type=int, default=24)
    parser.add_argument("--write-budgets", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--retrieve-topk", type=int, default=4)
    parser.add_argument("--route-width", type=int, default=2)
    parser.add_argument("--route-candidate-budget", type=int, default=100)
    parser.add_argument("--feedback-topk", type=int, default=4)
    parser.add_argument("--feedback-mass", type=float, default=1.0)
    parser.add_argument("--feedback-query-weight", type=float, default=0.35)
    parser.add_argument("--feedback-answer-weight", type=float, default=0.65)
    parser.add_argument("--feedback-temperature", type=float, default=0.10)
    parser.add_argument(
        "--feedback-mode", choices=("answer_proxy", "citation"),
        default="citation",
    )
    parser.add_argument("--feedback-citation-bonus", type=float, default=1.0)
    parser.add_argument("--demand-decay", type=float, default=0.25)
    parser.add_argument("--serve-decay", type=float, default=0.25)
    parser.add_argument("--replacement-target", type=float, default=0.25)
    parser.add_argument("--initial-dual-price", type=float, default=0.25)
    parser.add_argument("--hit-threshold", type=float, default=0.55)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--policies", nargs="+", choices=POLICY_NAMES,
        default=list(POLICY_NAMES),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for protocol, values in result["protocols"].items():
        print(protocol)
        for budget, summaries in values["write_budget_sweep"].items():
            print(" budget", budget, {
                name: (
                    row["persistent_support_hit_rate"],
                    row["service_recall_at_k"],
                    row["cache_writes"],
                )
                for name, row in summaries.items()
            })


if __name__ == "__main__":
    main()
