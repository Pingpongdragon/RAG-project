"""Calibration-only tuning for the controlled TopicDynamics trace.

Calibration and reported evaluation use disjoint stable source-family pools;
different workload seeds alone are not treated as data separation.
Only the replacement target and its initial dual price are selected; capacity,
write budget, topic partition, and proactive fraction remain fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.cache.params import PARAMS
from algorithms.drip import DRIP, DRIPConfig
from benchmarks.audit_semantic_pages import build_balanced_semantic_pages, prepare_dataset
from benchmarks.run_controlled_topic_trace import _feedback_window, _initial_cache
from experiments.common.factorized_workload import SOURCE_POOL_CALIBRATION


def _name(target: float, price: float) -> str:
    return f"target={target:g},price={price:g}"


def run(args):
    scores = defaultdict(list)
    writes = defaultdict(list)
    per_seed = {}
    PARAMS.WRITE_CAP = int(args.write_budget)

    configurations = list(itertools.product(
        args.replacement_targets, args.initial_prices
    ))
    effective_policy_configs = {}
    source_split = None
    for workload_seed in args.calibration_seeds:
        prep_args = argparse.Namespace(
            n_source=args.n_source,
            n_windows=args.windows,
            window_size=args.window_size,
            warmup_windows=args.warmup_windows,
            workload_seed=int(workload_seed),
            workload=args.workload,
            source_pool=SOURCE_POOL_CALIBRATION,
            source_pool_seed=int(args.source_pool_seed),
            source_pool_calibration_fraction=float(
                args.source_pool_calibration_fraction
            ),
        )
        dataset = prepare_dataset(args.dataset, prep_args)
        if dataset.source_split is None:
            raise AssertionError("calibration source-family split was not applied")
        if not dataset.source_split["overlap_assertion"]:
            raise AssertionError(f"source-pool leakage: {dataset.source_split}")
        if source_split is None:
            source_split = dict(dataset.source_split)
        elif dataset.source_split != source_split:
            raise AssertionError(
                "source-family split changed across calibration workload seeds"
            )
        labels, _, _ = build_balanced_semantic_pages(
            dataset,
            target_page_size=int(args.semantic_region_size),
            seed=int(args.partition_seed),
        )
        for position, document in enumerate(dataset.doc_pool):
            document["semantic_topic"] = int(labels[position])
        initial = _initial_cache(dataset, int(args.cache_size))

        policies = {}
        for target, price in configurations:
            name = _name(target, price)
            config = DRIPConfig.topic_dynamics(
                candidate_budget=int(args.candidate_budget),
                metadata_field="semantic_topic",
                replacement_target_rate=float(target),
                initial_dual_price=float(price),
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
            policy = DRIP(
                name,
                dataset.doc_pool,
                dataset.doc_embs,
                dataset.title_to_idx,
                config,
            )
            effective_policy_configs[name] = asdict(config)
            policy.set_kb(initial)
            policies[name] = policy

        object_hits = defaultdict(int)
        object_requests = 0
        ws = int(dataset.protocol.window_size)
        for window_index in range(int(dataset.protocol.n_windows)):
            start, stop = window_index * ws, (window_index + 1) * ws
            window = dataset.stream[start:stop]
            if len(window) < ws:
                break
            feedback, positions, _, feedback_query_rows = _feedback_window(
                dataset, window
            )
            query_rows = np.asarray(
                [int(query["qidx"]) for query in window], dtype=np.int64
            )
            embeddings = dataset.query_embs[query_rows][feedback_query_rows]
            object_requests += len(positions)
            for name, policy in policies.items():
                resident = {int(policy.d2p[doc_id]) for doc_id in policy.kb}
                object_hits[name] += sum(int(position) in resident for position in positions)
                policy.step(feedback, embeddings, window_index)

        seed_result = {}
        for name, policy in policies.items():
            hit_rate = object_hits[name] / max(1, object_requests)
            scores[name].append(hit_rate)
            writes[name].append(int(policy.update_cost))
            seed_result[name] = {
                "evidence_hit_rate": round(hit_rate, 6),
                "cache_writes": int(policy.update_cost),
            }
        per_seed[str(workload_seed)] = seed_result

    aggregate = {}
    for name in sorted(scores):
        aggregate[name] = {
            "mean_evidence_hit_rate": round(sum(scores[name]) / len(scores[name]), 6),
            "mean_cache_writes": round(sum(writes[name]) / len(writes[name]), 3),
        }
    selected = max(
        aggregate,
        key=lambda name: (
            aggregate[name]["mean_evidence_hit_rate"],
            -aggregate[name]["mean_cache_writes"],
            name,
        ),
    )
    return {
        "protocol": {
            "dataset": args.dataset,
            "workload": str(dataset.protocol.workload),
            "calibration_seeds": list(map(int, args.calibration_seeds)),
            "reported_test_seeds": list(map(int, args.reported_test_seeds)),
            "disjoint_seeds": not bool(
                set(args.calibration_seeds) & set(args.reported_test_seeds)
            ),
            "source_split": source_split,
            "disjoint_source_queries": source_split["query_overlap"] == 0,
            "disjoint_source_supports": source_split["support_overlap"] == 0,
            "disjoint_source_families": source_split["family_overlap"] == 0,
            "cache_size": int(args.cache_size),
            "warmup_windows": int(dataset.protocol.warmup_windows),
            "windows": int(dataset.protocol.n_windows),
            "window_size": int(dataset.protocol.window_size),
            "write_budget": int(args.write_budget),
            "candidate_budget": int(args.candidate_budget),
            "semantic_region_size": int(args.semantic_region_size),
            "partition_seed": int(args.partition_seed),
            "topic_document_decay": float(args.decay),
            "topic_forecast_mass": float(args.forecast_mass),
            "topic_min_forecast_confidence": float(
                args.min_forecast_confidence
            ),
            "topic_proactive_fraction": float(args.proactive_fraction),
            "provisional_placement": bool(args.enable_provisional_placement),
            "selection_metric": "mean calibration evidence hit; writes as tie-break",
            "selection_scope": (
                "residency objective only; proactive document reads are not "
                "part of the tuning objective"
            ),
            "method_version": str(DRIP.method_version),
            "effective_policy_configs": effective_policy_configs,
        },
        "selected_on_calibration": selected,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="squad_direct")
    parser.add_argument("--workload", default=None)
    parser.add_argument("--n-source", type=int, default=None)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=25)
    # ``None`` preserves each dataset's declared protocol (SQuAD uses one
    # warm-up window); calibration must not silently redefine the benchmark.
    parser.add_argument("--warmup-windows", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=24)
    parser.add_argument("--write-budget", type=int, default=3)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--semantic-region-size", type=int, default=128)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--source-pool-seed", type=int, default=1729)
    parser.add_argument(
        "--source-pool-calibration-fraction", type=float, default=0.5
    )
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=(11, 12, 13, 14, 15))
    parser.add_argument("--reported-test-seeds", nargs="+", type=int, default=(42, 43, 44, 45, 46))
    parser.add_argument("--replacement-targets", nargs="+", type=float, default=(0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--initial-prices", nargs="+", type=float, default=(0.0, 0.25, 0.5))
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--forecast-mass", type=float, default=20.0)
    parser.add_argument(
        "--min-forecast-confidence", type=float, default=0.5
    )
    parser.add_argument("--proactive-fraction", type=float, default=0.25)
    parser.add_argument(
        "--enable-provisional-placement", action="store_true"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if set(args.calibration_seeds) & set(args.reported_test_seeds):
        parser.error("calibration and reported test seeds must be disjoint")
    if not 0.0 < args.source_pool_calibration_fraction < 1.0:
        parser.error("--source-pool-calibration-fraction must be in (0, 1)")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": result["selected_on_calibration"],
        "aggregate": result["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
