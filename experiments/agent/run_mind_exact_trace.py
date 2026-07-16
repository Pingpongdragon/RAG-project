"""Natural chronological MIND evidence-residency and churn benchmark.

Positive clicks retain the official behavior timestamp order.  The current
impression slate is observable before service and acts as the query-conditioned
candidate set; the clicked news ID is revealed only after hit scoring.  This
exact trace therefore evaluates the same candidate-routing and priced-placement
mechanism without requiring a 51K-document embedding matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.agent.loaders import load_mind_news_context  # noqa: E402
from experiments.agent.run_mtrag_trace import (  # noqa: E402
    POLICY_NAMES,
    calibrate_capacity,
    evaluate_trace,
)
from experiments.common.stream_protocol import chronological_sample  # noqa: E402


def run(args):
    doc_pool, queries, _ = load_mind_news_context()
    selected, temporal = chronological_sample(
        queries,
        warmup_size=int(args.warmup_size),
        evaluation_size=int(args.evaluation_size),
        mode=str(args.temporal_sampling),
        block_size=int(args.window_size),
    )
    warmup = selected[: int(args.warmup_size)]
    evaluation = selected[int(args.warmup_size):]
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
    sweep = {}
    for write_budget in args.write_budgets:
        sweep[str(int(write_budget))] = evaluate_trace(
            warmup,
            evaluation,
            capacity=capacity,
            window_size=int(args.window_size),
            policy_names=args.policies,
            write_budget=int(write_budget),
            candidate_budget=int(args.candidate_budget),
            score_decay=float(args.score_decay),
        )
    return {
        "dataset": "mind_small_positive_clicks",
        "evaluation_scope": "exact clicked-news residency",
        "protocol": {
            "natural_chronology": True,
            "timestamp_field": "event_ts",
            "query_candidate_view": "official impression slate",
            "post_service_feedback": "clicked news ID",
            "documents": len(doc_pool),
            "warmup_events": len(warmup),
            "evaluation_events": len(evaluation),
            "window_size": int(args.window_size),
            "temporal_sampling": temporal.as_dict(),
            "candidate_budget": int(args.candidate_budget),
        },
        "capacity_calibration": capacity_audit,
        "cache_size": capacity,
        "write_budget_sweep": sweep,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-size", type=int, default=1500)
    parser.add_argument("--evaluation-size", type=int, default=25000)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument(
        "--temporal-sampling", choices=("prefix", "window_span"), default="prefix"
    )
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--capacity-coverage", type=float, default=0.90)
    parser.add_argument("--capacity-quantile", type=float, default=0.90)
    parser.add_argument("--candidate-budget", type=int, default=72)
    parser.add_argument("--write-budgets", nargs="+", type=int, default=[1, 5, 9])
    parser.add_argument("--score-decay", type=float, default=0.25)
    parser.add_argument(
        "--policies", nargs="+", choices=POLICY_NAMES,
        default=["LRU", "FIFO", "TinyLFU", "DRIP-Reactive", "DRIP-DomainAdapt"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("cache", result["cache_size"])
    for budget, summaries in result["write_budget_sweep"].items():
        print("budget", budget, {
            name: (summary["strict_all_support_hit_rate"], summary["cache_writes"])
            for name, summary in summaries.items()
        })


if __name__ == "__main__":
    main()
