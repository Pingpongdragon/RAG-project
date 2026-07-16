"""Temporal calibration/holdout tuning for MIND exact-access replacement.

Only the first ``calibration_windows`` after a causal warm-up select the
configuration.  Later windows are reported once as a locked temporal holdout.
The search is intentionally small and mechanism-level: replacement target and
initial dual price.  Topic forecasting remains disabled during this stage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.cache.params import PARAMS
from algorithms.cache.recency.lru import LRU
from algorithms.drip import DRIP, DRIPConfig
from experiments.common.stream_protocol import chronological_sample
from experiments.agent.loaders import load_mind_news_context
from experiments.agent.utils import compute_document_embeddings


def _initial_cache(doc_pool, title_to_idx, warmup, budget):
    counts = Counter(query["access_title"] for query in warmup)
    ranked = sorted(counts, key=lambda title: (-counts[title], title))
    resident = {
        doc_pool[title_to_idx[title]]["doc_id"] for title in ranked[:budget]
    }
    for document in doc_pool:
        if len(resident) >= budget:
            break
        resident.add(document["doc_id"])
    return resident


def run(args):
    doc_pool, queries, title_to_idx = load_mind_news_context()
    total_windows = int(args.calibration_windows) + int(args.holdout_windows)
    selected, temporal = chronological_sample(
        queries,
        warmup_size=int(args.warmup_windows) * int(args.window_size),
        evaluation_size=total_windows * int(args.window_size),
        mode="prefix",
        block_size=int(args.window_size),
    )
    del queries
    warmup_size = int(args.warmup_windows) * int(args.window_size)
    warmup, stream = selected[:warmup_size], selected[warmup_size:]
    doc_embs = compute_document_embeddings(doc_pool)
    positions = np.asarray(
        [title_to_idx[query["access_title"]] for query in stream],
        dtype=np.int64,
    )
    initial = _initial_cache(
        doc_pool, title_to_idx, warmup, int(args.cache_size)
    )
    PARAMS.WRITE_CAP = int(args.write_budget)

    configs = []
    for target, price in itertools.product(
        args.replacement_targets, args.initial_prices
    ):
        config = DRIPConfig.reactive(
            replacement_target_rate=float(target),
            initial_dual_price=float(price),
        )
        name = f"target={target:g},price={price:g}"
        policy = DRIP(name, doc_pool, doc_embs, title_to_idx, config)
        policy.set_kb(initial)
        configs.append((name, config, policy))
    lru = LRU("LRU", doc_pool, doc_embs, title_to_idx)
    lru.set_kb(initial)

    hits = {name: [] for name, _, _ in configs}
    hits["LRU"] = []
    writes = {name: [] for name, _, _ in configs}
    writes["LRU"] = []
    previous_writes = {name: 0 for name in hits}
    ws = int(args.window_size)
    for window_index in range(total_windows):
        start, stop = window_index * ws, (window_index + 1) * ws
        window = stream[start:stop]
        window_positions = positions[start:stop]
        feedback = doc_embs[window_positions]
        window_doc_ids = [
            doc_pool[int(position)]["doc_id"] for position in window_positions
        ]
        for name, _, policy in configs:
            hits[name].append(sum(doc_id in policy.kb for doc_id in window_doc_ids))
            policy.step(window, feedback, window_index)
            writes[name].append(policy.update_cost - previous_writes[name])
            previous_writes[name] = policy.update_cost
        hits["LRU"].append(sum(doc_id in lru.kb for doc_id in window_doc_ids))
        lru.step(window, feedback, window_index)
        writes["LRU"].append(lru.update_cost - previous_writes["LRU"])
        previous_writes["LRU"] = lru.update_cost

    calibration = slice(0, int(args.calibration_windows))
    holdout = slice(int(args.calibration_windows), total_windows)
    results = {}
    for name, config, policy in configs:
        results[name] = {
            "config": asdict(config),
            "calibration_hit_rate": round(
                sum(hits[name][calibration])
                / (int(args.calibration_windows) * ws), 6
            ),
            "holdout_hit_rate": round(
                sum(hits[name][holdout])
                / (int(args.holdout_windows) * ws), 6
            ),
            "calibration_writes": int(sum(writes[name][calibration])),
            "holdout_writes": int(sum(writes[name][holdout])),
            "per_window_hits": list(map(int, hits[name])),
            "per_window_writes": list(map(int, writes[name])),
            "candidate_reads": int(policy.maint_retrieval_cost),
        }
    results["LRU"] = {
        "calibration_hit_rate": round(
            sum(hits["LRU"][calibration])
            / (int(args.calibration_windows) * ws), 6
        ),
        "holdout_hit_rate": round(
            sum(hits["LRU"][holdout])
            / (int(args.holdout_windows) * ws), 6
        ),
        "calibration_writes": int(sum(writes["LRU"][calibration])),
        "holdout_writes": int(sum(writes["LRU"][holdout])),
        "per_window_hits": list(map(int, hits["LRU"])),
        "per_window_writes": list(map(int, writes["LRU"])),
        "candidate_reads": int(lru.maint_retrieval_cost),
    }
    selected_name = max(
        (name for name, _, _ in configs),
        key=lambda name: (
            results[name]["calibration_hit_rate"],
            -results[name]["calibration_writes"],
            name,
        ),
    )
    return {
        "protocol": {
            "dataset": "mind_news_context",
            "temporal_sampling": temporal.as_dict(),
            "warmup_windows": int(args.warmup_windows),
            "calibration_windows": int(args.calibration_windows),
            "holdout_windows": int(args.holdout_windows),
            "window_size": ws,
            "cache_size": int(args.cache_size),
            "write_budget": int(args.write_budget),
            "selection_metric": "calibration evidence hit; writes as tie-break",
        },
        "selected_on_calibration": selected_name,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-windows", type=int, default=3)
    parser.add_argument("--calibration-windows", type=int, default=20)
    parser.add_argument("--holdout-windows", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--cache-size", type=int, default=144)
    parser.add_argument("--write-budget", type=int, default=9)
    parser.add_argument(
        "--replacement-targets", nargs="+", type=float,
        default=(0.25, 0.5, 1.0),
    )
    parser.add_argument(
        "--initial-prices", nargs="+", type=float, default=(0.0, 0.25)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected": result["selected_on_calibration"],
        "scores": {
            name: {
                "cal": values["calibration_hit_rate"],
                "test": values["holdout_hit_rate"],
                "test_writes": values["holdout_writes"],
            }
            for name, values in result["results"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
