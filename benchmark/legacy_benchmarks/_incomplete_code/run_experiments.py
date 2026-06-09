"""
Experiment Runner CLI (v2)

Usage:
    # Quick test (random embeddings, small scale)
    python -m benchmarks.run_experiments --quick

    # Full run (real embeddings, full scale)
    python -m benchmarks.run_experiments --full

    # Single experiment
    python -m benchmarks.run_experiments --exp gradual_drift --full

    # Use pre-generated JSON datasets (skip building from raw data)
    python -m benchmarks.run_experiments --quick --from-json benchmarks/data/

    # Custom parameters
    python -m benchmarks.run_experiments --quick --kb-budget 30 --window-size 15 --queries 200
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.builders import (
    build_gradual_drift,
    build_sudden_shift,
    build_cyclic_return,
    build_hotpotqa_entity_walk,
)
from benchmarks.experiment_framework import (
    EmbeddingHelper,
    QARCAdapter,
    ComRAGAdapter,
    ERASEAdapter,
    StaticKBAdapter,
    RandomKBAdapter,
    run_comparison,
)

logger = logging.getLogger(__name__)


def _make_methods(kb_budget: int, window_size: int):
    """Instantiate all methods to compare."""
    candidate_top_k = 200  # 每个兴趣簇从pool取200候选，总候选≈200*m
    return [
        QARCAdapter(kb_budget=kb_budget, window_size=window_size,
                    candidate_top_k=candidate_top_k,
                    agent_warmup_windows=1,
                    agent_lambda_aggressive=0.5),
        ComRAGAdapter(kb_budget=kb_budget, top_k=10),
        ERASEAdapter(kb_budget=kb_budget, update_threshold=0.7, top_k=10),
        StaticKBAdapter(kb_budget=kb_budget, top_k=10),
        RandomKBAdapter(kb_budget=kb_budget, top_k=10, seed=42),
    ]


def run_gradual_drift(embedder, kb_budget, window_size, total_queries, out_dir, pool_size=50000, wow_split='train'):
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 1: Gradual Drift (Gaussian schedule)")
    logger.info("=" * 80)
    ds = build_gradual_drift(total_queries=total_queries, pool_size=pool_size, wow_split=wow_split)
    methods = _make_methods(kb_budget, window_size)
    return run_comparison(
        ds, methods, embedder,
        eval_window_size=window_size,
        output_path=os.path.join(out_dir, "exp1_gradual_drift.json"),
    )


def run_sudden_shift(embedder, kb_budget, window_size, total_queries, out_dir, pool_size=50000, wow_split='train'):
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 2: Sudden Shift (Sigmoid schedule)")
    logger.info("=" * 80)
    ds = build_sudden_shift(total_queries=total_queries, pool_size=pool_size, wow_split=wow_split)
    methods = _make_methods(kb_budget, window_size)
    return run_comparison(
        ds, methods, embedder,
        eval_window_size=window_size,
        output_path=os.path.join(out_dir, "exp2_sudden_shift.json"),
    )


def run_cyclic_return(embedder, kb_budget, window_size, total_queries, out_dir, pool_size=50000, wow_split='train'):
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 3: Cyclic Return (Periodic schedule)")
    logger.info("=" * 80)
    ds = build_cyclic_return(total_queries=total_queries, pool_size=pool_size, wow_split=wow_split)
    methods = _make_methods(kb_budget, window_size)
    return run_comparison(
        ds, methods, embedder,
        eval_window_size=window_size,
        output_path=os.path.join(out_dir, "exp3_cyclic_return.json"),
    )




def run_hotpotqa_walk(embedder, kb_budget, window_size, total_queries, out_dir, pool_size=50000):
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT 4: HotpotQA Entity Walk (no hard classification)")
    logger.info("=" * 80)
    ds = build_hotpotqa_entity_walk(total_queries=total_queries, max_pool=pool_size)
    methods = _make_methods(kb_budget, window_size)
    return run_comparison(
        ds, methods, embedder,
        eval_window_size=window_size,
        output_path=os.path.join(out_dir, "exp4_hotpotqa_walk.json"),
    )

EXPERIMENTS = {
    "gradual_drift":  run_gradual_drift,
    "sudden_shift":   run_sudden_shift,
    "cyclic_return":  run_cyclic_return,
    "hotpotqa_walk":  run_hotpotqa_walk,
}


def main():
    parser = argparse.ArgumentParser(description="Run KB curation experiments")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Quick test with random embeddings")
    mode.add_argument("--full", action="store_true",
                      help="Full run with real embeddings")
    parser.add_argument("--exp", choices=list(EXPERIMENTS.keys()),
                        help="Run a single experiment (default: all)")
    parser.add_argument("--kb-budget", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--queries", type=int, default=None,
                        help="Total queries per experiment (default: 200 quick / 5000 full)")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Document pool size (default: 5000 quick / 50000 full)")
    parser.add_argument("--wow-split", type=str, default=None,
                        choices=["validation", "train"],
                        help="WoW dataset split (default: validation quick / train full)")
    parser.add_argument("--from-json", type=str, default=None,
                        help="Load pre-built datasets from this directory (JSON files)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Defaults
    use_random = args.quick or not args.full
    total_queries = args.queries or (200 if use_random else 5000)
    pool_size = args.pool_size or (5000 if use_random else 50000)
    wow_split = args.wow_split or ("validation" if use_random else "train")
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = 'cpu' if use_random else 'cuda'

    logger.info(f"Mode: {'quick (random emb)' if use_random else 'full (real emb)'}")
    logger.info(f"KB budget: {args.kb_budget}, Window: {args.window_size}, "
                f"Queries: {total_queries}, Pool: {pool_size}")
    logger.info(f"WoW split: {wow_split}, Device: {device}")
    logger.info(f"Output: {out_dir}")

    embedder = EmbeddingHelper(use_random=use_random, device=device)

    all_results = {}
    exps_to_run = [args.exp] if args.exp else list(EXPERIMENTS.keys())

    # JSON pre-built dataset mapping
    JSON_NAME_MAP = {
        "gradual_drift": "gradual_drift.json",
        "sudden_shift": "sudden_shift.json",
        "cyclic_return": "cyclic_return.json",
        "hotpotqa_walk": "hotpotqa_entity_walk.json",
    }

    for exp_name in exps_to_run:
        if args.from_json:
            # Load pre-built dataset from JSON
            from benchmarks.data_structures import ExperimentDataset
            json_path = os.path.join(args.from_json, JSON_NAME_MAP[exp_name])
            logger.info(f"Loading pre-built dataset: {json_path}")
            ds = ExperimentDataset.load_json(json_path)
            methods = _make_methods(args.kb_budget, args.window_size)
            all_results[exp_name] = run_comparison(
                ds, methods, embedder,
                eval_window_size=args.window_size,
                output_path=os.path.join(out_dir, f"{exp_name}.json"),
            )
        else:
            fn = EXPERIMENTS[exp_name]
            kwargs = dict(embedder=embedder, kb_budget=args.kb_budget,
                          window_size=args.window_size,
                          total_queries=total_queries, out_dir=out_dir,
                          pool_size=pool_size)
            if exp_name != "hotpotqa_walk":
                kwargs["wow_split"] = wow_split
            all_results[exp_name] = fn(**kwargs)

    # Final summary
    print("\n" + "=" * 85)
    print("  FINAL SUMMARY — All Experiments")
    print("=" * 85)
    for exp_name, results in all_results.items():
        print(f"\n  [{exp_name}]")
        for name, r in results.items():
            print(f"    {name:<12}  "
                  f"recall={r.avg_recall:.4f}  "
                  f"gold_kb={r.avg_gold_in_kb:.4f}  "
                  f"align={r.avg_topic_alignment:.4f}  "
                  f"adapt={r.adaptation_speed:.1f}w  "
                  f"qry={r.query_time_s:.1f}s  "
                  f"upd={r.update_time_s:.1f}s  "
                  f"total={r.total_time_s:.1f}s")
    print()

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
