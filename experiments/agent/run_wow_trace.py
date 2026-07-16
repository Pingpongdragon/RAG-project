"""Ordered Wizard-of-Wikipedia evidence-residency benchmark.

WoW has turn order inside each dialogue but no global timestamp.  This runner
therefore reports two explicit protocols:

* ``session_round_robin`` preserves dialogue order and interleaves sessions;
* ``controlled_recurring_domain`` first clusters dialogue topic *names* into a
  small number of offline coarse domains, then cycles domain-sized blocks while
  preserving dialogue order.

The constructor never reads selected knowledge or future stream order.  The
selected knowledge sentence becomes post-service evidence feedback.  Candidate
knowledge IDs are observable current-query retrieval candidates and are used by
the exact-trace DomainAdapt adapter without revealing which candidate is gold.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.agent.loaders import load_wizard_of_wikipedia  # noqa: E402
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


def _topic_fingerprint(topics: list[str], model: str, domains: int, seed: int):
    digest = hashlib.sha256()
    for topic in topics:
        digest.update(topic.encode("utf-8"))
        digest.update(b"\0")
    digest.update(f"{model}|{domains}|{seed}".encode("utf-8"))
    return digest.hexdigest()[:16]


def cluster_dialogue_topics(
    queries,
    *,
    n_domains: int,
    seed: int,
    model_name: str,
):
    """Return copied queries with deterministic offline coarse-domain labels."""

    topics = sorted({str(query["topic"]) for query in queries})
    n_domains = int(n_domains)
    if not 1 <= n_domains <= len(topics):
        raise ValueError("n_domains must lie between 1 and unique topics")
    cache_dir = PROJECT_ROOT / "experiments" / "agent" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _topic_fingerprint(topics, model_name, n_domains, seed)
    cache_path = cache_dir / f"wow_topic_clusters_{fingerprint}.json"
    if cache_path.exists():
        topic_to_cluster = json.loads(cache_path.read_text())
    else:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans

        encoder = SentenceTransformer(model_name, device="cpu")
        embeddings = encoder.encode(
            topics,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        labels = KMeans(
            n_clusters=n_domains, random_state=int(seed), n_init=10
        ).fit_predict(np.asarray(embeddings, dtype=np.float32))
        topic_to_cluster = {
            topic: int(label) for topic, label in zip(topics, labels)
        }
        cache_path.write_text(
            json.dumps(topic_to_cluster, indent=2, sort_keys=True)
        )

    copied = []
    for source in queries:
        event = dict(source)
        cluster = int(topic_to_cluster[str(event["topic"])])
        event["domain"] = f"wow-domain-{cluster}"
        event["coarse_domain"] = cluster
        event["domain_constructor"] = (
            "offline topic-name embedding KMeans; no evidence or stream order"
        )
        copied.append(event)
    audit = {
        "unique_topics": len(topics),
        "coarse_domains": n_domains,
        "model": model_name,
        "seed": int(seed),
        "cache": str(cache_path.relative_to(PROJECT_ROOT)),
        "topics_per_domain": dict(sorted(Counter(
            topic_to_cluster.values()
        ).items())),
        "events_per_domain": dict(sorted(Counter(
            event["domain"] for event in copied
        ).items())),
    }
    return copied, audit


def _protocols(args, queries):
    requested = (
        [SESSION_ROUND_ROBIN, CONTROLLED_RECURRING_DOMAIN]
        if args.protocol == BOTH else [args.protocol]
    )
    for protocol in requested:
        common = dict(
            seed=int(args.seed),
            warmup_size=int(args.warmup_size),
            evaluation_size=int(args.evaluation_size),
            window_size=int(args.window_size),
        )
        if protocol == SESSION_ROUND_ROBIN:
            yield protocol, build_session_round_robin(queries, **common)
        else:
            yield protocol, build_recurring_domain_workload(
                queries, block_size=int(args.block_size), **common
            )


def run(args):
    _, raw_queries, _ = load_wizard_of_wikipedia(split=args.split)
    queries, domain_audit = cluster_dialogue_topics(
        raw_queries,
        n_domains=int(args.coarse_domains),
        seed=int(args.seed),
        model_name=str(args.topic_model),
    )
    result = {
        "dataset": "wizard_of_wikipedia",
        "evaluation_scope": "exact selected-knowledge sentence residency",
        "global_timestamp_available": False,
        "domain_partition": domain_audit,
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
        budget_results = {}
        for write_budget in args.write_budgets:
            budget_results[str(int(write_budget))] = evaluate_trace(
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
            "candidate_budget": int(args.candidate_budget),
            "write_budget_sweep": budget_results,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--protocol",
        choices=(SESSION_ROUND_ROBIN, CONTROLLED_RECURRING_DOMAIN, BOTH),
        default=BOTH,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coarse-domains", type=int, default=8)
    parser.add_argument("--topic-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--warmup-size", type=int, default=500)
    parser.add_argument("--evaluation-size", type=int, default=5000)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--cache-size", type=int, default=0)
    parser.add_argument("--capacity-coverage", type=float, default=0.90)
    parser.add_argument("--capacity-quantile", type=float, default=0.90)
    parser.add_argument("--candidate-budget", type=int, default=24)
    parser.add_argument("--write-budgets", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--score-decay", type=float, default=0.25)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=["LRU", "FIFO", "TinyLFU", "DRIP-Reactive", "DRIP-DomainAdapt"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for protocol, values in result["protocols"].items():
        print(protocol, "cache", values["cache_size"])
        for budget, summaries in values["write_budget_sweep"].items():
            print(" budget", budget, {
                name: (summary["strict_all_support_hit_rate"], summary["cache_writes"])
                for name, summary in summaries.items()
            })


if __name__ == "__main__":
    main()
