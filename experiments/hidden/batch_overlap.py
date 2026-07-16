#!/usr/bin/env python3
"""100-agent shared-cache pressure diagnostic.

This is not a real AgentBench/WebShop trace. It is the documented fallback
for the agentic setting: treat each sampled multi-hop QA item as one concurrent
agent request and measure whether hidden supporting documents are shared across
agents in the same batch.

Hidden support = a supporting-fact title that does not appear verbatim in the
question text. In entity-title datasets such as 2Wiki/HotpotQA, this is a
conservative proxy for bridge evidence that the query embedding may not expose.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from loaders import LOADERS
from config import DATA_DIR, log


def _norm(s):
    return " ".join(str(s).lower().split())


def _hidden_supports(q):
    question = _norm(q.get("question", ""))
    hidden = []
    visible = []
    for title in q.get("sf_titles", []):
        t = _norm(title)
        if t and t in question:
            visible.append(title)
        else:
            hidden.append(title)
    return hidden, visible


def _agent_shared_fraction(items, key):
    """Fraction of agents whose key-set intersects another agent's key-set."""
    counts = {}
    sets = []
    for it in items:
        s = set(it[key])
        sets.append(s)
        for x in s:
            counts[x] = counts.get(x, 0) + 1
    if not sets:
        return 0.0
    shared = sum(1 for s in sets if any(counts[x] > 1 for x in s))
    return shared / len(sets)


def _pairwise_overlap_rate(items, key):
    sets = [set(it[key]) for it in items]
    n = len(sets)
    if n < 2:
        return 0.0
    hit = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if sets[i] & sets[j]:
                hit += 1
    return hit / total


def _duplicate_ref_rate(items, key):
    refs = [x for it in items for x in it[key]]
    if not refs:
        return 0.0
    return (len(refs) - len(set(refs))) / len(refs)


def run(dataset, loader_name, q_type, n_source, batch_size, n_batches, seed):
    loader = LOADERS[loader_name]
    if loader_name == "2wiki_expanded":
        _, queries, _ = loader(n_source=n_source, q_type=q_type)
    else:
        _, queries, _ = loader()

    records = []
    for i, q in enumerate(queries):
        hidden, visible = _hidden_supports(q)
        records.append({
            "idx": i,
            "sf": list(q.get("sf_titles", [])),
            "hidden_sf": hidden,
            "visible_sf": visible,
            "question": q.get("question", ""),
        })

    rng = np.random.default_rng(seed)
    batch_metrics = []
    n = len(records)
    if n == 0:
        raise RuntimeError("no queries loaded")
    bsz = min(batch_size, n)
    for _ in range(n_batches):
        idx = rng.choice(n, bsz, replace=False)
        batch = [records[int(i)] for i in idx]
        batch_metrics.append({
            "agents_with_shared_sf": _agent_shared_fraction(batch, "sf"),
            "agents_with_shared_hidden_sf": _agent_shared_fraction(batch, "hidden_sf"),
            "pairwise_sf_overlap": _pairwise_overlap_rate(batch, "sf"),
            "pairwise_hidden_sf_overlap": _pairwise_overlap_rate(batch, "hidden_sf"),
            "duplicate_sf_ref_rate": _duplicate_ref_rate(batch, "sf"),
            "duplicate_hidden_sf_ref_rate": _duplicate_ref_rate(batch, "hidden_sf"),
            "mean_hidden_sf_per_agent": float(np.mean([len(x["hidden_sf"]) for x in batch])),
        })

    summary = {
        k: round(float(np.mean([m[k] for m in batch_metrics])), 4)
        for k in batch_metrics[0]
    }
    summary.update({
        f"{k}_std": round(float(np.std([m[k] for m in batch_metrics])), 4)
        for k in batch_metrics[0]
    })

    out = {
        "dataset": dataset,
        "loader": loader_name,
        "q_type": q_type,
        "n_source": n_source,
        "n_queries": len(records),
        "batch_size": bsz,
        "n_batches": n_batches,
        "seed": seed,
        "definition": {
            "agent": "one concurrent multi-hop QA request",
            "hidden_sf": "supporting title not appearing verbatim in the question",
            "agents_with_shared_hidden_sf": (
                "fraction of agents in a batch whose hidden support title is "
                "also needed by at least one other agent"
            ),
        },
        "summary": summary,
        "batches": batch_metrics,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wikimultihopqa",
                    choices=["2wikimultihopqa"])
    ap.add_argument("--expanded", action="store_true")
    ap.add_argument("--q-type", default=None)
    ap.add_argument("--n-source", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--n-batches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    loader_name = args.dataset
    if args.expanded:
        loader_name = "2wiki_expanded"
    out = run(
        args.dataset, loader_name, args.q_type, args.n_source,
        args.batch_size, args.n_batches, args.seed,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = DATA_DIR / out_path
    else:
        suffix = f"_{args.q_type}" if args.q_type else ""
        src = f"_n{args.n_source}" if args.expanded else ""
        out_path = DATA_DIR / f"batch_overlap_{args.dataset}{suffix}{src}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved %s", out_path)

    s = out["summary"]
    print("\n100-agent shared-cache pressure")
    print(f"dataset={out['dataset']} loader={out['loader']} q_type={out['q_type']} "
          f"queries={out['n_queries']} batches={out['n_batches']}x{out['batch_size']}")
    print(f"agents with shared hidden support: {s['agents_with_shared_hidden_sf']*100:.1f}%")
    print(f"pairwise hidden-support overlap:    {s['pairwise_hidden_sf_overlap']*100:.2f}%")
    print(f"duplicate hidden-support refs:      {s['duplicate_hidden_sf_ref_rate']*100:.1f}%")
    print(f"mean hidden supports per agent:     {s['mean_hidden_sf_per_agent']:.2f}")


if __name__ == "__main__":
    main()
