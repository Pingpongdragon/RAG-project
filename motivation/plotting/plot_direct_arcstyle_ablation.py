#!/usr/bin/env python3
"""ARC-style direct-branch plots with cache-size ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figs" / "intro"

DATASETS = {
    "StreamingQA": {
        200: ROOT / "motivation_1/data/qdirect_streamingqa_arcstyle_100w50_kb200_current.json",
        400: ROOT / "motivation_1/data/qdirect_streamingqa_arcstyle_100w50_kb400_current.json",
        800: ROOT / "motivation_1/data/qdirect_streamingqa_arcstyle_100w50_kb800_current.json",
    },
    "2Wiki": {
        1250: ROOT / "motivation_2/data/qdirect_2wiki_comparison_arcstyle_100w50_kb1250_dense_current.json",
        2500: ROOT / "motivation_2/data/qdirect_2wiki_comparison_arcstyle_100w50_kb2500_dense_current.json",
        6250: ROOT / "motivation_2/data/qdirect_2wiki_comparison_allbaselines_100w50_kb6250_dense_current.json",
    },
}

MAIN_KB = {"StreamingQA": 400, "2Wiki": 6250}

STRATEGIES = [
    "LRU",
    "FIFO",
    "TinyLFU",
    "Proximity",
    "GPTCacheStyle",
    "AgentRAGCache_NoHub",
    "ARC",
    "DRIP",
    "OnDemandFetch",
]

ABLATION_STRATEGIES = ["LRU", "FIFO", "TinyLFU", "ARC", "DRIP", "OnDemandFetch"]

LABEL = {
    "LRU": "LRU",
    "FIFO": "FIFO",
    "TinyLFU": "LFU",
    "Proximity": "Proximity",
    "GPTCacheStyle": "GPTCache",
    "AgentRAGCache_NoHub": "ARC w/o hub",
    "ARC": "ARC",
    "DRIP": "DRIP",
    "OnDemandFetch": "OnDemand",
}

COLOR = {
    "LRU": "#D97706",
    "FIFO": "#7C3AED",
    "TinyLFU": "#0284C7",
    "Proximity": "#64748B",
    "GPTCacheStyle": "#0891B2",
    "AgentRAGCache_NoHub": "#6B7280",
    "ARC": "#111827",
    "DRIP": "#0F766E",
    "OnDemandFetch": "#DC2626",
}


def _load(path: Path):
    data = json.loads(path.read_text())
    ds = next(iter(data))
    return data[ds]


def _metric(result, strategy, key, default=0.0):
    return result["summary"].get(strategy, {}).get(key, default)


def _kb_cov_h2(result, strategy):
    summary = result["summary"].get(strategy, {})
    return summary.get("kb_coverage_h2", summary.get("cov_h2", 0.0))


def _all_rows(results_by_dataset):
    rows = []
    for dataset, by_kb in results_by_dataset.items():
        for kb, result in by_kb.items():
            for strategy in STRATEGIES:
                if strategy not in result["summary"]:
                    continue
                s = result["summary"][strategy]
                rows.append(
                    {
                        "dataset": dataset,
                        "kb": kb,
                        "strategy": LABEL[strategy],
                        "has": s.get("has_answer_rate", 0.0),
                        "support": s.get("support_coverage_rate", 0.0),
                        "cold": s.get("cold_fetches_per_query", 0.0),
                        "r5h2": s.get("recall@5_h2", 0.0),
                        "covh2": _kb_cov_h2(result, strategy),
                        "writes": int(s.get("update_cost", 0)),
                        "serve": int(s.get("serve_retrieval_cost", 0)),
                    }
                )
    return rows


def _write_table(rows):
    lines = [
        "| Dataset | KB | Strategy | Hot Has-answer | SupportCov | ColdQ | R@5 H2 | KB Cov H2 | Writes | ServeR |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {dataset} | {kb} | {strategy} | {has:.1f} | {support:.1f} | "
            "{cold:.2f} | {r5h2:.1f} | {covh2:.1f} | {writes:d} | {serve:d} |".format(**r)
        )
    out = OUT_DIR / "direct_arcstyle_cache_ablation.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def _plot_main_panel(ax, dataset, result):
    x = np.arange(len(STRATEGIES))
    has = [_metric(result, s, "has_answer_rate") for s in STRATEGIES]
    r5 = [_metric(result, s, "recall@5_h2") for s in STRATEGIES]
    colors = [COLOR[s] for s in STRATEGIES]
    ax.bar(x, has, color=colors, alpha=0.86, label="hot-cache HasAnswer")
    ax.scatter(x, r5, color="#111827", marker="D", s=26, zorder=5, label="R@5 H2")
    ax.set_title(f"{dataset}, KB={result['config']['kb_budget']}")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in STRATEGIES], rotation=32, ha="right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")


def _plot_ablation_panel(ax, dataset, by_kb):
    kbs = sorted(by_kb)
    for strategy in ABLATION_STRATEGIES:
        values = [_metric(by_kb[kb], strategy, "has_answer_rate") for kb in kbs]
        lw = 2.8 if strategy == "DRIP" else 1.8
        marker = "*" if strategy == "DRIP" else "o"
        ax.plot(
            kbs,
            values,
            color=COLOR[strategy],
            marker=marker,
            lw=lw,
            label=LABEL[strategy],
        )
    ax.set_title(f"{dataset}: cache-size ablation")
    ax.set_xlabel("KB capacity (documents)")
    ax.set_ylabel("Hot-cache HasAnswer (%)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.set_ylim(0, 60 if dataset == "StreamingQA" else 35)
    ax.legend(frameon=False, ncol=2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        dataset: {kb: _load(path) for kb, path in by_kb.items()}
        for dataset, by_kb in DATASETS.items()
    }
    rows = _all_rows(results)
    table = _write_table(rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), constrained_layout=True)
    for col, dataset in enumerate(["StreamingQA", "2Wiki"]):
        _plot_main_panel(axes[0, col], dataset, results[dataset][MAIN_KB[dataset]])
        _plot_ablation_panel(axes[1, col], dataset, results[dataset])

    fig.suptitle(
        "Query-visible cache management under ARC-style metrics",
        fontsize=14,
    )
    pdf = OUT_DIR / "fig_direct_arcstyle_cache_ablation.pdf"
    png = OUT_DIR / "fig_direct_arcstyle_cache_ablation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"saved {pdf}")
    print(f"saved {png}")
    print(f"saved {table}")


if __name__ == "__main__":
    main()
