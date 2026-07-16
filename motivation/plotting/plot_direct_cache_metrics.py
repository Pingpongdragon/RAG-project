#!/usr/bin/env python3
"""Plot ARC-style cache metrics for the query-visible branch."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_figs" / "intro"
STREAMINGQA = (
    ROOT.parent
    / "experiments"
    / "direct"
    / "data"
    / "qdirect_streamingqa_temporal_fig1_metrics_100w50_current.json"
)
TWIKI = (
    ROOT.parent
    / "experiments"
    / "hidden"
    / "data"
    / "qdirect_2wiki_comparison_100w50_kb6250_dense_current.json"
)


LABEL = {
    "LRU": "LRU",
    "FIFO": "FIFO",
    "TinyLFU": "TinyLFU",
    "Proximity": "Proximity",
    "GPTCacheStyle": "GPTCache",
    "DocArrival": "DocArrival",
    "AgentRAGCache_NoHub": "ARC w/o hub",
    "ARC": "ARC",
    "DRIP-QueryVisible": "DRIP-Visible",
    "OnDemandFetch": "OnDemand",
    "Oracle": "Oracle",
}

COLOR = {
    "LRU": "#D97706",
    "FIFO": "#7C3AED",
    "TinyLFU": "#0284C7",
    "Proximity": "#64748B",
    "GPTCacheStyle": "#0891B2",
    "DocArrival": "#059669",
    "AgentRAGCache_NoHub": "#6B7280",
    "ARC": "#111827",
    "DRIP-QueryVisible": "#0F766E",
    "OnDemandFetch": "#17BECF",
    "Oracle": "#DC2626",
}


def _load(path: Path):
    data = json.loads(path.read_text())
    ds = next(iter(data))
    return data[ds]


def _summary_value(summary, strategy, key, default=0.0):
    if strategy not in summary:
        return default
    return summary[strategy].get(key, default)


def _write_table(rows):
    lines = [
        "| Dataset | Strategy | Has-answer | Support cov | Cold/query | R@5 H2 | KB cov H2 | Writes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {strategy} | {has_answer:.1f} | {support:.1f} | "
            "{cold:.2f} | {r5:.1f} | {cov:.1f} | {writes:d} |".format(**row)
        )
    out = OUT_DIR / "direct_query_visible_cache_metrics.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    sq = _load(STREAMINGQA)
    tw = _load(TWIKI)

    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.4, wspace=0.26)
    ax_stream = fig.add_subplot(gs[0, :])
    ax_has = fig.add_subplot(gs[1, 0])
    ax_cost = fig.add_subplot(gs[1, 1])

    stream_strategies = ["ARC", "AgentRAGCache_NoHub", "LRU", "FIFO", "DRIP-QueryVisible", "OnDemandFetch", "Oracle"]
    x = np.arange(1, sq["config"]["n_windows"] + 1)
    for name in stream_strategies:
        values = _summary_value(sq["summary"], name, "has_answer_per_window", [])
        if not values:
            continue
        lw = 2.8 if name in {"DRIP-QueryVisible", "Oracle"} else 1.8
        alpha = 1.0 if name in {"DRIP-QueryVisible", "Oracle", "ARC"} else 0.75
        ax_stream.plot(x, values, color=COLOR[name], lw=lw, alpha=alpha, label=LABEL[name])
    ax_stream.set_title("StreamingQA temporal stream: cache has-answer rate")
    ax_stream.set_ylabel("Has-answer (%)")
    ax_stream.set_xlabel("Window")
    ax_stream.set_ylim(0, 105)
    ax_stream.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax_stream.legend(ncol=4, frameon=False, loc="upper right")

    datasets = [
        ("StreamingQA", sq, ["ARC", "DRIP-QueryVisible", "LRU", "FIFO", "OnDemandFetch", "Oracle"]),
        ("2Wiki direct", tw, ["ARC", "DRIP-QueryVisible", "Oracle"]),
    ]
    rows = []
    bar_labels = []
    has_values = []
    support_values = []
    cold_values = []
    write_values = []
    colors = []
    for ds_label, result, strategies in datasets:
        summary = result["summary"]
        for name in strategies:
            if name not in summary:
                continue
            s = summary[name]
            cov = s.get("cov_h2", s.get("kb_coverage_h2", 0.0))
            row = {
                "dataset": ds_label,
                "strategy": LABEL.get(name, name),
                "has_answer": s.get("has_answer_rate", 0.0),
                "support": s.get("support_coverage_rate", 0.0),
                "cold": s.get("cold_fetches_per_query", 0.0),
                "r5": s.get("recall@5_h2", 0.0),
                "cov": cov,
                "writes": int(s.get("update_cost", 0)),
            }
            rows.append(row)
            bar_labels.append(f"{ds_label}\n{LABEL.get(name, name)}")
            has_values.append(row["has_answer"])
            support_values.append(row["support"])
            cold_values.append(row["cold"])
            write_values.append(row["writes"])
            colors.append(COLOR.get(name, "#888888"))

    xpos = np.arange(len(bar_labels))
    width = 0.38
    ax_has.bar(xpos - width / 2, has_values, width, color=colors, alpha=0.92, label="Has-answer")
    ax_has.bar(xpos + width / 2, support_values, width, color=colors, alpha=0.38, label="Support cov")
    ax_has.set_title("Answerability and support residency")
    ax_has.set_ylabel("%")
    ax_has.set_xticks(xpos)
    ax_has.set_xticklabels(bar_labels, rotation=35, ha="right")
    ax_has.set_ylim(0, 105)
    ax_has.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax_has.legend(frameon=False)

    ax_cost.bar(xpos - width / 2, cold_values, width, color="#475569", alpha=0.85, label="Cold/query")
    ax_cost_2 = ax_cost.twinx()
    ax_cost_2.plot(xpos + width / 2, write_values, color="#B91C1C", marker="o", lw=1.8, label="Writes")
    ax_cost.set_title("Miss pressure and update cost")
    ax_cost.set_ylabel("Cold fetches/query")
    ax_cost_2.set_ylabel("Writes")
    ax_cost.set_xticks(xpos)
    ax_cost.set_xticklabels(bar_labels, rotation=35, ha="right")
    ax_cost.grid(True, axis="y", linestyle=":", alpha=0.3)
    h1, l1 = ax_cost.get_legend_handles_labels()
    h2, l2 = ax_cost_2.get_legend_handles_labels()
    ax_cost.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.suptitle("Query-visible cache management under ARC-style metrics", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf = OUT_DIR / "fig_direct_query_visible_cache_metrics.pdf"
    png = OUT_DIR / "fig_direct_query_visible_cache_metrics.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    table = _write_table(rows)
    print(f"saved {pdf}")
    print(f"saved {png}")
    print(f"saved {table}")


if __name__ == "__main__":
    main()
