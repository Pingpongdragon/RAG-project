#!/usr/bin/env python3
"""从多个 runner JSON 生成 causal cache-ratio 四指标曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "LRU": ("#D97706", "v", "-."),
    "FIFO": ("#7C3AED", "^", "--"),
    "TinyLFU": ("#0284C7", "p", ":"),
    "GPTCacheStyle": ("#0891B2", "P", ":"),
    "Proximity": ("#65A30D", "h", "--"),
    "AgentRAGCache": ("#111827", "o", "-"),
    "DRIP": ("#0F766E", "*", "-"),
}

LABELS = {
    "GPTCacheStyle": "GPTCache-style",
    "AgentRAGCache": "ARC",
    "DRIP": "DRIP-Reactive",
}


def load_points(paths, dataset):
    points_by_ratio = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        result = payload[dataset]
        config = result["config"]
        if config.get("init_mode") != "causal-prefix":
            raise ValueError(f"{path} is not a causal-prefix run")
        ratio = 100.0 * float(config["kb_pool_ratio"])
        n_queries = int(config["n_windows"]) * int(config["window_size"])
        for method, summary in result["summary"].items():
            recall = np.mean(summary["recall@5_per_window"])
            replacement_rate = summary.get("replacement_rate_per_query")
            if replacement_rate is None:
                replacement_rate = (
                    float(summary["replacement_count"]) / max(1, n_queries)
                )
            points_by_ratio.setdefault(method, {})[ratio] = {
                "ratio": ratio,
                "has_answer": float(summary["has_answer_rate"]),
                "amat": float(summary["amat"]),
                "recall": float(recall),
                "replacement_per_1k": 1000.0 * float(replacement_rate),
                "replacement_count": int(summary["replacement_count"]),
                "churn_pct": float(summary.get("cache_churn_rate_pct", 0.0)),
            }
    return {
        method: [by_ratio[key] for key in sorted(by_ratio)]
        for method, by_ratio in points_by_ratio.items()
    }


def write_table(points, output_path):
    """Write the exact values behind the figure as a Markdown table."""

    lines = [
        "| Ratio | Method | Has-Answer | AMAT | Recall@5 | Repl./1K | Replacements |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for method, values in points.items():
        for item in values:
            rows.append((item["ratio"], method, item))
    for ratio, method, item in sorted(rows):
        lines.append(
            f"| {ratio:.2f}% | {LABELS.get(method, method)} | "
            f"{item['has_answer']:.2f} | {item['amat']:.3f} | "
            f"{item['recall']:.2f} | {item['replacement_per_1k']:.2f} | "
            f"{item['replacement_count']} |"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def plot(points, output_prefix, title):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    panels = [
        ("has_answer", "Has-Answer Rate (%)", "higher"),
        ("amat", "AMAT", "lower"),
        ("recall", "Recall@5 (%)", "higher"),
        ("replacement_per_1k", "Replacements / 1K queries", "lower"),
    ]
    for method, values in points.items():
        color, marker, linestyle = STYLES.get(
            method, ("#4B5563", "o", "-")
        )
        x = [item["ratio"] for item in values]
        for axis, (key, ylabel, direction) in zip(axes.flat, panels):
            axis.plot(
                x,
                [item[key] for item in values],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=6,
                label=LABELS.get(method, method),
            )
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", linestyle=":", alpha=0.35)
            axis.text(
                0.98, 0.94, direction,
                transform=axis.transAxes,
                ha="right", va="top", fontsize=8, color="#4B5563")
    for axis in axes[-1]:
        axis.set_xlabel("Hot-cache / cold-pool ratio (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Causal cache-ratio sweep")
    parser.add_argument("--table", default=None)
    args = parser.parse_args()
    points = load_points(args.paths, args.dataset)
    plot(points, args.output, args.title)
    if args.table:
        write_table(points, args.table)


if __name__ == "__main__":
    main()
