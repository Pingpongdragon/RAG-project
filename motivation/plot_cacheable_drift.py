#!/usr/bin/env python3
"""Build the paper motivation figure for the ARC -> shift -> DRIP chain.

The figure keeps three claims separate:

1. AgentRAGCache (ARC) established that a compact cache is useful when each
   benchmark is evaluated under a fixed query distribution.
2. On the same SQuAD source pool and cache budget, changing the evidence-domain
   mixture creates a residency mismatch for static/history-based caches.
3. Under the same hard capacity, per-window write cap, and cold-read accounting,
   DRIP-TopicState recovers hit rate on recurring evidence-domain shift.

Panel (a) uses the bge-small-en numbers reported in Table 1 of
"Cache Mechanism for Agent RAG Systems" (arXiv:2511.02919). Panels (b,c) read
the five held-out SQuAD seeds from experiments/hidden/data; no result value is
hard-coded for our methods.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "experiments" / "hidden" / "data"

ARC_PUBLISHED_HAS_ANSWER = {
    "MMLU": 62.63,
    "AdversarialQA": 71.18,
    "SQuAD": 79.80,
}
SEEDS = (42, 43, 44, 45, 46)
RECURRING_TEMPLATE = "formal_squad_disjoint_b32_w4_m32_test_s{seed}.json"
STATIONARY_TEMPLATE = (
    "control_squad_disjoint_factorized_stationary_"
    "b32_w4_m32_test_s{seed}.json"
)


def _load(template: str) -> list[dict]:
    payloads = []
    for seed in SEEDS:
        path = RESULT_DIR / template.format(seed=seed)
        if not path.exists():
            raise FileNotFoundError(f"missing formal result: {path}")
        payload = json.loads(path.read_text())
        protocol = payload["protocol"]
        if protocol["cache_size"] != 32 or protocol["write_budget"] != 4:
            raise ValueError(f"unexpected hard budget in {path}")
        if protocol["constructor"]["exact_query_duplicates"] != 0:
            raise ValueError(f"exact-query duplication in {path}")
        payloads.append(payload)
    return payloads


def _metric(payloads: list[dict], method: str, field: str, scale=1.0):
    return np.asarray(
        [float(payload["summary"][method][field]) * scale for payload in payloads],
        dtype=np.float64,
    )


def load_data() -> dict:
    recurring = _load(RECURRING_TEMPLATE)
    stationary = _load(STATIONARY_TEMPLATE)
    baseline_methods = ("LRU", "AgentRAGCache", "ClassicalARC")
    pareto_methods = ("LRU", "ClassicalARC", "AgentRAGCache", "DRIP-TopicState")

    return {
        "recurring": recurring,
        "stationary": stationary,
        "baseline_methods": baseline_methods,
        "pareto_methods": pareto_methods,
        "stationary_hit": {
            method: _metric(stationary, method, "strict_query_hit_rate", 100.0)
            for method in baseline_methods
        },
        "recurring_hit": {
            method: _metric(recurring, method, "strict_query_hit_rate", 100.0)
            for method in set(baseline_methods) | set(pareto_methods)
        },
        "recurring_reads": {
            method: _metric(recurring, method, "total_cold_store_reads")
            for method in pareto_methods
        },
        "recurring_writes": {
            method: _metric(recurring, method, "cache_writes")
            for method in pareto_methods
        },
    }


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def plot(data: dict, output: Path, overleaf_output: Path | None) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.58))
    ax_a, ax_b, ax_c = axes

    # (a) Published static-domain cache utility.
    names = list(ARC_PUBLISHED_HAS_ANSWER)
    values = [ARC_PUBLISHED_HAS_ANSWER[name] for name in names]
    bars = ax_a.bar(
        np.arange(len(names)), values, width=0.64,
        color=["#93C5FD", "#60A5FA", "#2563EB"],
        edgecolor="white", linewidth=0.6,
    )
    display_names = ("MMLU", "AdvQA", "SQuAD")
    ax_a.set_xticks(np.arange(len(names)), display_names, rotation=12, ha="right")
    ax_a.set_ylim(0, 100)
    ax_a.set_ylabel("Has-answer rate (%)")
    ax_a.set_title(
        "(a) Fixed-domain cache value",
        loc="left", fontweight="bold", fontsize=8.8,
    )
    ax_a.text(
        0.02, 0.96, "AgentRAGCache, published",
        transform=ax_a.transAxes, va="top", color="#475569", fontsize=7.4,
    )
    for bar, value in zip(bars, values):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}",
            ha="center", va="bottom", fontsize=7.6,
        )

    # (b) Same source pool/budget, stationary vs recurring evidence domains.
    labels = ("LRU", "AgentRAG\nCache", "Classical\nARC")
    methods = data["baseline_methods"]
    x = np.arange(len(methods), dtype=float)
    width = 0.34
    stationary_stats = [_mean_std(data["stationary_hit"][m]) for m in methods]
    recurring_stats = [_mean_std(data["recurring_hit"][m]) for m in methods]
    ax_b.bar(
        x - width / 2,
        [item[0] for item in stationary_stats],
        yerr=[item[1] for item in stationary_stats],
        width=width,
        color="#A7F3D0",
        edgecolor="white",
        capsize=2,
        label="Stationary mixture",
    )
    ax_b.bar(
        x + width / 2,
        [item[0] for item in recurring_stats],
        yerr=[item[1] for item in recurring_stats],
        width=width,
        color="#FCA5A5",
        edgecolor="white",
        capsize=2,
        label="Recurring shift",
    )
    ax_b.set_xticks(x, labels)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("Strict hit rate (%)")
    ax_b.set_title(
        "(b) Shift-induced mismatch",
        loc="left", fontweight="bold", fontsize=8.8,
    )
    ax_b.legend(frameon=False, fontsize=6.9, loc="lower right")
    ax_b.text(
        0.02, 0.96, "same SQuAD pool, B=32, W=4",
        transform=ax_b.transAxes, va="top", color="#475569", fontsize=7.4,
    )

    # (c) Quality/read trade-off under the recurring shift.
    colors = {
        "LRU": "#D97706",
        "ClassicalARC": "#7C3AED",
        "AgentRAGCache": "#111827",
        "DRIP-TopicState": "#0F766E",
    }
    labels = {
        "LRU": "LRU",
        "ClassicalARC": "Classical ARC",
        "AgentRAGCache": "AgentRAGCache",
        "DRIP-TopicState": "DRIP-TopicState",
    }
    label_positions = {
        "LRU": (350.0, 29.2),
        "ClassicalARC": (363.0, 21.2),
        "AgentRAGCache": (350.0, 25.8),
        "DRIP-TopicState": (358.2, 34.1),
    }
    for method in data["pareto_methods"]:
        hit_mean, hit_std = _mean_std(data["recurring_hit"][method])
        read_mean, read_std = _mean_std(data["recurring_reads"][method])
        write_mean = float(np.mean(data["recurring_writes"][method]))
        ax_c.errorbar(
            read_mean,
            hit_mean,
            xerr=read_std,
            yerr=hit_std,
            fmt="o",
            markersize=5.8 if method == "DRIP-TopicState" else 4.8,
            color=colors[method],
            ecolor=colors[method],
            elinewidth=0.9,
            capsize=2,
            zorder=3,
        )
        label_x, label_y = label_positions[method]
        ax_c.annotate(
            f"{labels[method]} ({write_mean:.1f} W)",
            (read_mean, hit_mean),
            xytext=(label_x, label_y),
            fontsize=6.8,
            color=colors[method],
            arrowprops={"arrowstyle": "-", "color": colors[method], "lw": 0.55},
        )
    ax_c.set_xlim(348, 393)
    ax_c.set_ylim(20, 36)
    ax_c.set_xlabel("Total cold-store reads")
    ax_c.set_ylabel("Strict hit rate (%)")
    ax_c.set_title(
        "(c) Budget-matched recovery",
        loc="left", fontweight="bold", fontsize=8.8,
    )
    ax_c.grid(True, linestyle=":", alpha=0.3)

    for axis in axes:
        axis.yaxis.grid(True, linestyle=":", alpha=0.25, zorder=0)
        axis.set_axisbelow(True)

    fig.tight_layout(w_pad=1.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    if overleaf_output is not None:
        overleaf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(overleaf_output, bbox_inches="tight")
    plt.close(fig)


def write_table(data: dict, output: Path) -> None:
    lines = [
        "# Motivation figure data",
        "",
        "Panel (a) reproduces Table 1 (bge-small-en) from arXiv:2511.02919.",
        "Panels (b,c) aggregate five held-out SQuAD construction seeds (42--46).",
        "All local methods use cache size B=32 and per-window write cap W=4.",
        "",
        "| Panel | Method / dataset | Condition | Mean | Std. | Reads | Writes |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset, value in ARC_PUBLISHED_HAS_ANSWER.items():
        lines.append(
            f"| a | AgentRAGCache / {dataset} | fixed benchmark | "
            f"{value:.2f} | -- | -- | -- |"
        )
    for method in data["baseline_methods"]:
        for condition, values in (
            ("stationary", data["stationary_hit"][method]),
            ("recurring", data["recurring_hit"][method]),
        ):
            mean, std = _mean_std(values)
            lines.append(
                f"| b | {method} | {condition} | {mean:.2f} | {std:.2f} | -- | -- |"
            )
    for method in data["pareto_methods"]:
        hit_mean, hit_std = _mean_std(data["recurring_hit"][method])
        read_mean = float(np.mean(data["recurring_reads"][method]))
        write_mean = float(np.mean(data["recurring_writes"][method]))
        lines.append(
            f"| c | {method} | recurring | {hit_mean:.2f} | {hit_std:.2f} | "
            f"{read_mean:.1f} | {write_mean:.1f} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR
        / "motivation/paper_figs/intro/fig1_intro_arc_cross_domain_chain",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=PROJECT_DIR
        / "motivation/paper_figs/intro/fig1_intro_arc_cross_domain_chain.md",
    )
    parser.add_argument(
        "--overleaf-output",
        type=Path,
        default=PROJECT_DIR
        / "overleaf-paper/motivation/fig1_intro_arc_cross_domain_chain.pdf",
    )
    args = parser.parse_args()

    data = load_data()
    plot(data, args.output, args.overleaf_output)
    write_table(data, args.table)
    print(f"WROTE {args.output.with_suffix('.pdf')}")
    print(f"WROTE {args.output.with_suffix('.png')}")
    print(f"WROTE {args.table}")
    print(f"WROTE {args.overleaf_output}")


if __name__ == "__main__":
    main()
