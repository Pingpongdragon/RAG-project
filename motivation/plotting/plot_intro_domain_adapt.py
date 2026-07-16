"""Draw Motivation Figure 2: the target streams are both drifting and reusable.

Panel (a) reports repeated evidence within the active domain or regime.  MIND
uses the stricter cross-user definition: an exact news item was previously
clicked by another user.  SQuAD uses the seed-42 controlled recurring audit.
WoW is the mean over seeds 42--46 of the controlled recurring-domain traces.
Panel (b) holds the SQuAD pool and budgets fixed and changes only request order.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "motivation" / "paper_figs" / "intro"


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.45))

    # (a) Reuse within the active domain/regime at each trace's cache object.
    datasets = ["MIND (natural users)", "SQuAD (controlled)",
                "WoW (controlled)"]
    reuse = [93.58, 79.8, 44.8]
    colors = ["#2B8CBE", "#7BCCC4", "#A8DDB5"]
    ax = axes[0]
    positions = np.arange(len(reuse))[::-1]
    bars = ax.barh(positions, reuse, color=colors, height=0.66)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Prior in-domain evidence (%)")
    ax.set_yticks(positions, datasets)
    ax.set_title("(a) Within-domain evidence reuse", loc="left")
    for bar, value in zip(bars, reuse):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", ha="left", va="center", fontsize=8.5)

    # (b) Same pool and marginal support families; only regime order changes.
    methods = ["LRU", "ARC", "AgentRAGCache"]
    # Causal-v2 seed-42 audit: current query embeddings are routed before
    # scoring; gold evidence embeddings are never used as route inputs.
    stationary = [76.2, 74.8, 43.4]
    recurring = [22.0, 19.2, 18.4]
    x = np.arange(len(methods))
    width = 0.36
    ax = axes[1]
    ax.bar(x - width / 2, stationary, width, label="stationary",
           color="#74A9CF")
    ax.bar(x + width / 2, recurring, width, label="recurring drift",
           color="#F16913")
    ax.set_ylim(0, 84)
    ax.set_ylabel("Evidence hit (%)")
    ax.set_xticks(x, ["LRU", "ARC", "Agent\nRAGCache"])
    ax.set_title("(b) Drift creates residency mismatch", loc="left")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout(h_pad=1.1)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT / f"fig2_cacheable_domain_drift.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
