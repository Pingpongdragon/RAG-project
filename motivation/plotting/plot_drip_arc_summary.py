"""One-slide summary figure for DRIP vs ARC discussion.

The figure intentionally combines two result families:
  1. StreamingQA temporal drift: ARC collapses after era changes; DRIP-Dense
     and DRIP-ESC-Lease recover because they admit documents from query failures.
  2. 2Wiki bridge-comparison: DRIP-ESC shows the bridge signal; DRIP-ESC-Lease
     adds pair-lease retention.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
M1 = ROOT.parent / "experiments" / "direct" / "data"
M2 = ROOT.parent / "experiments" / "hidden" / "data"
OUT = ROOT / "paper_figs" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "LRU": "#B45309",
    "FIFO": "#8B5CF6",
    "AgentRAGCache": "#111827",
    "DRIP-Dense": "#10B981",
    "DRIP-ESC": "#059669",
    "DRIP-ESC-Lease": "#2563EB",
    "Oracle": "#DC2626",
}

LABELS = {
    "AgentRAGCache": "ARC",
    "DRIP-Dense": "DRIP-Dense",
    "DRIP-ESC": "DRIP-ESC",
    "DRIP-ESC-Lease": "DRIP-ESC-Lease",
}

LEGACY_STRATEGY_ALIASES = {
    "".join(("Query", "Driven")): "DRIP-Dense",
    "".join(("Routed", "Cache")): "DRIP-ESC",
    "DRIP": "DRIP-ESC-Lease",
}


def load(path: Path):
    data = json.load(open(path))
    return data[list(data.keys())[0]]


def normalize_strategy_names(summary):
    for old, new in LEGACY_STRATEGY_ALIASES.items():
        if new not in summary and old in summary:
            summary[new] = summary[old]
    return summary


def smooth(values, w=5):
    arr = np.asarray(values, dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.concatenate([np.full(w - 1, arr[0]), arr])
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


temporal_base = load(M1 / "results_streamingqa_temporal_100w_arc_fifo_lru.json")
temporal_drip = load(M1 / "results_streamingqa_temporal_drip_final.json")
bridge = load(M2 / "results_50w_2wiki_bridge_dual_drip.json")

temporal = dict(temporal_base["summary"])
temporal.update(temporal_drip["summary"])
temporal = normalize_strategy_names(temporal)
bridge_summary = normalize_strategy_names(bridge["summary"])

fig = plt.figure(figsize=(13.6, 7.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], height_ratios=[1, 1],
                      hspace=0.36, wspace=0.32)
ax_line = fig.add_subplot(gs[:, 0])
ax_temp = fig.add_subplot(gs[0, 1])
ax_bridge = fig.add_subplot(gs[1, 1])

# Panel A: temporal per-window recovery.
line_order = ["LRU", "AgentRAGCache", "DRIP-Dense", "DRIP-ESC-Lease", "Oracle"]
for name in line_order:
    if name not in temporal:
        continue
    y = smooth(temporal[name]["recall@5_per_window"], w=5)
    ax_line.plot(
        np.arange(len(y)), y,
        label=LABELS.get(name, name),
        color=COLORS[name],
        lw=2.6 if name in {"DRIP-Dense", "DRIP-ESC-Lease", "Oracle"} else 2.0,
        alpha=0.95,
    )

cfg = temporal_base["config"]
era = cfg["n_windows"] // 5
era_labels = ["2008-10", "2011-13", "2014-16", "2017-18", "2019-20"]
for i, lab in enumerate(era_labels):
    x0, x1 = i * era, (i + 1) * era
    if i % 2 == 1:
        ax_line.axvspan(x0, x1, color="#E5E7EB", alpha=0.55, zorder=0)
    ax_line.text((x0 + x1) / 2, 97, lab, ha="center", va="top",
                 fontsize=8, color="#374151")

ax_line.set_title("(a) Temporal drift: per-window Recall@5", loc="left", fontsize=12)
ax_line.set_xlabel("Window index (StreamingQA temporal stream)")
ax_line.set_ylabel("Recall@5 (%)")
ax_line.set_xlim(0, cfg["n_windows"] - 1)
ax_line.set_ylim(0, 100)
ax_line.grid(alpha=0.25)
ax_line.legend(ncol=3, loc="lower left", frameon=True, framealpha=0.92)

# Panel B: temporal aggregate quality and writes.
temp_order = ["LRU", "AgentRAGCache", "DRIP-Dense", "DRIP-ESC-Lease", "Oracle"]
x = np.arange(len(temp_order))
width = 0.34
r5 = [temporal[m]["recall@5_h2"] for m in temp_order]
cov = [temporal[m]["cov_h2"] for m in temp_order]
writes = [temporal[m]["update_cost"] for m in temp_order]
ax_temp.bar(x - width / 2, r5, width, label="R@5 H2", color="#60A5FA",
            edgecolor="#111827", linewidth=0.5)
ax_temp.bar(x + width / 2, cov, width, label="Coverage H2", color="#34D399",
            edgecolor="#111827", linewidth=0.5)
for xi, w in zip(x, writes):
    ax_temp.text(xi, 3, f"W={w/1000:.1f}k", rotation=90, ha="center",
                 va="bottom", fontsize=7, color="#374151")
ax_temp.set_xticks(x)
ax_temp.set_xticklabels([LABELS.get(m, m) for m in temp_order], rotation=18, ha="right")
ax_temp.set_ylim(0, 105)
ax_temp.set_ylabel("Percent")
ax_temp.set_title("(b) StreamingQA temporal aggregate", loc="left", fontsize=12)
ax_temp.grid(axis="y", alpha=0.25)
ax_temp.legend(fontsize=8, loc="upper left")

# Panel C: bridge H2 quality vs write cost.
bridge_order = ["DRIP-Dense", "DRIP-ESC", "DRIP-ESC-Lease", "AgentRAGCache", "Oracle"]
x2 = np.arange(len(bridge_order))
width = 0.34
r5b = [bridge_summary[m]["recall@5_h2"] for m in bridge_order]
covb = [bridge_summary[m]["kb_coverage_h2"] for m in bridge_order]
writeb = [bridge_summary[m]["update_cost"] for m in bridge_order]
ax_bridge.bar(x2 - width / 2, r5b, width, label="R@5 H2", color="#93C5FD",
              edgecolor="#111827", linewidth=0.5)
ax_bridge.bar(x2 + width / 2, covb, width, label="Coverage H2", color="#6EE7B7",
              edgecolor="#111827", linewidth=0.5)
for xi, w in zip(x2, writeb):
    ax_bridge.text(xi, 3, f"W={w/1000:.1f}k", rotation=90, ha="center",
                   va="bottom", fontsize=7, color="#374151")
ax_bridge.set_xticks(x2)
ax_bridge.set_xticklabels([LABELS.get(m, m) for m in bridge_order], rotation=18, ha="right")
ax_bridge.set_ylim(0, 105)
ax_bridge.set_ylabel("Percent")
ax_bridge.set_title("(c) 2Wiki bridge-comparison aggregate", loc="left", fontsize=12)
ax_bridge.grid(axis="y", alpha=0.25)
ax_bridge.legend(fontsize=8, loc="upper left")

fig.suptitle("DRIP vs ARC: temporal drift and multi-hop bridge behavior",
             fontsize=14, y=0.985)
fig.text(
    0.02, 0.015,
    "Notes: H2 is the post-drift/new-era segment. W = total cache writes. "
    "Temporal uses StreamingQA 100 windows; bridge uses 2Wiki bridge-comparison 50 windows.",
    fontsize=8.5,
    color="#374151",
)

png = OUT / "fig_drip_arc_summary.png"
pdf = OUT / "fig_drip_arc_summary.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(png)
print(pdf)
