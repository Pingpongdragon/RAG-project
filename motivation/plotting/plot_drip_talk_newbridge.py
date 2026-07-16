"""Talk figure for the current DRIP ablation rerun.

This intentionally does not overwrite the paper Fig.2.  It uses the newbridge
rerun files and frames the current story for a progress talk:

  - HotpotQA comparison: DRIP-ESC-Lease helps direct multi-hop evidence.
  - 2Wiki bridge: current graph bridge/admission is not calibrated yet.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA2 = ROOT.parent / "experiments" / "hidden" / "data"
OUT = ROOT / "paper_figs" / "intro"
OUT.mkdir(parents=True, exist_ok=True)

HOTPOT_FILE = DATA2 / "results_hotpotqa_comp_full_gradual_q2k_newbridge.json"
WIKI_FILE = DATA2 / "results_2wiki_bc_entity_expand_gradual_q2k_newbridge.json"

COLOR = {
    "LRU": "#B45309",
    "TinyLFU": "#D97706",
    "Proximity": "#6366F1",
    "GPTCacheStyle": "#0891B2",
    "DocArrival": "#7C3AED",
    "AgentRAGCache": "#111827",
    "DRIP-Dense": "#10B981",
    "DRIP-ESC-Lease": "#2563EB",
    "OnDemandFetch": "#0F766E",
    "Oracle": "#DC2626",
}
LABEL = {
    "LRU": "LRU",
    "TinyLFU": "LFU",
    "Proximity": "Proximity",
    "GPTCacheStyle": "GPTCache",
    "DocArrival": "IndexGrowth",
    "AgentRAGCache": "ARC",
    "DRIP-Dense": "DRIP-Dense",
    "DRIP-ESC-Lease": "DRIP-ESC-Lease",
    "OnDemandFetch": "On-Demand",
    "Oracle": "Oracle",
}
LW = {
    "LRU": 1.35,
    "TinyLFU": 1.25,
    "Proximity": 1.25,
    "GPTCacheStyle": 1.25,
    "DocArrival": 1.25,
    "AgentRAGCache": 2.1,
    "DRIP-Dense": 2.3,
    "DRIP-ESC-Lease": 2.8,
    "OnDemandFetch": 2.0,
    "Oracle": 2.2,
}
LS = {
    "LRU": ":",
    "TinyLFU": ":",
    "Proximity": "-.",
    "GPTCacheStyle": ":",
    "DocArrival": "--",
    "AgentRAGCache": "-.",
    "DRIP-Dense": "-",
    "DRIP-ESC-Lease": "-",
    "OnDemandFetch": "--",
    "Oracle": "-",
}
ORDER = [
    "LRU",
    "TinyLFU",
    "Proximity",
    "GPTCacheStyle",
    "DocArrival",
    "AgentRAGCache",
    "DRIP-Dense",
    "DRIP-ESC-Lease",
    "OnDemandFetch",
    "Oracle",
]

LEGACY_STRATEGY_ALIASES = {
    "".join(("Query", "Driven")): "DRIP-Dense",
    "DRIP": "DRIP-ESC-Lease",
}


def normalize_strategy_names(summary):
    for old, new in LEGACY_STRATEGY_ALIASES.items():
        if new not in summary and old in summary:
            summary[new] = summary[old]
    return summary


def smooth_centered(values, w=7):
    a = np.array(values, dtype=float)
    if len(a) < w:
        return a
    c = np.convolve(a, np.ones(w) / w, mode="valid")
    pad_l = (len(a) - len(c)) // 2
    pad_r = len(a) - len(c) - pad_l
    return np.pad(c, (pad_l, pad_r), mode="edge")


def add_gradual_bg(ax, n_w, y_max):
    left = np.array([219, 234, 254], dtype=float) / 255.0
    right = np.array([254, 243, 199], dtype=float) / 255.0
    xs = np.linspace(0.0, 1.0, 512)
    grad = np.zeros((2, xs.size, 3), dtype=float)
    for ci in range(3):
        grad[:, :, ci] = left[ci] * (1 - xs) + right[ci] * xs
    ax.imshow(
        grad,
        extent=[0, n_w - 1, 0, y_max],
        aspect="auto",
        origin="lower",
        alpha=0.55,
        zorder=0,
    )
    ax.text(7, y_max * 0.96, "head topics", ha="left", va="top",
            fontsize=7.5, color="#1E40AF", style="italic")
    ax.text(n_w / 2, y_max * 0.96, "full gradual drift", ha="center",
            va="top", fontsize=7.5, color="#475569", style="italic")
    ax.text(n_w - 7, y_max * 0.96, "tail topics", ha="right", va="top",
            fontsize=7.5, color="#92400E", style="italic")


def route_diag(summary):
    drip = summary["DRIP-ESC-Lease"]
    routes = drip.get("route_log", [])
    out = {k: sum(x.get("routes", {}).get(k, 0) for x in routes)
           for k in ("SINGLE", "MULTI_DIRECT", "BRIDGE")}
    matched = sum(x.get("route_match", 0) for x in routes)
    labeled = sum(x.get("route_labeled", 0) for x in routes)
    return out, matched, labeled


def bridge_diag(summary):
    b = summary["DRIP-ESC-Lease"].get("bridge_log", [])
    total = sum(x.get("bridge_updates", 0) for x in b)
    gold = sum(x.get("bridge_gold_updates", 0) for x in b)
    direct_total = sum(x.get("bridge_direct_updates", 0) for x in b)
    direct_gold = sum(x.get("bridge_direct_gold_updates", 0) for x in b)
    rate = 100.0 * gold / max(total, 1)
    direct_rate = 100.0 * direct_gold / max(direct_total, 1)
    return gold, total, rate, direct_gold, direct_total, direct_rate


def draw_panel(ax, data, panel, title, note, note_color, note_bg, y_min=0):
    cfg = data["config"]
    summ = data["summary"]
    names = [n for n in ORDER if n in summ]
    wx = np.arange(cfg["n_windows"])
    all_vals = []
    for name in names:
        all_vals.extend(summ[name].get("recall@5_per_window", []))
    y_max = max(60, min(100, max(all_vals) * 1.15))
    add_gradual_bg(ax, cfg["n_windows"], y_max)

    for name in names:
        vals = summ[name].get("recall@5_per_window", [])
        alpha = 1.0 if name in {"DRIP-ESC-Lease", "DRIP-Dense", "AgentRAGCache", "Oracle", "OnDemandFetch"} else 0.62
        z = 5 if name in {"DRIP-ESC-Lease", "DRIP-Dense", "AgentRAGCache"} else 3
        ax.plot(
            wx,
            smooth_centered(vals),
            color=COLOR.get(name, "#888888"),
            lw=LW.get(name, 1.3),
            ls=LS.get(name, "-"),
            alpha=alpha,
            label=LABEL.get(name, name),
            zorder=z,
        )

    ax.text(
        0.04,
        0.055,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=note_color,
        bbox=dict(fc=note_bg, ec=note_color, boxstyle="round,pad=0.38", alpha=0.94),
    )
    ax.set_xlim(0, cfg["n_windows"] - 1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Window index", fontsize=9.5)
    ax.set_ylabel("Recall@5 (%)", fontsize=10)
    ax.set_title(
        f"$\\mathbf{{({panel})}}$  {title}\n"
        f"pool={cfg['pool_size']:,}   KB={cfg['kb_budget']:,}   stream={cfg['n_windows']}x{cfg['window_size']}",
        fontsize=10.2,
    )
    ax.grid(alpha=0.22, zorder=1)


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    hot = json.load(open(HOTPOT_FILE))["hotpotqa"]
    wiki = json.load(open(WIKI_FILE))["2wikimultihopqa"]
    hs = normalize_strategy_names(hot["summary"])
    ws = normalize_strategy_names(wiki["summary"])
    hot["summary"] = hs
    wiki["summary"] = ws

    h_routes, h_match, h_labeled = route_diag(hs)
    w_routes, w_match, w_labeled = route_diag(ws)
    wg, wt, wr, wdg, wdt, wdr = bridge_diag(ws)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.65), sharey=False)
    fig.subplots_adjust(wspace=0.28, top=0.84)

    draw_panel(
        axes[0],
        hot,
        "a",
        "Direct multi-hop: HotpotQA comparison",
        (
            f"DRIP-ESC-Lease improves H2 R@5: {hs['DRIP-ESC-Lease']['recall@5_h2']:.1f}%\n"
            f"vs ARC {hs['AgentRAGCache']['recall@5_h2']:.1f}%, "
            f"DRIP-Dense {hs['DRIP-Dense']['recall@5_h2']:.1f}%\n"
            f"router: MULTI_DIRECT {h_routes['MULTI_DIRECT']}/{h_labeled}"
        ),
        "#065F46",
        "#D1FAE5",
    )
    draw_panel(
        axes[1],
        wiki,
        "b",
        "Hidden bridge: 2WikiMultihopQA bridge-comparison",
        (
            f"Current bridge path fails: DRIP-ESC-Lease H2 R@5 {ws['DRIP-ESC-Lease']['recall@5_h2']:.1f}%\n"
            f"DRIP-Dense {ws['DRIP-Dense']['recall@5_h2']:.1f}%, "
            f"ARC {ws['AgentRAGCache']['recall@5_h2']:.1f}%, "
            f"Oracle {ws['Oracle']['recall@5_h2']:.1f}%\n"
            f"bridge gold rate {wg}/{wt} = {wr:.1f}% "
            f"(first-hop direct {wdr:.1f}%)"
        ),
        "#92400E",
        "#FEF3C7",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set(labels)
    for h, l in zip(*axes[1].get_legend_handles_labels()):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        fontsize=8.5,
        framealpha=0.93,
        columnspacing=1.0,
        handlelength=2.2,
    )
    fig.suptitle(
        "DRIP ablation rerun: direct multi-hop works; hidden bridge admission is still the bottleneck",
        fontsize=11.7,
        y=0.98,
    )

    for ext in ("pdf", "png"):
        out = OUT / f"fig2_intro_drip_newbridge_audit.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)

    print("Hotpot routes:", h_routes, h_match, "/", h_labeled)
    print("2Wiki routes:", w_routes, w_match, "/", w_labeled)
    print("2Wiki bridge gold:", wg, "/", wt, f"= {wr:.2f}%")


if __name__ == "__main__":
    main()
