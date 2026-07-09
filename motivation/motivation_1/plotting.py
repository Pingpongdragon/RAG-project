"""Motivation 1 的图生成工具。

主实验逻辑在 ``run.py``；这个文件只负责把 JSON summary 画成 coverage /
Recall@5 随窗口变化的图。
"""

import numpy as np

from config import FIG_DIR, STRATEGY_LABELS


PALETTE = {
    "Static": ("#7F7F7F", "-", "o"),
    "RandomFIFO": ("#9467BD", "-.", "^"),
    "DocArrival": ("#8C564B", ":", "s"),
    "KnowledgeEdit": ("#E377C2", "-.", "D"),
    "LRU": ("#D97706", "-.", "v"),
    "FIFO": ("#7C3AED", "--", "^"),
    "TinyLFU": ("#0284C7", ":", "p"),
    "TemporalAware": ("#2563EB", "-.", "o"),
    "OnDemandFetch": ("#17BECF", "--", "v"),
    "LogDrivenArrival": ("#BCBD22", ":", "P"),
    "AgentRAGCache": ("#111827", "-", "o"),
    "ARC": ("#111827", "-", "o"),
    "AgentRAGCache_NoHub": ("#6B7280", "--", "o"),
    "CostAwareDRIP": ("#2563EB", "-", "s"),
    "CostAwareDRIP-NoDrift": ("#60A5FA", "--", "s"),
    "CostAwareDRIP-NoChurn": ("#1D4ED8", "-.", "P"),
    "RoutedCache": ("#2563EB", "-", "s"),
    "DRIP-QueryVisible": ("#0F766E", "-", "*"),
    "DRIP-QueryHidden": ("#0D9488", "-.", "X"),
    "DRIP": ("#0F766E", "-", "*"),
    "DRIPNOdetector": ("#2563EB", "-", "s"),
    "Oracle": ("#D62728", "-", None),
}

EMPHASIS_STRATEGIES = {
    "Oracle",
    "DRIP",
    "DRIP-QueryVisible",
    "DRIP-QueryHidden",
    "CostAwareDRIP",
    "DRIPNOdetector",
}


def generate_figures(all_results, strategies_to_run, suffix=""):
    """生成每个数据集的 KB coverage 和 Recall@5 曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    datasets = list(all_results.keys())
    fig, axes = plt.subplots(
        max(len(datasets), 1),
        2,
        figsize=(13, 3.4 * max(len(datasets), 1)),
        sharex=True,
        squeeze=False,
    )

    for row, dataset_name in enumerate(datasets):
        dataset_result = all_results[dataset_name]
        config = dataset_result["config"]
        num_windows = config["n_windows"]
        half_window = num_windows // 2
        x_axis = np.arange(1, num_windows + 1)
        _plot_dataset_row(
            axes[row],
            dataset_name,
            dataset_result,
            config,
            strategies_to_run,
            x_axis,
            half_window,
        )

    title = _figure_title(suffix)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    top = 0.96 if title else 0.99
    fig.tight_layout(rect=[0, 0.06, 1, top])
    fig.savefig(FIG_DIR / f"coverage{suffix}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"coverage{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_row(axes, dataset_name, dataset_result, config,
                      strategies_to_run, x_axis, half_window):
    """画一个数据集的 coverage / recall 两个子图。"""
    for col, (series_key, ylabel) in enumerate((
        ("cov_per_window", "KB Coverage (%)"),
        ("recall@5_per_window", "Recall@5 (%)"),
    )):
        axis = axes[col]
        for strategy_name in strategies_to_run:
            if strategy_name not in dataset_result["summary"]:
                continue
            strategy_summary = dataset_result["summary"][strategy_name]
            color, line_style, marker = PALETTE.get(strategy_name, ("gray", "-", None))
            line_width = 2.4 if strategy_name == "Oracle" else (
                2.0 if strategy_name in EMPHASIS_STRATEGIES else 1.4
            )
            alpha = 1.0 if strategy_name in EMPHASIS_STRATEGIES else 0.85
            zorder = 5 if strategy_name in EMPHASIS_STRATEGIES else 3
            axis.plot(
                x_axis,
                strategy_summary[series_key],
                color=color,
                linestyle=line_style,
                marker=marker,
                linewidth=line_width,
                alpha=alpha,
                markersize=5,
                markevery=max(1, config["n_windows"] // 12),
                zorder=zorder,
                label=STRATEGY_LABELS.get(strategy_name, strategy_name),
            )
        _style_axis(axis, col, dataset_name, config, half_window, ylabel)


def _style_axis(axis, col, dataset_name, config, half_window, ylabel):
    """统一子图样式。"""
    axis.axvline(half_window + 0.5, color="#444444", ls="--", lw=0.9, alpha=0.6)
    axis.grid(True, axis="y", alpha=0.25, linestyle=":")
    axis.set_xlim(1, config["n_windows"])
    axis.set_ylim(0, 100)
    axis.set_ylabel(ylabel)
    if col == 0:
        axis.set_title(
            f"{dataset_name}  pool={config['pool_size']:,}  "
            f"KB={config['kb_budget']:,}  "
            f"{config['n_windows']}x{config['window_size']}",
            fontsize=11,
        )
        axis.text(
            half_window + 0.5,
            95,
            "  drift onset",
            va="top",
            ha="left",
            fontsize=9,
            color="#444444",
        )
    else:
        axis.set_title("Recall@5", fontsize=11)
    axis.set_xlabel("Window index")


def _figure_title(suffix):
    """根据输出 suffix 判断图标题。"""
    if "sudden" in suffix:
        return "Motivation 1 - Sudden Drift"
    if "gradual" in suffix:
        return "Motivation 1 - Gradual Drift"
    return None
