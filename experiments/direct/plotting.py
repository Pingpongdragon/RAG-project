"""Direct-evidence 正式实验的图生成工具。

主实验逻辑在 ``run.py``；这个文件只负责把 JSON summary 画成 coverage /
Recall@5 随窗口变化的图。
"""

import numpy as np

if __package__:
    from .config import FIG_DIR, STRATEGY_LABELS
else:
    from config import FIG_DIR, STRATEGY_LABELS


PALETTE = {
    "LRU": ("#D97706", "-.", "v"),
    "FIFO": ("#7C3AED", "--", "^"),
    "TinyLFU": ("#0284C7", ":", "p"),
    "AgentRAGCache": ("#111827", "-", "o"),
    "GPTCacheStyle": ("#0891B2", ":", "P"),
    "Proximity": ("#06B6D4", ":", "h"),
    "DRIP": ("#0F766E", "-", "*"),
    "Oracle": ("#D62728", "-", None),
}

EMPHASIS_STRATEGIES = {
    "Oracle",
    "DRIP",
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
        x_axis = np.arange(1, num_windows + 1)
        _plot_dataset_row(
            axes[row],
            dataset_name,
            dataset_result,
            config,
            strategies_to_run,
            x_axis,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.99])
    fig.savefig(FIG_DIR / f"coverage{suffix}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"coverage{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_row(axes, dataset_name, dataset_result, config,
                      strategies_to_run, x_axis):
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
        _style_axis(axis, col, dataset_name, config, ylabel)


def _style_axis(axis, col, dataset_name, config, ylabel):
    """统一子图样式。"""
    if config.get("workload") == "factorized_one_shot":
        midpoint = config["n_windows"] // 2 + 0.5
        axis.axvline(midpoint, color="#444444", ls="--", lw=0.9, alpha=0.6)
    axis.grid(True, axis="y", alpha=0.25, linestyle=":")
    axis.set_xlim(1, config["n_windows"])
    axis.set_ylim(0, 100)
    axis.set_ylabel(ylabel)
    if col == 0:
        axis.set_title(
            f"{dataset_name}  pool={config['pool_size']:,}  "
            f"KB={config['kb_budget']:,}  "
            f"{config['n_windows']}x{config['window_size']}  "
            f"{config.get('workload', '')}",
            fontsize=11,
        )
    else:
        axis.set_title("Recall@5", fontsize=11)
    axis.set_xlabel("Window index")
