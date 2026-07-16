"""
DRIP Experiment Results Visualization
Generates publication-quality figures from experiment JSON results.

Usage:
    python plot_results.py
    # Outputs: figures saved to benchmarks/archive_legacy/results/figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -- Configuration --
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

SCENARIOS = ['gradual_drift', 'sudden_shift', 'cyclic_return', 'hotpotqa_walk']
SCENARIO_LABELS = {
    'gradual_drift': 'Gradual Drift',
    'sudden_shift':  'Sudden Shift',
    'cyclic_return': 'Cyclic Return',
    'hotpotqa_walk': 'HotpotQA Walk',
}
METHODS = ['DRIP', 'ERASE', 'ComRAG', 'Static', 'Random']
METHOD_COLORS = {
    'DRIP':   '#E74C3C',
    'ERASE':  '#3498DB',
    'ComRAG': '#2ECC71',
    'Static': '#95A5A6',
    'Random': '#F39C12',
}
METHOD_MARKERS = {
    'DRIP': 'o', 'ERASE': 's', 'ComRAG': '^', 'Static': 'D', 'Random': 'v'
}


def load_all_results():
    data = {}
    for sc in SCENARIOS:
        path = os.path.join(RESULTS_DIR, f'{sc}_results.json')
        if os.path.exists(path):
            with open(path) as f:
                data[sc] = json.load(f)
    return data


def fig1_recall_bar(data):
    """Grouped bar chart: Recall@10 per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.15
    offsets = np.arange(len(METHODS)) - (len(METHODS) - 1) / 2

    for i, method in enumerate(METHODS):
        vals = [data[sc][method]['avg_recall'] for sc in SCENARIOS]
        ax.bar(x + offsets[i] * width, vals, width,
               label=method, color=METHOD_COLORS[method],
               edgecolor='white', linewidth=0.8)
        for j, sc in enumerate(SCENARIOS):
            all_recalls = {m: data[sc][m]['avg_recall'] for m in METHODS}
            winner = max(all_recalls, key=all_recalls.get)
            if method == winner:
                ax.annotate('\u2605', (x[j] + offsets[i] * width, vals[j]),
                           ha='center', va='bottom', fontsize=12, color='gold')

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Recall@10 Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(0.45, ax.get_ylim()[1] * 1.15))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_recall_bar.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_recall_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig1_recall_bar')


def fig2_f1_bar(data):
    """Grouped bar chart: Token F1 per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.15
    offsets = np.arange(len(METHODS)) - (len(METHODS) - 1) / 2

    for i, method in enumerate(METHODS):
        vals = [data[sc][method]['avg_f1'] for sc in SCENARIOS]
        ax.bar(x + offsets[i] * width, vals, width,
               label=method, color=METHOD_COLORS[method],
               edgecolor='white', linewidth=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Token F1', fontsize=12)
    ax.set_title('Token F1 (LLM Generation Quality) Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_f1_bar.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_f1_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig2_f1_bar')


def fig3_updates_efficiency(data):
    """Recall@10 vs Total Updates scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHODS:
        recalls = [data[sc][method]['avg_recall'] for sc in SCENARIOS]
        updates = [data[sc][method]['total_updates'] for sc in SCENARIOS]
        ax.scatter(updates, recalls,
                   c=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                   s=120, label=method, edgecolors='black', linewidths=0.5, zorder=3)
        for j, sc in enumerate(SCENARIOS):
            ax.annotate(SCENARIO_LABELS[sc][:3],
                       (updates[j], recalls[j]),
                       textcoords='offset points', xytext=(5, 5),
                       fontsize=7, alpha=0.7)

    ax.set_xlabel('Total KB Updates', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Update Efficiency: Recall vs. Number of Updates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_update_efficiency.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_update_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig3_update_efficiency')


def fig4_radar(data):
    """Radar chart: multi-metric comparison (averaged)."""
    metrics = ['Recall@10', 'Token F1', 'MRR', 'Gold_KB', '1-Turnover']
    active_methods = ['DRIP', 'ERASE', 'ComRAG']

    metric_vals = {}
    for method in active_methods:
        avgs = {}
        avgs['Recall@10'] = np.mean([data[sc][method]['avg_recall'] for sc in SCENARIOS])
        avgs['Token F1'] = np.mean([data[sc][method]['avg_f1'] for sc in SCENARIOS])
        avgs['MRR'] = np.mean([data[sc][method]['avg_mrr'] for sc in SCENARIOS])
        avgs['Gold_KB'] = np.mean([data[sc][method]['avg_gold_in_kb'] for sc in SCENARIOS])
        avgs['1-Turnover'] = 1.0 - np.mean([data[sc][method]['kb_turnover'] for sc in SCENARIOS])
        metric_vals[method] = avgs

    for metric in metrics:
        vals = [metric_vals[m][metric] for m in active_methods]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax > vmin else 1.0
        for m in active_methods:
            metric_vals[m][metric] = (metric_vals[m][metric] - vmin) / rng

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for method in active_methods:
        values = [metric_vals[method][m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, '-o', color=METHOD_COLORS[method],
                label=method, linewidth=2, markersize=6)
        ax.fill(angles, values, color=METHOD_COLORS[method], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title('Multi-Metric Comparison (Normalized)', fontsize=14,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_radar.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_radar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig4_radar')


def fig5_sliding_recall(data):
    """Sliding Recall over query windows (4 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=True)
    axes = axes.flatten()

    for idx, sc in enumerate(SCENARIOS):
        ax = axes[idx]
        for method in ['DRIP', 'ERASE', 'ComRAG']:
            recalls = data[sc][method]['recalls']
            window = 10
            if len(recalls) >= window:
                sliding = np.convolve(recalls, np.ones(window)/window, mode='valid')
                x_axis = np.arange(len(sliding)) + window
            else:
                sliding = recalls
                x_axis = np.arange(len(sliding)) + 1
            ax.plot(x_axis, sliding, color=METHOD_COLORS[method],
                    label=method, linewidth=1.5, alpha=0.85)

        ax.set_title(SCENARIO_LABELS[sc], fontsize=12, fontweight='bold')
        ax.set_xlabel('Query Index', fontsize=10)
        ax.set_ylabel('Sliding Recall@10', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('Sliding Recall@10 Over Time (Window=10)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_sliding_recall.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_sliding_recall.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig5_sliding_recall')


def fig6_overall_summary(data):
    """Combined horizontal bar: Overall Average summary."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_keys = [
        ('avg_recall', 'Recall@10'),
        ('avg_f1', 'Token F1'),
        ('total_updates', 'Total Updates (avg)')
    ]

    for ax_idx, (key, label) in enumerate(metric_keys):
        ax = axes[ax_idx]
        vals = []
        for method in METHODS:
            v = np.mean([data[sc][method][key] for sc in SCENARIOS])
            vals.append(v)

        if key == 'total_updates':
            order = np.argsort(vals)
        else:
            order = np.argsort(vals)[::-1]

        sorted_methods = [METHODS[i] for i in order]
        sorted_vals = [vals[i] for i in order]
        colors = [METHOD_COLORS[m] for m in sorted_methods]

        bars = ax.barh(range(len(sorted_methods)), sorted_vals, color=colors,
                       edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(sorted_methods)))
        ax.set_yticklabels(sorted_methods, fontsize=11)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, sorted_vals):
            txt = f'{val:.0f}' if key == 'total_updates' else f'{val:.4f}'
            ax.text(bar.get_width() + max(sorted_vals) * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    txt, va='center', fontsize=9)

    fig.suptitle('Overall Performance Summary (Averaged Across All Scenarios)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_overall_summary.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_overall_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig6_overall_summary')


def fig7_time_bar(data):
    """Runtime comparison bar chart per scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.15
    offsets = np.arange(len(METHODS)) - (len(METHODS) - 1) / 2

    for i, method in enumerate(METHODS):
        vals = [data[sc][method]['total_time_sec'] for sc in SCENARIOS]
        ax.bar(x + offsets[i] * width, vals, width,
               label=method, color=METHOD_COLORS[method],
               edgecolor='white', linewidth=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Total Time (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig7_time_bar.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig7_time_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig7_time_bar')


def fig8_turnover_bar(data):
    """KB Turnover comparison per scenario (adaptive methods only)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SCENARIOS))
    active = ['DRIP', 'ERASE', 'ComRAG']
    offsets = np.arange(len(active)) - (len(active) - 1) / 2
    width = 0.22

    for i, method in enumerate(active):
        vals = [data[sc][method]['kb_turnover'] for sc in SCENARIOS]
        ax.bar(x + offsets[i] * width, vals, width,
               label=method, color=METHOD_COLORS[method],
               edgecolor='white', linewidth=0.8)

    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('KB Turnover Rate', fontsize=12)
    ax.set_title('KB Turnover Rate Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig8_turnover_bar.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'fig8_turnover_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[OK] fig8_turnover_bar')


def main():
    print('Loading results...')
    data = load_all_results()
    print(f'Loaded {len(data)} scenarios: {list(data.keys())}')

    if len(data) < len(SCENARIOS):
        missing = set(SCENARIOS) - set(data.keys())
        print(f'WARNING: Missing scenarios: {missing}')

    print('\nGenerating figures...')
    fig1_recall_bar(data)
    fig2_f1_bar(data)
    fig3_updates_efficiency(data)
    fig4_radar(data)
    fig5_sliding_recall(data)
    fig6_overall_summary(data)
    fig7_time_bar(data)
    fig8_turnover_bar(data)

    print(f'\nAll figures saved to: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
