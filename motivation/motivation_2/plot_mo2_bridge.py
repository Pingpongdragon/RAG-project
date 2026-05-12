"""Mo2 figure - mo1-style 2x3 across three multi-hop bridge datasets.

Rows: KB Coverage, Recall@5
Cols: HotpotQA-bridge, 2WikiMultihopQA, MuSiQue
Setup: BGE-large + LightGraphRAG PPR, sudden drift, 100x50 windows, n_source=3500.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
FIGS = ROOT / 'figures'
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 11,
    'legend.fontsize': 9.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

PALETTE = {
    'Static':             ('#9E9E9E', (0, (2, 2))),
    'RandomFIFO':         ('#BCBD22', (0, (3, 2))),
    'DocArrival':         ('#43A047', (0, (6, 2))),
    'KnowledgeEdit':      ('#8E24AA', (0, (5, 2, 1, 2))),
    'OnDemandFetch':      ('#00ACC1', (0, (2, 1.5))),
    'LogDrivenArrival':   ('#EF8C00', (0, (7, 2))),
    'QueryDrivenCluster': ('#1565C0', '-'),
    'Oracle':             ('#C62828', '-'),
}
LABELS = {
    'Static':             'Static (frozen)',
    'RandomFIFO':         'Random-FIFO',
    'DocArrival':         'Doc-Arrival (supply-triggered)',
    'KnowledgeEdit':      'Knowledge-Edit (edit-based)',
    'OnDemandFetch':      'On-Demand (per-query online)',
    'LogDrivenArrival':   'Log-Driven (lagged feedback)',
    'QueryDrivenCluster': r'$\bf{QueryDriven\ (ours)}$',
    'Oracle':             'Oracle',
}
SHOW_MAIN = ['Static', 'DocArrival', 'KnowledgeEdit',
             'OnDemandFetch', 'LogDrivenArrival',
             'QueryDrivenCluster', 'Oracle']
SHOW_FULL = list(PALETTE.keys())

CURVE_FILES = {
    'hotpotqa': DATA / 'results_100w_hotpot_bridge_sudden_graph_bge_v2.json',
    '2wiki':    DATA / 'results_100w_2wiki_sudden_graph_bge_v2.json',
    'musique':  DATA / 'results_100w_musique_sudden_graph_bge_v2.json',
}
DS_TITLES = {
    'hotpotqa': 'HotpotQA-bridge',
    '2wiki':    '2WikiMultihopQA',
    'musique':  'MuSiQue',
}
DATASETS = ['hotpotqa', '2wiki', 'musique']
METRIC_KEYS = ['kb_coverage_per_window', 'recall@5_per_window']
METRIC_YLABELS = ['KB Coverage (%)', 'Recall@5 (%)']

PRE_BG  = '#EDF4FC'
POST_BG = '#FDF5E6'

curve_results = {}
for ds, path in CURVE_FILES.items():
    raw = json.load(open(path))
    curve_results[ds] = raw[list(raw.keys())[0]]


def _smooth_halves(arr, half, w=6):
    kernel = np.exp(-0.5 * (np.arange(-(w//2), w//2+1) / (w / 3.0))**2)
    kernel /= kernel.sum()
    pad = w // 2
    def _conv(seg):
        return np.convolve(np.pad(seg, pad, mode='edge'), kernel, mode='valid')
    return np.concatenate([_conv(arr[:half]), _conv(arr[half:])])


def _draw_panel(ax, res, show, metric_key, ylabel,
                lw_ours, lw_oracle, lw_other, smooth_w=6, first_panel=False):
    cfg  = res['config']
    nw   = cfg['n_windows']
    half = nw // 2
    x    = np.arange(1, nw + 1)
    ax.axvspan(0.5, half + 0.5, facecolor=PRE_BG,  alpha=1.0, zorder=0)
    ax.axvspan(half + 0.5, nw + 0.5, facecolor=POST_BG, alpha=1.0, zorder=0)
    for name in show:
        if name not in res['summary']:
            continue
        per = res['summary'][name].get(metric_key)
        if not per:
            continue
        color, ls = PALETTE[name]
        vals = _smooth_halves(np.asarray(per, dtype=float), half, smooth_w)
        if   name == 'QueryDrivenCluster': lw, alpha, z = lw_ours,   1.00, 6
        elif name == 'Oracle':             lw, alpha, z = lw_oracle, 0.95, 5
        elif name == 'Static':             lw, alpha, z = lw_other,  0.90, 3
        else:                              lw, alpha, z = lw_other,  0.80, 2
        ax.plot(x, vals, color=color, linestyle=ls, linewidth=lw,
                alpha=alpha, zorder=z, label=LABELS[name],
                solid_capstyle='round', dash_capstyle='round')
    ax.axvline(half + 0.5, color='#333', ls='-', lw=1.4, alpha=0.6, zorder=1)
    if first_panel:
        ax.text(half + 0.5, 102, 'drift onset',
                ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#333',
                bbox=dict(boxstyle='round,pad=0.22', fc='white',
                          ec='#333', lw=1.0, alpha=0.92))
        ax.text(half * 0.5, 3.5, 'pre-drift', ha='center', va='bottom',
                fontsize=9.5, color='#3A5F8A', fontweight='bold', alpha=0.95)
        ax.text(half + (nw - half) * 0.5, 3.5, 'post-drift',
                ha='center', va='bottom', fontsize=9.5,
                color='#9C5A00', fontweight='bold', alpha=0.95)
    ax.set_xlim(0.5, nw + 0.5)
    ax.set_ylim(0, 106)
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.20, linestyle=':', zorder=1)
    return cfg


def _build_figure(show, fig_w, fig_h, lw_ours, lw_oracle, lw_other,
                  smooth_w=6, legend_ncol=None):
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_h), sharex='col')
    for row, (mk, ylbl) in enumerate(zip(METRIC_KEYS, METRIC_YLABELS)):
        for col, ds in enumerate(DATASETS):
            ax  = axes[row, col]
            cfg = _draw_panel(ax, curve_results[ds], show, mk, ylbl,
                              lw_ours, lw_oracle, lw_other, smooth_w,
                              first_panel=(row == 0 and col == 0))
            if row == 0:
                ax.set_title(
                    f'{DS_TITLES[ds]}\n'
                    f'pool={cfg["pool_size"]:,},  KB budget={cfg["kb_budget"]:,}',
                    fontsize=12, fontweight='bold', pad=8)
            if row == len(METRIC_KEYS) - 1:
                ax.set_xlabel('Window index (time \u2192)', fontsize=12)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = min(len(labels_), 4)
    fig.legend(handles, labels_, loc='lower center', ncol=legend_ncol,
               frameon=False, bbox_to_anchor=(0.5, -0.03),
               fontsize=11, handlelength=3.5, columnspacing=1.5,
               handletextpad=0.6)
    fig.suptitle('Sudden drift across multi-hop bridge datasets '
                 '(BGE-large + LightGraphRAG PPR)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


fig = _build_figure(SHOW_MAIN, fig_w=15.5, fig_h=8.4,
                    lw_ours=3.2, lw_oracle=2.6, lw_other=1.8, smooth_w=6)
for ext in ('pdf', 'png'):
    pth = FIGS / f'mo2_bridge_sudden.{ext}'
    fig.savefig(pth, bbox_inches='tight',
                dpi=(None if ext == 'pdf' else 200))
    print(f'Saved {pth}')
plt.close(fig)

fig_app = _build_figure(SHOW_FULL, fig_w=16.0, fig_h=8.6,
                        lw_ours=3.0, lw_oracle=2.4, lw_other=1.4,
                        smooth_w=6, legend_ncol=4)
for ext in ('pdf', 'png'):
    pth = FIGS / f'mo2_bridge_sudden_appendix.{ext}'
    fig_app.savefig(pth, bbox_inches='tight',
                    dpi=(None if ext == 'pdf' else 200))
    print(f'Saved {pth}')
plt.close(fig_app)
