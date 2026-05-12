#!/usr/bin/env python3
"""Compare Dense vs Graph retrieval for Mo2 (HotpotQA, 50 windows)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / 'data'
FIG_DIR  = THIS_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

FILES = {
    ('sudden', 'dense'): 'results_50w_hotpot_sudden_dense.json',
    ('sudden', 'graph'): 'results_50w_hotpot_sudden_graph.json',
    ('gradual', 'dense'): 'results_50w_hotpot_gradual_dense.json',
    ('gradual', 'graph'): 'results_50w_hotpot_gradual_graph.json',
}
results = {k: json.load(open(DATA_DIR/v))['hotpotqa'] for k, v in FILES.items()}

STRATEGIES = ['Static','RandomFIFO','DocArrival','KnowledgeEdit',
              'OnDemandFetch','LogDrivenArrival','QueryDrivenCluster','Oracle']
LABELS = {
    'Static':'Static (no update)','RandomFIFO':'Random FIFO',
    'DocArrival':'Doc-Arrival (HippoRAG)','KnowledgeEdit':'Knowledge-Edit (RECIPE)',
    'OnDemandFetch':'On-Demand Fetch (CRAG)','LogDrivenArrival':'Log-Driven (lagging)',
    'QueryDrivenCluster':'Query-Driven (ours)','Oracle':'Oracle (upper bound)',
}
PALETTE = {
    'Static':('#7F7F7F','--','o'),'RandomFIFO':('#9467BD','-.', '^'),
    'DocArrival':('#8C564B',':','s'),'KnowledgeEdit':('#E377C2','-.','D'),
    'OnDemandFetch':('#17BECF','--','v'),'LogDrivenArrival':('#BCBD22',':','P'),
    'QueryDrivenCluster':('#10B981','-','D'),'Oracle':('#D62728','-',None),
}
EMPHASIS = {'Oracle','QueryDrivenCluster','OnDemandFetch'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,
    'xtick.labelsize':9,'ytick.labelsize':9,
    'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42})

# ── Fig 1: 2×2 per-window line chart ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
COL_TITLES = ['Dense Retrieval (cosine)', 'Graph Retrieval (PPR + IDF + EE-edges)']
ROW_DRIFTS = ['sudden','gradual']
ROW_TITLES = ['Sudden Drift','Gradual Drift']
for row, drift in enumerate(ROW_DRIFTS):
    for col, ret in enumerate(['dense','graph']):
        ax = axes[row, col]
        res = results[(drift, ret)]
        nw = res['config']['n_windows']; half = nw // 2
        x = np.arange(1, nw+1)
        for name in STRATEGIES:
            if name not in res['summary']: continue
            color, ls, marker = PALETTE[name]
            vals = res['summary'][name]['recall@5_per_window']
            lw = 2.4 if name=='Oracle' else (2.0 if name in EMPHASIS else 1.3)
            ax.plot(x, vals, color=color, linestyle=ls, marker=marker,
                    linewidth=lw, alpha=(1.0 if name in EMPHASIS else 0.78),
                    markersize=4, markevery=max(1,nw//10),
                    zorder=(5 if name in EMPHASIS else 3),
                    label=LABELS.get(name,name))
        ax.axvline(half+0.5, color='#444', ls='--', lw=0.9, alpha=0.6)
        if row==0: ax.text(half+0.8,96,'drift onset',va='top',ha='left',fontsize=8,color='#444')
        ax.grid(True,axis='y',alpha=0.22,ls=':')
        ax.set_xlim(1,nw); ax.set_ylim(0,100)
        ax.set_ylabel('Recall@5 (%)')
        if row==1: ax.set_xlabel('Window index')
        ax.set_title(f'{ROW_TITLES[row]} – {COL_TITLES[col]}\n'
                     f'pool={res["config"]["pool_size"]:,}  KB={res["config"]["kb_budget"]:,}',
                     fontsize=10)
handles, labels_ = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5,-0.01))
fig.suptitle('HotpotQA – Dense vs Graph Retrieval (Recall@5, 50 windows)',fontsize=13,y=1.01)
fig.tight_layout(rect=[0,0.08,1,1.00])
for ext in ('pdf','png'):
    fig.savefig(FIG_DIR/f'dense_vs_graph_recall5.{ext}', bbox_inches='tight', dpi=(200 if ext=='png' else None))
print(f"Saved dense_vs_graph_recall5.pdf/png")
plt.close(fig)

# ── Fig 2: Bar chart H2 summary ──────────────────────────────────────────
SHOW = ['Static','DocArrival','KnowledgeEdit','QueryDrivenCluster','OnDemandFetch','Oracle']
fig2, axes2 = plt.subplots(1, 2, figsize=(13,5), sharey=True)
x_pos = np.arange(len(SHOW)); bar_w = 0.35
for col, drift in enumerate(ROW_DRIFTS):
    ax = axes2[col]
    def get(ret, half):
        return [results[(drift,ret)]['summary'].get(n,{}).get(f'recall@5_{half}',0) for n in SHOW]
    dense_h2 = get('dense','h2'); graph_h2 = get('graph','h2')
    dense_h1 = get('dense','h1'); graph_h1 = get('graph','h1')
    ax.bar(x_pos-bar_w*0.5, dense_h2, bar_w, label='Dense H2', color='#6495ED', alpha=0.85, edgecolor='white')
    ax.bar(x_pos+bar_w*0.5, graph_h2, bar_w, label='Graph H2 (PPR)', color='#10B981', alpha=0.85, edgecolor='white')
    for i,(dh1,gh1) in enumerate(zip(dense_h1,graph_h1)):
        ax.hlines(dh1, i-bar_w, i, colors='#4070C0', linewidths=2.5)
        ax.hlines(gh1, i, i+bar_w, colors='#0a7a52', linewidths=2.5)
    for i,(d2,g2) in enumerate(zip(dense_h2,graph_h2)):
        delta = g2-d2
        if abs(delta)>=0.3:
            ax.text(x_pos[i]+bar_w*0.5, g2+1.5, f'{delta:+.1f}', ha='center', va='bottom',
                    fontsize=8, color=('#10B981' if delta>0 else '#D62728'), fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([LABELS.get(n,n) for n in SHOW], rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0,100); ax.set_ylabel('Recall@5 (%)')
    ax.set_title(f'{ROW_TITLES[col]}\nH2 recall (bars) with H1 reference (lines)', fontsize=10)
    ax.grid(True,axis='y',alpha=0.22,ls=':')
    ax.legend(fontsize=9)
fig2.suptitle('Dense vs Graph Retrieval – H2 Recall@5 Summary (HotpotQA)',fontsize=12,y=1.01)
fig2.tight_layout()
for ext in ('pdf','png'):
    fig2.savefig(FIG_DIR/f'dense_vs_graph_bar_summary.{ext}', bbox_inches='tight', dpi=(200 if ext=='png' else None))
print("Saved dense_vs_graph_bar_summary.pdf/png")
plt.close(fig2)

# ── Fig 3: Per-window delta (graph − dense) ───────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13,4.5), sharex=False)
for col, drift in enumerate(ROW_DRIFTS):
    ax = axes3[col]
    rd = results[(drift,'dense')]['summary']; rg = results[(drift,'graph')]['summary']
    nw = results[(drift,'dense')]['config']['n_windows']; half = nw//2
    x = np.arange(1,nw+1)
    for name in STRATEGIES:
        if name not in rd or name not in rg: continue
        color, ls, marker = PALETTE[name]
        delta = np.array(rg[name]['recall@5_per_window']) - np.array(rd[name]['recall@5_per_window'])
        ax.plot(x, delta, color=color, linestyle=ls,
                linewidth=(2.0 if name in EMPHASIS else 1.2),
                alpha=0.9, label=LABELS.get(name,name))
    ax.axhline(0, color='#333', lw=1.0, ls='--')
    ax.axvline(half+0.5, color='#444', ls='--', lw=0.9, alpha=0.6)
    ax.grid(True,axis='y',alpha=0.22,ls=':')
    ax.set_xlim(1,nw); ax.set_xlabel('Window index')
    ax.set_ylabel('Δ Recall@5 (graph − dense, pp)')
    ax.set_title(f'{ROW_TITLES[col]}\nGraph lift over dense per window',fontsize=10)
handles, labels_ = axes3[0].get_legend_handles_labels()
fig3.legend(handles, labels_, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5,-0.04))
fig3.suptitle('Graph vs Dense Recall@5 Delta per Window (HotpotQA)',fontsize=12,y=1.01)
fig3.tight_layout(rect=[0,0.10,1,1.00])
for ext in ('pdf','png'):
    fig3.savefig(FIG_DIR/f'dense_vs_graph_delta.{ext}', bbox_inches='tight', dpi=(200 if ext=='png' else None))
print("Saved dense_vs_graph_delta.pdf/png")
plt.close(fig3)

print('\nAll figures saved to', FIG_DIR)
