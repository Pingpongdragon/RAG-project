#!/usr/bin/env python3
"""Mo1 paper figures — all three outputs from one script.

Outputs
-------
figures/mo1_combined.pdf          2×2 coverage curves (HotpotQA + FEVER)
figures/dataset_delta_bar.pdf     QDC Δ bar chart across all datasets
figures/dataset_condition_space.pdf  q/SF vs alignment scatter
dataset_analysis_table.tex        LaTeX table
"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from pathlib import Path

DATA = Path(__file__).parent / 'data'
FIGS = Path(__file__).parent / 'figures'
OUT  = Path(__file__).parent
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 11,
    'legend.fontsize': 9.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# ════════════════════════════════════════════════════════════════════════════
# ── Figure 1: 2×2 coverage curves ───────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

PALETTE = {
    'Static':             ('#7F7F7F', '-',  'o'),
    'RandomFIFO':         ('#FF7F0E', '--', 's'),
    'DocArrival':         ('#2CA02C', ':',  '^'),
    'KnowledgeEdit':      ('#9467BD', '--', 'D'),
    'OnDemandFetch':      ('#17BECF', '--', 'v'),
    'LogDrivenArrival':   ('#BCBD22', ':',  'P'),
    'QueryDrivenCluster': ('#1F77B4', '-',  '*'),
    'Oracle':             ('#D62728', '-',  None),
}
LABELS = {
    'Static':             'Static',
    'RandomFIFO':         'Random-FIFO',
    'DocArrival':         'Doc-Arrival',
    'KnowledgeEdit':      'Knowledge-Edit',
    'OnDemandFetch':      'On-Demand',
    'LogDrivenArrival':   'Log-Driven',
    'QueryDrivenCluster': 'QueryDriven (ours)',
    'Oracle':             'Oracle',
}
SHOW = list(PALETTE.keys())

CURVE_FILES = {
    ('hotpotqa', 'sudden'):  DATA / 'results_100w_sudden.json',
    ('hotpotqa', 'gradual'): DATA / 'results_100w_gradual.json',
    ('fever', 'sudden'):     DATA / 'results_fever_100w_sudden.json',
    ('fever', 'gradual'):    DATA / 'results_fever_100w_gradual.json',
}
DS_LABELS    = {'hotpotqa': 'HotpotQA-comparison', 'fever': 'FEVER'}
DRIFT_LABELS = {'sudden': 'Sudden drift', 'gradual': 'Gradual drift'}

curve_results = {}
for key, path in CURVE_FILES.items():
    raw = json.load(open(path))
    curve_results[key] = raw[list(raw.keys())[0]]

DRIFTS   = ['sudden', 'gradual']
DATASETS = ['hotpotqa', 'fever']
fig1, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)

for row, drift in enumerate(DRIFTS):
    for col, ds in enumerate(DATASETS):
        ax  = axes[row, col]
        res = curve_results[(ds, drift)]
        cfg = res['config']
        nw  = cfg['n_windows']
        x   = np.arange(1, nw + 1)
        half = nw // 2

        for name in SHOW:
            if name not in res['summary']:
                continue
            color, ls, marker = PALETTE[name]
            cov = res['summary'][name]['cov_per_window']
            lw     = 2.4 if name == 'Oracle' else (2.2 if name == 'QueryDrivenCluster' else 1.3)
            alpha  = 1.0 if name in ('Oracle', 'QueryDrivenCluster') else 0.80
            zorder = 5 if name == 'QueryDrivenCluster' else (4 if name == 'Oracle' else 2)
            ax.plot(x, cov, color=color, linestyle=ls, marker=marker,
                    linewidth=lw, alpha=alpha, markersize=5,
                    markevery=max(1, nw // 10), zorder=zorder,
                    label=LABELS[name])

        ax.axvline(half + 0.5, color='#444', ls='--', lw=0.9, alpha=0.6)
        if row == 0 and col == 0:
            ax.text(half + 1, 96, 'drift\nonset', va='top', ha='left',
                    fontsize=8.5, color='#444')

        s_q = res['summary']['QueryDrivenCluster']
        s_s = res['summary']['Static']
        delta = s_q['cov_h2'] - s_s['cov_h2']
        sign  = '+' if delta >= 0 else ''
        ann_color = '#1F77B4' if delta > 5 else '#D62728'
        ax.annotate(f'QDC {sign}{delta:.1f}pp vs Static\n(post-drift cumul.)',
                    xy=(nw, s_q['cov_per_window'][-1]),
                    xytext=(nw - 25, 15),
                    fontsize=8.5, color=ann_color,
                    arrowprops=dict(arrowstyle='->', color=ann_color, lw=1.0))

        ax.set_xlim(1, nw); ax.set_ylim(0, 103)
        ax.grid(True, axis='y', alpha=0.22, linestyle=':')
        if row == 0:
            ax.set_title(f'{DS_LABELS[ds]}\npool={cfg["pool_size"]:,}  KB={cfg["kb_budget"]:,}',
                         fontsize=11)
        if col == 0:
            ax.set_ylabel(f'{DRIFT_LABELS[drift]}\nCoverage (%)', fontsize=11)
        if row == 1:
            ax.set_xlabel('Window index', fontsize=11)

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig1.legend(handles, labels_, loc='lower center', ncol=len(labels_),
            frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
fig1.tight_layout(rect=[0, 0.055, 1, 1])
for ext in ('pdf', 'png'):
    p = FIGS / f'mo1_combined.{ext}'
    fig1.savefig(p, bbox_inches='tight', dpi=(None if ext=='pdf' else 200))
    print(f'Saved {p}')
plt.close(fig1)

# ════════════════════════════════════════════════════════════════════════════
# ── Shared dataset-analysis data ────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# (ds_key, display_label, q/SF, query↔SF alignment, failure_reason)
# failure_reason: None = success; 'Cond-A' / 'Cond-B' / 'Cond-A+B' / 'Cond-C'
# Cond-A: q/SF ≥ 1.3  (enough cross-query demand signal)
# Cond-B: query text directly names SF entities (embedding alignment ≥ 0.6)
# Cond-C: pool has many distractor docs (RandomFIFO can't accidentally pick gold)
META = [
    ('hotpotqa_comparison',    'HotpotQA-comp',        1.35, 0.92, None),
    ('fever',                  'FEVER',                 8.15, 0.99, None),
    ('triviaqa_wikipedia',     'TriviaQA-Wiki',         1.36, 0.88, 'Cond-C'),
    ('2wiki_bridge_comparison','2wiki-bridge-comp',     1.66, 0.31, 'Cond-B'),
    ('2wiki_comparison',       '2wiki-comp',            1.02, 0.85, 'Cond-A'),
    ('2wiki_simple',           '2wiki-simple\n(mixed)', 1.19, 0.60, 'Cond-A+B'),
]

RESULT_FILES = {
    ('hotpotqa_comparison',    'sudden'):  'results_100w_sudden.json',
    ('hotpotqa_comparison',    'gradual'): 'results_100w_gradual.json',
    ('fever',                  'sudden'):  'results_fever_100w_sudden.json',
    ('fever',                  'gradual'): 'results_fever_100w_gradual.json',
    ('triviaqa_wikipedia',     'sudden'):  'results_triviaqa_wiki_100w_sudden.json',
    ('triviaqa_wikipedia',     'gradual'): 'results_triviaqa_wiki_100w_gradual.json',
    ('2wiki_bridge_comparison','sudden'):  'results_2wiki_bc_100w_sudden.json',
    ('2wiki_bridge_comparison','gradual'): 'results_2wiki_bc_100w_gradual.json',
    ('2wiki_comparison',       'sudden'):  'results_2wiki_comp_kb14000_100w_sudden.json',
    ('2wiki_comparison',       'gradual'): 'results_2wiki_comp_kb14000_100w_gradual.json',
    ('2wiki_simple',           'sudden'):  'results_2wiki_simple_100w_sudden.json',
}

def get_delta(ds, drift):
    fname = RESULT_FILES.get((ds, drift))
    if not fname: return None
    p = DATA / fname
    if not p.exists(): return None
    raw = json.load(open(p))
    k = list(raw.keys())[0]
    s = raw[k]['summary']
    return s.get('QueryDrivenCluster',{}).get('cov_h2',0) - s.get('Static',{}).get('cov_h2',0)

deltas = {(m[0], d): get_delta(m[0], d) for m in META for d in ('sudden','gradual')}

# ════════════════════════════════════════════════════════════════════════════
# ── Figure 2: Delta bar chart ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(9.5, 5.2))
x = np.arange(len(META)); w = 0.36
for i, (ds_id, label, qsf, align, fail) in enumerate(META):
    bc = '#1F77B4' if fail is None else '#D62728'
    lc = '#AEC7E8' if fail is None else '#F5A9A9'
    vs = deltas.get((ds_id,'sudden'))
    vg = deltas.get((ds_id,'gradual'))
    if vs is not None:
        ax2.bar(x[i]-w/2, vs, w, color=bc, zorder=3)
        ax2.text(x[i]-w/2, vs+(1.2 if vs>=0 else -3.0),
                 f'{vs:+.1f}', ha='center', va='bottom',
                 fontsize=9, color=bc, fontweight='bold')
    if vg is not None:
        ax2.bar(x[i]+w/2, vg, w, color=lc, edgecolor=bc, lw=1.2, zorder=3)
        ax2.text(x[i]+w/2, vg+(1.2 if vg>=0 else -3.0),
                 f'{vg:+.1f}', ha='center', va='bottom', fontsize=9, color=bc)
    if fail:
        ax2.text(x[i], -11.5, f'Fails: {fail}', ha='center', va='top',
                 fontsize=8.5, color='#D62728',
                 bbox=dict(boxstyle='round,pad=0.25', fc='#FFF0F0',
                           ec='#D62728', alpha=0.85))

ax2.axhline(0, color='#333', lw=0.9)
ax2.set_xticks(x)
ax2.set_xticklabels([m[1] for m in META], fontsize=10)
ax2.set_ylabel('QDC ΔCoverage H2 vs Static (pp)', fontsize=12)
ax2.set_title('QueryDriven KB Curation: gain over Static across all evaluated datasets',
              fontsize=11.5)
ax2.set_ylim(-16, 34)
ax2.grid(True, axis='y', alpha=0.22, linestyle=':')
legend_els = [
    Patch(fc='#1F77B4',              label='Sudden (success)'),
    Patch(fc='#AEC7E8',ec='#1F77B4', lw=1.2, label='Gradual (success)'),
    Patch(fc='#D62728',              label='Sudden (cond. not met)'),
    Patch(fc='#F5A9A9',ec='#D62728', lw=1.2, label='Gradual (cond. not met)'),
]
ax2.legend(handles=legend_els, loc='upper right', frameon=False, fontsize=9)
fig2.tight_layout()
for ext in ('pdf','png'):
    p = FIGS/f'dataset_delta_bar.{ext}'
    fig2.savefig(p, bbox_inches='tight', dpi=(None if ext=='pdf' else 200))
    print(f'Saved {p}')
plt.close(fig2)

# ════════════════════════════════════════════════════════════════════════════
# ── Figure 3: Condition space scatter (log x-axis for FEVER) ────────────────
# ════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(7.0, 5.2))
# shade the effective zone  (q/SF ≥ 1.3  AND  align ≥ 0.6  AND  Cond-C met)
rect = Rectangle((1.3, 0.6), 20, 0.5, lw=0, fc='#D9E8FF', alpha=0.35, zorder=0)
ax3.add_patch(rect)
ax3.text(3.0, 0.98, 'QDC effective zone\n(all 3 conditions met)',
         ha='center', va='top', fontsize=9, color='#1F77B4', alpha=0.85)

for ds_id, label, qsf, align, fail in META:
    vs = deltas.get((ds_id,'sudden')) or 0
    color  = '#1F77B4' if fail is None else '#D62728'
    marker = '*'       if fail is None else 'X'
    sz     = 260       if fail is None else 180
    ax3.scatter(qsf, align, s=sz, c=color, marker=marker, zorder=4)
    clean_label = label.replace('\n', ' ')
    dy = 0.05 if align < 0.85 else -0.08
    ax3.annotate(f'{clean_label}\n(Δ={vs:+.1f}pp)',
                 xy=(qsf, align), xytext=(qsf * 1.08, align + dy),
                 fontsize=8.5, color=color,
                 arrowprops=dict(arrowstyle='-', color=color, lw=0.7))

ax3.axvline(1.3, color='#888', ls='--', lw=1.0, alpha=0.7,
            label='Cond-A: q/SF ≥ 1.3')
ax3.axhline(0.6, color='#B8860B', ls=':', lw=1.0, alpha=0.7,
            label='Cond-B: query↔SF align ≥ 0.6')
ax3.set_xscale('log')
ax3.set_xlabel('q/SF reuse — log scale (post-subsample)', fontsize=12)
ax3.set_ylabel('Query ↔ SF semantic alignment (est.)', fontsize=12)
ax3.set_title('QDC Condition Space\n(★ = all conditions met, ✗ = at least one fails)',
              fontsize=11)
ax3.set_xlim(0.85, 15); ax3.set_ylim(0.10, 1.12)
ax3.grid(True, alpha=0.2, linestyle=':')
ax3.legend(fontsize=9, frameon=False, loc='lower right')
fig3.tight_layout()
for ext in ('pdf','png'):
    p = FIGS/f'dataset_condition_space.{ext}'
    fig3.savefig(p, bbox_inches='tight', dpi=(None if ext=='pdf' else 200))
    print(f'Saved {p}')
plt.close(fig3)

# ════════════════════════════════════════════════════════════════════════════
# ── LaTeX table ─────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

ROWS = [
    ('HotpotQA-comparison',    r'4{,}000',  '1.35', r'\ding{51}', r'\ding{51}', r'\ding{51}', '+25.6', '+13.8'),
    ('FEVER',                  r'8{,}000',  '8.15', r'\ding{51}', r'\ding{51}', r'\ding{51}', '+16.6', '+2.7'),
    None,
    ('TriviaQA-Wikipedia',     r'6{,}000',  '1.36', r'\ding{51}', r'\ding{51}', r'\ding{55}', '+30.1$^*$', '+3.7'),
    ('2wiki-comparison',       r'6{,}000',  '1.02', r'\ding{55}', r'\ding{51}', r'\ding{51}', '+9.3',  '+7.3'),
    ('2wiki-bridge-comp',      r'8{,}000',  '1.66', r'\ding{51}', r'\ding{55}', r'\ding{51}', '$-$1.2','$-$0.1'),
    (r'2wiki-simple (mixed)',  r'15{,}000', '1.19', r'\ding{55}', r'$\sim$',   r'$\sim$',   '+3.7',  'n/a'),
]
lines = [
    r'\begin{table}[t]',
    r'\centering',
    r'\caption{Dataset analysis: three conditions for QueryDriven KB curation effectiveness. '
    r'\textbf{Cond-A}: $q/\mathrm{SF} \geq 1.3$ (enough cross-query demand signal). '
    r'\textbf{Cond-B}: query text directly names SF entities (embedding alignment $\geq 0.6$). '
    r'\textbf{Cond-C}: pool contains many distractor documents. '
    r'$\Delta$ = QDC minus Static KB coverage H2 (pp). '
    r'$^*$TriviaQA pool has no distractors, so Random-FIFO also picks useful docs by chance. '
    r'Requires \texttt{\textbackslash usepackage\{pifont\}}.}',
    r'\label{tab:dataset_analysis}',
    r'\setlength{\tabcolsep}{5pt}',
    r'\begin{tabular}{lrrccccc}',
    r'\toprule',
    r'Dataset & $n$ & $q/\mathrm{SF}$ & Cond-A & Cond-B & Cond-C'
    r' & $\Delta$ sudden & $\Delta$ gradual \\',
    r'\midrule',
]
for row in ROWS:
    if row is None:
        lines.append(r'\midrule')
    else:
        lines.append('  ' + ' & '.join(row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
tex_path = OUT / 'dataset_analysis_table.tex'
tex_path.write_text('\n'.join(lines) + '\n')
print(f'Saved {tex_path}')

# ════════════════════════════════════════════════════════════════════════════
# ── Console summary ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

print()
print('='*75)
print(f'{"Dataset":<28} {"q/SF":>6} {"Align":>6}  {"Cond-A":^6} {"Cond-B":^6} {"Cond-C":^6}  {"Δsudden":>8} {"Δgradual":>9}')
print('-'*75)
for ds_id, label, qsf, align, fail in META:
    vs = deltas.get((ds_id,'sudden'))
    vg = deltas.get((ds_id,'gradual'))
    ca = '✓' if qsf  >= 1.3 else '✗'
    cb = '✓' if align >= 0.6 else '✗'
    cc = '✓' if fail != 'Cond-C' and fail is not None or fail is None else '✗'
    # Cond-C: None=success(has distractors), 'Cond-C'=TriviaQA no distractors
    cc = '✗' if fail == 'Cond-C' else '✓'
    clean = label.replace('\n',' ')
    print(f'{clean:<28} {qsf:>6.2f} {align:>6.2f}  {ca:^6} {cb:^6} {cc:^6}  '
          f'{(f"{vs:+.1f}pp" if vs is not None else "n/a"):>8} '
          f'{(f"{vg:+.1f}pp" if vg is not None else "n/a"):>9}')
print('='*75)
