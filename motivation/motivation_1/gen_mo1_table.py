#!/usr/bin/env python3
"""Generate LaTeX table for Mo1 method comparison (HotpotQA-comparison)."""
import json, numpy as np
from pathlib import Path

DATA = Path(__file__).parent / 'data'

s_res = json.load(open(DATA / 'results_100w_sudden.json'))
g_res = json.load(open(DATA / 'results_100w_gradual.json'))
sk = list(s_res.keys())[0]; gk = list(g_res.keys())[0]

DISPLAY = {
    'Static':             ('Static (no update)',      'none'),
    'RandomFIFO':         ('Random-FIFO',             'random'),
    'DocArrival':         ('Doc-Arrival',             'doc-side'),
    'KnowledgeEdit':      ('Knowledge-Edit',          'doc-side'),
    'LogDrivenArrival':   ('Log-Driven',              'query (lagged)'),
    'QueryDrivenCluster': ('\\textbf{QueryDriven (ours)}', 'query (live)'),
    'OnDemandFetch':      ('On-Demand Fetch$^\\dagger$', 'online'),
    'Oracle':             ('Oracle$^\\ddagger$',      '---'),
}

rows = []
for key, (label, cost) in DISPLAY.items():
    if key not in s_res[sk]['summary']: continue
    s = np.array(s_res[sk]['summary'][key]['cov_per_window'])
    g = np.array(g_res[gk]['summary'][key]['cov_per_window'])
    h1  = np.mean(s[:50])
    sh2 = np.mean(s[50:])
    gh2 = np.mean(g[50:])
    rows.append((label, cost, h1, sh2, gh2, key))

tex = r"""\begin{table}[t]
\centering
\caption{KB coverage (\%) under knowledge drift on HotpotQA-comparison.
H1/H2 = pre/post-drift mean over 50 windows each.
$^\dagger$On-Demand fetches fresh documents at query time (no persistent KB).
$^\ddagger$Oracle rebuilds KB from gold supporting facts each window.}
\label{tab:mo1_comparison}
\small
\begin{tabular}{llccc}
\toprule
\textbf{Method} & \textbf{Update signal} & \textbf{H1 mean} & \textbf{Sudden H2} & \textbf{Gradual H2} \\
\midrule
"""
for label, cost, h1, sh2, gh2, key in rows:
    bold_open  = r'\textbf{' if key == 'QueryDrivenCluster' else ''
    bold_close = r'}' if key == 'QueryDrivenCluster' else ''
    if key in ('OnDemandFetch', 'Oracle'):
        tex += r'\midrule' + '\n'
    tex += f'{label} & {cost} & {bold_open}{h1:.1f}{bold_close} & {bold_open}{sh2:.1f}{bold_close} & {bold_open}{gh2:.1f}{bold_close} \\\\\n'

tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

out = Path(__file__).parent / 'mo1_method_comparison_table.tex'
out.write_text(tex)
print(tex)
print(f'\nSaved {out}')
