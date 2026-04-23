#!/usr/bin/env python3
"""
Unified experiment runner for the motivation study.

Usage:
  python run.py                          # default 20-window sudden drift
  python run.py --n-windows 50 --window-size 50 --expanded
  python run.py --drift gradual
  python run.py --datasets hotpotqa musique
  python run.py --strategies Static QueryDrivenCluster Oracle

Supports:
  - Configurable window count and size
  - Expanded dataset loaders (full train+dev for 2Wiki and MuSiQue)
  - Sudden or gradual drift modes
  - Selective dataset and strategy execution
"""
import os, sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DATASET_CONFIGS, STRATEGY_ORDER, STRATEGY_LABELS,
                    STRATEGY_STYLES, K_LIST, DATA_DIR, FIG_DIR, log)
from loaders import LOADERS
from strategies import STRATEGY_FACTORIES
from utils import (compute_embeddings, cluster_and_build_stream,
                   focus_pool, head_biased_init_kb, recall_at_k)


def run_dataset(ds_name, cfg, strategies_to_run, drift_mode='sudden',
                loader_name=None, n_source=None, kb_budget_override=None):
    """Run one dataset experiment.

    Args:
        ds_name:          dataset key (for logging / output)
        cfg:              dict with n_windows, window_size, n_clusters, etc.
        strategies_to_run: list of strategy names to evaluate
        drift_mode:       'sudden' or 'gradual'
        loader_name:      override loader key (e.g. '2wiki_expanded')
        n_source:         override n_source for expanded loaders

    Returns:
        dict with config, per-strategy summary, and per-window results.
    """
    t0 = time.time()
    nw = cfg['n_windows']
    ws = cfg['window_size']

    # ── Load data ──
    lname = loader_name or ds_name
    loader = LOADERS[lname]
    if n_source and lname in ('musique_expanded', '2wiki_expanded', 'hotpotqa_expanded'):
        doc_pool, queries, title_to_idx = loader(n_source=n_source)
    else:
        doc_pool, queries, title_to_idx = loader()

    # ── Embeddings ──
    tag = f'{ds_name}_{nw}w_{ws}s'
    doc_embs, query_embs = compute_embeddings(doc_pool, queries, tag=tag)

    # ── Stream ──
    stream, centroids, head_set = cluster_and_build_stream(
        queries, query_embs, cfg, drift_mode=drift_mode)

    # ── KB init ──
    # KB budget = head_context_docs * kb_head_mult.
    # This standardises across datasets with very different pool sizes:
    # we ensure the KB can hold ~all head-context docs (so H1 isn't
    # capacity-bottlenecked) while leaving little slack for tail docs
    # (so H2 forces real strategy work). With mult=1.2 the init KB is
    # ~83% head SF coverage and ~10-20% tail coverage.
    t2l = {d['title']: i for i, d in enumerate(doc_pool)}
    head_ctx = set()
    for q in stream:
        if not q.get('is_tail', True):
            for t in q.get('ctx_titles', []):
                if t in t2l:
                    head_ctx.add(t2l[t])
    kb_head_mult = cfg.get('kb_head_mult', 1.2)
    kb_budget = max(300, int(round(len(head_ctx) * kb_head_mult / 50)) * 50)
    if kb_budget_override is not None:
        log.info(f"[{ds_name}] KB budget override: {kb_budget} -> {kb_budget_override} (head_ctx={len(head_ctx)})")
        kb_budget = kb_budget_override
    init_kb = head_biased_init_kb(
        doc_pool, doc_embs, centroids, head_set, kb_budget, stream)
    d2p = {d['doc_id']: i for i, d in enumerate(doc_pool)}

    # ── Strategies ──
    strategies = {}
    for name in strategies_to_run:
        s = STRATEGY_FACTORIES[name](doc_pool, doc_embs, title_to_idx)
        s.set_kb(set(init_kb))
        if hasattr(s, '_half'):
            s._half = nw // 2
        if hasattr(s, '_stream'):
            s._stream = stream
            s._query_embs = query_embs
        strategies[name] = s

    # ── Evaluate ──
    results = {n: {f'recall@{k}': [] for k in K_LIST} for n in strategies_to_run}
    kb_coverage = {n: [] for n in strategies_to_run}  # L1: fraction of window's gold SFs that sit in KB
    kb_coverage_cum = {n: [] for n in strategies_to_run}  # cumulative gold so far
    seen_gold = {n: set() for n in strategies_to_run}
    for w in range(nw):
        wq = stream[w * ws:(w + 1) * ws]
        if len(wq) == 0:
            log.warning(f"[{ds_name}] W{w+1}: empty window, stopping")
            break
        wqe = np.array([query_embs[q['qidx']] for q in wq])
        if wqe.ndim == 1:
            wqe = wqe.reshape(1, -1)
        # gold SF doc_ids for this window
        window_gold_did = set()
        for q in wq:
            for t in q.get('sf_titles', []):
                if t in title_to_idx:
                    window_gold_did.add(doc_pool[title_to_idx[t]]['doc_id'])
        for name in strategies_to_run:
            s = strategies[name]
            s.prepare_window(wq, wqe, w)
            # Oracle is non-causal: rebuild KB BEFORE evaluation so it represents
            # the per-window upper bound for the queries we are about to score.
            if name == 'Oracle':
                s.step(wq, wqe, w)
            effective_kb = s.get_effective_kb(wq, wqe) if hasattr(s, 'get_effective_kb') else s.kb
            r = recall_at_k(effective_kb, wq, d2p, doc_embs, title_to_idx, query_embs)
            for k in K_LIST:
                results[name][f'recall@{k}'].append(r[k])
            cov = len(effective_kb & window_gold_did) / max(1, len(window_gold_did))
            kb_coverage[name].append(cov)
            seen_gold[name] |= window_gold_did
            cum = len(effective_kb & seen_gold[name]) / max(1, len(seen_gold[name]))
            kb_coverage_cum[name].append(cum)
            if name != 'Oracle':
                s.step(wq, wqe, w)
        if w % 5 == 0 or w == nw - 1:
            r5 = {n: f"{results[n]['recall@5'][-1]*100:.1f}%"
                  for n in strategies_to_run}
            log.info(f"[{ds_name}] W{w+1}/{nw} R@5: {r5}")

    # ── Summarise ──
    actual_nw = len(results[strategies_to_run[0]]['recall@5'])
    elapsed = time.time() - t0
    half = actual_nw // 2
    summary = {}
    for name in strategies_to_run:
        r = results[name]
        s = {}
        for k in K_LIST:
            key = f'recall@{k}'
            vals = r[key]
            s[f'{key}_h1'] = round(float(np.mean(vals[:half])) * 100, 1)
            s[f'{key}_h2'] = round(float(np.mean(vals[half:])) * 100, 1)
            n_h2 = len(vals[half:])
            if n_h2 >= 5:
                s[f'{key}_h2_first5'] = round(
                    float(np.mean(vals[half:half+5])) * 100, 1)
                s[f'{key}_h2_last5'] = round(
                    float(np.mean(vals[-5:])) * 100, 1)
            s[f'{key}_per_window'] = [round(x * 100, 2) for x in vals]
        cov_vals = kb_coverage[name]
        s['kb_coverage_h1'] = round(float(np.mean(cov_vals[:half])) * 100, 1)
        s['kb_coverage_h2'] = round(float(np.mean(cov_vals[half:])) * 100, 1)
        s['kb_coverage_per_window'] = [round(x * 100, 2) for x in cov_vals]
        cum_vals = kb_coverage_cum[name]
        s['kb_coverage_cumulative_h1'] = round(float(np.mean(cum_vals[:half])) * 100, 1)
        s['kb_coverage_cumulative_h2'] = round(float(np.mean(cum_vals[half:])) * 100, 1)
        s['kb_coverage_cumulative_per_window'] = [round(x * 100, 2) for x in cum_vals]
        s['update_cost'] = strategies[name].update_cost
        s['maint_retrieval_cost'] = strategies[name].maint_retrieval_cost
        s['serve_retrieval_cost'] = strategies[name].serve_retrieval_cost
        s['retrieval_cost'] = strategies[name].retrieval_cost
        s['cost'] = strategies[name].cost
        summary[name] = s
    return {
        'dataset': ds_name,
        'config': {
            'kb_budget': kb_budget, 'pool_size': len(doc_pool),
            'n_windows': actual_nw, 'window_size': ws,
            'n_queries_available': len(queries),
            'drift': drift_mode, 'k_list': K_LIST,
        },
        'summary': summary, 'elapsed': round(elapsed, 1),
    }


def print_summary(all_results, strategies_to_run):
    ds_labels = {'hotpotqa': 'HotpotQA',
                 '2wikimultihopqa': '2WikiMultihopQA',
                 'musique': 'MuSiQue'}
    print("\n" + "=" * 115)
    print("  Recall@K Under Query Distribution Drift")
    print("=" * 115)
    for ds, res in all_results.items():
        cfg = res['config']
        print(f"\n{'_'*115}")
        print(f"  {ds_labels.get(ds, ds)}  |  pool={cfg['pool_size']}  "
              f"KB={cfg['kb_budget']}  "
              f"stream={cfg['n_windows']}x{cfg['window_size']}  "
              f"elapsed={res['elapsed']:.0f}s")
        print(f"{'_'*115}")
        hdr = (f"{'Strategy':>18s} |"
               f" {'R@5 H1':>6s} {'R@5 H2':>6s} {'D':>5s} |"
               f" {'Cov H1':>6s} {'Cov H2':>6s} {'D':>5s} |"
               f" {'Writes':>6s} {'MaintR':>7s} {'ServeR':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for name in strategies_to_run:
            if name not in res['summary'] or name == 'Oracle':
                continue
            s = res['summary'][name]
            h1 = s['recall@5_h1']; h2 = s['recall@5_h2']
            c1 = s['kb_coverage_h1']; c2 = s['kb_coverage_h2']
            line = (f"{name:>18s} |"
                    f" {h1:>5.1f}% {h2:>5.1f}% {h2-h1:>+5.1f} |"
                    f" {c1:>5.1f}% {c2:>5.1f}% {c2-c1:>+5.1f} |"
                    f" {s['update_cost']:>6d}"
                    f" {s['maint_retrieval_cost']:>7d}"
                    f" {s['serve_retrieval_cost']:>7d}")
            print(line)


def generate_figures(all_results, strategies_to_run, suffix=''):
    """Per-dataset KB Coverage + Recall@5 figure over windows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = list(all_results.keys())
    n_ds = len(datasets)
    ds_labels = {'hotpotqa': 'HotpotQA',
                 '2wikimultihopqa': '2WikiMultihopQA',
                 'musique': 'MuSiQue'}

    palette = {
        'Static':           ('#7F7F7F', '-',  'o'),
        'RandomFIFO':       ('#9467BD', '-.', '^'),
        'DocArrival':       ('#8C564B', ':',  's'),
        'KnowledgeEdit':    ('#E377C2', '-.', 'D'),
        'OnDemandFetch':    ('#17BECF', '--', 'v'),
        'LogDrivenArrival': ('#BCBD22', ':',  'P'),
        'QueryDrivenCluster': ('#10B981', '-',  'D'),
        'Oracle':           ('#D62728', '-',  None),
    }

    fig, axes = plt.subplots(n_ds, 2, figsize=(13, 3.4 * n_ds),
                             sharex=True, squeeze=False)
    emphasis = {'Oracle', 'QueryDrivenCluster'}

    for row, ds in enumerate(datasets):
        res = all_results[ds]
        cfg = res['config']
        nw = cfg['n_windows']
        half = nw // 2
        x = np.arange(1, nw + 1)

        for col, (series_key, ylabel) in enumerate((
            ('kb_coverage_per_window', 'KB Coverage (%)'),
            ('recall@5_per_window', 'Recall@5 (%)'),
        )):
            ax = axes[row, col]
            for name in strategies_to_run:
                if name not in res['summary']:
                    continue
                color, ls, marker = palette.get(name, ('gray', '-', None))
                values = res['summary'][name][series_key]
                lw = 2.4 if name == 'Oracle' else (2.0 if name == 'QueryDrivenCluster' else 1.4)
                alpha = 1.0 if name in emphasis else 0.85
                zorder = 5 if name in emphasis else 3
                mark_every = max(1, nw // 12)
                ax.plot(x, values, color=color, linestyle=ls, marker=marker,
                        linewidth=lw, alpha=alpha, markersize=5,
                        markevery=mark_every, zorder=zorder,
                        label=STRATEGY_LABELS.get(name, name))

            ax.axvline(half + 0.5, color='#444444', ls='--', lw=0.9, alpha=0.6)
            if col == 0:
                ax.text(half + 0.5, 95, '  drift onset', va='top', ha='left',
                        fontsize=9, color='#444444')
            ax.grid(True, axis='y', alpha=0.25, linestyle=':')
            ax.set_xlim(1, nw)
            ax.set_ylim(0, 100)
            ax.set_ylabel(ylabel)
            if col == 0:
                ax.set_title(
                    f'{ds_labels.get(ds, ds)}\n'
                    f'pool={cfg["pool_size"]:,}  KB={cfg["kb_budget"]:,}  '
                    f'{nw}×{cfg["window_size"]}',
                    fontsize=11)
            else:
                ax.set_title('Recall@5', fontsize=11)
            if row == n_ds - 1:
                ax.set_xlabel('Window index')

    drift_label = None
    if 'sudden' in suffix:
        drift_label = 'Sudden Drift'
    elif 'gradual' in suffix:
        drift_label = 'Gradual Drift'
    if drift_label:
        fig.suptitle('Motivation 2 - ' + drift_label, fontsize=14, y=1.01)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    top = 0.97 if drift_label else 1.00
    fig.tight_layout(rect=[0, 0.10, 1, top])
    fig.savefig(FIG_DIR / f'coverage_drift{suffix}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'coverage_drift{suffix}.png', dpi=200,
                bbox_inches='tight')
    log.info(f"Saved {FIG_DIR / f'coverage_drift{suffix}.pdf'}")
    plt.close(fig)
def main():
    parser = argparse.ArgumentParser(description='Motivation experiment runner')
    parser.add_argument('--n-windows', type=int, default=None)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--drift', choices=['sudden', 'gradual'], default='sudden')
    parser.add_argument('--expanded', action='store_true',
                        help='Use expanded loaders for 2Wiki and MuSiQue')
    parser.add_argument('--n-source', type=int, default=3500)
    parser.add_argument('--datasets', nargs='+',
                        default=['hotpotqa', '2wikimultihopqa', 'musique'])
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Strategies to run (default: all from STRATEGY_ORDER)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON filename (auto-generated if omitted)')
    parser.add_argument('--kb-budget', type=int, default=None,
                        help='Force absolute KB budget (overrides kb_head_mult formula)')
    args = parser.parse_args()

    strategies_to_run = args.strategies or STRATEGY_ORDER

    all_results = {}
    for ds in args.datasets:
        base_cfg = DATASET_CONFIGS.get(ds, DATASET_CONFIGS['musique']).copy()
        if args.n_windows:
            base_cfg['n_windows'] = args.n_windows
        if args.window_size:
            base_cfg['window_size'] = args.window_size

        loader_name = None
        n_source = None
        if args.expanded:
            loader_name = {
                'hotpotqa':        'hotpotqa_expanded',
                '2wikimultihopqa': '2wiki_expanded',
                'musique':         'musique_expanded',
            }[ds]
            n_source = args.n_source
            base_cfg['n_source'] = args.n_source

        log.info(f"\n{'='*60}\n  Running: {ds} ({args.drift})\n{'='*60}")
        all_results[ds] = run_dataset(
            ds, base_cfg, strategies_to_run,
            drift_mode=args.drift,
            loader_name=loader_name,
            n_source=n_source,
            kb_budget_override=args.kb_budget)

    # ── Save ──
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_name = args.output
    else:
        nw = args.n_windows or 20
        out_name = f'results_{nw}w_{args.drift}.json'
    out_path = DATA_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_path}")

    print_summary(all_results, strategies_to_run)

    suffix = f'_{args.n_windows or 20}w_{args.drift}'
    generate_figures(all_results, strategies_to_run, suffix=suffix)


if __name__ == '__main__':
    main()
