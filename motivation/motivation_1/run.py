#!/usr/bin/env python3
"""
Motivation 1 — Single-hop sanity runner.

Same 8 strategies as Motivation 2, but on a SINGLE-HOP query distribution
(HotpotQA 'comparison'). Primary metric = KB COVERAGE (retriever-orthogonal):

  per-window cov:  |effective_KB ∩ window_gold_SFs| / |window_gold_SFs|
  cumulative cov:  |effective_KB ∩ all_seen_gold_so_far| / |all_seen|

Recall@K is also recorded for completeness.

Usage:
  python run.py --n-windows 50 --window-size 50 --drift sudden
  python run.py --n-windows 50 --window-size 50 --drift gradual
"""
import sys, json, time, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DATASET_CONFIGS, STRATEGY_ORDER, STRATEGY_LABELS,
                    K_LIST, DATA_DIR, FIG_DIR, log)
from loaders import LOADERS
from strategies import STRATEGY_FACTORIES
from utils import (compute_embeddings, cluster_and_build_stream,
                   head_biased_init_kb, recall_at_k)


def run_dataset(ds_name, cfg, strategies_to_run, drift_mode='sudden',
                loader_name=None, n_source=None):
    t0 = time.time()
    nw = cfg['n_windows']; ws = cfg['window_size']
    lname = loader_name or ds_name
    loader = LOADERS[lname]
    doc_pool, queries, title_to_idx = loader(n_source=n_source or cfg.get('n_source'))

    tag = f'{ds_name}_{nw}w_{ws}s'
    doc_embs, query_embs = compute_embeddings(doc_pool, queries, tag=tag)
    stream, centroids, head_set = cluster_and_build_stream(
        queries, query_embs, cfg, drift_mode=drift_mode)

    t2l = {d['title']: i for i, d in enumerate(doc_pool)}
    head_ctx = set()
    for q in stream:
        if not q.get('is_tail', True):
            for t in q.get('ctx_titles', []):
                if t in t2l:
                    head_ctx.add(t2l[t])
    kb_head_mult = cfg.get('kb_head_mult', 1.2)
    kb_budget = max(300, int(round(len(head_ctx) * kb_head_mult / 50)) * 50)
    init_kb = head_biased_init_kb(
        doc_pool, doc_embs, centroids, head_set, kb_budget, stream)
    d2p = {d['doc_id']: i for i, d in enumerate(doc_pool)}

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

    results = {n: {f'recall@{k}': [] for k in K_LIST} for n in strategies_to_run}
    cov_per_window = {n: [] for n in strategies_to_run}

    for w in range(nw):
        wq = stream[w * ws:(w + 1) * ws]
        if len(wq) == 0:
            log.warning(f"[{ds_name}] W{w+1}: empty window, stopping")
            break
        wqe = np.array([query_embs[q['qidx']] for q in wq])
        if wqe.ndim == 1:
            wqe = wqe.reshape(1, -1)
        window_gold_did = set()
        for q in wq:
            for t in q.get('sf_titles', []):
                if t in title_to_idx:
                    window_gold_did.add(doc_pool[title_to_idx[t]]['doc_id'])
        for name in strategies_to_run:
            s = strategies[name]
            if name == 'Oracle':
                s.step(wq, wqe, w)
            effective_kb = (s.get_effective_kb(wq, wqe)
                            if hasattr(s, 'get_effective_kb') else s.kb)
            r = recall_at_k(effective_kb, wq, d2p, doc_embs, title_to_idx, query_embs)
            for k in K_LIST:
                results[name][f'recall@{k}'].append(r[k])
            cov = len(effective_kb & window_gold_did) / max(1, len(window_gold_did))
            cov_per_window[name].append(cov)
            if name != 'Oracle':
                s.step(wq, wqe, w)
        if w % 5 == 0 or w == nw - 1:
            cv = {n: f"{cov_per_window[n][-1]*100:.1f}%" for n in strategies_to_run}
            log.info(f"[{ds_name}] W{w+1}/{nw} Cov: {cv}")

    actual_nw = len(results[strategies_to_run[0]]['recall@5'])
    elapsed = time.time() - t0
    half = actual_nw // 2
    summary = {}
    for name in strategies_to_run:
        r = results[name]; s = {}
        for k in K_LIST:
            key = f'recall@{k}'
            vals = r[key]
            s[f'{key}_h1'] = round(float(np.mean(vals[:half])) * 100, 1)
            s[f'{key}_h2'] = round(float(np.mean(vals[half:])) * 100, 1)
            s[f'{key}_per_window'] = [round(x * 100, 2) for x in vals]
        cv = cov_per_window[name]
        s['cov_h1'] = round(float(np.mean(cv[:half])) * 100, 1)
        s['cov_h2'] = round(float(np.mean(cv[half:])) * 100, 1)
        s['cov_per_window'] = [round(x * 100, 2) for x in cv]
        s['update_cost']           = strategies[name].update_cost
        s['maint_retrieval_cost']  = strategies[name].maint_retrieval_cost
        s['serve_retrieval_cost']  = strategies[name].serve_retrieval_cost
        s['retrieval_cost']        = strategies[name].retrieval_cost
        s['cost']                  = strategies[name].cost
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
    print("\n" + "=" * 130)
    print("  KB Coverage of Window Gold SFs Under Single-Hop Query Drift")
    print("=" * 130)
    for ds, res in all_results.items():
        cfg = res['config']
        print(f"\n{'_'*130}")
        print(f"  {ds}  |  pool={cfg['pool_size']:,}  KB={cfg['kb_budget']:,}  "
              f"stream={cfg['n_windows']}x{cfg['window_size']}  "
              f"elapsed={res['elapsed']:.0f}s")
        print(f"{'_'*130}")
        hdr = (f"{'Strategy':>18s} |"
               f" {'Cov H1':>6s} {'Cov H2':>6s} {'D':>5s} |"
               f" {'R@5 H1':>6s} {'R@5 H2':>6s} |"
               f" {'Writes':>6s} {'MaintR':>7s} {'ServeR':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for name in strategies_to_run:
            if name not in res['summary']:
                continue
            s = res['summary'][name]
            c1, c2 = s['cov_h1'], s['cov_h2']
            r1, r2 = s['recall@5_h1'], s['recall@5_h2']
            print(f"{name:>18s} |"
                  f" {c1:>5.1f}% {c2:>5.1f}% {c2-c1:>+5.1f} |"
                  f" {r1:>5.1f}% {r2:>5.1f}% |"
                  f" {s['update_cost']:>6d}"
                  f" {s['maint_retrieval_cost']:>7d}"
                  f" {s['serve_retrieval_cost']:>7d}")


def generate_figures(all_results, strategies_to_run, suffix=''):
    """Two-row coverage figure: per-window + cumulative."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 12,
        'axes.titlesize': 12, 'legend.fontsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    palette = {
        'Static':           ('#7F7F7F', '-',  'o'),
        'RandomFIFO':       ('#9467BD', '-.', '^'),
        'DocArrival':       ('#8C564B', ':',  's'),
        'KnowledgeEdit':    ('#E377C2', '-.', 'D'),
        'OnDemandFetch':    ('#17BECF', '--', 'v'),
        'LogDrivenArrival': ('#BCBD22', ':',  'P'),
        'QueryDriven':      ('#1F77B4', '-',  '*'),
        'Oracle':           ('#D62728', '-',  None),
    }
    datasets = list(all_results.keys())
    n_ds = max(len(datasets), 1)
    fig, axes = plt.subplots(1, n_ds, figsize=(5.2 * n_ds, 3.8), squeeze=False)

    for col, ds in enumerate(datasets):
        res = all_results[ds]; cfg = res['config']
        nw = cfg['n_windows']; half = nw // 2
        x = np.arange(1, nw + 1)
        ax = axes[0, col]
        for name in strategies_to_run:
            if name not in res['summary']:
                continue
            color, ls, marker = palette.get(name, ('gray', '-', None))
            cov = res['summary'][name]['cov_per_window']
            lw = 2.4 if name == 'Oracle' else (2.0 if name == 'QueryDriven' else 1.4)
            alpha = 1.0 if name in ('Oracle', 'QueryDriven') else 0.85
            zorder = 5 if name in ('Oracle', 'QueryDriven') else 3
            mark_every = max(1, nw // 12)
            ax.plot(x, cov, color=color, linestyle=ls, marker=marker,
                    linewidth=lw, alpha=alpha, markersize=5,
                    markevery=mark_every, zorder=zorder,
                    label=STRATEGY_LABELS.get(name, name))
        ax.axvline(half + 0.5, color='#444444', ls='--', lw=0.9, alpha=0.6)
        ax.grid(True, axis='y', alpha=0.25, linestyle=':')
        ax.set_xlim(1, nw); ax.set_ylim(0, 100)
        ax.set_title(
            f'{ds}  pool={cfg["pool_size"]:,}  KB={cfg["kb_budget"]:,}  '
            f'{nw}×{cfg["window_size"]}', fontsize=11)
        if col == 0:
            ax.set_ylabel('Per-window SF coverage (%)')
            ax.text(half + 0.5, 95, '  drift onset', va='top',
                    ha='left', fontsize=9, color='#444444')
        ax.set_xlabel('Window index')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / f'coverage{suffix}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'coverage{suffix}.png', dpi=200, bbox_inches='tight')
    log.info(f"Saved {FIG_DIR / f'coverage{suffix}.pdf'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Motivation 1 (single-hop) runner')
    parser.add_argument('--n-windows', type=int, default=50)
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--drift', choices=['sudden', 'gradual'], default='sudden')
    parser.add_argument('--n-source', type=int, default=None)
    parser.add_argument('--datasets', nargs='+', default=['hotpotqa_comparison'])
    parser.add_argument('--strategies', nargs='+', default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    strategies_to_run = args.strategies or STRATEGY_ORDER
    all_results = {}
    for ds in args.datasets:
        base_cfg = DATASET_CONFIGS[ds].copy()
        base_cfg['n_windows'] = args.n_windows
        base_cfg['window_size'] = args.window_size
        log.info(f"\n{'='*60}\n  Running: {ds} ({args.drift})\n{'='*60}")
        all_results[ds] = run_dataset(
            ds, base_cfg, strategies_to_run,
            drift_mode=args.drift,
            loader_name=ds, n_source=args.n_source)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or f'results_{args.n_windows}w_{args.drift}.json'
    out_path = DATA_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_path}")
    print_summary(all_results, strategies_to_run)
    suffix = f'_{args.n_windows}w_{args.drift}'
    generate_figures(all_results, strategies_to_run, suffix=suffix)


if __name__ == '__main__':
    main()
