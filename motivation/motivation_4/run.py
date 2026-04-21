#!/usr/bin/env python3
"""
Unified experiment runner for the motivation study.

Usage:
  python run.py                          # default 20-window sudden drift
  python run.py --n-windows 50 --window-size 50 --expanded
  python run.py --drift gradual
  python run.py --datasets hotpotqa musique
  python run.py --strategies Static QueryDriven Oracle

Supports:
  - Configurable window count and size
  - Expanded dataset loaders (full train+dev for 2Wiki and MuSiQue)
  - Sudden or gradual drift modes
  - Selective dataset and strategy execution
"""
import sys, json, time, argparse
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
                loader_name=None, n_source=None):
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
    if n_source and lname in ('musique_expanded', '2wiki_expanded'):
        doc_pool, queries, title_to_idx = loader(n_source=n_source)
    else:
        doc_pool, queries, title_to_idx = loader()

    # ── Embeddings ──
    tag = f'{ds_name}_{nw}w_{ws}s'
    doc_embs, query_embs = compute_embeddings(doc_pool, queries, tag=tag)

    # ── Stream ──
    stream, centroids, head_set = cluster_and_build_stream(
        queries, query_embs, cfg, drift_mode=drift_mode)

    if cfg.get('use_focus_pool'):
        doc_pool, title_to_idx, doc_embs = focus_pool(
            doc_pool, title_to_idx, doc_embs, stream)

    # ── KB init ──
    kb_budget = max(300, round(len(doc_pool) * cfg['kb_budget_pct'] / 50) * 50)
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
    for w in range(nw):
        wq = stream[w * ws:(w + 1) * ws]
        if len(wq) == 0:
            log.warning(f"[{ds_name}] W{w+1}: empty window, stopping")
            break
        wqe = np.array([query_embs[q['qidx']] for q in wq])
        if wqe.ndim == 1:
            wqe = wqe.reshape(1, -1)
        for name in strategies_to_run:
            s = strategies[name]
            effective_kb = s.get_effective_kb(wq, wqe) if hasattr(s, 'get_effective_kb') else s.kb
            r = recall_at_k(effective_kb, wq, d2p, doc_embs, title_to_idx, query_embs)
            for k in K_LIST:
                results[name][f'recall@{k}'].append(r[k])
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
        s['update_cost'] = strategies[name].update_cost
        s['retrieval_cost'] = strategies[name].retrieval_cost
        s['cost'] = strategies[name].cost  # backward-compat: total
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
        hdr = f"{'Strategy':>20s} |"
        for k in K_LIST:
            hdr += f" {'R@'+str(k)+' H1':>8s} {'R@'+str(k)+' H2':>8s} {'D':>6s} |"
        hdr += f" {'KB-Writes':>9s} {'Retrievals':>10s}"
        print(hdr)
        print("-" * len(hdr))
        for name in strategies_to_run:
            if name not in res['summary']:
                continue
            s = res['summary'][name]
            line = f"{name:>20s} |"
            for k in K_LIST:
                h1 = s[f'recall@{k}_h1']
                h2 = s[f'recall@{k}_h2']
                d = h2 - h1
                line += f" {h1:>7.1f}% {h2:>7.1f}% {d:>+5.1f} |"
            line += f" {s['update_cost']:>9d} {s['retrieval_cost']:>10d}"
            print(line)


def generate_figures(all_results, strategies_to_run, suffix=''):
    """Two-row figure per dataset:
      Row 1: full y-range with Oracle ceiling shaded (shows total gap to oracle).
      Row 2: zoomed view excluding Oracle (reveals differences among practical
             strategies, especially QueryDriven vs supply-side baselines).
    Oracle is plotted across BOTH H1 and H2 so the ceiling is always visible.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = list(all_results.keys())
    n_ds = len(datasets)
    ds_labels = {'hotpotqa': 'HotpotQA',
                 '2wikimultihopqa': '2WikiMultihopQA',
                 'musique': 'MuSiQue'}

    fig, axes = plt.subplots(2, n_ds, figsize=(5.5 * n_ds, 7.5), sharex='col')
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    practical = [n for n in strategies_to_run if n != 'Oracle']

    for col, ds in enumerate(datasets):
        res = all_results[ds]
        cfg = res['config']
        nw = cfg['n_windows']
        half = nw // 2
        x = np.arange(1, nw + 1)

        ax_full = axes[0, col]
        ax_zoom = axes[1, col]

        # Oracle ceiling: shade between Static (floor at H2) and Oracle on full view
        oracle_y = (res['summary'].get('Oracle') or {}).get('recall@5_per_window')
        for name in strategies_to_run:
            if name not in res['summary']:
                continue
            st = STRATEGY_STYLES.get(name, {'color': 'gray', 'marker': '.', 'ls': '-'})
            y = res['summary'][name]['recall@5_per_window']
            lw = 2.6 if name == 'QueryDriven' else (2.0 if name == 'Oracle' else 1.6)
            alpha = 1.0 if name in ('QueryDriven', 'Oracle') else 0.85
            ax_full.plot(x, y, label=STRATEGY_LABELS.get(name, name),
                         color=st['color'], marker=st['marker'],
                         linestyle=st['ls'], linewidth=lw, alpha=alpha,
                         markersize=5, markevery=max(1, nw // 20))
            if name != 'Oracle':
                ax_zoom.plot(x, y, label=STRATEGY_LABELS.get(name, name),
                             color=st['color'], marker=st['marker'],
                             linestyle=st['ls'], linewidth=lw, alpha=alpha,
                             markersize=5, markevery=max(1, nw // 20))

        # Oracle ceiling band on full view
        if oracle_y is not None:
            ax_full.fill_between(x, 0, oracle_y, color='#DC2626',
                                 alpha=0.06, zorder=0)

        for ax in (ax_full, ax_zoom):
            ax.axvline(x=half + 0.5, color='gray', ls=':', lw=1, alpha=0.6)
            ax.grid(True, alpha=0.3)

        ax_full.set_title(
            f'{ds_labels.get(ds, ds)}\n'
            f'(pool={cfg["pool_size"]}, KB={cfg["kb_budget"]}, '
            f'stream={nw}x{cfg["window_size"]})', fontsize=11)
        if col == 0:
            ax_full.set_ylabel('Recall@5 (%) — full range', fontsize=10)
            ax_zoom.set_ylabel('Recall@5 (%) — practical (no Oracle)', fontsize=10)
        ax_zoom.set_xlabel('Window', fontsize=10)

        # Zoom y-limit: tight around practical strategies' max
        practical_vals = []
        for n in practical:
            if n in res['summary']:
                practical_vals.extend(res['summary'][n]['recall@5_per_window'])
        if practical_vals:
            ymax = max(practical_vals) * 1.15 + 1
            ax_zoom.set_ylim(0, ymax)

    # Single legend below
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_DIR / f'recall_drift{suffix}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'recall_drift{suffix}.png', dpi=150, bbox_inches='tight')
    log.info(f"Saved {FIG_DIR / f'recall_drift{suffix}.pdf'}")
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
            if ds == '2wikimultihopqa':
                loader_name = '2wiki_expanded'
                n_source = args.n_source
                base_cfg['use_focus_pool'] = False
            elif ds == 'musique':
                loader_name = 'musique_expanded'
                n_source = args.n_source
                base_cfg['use_focus_pool'] = False
            if args.n_windows:
                base_cfg['n_source'] = args.n_source

        log.info(f"\n{'='*60}\n  Running: {ds} ({args.drift})\n{'='*60}")
        all_results[ds] = run_dataset(
            ds, base_cfg, strategies_to_run,
            drift_mode=args.drift,
            loader_name=loader_name,
            n_source=n_source)

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
