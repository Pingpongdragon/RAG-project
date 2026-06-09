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
# project root for the shared algorithms.cache.* package (appended so local
# config.py / loaders.py keep import priority)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (DATASET_CONFIGS, STRATEGY_ORDER, STRATEGY_LABELS,
                    K_LIST, DATA_DIR, FIG_DIR, log)
import config as _mo1cfg
from loaders import LOADERS
from loaders_temporal import TEMPORAL_LOADERS
# Strategies now live in the shared algorithms.cache package (single source).
# Inject motivation_1's hyper-params (StreamingQA / MiniLM: SF_HIT_THRESH=0.55,
# EDIT_BATCH=8). QueryDriven uses the unified mo2 implementation.
from algorithms.cache.params import PARAMS as _P
_P.update(
    SEED=_mo1cfg.SEED, SF_HIT_THRESH=_mo1cfg.SF_HIT_THRESH,
    DOC_ARRIVE=_mo1cfg.DOC_ARRIVE, DOC_ADD_CAP=_mo1cfg.DOC_ADD_CAP,
    EDIT_BATCH=_mo1cfg.EDIT_BATCH, FETCH_TOP_K=_mo1cfg.FETCH_TOP_K,
)
from algorithms.cache.registry import STRATEGY_FACTORIES
from utils import (compute_embeddings, cluster_and_build_stream,
                   head_biased_init_kb, recall_at_k)


def run_dataset(ds_name, cfg, strategies_to_run, drift_mode='sudden',
                loader_name=None, n_source=None, kb_budget_override=None,
                n_stream_queries=None):
    t0 = time.time()
    nw = cfg['n_windows']; ws = cfg['window_size']
    lname = loader_name or ds_name
    if lname in TEMPORAL_LOADERS:
        loader = TEMPORAL_LOADERS[lname]
        doc_pool, queries, title_to_idx = loader()
    else:
        loader = LOADERS[lname]
        doc_pool, queries, title_to_idx = loader(n_source=n_source or cfg.get('n_source'))

    if n_stream_queries and len(queries) > n_stream_queries:
        import random as _r
        _rng = _r.Random(42)
        queries = _rng.sample(queries, n_stream_queries)
        log.info(f'[{ds_name}] capped stream queries -> {len(queries)} (pool stays {len(doc_pool)})')

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
    if kb_budget_override is not None:
        log.info(f"[{ds_name}] KB budget override: {kb_budget} -> {kb_budget_override} (head_ctx={len(head_ctx)})")
        kb_budget = kb_budget_override
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
    prepare_latency = {n: [] for n in strategies_to_run}  # online pre-retrieval work
    retrieval_latency = {n: [] for n in strategies_to_run}  # KB search / ranking work
    query_latency = {n: [] for n in strategies_to_run}      # prepare + retrieval
    step_latency  = {n: [] for n in strategies_to_run}      # background update seconds/window
    step_overhead = {n: [] for n in strategies_to_run}      # KB replacements per window

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
            _t0_prepare = time.perf_counter()
            s.prepare_window(wq, wqe, w)
            _prepare_dt = time.perf_counter() - _t0_prepare
            prepare_latency[name].append(_prepare_dt)
            if name == 'Oracle':
                s.step(wq, wqe, w)
            effective_kb = (s.get_effective_kb(wq, wqe)
                            if hasattr(s, 'get_effective_kb') else s.kb)
            _t0_retrieval = time.perf_counter()
            r = recall_at_k(effective_kb, wq, d2p, doc_embs, title_to_idx, query_embs)
            _retrieval_dt = time.perf_counter() - _t0_retrieval
            retrieval_latency[name].append(_retrieval_dt)
            query_latency[name].append(_prepare_dt + _retrieval_dt)
            for k in K_LIST:
                results[name][f'recall@{k}'].append(r[k])
            cov = len(effective_kb & window_gold_did) / max(1, len(window_gold_did))
            cov_per_window[name].append(cov)
            if name != 'Oracle':
                # freeze_until: skip updates before drift onset for ablation
                if w >= cfg.get('freeze_until', 0):
                    _cost_before = s.update_cost
                    _t0_step = time.perf_counter()
                    s.step(wq, wqe, w)
                    step_latency[name].append(time.perf_counter() - _t0_step)
                    step_overhead[name].append(s.update_cost - _cost_before)
                else:
                    step_latency[name].append(0.0)
                    step_overhead[name].append(0)
            else:
                step_latency[name].append(0.0)
                step_overhead[name].append(0)
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
        s['prepare_latency_mean_ms'] = round(float(np.mean(prepare_latency[name])) * 1000, 3) if prepare_latency[name] else 0.0
        s['prepare_latency_per_window'] = [round(x * 1000, 3) for x in prepare_latency[name]]
        s['retrieval_latency_mean_ms'] = round(float(np.mean(retrieval_latency[name])) * 1000, 3) if retrieval_latency[name] else 0.0
        s['retrieval_latency_per_window'] = [round(x * 1000, 3) for x in retrieval_latency[name]]
        # per-window total (prepare+retrieval for ws queries at once)
        s['query_latency_mean_ms'] = round(float(np.mean(query_latency[name])) * 1000, 3) if query_latency[name] else 0.0
        s['query_latency_per_window'] = [round(x * 1000, 3) for x in query_latency[name]]
        # per-query: window total / ws (recall_at_k loops sequentially)
        s['per_query_latency_mean_ms'] = round(s['query_latency_mean_ms'] / ws, 4) if ws else 0.0
        s['per_query_latency_per_window'] = [round(x / ws, 4) for x in s['query_latency_per_window']]
        s['step_latency_mean_ms']  = round(float(np.mean(step_latency[name])) * 1000, 3) if step_latency[name] else 0.0
        s['step_latency_per_window'] = [round(x * 1000, 3) for x in step_latency[name]]
        s['update_latency_mean_ms'] = s['step_latency_mean_ms']
        s['update_latency_per_window'] = s['step_latency_per_window']
        s['step_overhead_mean']    = round(float(np.mean(step_overhead[name])), 2) if step_overhead[name] else 0.0
        s['step_overhead_per_window'] = step_overhead[name]
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
               f" {'Writes':>6s} {'OvhW':>6s} {'Qlat':>8s} {'Ulat':>8s}")
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
                  f" {s.get('step_overhead_mean', 0):>6.1f}"
                  f" {s.get('query_latency_mean_ms', 0):>8.1f}"
                  f" {s.get('update_latency_mean_ms', 0):>8.1f}")


def generate_figures(all_results, strategies_to_run, suffix=''):
    """Per-dataset KB Coverage + Recall@5 figure over windows."""
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
        'TemporalAware':    ('#2563EB', '-.', 'o'),
        'OnDemandFetch':    ('#17BECF', '--', 'v'),
        'LogDrivenArrival': ('#BCBD22', ':',  'P'),
        'QueryDriven': ('#10B981', '-',  'D'),
        'Oracle':           ('#D62728', '-',  None),
    }
    datasets = list(all_results.keys())
    n_ds = max(len(datasets), 1)
    fig, axes = plt.subplots(n_ds, 2, figsize=(13, 3.4 * n_ds),
                             sharex=True, squeeze=False)
    emphasis = {'Oracle', 'QueryDriven'}

    for row, ds in enumerate(datasets):
        res = all_results[ds]; cfg = res['config']
        nw = cfg['n_windows']; half = nw // 2
        x = np.arange(1, nw + 1)
        for col, (series_key, ylabel) in enumerate((
            ('cov_per_window', 'KB Coverage (%)'),
            ('recall@5_per_window', 'Recall@5 (%)'),
        )):
            ax = axes[row, col]
            for name in strategies_to_run:
                if name not in res['summary']:
                    continue
                color, ls, marker = palette.get(name, ('gray', '-', None))
                values = res['summary'][name][series_key]
                lw = 2.4 if name == 'Oracle' else (2.0 if name == 'QueryDriven' else 1.4)
                alpha = 1.0 if name in emphasis else 0.85
                zorder = 5 if name in emphasis else 3
                mark_every = max(1, nw // 12)
                ax.plot(x, values, color=color, linestyle=ls, marker=marker,
                        linewidth=lw, alpha=alpha, markersize=5,
                        markevery=mark_every, zorder=zorder,
                        label=STRATEGY_LABELS.get(name, name))
            ax.axvline(half + 0.5, color='#444444', ls='--', lw=0.9, alpha=0.6)
            ax.grid(True, axis='y', alpha=0.25, linestyle=':')
            ax.set_xlim(1, nw)
            ax.set_ylim(0, 100)
            ax.set_ylabel(ylabel)
            if col == 0:
                ax.set_title(
                    f'{ds}  pool={cfg["pool_size"]:,}  KB={cfg["kb_budget"]:,}  '
                    f'{nw}×{cfg["window_size"]}',
                    fontsize=11,
                )
                ax.text(half + 0.5, 95, '  drift onset', va='top',
                        ha='left', fontsize=9, color='#444444')
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
        fig.suptitle('Motivation 1 - ' + drift_label, fontsize=14, y=1.01)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    top = 0.96 if drift_label else 0.99
    fig.tight_layout(rect=[0, 0.06, 1, top])
    fig.savefig(FIG_DIR / f'coverage{suffix}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'coverage{suffix}.png', dpi=200, bbox_inches='tight')
    log.info(f"Saved {FIG_DIR / f'coverage{suffix}.pdf'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Motivation 1 (single-hop) runner')
    parser.add_argument('--n-windows', type=int, default=50)
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--drift', choices=['sudden', 'gradual', 'hybrid', 'temporal', 'full_gradual'], default='sudden')
    parser.add_argument('--freeze-until', type=int, default=0,
                        help='Freeze non-Oracle updates for windows [0, freeze_until). '
                             'Use 50 with --n-windows 100 for freeze-before-drift ablation.')
    parser.add_argument('--n-source', type=int, default=None)
    parser.add_argument('--n-stream-queries', type=int, default=None,
                        help='Cap stream-query count (decouple from pool size). None=use all.')
    parser.add_argument('--datasets', nargs='+', default=['squad'])
    parser.add_argument('--strategies', nargs='+', default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--kb-budget', type=int, default=None, help='Override KB budget (bypass kb_head_mult formula).')
    args = parser.parse_args()

    strategies_to_run = args.strategies or STRATEGY_ORDER
    all_results = {}
    for ds in args.datasets:
        base_cfg = DATASET_CONFIGS[ds].copy()
        base_cfg['n_windows'] = args.n_windows
        base_cfg['window_size'] = args.window_size
        base_cfg['freeze_until'] = args.freeze_until
        log.info(f"\n{'='*60}\n  Running: {ds} ({args.drift})\n{'='*60}")
        all_results[ds] = run_dataset(
            ds, base_cfg, strategies_to_run,
            drift_mode=args.drift,
            loader_name=ds, n_source=args.n_source,
            kb_budget_override=args.kb_budget,
            n_stream_queries=args.n_stream_queries)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    freeze_tag = f'_freeze{args.freeze_until}' if args.freeze_until > 0 else ''
    out_name = args.output or f'results_{args.n_windows}w{freeze_tag}_{args.drift}.json'
    out_path = DATA_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_path}")
    print_summary(all_results, strategies_to_run)
    freeze_tag = f'_freeze{args.freeze_until}' if args.freeze_until > 0 else ''
    suffix = f'_{args.n_windows}w{freeze_tag}_{args.drift}'
    generate_figures(all_results, strategies_to_run, suffix=suffix)


if __name__ == '__main__':
    main()
