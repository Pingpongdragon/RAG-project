#!/usr/bin/env python3
"""
Unified experiment runner for the motivation study.

Usage:
  python run.py                          # default 20-window sudden drift
  python run.py --n-windows 50 --window-size 50 --expanded
  python run.py --drift gradual
  python run.py --datasets hotpotqa musique
  python run.py --strategies Static DRIP Oracle

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
# project root for the shared algorithms.cache.* package (appended, so the
# local motivation_2 modules — config.py, loaders.py — keep import priority)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import (DATASET_CONFIGS, STRATEGY_ORDER, STRATEGY_LABELS,
                    STRATEGY_STYLES, K_LIST, DATA_DIR, FIG_DIR, log)
import config as _mo2cfg
from loaders import LOADERS
# Strategies now live in the shared algorithms.cache package (single source).
# Inject this experiment's hyper-params so the same code reproduces mo2 numbers.
from algorithms.cache.params import PARAMS as _P
_P.update(
    SEED=_mo2cfg.SEED, SF_HIT_THRESH=_mo2cfg.SF_HIT_THRESH,
    DOC_ARRIVE=_mo2cfg.DOC_ARRIVE, DOC_ADD_CAP=_mo2cfg.DOC_ADD_CAP,
    EDIT_BATCH=_mo2cfg.EDIT_BATCH, FIFO_BATCH=_mo2cfg.FIFO_BATCH,
    FETCH_TOP_K=_mo2cfg.FETCH_TOP_K, LOG_FIX_TOP_K=_mo2cfg.LOG_FIX_TOP_K,
    LOG_FIX_CAP=_mo2cfg.LOG_FIX_CAP, LOG_LAG_WINDOWS=_mo2cfg.LOG_LAG_WINDOWS,
)
from algorithms.cache.registry import STRATEGY_FACTORIES
from utils import (compute_embeddings, build_query_stream,
                   focus_pool, head_biased_init_kb, recall_at_k)
from llm_expand import augment_with_llm, batch_decompose
from graph_retrieval import (extract_pool_entities, extract_query_entities,
                             LightGraphRAG, recall_at_k_graph,
                             recall_at_k_entity_expand)


def run_dataset(ds_name, cfg, strategies_to_run, drift_mode='sudden',
                loader_name=None, n_source=None, q_type=None, kb_budget_override=None,
                n_stream_queries=None,
                retrieval='dense', llm_expand=False, workload='cluster_shift',
                mask_stream_gold=True, init_stream_gold=False):
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
    if n_source is not None and lname in ('musique_expanded', '2wiki_expanded', 'hotpotqa_expanded'):
        if q_type and lname in ('hotpotqa_expanded', '2wiki_expanded'):
            doc_pool, queries, title_to_idx = loader(n_source=n_source, q_type=q_type)
        else:
            doc_pool, queries, title_to_idx = loader(n_source=n_source)
    else:
        doc_pool, queries, title_to_idx = loader()


    # Optional: cap query count for stream (decouple from pool size)
    if n_stream_queries and len(queries) > n_stream_queries:
        import random as _r
        _rng = _r.Random(42)
        queries = _rng.sample(queries, n_stream_queries)
        log.info(f'[{ds_name}] capped stream queries -> {len(queries)} (pool stays {len(doc_pool)})')
    # ── Embeddings ──
    tag = f'{ds_name}_{nw}w_{ws}s' + (f'_{q_type}' if q_type else '')
    doc_embs, query_embs = compute_embeddings(doc_pool, queries, tag=tag)

    # ── Stream ──
    stream, centroids, head_set = build_query_stream(
        queries, query_embs, cfg, workload=workload, drift_mode=drift_mode)

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
    if init_stream_gold:
        init_kb = _init_kb_with_stream_gold(
            init_kb, doc_pool, doc_embs, title_to_idx, stream, centroids, kb_budget)
    elif mask_stream_gold and workload in ('temporal_bridge_reuse', 'burst_bridge_reuse'):
        init_kb = _mask_stream_gold_from_init_kb(
            init_kb, doc_pool, doc_embs, title_to_idx, stream, centroids, kb_budget)
    d2p = {d['doc_id']: i for i, d in enumerate(doc_pool)}

    # ── Optional graph-retrieval setup (no-LLM HippoRAG/LightRAG analogue) ──
    pool_ents = query_ents = None
    strategy_graphs = {}
    if retrieval in ('graph', 'entity_expand'):
        pool_ents = extract_pool_entities(doc_pool, tag=tag)
    if retrieval == 'graph':
        query_ents = extract_query_entities(queries, tag=tag)
        if llm_expand:
            import spacy
            _nlp = spacy.load('en_core_web_sm')
            query_embs, query_ents = augment_with_llm(
                queries, query_embs, query_ents, _nlp,
                tag=tag + '_llmexp')
            log.info(f'[{ds_name}] LLM query expansion applied.')
            # Build per-query sub-question embedding lists for union retrieval
            # only when explicit LLM expansion is enabled.
            _sub_lists = batch_decompose(queries, tag=tag + '_llmexp')
            from sentence_transformers import SentenceTransformer as _ST
            from config import EMBED_MODEL as _EM, BGE_QUERY_PREFIX as _BP
            _sbert = _ST(_EM, device='cuda')
            _qpref = _BP if 'bge' in _EM.lower() else ''
            _flat = [sq for sqs in _sub_lists for sq in sqs]
            _flat_embs = _sbert.encode([_qpref + q for q in _flat], batch_size=256, show_progress_bar=False,
                                       normalize_embeddings=True)
            import spacy as _spacy
            _nlp2 = _spacy.load('en_core_web_sm')
            sub_query_embs, sub_query_ents = [], []
            _ptr = 0
            for _sqs in _sub_lists:
                _n = len(_sqs)
                sub_query_embs.append(_flat_embs[_ptr:_ptr + _n])
                sub_query_ents.append([[e.text for e in _nlp2(sq).ents] for sq in _sqs])
                _ptr += _n
        else:
            sub_query_embs = sub_query_ents = None
    else:
        sub_query_embs = sub_query_ents = None
    # ── Strategies ──
    strategies = {}
    for name in strategies_to_run:
        st = STRATEGY_FACTORIES[name](doc_pool, doc_embs, title_to_idx)
        st.set_kb(set(init_kb))
        if hasattr(st, '_half'):
            st._half = nw // 2
        if hasattr(st, '_stream'):
            st._stream = stream
            st._query_embs = query_embs
        # Inject pool entities into entity-aware strategies (e.g. RoutedCache R3).
        # Build lazily here if not already extracted for graph/entity_expand.
        if hasattr(st, '_pool_ents'):
            if pool_ents is None:
                pool_ents = extract_pool_entities(doc_pool, tag=tag)
            st._pool_ents = pool_ents
        strategies[name] = st
        if retrieval == 'graph':
            g = LightGraphRAG(pool_ents, query_ents,
                              doc_embs, query_embs,
                              {d['doc_id']: i for i, d in enumerate(doc_pool)},
                              {i: d['doc_id'] for i, d in enumerate(doc_pool)})
            g.index(set(init_kb))
            strategy_graphs[name] = g

    # ── Evaluate ──
    results = {n: {f'recall@{k}': [] for k in K_LIST} for n in strategies_to_run}
    kb_coverage = {n: [] for n in strategies_to_run}  # L1: fraction of window's gold SFs that sit in KB
    kb_coverage_cum = {n: [] for n in strategies_to_run}  # cumulative gold so far
    seen_gold = {n: set() for n in strategies_to_run}
    residency_metrics = {n: _new_residency_metrics() for n in strategies_to_run}
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
            st_i = strategies[name]
            st_i.prepare_window(wq, wqe, w)
            # Oracle is non-causal: rebuild KB BEFORE evaluation so it represents
            # the per-window upper bound for the queries we are about to score.
            if name == 'Oracle':
                st_i.step(wq, wqe, w)
            _record_residency_metrics(
                residency_metrics[name], st_i.kb, wq, doc_pool, title_to_idx)
            effective_kb = st_i.get_effective_kb(wq, wqe) if hasattr(st_i, 'get_effective_kb') else st_i.kb
            if retrieval == 'graph':
                strategy_graphs[name].sync_to(effective_kb)
                r = recall_at_k_graph(
                    strategy_graphs[name], wq, doc_pool, K_LIST,
                    sub_embs=sub_query_embs, sub_ents=sub_query_ents)
            elif retrieval == 'entity_expand':
                r = recall_at_k_entity_expand(
                    pool_ents, effective_kb, wq, d2p, doc_embs, title_to_idx, query_embs,
                    step1_k=3, k_list=K_LIST)
            else:
                r = recall_at_k(effective_kb, wq, d2p, doc_embs, title_to_idx, query_embs)
            for k in K_LIST:
                results[name][f'recall@{k}'].append(r[k])
            cov = len(effective_kb & window_gold_did) / max(1, len(window_gold_did))
            kb_coverage[name].append(cov)
            seen_gold[name] |= window_gold_did
            cum = len(effective_kb & seen_gold[name]) / max(1, len(seen_gold[name]))
            kb_coverage_cum[name].append(cum)
            if name != 'Oracle':
                st_i.step(wq, wqe, w)
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
        if hasattr(strategies[name], 'last_admission'):
            s['last_admission'] = strategies[name].last_admission
        if hasattr(strategies[name], 'bridge_log'):
            s['bridge_log'] = strategies[name].bridge_log
        if hasattr(strategies[name], 'route_log'):
            s['route_log'] = strategies[name].route_log
        if hasattr(strategies[name], 'drift_log'):
            s['drift_log'] = strategies[name].drift_log
        s.update(_summarise_residency_metrics(residency_metrics[name]))
        summary[name] = s
    return {
        'dataset': ds_name,
        'config': {
            'kb_budget': kb_budget, 'pool_size': len(doc_pool),
            'n_windows': actual_nw, 'window_size': ws,
            'n_queries_available': len(queries),
            'drift': drift_mode, 'workload': workload, 'k_list': K_LIST,
            'mask_stream_gold': bool(mask_stream_gold and not init_stream_gold),
            'init_stream_gold': bool(init_stream_gold),
        },
        'summary': summary, 'elapsed': round(elapsed, 1),
    }


def _new_residency_metrics():
    return {
        'cold_fetches': [],
        'reuse_hits': 0,
        'reuse_queries': 0,
        'exposure_cold_fetches': [],
        'reuse_cold_fetches': [],
    }


def _mask_stream_gold_from_init_kb(init_kb, doc_pool, doc_embs, title_to_idx,
                                   stream, centroids, kb_budget):
    """Prevent reuse workloads from starting with the answers already warm."""
    forbidden = set()
    for q in stream:
        for title in q.get('sf_titles', []):
            pi = title_to_idx.get(title)
            if pi is not None:
                forbidden.add(doc_pool[pi]['doc_id'])

    masked = set(init_kb) - forbidden
    removed = len(set(init_kb) & forbidden)
    if len(masked) >= kb_budget:
        log.info(
            f"Reuse workload init mask: removed {removed} stream-gold docs; "
            f"KB={len(masked)}"
        )
        return set(sorted(masked)[:kb_budget])

    forbidden_or_resident = forbidden | masked
    rest = [
        i for i, d in enumerate(doc_pool)
        if d['doc_id'] not in forbidden_or_resident
    ]
    need = kb_budget - len(masked)
    if rest and need > 0:
        norm_c = centroids / np.clip(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-10, None)
        scores = (doc_embs[rest] @ norm_c.T).max(axis=1)
        top = np.argsort(scores)[-min(need, len(rest)):]
        masked.update(doc_pool[rest[int(i)]]['doc_id'] for i in top)

    log.info(
        f"Reuse workload init mask: removed {removed} stream-gold docs; "
        f"forbidden={len(forbidden)} KB={len(masked)}/{kb_budget}"
    )
    return masked


def _init_kb_with_stream_gold(init_kb, doc_pool, doc_embs, title_to_idx,
                              stream, centroids, kb_budget):
    """Capacity sanity check: seed KB with stream support docs, then fill slack."""
    gold_doc_ids = []
    seen = set()
    for q in stream:
        for title in q.get('sf_titles', []):
            pi = title_to_idx.get(title)
            if pi is None:
                continue
            did = doc_pool[pi]['doc_id']
            if did not in seen:
                seen.add(did)
                gold_doc_ids.append(did)

    selected = set(gold_doc_ids[:kb_budget])
    if len(gold_doc_ids) > kb_budget:
        log.warning(
            f"Stream-gold init truncated: gold={len(gold_doc_ids)} "
            f"KB={kb_budget}"
        )
        return selected

    for did in sorted(init_kb):
        if len(selected) >= kb_budget:
            break
        selected.add(did)

    if len(selected) < kb_budget:
        selected_or_gold = set(selected)
        rest = [
            i for i, d in enumerate(doc_pool)
            if d['doc_id'] not in selected_or_gold
        ]
        need = kb_budget - len(selected)
        if rest and need > 0:
            norm_c = centroids / np.clip(
                np.linalg.norm(centroids, axis=1, keepdims=True), 1e-10, None)
            scores = (doc_embs[rest] @ norm_c.T).max(axis=1)
            top = np.argsort(scores)[-min(need, len(rest)):]
            selected.update(doc_pool[rest[int(i)]]['doc_id'] for i in top)

    log.info(
        f"Stream-gold init: gold={len(gold_doc_ids)} "
        f"resident_gold={len(set(gold_doc_ids) & selected)} "
        f"KB={len(selected)}/{kb_budget}"
    )
    return selected


def _missing_gold_count(kb_doc_ids, query, doc_pool, title_to_idx):
    missing = 0
    total = 0
    for title in query.get('sf_titles', []):
        pi = title_to_idx.get(title)
        if pi is None:
            continue
        total += 1
        if doc_pool[pi]['doc_id'] not in kb_doc_ids:
            missing += 1
    return missing, total


def _record_residency_metrics(metrics, kb_doc_ids, window_queries,
                              doc_pool, title_to_idx):
    for q in window_queries:
        missing, total = _missing_gold_count(kb_doc_ids, q, doc_pool, title_to_idx)
        if total <= 0:
            continue
        metrics['cold_fetches'].append(float(missing))
        role = q.get('reuse_role')
        if role == 'exposure':
            metrics['exposure_cold_fetches'].append(float(missing))
        elif role == 'reuse':
            metrics['reuse_cold_fetches'].append(float(missing))
            support_title = q.get('reuse_support_title')
            pi = title_to_idx.get(support_title) if support_title else None
            if pi is not None:
                metrics['reuse_queries'] += 1
                did = doc_pool[pi]['doc_id']
                metrics['reuse_hits'] += int(did in kb_doc_ids)


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _summarise_residency_metrics(metrics):
    reuse_q = int(metrics['reuse_queries'])
    reuse_hits = int(metrics['reuse_hits'])
    return {
        'cold_fetches_per_query': round(_mean_or_zero(metrics['cold_fetches']), 3),
        'reuse_hit_rate': round((reuse_hits / reuse_q) * 100, 1) if reuse_q else 0.0,
        'reuse_queries': reuse_q,
        'reuse_hits': reuse_hits,
        'first_exposure_cost': round(_mean_or_zero(metrics['exposure_cold_fetches']), 3),
        'amortized_cold_cost': round(_mean_or_zero(metrics['reuse_cold_fetches']), 3),
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
               f" {'ColdQ':>6s} {'Reuse':>6s} |"
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
                    f" {s.get('cold_fetches_per_query', 0.0):>6.2f}"
                    f" {s.get('reuse_hit_rate', 0.0):>5.1f}% |"
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
        'LRU':              ('#F59E0B', '-.', 'v'),
        'FIFO':             ('#7C3AED', '--', '^'),
        'TinyLFU':          ('#0284C7', ':',  'p'),
        'OnDemandFetch':    ('#17BECF', '--', 'v'),
        'LogDrivenArrival': ('#BCBD22', ':',  'P'),
        'AgentRAGCache':    ('#111827', '-',  'o'),
        'AgentRAGCache_NoHub': ('#6B7280', '--', 'o'),
        'RoutedCache':      ('#2563EB', '-',  's'),
        'DRIP':             ('#0F766E', '-',  '*'),
        'Oracle':           ('#D62728', '-',  None),
    }

    fig, axes = plt.subplots(n_ds, 2, figsize=(13, 3.4 * n_ds),
                             sharex=True, squeeze=False)
    emphasis = {'Oracle', 'DRIP'}

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
                lw = 2.4 if name == 'Oracle' else (2.0 if name == 'DRIP' else 1.4)
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
    parser.add_argument('--drift', choices=['sudden', 'gradual', 'full_gradual'], default='sudden')
    parser.add_argument('--workload',
                        choices=['cluster_shift', 'random_static',
                                 'temporal_bridge_reuse',
                                 'burst_bridge_reuse'],
                        default='cluster_shift',
                        help='Query stream constructor. cluster_shift preserves historical behavior.')
    parser.add_argument('--expanded', action='store_true',
                        help='Use expanded loaders for 2Wiki and MuSiQue')
    parser.add_argument('--q-type', default=None,
                        help='Filter to question type (hotpot: bridge|comparison; 2wiki: compositional|comparison|bridge_comparison|inference)')
    parser.add_argument('--n-source', type=int, default=3500)
    parser.add_argument('--n-stream-queries', type=int, default=None,
                        help='Cap stream-query count (decouple from pool size). None=use all.')
    parser.add_argument('--llm-expand', action='store_true',
                        help='Use LLM sub-question decomposition to augment retrieval')
    parser.add_argument('--datasets', nargs='+',
                        default=['hotpotqa', '2wikimultihopqa', 'musique'])
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Strategies to run (default: all from STRATEGY_ORDER)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON filename (auto-generated if omitted)')
    parser.add_argument('--kb-budget', type=int, default=None,
                        help='Force absolute KB budget (overrides kb_head_mult formula)')
    parser.add_argument('--no-mask-stream-gold', action='store_true',
                        help='Do not remove stream support docs from the initial KB. '
                             'Use for capacity/coverage upper-bound experiments.')
    parser.add_argument('--init-stream-gold', action='store_true',
                        help='Seed the initial KB with stream support docs before filling slack. '
                             'This is an oracle capacity sanity check, not a causal benchmark.')
    parser.add_argument('--retrieval', choices=['dense', 'graph', 'entity_expand'], default='dense',
                        help='Recall@K backend: dense cosine (default) or PPR over passage-entity graph')
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
            q_type=args.q_type,
            kb_budget_override=args.kb_budget,
            n_stream_queries=args.n_stream_queries,
            retrieval=args.retrieval,
            llm_expand=args.llm_expand,
            workload=args.workload,
            mask_stream_gold=not args.no_mask_stream_gold,
            init_stream_gold=args.init_stream_gold)

    # ── Save ──
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    actual_nw_for_name = (
        args.n_windows or
        next(iter(all_results.values()))['config']['n_windows']
    )
    if args.output:
        out_name = args.output
    else:
        ret_suf = '' if args.retrieval == 'dense' else f'_{args.retrieval}'
        workload_suf = '' if args.workload == 'cluster_shift' else f'_{args.workload}'
        out_name = f'results_{actual_nw_for_name}w_{args.drift}{workload_suf}{ret_suf}.json'
    out_path = DATA_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_path}")

    print_summary(all_results, strategies_to_run)

    workload_suf = '' if args.workload == 'cluster_shift' else f'_{args.workload}'
    suffix = f'_{actual_nw_for_name}w_{args.drift}{workload_suf}{"" if args.retrieval == "dense" else "_" + args.retrieval}'
    generate_figures(all_results, strategies_to_run, suffix=suffix)


if __name__ == '__main__':
    main()
