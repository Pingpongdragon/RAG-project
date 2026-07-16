#!/usr/bin/env python3
"""Hidden-evidence 与多跳 QA 正式实验入口。

真实日志严格保留官方时间顺序；普通 QA 数据统一使用 factorized evidence-regime
协议。旧 KMeans/head-tail 和手工 bridge-reuse 流已从 runner 中移除。
"""
import argparse
from dataclasses import asdict, is_dataclass
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 加入项目根目录；当前文件位于 experiments/hidden/。
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
    AMAT_HIT_COST=_mo2cfg.AMAT_HIT_COST,
    AMAT_MISS_PENALTY=_mo2cfg.AMAT_MISS_PENALTY,
)
from algorithms.cache.registry import STRATEGY_FACTORIES
from experiments.common.factorized_workload import (
    NATURAL,
    WORKLOAD_CHOICES,
    build_factorized_workload,
    resolve_workload,
)
from experiments.common.stream_protocol import (
    causal_prefix_init_kb,
    chronological_sample,
    stream_sampling_diagnostics,
    support_reuse_diagnostics,
    query_drift_diagnostics,
    workload_factor_diagnostics,
    warmup_overlap_diagnostics,
)
from utils import compute_embeddings, recall_at_k
from llm_expand import augment_with_llm, batch_decompose
from graph_retrieval import (extract_pool_entities, extract_query_entities,
                             LightGraphRAG, recall_at_k_graph,
                             recall_at_k_entity_expand)
from experiments.visibility import require_visibility


def run_dataset(ds_name, cfg, strategies_to_run,
                loader_name=None, n_source=None, q_type=None, kb_budget_override=None,
                kb_pool_ratio=None,
                retrieval='dense', llm_expand=False, workload='auto',
                warmup_windows=3,
                temporal_sampling='prefix',
                factorized_min_support_frequency=1,
                factorized_family_mode='auto'):
    """Run one dataset experiment.

    Args:
        ds_name:          dataset key (for logging / output)
        cfg:              dict with n_windows, window_size, and cache ratio
        strategies_to_run: list of strategy names to evaluate
        loader_name:      override loader key (e.g. '2wiki_expanded')
        n_source:         override n_source for expanded loaders

    Returns:
        dict with config, per-strategy summary, and per-window results.
    """
    t0 = time.time()
    nw = cfg['n_windows']
    ws = cfg['window_size']
    if cfg.get('sf_hit_thresh') is not None:
        _P.SF_HIT_THRESH = float(cfg['sf_hit_thresh'])
        log.info(f"[{ds_name}] SF hit threshold override -> {_P.SF_HIT_THRESH:.2f}")

    # ── Load data ──
    lname = loader_name or ds_name
    loader = LOADERS[lname]
    if lname == '2wiki_expanded':
        doc_pool, queries, title_to_idx = loader(
            n_source=n_source, q_type=q_type)
    else:
        doc_pool, queries, title_to_idx = loader()
    require_visibility('hidden', lname, queries)

    effective_workload = resolve_workload(queries, workload)
    requested_warmup_queries = int(warmup_windows) * ws
    evaluation_queries = nw * ws
    temporal_sampling_stats = None
    if effective_workload == NATURAL:
        queries, temporal_stats = chronological_sample(
            queries,
            warmup_size=requested_warmup_queries,
            evaluation_size=evaluation_queries,
            mode=temporal_sampling,
            block_size=ws,
        )
        temporal_sampling_stats = temporal_stats.as_dict()
        log.info(
            f'[{ds_name}] chronological {temporal_sampling} -> '
            f'{len(queries)} events ({requested_warmup_queries} warmup + '
            f'{evaluation_queries} eval); audit={temporal_sampling_stats}')
    # ── Embeddings ──
    tag = f'{ds_name}_{nw}w_{ws}s' + (f'_{q_type}' if q_type else '')
    doc_embs, query_embs = compute_embeddings(doc_pool, queries, tag=tag)

    # ── Stream ──
    factorized_construction = None
    if effective_workload == NATURAL:
        warmup_stream = list(queries[:requested_warmup_queries])
        stream = list(queries[requested_warmup_queries:])
    else:
        stream, warmup_stream, construction = build_factorized_workload(
            queries,
            doc_pool,
            title_to_idx,
            n_windows=nw,
            window_size=ws,
            workload=effective_workload,
            seed=int(_mo2cfg.DATA_SEED),
            min_support_frequency=factorized_min_support_frequency,
            family_mode=factorized_family_mode,
            warmup_size=requested_warmup_queries,
        )
        factorized_construction = construction.as_dict()
    stream_stats = stream_sampling_diagnostics(stream).as_dict()
    support_reuse_stats = support_reuse_diagnostics(stream, ws).as_dict()
    query_drift_stats = query_drift_diagnostics(
        stream, query_embs, ws).as_dict()
    factor_stats = workload_factor_diagnostics(stream, ws).as_dict()
    warmup_stats = warmup_overlap_diagnostics(
        warmup_stream, stream, requested_warmup_queries).as_dict()
    if warmup_stats['evaluation_overlap']:
        raise ValueError(
            'Causal protocol violation: warm-up shares exact queries with '
            f'evaluation stream: {warmup_stats}')
    log.info('Warm-up/evaluation audit: %s', warmup_stats)

    # ── KB init: 所有策略共用同一个因果 warm-up 初始化 ──
    default_ratio = float(cfg.get('kb_pool_ratio', 0.1))
    kb_budget = max(
        1,
        min(len(doc_pool), int(round(len(doc_pool) * default_ratio))),
    )
    if kb_budget_override is not None:
        log.info(f"[{ds_name}] KB budget override: {kb_budget} -> {kb_budget_override}")
        kb_budget = max(1, min(len(doc_pool), int(kb_budget_override)))
    elif kb_pool_ratio is not None:
        ratio_budget = int(round(len(doc_pool) * float(kb_pool_ratio)))
        ratio_budget = max(1, min(len(doc_pool), ratio_budget))
        log.info(
            f"[{ds_name}] KB/pool ratio override: {kb_budget} -> "
            f"{ratio_budget} (pool={len(doc_pool)}, ratio={float(kb_pool_ratio):.4f})"
        )
        kb_budget = ratio_budget
    init_kb = causal_prefix_init_kb(
        doc_pool, doc_embs, warmup_stream, query_embs, kb_budget,
        seed=int(_mo2cfg.DATA_SEED) + 313)
    log.info(
        f'[{ds_name}] causal-prefix init: warmup_queries='
        f'{len(warmup_stream)}, KB={len(init_kb)}')
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
            _device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
            _sbert = _ST(_EM, device=_device)
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
        if hasattr(st, '_stream'):
            st._stream = stream
            st._query_embs = query_embs
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
            s[key] = round(float(np.mean(vals)) * 100, 1)
            s[f'{key}_mean'] = s[key]
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
        s['kb_coverage'] = round(float(np.mean(cov_vals)) * 100, 1)
        s['kb_coverage_mean'] = s['kb_coverage']
        s['kb_coverage_h1'] = round(float(np.mean(cov_vals[:half])) * 100, 1)
        s['kb_coverage_h2'] = round(float(np.mean(cov_vals[half:])) * 100, 1)
        s['kb_coverage_per_window'] = [round(x * 100, 2) for x in cov_vals]
        cum_vals = kb_coverage_cum[name]
        s['kb_coverage_cumulative_h1'] = round(float(np.mean(cum_vals[:half])) * 100, 1)
        s['kb_coverage_cumulative_h2'] = round(float(np.mean(cum_vals[half:])) * 100, 1)
        s['kb_coverage_cumulative_per_window'] = [round(x * 100, 2) for x in cum_vals]
        replacement_count = int(strategies[name].update_cost)
        total_queries = max(1, int(actual_nw) * int(ws))
        s['update_cost'] = replacement_count
        s['cache_writes'] = replacement_count
        s['replacement_count'] = replacement_count
        s['cache_replacements'] = replacement_count
        s['replacement_rate_per_query'] = round(
            replacement_count / float(total_queries), 6)
        s['replacement_rate_per_window'] = round(
            replacement_count / float(max(1, actual_nw)), 3)
        s['cache_churn_rate'] = round(
            replacement_count / float(max(1, actual_nw * kb_budget)), 6)
        s['cache_churn_rate_pct'] = round(
            s['cache_churn_rate'] * 100, 3)
        s['maint_retrieval_cost'] = strategies[name].maint_retrieval_cost
        s['serve_retrieval_cost'] = strategies[name].serve_retrieval_cost
        s['retrieval_cost'] = strategies[name].retrieval_cost
        s['cost'] = strategies[name].cost
        if hasattr(strategies[name], 'method_version'):
            s['method_version'] = str(strategies[name].method_version)
        policy_config = getattr(strategies[name], 'config', None)
        if policy_config is not None and is_dataclass(policy_config):
            s['policy_config'] = asdict(policy_config)
        if hasattr(strategies[name], 'last_admission'):
            s['last_admission'] = strategies[name].last_admission
        if hasattr(strategies[name], 'cost_log'):
            cost_log = strategies[name].cost_log
            s['cost_log'] = cost_log
            s['evictions'] = int(getattr(strategies[name], 'total_evictions', 0))
            churn = [float(x.get('churn_rate', 0.0)) for x in cost_log]
            budgets = [int(x.get('write_budget', 0)) for x in cost_log]
            s['churn_rate_mean'] = round(float(np.mean(churn)), 6) if churn else 0.0
            s['write_budget_mean'] = round(float(np.mean(budgets)), 2) if budgets else 0.0
        if hasattr(strategies[name], 'route_log'):
            s['route_log'] = strategies[name].route_log
        if hasattr(strategies[name], 'drift_log'):
            s['drift_log'] = strategies[name].drift_log
        if hasattr(strategies[name], 'prefetch_log'):
            s['prefetch_log'] = strategies[name].prefetch_log
        if hasattr(strategies[name], 'expert_log'):
            s['expert_log'] = strategies[name].expert_log
        if hasattr(strategies[name], 'evidence_route_log'):
            s['evidence_route_log'] = strategies[name].evidence_route_log
        residency_summary = _summarise_residency_metrics(residency_metrics[name])
        s.update(residency_summary)
        s.update(_arc_compatible_metric_fields(strategies[name], residency_summary))
        summary[name] = s
    return {
        'dataset': ds_name,
        'config': {
            'kb_budget': kb_budget, 'pool_size': len(doc_pool),
            'kb_pool_ratio': round(float(kb_budget) / max(1, len(doc_pool)), 6),
            'n_windows': actual_nw, 'window_size': ws,
            'n_queries_available': len(queries),
            'seed': int(_mo2cfg.SEED),
            'data_seed': int(_mo2cfg.DATA_SEED),
            'workload': effective_workload, 'k_list': K_LIST,
            'initialization': 'causal-prefix',
            'warmup_windows': int(warmup_windows),
            'stream_sampling': stream_stats,
            'support_reuse': support_reuse_stats,
            'query_drift': query_drift_stats,
            'workload_factors': factor_stats,
            'factorized_construction': factorized_construction,
            'warmup_audit': warmup_stats,
            'temporal_sampling': temporal_sampling_stats,
            'write_cap_per_window': int(_P.WRITE_CAP),
        },
        'summary': summary, 'elapsed': round(elapsed, 1),
    }


def _new_residency_metrics():
    return {
        'cold_fetches': [],
        'support_coverage': [],
        'has_answer_hits': 0,
        'has_answer_queries': 0,
        'hidden_hits': 0,
        'hidden_queries': 0,
        'reuse_hits': 0,
        'reuse_queries': 0,
        'exposure_cold_fetches': [],
        'reuse_cold_fetches': [],
        'target_cold_fetches': [],
        'target_support_coverage': [],
        'target_has_answer_hits': 0,
        'target_has_answer_queries': 0,
        'background_cold_fetches': [],
        'background_support_coverage': [],
        'background_has_answer_hits': 0,
        'background_has_answer_queries': 0,
    }


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
        metrics['support_coverage'].append(float(total - missing) / float(total))
        metrics['has_answer_queries'] += 1
        metrics['has_answer_hits'] += int(missing == 0)
        support_title = q.get('reuse_support_title')
        pi = title_to_idx.get(support_title) if support_title else None
        if pi is not None:
            metrics['hidden_queries'] += 1
            did = doc_pool[pi]['doc_id']
            metrics['hidden_hits'] += int(did in kb_doc_ids)
        role = q.get('reuse_role')
        if role == 'exposure':
            metrics['exposure_cold_fetches'].append(float(missing))
        elif role == 'reuse':
            metrics['reuse_cold_fetches'].append(float(missing))
            if pi is not None:
                metrics['reuse_queries'] += 1
                metrics['reuse_hits'] += int(did in kb_doc_ids)
        if role in {'exposure', 'reuse'}:
            metrics['target_cold_fetches'].append(float(missing))
            metrics['target_support_coverage'].append(
                float(total - missing) / float(total))
            metrics['target_has_answer_queries'] += 1
            metrics['target_has_answer_hits'] += int(missing == 0)
        else:
            metrics['background_cold_fetches'].append(float(missing))
            metrics['background_support_coverage'].append(
                float(total - missing) / float(total))
            metrics['background_has_answer_queries'] += 1
            metrics['background_has_answer_hits'] += int(missing == 0)


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _summarise_residency_metrics(metrics):
    reuse_q = int(metrics['reuse_queries'])
    reuse_hits = int(metrics['reuse_hits'])
    has_q = int(metrics['has_answer_queries'])
    has_hits = int(metrics['has_answer_hits'])
    hidden_q = int(metrics['hidden_queries'])
    hidden_hits = int(metrics['hidden_hits'])
    target_q = int(metrics['target_has_answer_queries'])
    target_hits = int(metrics['target_has_answer_hits'])
    bg_q = int(metrics['background_has_answer_queries'])
    bg_hits = int(metrics['background_has_answer_hits'])
    support_coverage_rate = round(
        _mean_or_zero(metrics['support_coverage']) * 100, 1)
    strict_has_answer_rate = (
        round((has_hits / has_q) * 100, 1) if has_q else 0.0)
    return {
        'cold_fetches_per_query': round(_mean_or_zero(metrics['cold_fetches']), 3),
        'support_coverage_rate': support_coverage_rate,
        'arc_item_hit_rate': support_coverage_rate,
        'has_answer_rate': strict_has_answer_rate,
        'strict_has_answer_rate': strict_has_answer_rate,
        'has_answer_queries': has_q,
        'has_answer_hits': has_hits,
        'hidden_B_hit_rate': round((hidden_hits / hidden_q) * 100, 1) if hidden_q else 0.0,
        'hidden_B_queries': hidden_q,
        'hidden_B_hits': hidden_hits,
        'reuse_hit_rate': round((reuse_hits / reuse_q) * 100, 1) if reuse_q else 0.0,
        'reuse_queries': reuse_q,
        'reuse_hits': reuse_hits,
        'first_exposure_cost': round(_mean_or_zero(metrics['exposure_cold_fetches']), 3),
        'amortized_cold_cost': round(_mean_or_zero(metrics['reuse_cold_fetches']), 3),
        'target_cold_fetches_per_query': round(
            _mean_or_zero(metrics['target_cold_fetches']), 3),
        'target_support_coverage_rate': round(
            _mean_or_zero(metrics['target_support_coverage']) * 100, 1),
        'target_has_answer_rate': round((target_hits / target_q) * 100, 1)
        if target_q else 0.0,
        'target_has_answer_queries': target_q,
        'target_has_answer_hits': target_hits,
        'background_cold_fetches_per_query': round(
            _mean_or_zero(metrics['background_cold_fetches']), 3),
        'background_support_coverage_rate': round(
            _mean_or_zero(metrics['background_support_coverage']) * 100, 1),
        'background_has_answer_rate': round((bg_hits / bg_q) * 100, 1)
        if bg_q else 0.0,
        'background_has_answer_queries': bg_q,
        'background_has_answer_hits': bg_hits,
    }


def _arc_compatible_metric_fields(strategy, residency_summary):
    """Add ARC-compatible normalized AMAT to the result JSON.

    Classic AMAT is ``T_hot + miss_rate * T_miss``:
      1. hot-cache access costs ``AMAT_HIT_COST`` (default 1);
      2. L2/full-index access costs ``AMAT_MISS_PENALTY`` (default 10);
      3. serve-time full-index fetches, e.g. OnDemandFetch, correct the
         inferred miss rate with the actual external fetch rate.

    OnDemandFetch stores ``serve_retrieval_cost`` as fetched top-K document
    count, so divide by ``FETCH_TOP_K`` to recover full-index fetch calls.
    """
    n_queries = int(residency_summary.get('has_answer_queries', 0))
    if n_queries <= 0:
        miss_rate = 0.0
        serve_fetches_per_query = 0.0
    else:
        has_hits = float(residency_summary.get('has_answer_hits', 0))
        miss_rate = 1.0 - has_hits / float(n_queries)
        fetch_k = max(1, int(getattr(_P, 'FETCH_TOP_K', 1)))
        serve_fetches = float(getattr(strategy, 'serve_retrieval_cost', 0)) / fetch_k
        serve_fetches_per_query = serve_fetches / float(n_queries)

    l2_accesses_per_query = max(float(miss_rate), float(serve_fetches_per_query))
    hit_cost = float(getattr(_P, 'AMAT_HIT_COST', 1.0))
    miss_penalty = float(getattr(_P, 'AMAT_MISS_PENALTY', 10.0))
    amat = hit_cost + l2_accesses_per_query * miss_penalty
    return {
        'amat': round(float(amat), 3),
        'amat_normalized': round(float(amat), 3),
        'amat_hit_cost': round(float(hit_cost), 3),
        'amat_miss_penalty': round(float(miss_penalty), 3),
        'miss_rate': round(float(miss_rate), 6),
        'l2_accesses_per_query': round(float(l2_accesses_per_query), 3),
        'serve_fetches_per_query': round(float(serve_fetches_per_query), 3),
    }


def print_summary(all_results, strategies_to_run):
    ds_labels = {'2wikimultihopqa': '2WikiMultihopQA'}
    print("\n" + "=" * 115)
    print("  ARC-Compatible Cache Metrics Under Query Distribution Drift")
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
               f" {'HasAns':>6s} {'AMAT':>6s} |"
               f" {'R@5':>6s} {'Cov':>6s} |"
               f" {'Reuse':>6s} |"
               f" {'Repl':>6s} {'MaintR':>7s} {'ServeR':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for name in strategies_to_run:
            if name not in res['summary'] or name == 'Oracle':
                continue
            s = res['summary'][name]
            recall = s['recall@5']
            coverage = s['kb_coverage']
            line = (f"{name:>18s} |"
                    f" {s.get('has_answer_rate', 0.0):>5.1f}%"
                    f" {s.get('amat', 0.0):>6.2f} |"
                    f" {recall:>5.1f}% {coverage:>5.1f}% |"
                    f" {s.get('reuse_hit_rate', 0.0):>5.1f}% |"
                    f" {s['replacement_count']:>6d}"
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
    ds_labels = {'2wikimultihopqa': '2WikiMultihopQA'}

    palette = {
        'LRU':              ('#F59E0B', '-.', 'v'),
        'FIFO':             ('#7C3AED', '--', '^'),
        'TinyLFU':          ('#0284C7', ':',  'p'),
        'AgentRAGCache':    ('#111827', '-',  'o'),
        'GPTCacheStyle':    ('#0891B2', ':',  'P'),
        'Proximity':        ('#06B6D4', ':',  'h'),
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

            if cfg.get('workload') == 'factorized_one_shot':
                midpoint = nw // 2 + 0.5
                ax.axvline(
                    midpoint, color='#444444', ls='--', lw=0.9, alpha=0.6)
            ax.grid(True, axis='y', alpha=0.25, linestyle=':')
            ax.set_xlim(1, nw)
            ax.set_ylim(0, 100)
            ax.set_ylabel(ylabel)
            if col == 0:
                ax.set_title(
                    f'{ds_labels.get(ds, ds)}\n'
                    f'pool={cfg["pool_size"]:,}  KB={cfg["kb_budget"]:,}  '
                    f'{nw}×{cfg["window_size"]}  {cfg.get("workload", "")}',
                    fontsize=11)
            else:
                ax.set_title('Recall@5', fontsize=11)
            if row == n_ds - 1:
                ax.set_xlabel('Window index')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(labels), 4), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.10, 1, 1.00])
    fig.savefig(FIG_DIR / f'coverage_drift{suffix}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'coverage_drift{suffix}.png', dpi=200,
                bbox_inches='tight')
    log.info(f"Saved {FIG_DIR / f'coverage_drift{suffix}.pdf'}")
    plt.close(fig)
def main():
    parser = argparse.ArgumentParser(
        description='DRIP hidden-evidence 与 multi-hop QA 实验入口')
    parser.add_argument('--n-windows', type=int, default=None)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--workload',
                        choices=sorted(WORKLOAD_CHOICES),
                        default='auto',
                        help='auto keeps natural traces chronological and uses '
                             'factorized_recurring for controlled QA data.')
    parser.add_argument('--expanded', action='store_true',
                        help='Set an explicit source-sample size for 2Wiki')
    parser.add_argument('--q-type', default=None,
                        help='2Wiki hidden type: compositional, bridge_comparison, or inference')
    parser.add_argument('--n-source', type=int, default=3500)
    parser.add_argument(
        '--factorized-min-support-frequency', type=int, default=1,
        help='Required source-query frequency of every support in factorized workloads.')
    parser.add_argument(
        '--factorized-family-mode', choices=['auto', 'exact', 'anchor'],
        default='auto',
        help='Evidence-family definition for factorized workloads; labels stay offline.')
    parser.add_argument('--llm-expand', action='store_true',
                        help='Use LLM query expansion to augment retrieval')
    parser.add_argument('--datasets', nargs='+',
                        default=['2wikimultihopqa'],
                        choices=['2wikimultihopqa'])
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Strategies to run (default: all from STRATEGY_ORDER)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON filename (auto-generated if omitted)')
    parser.add_argument('--kb-budget', type=int, default=None,
                        help='Force absolute KB budget (overrides --kb-pool-ratio)')
    parser.add_argument('--kb-pool-ratio', type=float, default=None,
                        help='Set KB budget as a fraction of pool size, e.g. 0.1 for KB/pool=1/10. '
                             'Ignored when --kb-budget is set.')
    parser.add_argument(
        '--write-cap', type=int, default=None,
        help='Shared maximum physical cache writes per evaluation window.')
    parser.add_argument('--warmup-windows', type=int, default=3,
                        help='Pre-evaluation query-history windows used by causal init.')
    parser.add_argument('--temporal-sampling',
                        choices=['prefix', 'window_span'], default='prefix',
                        help='Natural trace selection: prefix uses the log head; '
                             'window_span preserves contiguous windows while '
                             'covering the full remaining time range.')
    parser.add_argument('--retrieval', choices=['dense', 'graph', 'entity_expand'], default='dense',
                        help='Recall@K backend: dense cosine (default), passage-entity graph, or entity expansion')
    args = parser.parse_args()

    if args.write_cap is not None:
        if args.write_cap < 0:
            parser.error('--write-cap must be non-negative')
        _P.WRITE_CAP = int(args.write_cap)

    strategies_to_run = args.strategies or STRATEGY_ORDER

    all_results = {}
    for ds in args.datasets:
        base_cfg = DATASET_CONFIGS['2wikimultihopqa'].copy()
        if args.n_windows:
            base_cfg['n_windows'] = args.n_windows
        if args.window_size:
            base_cfg['window_size'] = args.window_size

        loader_name = None
        n_source = None
        if args.expanded:
            loader_name = '2wiki_expanded'
            n_source = args.n_source
            base_cfg['n_source'] = args.n_source

        log.info(
            f"\n{'='*60}\n  Running: {ds} "
            f"(workload={args.workload})\n{'='*60}"
        )
        all_results[ds] = run_dataset(
            ds, base_cfg, strategies_to_run,
            loader_name=loader_name,
            n_source=n_source,
            q_type=args.q_type,
            kb_budget_override=args.kb_budget,
            kb_pool_ratio=args.kb_pool_ratio,
            retrieval=args.retrieval,
            llm_expand=args.llm_expand,
            workload=args.workload,
            warmup_windows=args.warmup_windows,
            temporal_sampling=args.temporal_sampling,
            factorized_min_support_frequency=(
                args.factorized_min_support_frequency),
            factorized_family_mode=args.factorized_family_mode)

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
        out_name = f'results_{actual_nw_for_name}w_{args.workload}{ret_suf}.json'
    out_path = DATA_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_path}")

    print_summary(all_results, strategies_to_run)

    # 图名绑定 JSON stem，避免不同数据集/消融共享窗口后缀时互相覆盖。
    suffix = f'_{Path(out_name).stem}'
    generate_figures(all_results, strategies_to_run, suffix=suffix)


if __name__ == '__main__':
    main()
