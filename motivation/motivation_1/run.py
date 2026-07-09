#!/usr/bin/env python3
"""Motivation 1 实验入口：可见时间漂移下的缓存评估。

这个文件只保留实验编排逻辑：

  1. 读取数据集并计算/加载 embedding；
  2. 构造带漂移的查询流；
  3. 用同一个初始热 KB 初始化所有策略；
  4. 按窗口做 causal evaluation；
  5. 输出 JSON summary 和曲线图。

缓存驻留指标放在 ``metrics.py``，画图逻辑放在 ``plotting.py``。
这样 ``run.py`` 主要展示实验流程，不把统计和绘图细节混在一起。
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 加入项目根目录，方便导入共享的 algorithms.cache.* 模块。
# 用 append 是为了让当前目录下的 config.py / loaders.py 仍然优先。
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import DATASET_CONFIGS, DATA_DIR, K_LIST, STRATEGY_ORDER, log
import config as _mo1cfg
from loaders import LOADERS
from loaders_temporal import TEMPORAL_LOADERS
from metrics import (
    new_residency_metrics,
    record_residency_metrics,
    summarise_residency_metrics,
)
from plotting import generate_figures

from algorithms.cache.params import PARAMS as _P
from algorithms.cache.registry import STRATEGY_FACTORIES
from utils import (
    cluster_and_build_stream,
    compute_embeddings,
    head_biased_init_kb,
    recall_at_k,
)


_P.update(
    SEED=_mo1cfg.SEED,
    SF_HIT_THRESH=_mo1cfg.SF_HIT_THRESH,
    DOC_ARRIVE=_mo1cfg.DOC_ARRIVE,
    DOC_ADD_CAP=_mo1cfg.DOC_ADD_CAP,
    EDIT_BATCH=_mo1cfg.EDIT_BATCH,
    FETCH_TOP_K=_mo1cfg.FETCH_TOP_K,
    AMAT_HIT_COST=_mo1cfg.AMAT_HIT_COST,
    AMAT_MISS_PENALTY=_mo1cfg.AMAT_MISS_PENALTY,
)


def run_dataset(
    dataset_name,
    dataset_config,
    strategies_to_run,
    drift_mode="sudden",
    loader_name=None,
    n_source=None,
    kb_budget_override=None,
    n_stream_queries=None,
):
    """在一个数据集上运行所有选中的 cache 策略。

    非 Oracle 策略使用因果评估：先服务当前窗口，再根据当前窗口
    更新 KB，更新结果只能影响未来窗口。Oracle 是上界，允许在评估前刷新。
    """
    start_time = time.time()
    num_windows = int(dataset_config["n_windows"])
    window_size = int(dataset_config["window_size"])

    doc_pool, queries, title_to_idx = load_query_dataset(
        dataset_name,
        dataset_config,
        loader_name=loader_name,
        n_source=n_source,
    )
    queries = cap_stream_queries(dataset_name, queries, n_stream_queries)

    embedding_tag = f"{dataset_name}_{num_windows}w_{window_size}s"
    doc_embs, query_embs = compute_embeddings(doc_pool, queries, tag=embedding_tag)
    stream, centroids, head_set = cluster_and_build_stream(
        queries,
        query_embs,
        dataset_config,
        drift_mode=drift_mode,
    )

    kb_budget, initial_kb = build_initial_kb(
        dataset_name,
        dataset_config,
        doc_pool,
        doc_embs,
        centroids,
        head_set,
        stream,
        kb_budget_override,
    )
    doc_id_to_pool_index = {
        doc["doc_id"]: pool_index for pool_index, doc in enumerate(doc_pool)
    }

    strategies = instantiate_strategies(
        strategies_to_run,
        doc_pool,
        doc_embs,
        title_to_idx,
        initial_kb,
        stream,
        query_embs,
        num_windows,
    )
    tracking = new_tracking_tables(strategies_to_run)

    for window_idx in range(num_windows):
        window_queries = stream[window_idx * window_size:(window_idx + 1) * window_size]
        if not window_queries:
            log.warning(f"[{dataset_name}] W{window_idx + 1}: empty window, stopping")
            break

        window_query_embs = window_embeddings(window_queries, query_embs)
        window_gold_doc_ids = collect_window_gold_doc_ids(
            window_queries,
            doc_pool,
            title_to_idx,
        )

        for strategy_name in strategies_to_run:
            evaluate_strategy_window(
                strategy_name,
                strategies[strategy_name],
                window_idx,
                window_queries,
                window_query_embs,
                window_gold_doc_ids,
                doc_pool,
                title_to_idx,
                doc_id_to_pool_index,
                query_embs,
                dataset_config,
                tracking,
            )

        if window_idx % 5 == 0 or window_idx == num_windows - 1:
            coverage = {
                name: f"{tracking['coverage'][name][-1] * 100:.1f}%"
                for name in strategies_to_run
            }
            log.info(f"[{dataset_name}] W{window_idx + 1}/{num_windows} Cov: {coverage}")

    elapsed_seconds = time.time() - start_time
    return summarise_run(
        dataset_name,
        drift_mode,
        doc_pool,
        queries,
        kb_budget,
        window_size,
        strategies_to_run,
        strategies,
        tracking,
        elapsed_seconds,
    )


def load_query_dataset(dataset_name, dataset_config, loader_name=None, n_source=None):
    """读取 temporal 或普通 QA 数据，并统一成 doc_pool / queries 格式。"""
    loader_key = loader_name or dataset_name
    if loader_key in TEMPORAL_LOADERS:
        return TEMPORAL_LOADERS[loader_key]()
    n_source = n_source or dataset_config.get("n_source")
    return LOADERS[loader_key](n_source=n_source)


def cap_stream_queries(dataset_name, queries, n_stream_queries):
    """可选地缩小查询流；doc pool 保持不变，方便做小规模调试。"""
    if not n_stream_queries or len(queries) <= n_stream_queries:
        return queries
    rng = random.Random(42)
    sampled = rng.sample(queries, n_stream_queries)
    log.info(
        f"[{dataset_name}] capped stream queries -> {len(sampled)} "
        f"(pool stays unchanged)"
    )
    return sampled


def build_initial_kb(
    dataset_name,
    dataset_config,
    doc_pool,
    doc_embs,
    centroids,
    head_set,
    stream,
    kb_budget_override,
):
    """构造所有策略共用的初始热 KB。"""
    title_to_pool_index = {doc["title"]: i for i, doc in enumerate(doc_pool)}
    head_context_positions = set()
    for query in stream:
        if query.get("is_tail", True):
            continue
        for title in query.get("ctx_titles", []):
            if title in title_to_pool_index:
                head_context_positions.add(title_to_pool_index[title])

    kb_head_mult = dataset_config.get("kb_head_mult", 1.2)
    kb_budget = max(
        300,
        int(round(len(head_context_positions) * kb_head_mult / 50)) * 50,
    )
    if kb_budget_override is not None:
        log.info(
            f"[{dataset_name}] KB budget override: {kb_budget} -> "
            f"{kb_budget_override} (head_ctx={len(head_context_positions)})"
        )
        kb_budget = int(kb_budget_override)

    initial_kb = head_biased_init_kb(
        doc_pool,
        doc_embs,
        centroids,
        head_set,
        kb_budget,
        stream,
    )
    return kb_budget, initial_kb


def instantiate_strategies(
    strategies_to_run,
    doc_pool,
    doc_embs,
    title_to_idx,
    initial_kb,
    stream,
    query_embs,
    num_windows,
):
    """从策略注册表创建策略对象，并注入同一个初始 KB。"""
    strategies = {}
    for strategy_name in strategies_to_run:
        strategy = STRATEGY_FACTORIES[strategy_name](
            doc_pool,
            doc_embs,
            title_to_idx,
        )
        strategy.set_kb(set(initial_kb))
        if hasattr(strategy, "_half"):
            strategy._half = num_windows // 2
        if hasattr(strategy, "_stream"):
            strategy._stream = stream
            strategy._query_embs = query_embs
        strategies[strategy_name] = strategy
    return strategies


def new_tracking_tables(strategies_to_run):
    """为每个策略创建逐窗口统计表。"""
    return {
        "recall": {
            name: {f"recall@{k}": [] for k in K_LIST}
            for name in strategies_to_run
        },
        "coverage": {name: [] for name in strategies_to_run},
        "residency": {
            name: new_residency_metrics() for name in strategies_to_run
        },
        "prepare_latency": {name: [] for name in strategies_to_run},
        "retrieval_latency": {name: [] for name in strategies_to_run},
        "query_latency": {name: [] for name in strategies_to_run},
        "step_latency": {name: [] for name in strategies_to_run},
        "step_overhead": {name: [] for name in strategies_to_run},
    }


def window_embeddings(window_queries, query_embs):
    """返回当前窗口查询的二维 embedding 矩阵。"""
    embeddings = np.array([query_embs[query["qidx"]] for query in window_queries])
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def collect_window_gold_doc_ids(window_queries, doc_pool, title_to_idx):
    """收集当前窗口所有 gold support 对应的稳定 doc_id。"""
    gold_doc_ids = set()
    for query in window_queries:
        for title in query.get("sf_titles", []):
            if title in title_to_idx:
                gold_doc_ids.add(doc_pool[title_to_idx[title]]["doc_id"])
    return gold_doc_ids


def evaluate_strategy_window(
    strategy_name,
    strategy,
    window_idx,
    window_queries,
    window_query_embs,
    window_gold_doc_ids,
    doc_pool,
    title_to_idx,
    doc_id_to_pool_index,
    query_embs,
    dataset_config,
    tracking,
):
    """对一个策略完成单个窗口的服务、评估和更新。"""
    prepare_start = time.perf_counter()
    strategy.prepare_window(window_queries, window_query_embs, window_idx)
    prepare_seconds = time.perf_counter() - prepare_start
    tracking["prepare_latency"][strategy_name].append(prepare_seconds)

    if strategy_name == "Oracle":
        # Oracle 是离线上界：它可以先看当前窗口，再评估当前窗口。
        strategy.step(window_queries, window_query_embs, window_idx)

    # 驻留指标只看真实热 KB，不看 get_effective_kb 里的临时扩展。
    record_residency_metrics(
        tracking["residency"][strategy_name],
        strategy.kb,
        window_queries,
        doc_pool,
        title_to_idx,
    )

    effective_kb = (
        strategy.get_effective_kb(window_queries, window_query_embs)
        if hasattr(strategy, "get_effective_kb")
        else strategy.kb
    )
    retrieval_start = time.perf_counter()
    recall_scores = recall_at_k(
        effective_kb,
        window_queries,
        doc_id_to_pool_index,
        strategy.doc_embs,
        title_to_idx,
        query_embs,
    )
    retrieval_seconds = time.perf_counter() - retrieval_start

    tracking["retrieval_latency"][strategy_name].append(retrieval_seconds)
    tracking["query_latency"][strategy_name].append(
        prepare_seconds + retrieval_seconds
    )
    for k in K_LIST:
        tracking["recall"][strategy_name][f"recall@{k}"].append(recall_scores[k])

    coverage = len(effective_kb & window_gold_doc_ids) / max(1, len(window_gold_doc_ids))
    tracking["coverage"][strategy_name].append(coverage)

    if strategy_name == "Oracle":
        tracking["step_latency"][strategy_name].append(0.0)
        tracking["step_overhead"][strategy_name].append(0)
        return

    if window_idx < dataset_config.get("freeze_until", 0):
        # freeze_until 用于消融实验：漂移发生前禁止策略更新。
        tracking["step_latency"][strategy_name].append(0.0)
        tracking["step_overhead"][strategy_name].append(0)
        return

    # 非 Oracle 策略在评估结束后才更新，保证当前窗口没有使用评估后信息。
    update_cost_before = strategy.update_cost
    step_start = time.perf_counter()
    strategy.step(window_queries, window_query_embs, window_idx)
    tracking["step_latency"][strategy_name].append(time.perf_counter() - step_start)
    tracking["step_overhead"][strategy_name].append(
        strategy.update_cost - update_cost_before
    )


def summarise_run(
    dataset_name,
    drift_mode,
    doc_pool,
    queries,
    kb_budget,
    window_size,
    strategies_to_run,
    strategies,
    tracking,
    elapsed_seconds,
):
    """把整次实验整理成可写入 JSON 的摘要。"""
    first_strategy = strategies_to_run[0]
    actual_num_windows = len(tracking["recall"][first_strategy]["recall@5"])
    half_window = actual_num_windows // 2
    summary = {}
    for strategy_name in strategies_to_run:
        summary[strategy_name] = summarise_strategy(
            strategy_name,
            strategies[strategy_name],
            tracking,
            half_window,
            window_size,
            actual_num_windows,
            kb_budget,
        )

    return {
        "dataset": dataset_name,
        "config": {
            "kb_budget": kb_budget,
            "pool_size": len(doc_pool),
            "n_windows": actual_num_windows,
            "window_size": window_size,
            "n_queries_available": len(queries),
            "drift": drift_mode,
            "k_list": K_LIST,
        },
        "summary": summary,
        "elapsed": round(elapsed_seconds, 1),
    }


def summarise_strategy(
    strategy_name,
    strategy,
    tracking,
    half_window,
    window_size,
    actual_num_windows,
    kb_budget,
):
    """汇总单个策略；保持历史 JSON 字段名不变，方便复用旧脚本。"""
    strategy_summary = {}
    for k in K_LIST:
        key = f"recall@{k}"
        values = tracking["recall"][strategy_name][key]
        strategy_summary[f"{key}_h1"] = percent_mean(values[:half_window])
        strategy_summary[f"{key}_h2"] = percent_mean(values[half_window:])
        strategy_summary[f"{key}_per_window"] = [round(x * 100, 2) for x in values]

    coverage_values = tracking["coverage"][strategy_name]
    strategy_summary["cov_h1"] = percent_mean(coverage_values[:half_window])
    strategy_summary["cov_h2"] = percent_mean(coverage_values[half_window:])
    strategy_summary["cov_per_window"] = [
        round(x * 100, 2) for x in coverage_values
    ]

    strategy_summary.update(
        strategy_cost_fields(strategy, actual_num_windows, window_size, kb_budget)
    )
    strategy_summary.update(latency_fields(strategy_name, tracking, window_size))
    strategy_summary.update(optional_strategy_logs(strategy))
    residency_summary = summarise_residency_metrics(
        tracking["residency"][strategy_name]
    )
    strategy_summary.update(residency_summary)
    strategy_summary.update(arc_compatible_metric_fields(strategy, residency_summary))
    return strategy_summary


def strategy_cost_fields(strategy, actual_num_windows, window_size, kb_budget):
    """所有缓存策略都会暴露的成本计数器和归一化 replacement 指标。"""
    replacement_count = int(strategy.update_cost)
    total_queries = max(1, int(actual_num_windows) * int(window_size))
    replacement_rate_per_query = replacement_count / float(total_queries)
    replacement_rate_per_window = replacement_count / float(max(1, actual_num_windows))
    cache_churn_rate = replacement_count / float(
        max(1, int(actual_num_windows) * int(kb_budget))
    )
    return {
        "update_cost": replacement_count,
        "cache_writes": replacement_count,
        "replacement_count": replacement_count,
        "cache_replacements": replacement_count,
        "replacement_rate_per_query": round(float(replacement_rate_per_query), 6),
        "replacement_rate_per_window": round(float(replacement_rate_per_window), 3),
        "cache_churn_rate": round(float(cache_churn_rate), 6),
        "cache_churn_rate_pct": round(float(cache_churn_rate) * 100, 3),
        "maint_retrieval_cost": strategy.maint_retrieval_cost,
        "serve_retrieval_cost": strategy.serve_retrieval_cost,
        "retrieval_cost": strategy.retrieval_cost,
        "cost": strategy.cost,
    }


def latency_fields(strategy_name, tracking, window_size):
    """按旧版 runner 的 JSON schema 汇总延迟。"""
    prepare = tracking["prepare_latency"][strategy_name]
    retrieval = tracking["retrieval_latency"][strategy_name]
    query = tracking["query_latency"][strategy_name]
    step = tracking["step_latency"][strategy_name]
    overhead = tracking["step_overhead"][strategy_name]
    query_ms = [round(x * 1000, 3) for x in query]
    return {
        "prepare_latency_mean_ms": mean_ms(prepare),
        "prepare_latency_per_window": [round(x * 1000, 3) for x in prepare],
        "retrieval_latency_mean_ms": mean_ms(retrieval),
        "retrieval_latency_per_window": [round(x * 1000, 3) for x in retrieval],
        "query_latency_mean_ms": mean_ms(query),
        "query_latency_per_window": query_ms,
        "per_query_latency_mean_ms": round(mean_ms(query) / window_size, 4)
        if window_size else 0.0,
        "per_query_latency_per_window": [
            round(x / window_size, 4) for x in query_ms
        ] if window_size else [],
        "step_latency_mean_ms": mean_ms(step),
        "step_latency_per_window": [round(x * 1000, 3) for x in step],
        "update_latency_mean_ms": mean_ms(step),
        "update_latency_per_window": [round(x * 1000, 3) for x in step],
        "step_overhead_mean": round(float(np.mean(overhead)), 2) if overhead else 0.0,
        "step_overhead_per_window": overhead,
    }


def arc_compatible_metric_fields(strategy, residency_summary):
    """补充 ARC-compatible 主指标里的 normalized AMAT。

    口径：
      - 经典 AMAT = T_hot + miss_rate * T_miss；
      - T_hot 默认 1；
      - T_miss 默认 10，表示 full-index/L2 访问远贵于 hot cache；
      - OnDemandFetch 这类策略如果实际做了 serve-time full-index fetch，
        就用实际 fetch rate 修正 miss rate。

    这比按 missing support 个数累计访问次数更接近系统论文里的 AMAT。
    需要模拟不同硬件/系统时，可通过环境变量调整：
      ``AMAT_HIT_COST`` 和 ``AMAT_MISS_PENALTY``。
    """
    n_queries = int(residency_summary.get("has_answer_queries", 0))
    if n_queries <= 0:
        miss_rate = 0.0
        serve_fetches_per_query = 0.0
    else:
        has_hits = float(residency_summary.get("has_answer_hits", 0))
        miss_rate = 1.0 - has_hits / float(n_queries)
        fetch_k = max(1, int(getattr(_P, "FETCH_TOP_K", 1)))
        serve_fetches = float(getattr(strategy, "serve_retrieval_cost", 0)) / fetch_k
        serve_fetches_per_query = serve_fetches / float(n_queries)

    l2_accesses_per_query = max(float(miss_rate), float(serve_fetches_per_query))
    hit_cost = float(getattr(_P, "AMAT_HIT_COST", 1.0))
    miss_penalty = float(getattr(_P, "AMAT_MISS_PENALTY", 10.0))
    amat = hit_cost + l2_accesses_per_query * miss_penalty
    return {
        "amat": round(float(amat), 3),
        "amat_normalized": round(float(amat), 3),
        "amat_hit_cost": round(float(hit_cost), 3),
        "amat_miss_penalty": round(float(miss_penalty), 3),
        "miss_rate": round(float(miss_rate), 6),
        "l2_accesses_per_query": round(float(l2_accesses_per_query), 3),
        "serve_fetches_per_query": round(float(serve_fetches_per_query), 3),
    }


def optional_strategy_logs(strategy):
    """如果策略暴露了诊断日志，就附加到结果 JSON 中。"""
    fields = {}
    if hasattr(strategy, "last_admission"):
        fields["last_admission"] = strategy.last_admission
    if hasattr(strategy, "drift_log"):
        fields["drift_log"] = strategy.drift_log
    if hasattr(strategy, "cost_log"):
        cost_log = strategy.cost_log
        fields["cost_log"] = cost_log
        fields["evictions"] = int(getattr(strategy, "total_evictions", 0))
        churn = [float(x.get("churn_rate", 0.0)) for x in cost_log]
        budgets = [int(x.get("write_budget", 0)) for x in cost_log]
        fields["churn_rate_mean"] = round(float(np.mean(churn)), 6) if churn else 0.0
        fields["write_budget_mean"] = round(float(np.mean(budgets)), 2) if budgets else 0.0
    return fields


def percent_mean(values):
    """把均值转成百分比，供摘要表使用。"""
    return round(float(np.mean(values)) * 100, 1) if values else 0.0


def mean_ms(values):
    """把秒级耗时均值转成毫秒。"""
    return round(float(np.mean(values)) * 1000, 3) if values else 0.0


def print_summary(all_results, strategies_to_run):
    """实验结束后打印一个紧凑的终端表格。"""
    print("\n" + "=" * 130)
    print("  ARC-Compatible Cache Metrics Under Single-Hop Query Drift")
    print("=" * 130)
    for dataset_name, dataset_result in all_results.items():
        config = dataset_result["config"]
        print(f"\n{'_' * 130}")
        print(
            f"  {dataset_name}  |  pool={config['pool_size']:,}  "
            f"KB={config['kb_budget']:,}  "
            f"stream={config['n_windows']}x{config['window_size']}  "
            f"elapsed={dataset_result['elapsed']:.0f}s"
        )
        print(f"{'_' * 130}")
        header = (
            f"{'Strategy':>18s} |"
            f" {'HasAns':>6s} {'AMAT':>6s} |"
            f" {'R@5 H2':>6s} {'Cov H2':>6s} |"
            f" {'Repl':>6s} {'Qlat':>8s} {'Ulat':>8s}"
        )
        print(header)
        print("-" * len(header))
        for strategy_name in strategies_to_run:
            if strategy_name not in dataset_result["summary"]:
                continue
            strategy_summary = dataset_result["summary"][strategy_name]
            cov_h2 = strategy_summary["cov_h2"]
            recall_h2 = strategy_summary["recall@5_h2"]
            print(
                f"{strategy_name:>18s} |"
                f" {strategy_summary.get('has_answer_rate', 0.0):>5.1f}%"
                f" {strategy_summary.get('amat', 0.0):>6.2f} |"
                f" {recall_h2:>5.1f}% {cov_h2:>5.1f}% |"
                f" {strategy_summary['replacement_count']:>6d}"
                f" {strategy_summary.get('query_latency_mean_ms', 0):>8.1f}"
                f" {strategy_summary.get('update_latency_mean_ms', 0):>8.1f}"
            )


def parse_args():
    """解析命令行参数；默认值保持不变，避免影响旧实验命令。"""
    parser = argparse.ArgumentParser(description="Motivation 1 实验入口")
    parser.add_argument("--n-windows", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument(
        "--drift",
        choices=["sudden", "gradual", "hybrid", "temporal", "full_gradual", "stationary"],
        default="sudden",
    )
    parser.add_argument(
        "--freeze-until",
        type=int,
        default=0,
        help=(
            "冻结非 Oracle 策略在 [0, freeze_until) 窗口内的更新。"
            "例如 --n-windows 100 时设为 50，可做 drift 前冻结 ablation。"
        ),
    )
    parser.add_argument("--n-source", type=int, default=None)
    parser.add_argument(
        "--n-stream-queries",
        type=int,
        default=None,
        help="限制 stream query 数量，但保持 document pool 不变。",
    )
    parser.add_argument("--datasets", nargs="+", default=["squad"])
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--kb-budget",
        type=int,
        default=None,
        help="手动指定 KB budget，跳过 kb_head_mult 自动估算公式。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategies_to_run = args.strategies or STRATEGY_ORDER
    all_results = {}
    for dataset_name in args.datasets:
        dataset_config = DATASET_CONFIGS[dataset_name].copy()
        dataset_config["n_windows"] = args.n_windows
        dataset_config["window_size"] = args.window_size
        dataset_config["freeze_until"] = args.freeze_until
        log.info(f"\n{'=' * 60}\n  Running: {dataset_name} ({args.drift})\n{'=' * 60}")
        all_results[dataset_name] = run_dataset(
            dataset_name,
            dataset_config,
            strategies_to_run,
            drift_mode=args.drift,
            loader_name=dataset_name,
            n_source=args.n_source,
            kb_budget_override=args.kb_budget,
            n_stream_queries=args.n_stream_queries,
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    freeze_tag = f"_freeze{args.freeze_until}" if args.freeze_until > 0 else ""
    output_name = args.output or f"results_{args.n_windows}w{freeze_tag}_{args.drift}.json"
    output_path = DATA_DIR / output_name
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {output_path}")

    print_summary(all_results, strategies_to_run)
    figure_suffix = f"_{args.n_windows}w{freeze_tag}_{args.drift}"
    generate_figures(all_results, strategies_to_run, suffix=figure_suffix)


if __name__ == "__main__":
    main()
