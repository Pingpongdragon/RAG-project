"""逐窗口记录的汇总、ARC-compatible 指标与终端表格。"""

import numpy as np

if __package__:
    from .config import K_LIST
    from . import config as experiment_config
    from .metrics import summarise_residency_metrics
else:
    from config import K_LIST
    import config as experiment_config
    from metrics import summarise_residency_metrics

from algorithms.cache.params import PARAMS as _P


def summarise_run(
    prepared,
    kb_budget,
    window_size,
    strategy_names,
    strategies,
    tracking,
    elapsed_seconds,
    warmup_windows,
):
    """把一次数据集实验整理成保持向后兼容的 JSON 对象。"""

    first_strategy = strategy_names[0]
    actual_windows = len(
        tracking["recall"][first_strategy]["recall@5"]
    )
    half_window = actual_windows // 2
    summary = {
        name: summarise_strategy(
            name,
            strategies[name],
            tracking,
            half_window,
            window_size,
            actual_windows,
            kb_budget,
        )
        for name in strategy_names
    }
    audit = prepared.diagnostics
    return {
        "dataset": prepared.dataset_name,
        "config": {
            "kb_budget": kb_budget,
            "pool_size": len(prepared.doc_pool),
            "kb_pool_ratio": round(
                float(kb_budget) / max(1, len(prepared.doc_pool)), 6
            ),
            "n_windows": actual_windows,
            "window_size": window_size,
            "n_queries_available": len(prepared.queries),
            "seed": int(experiment_config.SEED),
            "data_seed": int(experiment_config.DATA_SEED),
            "workload": prepared.workload,
            "initialization": "causal-prefix",
            "warmup_windows": int(warmup_windows),
            "stream_sampling": dict(audit["stream_sampling"]),
            "support_reuse": dict(audit["support_reuse"]),
            "query_drift": dict(audit["query_drift"]),
            "workload_factors": dict(audit["workload_factors"]),
            "factorized_construction": audit["factorized_construction"],
            "warmup_audit": dict(audit["warmup_audit"]),
            "temporal_sampling": audit["temporal_sampling"],
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
    """汇总单个策略，同时保留历史 JSON 字段名。"""

    result = {}
    for k in K_LIST:
        key = f"recall@{k}"
        values = tracking["recall"][strategy_name][key]
        result[key] = percent_mean(values)
        result[f"{key}_mean"] = result[key]
        result[f"{key}_h1"] = percent_mean(values[:half_window])
        result[f"{key}_h2"] = percent_mean(values[half_window:])
        result[f"{key}_per_window"] = [
            round(value * 100, 2) for value in values
        ]

    coverage = tracking["coverage"][strategy_name]
    result["coverage"] = percent_mean(coverage)
    result["coverage_mean"] = result["coverage"]
    result["cov_h1"] = percent_mean(coverage[:half_window])
    result["cov_h2"] = percent_mean(coverage[half_window:])
    result["cov_per_window"] = [
        round(value * 100, 2) for value in coverage
    ]

    result.update(
        strategy_cost_fields(
            strategy, actual_num_windows, window_size, kb_budget
        )
    )
    result.update(latency_fields(strategy_name, tracking, window_size))
    result.update(optional_strategy_logs(strategy))
    residency = summarise_residency_metrics(
        tracking["residency"][strategy_name]
    )
    result.update(residency)
    result.update(arc_compatible_metric_fields(strategy, residency))
    return result


def strategy_cost_fields(strategy, actual_num_windows, window_size, kb_budget):
    """汇总 replacement、churn 与底层访问计数器。"""

    replacements = int(strategy.update_cost)
    total_queries = max(1, int(actual_num_windows) * int(window_size))
    return {
        "update_cost": replacements,
        "cache_writes": replacements,
        "replacement_count": replacements,
        "cache_replacements": replacements,
        "replacement_rate_per_query": round(
            replacements / float(total_queries), 6
        ),
        "replacement_rate_per_window": round(
            replacements / float(max(1, actual_num_windows)), 3
        ),
        "cache_churn_rate": round(
            replacements
            / float(max(1, int(actual_num_windows) * int(kb_budget))),
            6,
        ),
        "cache_churn_rate_pct": round(
            replacements
            / float(max(1, int(actual_num_windows) * int(kb_budget)))
            * 100,
            3,
        ),
        "maint_retrieval_cost": strategy.maint_retrieval_cost,
        "serve_retrieval_cost": strategy.serve_retrieval_cost,
        "retrieval_cost": strategy.retrieval_cost,
        "cost": strategy.cost,
    }


def latency_fields(strategy_name, tracking, window_size):
    """将逐窗口秒级耗时转换成历史 JSON schema 使用的毫秒字段。"""

    prepare = tracking["prepare_latency"][strategy_name]
    retrieval = tracking["retrieval_latency"][strategy_name]
    query = tracking["query_latency"][strategy_name]
    step = tracking["step_latency"][strategy_name]
    overhead = tracking["step_overhead"][strategy_name]
    query_ms = [round(value * 1000, 3) for value in query]
    return {
        "prepare_latency_mean_ms": mean_ms(prepare),
        "prepare_latency_per_window": [
            round(value * 1000, 3) for value in prepare
        ],
        "retrieval_latency_mean_ms": mean_ms(retrieval),
        "retrieval_latency_per_window": [
            round(value * 1000, 3) for value in retrieval
        ],
        "query_latency_mean_ms": mean_ms(query),
        "query_latency_per_window": query_ms,
        "per_query_latency_mean_ms": round(
            mean_ms(query) / window_size, 4
        ) if window_size else 0.0,
        "per_query_latency_per_window": [
            round(value / window_size, 4) for value in query_ms
        ] if window_size else [],
        "step_latency_mean_ms": mean_ms(step),
        "step_latency_per_window": [
            round(value * 1000, 3) for value in step
        ],
        "update_latency_mean_ms": mean_ms(step),
        "update_latency_per_window": [
            round(value * 1000, 3) for value in step
        ],
        "step_overhead_mean": round(
            float(np.mean(overhead)), 2
        ) if overhead else 0.0,
        "step_overhead_per_window": overhead,
    }


def arc_compatible_metric_fields(strategy, residency_summary):
    """根据 hot miss 或实际 L2 fetch 计算 normalized AMAT。"""

    num_queries = int(residency_summary.get("has_answer_queries", 0))
    if num_queries <= 0:
        miss_rate = 0.0
        serve_fetches_per_query = 0.0
    else:
        hits = float(residency_summary.get("has_answer_hits", 0))
        miss_rate = 1.0 - hits / float(num_queries)
        fetch_k = max(1, int(getattr(_P, "FETCH_TOP_K", 1)))
        serve_fetches = float(strategy.serve_retrieval_cost) / fetch_k
        serve_fetches_per_query = serve_fetches / float(num_queries)

    l2_rate = max(float(miss_rate), float(serve_fetches_per_query))
    hit_cost = float(getattr(_P, "AMAT_HIT_COST", 1.0))
    miss_penalty = float(getattr(_P, "AMAT_MISS_PENALTY", 10.0))
    amat = hit_cost + l2_rate * miss_penalty
    return {
        "amat": round(float(amat), 3),
        "amat_normalized": round(float(amat), 3),
        "amat_hit_cost": round(float(hit_cost), 3),
        "amat_miss_penalty": round(float(miss_penalty), 3),
        "miss_rate": round(float(miss_rate), 6),
        "l2_accesses_per_query": round(float(l2_rate), 3),
        "serve_fetches_per_query": round(
            float(serve_fetches_per_query), 3
        ),
    }


def optional_strategy_logs(strategy):
    """把策略公开的诊断日志附加到结果 JSON。"""

    fields = {}
    for attribute in (
        "method_version",
        "last_admission",
        "drift_log",
        "prefetch_log",
        "expert_log",
        "evidence_route_log",
        "downstream_log",
    ):
        if hasattr(strategy, attribute):
            fields[attribute] = getattr(strategy, attribute)
    if hasattr(strategy, "cost_log"):
        cost_log = strategy.cost_log
        fields["cost_log"] = cost_log
        fields["evictions"] = int(getattr(strategy, "total_evictions", 0))
        churn = [float(item.get("churn_rate", 0.0)) for item in cost_log]
        budgets = [int(item.get("write_budget", 0)) for item in cost_log]
        fields["churn_rate_mean"] = round(
            float(np.mean(churn)), 6
        ) if churn else 0.0
        fields["write_budget_mean"] = round(
            float(np.mean(budgets)), 2
        ) if budgets else 0.0
    return fields


def percent_mean(values):
    return round(float(np.mean(values)) * 100, 1) if values else 0.0


def mean_ms(values):
    return round(float(np.mean(values)) * 1000, 3) if values else 0.0


def print_summary(all_results, strategy_names):
    """打印紧凑的主指标表。"""

    print("\n" + "=" * 130)
    print("  ARC-Compatible Cache Metrics Under Query-Workload Drift")
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
            f"{'Strategy':>18s} | {'HasAns':>6s} {'AMAT':>6s} |"
            f" {'R@5':>6s} {'Cov':>6s} | {'Repl':>6s}"
            f" {'Qlat':>8s} {'Ulat':>8s}"
        )
        print(header)
        print("-" * len(header))
        for name in strategy_names:
            if name not in dataset_result["summary"]:
                continue
            result = dataset_result["summary"][name]
            print(
                f"{name:>18s} |"
                f" {result.get('has_answer_rate', 0.0):>5.1f}%"
                f" {result.get('amat', 0.0):>6.2f} |"
                f" {result['recall@5']:>5.1f}%"
                f" {result['coverage']:>5.1f}% |"
                f" {result['replacement_count']:>6d}"
                f" {result.get('query_latency_mean_ms', 0):>8.1f}"
                f" {result.get('update_latency_mean_ms', 0):>8.1f}"
            )
