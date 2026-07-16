"""策略实例化与逐窗口因果评估控制流。

本模块只规定实验时序：普通策略必须先服务和计分，再看到当前窗口反馈并更新缓存；
Oracle 作为非因果上界，可以在计分前更新。指标的定义和最终汇总分别留在
``metrics.py`` 与 ``reporting.py``。
"""

import time

import numpy as np

if __package__:
    from .config import K_LIST, log
    from .metrics import new_residency_metrics, record_residency_metrics
    from .utils import recall_at_k
else:
    from config import K_LIST, log
    from metrics import new_residency_metrics, record_residency_metrics
    from utils import recall_at_k

from algorithms.cache.registry import STRATEGY_FACTORIES


def instantiate_strategies(
    strategy_names,
    prepared,
    initial_kb,
):
    """从统一注册表创建策略，并注入完全相同的初始缓存。"""

    strategies = {}
    for name in strategy_names:
        strategy = STRATEGY_FACTORIES[name](
            prepared.doc_pool,
            prepared.doc_embs,
            prepared.title_to_idx,
        )
        strategy.set_kb(set(initial_kb))
        if hasattr(strategy, "_stream"):
            strategy._stream = prepared.stream
            strategy._query_embs = prepared.query_embs
        strategies[name] = strategy
    return strategies


def evaluate_stream(
    prepared,
    dataset_config,
    strategy_names,
    initial_kb,
):
    """运行完整查询流，返回策略对象和未经汇总的逐窗口记录。"""

    num_windows = int(dataset_config["n_windows"])
    window_size = int(dataset_config["window_size"])
    strategies = instantiate_strategies(
        strategy_names,
        prepared,
        initial_kb,
    )
    tracking = new_tracking_tables(strategy_names)
    doc_id_to_pool_index = {
        document["doc_id"]: position
        for position, document in enumerate(prepared.doc_pool)
    }

    for window_idx in range(num_windows):
        start = window_idx * window_size
        window_queries = prepared.stream[start:start + window_size]
        if not window_queries:
            log.warning(
                "[%s] W%s: empty window, stopping",
                prepared.dataset_name,
                window_idx + 1,
            )
            break

        query_embs = window_embeddings(
            window_queries,
            prepared.query_embs,
        )
        gold_doc_ids = collect_window_gold_doc_ids(
            window_queries,
            prepared.doc_pool,
            prepared.title_to_idx,
        )
        for name in strategy_names:
            evaluate_strategy_window(
                name,
                strategies[name],
                window_idx,
                window_queries,
                query_embs,
                gold_doc_ids,
                prepared,
                doc_id_to_pool_index,
                dataset_config,
                tracking,
            )

        if window_idx % 5 == 0 or window_idx == num_windows - 1:
            coverage = {
                name: f"{tracking['coverage'][name][-1] * 100:.1f}%"
                for name in strategy_names
            }
            log.info(
                "[%s] W%s/%s Cov: %s",
                prepared.dataset_name,
                window_idx + 1,
                num_windows,
                coverage,
            )
    return strategies, tracking


def new_tracking_tables(strategy_names):
    """创建未经聚合的逐窗口统计容器。"""

    return {
        "recall": {
            name: {f"recall@{k}": [] for k in K_LIST}
            for name in strategy_names
        },
        "coverage": {name: [] for name in strategy_names},
        "residency": {
            name: new_residency_metrics() for name in strategy_names
        },
        "prepare_latency": {name: [] for name in strategy_names},
        "retrieval_latency": {name: [] for name in strategy_names},
        "query_latency": {name: [] for name in strategy_names},
        "step_latency": {name: [] for name in strategy_names},
        "step_overhead": {name: [] for name in strategy_names},
    }


def evaluate_strategy_window(
    strategy_name,
    strategy,
    window_idx,
    window_queries,
    window_query_embs,
    window_gold_doc_ids,
    prepared,
    doc_id_to_pool_index,
    dataset_config,
    tracking,
):
    """执行单个策略的一次“准备 -> 服务计分 -> 更新”。"""

    prepare_start = time.perf_counter()
    strategy.prepare_window(window_queries, window_query_embs, window_idx)
    prepare_seconds = time.perf_counter() - prepare_start
    tracking["prepare_latency"][strategy_name].append(prepare_seconds)

    if strategy_name == "Oracle":
        strategy.step(window_queries, window_query_embs, window_idx)

    # Has-Answer 等驻留指标只看持久 hot tier，不看临时 cold fetch。
    record_residency_metrics(
        tracking["residency"][strategy_name],
        strategy.kb,
        window_queries,
        prepared.doc_pool,
        prepared.title_to_idx,
    )
    effective_kb = (
        strategy.get_effective_kb(window_queries, window_query_embs)
        if hasattr(strategy, "get_effective_kb")
        else strategy.kb
    )

    retrieval_start = time.perf_counter()
    recalls = recall_at_k(
        effective_kb,
        window_queries,
        doc_id_to_pool_index,
        strategy.doc_embs,
        prepared.title_to_idx,
        prepared.query_embs,
    )
    retrieval_seconds = time.perf_counter() - retrieval_start
    tracking["retrieval_latency"][strategy_name].append(retrieval_seconds)
    tracking["query_latency"][strategy_name].append(
        prepare_seconds + retrieval_seconds
    )
    for k in K_LIST:
        tracking["recall"][strategy_name][f"recall@{k}"].append(recalls[k])

    coverage = len(effective_kb & window_gold_doc_ids) / max(
        1, len(window_gold_doc_ids)
    )
    tracking["coverage"][strategy_name].append(coverage)

    if strategy_name == "Oracle":
        _record_no_update(strategy_name, tracking)
        return
    if window_idx < int(dataset_config.get("freeze_until", 0)):
        _record_no_update(strategy_name, tracking)
        return

    # 关键因果边界：当前窗口已经完成计分，更新只能影响下一窗口。
    cost_before = strategy.update_cost
    step_start = time.perf_counter()
    strategy.step(window_queries, window_query_embs, window_idx)
    tracking["step_latency"][strategy_name].append(
        time.perf_counter() - step_start
    )
    tracking["step_overhead"][strategy_name].append(
        strategy.update_cost - cost_before
    )


def window_embeddings(window_queries, query_embs):
    """根据稳定 qidx 取出当前窗口的二维 embedding 矩阵。"""

    embeddings = np.asarray([
        query_embs[query["qidx"]] for query in window_queries
    ])
    return embeddings.reshape(1, -1) if embeddings.ndim == 1 else embeddings


def collect_window_gold_doc_ids(window_queries, doc_pool, title_to_idx):
    """只为离线 coverage 指标收集当前窗口 gold 文档。"""

    result = set()
    for query in window_queries:
        for title in query.get("sf_titles", []):
            if title in title_to_idx:
                result.add(doc_pool[title_to_idx[title]]["doc_id"])
    return result


def _record_no_update(strategy_name, tracking):
    tracking["step_latency"][strategy_name].append(0.0)
    tracking["step_overhead"][strategy_name].append(0)
