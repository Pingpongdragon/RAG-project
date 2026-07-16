"""Direct-evidence 实验的缓存驻留指标。

这些指标只看真实常驻的热 KB，不看 OnDemandFetch 在服务时临时扩展出来的
检索集合。这样可以区分：

  - 检索时临时拿到了答案；
  - 证据真的已经常驻在 L1/热缓存里。
"""

import numpy as np


def new_residency_metrics():
    """创建一个策略的逐查询 / 逐窗口驻留指标容器。"""
    return {
        "cold_fetches": [],
        "support_coverage": [],
        "has_answer": [],
        "cold_fetches_per_window": [],
        "support_coverage_per_window": [],
        "has_answer_per_window": [],
    }


def record_residency_metrics(metrics, kb_doc_ids, window_queries,
                             doc_pool, title_to_idx):
    """记录当前窗口里 gold support 是否已经在常驻 KB 中。

    参数：
        metrics: ``new_residency_metrics`` 返回的指标容器。
        kb_doc_ids: 当前策略的真实热 KB doc_id 集合。
        window_queries: 当前窗口查询。
        doc_pool/title_to_idx: 用于把 support title 映射回 doc_id。
    """
    window_cold_fetches = []
    window_support_coverage = []
    window_has_answer = []
    for query in window_queries:
        missing, total = missing_gold_count(kb_doc_ids, query, doc_pool, title_to_idx)
        if total <= 0:
            continue
        cold_fetches = float(missing)
        support_coverage = float(total - missing) / float(total)
        has_answer = float(missing == 0)

        metrics["cold_fetches"].append(cold_fetches)
        metrics["support_coverage"].append(support_coverage)
        metrics["has_answer"].append(has_answer)
        window_cold_fetches.append(cold_fetches)
        window_support_coverage.append(support_coverage)
        window_has_answer.append(has_answer)

    metrics["cold_fetches_per_window"].append(
        round(mean_or_zero(window_cold_fetches), 3)
    )
    metrics["support_coverage_per_window"].append(
        round(mean_or_zero(window_support_coverage) * 100, 2)
    )
    metrics["has_answer_per_window"].append(
        round(mean_or_zero(window_has_answer) * 100, 2)
    )


def summarise_residency_metrics(metrics):
    """把驻留指标汇总成写入结果 JSON 的字段。"""
    n_queries = len(metrics["has_answer"])
    n_has_answer = int(sum(metrics["has_answer"]))
    half = len(metrics["has_answer_per_window"]) // 2
    has_answer_windows = metrics["has_answer_per_window"]
    support_windows = metrics["support_coverage_per_window"]
    cold_fetch_windows = metrics["cold_fetches_per_window"]
    support_coverage_rate = round(
        mean_or_zero(metrics["support_coverage"]) * 100, 1
    )
    strict_has_answer_rate = (
        round((n_has_answer / n_queries) * 100, 1) if n_queries else 0.0
    )
    return {
        "cold_fetches_per_query": round(mean_or_zero(metrics["cold_fetches"]), 3),
        "support_coverage_rate": support_coverage_rate,
        # ARC 的 Has-Answer 按每个 retrieved item 是否驻留计数。对 gold support
        # 集，这等于 support coverage；显式输出别名，避免和严格 query 指标混淆。
        "arc_item_hit_rate": support_coverage_rate,
        "has_answer_rate": strict_has_answer_rate,
        "strict_has_answer_rate": strict_has_answer_rate,
        "has_answer_queries": n_queries,
        "has_answer_hits": n_has_answer,
        "has_answer_h1": round(mean_or_zero(has_answer_windows[:half]), 1),
        "has_answer_h2": round(mean_or_zero(has_answer_windows[half:]), 1),
        "has_answer_per_window": has_answer_windows,
        "support_coverage_h1": round(mean_or_zero(support_windows[:half]), 1),
        "support_coverage_h2": round(mean_or_zero(support_windows[half:]), 1),
        "support_coverage_per_window": support_windows,
        "cold_fetches_h1": round(mean_or_zero(cold_fetch_windows[:half]), 3),
        "cold_fetches_h2": round(mean_or_zero(cold_fetch_windows[half:]), 3),
        "cold_fetches_per_window": cold_fetch_windows,
    }


def missing_gold_count(kb_doc_ids, query, doc_pool, title_to_idx):
    """返回一个查询有多少 gold support 缺失于当前 KB。"""
    missing = 0
    total = 0
    for title in query.get("sf_titles", []):
        pool_index = title_to_idx.get(title)
        if pool_index is None:
            continue
        total += 1
        if doc_pool[pool_index]["doc_id"] not in kb_doc_ids:
            missing += 1
    return missing, total


def mean_or_zero(values):
    """空列表返回 0，避免 numpy warning。"""
    return float(np.mean(values)) if values else 0.0
