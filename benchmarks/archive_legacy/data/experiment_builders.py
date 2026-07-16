"""
实验数据集构建器 - 四种实验场景的高层接口。

每个 build_* 函数组装一个完整的 ExperimentDataset，
包括查询流（按 topic 调度采样）和候选文档池。

四种场景:
  1. Gradual Drift     - WoW 高斯漂移
  2. Sudden Shift      - WoW sigmoid 突变
  3. Cyclic Return     - WoW 周期回归
  4. HotpotQA Walk     - 实体图贪心游走
"""

import logging
import random
from typing import Dict, List, Optional

from benchmarks.archive_legacy.data.structures import ExperimentDataset, PoolDocument, QueryItem
from benchmarks.archive_legacy.data.wow_loader import (
    CyclicSchedule,
    GaussianDriftSchedule,
    SigmoidShiftSchedule,
)
from benchmarks.archive_legacy.data.structures import BenchmarkConfig
from benchmarks.archive_legacy.data.structures import (
    BIG_TOPICS, CYCLE_TOPICS, DIVERSE_TOPICS,
    GradualDriftConfig, SuddenShiftConfig, CyclicReturnConfig, HotpotQAConfig,
)
from benchmarks.archive_legacy.data.wow_loader import (
    build_stream,
    extract_topic_data,
    load_wow,
    merge_pool,
    select_topics,
)
from benchmarks.archive_legacy.data.hotpotqa_loader import (
    build_entity_graph,
    greedy_walk,
    hotpotqa_item_to_query_and_docs,
    load_hotpotqa,
)

logger = logging.getLogger(__name__)


def build_gradual_drift(
    total_queries: int = 300,
    topics: Optional[List[str]] = None,
    sigma: float = 0.18,
    seed: int = 42,
    pool_size: int = 5000,
    wow_split: str = "validation",
) -> ExperimentDataset:
    """
    Exp 1 - Gradual Drift.
    Gaussian activations => topics smoothly overlap and transition.
    """
    logger.info(f"Building Gradual Drift: {total_queries} queries, sigma={sigma}, pool_size={pool_size}")
    wow = load_wow(wow_split)
    if topics is None:
        topics = select_topics(wow, 10, preferred=BIG_TOPICS, seed=seed)
    logger.info(f"Topics: {topics}")

    tq, td = extract_topic_data(wow, topics)
    schedule = GaussianDriftSchedule(topics, sigma=sigma)
    stream, slog = build_stream(schedule, tq, total_queries, seed)

    gold_ids = {gid for q in stream for gid in q.gold_doc_ids}
    pool = merge_pool(td, gold_doc_ids=gold_ids, max_total=pool_size, seed=seed)
    logger.info(f"Gradual Drift: {len(stream)} queries, {len(pool)} pool docs")
    return ExperimentDataset(
        name="gradual_drift", description=f"Gaussian drift sigma={sigma}",
        query_stream=stream, document_pool=pool,
        topics=topics, schedule_type="gaussian",
        topic_schedule_log=slog,
    )


def build_sudden_shift(
    total_queries: int = 300,
    topics: Optional[List[str]] = None,
    steepness: float = 30.0,
    seed: int = 42,
    pool_size: int = 5000,
    wow_split: str = "validation",
) -> ExperimentDataset:
    """
    Exp 2 - Sudden Shift.
    Steep sigmoid transitions between semantically distant topics.
    """
    logger.info(f"Building Sudden Shift: {total_queries} queries, k={steepness}, pool_size={pool_size}")
    wow = load_wow(wow_split)
    if topics is None:
        topics = select_topics(wow, 10, preferred=DIVERSE_TOPICS, seed=seed)
    logger.info(f"Topics: {topics}")

    tq, td = extract_topic_data(wow, topics)
    schedule = SigmoidShiftSchedule(topics, steepness=steepness)
    stream, slog = build_stream(schedule, tq, total_queries, seed)

    gold_ids = {gid for q in stream for gid in q.gold_doc_ids}
    pool = merge_pool(td, gold_doc_ids=gold_ids, max_total=pool_size, seed=seed)
    logger.info(f"Sudden Shift: {len(stream)} queries, {len(pool)} pool docs")
    return ExperimentDataset(
        name="sudden_shift", description=f"Sigmoid shift k={steepness}",
        query_stream=stream, document_pool=pool,
        topics=topics, schedule_type="sigmoid",
        topic_schedule_log=slog,
    )


def build_cyclic_return(
    total_queries: int = 300,
    topics: Optional[List[str]] = None,
    n_cycles: int = 2,
    sigma: float = 0.08,
    seed: int = 42,
    pool_size: int = 5000,
    wow_split: str = "validation",
) -> ExperimentDataset:
    """
    Exp 3 - Cyclic Return.
    Topics re-appear periodically => tests re-adaptation speed.
    """
    logger.info(f"Building Cyclic Return: {total_queries} queries, "
                f"cycles={n_cycles}, sigma={sigma}")
    wow = load_wow("validation")
    if topics is None:
        topics = select_topics(wow, 5, preferred=CYCLE_TOPICS, seed=seed)
    logger.info(f"Topics: {topics}")

    tq, td = extract_topic_data(wow, topics)
    schedule = CyclicSchedule(topics, n_cycles=n_cycles, sigma=sigma)
    stream, slog = build_stream(schedule, tq, total_queries, seed)

    gold_ids = {gid for q in stream for gid in q.gold_doc_ids}
    pool = merge_pool(td, gold_doc_ids=gold_ids, max_total=pool_size, seed=seed)
    logger.info(f"Cyclic Return: {len(stream)} queries, {len(pool)} pool docs, "
                f"pattern={'->'.join(topics)} x{n_cycles}")
    return ExperimentDataset(
        name="cyclic_return",
        description=f"{topics} x{n_cycles} cycles",
        query_stream=stream, document_pool=pool,
        topics=topics, schedule_type="cyclic",
        topic_schedule_log=slog,
    )


def build_hotpotqa_entity_walk(
    total_queries: int = 400,
    seed: int = 42,
    max_pool: int = 50000,
) -> ExperimentDataset:
    """
    HotpotQA: entity-graph greedy walk.

    Questions that share supporting-fact titles are linked in a graph.
    A greedy walk creates natural topic drift without any hard classification.
    """
    logger.info(f"Building HotpotQA Entity Walk: {total_queries} queries")
    hp_data = load_hotpotqa()
    logger.info(f"Loaded {len(hp_data)} HotpotQA items")

    adj = build_entity_graph(hp_data)
    walk = greedy_walk(adj, len(hp_data), total_queries, seed=seed)
    logger.info(f"Greedy walk: {len(walk)} steps")

    stream: List[QueryItem] = []
    all_docs: List[PoolDocument] = []
    seen_docs: set = set()
    gold_ids_all: set = set()

    for idx in walk:
        item = hp_data[idx]
        q, docs = hotpotqa_item_to_query_and_docs(item, idx)
        stream.append(q)
        gold_ids_all.update(q.gold_doc_ids)
        for d in docs:
            if d.doc_id not in seen_docs:
                seen_docs.add(d.doc_id)
                all_docs.append(d)

    if len(all_docs) > max_pool:
        rng = random.Random(seed)
        gold = [d for d in all_docs if d.doc_id in gold_ids_all]
        other = [d for d in all_docs if d.doc_id not in gold_ids_all]
        rng.shuffle(other)
        budget = max(max_pool - len(gold), 0)
        all_docs = gold + other[:budget]
        logger.info(f"Pool subsampled: {len(all_docs)} docs "
                     f"({len(gold)} gold kept)")

    topics = list(set(d.topic for d in all_docs))

    logger.info(f"HotpotQA Entity Walk: {len(stream)} queries, {len(all_docs)} pool docs, "
                f"{len(topics)} unique article titles")
    return ExperimentDataset(
        name="hotpotqa_entity_walk",
        description=f"Entity-graph greedy walk, {total_queries} steps",
        query_stream=stream,
        document_pool=all_docs,
        topics=topics,
        schedule_type="entity_walk",
    )


def build_all_datasets(
    seed: int = 42,
    total_queries: int = 300,
    pool_size: int = 5000,
    wow_split: str = "validation",
) -> Dict[str, ExperimentDataset]:
    return {
        "gradual_drift": build_gradual_drift(total_queries=total_queries, seed=seed, pool_size=pool_size, wow_split=wow_split),
        "sudden_shift":  build_sudden_shift(total_queries=total_queries, seed=seed, pool_size=pool_size, wow_split=wow_split),
        "cyclic_return":  build_cyclic_return(total_queries=total_queries, seed=seed, pool_size=pool_size, wow_split=wow_split),
        "hotpotqa_walk":  build_hotpotqa_entity_walk(total_queries=total_queries, seed=seed, max_pool=pool_size),
    }
