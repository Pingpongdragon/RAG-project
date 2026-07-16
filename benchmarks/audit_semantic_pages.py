#!/usr/bin/env python3
"""审计“语义页缓存”是否提供了文档级 LRU 没有的可利用信号。

本脚本不是新的 DRIP 实现，也不读取未来 query 做在线决策。它回答三个更基础的
可行性问题：

1. 把语义相近文档打包后，page-level reuse 是否高于 exact-document reuse；
2. 只看当前 query embedding，能否把 required evidence 路由到少量语义页；
3. 在相同文档容量下，路由式 Page-LRU 的 Has-Answer 提升能否覆盖页加载造成的
   write amplification。

对照组 ``GoldDoc-LRU`` 和 ``GoldPage-LRU`` 使用事后 gold access trace，只是
reactive cacheability 诊断，并不是预取上界；``RoutedPage-LRU`` 才是只使用 query embedding 的
因果方法。所有 cache 都由同一组 warm-up query 做 query-conditioned 初始化，再按
“先评估、后更新”处理正式流。

示例：

    python benchmarks/audit_semantic_pages.py \
      --datasets squad_direct streamingqa_official \
      --page-sizes 128 --route-widths 1 2 4
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
MO1_DIR = PROJECT_DIR / "experiments" / "direct"
sys.path.insert(0, str(MO1_DIR))
sys.path.append(str(PROJECT_DIR))

# StreamingQA 的文本镜像已经缓存在本机；审计不应在运行时下载数据。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config import CACHE_DIR, DATA_SEED, SF_HIT_THRESH  # noqa: E402
from loaders import LOADERS  # noqa: E402
from loaders_temporal import TEMPORAL_LOADERS  # noqa: E402
from utils import compute_embeddings  # noqa: E402
from experiments.common.factorized_workload import (  # noqa: E402
    NATURAL,
    SOURCE_POOL_ALL,
    SOURCE_POOL_ROLES,
    build_factorized_workload,
    resolve_workload,
    split_disjoint_source_pool,
)
from experiments.common.stream_protocol import (  # noqa: E402
    causal_prefix_init_kb,
    chronological_sample,
    embedding_content_fingerprint,
    support_reuse_diagnostics,
    warmup_overlap_diagnostics,
)


AUDIT_VERSION = "semantic-page-feasibility-v1"


@dataclass(frozen=True)
class Protocol:
    """与当前主实验一致的数据规模和 stream 协议。"""

    dataset: str
    workload: str
    n_source: int | None
    n_windows: int
    window_size: int
    warmup_windows: int = 3
    temporal_sampling: str = "prefix"
    min_support_frequency: int = 1
    family_mode: str = "auto"


PROTOCOLS = {
    "squad_direct": Protocol(
        dataset="squad",
        workload="factorized_recurring",
        n_source=1000,
        n_windows=20,
        window_size=25,
        warmup_windows=1,
    ),
    "streamingqa_official": Protocol(
        dataset="streamingqa_official",
        workload="natural_temporal",
        n_source=None,
        n_windows=50,
        window_size=500,
        temporal_sampling="window_span",
    ),
    "fever_direct": Protocol(
        dataset="fever",
        workload="factorized_recurring",
        n_source=8000,
        n_windows=20,
        window_size=25,
        warmup_windows=4,
    ),
}


@dataclass
class PreparedDataset:
    alias: str
    protocol: Protocol
    doc_pool: list[dict]
    title_to_idx: dict[str, int]
    doc_embs: np.ndarray
    query_embs: np.ndarray
    warmup: list[dict]
    stream: list[dict]
    construction: dict | None
    temporal_sampling: dict | None
    source_split: dict | None


class VariableObjectLRU:
    """按文档容量约束的可变大小对象 LRU。

    文档缓存中一个 object 含一个文档；page cache 中一个 object 含一整页文档。
    因此两者都严格使用同一个 ``capacity_docs``，同时分别记录 object I/O 和 document
    traffic，避免把“一次加载 128 个文档”伪装成一次廉价 replacement。
    """

    def __init__(self, object_members: list[np.ndarray], capacity_docs: int):
        self.object_members = object_members
        self.capacity_docs = max(1, int(capacity_docs))
        self.objects: OrderedDict[int, None] = OrderedDict()
        self.resident_docs: set[int] = set()
        self.used_docs = 0

    def access(self, object_ids: Iterable[int]) -> dict:
        object_misses = 0
        object_loads = 0
        documents_loaded = 0
        objects_evicted = 0
        documents_evicted = 0
        oversized_objects = 0
        loaded_members: list[int] = []

        # 一个 query 内重复路由到同一 object 只访问一次，并保留路由顺序。
        unique_ids = list(dict.fromkeys(int(value) for value in object_ids))
        for object_id in unique_ids:
            if object_id in self.objects:
                self.objects.move_to_end(object_id)
                continue

            object_misses += 1
            members = self.object_members[object_id]
            object_size = int(len(members))
            if object_size > self.capacity_docs:
                oversized_objects += 1
                continue

            while self.objects and self.used_docs + object_size > self.capacity_docs:
                victim, _ = self.objects.popitem(last=False)
                victim_members = self.object_members[victim]
                self.resident_docs.difference_update(
                    int(value) for value in victim_members
                )
                self.used_docs -= int(len(victim_members))
                objects_evicted += 1
                documents_evicted += int(len(victim_members))

            self.objects[object_id] = None
            self.resident_docs.update(int(value) for value in members)
            self.used_docs += object_size
            object_loads += 1
            documents_loaded += object_size
            loaded_members.extend(int(value) for value in members)

        return {
            "object_misses": object_misses,
            "object_loads": object_loads,
            "documents_loaded": documents_loaded,
            "objects_evicted": objects_evicted,
            "documents_evicted": documents_evicted,
            "oversized_objects": oversized_objects,
            "loaded_members": loaded_members,
        }


def prepare_dataset(alias: str, args: argparse.Namespace) -> PreparedDataset:
    """完全复用主实验的 loader、warm-up 与 workload constructor。"""

    base = PROTOCOLS[alias]
    protocol = Protocol(
        dataset=base.dataset,
        workload=(getattr(args, "workload", None) or base.workload),
        n_source=args.n_source if args.n_source is not None else base.n_source,
        n_windows=args.n_windows or base.n_windows,
        window_size=args.window_size or base.window_size,
        warmup_windows=(
            args.warmup_windows
            if args.warmup_windows is not None
            else base.warmup_windows
        ),
        temporal_sampling=base.temporal_sampling,
        min_support_frequency=base.min_support_frequency,
        family_mode=base.family_mode,
    )
    print(f"[{alias}] loading {protocol.dataset}", flush=True)
    if protocol.dataset in TEMPORAL_LOADERS:
        doc_pool, queries, title_to_idx = TEMPORAL_LOADERS[protocol.dataset]()
    else:
        doc_pool, queries, title_to_idx = LOADERS[protocol.dataset](
            n_source=protocol.n_source
        )

    workload = resolve_workload(queries, protocol.workload)
    source_pool_role = str(
        getattr(args, "source_pool", SOURCE_POOL_ALL) or SOURCE_POOL_ALL
    )
    source_split = None
    if workload == NATURAL and source_pool_role != SOURCE_POOL_ALL:
        raise ValueError(
            "source-pool calibration/test splits apply only to controlled "
            "factorized workloads"
        )
    if workload != NATURAL and source_pool_role != SOURCE_POOL_ALL:
        queries, split_stats = split_disjoint_source_pool(
            queries,
            doc_pool,
            title_to_idx,
            role=source_pool_role,
            seed=int(getattr(args, "source_pool_seed", 1729)),
            calibration_fraction=float(
                getattr(args, "source_pool_calibration_fraction", 0.5)
            ),
            family_mode=protocol.family_mode,
        )
        source_split = split_stats.as_dict()
        if not source_split["overlap_assertion"]:
            raise AssertionError(f"source-pool leakage: {source_split}")
    warmup_size = protocol.warmup_windows * protocol.window_size
    evaluation_size = protocol.n_windows * protocol.window_size
    temporal_stats = None
    if workload == NATURAL:
        queries, sampling = chronological_sample(
            queries,
            warmup_size=warmup_size,
            evaluation_size=evaluation_size,
            mode=protocol.temporal_sampling,
            block_size=protocol.window_size,
        )
        temporal_stats = sampling.as_dict()

    embedding_tag = (
        f"{protocol.dataset}_{protocol.n_windows}w_{protocol.window_size}s"
    )
    doc_embs, query_embs = compute_embeddings(
        doc_pool, queries, tag=embedding_tag
    )

    construction = None
    if workload == NATURAL:
        warmup = list(queries[:warmup_size])
        stream = list(queries[warmup_size:])
    else:
        stream, warmup, stats = build_factorized_workload(
            queries,
            doc_pool,
            title_to_idx,
            n_windows=protocol.n_windows,
            window_size=protocol.window_size,
            workload=workload,
            seed=int(getattr(args, "workload_seed", DATA_SEED)),
            min_support_frequency=protocol.min_support_frequency,
            family_mode=protocol.family_mode,
            warmup_size=warmup_size,
        )
        construction = stats.as_dict()

    overlap = warmup_overlap_diagnostics(
        warmup, stream, warmup_size
    ).as_dict()
    if overlap["evaluation_overlap"]:
        raise ValueError(f"warm-up leakage detected: {overlap}")
    print(
        f"[{alias}] pool={len(doc_pool):,}, warmup={len(warmup):,}, "
        f"evaluation={len(stream):,}",
        flush=True,
    )
    return PreparedDataset(
        alias=alias,
        protocol=protocol,
        doc_pool=doc_pool,
        title_to_idx=title_to_idx,
        doc_embs=np.asarray(doc_embs, dtype=np.float32),
        query_embs=np.asarray(query_embs, dtype=np.float32),
        warmup=warmup,
        stream=stream,
        construction=construction,
        temporal_sampling=temporal_stats,
        source_split=source_split,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(
        np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None
    )


def build_balanced_semantic_pages(
    dataset: PreparedDataset,
    target_page_size: int,
    seed: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """先做 spherical KMeans，再把 overflow 重分配到未满页。

    普通 KMeans/IVF list 大小可能极不均衡，直接比较会违反固定容量。这里把每页
    限制在 ``target_page_size`` 个文档；重分配损失单独报告，不能隐藏平衡代价。
    """

    target_page_size = max(1, int(target_page_size))
    num_docs = len(dataset.doc_pool)
    num_pages = max(1, int(math.ceil(num_docs / target_page_size)))
    fingerprint = embedding_content_fingerprint(dataset.doc_pool, [])
    cache_path = CACHE_DIR / (
        f"semantic_pages_v1_{fingerprint}_p{target_page_size}_s{seed}.npz"
    )
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        labels = cached["labels"].astype(np.int32)
        centroids = cached["centroids"].astype(np.float32)
        packed = cached["quality"].astype(float).tolist()
        stats = partition_stats(labels, centroids, target_page_size)
        stats.update({
            "nearest_centroid_cosine_mean": round(packed[0], 6),
            "assigned_centroid_cosine_mean": round(packed[1], 6),
            "balancing_cosine_loss": round(packed[0] - packed[1], 6),
            "cache": str(cache_path.relative_to(PROJECT_DIR)),
        })
        return labels, centroids, stats

    from sklearn.cluster import MiniBatchKMeans

    embeddings = _normalize_rows(dataset.doc_embs)
    print(
        f"[{dataset.alias}] fitting {num_pages} balanced pages "
        f"(max {target_page_size} docs/page)",
        flush=True,
    )
    if num_pages == 1:
        raw_labels = np.zeros(num_docs, dtype=np.int32)
        raw_centroids = _normalize_rows(embeddings.mean(axis=0, keepdims=True))
    else:
        model = MiniBatchKMeans(
            n_clusters=num_pages,
            random_state=int(seed),
            batch_size=min(4096, max(512, num_pages * 4)),
            n_init=1,
            max_iter=50,
            max_no_improvement=10,
            reassignment_ratio=0.01,
        )
        raw_labels = model.fit_predict(embeddings).astype(np.int32)
        raw_centroids = _normalize_rows(model.cluster_centers_)

    scores = np.asarray(embeddings @ raw_centroids.T, dtype=np.float32)
    nearest_similarity = scores[np.arange(num_docs), raw_labels]
    labels = raw_labels.copy()
    remaining_capacity = np.full(num_pages, target_page_size, dtype=np.int32)
    overflow: list[int] = []

    for page_id in range(num_pages):
        members = np.flatnonzero(raw_labels == page_id)
        if len(members) <= target_page_size:
            remaining_capacity[page_id] -= len(members)
            continue
        order = members[np.argsort(scores[members, page_id])[::-1]]
        labels[order[target_page_size:]] = -1
        overflow.extend(int(value) for value in order[target_page_size:])
        remaining_capacity[page_id] = 0

    # 先放置“替代中心选择更明确”的 overflow 文档，减少末尾强制分配噪声。
    if overflow:
        overflow_array = np.asarray(overflow, dtype=np.int32)
        sorted_scores = np.sort(scores[overflow_array], axis=1)
        confidence = sorted_scores[:, -1] - sorted_scores[:, -2]
        overflow_array = overflow_array[np.argsort(confidence)[::-1]]
        for doc_idx in overflow_array:
            available = np.flatnonzero(remaining_capacity > 0)
            if not len(available):
                raise AssertionError("balanced page assignment exhausted capacity")
            page_id = int(available[np.argmax(scores[doc_idx, available])])
            labels[doc_idx] = page_id
            remaining_capacity[page_id] -= 1

    if np.any(labels < 0):
        raise AssertionError("semantic page assignment left documents unassigned")

    centroids = np.zeros((num_pages, embeddings.shape[1]), dtype=np.float32)
    for page_id in range(num_pages):
        members = embeddings[labels == page_id]
        if not len(members):
            raise AssertionError("balanced semantic packing created an empty page")
        centroids[page_id] = members.mean(axis=0)
    centroids = _normalize_rows(centroids)
    assigned_similarity = np.sum(embeddings * centroids[labels], axis=1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    quality = np.asarray(
        [nearest_similarity.mean(), assigned_similarity.mean()],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_path,
        labels=labels.astype(np.int32),
        centroids=centroids.astype(np.float32),
        quality=quality,
    )
    stats = partition_stats(labels, centroids, target_page_size)
    stats.update({
        "nearest_centroid_cosine_mean": round(float(quality[0]), 6),
        "assigned_centroid_cosine_mean": round(float(quality[1]), 6),
        "balancing_cosine_loss": round(float(quality[0] - quality[1]), 6),
        "cache": str(cache_path.relative_to(PROJECT_DIR)),
    })
    return labels, centroids, stats


def partition_stats(
    labels: np.ndarray,
    centroids: np.ndarray,
    target_page_size: int,
) -> dict:
    sizes = np.bincount(labels, minlength=len(centroids))
    return {
        "target_page_size": int(target_page_size),
        "num_pages": int(len(centroids)),
        "page_size_min": int(sizes.min()),
        "page_size_mean": round(float(sizes.mean()), 3),
        "page_size_p95": round(float(np.percentile(sizes, 95)), 3),
        "page_size_max": int(sizes.max()),
    }


def page_members(labels: np.ndarray, num_pages: int) -> list[np.ndarray]:
    return [
        np.flatnonzero(labels == page_id).astype(np.int32)
        for page_id in range(num_pages)
    ]


def query_routes(
    queries: list[dict],
    query_embs: np.ndarray,
    centroids: np.ndarray,
    max_width: int,
) -> np.ndarray:
    """只用当前 query embedding 路由 Top-L semantic pages。"""

    max_width = min(max(1, int(max_width)), len(centroids))
    indices = np.asarray([int(query["qidx"]) for query in queries], dtype=np.int64)
    routes = np.empty((len(indices), max_width), dtype=np.int32)
    for start in range(0, len(indices), 1024):
        block = _normalize_rows(query_embs[indices[start:start + 1024]])
        similarities = np.asarray(block @ centroids.T, dtype=np.float32)
        if max_width == 1:
            routes[start:start + len(block), 0] = np.argmax(
                similarities, axis=1
            )
            continue
        candidates = np.argpartition(
            similarities, -max_width, axis=1
        )[:, -max_width:]
        candidate_scores = np.take_along_axis(
            similarities, candidates, axis=1
        )
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        routes[start:start + len(block)] = np.take_along_axis(
            candidates, order, axis=1
        )
    return routes


def support_positions(
    queries: list[dict], title_to_idx: dict[str, int]
) -> tuple[list[set[int]], int]:
    supports = []
    missing = 0
    for query in queries:
        positions = set()
        for title in query.get("sf_titles", ()):
            if title in title_to_idx:
                positions.add(int(title_to_idx[title]))
            else:
                missing += 1
        supports.append(positions)
    return supports, missing


def generic_reuse_diagnostics(
    values_per_query: list[set[int]], window_size: int
) -> dict:
    seen: set[int] = set()
    occurrences = repeated = answerable = with_values = 0
    for values in values_per_query:
        if not values:
            continue
        with_values += 1
        occurrences += len(values)
        repeated += len(values & seen)
        answerable += int(values.issubset(seen))
        seen.update(values)

    windows = [
        set().union(*values_per_query[start:start + window_size])
        for start in range(0, len(values_per_query), window_size)
    ]
    jaccards = []
    for previous, current in zip(windows, windows[1:]):
        union = previous | current
        if union:
            jaccards.append(len(previous & current) / len(union))
    return {
        "queries_with_values": with_values,
        "occurrences": occurrences,
        "unique_values": len(seen),
        "repeated_occurrence_rate": round(repeated / max(1, occurrences), 6),
        "past_answerable_query_rate": round(answerable / max(1, with_values), 6),
        "adjacent_window_jaccard_mean": round(
            float(np.mean(jaccards)) if jaccards else 0.0, 6
        ),
    }


def page_semantic_stats(
    dataset: PreparedDataset,
    labels: np.ndarray,
    members: list[np.ndarray],
    eval_supports: list[set[int]],
) -> dict:
    """检查 page reuse 是真实局部性，还是把无关 topic 粗暴碰撞到同一页。"""

    eval_pages = [
        {int(labels[doc_idx]) for doc_idx in supports}
        for supports in eval_supports
    ]
    reuse = generic_reuse_diagnostics(
        eval_pages, dataset.protocol.window_size
    )

    used_docs = set().union(*eval_supports) if eval_supports else set()
    active_pages = {int(labels[doc_idx]) for doc_idx in used_docs}
    active_page_docs = sum(len(members[page]) for page in active_pages)
    reuse["active_page_count"] = len(active_pages)
    reuse["active_page_useful_doc_density"] = round(
        len(used_docs) / max(1, active_page_docs), 6
    )

    # 受控 workload 用 regime；自然时间流用月份 round。purity 越低，越可能是
    # coarsening collision，而不是一个可预测 working set。
    counts: dict[int, Counter] = defaultdict(Counter)
    total_labeled = 0
    for query, supports in zip(dataset.stream, eval_supports):
        state = query.get("workload_regime", query.get("round"))
        if state is None:
            continue
        for doc_idx in supports:
            counts[int(labels[doc_idx])][str(state)] += 1
            total_labeled += 1
    purity_numerator = sum(max(counter.values()) for counter in counts.values())
    reuse["page_state_purity"] = (
        round(purity_numerator / total_labeled, 6) if total_labeled else None
    )
    return reuse


def routing_quality(
    routes: np.ndarray,
    width: int,
    labels: np.ndarray,
    supports: list[set[int]],
    members: list[np.ndarray],
) -> dict:
    full_hits = supported_queries = support_hits = support_total = 0
    page_matches = routed_pages = routed_docs = 0
    for row, gold_docs in zip(routes[:, :width], supports):
        if not gold_docs:
            continue
        supported_queries += 1
        routed = {int(value) for value in row}
        gold_pages = {int(labels[doc_idx]) for doc_idx in gold_docs}
        matched = gold_pages & routed
        full_hits += int(gold_pages.issubset(routed))
        support_hits += sum(int(labels[doc_idx]) in routed for doc_idx in gold_docs)
        support_total += len(gold_docs)
        page_matches += len(matched)
        routed_pages += len(routed)
        routed_docs += sum(len(members[page_id]) for page_id in routed)
    return {
        "full_support_page_recall": round(
            full_hits / max(1, supported_queries), 6
        ),
        "support_document_recall": round(
            support_hits / max(1, support_total), 6
        ),
        "routed_page_precision": round(
            page_matches / max(1, routed_pages), 6
        ),
        "routed_docs_per_query": round(
            routed_docs / max(1, supported_queries), 3
        ),
    }


def simulate_lru(
    object_members: list[np.ndarray],
    capacity_docs: int,
    initial_objects: list[int],
    eval_accesses: list[list[int]],
    warmup_supports: list[set[int]],
    eval_supports: list[set[int]],
    window_size: int,
) -> dict:
    """装入 causal initializer，再按主 runner 的 window-causal 顺序运行。"""

    cache = VariableObjectLRU(object_members, capacity_docs)
    cache.access(initial_objects)
    initial_fill = cache.used_docs / max(1, capacity_docs)

    future_positions: dict[int, list[int]] = defaultdict(list)
    for query_index, supports in enumerate(eval_supports):
        for doc_idx in supports:
            future_positions[int(doc_idx)].append(query_index)

    seen_supports = set().union(*warmup_supports) if warmup_supports else set()
    supported_queries = has_answer = cold_queries = 0
    object_loads = documents_loaded = objects_evicted = documents_evicted = 0
    oversized_objects = 0
    useful_future_loads = total_loaded_members = 0
    novel_supports = novel_supports_prefetched = 0
    occupancy = []

    window_size = max(1, int(window_size))
    for window_start in range(0, len(eval_accesses), window_size):
        window_end = min(len(eval_accesses), window_start + window_size)
        window_support_union: set[int] = set()

        # 当前窗口所有 query 都看同一个 K_t；本窗口 access 只能影响 K_{t+1}。
        for query_index in range(window_start, window_end):
            accesses = eval_accesses[query_index]
            supports = eval_supports[query_index]
            if supports:
                supported_queries += 1
                has_answer += int(supports.issubset(cache.resident_docs))
                novel = supports - seen_supports
                novel_supports += len(novel)
                novel_supports_prefetched += len(novel & cache.resident_docs)
                window_support_union.update(supports)
            missing_before = any(
                int(object_id) not in cache.objects for object_id in accesses
            )
            cold_queries += int(bool(accesses) and missing_before)

        load_position = window_end - 1
        for query_index in range(window_start, window_end):
            step = cache.access(eval_accesses[query_index])
            object_loads += step["object_loads"]
            documents_loaded += step["documents_loaded"]
            objects_evicted += step["objects_evicted"]
            documents_evicted += step["documents_evicted"]
            oversized_objects += step["oversized_objects"]

            for doc_idx in step["loaded_members"]:
                total_loaded_members += 1
                positions = future_positions.get(int(doc_idx), ())
                next_offset = bisect_right(positions, load_position)
                if (
                    next_offset < len(positions)
                    and positions[next_offset] <= load_position + window_size
                ):
                    useful_future_loads += 1
        seen_supports.update(window_support_union)
        occupancy.extend([cache.used_docs] * (window_end - window_start))

    has_answer_rate = has_answer / max(1, supported_queries)
    return {
        "has_answer_rate": round(has_answer_rate, 6),
        "normalized_amat": round(1.0 + (1.0 - has_answer_rate) * 10.0, 6),
        "cold_object_query_rate": round(
            cold_queries / max(1, len(eval_accesses)), 6
        ),
        "object_loads": int(object_loads),
        "document_write_traffic": int(documents_loaded),
        "object_evictions": int(objects_evicted),
        "document_evictions": int(documents_evicted),
        "oversized_objects": int(oversized_objects),
        "future_useful_write_rate": round(
            useful_future_loads / max(1, total_loaded_members), 6
        ),
        "novel_support_prefetch_rate": round(
            novel_supports_prefetched / max(1, novel_supports), 6
        ),
        "initial_fill_ratio": round(float(initial_fill), 6),
        "evaluation_fill_ratio_mean": round(
            float(np.mean(occupancy)) / max(1, capacity_docs)
            if occupancy else 0.0,
            6,
        ),
    }


def initial_pages_from_documents(
    labels: np.ndarray,
    members: list[np.ndarray],
    initial_docs: list[int],
    capacity_docs: int,
) -> list[int]:
    """把共享的 causal initial document set 转成整页初始缓存。

    页按覆盖 initial documents 的数量排序，随后在同一文档容量内贪心装入。未被
    initial set 命中的页仍位于排序尾部，可用于填补剩余整页容量；全程不读取 gold。
    """

    covered = Counter(int(labels[doc_idx]) for doc_idx in initial_docs)
    order = sorted(
        range(len(members)),
        key=lambda page_id: (-covered[page_id], page_id),
    )
    selected = []
    used = 0
    for page_id in order:
        size = len(members[page_id])
        if used + size <= capacity_docs:
            selected.append(int(page_id))
            used += size
    return selected


def verdict(gold_doc: dict, gold_page: dict, routed_page: dict) -> str:
    """仅用于 go/no-go 阅读的保守启发式，不作为论文统计检验。"""

    if gold_page["oversized_objects"] or routed_page["oversized_objects"]:
        return "invalid-page-size"
    upper_gain = gold_page["has_answer_rate"] - gold_doc["has_answer_rate"]
    routed_gain = routed_page["has_answer_rate"] - gold_doc["has_answer_rate"]
    write_ratio = routed_page["document_write_traffic"] / max(
        1, gold_doc["document_write_traffic"]
    )
    if routed_gain >= 0.02:
        return "cost-limited" if write_ratio > 4.0 else "promising"
    if upper_gain >= 0.02:
        return "routing-limited"
    return "no-page-locality-gain"


def audit_dataset(dataset: PreparedDataset, args: argparse.Namespace) -> dict:
    protocol = dataset.protocol
    capacity_docs = max(
        1, min(len(dataset.doc_pool), round(len(dataset.doc_pool) * args.cache_ratio))
    )
    warmup_supports, warmup_missing = support_positions(
        dataset.warmup, dataset.title_to_idx
    )
    eval_supports, eval_missing = support_positions(
        dataset.stream, dataset.title_to_idx
    )
    if warmup_missing or eval_missing:
        raise ValueError(
            f"{dataset.alias}: support titles missing from pool "
            f"(warmup={warmup_missing}, eval={eval_missing})"
        )

    initial_doc_ids = causal_prefix_init_kb(
        dataset.doc_pool,
        dataset.doc_embs,
        dataset.warmup,
        dataset.query_embs,
        capacity_docs,
        seed=int(args.seed) + 313,
    )
    doc_id_to_position = {
        doc["doc_id"]: index for index, doc in enumerate(dataset.doc_pool)
    }
    initial_docs = sorted(
        doc_id_to_position[doc_id] for doc_id in initial_doc_ids
    )

    doc_objects = [np.asarray([idx], dtype=np.int32) for idx in range(len(dataset.doc_pool))]
    gold_doc_lru = simulate_lru(
        doc_objects,
        capacity_docs,
        initial_docs,
        [sorted(values) for values in eval_supports],
        warmup_supports,
        eval_supports,
        window_size=protocol.window_size,
    )
    document_reuse = support_reuse_diagnostics(
        dataset.stream, protocol.window_size
    ).as_dict()

    result = {
        "protocol": {
            "dataset": protocol.dataset,
            "workload": protocol.workload,
            "pool_size": len(dataset.doc_pool),
            "cache_capacity_docs": int(capacity_docs),
            "cache_pool_ratio": round(capacity_docs / len(dataset.doc_pool), 6),
            "warmup_queries": len(dataset.warmup),
            "evaluation_queries": len(dataset.stream),
            "n_windows": protocol.n_windows,
            "window_size": protocol.window_size,
            "factorized_min_support_frequency": protocol.min_support_frequency,
            "factorized_family_mode": protocol.family_mode,
            "sf_hit_threshold_reference": float(SF_HIT_THRESH),
            "constructor": dataset.construction,
            "source_split": dataset.source_split,
            "temporal_sampling": dataset.temporal_sampling,
        },
        "document_reuse": document_reuse,
        "gold_doc_lru": gold_doc_lru,
        "pages": {},
    }

    for target_page_size in args.page_sizes:
        labels, centroids, packing = build_balanced_semantic_pages(
            dataset,
            target_page_size,
            seed=args.seed,
            force=args.force_partitions,
        )
        members = page_members(labels, len(centroids))
        max_width = min(max(args.route_widths), len(centroids))
        eval_routes = query_routes(
            dataset.stream, dataset.query_embs, centroids, max_width
        )
        eval_gold_pages = [
            sorted({int(labels[doc_idx]) for doc_idx in supports})
            for supports in eval_supports
        ]
        initial_pages = initial_pages_from_documents(
            labels, members, initial_docs, capacity_docs
        )
        gold_page_lru = simulate_lru(
            members,
            capacity_docs,
            initial_pages,
            eval_gold_pages,
            warmup_supports,
            eval_supports,
            window_size=protocol.window_size,
        )
        page_result = {
            "partition": packing,
            "page_reuse": page_semantic_stats(
                dataset, labels, members, eval_supports
            ),
            "gold_page_lru": gold_page_lru,
            "route_widths": {},
        }

        for width in args.route_widths:
            effective_width = min(int(width), len(centroids))
            routed_page_lru = simulate_lru(
                members,
                capacity_docs,
                initial_pages,
                [list(row[:effective_width]) for row in eval_routes],
                warmup_supports,
                eval_supports,
                window_size=protocol.window_size,
            )
            comparison = {
                "has_answer_gain_vs_gold_doc_lru_pp": round(
                    100.0 * (
                        routed_page_lru["has_answer_rate"]
                        - gold_doc_lru["has_answer_rate"]
                    ),
                    3,
                ),
                "reactive_gold_page_gain_pp": round(
                    100.0 * (
                        gold_page_lru["has_answer_rate"]
                        - gold_doc_lru["has_answer_rate"]
                    ),
                    3,
                ),
                "gap_to_reactive_gold_page_pp": round(
                    100.0 * (
                        gold_page_lru["has_answer_rate"]
                        - routed_page_lru["has_answer_rate"]
                    ),
                    3,
                ),
                "document_write_amplification": round(
                    routed_page_lru["document_write_traffic"]
                    / max(1, gold_doc_lru["document_write_traffic"]),
                    3,
                ),
                "object_io_ratio": round(
                    routed_page_lru["object_loads"]
                    / max(1, gold_doc_lru["object_loads"]),
                    3,
                ),
                "verdict": verdict(
                    gold_doc_lru, gold_page_lru, routed_page_lru
                ),
            }
            page_result["route_widths"][str(width)] = {
                "routing": routing_quality(
                    eval_routes,
                    effective_width,
                    labels,
                    eval_supports,
                    members,
                ),
                "routed_page_lru": routed_page_lru,
                "comparison": comparison,
            }
        result["pages"][str(target_page_size)] = page_result
    return result


def percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def markdown_report(payload: dict, command: str) -> str:
    lines = [
        "# Semantic Page Cache Feasibility Audit",
        "",
        f"Audit version: `{payload['audit_version']}`",
        "",
        "This is a mechanism audit, not a main-paper result. `GoldDoc-LRU` and "
        "`GoldPage-LRU` are reactive gold-trace diagnostics, not prefetch upper "
        "bounds; only `RoutedPage-LRU` routes "
        "with the current query embedding. Capacity is equalized in documents, "
        "and page loads are also charged by document traffic. Writes are not "
        "capped in this feasibility run, so routed quality is optimistic while "
        "all resulting traffic remains visible.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Results",
        "",
        "| Dataset | Page | L | Route full support | Doc reuse | Page reuse | "
        "GoldDoc HA | GoldPage HA | RoutedPage HA | Gain | Doc write amp. | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    verdict_counts = Counter()
    best_by_dataset = {}
    for dataset_name, result in payload["datasets"].items():
        doc_reuse = result["document_reuse"]["repeated_support_rate"]
        gold_doc = result["gold_doc_lru"]
        for page_size, page_result in result["pages"].items():
            page_reuse = page_result["page_reuse"]["repeated_occurrence_rate"]
            gold_page = page_result["gold_page_lru"]
            for width, route_result in page_result["route_widths"].items():
                routed = route_result["routed_page_lru"]
                comparison = route_result["comparison"]
                verdict_counts[comparison["verdict"]] += 1
                candidate = {
                    "page_size": page_size,
                    "width": width,
                    "routed": routed,
                    "comparison": comparison,
                    "gold_doc": gold_doc,
                    "gold_page": gold_page,
                    "page_reuse": page_result["page_reuse"],
                }
                current = best_by_dataset.get(dataset_name)
                if current is None or (
                    routed["has_answer_rate"],
                    -routed["document_write_traffic"],
                ) > (
                    current["routed"]["has_answer_rate"],
                    -current["routed"]["document_write_traffic"],
                ):
                    best_by_dataset[dataset_name] = candidate
                lines.append(
                    f"| {dataset_name} | {page_size} | {width} | "
                    f"{percent(route_result['routing']['full_support_page_recall'])} | "
                    f"{percent(doc_reuse)} | {percent(page_reuse)} | "
                    f"{percent(gold_doc['has_answer_rate'])} | "
                    f"{percent(gold_page['has_answer_rate'])} | "
                    f"{percent(routed['has_answer_rate'])} | "
                    f"{comparison['has_answer_gain_vs_gold_doc_lru_pp']:+.1f} pp | "
                    f"{comparison['document_write_amplification']:.2f}x | "
                    f"`{comparison['verdict']}` |"
                )

    lines.extend([
        "",
        "## Audit Conclusion",
        "",
    ])
    if verdict_counts.get("promising", 0) == 0:
        lines.append(
            "No tested page size/route width passes the joint quality-cost "
            "criterion. Whole-page caching should therefore not replace the "
            "document-level DRIP policy in its current form."
        )
        lines.append("")
    for dataset_name, best in best_by_dataset.items():
        routed = best["routed"]
        comparison = best["comparison"]
        reuse = best["page_reuse"]
        lines.append(
            f"- **{dataset_name}:** best routed setting is page={best['page_size']}, "
            f"L={best['width']}; Has-Answer {percent(routed['has_answer_rate'])} "
            f"vs. {percent(best['gold_doc']['has_answer_rate'])} for GoldDoc-LRU, "
            f"with {comparison['document_write_amplification']:.2f}x document "
            f"traffic. Only {percent(routed['future_useful_write_rate'])} of "
            f"loaded documents are reused within one window; active-page useful "
            f"density is {percent(reuse['active_page_useful_doc_density'])}."
        )
    lines.extend([
        "",
        "Recommended use: retain semantic partitions as cold-index routing or "
        "candidate-pruning metadata, then admit selected documents (or very small "
        "evidence bundles) with the existing replacement controller. Do not make "
        "an entire semantic page the cache replacement unit without a learned "
        "within-page selector.",
        "",
        "## Reading The Verdict",
        "",
        "- `no-page-locality-gain`: routed pages do not gain 2 pp and reactive gold pages expose no latent page benefit.",
        "- `invalid-page-size`: at least one page is larger than the hot-tier document budget.",
        "- `routing-limited`: pages contain useful future evidence, but query-to-page routing cannot recover it.",
        "- `cost-limited`: routed pages improve answerability, but document traffic exceeds 4x.",
        "- `promising`: routed pages gain at least 2 pp with no more than 4x document traffic.",
        "",
        "The 2 pp and 4x thresholds are conservative audit heuristics, not statistical claims. "
        "A paper experiment must next compare against the repository's production LRU/ARC implementations, "
        "report multiple seeds, and assign measured byte/latency costs to page I/O.",
        "",
        "Verdict counts: " + ", ".join(
            f"`{name}`={count}" for name, count in sorted(verdict_counts.items())
        ),
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(PROTOCOLS),
        default=list(PROTOCOLS),
    )
    parser.add_argument("--page-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--route-widths", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--cache-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=int(DATA_SEED))
    parser.add_argument("--n-source", type=int, default=None)
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--warmup-windows", type=int, default=None)
    parser.add_argument(
        "--source-pool",
        choices=sorted(SOURCE_POOL_ROLES),
        default=SOURCE_POOL_ALL,
        help=(
            "optional stable support-family source split; controlled workloads "
            "only"
        ),
    )
    parser.add_argument("--source-pool-seed", type=int, default=1729)
    parser.add_argument(
        "--source-pool-calibration-fraction", type=float, default=0.5
    )
    parser.add_argument("--force-partitions", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "docs" / "experiments" /
        "SEMANTIC_PAGE_AUDIT_2026-07-15.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "docs" / "experiments" /
        "SEMANTIC_PAGE_AUDIT_2026-07-15.md",
    )
    args = parser.parse_args()
    if not 0.0 < args.cache_ratio <= 1.0:
        parser.error("--cache-ratio must be in (0, 1]")
    if any(value <= 0 for value in args.page_sizes + args.route_widths):
        parser.error("page sizes and route widths must be positive")
    if not 0.0 < args.source_pool_calibration_fraction < 1.0:
        parser.error("--source-pool-calibration-fraction must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    command = " ".join([sys.executable, *sys.argv])
    payload = {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "assumptions": {
            "causal_order": "evaluate current query before cache update",
            "initialization": "shared causal query-conditioned warm-up set",
            "page_partition_visibility": "cold corpus documents only",
            "online_route_visibility": "current query embedding only",
            "capacity_unit": "documents",
            "write_cost_units": ["page objects", "documents transferred"],
            "write_budget": (
                "unconstrained quality diagnostic; every load is charged, and "
                "a production write cap can only reduce routed-page quality"
            ),
            "gold_trace_role": "reactive diagnostic, not deployable policy or prefetch upper bound",
        },
        "datasets": {},
    }
    for alias in args.datasets:
        prepared = prepare_dataset(alias, args)
        payload["datasets"][alias] = audit_dataset(prepared, args)
        print(f"[{alias}] audit complete", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        markdown_report(payload, command), encoding="utf-8"
    )
    print(f"JSON: {args.output}")
    print(f"REPORT: {args.report}")


if __name__ == "__main__":
    main()
