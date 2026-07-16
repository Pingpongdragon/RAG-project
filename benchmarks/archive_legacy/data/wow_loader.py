"""
Wizard of Wikipedia (WoW) 数据加载与处理。

WoW 是一个开放域对话数据集，每段对话标注了自然 topic 标签。
本模块利用这些天然标签来构造 topic 漂移查询流，避免硬编码分类。

包含两部分:
  1. Topic 概率调度器 — TopicSchedule 及其子类 (原 topic_schedules.py)
  2. WoW 数据加载与流构建

数据来源: datasets/wizard_of_wikipedia/{split}.json
"""

import hashlib
import json
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarks.archive_legacy.data.structures import (
    PoolDocument, QueryItem,
    FOOD_CHAIN, HEALTH_CHAIN, DIVERSE_TOPICS, CYCLE_TOPICS, BIG_TOPICS,
)

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[3]


# ============================================================
#  Topic 概率调度器
# ============================================================

class TopicSchedule(ABC):
    """P(topic | t) 的抽象基类，t in [0, 1]。"""

    @abstractmethod
    def get_probs(self, t: float) -> Dict[str, float]:
        """返回时刻 t 的归一化 topic 概率分布。"""
        ...

    def get_probs_array(self, t: float, topics: List[str]) -> np.ndarray:
        """返回与 topics 列表对齐的概率数组。"""
        p = self.get_probs(t)
        a = np.array([p.get(tp, 0.0) for tp in topics])
        s = a.sum()
        return a / s if s > 0 else np.ones(len(topics)) / len(topics)


class GaussianDriftSchedule(TopicSchedule):
    """
    各 topic 拥有一个高斯激活曲线，中心等间距分布在 [0, 1] 上。

    w_i(t) = exp(-0.5 * ((t - c_i) / sigma)^2)
    sigma 控制重叠程度 - 越大，相邻 topic 共存越多。
    """

    def __init__(self, topics: List[str], sigma: float = 0.18):
        self.topics = topics
        n = len(topics)
        self.centres = [i / max(n - 1, 1) for i in range(n)]
        self.sigma = sigma

    def get_probs(self, t: float) -> Dict[str, float]:
        raw = {
            tp: math.exp(-0.5 * ((t - c) / self.sigma) ** 2)
            for tp, c in zip(self.topics, self.centres)
        }
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


class SigmoidShiftSchedule(TopicSchedule):
    """
    Topics 通过陡峭 sigmoid 曲线切换 - 模拟突然的兴趣转移。

    每对相邻 topic 之间有一个 sigmoid 过渡点，steepness 控制切换的陡峭程度。
    """

    def __init__(self, topics: List[str], steepness: float = 30.0):
        self.topics = topics
        n = len(topics)
        self.transitions = [(i + 0.5) / n for i in range(n - 1)]
        self.steepness = steepness

    def get_probs(self, t: float) -> Dict[str, float]:
        n = len(self.topics)
        w = np.zeros(n)
        for i in range(n):
            left = 1.0 if i == 0 else 1.0 / (1.0 + math.exp(
                -self.steepness * (t - self.transitions[i - 1])))
            right = 1.0 if i == n - 1 else 1.0 / (1.0 + math.exp(
                self.steepness * (t - self.transitions[i])))
            w[i] = left * right
        total = w.sum() or 1.0
        return {self.topics[i]: float(w[i] / total) for i in range(n)}


class CyclicSchedule(TopicSchedule):
    """
    Topics 周期性激活: A -> B -> C -> A -> B -> C -> ...

    每个 topic 拥有等间隔高斯峰，形成平滑的周期循环。
    """

    def __init__(self, topics: List[str], n_cycles: int = 2, sigma: float = 0.08):
        self.topics = topics
        self.n_cycles = n_cycles
        self.sigma = sigma
        n = len(topics)
        cycle_len = 1.0 / n_cycles
        self.peaks: Dict[str, List[float]] = {}
        for i, tp in enumerate(topics):
            offset = (i / n) * cycle_len
            self.peaks[tp] = [c * cycle_len + offset for c in range(n_cycles)]

    def get_probs(self, t: float) -> Dict[str, float]:
        raw = {}
        for tp, peaks in self.peaks.items():
            raw[tp] = sum(
                math.exp(-0.5 * ((t - p) / self.sigma) ** 2) for p in peaks
            )
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}


# ============================================================
#  数据加载
# ============================================================

def load_wow(split: str = "validation") -> List[Dict]:
    """加载 WoW 原始数据。"""
    path = BASE_DIR / f"datasets/wizard_of_wikipedia/{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#  Topic 数据提取
# ============================================================

def extract_topic_data(
    wow_data: List[Dict],
    topics: List[str],
) -> Tuple[Dict[str, List[QueryItem]], Dict[str, List[PoolDocument]]]:
    """
    从 WoW 数据中提取指定 topic 的查询和文档。

    遍历所有对话，如果对话的 primary topic 在目标列表中，
    则将其每个 turn 转为 QueryItem，将知识段落转为 PoolDocument。

    Returns:
        (topic_queries, topic_docs): 按 topic 分组的查询和文档字典
    """
    topic_set = set(topics)
    topic_queries: Dict[str, List[QueryItem]] = defaultdict(list)
    topic_docs: Dict[str, List[PoolDocument]] = defaultdict(list)
    seen_docs: set = set()

    for item in wow_data:
        conv_topics = item.get("topics", [])
        if not conv_topics:
            continue
        primary = conv_topics[0]
        if primary not in topic_set:
            continue

        conv_id = hashlib.md5(str(item["post"]).encode()).hexdigest()[:8]
        n_turns = len(item.get("post", []))

        for turn in range(n_turns):
            post = item["post"][turn]
            response = item["response"][turn]
            knowledge = item["knowledge"][turn] if turn < len(item["knowledge"]) else []
            label = item["labels"][turn] if turn < len(item["labels"]) else -1

            gold_ids: List[str] = []
            for k_idx, passage in enumerate(knowledge):
                if not passage or passage.startswith("no_passages_used"):
                    continue
                parts = passage.split(" __knowledge__ ", 1)
                doc_title, doc_text = (parts if len(parts) == 2
                                       else (primary, passage))
                doc_hash = hashlib.md5(doc_text.encode()).hexdigest()[:12]
                doc_id = f"wow_{doc_hash}"

                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    topic_docs[primary].append(PoolDocument(
                        doc_id=doc_id, text=doc_text,
                        topic=primary, title=doc_title,
                    ))
                if k_idx == label:
                    gold_ids.append(doc_id)

            topic_queries[primary].append(QueryItem(
                query_id=f"wow_{conv_id}_t{turn}",
                question=post, answer=response,
                topic=primary, gold_doc_ids=gold_ids,
                metadata={"conv_id": conv_id, "turn": turn},
            ))

    for tp in topics:
        logger.info(f"  {tp}: {len(topic_queries[tp])} queries, "
                     f"{len(topic_docs[tp])} pool docs")
    return dict(topic_queries), dict(topic_docs)


# ============================================================
#  Topic 选择
# ============================================================

def select_topics(
    wow_data: List[Dict],
    n_topics: int,
    min_conversations: int = 15,
    seed: int = 42,
    preferred: Optional[List[str]] = None,
) -> List[str]:
    """选择数据量充足的 topic。优先使用 preferred 列表，不足时自动补充。"""
    conv_counts: Counter = Counter()
    for item in wow_data:
        ts = item.get("topics", [])
        if ts:
            conv_counts[ts[0]] += 1

    if preferred:
        valid = [t for t in preferred
                 if conv_counts.get(t, 0) >= min_conversations]
        if len(valid) >= n_topics:
            return valid[:n_topics]
        logger.warning(f"Only {len(valid)}/{len(preferred)} preferred topics available")

    eligible = sorted(
        ((t, c) for t, c in conv_counts.items()
         if c >= min_conversations and t not in ("7", "")),
        key=lambda x: x[1], reverse=True,
    )
    rng = random.Random(seed)
    candidates = [t for t, _ in eligible[:n_topics * 3]]
    rng.shuffle(candidates)
    return candidates[:n_topics]


# ============================================================
#  查询流构建
# ============================================================

def build_stream(
    schedule: TopicSchedule,
    topic_queries: Dict[str, List[QueryItem]],
    total_queries: int,
    seed: int = 42,
) -> Tuple[List[QueryItem], List[Dict[str, float]]]:
    """
    按时变 topic 调度采样查询流。

    在流位置 i:
      1. 计算 P(topic | t)，t = i / (total - 1)
      2. 从该分布采样一个 topic
      3. 从该 topic 的查询池中取下一条（轮询式）

    Returns:
        (stream, schedule_log): 查询序列和每步的概率分布日志
    """
    rng = random.Random(seed)
    topics = list(topic_queries.keys())

    pools = {}
    for tp, qs in topic_queries.items():
        q_copy = list(qs)
        rng.shuffle(q_copy)
        pools[tp] = q_copy

    ptrs = {t: 0 for t in topics}
    stream: List[QueryItem] = []
    log: List[Dict[str, float]] = []

    for i in range(total_queries):
        t = i / max(total_queries - 1, 1)
        probs = schedule.get_probs(t)
        log.append(probs)

        prob_vec = [probs.get(tp, 0.0) for tp in topics]
        total_p = sum(prob_vec) or 1.0
        prob_vec = [p / total_p for p in prob_vec]

        chosen = rng.choices(topics, weights=prob_vec, k=1)[0]

        pool = pools[chosen]
        if not pool:
            for alt in topics:
                if pools[alt]:
                    chosen = alt
                    pool = pools[chosen]
                    break

        idx = ptrs[chosen] % len(pool)
        stream.append(pool[idx])
        ptrs[chosen] = idx + 1

    return stream, log


# ============================================================
#  文档池合并
# ============================================================

def merge_pool(
    topic_docs: Dict[str, List[PoolDocument]],
    gold_doc_ids: Optional[set] = None,
    max_total: int = 5000,
    seed: int = 42,
) -> List[PoolDocument]:
    """去重合并文档池，保留所有 gold 文档后随机采样到预算。"""
    seen: set = set()
    merged: List[PoolDocument] = []
    for docs in topic_docs.values():
        for d in docs:
            if d.doc_id not in seen:
                seen.add(d.doc_id)
                merged.append(d)

    if len(merged) > max_total:
        rng = random.Random(seed)
        gids = gold_doc_ids or set()
        gold = [d for d in merged if d.doc_id in gids]
        other = [d for d in merged if d.doc_id not in gids]
        rng.shuffle(other)
        budget = max(max_total - len(gold), 0)
        merged = gold + other[:budget]
        logger.info(f"Pool subsampled: {len(merged)} docs "
                     f"({len(gold)} gold kept, {min(budget, len(other))} sampled)")

    rng_shuffle = random.Random(seed)
    rng_shuffle.shuffle(merged)
    return merged
