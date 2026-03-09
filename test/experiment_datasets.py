"""
Natural Topic-Drift Experiment Datasets (v2)

Core design: **NO hard domain classification**.
Uses WoW's natural ``topics`` field to create smooth query streams where topic
proportions gradually shift over time via continuous probability schedules.

Three experiments:
  Exp 1  Gradual Drift:  Gaussian-centered topic activations, smooth transition
  Exp 2  Sudden Shift:   Steep sigmoid transitions between diverse topics
  Exp 3  Cyclic Return:  Periodic topic re-activation, tests re-adaptation

Each query is sampled from P(topic | t), where t is normalised stream position.
This creates natural, soft boundaries — never a hard cut between "domains".
"""

import json
import hashlib
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Data Structures
# ============================================================

@dataclass
class QueryItem:
    """A single query in the stream."""
    query_id: str
    question: str
    answer: str
    topic: str                                    # natural WoW topic label
    gold_doc_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolDocument:
    """A document available for KB curation."""
    doc_id: str
    text: str
    topic: str                                    # natural WoW topic label
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentDataset:
    """Complete dataset for one experiment."""
    name: str
    description: str
    query_stream: List[QueryItem]
    document_pool: List[PoolDocument]
    topics: List[str]                             # all topics involved
    schedule_type: str                            # gaussian / sigmoid / cyclic
    topic_schedule_log: List[Dict[str, float]] = field(default_factory=list)


# ============================================================
# Topic Probability Schedules
# ============================================================

class TopicSchedule(ABC):
    """P(topic | t)  for t in [0, 1]."""

    @abstractmethod
    def get_probs(self, t: float) -> Dict[str, float]:
        """Normalised probability distribution over topics at time *t*."""
        ...

    def get_probs_array(self, t: float, topics: List[str]) -> np.ndarray:
        p = self.get_probs(t)
        a = np.array([p.get(tp, 0.0) for tp in topics])
        s = a.sum()
        return a / s if s > 0 else np.ones(len(topics)) / len(topics)


class GaussianDriftSchedule(TopicSchedule):
    r"""
    Each topic has a Gaussian activation curve centred at evenly-spaced times.

    .. math::
        w_i(t) = \exp\!\Big(-\tfrac{1}{2}\big(\tfrac{t - c_i}{\sigma}\big)^2\Big)

    sigma controls overlap — larger sigma => more co-occurrence of adjacent topics.
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
    r"""
    Topics transition via steep sigmoid curves — simulates abrupt interest
    shift with narrow transition zones.

    ``steepness`` controls gradient: higher ⇒ more abrupt.
    """

    def __init__(self, topics: List[str], steepness: float = 30.0):
        self.topics = topics
        n = len(topics)
        # Transition mid-points between consecutive topics
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
    r"""
    Topics activate periodically:  A → B → C → A → B → C → …

    Each topic has Gaussian peaks at regular intervals, creating smooth cycles.
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
# WoW Data Loader  (uses natural ``topics`` field, no clustering)
# ============================================================

def _load_wow(split: str = "validation") -> List[Dict]:
    path = BASE_DIR / f"datasets/wizard_of_wikipedia/{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_topic_data(
    wow_data: List[Dict],
    topics: List[str],
) -> Tuple[Dict[str, List[QueryItem]], Dict[str, List[PoolDocument]]]:
    """
    Extract queries and pool documents for requested topics.
    Uses WoW's natural ``topics`` field — zero clustering.
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


def _select_topics(
    wow_data: List[Dict],
    n_topics: int,
    min_conversations: int = 15,
    seed: int = 42,
    preferred: Optional[List[str]] = None,
) -> List[str]:
    """Pick topics with enough data. Prefer ``preferred`` list when available."""
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

    # Fall back to largest topics (filter garbage like '7')
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
# Stream Builder  (probability-weighted sampling → soft boundaries)
# ============================================================

def _build_stream(
    schedule: TopicSchedule,
    topic_queries: Dict[str, List[QueryItem]],
    total_queries: int,
    seed: int = 42,
) -> Tuple[List[QueryItem], List[Dict[str, float]]]:
    """
    Sample queries according to a time-varying topic schedule.

    At position *i* in the stream:
      1. Compute P(topic | t)  where  t = i / (total - 1)
      2. Sample a topic from this distribution
      3. Draw the next unused query from that topic (round-robin)

    Returns (ordered stream, per-position probability log).
    """
    rng = random.Random(seed)
    topics = list(topic_queries.keys())

    # Pre-shuffle each topic's queries
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
        if not pool:          # fallback
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
# Pre-defined topic groups (semantically motivated)
# ============================================================

FOOD_CHAIN     = ["Pasta", "Pizza", "Baking", "Wine tasting"]
HEALTH_CHAIN   = ["Obesity", "Physical fitness", "Chronic fatigue syndrome"]
DIVERSE_TOPICS = ["Red", "Manta ray", "Superman", "Niagara Falls", "Ferrari"]
CYCLE_TOPICS   = ["Pasta", "Ferrari", "Hair coloring"]
BIG_TOPICS     = ["Pasta", "Brown hair", "Jazz", "Ferrari", "Obesity"]


# ============================================================
# Experiment Builders
# ============================================================

def build_gradual_drift(
    total_queries: int = 300,
    topics: Optional[List[str]] = None,
    sigma: float = 0.18,
    seed: int = 42,
) -> ExperimentDataset:
    """
    Exp 1 — Gradual Drift.
    Gaussian activations  =>  topics smoothly overlap and transition.
    """
    logger.info(f"Building Gradual Drift: {total_queries} queries, sigma={sigma}")
    wow = _load_wow("validation")
    if topics is None:
        topics = _select_topics(wow, 5, preferred=BIG_TOPICS, seed=seed)
    logger.info(f"Topics: {topics}")

    tq, td = _extract_topic_data(wow, topics)
    schedule = GaussianDriftSchedule(topics, sigma=sigma)
    stream, slog = _build_stream(schedule, tq, total_queries, seed)

    gold_ids = {gid for q in stream for gid in q.gold_doc_ids}
    pool = _merge_pool(td, gold_doc_ids=gold_ids, seed=seed)
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
) -> ExperimentDataset:
    """
    Exp 2 — Sudden Shift.
    Steep sigmoid transitions between semantically distant topics.
    """
    logger.info(f"Building Sudden Shift: {total_queries} queries, k={steepness}")
    wow = _load_wow("validation")
    if topics is None:
        topics = _select_topics(wow, 5, preferred=DIVERSE_TOPICS, seed=seed)
    logger.info(f"Topics: {topics}")

    tq, td = _extract_topic_data(wow, topics)
    schedule = SigmoidShiftSchedule(topics, steepness=steepness)
    stream, slog = _build_stream(schedule, tq, total_queries, seed)

    gold_ids = {gid for q in stream for gid in q.gold_doc_ids}
    pool = _merge_pool(td, gold_doc_ids=gold_ids, seed=seed)
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
) -> ExperimentDataset:
    """
    Exp 3 — Cyclic Return.
    Topics re-appear periodically  =>  tests re-adaptation speed.
    """
    logger.info(f"Building Cyclic Return: {total_queries} queries, "
                f"cycles={n_cycles}, sigma={sigma}")
    wow = _load_wow("validation")
    if topics is None:
        topics = _select_topics(wow, 3, preferred=CYCLE_TOPICS, seed=seed)
    logger.info(f"Topics: {topics}")

    tq, td = _extract_topic_data(wow, topics)
    schedule = CyclicSchedule(topics, n_cycles=n_cycles, sigma=sigma)
    stream, slog = _build_stream(schedule, tq, total_queries, seed)

    pool = _merge_pool(td)
    logger.info(f"Cyclic Return: {len(stream)} queries, {len(pool)} pool docs, "
                f"pattern={'->'.join(topics)} x{n_cycles}")
    return ExperimentDataset(
        name="cyclic_return",
        description=f"{topics} x{n_cycles} cycles",
        query_stream=stream, document_pool=pool,
        topics=topics, schedule_type="cyclic",
        topic_schedule_log=slog,
    )




# ============================================================
# HotpotQA: Entity-graph Greedy Walk  (no hard classification)
# ============================================================

def _load_hotpotqa(split: str = "validation_distractor") -> List[Dict]:
    path = BASE_DIR / f"datasets/hotpotqa/{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_entity_graph(items: List[Dict]) -> Dict[int, List[int]]:
    """
    Build adjacency list: items share an edge if they share >=1 supporting fact title.
    Returns {item_index: [neighbour_indices]}.
    """
    from collections import defaultdict

    title_to_items: Dict[str, List[int]] = defaultdict(list)
    for i, item in enumerate(items):
        sf_titles = set(item.get("supporting_facts", {}).get("title", []))
        for t in sf_titles:
            title_to_items[t].append(i)

    adj: Dict[int, set] = defaultdict(set)
    for _title, idx_list in title_to_items.items():
        for a in idx_list:
            for b in idx_list:
                if a != b:
                    adj[a].add(b)

    return {k: list(v) for k, v in adj.items()}


def _greedy_walk(
    adj: Dict[int, List[int]],
    n_items: int,
    total_steps: int,
    seed: int = 42,
) -> List[int]:
    """
    Greedy walk on entity graph: always prefer unvisited neighbours.
    When stuck (no unvisited neighbours), jump to nearest unvisited node
    that shares at least one title with any visited node.

    Creates a natural drift: closely related questions cluster together,
    gradually transitioning to distant topics as the walk progresses.
    """
    rng = random.Random(seed)
    visited_order: List[int] = []
    visited_set: set = set()

    # Start from the node with the most neighbours (densest area)
    start = max(range(n_items), key=lambda i: len(adj.get(i, [])))
    current = start

    for _ in range(total_steps):
        if current not in visited_set:
            visited_order.append(current)
            visited_set.add(current)

        if len(visited_order) >= total_steps:
            break

        # Try unvisited neighbours first
        neighbours = adj.get(current, [])
        unvisited_nbrs = [n for n in neighbours if n not in visited_set]

        if unvisited_nbrs:
            current = rng.choice(unvisited_nbrs)
        else:
            # Jump to any unvisited node that has edges
            candidates = [i for i in range(n_items)
                          if i not in visited_set and adj.get(i)]
            if candidates:
                current = rng.choice(candidates)
            else:
                # All connected nodes visited; pick any unvisited
                remaining = [i for i in range(n_items) if i not in visited_set]
                if remaining:
                    current = rng.choice(remaining)
                else:
                    break

    return visited_order


def _hotpotqa_item_to_query_and_docs(
    item: Dict,
    idx: int,
) -> Tuple[QueryItem, List[PoolDocument]]:
    """Convert a HotpotQA item to QueryItem + PoolDocuments. Topic = supporting fact titles."""
    ctx = item.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    sf_titles = set(item.get("supporting_facts", {}).get("title", []))

    # "Topic" is the concatenation of supporting fact titles (natural, no clustering)
    topic_label = " & ".join(sorted(sf_titles)) if sf_titles else "unknown"

    docs = []
    gold_ids = []
    for i in range(len(titles)):
        title = titles[i]
        sents = sentences[i] if i < len(sentences) else []
        text = " ".join(sents).strip()
        if not text:
            continue

        doc_id = f"hp_{hashlib.md5((title + text[:100]).encode()).hexdigest()[:10]}"
        docs.append(PoolDocument(
            doc_id=doc_id, text=text,
            topic=title,  # use Wikipedia article title as topic
            title=title,
        ))
        if title in sf_titles:
            gold_ids.append(doc_id)

    query = QueryItem(
        query_id=f"hp_{item['id']}",
        question=item["question"],
        answer=item["answer"],
        topic=topic_label,
        gold_doc_ids=gold_ids,
        metadata={"type": item.get("type", ""), "level": item.get("level", ""),
                  "sf_titles": list(sf_titles)},
    )
    return query, docs


def build_hotpotqa_entity_walk(
    total_queries: int = 400,
    seed: int = 42,
    max_pool: int = 5000,
) -> ExperimentDataset:
    """
    HotpotQA: entity-graph greedy walk.

    Questions that share supporting-fact titles are linked in a graph.
    A greedy walk creates natural topic drift without any hard classification.
    """
    logger.info(f"Building HotpotQA Entity Walk: {total_queries} queries")
    hp_data = _load_hotpotqa()
    logger.info(f"Loaded {len(hp_data)} HotpotQA items")

    adj = _build_entity_graph(hp_data)
    walk = _greedy_walk(adj, len(hp_data), total_queries, seed=seed)
    logger.info(f"Greedy walk: {len(walk)} steps")

    # Build stream
    stream: List[QueryItem] = []
    all_docs: List[PoolDocument] = []
    seen_docs: set = set()
    gold_ids_all: set = set()

    for idx in walk:
        item = hp_data[idx]
        q, docs = _hotpotqa_item_to_query_and_docs(item, idx)
        stream.append(q)
        gold_ids_all.update(q.gold_doc_ids)
        for d in docs:
            if d.doc_id not in seen_docs:
                seen_docs.add(d.doc_id)
                all_docs.append(d)

    # Subsample pool
    if len(all_docs) > max_pool:
        rng = random.Random(seed)
        gold = [d for d in all_docs if d.doc_id in gold_ids_all]
        other = [d for d in all_docs if d.doc_id not in gold_ids_all]
        rng.shuffle(other)
        budget = max(max_pool - len(gold), 0)
        all_docs = gold + other[:budget]
        logger.info(f"Pool subsampled: {len(all_docs)} docs "
                     f"({len(gold)} gold kept)")

    # Collect unique article titles as "topics"
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


# ============================================================
# Helpers
# ============================================================

def _merge_pool(
    topic_docs: Dict[str, List[PoolDocument]],
    gold_doc_ids: Optional[set] = None,
    max_total: int = 5000,
    seed: int = 42,
) -> List[PoolDocument]:
    """Deduplicate, merge, and subsample (keeping all gold docs)."""
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
    return merged


def build_all_datasets(
    seed: int = 42,
    total_queries: int = 300,
) -> Dict[str, ExperimentDataset]:
    return {
        "gradual_drift": build_gradual_drift(total_queries=total_queries, seed=seed),
        "sudden_shift":  build_sudden_shift(total_queries=total_queries, seed=seed),
        "cyclic_return":  build_cyclic_return(total_queries=total_queries, seed=seed),
        "hotpotqa_walk":  build_hotpotqa_entity_walk(total_queries=total_queries, seed=seed),
    }
