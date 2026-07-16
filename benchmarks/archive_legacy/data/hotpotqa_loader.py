"""
HotpotQA 数据加载与实体图游走。

HotpotQA 是一个多跳问答数据集，每个问题需要从多篇 Wikipedia 文章中
找到 supporting facts 才能回答。

本模块利用 supporting facts 的共享关系构建实体邻接图，
然后通过贪心游走产生自然的 topic 漂移序列——
相关问题聚在一起，逐渐过渡到远处的话题。

数据来源: datasets/hotpotqa/{split}.json
"""

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from benchmarks.archive_legacy.data.structures import PoolDocument, QueryItem

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[3]


def load_hotpotqa(split: str = "validation_distractor") -> List[Dict]:
    """加载 HotpotQA 原始数据。"""
    path = BASE_DIR / f"datasets/hotpotqa/{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_entity_graph(items: List[Dict]) -> Dict[int, List[int]]:
    """
    构建实体邻接图: 两个 item 共享至少一个 supporting fact title 则连边。

    Returns:
        {item_index: [neighbour_indices]}
    """
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


def greedy_walk(
    adj: Dict[int, List[int]],
    n_items: int,
    total_steps: int,
    seed: int = 42,
) -> List[int]:
    """
    实体图贪心游走: 优先访问未访问的邻居。

    当所有邻居都已访问时，随机跳转到仍有边的未访问节点。
    这产生了自然的 topic 漂移——相近问题聚在一起，逐渐过渡到新话题。

    Args:
        adj:         邻接表
        n_items:     总节点数
        total_steps: 游走步数
        seed:        随机种子

    Returns:
        按访问顺序排列的节点索引序列
    """
    rng = random.Random(seed)
    visited_order: List[int] = []
    visited_set: set = set()

    start = max(range(n_items), key=lambda i: len(adj.get(i, [])))
    current = start

    for _ in range(total_steps):
        if current not in visited_set:
            visited_order.append(current)
            visited_set.add(current)

        if len(visited_order) >= total_steps:
            break

        neighbours = adj.get(current, [])
        unvisited_nbrs = [n for n in neighbours if n not in visited_set]

        if unvisited_nbrs:
            current = rng.choice(unvisited_nbrs)
        else:
            candidates = [i for i in range(n_items)
                          if i not in visited_set and adj.get(i)]
            if candidates:
                current = rng.choice(candidates)
            else:
                remaining = [i for i in range(n_items) if i not in visited_set]
                if remaining:
                    current = rng.choice(remaining)
                else:
                    break

    return visited_order


def hotpotqa_item_to_query_and_docs(
    item: Dict,
    idx: int,
) -> Tuple[QueryItem, List[PoolDocument]]:
    """将 HotpotQA 条目转换为 QueryItem + PoolDocument 列表。"""
    ctx = item.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    sf_titles = set(item.get("supporting_facts", {}).get("title", []))

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
            topic=title,
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
