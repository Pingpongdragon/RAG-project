"""
benchmarks/archive_legacy/adapters.py — algorithms/ 方法的 KBUpdateStrategy 适配器

将 DRIP 内部逻辑适配到统一的 base.KBUpdateStrategy 接口，
使得 run_experiments.py 可以公平对比 DRIP / Static / Random。

适配策略:
  - DRIP: 包装 algorithms.drip.DRIP，即当前唯一论文策略

注: ComRAG / ERASE 适配器已移除 (跨范式, 非固定预算缓存换入换出, 指标不可比)。
    cache replacement 主 baseline 见 algorithms/cache/registry.py。
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algorithms.base import KBUpdateStrategy, ProcessResult, MethodMetrics, select_diverse_initial_kb

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _cosine_retrieve(query_emb: np.ndarray, doc_embs: np.ndarray,
                     doc_ids: List[str], top_k: int = 10) -> List[str]:
    """在给定的文档集合内做余弦检索"""
    if not doc_ids or doc_embs.shape[0] == 0:
        return []
    sims = doc_embs @ query_emb
    k = min(top_k, len(doc_ids))
    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    return [doc_ids[i] for i in top_idx]


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-10)


# ============================================================
# DRIP 适配器
# ============================================================

class DRIPStrategyAdapter(KBUpdateStrategy):
    """
    Adapter for the canonical cache-family DRIP policy.

    The old DRIPPipeline/KBUpdateAgent/DRIPKBCurator path and the simplified
    old cache policies were retired. This adapter keeps the benchmark CLI
    usable by feeding single-query windows into final DRIP.
    """

    def __init__(
        self,
        kb_budget: int = 50,
        window_size: int = 5,
        candidate_top_k: int = 100,
        eta: float = 0.05,
    ):
        self._kb_budget = kb_budget
        self._window_size = window_size
        self._candidate_top_k = candidate_top_k
        self._eta = eta
        self._metrics = MethodMetrics()

        # 内部状态 (在 initialize 中赋值)
        self._doc_pool = []
        self._doc_embeddings = None
        self._pool_id_to_idx = {}
        self._kb_doc_ids: Set[str] = set()

        self._strategy = None

    @property
    def name(self) -> str:
        return "DRIP"

    def initialize(self, doc_pool, doc_embeddings, kb_budget):
        self._doc_pool = doc_pool
        self._doc_embeddings = doc_embeddings
        self._kb_budget = kb_budget
        self._pool_id_to_idx = {d["doc_id"]: i for i, d in enumerate(doc_pool)}

        from algorithms.drip import DRIP

        title_to_idx = {
            d.get("title", d["doc_id"]): i for i, d in enumerate(doc_pool)
        }
        self._strategy = DRIP("DRIP", doc_pool, doc_embeddings, title_to_idx)
        self._kb_doc_ids = select_diverse_initial_kb(
            doc_pool, doc_embeddings, kb_budget
        )
        self._strategy.set_kb(self._kb_doc_ids)

    def process_query(self, query_text, query_embedding, step, gold_doc_ids=None):
        if self._strategy is None:
            return ProcessResult(kb_size=0)

        before = self._strategy.update_cost
        retrieved = self._retrieve_from_kb(query_embedding, top_k=10)
        q_emb = np.asarray(query_embedding, dtype=float).reshape(1, -1)
        self._strategy.step([{"question": query_text}], q_emb, step)
        self._kb_doc_ids = set(self._strategy.kb)
        update_done = self._strategy.update_cost > before
        if update_done:
            self._metrics.total_updates += 1

        self._metrics.total_queries += 1
        self._metrics.kb_size_history.append(len(self._kb_doc_ids))

        return ProcessResult(
            retrieved_doc_ids=retrieved[:10],
            update_performed=update_done,
            kb_size=len(self._kb_doc_ids),
            extra_metrics={
                "updates": self._strategy.update_cost,
            },
        )

    def get_kb_doc_ids(self):
        return set(self._kb_doc_ids)

    def get_kb_size(self):
        return len(self._kb_doc_ids)

    def _retrieve_from_kb(self, query_emb, top_k=10):
        kb_ids = list(self._kb_doc_ids)
        if not kb_ids:
            return []
        indices = [self._pool_id_to_idx[did] for did in kb_ids if did in self._pool_id_to_idx]
        if not indices:
            return []
        id_list = [did for did in kb_ids if did in self._pool_id_to_idx]
        # 对小 KB 直接 numpy; 大 KB 也没问题因为只在 KB 子集内检索
        kb_embs = self._doc_embeddings[indices]
        return _cosine_retrieve(query_emb, kb_embs, id_list, top_k)
