"""
benchmark/adapters.py — algorithms/ 方法的 KBUpdateStrategy 适配器

将 QARC 内部逻辑适配到统一的 base.KBUpdateStrategy 接口，
使得 run_experiments.py 可以公平对比 QARC / Static / Random。

适配策略:
  - QARC:   内部用 QARCPipeline (兴趣建模 + submodular KB curation + 三阶段生命周期)

注: ComRAG / ERASE 适配器已移除 (跨范式, 非固定预算缓存换入换出, 指标不可比)。
    cache replacement 主 baseline 见 algorithms/cache/registry.py。
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
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
# QARC 适配器
# ============================================================

class QARCStrategyAdapter(KBUpdateStrategy):
    """
    QARC — 查询对齐的检索增强知识库管理 (我们的方法)

    核心流程:
      1. Bootstrap: 从候选池中用 submodular 优化选出初始 KB
      2. 每条查询进入兴趣窗口, DriftLens 检测漂移 + 计算 AlignmentGap
      3. Agent 决策: 无操作 / 轻度更新 / 激进更新 / 重校准
      4. 执行 submodular KB 更新 (如有)
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

        # QARC 组件 (lazy init)
        self._pipeline = None
        self._curator = None

    @property
    def name(self) -> str:
        return "QARC"

    def initialize(self, doc_pool, doc_embeddings, kb_budget):
        self._doc_pool = doc_pool
        self._doc_embeddings = doc_embeddings
        self._kb_budget = kb_budget
        self._pool_id_to_idx = {d["doc_id"]: i for i, d in enumerate(doc_pool)}

        from algorithms.qarc.curation.kb_curator import DocumentPool, Document, QARCKBCurator
        from algorithms.qarc.pipeline import QARCPipeline

        # 1) 构建 QARC DocumentPool
        qarc_pool = DocumentPool()
        qarc_docs = []
        for i, d in enumerate(doc_pool):
            qarc_docs.append(Document(
                doc_id=d["doc_id"],
                text=d["text"],
                embedding=doc_embeddings[i],
                metadata={"topic": d.get("topic", "unknown")},
            ))
        qarc_pool.add_documents(qarc_docs)

        # 2) 构建 QARCKBCurator
        self._curator = QARCKBCurator(
            document_pool=qarc_pool,
            kb_budget=kb_budget,
            candidate_top_k=self._candidate_top_k,
        )

        # 3) 构建 QARCPipeline (通过 QARCConfig 传入所有参数)
        from algorithms.qarc.config import QARCConfig
        cfg = QARCConfig(
            window_size=self._window_size,
            retrieve_top_k=10,
            agent_warmup_windows=3,       # 前 3 窗口始终积极更新
            agent_cooldown_windows=1,     # 更新后冷却 1 窗口
            agent_gap_k=1.5,              # Gap 阈值灵敏度
            agent_lambda_mild=0.3,        # 轻度更新替换 30%
            agent_lambda_aggressive=0.6,  # 激进更新替换 60%
        )
        self._pipeline = QARCPipeline(
            curator=self._curator,
            cfg=cfg,
        )

        # 4) FAISS 索引: KB 检索加速
        import faiss
        d = doc_embeddings.shape[1]
        self._emb_dim = d

        # 5) Bootstrap: 用前 window_size 条文档的 embedding 作为初始 "queries"
        n_init = min(self._window_size, len(doc_pool))
        init_embeddings = [doc_embeddings[i] for i in range(n_init)]
        self._pipeline.bootstrap(historical_queries=init_embeddings)

        # 同步 KB doc IDs
        self._sync_kb_ids()

    def process_query(self, query_text, query_embedding, step, gold_doc_ids=None):
        if self._pipeline is None:
            return ProcessResult(kb_size=0)

        # QARC pipeline 处理查询
        result = self._pipeline.process_query(
            query_text=query_text,
            query_embedding=query_embedding,
        )

        # 检查是否触发了 curation (窗口满了才可能触发)
        window_event = result.get("window_event")
        update_done = False
        if window_event is not None and window_event.get("curation") is not None:
            update_done = True
            self._metrics.total_updates += 1

        # 同步 KB
        self._sync_kb_ids()

        # 从 result["documents"] 获取检索到的文档 ID
        retrieved = []
        docs = result.get("documents", [])
        for doc in docs:
            retrieved.append(doc.doc_id)

        # 如果 QARC 检索结果不够 10 个, 用 KB 内余弦补充
        if len(retrieved) < 10:
            extra = self._retrieve_from_kb(query_embedding, top_k=10)
            for did in extra:
                if did not in retrieved:
                    retrieved.append(did)
                if len(retrieved) >= 10:
                    break

        self._metrics.total_queries += 1
        self._metrics.kb_size_history.append(len(self._kb_doc_ids))

        return ProcessResult(
            retrieved_doc_ids=retrieved[:10],
            update_performed=update_done,
            kb_size=len(self._kb_doc_ids),
            extra_metrics={
                "phase": result.get("phase", "unknown"),
                "max_sim": result.get("max_sim", 0.0),
            },
        )

    def get_kb_doc_ids(self):
        return set(self._kb_doc_ids)

    def get_kb_size(self):
        return len(self._kb_doc_ids)

    def _sync_kb_ids(self):
        """从 QARCKBCurator 同步当前 KB 的文档 ID"""
        if self._curator is not None:
            self._kb_doc_ids = self._curator.get_kb_doc_ids()
        else:
            self._kb_doc_ids = set()

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

