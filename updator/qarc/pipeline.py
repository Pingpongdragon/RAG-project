"""
QARC Pipeline (v2) — 基于 Query-KB 对齐特征的漂移检测 + Agent 决策

=== 数据流 ===

Bootstrap:
  文档池 → 初始 KB
  (暂无 query 历史 → DriftLens 延迟到第一个 warmup 窗口结束后初始化)

Online Loop (每个查询窗口):
  1. AutoKMeans 聚类 → 兴趣中心 + 权重
  2. DriftLens 检测对齐漂移 (Part 1):
     每个 query → alignment_features = [sim(q,c1),..,sim(q,cK), top1,..,topN]
     FID(当前窗口对齐特征, 基线对齐特征) > threshold → 漂移
  3. 计算 AlignmentGap → 对齐度差距
  4. Agent 决策 (Part 2) → 更新动作
  5. 执行更新 (如有) → CurationResult → 重设 DriftLens 基线

=== Query 历史管理 ===
Pipeline 维护一个 query 历史 ring buffer:
  - 每个 query embedding 都会记录
  - DriftLens set_baseline 和 calibrate_threshold 使用这些历史
  - KB 更新后用最近 N 个 query 重建基线 (因为对齐特征的定义依赖当前 KB)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque

from updator.qarc.curation.interest_model import (
    QueryWindowBuffer,
    InterestCluster,
    auto_kmeans,
    compute_alignment_gap,
)
from updator.qarc.detection.drift_detector import DriftLensDetector, DriftResult
from updator.qarc.decision.kb_agent import KBUpdateAgent, AgentDecision, UpdateAction
from updator.qarc.config import QARCConfig
from updator.qarc.interfaces import BaseDriftDetector, BaseUpdateAgent, BaseKBCurator
from updator.qarc.curation.kb_curator import (
    QARCKBCurator,
    DocumentPool,
    Document,
    CurationResult,
)

logger = logging.getLogger(__name__)


class QARCPhase(Enum):
    BOOTSTRAP = "bootstrap"
    ONLINE = "online"


class QARCPipeline:
    """QARC 主流水线 — 对齐漂移检测 + Agent 决策。

    使用:
        pipeline = QARCPipeline(curator, window_size=8)
        pipeline.bootstrap()

        for query in query_stream:
            result = pipeline.process_query(query_text, query_embedding)
    """

    def __init__(
        self,
        curator: BaseKBCurator,
        cfg: Optional[QARCConfig] = None,
        detector: Optional[BaseDriftDetector] = None,
        agent: Optional[BaseUpdateAgent] = None,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        llm_fn: Optional[Callable[[str, List[str]], str]] = None,
    ):
        if cfg is None:
            cfg = QARCConfig()
        self.cfg = cfg
        self.curator = curator
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.window_size = cfg.window_size
        self.retrieve_top_k = cfg.retrieve_top_k

        # 窗口缓冲区
        self.buffer = QueryWindowBuffer(window_size=cfg.window_size)

        # Part 1: 漂移检测器 — 可注入自定义实现
        self.detector: BaseDriftDetector = detector or DriftLensDetector(
            n_clusters=cfg.drift_n_clusters,
            top_n_sims=cfg.drift_top_n_sims,
            threshold_percentile=cfg.drift_threshold_percentile,
            threshold_n_samples=cfg.drift_threshold_n_samples,
            cov_reg=cfg.drift_cov_reg,
            random_state=cfg.drift_random_state,
        )

        # Part 2: 更新决策 Agent — 可注入自定义实现
        self.agent: BaseUpdateAgent = agent or KBUpdateAgent(
            warmup_windows=cfg.agent_warmup_windows,
            gap_ema_beta=cfg.agent_gap_ema_beta,
            gap_k=cfg.agent_gap_k,
            lambda_mild=cfg.agent_lambda_mild,
            lambda_aggressive=cfg.agent_lambda_aggressive,
            eta_mild=cfg.agent_eta_mild,
            eta_aggressive=cfg.agent_eta_aggressive,
            cooldown_windows=cfg.agent_cooldown_windows,
            recalibrate_after_n_drifts=cfg.agent_recalibrate_after,
        )

        # Query 历史 (ring buffer, 用于 DriftLens 基线建立)
        self._query_history = deque(maxlen=cfg.query_history_max)

        # 状态
        self.phase = QARCPhase.BOOTSTRAP
        self.total_queries = 0
        self.total_recurations = 0
        self.window_count = 0
        self.gap_history: List[float] = []

    # ─────── Phase 0: Bootstrap ───────

    def bootstrap(
        self,
        historical_queries: Optional[List[np.ndarray]] = None,
    ):
        """Phase 0: 初始化 KB。

        DriftLens 此时不初始化 (没有 query 历史)。
        它会在 warmup 结束后用积累的 query 历史初始化。
        """
        logger.info("=" * 60)
        logger.info("QARC Bootstrap")
        logger.info("=" * 60)

        if historical_queries and len(historical_queries) >= 10:
            X = np.vstack(historical_queries)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(norms, 1e-10, None)
            centroids, labels, weights = auto_kmeans(X)
            self.curator.bootstrap_from_queries(
                query_embeddings=X,
                centroids=centroids,
                weights=weights,
                eta=0.05,
            )
            # 记入 query 历史
            for emb in X:
                self._query_history.append(emb)
            logger.info(f"热启动: {len(historical_queries)} 条历史查询")
        else:
            self.curator.bootstrap_diversity()
            logger.info("冷启动: 多样性最大化")

        self.phase = QARCPhase.ONLINE
        logger.info("进入 Online 阶段 (DriftLens 将在 warmup 后初始化)")

    # ─────── DriftLens 基线初始化 ───────

    def _init_detector(self):
        """用积累的 query 历史 + 当前 KB 初始化 DriftLens。
        需要至少 3 * window_size 条 query (约 3 个窗口) 才能有效校准阈值。
        """
        min_queries = max(3 * self.window_size, 20)
        if len(self._query_history) < min_queries:
            return

        kb_embs = self.curator.get_kb_embeddings()
        if kb_embs.shape[0] < 5:
            return

        q_history = np.array(list(self._query_history), dtype=np.float64)

        ok = self.detector.set_baseline(kb_embs, q_history)
        if ok:
            self.detector.calibrate_threshold(q_history, self.window_size)
            logger.info(
                f"DriftLens 初始化: {len(self._query_history)} 条 query 历史, "
                f"threshold={self.detector.threshold:.4f}"
            )

    def _recalibrate_detector(self):
        """KB 更新后重新校准 DriftLens (用近期 query + 新 KB)。"""
        min_queries = max(3 * self.window_size, 20)
        if len(self._query_history) < min_queries:
            return

        kb_embs = self.curator.get_kb_embeddings()
        if kb_embs.shape[0] < 5:
            return

        q_history = np.array(list(self._query_history), dtype=np.float64)

        ok = self.detector.set_baseline(kb_embs, q_history)
        if ok:
            self.detector.calibrate_threshold(q_history, self.window_size)

    # ─────── 主查询处理 ───────

    def process_query(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """处理单条查询。"""
        self.total_queries += 1

        # 1. 嵌入
        if query_embedding is None:
            if self.embed_fn is None:
                raise ValueError("未提供 query_embedding 且未配置 embed_fn")
            query_embedding = self.embed_fn(query_text)

        qnorm = np.linalg.norm(query_embedding)
        if qnorm > 1e-10:
            query_embedding = query_embedding / qnorm

        # 记入 query 历史
        self._query_history.append(query_embedding.copy())

        # 2. KB 检索
        retrieved = self.curator.retrieve(
            query_embedding, top_k=self.retrieve_top_k
        )
        documents = [doc for doc, sim in retrieved]
        max_sim = retrieved[0][1] if retrieved else 0.0

        # 3. LLM (可选)
        answer = None
        if self.llm_fn is not None and documents:
            try:
                answer = self.llm_fn(query_text, [doc.text for doc in documents])
            except Exception as e:
                logger.warning(f"LLM 失败: {e}")

        # 4. 缓冲
        self.buffer.add(
            embedding=query_embedding,
            text=query_text,
            max_sim_to_kb=max_sim,
        )

        result = {
            "documents": documents,
            "answer": answer,
            "max_sim": max_sim,
            "phase": self.phase.value,
            "query_index": self.total_queries,
            "window_event": None,
        }

        if self.buffer.is_full:
            result["window_event"] = self._process_window()

        return result

    # ─────── 窗口处理 ───────

    def _process_window(self) -> Dict[str, Any]:
        """窗口满时: 聚类 → 对齐检测 → Gap → Agent → 执行。"""
        self.window_count += 1
        embeddings, texts, sims = self.buffer.flush()

        logger.info(f"\n{'='*60}")
        logger.info(f"窗口 #{self.window_count} | 查询数: {len(embeddings)}")
        logger.info(f"{'='*60}")

        # 1. AutoKMeans 兴趣聚类
        X = np.vstack(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-10, None)

        centroids, labels, weights = auto_kmeans(X)

        clusters = []
        for i in range(len(centroids)):
            mask = labels == i
            clusters.append(InterestCluster(
                centroid=centroids[i],
                weight=float(weights[i]),
                query_count=int(mask.sum()),
                cluster_id=i,
                representative_queries=[
                    texts[j] for j in range(len(texts)) if mask[j]
                ][:3],
            ))
        logger.info(f"兴趣簇数: {len(clusters)}")

        # 2. DriftLens 对齐漂移检测 (Part 1)
        # 只在 warmup 结束后才尝试初始化 (warmup 期间 Agent 始终激进更新, 不需要检测)
        if not self.detector.is_ready and self.window_count > self.agent.warmup_windows:
            self._init_detector()

        drift_result = self.detector.detect(X)

        # 3. AlignmentGap
        kb_embs = self.curator.get_kb_embeddings()
        gap_result = compute_alignment_gap(X, kb_embs)
        self.gap_history.append(gap_result.gap)

        logger.info(
            f"Gap={gap_result.gap:.4f}, "
            f"FID={drift_result.fid_score:.4f}, "
            f"drifted={drift_result.is_drifted}"
        )

        # 4. Agent 决策 (Part 2)
        decision = self.agent.decide(drift_result, gap_result)

        # 5. 执行
        curation = None
        if decision.action != UpdateAction.NO_OP:
            # KB 更新
            curation = self.curator.recurate(
                centroids=centroids,
                weights=weights,
                lambda_max=decision.lambda_max,
                eta=decision.eta,
            )
            self.total_recurations += 1

            # KB 变了 → 对齐特征定义变了 → 重建基线
            self._recalibrate_detector()

            logger.info(
                f"Agent 执行: {decision.action.value} — "
                f"+{len(curation.added_ids)} / -{len(curation.removed_ids)}, "
                f"原因: {decision.reason}"
            )

        return {
            "window_index": self.window_count,
            "gap": gap_result.gap,
            "avg_max_sim": gap_result.avg_max_sim,
            "n_clusters": len(clusters),
            "clusters": [
                {"id": c.cluster_id, "weight": c.weight,
                 "n_queries": c.query_count}
                for c in clusters
            ],
            "drift": {
                "fid_score": drift_result.fid_score,
                "threshold": drift_result.threshold,
                "is_drifted": drift_result.is_drifted,
            },
            "decision": {
                "action": decision.action.value,
                "reason": decision.reason,
            },
            "curation": curation,
        }

    # ─────── 工具方法 ───────

    def get_current_kb_docs(self) -> List[Document]:
        return self.curator.get_kb_docs_list()

    def get_kb_size(self) -> int:
        return self.curator.kb_size

    def get_phase(self) -> str:
        return self.phase.value

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "total_queries": self.total_queries,
            "total_windows": self.window_count,
            "total_recurations": self.total_recurations,
            "gap_history": list(self.gap_history),
            "query_history_size": len(self._query_history),
            "detector_state": self.detector.get_state(),
            "agent_state": self.agent.get_statistics(),
            "curator": self.curator.get_statistics(),
        }

    def __repr__(self):
        return (
            f"QARCPipeline(phase={self.phase.value}, "
            f"queries={self.total_queries}, windows={self.window_count}, "
            f"kb_size={self.curator.kb_size})"
        )
