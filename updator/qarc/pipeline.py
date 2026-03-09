"""
QARC 流水线: 三阶段生命周期 — Bootstrap → Explore → Exploit

所属框架: QARC (Query-Aligned Retrieval-augmented Knowledge Curation)

=== 三阶段生命周期 ===

Phase 0 (Bootstrap): 冷启动
  - 在第一个用户查询之前执行
  - 从文档池 D_pool 中用多样性最大化选择初始 KB
  - 不需要任何用户信息

Phase 1 (Explore): 探索期
  - 前 N_warmup 个窗口
  - 每个窗口都触发重新策展 (ReCurate)
  - λ_max = 0.5 (激进替换，快速追踪兴趣)
  - η = 0.0 (纯兴趣匹配，不考虑多样性)
  - 积累 Alignment Gap 历史，为 Phase 2 初始化阈值

Phase 2 (Exploit): 利用期
  - 稳态运行
  - 仅当 Alignment Gap G(t) > EMA + k·MAD 时才触发重新策展
  - λ_max = 0.2 (保守替换，维持稳定性)
  - η = 0.1 (加入多样性，保持探索能力)
  - 有冷却期 (cooldown) 防止频繁重新策展

=== 阶段转换 ===

Phase 0 → 1: bootstrap() 后自动进入
Phase 1 → 2: 满足两个条件:
  1. 窗口数 ≥ N_warmup
  2. Gap 方差收敛: σ(recent) / σ(all) < ε_σ
     (近期 Gap 波动比整体波动小 → 说明兴趣已稳定)

Phase 2 → 1: 当连续 re_explore_trigger 次触发重新策展
  (说明兴趣发生了 Phase 2 无法处理的剧烈变化)

=== 漂移检测 (DriftLens-inspired) ===

除了 Alignment Gap 阈值外，还使用 GMM 漂移检测器:
  - 将 embedding 投影到"兴趣距离空间"
  - 用 GMM 建模基线分布 (KB 文档的) 和窗口分布 (新查询的)
  - 通过对称 KL 散度检测分布变化
  - 两个信号取 OR: Gap 超阈值 OR GMM 检测到漂移 → 触发重新策展

=== 与 ComRAG/ERASE 的对比 ===
- ComRAG 没有阶段概念，每条查询独立处理路由
- ERASE 没有阶段概念，每篇文档独立触发三步更新
- QARC 有完整的窗口级生命周期管理，根据兴趣变化的剧烈程度自适应调整策略
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

from updator.qarc.interest_model import (
    QueryWindowBuffer,
    InterestCluster,
    AlignmentGapResult,
    AdaptiveThreshold,
    GMMDriftDetector,
    auto_kmeans,
    compute_alignment_gap,
)
from updator.qarc.kb_curator import (
    QARCKBCurator,
    DocumentPool,
    Document,
    CurationResult,
)

logger = logging.getLogger(__name__)


# ============================================================
# 阶段枚举
# ============================================================

class QARCPhase(Enum):
    BOOTSTRAP = "bootstrap"
    EXPLORE = "explore"
    EXPLOIT = "exploit"


# ============================================================
# QARC 流水线
# ============================================================

class QARCPipeline:
    """
    QARC 主编排器 — 管理三阶段生命周期。

    协调以下组件:
      - QueryWindowBuffer:  积累查询直到窗口满
      - auto_kmeans():      对窗口查询做兴趣聚类
      - compute_alignment_gap(): 计算 KB 与兴趣的对齐度差距
      - GMMDriftDetector:   漂移检测 (DriftLens 启发)
      - QARCKBCurator:      子模选择 + 增量替换

    典型使用:
        pool = DocumentPool()
        pool.add_documents([...])
        curator = QARCKBCurator(pool, kb_budget=50)

        pipeline = QARCPipeline(curator, embed_fn=my_embed)
        pipeline.bootstrap()  # Phase 0 → Phase 1

        for query in query_stream:
            result = pipeline.process_query(query)
            # result["documents"]: 检索到的文档
            # result["answer"]:    LLM 生成的答案
            # result["window_event"]: 窗口事件 (如果窗口满了)
    """

    def __init__(
        self,
        curator: QARCKBCurator,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        llm_fn: Optional[Callable[[str, List[str]], str]] = None,
        # 窗口参数
        window_size: int = 50,
        # Phase 1 (Explore) 参数
        n_warmup_min: int = 5,
        epsilon_sigma: float = 0.3,
        explore_lambda_max: float = 0.5,
        explore_eta: float = 0.0,
        # Phase 2 (Exploit) 参数
        exploit_lambda_max: float = 0.2,
        exploit_eta: float = 0.1,
        cooldown_windows: int = 3,
        # 自适应阈值参数
        threshold_beta: float = 0.9,
        threshold_k: float = 2.0,
        # 重进入探索期的触发条件
        re_explore_trigger: int = 3,
        # RAG 检索参数
        retrieve_top_k: int = 5,
        # GMM 漂移检测参数 (DriftLens 启发)
        use_gmm_drift: bool = True,
        gmm_n_components_range: tuple = (1, 5),
        gmm_covariance_type: str = "diag",
        gmm_beta: float = 0.85,
        gmm_k_drift: float = 2.5,
    ):
        """
        参数:
            curator:             QARCKBCurator 实例
            embed_fn:            文本 → 向量的嵌入函数
            llm_fn:              (问题, 上下文文档列表) → 答案的 LLM 函数
            window_size:         每个窗口的查询数 W
            n_warmup_min:        Phase 1 最少窗口数
            epsilon_sigma:       Phase 1→2 收敛阈值 (σ_recent/σ_all < ε_σ)
            explore_lambda_max:  Phase 1 替换上限 (激进)
            explore_eta:         Phase 1 多样性系数 (0=纯兴趣)
            exploit_lambda_max:  Phase 2 替换上限 (保守)
            exploit_eta:         Phase 2 多样性系数 (保持探索)
            cooldown_windows:    Phase 2 重新策展后的冷却窗口数
            threshold_beta:      EMA 平滑因子
            threshold_k:         MAD 乘数
            re_explore_trigger:  连续触发次数 → 回退到 Phase 1
            retrieve_top_k:      RAG 检索 top-k
        """
        self.curator = curator
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn

        # 窗口缓冲区
        self.window_size = window_size
        self.buffer = QueryWindowBuffer(window_size=window_size)

        # Phase 1 参数
        self.n_warmup_min = n_warmup_min
        self.epsilon_sigma = epsilon_sigma
        self.explore_lambda_max = explore_lambda_max
        self.explore_eta = explore_eta

        # Phase 2 参数
        self.exploit_lambda_max = exploit_lambda_max
        self.exploit_eta = exploit_eta
        self.cooldown_windows = cooldown_windows
        self.re_explore_trigger = re_explore_trigger

        # RAG 参数
        self.retrieve_top_k = retrieve_top_k

        # GMM 漂移检测器 (DriftLens 启发)
        self.use_gmm_drift = use_gmm_drift
        self.gmm_drift = GMMDriftDetector(
            n_components_range=gmm_n_components_range,
            covariance_type=gmm_covariance_type,
            beta=gmm_beta,
            k_drift=gmm_k_drift,
        ) if use_gmm_drift else None

        # 状态变量
        self.phase = QARCPhase.BOOTSTRAP
        self.adaptive_threshold = AdaptiveThreshold(
            beta=threshold_beta, k=threshold_k
        )
        self.gap_history: List[float] = []
        self.window_count = 0
        self.cooldown_remaining = 0
        self.consecutive_triggers = 0  # 连续触发计数 (用于检测剧烈兴趣变化)

        # 统计
        self.total_queries = 0
        self.total_recurations = 0
        self.phase_history: List[Tuple[int, str]] = []

    # -------------------------------------------------------
    # Phase 0: Bootstrap (冷启动)
    # -------------------------------------------------------

    def bootstrap(
        self,
        historical_queries: Optional[List[np.ndarray]] = None,
    ):
        """
        Phase 0: 初始化 KB。

        两种模式:
          1. 冷启动 (无历史): 用多样性最大化从文档池选文档
          2. 热启动 (有历史查询): 先聚类历史查询得到兴趣模型，再选文档

        热启动的优势: 初始 KB 就已经对齐了已知的查询模式。

        参数:
            historical_queries: 可选的历史查询 embedding 列表
        """
        logger.info("=" * 60)
        logger.info("QARC Phase 0: Bootstrap (冷启动)")
        logger.info("=" * 60)

        if historical_queries and len(historical_queries) >= 10:
            # 热启动: 用历史查询聚类得到兴趣模型
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
            logger.info(f"热启动完成: 使用 {len(historical_queries)} 条历史查询")
        else:
            # 冷启动: 纯多样性
            self.curator.bootstrap_diversity()
            logger.info("冷启动完成: 使用多样性最大化")

        self.phase = QARCPhase.EXPLORE
        self.phase_history.append((0, "explore"))
        logger.info("进入 Phase 1 (Explore)")

        # GMM 基线将在第一个窗口结束后设置
        # (此时还没有兴趣中心，需要 AutoKMeans 输出)

    # -------------------------------------------------------
    # 主查询处理循环
    # -------------------------------------------------------

    def process_query(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        处理单条查询 — QARC 流水线的主入口。

        流程:
          1. 嵌入查询 (如果未提供)
          2. 从当前 KB 检索 top-k 文档
          3. LLM 生成答案 (如果配置了 llm_fn)
          4. 将查询加入窗口缓冲区
          5. 如果窗口满了 → 触发兴趣分析 + 可能的重新策展

        返回:
            包含 documents, answer, max_sim, phase, window_event 等信息的字典
        """
        self.total_queries += 1

        # 1. 嵌入
        if query_embedding is None:
            if self.embed_fn is None:
                raise ValueError("未提供 query_embedding 且未配置 embed_fn")
            query_embedding = self.embed_fn(query_text)

        # L2 归一化
        qnorm = np.linalg.norm(query_embedding)
        if qnorm > 1e-10:
            query_embedding = query_embedding / qnorm

        # 2. 从 KB 检索
        retrieved = self.curator.retrieve(query_embedding, top_k=self.retrieve_top_k)

        documents = [doc for doc, sim in retrieved]
        max_sim = retrieved[0][1] if retrieved else 0.0

        # 3. LLM 生成答案
        answer = None
        if self.llm_fn is not None and documents:
            context_texts = [doc.text for doc in documents]
            try:
                answer = self.llm_fn(query_text, context_texts)
            except Exception as e:
                logger.warning(f"LLM 调用失败: {e}")
                answer = None

        # 4. 加入窗口缓冲区
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

        # 5. 检查窗口是否满了
        if self.buffer.is_full:
            window_result = self._process_window()
            result["window_event"] = window_result

        return result

    # -------------------------------------------------------
    # 窗口处理 — 兴趣分析 + 阶段逻辑
    # -------------------------------------------------------

    def _process_window(self) -> Dict[str, Any]:
        """
        处理一个满窗口的查询。

        流程:
          1. 清空缓冲区，获取窗口内所有查询数据
          2. AutoKMeans 聚类 → 兴趣中心 + 权重
          3. 计算 Alignment Gap G(t)
          4. GMM 漂移检测 (可选)
          5. 根据当前阶段执行相应逻辑

        返回:
            窗口处理的详细结果
        """
        self.window_count += 1
        embeddings, texts, sims = self.buffer.flush()

        logger.info(f"\n{'='*60}")
        logger.info(
            f"窗口 #{self.window_count} | 阶段: {self.phase.value} | "
            f"查询数: {len(embeddings)}"
        )
        logger.info(f"{'='*60}")

        # 堆叠并归一化 embedding
        X = np.vstack(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-10, None)

        # 1. AutoKMeans 兴趣建模
        centroids, labels, weights = auto_kmeans(X)

        # 构建 InterestCluster 对象 (用于日志记录)
        clusters = []
        for i in range(len(centroids)):
            mask = labels == i
            cluster_texts = [texts[j] for j in range(len(texts)) if mask[j]]
            clusters.append(InterestCluster(
                centroid=centroids[i],
                weight=float(weights[i]),
                query_count=int(mask.sum()),
                cluster_id=i,
                representative_queries=cluster_texts[:3],
            ))

        logger.info(f"兴趣簇数: {len(clusters)}")
        for c in clusters:
            logger.info(f"  {c}")

        # 2. 计算 Alignment Gap
        # G(t) = 1 - avg max_sim(query, KB)
        kb_embs = self.curator.get_kb_embeddings()
        gap_result = compute_alignment_gap(X, kb_embs)
        self.gap_history.append(gap_result.gap)

        logger.info(
            f"Alignment Gap G={gap_result.gap:.4f} "
            f"(avg_max_sim={gap_result.avg_max_sim:.4f})"
        )

        # 2b. GMM 漂移检测 (DriftLens 启发)
        gmm_result = None
        if self.gmm_drift is not None:
            gmm_result = self.gmm_drift.compute_drift_score(X, centroids)

        # 3. 根据阶段执行逻辑
        curation_result = None
        phase_transition = None

        if self.phase == QARCPhase.EXPLORE:
            curation_result, phase_transition = self._explore_logic(
                centroids, weights, gap_result, gmm_result
            )
        elif self.phase == QARCPhase.EXPLOIT:
            curation_result, phase_transition = self._exploit_logic(
                centroids, weights, gap_result, gmm_result
            )

        return {
            "window_index": self.window_count,
            "phase": self.phase.value,
            "gap": gap_result.gap,
            "avg_max_sim": gap_result.avg_max_sim,
            "n_clusters": len(clusters),
            "clusters": [
                {"id": c.cluster_id, "weight": c.weight, "n_queries": c.query_count}
                for c in clusters
            ],
            "curation": curation_result,
            "phase_transition": phase_transition,
            "gmm_drift": gmm_result,
        }

    # -------------------------------------------------------
    # Phase 1 (Explore): 探索期逻辑
    # -------------------------------------------------------

    def _explore_logic(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        gap_result: AlignmentGapResult,
        gmm_result: Optional[Dict] = None,
    ) -> Tuple[Optional[CurationResult], Optional[str]]:
        """
        Phase 1 (Explore): 每个窗口都重新策展，检查是否可以转入 Phase 2。

        行为:
          - 每个窗口 → ReCurate (激进参数: λ_max=0.5, η=0)
          - 积累 Gap 历史
          - 检查 Gap 方差是否收敛 → 满足条件则转入 Phase 2

        Phase 1 → 2 转换条件:
          1. window_count ≥ n_warmup_min (至少跑 N_warmup 个窗口)
          2. σ(recent_5_windows) / σ(all_history) < ε_σ
             (近期 Gap 波动比整体小 → 兴趣已稳定)

        转换时:
          - 用积累的 Gap 历史初始化 AdaptiveThreshold
          - 用积累的漂移分数历史初始化 GMMDriftDetector 的 EMA
        """
        logger.info("Phase 1 (Explore): 触发重新策展")

        # 每个窗口都重新策展 (激进参数)
        curation = self.curator.recurate(
            centroids=centroids,
            weights=weights,
            lambda_max=self.explore_lambda_max,
            eta=self.explore_eta,
        )
        self.total_recurations += 1

        # 重新策展后更新 GMM 基线 (KB 变了，基线要重设)
        if self.gmm_drift is not None:
            kb_embs = self.curator.get_kb_embeddings()
            self.gmm_drift.set_reference(kb_embs, centroids)

        # 检查 Phase 1 → Phase 2 转换
        phase_transition = None
        if self.window_count >= self.n_warmup_min and len(self.gap_history) >= self.n_warmup_min:
            # 检查收敛: σ_recent / σ_all < ε_σ
            recent_window = min(5, len(self.gap_history))
            recent_std = float(np.std(self.gap_history[-recent_window:]))
            total_std = float(np.std(self.gap_history))

            ratio = recent_std / max(total_std, 1e-8)

            logger.info(
                f"  收敛检查: σ_recent={recent_std:.4f}, "
                f"σ_total={total_std:.4f}, 比值={ratio:.4f}, ε={self.epsilon_sigma}"
            )

            if ratio < self.epsilon_sigma:
                # 转入 Phase 2!
                self.phase = QARCPhase.EXPLOIT
                self.adaptive_threshold.initialize_from_history(self.gap_history)
                self.cooldown_remaining = 0
                self.consecutive_triggers = 0
                self.phase_history.append((self.window_count, "exploit"))
                phase_transition = "explore → exploit"

                # 初始化 GMM 漂移检测的 EMA/MAD
                if self.gmm_drift is not None:
                    self.gmm_drift.initialize_from_explore_history()

                logger.info(
                    f"*** 阶段转换: Explore → Exploit ***\n"
                    f"    阈值初始化: "
                    f"EMA={self.adaptive_threshold.g_ema:.4f}, "
                    f"MAD={self.adaptive_threshold.g_mad:.4f}"
                )

        return curation, phase_transition

    # -------------------------------------------------------
    # Phase 2 (Exploit): 利用期逻辑
    # -------------------------------------------------------

    def _exploit_logic(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        gap_result: AlignmentGapResult,
        gmm_result: Optional[Dict] = None,
    ) -> Tuple[Optional[CurationResult], Optional[str]]:
        """
        Phase 2 (Exploit): 自适应阈值触发重新策展。

        行为:
          - 每个窗口更新 EMA/MAD
          - G(t) > threshold OR GMM 检测到漂移 → 触发重新策展
          - 有冷却期 (cooldown) 防止频繁触发
          - 连续触发 ≥ re_explore_trigger 次 → 回退到 Phase 1

        触发条件 (取 OR):
          信号1: Gap > EMA + k·MAD  (Alignment Gap 异常)
          信号2: GMM KL 散度 > EMA_drift + k_drift·MAD_drift  (分布漂移)

        回退逻辑:
          如果 Phase 2 连续多次触发重新策展，说明兴趣发生了剧烈变化，
          Phase 2 的保守策略 (λ_max=0.2) 无法跟上。
          此时回退到 Phase 1 (λ_max=0.5)，重新激进地追踪兴趣。
        """
        # 更新阈值并检查触发
        gap_triggered = self.adaptive_threshold.update(gap_result.gap)
        gmm_triggered = (gmm_result or {}).get("triggered", False)

        # 组合信号: 任一触发即可
        triggered = gap_triggered or gmm_triggered

        trigger_src = []
        if gap_triggered:
            trigger_src.append("Gap")
        if gmm_triggered:
            trigger_src.append(f"GMM(D={gmm_result['drift_score']:.4f})")

        logger.info(
            f"Phase 2 (Exploit): G={gap_result.gap:.4f}, "
            f"阈值={self.adaptive_threshold.threshold:.4f}, "
            f"触发={triggered} [{'+'.join(trigger_src) if trigger_src else '-'}], "
            f"冷却={self.cooldown_remaining}"
        )

        curation = None
        phase_transition = None

        if triggered and self.cooldown_remaining <= 0:
            # 触发重新策展
            logger.info("  自适应阈值超过 → 执行重新策展")

            curation = self.curator.recurate(
                centroids=centroids,
                weights=weights,
                lambda_max=self.exploit_lambda_max,
                eta=self.exploit_eta,
            )
            self.total_recurations += 1
            self.cooldown_remaining = self.cooldown_windows
            self.consecutive_triggers += 1

            # 更新 GMM 基线
            if self.gmm_drift is not None:
                kb_embs = self.curator.get_kb_embeddings()
                self.gmm_drift.set_reference(kb_embs, centroids)

            # 检查是否需要回退到 Phase 1
            if self.consecutive_triggers >= self.re_explore_trigger:
                self.phase = QARCPhase.EXPLORE
                self.consecutive_triggers = 0
                self.phase_history.append((self.window_count, "re-explore"))
                phase_transition = "exploit → re-explore"

                logger.info(
                    f"*** 阶段转换: Exploit → Re-Explore ***\n"
                    f"    连续 {self.re_explore_trigger} 次触发, "
                    f"    回退到激进重新策展模式"
                )
        else:
            # 未触发或在冷却期
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
            if not triggered:
                self.consecutive_triggers = 0  # 重置连续触发计数

        return curation, phase_transition

    # -------------------------------------------------------
    # 工具方法
    # -------------------------------------------------------

    def force_recurate(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        lambda_max: Optional[float] = None,
        eta: Optional[float] = None,
    ) -> CurationResult:
        """强制重新策展 — 无视阶段/阈值 (用于测试)"""
        lam = lambda_max if lambda_max is not None else self.exploit_lambda_max
        e = eta if eta is not None else self.exploit_eta
        return self.curator.recurate(centroids=centroids, weights=weights,
                                     lambda_max=lam, eta=e)

    def get_current_kb_docs(self) -> List[Document]:
        """返回当前 KB 文档列表"""
        return self.curator.get_kb_docs_list()

    def get_kb_size(self) -> int:
        return self.curator.kb_size

    def get_statistics(self) -> Dict[str, Any]:
        """返回完整的流水线统计信息"""
        return {
            "phase": self.phase.value,
            "total_queries": self.total_queries,
            "total_windows": self.window_count,
            "total_recurations": self.total_recurations,
            "gap_history": list(self.gap_history),
            "phase_history": self.phase_history,
            "threshold_state": self.adaptive_threshold.get_state(),
            "gmm_drift_state": self.gmm_drift.get_state() if self.gmm_drift else None,
            "cooldown_remaining": self.cooldown_remaining,
            "consecutive_triggers": self.consecutive_triggers,
            "curator": self.curator.get_statistics(),
        }

    def get_phase(self) -> str:
        """返回当前阶段名"""
        return self.phase.value

    def __repr__(self):
        return (
            f"QARCPipeline(phase={self.phase.value}, "
            f"queries={self.total_queries}, windows={self.window_count}, "
            f"kb_size={self.curator.kb_size})"
        )
