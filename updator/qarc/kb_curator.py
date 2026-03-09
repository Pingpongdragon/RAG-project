"""
QARC KB 策展器: 子模函数文档选择 + 增量 KB 替换

所属框架: QARC (Query-Aligned Retrieval-augmented Knowledge Curation)

=== 本文件的核心职责 ===

这是 QARC 与 ComRAG/ERASE 最本质的区别所在:
QARC 的 KB 不是静态的，也不是修改事实的，而是"从文档池中动态选择文档"。

具体来说:
  文档池 D_pool (巨大的外部文档集) ──子模选择──→ KB K (小型、对齐用户兴趣)

本文件实现:
  1. Phase 0 Bootstrap: 多样性最大化初始化 KB (还没有用户查询时)
  2. ReCurate():        兴趣加权子模选择 + 增量替换 (检测到兴趣变化时)
  3. ERASE 一致性检查:  对新加入的文档执行 ERASE 风格的事实核查 (可选)

=== 子模目标函数 (Submodular Objective) ===

f(S) = f_interest(S) + η · f_diversity(S)

其中:
  f_interest(S) = Σ_{i=1}^{m} α_i · max_{d∈S} CosSim(c_i, d)
    - m 个兴趣簇（AutoKMeans 输出的中心 c_i 和权重 α_i）
    - 对每个兴趣簇，找 KB 中最匹配的文档
    - 加权求和: 权重大的兴趣簇贡献更多

  f_diversity(S) = (1/|D_pool|) · Σ_{d∈D_pool} max_{d'∈S} CosSim(d, d')
    - Facility Location 函数: 衡量 KB 对文档池的"覆盖度"
    - 保证 KB 不会过于狭窄地聚焦于某个话题

这两个都是单调递增的子模函数，因此:
  贪心算法保证 (1 - 1/e) ≈ 0.63 的近似比！

=== 增量替换 (Incremental Replacement) ===

每次重新策展时不是推翻整个 KB 重来，而是:
  K_ideal = GreedySubmodSelect(candidates, centroids, weights)
  to_add   = K_ideal - K_old
  to_remove = K_old - K_ideal
  实际替换量 ≤ λ_max × |K|  (防止灾难性 KB 震荡)

λ_max 的取值:
  Phase 1 (Explore): λ_max = 0.5 (激进替换，快速追踪兴趣)
  Phase 2 (Exploit): λ_max = 0.2 (保守替换，维持稳定性)

=== 与其他方法的对比 ===
- ERASE: 文档驱动 (document push) — 文档到来 → 修改事实
- ComRAG: 查询驱动但 KB 冻结 (KB frozen) — 只更新 QA 记忆
- QARC:  查询驱动 + 兴趣拉取 (interest pull) — 兴趣变化 → 重选文档
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Set, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Document:
    """文档池或 KB 中的一篇文档"""
    doc_id: str
    text: str
    embedding: np.ndarray        # L2 归一化的稠密向量
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"Document[{self.doc_id}]: {self.text[:60]}..."

    def __hash__(self):
        return hash(self.doc_id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self.doc_id == other.doc_id
        return False


@dataclass
class CurationResult:
    """一次 ReCurate 操作的结果"""
    added_ids: List[str]         # 新加入 KB 的文档 ID
    removed_ids: List[str]       # 从 KB 移除的文档 ID
    kb_size: int                 # 操作后 KB 大小
    objective_before: float      # 操作前子模目标值
    objective_after: float       # 操作后子模目标值
    replacement_ratio: float     # 实际替换比例 |added| / |KB|


# ============================================================
# 文档池 (Document Pool)
# ============================================================

class DocumentPool:
    """
    内存文档池 — 支持稠密向量检索。

    在生产环境中，这里会封装 FAISS 索引。
    为清晰起见，当前使用暴力 cosine similarity 检索。

    文档池 D_pool 通常很大（数千~数万篇），
    而 KB K 只从中选取一小部分（几十~几百篇）。
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self._embedding_matrix: Optional[np.ndarray] = None
        self._id_list: List[str] = []
        self._dirty = True  # 有新增文档时标记为脏，需要重建索引

    def add_document(self, doc: Document):
        """添加单篇文档到池中"""
        self.documents[doc.doc_id] = doc
        self._dirty = True

    def add_documents(self, docs: List[Document]):
        """批量添加文档"""
        for doc in docs:
            self.documents[doc.doc_id] = doc
        self._dirty = True

    def _rebuild_index(self):
        """重建 embedding 矩阵（懒加载，仅在脏标记时重建）"""
        if not self._dirty:
            return
        self._id_list = list(self.documents.keys())
        if self._id_list:
            embeddings = [self.documents[did].embedding for did in self._id_list]
            self._embedding_matrix = np.vstack(embeddings)
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            self._embedding_matrix = self._embedding_matrix / np.clip(norms, 1e-10, None)
        else:
            self._embedding_matrix = None
        self._dirty = False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        检索与查询最相似的 top-k 文档。

        参数:
            query_embedding: (d,) 归一化查询向量
            top_k:          返回数量
            exclude_ids:    要排除的文档 ID 集合

        返回:
            [(Document, similarity)] 按相似度降序
        """
        self._rebuild_index()

        if self._embedding_matrix is None:
            return []

        query = query_embedding.reshape(1, -1)
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query = query / qnorm

        sims = (self._embedding_matrix @ query.T).flatten()

        if exclude_ids:
            for i, did in enumerate(self._id_list):
                if did in exclude_ids:
                    sims[i] = -2.0

        if len(sims) <= top_k:
            top_idx = np.argsort(sims)[::-1]
        else:
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        results = []
        for i in top_idx:
            if sims[i] > -1.0:
                did = self._id_list[i]
                results.append((self.documents[did], float(sims[i])))

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """返回所有文档的 (n, d) embedding 矩阵"""
        self._rebuild_index()
        if self._embedding_matrix is None:
            return np.empty((0, 0))
        return self._embedding_matrix.copy()

    def get_all_ids(self) -> List[str]:
        """返回所有文档 ID"""
        self._rebuild_index()
        return self._id_list.copy()

    @property
    def size(self) -> int:
        return len(self.documents)


# ============================================================
# 子模目标函数
# ============================================================

def _interest_coverage(
    selected_embs: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    兴趣覆盖目标函数:
      f_interest(S) = Σ_{i=1}^{m} α_i · max_{d∈S} CosSim(c_i, d)

    直觉: 对每个兴趣簇 i (由 AutoKMeans 发现):
      - 找 KB 中最匹配该兴趣的文档
      - 乘以该兴趣的权重 α_i (查询占比)
      - 权重大的兴趣点贡献更多

    这是一个单调递增的子模函数:
      - 单调: 加入更多文档，覆盖只会增加不会减少
      - 子模: 边际收益递减 (已经覆盖的兴趣，再加文档收益小)

    参数:
        selected_embs: (|S|, d) 已选文档 embedding
        centroids:     (m, d) 兴趣簇中心
        weights:       (m,) 兴趣权重 α_i
    """
    if selected_embs.shape[0] == 0 or centroids.shape[0] == 0:
        return 0.0

    # (m, |S|) — 每个兴趣中心与每个文档的相似度
    sim_matrix = centroids @ selected_embs.T

    # 每个兴趣中心的最大相似度
    max_sims = sim_matrix.max(axis=1)  # (m,)

    return float((weights * max_sims).sum())


def _diversity_coverage(
    selected_embs: np.ndarray,
    pool_embs: np.ndarray,
) -> float:
    """
    多样性覆盖目标 (Facility Location):
      f_div(S) = (1/|D_pool|) · Σ_{d∈D_pool} max_{d'∈S} CosSim(d, d')

    直觉: 对文档池中的每篇文档:
      - 找 KB 中最"代表"它的那篇
      - 取平均 → 衡量 KB 对整个文档池的覆盖度

    这也是一个经典的子模函数 (Facility Location)。
    加入 η 系数可以防止 KB 过度聚焦于某个特定话题。

    参数:
        selected_embs: (|S|, d) 已选文档 embedding
        pool_embs:     (|D_pool|, d) 文档池所有 embedding
    """
    if selected_embs.shape[0] == 0 or pool_embs.shape[0] == 0:
        return 0.0

    sim_matrix = pool_embs @ selected_embs.T
    max_sims = sim_matrix.max(axis=1)

    return float(max_sims.mean())


def submodular_objective(
    selected_embs: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
    pool_embs: np.ndarray,
    eta: float = 0.1,
) -> float:
    """
    组合子模目标函数:
      f(S) = f_interest(S) + η · f_diversity(S)

    η 的作用:
      - η = 0:    纯兴趣匹配 (Phase 1 Explore 用这个，快速对齐兴趣)
      - η = 0.1:  兴趣 + 多样性 (Phase 2 Exploit 用这个，保持探索)
      - η = 1.0:  纯多样性 (Phase 0 Bootstrap 用这个，最大覆盖)

    因为 f_int 和 f_div 都是子模的，线性组合仍然是子模的，
    所以贪心算法仍然保证 (1-1/e) 近似比。
    """
    f_int = _interest_coverage(selected_embs, centroids, weights)
    f_div = _diversity_coverage(selected_embs, pool_embs) if eta > 0 else 0.0
    return f_int + eta * f_div


# ============================================================
# 贪心子模最大化
# ============================================================

def greedy_submodular_select(
    candidate_docs: List[Document],
    centroids: np.ndarray,
    weights: np.ndarray,
    pool_embs: np.ndarray,
    budget: int,
    eta: float = 0.1,
) -> List[Document]:
    """
    标准贪心子模最大化 — 每步加入边际增益最大的文档。

    算法:
      S = ∅
      for step in 1..budget:
          d* = argmax_{d∈C\\S} [f(S ∪ {d}) - f(S)]   # 边际增益最大化
          S = S ∪ {d*}

    理论保证: 单调子模函数下，贪心算法得到 ≥ (1-1/e) ≈ 0.632 的最优近似。
    复杂度: O(budget × |candidates|)，每步需要遍历所有候选计算边际增益。

    参数:
        candidate_docs: 候选文档列表
        centroids:      (m, d) 兴趣簇中心
        weights:        (m,) 兴趣权重
        pool_embs:      (|D_pool|, d) 文档池 embedding (多样性项用)
        budget:         最多选择的文档数
        eta:            多样性正则化系数

    返回:
        选中的文档列表 (最多 budget 篇)
    """
    if not candidate_docs:
        return []

    budget = min(budget, len(candidate_docs))

    # 构建候选 embedding 矩阵
    cand_embs = np.vstack([d.embedding for d in candidate_docs])
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    cand_embs = cand_embs / np.clip(cand_norms, 1e-10, None)

    selected_indices: List[int] = []
    remaining = set(range(len(candidate_docs)))

    for step in range(budget):
        best_gain = -np.inf
        best_idx = -1

        # 当前已选子集的 embedding
        if selected_indices:
            sel_embs = cand_embs[selected_indices]
        else:
            sel_embs = np.empty((0, cand_embs.shape[1]))

        current_val = submodular_objective(sel_embs, centroids, weights, pool_embs, eta)

        for idx in remaining:
            # 计算添加候选 d 后的边际增益
            new_sel_embs = np.vstack([sel_embs, cand_embs[idx:idx+1]]) if sel_embs.shape[0] > 0 else cand_embs[idx:idx+1]
            new_val = submodular_objective(new_sel_embs, centroids, weights, pool_embs, eta)
            gain = new_val - current_val

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx < 0 or best_gain <= 0:
            break  # 所有候选的边际增益 ≤ 0，提前终止

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

        if step < 3 or step == budget - 1:
            logger.debug(
                f"  贪心第 {step+1}/{budget} 步: "
                f"选入 doc={candidate_docs[best_idx].doc_id}, "
                f"边际增益={best_gain:.4f}"
            )

    return [candidate_docs[i] for i in selected_indices]


# ============================================================
# KB 策展器 (KB Curator)
# ============================================================

class QARCKBCurator:
    """
    QARC 知识库策展器 — 管理动态 KB。

    核心思想: KB 是从文档池 D_pool 中选出的子集，
    随用户兴趣变化而动态调整。

    主要接口:
      - bootstrap_diversity():     Phase 0 多样性最大化初始化
      - bootstrap_from_queries():  有历史查询时的热启动
      - recurate():                兴趣驱动的子模选择 + 增量替换
      - retrieve():                从当前 KB 中检索文档 (供 RAG 使用)

    与 ERASE 的区别:
      - ERASE 修改事实内容 (f_j 的文本和真假状态)
      - QARC 切换文档集合 (哪些文档在 KB 中)
      两者可以配合: QARC 选文档 → ERASE 检查新文档的事实一致性
    """

    def __init__(
        self,
        document_pool: DocumentPool,
        kb_budget: int = 50,
        lambda_max: float = 0.2,
        candidate_top_k: int = 100,
        erase_check_fn: Optional[Callable] = None,
    ):
        """
        参数:
            document_pool:   完整文档池 D_pool
            kb_budget:       KB 最大容量 (B)
            lambda_max:      单次重新策展的最大替换比例 (默认 0.2)
            candidate_top_k: 每个兴趣中心检索的候选文档数
            erase_check_fn:  可选的 ERASE 风格一致性检查回调
                             签名: fn(doc: Document, kb_docs: List[Document]) -> None
        """
        self.pool = document_pool
        self.kb_budget = kb_budget
        self.lambda_max = lambda_max
        self.candidate_top_k = candidate_top_k
        self.erase_check_fn = erase_check_fn

        # 当前 KB 状态: doc_id → Document
        self.kb_docs: Dict[str, Document] = {}

        # 统计
        self.recuration_count = 0

    @property
    def kb_size(self) -> int:
        return len(self.kb_docs)

    def get_kb_embeddings(self) -> np.ndarray:
        """返回当前 KB 所有文档的 (n, d) embedding 矩阵"""
        if not self.kb_docs:
            return np.empty((0, 0))
        embs = [doc.embedding for doc in self.kb_docs.values()]
        mat = np.vstack(embs)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-10, None)

    def get_kb_doc_ids(self) -> Set[str]:
        return set(self.kb_docs.keys())

    def get_kb_docs_list(self) -> List[Document]:
        return list(self.kb_docs.values())

    # -------------------------------------------------------
    # Phase 0: Bootstrap — 多样性最大化初始化
    # -------------------------------------------------------

    def bootstrap_diversity(self) -> List[Document]:
        """
        Phase 0: 用多样性最大化从文档池初始化 KB。

        在还没有任何用户查询时使用。
        目标: 让 KB 覆盖尽可能多的话题。

        使用 Facility Location 子模函数:
          f_div(S) = Σ_{d∈D_pool} max_{d'∈S} CosSim(d, d')

        直觉: 每加一篇文档，让它"代表"尽可能多的未被覆盖的文档池文档。

        返回:
            选中的文档列表
        """
        logger.info(
            f"Phase 0 Bootstrap: 从文档池 (大小={self.pool.size}) "
            f"选择 {self.kb_budget} 篇文档"
        )

        if self.pool.size == 0:
            logger.warning("文档池为空 — 无法初始化 KB")
            return []

        pool_embs = self.pool.get_all_embeddings()
        pool_ids = self.pool.get_all_ids()
        all_docs = [self.pool.documents[did] for did in pool_ids]
        budget = min(self.kb_budget, len(all_docs))

        # 贪心 Facility Location 选择
        selected = self._greedy_facility_location(all_docs, pool_embs, budget)

        self.kb_docs.clear()
        for doc in selected:
            self.kb_docs[doc.doc_id] = doc

        logger.info(f"Phase 0 Bootstrap 完成: KB 大小={self.kb_size}")
        return selected

    def bootstrap_from_queries(
        self,
        query_embeddings: np.ndarray,
        centroids: np.ndarray,
        weights: np.ndarray,
        eta: float = 0.05,
    ) -> List[Document]:
        """
        热启动: 利用历史查询日志初始化 KB。

        与纯多样性不同，这里使用已知的兴趣模型来选择文档，
        使得初始 KB 就已经较好地对齐了用户兴趣。

        参数:
            query_embeddings: 历史查询 embedding
            centroids:        兴趣中心
            weights:          兴趣权重
            eta:              小的多样性项 (防止过度聚焦)
        """
        logger.info(
            f"热启动: 使用 {len(centroids)} 个历史兴趣簇选择 {self.kb_budget} 篇文档"
        )

        candidates = self._gather_candidates(centroids)
        pool_embs = self.pool.get_all_embeddings()

        selected = greedy_submodular_select(
            candidate_docs=candidates,
            centroids=centroids,
            weights=weights,
            pool_embs=pool_embs,
            budget=self.kb_budget,
            eta=eta,
        )

        self.kb_docs.clear()
        for doc in selected:
            self.kb_docs[doc.doc_id] = doc

        logger.info(f"热启动完成: KB 大小={self.kb_size}")
        return selected

    # -------------------------------------------------------
    # ReCurate: 兴趣加权子模选择 + 增量替换
    # -------------------------------------------------------

    def recurate(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        lambda_max: Optional[float] = None,
        eta: float = 0.1,
    ) -> CurationResult:
        """
        基于当前兴趣模型重新策展 KB — QARC 的核心操作。

        四个步骤:
          A. 候选检索: 用每个兴趣中心去文档池中检索相似文档
          B. 子模选择: 贪心最大化 f(S) = f_interest(S) + η·f_diversity(S)
          C. 增量替换: K_ideal vs K_old 的差集，受 λ_max 限制
          D. ERASE 一致性检查: 对新加入的文档做事实核查 (可选)

        增量替换的设计意图:
          - 不能一次换掉太多文档，否则 RAG 系统的回答质量会突然变化
          - 通过 λ_max 控制每次最多替换 KB 的 20% (Phase 2) 或 50% (Phase 1)
          - 优先添加边际增益最大的文档，优先移除与当前兴趣最不相关的文档

        参数:
            centroids:  (m, d) AutoKMeans 输出的兴趣簇中心
            weights:    (m,) 兴趣权重
            lambda_max: 替换上限 (默认用 self.lambda_max)
            eta:        多样性正则化系数

        返回:
            CurationResult 包含添加/移除的文档 ID 和目标函数变化
        """
        self.recuration_count += 1
        lam = lambda_max if lambda_max is not None else self.lambda_max

        logger.info(
            f"ReCurate #{self.recuration_count}: "
            f"{len(centroids)} 个兴趣簇, λ_max={lam:.2f}, η={eta:.2f}"
        )

        # === A. 候选检索 ===
        # 每个兴趣中心去文档池检索 top-k 相似文档
        candidates = self._gather_candidates(centroids)
        logger.info(f"  候选检索: 共 {len(candidates)} 篇不重复候选文档")

        if not candidates:
            return CurationResult(
                added_ids=[], removed_ids=[],
                kb_size=self.kb_size,
                objective_before=0.0, objective_after=0.0,
                replacement_ratio=0.0,
            )

        # === B. 子模选择 ===
        # 贪心最大化: 选出"理想 KB"
        pool_embs = self.pool.get_all_embeddings()
        budget = max(self.kb_budget, self.kb_size)

        k_ideal = greedy_submodular_select(
            candidate_docs=candidates,
            centroids=centroids,
            weights=weights,
            pool_embs=pool_embs,
            budget=budget,
            eta=eta,
        )

        ideal_ids = {d.doc_id for d in k_ideal}
        current_ids = self.get_kb_doc_ids()

        # === C. 增量替换 ===
        # 计算需要添加和移除的差集
        to_add_ids = ideal_ids - current_ids     # 理想中有但当前没有的
        to_remove_ids = current_ids - ideal_ids  # 当前有但理想中没有的

        # 限制最大变化量
        max_changes = max(1, int(lam * self.kb_size)) if self.kb_size > 0 else len(to_add_ids)

        # 如果需要添加的超过限额 → 按边际增益排序，取 top-max_changes
        if len(to_add_ids) > max_changes:
            add_candidates = [d for d in k_ideal if d.doc_id in to_add_ids]
            add_candidates = self._rank_by_marginal_gain(
                add_candidates, centroids, weights, pool_embs, eta
            )
            to_add_ids = {d.doc_id for d in add_candidates[:max_changes]}

        # 如果需要移除的超过限额 → 按兴趣相关度升序，移除最不相关的
        if len(to_remove_ids) > max_changes:
            remove_candidates = [self.kb_docs[did] for did in to_remove_ids]
            remove_candidates = self._rank_by_interest_score(
                remove_candidates, centroids, weights
            )
            to_remove_ids = {d.doc_id for d in remove_candidates[:max_changes]}

        # 平衡: 不移除超过添加数量的文档 (保持 KB 大小稳定)
        n_add = len(to_add_ids)
        n_remove = min(len(to_remove_ids), n_add + max(0, self.kb_size - self.kb_budget))

        if n_remove < len(to_remove_ids):
            remove_candidates = [self.kb_docs[did] for did in to_remove_ids]
            remove_candidates = self._rank_by_interest_score(
                remove_candidates, centroids, weights
            )
            to_remove_ids = {d.doc_id for d in remove_candidates[:n_remove]}

        # 计算替换前的目标函数值
        obj_before = submodular_objective(
            self.get_kb_embeddings(), centroids, weights, pool_embs, eta
        ) if self.kb_size > 0 else 0.0

        # === D. 执行替换 + ERASE 一致性检查 ===
        for did in to_remove_ids:
            del self.kb_docs[did]

        for doc in k_ideal:
            if doc.doc_id in to_add_ids:
                self.kb_docs[doc.doc_id] = doc

                # 对新加入的文档执行 ERASE 风格的事实核查 (可选)
                if self.erase_check_fn is not None:
                    try:
                        self.erase_check_fn(doc, self.get_kb_docs_list())
                    except Exception as e:
                        logger.warning(f"ERASE 一致性检查失败 [{doc.doc_id}]: {e}")

        # 计算替换后的目标函数值
        obj_after = submodular_objective(
            self.get_kb_embeddings(), centroids, weights, pool_embs, eta
        ) if self.kb_size > 0 else 0.0

        replacement_ratio = len(to_add_ids) / max(self.kb_size, 1)

        result = CurationResult(
            added_ids=list(to_add_ids),
            removed_ids=list(to_remove_ids),
            kb_size=self.kb_size,
            objective_before=obj_before,
            objective_after=obj_after,
            replacement_ratio=replacement_ratio,
        )

        logger.info(
            f"  ReCurate 完成: +{len(to_add_ids)} / -{len(to_remove_ids)} 篇文档, "
            f"KB 大小={self.kb_size}, "
            f"目标函数 {obj_before:.4f} → {obj_after:.4f}"
        )

        return result

    # -------------------------------------------------------
    # 从 KB 检索 (供 RAG 使用)
    # -------------------------------------------------------

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        从当前 KB 中检索 top-k 文档 — RAG 的检索步骤。

        注意: 这里只在 KB (小型) 中检索，不是在整个文档池中检索。
        QARC 的思想: 先通过策展让 KB 对齐兴趣，然后检索就自然准确了。
        """
        if not self.kb_docs:
            return []

        kb_embs = self.get_kb_embeddings()
        kb_ids = list(self.kb_docs.keys())

        query = query_embedding.reshape(1, -1)
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query = query / qnorm

        sims = (kb_embs @ query.T).flatten()

        k = min(top_k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        results = []
        for i in top_idx:
            did = kb_ids[i]
            results.append((self.kb_docs[did], float(sims[i])))

        return results

    # -------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------

    def _gather_candidates(
        self,
        centroids: np.ndarray,
    ) -> List[Document]:
        """用每个兴趣中心在文档池中检索候选文档（去重）"""
        seen_ids: Set[str] = set()
        candidates: List[Document] = []

        for i, centroid in enumerate(centroids):
            results = self.pool.search(
                query_embedding=centroid,
                top_k=self.candidate_top_k,
                exclude_ids=None,  # 不排除任何文档，包括当前已在 KB 中的
            )
            for doc, sim in results:
                if doc.doc_id not in seen_ids:
                    seen_ids.add(doc.doc_id)
                    candidates.append(doc)

        return candidates

    def _greedy_facility_location(
        self,
        docs: List[Document],
        pool_embs: np.ndarray,
        budget: int,
    ) -> List[Document]:
        """
        贪心 Facility Location — Phase 0 多样性最大化。

        f_div(S) = Σ_{d∈D_pool} max_{d'∈S} CosSim(d, d')

        优化: 维护 current_max[j] = 当前 KB 对文档池第 j 篇文档的最大覆盖度，
        每步只需计算新候选的边际增益: max(0, sim(pool_j, candidate) - current_max[j])
        """
        n_pool = pool_embs.shape[0]
        doc_embs = np.vstack([d.embedding for d in docs])
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs = doc_embs / np.clip(doc_norms, 1e-10, None)

        # 预计算所有相似度: (n_pool, n_docs)
        all_sims = pool_embs @ doc_embs.T

        # 跟踪当前每个文档池文档的最大覆盖度
        current_max = np.full(n_pool, -np.inf)

        selected_indices = []
        remaining = set(range(len(docs)))

        for step in range(budget):
            best_gain = -np.inf
            best_idx = -1

            for idx in remaining:
                # 边际增益 = 新候选能增加多少覆盖度
                gains = np.maximum(0, all_sims[:, idx] - current_max)
                total_gain = gains.sum()

                if total_gain > best_gain:
                    best_gain = total_gain
                    best_idx = idx

            if best_idx < 0 or best_gain <= 0:
                break

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

            # 更新覆盖度
            current_max = np.maximum(current_max, all_sims[:, best_idx])

            if step < 3 or step == budget - 1:
                logger.debug(
                    f"  FacilityLoc 第 {step+1}/{budget} 步: "
                    f"doc={docs[best_idx].doc_id}, 增益={best_gain:.4f}"
                )

        return [docs[i] for i in selected_indices]

    def _rank_by_marginal_gain(
        self,
        docs: List[Document],
        centroids: np.ndarray,
        weights: np.ndarray,
        pool_embs: np.ndarray,
        eta: float,
    ) -> List[Document]:
        """按边际增益降序排列文档 — 用于限额时选择最有价值的添加"""
        if not docs:
            return []

        current_embs = self.get_kb_embeddings()
        base_val = submodular_objective(current_embs, centroids, weights, pool_embs, eta)

        gains = []
        for doc in docs:
            doc_emb = doc.embedding.reshape(1, -1)
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm > 1e-10:
                doc_emb = doc_emb / doc_norm
            augmented = np.vstack([current_embs, doc_emb]) if current_embs.shape[0] > 0 else doc_emb
            new_val = submodular_objective(augmented, centroids, weights, pool_embs, eta)
            gains.append((doc, new_val - base_val))

        gains.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in gains]

    def _rank_by_interest_score(
        self,
        docs: List[Document],
        centroids: np.ndarray,
        weights: np.ndarray,
    ) -> List[Document]:
        """
        按兴趣相关度升序排列文档 — 用于限额时选择最不相关的移除。

        score(d) = Σ α_i · CosSim(c_i, d)  
        分数低 = 与当前兴趣最不相关 → 优先移除
        """
        if not docs:
            return []

        scores = []
        for doc in docs:
            emb = doc.embedding.reshape(1, -1)
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 1e-10:
                emb = emb / emb_norm
            sims = (centroids @ emb.T).flatten()
            score = float((weights * sims).sum())
            scores.append((doc, score))

        # 升序: 最不相关的排在前面 (便于 [:n] 取最不相关的)
        scores.sort(key=lambda x: x[1])
        return [d for d, _ in scores]

    def get_statistics(self) -> Dict[str, Any]:
        """返回策展器统计信息"""
        return {
            "kb_size": self.kb_size,
            "kb_budget": self.kb_budget,
            "pool_size": self.pool.size,
            "recuration_count": self.recuration_count,
            "lambda_max": self.lambda_max,
            "kb_doc_ids": list(self.kb_docs.keys()),
        }
