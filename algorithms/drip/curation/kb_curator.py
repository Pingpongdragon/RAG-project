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

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from algorithms.qarc.interfaces import BaseKBCurator


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

    规模自适应索引策略:
      - pool ≤ 50k docs: FAISS IndexFlatIP（精确内积，快速）
      - pool  > 50k docs: FAISS IndexIVFFlat（ANN，nlist=√N，nprobe=nlist//4）
        构建时间 O(N)，查询时间 O(nprobe × N/nlist)，约快 20-50×

    文档池 D_pool 通常很大，
    而 KB K 只从中选取一小部分
    """

    # pool 大小超过此阈值，自动切换到 IVF-ANN 索引
    _IVF_THRESHOLD = 50_000

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self._id_list: List[str] = []
        self._faiss_index = None   # FAISS 索引（懒建）
        self._dirty = True

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
        """重建 FAISS 索引（懒加载，仅在脏标记时重建）"""
        if not self._dirty:
            return
        self._id_list = list(self.documents.keys())
        self._faiss_index = None

        if not self._id_list:
            self._dirty = False
            return

        embs = np.vstack([self.documents[did].embedding for did in self._id_list]).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-10, None)

        n, d = embs.shape

        if _FAISS_AVAILABLE:
            if n <= self._IVF_THRESHOLD:
                # 精确 Flat 索引（内积 ≡ 余弦相似度，因为向量已归一化）
                idx = faiss.IndexFlatIP(d)
            else:
                # IVF-ANN：nlist ≈ √N，nprobe = max(1, nlist//4)
                nlist = max(8, int(n ** 0.5))
                nprobe = max(1, nlist // 4)
                quantizer = faiss.IndexFlatIP(d)
                idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                idx.train(embs)
                idx.nprobe = nprobe
                logger.info(f"DocumentPool: IVF index built (n={n}, nlist={nlist}, nprobe={nprobe})")
            idx.add(embs)
            self._faiss_index = idx
        else:
            # fallback: 保存矩阵用于暴力搜索
            self._faiss_index = embs  # np.ndarray fallback

        self._dirty = False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        检索与查询最相似的 top-k 文档。

        使用 FAISS 索引（IVF-ANN 或 Flat），支持大规模 pool。
        exclude_ids: 后处理过滤（不影响 ANN 精度），适合少量排除。
        """
        self._rebuild_index()

        if not self._id_list or self._faiss_index is None:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query = query / qnorm

        # 多取一些候选以应对 exclude_ids 过滤
        fetch_k = top_k + len(exclude_ids) if exclude_ids else top_k
        fetch_k = min(fetch_k, len(self._id_list))

        if _FAISS_AVAILABLE and not isinstance(self._faiss_index, np.ndarray):
            sims_arr, idx_arr = self._faiss_index.search(query, fetch_k)
            sims_arr = sims_arr[0]
            idx_arr  = idx_arr[0]
        else:
            # numpy fallback
            mat = self._faiss_index if isinstance(self._faiss_index, np.ndarray) else None
            if mat is None:
                return []
            sims_all = (mat @ query.T).flatten()
            k = min(fetch_k, len(sims_all))
            raw_idx = np.argpartition(sims_all, -k)[-k:]
            raw_idx = raw_idx[np.argsort(sims_all[raw_idx])[::-1]]
            sims_arr = sims_all[raw_idx]
            idx_arr  = raw_idx

        results = []
        for sim, i in zip(sims_arr, idx_arr):
            if i < 0:
                continue
            did = self._id_list[i]
            if exclude_ids and did in exclude_ids:
                continue
            results.append((self.documents[did], float(sim)))
            if len(results) >= top_k:
                break

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """返回所有文档的 (n, d) embedding 矩阵（只读视图，不复制）"""
        self._rebuild_index()
        if not self._id_list:
            return np.empty((0, 0))
        if _FAISS_AVAILABLE and not isinstance(self._faiss_index, np.ndarray):
            # 从 FAISS 重建矩阵（用于子模计算，此时 pool 通常不超大）
            n = len(self._id_list)
            d = self._faiss_index.d
            mat = np.zeros((n, d), dtype=np.float32)
            self._faiss_index.reconstruct_n(0, n, mat)
            return mat
        elif isinstance(self._faiss_index, np.ndarray):
            return self._faiss_index  # numpy fallback，直接共享（只读）
        return np.empty((0, 0))

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
) -> float:
    """
    纯兴趣驱动子模目标函数:
      f(S) = f_interest(S)

    贪心算法保证 (1-1/e) 近似比。
    """
    return _interest_coverage(selected_embs, centroids, weights)


# ============================================================
# 贪心子模最大化
# ============================================================

def greedy_submodular_select(
    candidate_docs: List[Document],
    centroids: np.ndarray,
    weights: np.ndarray,
    budget: int,
) -> List[Document]:
    """
    增量贪心子模最大化 — 用增量边际增益替代全量重算。

    性能优化 (相比朴素版本):
      - 兴趣项: 维护 max_interest_sim[i] (每个簇的当前最大覆盖),
        边际增益 = Σ α_i · max(0, sim(c_i, d_new) - max_interest_sim[i])
      - 多样性项: 维护 max_div_coverage[j] (每个 pool 文档的当前最大覆盖),
        边际增益 = (1/pool_size) · Σ max(0, sim(pool_j, d_new) - max_div_coverage[j])
      - 复杂度: O(budget × candidates × (m + pool_size))
        相比朴素版 O(budget × candidates × |S| × pool_size) 快 100-200 倍

    理论保证: 单调子模函数下，贪心算法得到 ≥ (1-1/e) ≈ 0.632 的最优近似。

    参数:
        candidate_docs: 候选文档列表
        centroids:      (m, d) 兴趣簇中心
        weights:        (m,) 兴趣权重
        budget:         最多选择的文档数

    返回:
        选中的文档列表 (最多 budget 篇)
    """
    if not candidate_docs:
        return []

    budget = min(budget, len(candidate_docs))
    n_cand = len(candidate_docs)
    m = centroids.shape[0]  # 兴趣簇数

    # 构建候选 embedding 矩阵 (n_cand, dim)
    cand_embs = np.vstack([d.embedding for d in candidate_docs])
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    cand_embs = cand_embs / np.clip(cand_norms, 1e-10, None)

    # 预计算: 兴趣中心 vs 候选 (m, n_cand)
    interest_sims = centroids @ cand_embs.T

    # 增量状态
    max_interest_sim = np.full(m, -np.inf)       # 每个兴趣簇的当前最大覆盖

    selected_indices: List[int] = []
    remaining = set(range(n_cand))

    import heapq as _heapq

    # 转置为行优先访问
    interest_sims_T = interest_sims.T  # (n_cand, m)

    def _compute_gain(idx):
        int_deltas = np.maximum(0, interest_sims_T[idx] - max_interest_sim)
        return float((weights * int_deltas).sum())

    # 初始化堆
    heap = [(-_compute_gain(i), i, -1) for i in range(n_cand)]
    _heapq.heapify(heap)
    selected_set = set()

    for step in range(budget):
        best_idx = -1
        best_gain = -1.0

        while heap:
            neg_gain, idx, last_step = _heapq.heappop(heap)
            if idx in selected_set:
                continue
            if last_step == step:
                best_idx = idx
                best_gain = -neg_gain
                break
            gain = _compute_gain(idx)
            _heapq.heappush(heap, (-gain, idx, step))

        if best_idx < 0 or best_gain <= 0:
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)

        max_interest_sim = np.maximum(max_interest_sim, interest_sims_T[best_idx])

        if step % 500 == 0 or step == budget - 1:
            logger.debug(
                f"  贪心第 {step+1}/{budget} 步: "
                f"选入 doc={candidate_docs[best_idx].doc_id}, "
                f"边际增益={best_gain:.4f}"
            )

    return [candidate_docs[i] for i in selected_indices]



# ============================================================
# KB 策展器 (KB Curator)
# ============================================================

class QARCKBCurator(BaseKBCurator):
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

        # FAISS KB 索引缓存（增量维护，避免每次 retrieve 重建）
        self._kb_index = None   # FAISS IndexFlatIP 或 None
        self._kb_index_ids: List[str] = []   # 与索引行号对应的 doc_id

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
    ) -> List[Document]:
        """
        热启动: 利用历史查询日志初始化 KB (纯兴趣驱动)。

        参数:
            query_embeddings: 历史查询 embedding
            centroids:        兴趣中心
            weights:          兴趣权重
        """
        logger.info(
            f"热启动: 使用 {len(centroids)} 个历史兴趣簇选择 {self.kb_budget} 篇文档"
        )

        candidates = self._gather_candidates(centroids)

        selected = greedy_submodular_select(
            candidate_docs=candidates,
            centroids=centroids,
            weights=weights,
            budget=self.kb_budget,
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
    ) -> CurationResult:
        """
        基于当前兴趣模型重新策展 KB — QARC 的核心操作 (纯兴趣驱动)。

        三个步骤:
          A. 候选检索: 用每个兴趣中心去文档池中检索相似文档
          B. 子模选择: 贪心最大化 f(S) = f_interest(S)
          C. 增量替换: K_ideal vs K_old 的差集，受 λ_max 限制

        参数:
            centroids:  (m, d) AutoKMeans 输出的兴趣簇中心
            weights:    (m,) 兴趣权重
            lambda_max: 替换上限 (默认用 self.lambda_max)

        返回:
            CurationResult 包含添加/移除的文档 ID 和目标函数变化
        """
        self.recuration_count += 1
        lam = lambda_max if lambda_max is not None else self.lambda_max

        logger.info(
            f"ReCurate #{self.recuration_count}: "
            f"{len(centroids)} 个兴趣簇, λ_max={lam:.2f}"
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
        # 贪心最大化: 选出"理想 KB" (纯兴趣驱动)
        budget = max(self.kb_budget, self.kb_size)

        k_ideal = greedy_submodular_select(
            candidate_docs=candidates,
            centroids=centroids,
            weights=weights,
            budget=budget,
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
                add_candidates, centroids, weights
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
            self.get_kb_embeddings(), centroids, weights
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

        # KB 变化 → 使检索索引缓存失效（下次 retrieve 时重建）
        if to_add_ids or to_remove_ids:
            self._invalidate_kb_index()

        # 计算替换后的目标函数值
        obj_after = submodular_objective(
            self.get_kb_embeddings(), centroids, weights
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

    def _invalidate_kb_index(self):
        """KB 内容变化后调用，使缓存索引失效。"""
        self._kb_index = None
        self._kb_index_ids = []

    def _rebuild_kb_index(self):
        """懒加载：构建/更新 FAISS KB 索引。
        
        KB 通常几千到几万条，使用 IndexFlatIP（精确搜索，O(N_kb × d)）。
        相比每次 retrieve 都重新 np.vstack + matmul，这里只在 KB 变化时重建一次。
        """
        if self._kb_index is not None:
            return  # 未失效，直接复用

        self._kb_index_ids = list(self.kb_docs.keys())
        if not self._kb_index_ids:
            self._kb_index = None
            return

        embs = np.vstack([self.kb_docs[did].embedding for did in self._kb_index_ids]).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-10, None)

        if _FAISS_AVAILABLE:
            d = embs.shape[1]
            idx = faiss.IndexFlatIP(d)
            idx.add(embs)
            self._kb_index = idx
        else:
            self._kb_index = embs  # numpy fallback

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        从当前 KB 中检索 top-k 文档 — RAG 的检索步骤。

        使用增量维护的 FAISS IndexFlatIP 缓存，KB 不变时无需重建索引。
        支持 KB 规模从几百到几万条，每次查询 O(N_kb × d) → 实际∼1ms@50k。
        """
        if not self.kb_docs:
            return []

        self._rebuild_kb_index()

        if self._kb_index is None:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        qnorm = np.linalg.norm(query)
        if qnorm > 1e-10:
            query = query / qnorm

        k = min(top_k, len(self._kb_index_ids))

        if _FAISS_AVAILABLE and not isinstance(self._kb_index, np.ndarray):
            sims_arr, idx_arr = self._kb_index.search(query, k)
            sims_arr = sims_arr[0]
            idx_arr  = idx_arr[0]
        else:
            mat = self._kb_index
            sims_all = (mat @ query.T).flatten()
            raw_idx = np.argpartition(sims_all, -k)[-k:]
            raw_idx = raw_idx[np.argsort(sims_all[raw_idx])[::-1]]
            sims_arr = sims_all[raw_idx]
            idx_arr  = raw_idx

        results = []
        for sim, i in zip(sims_arr, idx_arr):
            if i < 0:
                continue
            did = self._kb_index_ids[i]
            results.append((self.kb_docs[did], float(sim)))

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
        使用 lazy greedy 加速 + 行优先访问优化缓存命中率。
        """
        import heapq

        n_pool = pool_embs.shape[0]
        n_docs = len(docs)
        doc_embs = np.vstack([d.embedding for d in docs])
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_embs = doc_embs / np.clip(doc_norms, 1e-10, None)

        if n_docs <= budget:
            return list(docs)

        # (n_docs, n_pool) 行优先，缓存友好
        logger.info(f"  FacilityLoc: 计算相似度矩阵 ({n_docs}, {n_pool})...")
        all_sims_T = doc_embs @ pool_embs.T  # (n_docs, n_pool)

        current_max = np.full(n_pool, -np.inf)
        selected_indices = []
        selected_set = set()

        # 初始化: 批量计算所有候选的初始增益
        logger.info(f"  FacilityLoc: 计算初始增益...")
        chunk_sz = 2000
        init_gains = np.empty(n_docs, dtype=np.float64)
        for cs in range(0, n_docs, chunk_sz):
            ce = min(cs + chunk_sz, n_docs)
            marginals = np.maximum(0, all_sims_T[cs:ce] - current_max[None, :])
            init_gains[cs:ce] = marginals.sum(axis=1)

        # 惰性贪心堆: (-gain, idx, last_step)
        heap = [(-init_gains[i], i, -1) for i in range(n_docs)]
        heapq.heapify(heap)

        logger.info(f"  FacilityLoc: 开始贪心选择 {budget} 篇文档...")
        for step in range(budget):
            best_idx = -1
            best_gain = -1.0

            while heap:
                neg_gain, idx, last_step = heapq.heappop(heap)
                if idx in selected_set:
                    continue
                if last_step == step:
                    best_idx = idx
                    best_gain = -neg_gain
                    break
                # 重新计算增益
                gain = float(np.maximum(0, all_sims_T[idx] - current_max).sum())
                heapq.heappush(heap, (-gain, idx, step))

            if best_idx < 0 or best_gain <= 0:
                break

            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            current_max = np.maximum(current_max, all_sims_T[best_idx])

            if step % 500 == 0 or step == budget - 1:
                logger.info(
                    f"  FacilityLoc 第 {step+1}/{budget} 步: "
                    f"doc={docs[best_idx].doc_id}, 增益={best_gain:.4f}"
                )

        return [docs[i] for i in selected_indices]

    def _rank_by_marginal_gain(
        self,
        docs: List[Document],
        centroids: np.ndarray,
        weights: np.ndarray,
    ) -> List[Document]:
        """按边际增益降序排列文档 — 用于限额时选择最有价值的添加 (纯兴趣)"""
        if not docs:
            return []

        current_embs = self.get_kb_embeddings()
        base_val = submodular_objective(current_embs, centroids, weights)

        gains = []
        for doc in docs:
            doc_emb = doc.embedding.reshape(1, -1)
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm > 1e-10:
                doc_emb = doc_emb / doc_norm
            augmented = np.vstack([current_embs, doc_emb]) if current_embs.shape[0] > 0 else doc_emb
            new_val = submodular_objective(augmented, centroids, weights)
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
