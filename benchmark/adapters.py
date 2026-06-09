"""
benchmark/adapters.py — algorithms/ 三个方法的 KBUpdateStrategy 适配器

每个适配器将对应方法的内部逻辑适配到统一的 base.KBUpdateStrategy 接口，
使得 run_experiments.py 可以公平对比 QARC / ComRAG / ERASE / Static / Random

适配策略:
  - QARC:   内部用 QARCPipeline (兴趣建模 + submodular KB curation + 三阶段生命周期)
  - ComRAG: 内部用 DynamicMemory (双向量库 + 质心路由)
  - ERASE:  内部用 ERASEKnowledgeBase (事实级编辑)
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


# ============================================================
# ComRAG 适配器
# ============================================================

class ComRAGStrategyAdapter(KBUpdateStrategy):
    """
    ComRAG — 对话式 RAG 动态记忆 (ACL 2025)

    核心流程:
      1. 维护双向量库: V_high (高分 QA 对) + V_low (低分 QA 对)
      2. 每次查询: 三层路由 → direct_reuse / reference_generation / kb_avoidance
      3. 回答后根据评分路由到对应向量库
      4. 自适应温度 T(Δ) = exp(-k · min_gap) 调节路由灵敏度
    """

    def __init__(self, kb_budget: int = 50):
        self._kb_budget = kb_budget
        self._metrics = MethodMetrics()
        self._doc_pool = []
        self._doc_embeddings = None
        self._pool_id_to_idx = {}
        self._kb_doc_ids: Set[str] = set()
        self._record_to_docid: Dict[str, str] = {}  # 增量维护: record_id → doc_id

        # ComRAG 组件 (lazy init)
        self._memory = None

    @property
    def name(self) -> str:
        return "ComRAG"

    def initialize(self, doc_pool, doc_embeddings, kb_budget):
        self._doc_pool = doc_pool
        self._doc_embeddings = doc_embeddings
        self._kb_budget = kb_budget
        self._pool_id_to_idx = {d["doc_id"]: i for i, d in enumerate(doc_pool)}

        # 构建 FAISS 索引用于快速 embedding → doc_id 映射
        import faiss
        d = doc_embeddings.shape[1]
        embs = np.ascontiguousarray(doc_embeddings, dtype=np.float32)
        faiss.normalize_L2(embs)
        self._pool_index = faiss.IndexFlatIP(d)
        self._pool_index.add(embs)
        self._pool_embs_normed = embs
        self._kb_ids_dirty = True  # 标记缓存是否需要刷新

        from algorithms.comrag.memory import DynamicMemory

        self._memory = DynamicMemory(tau=0.75, delta=0.9, gamma=0.6)

        # 基于 embedding 多样性选择冷启动文档
        init_ids = select_diverse_initial_kb(doc_pool, doc_embeddings, kb_budget)
        init_docs = [d for d in doc_pool if d["doc_id"] in init_ids]
        for d in init_docs:
            idx = self._pool_id_to_idx[d["doc_id"]]
            result = self._memory.add(
                question=d.get("text", "")[:100],
                answer=d.get("text", ""),
                embedding=doc_embeddings[idx],
                score=0.7,  # 高于 gamma=0.6 → 进入 V_high
            )

        # 批量初始化 record→docid 映射
        self._build_record_docid_mapping()

    def process_query(self, query_text, query_embedding, step, gold_doc_ids=None):
        if self._memory is None:
            return ProcessResult(kb_size=0)

        # 用 route_query 做三层路由决策
        route_result = self._memory.route_query(query_embedding)
        strategy = route_result.get("strategy", "kb_avoidance")

        # 从 V_high 和 V_low 检索
        results_high = self._memory.high_store.search_centroid_first(query_embedding, top_k=5)
        results_low = self._memory.low_store.search_centroid_first(query_embedding, top_k=5)

        # 合并检索结果, 映射回 doc_pool 的 doc_id (FAISS 快速 NN)
        all_results = results_high + results_low
        all_results.sort(key=lambda r: r.similarity, reverse=True)

        retrieved = []
        seen = set()
        if all_results:
            # 批量查 FAISS 索引
            query_vecs = np.array(
                [r.record.embedding for r in all_results[:10]], dtype=np.float32
            )
            import faiss as _faiss
            _faiss.normalize_L2(query_vecs)
            _, nn_ids = self._pool_index.search(query_vecs, 1)
            for i in range(min(10, len(all_results))):
                best_idx = int(nn_ids[i, 0])
                did = self._doc_pool[best_idx]["doc_id"]
                if did not in seen:
                    retrieved.append(did)
                    seen.add(did)

        # 将当前 query 加入 memory, 根据检索质量动态赋分
        top_sim = all_results[0].similarity if all_results else 0.0
        dynamic_score = max(0.0, min(1.0, top_sim))  # 检索相似度作为质量信号
        add_result = self._memory.add(
            question=query_text,
            answer="",
            embedding=query_embedding,
            score=dynamic_score,
        )

        # 只有真正改变记忆结构时才算更新 (替换旧记录或创建新簇)
        action = add_result.get("action", "skipped")
        update_done = action in ("replaced", "new_cluster")

        # 增量更新 record→docid 映射 (单次 FAISS 查询, 而非全量重建)
        if action == "replaced":
            old_rec = add_result.get("replaced_record")
            if old_rec and old_rec.record_id in self._record_to_docid:
                del self._record_to_docid[old_rec.record_id]
            self._incremental_map_record(query_embedding, add_result)
        elif action in ("added_to_cluster", "new_cluster"):
            self._incremental_map_record(query_embedding, add_result)
        # 从映射重建 kb_doc_ids
        self._kb_doc_ids = set(self._record_to_docid.values())

        self._metrics.total_queries += 1
        if update_done:
            self._metrics.total_updates += 1
        self._metrics.kb_size_history.append(len(self._kb_doc_ids))

        return ProcessResult(
            retrieved_doc_ids=retrieved[:10],
            update_performed=update_done,
            kb_size=len(self._kb_doc_ids),
            extra_metrics={
                "strategy": strategy,
                "action": action,
                "v_high_size": self._memory.high_store.total_records,
                "v_low_size": self._memory.low_store.total_records,
            },
        )

    def get_kb_doc_ids(self):
        return set(self._kb_doc_ids)

    def get_kb_size(self):
        return len(self._kb_doc_ids)

    def _build_record_docid_mapping(self):
        """批量构建 record_id→doc_id 映射 (初始化时调用一次)"""
        self._record_to_docid = {}
        if self._memory is None:
            self._kb_doc_ids = set()
            return
        all_embs = []
        all_rids = []
        for store in [self._memory.high_store, self._memory.low_store]:
            for cluster_records in store.clusters.values():
                for record in cluster_records:
                    all_embs.append(record.embedding)
                    all_rids.append(record.record_id)
        if not all_embs:
            self._kb_doc_ids = set()
            return
        import faiss as _faiss
        query_mat = np.array(all_embs, dtype=np.float32)
        _faiss.normalize_L2(query_mat)
        _, nn_ids = self._pool_index.search(query_mat, 1)
        for i in range(len(all_embs)):
            did = self._doc_pool[int(nn_ids[i, 0])]["doc_id"]
            self._record_to_docid[all_rids[i]] = did
        self._kb_doc_ids = set(self._record_to_docid.values())

    def _incremental_map_record(self, embedding, add_result):
        """增量映射单条新 record → doc_id (单次 FAISS 查询)"""
        import faiss as _faiss
        qv = np.array([embedding], dtype=np.float32)
        _faiss.normalize_L2(qv)
        _, nn_ids = self._pool_index.search(qv, 1)
        did = self._doc_pool[int(nn_ids[0, 0])]["doc_id"]
        # 用时间戳作为临时 record_id
        rid = f"qa_{int(time.time() * 1000)}"
        self._record_to_docid[rid] = did

    def _sync_kb_ids(self):
        """兼容接口: 全量重建 (仅 get_kb_doc_ids 外部调用时用)"""
        self._build_record_docid_mapping()


# ============================================================
# ERASE 适配器
# ============================================================

class ERASEStrategyAdapter(KBUpdateStrategy):
    """
    ERASE — 自一致编辑增强检索 (Li et al., 2024)

    核心流程:
      1. KB 以 fact 为单位存储, 每条 fact 带历史修改记录
      2. 新证据到达时: Retrieve 相关 facts → Update 分类+改写 → Add 新 facts
      3. 不使用 LLM 时退化为基于嵌入的近似编辑
    """

    def __init__(
        self,
        kb_budget: int = 50,
        update_threshold: float = 0.65,
        use_llm: bool = False,
    ):
        self._kb_budget = kb_budget
        self._update_threshold = update_threshold
        self._use_llm = use_llm
        self._metrics = MethodMetrics()
        self._doc_pool = []
        self._doc_embeddings = None
        self._pool_id_to_idx = {}
        self._kb_doc_ids: Set[str] = set()

        self._erase_kb = None
        self._step_counter = 0

    @property
    def name(self) -> str:
        return "ERASE"

    def initialize(self, doc_pool, doc_embeddings, kb_budget):
        self._doc_pool = doc_pool
        self._doc_embeddings = doc_embeddings
        self._kb_budget = kb_budget
        self._pool_id_to_idx = {d["doc_id"]: i for i, d in enumerate(doc_pool)}
        self._step_counter = 0

        # FAISS 索引: 快速 query→pool 检索
        import faiss
        d = doc_embeddings.shape[1]
        embs = np.ascontiguousarray(doc_embeddings, dtype=np.float32)
        faiss.normalize_L2(embs)
        self._pool_index = faiss.IndexFlatIP(d)
        self._pool_index.add(embs)

        from algorithms.erase.knowledge_base import ERASEKnowledgeBase

        self._erase_kb = ERASEKnowledgeBase(similarity_threshold=0.3)

        # 基于 embedding 多样性选择初始 fact 集合
        init_ids = select_diverse_initial_kb(doc_pool, doc_embeddings, kb_budget)
        init_docs = [d for d in doc_pool if d["doc_id"] in init_ids]
        for d in init_docs:
            idx = self._pool_id_to_idx[d["doc_id"]]
            self._erase_kb.add_fact(
                fact=d["text"],
                embedding=doc_embeddings[idx],
                timestamp=f"t0",
                source=d["doc_id"],
            )
        self._sync_kb_ids()

    def process_query(self, query_text, query_embedding, step, gold_doc_ids=None):
        if self._erase_kb is None:
            return ProcessResult(kb_size=0)

        self._step_counter += 1
        ts = f"t{self._step_counter}"

        # ERASE 检索
        results = self._erase_kb.retrieve(
            query_embedding=query_embedding,
            top_k=10,
            threshold=0.0,  # 不过滤, 让所有 fact 都参与排序
        )
        retrieved = [r.entry.source for r in results if r.entry.source]

        # 判断是否需要更新 (基于最高检索相似度)
        update_done = False
        if not results or results[0].similarity < self._update_threshold:
            # 检索质量低 → 从文档池中补充新 facts (FAISS 快速检索)
            import faiss as _faiss
            qv = np.array([query_embedding], dtype=np.float32)
            _faiss.normalize_L2(qv)
            _, top_pool_idx = self._pool_index.search(qv, 5)
            top_pool_idx = top_pool_idx[0]

            added = 0
            for idx in top_pool_idx:
                doc = self._doc_pool[idx]
                if doc["doc_id"] not in self._kb_doc_ids:
                    self._erase_kb.add_fact(
                        fact=doc["text"],
                        embedding=self._doc_embeddings[idx],
                        timestamp=ts,
                        source=doc["doc_id"],
                    )
                    added += 1
                    self._metrics.total_docs_added += 1

            # 超出预算时移除最早加入的 fact
            all_facts = self._erase_kb.get_all_facts()
            while len(all_facts) > int(self._kb_budget * 1.5):
                # 按 fact_id 排序, 移除最早的
                oldest = min(all_facts, key=lambda f: f.fact_id)
                self._erase_kb.remove_fact(oldest.fact_id)
                self._metrics.total_docs_removed += 1
                all_facts = self._erase_kb.get_all_facts()

            update_done = added > 0
            if update_done:
                self._metrics.total_updates += 1

        self._sync_kb_ids()
        self._metrics.total_queries += 1
        self._metrics.kb_size_history.append(len(self._kb_doc_ids))

        return ProcessResult(
            retrieved_doc_ids=retrieved,
            update_performed=update_done,
            kb_size=self._erase_kb.size(),
            extra_metrics={
                "erase_kb_size": self._erase_kb.size(),
            },
        )

    def get_kb_doc_ids(self):
        return set(self._kb_doc_ids)

    def get_kb_size(self):
        return self._erase_kb.size() if self._erase_kb else 0

    def _sync_kb_ids(self):
        if self._erase_kb:
            self._kb_doc_ids = {f.source for f in self._erase_kb.get_all_facts()
                                if f.source}
        else:
            self._kb_doc_ids = set()
