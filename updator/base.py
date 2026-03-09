"""
updator/base.py — 所有 KB 更新策略的抽象基类

设计思想:
  - 统一接口让 QARC / ComRAG / ERASE 可以公平对比
  - 与 core/ RAG 引擎解耦: base 只定义"决策"，执行由 pipeline 负责
  - 支持流式实验: process_query() 逐条处理，内部维护状态

                      ┌───────────────┐
                      │ KBUpdateStrategy│  (抽象基类)
                      │  .initialize() │
                      │  .process_query │
                      │  .get_kb_docs() │
                      └───────┬───────┘
                ┌─────────────┼──────────────┐
           ┌────┴────┐  ┌────┴────┐  ┌──────┴──────┐
           │  QARC   │  │ ComRAG  │  │   ERASE     │
           │Adapter  │  │Adapter  │  │  Adapter    │
           └─────────┘  └─────────┘  └─────────────┘
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ProcessResult:
    """process_query() 的返回值 — 封装一次查询的处理结果"""

    # 本次检索命中的文档 ID
    retrieved_doc_ids: List[str] = field(default_factory=list)

    # 本次是否触发了 KB 更新
    update_performed: bool = False

    # 更新后的 KB 大小
    kb_size: int = 0

    # 方法特有的额外指标 (如: QARC 的 alignment_gap, ComRAG 的 routing 类型)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodMetrics:
    """运行过程中累积的统计指标"""

    total_queries: int = 0
    total_updates: int = 0
    total_docs_added: int = 0
    total_docs_removed: int = 0

    # 每步的 KB 大小
    kb_size_history: List[int] = field(default_factory=list)

    # 方法特有的时间序列指标
    extra_series: Dict[str, List[float]] = field(default_factory=dict)


# ============================================================
# 抽象基类
# ============================================================

class KBUpdateStrategy(ABC):
    """
    所有 KB 更新方法的统一接口

    生命周期:
        1. __init__()        → 设置超参数
        2. initialize()      → 传入文档池和 embedding，构建初始 KB
        3. process_query()    → 逐条处理查询，内部决定是否更新 KB
        4. get_kb_doc_ids()   → 获取当前 KB 中的文档 ID 集合
        5. get_metrics()      → 获取累积统计指标
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称 (用于实验结果表头)"""
        ...

    @abstractmethod
    def initialize(
        self,
        doc_pool: List[Dict],
        doc_embeddings: np.ndarray,
        kb_budget: int,
    ) -> None:
        """
        初始化策略

        Args:
            doc_pool: 候选文档列表, 每条 dict 至少包含 {"doc_id", "text", "topic"}
            doc_embeddings: 形状 (N, D) 的文档嵌入矩阵，与 doc_pool 一一对应
            kb_budget: KB 容量上限 (文档数)
        """
        ...

    @abstractmethod
    def process_query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        step: int,
        gold_doc_ids: Optional[List[str]] = None,
    ) -> ProcessResult:
        """
        处理一条查询

        内部职责:
          1. 用当前 KB 进行检索
          2. 判断是否需要更新 KB (漂移检测/质量评估等)
          3. 如需更新，执行更新
          4. 返回检索结果和更新信息

        Args:
            query_text:      查询文本
            query_embedding: 查询的嵌入向量 shape=(D,)
            step:            当前时间步 (用于窗口/衰减计算)
            gold_doc_ids:    真实相关文档 ID (仅用于 oracle 评估，方法本身不应依赖)

        Returns:
            ProcessResult
        """
        ...

    @abstractmethod
    def get_kb_doc_ids(self) -> Set[str]:
        """返回当前 KB 包含的文档 ID 集合"""
        ...

    @abstractmethod
    def get_kb_size(self) -> int:
        """返回当前 KB 文档数"""
        ...

    def get_metrics(self) -> MethodMetrics:
        """返回累积指标 (子类可覆写以添加方法特有指标)"""
        return getattr(self, "_metrics", MethodMetrics())


# ============================================================
# Baseline: 静态 KB (不更新)
# ============================================================

class StaticKBStrategy(KBUpdateStrategy):
    """
    Baseline — 静态知识库，初始化后永不更新

    用途: 作为下界对比，展示不更新 KB 时性能随 drift 衰退
    """

    def __init__(self):
        self._metrics = MethodMetrics()
        self._kb_doc_ids: Set[str] = set()
        self._doc_pool: List[Dict] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._kb_budget: int = 0
        self._pool_id_to_idx: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "Static"

    def initialize(self, doc_pool, doc_embeddings, kb_budget):
        self._doc_pool = doc_pool
        self._doc_embeddings = doc_embeddings
        self._kb_budget = kb_budget
        self._pool_id_to_idx = {
            d["doc_id"]: i for i, d in enumerate(doc_pool)
        }
        # 取前 kb_budget 条作为初始 KB
        self._kb_doc_ids = {
            d["doc_id"] for d in doc_pool[:kb_budget]
        }

    def process_query(self, query_text, query_embedding, step, gold_doc_ids=None):
        # 在 KB 内检索 top-k
        retrieved = self._retrieve_from_kb(query_embedding, top_k=10)
        self._metrics.total_queries += 1
        self._metrics.kb_size_history.append(len(self._kb_doc_ids))
        return ProcessResult(
            retrieved_doc_ids=retrieved,
            update_performed=False,
            kb_size=len(self._kb_doc_ids),
        )

    def get_kb_doc_ids(self):
        return set(self._kb_doc_ids)

    def get_kb_size(self):
        return len(self._kb_doc_ids)

    def _retrieve_from_kb(self, query_emb, top_k=10):
        """在当前 KB 内做余弦检索"""
        kb_ids = list(self._kb_doc_ids)
        if not kb_ids:
            return []
        indices = [self._pool_id_to_idx[did] for did in kb_ids if did in self._pool_id_to_idx]
        if not indices:
            return []
        kb_embs = self._doc_embeddings[indices]
        sims = kb_embs @ query_emb
        k = min(top_k, len(indices))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [kb_ids[i] for i in top_idx]


# ============================================================
# Baseline: 随机替换 KB
# ============================================================

class RandomKBStrategy(KBUpdateStrategy):
    """
    Baseline — 随机替换知识库

    每隔 update_interval 步，从文档池中随机采样 kb_budget 条替换整个 KB
    用途: 作为随机基线
    """

    def __init__(self, seed: int = 42, update_interval: int = 50):
        self._rng = np.random.RandomState(seed)
        self._update_interval = update_interval
        self._metrics = MethodMetrics()
        self._kb_doc_ids: Set[str] = set()
        self._doc_pool: List[Dict] = []
        self._doc_embeddings: Optional[np.ndarray] = None
        self._kb_budget: int = 0
        self._pool_id_to_idx: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "Random"

    def initialize(self, doc_pool, doc_embeddings, kb_budget):
        self._doc_pool = doc_pool
        self._doc_embeddings = doc_embeddings
        self._kb_budget = kb_budget
        self._pool_id_to_idx = {
            d["doc_id"]: i for i, d in enumerate(doc_pool)
        }
        # 随机初始化
        chosen = self._rng.choice(len(doc_pool), size=min(kb_budget, len(doc_pool)), replace=False)
        self._kb_doc_ids = {doc_pool[i]["doc_id"] for i in chosen}

    def process_query(self, query_text, query_embedding, step, gold_doc_ids=None):
        # 随机替换
        update = (step > 0 and step % self._update_interval == 0)
        if update:
            chosen = self._rng.choice(
                len(self._doc_pool),
                size=min(self._kb_budget, len(self._doc_pool)),
                replace=False,
            )
            self._kb_doc_ids = {self._doc_pool[i]["doc_id"] for i in chosen}
            self._metrics.total_updates += 1

        retrieved = self._retrieve_from_kb(query_embedding, top_k=10)
        self._metrics.total_queries += 1
        self._metrics.kb_size_history.append(len(self._kb_doc_ids))
        return ProcessResult(
            retrieved_doc_ids=retrieved,
            update_performed=update,
            kb_size=len(self._kb_doc_ids),
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
        kb_embs = self._doc_embeddings[indices]
        sims = kb_embs @ query_emb
        k = min(top_k, len(indices))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [kb_ids[i] for i in top_idx]
