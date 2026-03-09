"""
ERASE 可编辑知识库: 支持事实级增删改查 + 时间线追踪

论文: "Language Modeling with Editable External Knowledge" (Li et al., 2024)
链接: https://arxiv.org/abs/2406.11830
简称: ERASE = Enhancing Retrieval Augmentation with Self-consistent Editing

=== 论文核心思想 ===

ERASE 的出发点: 世界知识会随时间变化！
例如 "英国女王是伊丽莎白二世" → 2022年后变成 "英国国王是查尔斯三世"。
传统 RAG 的 KB 是静态的，放进去什么就是什么，不处理知识过时的问题。

ERASE 让 KB 中的每条事实都变成"可编辑的":
  - 每条事实 f_j 附带一个历史记录 H_j = [(时间戳, 真/假), ...]
  - 新信息到来时，通过 LLM 判断已有事实的真假是否发生变化
  - 被推翻的事实可以被改写成新的真事实 (而非简单删除)

=== KB 数据模型 (论文 Section 3) ===

知识库 K = {(f_1, H_1), (f_2, H_2), ...}
- f_j:  原子事实 (atomic fact)，例如 "Elizabeth II is the Queen of England"
- H_j:  历史记录，例如 [(2020, True), (2022-09, False)]
         表示该事实在 2020年为真，2022年9月后为假

支持的操作:
  - add_fact():     添加新事实 (Step 3 提取的新原子事实)
  - reinforce():    标记事实仍然为真 (Step 2 的 "Reinforce" 判定)
  - make_false():   标记事实已变假 (Step 2 的 "Make False" 判定)
  - rewrite():      将假事实改写为真事实 (Step 2 第二轮 LLM 尝试)
  - retrieve():     稠密向量检索 top-k 相关事实

=== 与 ComRAG/QARC 的关键区别 ===
- ERASE 修改的是事实级别的内容 (f_j 的真假和文本)，而非文档集合
- ERASE 是文档驱动的 (document arrives → update KB)，不感知用户兴趣变化
- ComRAG 根本不修改 KB，只维护 QA 记忆
- QARC 修改的是文档级别的 KB 组成 (哪些文档在 KB 中)，不修改文档内容

=== 推理时的使用方式 (论文 Appendix A.3) ===
检索事实时，把历史变化信息一起给 LLM:
  "Elizabeth II is the Queen of England (true at 2020, false at 2022-09)"
这让 LLM 能理解知识的时间演变，做出更准确的回答。
"""

import numpy as np
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FactHistory:
    """事实的单条历史记录 — 记录某个时间点的真假状态。"""
    timestamp: str
    truth_value: bool  # True = 事实为真, False = 事实为假

    def __repr__(self):
        status = "true" if self.truth_value else "false"
        return f"({status} at {self.timestamp})"


@dataclass
class FactEntry:
    """
    ERASE 知识库中的一条事实条目。

    对应论文中知识库 K 的元素 (f_j, H_j):
    - fact (f_j):      原子事实的自然语言文本
    - history (H_j):   真假状态的时间序列
    - embedding:       事实文本的稠密向量 (用于检索)
    - fact_id:         唯一标识符
    - source:          引入/最后修改该事实的文档来源
    """
    fact: str
    embedding: np.ndarray
    history: List[FactHistory] = field(default_factory=list)
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_currently_true(self) -> bool:
        """当前是否为真 — 取历史记录中最新的状态。"""
        if not self.history:
            return True  # 无历史时默认为真
        return self.history[-1].truth_value

    @property
    def latest_timestamp(self) -> Optional[str]:
        if not self.history:
            return None
        return self.history[-1].timestamp

    def reinforce(self, timestamp: str):
        """Reinforce: 标记该事实在指定时间仍然为真。"""
        self.history.append(FactHistory(timestamp=timestamp, truth_value=True))

    def make_false(self, timestamp: str):
        """Make False: 标记该事实在指定时间已变为假。"""
        self.history.append(FactHistory(timestamp=timestamp, truth_value=False))

    def format_history_string(self) -> str:
        """将历史格式化为字符串 — 论文 Appendix A.3 推理时给 LLM 看。"""
        if not self.history:
            return "(no history)"
        parts = [str(h) for h in self.history]
        return ", ".join(parts)

    def __repr__(self):
        status = "TRUE" if self.is_currently_true else "FALSE"
        return f"FactEntry[{self.fact_id}|{status}]: {self.fact[:60]}..."


@dataclass
class RetrievalResult:
    """从知识库检索的单条结果。"""
    entry: FactEntry
    similarity: float


# ============================================================
# ERASE 知识库
# ============================================================

class ERASEKnowledgeBase:
    """
    可编辑外部知识库 — ERASE 的核心数据结构。

    支持稠密向量检索 + 事实级增删改查 + 历史追踪。

    核心操作:
    - add_fact():       添加新事实，初始状态为真
    - retrieve():       稠密向量检索 top-k 相似事实
    - reinforce():      标记事实仍为真 (Step 2 "Reinforce")
    - make_false():     标记事实已变假 (Step 2 "Make False")
    - rewrite():        将假事实改写为新的真事实 (Step 2 第二轮)
    - remove():         彻底删除事实
    - get_true_facts(): 获取所有当前为真的事实
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: 推理时的最低检索相似度阈值
                                  (论文 Appendix A.3 使用 0.7)
        """
        self.entries: Dict[str, FactEntry] = {}  # fact_id → FactEntry
        self.similarity_threshold = similarity_threshold

        # 统计
        self._total_added = 0
        self._total_reinforced = 0
        self._total_made_false = 0
        self._total_rewritten = 0
        self._total_removed = 0

    def add_fact(
        self,
        fact: str,
        embedding: np.ndarray,
        timestamp: str,
        source: str = "",
        metadata: Optional[Dict] = None,
    ) -> FactEntry:
        """
        添加一条新事实 — 对应 ERASE Step 3。

        新事实来源于 LLM 对新文档的原子事实提取:
          文档 → LLM → ["fact1", "fact2", ...] → 分别 embed → 加入 KB
        """
        embedding = self._normalize(embedding)
        entry = FactEntry(
            fact=fact,
            embedding=embedding,
            history=[FactHistory(timestamp=timestamp, truth_value=True)],
            source=source,
            metadata=metadata or {},
        )
        self.entries[entry.fact_id] = entry
        self._total_added += 1
        logger.debug(f"添加事实 [{entry.fact_id}]: {fact[:80]}")
        return entry

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
        only_true: bool = False,
    ) -> List[RetrievalResult]:
        """
        稠密向量检索 — 对应论文 Eq (2)。

        公式: Retrieve(K, d) = arg top-k_{(f_j, H_j) ∈ K} E(d)^T E(f_j)

        参数:
            query_embedding: 查询向量 (文档或问题的 embedding)
            top_k:           返回数量
            threshold:       最低相似度阈值
            only_true:       True = 只返回当前为真的事实 (推理时用)
        """
        if not self.entries:
            return []

        query_embedding = self._normalize(query_embedding)
        thresh = threshold if threshold is not None else self.similarity_threshold

        results = []
        for entry in self.entries.values():
            if only_true and not entry.is_currently_true:
                continue
            sim = float(np.dot(query_embedding, entry.embedding))
            if sim >= thresh:
                results.append(RetrievalResult(entry=entry, similarity=sim))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def retrieve_for_update(
        self,
        document_embedding: np.ndarray,
        top_k: int = 20,
    ) -> List[RetrievalResult]:
        """
        为更新步骤检索候选事实 — 对应 ERASE Step 1。

        使用更低的阈值 (0.3)，因为更新时需要找到所有"可能受影响"的事实，
        而不只是高度相关的。宁可多找一些让 LLM 判断，也不要漏掉。
        """
        return self.retrieve(
            query_embedding=document_embedding,
            top_k=top_k,
            threshold=0.3,  # 更新时用更低阈值
            only_true=False,  # 包含已变假的事实
        )

    def reinforce_fact(self, fact_id: str, timestamp: str):
        """Reinforce: 标记事实仍为真 — Step 2 第一轮 LLM 判定。"""
        if fact_id in self.entries:
            self.entries[fact_id].reinforce(timestamp)
            self._total_reinforced += 1

    def make_fact_false(self, fact_id: str, timestamp: str):
        """Make False: 标记事实已变假 — Step 2 第一轮 LLM 判定。"""
        if fact_id in self.entries:
            self.entries[fact_id].make_false(timestamp)
            self._total_made_false += 1

    def rewrite_fact(
        self,
        fact_id: str,
        new_fact: str,
        new_embedding: np.ndarray,
        timestamp: str,
    ) -> Optional[FactEntry]:
        """
        改写事实 — Step 2 第二轮 LLM 尝试将假事实改写为真。

        例如: "Elizabeth II is Queen" → "Charles III is King"
        保留原 fact_id 以便追踪改写历史。
        """
        if fact_id not in self.entries:
            return None

        old_entry = self.entries[fact_id]
        new_embedding = self._normalize(new_embedding)

        new_entry = FactEntry(
            fact=new_fact,
            embedding=new_embedding,
            history=[FactHistory(timestamp=timestamp, truth_value=True)],
            fact_id=fact_id,
            source=old_entry.source,
            metadata={
                **old_entry.metadata,
                "rewritten_from": old_entry.fact,
                "rewrite_timestamp": timestamp,
            },
        )
        self.entries[fact_id] = new_entry
        self._total_rewritten += 1
        logger.debug(f"改写事实 [{fact_id}]: '{old_entry.fact[:40]}' → '{new_fact[:40]}'")
        return new_entry

    def remove_fact(self, fact_id: str) -> bool:
        if fact_id in self.entries:
            del self.entries[fact_id]
            self._total_removed += 1
            return True
        return False

    def get_true_facts(self) -> List[FactEntry]:
        return [e for e in self.entries.values() if e.is_currently_true]

    def get_false_facts(self) -> List[FactEntry]:
        return [e for e in self.entries.values() if not e.is_currently_true]

    def get_all_facts(self) -> List[FactEntry]:
        return list(self.entries.values())

    def size(self) -> int:
        return len(self.entries)

    def format_facts_for_inference(self, facts: List[RetrievalResult]) -> str:
        """
        格式化事实 + 历史信息供 LLM 推理 — 论文 Appendix A.3。

        格式: f_i (v_{i0} at τ_{i0}, v_{i1} at τ_{i1}, ...)
        例: "Elizabeth II is Queen (true at 2020, false at 2022-09)"
        """
        if not facts:
            return "(No relevant facts found)"
        lines = []
        for i, r in enumerate(facts, 1):
            entry = r.entry
            hist_str = entry.format_history_string()
            lines.append(f"{entry.fact} ({hist_str})")
        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        true_count = sum(1 for e in self.entries.values() if e.is_currently_true)
        false_count = len(self.entries) - true_count
        return {
            "total_facts": len(self.entries),
            "true_facts": true_count,
            "false_facts": false_count,
            "total_added": self._total_added,
            "total_reinforced": self._total_reinforced,
            "total_made_false": self._total_made_false,
            "total_rewritten": self._total_rewritten,
            "total_removed": self._total_removed,
        }

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec
