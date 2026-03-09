"""
ComRAG 更新阶段 (Algorithm 2)

论文: ComRAG (ACL 2025 Industry Track)
链接: https://arxiv.org/abs/2506.21098

=== 论文 Algorithm 2: Update Phase ===

更新流程 — 每次 LLM 回答完一个问题后执行:
  1. 评分: s = Scorer(q, â)
     - 论文默认用 BERT-Score F1 来衡量回答质量
     - 有参考答案时计算 F1(â, a_ref)，无参考时默认 0.5
  2. 路由: 根据 s 与 γ 的比较，选择目标库
     - s >= γ → 存入 V_high (高质量库)
     - s <  γ → 存入 V_low  (低质量库)
  3. 放置: 在目标库内由 CentroidClusterStore.add() 处理
     - 近重复 (sim >= δ): 仅保留评分更高者
     - 文话题匹配 (sim >= τ): 加入簇，更新质心
     - 新话题 (sim < τ): 创建新簇

=== 这个类的职责 ===
DynamicMemory.add() 已经封装了步骤2+3的内部逻辑，
ComRAGUpdater 负责更上层的编排:
  - 获取/计算 embedding
  - 获取/计算评分
  - 调用 DynamicMemory.add()
  - 记录策略使用统计
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ComRAGUpdater:
    """
    ComRAG 更新阶段实现 — 编排 embedding/评分/记忆写入。

    与 DynamicMemory 配合工作:
    回答 → 评分 → 路由到目标库 → 质心聚类放置
    """

    def __init__(
        self,
        dynamic_memory,
        embed_fn: Optional[Callable] = None,
        score_fn: Optional[Callable] = None,
    ):
        """
        Args:
            dynamic_memory: DynamicMemory 实例 (来自 memory.py)
            embed_fn: 文本 → embedding 向量的函数
            score_fn: 评分函数 (question, answer, reference) → float ∈ [0,1]
                      默认使用 BERT-Score F1
        """
        self.memory = dynamic_memory
        self.embed_fn = embed_fn
        self.score_fn = score_fn or self._default_scorer

        # 统计
        self.update_count = 0
        self.strategy_counts = {
            "direct_reuse": 0,
            "reference_generation": 0,
            "kb_avoidance": 0,
        }

    def update(
        self,
        question: str,
        answer: str,
        score: Optional[float] = None,
        reference_answer: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        执行一次更新 — 对应 Algorithm 2 的一次迭代。

        典型调用时机: pipeline.answer_question() 生成回答后立刻调用。
        """
        # 1. 获取 embedding
        if embedding is None:
            if self.embed_fn is None:
                raise ValueError("未提供 embedding 且未配置 embed_fn")
            embedding = self.embed_fn(question)

        # 2. 计算评分
        if score is None:
            if reference_answer is not None:
                score = self.score_fn(question, answer, reference_answer)
            else:
                logger.warning("无评分且无参考答案，使用默认分数 0.5")
                score = 0.5

        # 3. 存入 DynamicMemory (Algorithm 2 的核心逻辑在 DynamicMemory.add() 内部)
        result = self.memory.add(
            question=question,
            answer=answer,
            embedding=embedding,
            score=score,
            metadata=metadata,
        )

        self.update_count += 1
        logger.info(
            f"[更新 #{self.update_count}] "
            f"目标库={result.get('target_store', '?')}, "
            f"动作={result.get('action', '?')}, "
            f"评分={score:.3f}"
        )

        return result

    def batch_update(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量更新多条 QA 对。"""
        results = []
        for item in qa_pairs:
            result = self.update(
                question=item["question"],
                answer=item["answer"],
                score=item.get("score"),
                reference_answer=item.get("reference_answer"),
                embedding=item.get("embedding"),
                metadata=item.get("metadata"),
            )
            results.append(result)
        return results

    def record_strategy(self, strategy: str):
        """记录路由策略使用次数（用于分析）。"""
        if strategy in self.strategy_counts:
            self.strategy_counts[strategy] += 1

    def get_statistics(self) -> Dict:
        memory_stats = self.memory.get_statistics()
        return {
            "update_count": self.update_count,
            "strategy_counts": self.strategy_counts.copy(),
            "memory": memory_stats,
        }

    @staticmethod
    def _default_scorer(question: str, answer: str, reference: str) -> float:
        """
        默认评分器 — 论文 Section 5.4 使用 BERT-Score F1。

        如果 bert_score 未安装，回退到简单的词重叠 F1。
        """
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                [answer], [reference],
                lang="en",
                verbose=False,
            )
            return float(F1[0])
        except ImportError:
            logger.warning(
                "bert_score 未安装，使用简单词重叠评分。"
                "安装: pip install bert-score"
            )
            answer_words = set(answer.lower().split())
            ref_words = set(reference.lower().split())
            if not ref_words:
                return 0.0
            overlap = len(answer_words & ref_words)
            precision = overlap / len(answer_words) if answer_words else 0.0
            recall = overlap / len(ref_words)
            if precision + recall == 0:
                return 0.0
            f1 = 2 * precision * recall / (precision + recall)
            return f1
