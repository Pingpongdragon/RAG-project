"""
ComRAG 完整管线: 查询阶段 (Algorithm 1) + 自适应温度 + 端到端编排

论文: ComRAG — Conversational Retrieval-Augmented Generation (ACL 2025 Industry Track)
链接: https://arxiv.org/abs/2506.21098

=== 查询阶段 (论文 Algorithm 1) ===
给定新查询 q，完整处理流程:
  q → embed → 路由决策 → 生成/复用回答 → 评分 → 更新记忆

三种策略的具体执行:
  策略1 (直接复用):   不调 LLM，直接返回 V_high 中的历史回答
  策略2 (参考生成):   将 V_high 中的相关 QA 对放入 prompt 作为 ICL 示例
                     → LLM 参考这些高质量回答来生成新答案
  策略3 (KB + 避免): 从 KB 检索文档 + 将 V_low 中的低质量 QA 作为反面教材
                     → LLM 被告知"这些回答是错的，不要模仿"

=== 自适应温度 (论文 Section 4.4) ===
根据检索到的历史回答的分数方差自动调节 LLM 的 temperature:
  T(Δ) = |exp(-k × min_gap)|  截断到 [T_min, T_max]

直觉:
  - 历史分数差异小 (min_gap ≈ 0) → 高温 → 鼓励 LLM 探索新回答方式
  - 历史分数差异大 (min_gap >> 0) → 低温 → 让 LLM 保持稳定，沿着高分方向走

=== 端到端编排循环 ===
对每个查询:  embed → route → answer → score → update memory
这形成了一个"越用越好"的正反馈循环:
  好回答 → 存入 V_high → 下次相似问题被直接复用或作为参考
  差回答 → 存入 V_low → 下次相似问题被告知避免这种回答
"""

import numpy as np
import math
import json
import logging
from typing import Dict, List, Optional, Any, Callable

from updator.comrag.memory import DynamicMemory, SearchResult
from updator.comrag.updater import ComRAGUpdater

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 模板 (论文 Appendix D)
# ============================================================

# 策略2 的 prompt: 用高质量 QA 作为参考
COMRAG_PROMPT_REFERENCE = """You are a helpful assistant answering questions based on your knowledge.

### Instructions:
1. Analyze the question carefully.
2. If previous_relevant_qa is highly similar to the current question, you can directly use its answer.
3. If previous_relevant_qa is not exactly the same, use it as reference but adjust for the current question.
4. Compare answers with higher and lower scores, and analyze reasons for improved scores.
   Avoid repeating mistakes from lower-scored answers.
5. Provide only the final answer.

### Context
- previous_relevant_qa:
{previous_relevant_qa}

### Question
{question}

Please return your answer directly.
"""

# 策略3 的 prompt: 用 KB 文档 + 低质量 QA 作为反面教材
COMRAG_PROMPT_KB_AVOIDANCE = """You are a helpful assistant answering questions based on your knowledge.

### Instructions:
1. Analyze the question carefully.
2. If knowledge_base_context exists, use it as the primary reference for your answer.
3. Analyze poor QA examples from bad_cqa_contexts (if available).
   Compare answers with higher and lower feedback scores.
   Avoid repeating errors from low-scored answers.
4. If insufficient context, respond with: "Unable to answer based on available knowledge."
5. Provide only the final answer.

### Context
- knowledge_base_context:
{knowledge_base_context}

- bad_cqa_contexts:
{bad_cqa_contexts}

### Question
{question}

Please return your answer directly.
"""


# ============================================================
# 自适应温度 (论文 Section 4.4)
# ============================================================

def compute_adaptive_temperature(
    scores: List[float],
    k: float = 250.0,
    t_min: float = 0.7,
    t_max: float = 1.2,
) -> float:
    """
    自适应温度调节 — 论文 Section 4.4。

    公式: T(Δ) = |exp(-k × min_{1≤i≤l-1}(s_{i+1} - s_i))|  截断到 [T_min, T_max]

    逻辑:
    - 对检索到的历史回答的评分排序
    - 计算相邻分数的最小差距 (min_gap)
    - min_gap 小 → 分数密集 → 模型不确定 → 提高温度鼓励探索
    - min_gap 大 → 分数分散 → 有明显好坏之分 → 降低温度保持稳定

    参数 (论文 Section 5.4 默认值):
      k:     缩放因子, 默认 250
      t_min: 最低温度, 默认 0.7
      t_max: 最高温度, 默认 1.2
    """
    if len(scores) < 2:
        return (t_min + t_max) / 2.0

    sorted_scores = sorted(scores)
    min_gap = min(
        sorted_scores[i + 1] - sorted_scores[i]
        for i in range(len(sorted_scores) - 1)
    )

    raw_temp = math.exp(-k * min_gap)
    return max(t_min, min(t_max, raw_temp))


# ============================================================
# ComRAG 完整管线
# ============================================================

class ComRAGPipeline:
    """
    ComRAG 端到端管线，编排查询 + 更新两个阶段。

    === 整体架构 ===
    外部输入:
      - embed_fn:  文本 → embedding 向量
      - llm_fn:    prompt + temperature → 生成文本
      - retriever: KB 检索器 (策略3 时使用)
      - score_fn:  评分器 (默认用 BERT-Score)

    内部组件:
      - DynamicMemory: 双向量库 (V_high + V_low)
      - ComRAGUpdater: 更新管理器 (评分 → 路由到目标库 → 聚类放置)

    处理流程:
      question → embed → 路由(三策略) → 回答 → 评分 → 记忆更新
    """

    def __init__(
        self,
        embed_fn: Callable,
        llm_fn: Callable,
        retriever=None,
        score_fn: Optional[Callable] = None,
        tau: float = 0.75,
        delta: float = 0.9,
        gamma: float = 0.6,
        adaptive_temp_k: float = 250.0,
        adaptive_temp_min: float = 0.7,
        adaptive_temp_max: float = 1.2,
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.retriever = retriever
        self.score_fn = score_fn

        # 自适应温度参数
        self.temp_k = adaptive_temp_k
        self.temp_min = adaptive_temp_min
        self.temp_max = adaptive_temp_max

        # 初始化动态记忆 + 更新器
        self.memory = DynamicMemory(tau=tau, delta=delta, gamma=gamma)
        self.updater = ComRAGUpdater(
            dynamic_memory=self.memory,
            embed_fn=embed_fn,
            score_fn=score_fn,
        )

        # 统计
        self._total_queries = 0
        self._reuse_count = 0

    def answer_question(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        kb_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理一个问题 — ComRAG 完整流程。

        步骤对应论文:
        1. Embed: Emb(q)
        2. Route: Algorithm 1 三层路由
        3. Answer: 根据策略生成或复用回答
        4. Score: s = Scorer(q, â) — 论文用 BERT-Score
        5. Update: Algorithm 2 存入记忆库
        """
        self._total_queries += 1

        # 步骤1: 编码问题
        query_embedding = self.embed_fn(question)

        # 步骤2: 路由决策
        route = self.memory.route_query(query_embedding)
        strategy = route["strategy"]
        self.updater.record_strategy(strategy)

        # 步骤3: 根据策略生成回答
        answer = ""
        temperature = (self.temp_min + self.temp_max) / 2.0

        if strategy == "direct_reuse":
            # ---- 策略1: 直接复用 ----
            # 不需要调 LLM，直接返回历史回答，省钱省时间
            best_match = route["best_match"]
            answer = best_match.record.answer
            self._reuse_count += 1
            temperature = 0.0

            logger.info(
                f"[查询 #{self._total_queries}] 直接复用 "
                f"(sim={route['max_similarity']:.3f}): {answer[:80]}..."
            )

        elif strategy == "reference_generation":
            # ---- 策略2: 参考生成 ----
            # 用 V_high 中的高质量 QA 作为 ICL 示例
            refs = route["high_q_references"]
            temperature = self._compute_temperature(refs)

            ref_text = self._format_references(refs)
            prompt = COMRAG_PROMPT_REFERENCE.format(
                previous_relevant_qa=ref_text,
                question=question,
            )
            answer = self.llm_fn(prompt, temperature)

            logger.info(
                f"[查询 #{self._total_queries}] 参考生成 "
                f"(sim={route['max_similarity']:.3f}, 参考数={len(refs)}, T={temperature:.2f})"
            )

        elif strategy == "kb_avoidance":
            # ---- 策略3: KB回退 + 低质量避免 ----
            # V_low 中的记录作为反面教材
            low_negs = route["low_q_negatives"]
            temperature = self._compute_temperature(low_negs)

            # 从 KB 检索上下文
            if kb_context is None and self.retriever is not None:
                try:
                    kb_results = self.retriever.retrieve(question)
                    kb_context = self._format_kb_results(kb_results)
                except Exception as e:
                    logger.warning(f"KB 检索失败: {e}")
                    kb_context = ""

            bad_text = self._format_negatives(low_negs)
            prompt = COMRAG_PROMPT_KB_AVOIDANCE.format(
                knowledge_base_context=kb_context or "(无 KB 上下文)",
                bad_cqa_contexts=bad_text or "(无反面教材)",
                question=question,
            )
            answer = self.llm_fn(prompt, temperature)

            logger.info(
                f"[查询 #{self._total_queries}] KB回退+避免 "
                f"(反面教材数={len(low_negs)}, T={temperature:.2f})"
            )

        # 步骤4: 评分
        score = None
        if reference_answer and self.score_fn:
            score = self.score_fn(question, answer, reference_answer)
        elif reference_answer:
            score = ComRAGUpdater._default_scorer(question, answer, reference_answer)

        # 步骤5: 更新记忆
        update_result = self.updater.update(
            question=question,
            answer=answer,
            score=score,
            reference_answer=reference_answer,
            embedding=query_embedding,
        )

        return {
            "question": question,
            "answer": answer,
            "strategy": strategy,
            "score": score,
            "temperature": temperature,
            "max_similarity": route["max_similarity"],
            "update_result": update_result,
        }

    def process_stream(
        self,
        questions: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        处理查询流 — 模拟真实的对话式 QA 场景。

        每个 item 需要:
        - "question": str (必填)
        - "reference_answer": str (可选，有则用于评分)
        - "kb_context": str (可选，有则直接使用，无则走 retriever)
        """
        results = []
        for i, item in enumerate(questions):
            result = self.answer_question(
                question=item["question"],
                reference_answer=item.get("reference_answer"),
                kb_context=item.get("kb_context"),
            )
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                stats = self.get_statistics()
                logger.info(
                    f"[流处理] 已处理 {i+1}/{len(questions)}, "
                    f"复用率={stats['reuse_rate']:.1%}, "
                    f"V_high={stats['memory']['high_store']['total_records']}, "
                    f"V_low={stats['memory']['low_store']['total_records']}"
                )

        return results

    def get_statistics(self) -> Dict:
        updater_stats = self.updater.get_statistics()
        reuse_rate = self._reuse_count / max(1, self._total_queries)
        return {
            "total_queries": self._total_queries,
            "reuse_count": self._reuse_count,
            "reuse_rate": reuse_rate,
            **updater_stats,
        }

    # ---- 内部辅助方法 ----

    def _compute_temperature(self, search_results: List[SearchResult]) -> float:
        if not search_results:
            return (self.temp_min + self.temp_max) / 2.0
        scores = [sr.record.score for sr in search_results]
        return compute_adaptive_temperature(
            scores, k=self.temp_k, t_min=self.temp_min, t_max=self.temp_max,
        )

    @staticmethod
    def _format_references(refs: List[SearchResult]) -> str:
        if not refs:
            return "(无相关历史)"
        parts = []
        for i, sr in enumerate(refs, 1):
            parts.append(
                f"[Reference {i}] (score={sr.record.score:.2f}, sim={sr.similarity:.3f})\n"
                f"Q: {sr.record.question}\n"
                f"A: {sr.record.answer}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_negatives(negs: List[SearchResult]) -> str:
        if not negs:
            return "(No negative examples)"
        parts = []
        for i, sr in enumerate(negs, 1):
            parts.append(
                f"[Bad Example {i}] (score={sr.record.score:.2f})\n"
                f"Q: {sr.record.question}\n"
                f"A (BAD - DO NOT follow): {sr.record.answer}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_kb_results(kb_results: List[Dict]) -> str:
        if not kb_results:
            return "(No knowledge base context available)"
        parts = []
        for i, item in enumerate(kb_results, 1):
            text = item.get("text", str(item))
            parts.append(f"[Document {i}]\n{text}")
        return "\n\n".join(parts)
