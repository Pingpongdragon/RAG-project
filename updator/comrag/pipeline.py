"""
ComRAG Pipeline: Query Phase (Algorithm 1) + Adaptive Temperature + Full Orchestration

Paper: ComRAG (ACL 2025 Industry Track)
https://arxiv.org/abs/2506.21098

Query Phase (Algorithm 1):
1) Direct Reuse:        sim >= delta  -> return historical answer directly
2) Reference Generation: tau <= sim < delta -> LLM with high-quality QA as reference
3) KB + Avoidance:       sim < tau  -> LLM with KB docs + low-quality QA as negatives

Adaptive Temperature (Section 4.4):
    T(delta) = |exp(-k * min_gap)|  clamped to [T_min, T_max]

Full orchestration loop:
    question -> embed -> route -> answer -> score -> update memory
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
# Prompt Templates (Appendix D of the paper)
# ============================================================

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
# Adaptive Temperature (Section 4.4)
# ============================================================

def compute_adaptive_temperature(
    scores: List[float],
    k: float = 250.0,
    t_min: float = 0.7,
    t_max: float = 1.2,
) -> float:
    """
    Adaptive temperature tuning from Section 4.4.

    T(Delta) = |exp(-k * min_{1<=i<=l-1}(s_{i+1} - s_i))| clamped to [T_min, T_max]

    - Low variance (scores are similar) -> high temperature (encourage exploration)
    - High variance (scores differ a lot) -> low temperature (ensure consistency)

    Args:
        scores: List of quality scores from retrieved evidence
        k:      Scaling factor (paper default: 250)
        t_min:  Minimum temperature (paper default: 0.7)
        t_max:  Maximum temperature (paper default: 1.2)

    Returns:
        Adaptive temperature value
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
# ComRAG Pipeline
# ============================================================

class ComRAGPipeline:
    """
    Full ComRAG pipeline orchestrating Query + Update phases.

    Usage:
        pipeline = ComRAGPipeline(
            embed_fn=my_embed,
            llm_fn=my_llm,
            retriever=my_retriever,  # for KB retrieval (strategy 3)
        )

        # Process a single question
        result = pipeline.answer_question("What is RAG?")

        # Process a stream of questions
        for q in questions:
            result = pipeline.answer_question(q, reference_answer=ref)
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
        """
        Args:
            embed_fn:  function(text: str) -> np.ndarray
            llm_fn:    function(prompt: str, temperature: float) -> str
            retriever: Object with .retrieve(query) method for KB retrieval
            score_fn:  function(question, answer, reference) -> float
            tau, delta, gamma: ComRAG thresholds (paper Section 5.4)
        """
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.retriever = retriever
        self.score_fn = score_fn

        # Adaptive temperature params
        self.temp_k = adaptive_temp_k
        self.temp_min = adaptive_temp_min
        self.temp_max = adaptive_temp_max

        # Initialize DynamicMemory + Updater
        self.memory = DynamicMemory(tau=tau, delta=delta, gamma=gamma)
        self.updater = ComRAGUpdater(
            dynamic_memory=self.memory,
            embed_fn=embed_fn,
            score_fn=score_fn,
        )

        # Statistics
        self._total_queries = 0
        self._reuse_count = 0

    def answer_question(
        self,
        question: str,
        reference_answer: Optional[str] = None,
        kb_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process one question through the full ComRAG pipeline.

        Flow:
        1. Embed question
        2. Route query (three-tier strategy)
        3. Generate/reuse answer
        4. Score answer
        5. Update memory

        Args:
            question:         The input question
            reference_answer: Ground truth (for scoring, optional)
            kb_context:       Pre-retrieved KB context (optional, or uses self.retriever)

        Returns:
            {
                "question": str,
                "answer": str,
                "strategy": str,
                "score": float,
                "temperature": float,
                "update_result": dict,
            }
        """
        self._total_queries += 1

        # Step 1: Embed question
        query_embedding = self.embed_fn(question)

        # Step 2: Route query
        route = self.memory.route_query(query_embedding)
        strategy = route["strategy"]
        self.updater.record_strategy(strategy)

        # Step 3: Generate answer based on strategy
        answer = ""
        temperature = (self.temp_min + self.temp_max) / 2.0  # default

        if strategy == "direct_reuse":
            # Strategy 1: Direct Reuse - no LLM call needed
            best_match = route["best_match"]
            answer = best_match.record.answer
            self._reuse_count += 1
            temperature = 0.0  # not used

            logger.info(
                f"[Query #{self._total_queries}] DIRECT REUSE "
                f"(sim={route['max_similarity']:.3f}): {answer[:80]}..."
            )

        elif strategy == "reference_generation":
            # Strategy 2: Reference Generation with high-quality QA
            refs = route["high_q_references"]
            temperature = self._compute_temperature(refs)

            ref_text = self._format_references(refs)
            prompt = COMRAG_PROMPT_REFERENCE.format(
                previous_relevant_qa=ref_text,
                question=question,
            )
            answer = self.llm_fn(prompt, temperature)

            logger.info(
                f"[Query #{self._total_queries}] REFERENCE GENERATION "
                f"(sim={route['max_similarity']:.3f}, refs={len(refs)}, T={temperature:.2f})"
            )

        elif strategy == "kb_avoidance":
            # Strategy 3: KB + Low-Quality Avoidance
            low_negs = route["low_q_negatives"]
            temperature = self._compute_temperature(low_negs)

            # Get KB context
            if kb_context is None and self.retriever is not None:
                try:
                    kb_results = self.retriever.retrieve(question)
                    kb_context = self._format_kb_results(kb_results)
                except Exception as e:
                    logger.warning(f"KB retrieval failed: {e}")
                    kb_context = ""

            bad_text = self._format_negatives(low_negs)
            prompt = COMRAG_PROMPT_KB_AVOIDANCE.format(
                knowledge_base_context=kb_context or "(No KB context available)",
                bad_cqa_contexts=bad_text or "(No negative examples available)",
                question=question,
            )
            answer = self.llm_fn(prompt, temperature)

            logger.info(
                f"[Query #{self._total_queries}] KB + AVOIDANCE "
                f"(negatives={len(low_negs)}, T={temperature:.2f})"
            )

        # Step 4: Score the answer
        score = None
        if reference_answer and self.score_fn:
            score = self.score_fn(question, answer, reference_answer)
        elif reference_answer:
            score = ComRAGUpdater._default_scorer(question, answer, reference_answer)

        # Step 5: Update memory
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
        Process a stream of questions (simulating real-time CQA).

        Each item should have:
        - "question": str (required)
        - "reference_answer": str (optional)
        - "kb_context": str (optional)

        Returns:
            List of answer results
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
                    f"[Stream] Processed {i+1}/{len(questions)}, "
                    f"reuse_rate={stats['reuse_rate']:.1%}, "
                    f"high_store={stats['memory']['high_store']['total_records']}, "
                    f"low_store={stats['memory']['low_store']['total_records']}"
                )

        return results

    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        updater_stats = self.updater.get_statistics()
        reuse_rate = self._reuse_count / max(1, self._total_queries)
        return {
            "total_queries": self._total_queries,
            "reuse_count": self._reuse_count,
            "reuse_rate": reuse_rate,
            **updater_stats,
        }

    # ---- Internal helpers ----

    def _compute_temperature(self, search_results: List[SearchResult]) -> float:
        """Compute adaptive temperature from evidence scores."""
        if not search_results:
            return (self.temp_min + self.temp_max) / 2.0

        scores = [sr.record.score for sr in search_results]
        return compute_adaptive_temperature(
            scores, k=self.temp_k, t_min=self.temp_min, t_max=self.temp_max,
        )

    @staticmethod
    def _format_references(refs: List[SearchResult]) -> str:
        """Format high-quality references for the prompt."""
        if not refs:
            return "(No relevant history found)"

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
        """Format low-quality examples as negative constraints."""
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
        """Format KB retrieval results for the prompt."""
        if not kb_results:
            return "(No knowledge base context available)"

        parts = []
        for i, item in enumerate(kb_results, 1):
            text = item.get("text", str(item))
            parts.append(f"[Document {i}]\n{text}")
        return "\n\n".join(parts)
