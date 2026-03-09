"""
ComRAG Update Phase (Algorithm 2)

Paper: ComRAG (ACL 2025 Industry Track)
https://arxiv.org/abs/2506.21098

Update Flow:
1. Generate answer: a_hat = LLM(q, context)
2. Score answer: s = Scorer(q, a_hat)  (e.g., BERT-Score)
3. Route to V_high (s >= gamma) or V_low (s < gamma)
4. Within target store:
   - If near-duplicate exists (sim >= delta): replace if new score > old score
   - Else if cluster match (sim >= tau): add to cluster, update centroid
   - Else: create new cluster
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ComRAGUpdater:
    """
    ComRAG Update Phase implementation.

    Works with DynamicMemory to manage the update cycle:
    answer -> score -> route to store -> centroid-based placement.

    The DynamicMemory.add() already implements Algorithm 2 internally,
    so this class orchestrates the higher-level update loop:
    scoring, embedding, and feeding results into DynamicMemory.
    """

    def __init__(
        self,
        dynamic_memory,
        embed_fn: Optional[Callable] = None,
        score_fn: Optional[Callable] = None,
    ):
        """
        Args:
            dynamic_memory: DynamicMemory instance (from clustering_detector.py)
            embed_fn: function(text: str) -> np.ndarray, for embedding questions
            score_fn: function(question: str, answer: str, reference: str) -> float
                      Returns quality score in [0, 1]. Default uses BERT-Score.
        """
        self.memory = dynamic_memory
        self.embed_fn = embed_fn
        self.score_fn = score_fn or self._default_scorer

        # Statistics
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
        Execute one update step (Algorithm 2).

        Args:
            question:         The question text
            answer:           The generated answer
            score:            Pre-computed quality score. If None, uses score_fn.
            reference_answer: Ground truth answer for scoring (if available)
            embedding:        Pre-computed question embedding. If None, uses embed_fn.
            metadata:         Optional metadata to store with the record

        Returns:
            Update result dict from DynamicMemory.add()
        """
        # 1. Get embedding
        if embedding is None:
            if self.embed_fn is None:
                raise ValueError("No embedding provided and no embed_fn configured")
            embedding = self.embed_fn(question)

        # 2. Get score
        if score is None:
            if reference_answer is not None:
                score = self.score_fn(question, answer, reference_answer)
            else:
                # Without reference, use a default moderate score
                logger.warning(
                    "No score or reference_answer provided, using default score 0.5"
                )
                score = 0.5

        # 3. Add to DynamicMemory (Algorithm 2 logic is inside)
        result = self.memory.add(
            question=question,
            answer=answer,
            embedding=embedding,
            score=score,
            metadata=metadata,
        )

        self.update_count += 1
        logger.info(
            f"[Update #{self.update_count}] "
            f"store={result.get('target_store', '?')}, "
            f"action={result.get('action', '?')}, "
            f"score={score:.3f}"
        )

        return result

    def batch_update(
        self,
        qa_pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Batch update multiple QA pairs.

        Each item in qa_pairs should have:
        - "question": str
        - "answer": str
        - "score": float (optional)
        - "reference_answer": str (optional)
        - "embedding": np.ndarray (optional)
        """
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
        """Record which routing strategy was used (for analytics)."""
        if strategy in self.strategy_counts:
            self.strategy_counts[strategy] += 1

    def get_statistics(self) -> Dict:
        """Get updater statistics."""
        memory_stats = self.memory.get_statistics()
        return {
            "update_count": self.update_count,
            "strategy_counts": self.strategy_counts.copy(),
            "memory": memory_stats,
        }

    @staticmethod
    def _default_scorer(question: str, answer: str, reference: str) -> float:
        """
        Default scoring using BERT-Score (paper Section 5.4).
        Falls back to simple overlap ratio if bert_score not available.
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
                "bert_score not installed, using simple word overlap scorer. "
                "Install with: pip install bert-score"
            )
            # Simple fallback: word overlap ratio
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
