"""
benchmarks/archive_legacy/rag_pipeline.py — core/ RAG 引擎的统一封装

将 core/ 的 retriever + reranker + generator + evaluator 组装成一条可调用的管线，
供 run_experiments.py 调用

典型用法:
    pipe = RAGPipeline(use_real_embeddings=True, device="cuda")
    doc_embs = pipe.encode_documents(doc_texts)
    q_embs   = pipe.encode_queries(query_texts)
    results  = pipe.retrieve_by_embeddings(q_emb, doc_embs, doc_ids)
    answer   = pipe.generate(query, context)
"""

import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# 把项目根目录加到 sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    封装 core/ 的完整 RAG 管线

    两种模式:
      1. lightweight (默认): 只用预计算的 embedding 做 cosine 检索，不调 LLM
         适合大规模对比实验，速度快
      2. full_rag: 构建 FAISS 索引 + BM25 + Reranker + LLM 生成
         适合端到端评测，速度慢但结果更真实
    """

    def __init__(
        self,
        use_real_embeddings: bool = True,
        use_reranker: bool = False,
        use_generator: bool = False,
        embedding_dim: int = 768,
        device: str = "cpu",
    ):
        self.use_real_embeddings = use_real_embeddings
        self.use_reranker = use_reranker
        self.use_generator = use_generator
        self.device = device

        self._embedder = None
        self._reranker = None
        self._generator_fn = None

        if use_real_embeddings:
            self._init_embedder()
        if use_reranker:
            self._init_reranker()
        if use_generator:
            self._init_generator()

    # ---- 初始化组件 ----

    def _init_embedder(self):
        """加载真实 Embedding 模型 (nomic-embed-text-v1.5)"""
        try:
            from models.embeddings import embedding_service
            self._embedder = embedding_service
            logger.info(f"Embedding model loaded: {type(self._embedder)}")
        except Exception as e:
            logger.warning(f"Real embedding model load failed: {e}, falling back to random")
            self.use_real_embeddings = False

    def _init_reranker(self):
        """加载 CrossEncoder Reranker"""
        try:
            from core.reranker import ReRanker
            self._reranker = ReRanker()
            logger.info("Reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker load failed: {e}")
            self.use_reranker = False

    def _init_generator(self):
        """加载 LLM 生成函数"""
        try:
            from core.generator import generate_llm_response
            self._generator_fn = generate_llm_response
            logger.info("Generator loaded")
        except Exception as e:
            logger.warning(f"Generator load failed: {e}")
            self.use_generator = False

    # ---- 核心功能 ----

    def encode(self, texts: List[str], batch_size: int = 32,
              prefix: str = "") -> np.ndarray:
        """
        编码文本为嵌入向量

        Args:
            texts: 待编码的文本列表
            batch_size: 批量大小
            prefix: nomic-embed task prefix (如 "search_query: " 或 "search_document: ")

        Returns:
            np.ndarray shape (N, D), L2 归一化
        """
        if self._embedder is not None:
            if prefix:
                texts = [f"{prefix}{t}" for t in texts]
            embs = self._embedder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embs
        else:
            # fallback: 随机嵌入 (用于快速测试)
            rng = np.random.RandomState(42)
            embs = rng.randn(len(texts), 768).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            return embs / norms

    def encode_queries(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码查询 (加 search_query: 前缀)"""
        return self.encode(texts, batch_size, prefix="search_query: ")

    def encode_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文档 (加 search_document: 前缀)"""
        return self.encode(texts, batch_size, prefix="search_document: ")

    def retrieve_by_embeddings(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        doc_ids: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        基于预计算 embedding 的余弦检索 (lightweight 模式)

        Args:
            query_embedding: 查询嵌入 shape (D,)
            doc_embeddings:  文档嵌入 shape (N, D)
            doc_ids:         文档 ID 列表, 与 doc_embeddings 一一对应
            top_k:           返回 top-k 个结果

        Returns:
            List[(doc_id, score)] 按 score 降序
        """
        if len(doc_ids) == 0:
            return []

        sims = doc_embeddings @ query_embedding
        k = min(top_k, len(doc_ids))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(doc_ids[i], float(sims[i])) for i in top_idx]

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        用 CrossEncoder 重排序

        Args:
            query: 查询文本
            candidates: 候选文档列表 [{"doc_id", "text", "score"}, ...]
            top_k: 重排后保留 top-k

        Returns:
            重排后的文档列表
        """
        if not self._reranker or not candidates:
            return candidates[:top_k]

        try:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self._reranker.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates[:top_k]
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            return candidates[:top_k]

    def generate(self, query: str, context: List[Dict], language: str = "en") -> str:
        """
        用 LLM 生成回答

        Args:
            query: 查询文本
            context: 检索到的文档列表 [{"text": ...}, ...]
            language: "en" 或 "zh"

        Returns:
            生成的回答文本
        """
        if not self._generator_fn:
            # 无生成器时返回空字符串 (只评估检索)
            return ""

        try:
            _, answer = self._generator_fn(
                query=query,
                context=context,
                language=language,
            )
            return answer or ""
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

    # ---- 评估工具 ----

    @staticmethod
    def compute_recall(
        retrieved_ids: List[str],
        gold_ids: List[str],
    ) -> float:
        """计算 Recall@K"""
        if not gold_ids:
            return 0.0
        hits = len(set(retrieved_ids) & set(gold_ids))
        return hits / len(gold_ids)

    @staticmethod
    def compute_em(prediction: str, gold: str) -> float:
        """Exact Match"""
        return float(prediction.strip().lower() == gold.strip().lower())

    @staticmethod
    def compute_f1(prediction: str, gold: str) -> float:
        """Token-level F1"""
        pred_tokens = set(prediction.lower().split())
        gold_tokens = set(gold.lower().split())
        if not gold_tokens:
            return 0.0
        if not pred_tokens:
            return 0.0
        common = pred_tokens & gold_tokens
        if not common:
            return 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)
