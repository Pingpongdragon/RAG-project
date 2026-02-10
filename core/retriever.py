from typing import List, Union, Dict
from abc import ABC, abstractmethod
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from RAG_project.config import settings
from RAG_project.config.logger_config import logger
from RAG_project.core.reranker import ReRanker 
from langchain_community.vectorstores.utils import DistanceStrategy
from rank_bm25 import BM25Okapi

class BaseRetriever(ABC):
    """检索器基类（最小化修改）"""
    @abstractmethod
    def get_results(self, query: str) -> List[Document]:
        pass


class DenseRetriever(BaseRetriever):
    def __init__(self, vector_db: FAISS):
        """
        直接存储原始 FAISS 实例，获得最高灵活性
        :param vector_db: 已初始化的 FAISS 向量数据库
        """
        self.vector_db = vector_db  # 直接持有原始向量库
        self.default_k = settings.DEFAULT_DENSE_K  # 可配置的默认返回数量
        self.score_threshold = settings.DENSE_SCORE_THRESHOLD  # 可配置的分数阈值

    def get_results(self, query: str) -> List[Document]:
        """
        执行带分数过滤的向量检索
        :param query: 输入查询文本
        :return: 符合条件的文档列表（带分数元数据）
        """
        if len(query.strip()) < 2:
            logger.warning(f"⚠️ 无效短查询: {query}（长度<2）")
            return []

        try:
            # 直接调用 FAISS 的带分数检索方法（返回 (doc, score) 元组列表）
            raw_results = self.vector_db.similarity_search_with_score(
                query=query,
                k=self.default_k,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
        except Exception as e:
            logger.error(f"🔍 向量检索失败: {str(e)}")
            return []

        processed = []
        for doc, score in raw_results:
            if score < self.score_threshold:
                continue
            # 存储转换后的余弦相似度（直接使用原始分数）
            doc.metadata["dense_score"] = float(score)
            doc.metadata["dense_metric"] = "cosine_similarity"  # 更新指标类型
            processed.append(doc)

        logger.info(f"🚀 向量检索完成，结果数: {len(processed)}")  
        return processed
    

class BM25Retriever(BaseRetriever):
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.tokenized_docs = [self._tokenize(doc.page_content) for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"✅ BM25 初始化完成，文档数: {len(self.docs)}")

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()  # 简单空格分词，按需替换

    def get_results(self, query: str) -> List[Document]:
        if len(query.strip()) < 1:
            logger.warning("⚠️ 空白查询")
            return []
        
        # 分词并处理未登录词（Out-Of-Vocabulary）
        tokenized_query = self._tokenize(query)
        valid_tokens = [
            token for token in tokenized_query 
            if any(token in doc_tokens for doc_tokens in self.tokenized_docs)
        ]
        if not valid_tokens:
            logger.warning(f"🚫 查询词未在语料库中出现: {query}")
            return []
        
        # 获取原始 BM25 得分
        raw_scores = self.bm25.get_scores(valid_tokens)  # shape = (n_docs,)
        
        # 处理边界情况：所有文档得分相同
        if len(set(raw_scores)) == 1:
            logger.warning("⚠️ 所有文档得分相同，可能查询词无区分度")
            normalized_scores = [0.5] * len(raw_scores)  # 赋予中间值
        else:
            # 归一化到 [0,1] 范围
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            normalized_scores = [
                (score - min_score) / (max_score - min_score + 1e-6)
                for score in raw_scores
            ]

        # 应用阈值过滤
        candidates = [
            (i, norm_score)
            for i, norm_score in enumerate(normalized_scores)
            if norm_score >= settings.SPARSE_SCORE_THRESHOLD  # 此时阈值应在0-1之间
        ]
        
        # 按归一化分数排序并截断
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:settings.DEFAULT_SPARSE_K]

        logger.info(
            f"🔍 BM25检索结果 | 原始分范围: {min(raw_scores):.2f}-{max(raw_scores):.2f} "
            f"归一化后阈值过滤: {len(candidates)} 条 (阈值={settings.SPARSE_SCORE_THRESHOLD})"
        )

        return [
            Document(
                page_content=self.docs[i].page_content,
                metadata={
                    "bm25_raw_score": float(raw_scores[i]),  # 保留原始分以便调试
                    "bm25_norm_score": float(norm_score),    # 归一化分用于阈值判断
                    **self.docs[i].metadata
                }
            ) for i, norm_score in candidates
        ]



class QARetriever:
    def __init__(
        self, 
        vector_db: FAISS,
        docs: List[Document], 
        reranker_model: Union[str, ReRanker] = settings.RERANKER_MODEL,
        hybrid_search: bool = True
    ):
        # 初始化子组件
        self.dense_retriever = DenseRetriever(vector_db)
        self.sparse_retriever = BM25Retriever(docs)
        self.hybrid_enabled = hybrid_search
        
        # 初始化重排序器
        self.reranker = (
            reranker_model 
            if isinstance(reranker_model, ReRanker) 
            else ReRanker(reranker_model)
        )

    def _merge_results(
        self, 
        dense_results: List[Document], 
        sparse_results: List[Document]
    ) -> List[Dict]:
        """混合检索结果融合"""
        # 结果去重（根据origin_id）
        merged = {}
        for doc in dense_results + sparse_results:
            uid = doc.metadata["doc_id"]
            if uid not in merged:
                merged[uid] = {
                    "dense_score": doc.metadata.get("dense_score", 0),
                    "sparse_score": doc.metadata.get("sparse_score", 0),
                    "doc": doc
                }
        
        # 计算混合分数
        weighted_results = []
        for uid, scores in merged.items():
            hybrid_score = (
                settings.HYBRID_DENSE_WEIGHT * scores["dense_score"] +
                (1 - settings.HYBRID_DENSE_WEIGHT) * scores["sparse_score"]
            )
            weighted_results.append({
                "text": scores["doc"].page_content,
                "hybrid_score": hybrid_score,
                "metadata": scores["doc"].metadata
            })
        
        return sorted(weighted_results, key=lambda x: x["hybrid_score"], reverse=True)[:settings.HYBRID_TOP_K]

    def retrieve(
        self, 
        query: str, 
        rerank_top_k: int = settings.DEFAULT_RERANK_K, 
        rerank_threshold: float = settings.RERANK_THRESHOLD
    ) -> List[Dict]:
        """保持不变的对外接口"""
        # 阶段1: 基础召回
        dense_results = self.dense_retriever.get_results(query)
        logger.info(f"🚀 向量召回数量: {len(dense_results)}")

        # 阶段2: 混合融合（必须配置了hybrid才触发）
        if self.hybrid_enabled:
            sparse_results = self.sparse_retriever.get_results(query)
            candidates = self._merge_results(dense_results, sparse_results)
            logger.info(f"🔀 混合后结果量: {len(candidates)}")
        else:
            candidates = [{
                "text": doc.page_content,
                "hybrid_score": doc.metadata["dense_score"],
                "metadata": doc.metadata
            } for doc in dense_results]
        
        # 阶段3: 深度重排
        rerank_inputs = [item["text"] for item in candidates]
        reranked = self.reranker.rerank(query, rerank_inputs, top_k=rerank_top_k)
        logger.info(f"🎯 精排后数量: {len(reranked)}")
        
        # 阶段4: 最终过滤
        final_results = []
        for res in reranked:
            original_item = next(
                item for item in candidates 
                if item["text"] == res["text"]
            )
            if res["rerank_score"] >= rerank_threshold:
                final_item = {
                    "text": res["text"],
                    "scores": {
                        "hybrid": original_item["hybrid_score"],
                        "rerank": res["rerank_score"]
                    },
                    "metadata": {
                        **original_item["metadata"],
                        **res.get("metadata", {})
                    }
                }
                final_results.append(final_item)
        
        logger.info(f"✅ 最终有效结果: {len(final_results)}条")
        if settings.RANK_ORDER == 0:  # 降序排序
            return sorted(
                final_results, 
                key=lambda x: 0.5 * x["scores"]["rerank"] + 0.5 * x["scores"]["hybrid"], 
                reverse=True
            )
        else:  # 升序排序
            return sorted(
                final_results, 
                key=lambda x: 0.5 * x["scores"]["rerank"] + 0.5 * x["scores"]["hybrid"]
            )

