from typing import List
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from config.logger_config import configure_logger

logger = configure_logger(__name__)

class QARetriever:
    def __init__(self, vector_db: FAISS):
        self.vector_db = vector_db
        self.embedder = vector_db.embeddings
        self.retriever = self._configure_retriever()
    
    def _configure_retriever(self):
        return self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 15, "score_threshold": 0.6}
        )
    
    def similarity_search(self, query: str, threshold: float = 0.65) -> List[dict]:
        """带余弦验证的增强检索"""
        candidates = self.retriever.invoke(query)
        query_embedding = np.array(self.embedder.embed_query(query))
        
        results = []
        for doc in candidates:
            doc_embedding = np.array(self.embedder.embed_query(doc.page_content))
            similarity = np.dot(query_embedding, doc_embedding)
            if similarity >= threshold:
                results.append({
                    "text": doc.page_content,
                    "similarity": round(float(similarity), 4),
                    "metadata": doc.metadata
                })
        logger.info(f"🔍 检索到 {len(results)} 条符合条件的结果")
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
