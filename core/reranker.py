# reranker.py
from typing import List, Dict
from sentence_transformers import CrossEncoder
from RAG_project.config.settings import RERANKER_MODEL,CACHE_FOLDER
import numpy as np

class ReRanker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            cache_dir=CACHE_FOLDER
        )
        self._warm_up()
    
    def _warm_up(self):
        """预热模型防止首次调用延迟"""
        self.model.predict([("warmup query", "warmup passage")])
    
    def rerank(
        self, 
        query: str, 
        passages: List[str], 
        top_k: int = 10
    ) -> List[Dict]:
        """
        Args:
            query: 用户查询语句
            passages: 待重排段落列表
            top_k: 返回结果数量
            
        Returns:
            排序后结果，格式: [{
                "text": 文本内容,
                "rerank_score": 重排序分数,
                "metadata": 附加信息
            }]
        """
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs, batch_size=32)
        
        sorted_indices = np.argsort(scores)[::-1]  # 降序排列
        return [{
            "text": passages[i],
            "rerank_score": float(scores[i]),
            "metadata": {}
        } for i in sorted_indices[:top_k]]
