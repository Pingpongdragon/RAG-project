"""DRIP cache manager 使用的 dense embedding index。"""
import numpy as np


class EmbeddingIndex:
    """用于 query-visible direct candidate 的语义检索。"""

    def __init__(self, doc_embs):
        self.doc_embs = doc_embs

    def search_one(self, query_emb, topk):
        topk = min(max(1, int(topk)), self.doc_embs.shape[0])
        sims = query_emb @ self.doc_embs.T
        top = np.argpartition(sims, -topk)[-topk:]
        top = top[np.argsort(sims[top])[::-1]]
        return [(int(pi), float(sims[pi])) for pi in top]
