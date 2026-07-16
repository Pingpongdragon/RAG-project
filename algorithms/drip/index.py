"""DRIP 使用的只读 cold-corpus inner-product index。"""

import numpy as np


class DenseIndex:
    """优先复用 FAISS；不可用时退化为 NumPy 精确检索。"""

    _shared = {}

    def __init__(self, document_embeddings):
        self.embeddings = np.ascontiguousarray(
            document_embeddings, dtype=np.float32
        )
        self.faiss = None
        try:
            import faiss

            key = (id(document_embeddings), self.embeddings.shape)
            self.faiss = self._shared.get(key)
            if self.faiss is None:
                self.faiss = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.faiss.add(self.embeddings)
                self._shared[key] = self.faiss
        except Exception:
            self.faiss = None

    def search(self, query_embedding, topk):
        topk = min(max(1, int(topk)), len(self.embeddings))
        query = np.ascontiguousarray(
            np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        )
        if self.faiss is not None:
            scores, indices = self.faiss.search(query, topk)
            return [
                (int(pool_idx), float(score))
                for pool_idx, score in zip(indices[0], scores[0])
                if int(pool_idx) >= 0
            ]

        similarities = query[0] @ self.embeddings.T
        selected = np.argpartition(similarities, -topk)[-topk:]
        selected = selected[np.argsort(similarities[selected])[::-1]]
        return [
            (int(pool_idx), float(similarities[pool_idx]))
            for pool_idx in selected
        ]
