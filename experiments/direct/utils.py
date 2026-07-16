"""Direct-evidence 实验的共享检索工具。

数据流构造与因果初始化统一放在 ``core.utils``；本文件只负责 embedding 缓存和
Recall@K，避免实验 runner 再藏一套旧的 KMeans/head-tail 构造逻辑。
"""

import hashlib

import numpy as np

if __package__:
    from .config import BGE_QUERY_PREFIX, CACHE_DIR, EMBED_MODEL, K_LIST, log
else:
    from config import BGE_QUERY_PREFIX, CACHE_DIR, EMBED_MODEL, K_LIST, log
from experiments.common.stream_protocol import embedding_content_fingerprint


def compute_embeddings(doc_pool, queries, tag):
    """编码文档和 query，并按真实输入内容缓存归一化向量。"""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    content_key = embedding_content_fingerprint(doc_pool, queries)
    key = hashlib.md5(
        f"{content_key}_{tag}_{EMBED_MODEL}_v13".encode()
    ).hexdigest()[:12]
    doc_cache = CACHE_DIR / f"de_{key}.npy"
    query_cache = CACHE_DIR / f"qe_{key}.npy"
    if doc_cache.exists() and query_cache.exists():
        log.info("Loading cached embeddings")
        return np.load(doc_cache).astype("f"), np.load(query_cache).astype("f")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    log.info("Encoding %d docs...", len(doc_pool))
    doc_embs = model.encode(
        [f"{doc['title']}: {doc['text'][:256]}" for doc in doc_pool],
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    query_prefix = BGE_QUERY_PREFIX if "bge" in EMBED_MODEL.lower() else ""
    log.info("Encoding %d queries...", len(queries))
    query_embs = model.encode(
        [query_prefix + query["question"] for query in queries],
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(doc_cache, doc_embs)
    np.save(query_cache, query_embs)
    return doc_embs.astype("f"), query_embs.astype("f")


def recall_at_k(
    kb_doc_ids,
    queries,
    doc_id_to_pool_index,
    doc_embs,
    title_to_idx,
    query_embs,
    k_list=K_LIST,
):
    """在当前 hot cache 内执行统一 cosine retrieval，并计算 Recall@K。"""

    if not kb_doc_ids:
        return {k: 0.0 for k in k_list}
    kb_doc_ids = sorted(kb_doc_ids)
    kb_indices = [doc_id_to_pool_index[doc_id] for doc_id in kb_doc_ids]
    kb_embs = doc_embs[kb_indices]
    pool_index_to_title = {value: key for key, value in title_to_idx.items()}
    local_index_to_title = {
        local_index: pool_index_to_title[pool_index]
        for local_index, pool_index in enumerate(kb_indices)
        if pool_index in pool_index_to_title
    }
    max_k = max(k_list)
    recalls = {k: [] for k in k_list}
    for query in queries:
        gold = set(query["sf_titles"])
        if not gold:
            continue
        similarities = kb_embs @ query_embs[int(query["qidx"])]
        if len(similarities) <= max_k:
            top = np.argsort(similarities)[::-1]
        else:
            top = np.argpartition(similarities, -max_k)[-max_k:]
            top = top[np.argsort(similarities[top])[::-1]]
        for k in k_list:
            retrieved = {
                local_index_to_title[index]
                for index in top[:k]
                if index in local_index_to_title
            }
            recalls[k].append(len(gold & retrieved) / len(gold))
    return {
        k: float(np.mean(values)) if values else 0.0
        for k, values in recalls.items()
    }
