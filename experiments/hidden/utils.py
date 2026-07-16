"""Hidden-evidence 与多跳 QA 实验的共享检索工具。

stream 构造和初始 KB 协议统一放在 ``core.utils``。这里仅保留 embedding 缓存与
Recall@K，旧的 KMeans/head-tail、bridge-reuse 手工流和未来标签初始化已移除。
"""

import hashlib

import numpy as np

from config import BGE_QUERY_PREFIX, CACHE_DIR, EMBED_MODEL, K_LIST, log


def _embedding_input_fingerprint(texts):
    """为编码器的有序输入生成稳定指纹。"""

    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def compute_document_embeddings(doc_pool):
    """编码冷库文档，且不强制构造当前实验不需要的 query embedding。

    部分 hidden 实验只需要冷库文档表示；把这个步骤独立出来可避免无条件编码
    当前协议不使用的 query embedding。
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doc_texts = [f"{doc['title']}: {doc['text'][:256]}" for doc in doc_pool]
    doc_key = hashlib.md5(
        f"{_embedding_input_fingerprint(doc_texts)}_{EMBED_MODEL}_doc_v14".encode()
    ).hexdigest()[:12]
    doc_cache = CACHE_DIR / f"de_{doc_key}.npy"
    doc_embs = np.load(doc_cache).astype("f") if doc_cache.exists() else None
    if doc_embs is not None:
        log.info("Loading cached document embeddings: %s", doc_cache.name)
        return doc_embs

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    log.info("Encoding %d docs...", len(doc_texts))
    doc_embs = model.encode(
        doc_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(doc_cache, doc_embs)
    return np.asarray(doc_embs, dtype="f")


def compute_embeddings(doc_pool, queries, tag):
    """编码文档和 query；两类向量按各自真实输入独立缓存。"""

    del tag  # cache key 已绑定完整输入，不再依赖容易冲突的人工 tag。
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    query_prefix = BGE_QUERY_PREFIX if "bge" in EMBED_MODEL.lower() else ""
    query_texts = [query_prefix + query["question"] for query in queries]
    query_key = hashlib.md5(
        f"{_embedding_input_fingerprint(query_texts)}_{EMBED_MODEL}_query_v14".encode()
    ).hexdigest()[:12]
    query_cache = CACHE_DIR / f"qe_{query_key}.npy"

    doc_embs = compute_document_embeddings(doc_pool)
    query_embs = (
        np.load(query_cache).astype("f") if query_cache.exists() else None
    )
    if query_embs is not None:
        log.info("Loading cached query embeddings: %s", query_cache.name)
        return doc_embs, query_embs

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    unique_texts = list(dict.fromkeys(query_texts))
    text_to_index = {text: index for index, text in enumerate(unique_texts)}
    log.info(
        "Encoding %d unique queries (%d events)...",
        len(unique_texts),
        len(query_texts),
    )
    unique_embs = model.encode(
        unique_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    inverse = np.asarray(
        [text_to_index[text] for text in query_texts], dtype=np.int64
    )
    query_embs = np.asarray(unique_embs)[inverse]
    np.save(query_cache, query_embs)

    return np.asarray(doc_embs, dtype="f"), np.asarray(query_embs, dtype="f")


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
