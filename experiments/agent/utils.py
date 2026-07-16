"""Agent evidence-access replay 使用的文档编码工具。"""

import hashlib

import numpy as np

from experiments.agent.config import CACHE_DIR, EMBED_MODEL, log


def _input_fingerprint(texts):
    """为有序编码输入生成稳定指纹，防止误用旧 embedding cache。"""

    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def compute_document_embeddings(doc_pool):
    """只编码冷库文档，不把 history query 的检索质量混入驻留评估。"""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doc_texts = [f"{doc['title']}: {doc['text'][:256]}" for doc in doc_pool]
    cache_key = hashlib.md5(
        f"{_input_fingerprint(doc_texts)}_{EMBED_MODEL}_doc_v14".encode()
    ).hexdigest()[:12]
    cache_path = CACHE_DIR / f"de_{cache_key}.npy"
    if cache_path.exists():
        log.info("Loading cached document embeddings: %s", cache_path.name)
        return np.load(cache_path).astype("f")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    log.info("Encoding %d docs...", len(doc_texts))
    embeddings = model.encode(
        doc_texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, embeddings)
    return np.asarray(embeddings, dtype="f")

