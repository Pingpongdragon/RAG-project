"""
公共工具函数: 数据加载、评估、文本处理
"""
import sys
import os
import gc
import json
import string
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

# ==========================================
# 路径设置
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ==========================================
# 文本处理
# ==========================================
def normalize_answer(s: str) -> str:
    exclude = set(string.punctuation)
    return ' '.join(''.join(ch for ch in s.lower() if ch not in exclude).split())


def calculate_containment(prediction: str, ground_truths) -> float:
    norm_pred = normalize_answer(prediction)
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if len(norm_gt) >= 2 and norm_gt in norm_pred:
            return 1.0
    return 0.0


# ==========================================
# RAG Pipeline (按需导入，避免无 GPU 时报错)
# ==========================================
def build_pipeline(docs, model_name: str, use_hybrid: bool = False):
    """构建 RAG pipeline"""
    from config import settings
    from core.data_processor import _build_hybrid_vector_index
    from core.retriever import QARetriever

    gc.collect()
    torch.cuda.empty_cache()
    settings.EMBEDDING_MODEL = model_name
    settings.KNOWLEDGE_DATASET_CONFIG = {'chunk_size': 512, 'chunk_overlap': 50}
    settings.TEMPERATURE = 0.01
    if not docs:
        return None
    vector_db = _build_hybrid_vector_index(docs)
    settings.DEFAULT_RERANK_K = 5
    return QARetriever(vector_db=vector_db, docs=docs, hybrid_search=use_hybrid)


def evaluate_batch(retriever, test_set: List[Dict], batch_size: int = 32) -> float:
    """批量评估准确率"""
    from core.generator import generate_batch_llm_response

    if not test_set or not retriever:
        return 0.0

    total_acc = 0
    num_batches = (len(test_set) + batch_size - 1) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(test_set))
        batch = test_set[start:end]

        qs = [x['q'] for x in batch]
        golds = [x['a'] for x in batch]

        ctxs = []
        for q in qs:
            try:
                res = retriever.retrieve(q, rerank_top_k=5)
                ctxs.append([{"text": r['text'], "score": r['scores']['rerank']} for r in res])
            except Exception:
                ctxs.append([])

        try:
            resps = generate_batch_llm_response(qs, ctxs, language="en")
        except Exception:
            resps = [""] * len(qs)

        for r, g in zip(resps, golds):
            total_acc += calculate_containment(r, g)

    return total_acc / len(test_set)


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {path}")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)