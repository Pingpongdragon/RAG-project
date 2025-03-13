from pathlib import Path
from ollama import Client
import torch

# -------------------------
# 数据集配置区
# -------------------------
DATASET_CONFIG = {
    "dataset_name": "rag-datasets/rag-mini-wikipedia",  # HF数据集名称
    "split": "passages",        # 使用的数据拆分
    "config_name": "text-corpus",   # 预训练模型配置
    "text_columns": ["passage"],# 文本字段名（支持多个）
    "id_column": "id",          # ID字段名
    
    # 索引构建参数
    "chunk_size": 1024,         # 文本切分块大小
    "chunk_overlap": 128,       # 切分块重叠长度
    "index_save_path": "./data/RAG_Wikipedia_index",  # 默认保存路径
}

DATA_CACHE_DIR = "./data/raw_data"  # 数据集缓存目录


# -------------------------
# 检索参数配置区
# -------------------------
DEFAULT_DENSE_K = 10      # 向量检索候选数
DEFAULT_SPARSE_K = 10     # 关键词检索候选数
DENSE_SCORE_THRESHOLD = 0.65  # 向量初筛阈值
SPARSE_SCORE_THRESHOLD = 0.65  # 关键词初筛阈值
HYBRID_DENSE_WEIGHT = 0.7    # 混合权重
HYBRID_TOP_K = 30           # 混合后保留数量
DEFAULT_RERANK_K = 10      # 重排序数量
FINAL_SCORE_THRESHOLD = 0.65  # 最终阈值


# -------------------------
# 生成参数配置区
# -------------------------
GENERATION_CONFIG = {
    "max_retries": 3,            # 最大重试次数
    "temperature": 0.4,          # 温度参数（控制生成随机性）
    "max_tokens": 300,           # 生成最大token数
    "context_top_n": 3,          # 上下文取前N条
    "top_p": 0.9,                # 核采样参数
    "stream": False,             # 是否流式输出
}

# -------------------------
# 模型配置
# -------------------------
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_CONFIG = {
    "model_kwargs": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,
        "revision":"ac6fcd72429d86ff25c17895e47a9bfcfc50c1b2",
        "model_kwargs": {
            "weights_only":True,
        }
    },
    "encode_kwargs": {"normalize_embeddings": True},
}
CACHE_FOLDER = "./models" 

RERANKER_MODEL = "BAAI/bge-reranker-base"  # 默认模型名称

OLLAMA_MODEL = "llama3.1"
OLLAMA_CLIENT = Client(host='http://127.0.0.1:11434')
