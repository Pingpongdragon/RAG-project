from pathlib import Path
from ollama import Client
import torch

# -------------------------
# 路径配置
# -------------------------
DEFAULT_INDEX_PATH = Path("RAG_QA_index").absolute()

# -------------------------
# 模型配置
# -------------------------
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_CONFIG = {
    "model_kwargs": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True
    },
    "encode_kwargs": {"normalize_embeddings": True}
}

OLLAMA_MODEL = "llama3.1"
OLLAMA_CLIENT = Client(host='http://127.0.0.1:11434')
