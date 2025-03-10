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

OLLAMA_MODEL = "llama3.1"
OLLAMA_CLIENT = Client(host='http://127.0.0.1:11434')
