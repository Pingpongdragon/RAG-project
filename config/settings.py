from pathlib import Path
from .logger_config import configure_console_logger
from dotenv import load_dotenv
import torch
import os

logger = configure_console_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# 修改这一行，使用 BASE_DIR 定位 .env 文件
load_dotenv(dotenv_path=BASE_DIR / ".env")


IS_MMLU = True  # 是否使用MMLU数据集

# -------------------------
# 数据集配置区
# -------------------------
KNOWLEDGE_DATASET_CONFIG = {
    "dataset_source": "huggingface",  # 数据集来源（hugginface或local）
    "dataset_name": "cais/mmlu",  # 数据集名称
    "split": "test",        # 使用的数据拆分
    "config_name": [
    "global_facts",
    "astronomy",
    "prehistory",
    "college_biology",
    "college_chemistry",
    "college_physics",
    "college_medicine",
    "clinical_knowledge",
    "philosophy",
    "world_religions"
  ],   # 预训练模型配置
    "text_columns": ["question","choices","answer"],# 文本字段名（支持多个）
    "id_column": "id",          # ID字段名
    
    # 索引构建参数
    "chunk_size": 1024,         # 文本切分块大小
    "chunk_overlap": 128,       # 切分块重叠长度
    "index_save_path": str(BASE_DIR / "data/mmlu_index"),  # 默认保存路径
}

DATA_CACHE_DIR = str(BASE_DIR / "data/raw_data")  # 数据集缓存目录

WMT_DIR = str(BASE_DIR / "streamingqa/WMT docs")  # WMT数据集目录
STREAMINGQA_DIR = str(BASE_DIR / "streamingqa/streamingQAdatasets")  # StreamingQA数据集目录


# -------------------------
# 检索参数配置区
# -------------------------
DEFAULT_DENSE_K = 10      # 向量检索候选数
DEFAULT_SPARSE_K = 10     # 关键词检索候选数
DENSE_SCORE_THRESHOLD = 0.6  # 向量初筛阈值
SPARSE_SCORE_THRESHOLD = 0.8  # 关键词初筛阈值
HYBRID_DENSE_WEIGHT = 0.6    # 混合权重
HYBRID_TOP_K = 5           # 混合后保留数量
DEFAULT_RERANK_K = 5      # 重排序数量
RERANK_THRESHOLD = 0.4  # 最终阈值
RANK_ORDER = 0 # 排序方式（0:降序, 1:升序）


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
CACHE_FOLDER = str(BASE_DIR / "models/") 

RERANKER_MODEL = "BAAI/bge-reranker-base"  # 默认模型名称


# -------------------------
# 推理配置
# -------------------------
try:
    from swift.llm import VllmEngine, PtEngine
    USE_SWIFT_LLM = True
except Exception as e:
    print(f"⚠️  Warning: swift.llm import failed ({e}). Local model inference disabled.")
    VllmEngine = None
    PtEngine = None
    USE_SWIFT_LLM = False
MODEL_DIR = str(BASE_DIR.parent / "llm/Qwen/Qwen3-8B")
MODEL_TYPE = None  # 模型类型
ADAPTER_DIR = None # 适配器目录

# 本地模型配置
MODEL_DIR = str(BASE_DIR.parent / "llm/Qwen/Qwen3-8B")
MODEL_TYPE = 'qwen25-32b-instruct'
ADAPTER_DIR = None

# DeepSeek API 配置
DEEPSEEK_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL"),
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
}

# Qwen-30B API 配置
QWEN_API_CONFIG = {
    "api_key": os.getenv("QWEN_API_KEY", "none"),
    "base_url": os.getenv("QWEN_BASE_URL"),
    "model": os.getenv("QWEN_MODEL_NAME"),
}


# 选择使用的模型类型: "local" | "deepseek" | "qwen"
ACTIVE_MODEL_TYPE = os.getenv("ACTIVE_MODEL_TYPE", "qwen")  # 默认使用 Qwen

print(f"当前使用的模型类型: {ACTIVE_MODEL_TYPE}")
print(f"Qwen API 配置: {QWEN_API_CONFIG}")

class ModelManager:
    def __init__(self):
        self.engine = None
        self.current_config = None
    
    def get_engine(self, model_dir=None, adapters=None, model_type=None):
        """获取推理引擎，如果配置发生变化则重新加载"""     
        config = (model_dir, adapters, model_type)
        
        # 如果配置不存在或已更改，则创建新引擎
        if not self.current_config or config != self.current_config:
            logger.info(f"加载模型配置: {model_dir}, 适配器: {adapters}")
            self.engine = PtEngine(model_dir, 
                                    adapters=adapters,
                                    # gpu_memory_utilization=0.5,
                                    max_batch_size=MAX_BATCH_SIZE,
                                      model_type=model_type)
        
        self.current_config = config
        return self.engine
    
    def reload_current(self):
        """强制重新加载当前配置的模型"""
        if self.current_config:
            model_dir, adapters, model_type = self.current_config
            logger.info(f"重新加载模型: {model_dir}, 适配器: {adapters}")
            self.engines[self.current_config] = PtEngine(model_dir, adapters=adapters, model_type=model_type)
            return self.engines[self.current_config]
        return None

# 创建全局模型管理器实例
# ✅ 仅在需要时初始化 ModelManager
if USE_SWIFT_LLM:
    model_manager = ModelManager()
else:
    model_manager = None

CONTEXT_TOP_N = 3  # 上下文数量
MAX_NEW_TOKEN = 2048  # 最大生成token数
MAX_BATCH_SIZE = 8  # 最大批量大小
TEMPERATURE = None  # 温度参数
MAX_RETRIES = 2  # 最大重试次数




# -------------------------
# RAG评估配置区
# -------------------------
EVAL_CONFIG = {
    # 测试问题集配置 (使用同一仓库下的questions split)
    "eval_dataset": {
        "name": "rag-datasets/rag-mini-wikipedia",
        "split": "test",         
        "config_name": "question-answer",
        "question_col": "question",  # 输入问题字段
        "answer_col": "answer"       # 参考答案字段
    },
    
    # RAGAS评估参数
    "metrics": ["answer_relevancy", "answer_correctness","context_recall","context_precision"],
}




# -------------------------
# Advanced RAG Strategies (参考 ottomator-agents/all-rag-strategies)
# -------------------------

# Query Expansion: 用 LLM 将简短 query 扩展为详细版本
ENABLE_QUERY_EXPANSION = False
QUERY_EXPANSION_TEMPERATURE = 0.3

# Multi-Query RAG: 生成多个查询变体并行检索
ENABLE_MULTI_QUERY = False
MULTI_QUERY_NUM_VARIATIONS = 3

# Self-Reflective RAG: 检索后自评注, 低分则修正查询重试
ENABLE_SELF_REFLECTION = False
SELF_REFLECTION_THRESHOLD = 3  # 1-5评分, 低于此值触发修正
SELF_REFLECTION_MAX_RETRIES = 1  # 最多修正几轮


# -------------------------
# ComRAG Hyperparameters (Paper Section 5.4)
# https://arxiv.org/abs/2506.21098
# -------------------------

# Cluster threshold: cosine similarity for cluster assignment
COMRAG_TAU = 0.75

# Near-duplicate threshold: sim >= delta -> direct reuse / replacement
COMRAG_DELTA = 0.9

# Quality boundary: score >= gamma -> V_high, else -> V_low
COMRAG_GAMMA = 0.6

# Adaptive temperature parameters (Section 4.4)
COMRAG_TEMP_K = 250.0       # Scaling factor for temperature function
COMRAG_TEMP_MIN = 0.7       # Minimum temperature (high variance -> consistency)
COMRAG_TEMP_MAX = 1.2       # Maximum temperature (low variance -> exploration)


# -------------------------
# ERASE Hyperparameters (Li et al., 2024)
# https://arxiv.org/abs/2406.11830
# -------------------------

# Retrieval threshold for inference (Appendix A.3)
ERASE_INFERENCE_THRESHOLD = 0.7

# Top-k facts retrieved for update step
ERASE_UPDATE_TOP_K = 20

# Top-k facts retrieved for inference
ERASE_INFERENCE_TOP_K = 10


# -------------------------
# ERASE Hyperparameters (Li et al., 2024)
# https://arxiv.org/abs/2406.11830
# -------------------------
ERASE_INFERENCE_THRESHOLD = 0.7
ERASE_UPDATE_TOP_K = 20
ERASE_INFERENCE_TOP_K = 10


# -------------------------
# QARC Hyperparameters (Query-Aligned Retrieval-augmented Knowledge Curation)
# Our proposed framework — three-phase adaptive KB curation
# -------------------------

# Window parameters
QARC_WINDOW_SIZE = 50           # Queries per window (W_size)

# Phase 1 (Explore) parameters
QARC_N_WARMUP_MIN = 5           # Minimum windows before Phase 1→2 transition
QARC_EPSILON_SIGMA = 0.3        # Convergence threshold for Gap variance ratio
QARC_EXPLORE_LAMBDA_MAX = 0.5   # Replacement ratio in Phase 1 (aggressive)
QARC_EXPLORE_ETA = 0.0          # Diversity term in Phase 1 (pure interest)

# Phase 2 (Exploit) parameters
QARC_EXPLOIT_LAMBDA_MAX = 0.2   # Replacement ratio in Phase 2 (conservative)
QARC_EXPLOIT_ETA = 0.1          # Diversity regularization in Phase 2
QARC_COOLDOWN_WINDOWS = 3       # Cooldown windows after re-curation

# Adaptive threshold (EMA + k·MAD)
QARC_THRESHOLD_BETA = 0.9       # EMA smoothing factor
QARC_THRESHOLD_K = 2.0          # MAD multiplier (sensitivity)

# Re-explore trigger
QARC_RE_EXPLORE_TRIGGER = 3     # Consecutive Phase 2 triggers before re-explore

# KB parameters
QARC_KB_BUDGET = 50             # Maximum KB documents
QARC_CANDIDATE_TOP_K = 100      # Candidates per interest centroid from pool

# Retrieval parameters
QARC_RETRIEVE_TOP_K = 5         # Docs retrieved per query for RAG
