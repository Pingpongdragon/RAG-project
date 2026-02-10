from pathlib import Path
from .logger_config import configure_console_logger
from swift.llm import VllmEngine,PtEngine
import torch

logger = configure_console_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

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
MODEL_DIR = str(BASE_DIR.parent / "llm/Qwen/Qwen3-8B")
# MODEL_DIR = "/home/users/zhangxx/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B"
MODEL_TYPE = None  # 模型类型
ADAPTER_DIR = None # 适配器目录

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
model_manager = ModelManager()

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


