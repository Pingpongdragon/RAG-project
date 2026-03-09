from sentence_transformers import SentenceTransformer
from config import settings
import torch
import logging
import os

logger = logging.getLogger(__name__)

class SafeHuggingFaceEmbedder:
    def __init__(self):
        self._validate_environment()
        self.embedder = self._init_embedder()


    def _validate_environment(self):
        """执行前置环境校验"""
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA不可用，将回退到CPU模式")
        try:
            import sentence_transformers  # noqa
        except ImportError:
            raise RuntimeError("⚠️ 需要安装sentence-transformers: pip install sentence-transformers")

    def _init_embedder(self) -> SentenceTransformer:
        """带异常捕获的初始化流程"""
        try:
            return SentenceTransformer(
                settings.EMBEDDING_MODEL,
                # model_kwargs=settings.EMBEDDING_CONFIG["model_kwargs"],
                # encode_kwargs=self._get_encode_kwargs(),
                cache_folder=settings.CACHE_FOLDER,
                trust_remote_code=True,
                local_files_only=False,
            )
        except Exception as e:
            logger.error(f"❌ 嵌入模型初始化失败: {str(e)}")
            raise

    def _get_encode_kwargs(self) -> dict:
        """动态生成encode参数"""
        base_args = settings.EMBEDDING_CONFIG["encode_kwargs"]
        # 自动禁用batch加速当使用CPU时
        if "cuda" not in settings.EMBEDDING_CONFIG["model_kwargs"]["device"]:
            base_args["batch_size"] = 1  # CPU模式禁用批量处理
            logger.info("⚠️ CPU模式运行，批量处理已禁用")
        return base_args

# 全局单例访问点
embedding_service = SafeHuggingFaceEmbedder().embedder
