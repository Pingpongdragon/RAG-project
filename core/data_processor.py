from typing import List, Union, Tuple
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from datasets import load_dataset
from models.embeddings import embedding_service
from config import settings
import pickle
from config.logger_config import configure_logger
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import CharacterTextSplitter

logger = configure_logger(__name__)  # 使用统一配置的logger

# -------------------------
# 数据集处理相关
# -------------------------
def _load_hf_dataset() -> List[Document]:
    """根据配置加载数据集（适配当前测试集结构）"""
    
    # 从配置读取参数
    cfg = settings.DATASET_CONFIG
    logger.info(f"🚀 正在加载数据集 {cfg['dataset_name']}")
    
    try:
        # 自动检测本地缓存路径
        cache_dir = Path(settings.DATA_CACHE_DIR) 
        raw_data = load_dataset(
            cfg['dataset_name'], 
            cfg['config_name'],
            split=cfg['split'],
            cache_dir=str(cache_dir)
        )
    except Exception as e:
        logger.error(f"🔥 数据集加载失败: {str(e)}")
        return []

    # 有效性校验（动态检查字段）
    required_columns = cfg['text_columns'] + [cfg['id_column']]
    valid_items = [
        item for item in raw_data 
        if all(k in item for k in required_columns)
    ]
    invalid_count = len(raw_data) - len(valid_items)
    
    logger.info(f"📊 加载完成: 有效 {len(valid_items)}条, 跳过无效 {invalid_count}条")
    
    # 组合多个文本字段
    return [
        Document(
            page_content= "\n".join(str(item[col]) for col in cfg['text_columns']),
            metadata={
                "doc_id": item[cfg['id_column']],
                "source": cfg['dataset_name']
            }
        ) for item in valid_items
    ]


def _build_hybrid_vector_index(docs: List[Document]) -> FAISS:
    """基于纯文本的混合索引构建"""
    
    cfg = settings.DATASET_CONFIG
    text_splitter = CharacterTextSplitter(
        chunk_size=cfg['chunk_size'],
        chunk_overlap=cfg['chunk_overlap']
    )
    
    processed = []
    for doc in docs:
        try:
            # 直接分块原始文本内容
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                processed.append(Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc.metadata["doc_id"],
                        "chunk_id": i,
                        "source": doc.metadata["source"]
                    }
                ))
        except KeyError as e:
            logger.error(f"元数据字段缺失: {e} 于文档 {doc.metadata}")
        except Exception as e:
            logger.error(f"处理异常: {str(e)}")
    
    logger.info(f"🎯 生成 {len(processed)} 个分块")
    return FAISS.from_documents(
        processed, 
        embedding_service,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )




# -------------------------
# 索引管理主函数
# -------------------------
def load_or_build_index() -> Tuple[FAISS, List[Document]]:
    """自动化索引全生命周期管理"""
    
    cfg = settings.DATASET_CONFIG
    index_path = Path(cfg['index_save_path'])
    docs_file = Path(settings.DATA_CACHE_DIR)  / "docs.pkl"
    
    # 当存在完整索引时加载
    if (index_path / "index.faiss").exists() and docs_file.exists():
        logger.info(f"🔄 加载现有索引: {index_path}")
        vector_db = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embedding_service,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        with open(docs_file, "rb") as f:
            docs = pickle.load(f)
        return vector_db, docs
    
    # 新建索引流程
    logger.warning("⚠️ 无可用索引，启动构建流程...")
    try:
        docs = _load_hf_dataset()
        if not docs:
            raise ValueError("数据集加载后为空，请检查配置")
            
        index = _build_hybrid_vector_index(docs)
        
        # 保证保存路径存在
        index_path.mkdir(parents=True, exist_ok=True)
        index.save_local(str(index_path))
        
        with open(docs_file, "wb") as f:
            pickle.dump(docs, f)
            
        logger.success(f"✅ 新索引保存至项目下目录 {index_path}")
        return index, docs
    except Exception as e:
        logger.critical(f"❌ 索引构建失败: {str(e)}", exc_info=True)
        raise


