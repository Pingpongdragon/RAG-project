from typing import List, Union
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from datasets import load_dataset
from models.embeddings import embedding_service
from config import settings
from config.logger_config import configure_logger

logger = configure_logger(__name__)  # 使用统一配置的logger

# -------------------------
# 数据集处理相关
# -------------------------
def _load_hf_dataset(dataset_name: str) -> List[Document]:
    """数据加载与基础处理 (带完整日志)"""
    logger.info(f"🚀 正在加载数据集: {dataset_name}")
    try:
        raw_data = load_dataset(dataset_name, "question-answer", split="test")
    except Exception as e:
        logger.error(f"🔥 数据集加载失败: {str(e)}")
        return []

    valid_items = [item for item in raw_data 
                  if all(k in item for k in ["id", "question", "answer"])]
    invalid_count = len(raw_data) - len(valid_items)
    
    logger.info(f"📊 有效记录: {len(valid_items)}条, 无效跳过: {invalid_count}条")
    
    return [
        Document(
            page_content=f"Question:{item['question']}",
            metadata={
                "q_id": item["id"],
                "question": item["question"],
                "raw_answer": item["answer"],  # 完整保留原始回答
                "source": dataset_name
            }
        ) for item in valid_items
    ]

def _build_hybrid_vector_index(docs: List[Document]) -> FAISS:
    """问答融合索引构建 (使用raw_answer)"""
    from langchain.text_splitter import CharacterTextSplitter
    
    logger.debug("🔧 初始化混合索引处理器")
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    
    processed = []
    for doc in docs:
        try:
            answer_chunk = text_splitter.split_text(doc.metadata["raw_answer"])[0]  # 直接使用原始回答
            processed.append(Document(
                page_content=f"Q:{doc.metadata['question']}\nA:{answer_chunk}",
                metadata={
                    "source": "hybrid-index",
                    "origin_id": doc.metadata["q_id"]
                }
            ))
            logger.debug(f"✅ 成功处理文档: {doc.metadata['q_id']}")
        except Exception as e:
            logger.warning(f"⚠️ 处理异常 {doc.metadata.get('q_id')}: {str(e)}")
    
    logger.info(f"🎯 完成索引构建，总文档数: {len(processed)}")
    return FAISS.from_documents(processed, embedding_service)



# -------------------------
# 索引管理主函数
# -------------------------
def load_or_build_index(
    index_path: Union[str, Path, None] = None
) -> FAISS:
    index_path = Path(index_path) if index_path else settings.DEFAULT_INDEX_PATH
    
    if index_path.exists():
        logger.info(f"📂 加载现有索引: {index_path}")
        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embedding_service,  # 使用全局单例
            allow_dangerous_deserialization=True
        )
    
    logger.warning("⚠️ 未找到现有索引，开始新构建流程...")
    try:
        docs = _load_hf_dataset("rag-datasets/rag-mini-wikipedia")
        index = _build_hybrid_vector_index(docs)
        index.save_local(str(index_path))
        logger.success(f"🎉 新索引已保存至 {index_path}")
        return index
    except Exception as e:
        logger.critical(f"❌ 索引构建失败: {str(e)}", stack_info=True)  # 添加详细堆栈
        raise

