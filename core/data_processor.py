from typing import List, Union
from pathlib import Path
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from datasets import load_dataset
from config import settings
from config.logger_config import configure_logger

logger = configure_logger(__name__)  # 使用统一配置的logger

# -------------------------
# 数据集处理相关
# -------------------------
def _load_hf_dataset(dataset_name: str) -> List[Document]:
    """私有方法：加载并转换数据集"""
    logger.debug(f"🔍 开始加载数据集: {dataset_name}")
    raw_data = load_dataset(dataset_name, "question-answer", split="test")
    
    docs = []
    for item in raw_data:
        try:
            doc = Document(
                page_content=f"Question: {item['question']}\nAnswer: {item['answer']}",
                metadata={
                    "q_id": int(item["id"]),
                    "source": "rag-wiki-qa",
                    "question": item["question"]
                }
            )
            docs.append(doc)
        except KeyError as e:
            logger.error(f"❌ 数据格式错误，跳过条目ID {item.get('id', 'unknown')}: {str(e)}")
    
    logger.info(f"✅ 成功加载 {len(docs)} 条文档")
    return docs

def _build_hybrid_vector_index(docs: List[Document], embedder: HuggingFaceEmbeddings) -> FAISS:
    """私有方法：构建问题-答案分离的索引结构"""
    from langchain.text_splitter import CharacterTextSplitter  # 延迟导入减少内存开销
    
    logger.debug("🛠️ 开始构建混合索引...")
    
    # 配置分割器
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=30
    )
    
    processed_docs = []
    for doc in docs:
        try:
            # 分割问题与答案
            question = doc.metadata["question"]
            answer = doc.page_content.split("\nAnswer: ")[1]
            
            # 生成两种类型文档
            processed_docs.extend([
                Document(
                    page_content=question,
                    metadata={**doc.metadata, "content_type": "question"}
                ),
                Document(
                    page_content=text_splitter.split_text(answer)[0],  # 取第一段答案
                    metadata={**doc.metadata, "content_type": "answer"}
                )
            ])
        except Exception as e:
            logger.warning(f"⚠️ 文档处理失败ID {doc.metadata.get('q_id', 'unknown')}: {str(e)}")
    
    logger.info(f"📊 构建索引使用文档数: {len(processed_docs)}")
    return FAISS.from_documents(processed_docs, embedder)

# -------------------------
# 索引管理主函数
# -------------------------
def load_or_build_index(
    embedder: Union[HuggingFaceEmbeddings, None] = None,
    index_path: Union[str, Path, None] = None
) -> FAISS:
    """智能索引管理系统"""
    # 参数默认值处理
    if embedder is None:
        embedder = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            **settings.EMBEDDING_CONFIG
        )
    
    index_path = Path(index_path) if index_path else settings.DEFAULT_INDEX_PATH
    
    # 存在性检查
    if index_path.exists():
        try:
            logger.info(f"📂 加载现有索引: : {index_path}")
            return FAISS.load_local(
                folder_path=str(index_path),
                embeddings=embedder,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"🚨 索引加载失败: {str(e)}")
            raise RuntimeError("索引文件可能已损坏，请删除后重新构建") from e
    
    # 新建索引流程
    logger.warning("⚠️ 未找到现有索引，开始新构建流程...")
    try:
        docs = _load_hf_dataset("rag-datasets/rag-mini-wikipedia")
        index = _build_hybrid_vector_index(docs, embedder)
        index.save_local(str(index_path))
        logger.success(f"🎉 新索引已保存至 {index_path}")  # 使用增强日志
        return index
    except Exception as e:
        logger.critical(f"🔥 索引构建失败: {str(e)}")
        raise
