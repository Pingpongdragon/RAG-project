from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from typing import List
from config.logger_config import configure_console_logger

logger = configure_console_logger(__name__)

def validate_data_consistency(vector_db: FAISS, docs: List[Document]) -> bool:
    """检查文档数量是否匹配"""
    # Faiss索引自带文档数统计
    index_count = vector_db.index.ntotal  
    actual_count = len(docs)
    
    if index_count != actual_count:
        logger.error(f"❗数据不一致: 索引包含{index_count}条，文档集有{actual_count}条")
        return False
    return True
