from config import settings
from config.logger_config import configure_logger
from core.data_processor import load_or_build_index
from core.retriever import QARetriever
from core.generator import generate_llm_response
from core.utils.validation_utils import validate_data_consistency

logger = configure_logger(__name__)

def main():
    # 核心逻辑保持最简
    vector_db, docs = load_or_build_index()
    if not validate_data_consistency(vector_db, docs):
        raise RuntimeError("索引与文档数据不匹配")
    retriever = QARetriever(vector_db, docs)
    
    test_queries = [
        "Was Abraham Lincoln the sixteenth President of the United States?",
        "量子计算机的主要优势是什么？",
    ] # 测试查询
    
    for query in test_queries:
        context = retriever.retrieve(query)
        answer = generate_llm_response(query, context)
        logger.info(f"Q: {query}\nA: {answer}")

if __name__ == "__main__":
    main()
