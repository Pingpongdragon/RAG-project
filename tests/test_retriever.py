import pytest
from core.retriever import QARetriever
from config.settings import EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings

@pytest.fixture
def sample_retriever():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # 这里可以加载测试用的小型索引
    return QARetriever(...)

def test_similarity_search(sample_retriever):
    results = sample_retriever.similarity_search("测试查询")
    assert len(results) > 0, "应该返回至少一个结果"
