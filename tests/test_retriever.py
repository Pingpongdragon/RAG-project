import pytest
from core.retriever import QARetriever
from core.data_processor import load_or_build_index

def test_similarity_search():
    retriever = QARetriever(load_or_build_index())
    query = "Was Abraham Lincoln the sixteenth President of the United States?"
    results = retriever.similarity_search(query, threshold=0.7)
    print(results)
