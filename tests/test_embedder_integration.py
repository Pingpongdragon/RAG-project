from core.data_processor import load_or_build_index

def test_full_pipeline():
    # 首次运行构建索引
    index_v1 = load_or_build_index()
    assert len(index_v1.index_to_docstore_id) > 0
    
    # 二次运行加载索引
    index_v2 = load_or_build_index()
    assert index_v1.index_to_docstore_id == index_v2.index_to_docstore_id

    # 获取底层的 FAISS 索引对象
    faiss_index = index_v2.index

    # 查看索引中向量总数
    print(f"NTOTAL: {faiss_index.ntotal}")

    # # 如果想遍历并查看每条向量，可以 reconstruct
    # for i in range(faiss_index.ntotal):
    #     vector = faiss_index.reconstruct(i)
    #     print(f"向量 {i}:", vector)

    # 如果想查看对应的文本或元数据，可访问 docstore
    for doc_id in index_v2.docstore._dict:
        print(f"文档ID: {doc_id}")
        print("内容:", index_v2.docstore.search(doc_id).page_content)
    
    print(f"✅ 测试通过")
