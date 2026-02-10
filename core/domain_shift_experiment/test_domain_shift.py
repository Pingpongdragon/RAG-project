import json
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# 向量库与模型
import faiss
from sentence_transformers import SentenceTransformer

os.environ["HF-ENDPOINT"] = "https://hf-mirror.com"

# ================= 配置 =================
DATA_ROOT = Path("./data")
FIQA_DIR = DATA_ROOT / "raw_data" / "fiqa"
MODEL_NAME = "all-MiniLM-L6-v2"  # 一个典型的通用小模型，容易在专业领域翻车
SAMPLE_SIZE = 100000  # 为了速度，只取前2000个文档做实验
TEST_QUERIES_COUNT = 500 # 测试50个问题

def load_fiqa_data():
    """读取 FiQA 的原始 JSONL 数据"""
    print("📖 正在加载语料库 (Corpus)...")
    corpus = {}
    with open(FIQA_DIR / "corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc['text']
            if len(corpus) >= SAMPLE_SIZE: break
    
    print("📖 正在加载查询集 (Queries)...")
    queries = {}
    with open(FIQA_DIR / "queries.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
            
    print("📖 正在加载标准答案 (Qrels)...")
    qrels = {} # query_id -> [doc_id, ...]
    with open(FIQA_DIR / "qrels" / "test.tsv", 'r', encoding='utf-8') as f:
        next(f) # skip header
        for line in f:
            qid, docid, score = line.strip().split('\t')
            if qid not in qrels: qrels[qid] = []
            qrels[qid].append(docid)
            
    return corpus, queries, qrels

def run_experiment():
    print(f"\n🔬 启动 Domain Shift 诊断实验 (Model: {MODEL_NAME})")
    print("-" * 60)
    
    # 1. 加载模型
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. 准备数据
    corpus, queries, qrels = load_fiqa_data()
    
    # 过滤：只保留我们在 Sample Corpus 里有答案的问题
    valid_qids = []
    for qid, docids in qrels.items():
        if qid in queries and any(did in corpus for did in docids):
            valid_qids.append(qid)
    
    test_qids = valid_qids[:TEST_QUERIES_COUNT]
    print(f"📊 实验规模: 文档库 {len(corpus)} 条, 测试问题 {len(test_qids)} 个")

    if len(test_qids) == 0:
        print("❌ 错误: 采样太小，没有找到匹配的问题-文档对，请增加 SAMPLE_SIZE。")
        return

    # 3. [Dynamic Indexing] 模拟更新向量库
    print("\n🏗️ [Step 1] 正在构建动态索引 (Embedding + Indexing)...")
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())
    
    # 编码
    doc_embeddings = model.encode(doc_texts, show_progress_bar=True, batch_size=32)
    
    # 建库
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product (Sim to Cosine if normalized)
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)
    
    print(f"✅ 索引构建完成。")

    # 4. [Retrieval Evaluation] 评估检索效果
    print("\n🔍 [Step 2] 执行检索并评估 Hit Rate...")
    
    hits = 0
    mrr_sum = 0
    top_k = 10
    
    failure_cases = []
    success_cases = []

    for qid in tqdm(test_qids):
        query_text = queries[qid]
        target_doc_ids = set(qrels[qid])
        
        # 编码查询
        q_emb = model.encode([query_text])
        faiss.normalize_L2(q_emb)
        
        # 搜索
        distances, indices = index.search(q_emb, k=top_k)
        retrieved_indices = indices[0]
        
        # 检查命中
        is_hit = False
        for rank, idx in enumerate(retrieved_indices):
            if idx == -1: continue
            retrieved_doc_id = doc_ids[idx]
            
            if retrieved_doc_id in target_doc_ids:
                hits += 1
                mrr_sum += 1.0 / (rank + 1)
                is_hit = True
                
                # 记录一个成功案例
                if len(success_cases) < 2:
                    success_cases.append({
                        "q": query_text, 
                        "d": corpus[retrieved_doc_id][:100] + "..."
                    })
                break
        
        if not is_hit:
            # 记录失败案例用于分析
            if len(failure_cases) < 3:
                # 找到正确答案的内容（如果有的话）
                correct_doc_content = "N/A"
                for did in target_doc_ids:
                    if did in corpus:
                        correct_doc_content = corpus[did][:100] + "..."
                        break
                
                failure_cases.append({
                    "q": query_text,
                    "target_doc": correct_doc_content,
                    "retrieved_doc": corpus[doc_ids[retrieved_indices[0]]][:100] + "..."
                })

    # 5. 计算指标
    hit_rate = hits / len(test_qids)
    mrr = mrr_sum / len(test_qids)

    print("\n" + "="*60)
    print(f"📈 实验结果 (Domain: Finance/FiQA)")
    print(f"   Hit Rate @ {top_k}: {hit_rate:.2%}  (目标文档在前10个结果里出现的概率)")
    print(f"   MRR @ {top_k}     : {mrr:.4f}     (平均倒数排名)")
    print("="*60)

    # 6. 自动诊断与建议
    print("\n🩺 [诊断报告]")
    
    if hit_rate > 0.7:
        print("✅ 结论: 单纯更新 Vectorbase 足够有效。")
        print("   原因: 通用模型在这个子领域表现尚可，术语重叠度较高。")
    elif hit_rate > 0.4:
        print("⚠️ 结论: 效果一般，建议尝试微调检索器 (Retriever Fine-tuning)。")
        print("   原因: 模型能懂部分内容，但在处理专业匹配时有困难。")
    else:
        print("❌ 结论: 单纯更新 Vectorbase 失败！必须微调检索器或使用领域专用模型。")
        print("   原因: 发生了严重的 Semantic Shift，模型完全抓瞎。")

    print("\n📝 [失败案例分析 - 为什么需要微调?]")
    for i, case in enumerate(failure_cases):
        print(f"\nCase {i+1}:")
        print(f"❓ Query: {case['q']}")
        print(f"✅ 目标文档 (模型没选): {case['target_doc']}")
        print(f"❌ 检索结果 (模型选了): {case['retrieved_doc']}")
        print("   👉 分析: 如果目标文档里没有Query的关键词（纯语义匹配），通用模型通常会挂。")

if __name__ == "__main__":
    if not (FIQA_DIR / "corpus.jsonl").exists():
        print("❌ 数据未找到。请先运行 01_download_real_data.py")
    else:
        run_experiment()