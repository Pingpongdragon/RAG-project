import sys
import os
from pathlib import Path
import random
import re
import string
import matplotlib.pyplot as plt
import torch
import gc
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from langchain.schema import Document

# ================= 路径自动配置 =================
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
while not (project_root / "core").exists():
    if project_root == project_root.parent: 
        project_root = current_file_path.parent 
        break
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 导入 core
import core.generator
from core.data_processor import _build_hybrid_vector_index
from core.retriever import QARetriever
from core.generator import generate_llm_response
from RAG_project.config import settings

# 🟢 1. 修改 Prompt：强制简短回答，方便做包含匹配
core.generator.CONTEXT_PROMPT_TEMPLATE_EN = '''
Based on the provided context, answer the question using ONLY a few words (e.g., a name, date, or entity).
Do NOT write full sentences.

[Context]
{context}

[Question]
{query}

[Answer]
'''

# 🟢 2. 配置调整
settings.KNOWLEDGE_DATASET_CONFIG = {
    'chunk_size': 512,
    'chunk_overlap': 50
}
settings.TEMPERATURE = 0.01
settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # 小模型更容易体现噪音干扰

# 实验参数
QUERIES_PER_STEP = 40    # 每个时间步测试 40 个问题 (20 old / 20 new 动态变化)
DOCS_PER_DOMAIN = 2000   # 每个领域最多 2000 条文档 (节省时间)

# ================= 指标工具: Containment =================
def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def calculate_containment(prediction, ground_truths):
    """如果生成的答案包含了任意一个标准答案，记为 1.0"""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        norm_gt = normalize_answer(gt)
        if len(norm_gt) < 2: continue # 跳过过短的答案
        if norm_gt in norm_pred:
            return 1.0
    return 0.0

# ================= 数据准备 =================
def prepare_data():
    print("📦 准备双领域数据 (Wiki vs SQuAD)...")
    
    # 1. Domain A: Wiki (Old)
    wiki_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
    wiki_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
    
    w_docs = []
    for i, item in enumerate(wiki_ds):
        if i >= DOCS_PER_DOMAIN: break
        w_docs.append(Document(page_content=item['passage'], metadata={"doc_id": f"wiki_{item['id']}", "source": "wiki"}))
        
    w_qs = []
    for item in wiki_qa.select(range(QUERIES_PER_STEP)): 
        w_qs.append({"question": item['question'], "answers": [item['answer']] if 'answer' in item else []})

    # 2. Domain B: SQuAD (New)
    squad_ds = load_dataset("squad_v2", split="validation")
    s_docs = []
    s_qs = []
    seen = set()
    count = 0
    
    for item in squad_ds:
        ctx = item['context']
        if ctx not in seen and len(s_docs) < DOCS_PER_DOMAIN:
            s_docs.append(Document(page_content=ctx, metadata={"doc_id": f"squad_{item['id']}", "source": "squad"}))
            seen.add(ctx)
        
        if len(item['answers']['text']) > 0 and count < QUERIES_PER_STEP:
            s_qs.append({"question": item['question'], "answers": item['answers']['text']})
            count += 1
            
    print(f"✅ Ready: {len(w_docs)} Wiki Docs, {len(s_docs)} SQuAD Docs")
    return w_docs, w_qs, s_docs, s_qs

# ================= Pipeline =================
def run_pipeline(docs, test_set, desc):
    if not docs: return 0.0
    
    # 禁用 Hybrid，只用 Dense，因为 Dense 对噪音最敏感
    vector_db = _build_hybrid_vector_index(docs)
    retriever = QARetriever(vector_db=vector_db, docs=docs, hybrid_search=False)
    
    total_acc = 0
    # 使用 tqdm 显示进度
    pbar = tqdm(test_set, desc=f"   Running {desc}", leave=False)
    
    for item in pbar:
        try:
            results = retriever.retrieve(item['question'], rerank_top_k=3)
            ctx = [{"text": r['text'], "score": r['scores']['rerank']} for r in results]
            _, resp = generate_llm_response(query=item['question'], context=ctx, language="en")
            
            score = calculate_containment(resp, item['answers'])
            total_acc += score
            
            pbar.set_postfix({"Acc": f"{total_acc / (pbar.n + 1):.2%}"})
        except: 
            pass
            
    avg_acc = total_acc / len(test_set)
    
    del vector_db, retriever
    gc.collect()
    torch.cuda.empty_cache()
    
    return avg_acc

# ================= 主实验逻辑 =================
def run_smooth_shift():
    w_docs, w_qs, s_docs, s_qs = prepare_data()
    
    # 模拟平滑迁移的时间步：新业务占比 (Alpha)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    res_cum = []  # Cumulative (Add Only)
    res_agg = []  # Aggressive (Delete All Old)
    res_ada = []  # Adaptive (Proportional)
    x_labels = []
    
    print("\n🚀 开始 Smooth Shift 实验...")
    
    for alpha in alphas:
        label = f"{int(alpha*100)}% New"
        x_labels.append(label)
        print(f"\n⏳ Time Step: {label} (Query Mix)")
        
        # 1. 构造当前的 Query 分布 (混合 Old & New 问题)
        n_new = int(len(s_qs) * alpha)
        n_old = int(len(w_qs) * (1 - alpha))
        # 截取对应数量的问题
        current_test_set = w_qs[:n_old] + s_qs[:n_new]
        
        # --- A. Cumulative (只加不删) ---
        # 始终保留全部旧数据 (2000条)，只按比例加新数据
        kb_cum = w_docs[:] + s_docs[:int(len(s_docs)*alpha)]
        if not kb_cum: kb_cum = w_docs[:] 
        acc = run_pipeline(kb_cum, current_test_set, "🔴 Cumulative")
        res_cum.append(acc)
        
        # --- B. Aggressive (全删旧的) ---
        # 只要开始转型，就只保留新数据
        kb_agg = s_docs[:int(len(s_docs)*alpha)]
        if not kb_agg: 
            acc = 0.0 # 空库
        else:
            acc = run_pipeline(kb_agg, current_test_set, "🔵 Aggressive")
        res_agg.append(acc)
        
        # --- C. Adaptive (动态平衡) ---
        # 删一部分旧的，加一部分新的，保持 KB 分布 = Query 分布
        n_w_keep = int(len(w_docs) * (1 - alpha))
        n_s_keep = int(len(s_docs) * alpha)
        kb_ada = w_docs[:n_w_keep] + s_docs[:n_s_keep]
        if not kb_ada: kb_ada = w_docs[:1] # 防止空
        
        acc = run_pipeline(kb_ada, current_test_set, "🟢 Adaptive")
        res_ada.append(acc)
        
    return res_cum, res_agg, res_ada, x_labels

# ================= 绘图 =================
def plot_results(rc, rg, ra, labels):
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(labels, ra, marker='*', markersize=12, color='green', linewidth=3, label='Adaptive (Add & Partial Delete)')
    plt.plot(labels, rc, marker='o', color='red', linestyle='--', label='Cumulative (Add Only)')
    plt.plot(labels, rg, marker='x', color='blue', linestyle='-.', label='Aggressive (Delete All Old)')
    
    # 标注中间的噪音干扰
    mid = 2 # 50% New
    gap_noise = ra[mid] - rc[mid]
    plt.annotate(f"Noise Penalty\n(-{gap_noise:.1%})", 
                 xy=(mid, rc[mid]), xytext=(mid, rc[mid]-0.1),
                 arrowprops=dict(facecolor='red', shrink=0.05), ha='center', color='red')
    
    # 标注前期的遗忘
    early = 1 # 25% New
    gap_forget = ra[early] - rg[early]
    plt.annotate(f"Forgetting\n(-{gap_forget:.1%})", 
                 xy=(early, rg[early]), xytext=(early, rg[early]-0.15),
                 arrowprops=dict(facecolor='blue', shrink=0.05), ha='center', color='blue')

    plt.ylabel('Accuracy (Answer Containment)')
    plt.xlabel('Shift Progress (Old -> New Domain)')
    plt.title('RAG Data Strategy during Smooth Domain Shift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out = 'rag_smooth_shift_containment.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\n✅ 结果已保存: {out}")

if __name__ == "__main__":
    c, g, a, l = run_smooth_shift()
    plot_results(c, g, a, l)