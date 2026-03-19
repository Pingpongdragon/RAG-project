"""
Motivation 2: KB-Query Misalignment 实验 (聚类版)

实验设计:
  用 embedding 聚类自动发现 HotpotQA 的话题结构，替代硬编码 4 领域分类。

  防偏差措施:
  1. 聚类用 all-MiniLM-L6-v2，检索用 BAAI/bge-small-en-v1.5 — 不同模型避免 embedding bias
  2. 对全量加载数据做聚类 (不抽子集) — 避免采样偏差
  3. LLM (Qwen) 给聚类赋予语义标签 — 提供可解释性

  核心实验逻辑 (与旧版一致):
  1. 加载 HotpotQA，提取 question + gold/context docs
  2. Embedding → Auto-KMeans 自动聚类发现话题
  3. Query 分布由聚类自然决定 (不人工干预)
  4. KB 分布从 "Aligned" 到 "Extreme" 逐步偏移
  5. 测量 Gold Coverage / Hit@K

用法:
  python gen_misalignment.py            # 默认: quick verify (仅 coverage)
  python gen_misalignment.py --full     # 完整实验: coverage + Hit@K

输出:
  results_misalignment.json  — 实验结果
  cluster_info.json          — 聚类详情 (K, 语义标签, 分布)
"""
import sys
import os
import gc
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
from scipy.spatial.distance import jensenshannon

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # RAG-project/
sys.path.insert(0, str(SCRIPT_DIR.parent))  # motivation/

# 加载 .env (Qwen API 配置)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from plot_config import setup_style, COLORS, save_fig
from utils import normalize_answer, save_results, build_pipeline

OUT_DIR = SCRIPT_DIR
DATA_DIR = SCRIPT_DIR / "data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# 实验配置
# ============================================================
LOAD_ITEMS        = 20000      # HotpotQA streaming 加载条目数
CLUSTER_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"   # 聚类用
RETRIEVER_MODEL   = "BAAI/bge-small-en-v1.5"                  # 检索用 (不同模型!)
USE_HYBRID        = False
K_MIN, K_MAX      = 3, 8      # 自动选 K 的搜索范围
N_Q_PER_CLUSTER   = 200       # 每聚类测试查询数
KB_TOTAL_SIZE     = 60000     # KB 总文档数
TOP_K             = 10
BATCH_SIZE        = 32
EMBED_BATCH_SIZE  = 256
RANDOM_SEED       = 42


# ============================================================
# 工具函数
# ============================================================
def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"🎲 Random seed: {seed}")


def compute_jsd(p, q):
    p = np.clip(p, 1e-10, None); p /= p.sum()
    q = np.clip(q, 1e-10, None); q /= q.sum()
    return float(jensenshannon(p, q))


# ============================================================
# Step 1: 加载 HotpotQA
# ============================================================
def load_hotpotqa(max_items=LOAD_ITEMS):
    """Stream HotpotQA, 提取 question + gold docs + all context docs"""
    from datasets import load_dataset
    import itertools

    print(f"📥 Loading HotpotQA (streaming, up to {max_items} items)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train",
                      streaming=True, trust_remote_code=True)

    items = []
    for raw in itertools.islice(ds, max_items):
        titles = raw['context']['title']
        sentences = raw['context']['sentences']
        sf_titles = set(raw['supporting_facts']['title'])

        gold_texts, all_texts = [], []
        for title, sent_list in zip(titles, sentences):
            text = f"{title}: " + " ".join(sent_list)
            all_texts.append(text)
            if title in sf_titles:
                gold_texts.append(text)

        if not gold_texts:
            continue

        items.append({
            "question": raw['question'],
            "answer": raw['answer'],
            "gold_texts": gold_texts,
            "all_texts": all_texts,
        })

        if len(items) % 5000 == 0:
            print(f"  ... {len(items)} valid items")

    del ds
    gc.collect()

    print(f"  ✅ {len(items)} items loaded (with valid gold docs)")
    return items


# ============================================================
# Step 2: Embedding (与检索模型不同，避免 embedding 偏差)
# ============================================================
def embed_questions(items, model_name=CLUSTER_MODEL):
    from sentence_transformers import SentenceTransformer

    print(f"🔢 Embedding {len(items)} questions with {model_name.split('/')[-1]}")
    print(f"   (Different from retrieval model {RETRIEVER_MODEL.split('/')[-1]} → no embedding bias)")

    model = SentenceTransformer(model_name)
    questions = [it["question"] for it in items]
    embeddings = model.encode(
        questions, batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True
    )

    del model
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"  ✅ Embedding shape: {embeddings.shape}")
    return embeddings


# ============================================================
# Step 3: 自动聚类 (全量数据, 不抽子集, 避免采样偏差)
# ============================================================
def auto_cluster(embeddings, k_min=K_MIN, k_max=K_MAX, seed=RANDOM_SEED):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = embeddings.shape[0]
    k_max = min(k_max, n - 1)

    print(f"🔍 Auto clustering on ALL {n} samples (no subset → no bias)")
    print(f"   Searching K ∈ [{k_min}, {k_max}]")

    best_k, best_score, best_labels = k_min, -1.0, None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(
            embeddings, labels, metric='cosine',
            sample_size=min(5000, n), random_state=seed
        )
        sizes = sorted([(labels == i).sum() for i in range(k)], reverse=True)
        print(f"  K={k}: silhouette={score:.4f}, sizes={sizes}")

        if score > best_score:
            best_k, best_score, best_labels = k, score, labels.copy()

    print(f"  ✅ Selected K={best_k} (silhouette={best_score:.4f})")

    # 按簇大小排序 (最大簇 → id=0)
    cluster_sizes = [(best_labels == i).sum() for i in range(best_k)]
    sorted_ids = sorted(range(best_k), key=lambda i: cluster_sizes[i], reverse=True)
    id_map = {old: new for new, old in enumerate(sorted_ids)}
    remapped = np.array([id_map[int(l)] for l in best_labels])

    return remapped, best_k, best_score


# ============================================================
# Step 4: LLM 赋予语义标签
# ============================================================
def label_clusters_llm(items, labels, k):
    from openai import OpenAI

    cluster_queries = {i: [] for i in range(k)}
    for item, label in zip(items, labels):
        cluster_queries[int(label)].append(item["question"])

    try:
        api_key = os.getenv("QWEN_API_KEY")
        base_url = os.getenv("QWEN_BASE_URL")
        model_name = os.getenv("QWEN_MODEL_NAME")

        if not all([api_key, base_url, model_name]):
            raise ValueError("Qwen API not configured")

        client = OpenAI(api_key=api_key, base_url=base_url)
        names = {}

        print("🏷️  LLM labeling clusters...")
        for cid in range(k):
            sample = random.sample(
                cluster_queries[cid],
                min(15, len(cluster_queries[cid]))
            )
            sample_text = "\n".join(f"  - {q}" for q in sample)

            prompt = (
                f"Below are example questions from a topic cluster in a QA dataset:\n"
                f"{sample_text}\n\n"
                f"Give a concise topic label (2-4 English words) for this cluster. "
                f"Reply with ONLY the label, nothing else."
            )

            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )
            raw_name = resp.choices[0].message.content.strip()
            name = raw_name.strip('"\'.,!? ').title()
            names[cid] = name
            print(f"  Cluster {cid} ({len(cluster_queries[cid]):>5d} Q): {name}")

        return names

    except Exception as e:
        print(f"  ⚠️ LLM labeling failed ({e}), using default names")
        return {i: f"Topic_{i}" for i in range(k)}


# ============================================================
# Step 5: 构建聚类池
# ============================================================
def build_cluster_pools(items, labels, cluster_names):
    k = len(cluster_names)
    domain_names = [cluster_names[i] for i in range(k)]

    cluster_questions = {name: [] for name in domain_names}
    cluster_doc_pool = {name: set() for name in domain_names}

    for item, label in zip(items, labels):
        name = cluster_names[int(label)]
        cluster_questions[name].append({
            "question": item["question"],
            "answer": item["answer"],
            "gold_texts": item["gold_texts"],
            "all_texts": item["all_texts"],
            "domain": name,
        })
        for text in item["all_texts"]:
            cluster_doc_pool[name].add(text)

    cluster_doc_pool = {n: list(docs) for n, docs in cluster_doc_pool.items()}

    print("\n📊 Cluster Pools:")
    for name in domain_names:
        print(f"  [{name:>25s}]: {len(cluster_questions[name]):>5d} Q, "
              f"{len(cluster_doc_pool[name]):>6d} docs")

    return cluster_questions, cluster_doc_pool, domain_names


# ============================================================
# Step 6: 自动生成 KB 分布设置 (池容量感知, 避免溢出)
# ============================================================
def generate_kb_distributions(query_dist, pool_sizes, kb_total=KB_TOTAL_SIZE):
    """从 Aligned 到 Extreme, 自动生成 KB 分布. 确保不超出各簇文档池容量."""
    K = len(query_dist)
    max_idx = int(query_dist.argmax())

    # 极端分布目标: 集中在最大簇, 其余每簇最少 2%
    max_concentration = min(
        pool_sizes[max_idx] / kb_total,   # 不超过最大簇的池容量
        1.0 - 0.02 * (K - 1)              # 其余簇至少 2%
    )
    extreme = np.full(K, 0.02)
    extreme[max_idx] = max_concentration
    extreme /= extreme.sum()

    blends = [
        ("Aligned",   0.00),
        ("Mild",      0.30),
        ("Moderate",  0.50),
        ("Severe",    0.70),
        ("Extreme",   0.85),
    ]

    settings = []
    for label, alpha in blends:
        d = (1 - alpha) * query_dist + alpha * extreme
        d = np.clip(d, 0.01, None)
        d /= d.sum()
        settings.append((label, d))

    return settings


# ============================================================
# Step 7: 采样测试查询
# ============================================================
def sample_test_queries(cluster_questions, domain_names, n_per=N_Q_PER_CLUSTER):
    test_q = []
    for name in domain_names:
        pool = cluster_questions[name]
        n = min(n_per, len(pool))
        selected = pool[-n:]
        test_q.extend(selected)
        print(f"  🎯 {name}: {len(selected)} test queries")
    random.shuffle(test_q)
    return test_q


# ============================================================
# Step 8: 按分布构建 KB
# ============================================================
def build_kb_by_distribution(cluster_doc_pool, kb_dist, domain_names,
                             kb_total=KB_TOTAL_SIZE):
    from langchain_core.documents import Document

    kb_docs = []
    domain_counts = {}

    for i, name in enumerate(domain_names):
        n_need = int(kb_total * kb_dist[i])
        pool = cluster_doc_pool[name]

        if len(pool) < n_need:
            print(f"    ⚠️ {name}: need {n_need}, have {len(pool)}, taking all")
            sampled = pool[:]
        else:
            sampled = random.sample(pool, n_need)

        domain_counts[name] = len(sampled)
        for j, text in enumerate(sampled):
            kb_docs.append(Document(
                page_content=text,
                metadata={"doc_id": f"{name}_{j}", "domain": name, "source": "hotpotqa"},
            ))

    random.shuffle(kb_docs)
    print(f"    📦 KB: {len(kb_docs)} docs | " +
          ", ".join(f"{n}={domain_counts[n]}" for n in domain_names))
    return kb_docs, domain_counts


# ============================================================
# Step 9: Gold Coverage (精确匹配, O(n) 而非 O(n*m))
# ============================================================
def compute_gold_coverage(test_queries, kb_docs, domain_names):
    # 构建 normalized KB 文本集合 → O(1) 查找
    kb_texts = set(normalize_answer(doc.page_content) for doc in kb_docs)

    total = len(test_queries)
    covered = 0
    domain_stats = {n: {"total": 0, "covered": 0} for n in domain_names}

    for item in test_queries:
        domain = item.get("domain", "unknown")
        golds = item["gold_texts"]

        # 精确集合匹配 O(1), 替代旧版 O(n) 子串扫描
        found = any(normalize_answer(g) in kb_texts for g in golds)

        if found:
            covered += 1
        if domain in domain_stats:
            domain_stats[domain]["total"] += 1
            if found:
                domain_stats[domain]["covered"] += 1

    q_cov = covered / total if total > 0 else 0

    print(f"    📊 Coverage: {covered}/{total} ({q_cov:.1%})")
    for n in domain_names:
        s = domain_stats[n]
        if s["total"] > 0:
            print(f"       {n}: {s['covered']}/{s['total']} ({s['covered']/s['total']:.1%})")

    return {
        "query_coverage": q_cov,
        "domain_coverage": {
            n: (domain_stats[n]["covered"] / domain_stats[n]["total"]
                if domain_stats[n]["total"] > 0 else 0)
            for n in domain_names
        },
    }


# ============================================================
# Step 10: Hit@K 检索评估
# ============================================================
def evaluate_hit_rate(test_queries, kb_docs, domain_names,
                      model_name=RETRIEVER_MODEL, use_hybrid=USE_HYBRID, top_k=TOP_K):
    pipe = build_pipeline(kb_docs, model_name, use_hybrid)

    domain_hits = {n: 0 for n in domain_names}
    domain_total = {n: 0 for n in domain_names}
    total_done = 0

    for item in test_queries:
        query = item["question"]
        domain = item.get("domain", "unknown")
        gold_set = set(normalize_answer(g) for g in item["gold_texts"])

        if domain in domain_total:
            domain_total[domain] += 1

        try:
            results = pipe.retrieve(query, rerank_top_k=top_k, rerank_threshold=0.0)
            ret_texts = [normalize_answer(r["text"]) for r in results]
        except Exception as e:
            print(f"    Retrieve error: {query[:50]}... | {e}")
            ret_texts = []

        hit = any(
            any(gt in rt or rt in gt for rt in ret_texts)
            for gt in gold_set
        )
        if hit and domain in domain_hits:
            domain_hits[domain] += 1

        total_done += 1
        if total_done % 200 == 0:
            print(f"    ... {total_done}/{len(test_queries)} queries evaluated")

    del pipe
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    total = len(test_queries)
    total_hits = sum(domain_hits.values())
    overall = total_hits / total if total > 0 else 0

    domain_rates = {}
    for n in domain_names:
        domain_rates[n] = domain_hits[n] / domain_total[n] if domain_total[n] > 0 else 0

    print(f"    📊 Hit@{top_k}: {total_hits}/{total} ({overall:.1%})")
    for n in domain_names:
        s = domain_total[n]
        h = domain_hits[n]
        if s > 0:
            print(f"       {n}: {h}/{s} ({h/s:.1%})")

    return overall, domain_rates


# ============================================================
# 数据准备 (load → embed → cluster → label)
# ============================================================
def prepare_data():
    """加载 → 编码 → 聚类 → 标注, 所有模式共用"""
    set_random_seed()

    items = load_hotpotqa()
    embeddings = embed_questions(items)
    labels, k, sil_score = auto_cluster(embeddings)

    del embeddings
    gc.collect()

    cluster_names = label_clusters_llm(items, labels, k)
    cluster_questions, cluster_doc_pool, domain_names = \
        build_cluster_pools(items, labels, cluster_names)

    # Query 分布 (由聚类自然决定)
    query_dist = np.array([len(cluster_questions[n]) for n in domain_names], dtype=float)
    query_dist /= query_dist.sum()

    # 各簇文档池大小 (用于安全生成 KB 分布)
    pool_sizes = np.array([len(cluster_doc_pool[n]) for n in domain_names], dtype=float)

    print(f"\n📊 Query Distribution (natural, from clustering):")
    for n, p in zip(domain_names, query_dist):
        print(f"   {n}: {p:.1%}")

    # 保存聚类详情
    cluster_info = {
        "k": k,
        "silhouette": sil_score,
        "cluster_model": CLUSTER_MODEL,
        "retrieval_model": RETRIEVER_MODEL,
        "bias_prevention": {
            "embedding_bias": "cluster model (MiniLM) != retrieval model (BGE-Small)",
            "subset_bias": f"clustering on ALL {len(items)} items, no subsampling",
        },
        "cluster_names": {str(cid): name for cid, name in cluster_names.items()},
        "cluster_sizes": {n: len(cluster_questions[n]) for n in domain_names},
        "query_distribution": dict(zip(domain_names, query_dist.tolist())),
        "total_items": len(items),
    }
    save_results(cluster_info, str(DATA_DIR / "cluster_info.json"))

    return cluster_questions, cluster_doc_pool, domain_names, query_dist, pool_sizes


# ============================================================
# Quick Verify (仅 Coverage, 不调 retriever)
# ============================================================
def quick_verify():
    print("=" * 60)
    print("⚡ Quick Verify: Gold Coverage Only (no retriever)")
    print("=" * 60)

    data = prepare_data()
    cluster_questions, cluster_doc_pool, domain_names, query_dist, pool_sizes = data

    print("\n📝 Sampling test queries...")
    test_queries = sample_test_queries(cluster_questions, domain_names)
    print(f"   Total: {len(test_queries)} queries")

    kb_settings = generate_kb_distributions(query_dist, pool_sizes)
    results = []

    for label, kb_dist in kb_settings:
        jsd = compute_jsd(query_dist, kb_dist)
        print(f"\n  [{label}] KB dist={[f'{d:.2f}' for d in kb_dist]}, JSD={jsd:.4f}")

        kb_docs, domain_counts = build_kb_by_distribution(
            cluster_doc_pool, kb_dist, domain_names)
        coverage = compute_gold_coverage(test_queries, kb_docs, domain_names)

        results.append({
            "label": label,
            "jsd": jsd,
            "kb_dist": kb_dist.tolist(),
            "kb_size": len(kb_docs),
            "domain_counts": domain_counts,
            "query_coverage": coverage["query_coverage"],
            "domain_coverage": coverage["domain_coverage"],
            "domain_names": domain_names,
        })

    # 汇总
    print("\n" + "=" * 70)
    print("📋 Quick Verify Summary")
    print(f"  {'Setting':<12s} {'JSD':>6s} {'Q Coverage':>12s}")
    print(f"  {'-'*12} {'-'*6} {'-'*12}")
    for r in results:
        print(f"  {r['label']:<12s} {r['jsd']:>6.3f} {r['query_coverage']:>10.1%}")

    coverages = [r["query_coverage"] for r in results]
    if coverages[0] > coverages[-1]:
        d = coverages[0] - coverages[-1]
        print(f"\n  ✅ Trend correct! {coverages[0]:.1%} → {coverages[-1]:.1%} (Δ={d:.1%})")
        print(f"  💡 Run --full for Hit@K evaluation")
    else:
        print(f"\n  ❌ Trend issue: {coverages[0]:.1%} → {coverages[-1]:.1%}")

    save_results(results, str(DATA_DIR / "results_misalignment_quick.json"))
    return results


# ============================================================
# Full Experiment (Coverage + Hit@K)
# ============================================================
def run_full():
    print("=" * 60)
    print("📊 Full Experiment: Coverage + Hit@K")
    print(f"   Cluster model: {CLUSTER_MODEL} (NOT used for retrieval)")
    print(f"   Retriever:     {RETRIEVER_MODEL}")
    print(f"   KB Size:       {KB_TOTAL_SIZE}")
    print("=" * 60)

    data = prepare_data()
    cluster_questions, cluster_doc_pool, domain_names, query_dist, pool_sizes = data

    print("\n📝 Sampling test queries...")
    test_queries = sample_test_queries(cluster_questions, domain_names)
    print(f"   Total: {len(test_queries)} queries")

    kb_settings = generate_kb_distributions(query_dist, pool_sizes)
    results = []

    for label, kb_dist in kb_settings:
        jsd = compute_jsd(query_dist, kb_dist)
        print(f"\n{'='*55}")
        print(f"  [{label}] JSD={jsd:.4f}")

        kb_docs, domain_counts = build_kb_by_distribution(
            cluster_doc_pool, kb_dist, domain_names)

        coverage = compute_gold_coverage(test_queries, kb_docs, domain_names)

        print(f"  [{label}] Evaluating Hit@{TOP_K}...")
        hit_rate, domain_hit_rates = evaluate_hit_rate(
            test_queries, kb_docs, domain_names)

        r = {
            "label": label,
            "jsd": jsd,
            "kb_dist": kb_dist.tolist(),
            "kb_size": len(kb_docs),
            "domain_counts": domain_counts,
            "query_coverage": coverage["query_coverage"],
            "domain_coverage": coverage["domain_coverage"],
            "hit_rate": hit_rate,
            "domain_hit_rates": domain_hit_rates,
            "domain_names": domain_names,
        }
        results.append(r)
        print(f"  ✅ {label}: JSD={jsd:.4f}, Coverage={coverage['query_coverage']:.1%}, "
              f"Hit@{TOP_K}={hit_rate:.1%}")

    save_results(results, str(DATA_DIR / "results_misalignment.json"))

    # 汇总
    print("\n" + "=" * 70)
    print("📋 Full Results")
    print(f"  {'Setting':<12s} {'JSD':>6s} {'Coverage':>10s} {'Hit@10':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['label']:<12s} {r['jsd']:>6.3f} {r['query_coverage']:>8.1%} {r['hit_rate']:>8.1%}")

    return results


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KB-Query Misalignment Experiment (Clustering)")
    parser.add_argument("--quick", action="store_true", help="Quick verify: coverage only")
    parser.add_argument("--full", action="store_true", help="Full experiment: coverage + Hit@K")
    args = parser.parse_args()

    if args.full:
        run_full()
    else:
        quick_verify()

    # 避免 HotpotQA streaming 的 C++ abort
    os._exit(0)
