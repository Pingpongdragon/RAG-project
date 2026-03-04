"""
诊断: 为什么 KB-Query Misalignment 实验中 Hit@10 不下降？

假说 1: Retriever 太强 → 检查 gold doc 的检索排名分布
假说 2: 跨领域噪声语义差距太大 → 检查噪声 vs gold 的相似度分布
假说 3: Hot Document 效应 → 检查 gold doc 去重率和复用率
假说 4: 噪声总量不够 → 不同噪声量级下的 hit rate 变化
"""
import sys
import gc
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import normalize_answer, build_pipeline

from common import (
    get_detector, classify_queries,
    LABEL_MAP, DOMAIN_NAMES, NUM_DOMAINS, OUT_DIR,
)

# ==========================================
# 配置
# ==========================================
RANDOM_SEED = 42
MAX_PER_DOMAIN = 5000
N_Q_PER_DOMAIN = 100
CLASSIFY_BATCH_SIZE = 512
RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K_EXTENDED = 100  # 查看 top-100 排名分布

DIAG_DIR = OUT_DIR / "diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================
# 数据加载（复用 misalignment 的逻辑）
# ==========================================
def load_data():
    """加载并分类数据，与主实验一致"""
    from datasets import load_dataset

    detector = get_detector()
    ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)

    domain_questions: Dict[str, List[Dict]] = {d: [] for d in DOMAIN_NAMES}
    domain_all_docs: Dict[str, set] = {d: set() for d in DOMAIN_NAMES}

    buffer_q, buffer_items = [], []
    total = 0

    for item in ds:
        buffer_q.append(item['question'])
        buffer_items.append(item)

        if len(buffer_q) >= CLASSIFY_BATCH_SIZE:
            labels = classify_queries(buffer_q, detector)
            for lbl, it in zip(labels, buffer_items):
                if lbl not in LABEL_MAP or len(domain_questions[lbl]) >= MAX_PER_DOMAIN:
                    continue
                titles = it['context']['title']
                sentences = it['context']['sentences']
                sf_titles = set(it['supporting_facts']['title'])
                gold_texts, all_texts = [], []
                for title, sent_list in zip(titles, sentences):
                    text = f"{title}: " + " ".join(sent_list)
                    all_texts.append(text)
                    if title in sf_titles:
                        gold_texts.append(text)
                if not gold_texts:
                    continue
                domain_questions[lbl].append({
                    "question": it['question'],
                    "answer": it['answer'],
                    "gold_texts": gold_texts,
                    "all_texts": all_texts,
                    "domain": lbl,
                })
                for text in all_texts:
                    domain_all_docs[lbl].add(text)

            total += len(buffer_q)
            buffer_q.clear()
            buffer_items.clear()

            if total % 5000 == 0:
                stats = {d: len(domain_questions[d]) for d in DOMAIN_NAMES}
                print(f"  Processed {total}, per domain: {stats}")

            if all(len(domain_questions[d]) >= MAX_PER_DOMAIN for d in DOMAIN_NAMES):
                break

    for d in DOMAIN_NAMES:
        print(f"  {d}: {len(domain_questions[d])} Q, {len(domain_all_docs[d])} docs")

    return domain_questions, {d: list(v) for d, v in domain_all_docs.items()}


def sample_queries(domain_questions, n_per_domain=N_Q_PER_DOMAIN):
    test_q = []
    for d in DOMAIN_NAMES:
        pool = domain_questions[d]
        start = max(0, len(pool) - n_per_domain)
        test_q.extend(pool[start:start + n_per_domain])
    random.shuffle(test_q)
    return test_q


def build_noise_pools(domain_all_docs, test_queries):
    all_gold = set()
    for item in test_queries:
        for g in item["gold_texts"]:
            all_gold.add(g)
    return {d: [t for t in docs if t not in all_gold]
            for d, docs in domain_all_docs.items()}


def build_kb_docs(test_queries, noise_pools, noise_dist, noise_ratio):
    from langchain_core.documents import Document

    gold_set = set()
    for item in test_queries:
        for g in item["gold_texts"]:
            gold_set.add(g)
    gold_list = list(gold_set)

    n_noise = len(gold_list) * noise_ratio
    noise_docs = []
    for i, d in enumerate(DOMAIN_NAMES):
        n_need = int(n_noise * noise_dist[i])
        pool = noise_pools[d]
        sampled = random.sample(pool, min(n_need, len(pool)))
        noise_docs.extend(sampled)

    kb = []
    for j, t in enumerate(gold_list):
        kb.append(Document(page_content=t, metadata={"source": "gold", "doc_id": j}))
    for j, t in enumerate(noise_docs):
        kb.append(Document(page_content=t, metadata={"source": "noise", "doc_id": j}))
    random.shuffle(kb)
    return kb, len(gold_list), len(noise_docs)


# ==========================================
# 假说 1: 检查 gold doc 的检索排名分布
# ==========================================
def diagnose_rank_distribution(test_queries, kb_docs):
    """
    对每个 query，检索 top-100，看 gold doc 排在第几位
    如果 gold doc 几乎都在 top-3，说明 retriever 太强，噪声无法干扰
    """
    print("\n" + "=" * 60)
    print("🔬 假说 1: Gold doc 排名分布 (Top-100)")
    print("=" * 60)

    pipe = build_pipeline(kb_docs, RETRIEVER_MODEL, False)

    ranks = []  # gold doc 的排名列表
    scores_gold = []  # gold doc 的得分
    scores_noise_top = []  # 排名最高的 noise doc 得分
    not_found = 0

    for item in test_queries:
        query = item["question"]
        gold_set = set(normalize_answer(g) for g in item["gold_texts"])

        try:
            results = pipe.retrieve(
                query, rerank_top_k=TOP_K_EXTENDED, rerank_threshold=0.0
            )
        except Exception as e:
            print(f"  检索失败: {e}")
            continue

        found = False
        for rank, res in enumerate(results):
            norm_text = normalize_answer(res["text"])
            if any(g in norm_text or norm_text in g for g in gold_set):
                ranks.append(rank + 1)
                scores_gold.append(res.get("score", 0))
                found = True
                break

        if not found:
            not_found += 1
            ranks.append(TOP_K_EXTENDED + 1)

        # 最高噪声得分
        for res in results:
            norm_text = normalize_answer(res["text"])
            if not any(g in norm_text or norm_text in g for g in gold_set):
                scores_noise_top.append(res.get("score", 0))
                break

    del pipe
    gc.collect()

    ranks = np.array(ranks)
    print(f"\n📊 排名分布统计:")
    print(f"  Total queries: {len(ranks)}")
    print(f"  Not found in top-{TOP_K_EXTENDED}: {not_found}")
    print(f"  Rank 1 (完美匹配): {np.sum(ranks == 1)} ({np.mean(ranks == 1):.1%})")
    print(f"  Rank 1-3: {np.sum(ranks <= 3)} ({np.mean(ranks <= 3):.1%})")
    print(f"  Rank 1-10: {np.sum(ranks <= 10)} ({np.mean(ranks <= 10):.1%})")
    print(f"  Rank 11-50: {np.sum((ranks > 10) & (ranks <= 50))} ({np.mean((ranks > 10) & (ranks <= 50)):.1%})")
    print(f"  Rank 51-100: {np.sum((ranks > 50) & (ranks <= 100))} ({np.mean((ranks > 50) & (ranks <= 100)):.1%})")
    print(f"  Mean rank: {np.mean(ranks):.1f}")
    print(f"  Median rank: {np.median(ranks):.1f}")

    if scores_gold and scores_noise_top:
        print(f"\n📊 得分对比:")
        print(f"  Gold doc 平均得分:     {np.mean(scores_gold):.4f} ± {np.std(scores_gold):.4f}")
        print(f"  Top noise 平均得分:    {np.mean(scores_noise_top):.4f} ± {np.std(scores_noise_top):.4f}")
        print(f"  得分差距 (gold-noise): {np.mean(scores_gold) - np.mean(scores_noise_top):.4f}")

    # 绘制排名分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    valid_ranks = ranks[ranks <= TOP_K_EXTENDED]
    ax1.hist(valid_ranks, bins=range(1, TOP_K_EXTENDED + 2), edgecolor='black',
             color='steelblue', alpha=0.7)
    ax1.axvline(x=10.5, color='red', linestyle='--', linewidth=2, label='Top-K=10 cutoff')
    ax1.set_xlabel("Gold Doc Rank", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("假说1: Gold Doc 排名分布\n(大部分在 rank 1 = retriever 太强)", fontsize=12)
    ax1.legend()
    ax1.set_xlim(0, 50)

    ax2 = axes[1]
    if scores_gold and scores_noise_top:
        ax2.hist(scores_gold, bins=30, alpha=0.6, label='Gold doc scores', color='green')
        ax2.hist(scores_noise_top, bins=30, alpha=0.6, label='Top noise scores', color='red')
        ax2.set_xlabel("Retrieval Score", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Gold vs Noise 得分分布\n(差距大 = 噪声无法干扰)", fontsize=12)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(str(DIAG_DIR / "h1_rank_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 图已保存: {DIAG_DIR / 'h1_rank_distribution.png'}")

    return {"ranks": ranks.tolist(), "scores_gold": scores_gold, "scores_noise_top": scores_noise_top}


# ==========================================
# 假说 2: 跨领域噪声语义距离
# ==========================================
def diagnose_cross_domain_similarity(test_queries, noise_pools):
    """
    计算每个 query 与同领域噪声 vs 跨领域噪声的 embedding 相似度
    如果跨领域噪声相似度极低，说明跨领域噪声完全无法干扰检索
    """
    print("\n" + "=" * 60)
    print("🔬 假说 2: 跨领域噪声的语义距离")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(RETRIEVER_MODEL)
    N_SAMPLE_NOISE = 200  # 每个领域采样 200 个噪声文档

    # 采样少量 query 和 noise 来计算
    n_sample_q = min(50, len(test_queries))
    sample_queries = random.sample(test_queries, n_sample_q)

    results_by_domain = {}

    for q_domain in DOMAIN_NAMES:
        domain_queries = [q for q in sample_queries if q.get("domain") == q_domain]
        if len(domain_queries) < 5:
            continue

        q_texts = [q["question"] for q in domain_queries]
        q_embs = model.encode(q_texts, normalize_embeddings=True, convert_to_numpy=True)

        sims = {}
        for n_domain in DOMAIN_NAMES:
            pool = noise_pools[n_domain]
            if len(pool) == 0:
                continue
            sample_noise = random.sample(pool, min(N_SAMPLE_NOISE, len(pool)))
            n_embs = model.encode(sample_noise, normalize_embeddings=True, convert_to_numpy=True)

            # 计算每个 query 与所有噪声的最大相似度
            sim_matrix = q_embs @ n_embs.T  # (n_q, n_noise)
            max_sims = sim_matrix.max(axis=1)  # 每个 query 的最高噪声相似度
            mean_sims = sim_matrix.mean(axis=1)
            sims[n_domain] = {
                "max_sim_mean": float(max_sims.mean()),
                "max_sim_std": float(max_sims.std()),
                "mean_sim_mean": float(mean_sims.mean()),
            }

        results_by_domain[q_domain] = sims

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 打印结果
    print(f"\n📊 Query→Noise 最大相似度 (越高 = 噪声越可能干扰检索):")
    print(f"  {'Q Domain':<15s}", end="")
    for d in DOMAIN_NAMES:
        print(f"  {d[:8]:>10s}", end="")
    print()

    for q_d in DOMAIN_NAMES:
        if q_d not in results_by_domain:
            continue
        print(f"  {q_d:<15s}", end="")
        for n_d in DOMAIN_NAMES:
            if n_d in results_by_domain[q_d]:
                val = results_by_domain[q_d][n_d]["max_sim_mean"]
                marker = "⬅同域" if q_d == n_d else ""
                print(f"  {val:>8.4f}{marker}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print()

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = np.zeros((len(DOMAIN_NAMES), len(DOMAIN_NAMES)))
    for i, q_d in enumerate(DOMAIN_NAMES):
        for j, n_d in enumerate(DOMAIN_NAMES):
            if q_d in results_by_domain and n_d in results_by_domain[q_d]:
                matrix[i, j] = results_by_domain[q_d][n_d]["max_sim_mean"]

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha='center', va='center',
                    fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(DOMAIN_NAMES)))
    ax.set_xticklabels([d[:8] for d in DOMAIN_NAMES], fontsize=10)
    ax.set_yticks(range(len(DOMAIN_NAMES)))
    ax.set_yticklabels([d[:8] for d in DOMAIN_NAMES], fontsize=10)
    ax.set_xlabel("Noise Domain", fontsize=12)
    ax.set_ylabel("Query Domain", fontsize=12)
    ax.set_title("假说2: Query→Noise Max Cosine Similarity\n(同域高、跨域低 = 跨域噪声无法干扰)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Max Cosine Similarity")

    plt.tight_layout()
    fig.savefig(str(DIAG_DIR / "h2_cross_domain_similarity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 图已保存: {DIAG_DIR / 'h2_cross_domain_similarity.png'}")

    return results_by_domain


# ==========================================
# 假说 3: Hot Document 效应
# ==========================================
def diagnose_hot_documents(test_queries):
    """
    分析 gold 文档的复用率
    如果很多 query 共享少量 gold doc，说明有 hot document 效应
    """
    print("\n" + "=" * 60)
    print("🔬 假说 3: Hot Document 效应")
    print("=" * 60)

    # 统计每个 gold doc 被多少 query 引用
    gold_doc_counter = Counter()
    total_gold_refs = 0
    per_query_gold_count = []

    for item in test_queries:
        per_query_gold_count.append(len(item["gold_texts"]))
        for g in item["gold_texts"]:
            gold_doc_counter[g] += 1
            total_gold_refs += 1

    n_unique_gold = len(gold_doc_counter)
    n_total_queries = len(test_queries)

    print(f"\n📊 Gold Document 统计:")
    print(f"  Total queries: {n_total_queries}")
    print(f"  Total gold references: {total_gold_refs}")
    print(f"  Unique gold docs: {n_unique_gold}")
    print(f"  Dedup ratio: {n_unique_gold / total_gold_refs:.1%} "
          f"(100%=无复用, 低=高复用)")
    print(f"  Avg gold docs per query: {np.mean(per_query_gold_count):.1f}")

    # 复用分布
    usage_counts = list(gold_doc_counter.values())
    print(f"\n📊 Gold doc 被引用次数分布:")
    print(f"  被 1 个 query 引用: {sum(1 for c in usage_counts if c == 1)} docs")
    print(f"  被 2 个 query 引用: {sum(1 for c in usage_counts if c == 2)} docs")
    print(f"  被 3+ 个 query 引用: {sum(1 for c in usage_counts if c >= 3)} docs")
    print(f"  被 5+ 个 query 引用: {sum(1 for c in usage_counts if c >= 5)} docs")

    # Top-10 最热门文档
    print(f"\n📊 Top-10 最热门 gold docs:")
    for doc_text, count in gold_doc_counter.most_common(10):
        print(f"  [{count:>3d} queries] {doc_text[:80]}...")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(usage_counts, bins=range(1, max(usage_counts) + 2),
             edgecolor='black', color='coral', alpha=0.7)
    ax1.set_xlabel("被引用次数", fontsize=12)
    ax1.set_ylabel("Gold doc 数量", fontsize=12)
    ax1.set_title(f"假说3: Gold Doc 引用次数分布\n"
                  f"(Unique={n_unique_gold}, Total refs={total_gold_refs})", fontsize=12)

    ax2 = axes[1]
    ax2.hist(per_query_gold_count, bins=range(1, max(per_query_gold_count) + 2),
             edgecolor='black', color='skyblue', alpha=0.7)
    ax2.set_xlabel("每个 query 的 gold doc 数量", fontsize=12)
    ax2.set_ylabel("Query 数量", fontsize=12)
    ax2.set_title("每个 query 包含几个 gold doc", fontsize=12)

    plt.tight_layout()
    fig.savefig(str(DIAG_DIR / "h3_hot_documents.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 图已保存: {DIAG_DIR / 'h3_hot_documents.png'}")

    return {
        "n_unique_gold": n_unique_gold,
        "total_gold_refs": total_gold_refs,
        "dedup_ratio": n_unique_gold / total_gold_refs,
        "top10": gold_doc_counter.most_common(10),
    }


# ==========================================
# 假说 4: 噪声量级扫描
# ==========================================
def diagnose_noise_scale(test_queries, noise_pools):
    """
    固定噪声分布 (Extreme=[0.85,0.10,0.03,0.02])，
    扫描噪声比从 10 到 2000，看 hit rate 何时开始下降
    """
    print("\n" + "=" * 60)
    print("🔬 假说 4: 噪声量级扫描 (固定 Extreme 分布)")
    print("=" * 60)

    noise_dist = np.array([0.85, 0.10, 0.03, 0.02])
    noise_ratios = [10, 50, 100, 200, 500, 1000]

    # 检查噪声池容量
    pool_sizes = {d: len(noise_pools[d]) for d in DOMAIN_NAMES}
    print(f"  噪声池大小: {pool_sizes}")

    gold_set = set()
    for item in test_queries:
        for g in item["gold_texts"]:
            gold_set.add(g)
    n_gold = len(gold_set)

    max_possible_noise = sum(int(pool_sizes[d] * noise_dist[i])
                             for i, d in enumerate(DOMAIN_NAMES))
    max_possible_ratio = max_possible_noise // n_gold if n_gold > 0 else 0
    print(f"  Gold docs: {n_gold}")
    print(f"  最大可用噪声量: {max_possible_noise} (ratio ≈ 1:{max_possible_ratio})")

    # 过滤掉超出容量的 ratio
    valid_ratios = [r for r in noise_ratios if r <= max_possible_ratio]
    if len(valid_ratios) < len(noise_ratios):
        print(f"  ⚠️ 可用 ratio: {valid_ratios} (跳过超出容量的)")
    noise_ratios = valid_ratios

    if not noise_ratios:
        print("  ❌ 噪声池太小，无法进行扫描")
        return {}

    results = []
    for ratio in noise_ratios:
        print(f"\n  🔍 Noise ratio 1:{ratio}...")
        kb_docs, n_g, n_n = build_kb_docs(test_queries, noise_pools, noise_dist, ratio)
        print(f"    KB size: {len(kb_docs)} (gold={n_g}, noise={n_n})")

        pipe = build_pipeline(kb_docs, RETRIEVER_MODEL, False)
        hits = 0
        for item in test_queries:
            query = item["question"]
            gold_norm = set(normalize_answer(g) for g in item["gold_texts"])
            try:
                ret = pipe.retrieve(query, rerank_top_k=10, rerank_threshold=0.0)
                ret_texts = [normalize_answer(r["text"]) for r in ret]
            except:
                ret_texts = []
            for g in gold_norm:
                if any(g in r or r in g for r in ret_texts):
                    hits += 1
                    break

        hit_rate = hits / len(test_queries)
        results.append({"ratio": ratio, "hit_rate": hit_rate, "kb_size": len(kb_docs)})
        print(f"    ✅ Hit@10 = {hit_rate:.2%}")

        del pipe
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = [r["ratio"] for r in results]
    hit_rates = [r["hit_rate"] for r in results]
    kb_sizes = [r["kb_size"] for r in results]

    ax.plot(ratios, hit_rates, 'o-', color='steelblue', linewidth=2.5,
            markersize=10, label='Hit@10')

    for r, h, k in zip(ratios, hit_rates, kb_sizes):
        ax.annotate(f"{h:.1%}\n(KB={k})", (r, h),
                    textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel("Noise Ratio (1:X)", fontsize=13)
    ax.set_ylabel("Hit@10", fontsize=13)
    ax.set_title("假说4: 噪声量级 vs Hit Rate\n(Extreme 分布 [0.85, 0.10, 0.03, 0.02])", fontsize=13)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(str(DIAG_DIR / "h4_noise_scale.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📈 图已保存: {DIAG_DIR / 'h4_noise_scale.png'}")

    return results


# ==========================================
# 综合诊断报告
# ==========================================
def generate_report(h1_result, h2_result, h3_result, h4_result):
    """生成综合诊断报告"""
    print("\n" + "=" * 70)
    print("📋 综合诊断报告")
    print("=" * 70)

    # 假说 1
    ranks = np.array(h1_result["ranks"])
    top1_pct = np.mean(ranks == 1)
    top10_pct = np.mean(ranks <= 10)
    print(f"\n🔬 假说 1 - Retriever 太强:")
    print(f"   Gold doc 在 rank 1 的比例: {top1_pct:.1%}")
    print(f"   Gold doc 在 top-10 的比例: {top10_pct:.1%}")
    if top1_pct > 0.7:
        print(f"   ⚠️ 诊断: CONFIRMED - {top1_pct:.0%} 的 gold doc 直接排第一，噪声完全无法干扰")
    elif top10_pct > 0.95:
        print(f"   ⚠️ 诊断: LIKELY - top-10 命中率 {top10_pct:.0%}，噪声干扰很弱")
    else:
        print(f"   ✅ 诊断: UNLIKELY")

    if h1_result["scores_gold"] and h1_result["scores_noise_top"]:
        gap = np.mean(h1_result["scores_gold"]) - np.mean(h1_result["scores_noise_top"])
        print(f"   Gold-Noise 得分差距: {gap:.4f}")
        if gap > 0.1:
            print(f"   ⚠️ 相大得分差距说明 retriever 语义区分度极高")

    # 假说 2
    print(f"\n🔬 假说 2 - 跨领域噪声语义差距:")
    if h2_result:
        same_domain_sims = []
        cross_domain_sims = []
        for q_d in DOMAIN_NAMES:
            if q_d not in h2_result:
                continue
            for n_d in DOMAIN_NAMES:
                if n_d in h2_result[q_d]:
                    val = h2_result[q_d][n_d]["max_sim_mean"]
                    if q_d == n_d:
                        same_domain_sims.append(val)
                    else:
                        cross_domain_sims.append(val)
        if same_domain_sims and cross_domain_sims:
            same_avg = np.mean(same_domain_sims)
            cross_avg = np.mean(cross_domain_sims)
            print(f"   同域噪声平均 max-sim: {same_avg:.4f}")
            print(f"   跨域噪声平均 max-sim: {cross_avg:.4f}")
            print(f"   差距: {same_avg - cross_avg:.4f}")
            if same_avg - cross_avg > 0.05:
                print(f"   ⚠️ 诊断: CONFIRMED - 跨域噪声语义距离远，无法干扰")
            else:
                print(f"   ✅ 诊断: UNLIKELY - 跨域噪声也有一定相似度")

    # 假说 3
    print(f"\n🔬 假说 3 - Hot Document 效应:")
    if h3_result:
        dedup = h3_result["dedup_ratio"]
        print(f"   去重率: {dedup:.1%} (100%=无复用)")
        if dedup < 0.5:
            print(f"   ⚠️ 诊断: CONFIRMED - 大量 query 共享 gold doc")
        else:
            print(f"   ✅ 诊断: UNLIKELY - gold doc 复用率不高")

    # 假说 4
    print(f"\n🔬 假说 4 - 噪声量不够:")
    if h4_result:
        for r in h4_result:
            print(f"   Ratio 1:{r['ratio']}: Hit@10={r['hit_rate']:.2%} (KB={r['kb_size']})")
        first_hit = h4_result[0]["hit_rate"] if h4_result else 1.0
        last_hit = h4_result[-1]["hit_rate"] if h4_result else 1.0
        drop = first_hit - last_hit
        if drop < 0.05:
            print(f"   ⚠️ 诊断: CONFIRMED - 增大噪声比到 1:{h4_result[-1]['ratio']} 仍无显著下降")
            print(f"   💡 建议: 可能需要 1:5000+ 或换更弱的 retriever")
        elif drop < 0.15:
            print(f"   ⚠️ 诊断: PARTIAL - 有下降但不够显著 (Δ={drop:.1%})")
        else:
            print(f"   ✅ 诊断: UNLIKELY - 噪声量增大确实导致显著下降 (Δ={drop:.1%})")

    # 综合建议
    print(f"\n{'='*70}")
    print(f"💡 综合建议:")
    print(f"  1. 如果假说1/2 confirmed → 改用同领域噪声（而非跨领域），让噪声更 confusing")
    print(f"  2. 如果假说4 confirmed → 大幅增加噪声量 或 换弱 retriever (e.g., BM25)")
    print(f"  3. 考虑换实验设计: 不做跨领域噪声，而是用同 topic 的 hard negatives")
    print(f"  4. 或者反转思路: 展示 'dense retriever 对分布偏移鲁棒' 也是一个发现")
    print(f"{'='*70}")


# ==========================================
# 主函数
# ==========================================
def main():
    print("🔬 KB-Query Misalignment 诊断工具")
    print("=" * 60)

    set_seed()

    # 加载数据
    print("\n📥 加载数据...")
    domain_questions, domain_all_docs = load_data()

    # 采样 query
    print("\n📝 采样 test queries...")
    test_queries = sample_queries(domain_questions)
    print(f"   Total: {len(test_queries)}")

    # 构建噪声池
    print("\n🗃️ 构建噪声池...")
    noise_pools = build_noise_pools(domain_all_docs, test_queries)

    # 使用 Extreme 分布构建 KB（最偏斜的设置）
    print("\n📦 构建 KB (Extreme 分布, ratio=1:100)...")
    noise_dist = np.array([0.85, 0.10, 0.03, 0.02])
    kb_docs, n_g, n_n = build_kb_docs(test_queries, noise_pools, noise_dist, 100)
    print(f"   KB size: {len(kb_docs)}")

    # ===== 运行所有诊断 =====

    # 假说 3 最快，先跑
    h3_result = diagnose_hot_documents(test_queries)

    # 假说 2 需要 embedding 但不需要 retriever
    h2_result = diagnose_cross_domain_similarity(test_queries, noise_pools)

    # 假说 1 需要构建 retriever index
    h1_result = diagnose_rank_distribution(test_queries, kb_docs)

    # 假说 4 最慢（多次构建 retriever）
    h4_result = diagnose_noise_scale(test_queries, noise_pools)

    # 综合报告
    generate_report(h1_result, h2_result, h3_result, h4_result)

    # 保存原始结果
    report = {
        "h1_rank_dist": {
            "top1_pct": float(np.mean(np.array(h1_result["ranks"]) == 1)),
            "top10_pct": float(np.mean(np.array(h1_result["ranks"]) <= 10)),
            "mean_rank": float(np.mean(h1_result["ranks"])),
        },
        "h3_hot_docs": {
            "n_unique_gold": h3_result["n_unique_gold"],
            "total_gold_refs": h3_result["total_gold_refs"],
            "dedup_ratio": h3_result["dedup_ratio"],
        },
        "h4_noise_scale": h4_result,
    }
    with open(str(DIAG_DIR / "diagnosis_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 诊断报告已保存: {DIAG_DIR / 'diagnosis_report.json'}")


if __name__ == "__main__":
    main()