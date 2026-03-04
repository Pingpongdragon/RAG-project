"""
Motivation 2B v2: KB-Query Distribution Misalignment → Retrieval Accuracy Drop

核心修改:
  - gold 文档不再强制放入 KB
  - KB 完全按领域分布从文档池中随机采样
  - 如果 KB 分布偏向某领域，其他领域的 gold doc 自然不在 KB 中
  - 这才是真实场景: 你无法预知 query 需要哪些文档

快速验证版: 减少 query 数量，先确认趋势正确
"""
import sys
import gc
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, save_fig
from utils import normalize_answer, save_results, build_pipeline, evaluate_batch

from common import (
    get_detector, classify_queries,
    get_domain_distribution, compute_jsd,
    LABEL_MAP, DOMAIN_NAMES, NUM_DOMAINS, OUT_DIR,
)


# ==========================================
# 实验配置 - 快速验证版
# ==========================================
MAX_PER_DOMAIN = 20000       # 每个领域采集文档
CLASSIFY_BATCH_SIZE = 512
N_Q_PER_DOMAIN = 200         # 快速验证: 每领域 50 query, 共 200
KB_TOTAL_SIZE = 80000       # KB 总文档数（固定大小，按分布采样）
TOP_K = 10
BATCH_SIZE = 32
RANDOM_SEED = 42

RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
RETRIEVER_DISPLAY = "BGE-Small (Dense)"
USE_HYBRID = False

# KB 分布设置: 控制 KB 中各领域文档的比例
# query 分布是均匀的 [0.25, 0.25, 0.25, 0.25]
KB_DIST_SETTINGS = [
    ("Aligned",   [0.25, 0.25, 0.25, 0.25]),
    ("Mild",      [0.40, 0.25, 0.20, 0.15]),
    ("Moderate",  [0.55, 0.25, 0.12, 0.08]),
    ("Severe",    [0.70, 0.15, 0.10, 0.05]),
    ("Extreme",   [0.85, 0.10, 0.03, 0.02]),
]


# ==========================================
# 随机种子
# ==========================================
def set_random_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"🎲 Random seed set to {seed}")


# ==========================================
# 数据加载与领域分类
# ==========================================
def load_and_classify_hotpotqa(max_per_domain: int = MAX_PER_DOMAIN) -> tuple:
    """
    加载 HotpotQA 并分类

    Returns:
        domain_questions: {domain: [{"question", "answer", "gold_texts"}, ...]}
        domain_doc_pool:  {domain: [text, ...]}  每个领域的全部文档池
    """
    from datasets import load_dataset

    print("🔍 Loading distilled detector...")
    detector = get_detector()

    print(f"📥 Loading HotpotQA (max {max_per_domain}/domain)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)

    domain_questions: Dict[str, List[Dict]] = {d: [] for d in DOMAIN_NAMES}
    domain_doc_pool: Dict[str, set] = {d: set() for d in DOMAIN_NAMES}

    buffer_q, buffer_items = [], []
    total = 0

    for item in ds:
        buffer_q.append(item['question'])
        buffer_items.append(item)

        if len(buffer_q) >= CLASSIFY_BATCH_SIZE:
            labels = classify_queries(buffer_q, detector)
            for lbl, it in zip(labels, buffer_items):
                if lbl not in LABEL_MAP or len(domain_questions[lbl]) >= max_per_domain:
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
                    domain_doc_pool[lbl].add(text)

            total += len(buffer_q)
            buffer_q.clear()
            buffer_items.clear()

            if total % 5000 == 0:
                stats = {d: len(domain_questions[d]) for d in DOMAIN_NAMES}
                print(f"    📈 Processed {total}, Q per domain: {stats}")

            if all(len(domain_questions[d]) >= max_per_domain for d in DOMAIN_NAMES):
                break

    domain_doc_pool_list = {d: list(domain_doc_pool[d]) for d in DOMAIN_NAMES}

    print("  📊 Domain statistics:")
    for d in DOMAIN_NAMES:
        print(f"    [{d:>15s}]: {len(domain_questions[d]):>5d} Q, "
              f"{len(domain_doc_pool_list[d]):>6d} unique docs")

    return domain_questions, domain_doc_pool_list


# ==========================================
# 采样 test queries（从末尾取，与文档池减少重叠）
# ==========================================
def sample_test_queries(
    domain_questions: Dict[str, List[Dict]],
    n_per_domain: int = N_Q_PER_DOMAIN,
) -> List[Dict]:
    test_q = []
    for d in DOMAIN_NAMES:
        pool = domain_questions[d]
        start = max(0, len(pool) - n_per_domain)
        selected = pool[start:start + n_per_domain]
        test_q.extend(selected)
        print(f"    🎯 {d}: {len(selected)} queries")
    random.shuffle(test_q)
    return test_q


# ==========================================
# 核心变化: KB 完全按分布采样（不强制包含 gold）
# ==========================================
def build_kb_by_distribution(
    domain_doc_pool: Dict[str, List[str]],
    kb_dist: np.ndarray,
    kb_total: int = KB_TOTAL_SIZE,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    完全按领域分布从文档池中采样构建 KB
    
    关键区别: 不强制包含 gold 文档！
    如果某个领域分配到的文档数很少，该领域 query 的 gold doc
    很可能不在 KB 中 → 检索自然失败
    
    Args:
        domain_doc_pool: {domain: [doc_text, ...]}
        kb_dist: 各领域在 KB 中的比例 [p1, p2, p3, p4]
        kb_total: KB 总文档数
    
    Returns:
        kb_docs: Document 列表
        domain_counts: {domain: 实际采样数}
    """
    from langchain_core.documents import Document

    kb_docs = []
    domain_counts = {}

    for i, d in enumerate(DOMAIN_NAMES):
        n_need = int(kb_total * kb_dist[i])
        pool = domain_doc_pool[d]

        if len(pool) < n_need:
            print(f"    ⚠️ {d}: 需要 {n_need}, 可用 {len(pool)}, 取全部")
            sampled = pool[:]
        else:
            sampled = random.sample(pool, n_need)

        domain_counts[d] = len(sampled)
        for j, text in enumerate(sampled):
            
            kb_docs.append(Document(
                page_content=text,
                metadata={"doc_id": f"{d}_{j}", "domain": d,"source":"hotpotqa"},
            ))

    random.shuffle(kb_docs)

    print(f"    📦 KB: total={len(kb_docs)}, "
          f"per domain: {', '.join(f'{d}={domain_counts[d]}' for d in DOMAIN_NAMES)}")

    return kb_docs, domain_counts


# ==========================================
# 计算 gold 文档覆盖率（新增关键指标）
# ==========================================
def compute_gold_coverage(
    test_queries: List[Dict],
    kb_docs: List[Any],
) -> Dict[str, float]:
    """
    计算 KB 中包含了多少 test query 的 gold 文档
    
    这是核心指标: 如果 gold doc 不在 KB 中，检索必然失败
    """
    kb_texts = set(normalize_answer(doc.page_content) for doc in kb_docs)

    total_queries = len(test_queries)
    queries_with_gold = 0  # 至少有一个 gold doc 在 KB 中的 query 数
    total_gold_docs = 0
    gold_docs_in_kb = 0

    # 按领域统计
    domain_stats = {d: {"total": 0, "covered": 0} for d in DOMAIN_NAMES}

    for item in test_queries:
        domain = item.get("domain", "unknown")
        gold_texts = item["gold_texts"]
        total_gold_docs += len(gold_texts)

        found_any = False
        for g in gold_texts:
            g_norm = normalize_answer(g)
            if any(g_norm in kb_t or kb_t in g_norm for kb_t in kb_texts):
                gold_docs_in_kb += 1
                found_any = True

        if found_any:
            queries_with_gold += 1

        if domain in domain_stats:
            domain_stats[domain]["total"] += 1
            if found_any:
                domain_stats[domain]["covered"] += 1

    coverage = queries_with_gold / total_queries if total_queries > 0 else 0
    doc_coverage = gold_docs_in_kb / total_gold_docs if total_gold_docs > 0 else 0

    print(f"    📊 Gold Coverage: {queries_with_gold}/{total_queries} queries "
          f"({coverage:.1%}), {gold_docs_in_kb}/{total_gold_docs} docs ({doc_coverage:.1%})")
    for d in DOMAIN_NAMES:
        s = domain_stats[d]
        if s["total"] > 0:
            pct = s["covered"] / s["total"]
            print(f"       {d}: {s['covered']}/{s['total']} ({pct:.1%})")

    return {
        "query_coverage": coverage,
        "doc_coverage": doc_coverage,
        "domain_coverage": {
            d: (domain_stats[d]["covered"] / domain_stats[d]["total"]
                if domain_stats[d]["total"] > 0 else 0)
            for d in DOMAIN_NAMES
        },
    }


# ==========================================
# 检索评估
# ==========================================
def evaluate_retrieval_hit_rate(
    test_queries: List[Dict],
    kb_docs: List[Any],
    model_name: str = RETRIEVER_MODEL,
    use_hybrid: bool = USE_HYBRID,
    top_k: int = TOP_K,
) -> Tuple[float, Dict[str, float]]:
    """评估 Hit@K，同时返回各领域的 hit rate"""
    pipe = build_pipeline(kb_docs, model_name, use_hybrid)

    domain_hits = {d: 0 for d in DOMAIN_NAMES}
    domain_total = {d: 0 for d in DOMAIN_NAMES}

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
            print(f"  检索失败: {query[:50]}... | {e}")
            ret_texts = []

        hit = False
        for gt in gold_set:
            if any(gt in rd or rd in gt for rd in ret_texts):
                hit = True
                break
        if hit and domain in domain_hits:
            domain_hits[domain] += 1

    del pipe
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    total = len(test_queries)
    total_hits = sum(domain_hits.values())
    overall = total_hits / total if total > 0 else 0

    domain_rates = {}
    for d in DOMAIN_NAMES:
        if domain_total[d] > 0:
            domain_rates[d] = domain_hits[d] / domain_total[d]
        else:
            domain_rates[d] = 0

    return overall, domain_rates


def evaluate_end2end_accuracy(
    test_queries: List[Dict],
    kb_docs: List[Any],
    model_name: str = RETRIEVER_MODEL,
    use_hybrid: bool = USE_HYBRID,
) -> float:
    """端到端准确率"""
    eval_items = [{"q": item["question"], "a": item["answer"]} for item in test_queries]
    pipe = build_pipeline(kb_docs, model_name, use_hybrid)
    acc = evaluate_batch(pipe, eval_items, BATCH_SIZE)

    del pipe
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return acc


# ==========================================
# 快速验证: 不调 retriever，仅检查 gold 覆盖率
# ==========================================
def quick_verify():
    """
    快速验证模式: 不构建 retriever，只检查 gold doc 覆盖率
    如果覆盖率随 JSD 下降，说明实验逻辑正确，再跑完整实验
    预计耗时: 数据加载 ~5min，验证 ~10s
    """
    print("=" * 60)
    print("⚡ Quick Verify: 仅检查 Gold Coverage (不调 retriever)")
    print("=" * 60)

    set_random_seed()

    # 加载数据
    domain_questions, domain_doc_pool = load_and_classify_hotpotqa()

    # 采样 queries
    print("\n📝 Sampling test queries...")
    test_queries = sample_test_queries(domain_questions)
    print(f"   Total: {len(test_queries)}")

    query_dist = np.array([0.25, 0.25, 0.25, 0.25])

    results = []
    print(f"\n🔬 Checking gold coverage for each KB distribution...\n")

    for label, kb_dist_raw in KB_DIST_SETTINGS:
        kb_dist = np.array(kb_dist_raw, dtype=float)
        kb_dist /= kb_dist.sum()
        jsd = compute_jsd(query_dist, kb_dist)

        print(f"  [{label}] KB dist: {kb_dist.tolist()}, JSD={jsd:.4f}")

        kb_docs, domain_counts = build_kb_by_distribution(domain_doc_pool, kb_dist)
        coverage = compute_gold_coverage(test_queries, kb_docs)

        results.append({
            "label": label,
            "jsd": jsd,
            "kb_dist": kb_dist.tolist(),
            "kb_size": len(kb_docs),
            "domain_counts": domain_counts,
            "query_coverage": coverage["query_coverage"],
            "doc_coverage": coverage["doc_coverage"],
            "domain_coverage": coverage["domain_coverage"],
        })
        print()

    # 打印汇总
    print("\n" + "=" * 70)
    print("📋 Quick Verify 汇总")
    print("=" * 70)
    print(f"  {'Setting':<12s} {'JSD':>6s} {'Query Coverage':>16s} {'Doc Coverage':>14s}  Per-Domain Coverage")
    print(f"  {'-'*12} {'-'*6} {'-'*16} {'-'*14}  {'-'*40}")
    for r in results:
        domain_str = "  ".join(f"{d[:3]}={r['domain_coverage'][d]:.0%}" for d in DOMAIN_NAMES)
        print(f"  {r['label']:<12s} {r['jsd']:>6.3f} {r['query_coverage']:>14.1%} "
              f"{r['doc_coverage']:>14.1%}  {domain_str}")

    # 检查趋势
    coverages = [r["query_coverage"] for r in results]
    if coverages[0] > coverages[-1]:
        drop = coverages[0] - coverages[-1]
        print(f"\n  ✅ 趋势正确! Coverage 从 {coverages[0]:.1%} 降到 {coverages[-1]:.1%} (Δ={drop:.1%})")
        print(f"  💡 可以跑完整实验了 (python {Path(__file__).name} --full)")
    else:
        print(f"\n  ❌ 趋势不对: Coverage 从 {coverages[0]:.1%} 到 {coverages[-1]:.1%}")
        print(f"  💡 可能需要调整 KB_TOTAL_SIZE 或文档池大小")

    # 保存
    out_path = OUT_DIR / "results_misalignment_v2_quick.json"
    save_results(results, str(out_path))
    print(f"\n💾 Saved: {out_path}")

    # 绘制 coverage 趋势图
    plot_quick_verify(results)

    return results


def plot_quick_verify(results):
    """绘制快速验证的 coverage 趋势图"""
    setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = [r["label"] for r in results]
    jsd_vals = [r["jsd"] for r in results]
    q_cov = [r["query_coverage"] for r in results]
    d_cov = [r["doc_coverage"] for r in results]
    x = range(len(results))

    # ===== 左图: Coverage vs JSD =====
    ax1.bar(x, q_cov, width=0.4, color=COLORS['primary'], alpha=0.8,
            edgecolor='black', label='Query Coverage')

    for xi, yi in zip(x, q_cov):
        ax1.text(xi, yi + 0.015, f"{yi:.1%}", ha='center', fontsize=10, fontweight='bold')

    ax1_jsd = ax1.twinx()
    ax1_jsd.plot(list(x), jsd_vals, 'D-', color=COLORS['secondary'],
                 linewidth=2.5, markersize=8, label='JSD')
    for xi, yi in zip(x, jsd_vals):
        ax1_jsd.text(xi + 0.15, yi + 0.01, f"{yi:.3f}", fontsize=9,
                     color=COLORS['secondary'], fontweight='bold')

    ax1.set_ylabel("Gold Doc Coverage", fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1_jsd.set_ylabel("JSD", fontsize=12, color=COLORS['secondary'])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_title("JSD↑ → Gold Coverage↓ (Quick Verify)", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_jsd.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # ===== 右图: 各领域 coverage 对比 =====
    bar_width = 0.15
    for i, d in enumerate(DOMAIN_NAMES):
        d_vals = [r["domain_coverage"][d] for r in results]
        positions = [xi + i * bar_width for xi in x]
        ax2.bar(positions, d_vals, width=bar_width, label=d[:12], alpha=0.8, edgecolor='black')

    ax2.set_ylabel("Per-Domain Query Coverage", fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks([xi + bar_width * 1.5 for xi in x])
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_title("Per-Domain Coverage Breakdown", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = str(OUT_DIR / "fig_misalignment_v2_quick_verify.png")
    save_fig(fig, out)
    plt.close()
    print(f"  ✅ Quick verify figure saved: {out}")


# ==========================================
# 完整实验
# ==========================================
def run_full():
    """完整实验: 构建 retriever，评估 Hit Rate + Accuracy"""
    print("=" * 60)
    print("📊 Motivation 2B v2: KB-Query Misalignment (Full)")
    print(f"   Retriever: {RETRIEVER_DISPLAY}")
    print(f"   KB Size: {KB_TOTAL_SIZE}")
    print(f"   Queries: {N_Q_PER_DOMAIN * NUM_DOMAINS}")
    print("=" * 60)

    set_random_seed()

    domain_questions, domain_doc_pool = load_and_classify_hotpotqa()

    print("\n📝 Sampling test queries...")
    test_queries = sample_test_queries(domain_questions)
    print(f"   Total: {len(test_queries)}")

    query_dist = np.array([0.25, 0.25, 0.25, 0.25])

    results = []

    for label, kb_dist_raw in KB_DIST_SETTINGS:
        kb_dist = np.array(kb_dist_raw, dtype=float)
        kb_dist /= kb_dist.sum()
        jsd = compute_jsd(query_dist, kb_dist)

        print(f"\n{'='*55}")
        print(f"  [{label}] KB dist: {kb_dist.tolist()}, JSD={jsd:.4f}")

        # 按分布构建 KB（不强制包含 gold）
        kb_docs, domain_counts = build_kb_by_distribution(domain_doc_pool, kb_dist)

        # 检查 gold 覆盖率
        coverage = compute_gold_coverage(test_queries, kb_docs)

        # Hit@K
        print(f"  [{label}] Evaluating Hit@{TOP_K}...")
        hit_rate, domain_hit_rates = evaluate_retrieval_hit_rate(test_queries, kb_docs)

        # End-to-end accuracy
        print(f"  [{label}] Evaluating end-to-end accuracy...")
        accuracy = evaluate_end2end_accuracy(test_queries, kb_docs)

        r = {
            "label": label,
            "jsd": jsd,
            "kb_dist": kb_dist.tolist(),
            "kb_size": len(kb_docs),
            "domain_counts": domain_counts,
            "query_coverage": coverage["query_coverage"],
            "doc_coverage": coverage["doc_coverage"],
            "domain_coverage": coverage["domain_coverage"],
            "hit_rate": hit_rate,
            "domain_hit_rates": domain_hit_rates,
            "accuracy": accuracy,
        }
        results.append(r)
        print(f"  ✅ {label:10s}: JSD={jsd:.4f}, Coverage={coverage['query_coverage']:.1%}, "
              f"Hit@{TOP_K}={hit_rate:.2%}, Acc={accuracy:.2%}")

    # 保存
    out_path = OUT_DIR / "results_misalignment_v2.json"
    save_results(results, str(out_path))
    print(f"\n💾 Saved: {out_path}")

    plot_full(results)
    return results


# ==========================================
# 完整版绘图
# ==========================================
def plot_full(results=None):
    """绘制完整实验结果"""
    if results is None:
        data = json.loads((OUT_DIR / "results_misalignment_v2.json").read_text())
        results = data if isinstance(data, list) else data.get("results", data)

    setup_style()

    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.8, 0.8], wspace=0.35)

    ax_main = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cov = fig.add_subplot(gs[2])

    x = range(len(results))
    labels = [r["label"] for r in results]
    jsd_vals = [r["jsd"] for r in results]
    hit_vals = [r["hit_rate"] for r in results]
    acc_vals = [r.get("accuracy", 0) for r in results]
    cov_vals = [r["query_coverage"] for r in results]

    # ========== 左图: Hit + Accuracy + JSD ==========
    bar_width = 0.3
    x_hit = [xi - bar_width / 2 for xi in x]
    x_acc = [xi + bar_width / 2 for xi in x]

    ax_main.bar(x_hit, hit_vals, width=bar_width,
                color=COLORS['primary'], edgecolor='black', alpha=0.8,
                label=f'Hit@{TOP_K}')
    ax_main.bar(x_acc, acc_vals, width=bar_width,
                color=COLORS['accent1'], edgecolor='black', alpha=0.8,
                label='End-to-End Acc')

    for xi, yi in zip(x_hit, hit_vals):
        ax_main.text(xi, yi + 0.01, f"{yi:.1%}", ha='center', fontsize=8, fontweight='bold')
    for xi, yi in zip(x_acc, acc_vals):
        ax_main.text(xi, yi + 0.01, f"{yi:.1%}", ha='center', fontsize=8, fontweight='bold')

    ax_main.set_ylabel("Hit Rate / Accuracy", fontsize=12)
    ax_main.set_ylim(0, max(hit_vals + acc_vals) * 1.35)
    ax_main.legend(loc='upper left', fontsize=9)

    ax_jsd = ax_main.twinx()
    ax_jsd.plot(list(x), jsd_vals, 'D-', color=COLORS['secondary'],
                linewidth=2.5, markersize=8, label='JSD(KB ∥ Query)')
    ax_jsd.set_ylabel("JS Divergence", fontsize=12, color=COLORS['secondary'])
    ax_jsd.set_ylim(0, max(jsd_vals) * 1.4)
    ax_jsd.legend(loc='upper right', fontsize=9)

    x_labels = []
    for r in results:
        dist = r["kb_dist"]
        dist_str = "[" + ",".join(f".{int(d * 100):02d}" for d in dist) + "]"
        x_labels.append(f"{r['label']}\n{dist_str}\nKB={r['kb_size']}")

    ax_main.set_xticks(list(x))
    ax_main.set_xticklabels(x_labels, fontsize=7)
    ax_main.set_xlabel("KB Distribution Setting", fontsize=11)
    ax_main.set_title("(a) JSD↑ → Hit Rate↓ & Accuracy↓\n(KB sampled by distribution, no forced gold)",
                      fontsize=12, fontweight='bold')
    ax_main.grid(axis='y', alpha=0.3)

    # ========== 中图: 分布热力图 ==========
    query_dist = [0.25, 0.25, 0.25, 0.25]
    row_labels = ["Query\n(uniform)"]
    for r in results:
        row_labels.append(f"{r['label']}\nJSD={r['jsd']:.3f}")

    matrix = np.array([query_dist] + [r["kb_dist"] for r in results])
    im = ax_heat.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax_heat.text(j, i, f"{val:.2f}", ha='center', va='center',
                         fontsize=9, fontweight='bold', color=color)

    domain_short = [d[:3].upper() for d in DOMAIN_NAMES]
    ax_heat.set_xticks(range(NUM_DOMAINS))
    ax_heat.set_xticklabels(domain_short, fontsize=9)
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=7)
    ax_heat.set_title("(b) KB Distribution", fontsize=12, fontweight='bold')
    ax_heat.axhline(y=0.5, color='white', linewidth=3)
    fig.colorbar(im, ax=ax_heat, shrink=0.7, label="Probability")

    # ========== 右图: Gold Coverage + 各领域 ==========
    ax_cov.bar(x, cov_vals, width=0.5, color='coral', edgecolor='black', alpha=0.8,
               label='Query Coverage')
    for xi, yi in zip(x, cov_vals):
        ax_cov.text(xi, yi + 0.015, f"{yi:.1%}", ha='center', fontsize=9, fontweight='bold')

    ax_cov.set_ylabel("Gold Doc Coverage", fontsize=12)
    ax_cov.set_ylim(0, 1.15)
    ax_cov.set_xticks(list(x))
    ax_cov.set_xticklabels(labels, fontsize=9)
    ax_cov.set_title("(c) Gold Coverage in KB\n(the real reason)", fontsize=12, fontweight='bold')
    ax_cov.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = str(OUT_DIR / "fig_kb_query_misalignment_v2.png")
    save_fig(fig, out)
    plt.close()
    print(f"  ✅ Figure saved: {out}")


# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick verify: 只检查 gold coverage，不调 retriever")
    parser.add_argument("--full", action="store_true",
                        help="Full experiment: 构建 retriever + 评估")
    parser.add_argument("--plot-only", action="store_true",
                        help="仅绘图")
    args = parser.parse_args()

    if args.plot_only:
        plot_full()
    elif args.full:
        run_full()
    else:
        # 默认跑 quick verify
        quick_verify()