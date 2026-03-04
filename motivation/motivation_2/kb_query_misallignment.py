"""
Motivation 2B: KB-Query Distribution Misalignment → Retrieval Accuracy Drop

核心思路:
  - 每个 test query 的 gold 文档始终在 KB 中（不人为排除）
  - 按不同领域比例向 KB 注入大量噪声文档
  - 噪声越多 / 分布越偏 → gold 被淹没 → Hit Rate & Accuracy 下降
  - 展示 JSD↑ → Hit Rate↓ / Accuracy↓ 的因果关系

实验流程:
  1. 加载数据，按领域分类
  2. 采样 test queries（各领域均匀）
  3. 构建噪声池（每个领域的非 gold 文档）
  4. 对每种 KB 分布设置：gold docs + 按比例采样噪声 → 组成 KB
  5. 用真实 retriever 评估

输出: motivation_2/fig_kb_query_misalignment.png
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
# 实验配置
# ==========================================
MAX_PER_DOMAIN = 5000       # 每个领域采集 5000 个 question
CLASSIFY_BATCH_SIZE = 512   # 批量分类大小
N_Q_PER_DOMAIN = 100        # 每个领域 100 个 test query，总共 400 个
NOISE_RATIO = 100           # 噪声比 = 噪声文档数 / gold文档数 (1:100)
TOP_K = 10                  # Hit@K
BATCH_SIZE = 32             # 评估批量大小
RANDOM_SEED = 42

# Retriever 配置
RETRIEVER_MODEL = "BAAI/bge-small-en-v1.5"
RETRIEVER_DISPLAY = "BGE-Small (Dense)"
USE_HYBRID = False

# KB 噪声分布设置：控制噪声文档在各领域的比例
# query 分布是均匀的 [0.25, 0.25, 0.25, 0.25]
# 噪声分布从对齐到极端偏斜
KB_NOISE_SETTINGS = [
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
def load_and_classify_hotpotqa(
    max_per_domain: int = MAX_PER_DOMAIN,
) -> tuple:
    """
    加载 HotpotQA 数据并用蒸馏分类器进行真实领域划分

    Returns:
        domain_questions: {domain: [{"question", "answer", "gold_texts", "all_texts"}, ...]}
        domain_noise_pool: {domain: [text, ...]}  各领域的噪声文档池（去重）
    """
    from datasets import load_dataset

    print("🔍 Loading distilled detector...")
    detector = get_detector()

    print(f"📥 Loading HotpotQA and classifying by domain (max {max_per_domain}/domain)...")
    ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)

    domain_questions: Dict[str, List[Dict]] = {d: [] for d in DOMAIN_NAMES}
    # 收集每个领域的所有文档（用于构建噪声池）
    domain_all_docs: Dict[str, set] = {d: set() for d in DOMAIN_NAMES}

    buffer_questions: List[str] = []
    buffer_items: List[Any] = []
    total_processed = 0

    for item in ds:
        buffer_questions.append(item['question'])
        buffer_items.append(item)

        if len(buffer_questions) >= CLASSIFY_BATCH_SIZE:
            _process_batch(
                buffer_questions, buffer_items,
                detector, domain_questions, domain_all_docs,
                max_per_domain
            )
            total_processed += len(buffer_questions)
            buffer_questions.clear()
            buffer_items.clear()

            if total_processed % 5000 == 0:
                stats = {d: len(domain_questions[d]) for d in DOMAIN_NAMES}
                print(f"    📈 Processed {total_processed}, Q per domain: {stats}")

        if all(len(domain_questions[d]) >= max_per_domain for d in DOMAIN_NAMES):
            break

    if buffer_questions:
        _process_batch(
            buffer_questions, buffer_items,
            detector, domain_questions, domain_all_docs,
            max_per_domain
        )

    # 转换 set → list
    domain_all_docs_list: Dict[str, List[str]] = {
        d: list(domain_all_docs[d]) for d in DOMAIN_NAMES
    }

    print("  📊 Domain statistics:")
    for d in DOMAIN_NAMES:
        print(f"    [{d:>15s}]: {len(domain_questions[d]):>5d} Q, "
              f"{len(domain_all_docs_list[d]):>6d} unique docs")

    return domain_questions, domain_all_docs_list


def _process_batch(
    questions: List[str],
    items: List[Any],
    detector,
    domain_questions: Dict[str, List],
    domain_all_docs: Dict[str, set],
    max_per_domain: int,
):
    """处理一个批次的分类和数据提取"""
    labels = classify_queries(questions, detector)

    for lbl, item in zip(labels, items):
        if lbl not in LABEL_MAP:
            continue
        if len(domain_questions[lbl]) >= max_per_domain:
            continue

        titles = item['context']['title']
        sentences = item['context']['sentences']
        sf_titles = set(item['supporting_facts']['title'])

        gold_texts, all_texts = [], []
        for title, sent_list in zip(titles, sentences):
            text = f"{title}: " + " ".join(sent_list)
            all_texts.append(text)
            if title in sf_titles:
                gold_texts.append(text)

        if not gold_texts:
            continue

        domain_questions[lbl].append({
            "question": item['question'],
            "answer": item['answer'],
            "gold_texts": gold_texts,
            "all_texts": all_texts,
        })
        # 所有文档都加入该领域的文档池
        for text in all_texts:
            domain_all_docs[lbl].add(text)


# ==========================================
# 采样 test queries
# ==========================================
def _sample_test_queries(
    domain_questions: Dict[str, List[Dict]],
    n_per_domain: int = N_Q_PER_DOMAIN,
) -> List[Dict]:
    """
    从各领域均匀采样测试 query。
    从列表末尾取，减少与噪声池的重叠。
    """
    test_q = []
    for d in DOMAIN_NAMES:
        pool = domain_questions[d]
        start = max(0, len(pool) - n_per_domain)
        selected = pool[start:start + n_per_domain]
        test_q.extend(selected)
        print(f"    🎯 {d}: 选取 query [{start}:{start + len(selected)}] = {len(selected)} 条")
    random.shuffle(test_q)
    return test_q


# ==========================================
# 构建噪声池（排除 gold 文档）
# ==========================================
def _build_noise_pools(
    domain_all_docs: Dict[str, List[str]],
    test_queries: List[Dict],
) -> Dict[str, List[str]]:
    """
    为每个领域构建噪声文档池 = 全部文档 - test queries 的 gold 文档
    
    这样可以保证：
    - gold 文档只通过 test_queries 注入 KB（作为信号）
    - 噪声池里不包含 gold 文档（纯粹的噪声）
    """
    # 收集所有 gold 文档
    all_gold = set()
    for item in test_queries:
        for gt in item["gold_texts"]:
            all_gold.add(gt)

    noise_pools = {}
    for d in DOMAIN_NAMES:
        pool = [t for t in domain_all_docs[d] if t not in all_gold]
        noise_pools[d] = pool
        print(f"    🗃️ {d} noise pool: {len(pool)} docs "
              f"(removed {len(domain_all_docs[d]) - len(pool)} gold docs)")

    return noise_pools


# ==========================================
# 按比例构建 KB: gold + 噪声
# ==========================================
def _build_kb(
    test_queries: List[Dict],
    noise_pools: Dict[str, List[str]],
    noise_dist: np.ndarray,
    noise_ratio: int = NOISE_RATIO,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    构建 KB = gold 文档 + 按比例采样的噪声文档
    
    参数:
        test_queries: 测试 query 列表（包含 gold_texts）
        noise_pools: {domain: [噪声文档列表]}
        noise_dist: 噪声在各领域的分布比例 [p1, p2, p3, p4]
        noise_ratio: 噪声文档总数 / gold文档总数
    
    Returns:
        kb_docs: Document 列表
        stats: 统计信息
    """
    from langchain_core.documents import Document

    # ===== Step 1: 收集所有 gold 文档（去重） =====
    gold_doc_set = set()
    for item in test_queries:
        for gt in item["gold_texts"]:
            gold_doc_set.add(gt)
    gold_docs_list = list(gold_doc_set)
    n_gold = len(gold_docs_list)

    # ===== Step 2: 计算噪声文档总数 =====
    n_noise_total = n_gold * noise_ratio
    print(f"    📊 Gold docs: {n_gold}, Noise total: {n_noise_total} "
          f"(ratio 1:{noise_ratio})")

    # ===== Step 3: 按分布从各领域采样噪声文档 =====
    noise_docs_list = []
    noise_counts = {}
    for i, d in enumerate(DOMAIN_NAMES):
        n_need = int(n_noise_total * noise_dist[i])
        pool = noise_pools[d]

        if len(pool) < n_need:
            print(f"    ⚠️ {d}: 需要 {n_need} 噪声 docs, 可用 {len(pool)}, 取全部")
            sampled = pool[:]
        else:
            sampled = random.sample(pool, n_need)

        noise_counts[d] = len(sampled)
        noise_docs_list.extend(sampled)

    # ===== Step 4: 合并 gold + noise → KB =====
    kb_docs = []
    # 添加 gold 文档
    for j, text in enumerate(gold_docs_list):
        kb_docs.append(Document(
            page_content=text,
            metadata={"doc_id": f"gold_{j}", "source": "gold"},
        ))
    # 添加噪声文档
    for j, text in enumerate(noise_docs_list):
        kb_docs.append(Document(
            page_content=text,
            metadata={"doc_id": f"noise_{j}", "source": "noise"},
        ))

    # 打乱顺序
    random.shuffle(kb_docs)

    total_kb = len(kb_docs)
    print(f"    📦 KB 构建完成: gold={n_gold}, "
          f"noise={len(noise_docs_list)} "
          f"({', '.join(f'{d}={noise_counts[d]}' for d in DOMAIN_NAMES)}), "
          f"total={total_kb}")

    stats = {
        "n_gold": n_gold,
        "n_noise": len(noise_docs_list),
        "n_total": total_kb,
        "noise_per_domain": noise_counts,
    }
    return kb_docs, stats


# ==========================================
# 真实检索评估
# ==========================================
def evaluate_retrieval_hit_rate(
    test_queries: List[Dict],
    kb_docs: List[Any],
    model_name: str = RETRIEVER_MODEL,
    use_hybrid: bool = USE_HYBRID,
    top_k: int = TOP_K,
) -> float:
    """用真实 Dense retriever 评估 Hit@K"""
    pipe = build_pipeline(kb_docs, model_name, use_hybrid)

    hits = 0
    total = len(test_queries)

    for i in range(0, total, BATCH_SIZE):
        batch = test_queries[i:i + BATCH_SIZE]
        for item in batch:
            query = item["question"]
            gold_set = set(normalize_answer(g) for g in item["gold_texts"])

            try:
                retrieved_results = pipe.retrieve(
                    query,
                    rerank_top_k=top_k,
                    rerank_threshold=0.0
                )
                retrieved_texts = [
                    normalize_answer(res["text"])
                    for res in retrieved_results
                ]
            except Exception as e:
                print(f"检索失败: {query[:50]}... | {e}")
                retrieved_texts = []

            for gt in gold_set:
                if any(gt in rd or rd in gt for rd in retrieved_texts):
                    hits += 1
                    break

    del pipe
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    return hits / total if total > 0 else 0.0


def evaluate_end2end_accuracy(
    test_queries: List[Dict],
    kb_docs: List[Any],
    model_name: str = RETRIEVER_MODEL,
    use_hybrid: bool = USE_HYBRID,
) -> float:
    """用真实 RAG pipeline 评估端到端准确率"""
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
# 核心实验流程
# ==========================================
def run():
    """Part B: KB-Query Distribution Misalignment 实验"""
    print("=" * 60)
    print("📊 Motivation 2B: KB-Query Distribution Misalignment")
    print(f"   Retriever: {RETRIEVER_DISPLAY}")
    print(f"   Noise Ratio: 1:{NOISE_RATIO}")
    print(f"   Queries: {N_Q_PER_DOMAIN * NUM_DOMAINS}")
    print("=" * 60)

    set_random_seed()

    # ===== 1. 数据加载与分类 =====
    domain_questions, domain_all_docs = load_and_classify_hotpotqa()

    # ===== 2. 采样 test queries（所有设置共享同一组 query）=====
    print("\n📝 Sampling test queries...")
    test_queries = _sample_test_queries(domain_questions)
    print(f"   Total test queries: {len(test_queries)}")

    # ===== 3. 构建噪声池 =====
    print("\n🗃️ Building noise pools (excluding gold docs)...")
    noise_pools = _build_noise_pools(domain_all_docs, test_queries)

    # ===== 4. Query 分布（均匀）=====
    query_dist = np.array([0.25, 0.25, 0.25, 0.25])

    # ===== 5. 对每种噪声分布设置：构建 KB → 评估 =====
    results = []
    print(f"\n🔬 Running misalignment experiment...")

    for label, noise_dist_raw in KB_NOISE_SETTINGS:
        noise_dist = np.array(noise_dist_raw, dtype=float)
        noise_dist /= noise_dist.sum()

        jsd = compute_jsd(query_dist, noise_dist)
        print(f"\n{'='*55}")
        print(f"  [{label}] Noise dist: {noise_dist.tolist()}, JSD={jsd:.4f}")

        # 构建 KB = gold(固定) + noise(按分布采样)
        kb_docs, kb_stats = _build_kb(test_queries, noise_pools, noise_dist)

        # 评估 Hit@K
        print(f"  [{label}] Evaluating Hit@{TOP_K}...")
        hit_rate = evaluate_retrieval_hit_rate(test_queries, kb_docs)

        # 评估端到端准确率
        print(f"  [{label}] Evaluating end-to-end accuracy...")
        accuracy = evaluate_end2end_accuracy(test_queries, kb_docs)

        results.append({
            "label": label,
            "jsd": jsd,
            "hit_rate": hit_rate,
            "accuracy": accuracy,
            "noise_dist": noise_dist.tolist(),
            "kb_stats": kb_stats,
        })
        print(f"  ✅ {label:10s}: JSD={jsd:.4f}, Hit@{TOP_K}={hit_rate:.2%}, "
              f"Acc={accuracy:.2%}, KB={kb_stats['n_total']}")

    # ===== 保存结果 =====
    out_path = OUT_DIR / "results_misalignment.json"
    save_results(results, str(out_path))
    print(f"\n💾 Results saved: {out_path}")

    # ===== 绘图 =====
    plot(results)

    return results


# ==========================================
# 绘图
# ==========================================
def plot(results=None):
    """绘制 KB-Query Misalignment 图: JSD↑ → Hit Rate↓ / Accuracy↓"""

    if results is None:
        data = json.loads((OUT_DIR / "results_misalignment.json").read_text())
        results = data if isinstance(data, list) else data.get("results", data)

    setup_style()

    fig, (ax_main, ax_heat) = plt.subplots(
        1, 2, figsize=(16, 6.5),
        gridspec_kw={'width_ratios': [1.5, 1]}
    )

    x = range(len(results))
    labels = [r["label"] for r in results]
    jsd_vals = [r["jsd"] for r in results]
    hit_vals = [r["hit_rate"] for r in results]
    acc_vals = [r.get("accuracy", None) for r in results]
    has_accuracy = all(a is not None for a in acc_vals)

    # ========== 左图: Hit Rate + Accuracy 柱状 + JSD 折线 ==========
    if has_accuracy:
        bar_width = 0.35
        x_hit = [xi - bar_width / 2 for xi in x]
        x_acc = [xi + bar_width / 2 for xi in x]

        ax_main.bar(x_hit, hit_vals, width=bar_width,
                    color=COLORS['primary'], edgecolor='black',
                    linewidth=1.0, alpha=0.8, zorder=3, label=f'Hit@{TOP_K}')
        ax_main.bar(x_acc, acc_vals, width=bar_width,
                    color=COLORS['accent1'], edgecolor='black',
                    linewidth=1.0, alpha=0.8, zorder=3, label='End-to-End Acc')

        for xi, yi in zip(x_hit, hit_vals):
            ax_main.text(xi, yi + 0.012, f"{yi:.1%}", ha='center', fontsize=9,
                         fontweight='bold', color=COLORS['primary'])
        for xi, yi in zip(x_acc, acc_vals):
            ax_main.text(xi, yi + 0.012, f"{yi:.1%}", ha='center', fontsize=9,
                         fontweight='bold', color=COLORS['accent1'])

        all_vals = hit_vals + acc_vals
    else:
        cmap = plt.cm.RdYlGn_r
        norm = plt.Normalize(0, len(results) - 1)
        bar_colors = [cmap(norm(i)) for i in range(len(results))]

        ax_main.bar(x, hit_vals, width=0.5, color=bar_colors,
                    edgecolor='black', linewidth=1.2, alpha=0.85, zorder=3)
        for xi, yi in zip(x, hit_vals):
            ax_main.text(xi, yi + 0.015, f"{yi:.1%}", ha='center', fontsize=11,
                         fontweight='bold', color=COLORS['primary'])
        all_vals = hit_vals

    ax_main.set_ylabel("Hit Rate / Accuracy", fontsize=13, color=COLORS['primary'])
    ax_main.set_ylim(0, max(all_vals) * 1.35)
    ax_main.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax_main.legend(loc='upper left', fontsize=10)

    # JSD 折线 (右 Y 轴)
    ax_jsd = ax_main.twinx()
    ax_jsd.plot(list(x), jsd_vals, color=COLORS['secondary'], linewidth=2.5,
                marker='D', markersize=8, zorder=4, label='JSD(Noise ∥ Query)')
    ax_jsd.set_ylabel("JS Divergence", fontsize=13, color=COLORS['secondary'])
    ax_jsd.set_ylim(0, max(jsd_vals) * 1.4)
    ax_jsd.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax_jsd.spines['right'].set_color(COLORS['secondary'])

    for xi, yi in zip(x, jsd_vals):
        ax_jsd.text(xi + 0.15, yi + max(jsd_vals) * 0.03, f"{yi:.3f}", fontsize=9,
                    color=COLORS['secondary'], fontweight='bold')

    # 趋势箭头
    ax_main.annotate('', xy=(len(results) - 1, hit_vals[-1] + max(all_vals) * 0.18),
                     xytext=(0, hit_vals[0] + max(all_vals) * 0.18),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax_main.text(len(results) / 2 - 0.3, max(all_vals) * 1.28,
                 "↑ Distribution Misalignment → ↓ Retrieval & Accuracy",
                 fontsize=11, color='red', ha='center', fontstyle='italic',
                 fontweight='bold')

    # X 轴标签（显示噪声分布）
    x_tick_labels = []
    for r in results:
        dist = r["noise_dist"]
        dist_str = "[" + ",".join(f".{int(d * 100):02d}" for d in dist) + "]"
        kb_total = r.get("kb_stats", {}).get("n_total", "?")
        x_tick_labels.append(f"{r['label']}\n{dist_str}\nKB={kb_total}")

    ax_main.set_xticks(list(x))
    ax_main.set_xticklabels(x_tick_labels, fontsize=7)
    ax_main.set_xlabel(
        f"Noise Distribution Setting (noise ratio 1:{NOISE_RATIO})",
        fontsize=11
    )
    ax_main.set_title(
        f"(a) KB-Query Misalignment → Retrieval & Accuracy Drop\n"
        f"(Gold docs always in KB, drowned by noise)",
        fontsize=13, fontweight='bold'
    )
    ax_jsd.legend(loc='upper right', fontsize=10)
    ax_main.grid(axis='y', alpha=0.3)

    # ========== 右图: 分布对比热力图 ==========
    query_dist = [0.25, 0.25, 0.25, 0.25]

    row_labels = ["Query\n(uniform)"]
    for r in results:
        row_labels.append(f"{r['label']}\nJSD={r['jsd']:.3f}")

    matrix = np.array([query_dist] + [r["noise_dist"] for r in results])

    im = ax_heat.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax_heat.text(j, i, f"{val:.2f}", ha='center', va='center',
                         fontsize=10, fontweight='bold', color=color)

    domain_short = [d[:3].upper() for d in DOMAIN_NAMES]
    ax_heat.set_xticks(range(NUM_DOMAINS))
    ax_heat.set_xticklabels(domain_short, fontsize=10)
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_xlabel("Domain (classified by detector)", fontsize=11)
    ax_heat.set_title("(b) Noise Distribution Heatmap\n(Query vs KB Noise)", fontsize=13, fontweight='bold')

    ax_heat.axhline(y=0.5, color='white', linewidth=3)
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Probability")

    plt.tight_layout()
    out = str(OUT_DIR / "fig_kb_query_misalignment.png")
    save_fig(fig, out)
    plt.close()
    print(f"  ✅ Figure saved: {out}")


# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="仅绘图 (需已有结果文件)")
    args = parser.parse_args()

    if args.plot_only:
        plot()
    else:
        run()