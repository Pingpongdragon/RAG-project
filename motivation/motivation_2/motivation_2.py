"""
Motivation 2: Domain Shift → Accuracy Drop → Scaling Recovery

实验设计:
  Part A: Domain Shift + Scaling Recovery (多检索模型对比)
    - Source: SQuAD (阅读理解, 短答案)
    - Target: HotpotQA (多跳推理, 1:600 噪声比)
    - 用蒸馏分类器对 query 做领域标注，展示 shift 前后分布差异
    - 多种 retriever (Dense/Hybrid) 对比
    - 逐步增加 target 文档，观察恢复曲线

  Part B: KB-Query Distribution Misalignment → Retrieval Accuracy Drop
    - 用蒸馏分类器对 query 做真实领域分类
    - 构造不同程度的 KB 偏斜
    - 展示 JSD↑ → Hit Rate↓

输出:
  motivation_2/fig_scaling_law_recovery.png
  motivation_2/fig_kb_query_misalignment.png
  motivation_2/fig_motivation2_combined.png
"""
import sys
import gc
import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, MODEL_STYLES, save_fig
from utils import (
    normalize_answer, calculate_containment,
    build_pipeline, evaluate_batch, save_results
)

# 导入蒸馏分类器
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from detector.distill_trainer import OnlineDetector, LABEL_MAP, ID2LABEL

# ==========================================
# 全局配置
# ==========================================
TEST_SAMPLE_SIZE = 100
BATCH_SIZE = 32
NOISE_RATIO = 600               # 1:600 噪声比
SCALE_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

RETRIEVER_CONFIGS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6 (Dense)", False),
    ("BAAI/bge-small-en-v1.5",                 "BGE-Small (Dense)", False),
    ("BAAI/bge-large-en-v1.5",                 "BGE-Large (Dense)", False),
    ("BAAI/bge-small-en-v1.5",                 "BGE-Small (Hybrid)", True),
]

DETECTOR_PATH = "/home/jyliu/RAG_project/detector/mini_router_best"
DOMAIN_NAMES = list(LABEL_MAP.keys())  # ['entertainment', 'stem', 'humanities', 'lifestyle']
NUM_DOMAINS = len(DOMAIN_NAMES)

OUT_DIR = Path(__file__).resolve().parent


# ==========================================
# 领域分类工具
# ==========================================
def classify_queries(queries: List[str], detector: OnlineDetector) -> List[str]:
    """用蒸馏分类器批量分类 query 的领域"""
    results = detector.predict_batch(queries, batch_size=64)
    return [r["top_label"] for r in results]


def get_domain_distribution(labels: List[str]) -> np.ndarray:
    """计算领域分布向量 [entertainment, stem, humanities, lifestyle]"""
    counts = np.zeros(NUM_DOMAINS, dtype=float)
    for lbl in labels:
        if lbl in LABEL_MAP:
            counts[LABEL_MAP[lbl]] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    else:
        counts = np.ones(NUM_DOMAINS) / NUM_DOMAINS
    return counts


def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个分布之间的 JS 散度"""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


# ==========================================
# Part A: Domain Shift + Scaling Recovery
# ==========================================
def load_squad_and_hotpotqa():
    """加载 SQuAD + HotpotQA 数据"""
    from datasets import load_dataset
    from langchain.schema import Document

    print("🔥 Loading SQuAD + HotpotQA...")

    # Source: SQuAD validation (短答案)
    ds_squad = load_dataset("squad", split="validation")
    src_test, src_docs = [], []
    count = 0
    for item in list(ds_squad):
        if count >= TEST_SAMPLE_SIZE:
            break
        answers = item['answers']['text']
        if not answers or len(answers[0].split()) > 5:
            continue
        src_test.append({"q": item['question'], "a": answers})
        src_docs.append(Document(
            page_content=item['context'],
            metadata={"doc_id": item['id'], "source": "squad"},
        ))
        count += 1

    # Target: HotpotQA (1:600 噪声比)
    ds = load_dataset("hotpot_qa", "distractor", split="train")
    ds_list = list(ds)
    random.seed(42)
    random.shuffle(ds_list)

    target_test, gold_docs, noise_pool = [], [], []

    for item in ds_list[:TEST_SAMPLE_SIZE]:
        target_test.append({"q": item['question'], "a": item['answer']})
        titles = item['context']['title']
        sentences = item['context']['sentences']
        gold_titles = set(item['supporting_facts']['title'])
        for t, s_list in zip(titles, sentences):
            text = f"{t}: " + "".join(s_list)
            if t in gold_titles:
                gold_docs.append(Document(
                    page_content=text,
                    metadata={"doc_id": f"hp_{t}", "source": "hotpot_gold"},
                ))

    # 噪声文档 (1:600)
    target_noise_count = len(gold_docs) * NOISE_RATIO
    for item in ds_list[TEST_SAMPLE_SIZE:]:
        if len(noise_pool) >= target_noise_count:
            break
        for t, s_list in zip(item['context']['title'], item['context']['sentences']):
            if len(noise_pool) >= target_noise_count:
                break
            text = f"{t}: " + "".join(s_list)
            noise_pool.append(Document(
                page_content=text,
                metadata={"doc_id": f"noise_{len(noise_pool)}", "source": "hotpot_noise"},
            ))

    mixed_pool = gold_docs + noise_pool
    random.shuffle(mixed_pool)

    print(f"  Source: {len(src_docs)} docs, {len(src_test)} Q")
    print(f"  Target: {len(gold_docs)} gold + {len(noise_pool)} noise = {len(mixed_pool)} docs, {len(target_test)} Q")
    print(f"  Noise ratio: 1:{len(noise_pool) // max(len(gold_docs), 1)}")
    return src_docs, src_test, mixed_pool, target_test


def run_scaling_experiment():
    """Part A: Domain Shift + Scaling Recovery"""
    import torch

    src_docs, src_test, mixed_pool, target_test = load_squad_and_hotpotqa()
    total_target = len(mixed_pool)

    # 用蒸馏分类器分析 query 领域分布
    print("\n🔍 Classifying queries with distilled detector...")
    detector = OnlineDetector(DETECTOR_PATH)
    src_queries = [item["q"] for item in src_test]
    tgt_queries = [item["q"] for item in target_test]
    src_labels = classify_queries(src_queries, detector)
    tgt_labels = classify_queries(tgt_queries, detector)

    src_dist = get_domain_distribution(src_labels)
    tgt_dist = get_domain_distribution(tgt_labels)
    shift_jsd = compute_jsd(src_dist, tgt_dist)

    print(f"  Source distribution: {dict(zip(DOMAIN_NAMES, src_dist.round(3)))}")
    print(f"  Target distribution: {dict(zip(DOMAIN_NAMES, tgt_dist.round(3)))}")
    print(f"  Domain Shift (JSD):  {shift_jsd:.4f}")

    x_labels = ["Source\nBaseline", "Domain\nShift"] + [f"{r:.0%}" for r in SCALE_RATIOS]
    all_results = {}

    for model_name, display_name, use_hybrid in RETRIEVER_CONFIGS:
        print(f"\n🚀 {display_name}")
        accs = []

        # Phase 1: Source Baseline
        pipe = build_pipeline(src_docs, model_name, use_hybrid)
        acc = evaluate_batch(pipe, src_test, BATCH_SIZE)
        accs.append(acc)
        print(f"  Source Baseline: {acc:.2%}")
        del pipe

        # Phase 2: Domain Shift (用 source KB 回答 target 问题)
        pipe = build_pipeline(src_docs, model_name, use_hybrid)
        acc = evaluate_batch(pipe, target_test, BATCH_SIZE)
        accs.append(acc)
        print(f"  Domain Shift:   {acc:.2%}")
        del pipe

        # Phase 3: Scaling Recovery (逐步增加 target 文档)
        for ratio in SCALE_RATIOS:
            n = int(total_target * ratio)
            docs = src_docs + mixed_pool[:n]
            pipe = build_pipeline(docs, model_name, use_hybrid)
            acc = evaluate_batch(pipe, target_test, BATCH_SIZE)
            accs.append(acc)
            print(f"  Scale {ratio:.0%} ({n:,} docs): {acc:.2%}")
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

        all_results[display_name] = accs

    domain_info = {
        "src_dist": src_dist.tolist(),
        "tgt_dist": tgt_dist.tolist(),
        "shift_jsd": shift_jsd,
    }
    return all_results, x_labels, domain_info


def plot_scaling(results, x_labels, domain_info=None):
    """Part A 图: Scaling Recovery + 领域分布对比"""
    setup_style()

    if domain_info:
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.25)
        ax = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(13, 6))

    x = range(len(x_labels))

    # ---- 主图: Scaling Recovery ----
    # 背景分区
    ax.axvspan(-0.5, 0.5, alpha=0.06, color=COLORS['accent1'])
    ax.axvspan(0.5, 1.5, alpha=0.08, color=COLORS['secondary'])
    ax.axvspan(1.5, len(x_labels) - 0.5, alpha=0.04, color=COLORS['primary'])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    for name, accs in results.items():
        style = MODEL_STYLES.get(name, {'color': 'gray', 'marker': 'o'})
        ax.plot(x, accs, color=style['color'], marker=style['marker'],
                linewidth=2.5, markersize=8, label=name,
                markeredgecolor='white', markeredgewidth=1.5)
        for xi, yi in zip(x, accs):
            ax.text(xi, yi + 0.012, f"{yi:.0%}", ha='center', fontsize=8,
                    fontweight='bold', color=style['color'])

    # Drop 箭头
    all_base = [results[m][0] for m in results]
    all_shift = [results[m][1] for m in results]
    max_b, min_s = max(all_base), min(all_shift)
    ax.annotate('', xy=(1, min_s), xytext=(1, max_b),
                arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2.5))
    drop = max_b - min_s
    ax.text(1.3, (min_s + max_b) / 2, f'−{drop:.0%}\ndrop',
            fontsize=12, fontweight='bold', color=COLORS['secondary'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_red'],
                      edgecolor=COLORS['secondary'], alpha=0.9))

    # 区域标签
    ax.text(0, 0.72, 'Baseline', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['accent1'], alpha=0.8)
    ax.text(1, 0.28, 'Domain\nShift', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['secondary'], alpha=0.9)
    mid_scale = (1.5 + len(x_labels) - 0.5) / 2
    ax.text(mid_scale, 0.72, 'Knowledge Scaling (Recovery)',
            ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'], alpha=0.8)

    ax.set_xlabel('Experiment Phase', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('(a) Domain Shift Impact & Scaling Recovery\n'
                 f'(Target KB noise ratio = 1:{NOISE_RATIO})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim([0.20, 0.78])
    ax.legend(loc='lower right', ncol=2, framealpha=0.95, fontsize=9)

    # ---- 右侧小图: 领域分布对比 (分类器结果) ----
    if domain_info:
        src_dist = np.array(domain_info["src_dist"])
        tgt_dist = np.array(domain_info["tgt_dist"])
        jsd_val = domain_info["shift_jsd"]

        bar_width = 0.35
        x_dom = np.arange(NUM_DOMAINS)
        ax_dist.barh(x_dom - bar_width / 2, src_dist, bar_width,
                     color=COLORS['accent1'], edgecolor='black', linewidth=0.8,
                     label='Source (SQuAD)')
        ax_dist.barh(x_dom + bar_width / 2, tgt_dist, bar_width,
                     color=COLORS['secondary'], edgecolor='black', linewidth=0.8,
                     label='Target (HotpotQA)')

        ax_dist.set_yticks(x_dom)
        ax_dist.set_yticklabels([d.capitalize() for d in DOMAIN_NAMES], fontsize=10)
        ax_dist.set_xlabel('Proportion', fontsize=11)
        ax_dist.set_title(f'(b) Query Domain Distribution\n(JSD = {jsd_val:.4f})',
                          fontsize=12, fontweight='bold')
        ax_dist.legend(fontsize=8, loc='lower right')
        ax_dist.set_xlim(0, max(max(src_dist), max(tgt_dist)) * 1.3)

        # 数值标注
        for i, (s, t) in enumerate(zip(src_dist, tgt_dist)):
            ax_dist.text(s + 0.01, i - bar_width / 2, f"{s:.2f}",
                         va='center', fontsize=8, color=COLORS['accent1'], fontweight='bold')
            ax_dist.text(t + 0.01, i + bar_width / 2, f"{t:.2f}",
                         va='center', fontsize=8, color=COLORS['secondary'], fontweight='bold')

        ax_dist.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out = str(OUT_DIR / "fig_scaling_law_recovery.png")
    save_fig(fig, out)
    plt.close()
    return fig


# ==========================================
# Part B: KB-Query Distribution Misalignment
# ==========================================
def run_misalignment_experiment():
    """
    核心实验: 用蒸馏分类器做真实领域分类
    构造不同程度的 KB 偏斜，展示 JSD↑ → Hit Rate↓
    """
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("📊 Part B: KB-Query Distribution Misalignment")
    print("=" * 60)

    # 加载蒸馏分类器
    print("  🔍 Loading distilled detector...")
    detector = OnlineDetector(DETECTOR_PATH)

    # 加载 HotpotQA，用分类器做真实领域划分
    ds = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)

    domain_questions = {d: [] for d in DOMAIN_NAMES}
    domain_contexts = {d: [] for d in DOMAIN_NAMES}

    MAX_PER_DOMAIN = 300
    buffer_questions = []
    buffer_items = []

    print("  📥 Loading and classifying data...")
    for item in ds:
        q = item['question']
        buffer_questions.append(q)
        buffer_items.append(item)

        # 批量分类
        if len(buffer_questions) >= 256:
            labels = classify_queries(buffer_questions, detector)
            for lbl, bi in zip(labels, buffer_items):
                if lbl not in LABEL_MAP:
                    continue
                if len(domain_questions[lbl]) >= MAX_PER_DOMAIN:
                    continue

                titles = bi['context']['title']
                sentences = bi['context']['sentences']
                sf_titles = set(bi['supporting_facts']['title'])

                gold_texts, all_texts = [], []
                for t, s_list in zip(titles, sentences):
                    text = f"{t}: " + "".join(s_list)
                    all_texts.append(text)
                    if t in sf_titles:
                        gold_texts.append(text)

                if not gold_texts:
                    continue

                domain_questions[lbl].append({
                    "question": bi['question'], "answer": bi['answer'],
                    "gold_texts": gold_texts, "all_texts": all_texts,
                })
                domain_contexts[lbl].extend(all_texts)

            buffer_questions.clear()
            buffer_items.clear()

        if all(len(domain_questions[d]) >= MAX_PER_DOMAIN for d in DOMAIN_NAMES):
            break

    for d in DOMAIN_NAMES:
        print(f"  Domain [{d}]: {len(domain_questions[d])} Q, {len(domain_contexts[d])} docs")

    # 实验条件: 从对齐到极端偏斜
    query_dist = np.array([0.25, 0.25, 0.25, 0.25])
    kb_settings = [
        ("Aligned",    [0.25, 0.25, 0.25, 0.25]),
        ("Mild",       [0.35, 0.30, 0.20, 0.15]),
        ("Moderate",   [0.50, 0.25, 0.15, 0.10]),
        ("Severe",     [0.70, 0.15, 0.10, 0.05]),
        ("Extreme",    [0.85, 0.10, 0.03, 0.02]),
    ]

    def keyword_overlap(q: str, text: str) -> float:
        qt = set(normalize_answer(q).split())
        tt = set(normalize_answer(text).split())
        return len(qt & tt) / len(qt) if qt else 0.0

    TOTAL_KB = 2000
    N_Q_PER_DOMAIN = 50
    TOP_K = 10

    results = []
    for label, kb_dist_raw in kb_settings:
        kb_dist = np.array(kb_dist_raw, dtype=float)
        kb_dist /= kb_dist.sum()

        jsd = compute_jsd(query_dist, kb_dist)

        # 按 KB 分布采样文档
        kb_docs = []
        for i, d in enumerate(DOMAIN_NAMES):
            n = int(TOTAL_KB * kb_dist[i])
            pool = domain_contexts[d]
            if len(pool) < n:
                sampled = (pool * (n // len(pool) + 1))[:n]
            else:
                sampled = random.sample(pool, n)
            kb_docs.extend(sampled)
        random.shuffle(kb_docs)

        # 均匀采样 query (保证各领域平衡)
        test_q = []
        for d in DOMAIN_NAMES:
            test_q.extend(domain_questions[d][:N_Q_PER_DOMAIN])
        random.shuffle(test_q)

        # 计算 Hit@10
        hits = 0
        for item in test_q:
            gold_set = set(normalize_answer(g) for g in item["gold_texts"])
            scored = [(keyword_overlap(item["question"], doc), doc) for doc in kb_docs]
            scored.sort(key=lambda x: -x[0])
            top = [normalize_answer(d) for _, d in scored[:TOP_K]]
            for gt in gold_set:
                if any(gt in td or td in gt for td in top):
                    hits += 1
                    break

        hit_rate = hits / len(test_q) if test_q else 0
        results.append({
            "label": label, "jsd": jsd, "hit_rate": hit_rate,
            "kb_dist": kb_dist.tolist(),
        })
        print(f"  {label:10s}: JSD={jsd:.4f}, Hit@10={hit_rate:.2%}")

    return results


def plot_misalignment(results):
    """Part B 图: 清晰展示 JSD↑ → Hit Rate↓ 的因果关系"""
    setup_style()

    fig, (ax_main, ax_heat) = plt.subplots(1, 2, figsize=(15, 6),
                                            gridspec_kw={'width_ratios': [1.4, 1]})

    x = range(len(results))
    labels = [r["label"] for r in results]
    jsd_vals = [r["jsd"] for r in results]
    hit_vals = [r["hit_rate"] for r in results]

    # ========== 左图: Hit Rate 柱状 + JSD 折线 ==========
    # Hit Rate 柱子 (颜色: 绿→红渐变，直观反映好→差)
    cmap = plt.cm.RdYlGn_r  # 绿→红
    norm = plt.Normalize(0, len(results) - 1)
    bar_colors = [cmap(norm(i)) for i in range(len(results))]

    bars = ax_main.bar(x, hit_vals, width=0.5, color=bar_colors,
                       edgecolor='black', linewidth=1.2, alpha=0.85, zorder=3)
    ax_main.set_ylabel("Retrieval Hit Rate @ 10", fontsize=13, color=COLORS['primary'])
    ax_main.set_ylim(0, max(hit_vals) * 1.3)
    ax_main.tick_params(axis='y', labelcolor=COLORS['primary'])

    for xi, yi in zip(x, hit_vals):
        ax_main.text(xi, yi + 0.015, f"{yi:.1%}", ha='center', fontsize=11,
                     fontweight='bold', color=COLORS['primary'])

    # JSD 折线 (右 Y 轴)
    ax_jsd = ax_main.twinx()
    ax_jsd.plot(x, jsd_vals, color=COLORS['secondary'], linewidth=2.5,
                marker='D', markersize=8, zorder=4, label='JSD(KB ∥ Query)')
    ax_jsd.set_ylabel("JS Divergence", fontsize=13, color=COLORS['secondary'])
    ax_jsd.set_ylim(0, max(jsd_vals) * 1.4)
    ax_jsd.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax_jsd.spines['right'].set_color(COLORS['secondary'])

    for xi, yi in zip(x, jsd_vals):
        ax_jsd.text(xi + 0.15, yi + max(jsd_vals) * 0.03, f"{yi:.3f}", fontsize=9,
                    color=COLORS['secondary'], fontweight='bold')

    # 趋势箭头
    ax_main.annotate('', xy=(len(results) - 1, hit_vals[-1] + max(hit_vals) * 0.15),
                     xytext=(0, hit_vals[0] + max(hit_vals) * 0.15),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax_main.text(len(results) / 2 - 0.3, max(hit_vals) * 1.22,
                 "↑ Distribution Misalignment → ↓ Retrieval Accuracy",
                 fontsize=11, color='red', ha='center', fontstyle='italic',
                 fontweight='bold')

    # X 轴: 同时显示 label 和分布向量
    x_tick_labels = []
    for r in results:
        dist_str = "[" + ",".join(f".{int(d*100):02d}" for d in r["kb_dist"]) + "]"
        x_tick_labels.append(f"{r['label']}\n{dist_str}")

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(x_tick_labels, fontsize=8)
    ax_main.set_xlabel("KB Distribution Setting (classified by distilled detector)", fontsize=11)
    ax_main.set_title("(a) KB-Query Misalignment → Retrieval Accuracy Drop",
                      fontsize=13, fontweight='bold')
    ax_jsd.legend(loc='upper left', fontsize=10)
    ax_main.grid(axis='y', alpha=0.3)

    # ========== 右图: 分布对比热力图 ==========
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
                         fontsize=10, fontweight='bold', color=color)

    domain_short = [d[:3].upper() for d in DOMAIN_NAMES]  # ENT, STE, HUM, LIF
    ax_heat.set_xticks(range(NUM_DOMAINS))
    ax_heat.set_xticklabels(domain_short, fontsize=10)
    ax_heat.set_yticks(range(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_xlabel("Domain (classified by detector)", fontsize=11)
    ax_heat.set_title("(b) Distribution Heatmap\n(Query vs KB)", fontsize=13, fontweight='bold')

    # query 行加粗分隔线
    ax_heat.axhline(y=0.5, color='white', linewidth=3)

    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Probability")

    plt.tight_layout()
    out = str(OUT_DIR / "fig_kb_query_misalignment.png")
    save_fig(fig, out)
    plt.close()
    return fig


# ==========================================
# Combined Figure
# ==========================================
def plot_combined(scaling_results, scaling_labels, misalignment_results, domain_info=None):
    """合并 Part A + Part B"""
    setup_style()

    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1],
                           width_ratios=[3, 1], hspace=0.35, wspace=0.25)

    # ========== 上左: Scaling Recovery ==========
    ax_top = fig.add_subplot(gs[0, 0])
    x = range(len(scaling_labels))

    ax_top.axvspan(-0.5, 0.5, alpha=0.06, color=COLORS['accent1'])
    ax_top.axvspan(0.5, 1.5, alpha=0.08, color=COLORS['secondary'])
    ax_top.axvspan(1.5, len(scaling_labels) - 0.5, alpha=0.04, color=COLORS['primary'])
    ax_top.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_top.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    for name, accs in scaling_results.items():
        style = MODEL_STYLES.get(name, {'color': 'gray', 'marker': 'o'})
        ax_top.plot(x, accs, color=style['color'], marker=style['marker'],
                    linewidth=2.5, markersize=8, label=name,
                    markeredgecolor='white', markeredgewidth=1.5)
        for xi, yi in zip(x, accs):
            ax_top.text(xi, yi + 0.012, f"{yi:.0%}", ha='center', fontsize=8,
                        fontweight='bold', color=style['color'])

    all_base = [scaling_results[m][0] for m in scaling_results]
    all_shift = [scaling_results[m][1] for m in scaling_results]
    max_b, min_s = max(all_base), min(all_shift)
    ax_top.annotate('', xy=(1, min_s), xytext=(1, max_b),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2.5))
    drop = max_b - min_s
    ax_top.text(1.3, (min_s + max_b) / 2, f'−{drop:.0%}\ndrop',
                fontsize=11, fontweight='bold', color=COLORS['secondary'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_red'],
                          edgecolor=COLORS['secondary'], alpha=0.9))

    ax_top.text(0, 0.73, 'Baseline', ha='center', fontsize=10,
                fontweight='bold', color=COLORS['accent1'], alpha=0.8)
    ax_top.text(1, 0.27, 'Domain\nShift', ha='center', fontsize=10,
                fontweight='bold', color=COLORS['secondary'], alpha=0.9)
    mid = (1.5 + len(scaling_labels) - 0.5) / 2
    ax_top.text(mid, 0.73, 'Knowledge Scaling (Recovery)',
                ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'], alpha=0.8)

    ax_top.set_xlabel('Experiment Phase', fontsize=12)
    ax_top.set_ylabel('Accuracy', fontsize=12)
    ax_top.set_title('(a) Domain Shift Impact & Scaling Recovery',
                     fontsize=14, fontweight='bold')
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(scaling_labels)
    ax_top.set_ylim([0.20, 0.78])
    ax_top.legend(loc='lower right', ncol=2, framealpha=0.95, fontsize=8)

    # ========== 上右: 领域分布对比 ==========
    ax_dist = fig.add_subplot(gs[0, 1])
    if domain_info:
        src_dist = np.array(domain_info["src_dist"])
        tgt_dist = np.array(domain_info["tgt_dist"])
        jsd_val = domain_info["shift_jsd"]

        bar_width = 0.35
        x_dom = np.arange(NUM_DOMAINS)
        ax_dist.barh(x_dom - bar_width / 2, src_dist, bar_width,
                     color=COLORS['accent1'], edgecolor='black', linewidth=0.8,
                     label='Source')
        ax_dist.barh(x_dom + bar_width / 2, tgt_dist, bar_width,
                     color=COLORS['secondary'], edgecolor='black', linewidth=0.8,
                     label='Target')

        ax_dist.set_yticks(x_dom)
        ax_dist.set_yticklabels([d.capitalize() for d in DOMAIN_NAMES], fontsize=9)
        ax_dist.set_xlabel('Proportion', fontsize=10)
        ax_dist.set_title(f'(b) Query Domain Shift\n(JSD = {jsd_val:.4f})',
                          fontsize=12, fontweight='bold')
        ax_dist.legend(fontsize=8, loc='lower right')
        ax_dist.set_xlim(0, max(max(src_dist), max(tgt_dist)) * 1.4)

        for i, (s, t) in enumerate(zip(src_dist, tgt_dist)):
            ax_dist.text(s + 0.01, i - bar_width / 2, f"{s:.2f}",
                         va='center', fontsize=8, color=COLORS['accent1'], fontweight='bold')
            ax_dist.text(t + 0.01, i + bar_width / 2, f"{t:.2f}",
                         va='center', fontsize=8, color=COLORS['secondary'], fontweight='bold')
        ax_dist.grid(axis='x', alpha=0.3)
    else:
        ax_dist.text(0.5, 0.5, 'No domain info', transform=ax_dist.transAxes,
                     ha='center', va='center', fontsize=12, color='gray')

    # ========== 下左: Misalignment ==========
    ax_bot = fig.add_subplot(gs[1, 0])

    x2 = range(len(misalignment_results))
    labels2 = [r["label"] for r in misalignment_results]
    jsd_vals = [r["jsd"] for r in misalignment_results]
    hit_vals = [r["hit_rate"] for r in misalignment_results]

    cmap = plt.cm.RdYlGn_r
    norm_c = plt.Normalize(0, len(misalignment_results) - 1)
    bar_colors = [cmap(norm_c(i)) for i in range(len(misalignment_results))]

    ax_bot.bar(x2, hit_vals, width=0.5, color=bar_colors,
               edgecolor='black', linewidth=1.2, alpha=0.85, zorder=3)
    ax_bot.set_ylabel("Retrieval Hit Rate @ 10", fontsize=13, color=COLORS['primary'])
    ax_bot.set_ylim(0, max(hit_vals) * 1.3)
    ax_bot.tick_params(axis='y', labelcolor=COLORS['primary'])

    for xi, yi in zip(x2, hit_vals):
        ax_bot.text(xi, yi + 0.015, f"{yi:.1%}", ha='center', fontsize=11,
                    fontweight='bold', color=COLORS['primary'])

    ax_jsd = ax_bot.twinx()
    ax_jsd.plot(x2, jsd_vals, color=COLORS['secondary'], linewidth=2.5,
                marker='D', markersize=8, zorder=4, label='JSD(KB ∥ Query)')
    ax_jsd.set_ylabel("JS Divergence", fontsize=13, color=COLORS['secondary'])
    ax_jsd.set_ylim(0, max(jsd_vals) * 1.4)
    ax_jsd.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax_jsd.spines['right'].set_color(COLORS['secondary'])

    for xi, yi in zip(x2, jsd_vals):
        ax_jsd.text(xi + 0.15, yi + max(jsd_vals) * 0.03, f"{yi:.3f}", fontsize=9,
                    color=COLORS['secondary'], fontweight='bold')

    ax_bot.annotate('', xy=(len(misalignment_results) - 1, hit_vals[-1] + max(hit_vals) * 0.15),
                    xytext=(0, hit_vals[0] + max(hit_vals) * 0.15),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax_bot.text(len(misalignment_results) / 2 - 0.3, max(hit_vals) * 1.22,
                "↑ Misalignment → ↓ Accuracy",
                fontsize=11, color='red', ha='center', fontstyle='italic', fontweight='bold')

    x_tick_labels2 = []
    for r in misalignment_results:
        dist_str = "[" + ",".join(f".{int(d * 100):02d}" for d in r["kb_dist"]) + "]"
        x_tick_labels2.append(f"{r['label']}\n{dist_str}")
    ax_bot.set_xticks(x2)
    ax_bot.set_xticklabels(x_tick_labels2, fontsize=8)
    ax_bot.set_xlabel("KB Distribution Setting", fontsize=11)
    ax_bot.set_title("(c) KB-Query Misalignment → Retrieval Accuracy Drop",
                     fontsize=14, fontweight='bold')
    ax_jsd.legend(loc='upper left', fontsize=10)
    ax_bot.grid(axis='y', alpha=0.3)

    # ========== 下右: 热力图 ==========
    ax_heat = fig.add_subplot(gs[1, 1])
    query_dist = [0.25, 0.25, 0.25, 0.25]
    row_labels_h = ["Query"]
    for r in misalignment_results:
        row_labels_h.append(r['label'])
    matrix = np.array([query_dist] + [r["kb_dist"] for r in misalignment_results])

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
    ax_heat.set_yticks(range(len(row_labels_h)))
    ax_heat.set_yticklabels(row_labels_h, fontsize=8)
    ax_heat.set_title("(d) Distribution Heatmap", fontsize=12, fontweight='bold')
    ax_heat.axhline(y=0.5, color='white', linewidth=3)
    fig.colorbar(im, ax=ax_heat, shrink=0.7, label="Prob")

    out = str(OUT_DIR / "fig_motivation2_combined.png")
    save_fig(fig, out)
    plt.close()


# ==========================================
# 主入口
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all", choices=["a", "b", "all", "plot_only"])
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    results_path_a = OUT_DIR / "results_scaling.json"
    results_path_b = OUT_DIR / "results_misalignment.json"

    # Part A
    if args.part in ("a", "all"):
        scaling_results, x_labels, domain_info = run_scaling_experiment()
        save_results({"results": scaling_results, "labels": x_labels, "domain_info": domain_info},
                     str(results_path_a))
        plot_scaling(scaling_results, x_labels, domain_info)

    # Part B
    if args.part in ("b", "all"):
        misalignment_results = run_misalignment_experiment()
        save_results(misalignment_results, str(results_path_b))
        plot_misalignment(misalignment_results)

    # Combined
    if args.part in ("all", "plot_only"):
        try:
            if not results_path_a.exists() or not results_path_b.exists():
                print("⚠️ 需要先运行 --part all 生成结果")
            else:
                data_a = json.loads(results_path_a.read_text())
                data_b = json.loads(results_path_b.read_text())
                plot_combined(data_a["results"], data_a["labels"], data_b,
                              domain_info=data_a.get("domain_info"))
        except Exception as e:
            print(f"⚠️ Combined plot failed: {e}")

    print("\n🎉 Motivation 2 实验完成！")