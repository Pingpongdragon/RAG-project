"""
Motivation 2A: Domain Shift → Accuracy Drop → Scaling Recovery

实验设计:
  - Source: SQuAD (阅读理解, 短答案)
  - Target: HotpotQA (多跳推理, 1:600 噪声比)
  - 用蒸馏分类器对 query 做领域标注，展示 shift 前后分布差异
  - 多种 retriever (Dense/Hybrid) 对比
  - 逐步增加 target 文档，观察恢复曲线

输出: motivation_2/fig_scaling_law_recovery.png
"""
import sys
import gc
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, MODEL_STYLES, save_fig
from utils import build_pipeline, evaluate_batch, save_results

from common import (
    get_detector, classify_queries,
    get_domain_distribution, compute_jsd,
    DOMAIN_NAMES, NUM_DOMAINS, OUT_DIR,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ✅ 新增：避免 tokenizer 多线程死锁
os.environ["OMP_NUM_THREADS"] = "1"              # ✅ 新增：限制 OpenMP 线程数
os.environ["MKL_NUM_THREADS"] = "1"  

# ==========================================
# 实验配置
# ==========================================
TEST_SAMPLE_SIZE = 100
BATCH_SIZE = 32
NOISE_RATIO = 600               # 1:600 噪声比
SCALE_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
RANDOM_SEED = 42

RETRIEVER_CONFIGS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6 (Dense)", False),
    ("BAAI/bge-small-en-v1.5",                 "BGE-Small (Dense)", False),
    ("BAAI/bge-large-en-v1.5",                 "BGE-Large (Dense)", False),
    ("BAAI/bge-small-en-v1.5",                 "BGE-Small (Hybrid)", True),
]


# ==========================================
# 随机种子初始化
# ==========================================
def set_random_seed(seed: int = RANDOM_SEED):
    """统一设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"🎲 Random seed set to {seed}")


# ==========================================
# 数据加载 - 阶段 1: SQuAD (Source Domain)
# ==========================================
def load_squad_source(max_samples: int = TEST_SAMPLE_SIZE) -> Tuple[List[Any], List[Dict]]:
    """
    加载 SQuAD 数据集作为 source domain
    
    Args:
        max_samples: 最大样本数
    
    Returns:
        src_docs: Document 对象列表
        src_test: 测试样本列表 [{"q": question, "a": answers}, ...]
    """
    from datasets import load_dataset
    from langchain_core.documents import Document

    print("📚 Loading SQuAD (Source Domain)...")
    
    ds_squad = load_dataset("squad", split="validation")
    src_test, src_docs = [], []
    
    for item in ds_squad:
        if len(src_test) >= max_samples:
            break
        
        answers = item['answers']['text']
        # 过滤：只保留短答案（≤5词）
        if not answers or len(answers[0].split()) > 5:
            continue
        
        src_test.append({"q": item['question'], "a": answers})
        src_docs.append(Document(
            page_content=item['context'],
            metadata={"doc_id": item['id'], "source": "squad"},
        ))
    
    print(f"  ✅ Loaded {len(src_docs)} docs, {len(src_test)} questions")
    return src_docs, src_test


# ==========================================
# 数据加载 - 阶段 2: HotpotQA (Target Domain)
# ==========================================
def load_hotpotqa_target(
    max_samples: int = TEST_SAMPLE_SIZE,
    noise_ratio: int = NOISE_RATIO
) -> Tuple[List[Dict], List[Any], List[Any]]:
    """
    加载 HotpotQA 数据集作为 target domain
    
    Args:
        max_samples: 测试集样本数
        noise_ratio: 噪声文档与金标文档的比例
    
    Returns:
        target_test: 测试样本列表
        gold_docs: 金标文档列表
        noise_docs: 噪声文档列表
    """
    from datasets import load_dataset
    from langchain_core.documents import Document
    import itertools

    print("📚 Loading HotpotQA (Target Domain)...")
    
    # 使用流式加载避免缓存兼容性问题
    ds_stream = load_dataset(
        "hotpot_qa", 
        "distractor", 
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # ===== 第一遍：取测试样本，统计 gold docs 数量 =====
    print(f"  📥 Phase 1: Extracting {max_samples} test samples...")
    buffer = list(itertools.islice(ds_stream, max_samples))
    
    target_test = []
    gold_docs = []
    
    for item in buffer:
        target_test.append({
            "q": item['question'], 
            "a": item['answer']
        })
        
        titles = item['context']['title']
        sentences = item['context']['sentences']
        gold_titles = set(item['supporting_facts']['title'])
        
        for title, sent_list in zip(titles, sentences):
            if title in gold_titles:
                text = f"{title}: " + " ".join(sent_list)
                gold_docs.append(Document(
                    page_content=text,
                    metadata={"doc_id": f"hp_gold_{title}", "source": "hotpot_gold"},
                ))
    
    # ===== 第二遍：继续从流中读取，构造足量噪声文档 =====
    target_noise_count = len(gold_docs) * noise_ratio
    noise_docs = []
    
    # 每条 HotpotQA 大约能提供 10 个段落，预估需要读取的条目数
    estimated_items_needed = (target_noise_count // 8) + 1000  # 留余量
    print(f"  🔧 Phase 2: Mining {target_noise_count} noise docs "
          f"(streaming ~{estimated_items_needed} more items)...")
    
    batch_count = 0
    for item in ds_stream:  # 从上次断点继续读取（流式迭代器自动续接）
        if len(noise_docs) >= target_noise_count:
            break
        
        titles = item['context']['title']
        sentences = item['context']['sentences']
        
        for title, sent_list in zip(titles, sentences):
            if len(noise_docs) >= target_noise_count:
                break
            
            text = f"{title}: " + " ".join(sent_list)
            noise_docs.append(Document(
                page_content=text,
                metadata={"doc_id": f"hp_noise_{len(noise_docs)}", "source": "hotpot_noise"},
            ))
        
        batch_count += 1
        if batch_count % 5000 == 0:
            print(f"    ... processed {batch_count} items, noise docs: {len(noise_docs)}/{target_noise_count}")
    
    print(f"  ✅ Loaded {len(target_test)} questions")
    print(f"  ✅ Gold docs: {len(gold_docs)}, Noise docs: {len(noise_docs)}")
    actual_ratio = len(noise_docs) // max(len(gold_docs), 1)
    print(f"  ✅ Noise ratio: 1:{actual_ratio}")
    
    if actual_ratio < noise_ratio:
        print(f"  ⚠️  Warning: target ratio 1:{noise_ratio}, actual 1:{actual_ratio} "
              f"(dataset may not have enough data)")
    
    return target_test, gold_docs, noise_docs


# ==========================================
# 数据整合
# ==========================================
def prepare_experiment_data() -> Tuple[List[Any], List[Dict], List[Any], List[Dict]]:
    """
    整合所有实验数据
    
    Returns:
        src_docs: Source domain 文档
        src_test: Source domain 测试集
        mixed_pool: Target domain 混合文档池（gold + noise）
        target_test: Target domain 测试集
    """
    set_random_seed()
    
    # 加载 source domain
    src_docs, src_test = load_squad_source()
    
    # 加载 target domain
    target_test, gold_docs, noise_docs = load_hotpotqa_target()
    
    # 混合金标 + 噪声文档
    mixed_pool = gold_docs + noise_docs
    random.shuffle(mixed_pool)
    
    print(f"\n📦 Total KB size: {len(mixed_pool)} docs")
    print(f"   Source: {len(src_docs)} docs")
    print(f"   Target: {len(gold_docs)} gold + {len(noise_docs)} noise")
    
    return src_docs, src_test, mixed_pool, target_test


# ==========================================
# 实验阶段函数
# ==========================================
def evaluate_source_baseline(
    src_docs: List[Any],
    src_test: List[Dict],
    model_name: str,
    use_hybrid: bool
) -> float:
    """Phase 1: 评估 source domain 基线性能"""
    pipe = build_pipeline(src_docs, model_name, use_hybrid)
    acc = evaluate_batch(pipe, src_test, BATCH_SIZE)
    del pipe
    gc.collect()
    return acc


def evaluate_domain_shift(
    src_docs: List[Any],
    target_test: List[Dict],
    model_name: str,
    use_hybrid: bool
) -> float:
    """Phase 2: 评估 domain shift 后的性能下降"""
    pipe = build_pipeline(src_docs, model_name, use_hybrid)
    acc = evaluate_batch(pipe, target_test, BATCH_SIZE)
    del pipe
    gc.collect()
    return acc


def evaluate_scaling_recovery(
    src_docs: List[Any],
    mixed_pool: List[Any],
    target_test: List[Dict],
    model_name: str,
    use_hybrid: bool,
    scale_ratios: List[float]
) -> List[float]:
    """Phase 3: 评估通过扩充 target 知识的恢复曲线"""
    import torch
    
    total_target = len(mixed_pool)
    accs = []
    
    for ratio in scale_ratios:
        n = int(total_target * ratio)
        docs = src_docs + mixed_pool[:n]
        
        pipe = build_pipeline(docs, model_name, use_hybrid)
        acc = evaluate_batch(pipe, target_test, BATCH_SIZE)
        accs.append(acc)
        
        print(f"    Scale {ratio:.0%} ({n:,} docs): {acc:.2%}")
        
        del pipe
        gc.collect()
        # torch.cuda.empty_cache()
    
    return accs


# ==========================================
# 核心实验流程
# ==========================================
def run():
    """Part A: Domain Shift + Scaling Recovery"""
    print("=" * 60)
    print("📊 Motivation 2A: Domain Shift + Scaling Recovery")
    print("=" * 60)

    # ===== 数据准备 =====
    src_docs, src_test, mixed_pool, target_test = prepare_experiment_data()

    # ===== 领域分布分析 =====
    print("\n🔍 Analyzing domain shift with distilled detector...")
    detector = get_detector()
    
    src_queries = [item["q"] for item in src_test]
    tgt_queries = [item["q"] for item in target_test]
    
    src_labels = classify_queries(src_queries, detector)
    tgt_labels = classify_queries(tgt_queries, detector)

    src_dist = get_domain_distribution(src_labels)
    tgt_dist = get_domain_distribution(tgt_labels)
    shift_jsd = compute_jsd(src_dist, tgt_dist)

    print(f"  Source dist: {dict(zip(DOMAIN_NAMES, src_dist.round(3)))}")
    print(f"  Target dist: {dict(zip(DOMAIN_NAMES, tgt_dist.round(3)))}")
    print(f"  JSD:         {shift_jsd:.4f}")

    # ===== 多模型实验 =====
    x_labels = ["Source\nBaseline", "Domain\nShift"] + [f"{r:.0%}" for r in SCALE_RATIOS]
    all_results = {}

    for model_name, display_name, use_hybrid in RETRIEVER_CONFIGS:
        print(f"\n🚀 Evaluating: {display_name}")
        accs = []

        # Phase 1: Source Baseline
        acc_base = evaluate_source_baseline(src_docs, src_test, model_name, use_hybrid)
        accs.append(acc_base)
        print(f"  ✅ Source Baseline: {acc_base:.2%}")

        # Phase 2: Domain Shift
        acc_shift = evaluate_domain_shift(src_docs, target_test, model_name, use_hybrid)
        accs.append(acc_shift)
        print(f"  ⚠️  Domain Shift:    {acc_shift:.2%} (drop: {acc_base - acc_shift:.2%})")

        # Phase 3: Scaling Recovery
        print(f"  🔄 Scaling Recovery:")
        accs_recovery = evaluate_scaling_recovery(
            src_docs, mixed_pool, target_test, 
            model_name, use_hybrid, SCALE_RATIOS
        )
        accs.extend(accs_recovery)

        all_results[display_name] = accs

    # ===== 保存结果 =====
    domain_info = {
        "src_dist": src_dist.tolist(),
        "tgt_dist": tgt_dist.tolist(),
        "shift_jsd": shift_jsd,
    }

    out_path = OUT_DIR / "results_scaling.json"
    save_results({
        "results": all_results, 
        "labels": x_labels, 
        "domain_info": domain_info
    }, str(out_path))
    print(f"\n💾 Results saved: {out_path}")

    # ===== 绘图 =====
    plot(all_results, x_labels, domain_info)

    return all_results, x_labels, domain_info


# ==========================================
# 绘图
# ==========================================
def plot(results=None, x_labels=None, domain_info=None):
    """绘制 Scaling Recovery + 领域分布对比图"""

    # 从文件加载
    if results is None:
        data = json.loads((OUT_DIR / "results_scaling.json").read_text())
        results = data["results"]
        x_labels = data["labels"]
        domain_info = data.get("domain_info")

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

    # ---- 右侧小图: 领域分布对比 ----
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