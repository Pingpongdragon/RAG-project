"""
Motivation 1A: TopiOCQA 对话中的 Topic Shift 统计
证明: 多轮对话中主题偏移是普遍现象，静态检索策略不可靠

输出: motivation_1/fig_topic_shift_distribution.png
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, save_fig

OUT_DIR = Path(__file__).resolve().parent


def load_topiocqa():
    """加载 TopiOCQA 数据集 (兼容新版 datasets 库)"""
    from datasets import load_dataset

    try:
        # 方法 1: 直接加载 (旧版 datasets)
        ds = load_dataset("McGill-NLP/TopiOCQA", split="train")
    except RuntimeError:
        try:
            # 方法 2: trust_remote_code (中间版本 datasets)
            ds = load_dataset("McGill-NLP/TopiOCQA", split="train", trust_remote_code=True)
        except Exception:
            # 方法 3: 直接加载原始 JSON (新版 datasets 3.0+)
            print("  📥 Fetching raw JSON via generic loader...")
            data_url = "https://huggingface.co/datasets/McGill-NLP/TopiOCQA/resolve/main/TopiOCQA_train.json"
            ds = load_dataset("json", data_files=data_url, split="train")
    return ds


def compute_shift_stats(ds):
    """按对话分组并统计主题切换次数"""
    dialogs = defaultdict(list)
    for ex in ds:
        did = ex["Conversation_no"]
        tid = ex.get("Turn_no", len(dialogs[did]))
        topic = ex.get("Topic", "").strip()
        dialogs[did].append((tid, topic))

    shift_counts = []
    for did, turns in dialogs.items():
        turns_sorted = sorted(turns, key=lambda x: x[0])
        last_topic, n_shift = None, 0
        for _, topic in turns_sorted:
            if not topic:
                continue
            if last_topic is not None and topic != last_topic:
                n_shift += 1
            last_topic = topic
        shift_counts.append(n_shift)

    return pd.Series(shift_counts)


def plot_topic_shift(s):
    """绘制主题切换分布图"""
    setup_style()

    total = len(s)
    with_shift = int((s > 0).sum())
    avg = s.mean()

    vc = s.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))

    # ========== 柱状图 (渐变色) ==========
    norm = plt.Normalize(0, vc.index.max())
    cmap = plt.cm.YlOrRd
    bar_colors = [cmap(norm(xi)) for xi in vc.index]

    bars = ax.bar(vc.index, vc.values, color=bar_colors,
                  edgecolor='black', linewidth=0.8, zorder=3)

    # 柱状图数值标注
    for xi, yi in zip(vc.index, vc.values):
        ax.text(xi, yi + max(vc.values) * 0.01, str(int(yi)),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ========== CDF 折线 (降低 zorder，避免遮挡) ==========
    ax2 = ax.twinx()
    cumsum = np.cumsum([vc.get(i, 0) for i in range(vc.index.max() + 1)])
    cdf = cumsum / total * 100
    ax2.plot(range(vc.index.max() + 1), cdf, color=COLORS['primary'],
             linewidth=2.5, marker='o', markersize=4, zorder=2)  # zorder 低于统计框
    ax2.set_ylabel('Cumulative %', color=COLORS['primary'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax2.set_ylim(0, 115)  # 留出更多顶部空间
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLORS['primary'])
    ax2.spines['top'].set_visible(False)

    # ========== 统计框 (移到左上角，避免被蓝线遮挡) ==========
    stats_text = (
        f"Total Dialogs: {total}\n"
        f"With Shifts: {with_shift} ({with_shift / total * 100:.1f}%)\n"
        f"Avg Shifts/Dialog: {avg:.2f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            va='top', ha='right', zorder=10,  # zorder 最高，浮在蓝线之上
            bbox=dict(boxstyle='round', facecolor='wheat',
                      alpha=0.95,  # 提高不透明度，完全遮住后面的蓝线
                      edgecolor=COLORS['accent2']))

    # ========== 增加 Y 轴上限 (给柱状图顶部数字和统计框留空间) ==========
    ax.set_ylim(0, max(vc.values) * 1.2)

    ax.set_xlabel('Number of Topic Shifts per Dialog', fontsize=12)
    ax.set_ylabel('Number of Dialogs', fontsize=12)
    ax.set_title('Topic Shift Distribution in Conversations (TopiOCQA, train)',
                 fontsize=14, fontweight='bold')

    out_path = OUT_DIR / "fig_topic_shift_distribution.png"
    save_fig(fig, str(out_path))
    plt.close()

    return out_path


def run():
    setup_style()

    print("📊 Motivation 1A: TopiOCQA Topic Shift")
    print("=" * 60)

    # 1. 加载数据
    ds = load_topiocqa()
    print(f"  加载完成: {len(ds)} 条")

    # 2. 计算统计
    s = compute_shift_stats(ds)
    total = len(s)
    with_shift = int((s > 0).sum())
    avg = s.mean()

    print(f"  总对话: {total}")
    print(f"  有 shift: {with_shift} ({with_shift / total * 100:.1f}%)")
    print(f"  平均 shift/对话: {avg:.2f}")

    # 3. 绘图
    out_path = plot_topic_shift(s)
    print(f"  ✅ 已保存: {out_path}")


if __name__ == "__main__":
    run()