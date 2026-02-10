# filepath: \home\ljy\RAG_FT_project\RAG_project\core\rag_scale_law\cal.py
import argparse
from pathlib import Path
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


def download_topiocqa(split: str = "train"):
    """下载 TopiOCQA plain_text 指定 split"""
    print("📥 加载 TopiOCQA 数据集 (plain_text)...")
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        ds = load_dataset("McGill-NLP/TopiOCQA", "plain_text", split=split)
        print(f"✅ 镜像加载完成：{split}，{len(ds)} 条")
        return ds
    except Exception as e:
        print(f"⚠️ 镜像失败，改用官方源：{e}")
        os.environ.pop("HF_ENDPOINT", None)
        ds = load_dataset("McGill-NLP/TopiOCQA", "plain_text", split=split)
        print(f"✅ 官方源加载完成：{split}，{len(ds)} 条")
        return ds


def detect_keys(ds):
    """自动检测 topic / dialog / turn 的字段名"""
    if len(ds) == 0:
        raise RuntimeError("数据集为空。")
    
    cols = list(ds.features.keys())
    print(f"[DEBUG] 可用字段: {cols}")
    
    # TopiOCQA 的字段名
    topic_key = "Topic"
    dialog_key = "Conversation_no"
    turn_key = "Turn_no"
    
    if topic_key not in cols or dialog_key not in cols:
        raise RuntimeError(f"找不到必需字段，可用列：{cols}")
    
    print(f"[INFO] 使用字段: topic={topic_key}, dialog={dialog_key}, turn={turn_key}")
    return topic_key, dialog_key, turn_key


def compute_topic_shifts(ds):
    """
    统计每个对话内的 topic shift 次数
    返回: 
        - shift_counts: 每个对话的 shift 次数 Series
        - total_shifts: 总 shift 次数
        - total_dialogs: 总对话数
        - dialogs_with_shift: 有 shift 的对话数
    """
    topic_key, dialog_key, turn_key = detect_keys(ds)

    # 按对话分组
    dialogs = defaultdict(list)
    for ex in ds:
        did = ex[dialog_key]
        tid = ex.get(turn_key, len(dialogs[did]))
        topic = ex.get(topic_key, "")
        if isinstance(topic, str):
            topic = topic.strip()
        dialogs[did].append((tid, topic))

    # 统计每个对话的 shift 次数
    shift_counts = []
    for did, turns in dialogs.items():
        # 按 turn_id 排序
        turns_sorted = sorted(turns, key=lambda x: x[0])
        
        last_topic = None
        shift_count = 0
        
        for _, topic in turns_sorted:
            if not topic:  # 跳过空 topic
                continue
            
            if last_topic is not None and topic != last_topic:
                shift_count += 1
            
            last_topic = topic
        
        shift_counts.append(shift_count)

    shift_series = pd.Series(shift_counts, name="num_shifts")
    
    stats = {
        "shift_counts": shift_series,
        "total_shifts": shift_series.sum(),
        "total_dialogs": len(shift_series),
        "dialogs_with_shift": (shift_series > 0).sum(),
        "dialogs_without_shift": (shift_series == 0).sum(),
        "avg_shifts_per_dialog": shift_series.mean(),
        "max_shifts": shift_series.max()
    }
    
    return stats


def plot_shift_distribution(stats: dict, out_png: Path, split: str):
    """画 topic shift 次数的分布直方图"""
    shift_counts = stats["shift_counts"]
    vc = shift_counts.value_counts().sort_index()
    
    sns.set(style="whitegrid")
    sns.set_context("talk", font_scale=0.9)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 画柱状图
    bars = ax.bar(vc.index, vc.values, 
                  color=sns.color_palette("deep")[1], 
                  edgecolor="black", linewidth=1.2)
    
    # 在柱子上标数字
    for i, (idx, val) in enumerate(zip(vc.index, vc.values)):
        ax.text(idx, val + max(vc.values)*0.01, str(int(val)), 
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Number of Topic Shifts per Dialog", fontsize=12)
    ax.set_ylabel("Number of Dialogs", fontsize=12)
    ax.set_title(f"Topic Shift Distribution in Conversations (TopiOCQA, {split})", 
                 fontsize=14, fontweight="bold")
    
    # 添加统计信息文本框
    textstr = '\n'.join([
        f"Total Dialogs: {stats['total_dialogs']}",
        f"Dialogs with Shifts: {stats['dialogs_with_shift']} ({stats['dialogs_with_shift']/stats['total_dialogs']*100:.1f}%)",
        f"Total Shifts: {stats['total_shifts']}",
        f"Avg Shifts/Dialog: {stats['avg_shifts_per_dialog']:.2f}"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=props)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Topic shift 分布图已保存: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", 
                        choices=["train", "validation", "test"])
    parser.add_argument("--outdir", type=str, default=str(Path("figures")))
    parser.add_argument("--save_csv", action="store_true", 
                        help="保存统计数据为 CSV")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据并统计
    ds = download_topiocqa(args.split)
    stats = compute_topic_shifts(ds)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("📊 Topic Shift 统计结果:")
    print("="*60)
    print(f"  总对话数: {stats['total_dialogs']}")
    print(f"  有 shift 的对话数: {stats['dialogs_with_shift']} ({stats['dialogs_with_shift']/stats['total_dialogs']*100:.1f}%)")
    print(f"  无 shift 的对话数: {stats['dialogs_without_shift']} ({stats['dialogs_without_shift']/stats['total_dialogs']*100:.1f}%)")
    print(f"  总 shift 次数: {stats['total_shifts']}")
    print(f"  平均每对话 shift 次数: {stats['avg_shifts_per_dialog']:.2f}")
    print(f"  最多 shift 次数: {stats['max_shifts']}")
    print("="*60)
    
    print("\n📈 Shift 次数分布 (Top 10):")
    shift_dist = stats["shift_counts"].value_counts().sort_index()
    for shifts, count in shift_dist.head(10).items():
        print(f"  {shifts} shifts: {count} dialogs ({count/stats['total_dialogs']*100:.1f}%)")
    print("="*60 + "\n")

    # 画图
    out_png = out_dir / f"topiocqa_topic_shift_distribution_{args.split}.png"
    plot_shift_distribution(stats, out_png, args.split)
    
    # 保存 CSV
    if args.save_csv:
        out_csv = out_dir / f"topiocqa_topic_shift_stats_{args.split}.csv"
        stats["shift_counts"].to_csv(out_csv, header=["num_shifts"], index=False)
        print(f"✅ Shift 统计已保存: {out_csv}")


if __name__ == "__main__":
    main()
