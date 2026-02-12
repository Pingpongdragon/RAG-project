"""
Motivation 3: Naive KB Scaling Cost Benchmark
测量 KB 规模增长对检索延迟和内存的线性影响

输出: fig_kb_size_vs_latency_memory.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_config import setup_style, COLORS, save_fig

OUT_DIR = Path(__file__).resolve().parent


def benchmark_retrieval_overhead():
    """基准测试: KB规模 vs 检索开销"""
    setup_style()

    print("📊 Motivation 3: Naive Scaling Cost Benchmark")
    print("=" * 70)

    EMBED_DIM = 384
    KB_SIZES = [1_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    NUM_QUERIES = 200
    TOP_K = 10

    # 生成查询向量
    query_vecs = np.random.randn(NUM_QUERIES, EMBED_DIM).astype(np.float32)
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

    results = []
    for kb_size in KB_SIZES:
        print(f"  Testing KB size: {kb_size:,} documents...")

        # 生成 KB 向量
        kb_vecs = np.random.randn(kb_size, EMBED_DIM).astype(np.float32)
        kb_vecs /= np.linalg.norm(kb_vecs, axis=1, keepdims=True)

        # 内存占用 (MB)
        mem_mb = kb_vecs.nbytes / (1024 * 1024)

        # Warmup (消除缓存冷启动影响)
        for i in range(min(20, NUM_QUERIES)):
            sims = kb_vecs @ query_vecs[i]
            _ = np.argpartition(sims, -TOP_K)[-TOP_K:]

        # 延迟测量
        latencies_ms = []
        for i in range(NUM_QUERIES):
            t_start = time.perf_counter()
            sims = kb_vecs @ query_vecs[i]
            _ = np.argpartition(sims, -TOP_K)[-TOP_K:]
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000)

        avg_latency = np.mean(latencies_ms)
        p99_latency = np.percentile(latencies_ms, 99)

        results.append({
            "kb_size": kb_size,
            "avg_latency_ms": avg_latency,
            "p99_latency_ms": p99_latency,
            "memory_mb": mem_mb,
        })
        print(f"    └─ Avg: {avg_latency:.2f}ms | P99: {p99_latency:.2f}ms | Memory: {mem_mb:.1f}MB")
        del kb_vecs

    return results


def plot_scaling_cost(results):
    """绘制: KB规模 vs 延迟/内存"""
    sizes_k = [r["kb_size"] / 1000 for r in results]
    avg_lats = [r["avg_latency_ms"] for r in results]
    p99_lats = [r["p99_latency_ms"] for r in results]
    memories = [r["memory_mb"] for r in results]

    fig, (ax_latency, ax_memory) = plt.subplots(1, 2, figsize=(14, 6.2))  # 增加高度

    # ========== 左图: 延迟 vs KB规模 ==========
    ax_latency.plot(sizes_k, avg_lats, 'o-', color=COLORS['primary'], 
                    linewidth=2.5, markersize=9, markeredgecolor='white', 
                    markeredgewidth=1.5, label='Avg Latency (mean)', zorder=5)
    ax_latency.plot(sizes_k, p99_lats, 's--', color=COLORS['accent2'], 
                    linewidth=2, markersize=7, alpha=0.7, 
                    label='P99 Latency (99th percentile)', zorder=4)
    ax_latency.fill_between(sizes_k, avg_lats, alpha=0.06, color=COLORS['primary'])

    # 标注数值 (修复: 添加 ms 单位)
    for x, y in zip(sizes_k, avg_lats):
        ax_latency.text(x, y + max(avg_lats) * 0.04, f"{y:.1f}ms", 
                        ha='center', fontsize=9, fontweight='bold', 
                        color=COLORS['primary'])
    
    # 标注数值 (P99 Latency) - 新增
    for x, y in zip(sizes_k, p99_lats):
        ax_latency.text(x, y + max(p99_lats) * 0.04, f"{y:.1f}ms", 
                        ha='center', fontsize=9, fontweight='bold', 
                        color=COLORS['accent2'])

    # 增长倍数标注
    speedup = avg_lats[-1] / avg_lats[0]
    ax_latency.annotate(
        f'{speedup:.1f}× slower',
        xy=(sizes_k[-1], avg_lats[-1]),
        xytext=(sizes_k[-1] - 20, avg_lats[-1] * 0.6),
        fontsize=12, fontweight='bold', color=COLORS['secondary'],
        arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_red'],
                  edgecolor=COLORS['secondary'], alpha=0.9),
    )

    ax_latency.set_xlabel("KB Size (×1K documents)", fontsize=13)
    ax_latency.set_ylabel("Retrieval Latency (ms/query)", fontsize=13)
    ax_latency.set_title("(a) Retrieval Latency vs KB Size", fontsize=14, fontweight='bold')
    ax_latency.legend(fontsize=10, loc='upper left')
    ax_latency.set_ylim(bottom=0)
    ax_latency.grid(True, alpha=0.3)

    # ========== 右图: 内存 vs KB规模 ==========
    ax_memory.plot(sizes_k, memories, 's-', color=COLORS['secondary'], 
                   linewidth=2.5, markersize=9, markeredgecolor='white', 
                   markeredgewidth=1.5, zorder=5)
    ax_memory.fill_between(sizes_k, memories, alpha=0.06, color=COLORS['secondary'])

    # 标注数值 (修复: 添加 MB 单位)
    for x, y in zip(sizes_k, memories):
        label = f"{y:.0f}MB" if y >= 1 else f"{y:.1f}MB"
        ax_memory.text(x, y + max(memories) * 0.04, label, 
                       ha='center', fontsize=9, fontweight='bold', 
                       color=COLORS['secondary'])

    # 内存增长倍数
    mem_growth = memories[-1] / memories[0]
    ax_memory.annotate(
        f'{mem_growth:.0f}× memory',
        xy=(sizes_k[-1], memories[-1]),
        xytext=(sizes_k[-1] - 25, memories[-1] * 0.5),
        fontsize=12, fontweight='bold', color=COLORS['accent2'],
        arrowprops=dict(arrowstyle='->', color=COLORS['accent2'], lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light_yellow'],
                  edgecolor=COLORS['accent2'], alpha=0.9),
    )

    ax_memory.set_xlabel("KB Size (×1K documents)", fontsize=13)
    ax_memory.set_ylabel("Memory Usage (MB)", fontsize=13)
    ax_memory.set_title("(b) Memory Footprint vs KB Size", fontsize=14, fontweight='bold')
    ax_memory.set_ylim(bottom=0)
    ax_memory.grid(True, alpha=0.3)

    # ========== 底部警告文本 (修复位置) ==========
    fig.text(0.5, 0.01,  # 从 -0.02 改为 0.01
             "WARNING: Naive scaling is unsustainable -> Need drift-aware selective updates",
             ha='center', fontsize=10, fontstyle='italic', color=COLORS['gray'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F9FAFB',
                       edgecolor=COLORS['gray'], alpha=0.8))

    # 调整布局 (给底部文本留出空间)
    fig.tight_layout(rect=[0, 0.06, 1, 1])  # 底部从 0.04 改为 0.06
    
    # 保存图片
    out_path = OUT_DIR / "fig_kb_size_vs_latency_memory.png"
    save_fig(fig, str(out_path))
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    results = benchmark_retrieval_overhead()
    plot_scaling_cost(results)
    print("\n✅ Motivation 3 实验完成!")
    print(f"📊 图片已保存至: {OUT_DIR / 'fig_kb_size_vs_latency_memory.png'}")