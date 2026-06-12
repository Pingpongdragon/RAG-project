"""
=============================================================
测试 1: interest_model.py — 兴趣建模基础工具
=============================================================

这个文件提供 QARC 最底层的三个工具:

┌─────────────────────────────────────────────────────┐
│ 1. QueryWindowBuffer  — 窗口缓冲区                   │
│    攒够 W 个 query 再一起分析                        │
│                                                     │
│ 2. auto_kmeans        — 自动聚类                     │
│    把一批 query 分成若干组, 找出用户关心哪些主题     │
│                                                     │
│ 3. compute_alignment_gap — KB 对齐度                 │
│    衡量当前 KB 和用户兴趣之间的"差距"                │
│    Gap≈0 → KB完美覆盖, Gap≈1 → KB完全不匹配        │
└─────────────────────────────────────────────────────┘

运行: python -m algorithms.qarc.tests.test_1_interest_model
"""

import numpy as np
np.random.seed(42)

DIM = 32  # 用小维度便于演示

def make_cluster(center, n=10, noise=0.1):
    """造一组聚集在 center 附近的向量"""
    X = center + np.random.randn(n, DIM) * noise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-10, None)  # L2 归一化


# ============================================================
# 测试 1: QueryWindowBuffer — 窗口缓冲区
# ============================================================
print("=" * 60)
print("测试 1: QueryWindowBuffer — 窗口缓冲区")
print("=" * 60)
print("""
用途: 把一个一个来的 query 攒成一批再处理
就像排队坐车: 攒够人数再发车

流程: add() → is_full? → flush() 取出数据
""")

from algorithms.qarc.curation.interest_model import QueryWindowBuffer

buf = QueryWindowBuffer(window_size=5)
print(f"窗口大小: {buf.window_size}")
print(f"已有: {buf.size} 条")

for i in range(5):
    emb = np.random.randn(DIM)
    buf.add(embedding=emb, text=f"问题{i}", max_sim_to_kb=0.5 + i * 0.1)
    print(f"  添加第 {i+1} 条, 缓冲区={buf.size}/{buf.window_size}, 满了? {buf.is_full}")

embeddings, texts, sims = buf.flush()
print(f"\nflush 后: {len(embeddings)} 条 embedding, 文本={texts}")
print(f"相似度: {sims}")
print(f"缓冲区清空: {buf.size} 条")
print()


# ============================================================
# 测试 2: auto_kmeans — 自动聚类
# ============================================================
print("=" * 60)
print("测试 2: auto_kmeans — 自动聚类")
print("=" * 60)
print("""
用途: 发现用户关心哪些主题, 以及每个主题有多重要

输入: 一批 query 的 embedding
输出: centroids (主题中心), weights (主题重要性)

例子: 10个体育问题 + 5个美食问题
  → 发现 2 个主题簇
  → 体育 weight=0.67, 美食 weight=0.33
""")

from algorithms.qarc.curation.interest_model import auto_kmeans

# 制造两簇不同方向的向量 (模拟两个不同主题)
center_sports = np.zeros(DIM); center_sports[0:8] = 1.0  # "体育"方向
center_food   = np.zeros(DIM); center_food[8:16] = 1.0   # "美食"方向

queries_sports = make_cluster(center_sports, n=10)
queries_food   = make_cluster(center_food, n=5)
X = np.vstack([queries_sports, queries_food])  # 15个query: 10体育+5美食

centroids, labels, weights = auto_kmeans(X)

print(f"输入: {X.shape[0]} 个 query embedding")
print(f"自动选择: {len(centroids)} 个主题簇")
print(f"各簇权重: {weights}")
print(f"各簇大小: {[int((labels == i).sum()) for i in range(len(centroids))]}")
print()

# 验证: 应该发现 ~2 个簇, 一个大(体育) 一个小(美食)
# 权重大的簇 = 用户更关心的话题
max_w_idx = np.argmax(weights)
print(f"最大权重簇 (idx={max_w_idx}): weight={weights[max_w_idx]:.2f}")
print(f"  → 这个簇被分配了 {int((labels == max_w_idx).sum())} 个query = 用户最关心的话题")
print()


# ============================================================
# 测试 3: compute_alignment_gap — KB 对齐度
# ============================================================
print("=" * 60)
print("测试 3: compute_alignment_gap — KB 对齐度")
print("=" * 60)
print("""
用途: 衡量 KB 与用户兴趣匹配不匹配

公式: G(t) = 1 - avg(max_sim(query, KB_doc))
  G ≈ 0 → 每个query都能在KB里找到高相似的文档 → 匹配好
  G ≈ 1 → query和KB文档完全不搭 → 匹配差, 需要更新KB
""")

from algorithms.qarc.curation.interest_model import compute_alignment_gap

# 场景 A: KB 和 query 匹配得很好 (KB包含体育和美食文档)
kb_good = np.vstack([
    make_cluster(center_sports, n=5),   # KB有体育文档
    make_cluster(center_food, n=5),     # KB有美食文档
])
gap_good = compute_alignment_gap(X, kb_good)
print(f"场景 A: KB 包含体育+美食文档")
print(f"  Gap = {gap_good.gap:.4f} (越小越好)")
print(f"  平均最大相似度 = {gap_good.avg_max_sim:.4f}")
print()

# 场景 B: KB 只有美食, 没有体育 (部分匹配)
kb_partial = make_cluster(center_food, n=10)
gap_partial = compute_alignment_gap(X, kb_partial)
print(f"场景 B: KB 只有美食文档, 没有体育")
print(f"  Gap = {gap_partial.gap:.4f} (比场景A大, 因为体育query没覆盖)")
print()

# 场景 C: KB 是完全不相关的内容
center_random = np.zeros(DIM); center_random[24:32] = 1.0
kb_bad = make_cluster(center_random, n=10)
gap_bad = compute_alignment_gap(X, kb_bad)
print(f"场景 C: KB 内容完全不相关")
print(f"  Gap = {gap_bad.gap:.4f} (最大, KB完全没覆盖)")
print()

print(f"对比: 好={gap_good.gap:.3f} < 部分={gap_partial.gap:.3f} < 差={gap_bad.gap:.3f}")
print(f"→ Gap 越大, KB越需要更新!")

print("\n✓ interest_model.py 测试全部通过")
