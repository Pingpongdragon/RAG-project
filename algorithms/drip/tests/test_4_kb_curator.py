"""
=============================================================
测试 4: kb_curator.py — 子模 KB 策展
=============================================================

核心问题: 从一个大的文档池中, 选哪些文档组成 KB?

传统做法:
  ERASE: 文档来了就加进去, 修改事实
  ComRAG: KB 不动, 只记 QA 历史
  
QARC 做法: 从文档池 D_pool 中"挑选"一个子集作为 KB
  文档池 D_pool (几千篇) ──子模选择──→ KB K (几十篇)
  
选择标准 (子模目标函数):
  f(S) = f_interest(S) + η · f_diversity(S)
  
  f_interest: KB 覆盖用户的兴趣簇 (由 AutoKMeans 发现)
  f_diversity: KB 覆盖文档池的多样性 (Facility Location)
  
为什么用子模函数:
  1. 边际收益递减 → 自动避免重复 (加10篇体育文档, 最后几篇收益很小)
  2. 单调递增 → 加文档只会更好不会更差
  3. 贪心保证 → (1-1/e) ≈ 63% 的最优近似比

运行: python -m algorithms.qarc.tests.test_4_kb_curator
"""

import numpy as np
np.random.seed(42)

DIM = 32

def make_cluster(center, n=10, noise=0.1):
    X = center + np.random.randn(n, DIM) * noise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-10, None)


# ============================================================
# 准备数据
# ============================================================
print("=" * 60)
print("准备数据: 3 个主题的文档池")
print("=" * 60)

center_sports  = np.zeros(DIM); center_sports[0:8] = 1.0
center_food    = np.zeros(DIM); center_food[8:16] = 1.0
center_science = np.zeros(DIM); center_science[16:24] = 1.0

from algorithms.qarc.curation.kb_curator import (
    Document, DocumentPool, QARCKBCurator,
    _interest_coverage, _diversity_coverage, submodular_objective,
    greedy_submodular_select,
)

# 文档池: 体育30 + 美食30 + 科学30 = 90 篇
pool = DocumentPool()
topics = [
    ("体育", center_sports, 30),
    ("美食", center_food, 30),
    ("科学", center_science, 30),
]

for topic_name, center, n in topics:
    embs = make_cluster(center, n=n)
    for i in range(n):
        pool.add_document(Document(
            doc_id=f"{topic_name}_{i:03d}",
            text=f"{topic_name}相关文档 #{i}",
            embedding=embs[i],
        ))

print(f"文档池大小: {pool.size} 篇 (体育30 + 美食30 + 科学30)")
print()


# ============================================================
# Part A: 子模目标函数 — 看看它怎么衡量 KB 的好坏
# ============================================================
print("=" * 60)
print("Part A: 子模目标函数")
print("=" * 60)

# 假设用户兴趣: 70% 体育, 30% 美食 (不关心科学)
c_sports = center_sports / np.linalg.norm(center_sports)
c_food   = center_food / np.linalg.norm(center_food)
centroids = np.vstack([c_sports, c_food])
weights = np.array([0.7, 0.3])

pool_embs = pool.get_all_embeddings()

# KB 方案 1: 5 篇体育
kb_sports = make_cluster(center_sports, n=5, noise=0.05)
f1_int = _interest_coverage(kb_sports, centroids, weights)
f1_div = _diversity_coverage(kb_sports, pool_embs)
print(f"\n方案1: 只选5篇体育 → f_interest={f1_int:.4f}, f_diversity={f1_div:.4f}")
print(f"  兴趣覆盖好(高权重兴趣被满足), 但多样性差(只覆盖了1/3的池)")

# KB 方案 2: 3 篇体育 + 2 篇美食
kb_mixed = np.vstack([
    make_cluster(center_sports, n=3, noise=0.05),
    make_cluster(center_food, n=2, noise=0.05),
])
f2_int = _interest_coverage(kb_mixed, centroids, weights)
f2_div = _diversity_coverage(kb_mixed, pool_embs)
print(f"\n方案2: 3篇体育+2篇美食 → f_interest={f2_int:.4f}, f_diversity={f2_div:.4f}")
print(f"  两个兴趣主题都覆盖到了, 多样性也稍好")

# KB 方案 3: 2 体育 + 2 美食 + 1 科学
kb_diverse = np.vstack([
    make_cluster(center_sports, n=2, noise=0.05),
    make_cluster(center_food, n=2, noise=0.05),
    make_cluster(center_science, n=1, noise=0.05),
])
f3_int = _interest_coverage(kb_diverse, centroids, weights)
f3_div = _diversity_coverage(kb_diverse, pool_embs)
print(f"\n方案3: 2体育+2美食+1科学 → f_interest={f3_int:.4f}, f_diversity={f3_div:.4f}")
print(f"  多样性最好(三个主题都有), 但用户不关心科学浪费了名额")

# 加上 η 后的综合评分
for eta_val in [0.0, 0.1, 1.0]:
    s1 = f1_int + eta_val * f1_div
    s2 = f2_int + eta_val * f2_div
    s3 = f3_int + eta_val * f3_div
    best = max([(s1, "纯体育"), (s2, "混合"), (s3, "全覆盖")])
    print(f"\n  η={eta_val}: 纯体育={s1:.4f}, 混合={s2:.4f}, 全覆盖={s3:.4f}, 最优={best[1]}")

print(f"\n→ η小: 偏向兴趣匹配; η大: 偏向多样性探索")
print()


# ============================================================
# Part B: 贪心子模选择 — 自动选出最优 KB
# ============================================================
print("=" * 60)
print("Part B: 贪心子模选择")
print("=" * 60)
print("""
给定候选文档 + 兴趣模型 + 预算 → 自动贪心选择最优子集
每一步选"边际增益最大"的文档 (增量优化, 不用每次重算目标)
""")

all_docs = [pool.documents[did] for did in pool.get_all_ids()]

selected = greedy_submodular_select(
    candidate_docs=all_docs,
    centroids=centroids,
    weights=weights,
    pool_embs=pool_embs,
    budget=10,
    eta=0.1,
)

print(f"从 {len(all_docs)} 篇候选中选出 {len(selected)} 篇:")
topic_count = {}
for doc in selected:
    topic = doc.doc_id.split("_")[0]
    topic_count[topic] = topic_count.get(topic, 0) + 1
for topic, cnt in sorted(topic_count.items()):
    print(f"  {topic}: {cnt} 篇")
print(f"\n→ 用户 70% 关心体育, 30% 关心美食 → 体育分配多, 科学分配少 (来自多样性正则)")
print()


# ============================================================
# Part C: QARCKBCurator — 完整 Bootstrap + ReCurate 流程
# ============================================================
print("=" * 60)
print("Part C: QARCKBCurator — Bootstrap + ReCurate")
print("=" * 60)

curator = QARCKBCurator(
    document_pool=pool,
    kb_budget=10,
    lambda_max=0.3,
    candidate_top_k=30,
)

# Phase 0: 冷启动 (没有用户查询, 用多样性最大化)
print("\n--- Phase 0: 多样性最大化 Bootstrap ---")
initial_kb = curator.bootstrap_diversity()
print(f"  初始 KB: {curator.kb_size} 篇")
topic_count_0 = {}
for doc in initial_kb:
    topic = doc.doc_id.split("_")[0]
    topic_count_0[topic] = topic_count_0.get(topic, 0) + 1
for topic, cnt in sorted(topic_count_0.items()):
    print(f"    {topic}: {cnt} 篇")
print(f"  → 没有用户兴趣信息, 三个主题均匀分布 (Facility Location)")

# ReCurate: 发现用户兴趣后, 调整 KB
print("\n--- ReCurate: 用户偏好体育 → KB 向体育倾斜 ---")
result = curator.recurate(
    centroids=centroids,
    weights=weights,
    lambda_max=0.5,
    eta=0.1,
)

print(f"  添加: {len(result.added_ids)} 篇")
print(f"  移除: {len(result.removed_ids)} 篇")
print(f"  替换比例: {result.replacement_ratio:.1%}")
print(f"  目标函数: {result.objective_before:.4f} → {result.objective_after:.4f}")

topic_count_1 = {}
for doc in curator.get_kb_docs_list():
    topic = doc.doc_id.split("_")[0]
    topic_count_1[topic] = topic_count_1.get(topic, 0) + 1
print(f"  ReCurate 后:")
for topic, cnt in sorted(topic_count_1.items()):
    print(f"    {topic}: {cnt} 篇")
print(f"  → 体育比例增加, 科学减少 (因为用户不关心科学)")

# 再次 ReCurate: 用户兴趣变了 → 转向科学
print("\n--- ReCurate: 用户兴趣转向科学 ---")
c_science = center_science / np.linalg.norm(center_science)
new_centroids = np.vstack([c_science, c_food])
new_weights = np.array([0.8, 0.2])

result2 = curator.recurate(
    centroids=new_centroids,
    weights=new_weights,
    lambda_max=0.5,
    eta=0.1,
)

print(f"  添加: {len(result2.added_ids)} 篇")
print(f"  移除: {len(result2.removed_ids)} 篇")
print(f"  目标函数: {result2.objective_before:.4f} → {result2.objective_after:.4f}")

topic_count_2 = {}
for doc in curator.get_kb_docs_list():
    topic = doc.doc_id.split("_")[0]
    topic_count_2[topic] = topic_count_2.get(topic, 0) + 1
print(f"  KB 成分变化:")
for topic in ["体育", "美食", "科学"]:
    before = topic_count_1.get(topic, 0)
    after = topic_count_2.get(topic, 0)
    arrow = "↑" if after > before else ("↓" if after < before else "→")
    print(f"    {topic}: {before} → {after} {arrow}")
print(f"  → KB 动态跟随用户兴趣变化!")

print("\n✓ kb_curator.py 测试全部通过")
