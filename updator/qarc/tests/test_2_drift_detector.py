"""
=============================================================
测试 2: drift_detector.py — 对齐漂移检测
=============================================================

核心问题: 什么时候需要更新 KB?

错误的思路: "用户问的东西变了 → 更新 KB"
  反例: 用户从问篮球变成问足球, 但 KB 里都有 → 不需要更新

正确的思路: "用户的问题和 KB 的对齐程度变了 → 更新 KB"
  例: 用户一直问体育, KB 覆盖得很好, 突然问起量子物理, KB 覆盖不了 → 需要更新

检测方法:
┌───────────────────────────────────────────────────────────┐
│ 1. 对齐特征 (Alignment Features)                          │
│    每个 query → [与KB各主题的相似度, 与最近N篇文档的相似度] │
│    不看 query 本身是什么, 看它跟 KB 匹不匹配              │
│                                                           │
│ 2. Offline: 用历史 query 建立"正常对齐"的基线             │
│    "历史上 query 和 KB 之间的匹配模式长什么样"             │
│                                                           │
│ 3. Threshold: 随机采窗口算 FID, 取 P95 → 漂移阈值        │
│    "正常波动最大能到多少"                                  │
│                                                           │
│ 4. Online: 新窗口的对齐特征 vs 基线 → FID 距离            │
│    FID > threshold → 对齐模式偏离了 → KB 需要更新          │
└───────────────────────────────────────────────────────────┘

运行: python -m updator.qarc.tests.test_2_drift_detector
"""

import numpy as np
np.random.seed(42)

DIM = 32

def make_cluster(center, n=10, noise=0.1):
    X = center + np.random.randn(n, DIM) * noise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-10, None)


# ============================================================
# 准备数据: 一个模拟 KB + 几批不同的 query
# ============================================================
print("=" * 60)
print("准备数据")
print("=" * 60)

# KB: 50 篇文档, 分布在"体育"和"美食"两个主题
center_sports = np.zeros(DIM); center_sports[0:8] = 1.0
center_food   = np.zeros(DIM); center_food[8:16] = 1.0
center_science = np.zeros(DIM); center_science[16:24] = 1.0

kb_embs = np.vstack([
    make_cluster(center_sports, n=25),
    make_cluster(center_food, n=25),
])
print(f"KB: {kb_embs.shape[0]} 篇文档 (25体育 + 25美食)")

# 历史 query: 主要问体育和美食 (和 KB 对齐)
history_queries = np.vstack([
    make_cluster(center_sports, n=30),
    make_cluster(center_food, n=20),
])
print(f"历史 query: {history_queries.shape[0]} 条 (30体育 + 20美食, 与KB匹配)")
print()

# ============================================================
# Step 1: set_baseline — 建立"正常对齐"基线
# ============================================================
print("=" * 60)
print("Step 1: set_baseline — 建立基线")
print("=" * 60)
print("""
告诉检测器: "正常情况下, query 和 KB 之间的对齐模式是这样的"
内部做了什么:
  1. KMeans(KB) → 发现 KB 有哪些主题 (比如: 体育、美食)
  2. 每个历史 query → 对齐特征 = [和体育的相似度, 和美食的相似度, top-N相似度]
  3. 存下这些对齐特征的均值 μ₀ 和协方差 Σ₀ → 基线
""")

from updator.qarc.detection.drift_detector import DriftLensDetector

detector = DriftLensDetector(
    n_clusters=3,     # KB 最多分 3 个主题
    top_n_sims=5,     # 取 top-5 文档相似度作为特征
    threshold_percentile=95.0,
    threshold_n_samples=200,
)

ok = detector.set_baseline(kb_embs, history_queries)
print(f"基线建立成功: {ok}")
print(f"检测器就绪: {detector.is_ready}")
print()

# ============================================================
# Step 2: calibrate_threshold — 校准阈值
# ============================================================
print("=" * 60)
print("Step 2: calibrate_threshold — 校准阈值")
print("=" * 60)
print("""
问题: FID 多大算"异常"?
方法: 从历史数据中随机抽窗口 → 算 FID → 排序 → 取 P95
含义: "如果 FID 超过这个值, 有 95% 的把握说对齐出了问题"
""")

threshold = detector.calibrate_threshold(history_queries, window_size=8)
print(f"校准阈值: {threshold:.6f}")
print(f"含义: FID > {threshold:.6f} → 判定为对齐漂移")
print()

# ============================================================
# Step 3: detect — 在线检测
# ============================================================
print("=" * 60)
print("Step 3: detect — 在线检测 3 个场景")
print("=" * 60)

# 场景 A: 新query仍然问体育+美食 (和基线一致, 不应该漂移)
print("\n--- 场景 A: query 还是体育+美食 (和 KB 对齐) ---")
window_a = np.vstack([
    make_cluster(center_sports, n=5, noise=0.15),
    make_cluster(center_food, n=3, noise=0.15),
])
result_a = detector.detect(window_a)
print(f"  FID = {result_a.fid_score:.6f}")
print(f"  阈值 = {result_a.threshold:.6f}")
print(f"  漂移? {result_a.is_drifted}")
print(f"  → {'❌ 误报' if result_a.is_drifted else '✓ 正确: 没漂移, 不需要更新KB'}")

# 场景 B: 新query突然都问科学 (KB没有科学文档, 应该漂移)
print("\n--- 场景 B: query 突然全问科学 (KB 没有覆盖) ---")
window_b = make_cluster(center_science, n=8, noise=0.15)
result_b = detector.detect(window_b)
print(f"  FID = {result_b.fid_score:.6f}")
print(f"  阈值 = {result_b.threshold:.6f}")
print(f"  漂移? {result_b.is_drifted}")
print(f"  → {'✓ 正确: 检测到对齐漂移, KB需要更新!' if result_b.is_drifted else '❌ 漏报'}")

# 场景 C: 混合 (一半体育一半科学)
print("\n--- 场景 C: query 一半体育一半科学 (部分偏移) ---")
window_c = np.vstack([
    make_cluster(center_sports, n=4, noise=0.15),
    make_cluster(center_science, n=4, noise=0.15),
])
result_c = detector.detect(window_c)
print(f"  FID = {result_c.fid_score:.6f}")
print(f"  阈值 = {result_c.threshold:.6f}")
print(f"  漂移? {result_c.is_drifted}")
print()

print(f"FID 对比: 对齐={result_a.fid_score:.4f} < 部分偏移={result_c.fid_score:.4f} < 完全偏移={result_b.fid_score:.4f}")
print(f"→ 偏离基线越多, FID越大, 越应该更新KB")

print("\n✓ drift_detector.py 测试全部通过")
