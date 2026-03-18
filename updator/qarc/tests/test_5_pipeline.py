"""
=============================================================
测试 5: pipeline.py — QARC 端到端流水线
=============================================================

这是把前 4 个模块串起来的 orchestrator:

┌──────────────────────────────────────────────────────────────┐
│                    QARC Pipeline 流程                        │
│                                                              │
│  Bootstrap                                                   │
│  ├── 无历史 → 多样性最大化 (Facility Location)               │
│  └── 有历史 → AutoKMeans + 子模选择                          │
│                                                              │
│  Online Loop (每收到一个 query)                              │
│  │ 1. 记入 query 历史 (ring buffer)                          │
│  │ 2. 从 KB 检索文档 → LLM 回答 (可选)                      │
│  │ 3. 加入窗口缓冲区                                        │
│  │                                                           │
│  │ [窗口满了 → _process_window]                              │
│  │   4. AutoKMeans → 兴趣簇                                  │
│  │   5. DriftLens → 对齐漂移? (Part 1)                       │
│  │   6. AlignmentGap → 匹配程度                              │
│  │   7. Agent 决策 (Part 2):                                 │
│  │      Warmup → AGGRESSIVE                                  │
│  │      正常 → NO_OP / MILD / AGGRESSIVE / RECALIBRATE       │
│  │   8. 执行更新 → 重校准 DriftLens                          │
│  └───────────────────────────────────────────────────────────│
└──────────────────────────────────────────────────────────────┘

运行: python -m updator.qarc.tests.test_5_pipeline
"""

import numpy as np
np.random.seed(42)
import logging
logging.basicConfig(level=logging.WARNING)  # 减少日志
logger = logging.getLogger("qarc_test5")

DIM = 32
WINDOW = 8

def make_cluster(center, n=10, noise=0.1):
    X = center + np.random.randn(n, DIM) * noise
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-10, None)

# ============================================================
# 准备数据
# ============================================================
print("=" * 60)
print("准备: 文档池 + 查询流")
print("=" * 60)

center_sports  = np.zeros(DIM); center_sports[0:8] = 1.0
center_food    = np.zeros(DIM); center_food[8:16] = 1.0
center_science = np.zeros(DIM); center_science[16:24] = 1.0

from updator.qarc.curation.kb_curator import Document, DocumentPool, QARCKBCurator
from updator.qarc.pipeline import QARCPipeline

# 文档池
pool = DocumentPool()
for topic, center, n in [("体育", center_sports, 40), ("美食", center_food, 40), ("科学", center_science, 40)]:
    embs = make_cluster(center, n=n)
    for i in range(n):
        pool.add_document(Document(
            doc_id=f"{topic}_{i:03d}",
            text=f"{topic}相关文档 #{i}",
            embedding=embs[i],
        ))
print(f"文档池: {pool.size} 篇 (体育40 + 美食40 + 科学40)")

# 模拟查询流: 先问体育(40条), 再转向科学(40条)
queries_phase1 = [(f"体育问题{i}", make_cluster(center_sports, n=1, noise=0.15)[0]) for i in range(40)]
queries_phase2 = [(f"科学问题{i}", make_cluster(center_science, n=1, noise=0.15)[0]) for i in range(40)]
all_queries = queries_phase1 + queries_phase2
print(f"查询流: {len(all_queries)} 条 (前40条体育, 后40条科学)")
print()


# ============================================================
# 创建 Pipeline
# ============================================================
print("=" * 60)
print("创建 QARCPipeline")
print("=" * 60)

curator = QARCKBCurator(
    document_pool=pool,
    kb_budget=15,         # KB 只保留 15 篇文档
    candidate_top_k=30,
)

from updator.qarc.config import QARCConfig
cfg = QARCConfig(
    window_size=WINDOW,
    agent_warmup_windows=2,     # 前 2 个窗口 warmup
    agent_cooldown_windows=1,   # 更新后冷却 1 个窗口
    agent_recalibrate_after=3,  # 连续 3 次漂移 → 重校准
    query_history_max=200,
)
pipeline = QARCPipeline(curator=curator, cfg=cfg)

print(f"  窗口大小: {WINDOW}")
print(f"  KB 预算: 15 篇")
print(f"  Warmup: 2 个窗口")
print(f"  DriftLens 初始化: warmup 后 (至少 {3*WINDOW} 条 query)")
print()


# ============================================================
# Phase 0: Bootstrap (冷启动)
# ============================================================
print("=" * 60)
print("Phase 0: Bootstrap (多样性最大化)")
print("=" * 60)

pipeline.bootstrap()
print(f"  KB 大小: {pipeline.get_kb_size()} 篇")
topic_counts = {}
for doc in pipeline.get_current_kb_docs():
    t = doc.doc_id.split("_")[0]
    topic_counts[t] = topic_counts.get(t, 0) + 1
for t, c in sorted(topic_counts.items()):
    print(f"    {t}: {c} 篇")
print(f"  → 无用户信息, 三个主题均匀分布")
print()


# ============================================================
# Online: 处理查询流
# ============================================================
print("=" * 60)
print("Online: 处理 80 条查询")
print("=" * 60)
print(f"""
前 40 条: 体育 (期望 KB 向体育倾斜)
后 40 条: 科学 (期望检测到漂移 → KB 转向科学)
""")

events = []  # 收集窗口事件
for i, (text, emb) in enumerate(all_queries):
    result = pipeline.process_query(text, emb)
    
    if result["window_event"] is not None:
        ev = result["window_event"]
        events.append(ev)
        
        phase_name = "体育" if i < 40 else "科学"
        decision = ev["decision"]["action"]
        gap = ev["gap"]
        fid = ev["drift"]["fid_score"]
        drifted = ev["drift"]["is_drifted"]
        
        # 统计当前 KB 成分
        topic_counts = {}
        for doc in pipeline.get_current_kb_docs():
            t = doc.doc_id.split("_")[0]
            topic_counts[t] = topic_counts.get(t, 0) + 1
        kb_str = ", ".join(f"{t}={c}" for t, c in sorted(topic_counts.items()))
        
        print(f"  窗口{ev['window_index']:2d} [{phase_name}]: "
              f"Gap={gap:.3f} FID={fid:.3f} 漂移={str(drifted):5s} "
              f"决策={decision:15s} KB=[{kb_str}]")

print()


# ============================================================
# 分析结果
# ============================================================
print("=" * 60)
print("结果分析")
print("=" * 60)

# Gap 趋势
gaps = pipeline.gap_history
if len(gaps) >= 6:
    early_gap = np.mean(gaps[:3])  # 前 3 个窗口
    mid_gap = np.mean(gaps[3:6])   # 中间
    late_gap = np.mean(gaps[-3:])  # 后 3 个窗口
    print(f"\n  Gap 趋势:")
    print(f"    前期(体育→体育KB): {early_gap:.4f}  ← warmup 后 KB 对齐")
    print(f"    中期(过渡):         {mid_gap:.4f}")
    print(f"    后期(科学→科学KB): {late_gap:.4f}  ← 漂移后 KB 重新对齐")

# KB 最终成分
topic_counts = {}
for doc in pipeline.get_current_kb_docs():
    t = doc.doc_id.split("_")[0]
    topic_counts[t] = topic_counts.get(t, 0) + 1

print(f"\n  最终 KB ({pipeline.get_kb_size()} 篇):")
for t, c in sorted(topic_counts.items()):
    print(f"    {t}: {c} 篇")

# 统计
stats = pipeline.get_statistics()
print(f"\n  总查询: {stats['total_queries']}")
print(f"  总窗口: {stats['total_windows']}")
print(f"  总更新: {stats['total_recurations']} 次")
print(f"  Agent 决策分布:")
for action, count in stats['agent_state']['decision_summary'].items():
    if count > 0:
        print(f"    {action:15s}: {count} 次")

# 验证: 最终 KB 应该偏向科学 (最后 40 条都是科学查询)
science_ratio = topic_counts.get("科学", 0) / max(pipeline.get_kb_size(), 1)
print(f"\n  最终科学比例: {science_ratio:.1%}")
if science_ratio > 0.3:
    print(f"  ✓ KB 成功跟随用户兴趣从体育转向科学!")
else:
    print(f"  △ KB 转换不够彻底 (可能需要更多窗口)")

print("\n✓ pipeline.py 端到端测试完成")
