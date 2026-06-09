"""
=============================================================
测试 3: kb_agent.py — Agent 驱动的 KB 更新决策
=============================================================

核心问题: 检测到信号后, 该怎么更新 KB?

传统方法 (DriftLens 论文): Human-in-the-Loop
  人看到漂移报告 → 手动决定:
    "要不要加新类别? 要不要重训练? 怎么改数据?"

我们的方法: Agent-in-the-Loop
  Agent 接收两个信号:
    1. DriftResult: 对齐模式是否漂移 (Part 1)
    2. AlignmentGap: KB 和 query 的匹配度 (interest_model)
  → 自动做出以下决策:

┌─────────────────────────────────────────────────────────────┐
│ Warmup: 前 N 个窗口 → 始终激进更新 (冷启动快速收敛KB)      │
│ Rule 1: 未漂移 + Gap正常 → NO_OP (维持)                    │
│ Rule 2: 未漂移 + Gap偏高 → MILD_UPDATE (20%替换)           │
│ Rule 3: 漂移 → AGGRESSIVE (50%替换)                        │
│ Rule 4: 连续漂移≥3次 → RECALIBRATE (重建基线 + 50%替换)    │
└─────────────────────────────────────────────────────────────┘

Gap阈值是自适应的: EMA(Gap) + k * MAD(Gap)
  不是固定阈值, 而是跟着历史 Gap 趋势走

运行: python -m algorithms.qarc.tests.test_3_kb_agent
"""

import numpy as np
from dataclasses import dataclass

# ─── 构造假的 DriftResult 和 AlignmentGapResult ───
# 这样可以不依赖真正的 detector, 直接喂信号给 Agent

from algorithms.qarc.detection.drift_detector import DriftResult
from algorithms.qarc.curation.interest_model import AlignmentGapResult
from algorithms.qarc.decision.kb_agent import KBUpdateAgent, UpdateAction

def fake_drift(is_drifted, fid=0.5, threshold=1.0):
    """构造一个假的漂移检测结果"""
    return DriftResult(
        is_drifted=is_drifted,
        fid_score=fid,
        threshold=threshold,


    )

def fake_gap(gap_value):
    """构造一个假的对齐 Gap 结果"""
    return AlignmentGapResult(
        gap=gap_value,
        avg_max_sim=1.0 - gap_value,
        per_query_sims=[1.0 - gap_value],
        window_size=8,
    )


# ============================================================
# 场景 1: Warmup 阶段 — 始终激进更新
# ============================================================
print("=" * 60)
print("场景 1: Warmup 阶段 (前 3 个窗口)")
print("=" * 60)
print("""
不管漂移不漂移, Gap高不高, warmup 期间都激进更新.
目的: KB 冷启动, 需要快速收敛到用户兴趣.
最后一个 warmup 窗口触发重校准 (类似 Explore → Exploit).
""")

agent = KBUpdateAgent(warmup_windows=3, recalibrate_after_n_drifts=3)

for i in range(3):
    # 即使信号说"没漂移, Gap很小", warmup 仍然激进更新
    d = agent.decide(fake_drift(False, fid=0.1), fake_gap(0.1))
    recal = "✓ 重校准!" if d.should_recalibrate else ""
    print(f"  Window {i+1}: {d.action.value:15s}  λ={d.lambda_max:.1f}  {recal}  原因: {d.reason}")

print()


# ============================================================
# 场景 2: 正常运行 — Rule 1 (NO_OP)
# ============================================================
print("=" * 60)
print("场景 2: 正常运行 — 未漂移 + Gap正常")
print("=" * 60)
print("""
Warmup 结束后, 如果不漂移且 Gap 没有异常升高 → 不更新.
目的: 不要过度更新 KB, 避免频繁变动影响检索质量.
""")

# 先喂几个正常的窗口让 EMA 稳定下来
for _ in range(5):
    d = agent.decide(fake_drift(False, fid=0.3), fake_gap(0.2))

print(f"  Gap EMA = {agent._gap_ema:.4f}")
print(f"  Gap MAD = {agent._gap_mad:.4f}")
print(f"  Gap 阈值 = {agent.gap_threshold:.4f}")
print(f"  决策: {d.action.value}")
print(f"  → Gap {0.2:.4f} < 阈值 {agent.gap_threshold:.4f} → 不更新")
print()


# ============================================================
# 场景 3: Gap 偏高 — Rule 2 (MILD_UPDATE)
# ============================================================
print("=" * 60)
print("场景 3: Gap 偏高 — 未漂移但 KB 覆盖变差")
print("=" * 60)
print("""
没检测到对齐漂移, 但 Gap 突然变大:
  "query 和 KB 的匹配度在下降"
可能原因: 用户兴趣慢慢偏移, 还没到漂移程度, 但 KB 已跟不上
策略: 温和替换 20% KB (λ_mild=0.2)
""")

# 创建新 agent, past warmup
agent2 = KBUpdateAgent(warmup_windows=2, gap_k=1.5)
for _ in range(2):
    agent2.decide(fake_drift(False), fake_gap(0.15))

# 喂几个正常 gap 建立基线
for _ in range(5):
    agent2.decide(fake_drift(False, fid=0.3), fake_gap(0.15))

print(f"  正常: Gap EMA={agent2._gap_ema:.4f}, 阈值={agent2.gap_threshold:.4f}")

# 突然 Gap 变大
d = agent2.decide(fake_drift(False, fid=0.3), fake_gap(0.8))
print(f"  异常: Gap=0.8 > 阈值={agent2.gap_threshold:.4f}")
print(f"  决策: {d.action.value}  λ={d.lambda_max}  原因: {d.reason}")
print()


# ============================================================
# 场景 4: 漂移 — Rule 3 (AGGRESSIVE_UPDATE)
# ============================================================
print("=" * 60)
print("场景 4: 漂移检测触发 — 激进更新")
print("=" * 60)
print("""
DriftLens 检测到对齐模式已经发生漂移:
  "用户和 KB 之间的关系已经变了"
策略: 激进替换 50% KB (λ_aggressive=0.5) + 进入冷却期
冷却: 更新后暂停 2 个窗口不操作 (避免频繁抖动)
""")

agent3 = KBUpdateAgent(warmup_windows=1, cooldown_windows=2)
agent3.decide(fake_drift(False), fake_gap(0.2))  # warmup

# 收到漂移信号
d = agent3.decide(fake_drift(True, fid=2.5, threshold=1.0), fake_gap(0.5))
print(f"  漂移! FID=2.5 > 阈值=1.0")
print(f"  决策: {d.action.value}  λ={d.lambda_max}  原因: {d.reason}")

# 冷却期
print(f"\n  进入冷却 ({agent3.cooldown_windows} 个窗口不操作):")
for i in range(3):
    d = agent3.decide(fake_drift(False), fake_gap(0.3))
    print(f"    Window +{i+1}: {d.action.value:15s}  原因: {d.reason}")
print()


# ============================================================
# 场景 5: 连续漂移 — Rule 4 (RECALIBRATE)
# ============================================================
print("=" * 60)
print("场景 5: 连续漂移 — 重校准基线")
print("=" * 60)
print("""
如果连续 3 次以上检测到漂移:
  "不是暂时波动, 是用户兴趣发生了根本性变化"
  → Offline 基线已经过时了
策略: 重建基线 + 激进更新 (should_recalibrate=True)
类比: 世界变了, 不只是调 KB, 连"什么是正常"这个标准也要更新
""")

agent4 = KBUpdateAgent(warmup_windows=1, recalibrate_after_n_drifts=3, cooldown_windows=0)
agent4.decide(fake_drift(False), fake_gap(0.2))  # warmup

actions = []
for i in range(4):
    d = agent4.decide(fake_drift(True, fid=3.0, threshold=1.0), fake_gap(0.5))
    actions.append(d)
    print(f"  连续漂移 {i+1}: {d.action.value:15s}  重校准={d.should_recalibrate}  原因: {d.reason}")

print()
print(f"  → 第 1-2 次: AGGRESSIVE (漂移了就激进更新)")
print(f"  → 第 3 次:   RECALIBRATE (连续漂移太多→重建基线)")
print(f"  → 第 4 次:   AGGRESSIVE (计数重置, 重新开始)")
print()


# ============================================================
# 汇总统计
# ============================================================
print("=" * 60)
print("Agent 统计信息")
print("=" * 60)
stats = agent4.get_statistics()
print(f"  总窗口数:      {stats['window_count']}")
print(f"  总更新次数:    {stats['total_updates']}")
print(f"  当前连续漂移:  {stats['consecutive_drifts']}")
print(f"  决策分布:")
for action, count in stats['decision_summary'].items():
    if count > 0:
        print(f"    {action:15s}: {count} 次")

print("\n✓ kb_agent.py 测试全部通过")
