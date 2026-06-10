"""
algorithms — 所有 RAG KB 更新方法的统一包

Sub-packages:
  qarc/    — QARC:   对齐漂移检测 + Agent 决策 + 子模 KB 策展
  cache/   — 窗口级缓存策略族 (replacement baselines + ours + AgentRAGCache)

注: ComRAG / ERASE 已移除 —— 它们是知识更新 / 对话记忆范式, 不是固定预算下的
    缓存换入换出方法, 与 cache 主战场指标不可比 (避免稻草人对比)。
"""

# ── QARC (our method) ──
from .qarc import (
    # 接口
    BaseDriftDetector, BaseUpdateAgent, BaseKBCurator,
    DriftResult, AgentDecision,
    # 配置
    QARCConfig,
    # 实现
    DriftLensDetector, KBUpdateAgent, UpdateAction,
    QueryWindowBuffer, InterestCluster, AlignmentGapResult,
    auto_kmeans, compute_alignment_gap,
    QARCKBCurator, DocumentPool, Document, CurationResult,
    QARCPipeline, QARCPhase,
)
