"""
updator — 所有 RAG KB 更新方法的统一包

Sub-packages:
  comrag/  — ComRAG: 动态记忆 + centroid 检测 + 路由更新
  erase/   — ERASE:  可编辑知识库 + 三步一致性更新
  qarc/    — QARC:   对齐漂移检测 + Agent 决策 + 子模 KB 策展
"""

# ── ComRAG ──
from .comrag import (
    QARecord, SearchResult, CentroidClusterStore, DynamicMemory,
    ComRAGPipeline, compute_adaptive_temperature,
    ComRAGUpdater,
)

# ── ERASE ──
from .erase import (
    ERASEKnowledgeBase, FactEntry, FactHistory, RetrievalResult,
    ERASEUpdater,
)

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
