"""
QARC — Query-Aligned Retrieval-augmented Knowledge Curation (our method)

Modules:
  interfaces.py     — 抽象接口 (BaseDriftDetector, BaseUpdateAgent, BaseKBCurator)
  config.py         — 所有超参数集中配置 (QARCConfig)
  drift_detector.py — Part 1: 对齐漂移检测 (正则化 FID)
  kb_agent.py       — Part 2: Agent 驱动的 KB 更新 (规则引擎)
  interest_model.py — 兴趣建模工具 (AutoKMeans + AlignmentGap)
  kb_curator.py     — Submodular KB 策展 (bootstrap + 增量替换)
  pipeline.py       — 主流水线: Bootstrap → DriftLens → Agent → 更新
"""

# ─── 接口层 ───
from .interfaces import (
    BaseDriftDetector,
    BaseUpdateAgent,
    BaseKBCurator,
    DriftResult,
    AgentDecision,
)

# ─── 配置 ───
from .config import QARCConfig

# ─── 具体实现 ───
from .detection.drift_detector import DriftLensDetector
from .decision.kb_agent import KBUpdateAgent, UpdateAction
from .curation.interest_model import (
    QueryWindowBuffer,
    InterestCluster,
    AlignmentGapResult,
    auto_kmeans,
    compute_alignment_gap,
)
from .curation.kb_curator import QARCKBCurator, DocumentPool, Document, CurationResult

# ─── 流水线 ───
from .pipeline import QARCPipeline, QARCPhase
