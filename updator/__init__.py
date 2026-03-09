"""
updator — Unified package for all RAG update/curation methods.

Sub-packages:
  comrag/  — ComRAG: Dynamic Memory + centroid-based detection & routing + update loop
  erase/   — ERASE:  Editable Knowledge Base + three-step consistency update
  qarc/    — QARC:   Interest profiling (AutoKMeans + GMM drift detection) + KB curation lifecycle
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
    QueryWindowBuffer, InterestCluster, AlignmentGapResult,
    AdaptiveThreshold, GMMDriftDetector,
    auto_kmeans, compute_alignment_gap,
    QARCKBCurator, DocumentPool, Document, CurationResult,
    QARCPipeline, QARCPhase,
)
