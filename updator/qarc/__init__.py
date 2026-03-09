"""
QARC — Query-Aligned Retrieval-augmented Knowledge Curation (our method)

Modules:
  interest_model.py — AutoKMeans interest profiling + AlignmentGap + AdaptiveThreshold
                      + GMMDriftDetector (DriftLens-inspired mixture-of-Gaussians drift detection)
  kb_curator.py     — Submodular KB curation: bootstrap + interest-weighted re-curation
  pipeline.py       — Three-phase lifecycle: Bootstrap -> Explore -> Exploit
"""
from .interest_model import (
    QueryWindowBuffer,
    InterestCluster,
    AlignmentGapResult,
    AdaptiveThreshold,
    GMMDriftDetector,
    auto_kmeans,
    compute_alignment_gap,
)
from .kb_curator import QARCKBCurator, DocumentPool, Document, CurationResult
from .pipeline import QARCPipeline, QARCPhase
