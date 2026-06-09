"""
ComRAG — Conversational RAG with Dynamic Memory (ACL 2025 Industry Track)
https://arxiv.org/abs/2506.21098

Modules:
  memory.py   — DynamicMemory (dual vector store) + CentroidClusterStore (centroid-based detection & routing)
  pipeline.py — Full pipeline: three-tier query routing + adaptive temperature + orchestration loop
  updater.py  — Update phase (Algorithm 2): score -> route to V_high/V_low -> centroid placement
"""
from .memory import QARecord, SearchResult, CentroidClusterStore, DynamicMemory
from .pipeline import ComRAGPipeline, compute_adaptive_temperature
from .updater import ComRAGUpdater
