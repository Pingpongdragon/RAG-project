"""
ERASE — Enhancing Retrieval Augmentation with Self-consistent Editing (Li et al., 2024)
https://arxiv.org/abs/2406.11830

Modules:
  knowledge_base.py — Editable external knowledge store with fact-level CRUD & history tracking
  updater.py        — Three-step pipeline: Retrieve -> Update (classify+rewrite) -> Add new facts
"""
from .knowledge_base import ERASEKnowledgeBase, FactEntry, FactHistory, RetrievalResult
from .updater import ERASEUpdater
