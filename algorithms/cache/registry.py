"""Central registry of window-level cache policies (single source).

All experiments (motivation_1 / motivation_2 / benchmark) import
``STRATEGY_FACTORIES`` from here. Each factory takes
``(doc_pool, doc_embs, title_to_idx)`` and returns a policy instance.

Strategy code lives in family submodules:
  recency/      LRU (access-history, time-aware)
  frequency/    TinyLFU (miss-driven admission + LFU eviction)
  semantic/     GPTCacheStyle
  oracle/       Oracle (Belady upper bound)
  ours/         QueryDriven (minimal SemFlow / motivation baseline)
  drip/         DRIPCore / SupportFlow
  paradigm_ref/ Static, DocArrival, KnowledgeEdit, RandomFIFO,
                OnDemandFetch, LogDrivenArrival, MemGPTStyle

Experiment params (SF_HIT_THRESH etc.) are injected via
``algorithms.cache.params.PARAMS.update(...)`` before running.
"""
from algorithms.cache.recency.lru import LRU
from algorithms.cache.recency.fifo import FIFO
from algorithms.cache.recency.temporal import TemporalAware, RecencyTTL
from algorithms.cache.frequency.tinylfu import TinyLFU
from algorithms.cache.semantic.gptcache import GPTCacheStyle
from algorithms.cache.semantic.proximity import Proximity
from algorithms.cache.oracle.belady import Oracle
from algorithms.cache.ours.query_driven import QueryDriven, QueryDrivenLoose
from algorithms.drip.support_flow import DRIPCore, SupportFlow
from algorithms.cache.paradigm_ref.supply_side import (
    Static, DocArrival, KnowledgeEdit, RandomFIFO)
from algorithms.cache.paradigm_ref.reactive import (
    OnDemandFetch, LogDrivenArrival, MemGPTStyle)
from algorithms.cache.paradigm_ref.agent_rag_cache import AgentRAGCache


def _f(cls):
    return lambda doc_pool, doc_embs, title_to_idx: cls(
        cls.__name__, doc_pool, doc_embs, title_to_idx)


STRATEGY_FACTORIES = {
    # paradigm references / supply-side
    'Static':           _f(Static),
    'RandomFIFO':       _f(RandomFIFO),
    'DocArrival':       _f(DocArrival),
    'KnowledgeEdit':    _f(KnowledgeEdit),
    'OnDemandFetch':    _f(OnDemandFetch),
    'LogDrivenArrival': _f(LogDrivenArrival),
    'MemGPTStyle':      _f(MemGPTStyle),
    # cache replacement families
    'LRU':              _f(LRU),
    'FIFO':             _f(FIFO),
    'TemporalAware':    _f(TemporalAware),
    'RecencyTTL':       _f(RecencyTTL),
    'TinyLFU':          _f(TinyLFU),
    'GPTCacheStyle':    _f(GPTCacheStyle),
    # Proximity (Bergman 2025) — approximate cache keyed on past queries
    'Proximity':        _f(Proximity),
    # Agent-RAG ARC baseline (Lin et al. 2511.02919) — DRF + hubness, no drift
    'AgentRAGCache':    _f(AgentRAGCache),
    # ARC ablation: DRF only, no hubness centrality (paper's "ARC w/o hubness")
    'AgentRAGCache_NoHub': lambda doc_pool, doc_embs, title_to_idx: AgentRAGCache(
        'AgentRAGCache_NoHub', doc_pool, doc_embs, title_to_idx, use_hubness=False),
    # minimal SemFlow / motivation baseline
    'QueryDriven':      _f(QueryDriven),
    'QueryDrivenLoose': _f(QueryDrivenLoose),
    # DRIP core
    'SupportFlow':      _f(SupportFlow),
    'DRIPCore':         _f(DRIPCore),
    'DRIP':             _f(DRIPCore),
    # oracle
    'Oracle':           _f(Oracle),
}
