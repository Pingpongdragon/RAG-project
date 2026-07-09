"""实验策略注册表。

这里故意只保留当前还需要比较的少量策略，避免旧 ablation / 旧 prototype
继续出现在实验命令里造成混乱。

每个 factory 都接收：

    (doc_pool, doc_embs, title_to_idx)

并返回一个可运行的 cache strategy 实例。
"""
from algorithms.cache.recency.lru import LRU
from algorithms.cache.recency.fifo import FIFO
from algorithms.cache.frequency.tinylfu import TinyLFU
from algorithms.cache.semantic.gptcache import GPTCacheStyle
from algorithms.cache.semantic.proximity import Proximity
from algorithms.cache.oracle.belady import Oracle
from algorithms.drip import (
    DRIP,
    DRIPNOdetector,
)
from algorithms.cache.paradigm_ref.agent_rag_cache import AgentRAGCache


def _f(cls):
    """用类名作为策略名的普通注册方式。"""
    return lambda doc_pool, doc_embs, title_to_idx: cls(
        cls.__name__, doc_pool, doc_embs, title_to_idx)


STRATEGY_FACTORIES = {
    # 经典 cache replacement baseline。
    'LRU':              _f(LRU),
    'FIFO':             _f(FIFO),
    'TinyLFU':          _f(TinyLFU),

    # 语义 cache baseline。
    'GPTCacheStyle':    _f(GPTCacheStyle),
    'Proximity':        _f(Proximity),

    # Agent-RAG Cache baseline。
    'AgentRAGCache':    _f(AgentRAGCache),

    # DRIP: 保留两个入口。
    # DRIP = 以后接 drift detector 的版本；现在 detector 还没调好，先少用。
    # DRIPNOdetector = 当前重点实验版本，不依赖 detector。
    'DRIP':             _f(DRIP),
    'DRIPNOdetector':   _f(DRIPNOdetector),

    # 上界。
    'Oracle':           _f(Oracle),
}
