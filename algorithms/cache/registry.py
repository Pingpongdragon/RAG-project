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
from algorithms.cache.adaptive.arc import ClassicalARC
from algorithms.cache.semantic.gptcache import GPTCacheStyle
from algorithms.cache.semantic.proximity import Proximity
from algorithms.cache.oracle.belady import Oracle
from algorithms.drip import DRIP
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
    'ClassicalARC':     _f(ClassicalARC),

    # 语义 cache baseline。
    'GPTCacheStyle':    _f(GPTCacheStyle),
    'Proximity':        _f(Proximity),

    # Agent-RAG Cache baseline。
    'AgentRAGCache':    _f(AgentRAGCache),

    # 唯一论文主方法：MEF/HSU + Fixed-Share + primal-dual admission。
    'DRIP':             _f(DRIP),

    # 上界。
    'Oracle':           _f(Oracle),
}
