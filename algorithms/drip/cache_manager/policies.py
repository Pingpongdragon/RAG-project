"""DRIP 实验策略入口。

这个文件不是算法主循环，而是把实验中可选的策略名映射到具体 core：

  - ``DRIPNOdetector``：当前主实验方法，不依赖 drift detector；
  - ``DRIP``：保留给后续接 detector 的 hidden/bridge 版本。

真正的窗口循环在 ``core.py``，direct/hidden evidence writer 在
``evidence_core.py``。
"""

from .drip_config import DRIPCoreConfig, DRIPHiddenDiagnosticConfig
from .evidence_core import EmbeddingOnlyDRIPCore, EvidenceConditionedDRIPCore


def _default_config():
    """DRIP / hidden diagnostic 使用完整参数表。"""
    return DRIPHiddenDiagnosticConfig()


def _no_detector_config():
    """当前主实验先不用 detector，只保留 demand/serve admission 信号。"""
    return DRIPCoreConfig()


class DRIPQueryHidden(EvidenceConditionedDRIPCore):
    """DRIP-QueryHidden：从可见 anchor A 补全 hidden support B。

    hidden diagnostic 的逻辑链：

        query-visible A -> evidence-conditioned B -> pair lease for A+B

    pair lease 是内部保留机制，不是公开 query 类型。
    """

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or _default_config(),
            use_bridge=True,
            use_pair_lease=True,
            use_text_encoder=True,
        )


class DRIP(DRIPQueryHidden):
    """旧 DRIP shorthand；等价于 DRIP-QueryHidden。"""


class DRIPQueryVisible(EmbeddingOnlyDRIPCore):
    """DRIP-QueryVisible：只使用 dense/direct admission。"""

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or _default_config(),
        )


class DRIPNOdetector(DRIPQueryVisible):
    """不依赖 detector 的 DRIP。

    这是当前建议优先跑的版本：使用旧 DRIP 的 demand/serve 账本，但不使用
    drift detector 去调 write cap、decay 或 admission margin。
    """

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or _no_detector_config(),
        )


class DRIPESC(EvidenceConditionedDRIPCore):
    """消融：使用 evidence-conditioned support completion，但不使用 pair lease。"""

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or _default_config(),
            use_bridge=True,
            use_pair_lease=False,
            use_text_encoder=True,
        )


class DRIPDense(DRIPQueryVisible):
    """论文别名：query-visible dense/direct 分支。"""


class DRIPESCLease(DRIP):
    """论文别名：完整 ESC + Pair Lease 分支。"""
