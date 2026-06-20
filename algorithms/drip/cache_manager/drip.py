"""Final DRIP cache policy."""

from .config import DRIPCoreConfig
from .local_ppr import PPRSplitAdmissionDRIPCore


class DRIP(PPRSplitAdmissionDRIPCore):
    """Final DRIP method: local-PPR evidence with split direct/bridge admission."""

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or DRIPCoreConfig(gain_margin=1.5),
            ppr_kwargs=dict(c=0.5, L=3, R=2, K0=5, d_cap=30),
            bridge_reserve=0.65,
            bridge_margin=0.85,
            bridge_stickiness=0.0,
            echo_weight=0.75,
            echo_decay=0.99,
            serve_weight=1.0,
            write_budget_scale=0.8,
        )
