"""Final DRIP cache policy."""

from .config import DRIPCoreConfig
from .local_ppr import PPRBridgeDebtDRIPCore


class DRIP(PPRBridgeDebtDRIPCore):
    """Final DRIP method: PPR candidates with bridge-debt admission."""

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or DRIPCoreConfig(gain_margin=1.5),
            ppr_kwargs=dict(c=0.5, L=3, R=2, K0=5, d_cap=30),
            bridge_reserve=0.15,
            bridge_margin=1.0,
            bridge_stickiness=2.0,
            echo_weight=0.75,
            echo_decay=0.99,
            serve_weight=1.0,
            write_budget_scale=1.0,
            debt_topk=3,
            direct_debt_topk=3,
            direct_debt_gain=1.4,
            debt_decay=0.96,
            debt_gain=1.8,
            debt_specificity=0.5,
        )
