"""Paper-facing DRIP cache policies."""

from .config import DRIPCoreConfig
from .support_completion import EmbeddingOnlyDRIPCore, EvidenceConditionedDRIPCore


def _default_config():
    return DRIPCoreConfig(
        gain_margin=1.35,
        hidden_comparison_slots=4,
        use_oracle_route_hint=True,
    )


class DRIPQueryHidden(EvidenceConditionedDRIPCore):
    """DRIP-QueryHidden: complete evidence hidden behind a visible anchor.

    This is the current DRIP method:

        query-visible A -> evidence-conditioned B -> pair lease for A+B

    The pair lease is an internal retention mechanism, not a separate public
    query class.
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
    """Current DRIP method; shorthand for DRIP-QueryHidden."""


class DRIPQueryVisible(EmbeddingOnlyDRIPCore):
    """DRIP-QueryVisible: dense/direct admission for query-visible evidence."""

    def __init__(self, name, doc_pool, doc_embs, title_to_idx, config=None):
        super().__init__(
            name,
            doc_pool,
            doc_embs,
            title_to_idx,
            config=config or _default_config(),
        )


class DRIPESC(EvidenceConditionedDRIPCore):
    """Ablation: evidence-conditioned support completion without pair lease."""

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
    """Paper alias for the query-visible dense/direct branch."""


class DRIPESCLease(DRIP):
    """Paper alias for the full ESC + Pair Lease method."""
