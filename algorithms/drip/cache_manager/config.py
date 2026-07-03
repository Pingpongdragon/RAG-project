"""Configuration for the canonical DRIP cache manager."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DRIPCoreConfig:
    """DRIP cache-manager configuration.

    The defaults are design-level settings, not per-dataset tuned constants.
    DRIP separates two evidence visibility regimes:

    * query-visible evidence: the needed document is reachable from the query
      embedding, so dense/direct demand is enough;
    * query-hidden evidence: the query first exposes an anchor A, then DRIP
      completes the missing support B through evidence conditioned on A.

    Public benchmark query types are diagnostics only by default. Main runs
    should not route from ``qtype`` or ``route_hint`` unless an explicit oracle
    ablation enables ``use_oracle_route_hint``.
    """

    demand_decay: float = 0.92
    serve_decay: float = 0.75
    min_stat: float = 0.01
    serve_prior: float = 1.0
    demand_ledger_cap: int = 10000

    singlehop_slots: int = 1
    multihop_slots: int = 2
    hidden_comparison_slots: int = 4
    serve_topk: int = 1

    direct_topk: int = 8
    direct_gamma: float = 1.0
    direct_top1_bonus: float = 1.0
    bridge_direct_gamma: float = 0.2
    bridge_step1_k: int = 3
    bridge_alpha: float = 0.6
    bridge_demand_gain: float = 2.0
    bridge_max_docs: int = 20
    max_entity_degree: int = 200
    entity_degree_power: float = 0.5
    min_entity_len: int = 3
    bridge_abs_threshold: float = 0.08
    bridge_score_saturation: float = 1.0
    bridge_evidence_alpha: float = 0.45
    bridge_evidence_beta: float = 0.30
    bridge_evidence_gamma: float = 0.25
    bridge_mmr_mu: float = 0.02
    bridge_relation_floor: float = 0.05
    bridge_relation_overlap_weight: float = 0.35
    bridge_min_relation_overlap: int = 1
    # Additive bonus to the relation score when the bridge entity matches the
    # candidate B's title (typical for Wikipedia title-based bridges, e.g. 2Wiki).
    bridge_title_relation_bonus: float = 0.25
    # Gate first-hop documents A by query similarity before fanning out paths,
    # so weak/incorrect first hops do not spawn bridge candidates.
    bridge_min_firsthop_sim: float = 0.0
    bridge_max_seed_entities: int = 12
    use_oracle_route_hint: bool = False
    router_min_entities: int = 2
    router_dense_diversity: float = 0.65
    router_hidden_anchor_ratio: float = 0.8
    graph_novelty_floor: float = 0.05

    use_drift_detector: bool = True
    drift_warmup_windows: int = 3
    drift_min_agent_queries: int = 2
    drift_z_threshold: float = 2.0
    drift_centroid_threshold: float = 0.25
    drift_write_boost: float = 1.0
    drift_decay_boost: float = 0.25
    drift_margin_discount: float = 0.25

    tau_duplicate: float = 0.95
    redundancy_threshold: float = 0.85
    redundancy_penalty: float = 1.5
    gain_margin: float = 1.0
