"""DRIP cache manager 参数。

当前主实验只看 ``DRIPNOdetector``，所以默认配置只保留主公式需要的参数：

  - serve / demand 账本；
  - query-visible direct evidence；
  - replacement-aware admission；
  - 少量 query slot / router 参数。

hidden evidence、GraphIndex、drift detector 的旧参数没有混在默认配置里，统一放到
``DRIPHiddenDiagnosticConfig``。这样日常调主实验时，不会被旧分支参数淹没。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DRIPCoreConfig:
    """当前主实验 ``DRIPNOdetector`` 的最小参数表。

    对应论文公式：

      D_t(d) = beta_d D_{t-1}(d) + E_t(q,d)
      S_t(d) = beta_s S_{t-1}(d) + A_t(d)
      Delta_t(c,v) = D_t(c) - gain_margin * P_t(v) - C_t
      C_t = replacement_cost * (1 + replacement_pressure_mu * phi_t)
    """

    # ========== 1. serve / demand 账本 ==========
    # demand_decay = beta_d：历史需求 D_{t-1}(d) 的保留比例。
    demand_decay: float = 0.92
    # serve_decay = beta_s：resident serve 账本 S_{t-1}(d) 的保留比例。
    serve_decay: float = 0.75
    # min_stat：账本低于该值后清理，避免 demand/serve 无限增长。
    min_stat: float = 0.01
    # serve_prior：初始 KB resident 的服务价值先验。
    serve_prior: float = 1.0
    # demand_ledger_cap：demand 账本最多保留多少候选文档。
    demand_ledger_cap: int = 10000
    # serve_topk：每个命中 query 给几个 resident 记 serve credit。
    serve_topk: int = 1

    # ========== 2. query-visible direct evidence ==========
    # direct_topk：每个 under-covered query 检索多少 dense 候选。
    direct_topk: int = 8
    # direct_gamma：E_t(q,d) 写入 D_t(d) 的整体权重。
    direct_gamma: float = 1.0
    # direct_top1_bonus：top-1 dense 候选的额外 evidence credit。
    direct_top1_bonus: float = 1.0
    # direct_evidence_alpha：ARC-style 距离折扣强度。
    direct_evidence_alpha: float = 0.5
    # direct_evidence_epsilon：防止 1 - sim 接近 0 时 evidence 爆炸。
    direct_evidence_epsilon: float = 0.05

    # ========== 3. query slot / router ==========
    # singlehop_slots：单跳 / temporal direct query 至少需要几个 support。
    singlehop_slots: int = 1
    # multihop_slots：comparison / 普通多跳 visible query 至少需要几个 support。
    multihop_slots: int = 2
    # hidden_comparison_slots：只用于 route hint 诊断；主实验会强制 visible。
    hidden_comparison_slots: int = 4
    # use_oracle_route_hint：实验里可用 qtype/route_hint 固定 support slot。
    use_oracle_route_hint: bool = True
    # router_min_entities/router_dense_diversity/router_hidden_anchor_ratio：
    # 无 qtype 时的轻量文本/first-hop 路由启发式。
    router_min_entities: int = 2
    router_dense_diversity: float = 0.65
    router_hidden_anchor_ratio: float = 0.8

    # ========== 4. replacement-aware admission ==========
    # gain_margin：candidate 需要超过 victim priority 的收益倍率。
    gain_margin: float = 1.35
    # tau_duplicate：候选和当前 KB 太相似时跳过，避免重复 evidence。
    tau_duplicate: float = 0.95
    # replacement_cost：统一替换惩罚 C_t 的基础项。
    replacement_cost: float = 0.25
    # replacement_pressure_mu：最近 replacement 越多，C_t 放大的强度。
    replacement_pressure_mu: float = 1.0
    # replacement_ema_decay：replacement pressure 的 EMA 保留比例。
    replacement_ema_decay: float = 0.75

    # ========== 5. detector 开关 ==========
    # 当前主实验默认不用 detector。其他 detector 参数见 DRIPHiddenDiagnosticConfig。
    use_drift_detector: bool = False


@dataclass(frozen=True)
class DRIPHiddenDiagnosticConfig(DRIPCoreConfig):
    """旧 hidden evidence / detector diagnostic 参数。

    这些参数不是当前主实验必须项；保留它们只是为了 appendix、diagnostic 和后续
    detector 版本。平时调 ``DRIPNOdetector`` 可以先忽略整个类。
    """

    # ========== hidden / bridge support completion ==========
    bridge_step1_k: int = 3
    bridge_direct_gamma: float = 0.2
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
    bridge_title_relation_bonus: float = 0.25
    bridge_min_firsthop_sim: float = 0.0
    bridge_max_seed_entities: int = 12
    graph_novelty_floor: float = 0.05

    # ========== drift detector/controller ==========
    use_drift_detector: bool = True
    drift_warmup_windows: int = 3
    drift_min_agent_queries: int = 2
    drift_z_threshold: float = 2.0
    drift_centroid_threshold: float = 0.25
    drift_write_boost: float = 1.0
    drift_decay_boost: float = 0.25
    drift_margin_discount: float = 0.25

    # ========== 旧 DRIPCore redundancy penalty ==========
    # 只给 base DRIPCore / diagnostic tests 使用；当前 EvidenceConditionedDRIPCore
    # 的主 writer 不用这两个参数。
    redundancy_threshold: float = 0.85
    redundancy_penalty: float = 1.5
