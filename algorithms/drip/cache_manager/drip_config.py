"""DRIP cache manager 配置。

这个配置仍然是 active 的，因为 hidden evidence 分支需要它：

  - ``QueryRouter`` 用这里的 route 参数判断 query-visible / query-hidden；
  - ``DRIPCore`` 用 direct / bridge 参数分别更新 visible 和 hidden evidence；
  - ``DRIPNOdetector`` 使用这里的 direct evidence 和 replacement 参数。

当前主实验方法是 ``DRIPNOdetector``：先不依赖 drift detector，只验证
direct evidence + serve/demand + replacement control。hidden 分支不删除，作为
bridge / query-hidden diagnostic 和后续 appendix 实验保留。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DRIPCoreConfig:
    """旧 DRIPCore + hidden diagnostic 的核心参数。

    参数分成三类：
      1. demand/serve 账本：旧 DRIPCore 的 ``D_t(d)`` 和 ``Serve_t(d)``；
      2. query router：判断 evidence 是 visible 还是 hidden；
      3. bridge completion：从 visible anchor A 补全 hidden support B。

    数据集里的 ``qtype`` / ``route_hint`` 默认只是诊断标签；只有显式打开
    ``use_oracle_route_hint`` 时，router 才会用这些标签固定 visible/hidden 分支。
    """

    # ========== 1. 旧 DRIPCore 的 demand / serve 账本 ==========
    # demand_decay: 历史需求 D_{t-1}(d) 的保留比例。
    demand_decay: float = 0.92
    # serve_decay: resident serve 账本的衰减比例。
    serve_decay: float = 0.75
    # min_stat: 账本低于该值后清理，避免无限增长。
    min_stat: float = 0.01
    # serve_prior: 初始 KB resident 的服务价值先验。
    serve_prior: float = 1.0
    # demand_ledger_cap: demand 账本最多保留多少候选文档。
    demand_ledger_cap: int = 10000

    # ========== 2. query 需要的 support slot 数 ==========
    # singlehop_slots: 单跳 / temporal direct query 至少需要几个 support。
    singlehop_slots: int = 1
    # multihop_slots: comparison / 普通多跳 visible query 至少需要几个 support。
    multihop_slots: int = 2
    # hidden_comparison_slots: bridge-comparison 这类 hidden query 的 support 预算。
    hidden_comparison_slots: int = 4
    # serve_topk: 旧 serve 账本里每个 query 给多少 resident 记服务 credit。
    serve_topk: int = 1

    # ========== 3. query-visible direct evidence credit ==========
    # direct_topk: visible/direct 分支每个 under-covered query 检索多少 dense 候选。
    direct_topk: int = 8
    # direct_gamma: dense similarity 写入 D_t(d) 的主权重。
    direct_gamma: float = 1.0
    # direct_top1_bonus: top-1 dense 候选的额外 credit。
    direct_top1_bonus: float = 1.0
    # direct_evidence_alpha: ARC-style 距离折扣强度；越大越偏向高相似候选。
    direct_evidence_alpha: float = 0.5
    # direct_evidence_epsilon: 防止 1 - sim 接近 0 时 evidence 爆炸。
    direct_evidence_epsilon: float = 0.05
    # bridge_direct_gamma: hidden 分支里 first-hop anchor A 的 dense credit 权重。
    bridge_direct_gamma: float = 0.2

    # ========== 4. query-hidden / bridge support completion ==========
    # bridge_step1_k: 先从 query 找多少个 visible anchor A。
    bridge_step1_k: int = 3
    # bridge_alpha: hidden support B 的 graph/evidence score 混合强度。
    bridge_alpha: float = 0.6
    # bridge_demand_gain: hidden support 候选写入 demand 的整体增益。
    bridge_demand_gain: float = 2.0
    # bridge_max_docs: 每个 hidden query 最多保留多少个 bridge candidates。
    bridge_max_docs: int = 20
    # max_entity_degree: 实体邻居过多时截断，防止泛化实体支配候选。
    max_entity_degree: int = 200
    # entity_degree_power: 对高频实体的 degree penalty 强度。
    entity_degree_power: float = 0.5
    # min_entity_len: 过短实体字符串过滤阈值。
    min_entity_len: int = 3
    # bridge_abs_threshold: bridge candidate 的最低绝对分数。
    bridge_abs_threshold: float = 0.08
    # bridge_score_saturation: bridge score 饱和归一化常数。
    bridge_score_saturation: float = 1.0
    # bridge_evidence_alpha/beta/gamma: hidden B 评分的证据项权重。
    bridge_evidence_alpha: float = 0.45
    bridge_evidence_beta: float = 0.30
    bridge_evidence_gamma: float = 0.25
    # bridge_mmr_mu: bridge candidates 的轻量多样性惩罚。
    bridge_mmr_mu: float = 0.02
    # bridge_relation_floor: relation score 的最低保留门槛。
    bridge_relation_floor: float = 0.05
    # bridge_relation_overlap_weight: anchor A 与 candidate B 的实体重合权重。
    bridge_relation_overlap_weight: float = 0.35
    # bridge_min_relation_overlap: 至少共享多少关系/实体线索才进入 hidden 候选。
    bridge_min_relation_overlap: int = 1
    # bridge_title_relation_bonus: bridge entity 命中文档标题时的加分。
    bridge_title_relation_bonus: float = 0.25
    # bridge_min_firsthop_sim: first-hop anchor A 的最低 query 相似度。
    bridge_min_firsthop_sim: float = 0.0
    # bridge_max_seed_entities: 每个 query 最多使用多少个 seed entity 扩展。
    bridge_max_seed_entities: int = 12

    # ========== 5. route / evidence visibility ==========
    # use_oracle_route_hint: True 时使用 qtype/route_hint 固定 visible/hidden 分支。
    use_oracle_route_hint: bool = False
    # router_min_entities: 文本里粗略实体数达到该阈值时视为可能多跳。
    router_min_entities: int = 2
    # router_dense_diversity: first-hop entity set 足够多样时判为 visible dense。
    router_dense_diversity: float = 0.65
    # router_hidden_anchor_ratio: 有 anchor 但 support 不全时判 hidden 的相似度比例。
    router_hidden_anchor_ratio: float = 0.8
    # graph_novelty_floor: hidden graph candidate 相对当前 KB 的最低新颖性。
    graph_novelty_floor: float = 0.05

    # ========== 6. 旧 DRIPCore drift controller ==========
    # use_drift_detector: 旧 DRIPCore 是否启用 multi-agent drift detector。
    use_drift_detector: bool = True
    # drift_warmup_windows: detector 前几个窗口只建 baseline。
    drift_warmup_windows: int = 3
    # drift_min_agent_queries: 每个 agent 至少多少 query 才单独估计 drift。
    drift_min_agent_queries: int = 2
    # drift_z_threshold: alignment 特征超过历史多少标准差触发 drift pressure。
    drift_z_threshold: float = 2.0
    # drift_centroid_threshold: query centroid shift 超过多少触发 drift pressure。
    drift_centroid_threshold: float = 0.25
    # drift_write_boost: 旧 DRIPCore 在 drift 下放宽写入预算的系数。
    drift_write_boost: float = 1.0
    # drift_decay_boost: 旧 DRIPCore 在 drift 下加快 demand 衰减的系数。
    drift_decay_boost: float = 0.25
    # drift_margin_discount: 旧 DRIPCore 在 drift 下减少 gain margin 的系数。
    drift_margin_discount: float = 0.25

    # ========== 7. 旧 DRIPCore admission / redundancy ==========
    # tau_duplicate: 候选和当前 KB 过于相似时跳过。
    tau_duplicate: float = 0.95
    # redundancy_threshold: resident 相互相似超过该值时认为冗余。
    redundancy_threshold: float = 0.85
    # redundancy_penalty: 冗余 resident 的保留优先级惩罚。
    redundancy_penalty: float = 1.5
    # gain_margin: 旧 DRIPCore 写入时需要超过 victim priority 的最小收益。
    gain_margin: float = 1.0
    # replacement_cost: 统一替换惩罚 C_t 的基础项。
    replacement_cost: float = 0.25
    # replacement_pressure_mu: 最近 replacement 越多，C_t 放大的强度。
    replacement_pressure_mu: float = 1.0
    # replacement_ema_decay: replacement pressure 的 EMA 保留比例。
    replacement_ema_decay: float = 0.75
