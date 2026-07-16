"""DRIP-Reactive 与可选 TopicDynamics 分支的唯一参数表。

这里只保留论文当前使用的机制：MEF/HSU、bounded cold probe、
长期写入约束，以及由显式 drift evidence 驱动的 soft topic transition。
"""

from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class DRIPConfig:
    """Evidence-only DRIP 及其 TopicDynamics 扩展的共享配置。"""

    # MEF / HSU 的窗口存活率。
    demand_decay: float = 0.92
    serve_decay: float = 0.75
    min_stat: float = 0.01
    serve_prior: float = 0.25
    serve_topk: int = 3
    demand_ledger_cap: int = 10000

    # Miss 后的 bounded cold probe。论文默认统一使用 Top-4。
    direct_topk: int = 4
    direct_top1_bonus: float = 1.0
    direct_evidence_alpha: float = 0.5
    direct_evidence_epsilon: float = 0.05

    # 当前请求完成后的 generator/evidence attribution。输入必须由 runner 在
    # 当前请求计分后写入 ``downstream_feedback``，不能读取未来请求或历史 topic。
    # mass=0 保留旧的 similarity-only 行为；正式 feedback 实验显式设为 1。
    downstream_feedback_mass: float = 0.0
    downstream_feedback_topk: int = 4
    downstream_feedback_min_utility: float = 0.0

    # DomainAdapt：冷库分区只用于 query-conditioned routing 与 document-level
    # placement，不把整区载入 hot tier，也不预测下一 topic。候选预算为 0 时关闭。
    domain_candidate_budget: int = 0
    domain_metadata_field: str = ""
    domain_semantic_topics: int = 0
    domain_soft_memberships: int = 2
    domain_partition_metric: str = "cosine"
    domain_partition_temperature: float = 0.10
    domain_prior_rate: float = 0.25
    domain_reliability_rate: float = 0.25
    # Main method is query-adaptive rather than future-topic predictive.
    # Non-zero values are retained only for an explicit history-prior ablation.
    domain_prior_weight: float = 0.0
    domain_route_width: int = 2
    domain_retrieve_topk: int = 4

    # 粗到细的主动 evidence 预测：topic 分区把已完成窗口编码成
    # soft demand distribution；显式 drift statistic 决定何时更新 soft transition，
    # 具体文档仍只能来自因果历史。``topic_candidate_budget=0`` 时关闭该分支。
    topic_candidate_budget: int = 0
    topic_metadata_field: str = ""
    topic_semantic_topics: int = 0
    topic_soft_memberships: int = 2
    topic_partition_metric: str = "cosine"
    topic_partition_temperature: float = 0.10
    topic_drift_reference_rate: float = 0.05
    topic_drift_slack: float = 0.01
    topic_drift_threshold: float = 0.10
    topic_transition_decay: float = 0.95
    topic_document_decay: float = 0.50
    topic_min_transition_support: float = 1.0
    topic_min_forecast_confidence: float = 0.50
    topic_forecast_mass: float = 8.0
    topic_max_cache_fraction: float = 0.25
    # 新 topic layer 默认只产出可审计候选；placement 定稿前不得静默写入。
    topic_apply_forecast_to_cache: bool = False

    # Admission 与长期 replacement 约束。
    duplicate_threshold: float = 0.95
    replacement_target_rate: float = 0.25
    initial_dual_price: float = 0.25

    def __post_init__(self):
        if not 0.0 <= self.demand_decay <= 1.0:
            raise ValueError("demand_decay must be in [0, 1]")
        if not 0.0 <= self.serve_decay <= 1.0:
            raise ValueError("serve_decay must be in [0, 1]")
        if self.direct_topk <= 0:
            raise ValueError("direct_topk must be positive")
        if self.downstream_feedback_mass < 0.0:
            raise ValueError("downstream_feedback_mass must be non-negative")
        if self.downstream_feedback_topk < 1:
            raise ValueError("downstream_feedback_topk must be positive")
        if self.downstream_feedback_min_utility < 0.0:
            raise ValueError(
                "downstream_feedback_min_utility must be non-negative"
            )
        if self.domain_candidate_budget < 0:
            raise ValueError("domain_candidate_budget must be non-negative")
        if self.domain_semantic_topics < 0:
            raise ValueError("domain_semantic_topics must be non-negative")
        if self.domain_soft_memberships < 1:
            raise ValueError("domain_soft_memberships must be positive")
        if self.domain_partition_metric not in {"cosine", "euclidean"}:
            raise ValueError(
                "domain_partition_metric must be 'cosine' or 'euclidean'"
            )
        if self.domain_partition_temperature <= 0.0:
            raise ValueError("domain_partition_temperature must be positive")
        if not 0.0 <= self.domain_prior_rate <= 1.0:
            raise ValueError("domain_prior_rate must be in [0, 1]")
        if not 0.0 <= self.domain_reliability_rate <= 1.0:
            raise ValueError("domain_reliability_rate must be in [0, 1]")
        if self.domain_prior_weight < 0.0:
            raise ValueError("domain_prior_weight must be non-negative")
        if self.domain_route_width < 1:
            raise ValueError("domain_route_width must be positive")
        if self.domain_retrieve_topk < 1:
            raise ValueError("domain_retrieve_topk must be positive")
        if self.topic_candidate_budget < 0:
            raise ValueError("topic_candidate_budget must be non-negative")
        if self.topic_semantic_topics < 0:
            raise ValueError("topic_semantic_topics must be non-negative")
        if self.topic_soft_memberships < 1:
            raise ValueError("topic_soft_memberships must be positive")
        if self.topic_partition_metric not in {"cosine", "euclidean"}:
            raise ValueError(
                "topic_partition_metric must be 'cosine' or 'euclidean'"
            )
        if self.topic_partition_temperature <= 0.0:
            raise ValueError("topic_partition_temperature must be positive")
        if not 0.0 < self.topic_drift_reference_rate <= 1.0:
            raise ValueError("topic_drift_reference_rate must be in (0, 1]")
        if self.topic_drift_slack < 0.0:
            raise ValueError("topic_drift_slack must be non-negative")
        if self.topic_drift_threshold <= 0.0:
            raise ValueError("topic_drift_threshold must be positive")
        if not 0.0 <= self.topic_transition_decay <= 1.0:
            raise ValueError("topic_transition_decay must be in [0, 1]")
        if not 0.0 <= self.topic_document_decay <= 1.0:
            raise ValueError("topic_document_decay must be in [0, 1]")
        if self.topic_min_transition_support < 0.0:
            raise ValueError(
                "topic_min_transition_support must be non-negative"
            )
        if not 0.0 <= self.topic_min_forecast_confidence <= 1.0:
            raise ValueError(
                "topic_min_forecast_confidence must be in [0, 1]"
            )
        if self.topic_forecast_mass < 0.0:
            raise ValueError("topic_forecast_mass must be non-negative")
        if not 0.0 <= self.topic_max_cache_fraction <= 1.0:
            raise ValueError("topic_max_cache_fraction must be in [0, 1]")
        if not isinstance(self.topic_apply_forecast_to_cache, bool):
            raise TypeError("topic_apply_forecast_to_cache must be boolean")
        if not 0.0 <= self.replacement_target_rate <= 1.0:
            raise ValueError("replacement_target_rate must be in [0, 1]")

        topic_enabled = self.topic_candidate_budget > 0
        partition_count = int(bool(self.topic_metadata_field)) + int(
            self.topic_semantic_topics > 0
        )
        if topic_enabled and partition_count != 1:
            raise ValueError(
                "enabled TopicDynamics requires exactly one of "
                "topic_metadata_field or topic_semantic_topics"
            )
        domain_enabled = self.domain_candidate_budget > 0
        domain_partition_count = int(bool(self.domain_metadata_field)) + int(
            self.domain_semantic_topics > 0
        )
        if domain_enabled and domain_partition_count != 1:
            raise ValueError(
                "enabled DomainAdapt requires exactly one of "
                "domain_metadata_field or domain_semantic_topics"
            )
        if domain_enabled and topic_enabled:
            raise ValueError(
                "DomainAdapt and legacy TopicDynamics cannot be enabled together"
            )

    @property
    def domain_adaptation_enabled(self):
        """Return whether partition-routed DomainAdapt is enabled."""

        return self.domain_candidate_budget > 0

    @property
    def topic_dynamics_enabled(self):
        """返回 soft topic dynamics 主动预测分支是否启用。"""

        return self.topic_candidate_budget > 0

    @classmethod
    def reactive(cls, **overrides):
        """构造不启用预取的 DRIP-Reactive 配置。"""

        return cls(topic_candidate_budget=0, **overrides)

    @classmethod
    def domain_adapt(
        cls,
        *,
        candidate_budget,
        metadata_field="",
        semantic_topics=0,
        **overrides,
    ):
        """Construct continuous DomainAdapt without shift prediction."""

        return cls(
            domain_candidate_budget=int(candidate_budget),
            domain_metadata_field=str(metadata_field),
            domain_semantic_topics=int(semantic_topics),
            topic_candidate_budget=0,
            **overrides,
        )

    @classmethod
    def topic_dynamics(
        cls,
        *,
        candidate_budget,
        metadata_field="",
        semantic_topics=0,
        **overrides,
    ):
        """构造显式启用的 TopicDynamics 配置。

        ``metadata_field`` 可以复用离线建立的冷库分区，例如新闻子类别或预计算的
        semantic-page ID。若没有这类元数据，可以通过 ``semantic_topics`` 在初始化时
        对冷库 embedding 聚类一次。
        """

        return cls(
            topic_candidate_budget=int(candidate_budget),
            topic_metadata_field=str(metadata_field),
            topic_semantic_topics=int(semantic_topics),
            **overrides,
        )

    @classmethod
    def from_env(cls):
        """开放具有直接系统含义的实验旋钮。"""

        config = cls()
        updates = {}
        if "DRIP_DIRECT_TOPK" in os.environ:
            updates["direct_topk"] = int(os.environ["DRIP_DIRECT_TOPK"])
        if "DRIP_DOWNSTREAM_FEEDBACK_MASS" in os.environ:
            updates["downstream_feedback_mass"] = float(
                os.environ["DRIP_DOWNSTREAM_FEEDBACK_MASS"]
            )
        if "DRIP_DOWNSTREAM_FEEDBACK_TOPK" in os.environ:
            updates["downstream_feedback_topk"] = int(
                os.environ["DRIP_DOWNSTREAM_FEEDBACK_TOPK"]
            )
        if "DRIP_DOWNSTREAM_FEEDBACK_MIN_UTILITY" in os.environ:
            updates["downstream_feedback_min_utility"] = float(
                os.environ["DRIP_DOWNSTREAM_FEEDBACK_MIN_UTILITY"]
            )
        if "DRIP_DOMAIN_CANDIDATE_BUDGET" in os.environ:
            updates["domain_candidate_budget"] = int(
                os.environ["DRIP_DOMAIN_CANDIDATE_BUDGET"]
            )
        if "DRIP_DOMAIN_METADATA_FIELD" in os.environ:
            updates["domain_metadata_field"] = str(
                os.environ["DRIP_DOMAIN_METADATA_FIELD"]
            )
        if "DRIP_DOMAIN_SEMANTIC_TOPICS" in os.environ:
            updates["domain_semantic_topics"] = int(
                os.environ["DRIP_DOMAIN_SEMANTIC_TOPICS"]
            )
        if "DRIP_DOMAIN_PRIOR_RATE" in os.environ:
            updates["domain_prior_rate"] = float(
                os.environ["DRIP_DOMAIN_PRIOR_RATE"]
            )
        if "DRIP_DOMAIN_RELIABILITY_RATE" in os.environ:
            updates["domain_reliability_rate"] = float(
                os.environ["DRIP_DOMAIN_RELIABILITY_RATE"]
            )
        if "DRIP_DOMAIN_PRIOR_WEIGHT" in os.environ:
            updates["domain_prior_weight"] = float(
                os.environ["DRIP_DOMAIN_PRIOR_WEIGHT"]
            )
        if "DRIP_DOMAIN_ROUTE_WIDTH" in os.environ:
            updates["domain_route_width"] = int(
                os.environ["DRIP_DOMAIN_ROUTE_WIDTH"]
            )
        if "DRIP_DOMAIN_RETRIEVE_TOPK" in os.environ:
            updates["domain_retrieve_topk"] = int(
                os.environ["DRIP_DOMAIN_RETRIEVE_TOPK"]
            )
        if "DRIP_REPLACEMENT_TARGET" in os.environ:
            updates["replacement_target_rate"] = float(
                os.environ["DRIP_REPLACEMENT_TARGET"]
            )
        if "DRIP_DUPLICATE_THRESHOLD" in os.environ:
            updates["duplicate_threshold"] = float(
                os.environ["DRIP_DUPLICATE_THRESHOLD"]
            )
        if "DRIP_TOPIC_CANDIDATE_BUDGET" in os.environ:
            updates["topic_candidate_budget"] = int(
                os.environ["DRIP_TOPIC_CANDIDATE_BUDGET"]
            )
        if "DRIP_TOPIC_METADATA_FIELD" in os.environ:
            updates["topic_metadata_field"] = str(
                os.environ["DRIP_TOPIC_METADATA_FIELD"]
            )
        if "DRIP_TOPIC_SEMANTIC_TOPICS" in os.environ:
            updates["topic_semantic_topics"] = int(
                os.environ["DRIP_TOPIC_SEMANTIC_TOPICS"]
            )
        if "DRIP_TOPIC_PARTITION_METRIC" in os.environ:
            updates["topic_partition_metric"] = str(
                os.environ["DRIP_TOPIC_PARTITION_METRIC"]
            )
        if "DRIP_TOPIC_PARTITION_TEMPERATURE" in os.environ:
            updates["topic_partition_temperature"] = float(
                os.environ["DRIP_TOPIC_PARTITION_TEMPERATURE"]
            )
        if "DRIP_TOPIC_DRIFT_REFERENCE_RATE" in os.environ:
            updates["topic_drift_reference_rate"] = float(
                os.environ["DRIP_TOPIC_DRIFT_REFERENCE_RATE"]
            )
        if "DRIP_TOPIC_DRIFT_SLACK" in os.environ:
            updates["topic_drift_slack"] = float(
                os.environ["DRIP_TOPIC_DRIFT_SLACK"]
            )
        if "DRIP_TOPIC_DRIFT_THRESHOLD" in os.environ:
            updates["topic_drift_threshold"] = float(
                os.environ["DRIP_TOPIC_DRIFT_THRESHOLD"]
            )
        if "DRIP_TOPIC_TRANSITION_DECAY" in os.environ:
            updates["topic_transition_decay"] = float(
                os.environ["DRIP_TOPIC_TRANSITION_DECAY"]
            )
        if "DRIP_TOPIC_DOCUMENT_DECAY" in os.environ:
            updates["topic_document_decay"] = float(
                os.environ["DRIP_TOPIC_DOCUMENT_DECAY"]
            )
        if "DRIP_TOPIC_MIN_TRANSITION_SUPPORT" in os.environ:
            updates["topic_min_transition_support"] = float(
                os.environ["DRIP_TOPIC_MIN_TRANSITION_SUPPORT"]
            )
        if "DRIP_TOPIC_MIN_FORECAST_CONFIDENCE" in os.environ:
            updates["topic_min_forecast_confidence"] = float(
                os.environ["DRIP_TOPIC_MIN_FORECAST_CONFIDENCE"]
            )
        if "DRIP_TOPIC_FORECAST_MASS" in os.environ:
            updates["topic_forecast_mass"] = float(
                os.environ["DRIP_TOPIC_FORECAST_MASS"]
            )
        if "DRIP_TOPIC_MAX_CACHE_FRACTION" in os.environ:
            updates["topic_max_cache_fraction"] = float(
                os.environ["DRIP_TOPIC_MAX_CACHE_FRACTION"]
            )
        return replace(config, **updates) if updates else config
