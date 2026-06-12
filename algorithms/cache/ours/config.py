"""Configuration objects for our cache-admission policies.

Only tunable hyperparameters live here. Runtime state such as demand ledgers,
detectors, query history, and cache contents stays inside the policy classes.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QueryDrivenConfig:
    """Hyperparameters for query-demand admission."""

    demand_decay: float = 0.92
    serve_decay: float = 0.92
    min_stat: float = 0.01
    serve_prior: float = 1.0
    prefetch_topk: int = 5
    tau_admit: float = 0.95
    lambda_red: float = 1.5
    red_thresh: float = 0.85
    neigh_gamma: float = 0.4
    support_slots: int = 1
    serve_topk: int = 1
    admission_gain_margin: float = 1.0

    def as_dict(self):
        return asdict(self)
