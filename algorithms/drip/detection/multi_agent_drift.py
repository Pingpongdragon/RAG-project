"""Multi-agent drift controller for DRIP cache management.

The detector is deliberately a controller, not a query router.  It observes
whether each agent's current query window is still aligned with the shared hot
cache, then returns a drift severity used to tune cache update aggressiveness.

Signals are derived only from query embeddings and current cache similarities:

  - unsupported_rate: fraction of queries below the cache-hit threshold;
  - misalignment: one minus top-1 / top-k cache similarity;
  - centroid_shift: movement of the agent's query centroid from its baseline.

No qtype, route_hint, support labels, or answer labels are used.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from algorithms.cache.params import PARAMS as _P


GLOBAL_AGENT = "__global__"


@dataclass
class AgentDriftSignal:
    """Drift signal for one global or per-agent window."""

    agent_id: str
    n_queries: int
    severity: float
    drifted: bool
    unsupported_rate: float
    mean_top1: float
    mean_topk: float
    centroid_shift: float
    warmup: bool
    reason: str

    def to_log(self):
        return {
            "agent_id": self.agent_id,
            "n_queries": int(self.n_queries),
            "severity": round(float(self.severity), 4),
            "drifted": bool(self.drifted),
            "unsupported_rate": round(float(self.unsupported_rate), 4),
            "mean_top1": round(float(self.mean_top1), 4),
            "mean_topk": round(float(self.mean_topk), 4),
            "centroid_shift": round(float(self.centroid_shift), 4),
            "warmup": bool(self.warmup),
            "reason": self.reason,
        }


@dataclass
class MultiAgentDriftResult:
    """Aggregated controller output for one query window."""

    severity: float
    drifted: bool
    global_signal: AgentDriftSignal
    agent_signals: Dict[str, AgentDriftSignal] = field(default_factory=dict)

    @property
    def affected_agents(self) -> List[str]:
        return sorted(
            aid for aid, sig in self.agent_signals.items()
            if sig.drifted and not sig.warmup
        )

    def to_log(self, window_idx):
        return {
            "w": int(window_idx),
            "severity": round(float(self.severity), 4),
            "drifted": bool(self.drifted),
            "affected_agents": self.affected_agents,
            "global": self.global_signal.to_log(),
            "agents": {
                aid: sig.to_log() for aid, sig in sorted(self.agent_signals.items())
            },
        }


class _RunningScope:
    """Running baseline for one agent or global stream."""

    def __init__(self, feature_dim):
        self.n_windows = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.m2 = np.zeros(feature_dim, dtype=np.float64)
        self.centroid = None

    def update(self, features, centroid):
        features = np.asarray(features, dtype=np.float64)
        self.n_windows += 1
        delta = features - self.mean
        self.mean += delta / float(self.n_windows)
        self.m2 += delta * (features - self.mean)

        centroid = np.asarray(centroid, dtype=np.float64)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            centroid = centroid / norm
        if self.centroid is None:
            self.centroid = centroid
        else:
            eta = 1.0 / float(self.n_windows)
            self.centroid = (1.0 - eta) * self.centroid + eta * centroid
            c_norm = float(np.linalg.norm(self.centroid))
            if c_norm > 0.0:
                self.centroid = self.centroid / c_norm

    @property
    def std(self):
        if self.n_windows < 2:
            return np.ones_like(self.mean) * 1e-6
        return np.sqrt(self.m2 / float(self.n_windows - 1) + 1e-6)


class MultiAgentDriftDetector:
    """Online detector for shared-cache multi-agent demand drift.

    The detector scores a current query window before adding it to the running
    baseline.  It is intended to run once per cache-maintenance window.
    """

    FEATURE_DIM = 3

    def __init__(
        self,
        warmup_windows=3,
        min_agent_queries=2,
        z_threshold=2.0,
        centroid_threshold=0.25,
        topk=3,
        agent_id_fields=("agent_id", "user_id", "agent", "user", "client_id"),
    ):
        self.warmup_windows = int(warmup_windows)
        self.min_agent_queries = int(min_agent_queries)
        self.z_threshold = float(z_threshold)
        self.centroid_threshold = float(centroid_threshold)
        self.topk = int(topk)
        self.agent_id_fields = tuple(agent_id_fields)
        self._scopes: Dict[str, _RunningScope] = {}
        self._last_result = None

    def agent_id(self, query):
        if isinstance(query, dict):
            for field in self.agent_id_fields:
                value = query.get(field)
                if value is not None and str(value):
                    return str(value)
        return GLOBAL_AGENT

    def agent_ids(self, queries: Sequence[object]) -> List[str]:
        return [self.agent_id(query) for query in queries]

    def detect(
        self,
        window_query_embs: np.ndarray,
        q_kb_sims: np.ndarray,
        queries: Sequence[object] = (),
    ) -> MultiAgentDriftResult:
        """Score the current window and update running baselines."""

        q = np.asarray(window_query_embs, dtype=np.float64)
        sims = np.asarray(q_kb_sims, dtype=np.float64)
        if q.shape[0] == 0 or sims.shape[0] == 0 or sims.shape[1] == 0:
            result = self._empty_result()
            self._last_result = result
            return result

        ids = self.agent_ids(queries) if queries else [GLOBAL_AGENT] * q.shape[0]
        if len(ids) != q.shape[0]:
            ids = [GLOBAL_AGENT] * q.shape[0]

        global_signal = self._score_scope(GLOBAL_AGENT, q, sims)
        agent_signals = {}
        for aid in sorted(set(ids)):
            idx = [i for i, x in enumerate(ids) if x == aid]
            if len(idx) < self.min_agent_queries:
                continue
            agent_signals[aid] = self._score_scope(aid, q[idx], sims[idx])

        severity = global_signal.severity
        for sig in agent_signals.values():
            if sig.warmup:
                continue
            share = sig.n_queries / max(1.0, float(q.shape[0]))
            weight = min(1.0, 0.5 + share)
            severity = max(severity, weight * sig.severity)
        severity = float(np.clip(severity, 0.0, 1.0))

        result = MultiAgentDriftResult(
            severity=severity,
            drifted=bool(severity > 0.0),
            global_signal=global_signal,
            agent_signals=agent_signals,
        )
        self._last_result = result
        return result

    def _empty_result(self):
        empty = AgentDriftSignal(
            agent_id=GLOBAL_AGENT,
            n_queries=0,
            severity=0.0,
            drifted=False,
            unsupported_rate=0.0,
            mean_top1=0.0,
            mean_topk=0.0,
            centroid_shift=0.0,
            warmup=True,
            reason="empty_window",
        )
        return MultiAgentDriftResult(
            severity=0.0,
            drifted=False,
            global_signal=empty,
            agent_signals={},
        )

    def _scope(self, agent_id):
        scope = self._scopes.get(agent_id)
        if scope is None:
            scope = _RunningScope(self.FEATURE_DIM)
            self._scopes[agent_id] = scope
        return scope

    def _score_scope(self, agent_id, query_embs, q_kb_sims):
        features, centroid, raw = self._features(query_embs, q_kb_sims)
        scope = self._scope(agent_id)
        warmup = scope.n_windows < self.warmup_windows or scope.centroid is None
        if warmup:
            severity = 0.0
            reason = "warming_baseline"
            drifted = False
        else:
            std = scope.std
            excess = np.maximum(
                0.0,
                (features - scope.mean - self.z_threshold * std)
                / np.clip(std, 1e-6, None),
            )
            centroid_shift = self._centroid_shift(centroid, scope.centroid)
            centroid_excess = max(
                0.0,
                (centroid_shift - self.centroid_threshold)
                / max(1e-6, self.centroid_threshold),
            )
            pressure = (
                0.45 * excess[0]
                + 0.35 * excess[1]
                + 0.10 * excess[2]
                + 0.10 * centroid_excess
            )
            severity = float(np.clip(1.0 - math.exp(-pressure), 0.0, 1.0))
            drifted = severity > 0.0
            reason = self._reason(excess, centroid_excess)

        centroid_shift = (
            0.0 if scope.centroid is None
            else self._centroid_shift(centroid, scope.centroid)
        )
        scope.update(features, centroid)
        return AgentDriftSignal(
            agent_id=agent_id,
            n_queries=int(query_embs.shape[0]),
            severity=float(severity),
            drifted=bool(drifted),
            unsupported_rate=float(raw["unsupported_rate"]),
            mean_top1=float(raw["mean_top1"]),
            mean_topk=float(raw["mean_topk"]),
            centroid_shift=float(centroid_shift),
            warmup=bool(warmup),
            reason=reason,
        )

    def _features(self, query_embs, q_kb_sims):
        sims = np.asarray(q_kb_sims, dtype=np.float64)
        top1 = sims.max(axis=1)
        k = min(max(1, self.topk), sims.shape[1])
        if sims.shape[1] <= k:
            topk = np.sort(sims, axis=1)[:, ::-1]
        else:
            idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            topk = np.take_along_axis(sims, idx, axis=1)
        mean_topk = topk.mean(axis=1)
        unsupported_rate = float(np.mean(top1 < _P.SF_HIT_THRESH))
        mean_top1 = float(np.mean(top1))
        mean_topk_val = float(np.mean(mean_topk))
        features = np.array(
            [
                unsupported_rate,
                1.0 - mean_top1,
                1.0 - mean_topk_val,
            ],
            dtype=np.float64,
        )
        centroid = np.mean(query_embs, axis=0)
        return features, centroid, {
            "unsupported_rate": unsupported_rate,
            "mean_top1": mean_top1,
            "mean_topk": mean_topk_val,
        }

    @staticmethod
    def _centroid_shift(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            return 0.0
        return float(1.0 - np.clip((a @ b) / denom, -1.0, 1.0))

    @staticmethod
    def _reason(excess, centroid_excess):
        names = ["unsupported_rate", "top1_misalignment", "topk_misalignment"]
        values = list(float(v) for v in excess) + [float(centroid_excess)]
        labels = names + ["centroid_shift"]
        best = int(np.argmax(values))
        if values[best] <= 0.0:
            return "within_baseline"
        return labels[best]

    def get_state(self):
        return {
            "n_scopes": len(self._scopes),
            "warmup_windows": self.warmup_windows,
            "min_agent_queries": self.min_agent_queries,
            "z_threshold": self.z_threshold,
            "centroid_threshold": self.centroid_threshold,
            "last_severity": (
                None if self._last_result is None
                else round(float(self._last_result.severity), 4)
            ),
        }
