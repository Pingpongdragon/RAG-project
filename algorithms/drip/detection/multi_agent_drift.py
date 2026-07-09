"""DRIP 的多 agent 漂移 controller。

这个 detector 只回答一个问题：

    当前窗口的 query 是否还被共享 hot cache 覆盖？

它是 controller，不是 query router；也不判断 visible/hidden evidence。
输出的 ``severity`` 可以看成论文里的 ``rho_t``，用于调节 cache 更新激进程度。

只使用可观察信号：
  - unsupported_rate: 低于 cache-hit threshold 的 query 比例；
  - top1/topk misalignment: query 和当前 KB 的相似度下降；
  - centroid_shift: 当前 agent query centroid 相对历史 baseline 的移动。

不使用 qtype、route_hint、support labels 或 answer labels。
"""
from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from algorithms.cache.params import PARAMS as _P


GLOBAL_AGENT = "__global__"


@dataclass
class AgentDriftSignal:
    """单个 agent 或 global stream 在一个窗口里的漂移信号。"""

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
    """一个窗口的聚合 controller 输出。"""

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
    """某个 agent/global stream 的在线历史 baseline。

    ``mean`` / ``std`` 记录 alignment feature 的历史分布；
    ``centroid`` 记录 query embedding centroid 的历史方向。
    """

    def __init__(self, feature_dim):
        self.n_windows = 0
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.m2 = np.zeros(feature_dim, dtype=np.float64)
        self.centroid = None

    def update(self, features, centroid):
        """用当前窗口更新在线均值/方差和 centroid baseline。"""
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
    """共享 cache 的多 agent 在线漂移 detector。

    detector 会先给当前窗口打分，再把该窗口加入历史 baseline。这样当前窗口
    如果是异常窗口，不会在打分前污染 baseline。
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
        # warmup_windows: 前几个窗口只建立 baseline，不报告 drift。
        self.warmup_windows = int(warmup_windows)
        # min_agent_queries: agent 单独成组所需的最少 query 数。
        self.min_agent_queries = int(min_agent_queries)
        # z_threshold: alignment feature 超过历史均值多少标准差才算异常压力。
        self.z_threshold = float(z_threshold)
        # centroid_threshold: query centroid 位移超过多少才产生额外漂移压力。
        self.centroid_threshold = float(centroid_threshold)
        # topk: 计算 top-k alignment 时使用的 k。
        self.topk = int(topk)
        # agent_id_fields: 从 query dict 中依次尝试读取 agent/user id 的字段名。
        self.agent_id_fields = tuple(agent_id_fields)
        self._scopes: Dict[str, _RunningScope] = {}
        self._last_result = None

    def agent_id(self, query):
        """从 query dict 中提取 agent id；没有则归到 global stream。"""
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
        """计算当前窗口的 rho_t/severity，并更新历史 baseline。"""

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
            # agent 占比越大，它的局部漂移越应该影响全局 severity。
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
        """空窗口返回无漂移。"""
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
        """获取或创建某个 agent/global 的 running baseline。"""
        scope = self._scopes.get(agent_id)
        if scope is None:
            scope = _RunningScope(self.FEATURE_DIM)
            self._scopes[agent_id] = scope
        return scope

    def _score_scope(self, agent_id, query_embs, q_kb_sims):
        """对某个 agent/global scope 计算漂移强度。

        features = [unsupported_rate, 1-mean_top1, 1-mean_topk]。
        如果这些特征超过历史均值 + z_threshold * std，就产生 drift pressure。
        """
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
            # 指数压缩到 [0,1]，避免极端 z-score 让 severity 爆掉。
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
        """把 query-cache 相似度矩阵转成 drift feature。"""
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
        """两个 centroid 之间的 cosine distance。"""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            return 0.0
        return float(1.0 - np.clip((a @ b) / denom, -1.0, 1.0))

    @staticmethod
    def _reason(excess, centroid_excess):
        """返回最主要的 drift 原因，写入日志用于诊断。"""
        names = ["unsupported_rate", "top1_misalignment", "topk_misalignment"]
        values = list(float(v) for v in excess) + [float(centroid_excess)]
        labels = names + ["centroid_shift"]
        best = int(np.argmax(values))
        if values[best] <= 0.0:
            return "within_baseline"
        return labels[best]

    def get_state(self):
        """返回轻量状态，方便实验日志记录。"""
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
