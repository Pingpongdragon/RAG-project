"""
QARC 可插拔组件的抽象接口

QARC 架构分 3 个可插拔组件, 每个都有 ABC:

    ┌────────────────────┐
    │   QARCPipeline     │  协调器 (不可替换)
    │                    │
    │  ┌──────────────┐  │    BaseDriftDetector
    │  │  detector     │◄─┼─── ├─ DriftLensDetector (FID)
    │  └──────────────┘  │    └─ 可扩展: MMD / KL / ...
    │                    │
    │  ┌──────────────┐  │    BaseUpdateAgent
    │  │  agent        │◄─┼─── ├─ KBUpdateAgent (规则引擎)
    │  └──────────────┘  │    └─ 可扩展: LLM Agent / RL / ...
    │                    │
    │  ┌──────────────┐  │    BaseKBCurator
    │  │  curator      │◄─┼─── ├─ QARCKBCurator (子模优化)
    │  └──────────────┘  │    └─ 可扩展: Bandit / RL / ...
    └────────────────────┘

扩展方式:
    1. 继承下面的 ABC
    2. 实现所有 @abstractmethod
    3. 传入 QARCPipeline(detector=MyDetector(), agent=MyAgent(), ...)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
from dataclasses import dataclass


# ═══════ 数据结构 (共用) ═══════

@dataclass
class DriftResult:
    """漂移检测结果。"""
    fid_score: float       # 距离度量 (FID / KL / MMD / ...)
    threshold: float       # 校准阈值
    is_drifted: bool       # 是否判定为漂移


# ═══════ 1. 漂移检测器接口 ═══════

class BaseDriftDetector(ABC):
    """漂移检测器抽象接口。

    职责: 判断"query 与 KB 的对齐模式是否偏离历史正常水平"

    生命周期:
        1. set_baseline(kb_embs, query_embs)     — Offline: 建立正常基线
        2. calibrate_threshold(query_embs, ws)   — Offline: 校准阈值
        3. detect(window_query_embs)             — Online:  检测漂移

    扩展示例:
        - 当前实现: DriftLensDetector (正则化 FID)
        - 可选实现: MMDDetector, KLDriftDetector, ...
    """

    @abstractmethod
    def set_baseline(
        self, kb_embeddings: np.ndarray, query_embeddings: np.ndarray,
    ) -> bool:
        """建立"正常对齐"的基线分布。

        Args:
            kb_embeddings:    (n_kb, d) 当前 KB embedding
            query_embeddings: (n_q, d) 历史 query embedding

        Returns:
            是否成功
        """
        ...

    @abstractmethod
    def calibrate_threshold(
        self, query_embeddings: np.ndarray, window_size: int,
    ) -> Optional[float]:
        """校准漂移阈值。

        Args:
            query_embeddings: (n_q, d) 历史 query embedding
            window_size:      窗口大小

        Returns:
            校准的阈值, 或 None
        """
        ...

    @abstractmethod
    def detect(self, window_query_embs: np.ndarray) -> DriftResult:
        """检测当前窗口是否发生对齐漂移。

        Args:
            window_query_embs: (w, d) 当前窗口 query embedding

        Returns:
            DriftResult
        """
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """基线是否已建立、可以做检测。"""
        ...

    @property
    @abstractmethod
    def threshold(self) -> float:
        """当前阈值 (未校准时返回 inf)。"""
        ...

    def get_state(self) -> Dict[str, Any]:
        """返回内部状态 (可选覆写, 用于日志/调试)。"""
        return {"is_ready": self.is_ready, "threshold": self.threshold}


# ═══════ 2. 更新决策 Agent 接口 ═══════

@dataclass
class AgentDecision:
    """Agent 的更新决策。"""

    class Action:
        NO_OP = "no_op"
        MILD_UPDATE = "mild_update"
        AGGRESSIVE_UPDATE = "aggressive"
        RECALIBRATE = "recalibrate"

    action: str              # Action.NO_OP / MILD_UPDATE / ...
    lambda_max: float        # 替换上限 (占 KB 比例)
    eta: float               # 多样性系数
    should_recalibrate: bool # 是否需要重新校准检测器基线
    reason: str              # 可读的决策原因


class BaseUpdateAgent(ABC):
    """KB 更新决策 Agent 抽象接口。

    职责: 根据漂移信号和对齐度, 决定"是否更新 KB、如何更新"

    扩展示例:
        - 当前实现: KBUpdateAgent (4 条规则引擎)
        - 可选实现: LLMUpdateAgent (LLM 推理), RLUpdateAgent (强化学习)
    """

    @abstractmethod
    def decide(
        self, drift_result: DriftResult, gap_result: Any,
    ) -> AgentDecision:
        """核心决策: 观察 → 推理 → 输出更新策略。

        Args:
            drift_result: 漂移检测结果
            gap_result:   对齐度差距 (AlignmentGapResult)

        Returns:
            AgentDecision
        """
        ...

    @property
    @abstractmethod
    def warmup_windows(self) -> int:
        """Warmup 窗口数 (前 N 个窗口始终积极更新)。"""
        ...

    def get_statistics(self) -> Dict[str, Any]:
        """返回累积统计 (可选覆写)。"""
        return {}


# ═══════ 3. KB 策展器接口 ═══════

class BaseKBCurator(ABC):
    """KB 策展器抽象接口。

    职责: 从文档池中选择最优文档子集作为 KB

    扩展示例:
        - 当前实现: QARCKBCurator (子模优化 + 增量替换)
        - 可选实现: BanditCurator, RLCurator, ...
    """

    @abstractmethod
    def bootstrap_diversity(self) -> None:
        """冷启动: 多样性最大化初始化 KB。"""
        ...

    @abstractmethod
    def bootstrap_from_queries(
        self,
        query_embeddings: np.ndarray,
        centroids: np.ndarray,
        weights: np.ndarray,
        eta: float,
    ) -> None:
        """热启动: 基于历史查询初始化 KB。"""
        ...

    @abstractmethod
    def recurate(
        self,
        centroids: np.ndarray,
        weights: np.ndarray,
        lambda_max: float,
        eta: float,
    ) -> Any:
        """重新策展: 根据最新兴趣更新 KB。

        Returns:
            CurationResult (或子类)
        """
        ...

    @abstractmethod
    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 10,
    ) -> List[Tuple[Any, float]]:
        """从当前 KB 检索与查询最相似的文档。

        Returns:
            [(Document, similarity)] 按相似度降序
        """
        ...

    @abstractmethod
    def get_kb_embeddings(self) -> np.ndarray:
        """返回当前 KB 所有文档的 (n, d) embedding 矩阵。"""
        ...

    @property
    @abstractmethod
    def kb_size(self) -> int:
        """当前 KB 文档数。"""
        ...

    def get_kb_docs_list(self) -> list:
        """返回当前 KB 文档列表 (可选覆写)。"""
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """返回累积统计 (可选覆写)。"""
        return {}
