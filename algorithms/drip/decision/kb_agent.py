"""
QARC Part 2: Agent 驱动的 KB 更新模块

DriftLens 论文中, 检测到漂移后由 Human-in-the-Loop 决定:
  - 是否需要新增类别? (新主题出现)
  - 是否需要重新训练? (分布已过时)
  - 如何更新数据集? (增删/替换)

本模块用 Agent 替代 Human, 自动做出以上决策:
  1. 观察: DriftResult (漂移信号) + AlignmentGap (KB 对齐度) + 历史趋势
  2. 推理: 规则引擎 (可扩展为 LLM-based Agent)
  3. 决策: 无操作 / 轻度更新 / 激进更新 / 重校准
  4. 执行: 调用 QARCKBCurator 做 submodular KB 增删

=== Agent 决策规则 ===

Warmup 期 (前 N 个窗口):
  始终积极更新 — KB 从冷启动快速收敛到用户兴趣
  最后一个 warmup 窗口后重校准阈值 (类似旧版 Explore → Exploit 转换)

Rule 1 (无操作): 未漂移 && Gap 正常 → 保持不动
Rule 2 (轻度更新): 未漂移 && Gap 异常偏高 → 适度替换 KB
Rule 3 (激进更新): 漂移 → 大幅替换 KB (用户兴趣明显变化)
Rule 4 (重校准): 连续多次漂移 → 更新 Offline 基线 + 大幅替换

=== Gap 阈值: 自适应 EMA + k*MAD ===
不使用固定阈值, 而是跟踪 Gap 的指数移动平均和偏差:
  gap_threshold = EMA(Gap) + k * MAD(Gap)
当 Gap 超出 "正常范围" 时触发轻度更新
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from algorithms.qarc.interfaces import BaseDriftDetector, BaseUpdateAgent, DriftResult, AgentDecision as _BaseDecision
from algorithms.qarc.detection.drift_detector import DriftLensDetector
from algorithms.qarc.curation.interest_model import AlignmentGapResult
from algorithms.qarc.curation.kb_curator import QARCKBCurator, CurationResult

logger = logging.getLogger(__name__)


# ─── 数据结构 ───

class UpdateAction(Enum):
    """Agent 可执行的更新动作"""
    NO_OP = "no_op"                      # 不动
    MILD_UPDATE = "mild_update"          # 轻度替换 (Gap 偏高)
    AGGRESSIVE_UPDATE = "aggressive"     # 激进替换 (漂移/warmup)
    RECALIBRATE = "recalibrate"          # 重校准基线 + 激进替换


@dataclass
class AgentDecision:
    """Agent 的更新决策。

    Attributes:
        action:             更新动作类型
        lambda_max:         替换上限 (占 KB 大小的比例)
        should_recalibrate: 是否需要重新校准 DriftLens 基线和阈值
        reason:             可读的决策原因 (日志用)
    """
    action: UpdateAction
    lambda_max: float
    should_recalibrate: bool
    reason: str


# ─── Agent ───

class KBUpdateAgent(BaseUpdateAgent):
    """Agent-in-the-Loop KB 更新决策器。

    替代 DriftLens 的 Human-in-the-Loop:
      Human 看到漂移报告后手动决定更新策略
      Agent 接收漂移检测 + KB 对齐状态后自动决策

    三个观察信号:
      1. drift_result.is_drifted — DriftLens 分布是否漂移
      2. gap_result.gap — AlignmentGap 对齐度差距
      3. 历史趋势 — 连续漂移次数, Gap 的 EMA/MAD 走势
    """

    @property
    def warmup_windows(self) -> int:
        return self._warmup_windows

    @property
    def warmup_windows(self) -> int:
        return self._warmup_windows

    def __init__(
        self,
        # Warmup 参数
        warmup_windows: int = 3,
        # Gap 自适应阈值 (EMA + k*MAD)
        gap_ema_beta: float = 0.85,
        gap_k: float = 1.5,
        # 替换比例
        lambda_mild: float = 0.2,
        lambda_aggressive: float = 0.5,
        # 多样性系数

        # 重校准触发
        recalibrate_after_n_drifts: int = 3,
        # 冷却
        cooldown_windows: int = 2,
    ):
        """
        Args:
            warmup_windows:  前 N 个窗口始终积极更新 (类似旧版 Explore 阶段)
            gap_ema_beta:    Gap EMA 平滑因子 (越大越平滑)
            gap_k:           Gap MAD 乘数 (越大阈值越宽松)
            lambda_mild:     轻度更新替换上限 (20%)
            lambda_aggressive: 激进更新替换上限 (50%)

            recalibrate_after_n_drifts: 连续漂移 N 次后重校准
            cooldown_windows: 更新后冷却窗口数
        """
        self._warmup_windows = warmup_windows
        self.gap_ema_beta = gap_ema_beta
        self.gap_k = gap_k
        self.lambda_mild = lambda_mild
        self.lambda_aggressive = lambda_aggressive

        self.recalibrate_after_n_drifts = recalibrate_after_n_drifts
        self.cooldown_windows = cooldown_windows

        # 内部状态
        self._window_count = 0
        self._consecutive_drifts = 0
        self._cooldown_remaining = 0
        self._total_updates = 0

        # Gap 的 EMA 跟踪 (自适应阈值)
        self._gap_ema: Optional[float] = None
        self._gap_mad: Optional[float] = None
        self._gap_history: List[float] = []
        self._decision_history: List[AgentDecision] = []

    def _update_gap_stats(self, gap: float):
        """更新 Gap 的 EMA 和 MAD (指数移动平均 + 平均绝对偏差)"""
        self._gap_history.append(gap)
        if self._gap_ema is None:
            self._gap_ema = gap
            self._gap_mad = 0.0
        else:
            self._gap_ema = (
                self.gap_ema_beta * self._gap_ema
                + (1 - self.gap_ema_beta) * gap
            )
            self._gap_mad = (
                self.gap_ema_beta * self._gap_mad
                + (1 - self.gap_ema_beta) * abs(gap - self._gap_ema)
            )

    @property
    def gap_threshold(self) -> float:
        """自适应 Gap 阈值: EMA + k * MAD"""
        if self._gap_ema is None:
            return float('inf')
        return self._gap_ema + self.gap_k * max(self._gap_mad or 0, 0.01)

    # ─── 核心决策 ───

    def decide(
        self,
        drift_result: DriftResult,
        gap_result: AlignmentGapResult,
    ) -> AgentDecision:
        """核心决策: 观察 → 推理 → 输出更新策略。

        Args:
            drift_result: DriftLens 漂移检测结果 (Part 1 输出)
            gap_result:   AlignmentGap 计算结果

        Returns:
            AgentDecision 更新决策
        """
        gap = gap_result.gap
        drifted = drift_result.is_drifted

        self._window_count += 1
        self._update_gap_stats(gap)

        # 更新连续漂移计数
        if drifted:
            self._consecutive_drifts += 1
        else:
            self._consecutive_drifts = 0

        # ─── Warmup: 前 N 个窗口始终积极更新 ───
        # 类似旧版 Explore 阶段: 快速收敛 KB 到用户兴趣
        # 最后一个 warmup 窗口后重校准阈值 (类似 Explore→Exploit 转换)
        if self._window_count <= self._warmup_windows:
            is_last_warmup = (self._window_count == self.warmup_windows)
            self._total_updates += 1
            decision = AgentDecision(
                action=UpdateAction.AGGRESSIVE_UPDATE,
                lambda_max=self.lambda_aggressive,
                should_recalibrate=is_last_warmup,
                reason=f"Warmup {self._window_count}/{self.warmup_windows}"
                       + (" + 重校准" if is_last_warmup else ""),
            )
            self._decision_history.append(decision)
            logger.info(f"Agent: {decision.action.value} — {decision.reason}")
            return decision

        # ─── 冷却期: 无操作 ───
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            decision = AgentDecision(
                action=UpdateAction.NO_OP,
                lambda_max=0.0,
                should_recalibrate=False,
                reason=f"冷却中 (剩余 {self._cooldown_remaining + 1} 窗口)",
            )
            self._decision_history.append(decision)
            return decision

        # ─── Rule 4: 连续漂移 → 重校准基线 ───
        if self._consecutive_drifts >= self.recalibrate_after_n_drifts:
            self._consecutive_drifts = 0
            self._cooldown_remaining = self.cooldown_windows
            self._total_updates += 1
            decision = AgentDecision(
                action=UpdateAction.RECALIBRATE,
                lambda_max=self.lambda_aggressive,
                should_recalibrate=True,
                reason=(
                    f"连续 {self.recalibrate_after_n_drifts} 次漂移, "
                    f"Offline 基线已过时 → 重校准"
                ),
            )
            self._decision_history.append(decision)
            logger.info(f"Agent: {decision.action.value} — {decision.reason}")
            return decision

        # ─── Rule 3: 漂移 → 激进更新 ───
        if drifted:
            self._cooldown_remaining = self.cooldown_windows
            self._total_updates += 1
            decision = AgentDecision(
                action=UpdateAction.AGGRESSIVE_UPDATE,
                lambda_max=self.lambda_aggressive,
                should_recalibrate=False,
                reason=(
                    f"漂移 (FID={drift_result.fid_score:.4f} "
                    f"> {drift_result.threshold:.4f})"
                ),
            )
            self._decision_history.append(decision)
            logger.info(f"Agent: {decision.action.value} — {decision.reason}")
            return decision

        # ─── Rule 2: Gap 异常偏高 → 轻度更新 ───
        if gap > self.gap_threshold:
            self._cooldown_remaining = self.cooldown_windows
            self._total_updates += 1
            decision = AgentDecision(
                action=UpdateAction.MILD_UPDATE,
                lambda_max=self.lambda_mild,
                should_recalibrate=False,
                reason=(
                    f"Gap 偏高 ({gap:.4f} > "
                    f"阈值 {self.gap_threshold:.4f})"
                ),
            )
            self._decision_history.append(decision)
            logger.info(f"Agent: {decision.action.value} — {decision.reason}")
            return decision

        # ─── Rule 1: 正常 → 无操作 ───
        decision = AgentDecision(
            action=UpdateAction.NO_OP,
            lambda_max=0.0,
            should_recalibrate=False,
            reason=f"正常 (FID={drift_result.fid_score:.4f}, Gap={gap:.4f})",
        )
        self._decision_history.append(decision)
        return decision

    # ─── 执行 ───

    def execute(
        self,
        decision: AgentDecision,
        curator: QARCKBCurator,
        centroids: np.ndarray,
        weights: np.ndarray,
        detector: DriftLensDetector,
        query_history: Optional[np.ndarray] = None,
        window_size: int = 8,
    ) -> Optional[CurationResult]:
        """执行 Agent 决策。

        1. 调用 QARCKBCurator.recurate() 做 submodular KB 增删
        2. 重设 DriftLens 基线 (KB 变了, 基线需更新)
        3. 如需重校准, 重新跑 random_sampling_threshold_estimation

        Args:
            decision:        Agent 决策
            curator:         KB 管理器
            centroids:       兴趣聚类中心 (auto_kmeans 输出)
            weights:         兴趣权重
            detector:        DriftLens 检测器
            query_history:   Query 历史 embedding (对齐基线重建用)
            window_size:     窗口大小 (重校准用)

        Returns:
            CurationResult, 或 None (如果是 NO_OP)
        """
        if decision.action == UpdateAction.NO_OP:
            return None

        # Step 1: submodular KB 更新
        curation = curator.recurate(
            centroids=centroids,
            weights=weights,
            lambda_max=decision.lambda_max,

        )

        # Step 2+3: 重设 DriftLens 对齐基线 (KB 改变 → 对齐特征定义变了)
        if query_history is not None and len(query_history) >= window_size:
            kb_embs = curator.get_kb_embeddings()
            if kb_embs.shape[0] > 0:
                ok = detector.set_baseline(kb_embs, query_history)
                if ok and decision.should_recalibrate:
                    detector.calibrate_threshold(query_history, window_size)

        logger.info(
            f"Agent 执行: {decision.action.value} — "
            f"+{len(curation.added_ids)} / -{len(curation.removed_ids)}, "
            f"原因: {decision.reason}"
        )
        return curation

    # ─── 统计 ───

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_updates": self._total_updates,
            "window_count": self._window_count,
            "consecutive_drifts": self._consecutive_drifts,
            "cooldown_remaining": self._cooldown_remaining,
            "gap_ema": self._gap_ema,
            "gap_mad": self._gap_mad,
            "gap_threshold": self.gap_threshold,
            "gap_history": list(self._gap_history),
            "decision_summary": {
                action.value: sum(
                    1 for d in self._decision_history if d.action == action
                )
                for action in UpdateAction
            },
        }
