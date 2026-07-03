"""DRIP algorithm package.

This package contains the canonical DRIP components.  The current main engine is
DRIPCore, a support-cache manager with an optional multi-agent drift controller.
The controller tunes update aggressiveness; it does not route query evidence.
"""

from .interfaces import BaseDriftDetector, DriftResult
from .detection.baseline_detectors import ADWINDetector, MMDDetector, NoDetector
from .detection.multi_agent_drift import (
    AgentDriftSignal,
    MultiAgentDriftDetector,
    MultiAgentDriftResult,
)
from .cache_manager import (
    DRIP,
    DRIPDense,
    DRIPESC,
    DRIPESCLease,
    DRIPQueryHidden,
    DRIPQueryVisible,
    EmbeddingIndex,
    GraphIndex,
    QUERY_HIDDEN,
    QUERY_VISIBLE,
    QueryRouter,
    RouteDecision,
    DRIPCore,
    DRIPCoreConfig,
)

try:
    from .detection.drift_detector import DriftLensDetector
except Exception:
    DriftLensDetector = None

__all__ = [
    "BaseDriftDetector",
    "DriftResult",
    "DriftLensDetector",
    "NoDetector",
    "ADWINDetector",
    "MMDDetector",
    "AgentDriftSignal",
    "MultiAgentDriftDetector",
    "MultiAgentDriftResult",
    "DRIPCore",
    "DRIPCoreConfig",
    "DRIP",
    "DRIPDense",
    "DRIPESC",
    "DRIPESCLease",
    "DRIPQueryHidden",
    "DRIPQueryVisible",
    "QueryRouter",
    "EmbeddingIndex",
    "GraphIndex",
    "RouteDecision",
    "QUERY_VISIBLE",
    "QUERY_HIDDEN",
]
