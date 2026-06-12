"""DRIP algorithm package.

This package contains the canonical DRIP components.  The current main engine is
DRIPCore, a detector-free support-cache manager.  Drift detectors live here as
optional epoch-controller utilities for a future detector-wrapped DRIP.
"""

from .interfaces import BaseDriftDetector, DriftResult
from .detection.baseline_detectors import ADWINDetector, MMDDetector, NoDetector
from .support_flow import (
    BRIDGE,
    EmbeddingIndex,
    GraphIndex,
    MULTI_DIRECT,
    QueryRouter,
    RouteDecision,
    SINGLE,
    DRIPCore,
    SupportFlow,
    SupportFlowConfig,
)

try:
    from .detection.drift_detector import DriftLensDetector
except Exception:
    DriftLensDetector = None

DRIP = DRIPCore

__all__ = [
    "BaseDriftDetector",
    "DriftResult",
    "DriftLensDetector",
    "NoDetector",
    "ADWINDetector",
    "MMDDetector",
    "DRIPCore",
    "DRIP",
    "SupportFlow",
    "SupportFlowConfig",
    "QueryRouter",
    "EmbeddingIndex",
    "GraphIndex",
    "RouteDecision",
    "SINGLE",
    "MULTI_DIRECT",
    "BRIDGE",
]
