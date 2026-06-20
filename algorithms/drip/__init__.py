"""DRIP algorithm package.

This package contains the canonical DRIP components.  The current main engine is
DRIPCore, a detector-free support-cache manager.  Drift detectors live here as
optional epoch-controller utilities for a future detector-wrapped DRIP.
"""

from .interfaces import BaseDriftDetector, DriftResult
from .detection.baseline_detectors import ADWINDetector, MMDDetector, NoDetector
from .cache_manager import (
    BRIDGE,
    DRIP,
    EmbeddingIndex,
    GraphIndex,
    MULTI_DIRECT,
    QueryRouter,
    RouteDecision,
    SINGLE,
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
    "DRIPCore",
    "DRIPCoreConfig",
    "DRIP",
    "QueryRouter",
    "EmbeddingIndex",
    "GraphIndex",
    "RouteDecision",
    "SINGLE",
    "MULTI_DIRECT",
    "BRIDGE",
]
