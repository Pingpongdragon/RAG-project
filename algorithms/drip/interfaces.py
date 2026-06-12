"""Shared detector interfaces for DRIP cache policies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DriftResult:
    """Result returned by a query-cache alignment drift detector."""

    fid_score: float
    threshold: float
    is_drifted: bool


class BaseDriftDetector(ABC):
    """Detector for query-cache alignment drift.

    Detectors establish a baseline over historical query embeddings and the
    current KB embeddings, then decide whether a new query window departs from
    that baseline.
    """

    @abstractmethod
    def set_baseline(
        self,
        kb_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> bool:
        """Fit the reference query-cache alignment distribution."""

    @abstractmethod
    def calibrate_threshold(
        self,
        query_embeddings: np.ndarray,
        window_size: int,
    ) -> Optional[float]:
        """Calibrate the detector threshold from historical query windows."""

    @abstractmethod
    def detect(self, window_query_embs: np.ndarray) -> DriftResult:
        """Detect whether the current query window is drifted."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the detector has a usable baseline."""

    @property
    @abstractmethod
    def threshold(self) -> float:
        """Current detector threshold."""

    def get_state(self) -> Dict[str, Any]:
        return {"is_ready": self.is_ready, "threshold": self.threshold}
