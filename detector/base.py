"""
基础数据类和常量
"""
import numpy as np
from typing import Dict
from dataclasses import dataclass, field

LABEL_MAP = {
    "entertainment": 0,
    "stem": 1,
    "humanities": 2,
    "lifestyle": 3
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
DOMAINS = list(LABEL_MAP.keys())
NUM_CLASSES = len(DOMAINS)


@dataclass
class DetectionResult:
    step: int
    query: str

    # Individual
    predicted_domain: str
    confidence: float
    calibrated_probs: Dict[str, float]
    entropy: float

    # Global Shift
    is_global_shift: bool = False
    # B1
    jsd_score: float = 0.0
    psi_score: float = 0.0
    # B2: Per-Label FID
    fid_global: float = 0.0
    fid_per_label: Dict[str, float] = field(default_factory=dict)
    # Ours
    soft_fid_global: float = 0.0
    soft_fid_per_label: Dict[str, float] = field(default_factory=dict)
    bbse_l1: float = 0.0
    estimated_true_distribution: Dict[str, float] = field(default_factory=dict)

    query_distribution: Dict[str, float] = field(default_factory=dict)

    # Local Shift
    local_accuracy: Dict[str, float] = field(default_factory=dict)
    local_shifts: Dict[str, bool] = field(default_factory=dict)