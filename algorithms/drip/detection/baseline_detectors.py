"""Drift-detector baselines (same BaseDriftDetector interface as DriftLensDetector).

These exist so the ablation "what does alignment-feature FID buy us?" has real
comparators rather than only the FID detector. All three are drop-in: the
DRIP/DRIP pipeline can swap DriftLensDetector for any of these without other
changes (same set_baseline / calibrate_threshold / detect -> DriftResult API).

Family coverage (one representative per family, newest/most-standard):
  - NoDetector       : ablation floor. Never fires drift. Isolates the value of
                       *any* detection vs none.
  - ADWINDetector    : scalar change-point family (Bifet & Gavalda, classic
                       adaptive-windowing). Runs on a scalar alignment signal
                       (mean query-KB similarity per query) — the natural way a
                       streaming change-point detector would be applied here.
  - MMDDetector      : distribution-distance family. Two-sample MMD on the
                       *raw query embeddings* (NOT alignment features). This is
                       the key contrast for our claim that alignment-feature
                       FID beats raw-embedding distance.
"""
from typing import Optional, List, Dict, Any
import numpy as np

from algorithms.drip.interfaces import BaseDriftDetector, DriftResult


# ════════════════════════════════════════════════════════════
# NoDetector — ablation floor
# ════════════════════════════════════════════════════════════

class NoDetector(BaseDriftDetector):
    """Never reports drift. Ablation floor: detection disabled."""

    def __init__(self, **kwargs):
        self._ready = False

    def set_baseline(self, kb_embeddings, query_embeddings) -> bool:
        self._ready = True
        return True

    def calibrate_threshold(self, query_embeddings, window_size) -> Optional[float]:
        return float("inf")

    def detect(self, window_query_embs: np.ndarray) -> DriftResult:
        return DriftResult(fid_score=0.0, threshold=float("inf"), is_drifted=False)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def threshold(self) -> float:
        return float("inf")


# ════════════════════════════════════════════════════════════
# ADWIN — scalar change-point on the alignment signal
# ════════════════════════════════════════════════════════════

class ADWINDetector(BaseDriftDetector):
    """ADWIN-style adaptive windowing on a scalar alignment signal.

    Bifet & Gavalda (2007). We feed it one scalar per query: the mean
    similarity of that query to the baseline KB centroids (lower = the query is
    drifting away from what the KB covers). ADWIN keeps a window and splits it
    into two sub-windows; if their means differ by more than a Hoeffding-style
    bound, it drops the older half and flags change.

    A lightweight self-contained implementation (no river/skmultiflow dep).
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5, **kwargs):
        self.delta = delta
        self.max_buckets = max_buckets
        self._kb: Optional[np.ndarray] = None
        self._window: List[float] = []
        self._ready = False
        self._last_score = 0.0

    def set_baseline(self, kb_embeddings, query_embeddings) -> bool:
        # keep full KB to compute per-query top-1 alignment (drops under drift)
        kb = np.asarray(kb_embeddings, dtype=np.float64)
        self._kb = kb / np.clip(np.linalg.norm(kb, axis=1, keepdims=True), 1e-10, None)
        self._window = []
        self._ready = True
        return True

    def calibrate_threshold(self, query_embeddings, window_size) -> Optional[float]:
        # ADWIN self-calibrates via delta; no separate threshold needed.
        return self.delta

    def _signal(self, window_query_embs: np.ndarray) -> np.ndarray:
        q = np.asarray(window_query_embs, dtype=np.float64)
        nq = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-10, None)
        # alignment = max similarity to any KB doc (per query). Falls
        # systematically as queries drift away from KB coverage — the signal a
        # streaming change-point detector should track.
        sims = nq @ self._kb.T              # (n_q, n_kb)
        return sims.max(axis=1)

    def detect(self, window_query_embs: np.ndarray) -> DriftResult:
        if not self._ready or self._kb is None:
            return DriftResult(fid_score=0.0, threshold=self.delta, is_drifted=False)

        sig = self._signal(window_query_embs)
        drifted = False
        max_gap = 0.0
        MAXLEN = 400  # bound memory; ADWIN drops old data on drift anyway
        for x in sig:
            self._window.append(float(x))
            if len(self._window) > MAXLEN:
                self._window.pop(0)
            n = len(self._window)
            if n < 8:
                continue
            arr = np.array(self._window)
            var = float(arr.var()) + 1e-9
            cut_found = False
            # check a few split points (coarse grid is enough and cheaper)
            for cut in range(2, n - 1):
                w0, w1 = arr[:cut], arr[cut:]
                n0, n1 = len(w0), len(w1)
                m = 1.0 / (1.0 / n0 + 1.0 / n1)
                # ADWIN2 Hoeffding-Bernstein bound using empirical variance
                dd = np.log(2.0 * np.log(n) / self.delta) if n > 2 else np.log(2.0 / self.delta)
                eps = np.sqrt((2.0 / m) * var * dd) + (2.0 / (3.0 * m)) * dd
                gap = abs(w0.mean() - w1.mean())
                max_gap = max(max_gap, gap)
                if gap > eps:
                    self._window = list(w1)
                    drifted = True
                    cut_found = True
                    break
            if cut_found:
                break

        self._last_score = float(max_gap)
        return DriftResult(fid_score=float(max_gap), threshold=self.delta,
                           is_drifted=drifted)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def threshold(self) -> float:
        return self.delta


# ════════════════════════════════════════════════════════════
# MMD — two-sample distance on RAW query embeddings
# ════════════════════════════════════════════════════════════

class MMDDetector(BaseDriftDetector):
    """Maximum Mean Discrepancy on raw query embeddings (RBF kernel).

    Contrast for the ablation: this detects shift in the *raw* query
    distribution, ignoring how those queries align with the KB. Our claim is
    that alignment-feature FID (DriftLensDetector) is more sensitive to the
    drift that actually matters for caching, because a query distribution can
    move while still being covered by the KB (no update needed) — raw MMD would
    false-alarm there, whereas alignment FID would not.

    Threshold calibrated by bootstrapping MMD over random baseline sub-windows
    and taking the P95, mirroring DriftLensDetector's calibration.
    """

    def __init__(self, gamma: Optional[float] = None,
                 threshold_percentile: float = 95.0,
                 threshold_n_samples: int = 300,
                 random_state: int = 42, **kwargs):
        self.gamma = gamma
        self.threshold_percentile = threshold_percentile
        self.threshold_n_samples = threshold_n_samples
        self.rng = np.random.default_rng(random_state)
        self._baseline_q: Optional[np.ndarray] = None
        self._threshold: Optional[float] = None
        self._ready = False

    @staticmethod
    def _median_gamma(x: np.ndarray) -> float:
        # median heuristic for RBF bandwidth
        n = min(len(x), 200)
        sub = x[:n]
        d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return 1.0 / max(med, 1e-8)

    def _mmd2(self, x: np.ndarray, y: np.ndarray) -> float:
        g = self.gamma
        kxx = np.exp(-g * np.sum((x[:, None] - x[None]) ** 2, axis=-1))
        kyy = np.exp(-g * np.sum((y[:, None] - y[None]) ** 2, axis=-1))
        kxy = np.exp(-g * np.sum((x[:, None] - y[None]) ** 2, axis=-1))
        return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())

    def set_baseline(self, kb_embeddings, query_embeddings) -> bool:
        q = np.asarray(query_embeddings, dtype=np.float64)
        if q.shape[0] < 4:
            return False
        self._baseline_q = q
        if self.gamma is None:
            self.gamma = self._median_gamma(q)
        self._ready = True
        return True

    def calibrate_threshold(self, query_embeddings, window_size) -> Optional[float]:
        if self._baseline_q is None:
            return None
        q = np.asarray(query_embeddings, dtype=np.float64)
        w = max(2, min(window_size, q.shape[0] // 2))
        scores = []
        for _ in range(self.threshold_n_samples):
            a = q[self.rng.choice(q.shape[0], w, replace=False)]
            b = q[self.rng.choice(q.shape[0], w, replace=False)]
            scores.append(self._mmd2(a, b))
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        return self._threshold

    def detect(self, window_query_embs: np.ndarray) -> DriftResult:
        thr = self._threshold if self._threshold is not None else float("inf")
        if not self._ready or self._baseline_q is None:
            return DriftResult(fid_score=0.0, threshold=thr, is_drifted=False)
        w = np.asarray(window_query_embs, dtype=np.float64)
        # compare window against a same-size baseline sub-sample
        n = min(len(w), self._baseline_q.shape[0])
        base = self._baseline_q[self.rng.choice(self._baseline_q.shape[0], n,
                                                replace=False)]
        score = self._mmd2(w[:n], base)
        return DriftResult(fid_score=score, threshold=thr,
                           is_drifted=score > thr)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else float("inf")
