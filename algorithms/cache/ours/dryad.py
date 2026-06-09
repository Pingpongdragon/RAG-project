"""DRYAD: drift detection + decision + dual-tier admission (ours, final)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.ours.routed_cache import RoutedCache

class DRYAD(RoutedCache):
    """DRYAD — Drift-aware Demand-driven Admission with entity-chained prefetch.

    The full final method = three modules on one pipeline:

      ① DETECT  — explicit drift signal from the **alignment-feature FID**
                  detector (algorithms/qarc DriftLensDetector): per query we
                  build [sim(q,c_1..c_K), top1..topN] against the current KB,
                  fit a baseline (mu0, Sigma0) over a warmup history, calibrate
                  a P95 FID threshold, then each window compute FID(window vs
                  baseline). FID > threshold => alignment drift. The baseline is
                  rebuilt after every KB write (KB changed -> centroids changed).
                  Falls back to the scalar AlignmentGap only if the detector is
                  unavailable (e.g. sklearn missing).
      ② DECIDE  — rule agent maps the drift signal to an action + replacement
                  ratio λ, which becomes the per-window write budget λ·B:
                    NoOp        (not drifted, gap normal)     -> budget 0
                    Mild        (gap above EMA+k·MAD)         -> λ_mild·B
                    Aggressive  (FID drift OR gap >> baseline)-> λ_aggr·B
                    warmup windows                            -> Aggressive
      ③ ADMIT   — inherited from RoutedCache: R1 (SemFlow neighborhood demand)
                  + R3 (entity-chained bridge prefetch) competing in one
                  demand/serve ledger under the unified gate, capped by λ·B.

    This replaces SemFlow's implicit "write cap = #failures" with an explicit
    detect→decide→admit loop driven by a real alignment-feature FID, so the
    system updates *when query-KB alignment drifts* and *as hard as the drift
    warrants*, not on every miss.
    """
    # Module ② decision hyperparameters (mirror algorithms/qarc kb_agent defaults)
    WARMUP_WINDOWS = 3
    GAP_EMA_BETA   = 0.85   # EMA smoothing for the gap baseline (fallback path)
    GAP_K          = 1.0    # MAD multiplier for the "gap high" threshold
    LAMBDA_MILD    = 0.05   # mild update: 5% of budget B
    LAMBDA_AGGR    = 0.15   # aggressive update: 15% of budget B
    DRIFT_GAP_MULT = 1.5    # gap > DRIFT_GAP_MULT * EMA  => treat as drift (fallback)
    # Module ① detector config
    DRIFT_N_CLUSTERS = 5
    DRIFT_TOP_N      = 10
    DRIFT_PCTL       = 95.0
    REBUILD_EVERY    = 5    # rebuild FID baseline at most every N writes (cost control)

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self._budget = None          # B = KB capacity, set on first step
        self._gap_ema = None         # EMA of AlignmentGap (fallback signal)
        self._gap_mad = 0.0
        self._win = 0                # window counter (for warmup)
        self.drift_log = []          # per-window dict for analysis
        # Module ① real FID detector (lazy init after warmup history collected)
        self._detector = None
        self._det_ready = False
        self._q_history = []         # accumulated query embeddings (warmup + ongoing)
        self._writes_since_rebuild = 0
        self._use_fid = True         # set False if detector deps unavailable

    # ── Module ① helpers ──────────────────────────────────────────────
    def _kb_emb_now(self):
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        return self.doc_embs[kb_idx]

    def _init_detector(self):
        """Build the alignment-feature FID detector from accumulated history."""
        try:
            from algorithms.qarc.detection.drift_detector import DriftLensDetector
        except Exception:
            self._use_fid = False
            return
        q_hist = np.asarray(self._q_history, dtype=np.float64)
        kb_emb = self._kb_emb_now().astype(np.float64)
        det = DriftLensDetector(
            n_clusters=self.DRIFT_N_CLUSTERS, top_n_sims=self.DRIFT_TOP_N,
            threshold_percentile=self.DRIFT_PCTL,
        )
        ok = det.set_baseline(kb_emb, q_hist)
        if not ok:
            self._use_fid = False
            return
        # calibrate threshold on the warmup history (needs >= 2*window samples)
        win = max(2, len(self._q_history) // max(1, self._win))
        det.calibrate_threshold(q_hist, window_size=win)
        self._detector = det
        self._det_ready = True

    def _rebuild_baseline(self):
        """KB changed after a write -> rebuild FID baseline on recent history."""
        if not (self._use_fid and self._detector is not None):
            return
        # use the most recent history (cap to keep it cheap)
        hist = np.asarray(self._q_history[-2000:], dtype=np.float64)
        kb_emb = self._kb_emb_now().astype(np.float64)
        try:
            self._detector.set_baseline(kb_emb, hist)
        except Exception:
            pass

    def _alignment_gap(self, nqe, kb_emb):
        """G(t) = 1 - mean_q max_{d in KB} sim(q, d)  (scalar fallback signal)."""
        q_kb = nqe @ kb_emb.T
        return float(1.0 - np.mean(np.max(q_kb, axis=1)))

    # ── Module ② decision ─────────────────────────────────────────────
    def _decide(self, gap, fid_drift):
        """Map (gap, fid_drift) -> (action, lambda).

        fid_drift: True/False from the real FID detector, or None if unavailable
        (then we fall back to the EMA/MAD gap rule).
        """
        if self._win < self.WARMUP_WINDOWS:
            action, lam = 'Aggressive', self.LAMBDA_AGGR
        else:
            ema = self._gap_ema if self._gap_ema is not None else gap
            thresh = ema + self.GAP_K * self._gap_mad
            if fid_drift is True:
                action, lam = 'Aggressive', self.LAMBDA_AGGR        # FID drift
            elif fid_drift is None and gap > self.DRIFT_GAP_MULT * ema:
                action, lam = 'Aggressive', self.LAMBDA_AGGR        # fallback drift
            elif gap > thresh:
                action, lam = 'Mild', self.LAMBDA_MILD              # gap high
            else:
                action, lam = 'NoOp', 0.0                          # stable
        # update gap EMA/MAD trackers (always, for the fallback + logging)
        if self._gap_ema is None:
            self._gap_ema = gap
        else:
            dev = abs(gap - self._gap_ema)
            self._gap_mad = self.GAP_EMA_BETA * self._gap_mad + (1 - self.GAP_EMA_BETA) * dev
            self._gap_ema = self.GAP_EMA_BETA * self._gap_ema + (1 - self.GAP_EMA_BETA) * gap
        return action, lam

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        if self._budget is None:
            self._budget = len(self.kb)
        # normalise queries once
        norms = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe = window_query_embs / np.clip(norms, 1e-10, None)
        kb_emb = self._kb_emb_now()
        # accumulate history for the detector
        self._q_history.extend(nqe.tolist())

        # ── Module ① DETECT ──
        gap = self._alignment_gap(nqe, kb_emb)
        fid_val, fid_drift = None, None
        if self._use_fid:
            if not self._det_ready and self._win + 1 >= self.WARMUP_WINDOWS:
                self._init_detector()          # build baseline after warmup
            if self._det_ready:
                res = self._detector.detect(nqe.astype(np.float64))
                fid_val, fid_drift = float(res.fid_score), bool(res.is_drifted)

        # ── Module ② DECIDE ──
        action, lam = self._decide(gap, fid_drift)
        self._win += 1
        self.drift_log.append({
            'w': window_idx, 'gap': round(gap, 4),
            'fid': round(fid_val, 4) if fid_val is not None else None,
            'fid_drift': fid_drift, 'action': action,
        })

        # ── Module ③ ADMIT/EVICT under budget λ·B ──
        if action == 'NoOp':
            self._write_budget = 0
            super().step(window_queries, window_query_embs, window_idx)
            self._write_budget = None
            return
        prev_writes = self.update_cost
        self._write_budget = max(1, int(lam * self._budget))
        super().step(window_queries, window_query_embs, window_idx)
        self._write_budget = None
        # rebuild FID baseline if KB actually changed (throttled)
        if self.update_cost > prev_writes:
            self._writes_since_rebuild += (self.update_cost - prev_writes)
            if self._writes_since_rebuild >= self.REBUILD_EVERY:
                self._rebuild_baseline()
                self._writes_since_rebuild = 0


